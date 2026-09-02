from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import qupy as qp

_MIN_REPORTS = 3
_MIN_DECISION_SAMPLES = 16
_MAX_DECISION_REGRET = 1.10
_MAX_SEMANTIC_ERROR = 2e-11
_POLICY_VERSION = 1
_EXPECTED_BASELINES = {"native-cpu", "native-mps"}


@dataclass(frozen=True, slots=True)
class PairObservation:
    report: str
    fingerprint: str
    workload: str
    baseline_backend: str
    baseline_median_ns: float
    adaptive_median_ns: float
    regret: float


@dataclass(frozen=True, slots=True)
class MpsDecisionEvidence:
    samples: int
    mistakes: int
    max_regret: float
    rows: tuple[dict[str, Any], ...]

    @property
    def validated(self) -> bool:
        return (
            self.samples >= _MIN_DECISION_SAMPLES
            and self.mistakes == 0
            and self.max_regret <= _MAX_DECISION_REGRET
        )


@dataclass(frozen=True, slots=True)
class MpsCalibration:
    engine_version: str
    workload_version: int
    host: dict[str, str]
    report_count: int
    decision: MpsDecisionEvidence

    @property
    def validated(self) -> bool:
        return self.report_count >= _MIN_REPORTS and self.decision.validated

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "engine_version": self.engine_version,
            "workload_version": self.workload_version,
            "host": self.host,
            "report_count": self.report_count,
            "policy_version": _POLICY_VERSION,
            "validated": self.validated,
            "decision": asdict(self.decision) | {"validated": self.decision.validated},
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_planner_text(self, base_artifact: Path) -> str:
        if not self.validated:
            raise ValueError("MPS planner promotion requires validated policy evidence")
        base = qp.load_planner_cost_model(str(base_artifact))
        if base.schema_version not in (1, 2):
            raise ValueError("MPS planner promotion requires a schema-v1 or schema-v2 artifact")
        if (
            base.engine_version != self.engine_version
            or base.workload_version != self.workload_version
            or base.host_fingerprint != self.host["planner_host_fingerprint"]
        ):
            raise ValueError("base planner artifact does not match MPS calibration")

        base_lines = base_artifact.read_text(encoding="utf-8").splitlines()
        model_lines = sorted(line for line in base_lines if line.startswith("model "))
        cuda_host_lines = [line for line in base_lines if line.startswith("cuda-host ")]
        cuda_decisions = [
            line for line in base_lines if line.startswith("decision statevector-auto ")
        ]
        if base.schema_version == 1 and (cuda_host_lines or cuda_decisions):
            raise ValueError("schema-v1 base artifact unexpectedly contains CUDA evidence")
        if base.schema_version == 2 and (len(cuda_host_lines) != 1 or len(cuda_decisions) != 1):
            raise ValueError("schema-v2 base artifact is missing CUDA promotion evidence")

        lines = [
            "qupy-planner-cost 3",
            f"engine {self.engine_version}",
            f"workload {self.workload_version}",
            f"host {self.host['planner_host_fingerprint']}",
            *cuda_host_lines,
            "validated 1",
            *model_lines,
            *cuda_decisions,
            f"policy adaptive-mps {_POLICY_VERSION}",
            (
                "decision observable-auto "
                f"{self.decision.samples} {self.decision.mistakes} "
                f"{format(self.decision.max_regret, '.17g')}"
            ),
        ]
        return "\n".join(lines) + "\n"


def _positive_samples(value: object, *, field: str) -> list[int]:
    if not isinstance(value, list) or len(value) < 2 or len(value) % 2 != 0:
        raise ValueError(f"MPS policy {field} must contain a positive even sample count")
    if not all(isinstance(item, int) and item > 0 for item in value):
        raise ValueError(f"MPS policy {field} contains an invalid timing")
    return value


def _load_report(path: Path) -> tuple[dict[str, Any], list[PairObservation]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("profile") != "policy":
        raise ValueError("MPS calibration requires schema-v1 policy reports")
    if payload.get("policy_version") != _POLICY_VERSION:
        raise ValueError("MPS policy report uses an unsupported policy version")
    if payload.get("engine_version") != qp.core_version() or payload.get("workload_version") != 1:
        raise ValueError("MPS policy report does not match this QuPy runtime")
    iterations = payload.get("iterations")
    warmups = payload.get("warmups")
    if not isinstance(iterations, int) or iterations < 2 or iterations % 2 != 0:
        raise ValueError("MPS policy report requires a positive even iteration count")
    if not isinstance(warmups, int) or warmups < 0:
        raise ValueError("MPS policy report has an invalid warmup count")
    host = payload.get("host")
    if not isinstance(host, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in host.items()
    ):
        raise TypeError("MPS policy report is missing host metadata")
    if host.get("planner_host_fingerprint") != qp.planner_host_fingerprint():
        raise ValueError("MPS policy report host does not match this runtime")

    validations = payload.get("validations")
    if not isinstance(validations, list) or not validations:
        raise ValueError("MPS policy report is missing exactness validation")
    validation_workloads: set[str] = set()
    for validation in validations:
        if not isinstance(validation, dict):
            raise TypeError("MPS policy validation row is invalid")
        workload = validation.get("workload")
        error = validation.get("max_abs_error")
        if not isinstance(workload, str) or not workload or workload in validation_workloads:
            raise ValueError("MPS policy validation workload is invalid or duplicated")
        if (
            not isinstance(error, (int, float))
            or error < 0.0
            or error > _MAX_SEMANTIC_ERROR
        ):
            raise ValueError("MPS policy report failed exactness validation")
        validation_workloads.add(workload)

    evidence = payload.get("policy_evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("MPS policy report is missing paired decision evidence")
    observations: list[PairObservation] = []
    seen: set[tuple[str, str]] = set()
    evidence_workloads: set[str] = set()
    for row in evidence:
        if not isinstance(row, dict):
            raise TypeError("MPS policy evidence row is invalid")
        workload = row.get("workload")
        fingerprint = row.get("workload_fingerprint")
        comparisons = row.get("comparisons")
        if (
            not isinstance(workload, str)
            or not workload
            or workload in evidence_workloads
            or not isinstance(fingerprint, str)
            or len(fingerprint) != 64
        ):
            raise ValueError("MPS policy workload identity is invalid or duplicated")
        evidence_workloads.add(workload)
        if not isinstance(comparisons, list) or len(comparisons) != 2:
            raise ValueError("MPS policy workload must contain two paired comparisons")
        baselines: set[str] = set()
        for comparison in comparisons:
            if not isinstance(comparison, dict):
                raise TypeError("MPS policy comparison is invalid")
            baseline = comparison.get("baseline_backend")
            if not isinstance(baseline, str) or baseline not in _EXPECTED_BASELINES:
                raise ValueError("MPS policy comparison has an invalid baseline backend")
            if baseline in baselines or (fingerprint, baseline) in seen:
                raise ValueError("MPS policy comparison identity is duplicated")
            baselines.add(baseline)
            seen.add((fingerprint, baseline))
            baseline_samples = _positive_samples(
                comparison.get("baseline_timings_ns"), field="baseline timings"
            )
            adaptive_samples = _positive_samples(
                comparison.get("adaptive_timings_ns"), field="adaptive timings"
            )
            if (
                len(baseline_samples) != len(adaptive_samples)
                or len(baseline_samples) != iterations
            ):
                raise ValueError("MPS policy paired timing counts do not match the report")
            baseline_median = float(statistics.median(baseline_samples))
            adaptive_median = float(statistics.median(adaptive_samples))
            regret = max(1.0, adaptive_median / baseline_median)
            observations.append(
                PairObservation(
                    report=str(path),
                    fingerprint=fingerprint,
                    workload=workload,
                    baseline_backend=baseline,
                    baseline_median_ns=baseline_median,
                    adaptive_median_ns=adaptive_median,
                    regret=regret,
                )
            )
        if baselines != _EXPECTED_BASELINES:
            raise ValueError("MPS policy workload does not cover both decision baselines")
    if validation_workloads != evidence_workloads:
        raise ValueError("MPS policy validation does not cover the decision workload set")
    return dict(host), observations


def _decision_evidence(observations: list[PairObservation]) -> MpsDecisionEvidence:
    grouped: dict[str, dict[str, list[PairObservation]]] = defaultdict(lambda: defaultdict(list))
    workloads: dict[str, str] = {}
    for observation in observations:
        previous = workloads.setdefault(observation.fingerprint, observation.workload)
        if previous != observation.workload:
            raise ValueError("MPS policy fingerprint maps to inconsistent workload names")
        grouped[observation.fingerprint][observation.baseline_backend].append(observation)

    rows: list[dict[str, Any]] = []
    mistakes = 0
    max_regret = 1.0
    for fingerprint in sorted(grouped):
        pairs = grouped[fingerprint]
        if set(pairs) != _EXPECTED_BASELINES:
            raise ValueError("MPS policy workload does not have both baseline comparisons")
        comparison_rows: list[dict[str, Any]] = []
        workload_regret = 1.0
        for baseline in sorted(_EXPECTED_BASELINES):
            pair = pairs[baseline]
            regrets = [observation.regret for observation in pair]
            regret = float(statistics.median(regrets))
            workload_regret = max(workload_regret, regret)
            comparison_rows.append(
                {
                    "baseline_backend": baseline,
                    "reports": len(pair),
                    "median_regret": regret,
                    "max_observed_regret": max(regrets),
                    "median_baseline_ns": statistics.median(
                        observation.baseline_median_ns for observation in pair
                    ),
                    "median_adaptive_ns": statistics.median(
                        observation.adaptive_median_ns for observation in pair
                    ),
                }
            )
        if workload_regret > _MAX_DECISION_REGRET:
            mistakes += 1
        max_regret = max(max_regret, workload_regret)
        rows.append(
            {
                "fingerprint": fingerprint,
                "workload": workloads[fingerprint],
                "comparisons": comparison_rows,
                "regret": workload_regret,
            }
        )
    return MpsDecisionEvidence(
        samples=len(rows), mistakes=mistakes, max_regret=max_regret, rows=tuple(rows)
    )


def calibrate_files(paths: tuple[Path, ...]) -> MpsCalibration:
    if len(paths) < _MIN_REPORTS:
        raise ValueError(f"MPS policy calibration requires at least {_MIN_REPORTS} reports")
    hosts: list[dict[str, Any]] = []
    observations: list[PairObservation] = []
    identities: set[frozenset[tuple[str, str]]] = set()
    for path in paths:
        host, report_observations = _load_report(path)
        hosts.append(host)
        identities.add(
            frozenset((observation.fingerprint, observation.baseline_backend) for observation in report_observations)
        )
        observations.extend(report_observations)
    if any(host != hosts[0] for host in hosts[1:]):
        raise ValueError("MPS policy reports came from different hosts")
    if len(identities) != 1:
        raise ValueError("MPS policy reports do not cover the same workload set")
    decision = _decision_evidence(observations)
    return MpsCalibration(
        engine_version=qp.core_version(),
        workload_version=1,
        host=dict(hosts[0]),
        report_count=len(paths),
        decision=decision,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate QuPy adaptive MPS planner evidence")
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--base-artifact", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--planner-output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    calibration = calibrate_files(tuple(args.reports))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(calibration.to_json(), encoding="utf-8")
    if not calibration.validated:
        return 1
    planner_text = calibration.to_planner_text(args.base_artifact)
    args.planner_output.parent.mkdir(parents=True, exist_ok=True)
    args.planner_output.write_text(planner_text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
