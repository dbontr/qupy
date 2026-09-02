from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

import qupy as qp
from benchmarks.cost_model import ErrorMetrics

_MAX_MEDIAN_FACTOR = 1.5
_MAX_FACTOR = 2.0
_MIN_DECISION_SAMPLES = 8
_MAX_DECISION_REGRET = 1.10


@dataclass(frozen=True, slots=True)
class ReturnObservation:
    fingerprint: str
    workload: str
    backend: str
    qubits: int
    single_qubit_operations: int
    two_qubit_operations: int
    compiled_steps: int
    threads: int
    median_ns: float

    @property
    def cost_class(self) -> str:
        return (
            "statevector-return-cuda" if self.backend == "native-cuda" else "statevector-return-cpu"
        )

    @property
    def sort_work(self) -> float:
        return (
            math.log(max(self.compiled_steps, 1))
            + self.qubits * math.log(2.0)
            - math.log(max(self.threads, 1))
        )


@dataclass(frozen=True, slots=True)
class ReturnCostModel:
    cost_class: str
    feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    training_fingerprints: tuple[str, ...]
    holdout_fingerprints: tuple[str, ...]
    training_error: ErrorMetrics
    holdout_error: ErrorMetrics
    validated: bool

    def predict_ns(self, observation: ReturnObservation) -> float:
        features = _features(observation)
        if len(features) != len(self.coefficients):
            raise ValueError("return-cost feature shape does not match coefficients")
        return math.exp(sum(a * b for a, b in zip(self.coefficients, features, strict=True)))


@dataclass(frozen=True, slots=True)
class DecisionEvidence:
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
class ReturnCalibration:
    engine_version: str
    workload_version: int
    host: dict[str, str]
    report_count: int
    models: tuple[ReturnCostModel, ReturnCostModel]
    decision: DecisionEvidence

    @property
    def validated(self) -> bool:
        return all(model.validated for model in self.models) and self.decision.validated

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "engine_version": self.engine_version,
            "workload_version": self.workload_version,
            "host": self.host,
            "report_count": self.report_count,
            "validated": self.validated,
            "models": [asdict(model) for model in self.models],
            "decision": asdict(self.decision) | {"validated": self.decision.validated},
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_planner_text(self, cpu_artifact: Path) -> str:
        if not self.validated:
            raise ValueError("CUDA planner promotion requires validated return-cost evidence")
        legacy = qp.load_planner_cost_model(str(cpu_artifact))
        if legacy.schema_version != 1:
            raise ValueError("CUDA planner promotion requires a schema-v1 CPU artifact")
        if (
            legacy.engine_version != self.engine_version
            or legacy.workload_version != self.workload_version
        ):
            raise ValueError("CPU artifact version does not match state-vector calibration")
        if legacy.host_fingerprint != self.host["planner_host_fingerprint"]:
            raise ValueError("CPU artifact host does not match state-vector calibration")
        legacy_lines = cpu_artifact.read_text(encoding="utf-8").splitlines()
        legacy_models = sorted(line for line in legacy_lines if line.startswith("model "))
        expected_classes = {
            "pauli-propagation",
            "statevector-parallel",
            "statevector-serial",
        }
        found_classes = {line.split()[1] for line in legacy_models}
        if found_classes != expected_classes:
            raise ValueError("CPU artifact does not contain the expected legacy cost classes")
        lines = [
            "qupy-planner-cost 2",
            f"engine {self.engine_version}",
            f"workload {self.workload_version}",
            f"host {self.host['planner_host_fingerprint']}",
            f"cuda-host {self.host['planner_cuda_host_fingerprint']}",
            "validated 1",
            *legacy_models,
        ]
        for model in sorted(self.models, key=lambda item: item.cost_class):
            coefficients = " ".join(format(value, ".17g") for value in model.coefficients)
            lines.append(
                f"model {model.cost_class} {len(model.coefficients)} {coefficients} "
                f"{format(model.holdout_error.median_factor, '.17g')} "
                f"{format(model.holdout_error.max_factor, '.17g')}"
            )
        lines.append(
            "decision statevector-auto "
            f"{self.decision.samples} {self.decision.mistakes} "
            f"{format(self.decision.max_regret, '.17g')}"
        )
        return "\n".join(lines) + "\n"


def _error_metrics(actual: list[float], predicted: list[float]) -> ErrorMetrics:
    if not actual or len(actual) != len(predicted):
        raise ValueError("cost error metrics require paired non-empty samples")
    factors = [max(a / p, p / a) for a, p in zip(actual, predicted, strict=True)]
    percentages = [abs(a - p) / a for a, p in zip(actual, predicted, strict=True)]
    ordered = sorted(factors)
    p90_index = min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)
    return ErrorMetrics(
        samples=len(actual),
        median_factor=statistics.median(factors),
        p90_factor=ordered[p90_index],
        max_factor=max(factors),
        mean_absolute_percentage_error=statistics.fmean(percentages),
    )


def load_observations(
    paths: tuple[Path, ...],
) -> tuple[list[ReturnObservation], dict[str, str], str, int, int]:
    if not paths:
        raise ValueError("at least one state-vector report is required")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    for report in reports:
        if report.get("schema_version") != 1:
            raise ValueError("unsupported state-vector benchmark schema")
    hosts = [report.get("host") for report in reports]
    if any(host != hosts[0] for host in hosts[1:]):
        raise ValueError("state-vector reports came from different hosts")
    host = hosts[0]
    if not isinstance(host, dict):
        raise TypeError("state-vector report is missing host metadata")
    engine_versions = {report.get("engine_version") for report in reports}
    workload_versions = {report.get("workload_version") for report in reports}
    if len(engine_versions) != 1 or len(workload_versions) != 1:
        raise ValueError("state-vector reports use different engine or workload versions")
    engine_version = next(iter(engine_versions))
    workload_version = next(iter(workload_versions))
    if not isinstance(engine_version, str) or not isinstance(workload_version, int):
        raise TypeError("state-vector report version metadata is invalid")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for report in reports:
        results = report.get("results")
        if not isinstance(results, list):
            raise TypeError("state-vector report is missing results")
        for row in results:
            if not isinstance(row, dict):
                raise TypeError("state-vector result row is invalid")
            fingerprint = row.get("workload_fingerprint")
            backend = row.get("backend")
            if not isinstance(fingerprint, str) or backend not in {"native-cpu", "native-cuda"}:
                raise ValueError("state-vector result identity is invalid")
            grouped[(fingerprint, backend)].append(row)

    observations: list[ReturnObservation] = []
    for (fingerprint, backend), rows in grouped.items():
        first = rows[0]
        structural = (
            "workload",
            "qubits",
            "single_qubit_operations",
            "two_qubit_operations",
            "compiled_steps",
            "threads",
        )
        if any(any(row.get(key) != first.get(key) for key in structural) for row in rows[1:]):
            raise ValueError("state-vector report structure changed between repeated runs")
        medians = [float(row["median_ns"]) for row in rows]
        if any(not math.isfinite(value) or value <= 0.0 for value in medians):
            raise ValueError("state-vector report contains invalid timings")
        observations.append(
            ReturnObservation(
                fingerprint=fingerprint,
                workload=str(first["workload"]),
                backend=backend,
                qubits=int(first["qubits"]),
                single_qubit_operations=int(first["single_qubit_operations"]),
                two_qubit_operations=int(first["two_qubit_operations"]),
                compiled_steps=int(first["compiled_steps"]),
                threads=int(first["threads"]),
                median_ns=statistics.median(medians),
            )
        )
    paired: dict[str, set[str]] = defaultdict(set)
    for observation in observations:
        paired[observation.fingerprint].add(observation.backend)
    if any(backends != {"native-cpu", "native-cuda"} for backends in paired.values()):
        raise ValueError("every state-vector workload requires paired CPU and CUDA results")
    return observations, dict(host), engine_version, workload_version, len(reports)


def _features(observation: ReturnObservation) -> tuple[float, ...]:
    operations = max(observation.single_qubit_operations + observation.two_qubit_operations, 1)
    common = (
        1.0,
        float(observation.qubits),
        math.log(max(observation.compiled_steps, 1)),
        observation.two_qubit_operations / operations,
    )
    if observation.backend == "native-cpu":
        return (*common, math.log(max(observation.threads, 1)))
    return common


def _feature_names(cost_class: str) -> tuple[str, ...]:
    common = ("bias", "qubits", "log_compiled_steps", "two_qubit_fraction")
    if cost_class == "statevector-return-cpu":
        return (*common, "log_threads")
    if cost_class == "statevector-return-cuda":
        return common
    raise ValueError(f"unsupported return-cost class: {cost_class}")


def _holdout_fingerprints(observations: list[ReturnObservation]) -> set[str]:
    cpu_rows = sorted(
        (row for row in observations if row.backend == "native-cpu"),
        key=lambda row: (row.sort_work, row.workload),
    )
    minimum_training = 5
    if len(cpu_rows) < _MIN_DECISION_SAMPLES + minimum_training:
        raise ValueError("state-vector calibration requires at least 13 paired workloads")
    count = min(
        max(_MIN_DECISION_SAMPLES, len(cpu_rows) // 3),
        len(cpu_rows) - minimum_training,
    )
    indices = {
        round(position * (len(cpu_rows) - 1) / (count + 1)) for position in range(1, count + 1)
    }
    candidate = 1
    while len(indices) < count:
        if candidate < len(cpu_rows) - 1:
            indices.add(candidate)
        candidate += 1
    return {cpu_rows[index].fingerprint for index in indices}


def _fit_model(
    rows: list[ReturnObservation],
    holdout_fingerprints: set[str],
) -> ReturnCostModel:
    if not rows:
        raise ValueError("return-cost model requires observations")
    training = [row for row in rows if row.fingerprint not in holdout_fingerprints]
    holdout = [row for row in rows if row.fingerprint in holdout_fingerprints]
    feature_names = _feature_names(rows[0].cost_class)
    if len(training) < len(feature_names) or len(holdout) < _MIN_DECISION_SAMPLES:
        raise ValueError("return-cost model has too little training or holdout data")
    design = np.asarray([_features(row) for row in training], dtype=np.float64)
    targets = np.log(np.asarray([row.median_ns for row in training], dtype=np.float64))
    coefficients_array, _, rank, _ = np.linalg.lstsq(design, targets, rcond=None)
    if rank < len(feature_names):
        raise ValueError("return-cost calibration set is rank deficient")
    coefficients = tuple(float(value) for value in coefficients_array)
    training_predicted = [
        math.exp(sum(a * b for a, b in zip(coefficients, _features(row), strict=True)))
        for row in training
    ]
    holdout_predicted = [
        math.exp(sum(a * b for a, b in zip(coefficients, _features(row), strict=True)))
        for row in holdout
    ]
    training_error = _error_metrics([row.median_ns for row in training], training_predicted)
    holdout_error = _error_metrics([row.median_ns for row in holdout], holdout_predicted)
    return ReturnCostModel(
        cost_class=rows[0].cost_class,
        feature_names=feature_names,
        coefficients=coefficients,
        training_fingerprints=tuple(sorted(row.fingerprint for row in training)),
        holdout_fingerprints=tuple(sorted(row.fingerprint for row in holdout)),
        training_error=training_error,
        holdout_error=holdout_error,
        validated=(
            holdout_error.median_factor <= _MAX_MEDIAN_FACTOR
            and holdout_error.max_factor <= _MAX_FACTOR
        ),
    )


def _decision_evidence(
    observations: list[ReturnObservation],
    models: tuple[ReturnCostModel, ReturnCostModel],
    holdout_fingerprints: set[str],
) -> DecisionEvidence:
    by_class = {model.cost_class: model for model in models}
    by_fingerprint: dict[str, dict[str, ReturnObservation]] = defaultdict(dict)
    for row in observations:
        if row.fingerprint in holdout_fingerprints:
            by_fingerprint[row.fingerprint][row.backend] = row
    rows: list[dict[str, Any]] = []
    material_mistakes = 0
    max_regret = 1.0
    for fingerprint in sorted(by_fingerprint):
        pair = by_fingerprint[fingerprint]
        cpu = pair["native-cpu"]
        cuda = pair["native-cuda"]
        cpu_predicted = by_class[cpu.cost_class].predict_ns(cpu)
        cuda_predicted = by_class[cuda.cost_class].predict_ns(cuda)
        selected = "native-cuda" if cuda_predicted < cpu_predicted else "native-cpu"
        selected_actual = cuda.median_ns if selected == "native-cuda" else cpu.median_ns
        best_actual = min(cpu.median_ns, cuda.median_ns)
        regret = selected_actual / best_actual
        max_regret = max(max_regret, regret)
        if regret > _MAX_DECISION_REGRET:
            material_mistakes += 1
        rows.append(
            {
                "fingerprint": fingerprint,
                "workload": cpu.workload,
                "selected_backend": selected,
                "actual_best_backend": (
                    "native-cuda" if cuda.median_ns < cpu.median_ns else "native-cpu"
                ),
                "cpu_actual_ns": cpu.median_ns,
                "cuda_actual_ns": cuda.median_ns,
                "cpu_predicted_ns": cpu_predicted,
                "cuda_predicted_ns": cuda_predicted,
                "regret": regret,
            }
        )
    return DecisionEvidence(
        samples=len(rows),
        mistakes=material_mistakes,
        max_regret=max_regret,
        rows=tuple(rows),
    )


def fit_return_costs(
    observations: list[ReturnObservation],
    *,
    host: dict[str, str],
    engine_version: str,
    workload_version: int,
    report_count: int,
) -> ReturnCalibration:
    if report_count < 1:
        raise ValueError("return-cost calibration requires at least one report")
    if engine_version != qp.core_version() or workload_version != 1:
        raise ValueError("state-vector calibration does not match this QuPy runtime")
    for key in ("planner_host_fingerprint", "planner_cuda_host_fingerprint"):
        value = host.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"state-vector calibration host metadata {key} is invalid")
    holdout_fingerprints = _holdout_fingerprints(observations)
    cpu_rows = [row for row in observations if row.backend == "native-cpu"]
    cuda_rows = [row for row in observations if row.backend == "native-cuda"]
    models = (
        _fit_model(cpu_rows, holdout_fingerprints),
        _fit_model(cuda_rows, holdout_fingerprints),
    )
    decision = _decision_evidence(observations, models, holdout_fingerprints)
    return ReturnCalibration(
        engine_version=engine_version,
        workload_version=workload_version,
        host=dict(host),
        report_count=report_count,
        models=models,
        decision=decision,
    )


def calibrate_files(paths: tuple[Path, ...]) -> ReturnCalibration:
    observations, host, engine_version, workload_version, report_count = load_observations(paths)
    return fit_return_costs(
        observations,
        host=host,
        engine_version=engine_version,
        workload_version=workload_version,
        report_count=report_count,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate paired CPU/CUDA QuPy state-vector return costs"
    )
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--cpu-artifact", required=True, type=Path)
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
    planner_text = calibration.to_planner_text(args.cpu_artifact)
    args.planner_output.parent.mkdir(parents=True, exist_ok=True)
    args.planner_output.write_text(planner_text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
