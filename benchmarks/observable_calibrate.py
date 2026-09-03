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

_MIN_REPORTS = 3
_MIN_DECISION_SAMPLES = 12
_MAX_DECISION_REGRET = 1.10
_MAX_MEDIAN_FACTOR = 1.5
_MAX_FACTOR = 2.0
_MAX_SEMANTIC_ERROR = 2e-11
_POLICY_VERSION = 1


@dataclass(frozen=True, slots=True)
class Observation:
    fingerprint: str
    workload: str
    backend: str
    qubits: int
    single_qubit_operations: int
    two_qubit_operations: int
    compiled_steps: int
    threads: int
    term_evaluations: int
    state_passes: int
    median_ns: float

    @property
    def cost_class(self) -> str:
        return "observable-return-cuda" if self.backend == "native-cuda" else "observable-return-cpu"


@dataclass(frozen=True, slots=True)
class CostModel:
    cost_class: str
    feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    training_fingerprints: tuple[str, ...]
    holdout_fingerprints: tuple[str, ...]
    training_error: ErrorMetrics
    holdout_error: ErrorMetrics
    validated: bool

    def predict_ns(self, row: Observation) -> float:
        score = sum(a * b for a, b in zip(self.coefficients, _features(row), strict=True))
        return score if self.cost_class == "observable-return-cuda" else math.exp(score)


@dataclass(frozen=True, slots=True)
class DecisionEvidence:
    samples: int
    mistakes: int
    max_regret: float
    rows: tuple[dict[str, Any], ...]

    @property
    def validated(self) -> bool:
        return self.samples >= _MIN_DECISION_SAMPLES and self.mistakes == 0 and self.max_regret <= _MAX_DECISION_REGRET


@dataclass(frozen=True, slots=True)
class ObservableCalibration:
    engine_version: str
    workload_version: int
    host: dict[str, str]
    report_count: int
    models: tuple[CostModel, CostModel]
    decision: DecisionEvidence

    @property
    def validated(self) -> bool:
        return self.report_count >= _MIN_REPORTS and all(model.validated for model in self.models) and self.decision.validated

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "policy_version": _POLICY_VERSION,
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

    def to_planner_text(self, base_artifact: Path) -> str:
        if not self.validated:
            raise ValueError("observable planner promotion requires validated evidence")
        base = qp.load_planner_cost_model(str(base_artifact))
        if base.schema_version != 3:
            raise ValueError("observable planner promotion requires a schema-v3 artifact")
        if not base.cuda_auto_validated or not base.mps_auto_validated:
            raise ValueError("schema-v3 base artifact is missing validated CUDA or MPS evidence")
        if base.engine_version != self.engine_version or base.workload_version != self.workload_version:
            raise ValueError("base planner artifact version does not match observable calibration")
        if base.host_fingerprint != self.host["planner_host_fingerprint"]:
            raise ValueError("base planner artifact host does not match observable calibration")
        if base.cuda_host_fingerprint != self.host["planner_cuda_host_fingerprint"]:
            raise ValueError("base planner CUDA host does not match observable calibration")
        lines = base_artifact.read_text(encoding="utf-8").splitlines()
        if not lines or lines[0] != "qupy-planner-cost 3":
            raise ValueError("base planner artifact text is not schema v3")
        if any("rich-observable" in line or "observable-return-" in line for line in lines):
            raise ValueError("base planner artifact already contains observable policy evidence")
        lines[0] = "qupy-planner-cost 4"
        for model in sorted(self.models, key=lambda item: item.cost_class):
            coefficients = " ".join(format(value, ".17g") for value in model.coefficients)
            lines.append(f"model {model.cost_class} {len(model.coefficients)} {coefficients} {model.holdout_error.median_factor:.17g} {model.holdout_error.max_factor:.17g}")
        lines.append(f"policy rich-observable {_POLICY_VERSION}")
        lines.append(f"decision rich-observable-auto {self.decision.samples} {self.decision.mistakes} {self.decision.max_regret:.17g}")
        return "\n".join(lines) + "\n"


def _cuda_reduction_kernels(qubits: int) -> int:
    count = max(1, (1 << qubits) // 256)
    kernels = 1
    while count > 1:
        count = (count + 255) // 256
        kernels += 1
    return kernels


def _features(row: Observation) -> tuple[float, ...]:
    operations = max(row.single_qubit_operations + row.two_qubit_operations, 1)
    qubits = float(row.qubits)
    if row.backend == "native-cuda":
        dimension = float(1 << row.qubits)
        compiled = float(row.compiled_steps)
        two_qubit = float(row.two_qubit_operations)
        single_qubit = max(compiled - two_qubit, 0.0)
        terms = float(row.term_evaluations)
        return (
            1.0,
            dimension,
            dimension * (0.5 * single_qubit + two_qubit),
            compiled,
            dimension * terms,
            terms * _cuda_reduction_kernels(row.qubits),
            terms,
        )
    log_steps = math.log(max(row.compiled_steps, 1))
    log_terms = math.log1p(row.term_evaluations)
    return (
        1.0,
        qubits,
        log_steps,
        row.two_qubit_operations / operations,
        log_terms,
        qubits * log_terms,
        qubits * qubits,
        log_terms * log_terms,
        log_steps * log_steps,
        float(row.state_passes),
        math.log(max(row.threads, 1)),
    )


def _feature_names(cost_class: str) -> tuple[str, ...]:
    if cost_class == "observable-return-cpu":
        return (
            "bias", "qubits", "log_compiled_steps", "two_qubit_fraction",
            "log1p_term_evaluations", "qubits_x_log1p_term_evaluations",
            "qubits_squared", "log1p_term_evaluations_squared",
            "log_compiled_steps_squared", "state_passes", "log_threads",
        )
    if cost_class == "observable-return-cuda":
        return (
            "bias", "dimension", "state_work_items", "compiled_steps",
            "term_work_items", "term_kernel_launches", "term_evaluations",
        )
    raise ValueError(f"unsupported observable cost class: {cost_class}")


def _error_metrics(actual: list[float], predicted: list[float]) -> ErrorMetrics:
    factors = [max(a / p, p / a) for a, p in zip(actual, predicted, strict=True)]
    percentages = [abs(a - p) / a for a, p in zip(actual, predicted, strict=True)]
    ordered = sorted(factors)
    p90_index = min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)
    return ErrorMetrics(len(actual), statistics.median(factors), ordered[p90_index], max(factors), statistics.fmean(percentages))


def _positive_samples(value: object, *, field: str, iterations: int) -> list[int]:
    if not isinstance(value, list) or len(value) != iterations:
        raise ValueError(f"observable policy {field} count does not match iterations")
    if not all(isinstance(item, int) and item > 0 for item in value):
        raise ValueError(f"observable policy {field} contains an invalid timing")
    return value


def _load_report(path: Path) -> tuple[dict[str, str], list[Observation], set[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("policy_version") != _POLICY_VERSION or payload.get("profile") != "policy":
        raise ValueError("observable calibration requires schema-v1 policy reports")
    if payload.get("engine_version") != qp.core_version() or payload.get("workload_version") != 1:
        raise ValueError("observable policy report does not match this QuPy runtime")
    iterations = payload.get("iterations")
    if not isinstance(iterations, int) or iterations < 2 or iterations % 2 != 0:
        raise ValueError("observable policy report requires a positive even iteration count")
    host = payload.get("host")
    if not isinstance(host, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in host.items()):
        raise TypeError("observable policy report is missing host metadata")
    for key in ("planner_host_fingerprint", "planner_cuda_host_fingerprint"):
        if len(host.get(key, "")) != 64:
            raise ValueError(f"observable policy host metadata {key} is invalid")
    validations = payload.get("validations")
    evidence = payload.get("policy_evidence")
    if not isinstance(validations, list) or not isinstance(evidence, list) or not evidence:
        raise ValueError("observable policy report is missing validation or decision evidence")
    validated_names: set[str] = set()
    for row in validations:
        if not isinstance(row, dict) or not isinstance(row.get("workload"), str):
            raise TypeError("observable policy validation row is invalid")
        error = row.get("max_abs_error")
        if not isinstance(error, (int, float)) or error < 0 or error > _MAX_SEMANTIC_ERROR:
            raise ValueError("observable policy report failed exactness validation")
        if row["workload"] in validated_names:
            raise ValueError("observable policy validation workload is duplicated")
        validated_names.add(row["workload"])

    observations: list[Observation] = []
    evidence_names: set[str] = set()
    for row in evidence:
        if not isinstance(row, dict):
            raise TypeError("observable policy evidence row is invalid")
        workload = row.get("workload")
        fingerprint = row.get("fingerprint")
        if not isinstance(workload, str) or not workload or workload in evidence_names or not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise ValueError("observable policy workload identity is invalid or duplicated")
        evidence_names.add(workload)
        structural = {key: row.get(key) for key in ("qubits", "single_qubit_operations", "two_qubit_operations", "compiled_steps", "threads")}
        if not all(isinstance(value, int) and value >= 0 for value in structural.values()):
            raise ValueError("observable policy workload structure is invalid")
        for backend, prefix, timing_key in (
            ("native-cpu", "cpu", "cpu_timings_ns"),
            ("native-cuda", "cuda", "cuda_timings_ns"),
        ):
            term_evaluations = row.get(f"{prefix}_term_evaluations")
            state_passes = row.get(f"{prefix}_state_passes")
            if (
                not isinstance(term_evaluations, int)
                or term_evaluations < 0
                or not isinstance(state_passes, int)
                or state_passes < 0
                or (term_evaluations == 0 and state_passes == 0)
            ):
                raise ValueError("observable policy work descriptor is invalid")
            samples = _positive_samples(row.get(timing_key), field=timing_key, iterations=iterations)
            observations.append(
                Observation(
                    fingerprint=fingerprint,
                    workload=workload,
                    backend=backend,
                    qubits=int(structural["qubits"]),
                    single_qubit_operations=int(structural["single_qubit_operations"]),
                    two_qubit_operations=int(structural["two_qubit_operations"]),
                    compiled_steps=int(structural["compiled_steps"]),
                    threads=int(structural["threads"]),
                    term_evaluations=term_evaluations,
                    state_passes=state_passes,
                    median_ns=float(statistics.median(samples)),
                )
            )
    if validated_names != evidence_names:
        raise ValueError("observable exactness validation does not cover the decision workload set")
    return dict(host), observations, evidence_names


def _aggregate(rows: list[Observation]) -> list[Observation]:
    grouped: dict[tuple[str, str], list[Observation]] = defaultdict(list)
    for row in rows:
        grouped[(row.fingerprint, row.backend)].append(row)
    result: list[Observation] = []
    for group in grouped.values():
        first = group[0]
        shape = {
            (
                x.workload, x.qubits, x.single_qubit_operations, x.two_qubit_operations,
                x.compiled_steps, x.threads, x.term_evaluations, x.state_passes,
            )
            for x in group
        }
        if len(shape) != 1:
            raise ValueError("observable workload structure changed between reports")
        result.append(
            Observation(
                first.fingerprint, first.workload, first.backend, first.qubits,
                first.single_qubit_operations, first.two_qubit_operations,
                first.compiled_steps, first.threads, first.term_evaluations,
                first.state_passes, float(statistics.median(x.median_ns for x in group)),
            )
        )
    return result


def _validation_folds(rows: list[Observation]) -> tuple[frozenset[str], ...]:
    cpu = sorted(
        (row for row in rows if row.backend == "native-cpu"),
        key=lambda row: row.workload,
    )
    if len(cpu) < 18:
        raise ValueError("observable calibration requires at least 18 paired workloads")
    return tuple(frozenset((row.fingerprint,)) for row in cpu)


def _nonnegative_least_squares(
    design: np.ndarray,
    targets: np.ndarray,
) -> tuple[float, ...]:
    scale = np.linalg.norm(design, axis=0)
    scale[scale == 0.0] = 1.0
    normalized = design / scale
    if np.linalg.matrix_rank(normalized) < normalized.shape[1]:
        raise ValueError("observable CUDA cost calibration set is rank deficient")
    coefficients = np.maximum(np.linalg.lstsq(normalized, targets, rcond=None)[0], 0.0)
    prediction = normalized @ coefficients
    for _ in range(20000):
        previous = coefficients.copy()
        for column in range(normalized.shape[1]):
            values = normalized[:, column]
            residual = targets - prediction + values * coefficients[column]
            denominator = float(values @ values)
            updated = 0.0 if denominator == 0.0 else max(0.0, float(values @ residual) / denominator)
            prediction += values * (updated - coefficients[column])
            coefficients[column] = updated
        delta = float(np.max(np.abs(coefficients - previous)))
        magnitude = max(1.0, float(np.max(np.abs(coefficients))))
        if delta <= 1e-10 * magnitude:
            break
    result = coefficients / scale
    if not np.all(np.isfinite(result)) or not np.any(result > 0.0):
        raise ValueError("observable CUDA cost calibration produced invalid coefficients")
    return tuple(float(value) for value in result)


def _fit_coefficients(rows: list[Observation]) -> tuple[float, ...]:
    if not rows:
        raise ValueError("observable cost model requires observations")
    names = _feature_names(rows[0].cost_class)
    if len(rows) < len(names):
        raise ValueError("observable cost model has too little training data")
    design = np.asarray([_features(row) for row in rows], dtype=np.float64)
    raw_targets = np.asarray([row.median_ns for row in rows], dtype=np.float64)
    if rows[0].cost_class == "observable-return-cuda":
        return _nonnegative_least_squares(design, raw_targets)
    targets = np.log(raw_targets)
    coefficients_array, _, rank, _ = np.linalg.lstsq(design, targets, rcond=None)
    if rank < len(names):
        raise ValueError("observable cost calibration set is rank deficient")
    return tuple(float(value) for value in coefficients_array)


def _predict(coefficients: tuple[float, ...], row: Observation) -> float:
    score = sum(a * b for a, b in zip(coefficients, _features(row), strict=True))
    prediction = score if row.cost_class == "observable-return-cuda" else math.exp(score)
    if not math.isfinite(prediction) or prediction <= 0.0:
        raise ValueError("observable cost calibration produced an invalid prediction")
    return prediction


def _cross_validated_model(
    rows: list[Observation],
    folds: tuple[frozenset[str], ...],
) -> tuple[CostModel, dict[str, float]]:
    names = _feature_names(rows[0].cost_class)
    predictions: dict[str, float] = {}
    for fold in folds:
        training = [row for row in rows if row.fingerprint not in fold]
        held = [row for row in rows if row.fingerprint in fold]
        coefficients = _fit_coefficients(training)
        for row in held:
            if row.fingerprint in predictions:
                raise ValueError("observable validation fingerprint appears in multiple folds")
            predictions[row.fingerprint] = _predict(coefficients, row)
    if set(predictions) != {row.fingerprint for row in rows}:
        raise ValueError("observable cross-validation did not cover every workload")

    final_coefficients = _fit_coefficients(rows)
    training_predicted = [_predict(final_coefficients, row) for row in rows]
    validation_predicted = [predictions[row.fingerprint] for row in rows]
    training_error = _error_metrics([row.median_ns for row in rows], training_predicted)
    holdout_error = _error_metrics([row.median_ns for row in rows], validation_predicted)
    fingerprints = tuple(sorted(row.fingerprint for row in rows))
    return (
        CostModel(
            rows[0].cost_class,
            names,
            final_coefficients,
            fingerprints,
            fingerprints,
            training_error,
            holdout_error,
            holdout_error.median_factor <= _MAX_MEDIAN_FACTOR
            and holdout_error.max_factor <= _MAX_FACTOR,
        ),
        predictions,
    )


def _decision(
    rows: list[Observation],
    cpu_predictions: dict[str, float],
    cuda_predictions: dict[str, float],
) -> DecisionEvidence:
    paired: dict[str, dict[str, Observation]] = defaultdict(dict)
    for row in rows:
        paired[row.fingerprint][row.backend] = row
    result: list[dict[str, Any]] = []
    mistakes = 0
    max_regret = 1.0
    for fingerprint in sorted(paired):
        pair = paired[fingerprint]
        cpu, cuda = pair["native-cpu"], pair["native-cuda"]
        cpu_predicted = cpu_predictions[fingerprint]
        cuda_predicted = cuda_predictions[fingerprint]
        selected = cuda if cuda_predicted < cpu_predicted else cpu
        best = min(cpu.median_ns, cuda.median_ns)
        regret = selected.median_ns / best
        max_regret = max(max_regret, regret)
        if regret > _MAX_DECISION_REGRET:
            mistakes += 1
        result.append({
            "fingerprint": fingerprint,
            "workload": cpu.workload,
            "selected_backend": selected.backend,
            "actual_best_backend": "native-cuda" if cuda.median_ns < cpu.median_ns else "native-cpu",
            "cpu_actual_ns": cpu.median_ns,
            "cuda_actual_ns": cuda.median_ns,
            "cpu_predicted_ns": cpu_predicted,
            "cuda_predicted_ns": cuda_predicted,
            "regret": regret,
        })
    return DecisionEvidence(len(result), mistakes, max_regret, tuple(result))


def calibrate_files(paths: tuple[Path, ...]) -> ObservableCalibration:
    if len(paths) < _MIN_REPORTS:
        raise ValueError(f"observable policy calibration requires at least {_MIN_REPORTS} reports")
    hosts: list[dict[str, str]] = []
    observations: list[Observation] = []
    identities: list[set[str]] = []
    for path in paths:
        host, rows, names = _load_report(path)
        hosts.append(host)
        observations.extend(rows)
        identities.append(names)
    if any(host != hosts[0] for host in hosts[1:]):
        raise ValueError("observable policy reports came from different hosts")
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("observable policy reports do not cover the same workload set")
    aggregated = _aggregate(observations)
    by_fingerprint: dict[str, set[str]] = defaultdict(set)
    for row in aggregated:
        by_fingerprint[row.fingerprint].add(row.backend)
    if any(backends != {"native-cpu", "native-cuda"} for backends in by_fingerprint.values()):
        raise ValueError("every observable workload requires paired CPU and CUDA evidence")
    folds = _validation_folds(aggregated)
    cpu_model, cpu_predictions = _cross_validated_model(
        [row for row in aggregated if row.backend == "native-cpu"], folds
    )
    cuda_model, cuda_predictions = _cross_validated_model(
        [row for row in aggregated if row.backend == "native-cuda"], folds
    )
    models = (cpu_model, cuda_model)
    decision = _decision(aggregated, cpu_predictions, cuda_predictions)
    return ObservableCalibration(
        qp.core_version(), 1, hosts[0], len(paths), models, decision
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate QuPy rich-observable planner evidence")
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
