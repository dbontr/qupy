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
_MIN_DECISION_SAMPLES = 18
_MIN_WORKLOADS = 24
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
    noise_events: int
    kraus_evaluations: int
    median_ns: float

    @property
    def cost_class(self) -> str:
        return "density-return-cuda" if self.backend == "native-cuda" else "density-return-cpu"


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
        features = _speedup_features(row) if self.cost_class == "density-speedup" else _features(row)
        score = sum(a * b for a, b in zip(self.coefficients, features, strict=True))
        return score if self.cost_class == "density-return-cuda" else math.exp(score)


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
class DensityCalibration:
    engine_version: str
    workload_version: int
    host: dict[str, str]
    report_count: int
    models: tuple[CostModel, ...]
    decision: DecisionEvidence

    @property
    def validated(self) -> bool:
        return self.report_count >= _MIN_REPORTS and all(
            model.validated for model in self.models
        ) and self.decision.validated

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
            raise ValueError("density planner promotion requires validated evidence")
        base = qp.load_planner_cost_model(str(base_artifact))
        if base.schema_version != 4:
            raise ValueError("density planner promotion requires a schema-v4 artifact")
        if not base.cuda_auto_validated or not base.mps_auto_validated or not base.observable_auto_validated:
            raise ValueError("schema-v4 base artifact is missing validated planner evidence")
        if base.engine_version != self.engine_version or base.workload_version != self.workload_version:
            raise ValueError("base planner artifact version does not match density calibration")
        if base.host_fingerprint != self.host["planner_host_fingerprint"]:
            raise ValueError("base planner artifact host does not match density calibration")
        if base.cuda_host_fingerprint != self.host["planner_cuda_host_fingerprint"]:
            raise ValueError("base planner CUDA host does not match density calibration")
        lines = base_artifact.read_text(encoding="utf-8").splitlines()
        if not lines or lines[0] != "qupy-planner-cost 4":
            raise ValueError("base planner artifact text is not schema v4")
        if any("noisy-density" in line or "density-return-" in line for line in lines):
            raise ValueError("base planner artifact already contains density policy evidence")
        lines[0] = "qupy-planner-cost 5"
        for model in sorted(self.models, key=lambda item: item.cost_class):
            coefficients = " ".join(format(value, ".17g") for value in model.coefficients)
            lines.append(
                f"model {model.cost_class} {len(model.coefficients)} {coefficients} "
                f"{model.holdout_error.median_factor:.17g} {model.holdout_error.max_factor:.17g}"
            )
        lines.append(f"policy noisy-density {_POLICY_VERSION}")
        lines.append(
            f"decision noisy-density-auto {self.decision.samples} {self.decision.mistakes} "
            f"{self.decision.max_regret:.17g}"
        )
        return "\n".join(lines) + "\n"


def _features(row: Observation) -> tuple[float, ...]:
    qubits = float(row.qubits)
    operations = float(row.single_qubit_operations + row.two_qubit_operations)
    if row.backend == "native-cuda":
        elements = float(4**row.qubits)
        return (
            1.0,
            elements,
            elements * row.single_qubit_operations,
            elements * row.two_qubit_operations,
            elements * row.noise_events,
            operations,
            float(row.noise_events),
        )
    log_kraus = math.log1p(row.kraus_evaluations)
    return (
        1.0,
        qubits,
        qubits * qubits,
        math.log1p(operations),
        log_kraus,
        qubits * log_kraus,
    )


def _speedup_features(row: Observation) -> tuple[float, ...]:
    q = float(row.qubits)
    operations = float(row.single_qubit_operations + row.two_qubit_operations)
    return (
        1.0,
        q,
        q * q,
        math.log1p(operations),
        math.log1p(row.noise_events),
        math.log1p(row.kraus_evaluations),
    )


def _feature_names(cost_class: str) -> tuple[str, ...]:
    if cost_class == "density-return-cpu":
        return ("bias", "qubits", "qubits_squared", "log1p_operations", "log1p_kraus", "qubits_x_log1p_kraus")
    if cost_class == "density-return-cuda":
        return (
            "bias",
            "density_elements",
            "single_gate_work",
            "two_gate_work",
            "noise_work",
            "operations",
            "noise_events",
        )
    if cost_class == "density-speedup":
        return (
            "bias",
            "qubits",
            "qubits_squared",
            "log1p_operations",
            "log1p_noise_events",
            "log1p_kraus",
        )
    raise ValueError(f"unsupported density cost class: {cost_class}")


def _error_metrics(actual: list[float], predicted: list[float]) -> ErrorMetrics:
    factors = [max(a / p, p / a) for a, p in zip(actual, predicted, strict=True)]
    percentages = [abs(a - p) / a for a, p in zip(actual, predicted, strict=True)]
    ordered = sorted(factors)
    p90_index = min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)
    return ErrorMetrics(
        len(actual),
        statistics.median(factors),
        ordered[p90_index],
        max(factors),
        statistics.fmean(percentages),
    )


def _positive_samples(value: object, *, field: str, iterations: int) -> list[int]:
    if not isinstance(value, list) or len(value) != iterations:
        raise ValueError(f"density policy {field} count does not match iterations")
    if not all(isinstance(item, int) and item > 0 for item in value):
        raise ValueError(f"density policy {field} contains an invalid timing")
    return value


def _load_report(path: Path) -> tuple[dict[str, str], list[Observation], set[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 1
        or payload.get("policy_version") != _POLICY_VERSION
        or payload.get("profile") != "policy"
    ):
        raise ValueError("density calibration requires schema-v1 policy reports")
    if payload.get("engine_version") != qp.core_version() or payload.get("workload_version") != 1:
        raise ValueError("density policy report does not match this QuPy runtime")
    iterations = payload.get("iterations")
    if not isinstance(iterations, int) or iterations < 2 or iterations % 2 != 0:
        raise ValueError("density policy report requires a positive even iteration count")
    host = payload.get("host")
    if not isinstance(host, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in host.items()):
        raise TypeError("density policy report is missing host metadata")
    for key in ("planner_host_fingerprint", "planner_cuda_host_fingerprint"):
        if len(host.get(key, "")) != 64:
            raise ValueError(f"density policy host metadata {key} is invalid")
    validations = payload.get("validations")
    evidence = payload.get("policy_evidence")
    if not isinstance(validations, list) or not isinstance(evidence, list) or not evidence:
        raise ValueError("density policy report is missing validation or decision evidence")
    validated_names: set[str] = set()
    for row in validations:
        if not isinstance(row, dict) or not isinstance(row.get("workload"), str):
            raise TypeError("density policy validation row is invalid")
        error = row.get("max_abs_error")
        if not isinstance(error, (int, float)) or error < 0 or error > _MAX_SEMANTIC_ERROR:
            raise ValueError("density policy report failed exactness validation")
        workload = row["workload"]
        if workload in validated_names:
            raise ValueError("density policy validation workload is duplicated")
        validated_names.add(workload)

    observations: list[Observation] = []
    evidence_names: set[str] = set()
    for row in evidence:
        if not isinstance(row, dict):
            raise TypeError("density policy evidence row is invalid")
        workload = row.get("workload")
        fingerprint = row.get("fingerprint")
        if (
            not isinstance(workload, str)
            or not workload
            or workload in evidence_names
            or not isinstance(fingerprint, str)
            or len(fingerprint) != 64
        ):
            raise ValueError("density policy workload identity is invalid or duplicated")
        evidence_names.add(workload)
        structural = {
            key: row.get(key)
            for key in (
                "qubits",
                "single_qubit_operations",
                "two_qubit_operations",
                "noise_events",
                "kraus_evaluations",
            )
        }
        if not all(isinstance(value, int) and value >= 0 for value in structural.values()):
            raise ValueError("density policy workload structure is invalid")
        if structural["noise_events"] == 0 or structural["kraus_evaluations"] == 0:
            raise ValueError("density policy requires non-zero noisy execution work")
        for backend, timing_key in (
            ("native-cpu", "cpu_timings_ns"),
            ("native-cuda", "cuda_timings_ns"),
        ):
            samples = _positive_samples(
                row.get(timing_key), field=timing_key, iterations=iterations
            )
            observations.append(
                Observation(
                    fingerprint=fingerprint,
                    workload=workload,
                    backend=backend,
                    qubits=int(structural["qubits"]),
                    single_qubit_operations=int(structural["single_qubit_operations"]),
                    two_qubit_operations=int(structural["two_qubit_operations"]),
                    noise_events=int(structural["noise_events"]),
                    kraus_evaluations=int(structural["kraus_evaluations"]),
                    median_ns=float(statistics.median(samples)),
                )
            )
    if validated_names != evidence_names:
        raise ValueError("density exactness validation does not cover the decision workload set")
    return dict(host), observations, evidence_names


def _aggregate(rows: list[Observation]) -> list[Observation]:
    grouped: dict[tuple[str, str], list[Observation]] = defaultdict(list)
    for row in rows:
        grouped[(row.fingerprint, row.backend)].append(row)
    result: list[Observation] = []
    for group in grouped.values():
        first = group[0]
        structure = {
            (
                row.workload,
                row.qubits,
                row.single_qubit_operations,
                row.two_qubit_operations,
                row.noise_events,
                row.kraus_evaluations,
            )
            for row in group
        }
        if len(structure) != 1:
            raise ValueError("density workload structure changed between reports")
        result.append(
            Observation(
                first.fingerprint,
                first.workload,
                first.backend,
                first.qubits,
                first.single_qubit_operations,
                first.two_qubit_operations,
                first.noise_events,
                first.kraus_evaluations,
                float(statistics.median(row.median_ns for row in group)),
            )
        )
    return result


def _validation_folds(rows: list[Observation]) -> tuple[frozenset[str], ...]:
    cpu = sorted(
        (row for row in rows if row.backend == "native-cpu"),
        key=lambda row: row.workload,
    )
    if len(cpu) < _MIN_WORKLOADS:
        raise ValueError(
            f"density calibration requires at least {_MIN_WORKLOADS} paired workloads"
        )
    return tuple(frozenset((row.fingerprint,)) for row in cpu)


def _nonnegative_least_squares(
    design: np.ndarray,
    targets: np.ndarray,
) -> tuple[float, ...]:
    scale = np.linalg.norm(design, axis=0)
    scale[scale == 0.0] = 1.0
    normalized = design / scale
    if np.linalg.matrix_rank(normalized) < normalized.shape[1]:
        raise ValueError("density CUDA cost calibration set is rank deficient")
    coefficients = np.maximum(np.linalg.lstsq(normalized, targets, rcond=None)[0], 0.0)
    prediction = normalized @ coefficients
    for _ in range(20000):
        previous = coefficients.copy()
        for column in range(normalized.shape[1]):
            values = normalized[:, column]
            residual = targets - prediction + values * coefficients[column]
            denominator = float(values @ values)
            updated = (
                0.0
                if denominator == 0.0
                else max(0.0, float(values @ residual) / denominator)
            )
            prediction += values * (updated - coefficients[column])
            coefficients[column] = updated
        delta = float(np.max(np.abs(coefficients - previous)))
        magnitude = max(1.0, float(np.max(np.abs(coefficients))))
        if delta <= 1e-10 * magnitude:
            break
    result = coefficients / scale
    if not np.all(np.isfinite(result)) or not np.any(result > 0.0):
        raise ValueError("density CUDA cost calibration produced invalid coefficients")
    return tuple(float(value) for value in result)


def _fit_coefficients(rows: list[Observation]) -> tuple[float, ...]:
    if not rows:
        raise ValueError("density cost model requires observations")
    names = _feature_names(rows[0].cost_class)
    if len(rows) < len(names):
        raise ValueError("density cost model has too little training data")
    design = np.asarray([_features(row) for row in rows], dtype=np.float64)
    raw_targets = np.asarray([row.median_ns for row in rows], dtype=np.float64)
    if rows[0].cost_class == "density-return-cuda":
        return _nonnegative_least_squares(design, raw_targets)
    targets = np.log(raw_targets)
    coefficients, _, rank, _ = np.linalg.lstsq(design, targets, rcond=None)
    if rank < len(names):
        raise ValueError("density CPU cost calibration set is rank deficient")
    return tuple(float(value) for value in coefficients)


def _predict(coefficients: tuple[float, ...], row: Observation) -> float:
    score = sum(a * b for a, b in zip(coefficients, _features(row), strict=True))
    prediction = score if row.cost_class == "density-return-cuda" else math.exp(score)
    if not math.isfinite(prediction) or prediction <= 0.0:
        raise ValueError("density cost calibration produced an invalid prediction")
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
                raise ValueError("density validation fingerprint appears in multiple folds")
            predictions[row.fingerprint] = _predict(coefficients, row)
    if set(predictions) != {row.fingerprint for row in rows}:
        raise ValueError("density cross-validation did not cover every workload")

    final_coefficients = _fit_coefficients(rows)
    training_predicted = [_predict(final_coefficients, row) for row in rows]
    validation_predicted = [predictions[row.fingerprint] for row in rows]
    actual = [row.median_ns for row in rows]
    training_error = _error_metrics(actual, training_predicted)
    holdout_error = _error_metrics(actual, validation_predicted)
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


def _paired_observations(
    rows: list[Observation],
) -> dict[str, dict[str, Observation]]:
    paired: dict[str, dict[str, Observation]] = defaultdict(dict)
    for row in rows:
        pair = paired[row.fingerprint]
        if row.backend in pair:
            raise ValueError("density workload contains duplicate backend evidence")
        pair[row.backend] = row
    if any(set(pair) != {"native-cpu", "native-cuda"} for pair in paired.values()):
        raise ValueError("every density workload requires paired CPU and CUDA evidence")
    return paired


def _fit_speedup_coefficients(rows: list[Observation]) -> tuple[float, ...]:
    pairs = _paired_observations(rows)
    names = _feature_names("density-speedup")
    if len(pairs) < len(names):
        raise ValueError("density speedup model has too little training data")
    design: list[tuple[float, ...]] = []
    targets: list[float] = []
    for fingerprint in sorted(pairs):
        pair = pairs[fingerprint]
        cpu = pair["native-cpu"]
        cuda = pair["native-cuda"]
        design.append(_speedup_features(cpu))
        targets.append(math.log(cpu.median_ns / cuda.median_ns))
    matrix = np.asarray(design, dtype=np.float64)
    coefficients, _, rank, _ = np.linalg.lstsq(
        matrix, np.asarray(targets, dtype=np.float64), rcond=None
    )
    if rank < len(names):
        raise ValueError("density speedup calibration set is rank deficient")
    return tuple(float(value) for value in coefficients)


def _predict_speedup(coefficients: tuple[float, ...], row: Observation) -> float:
    score = sum(
        a * b for a, b in zip(coefficients, _speedup_features(row), strict=True)
    )
    prediction = math.exp(score)
    if not math.isfinite(prediction) or prediction <= 0.0:
        raise ValueError("density speedup calibration produced an invalid prediction")
    return prediction


def _cross_validated_speedup(
    rows: list[Observation],
    folds: tuple[frozenset[str], ...],
) -> tuple[CostModel, dict[str, float]]:
    pairs = _paired_observations(rows)
    predictions: dict[str, float] = {}
    for fold in folds:
        coefficients = _fit_speedup_coefficients(
            [row for row in rows if row.fingerprint not in fold]
        )
        for fingerprint in fold:
            if fingerprint in predictions:
                raise ValueError("density speedup fingerprint appears in multiple folds")
            predictions[fingerprint] = _predict_speedup(
                coefficients, pairs[fingerprint]["native-cpu"]
            )
    if set(predictions) != set(pairs):
        raise ValueError("density speedup validation did not cover every workload")

    final_coefficients = _fit_speedup_coefficients(rows)
    fingerprints = tuple(sorted(pairs))
    actual = [
        pairs[fingerprint]["native-cpu"].median_ns /
        pairs[fingerprint]["native-cuda"].median_ns
        for fingerprint in fingerprints
    ]
    training_predicted = [
        _predict_speedup(final_coefficients, pairs[fingerprint]["native-cpu"])
        for fingerprint in fingerprints
    ]
    validation_predicted = [predictions[fingerprint] for fingerprint in fingerprints]
    training_error = _error_metrics(actual, training_predicted)
    holdout_error = _error_metrics(actual, validation_predicted)
    return (
        CostModel(
            "density-speedup",
            _feature_names("density-speedup"),
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
    speedup_predictions: dict[str, float],
) -> DecisionEvidence:
    paired = _paired_observations(rows)
    result: list[dict[str, Any]] = []
    mistakes = 0
    max_regret = 1.0
    for fingerprint in sorted(paired):
        pair = paired[fingerprint]
        cpu, cuda = pair["native-cpu"], pair["native-cuda"]
        cpu_predicted = cpu_predictions[fingerprint]
        cuda_predicted = cuda_predictions[fingerprint]
        predicted_speedup = speedup_predictions[fingerprint]
        selected = cuda if predicted_speedup > 1.0 else cpu
        best = min(cpu.median_ns, cuda.median_ns)
        regret = selected.median_ns / best
        max_regret = max(max_regret, regret)
        if regret > _MAX_DECISION_REGRET:
            mistakes += 1
        result.append(
            {
                "fingerprint": fingerprint,
                "workload": cpu.workload,
                "selected_backend": selected.backend,
                "actual_best_backend": (
                    "native-cuda" if cuda.median_ns < cpu.median_ns else "native-cpu"
                ),
                "cpu_actual_ns": cpu.median_ns,
                "cuda_actual_ns": cuda.median_ns,
                "cpu_predicted_ns": cpu_predicted,
                "cuda_predicted_ns": cuda_predicted,
                "predicted_speedup": predicted_speedup,
                "regret": regret,
            }
        )
    return DecisionEvidence(len(result), mistakes, max_regret, tuple(result))


def calibrate_files(paths: tuple[Path, ...]) -> DensityCalibration:
    if len(paths) < _MIN_REPORTS:
        raise ValueError(f"density policy calibration requires at least {_MIN_REPORTS} reports")
    hosts: list[dict[str, str]] = []
    observations: list[Observation] = []
    identities: list[set[str]] = []
    for path in paths:
        host, rows, names = _load_report(path)
        hosts.append(host)
        observations.extend(rows)
        identities.append(names)
    if any(host != hosts[0] for host in hosts[1:]):
        raise ValueError("density policy reports came from different hosts")
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("density policy reports do not cover the same workload set")
    aggregated = _aggregate(observations)
    by_fingerprint: dict[str, set[str]] = defaultdict(set)
    for row in aggregated:
        by_fingerprint[row.fingerprint].add(row.backend)
    if any(backends != {"native-cpu", "native-cuda"} for backends in by_fingerprint.values()):
        raise ValueError("every density workload requires paired CPU and CUDA evidence")
    folds = _validation_folds(aggregated)
    cpu_model, cpu_predictions = _cross_validated_model(
        [row for row in aggregated if row.backend == "native-cpu"], folds
    )
    cuda_model, cuda_predictions = _cross_validated_model(
        [row for row in aggregated if row.backend == "native-cuda"], folds
    )
    speedup_model, speedup_predictions = _cross_validated_speedup(aggregated, folds)
    decision = _decision(
        aggregated, cpu_predictions, cuda_predictions, speedup_predictions
    )
    return DensityCalibration(
        qp.core_version(),
        1,
        hosts[0],
        len(paths),
        (cpu_model, cuda_model, speedup_model),
        decision,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate QuPy noisy-density planner evidence")
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
