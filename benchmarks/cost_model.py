from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

_COST_MODEL_SCHEMA_VERSION = 1
_BENCHMARK_SCHEMA_VERSION = 1
_WORKLOAD_VERSION = 1
_PLANNER_PROMOTION_MAX_HOLDOUT_FACTOR = 2.0
_PLANNER_PROMOTION_MAX_HOLDOUT_MEDIAN_FACTOR = 1.5
_LN2 = math.log(2.0)


@dataclass(frozen=True, slots=True)
class CostObservation:
    workload_fingerprint: str
    workload: str
    engine_version: str
    method: str
    cost_class: str
    median_ns: float
    active_qubits: int
    active_operations: int
    compiled_steps: int
    threads: int

    @property
    def log_work(self) -> float:
        if self.cost_class.startswith("statevector-"):
            return (
                math.log(max(self.compiled_steps, 1))
                + self.active_qubits * _LN2
                - math.log(max(self.threads, 1))
            )
        if self.cost_class == "pauli-propagation":
            return math.log(max(self.active_operations, 1))
        raise ValueError(f"unsupported cost class: {self.cost_class}")


@dataclass(frozen=True, slots=True)
class ErrorMetrics:
    samples: int
    median_factor: float
    p90_factor: float
    max_factor: float
    mean_absolute_percentage_error: float


@dataclass(frozen=True, slots=True)
class MethodCostModel:
    cost_class: str
    methods: tuple[str, ...]
    feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    training_fingerprints: tuple[str, ...]
    holdout_fingerprints: tuple[str, ...]
    training_error: ErrorMetrics
    holdout_error: ErrorMetrics
    validated: bool

    def predict_ns(self, observation: CostObservation) -> float:
        if observation.cost_class != self.cost_class:
            raise ValueError(
                f"cost model {self.cost_class} cannot predict {observation.cost_class}"
            )
        features = _feature_vector(observation)
        if len(features) != len(self.coefficients):
            raise ValueError("cost model feature shape does not match coefficients")
        log_runtime = sum(
            coefficient * feature
            for coefficient, feature in zip(self.coefficients, features, strict=True)
        )
        return math.exp(log_runtime)


@dataclass(frozen=True, slots=True)
class CostModelReport:
    schema_version: int
    benchmark_schema_version: int
    workload_version: int
    generated_at: str
    host: dict[str, str]
    engine: str
    engine_version: str
    source_report_count: int
    observations: int
    validated: bool
    models: tuple[MethodCostModel, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "benchmark_schema_version": self.benchmark_schema_version,
            "workload_version": self.workload_version,
            "generated_at": self.generated_at,
            "host": dict(self.host),
            "engine": self.engine,
            "engine_version": self.engine_version,
            "source_report_count": self.source_report_count,
            "observations": self.observations,
            "validated": self.validated,
            "models": [asdict(model) for model in self.models],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def to_planner_text(self) -> str:
        if not self.validated:
            raise ValueError("planner artifacts require a validated cost-model report")
        host_fingerprint = self.host.get("planner_host_fingerprint")
        if (
            not isinstance(host_fingerprint, str)
            or len(host_fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in host_fingerprint)
        ):
            raise ValueError("planner artifacts require a valid native host fingerprint")
        lines = [
            "qupy-planner-cost 1",
            f"engine {self.engine_version}",
            f"workload {self.workload_version}",
            f"host {host_fingerprint}",
            "validated 1",
        ]
        for model in sorted(self.models, key=lambda item: item.cost_class):
            if (
                model.holdout_error.median_factor > _PLANNER_PROMOTION_MAX_HOLDOUT_MEDIAN_FACTOR
                or model.holdout_error.max_factor > _PLANNER_PROMOTION_MAX_HOLDOUT_FACTOR
            ):
                raise ValueError("planner artifact does not meet fixed promotion thresholds")
            coefficients = " ".join(format(value, ".17g") for value in model.coefficients)
            lines.append(
                f"model {model.cost_class} {len(model.coefficients)} {coefficients} "
                f"{format(model.holdout_error.median_factor, '.17g')} "
                f"{format(model.holdout_error.max_factor, '.17g')}"
            )
        return "\n".join(lines) + "\n"


def _cost_class(method: str, threads: int) -> str:
    if method in {"statevector", "statevector-lightcone"}:
        return "statevector-serial" if threads == 1 else "statevector-parallel"
    if method == "pauli-propagation":
        return method
    raise ValueError(f"unsupported QuPy execution method for cost calibration: {method}")


def _require_int(metadata: dict[str, Any], key: str) -> int:
    value = metadata.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"benchmark metadata {key} must be a non-negative integer")
    return value


def _observation_from_result(result: dict[str, Any]) -> CostObservation | None:
    if result.get("engine") != "qupy" or result.get("skipped") is True:
        return None
    if result.get("valid") is not True:
        raise ValueError("cost calibration requires valid QuPy benchmark results")
    method = result.get("method")
    engine_version = result.get("engine_version")
    median_ns = result.get("median_ns")
    workload = result.get("workload")
    metadata = result.get("metadata")
    if not isinstance(method, str) or not method:
        raise ValueError("QuPy benchmark result is missing its execution method")
    if not isinstance(engine_version, str) or not engine_version:
        raise ValueError("QuPy benchmark result is missing its engine version")
    if not isinstance(median_ns, (int, float)) or median_ns <= 0 or not math.isfinite(median_ns):
        raise ValueError("QuPy benchmark result median_ns must be positive and finite")
    if not isinstance(workload, str) or not workload:
        raise ValueError("QuPy benchmark result is missing its workload name")
    if not isinstance(metadata, dict):
        raise TypeError("QuPy benchmark result is missing planner metadata")
    if metadata.get("workload_version") != _WORKLOAD_VERSION:
        raise ValueError("cost calibration requires workload fingerprint version 1")
    fingerprint = metadata.get("workload_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("QuPy benchmark result has an invalid workload fingerprint")
    threads = max(_require_int(metadata, "threads"), 1)
    return CostObservation(
        workload_fingerprint=fingerprint,
        workload=workload,
        engine_version=engine_version,
        method=method,
        cost_class=_cost_class(method, threads),
        median_ns=float(median_ns),
        active_qubits=_require_int(metadata, "active_qubits"),
        active_operations=_require_int(metadata, "active_operations"),
        compiled_steps=_require_int(metadata, "compiled_steps"),
        threads=threads,
    )


def _feature_vector(observation: CostObservation) -> tuple[float, ...]:
    log_work = observation.log_work
    if observation.cost_class.startswith("statevector-"):
        return (1.0, log_work, log_work * log_work)
    if observation.cost_class == "pauli-propagation":
        return (1.0, log_work)
    raise ValueError(f"unsupported cost class: {observation.cost_class}")


def _feature_names(cost_class: str) -> tuple[str, ...]:
    if cost_class.startswith("statevector-"):
        return ("bias", "log_work", "log_work_squared")
    if cost_class == "pauli-propagation":
        return ("bias", "log_active_operations")
    raise ValueError(f"unsupported cost class: {cost_class}")


def _aggregate_observations(observations: Iterable[CostObservation]) -> tuple[CostObservation, ...]:
    grouped: dict[tuple[str, str], list[CostObservation]] = defaultdict(list)
    for observation in observations:
        grouped[(observation.workload_fingerprint, observation.method)].append(observation)

    aggregated: list[CostObservation] = []
    for group in grouped.values():
        first = group[0]
        structural = {
            (
                item.workload,
                item.engine_version,
                item.cost_class,
                item.active_qubits,
                item.active_operations,
                item.compiled_steps,
                item.threads,
            )
            for item in group
        }
        if len(structural) != 1:
            raise ValueError("the same workload fingerprint has conflicting planner features")
        aggregated.append(
            CostObservation(
                workload_fingerprint=first.workload_fingerprint,
                workload=first.workload,
                engine_version=first.engine_version,
                method=first.method,
                cost_class=first.cost_class,
                median_ns=float(statistics.median(item.median_ns for item in group)),
                active_qubits=first.active_qubits,
                active_operations=first.active_operations,
                compiled_steps=first.compiled_steps,
                threads=first.threads,
            )
        )
    return tuple(sorted(aggregated, key=lambda item: (item.cost_class, item.log_work, item.workload)))


def _holdout_indices(size: int) -> set[int]:
    if size < 5:
        raise ValueError("each cost class requires at least five distinct workload fingerprints")
    holdout_count = max(2, size // 4)
    indices: set[int] = set()
    for position in range(1, holdout_count + 1):
        index = round(position * (size - 1) / (holdout_count + 1))
        index = min(max(index, 1), size - 2)
        indices.add(index)
    candidate = 1
    while len(indices) < holdout_count:
        if candidate < size - 1:
            indices.add(candidate)
        candidate += 1
    return indices


def _error_metrics(
    coefficients: Sequence[float],
    observations: Sequence[CostObservation],
) -> ErrorMetrics:
    if not observations:
        raise ValueError("error metrics require at least one observation")
    factors: list[float] = []
    percentage_errors: list[float] = []
    for observation in observations:
        features = _feature_vector(observation)
        predicted = math.exp(
            sum(
                coefficient * feature
                for coefficient, feature in zip(coefficients, features, strict=True)
            )
        )
        actual = observation.median_ns
        factors.append(max(predicted / actual, actual / predicted))
        percentage_errors.append(abs(predicted - actual) / actual)
    ordered = sorted(factors)
    p90_index = min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)
    return ErrorMetrics(
        samples=len(observations),
        median_factor=float(statistics.median(ordered)),
        p90_factor=float(ordered[p90_index]),
        max_factor=float(max(ordered)),
        mean_absolute_percentage_error=float(statistics.fmean(percentage_errors)),
    )


def _fit_cost_class(
    observations: Sequence[CostObservation],
    *,
    max_holdout_factor: float,
    max_holdout_median_factor: float,
) -> MethodCostModel:
    ordered = tuple(sorted(observations, key=lambda item: (item.log_work, item.workload)))
    holdout_indices = _holdout_indices(len(ordered))
    training = tuple(item for index, item in enumerate(ordered) if index not in holdout_indices)
    holdout = tuple(item for index, item in enumerate(ordered) if index in holdout_indices)
    feature_names = _feature_names(ordered[0].cost_class)
    if len(training) < len(feature_names):
        raise ValueError(
            f"cost class {ordered[0].cost_class} has too little training data for its feature model"
        )

    design = np.asarray([_feature_vector(item) for item in training], dtype=np.float64)
    targets = np.log(np.asarray([item.median_ns for item in training], dtype=np.float64))
    coefficients_array, _, rank, _ = np.linalg.lstsq(design, targets, rcond=None)
    if rank < len(feature_names):
        raise ValueError(f"cost class {ordered[0].cost_class} has a rank-deficient calibration set")
    coefficients = tuple(float(value) for value in coefficients_array)
    training_error = _error_metrics(coefficients, training)
    holdout_error = _error_metrics(coefficients, holdout)
    validated = (
        holdout_error.max_factor <= max_holdout_factor
        and holdout_error.median_factor <= max_holdout_median_factor
    )
    return MethodCostModel(
        cost_class=ordered[0].cost_class,
        methods=tuple(sorted({item.method for item in ordered})),
        feature_names=feature_names,
        coefficients=coefficients,
        training_fingerprints=tuple(item.workload_fingerprint for item in training),
        holdout_fingerprints=tuple(item.workload_fingerprint for item in holdout),
        training_error=training_error,
        holdout_error=holdout_error,
        validated=validated,
    )


def fit_cost_models(
    observations: Iterable[CostObservation],
    *,
    host: dict[str, str],
    source_report_count: int = 1,
    max_holdout_factor: float = 2.0,
    max_holdout_median_factor: float = 1.5,
) -> CostModelReport:
    if source_report_count < 1:
        raise ValueError("source_report_count must be at least one")
    if max_holdout_factor < 1.0 or max_holdout_median_factor < 1.0:
        raise ValueError("holdout factor thresholds must be at least one")
    aggregated = _aggregate_observations(observations)
    if not aggregated:
        raise ValueError("cost calibration found no QuPy benchmark observations")
    engine_versions = {observation.engine_version for observation in aggregated}
    if len(engine_versions) != 1:
        raise ValueError("cost calibration cannot mix QuPy engine versions")
    engine_version = next(iter(engine_versions))
    by_class: dict[str, list[CostObservation]] = defaultdict(list)
    for observation in aggregated:
        by_class[observation.cost_class].append(observation)
    models = tuple(
        _fit_cost_class(
            group,
            max_holdout_factor=max_holdout_factor,
            max_holdout_median_factor=max_holdout_median_factor,
        )
        for _, group in sorted(by_class.items())
    )
    return CostModelReport(
        schema_version=_COST_MODEL_SCHEMA_VERSION,
        benchmark_schema_version=_BENCHMARK_SCHEMA_VERSION,
        workload_version=_WORKLOAD_VERSION,
        generated_at=datetime.now(UTC).isoformat(),
        host=dict(host),
        engine="qupy",
        engine_version=engine_version,
        source_report_count=source_report_count,
        observations=len(aggregated),
        validated=all(model.validated for model in models),
        models=models,
    )


def load_observations(paths: Iterable[Path]) -> tuple[tuple[CostObservation, ...], dict[str, str], int]:
    observations: list[CostObservation] = []
    hosts: list[dict[str, str]] = []
    report_count = 0
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != _BENCHMARK_SCHEMA_VERSION:
            raise ValueError(f"{path} is not benchmark schema version 1")
        host = payload.get("host")
        results = payload.get("results")
        if not isinstance(host, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in host.items()
        ):
            raise ValueError(f"{path} has invalid host metadata")
        if not isinstance(results, list):
            raise TypeError(f"{path} has invalid benchmark results")
        normalized_host = dict(sorted(host.items()))
        hosts.append(normalized_host)
        report_count += 1
        for raw_result in results:
            if not isinstance(raw_result, dict):
                raise TypeError(f"{path} contains a malformed benchmark result")
            observation = _observation_from_result(raw_result)
            if observation is not None:
                observations.append(observation)

    if report_count == 0:
        raise ValueError("at least one benchmark report is required")
    first_host = hosts[0]
    if any(host != first_host for host in hosts[1:]):
        raise ValueError("cost calibration cannot mix benchmark reports from different hosts")
    return tuple(observations), first_host, report_count


def calibrate_files(
    paths: Iterable[Path],
    *,
    max_holdout_factor: float = 2.0,
    max_holdout_median_factor: float = 1.5,
) -> CostModelReport:
    observations, host, report_count = load_observations(paths)
    return fit_cost_models(
        observations,
        host=host,
        source_report_count=report_count,
        max_holdout_factor=max_holdout_factor,
        max_holdout_median_factor=max_holdout_median_factor,
    )
