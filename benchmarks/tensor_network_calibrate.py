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
_MIN_BACKEND_WINS = 3
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
    observable_count: int
    term_count: int
    operation_count: int
    two_qubit_operations: int
    compiled_steps: int
    threads: int
    cpu_active_qubits: int
    cpu_estimated_state_bytes: int
    tn_contractions: int
    tn_peak_tensor_rank: int
    tn_peak_tensor_bytes: int
    tn_scalar_multiplications: float
    median_ns: float

    @property
    def cost_class(self) -> str:
        if self.backend == "native-cpu":
            return "tensor-network-baseline-cpu"
        if self.backend == "native-tn":
            return "tensor-network-return-cpu"
        raise ValueError(f"unsupported tensor-network calibration backend: {self.backend}")


@dataclass(frozen=True, slots=True)
class CostModel:
    cost_class: str
    feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    training_fingerprints: tuple[str, ...]
    training_error: ErrorMetrics
    holdout_error: ErrorMetrics
    validated: bool

    def predict_ns(self, row: Observation) -> float:
        if row.cost_class != self.cost_class:
            raise ValueError(f"cost model {self.cost_class} cannot predict {row.cost_class}")
        score = sum(
            coefficient * feature
            for coefficient, feature in zip(
                self.coefficients,
                _features(row),
                strict=True,
            )
        )
        prediction = math.exp(score)
        if not math.isfinite(prediction) or prediction <= 0.0:
            raise OverflowError("tensor-network cost model produced an invalid prediction")
        return prediction


@dataclass(frozen=True, slots=True)
class DecisionEvidence:
    samples: int
    mistakes: int
    max_regret: float
    cpu_wins: int
    tn_wins: int
    rows: tuple[dict[str, Any], ...]

    @property
    def validated(self) -> bool:
        return (
            self.samples >= _MIN_DECISION_SAMPLES
            and self.mistakes == 0
            and self.max_regret <= _MAX_DECISION_REGRET
            and self.cpu_wins >= _MIN_BACKEND_WINS
            and self.tn_wins >= _MIN_BACKEND_WINS
        )


@dataclass(frozen=True, slots=True)
class TensorNetworkCalibration:
    engine_version: str
    workload_version: int
    host: dict[str, str]
    report_count: int
    models: tuple[CostModel, CostModel]
    decision: DecisionEvidence

    @property
    def validated(self) -> bool:
        return (
            self.report_count >= _MIN_REPORTS
            and all(model.validated for model in self.models)
            and self.decision.validated
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "policy_version": _POLICY_VERSION,
            "engine_version": self.engine_version,
            "workload_version": self.workload_version,
            "host": dict(self.host),
            "report_count": self.report_count,
            "validated": self.validated,
            "models": [asdict(model) for model in self.models],
            "decision": asdict(self.decision) | {"validated": self.decision.validated},
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def _feature_names(cost_class: str) -> tuple[str, ...]:
    if cost_class == "tensor-network-baseline-cpu":
        return (
            "bias",
            "active_qubits",
            "log1p_compiled_steps",
            "two_qubit_fraction",
            "log1p_term_count",
            "log_threads",
        )
    if cost_class == "tensor-network-return-cpu":
        return (
            "bias",
            "log1p_scalar_multiplications",
            "log1p_contractions",
            "peak_tensor_rank",
            "log1p_peak_tensor_bytes",
            "log1p_term_count",
        )
    raise ValueError(f"unsupported tensor-network cost class: {cost_class}")


def _features(row: Observation) -> tuple[float, ...]:
    if row.cost_class == "tensor-network-baseline-cpu":
        operations = max(row.operation_count, 1)
        return (
            1.0,
            float(row.cpu_active_qubits),
            math.log1p(row.compiled_steps),
            row.two_qubit_operations / operations,
            math.log1p(row.term_count),
            math.log(max(row.threads, 1)),
        )
    if row.cost_class == "tensor-network-return-cpu":
        return (
            1.0,
            math.log1p(row.tn_scalar_multiplications),
            math.log1p(row.tn_contractions),
            float(row.tn_peak_tensor_rank),
            math.log1p(row.tn_peak_tensor_bytes),
            math.log1p(row.term_count),
        )
    raise ValueError(f"unsupported tensor-network cost class: {row.cost_class}")


def _error_metrics(actual: list[float], predicted: list[float]) -> ErrorMetrics:
    if not actual or len(actual) != len(predicted):
        raise ValueError("tensor-network error metrics require matching non-empty samples")
    factors = [max(a / p, p / a) for a, p in zip(actual, predicted, strict=True)]
    percentages = [abs(a - p) / a for a, p in zip(actual, predicted, strict=True)]
    ordered = sorted(factors)
    p90_index = min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)
    return ErrorMetrics(
        samples=len(actual),
        median_factor=float(statistics.median(ordered)),
        p90_factor=float(ordered[p90_index]),
        max_factor=float(max(ordered)),
        mean_absolute_percentage_error=float(statistics.fmean(percentages)),
    )


def _fit_coefficients(rows: list[Observation]) -> tuple[float, ...]:
    if not rows:
        raise ValueError("tensor-network cost model requires observations")
    cost_classes = {row.cost_class for row in rows}
    if len(cost_classes) != 1:
        raise ValueError("tensor-network cost fit cannot mix cost classes")
    feature_names = _feature_names(rows[0].cost_class)
    if len(rows) < len(feature_names):
        raise ValueError("tensor-network cost model has too little training data")
    design = np.asarray([_features(row) for row in rows], dtype=np.float64)
    targets = np.log(np.asarray([row.median_ns for row in rows], dtype=np.float64))
    coefficients, _, rank, _ = np.linalg.lstsq(design, targets, rcond=None)
    if rank < len(feature_names):
        raise ValueError(f"tensor-network cost class {rows[0].cost_class} is rank deficient")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("tensor-network cost calibration produced invalid coefficients")
    return tuple(float(value) for value in coefficients)


def _predict(coefficients: tuple[float, ...], row: Observation) -> float:
    score = sum(
        coefficient * feature
        for coefficient, feature in zip(coefficients, _features(row), strict=True)
    )
    prediction = math.exp(score)
    if not math.isfinite(prediction) or prediction <= 0.0:
        raise OverflowError("tensor-network cost calibration produced an invalid prediction")
    return prediction


def _positive_samples(value: object, *, field: str, iterations: int) -> list[int]:
    if not isinstance(value, list) or len(value) != iterations:
        raise ValueError(f"tensor-network policy {field} count does not match iterations")
    if not all(isinstance(item, int) and item > 0 for item in value):
        raise ValueError(f"tensor-network policy {field} contains an invalid timing")
    return value


def _nonnegative_int(row: dict[str, Any], key: str) -> int:
    value = row.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"tensor-network policy {key} must be a non-negative integer")
    return value


def _load_report(
    path: Path,
) -> tuple[dict[str, str], list[Observation], dict[str, tuple[str, str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 1
        or payload.get("policy_version") != _POLICY_VERSION
        or payload.get("profile") != "policy"
    ):
        raise ValueError("tensor-network calibration requires schema-v1 policy reports")
    if payload.get("engine_version") != qp.core_version() or payload.get("workload_version") != 1:
        raise ValueError("tensor-network policy report does not match this QuPy runtime")
    iterations = payload.get("iterations")
    if not isinstance(iterations, int) or iterations < 2 or iterations % 2 != 0:
        raise ValueError("tensor-network policy report requires a positive even iteration count")

    host = payload.get("host")
    if not isinstance(host, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in host.items()
    ):
        raise TypeError("tensor-network policy report is missing host metadata")
    if len(host.get("planner_host_fingerprint", "")) != 64:
        raise ValueError("tensor-network policy host fingerprint is invalid")

    validations = payload.get("validations")
    evidence = payload.get("policy_evidence")
    if not isinstance(validations, list) or not isinstance(evidence, list) or not evidence:
        raise ValueError("tensor-network policy report is missing validation or timing evidence")

    validated_names: set[str] = set()
    for validation in validations:
        if not isinstance(validation, dict) or not isinstance(validation.get("workload"), str):
            raise TypeError("tensor-network policy validation row is invalid")
        error = validation.get("max_abs_error")
        if (
            not isinstance(error, (int, float))
            or isinstance(error, bool)
            or error < 0
            or error > _MAX_SEMANTIC_ERROR
        ):
            raise ValueError("tensor-network policy report failed exactness validation")
        workload = validation["workload"]
        if workload in validated_names:
            raise ValueError("tensor-network policy validation workload is duplicated")
        validated_names.add(workload)

    observations: list[Observation] = []
    identities: dict[str, tuple[str, str]] = {}
    seen_fingerprints: set[str] = set()
    for row in evidence:
        if not isinstance(row, dict):
            raise TypeError("tensor-network policy evidence row is invalid")
        workload = row.get("workload")
        fingerprint = row.get("fingerprint")
        plan_fingerprint = row.get("tn_plan_fingerprint")
        if (
            not isinstance(workload, str)
            or not workload
            or workload in identities
            or not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or fingerprint in seen_fingerprints
            or not isinstance(plan_fingerprint, str)
            or len(plan_fingerprint) != 64
        ):
            raise ValueError("tensor-network policy workload identity is invalid or duplicated")
        identities[workload] = (fingerprint, plan_fingerprint)
        seen_fingerprints.add(fingerprint)

        scalar_multiplications = row.get("tn_scalar_multiplications")
        if (
            not isinstance(scalar_multiplications, (int, float))
            or isinstance(scalar_multiplications, bool)
            or scalar_multiplications < 0
            or not math.isfinite(float(scalar_multiplications))
        ):
            raise ValueError("tensor-network scalar multiplication count is invalid")

        common = {
            "qubits": _nonnegative_int(row, "qubits"),
            "observable_count": _nonnegative_int(row, "observable_count"),
            "term_count": _nonnegative_int(row, "term_count"),
            "operation_count": _nonnegative_int(row, "operation_count"),
            "two_qubit_operations": _nonnegative_int(row, "two_qubit_operations"),
            "compiled_steps": _nonnegative_int(row, "compiled_steps"),
            "threads": _nonnegative_int(row, "threads"),
            "cpu_active_qubits": _nonnegative_int(row, "cpu_active_qubits"),
            "cpu_estimated_state_bytes": _nonnegative_int(row, "cpu_estimated_state_bytes"),
            "tn_contractions": _nonnegative_int(row, "tn_contractions"),
            "tn_peak_tensor_rank": _nonnegative_int(row, "tn_peak_tensor_rank"),
            "tn_peak_tensor_bytes": _nonnegative_int(row, "tn_peak_tensor_bytes"),
        }
        if common["observable_count"] == 0 or common["term_count"] == 0:
            raise ValueError("tensor-network policy query work must be non-zero")
        if common["threads"] == 0:
            raise ValueError("tensor-network policy CPU thread count must be positive")
        if common["two_qubit_operations"] > common["operation_count"]:
            raise ValueError("tensor-network policy two-qubit operation count is invalid")

        for backend, timing_key in (
            ("native-cpu", "cpu_timings_ns"),
            ("native-tn", "tn_timings_ns"),
        ):
            samples = _positive_samples(row.get(timing_key), field=timing_key, iterations=iterations)
            observations.append(
                Observation(
                    fingerprint=fingerprint,
                    workload=workload,
                    backend=backend,
                    qubits=common["qubits"],
                    observable_count=common["observable_count"],
                    term_count=common["term_count"],
                    operation_count=common["operation_count"],
                    two_qubit_operations=common["two_qubit_operations"],
                    compiled_steps=common["compiled_steps"],
                    threads=common["threads"],
                    cpu_active_qubits=common["cpu_active_qubits"],
                    cpu_estimated_state_bytes=common["cpu_estimated_state_bytes"],
                    tn_contractions=common["tn_contractions"],
                    tn_peak_tensor_rank=common["tn_peak_tensor_rank"],
                    tn_peak_tensor_bytes=common["tn_peak_tensor_bytes"],
                    tn_scalar_multiplications=float(scalar_multiplications),
                    median_ns=float(statistics.median(samples)),
                )
            )

    if validated_names != set(identities):
        raise ValueError("tensor-network exactness validation does not cover the decision workload set")
    return dict(host), observations, identities


def _aggregate(rows: list[Observation]) -> list[Observation]:
    grouped: dict[tuple[str, str], list[Observation]] = defaultdict(list)
    for row in rows:
        grouped[(row.fingerprint, row.backend)].append(row)

    result: list[Observation] = []
    for group in grouped.values():
        first = group[0]
        shape = {
            (
                row.workload,
                row.qubits,
                row.observable_count,
                row.term_count,
                row.operation_count,
                row.two_qubit_operations,
                row.compiled_steps,
                row.threads,
                row.cpu_active_qubits,
                row.cpu_estimated_state_bytes,
                row.tn_contractions,
                row.tn_peak_tensor_rank,
                row.tn_peak_tensor_bytes,
                row.tn_scalar_multiplications,
            )
            for row in group
        }
        if len(shape) != 1:
            raise ValueError("tensor-network workload structure changed between reports")
        result.append(
            Observation(
                fingerprint=first.fingerprint,
                workload=first.workload,
                backend=first.backend,
                qubits=first.qubits,
                observable_count=first.observable_count,
                term_count=first.term_count,
                operation_count=first.operation_count,
                two_qubit_operations=first.two_qubit_operations,
                compiled_steps=first.compiled_steps,
                threads=first.threads,
                cpu_active_qubits=first.cpu_active_qubits,
                cpu_estimated_state_bytes=first.cpu_estimated_state_bytes,
                tn_contractions=first.tn_contractions,
                tn_peak_tensor_rank=first.tn_peak_tensor_rank,
                tn_peak_tensor_bytes=first.tn_peak_tensor_bytes,
                tn_scalar_multiplications=first.tn_scalar_multiplications,
                median_ns=float(statistics.median(row.median_ns for row in group)),
            )
        )
    return sorted(result, key=lambda row: (row.workload, row.backend))


def _fit_model(rows: list[Observation], holdout_predictions: dict[str, float]) -> CostModel:
    if not rows:
        raise ValueError("tensor-network model requires observations")
    cost_class = rows[0].cost_class
    if any(row.cost_class != cost_class for row in rows):
        raise ValueError("tensor-network model cannot mix cost classes")
    coefficients = _fit_coefficients(rows)
    training_predictions = [_predict(coefficients, row) for row in rows]
    training_error = _error_metrics([row.median_ns for row in rows], training_predictions)
    holdout_actual = [row.median_ns for row in rows]
    holdout_predicted = [holdout_predictions[row.fingerprint] for row in rows]
    holdout_error = _error_metrics(holdout_actual, holdout_predicted)
    validated = (
        holdout_error.median_factor <= _MAX_MEDIAN_FACTOR
        and holdout_error.max_factor <= _MAX_FACTOR
    )
    return CostModel(
        cost_class=cost_class,
        feature_names=_feature_names(cost_class),
        coefficients=coefficients,
        training_fingerprints=tuple(sorted(row.fingerprint for row in rows)),
        training_error=training_error,
        holdout_error=holdout_error,
        validated=validated,
    )


def _leave_one_out(
    rows: list[Observation],
) -> tuple[dict[tuple[str, str], float], DecisionEvidence]:
    by_fingerprint: dict[str, dict[str, Observation]] = defaultdict(dict)
    for row in rows:
        if row.backend in by_fingerprint[row.fingerprint]:
            raise ValueError("tensor-network calibration has duplicate backend evidence")
        by_fingerprint[row.fingerprint][row.backend] = row
    if len(by_fingerprint) < _MIN_DECISION_SAMPLES:
        raise ValueError(
            f"tensor-network calibration requires at least {_MIN_DECISION_SAMPLES} paired workloads"
        )
    if any(set(pair) != {"native-cpu", "native-tn"} for pair in by_fingerprint.values()):
        raise ValueError("tensor-network calibration requires paired CPU and TN observations")

    predictions: dict[tuple[str, str], float] = {}
    decision_rows: list[dict[str, Any]] = []
    mistakes = 0
    max_regret = 1.0
    cpu_wins = 0
    tn_wins = 0

    for fingerprint in sorted(by_fingerprint):
        held_out = by_fingerprint[fingerprint]
        for backend in ("native-cpu", "native-tn"):
            training = [
                row
                for row in rows
                if row.backend == backend and row.fingerprint != fingerprint
            ]
            coefficients = _fit_coefficients(training)
            predictions[(fingerprint, backend)] = _predict(coefficients, held_out[backend])

        predicted_cpu = predictions[(fingerprint, "native-cpu")]
        predicted_tn = predictions[(fingerprint, "native-tn")]
        actual_cpu = held_out["native-cpu"].median_ns
        actual_tn = held_out["native-tn"].median_ns
        selected = "native-cpu" if predicted_cpu <= predicted_tn else "native-tn"
        optimal = "native-cpu" if actual_cpu <= actual_tn else "native-tn"
        if optimal == "native-cpu":
            cpu_wins += 1
        else:
            tn_wins += 1
        selected_actual = actual_cpu if selected == "native-cpu" else actual_tn
        optimal_actual = min(actual_cpu, actual_tn)
        regret = selected_actual / optimal_actual
        if selected != optimal:
            mistakes += 1
        max_regret = max(max_regret, regret)
        decision_rows.append(
            {
                "workload": held_out["native-cpu"].workload,
                "fingerprint": fingerprint,
                "selected_backend": selected,
                "optimal_backend": optimal,
                "regret": regret,
                "actual_cpu_ns": actual_cpu,
                "actual_tn_ns": actual_tn,
                "predicted_cpu_ns": predicted_cpu,
                "predicted_tn_ns": predicted_tn,
            }
        )

    return predictions, DecisionEvidence(
        samples=len(decision_rows),
        mistakes=mistakes,
        max_regret=float(max_regret),
        cpu_wins=cpu_wins,
        tn_wins=tn_wins,
        rows=tuple(decision_rows),
    )


def calibrate_files(paths: tuple[Path, ...]) -> TensorNetworkCalibration:
    if len(paths) < _MIN_REPORTS:
        raise ValueError(f"tensor-network calibration requires at least {_MIN_REPORTS} reports")

    all_rows: list[Observation] = []
    hosts: list[dict[str, str]] = []
    identities: list[dict[str, tuple[str, str]]] = []
    for path in paths:
        host, rows, report_identities = _load_report(path)
        hosts.append(host)
        all_rows.extend(rows)
        identities.append(report_identities)

    if any(host != hosts[0] for host in hosts[1:]):
        raise ValueError("tensor-network calibration cannot mix hosts")
    if any(value != identities[0] for value in identities[1:]):
        raise ValueError("tensor-network workload identities changed between reports")

    aggregated = _aggregate(all_rows)
    predictions, decision = _leave_one_out(aggregated)
    models: list[CostModel] = []
    for backend in ("native-cpu", "native-tn"):
        rows = [row for row in aggregated if row.backend == backend]
        holdout = {row.fingerprint: predictions[(row.fingerprint, backend)] for row in rows}
        models.append(_fit_model(rows, holdout))

    return TensorNetworkCalibration(
        engine_version=qp.core_version(),
        workload_version=1,
        host=hosts[0],
        report_count=len(paths),
        models=(models[0], models[1]),
        decision=decision,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate held-out dense-CPU and tensor-network routing evidence"
    )
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    calibration = calibrate_files(tuple(args.reports))
    if not calibration.validated:
        raise SystemExit("tensor-network routing evidence did not meet promotion thresholds")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(calibration.to_json(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
