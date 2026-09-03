from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import qupy as qp
from benchmarks.observable_calibrate import calibrate_files


def _features(row: dict[str, int], *, cuda: bool) -> tuple[float, ...]:
    operations = row["single_qubit_operations"] + row["two_qubit_operations"]
    qubits = float(row["qubits"])
    if cuda:
        dimension = float(1 << row["qubits"])
        compiled = float(row["compiled_steps"])
        two_qubit = float(row["two_qubit_operations"])
        single_qubit = max(compiled - two_qubit, 0.0)
        terms = float(row["cuda_term_evaluations"])
        count = max(1, (1 << row["qubits"]) // 256)
        kernels = 1
        while count > 1:
            count = (count + 255) // 256
            kernels += 1
        return (
            1.0, dimension, dimension * (0.5 * single_qubit + two_qubit), compiled,
            dimension * terms, terms * kernels, terms,
        )
    log_steps = math.log(row["compiled_steps"])
    log_terms = math.log1p(row["cpu_term_evaluations"])
    return (
        1.0, qubits, log_steps, row["two_qubit_operations"] / operations,
        log_terms, qubits * log_terms, qubits * qubits,
        log_terms * log_terms, log_steps * log_steps,
        float(row["cpu_state_passes"]), math.log(row["threads"]),
    )


def _runtime(
    features: tuple[float, ...], coefficients: tuple[float, ...], *, additive: bool = False
) -> int:
    score = sum(a * b for a, b in zip(features, coefficients, strict=True))
    return max(1, round(score if additive else math.exp(score)))


def _report(path: Path, *, claimed_scale: int = 1) -> None:
    cpu_coefficients = (
        8.0, 0.10, 0.25, 0.30, 0.12, 0.004, 0.002, 0.005, 0.01, 0.08, -0.10
    )
    cuda_coefficients = (50000.0, 0.5, 0.02, 2000.0, 0.03, 5000.0, 10000.0)
    evidence: list[dict[str, object]] = []
    validations: list[dict[str, object]] = []
    for index in range(20):
        row: dict[str, int] = {
            "qubits": 9 + index % 10,
            "single_qubit_operations": 11 + index * 2 + index % 3,
            "two_qubit_operations": 2 + (index * 5) % 13,
            "compiled_steps": 17 + index * 3 + (index % 4) * 2,
            "threads": (1, 2, 4, 8)[index % 4],
            "cpu_term_evaluations": 1 + (index * 7) % 31,
            "cpu_state_passes": index % 2,
            "cuda_term_evaluations": 1 + (index * 11) % 37,
            "cuda_state_passes": 0,
        }
        cpu = _runtime(_features(row, cuda=False), cpu_coefficients)
        cuda = _runtime(_features(row, cuda=True), cuda_coefficients, additive=True)
        workload = f"synthetic-observable-{index:02d}"
        evidence.append({
            "workload": workload,
            "fingerprint": f"{index + 1:064x}",
            **row,
            "cpu_timings_ns": [cpu - 1, cpu + 1, cpu, cpu],
            "cuda_timings_ns": [cuda + 1, cuda - 1, cuda, cuda],
            "cpu_median_ns": cpu * claimed_scale,
            "cuda_median_ns": cuda * claimed_scale,
        })
        validations.append({"workload": workload, "max_abs_error": 0.0})
    payload = {
        "schema_version": 1,
        "policy_version": 1,
        "engine_version": qp.core_version(),
        "workload_version": 1,
        "host": {
            "planner_host_fingerprint": "1" * 64,
            "planner_cuda_host_fingerprint": "2" * 64,
        },
        "profile": "policy",
        "warmups": 1,
        "iterations": 4,
        "validations": validations,
        "policy_evidence": evidence,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _reports(tmp_path: Path, *, claimed_scale: int = 1) -> tuple[Path, ...]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = tuple(tmp_path / f"observable-{index}.json" for index in range(3))
    for path in paths:
        _report(path, claimed_scale=claimed_scale)
    return paths


def test_observable_calibration_uses_raw_samples_and_validates(tmp_path: Path) -> None:
    baseline = calibrate_files(_reports(tmp_path / "baseline"))
    claimed = calibrate_files(_reports(tmp_path / "claimed", claimed_scale=1000))

    assert baseline.validated
    assert baseline.decision.samples == 20
    assert baseline.decision.mistakes == 0
    assert baseline.decision.max_regret <= 1.10
    assert baseline.models == claimed.models
    assert baseline.decision == claimed.decision


def test_observable_calibration_requires_three_reports(tmp_path: Path) -> None:
    reports = _reports(tmp_path)
    with pytest.raises(ValueError, match="at least 3 reports"):
        calibrate_files(reports[:2])


def test_observable_calibration_rejects_incomplete_exactness(tmp_path: Path) -> None:
    reports = list(_reports(tmp_path))
    payload = json.loads(reports[0].read_text(encoding="utf-8"))
    payload["validations"].pop()
    reports[0].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="does not cover"):
        calibrate_files(tuple(reports))
