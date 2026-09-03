from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import qupy as qp
from benchmarks.density_calibrate import calibrate_files


def _features(row: dict[str, int], *, cuda: bool) -> tuple[float, ...]:
    qubits = float(row["qubits"])
    operations = row["single_qubit_operations"] + row["two_qubit_operations"]
    if cuda:
        elements = float(4 ** row["qubits"])
        return (
            1.0,
            elements,
            elements * row["single_qubit_operations"],
            elements * row["two_qubit_operations"],
            elements * row["noise_events"],
            float(operations),
            float(row["noise_events"]),
        )
    log_kraus = math.log1p(row["kraus_evaluations"])
    return (
        1.0,
        qubits,
        qubits * qubits,
        math.log1p(operations),
        log_kraus,
        qubits * log_kraus,
    )


def _runtime(
    features: tuple[float, ...],
    coefficients: tuple[float, ...],
    *,
    additive: bool = False,
) -> int:
    score = sum(a * b for a, b in zip(features, coefficients, strict=True))
    return max(1, round(score if additive else math.exp(score)))


def _report(path: Path, *, claimed_scale: int = 1) -> None:
    cpu_coefficients = (7.2, 0.18, 0.012, 0.11, 0.16, 0.014)
    cuda_coefficients = (45000.0, 0.02, 0.012, 0.028, 0.015, 1800.0, 9000.0)
    evidence: list[dict[str, object]] = []
    validations: list[dict[str, object]] = []
    for index in range(30):
        row: dict[str, int] = {
            "qubits": 4 + index % 6,
            "single_qubit_operations": 5 + (index * 3) % 17,
            "two_qubit_operations": 2 + (index * 5) % 11,
            "noise_events": 1 + index % 5,
            "kraus_evaluations": 2 + (index * 7) % 19,
        }
        cpu = _runtime(_features(row, cuda=False), cpu_coefficients)
        cuda = _runtime(_features(row, cuda=True), cuda_coefficients, additive=True)
        workload = f"synthetic-density-{index:02d}"
        evidence.append(
            {
                "workload": workload,
                "fingerprint": f"{index + 1:064x}",
                **row,
                "cpu_timings_ns": [cpu - 1, cpu + 1, cpu, cpu],
                "cuda_timings_ns": [cuda + 1, cuda - 1, cuda, cuda],
                "cpu_median_ns": cpu * claimed_scale,
                "cuda_median_ns": cuda * claimed_scale,
            }
        )
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
    paths = tuple(tmp_path / f"density-{index}.json" for index in range(3))
    for path in paths:
        _report(path, claimed_scale=claimed_scale)
    return paths


def test_density_calibration_uses_raw_samples_and_validates(tmp_path: Path) -> None:
    baseline = calibrate_files(_reports(tmp_path / "baseline"))
    claimed = calibrate_files(_reports(tmp_path / "claimed", claimed_scale=1000))

    assert baseline.validated
    assert {model.cost_class for model in baseline.models} == {
        "density-return-cpu",
        "density-return-cuda",
        "density-speedup",
    }
    assert all(model.validated for model in baseline.models)
    assert baseline.decision.samples == 30
    assert baseline.decision.mistakes == 0
    assert baseline.decision.max_regret <= 1.10
    assert baseline.models == claimed.models
    assert baseline.decision == claimed.decision


def test_density_calibration_requires_three_reports(tmp_path: Path) -> None:
    reports = _reports(tmp_path)
    with pytest.raises(ValueError, match="at least 3 reports"):
        calibrate_files(reports[:2])


def test_density_calibration_rejects_incomplete_exactness(tmp_path: Path) -> None:
    reports = list(_reports(tmp_path))
    payload = json.loads(reports[0].read_text(encoding="utf-8"))
    payload["validations"].pop()
    reports[0].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="does not cover"):
        calibrate_files(tuple(reports))


def test_density_calibration_rejects_timing_count_mismatch(tmp_path: Path) -> None:
    reports = list(_reports(tmp_path))
    payload = json.loads(reports[0].read_text(encoding="utf-8"))
    payload["policy_evidence"][0]["cuda_timings_ns"].pop()
    reports[0].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="count does not match iterations"):
        calibrate_files(tuple(reports))


def test_density_calibration_rejects_changed_host(tmp_path: Path) -> None:
    reports = list(_reports(tmp_path))
    payload = json.loads(reports[-1].read_text(encoding="utf-8"))
    payload["host"]["machine"] = "other-host"
    reports[-1].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="different hosts"):
        calibrate_files(tuple(reports))
