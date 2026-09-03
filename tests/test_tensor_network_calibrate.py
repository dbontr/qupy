from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from benchmarks.tensor_network_calibrate import (
    Observation,
    _features,
    calibrate_files,
)
import qupy as qp


_CPU_COEFFICIENTS = (8.1, 0.12, 0.18, 0.25, 0.09, -0.06)
_TN_COEFFICIENTS = (7.0, 0.21, 0.11, 0.045, 0.075, 0.10)


def _runtime(row: Observation, coefficients: tuple[float, ...]) -> int:
    score = sum(
        coefficient * feature
        for coefficient, feature in zip(coefficients, _features(row), strict=True)
    )
    return max(1, round(math.exp(score)))


def _row(index: int, backend: str) -> Observation:
    qubits = 8 + index % 9
    term_count = 1 + (index * 3) % 8
    operation_count = qubits * 2 + 3 + (index * 5) % 11
    two_qubit_operations = qubits - 1 + index % 4
    compiled_steps = qubits + two_qubit_operations + 2 + index % 5
    complexity_phase = 1 if index < 9 else 4096
    return Observation(
        fingerprint=f"{index + 1:064x}",
        workload=f"synthetic-tensor-network-{index:02d}",
        backend=backend,
        qubits=qubits,
        observable_count=1 + index % 2,
        term_count=term_count,
        operation_count=operation_count,
        two_qubit_operations=two_qubit_operations,
        compiled_steps=compiled_steps,
        threads=(1, 2, 4, 8)[index % 4],
        cpu_active_qubits=qubits,
        cpu_estimated_state_bytes=(1 << qubits) * 16,
        tn_contractions=24 + index * 7 + (index % 3) * 5,
        tn_peak_tensor_rank=2 + (index * 5) % 7,
        tn_peak_tensor_bytes=32 << ((index * 3) % 8),
        tn_scalar_multiplications=float(
            (index + 2) ** 2 * (16 + 9 * (index % 4)) * complexity_phase
        ),
        median_ns=1.0,
    )


def _report(path: Path, *, claimed_scale: int = 1, host_suffix: str = "a") -> None:
    evidence: list[dict[str, object]] = []
    validations: list[dict[str, object]] = []
    for index in range(18):
        cpu_row = _row(index, "native-cpu")
        tn_row = _row(index, "native-tn")
        cpu = _runtime(cpu_row, _CPU_COEFFICIENTS)
        tn = _runtime(tn_row, _TN_COEFFICIENTS)
        evidence.append(
            {
                "workload": cpu_row.workload,
                "fingerprint": cpu_row.fingerprint,
                "query": "expectation" if index % 2 == 0 else "batch",
                "qubits": cpu_row.qubits,
                "variant": ("chain", "ring", "star", "ladder")[index % 4],
                "observable_count": cpu_row.observable_count,
                "term_count": cpu_row.term_count,
                "operation_count": cpu_row.operation_count,
                "two_qubit_operations": cpu_row.two_qubit_operations,
                "compiled_steps": cpu_row.compiled_steps,
                "threads": cpu_row.threads,
                "cpu_active_qubits": cpu_row.cpu_active_qubits,
                "cpu_estimated_state_bytes": cpu_row.cpu_estimated_state_bytes,
                "tn_contractions": cpu_row.tn_contractions,
                "tn_peak_tensor_rank": cpu_row.tn_peak_tensor_rank,
                "tn_peak_tensor_bytes": cpu_row.tn_peak_tensor_bytes,
                "tn_scalar_multiplications": cpu_row.tn_scalar_multiplications,
                "tn_plan_fingerprint": f"{1000 + index:064x}",
                "cpu_timings_ns": [cpu - 1, cpu + 1, cpu, cpu],
                "tn_timings_ns": [tn + 1, tn - 1, tn, tn],
                "cpu_median_ns": cpu * claimed_scale,
                "tn_median_ns": tn * claimed_scale,
            }
        )
        validations.append({"workload": cpu_row.workload, "max_abs_error": 0.0})

    payload = {
        "schema_version": 1,
        "policy_version": 1,
        "engine_version": qp.core_version(),
        "workload_version": 1,
        "host": {
            "platform": "synthetic",
            "machine": "synthetic",
            "python": "3.12",
            "planner_host_fingerprint": host_suffix * 64,
        },
        "profile": "policy",
        "warmups": 1,
        "iterations": 4,
        "validations": validations,
        "policy_evidence": evidence,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _reports(
    tmp_path: Path,
    *,
    claimed_scale: int = 1,
    host_suffix: str = "a",
) -> tuple[Path, ...]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = tuple(tmp_path / f"tensor-network-{index}.json" for index in range(3))
    for path in paths:
        _report(path, claimed_scale=claimed_scale, host_suffix=host_suffix)
    return paths


def test_tensor_network_calibration_uses_raw_samples_and_validates(tmp_path: Path) -> None:
    baseline = calibrate_files(_reports(tmp_path / "baseline"))
    claimed = calibrate_files(_reports(tmp_path / "claimed", claimed_scale=1000))

    assert baseline.validated
    assert baseline.report_count == 3
    assert baseline.decision.samples == 18
    assert baseline.decision.mistakes == 0
    assert baseline.decision.max_regret <= 1.10
    assert baseline.decision.cpu_wins == 9
    assert baseline.decision.tn_wins == 9
    assert all(model.validated for model in baseline.models)
    assert baseline.models == claimed.models
    assert baseline.decision == claimed.decision


def test_tensor_network_calibration_requires_three_reports(tmp_path: Path) -> None:
    reports = _reports(tmp_path)
    with pytest.raises(ValueError, match="at least 3 reports"):
        calibrate_files(reports[:2])


def test_tensor_network_calibration_rejects_incomplete_exactness(tmp_path: Path) -> None:
    reports = list(_reports(tmp_path))
    payload = json.loads(reports[0].read_text(encoding="utf-8"))
    payload["validations"].pop()
    reports[0].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="does not cover"):
        calibrate_files(tuple(reports))


def test_tensor_network_calibration_rejects_mixed_hosts(tmp_path: Path) -> None:
    reports = list(_reports(tmp_path / "first", host_suffix="a"))
    replacement = tmp_path / "other-host.json"
    _report(replacement, host_suffix="b")
    reports[-1] = replacement
    with pytest.raises(ValueError, match="cannot mix hosts"):
        calibrate_files(tuple(reports))


def test_tensor_network_calibration_rejects_cross_report_identity_drift(tmp_path: Path) -> None:
    reports = list(_reports(tmp_path))
    payload = json.loads(reports[-1].read_text(encoding="utf-8"))
    payload["policy_evidence"][0]["tn_plan_fingerprint"] = "f" * 64
    reports[-1].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="identities changed"):
        calibrate_files(tuple(reports))
