from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import qupy as qp
from benchmarks.statevector_calibrate import (
    ReturnObservation,
    fit_return_costs,
    load_observations,
)


def _threads(qubits: int) -> int:
    if qubits < 16:
        return 1
    return min(16, 1 << min(qubits - 16, 4))


def _observation(index: int, qubits: int, backend: str) -> ReturnObservation:
    single = qubits + 1 if index % 2 == 0 else 2 * qubits
    two = qubits - 1 + qubits // 2 if index % 2 == 0 else qubits - 1
    compiled_steps = single + two
    threads = _threads(qubits) if backend == "native-cpu" else 1
    two_fraction = two / (single + two)
    if backend == "native-cpu":
        coefficients = (4.0, 0.65, 0.4, 0.4, -0.7)
        features = (1.0, float(qubits), math.log(compiled_steps), two_fraction, math.log(threads))
    else:
        coefficients = (8.0, 0.35, 0.4, 0.2)
        features = (1.0, float(qubits), math.log(compiled_steps), two_fraction)
    runtime = math.exp(sum(a * b for a, b in zip(coefficients, features, strict=True)))
    observation = ReturnObservation(
        fingerprint=f"{index:064x}",
        workload=f"paired-{qubits}-{index % 2}",
        backend=backend,
        qubits=qubits,
        single_qubit_operations=single,
        two_qubit_operations=two,
        compiled_steps=compiled_steps,
        threads=threads,
        median_ns=runtime,
    )
    return ReturnObservation(
        fingerprint=observation.fingerprint,
        workload=observation.workload,
        backend=observation.backend,
        qubits=observation.qubits,
        single_qubit_operations=observation.single_qubit_operations,
        two_qubit_operations=observation.two_qubit_operations,
        compiled_steps=observation.compiled_steps,
        threads=observation.threads,
        median_ns=runtime,
    )


def _paired_observations() -> list[ReturnObservation]:
    rows: list[ReturnObservation] = []
    for index, qubits in enumerate(range(10, 28)):
        rows.append(_observation(index, qubits, "native-cpu"))
        rows.append(_observation(index, qubits, "native-cuda"))
    return rows


def _host() -> dict[str, str]:
    return {
        "planner_host_fingerprint": qp.planner_host_fingerprint(),
        "planner_cuda_host_fingerprint": "a" * 64,
    }


def test_return_cost_calibration_recovers_paired_curves() -> None:
    calibration = fit_return_costs(
        _paired_observations(),
        host=_host(),
        engine_version=qp.core_version(),
        workload_version=1,
        report_count=3,
    )

    assert calibration.validated
    assert calibration.report_count == 3
    assert calibration.decision.samples >= 8
    assert calibration.decision.mistakes == 0
    assert calibration.decision.max_regret == pytest.approx(1.0, abs=1e-9)
    assert {model.cost_class for model in calibration.models} == {
        "statevector-return-cpu",
        "statevector-return-cuda",
    }
    for model in calibration.models:
        assert model.validated
        assert model.holdout_error.max_factor == pytest.approx(1.0, abs=1e-8)
        assert set(model.training_fingerprints).isdisjoint(model.holdout_fingerprints)
    assert calibration.models[0].holdout_fingerprints == calibration.models[1].holdout_fingerprints


def test_return_cost_calibration_rejects_material_backend_regret() -> None:
    observations = _paired_observations()
    baseline = fit_return_costs(
        observations,
        host=_host(),
        engine_version=qp.core_version(),
        workload_version=1,
        report_count=1,
    )
    row = baseline.decision.rows[0]
    fingerprint = str(row["fingerprint"])
    selected = str(row["selected_backend"])
    degraded = []
    for observation in observations:
        median_ns = observation.median_ns
        if observation.fingerprint == fingerprint and observation.backend == selected:
            alternate = "cuda_actual_ns" if selected == "native-cpu" else "cpu_actual_ns"
            median_ns = float(row[alternate]) * 2.0
        degraded.append(
            ReturnObservation(
                fingerprint=observation.fingerprint,
                workload=observation.workload,
                backend=observation.backend,
                qubits=observation.qubits,
                single_qubit_operations=observation.single_qubit_operations,
                two_qubit_operations=observation.two_qubit_operations,
                compiled_steps=observation.compiled_steps,
                threads=observation.threads,
                median_ns=median_ns,
            )
        )
    calibration = fit_return_costs(
        degraded,
        host=_host(),
        engine_version=qp.core_version(),
        workload_version=1,
        report_count=1,
    )
    assert not calibration.decision.validated
    assert calibration.decision.mistakes >= 1
    assert calibration.decision.max_regret > 1.10


def test_return_cost_loader_requires_paired_backends(tmp_path: Path) -> None:
    observation = _observation(1, 18, "native-cpu")
    report = {
        "schema_version": 1,
        "engine_version": qp.core_version(),
        "workload_version": 1,
        "host": _host(),
        "results": [
            {
                "workload_fingerprint": observation.fingerprint,
                "workload": observation.workload,
                "backend": observation.backend,
                "qubits": observation.qubits,
                "single_qubit_operations": observation.single_qubit_operations,
                "two_qubit_operations": observation.two_qubit_operations,
                "compiled_steps": observation.compiled_steps,
                "threads": observation.threads,
                "median_ns": observation.median_ns,
            }
        ],
    }
    path = tmp_path / "unpaired.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="paired CPU and CUDA"):
        load_observations((path,))
