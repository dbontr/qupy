from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

import qupy as qp


def _write_v3(
    path: Path,
    *,
    policy_version: int = 1,
    decision: str = "29 0 1.0",
) -> None:
    lines = (
        "qupy-planner-cost 3",
        f"engine {qp.core_version()}",
        "workload 1",
        f"host {qp.planner_host_fingerprint()}",
        "validated 1",
        "model pauli-propagation 2 5 0.85 1.1 1.2",
        "model statevector-parallel 3 4.5 0.55 0.012 1.1 1.2",
        "model statevector-serial 3 4.5 0.55 0.012 1.1 1.2",
        f"policy adaptive-mps {policy_version}",
        f"decision observable-auto {decision}",
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _chain(qubits: int) -> qp.Program:
    program = qp.ry(qp.Program(qubits), 0.371, 0)
    for qubit in range(qubits - 1):
        program = qp.cx(program, qubit, qubit + 1)
    return program


def test_schema_v3_accepts_host_bound_mps_policy_without_cuda(tmp_path: Path) -> None:
    artifact = tmp_path / "mps-policy.qpcost"
    _write_v3(artifact)
    model = qp.load_planner_cost_model(str(artifact))

    assert model.schema_version == 3
    assert model.mps_auto_validated
    assert model.mps_policy_version == 1
    assert not model.cuda_auto_validated
    assert model.cuda_host_fingerprint == ""


def test_validated_mps_policy_selects_and_executes_exactly(tmp_path: Path) -> None:
    artifact = tmp_path / "mps-policy.qpcost"
    _write_v3(artifact)
    model = qp.load_planner_cost_model(str(artifact))
    program = _chain(15)
    observable = qp.Z(14)

    default_plan = qp.expectation_plan(program, observable)
    selected_plan = qp.expectation_plan(program, observable, cost_model=model)
    assert default_plan.backend == "native-cpu"
    assert selected_plan.backend == "native-adaptive-mps"
    assert selected_plan.method == "mps"
    assert selected_plan.cost_model_class == "adaptive-mps-policy"
    assert selected_plan.cost_model_fingerprint == model.artifact_fingerprint
    assert selected_plan.predicted_ns is None

    cpu = qp.expect(program, observable, backend="native-cpu")
    selected = qp.expect(program, observable, cost_model=model)
    assert selected.backend == "native-adaptive-mps"
    assert selected.value == pytest.approx(cpu.value, abs=5e-12)

    cpu_variance = qp.variance(program, observable, backend="native-cpu")
    selected_variance = qp.variance(program, observable, cost_model=model)
    assert selected_variance.backend == "native-adaptive-mps"
    assert selected_variance.value == pytest.approx(cpu_variance.value, abs=5e-12)


def test_schema_v3_rejects_unvalidated_mps_policy(tmp_path: Path) -> None:
    cases = (
        (2, "29 0 1.0", "policy metadata"),
        (1, "15 0 1.0", "MPS decision evidence"),
        (1, "29 1 1.0", "MPS decision evidence"),
        (1, "29 0 1.11", "MPS decision evidence"),
    )
    for index, (policy_version, decision, message) in enumerate(cases):
        artifact = tmp_path / f"invalid-{index}.qpcost"
        _write_v3(artifact, policy_version=policy_version, decision=decision)
        with pytest.raises(ValueError, match=message):
            qp.load_planner_cost_model(str(artifact))


def test_adaptive_mps_target_fails_closed_outside_observables() -> None:
    target = qp.adaptive_mps_target()
    assert target.name == "native-adaptive-mps"
    assert target.supports_result(qp.ResultMode.EXPECTATION)
    assert target.supports_result(qp.ResultMode.VARIANCE)
    assert not target.supports_result(qp.ResultMode.STATEVECTOR)
    assert not target.supports_result(qp.ResultMode.SAMPLE)
    assert not target.parameter_batches

    program = _chain(4)
    with pytest.raises(ValueError, match="result mode"):
        qp.statevector(program, backend="native-adaptive-mps")
    with pytest.raises(ValueError, match="result mode"):
        qp.sample(program, shots=8, seed=7, backend="native-adaptive-mps")

    slots = [qp.ParameterSlot(0)]
    values = np.array([[0.2]], dtype=np.float64)
    parameterized = qp.ry(qp.Program(1), 0.1, 0)
    with pytest.raises(ValueError, match="parameter batches"):
        qp.expect_batch(
            parameterized,
            qp.Z(0),
            slots,
            values,
            backend="native-adaptive-mps",
        )


def test_adaptive_mps_large_low_bond_plan_does_not_require_dense_address_space() -> None:
    qubits = 128
    program = _chain(qubits)
    observable = qp.Z(qubits - 1)

    execution_plan = qp.expectation_plan(
        program,
        observable,
        backend="native-adaptive-mps",
    )
    assert execution_plan.method == "mps"
    assert execution_plan.tensor_network_max_bond == 2
    assert execution_plan.estimated_state_bytes < 1 << 20

    result = qp.expect(program, observable, backend="native-adaptive-mps")
    assert result.value == pytest.approx(math.cos(0.371), abs=5e-12)


def test_adaptive_mps_checkpoint_fallback_matches_dense_cpu() -> None:
    qubits = 15
    program = qp.Program(qubits)
    for qubit in range(qubits):
        program = qp.ry(program, 0.019 * (qubit + 1), qubit)
    for layer in range(7):
        parity = layer % 2
        for qubit in range(parity, qubits - 1, 2):
            program = qp.cz(program, qubit, qubit + 1)
        for qubit in range(qubits):
            program = qp.rz(program, 0.007 * (layer + qubit + 1), qubit)
    for qubit in range(qubits - 1):
        program = qp.cx(program, qubit, qubit + 1)

    observable = qp.Z(qubits - 1)
    execution_plan = qp.expectation_plan(
        program, observable, backend="native-adaptive-mps"
    )
    assert execution_plan.method == "adaptive-mps"
    assert execution_plan.tensor_network_max_bond > 16

    adaptive = qp.expect(program, observable, backend="native-adaptive-mps")
    dense = qp.expect(program, observable, backend="native-cpu")
    assert adaptive.value == pytest.approx(dense.value, abs=5e-12)
