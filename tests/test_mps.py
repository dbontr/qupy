from __future__ import annotations

import math

import numpy as np
import pytest

import qupy as qp


def _mixed_program(qubits: int = 4) -> qp.Program:
    program = qp.Program(qubits)
    program = qp.ry(program, 0.371, 0)
    program = qp.cx(program, 0, qubits - 1)
    if qubits > 2:
        program = qp.cz(program, 1, qubits - 1)
        program = qp.swap(program, 0, 1)
    return program


def test_mps_target_and_plan_are_explicit() -> None:
    target = qp.mps_target()
    assert target.name == "native-mps"
    assert target.supports_result(qp.ResultMode.STATEVECTOR)
    assert target.supports_result(qp.ResultMode.EXPECTATION)
    assert target.supports_result(qp.ResultMode.VARIANCE)
    assert not target.supports_result(qp.ResultMode.SAMPLE)

    program = _mixed_program()
    execution_plan = qp.plan(program, qp.ResultMode.STATEVECTOR, backend="native-mps")
    assert execution_plan.backend == "native-mps"
    assert execution_plan.method == "mps-statevector"
    assert execution_plan.exact
    assert execution_plan.threads == 1
    assert execution_plan.tensor_network_max_bond >= 2
    assert execution_plan.tensor_network_routed_swaps >= 2
    assert execution_plan.tensor_network_contraction_work > 0.0

    automatic = qp.plan(program, qp.ResultMode.STATEVECTOR)
    assert automatic.backend != "native-mps"


def test_mps_statevector_expectation_and_variance_match_cpu() -> None:
    program = _mixed_program(5)
    cpu_state = qp.statevector(program, backend="native-cpu")
    mps_state = qp.statevector(program, backend="native-mps")
    np.testing.assert_allclose(mps_state.values, cpu_state.values, atol=5e-12, rtol=5e-12)
    assert mps_state.backend == "native-mps"

    observable = qp.Z(4)
    cpu_expectation = qp.expect(program, observable, backend="native-cpu")
    mps_expectation = qp.expect(program, observable, backend="native-mps")
    assert mps_expectation.value == pytest.approx(cpu_expectation.value, abs=5e-12)
    cpu_variance = qp.variance(program, observable, backend="native-cpu")
    mps_variance = qp.variance(program, observable, backend="native-mps")
    assert mps_variance.value == pytest.approx(cpu_variance.value, abs=5e-12)


def test_mps_large_low_bond_expectation_stays_compact() -> None:
    qubits = 128
    angle = 0.371
    program = qp.ry(qp.Program(qubits), angle, 0)
    for qubit in range(qubits - 1):
        program = qp.cx(program, qubit, qubit + 1)

    observable = qp.Z(qubits - 1)
    execution_plan = qp.expectation_plan(program, observable, backend="native-mps")
    assert execution_plan.method == "mps"
    assert execution_plan.tensor_network_max_bond == 2
    assert execution_plan.tensor_network_routed_swaps == 0
    assert execution_plan.estimated_state_bytes < 1 << 20
    assert execution_plan.tensor_network_contraction_work > 0.0

    result = qp.expect(program, observable, backend="native-mps")
    assert result.value == pytest.approx(math.cos(angle), abs=5e-12)


def test_mps_large_routing_estimate_does_not_block_execution() -> None:
    qubits = 64
    angle = 0.371
    program = qp.ry(qp.Program(qubits), angle, 0)
    for qubit in range(1, qubits):
        program = qp.cx(program, 0, qubit)

    observable = qp.Z(qubits - 1)
    execution_plan = qp.expectation_plan(program, observable, backend="native-mps")
    assert execution_plan.tensor_network_routed_swaps > 0
    assert execution_plan.tensor_network_max_bond > 1
    assert execution_plan.estimated_state_bytes > 0

    result = qp.expect(program, observable, backend="native-mps")
    assert result.value == pytest.approx(math.cos(angle), abs=5e-12)


def test_mps_batch_requests_fail_closed() -> None:
    program = qp.ry(qp.Program(1), 0.1, 0)
    slots = [qp.ParameterSlot(0)]
    values = np.array([[0.2]], dtype=np.float64)

    with pytest.raises(ValueError, match="parameter batches"):
        qp.expect_batch(program, qp.Z(0), slots, values, backend="native-mps")
    with pytest.raises(ValueError, match="result mode"):
        qp.sample_batch(program, slots, values, shots=8, seed=7, backend="native-mps")


def test_mps_rejects_unsupported_sampling() -> None:
    with pytest.raises(ValueError, match="result mode"):
        qp.sample(_mixed_program(), shots=8, seed=7, backend="native-mps")
