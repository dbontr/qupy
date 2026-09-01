import math

import numpy as np
import pytest

import qupy as qp


def bell_program() -> qp.Program:
    program = qp.Program(2)
    program = qp.h(program, 0)
    return qp.cx(program, 0, 1)


def test_native_core_identity_and_bell_statevector() -> None:
    assert qp.core_language() == "C++20"
    assert qp.parallel_threads() >= 1

    state = qp.statevector(bell_program())
    expected = np.array([1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)], dtype=np.complex128)
    np.testing.assert_allclose(state.values, expected, atol=1e-12)
    assert state.backend == "native-cpu"
    assert not state.values.flags.writeable


def test_bell_samples_are_correlated_and_reproducible() -> None:
    first = qp.sample(bell_program(), shots=512, seed=7)
    second = qp.sample(bell_program(), shots=512, seed=7)
    assert set(first.counts()) == {"00", "11"}
    assert first.shots == 512
    np.testing.assert_array_equal(first.values, second.values)


def test_bell_z_expectation_is_zero() -> None:
    result = qp.expect(bell_program(), qp.Z(0))
    assert result.value == pytest.approx(0.0, abs=1e-12)
    assert result.backend == "native-cpu"


def test_single_qubit_gate_conventions() -> None:
    x_program = qp.x(qp.Program(1), 0)
    np.testing.assert_allclose(qp.statevector(x_program).values, [0, 1], atol=1e-12)

    y_program = qp.y(qp.Program(1), 0)
    np.testing.assert_allclose(qp.statevector(y_program).values, [0, 1j], atol=1e-12)

    rx_program = qp.rx(qp.Program(1), math.pi, 0)
    np.testing.assert_allclose(qp.statevector(rx_program).values, [0, -1j], atol=1e-12)

    ry_program = qp.ry(qp.Program(1), math.pi, 0)
    np.testing.assert_allclose(qp.statevector(ry_program).values, [0, 1], atol=1e-12)

    rz_program = qp.rz(qp.x(qp.Program(1), 0), math.pi, 0)
    np.testing.assert_allclose(qp.statevector(rz_program).values, [0, 1j], atol=1e-12)


def test_two_qubit_gates() -> None:
    swap_program = qp.swap(qp.x(qp.Program(2), 0), 0, 1)
    np.testing.assert_allclose(qp.statevector(swap_program).values, [0, 0, 1, 0], atol=1e-12)

    cz_program = qp.Program(2)
    cz_program = qp.x(cz_program, 0)
    cz_program = qp.x(cz_program, 1)
    cz_program = qp.cz(cz_program, 0, 1)
    np.testing.assert_allclose(qp.statevector(cz_program).values, [0, 0, 0, -1], atol=1e-12)


def test_native_target_and_planner_are_explicit() -> None:
    target = qp.native_target()
    assert target.name == "native-cpu"
    assert target.simulator
    assert target.supports_operation(qp.OperationCode.CX)
    assert target.supports_result(qp.ResultMode.STATEVECTOR)

    execution_plan = qp.plan(bell_program(), qp.ResultMode.SAMPLE)
    assert execution_plan.backend == "native-cpu"
    assert execution_plan.method == "statevector"
    assert execution_plan.exact
    assert execution_plan.threads >= 1
    assert execution_plan.original_operations == 2
    assert execution_plan.compiled_steps == 2
    assert execution_plan.active_qubits == 2
    assert execution_plan.estimated_state_bytes == 64


def test_invalid_programs_and_execution_options_are_rejected() -> None:
    with pytest.raises(ValueError, match="outside this program"):
        qp.h(qp.Program(2), 2)

    with pytest.raises(ValueError, match="same qubit twice"):
        qp.cx(qp.Program(2), 0, 0)

    with pytest.raises(ValueError, match="shots must be at least 1"):
        qp.sample(qp.Program(1), shots=0)

    with pytest.raises(ValueError, match="unknown backend"):
        qp.statevector(qp.Program(1), backend="missing")


def test_program_ir_is_native_and_immutable() -> None:
    base = qp.Program(2)
    derived = qp.h(base, 1)
    assert base.operations == []
    assert len(derived.operations) == 1
    assert derived.operations[0].name == "h"
    assert derived.operations[0].qubits == [1]


def test_native_compiler_fuses_single_qubit_runs() -> None:
    program = qp.Program(2)
    program = qp.h(program, 0)
    program = qp.rx(program, 0.2, 0)
    program = qp.rz(program, -0.4, 0)
    program = qp.x(program, 1)
    program = qp.ry(program, 0.3, 1)

    execution_plan = qp.plan(program, qp.ResultMode.STATEVECTOR)
    assert execution_plan.original_operations == 5
    assert execution_plan.compiled_steps == 2
    assert execution_plan.active_qubits == 2


def test_expectation_planner_reduces_to_exact_reverse_lightcone() -> None:
    program = qp.Program(100)
    program = qp.h(program, 0)
    program = qp.x(program, 98)
    program = qp.ry(program, 0.7, 99)

    execution_plan = qp.expectation_plan(program, qp.Z(0))
    assert execution_plan.method == "statevector-lightcone"
    assert execution_plan.exact
    assert execution_plan.active_qubits == 1
    assert execution_plan.compiled_steps == 1
    assert execution_plan.estimated_state_bytes == 32
    result = qp.expect(program, qp.Z(0))
    assert result.value == pytest.approx(0.0, abs=1e-12)
    assert result.active_qubits == 1
    assert result.compiled_steps == 1
    assert result.estimated_state_bytes == 32


def test_expectation_lightcone_expands_across_entanglement() -> None:
    program = qp.Program(64)
    program = qp.h(program, 0)
    program = qp.cx(program, 0, 37)
    program = qp.x(program, 63)

    execution_plan = qp.expectation_plan(program, qp.Z(37))
    assert execution_plan.method == "statevector-lightcone"
    assert execution_plan.active_qubits == 2
    assert execution_plan.estimated_state_bytes == 64

    result = qp.expect(program, qp.Z(37))
    assert result.value == pytest.approx(0.0, abs=1e-12)
    assert result.active_qubits == 2
