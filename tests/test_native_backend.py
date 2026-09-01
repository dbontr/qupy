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
    assert qp.core_version() == "0.3.0a0"
    assert qp.ir_version() == 1
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

    deterministic = qp.sample(bell_program(), shots=16, seed=7)
    expected = np.array(
        [
            [0, 0], [1, 1], [1, 1], [1, 1],
            [1, 1], [1, 1], [0, 0], [0, 0],
            [1, 1], [0, 0], [1, 1], [0, 0],
            [1, 1], [0, 0], [0, 0], [0, 0],
        ],
        dtype=np.int8,
    )
    np.testing.assert_array_equal(deterministic.values, expected)


def test_parameter_binding_and_native_batches() -> None:
    template = qp.ry(qp.Program(1), 0.0, 0)
    template_fingerprint = template.fingerprint
    slots = [qp.ParameterSlot(0)]

    bound = template.bind(slots, [math.pi])
    assert template.fingerprint == template_fingerprint
    assert bound.fingerprint != template_fingerprint
    assert qp.expect(bound, qp.Z(0)).value == pytest.approx(-1.0, abs=1e-12)

    parameters = np.array([[0.0], [math.pi / 2.0], [math.pi]], dtype=np.float64)
    expectations = qp.expect_batch(template, qp.Z(0), slots, parameters)
    np.testing.assert_allclose(expectations.values, [1.0, 0.0, -1.0], atol=1e-12)
    assert expectations.values.shape == (3,)
    assert not expectations.values.flags.writeable
    assert expectations.batch_size == 3
    assert expectations.parameter_count == 1
    assert expectations.active_qubits == 1
    assert expectations.compiled_steps == 1
    assert expectations.estimated_state_bytes == 32

    sampled = qp.cx(qp.ry(qp.Program(2), 0.0, 0), 0, 1)
    sample_parameters = np.array([[0.0], [math.pi]], dtype=np.float64)
    first = qp.sample_batch(sampled, slots, sample_parameters, shots=32, seed=7)
    second = qp.sample_batch(sampled, slots, sample_parameters, shots=32, seed=7)
    assert first.values.shape == (2, 32, 2)
    assert not first.values.flags.writeable
    np.testing.assert_array_equal(first.values, second.values)
    assert first.counts(0) == {"00": 32}
    assert first.counts(1) == {"11": 32}
    assert first.batch_size == 2
    assert first.parameter_count == 1
    assert first.compiled_steps == 2
    assert first.estimated_state_bytes == 64

    one_row = qp.sample_batch(
        sampled,
        slots,
        np.array([[math.pi / 2.0]], dtype=np.float64),
        shots=32,
        seed=19,
    )
    scalar = qp.sample(sampled.bind(slots, [math.pi / 2.0]), shots=32, seed=19)
    np.testing.assert_array_equal(one_row.values[0], scalar.values)

    with pytest.raises(ValueError, match="columns must match"):
        qp.expect_batch(template, qp.Z(0), slots, np.empty((2, 0), dtype=np.float64))


def test_bell_z_expectation_is_zero() -> None:
    result = qp.expect(bell_program(), qp.Z(0))
    assert result.value == pytest.approx(0.0, abs=1e-12)
    assert result.backend == "native-cpu"
    assert result.estimated_state_bytes == 0


def test_statevector_storage_is_independent_of_internal_workspace() -> None:
    owned_program = qp.ry(qp.Program(1), math.pi, 0)
    owned = qp.statevector(owned_program)
    expected = owned.values.copy()

    workspace_program = qp.ry(qp.Program(1), 0.0, 0)
    assert qp.expect(workspace_program, qp.Z(0)).value == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(owned.values, expected, atol=1e-12)


def test_native_probabilities_and_variance() -> None:
    program = bell_program()
    probabilities = qp.probabilities(program)
    np.testing.assert_allclose(probabilities.values, [0.5, 0.0, 0.0, 0.5], atol=1e-12)
    assert probabilities.backend == "native-cpu"
    assert not probabilities.values.flags.writeable

    result = qp.variance(program, qp.Z(0))
    assert result.value == pytest.approx(1.0, abs=1e-12)
    assert result.active_qubits == 2
    assert result.estimated_state_bytes == 0

    expectation_plan = qp.expectation_plan(program, qp.Z(0))
    variance_plan = qp.variance_plan(program, qp.Z(0))
    assert variance_plan.result_mode == qp.ResultMode.VARIANCE
    assert expectation_plan.cache_key != variance_plan.cache_key


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
    assert target.state_access
    assert not target.mid_circuit_measurement
    assert not target.reset
    assert not target.dynamic_control
    assert target.parameter_batches
    assert len(target.fingerprint) == 64
    assert target.supports_operation(qp.OperationCode.CX)
    assert target.supports_result(qp.ResultMode.STATEVECTOR)
    assert target.supports_result(qp.ResultMode.PROBABILITIES)
    assert target.supports_result(qp.ResultMode.VARIANCE)

    program = bell_program()
    assert program.fingerprint == "ab7840ba9d0cd5353fe9e66c9100b195a8f5ad566f13e82f8d775e350f7e8009"
    execution_plan = qp.plan(program, qp.ResultMode.SAMPLE)
    assert execution_plan.backend == "native-cpu"
    assert execution_plan.method == "statevector"
    assert execution_plan.exact
    assert execution_plan.threads == 1
    assert execution_plan.original_qubits == 2
    assert execution_plan.original_operations == 2
    assert execution_plan.active_qubits == 2
    assert execution_plan.active_operations == 2
    assert execution_plan.single_qubit_operations == 1
    assert execution_plan.two_qubit_operations == 1
    assert execution_plan.parameterized_operations == 0
    assert execution_plan.non_clifford_operations == 0
    assert execution_plan.compiled_steps == 2
    assert execution_plan.estimated_state_bytes == 64
    assert execution_plan.result_mode == qp.ResultMode.SAMPLE
    assert execution_plan.workload_version == 1
    assert len(execution_plan.workload_fingerprint) == 64
    assert execution_plan.workload_fingerprint == (
        "6146406a5bd9baf7b57435a3815bc8427a2a6990b1c183fd10db46cebb841b0d"
    )
    assert execution_plan.program_fingerprint == program.fingerprint
    assert execution_plan.target_fingerprint == target.fingerprint
    assert execution_plan.cache_key.startswith("qupy-cache/1/0.3.0a0/")

    parallel_plan = qp.plan(qp.Program(16), qp.ResultMode.STATEVECTOR)
    assert parallel_plan.threads == min(qp.parallel_threads(), 8)

    large_parallel_plan = qp.plan(qp.Program(20), qp.ResultMode.STATEVECTOR)
    assert large_parallel_plan.threads == min(qp.parallel_threads(), 16)


def test_workload_fingerprint_is_structural_and_result_aware() -> None:
    first = qp.Program(4)
    first = qp.ry(first, 0.1, 0)
    first = qp.cx(first, 0, 1)
    first = qp.rz(first, 0.2, 3)

    second = qp.Program(4)
    second = qp.ry(second, 1.1, 0)
    second = qp.cx(second, 0, 1)
    second = qp.rz(second, -0.7, 3)

    first_dense = qp.plan(first, qp.ResultMode.STATEVECTOR)
    second_dense = qp.plan(second, qp.ResultMode.STATEVECTOR)
    assert first.fingerprint != second.fingerprint
    assert first_dense.workload_fingerprint == second_dense.workload_fingerprint
    assert first_dense.active_operations == 3
    assert first_dense.single_qubit_operations == 2
    assert first_dense.two_qubit_operations == 1
    assert first_dense.parameterized_operations == 2
    assert first_dense.non_clifford_operations == 2

    lightcone = qp.expectation_plan(first, qp.Z(1))
    assert lightcone.workload_fingerprint != first_dense.workload_fingerprint
    assert lightcone.active_qubits == 2
    assert lightcone.active_operations == 2
    assert lightcone.single_qubit_operations == 1
    assert lightcone.two_qubit_operations == 1
    assert lightcone.parameterized_operations == 1
    assert lightcone.non_clifford_operations == 1

    sample_plan = qp.plan(first, qp.ResultMode.SAMPLE)
    assert sample_plan.workload_fingerprint != first_dense.workload_fingerprint


def test_invalid_programs_and_execution_options_are_rejected() -> None:
    with pytest.raises(ValueError, match="outside this program"):
        qp.h(qp.Program(2), 2)

    with pytest.raises(ValueError, match="same qubit twice"):
        qp.cx(qp.Program(2), 0, 0)

    with pytest.raises(ValueError, match="shots must be at least 1"):
        qp.sample(qp.Program(1), shots=0)

    with pytest.raises(ValueError, match="unknown backend"):
        qp.statevector(qp.Program(1), backend="missing")

    with pytest.raises(ValueError, match="observable result mode requires"):
        qp.plan(qp.Program(1), qp.ResultMode.EXPECTATION)


def test_program_ir_is_native_and_immutable() -> None:
    base = qp.Program(2)
    derived = qp.h(base, 1)
    repeated = qp.h(qp.Program(2), 1)
    assert base.operations == []
    assert len(derived.operations) == 1
    assert derived.operations[0].name == "h"
    assert derived.operations[0].qubits == [1]
    assert derived.canonical_text.startswith("qupy-ir 1\nqubits 2\n")
    assert derived.canonical_text == repeated.canonical_text
    assert derived.fingerprint == repeated.fingerprint
    assert derived.fingerprint != base.fingerprint
    assert len(derived.fingerprint) == 64


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


def test_expectation_planner_uses_zero_state_pauli_propagation() -> None:
    program = qp.Program(100)
    program = qp.h(program, 0)
    program = qp.x(program, 98)
    program = qp.ry(program, 0.7, 99)

    execution_plan = qp.expectation_plan(program, qp.Z(0))
    assert execution_plan.method == "pauli-propagation"
    assert execution_plan.exact
    assert execution_plan.active_qubits == 1
    assert execution_plan.compiled_steps == 1
    assert execution_plan.estimated_state_bytes == 0
    result = qp.expect(program, qp.Z(0))
    assert result.value == pytest.approx(0.0, abs=1e-12)
    assert result.active_qubits == 1
    assert result.compiled_steps == 1
    assert result.estimated_state_bytes == 0


def test_expectation_planner_falls_back_for_relevant_non_clifford() -> None:
    program = qp.Program(64)
    program = qp.h(program, 0)
    program = qp.ry(program, 0.7, 0)
    program = qp.x(program, 63)

    execution_plan = qp.expectation_plan(program, qp.Z(0))
    assert execution_plan.method == "statevector-lightcone"
    assert execution_plan.exact
    assert execution_plan.active_qubits == 1
    assert execution_plan.estimated_state_bytes == 32
    result = qp.expect(program, qp.Z(0))
    assert result.value == pytest.approx(-math.sin(0.7), abs=1e-12)
    assert result.estimated_state_bytes == 32


def test_pauli_propagation_tracks_entanglement_and_ignores_unrelated_rotations() -> None:
    program = qp.Program(64)
    program = qp.h(program, 0)
    program = qp.cx(program, 0, 37)
    program = qp.ry(program, 0.7, 63)

    execution_plan = qp.expectation_plan(program, qp.Z(37))
    assert execution_plan.method == "pauli-propagation"
    assert execution_plan.active_qubits == 2
    assert execution_plan.estimated_state_bytes == 0

    result = qp.expect(program, qp.Z(37))
    assert result.value == pytest.approx(0.0, abs=1e-12)
    assert result.active_qubits == 2
    assert result.estimated_state_bytes == 0
