from __future__ import annotations

import itertools
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import qupy as qp


def _term(coefficient: float, *factors: tuple[int, qp.Pauli]) -> qp.PauliTerm:
    return qp.PauliTerm(
        coefficient,
        [qp.PauliFactor(qubit, pauli) for qubit, pauli in factors],
    )


def _observable(*terms: qp.PauliTerm) -> qp.Observable:
    return qp.Observable(list(terms))


def _bell_program() -> qp.Program:
    program = qp.Program(2)
    program = qp.h(program, 0)
    return qp.cx(program, 0, 1)
def test_pauli_sum_observables_cover_expectation_variance_and_covariance() -> None:
    program = _bell_program()
    xx = _observable(_term(1.0, (0, qp.Pauli.X), (1, qp.Pauli.X)))
    yy = _observable(_term(1.0, (0, qp.Pauli.Y), (1, qp.Pauli.Y)))
    zz = _observable(_term(1.0, (0, qp.Pauli.Z), (1, qp.Pauli.Z)))
    hamiltonian = _observable(
        _term(0.5, (0, qp.Pauli.X), (1, qp.Pauli.X)),
        _term(0.25, (0, qp.Pauli.Z), (1, qp.Pauli.Z)),
        _term(-0.125),
    )

    assert qp.expect(program, xx).value == pytest.approx(1.0, abs=1e-12)
    assert qp.expect(program, yy).value == pytest.approx(-1.0, abs=1e-12)
    assert qp.expect(program, zz).value == pytest.approx(1.0, abs=1e-12)
    assert qp.expect(program, hamiltonian).value == pytest.approx(0.625, abs=1e-12)

    stabilizer_sum = _observable(*xx.terms, *zz.terms)
    assert qp.variance(program, stabilizer_sum).value == pytest.approx(0.0, abs=1e-12)
    assert qp.covariance(program, xx, zz).value == pytest.approx(0.0, abs=1e-12)


def test_observable_grouping_and_union_causal_cone() -> None:
    observable = _observable(
        _term(1.0, (0, qp.Pauli.X), (1, qp.Pauli.X)),
        _term(1.0, (0, qp.Pauli.Y), (1, qp.Pauli.Y)),
        _term(1.0, (0, qp.Pauli.Z), (1, qp.Pauli.Z)),
        _term(1.0, (0, qp.Pauli.X)),
    )
    groups = qp.commuting_groups(observable)
    assert sorted(index for group in groups for index in group) == [0, 1, 2, 3]
    assert any(set(group) == {0, 1, 2} for group in groups)

    program = qp.Program(6)
    program = qp.h(program, 0)
    program = qp.cx(program, 0, 1)
    program = qp.ry(program, 0.37, 5)
    z1 = _observable(_term(1.0, (1, qp.Pauli.Z)))
    xx = _observable(_term(1.0, (0, qp.Pauli.X), (1, qp.Pauli.X)))
    batch = qp.expect_observables(program, [z1, xx])

    assert batch.observable_count == 2
    assert batch.active_qubits == 2
    np.testing.assert_allclose(batch.values, [0.0, 1.0], atol=1e-12)


@pytest.mark.parametrize(
    ("method", "expected_name", "tolerance"),
    [
        (qp.GradientMethod.AUTO, "adjoint", 1e-12),
        (qp.GradientMethod.ADJOINT, "adjoint", 1e-12),
        (qp.GradientMethod.PARAMETER_SHIFT, "parameter-shift", 1e-12),
        (qp.GradientMethod.FINITE_DIFFERENCE, "finite-difference", 2e-8),
    ],
)
def test_value_and_grad_matches_analytic_rotation(
    method: qp.GradientMethod,
    expected_name: str,
    tolerance: float,
) -> None:
    theta = 0.371
    program = qp.ry(qp.Program(1), 0.0, 0)
    observable = _observable(_term(1.0, (0, qp.Pauli.Z)))
    result = qp.value_and_grad(
        program,
        observable,
        [qp.ParameterSlot(0, 0)],
        np.array([theta], dtype=np.float64),
        method=method,
    )

    assert result.method == expected_name
    assert result.value == pytest.approx(math.cos(theta), abs=tolerance)
    assert result.gradient[0] == pytest.approx(-math.sin(theta), abs=tolerance)


def test_adjoint_matches_parameter_shift_for_entangled_parameters() -> None:
    program = qp.Program(2)
    program = qp.ry(program, 0.0, 0)
    program = qp.rx(program, 0.0, 1)
    program = qp.cx(program, 0, 1)
    observable = _observable(
        _term(0.7, (0, qp.Pauli.Z)),
        _term(-0.4, (0, qp.Pauli.X), (1, qp.Pauli.X)),
        _term(0.2, (1, qp.Pauli.Y)),
    )
    slots = [qp.ParameterSlot(0, 0), qp.ParameterSlot(1, 0)]
    values = np.array([0.42, -0.31], dtype=np.float64)

    adjoint = qp.value_and_grad(
        program, observable, slots, values, method=qp.GradientMethod.ADJOINT
    )
    shifted = qp.value_and_grad(
        program, observable, slots, values, method=qp.GradientMethod.PARAMETER_SHIFT
    )

    assert adjoint.value == pytest.approx(shifted.value, abs=1e-12)
    np.testing.assert_allclose(adjoint.gradient, shifted.gradient, atol=1e-12)
    assert adjoint.evaluations == 1
    assert shifted.evaluations == 5


def test_optimizer_reduces_self_inverse_and_rotation_runs() -> None:
    program = qp.Program(2)
    program = qp.x(program, 0)
    program = qp.x(program, 0)
    program = qp.ry(program, 0.2, 1)
    program = qp.ry(program, 0.3, 1)
    program = qp.cz(program, 0, 1)
    program = qp.cz(program, 1, 0)

    report = qp.optimize(program)
    before = qp.statevector(program, "native-cpu")
    after = qp.statevector(report.program, "native-cpu")

    assert report.original_operations == 6
    assert report.optimized_operations == 1
    assert set(report.passes) == {"inverse-cancellation", "rotation-merge"}
    np.testing.assert_allclose(after.values, before.values, atol=1e-12)


def test_density_matrix_matches_pure_state_outer_product() -> None:
    program = qp.ry(qp.h(qp.Program(2), 0), 0.37, 1)
    state = qp.statevector(program, "native-cpu").values
    density = qp.density_matrix(program)

    assert density.backend == "native-density"
    np.testing.assert_allclose(density.values, np.outer(state, state.conj()), atol=1e-12)


def test_kraus_noise_channels_are_trace_preserving() -> None:
    program = qp.x(qp.Program(1), 0)
    noisy = qp.NoisyProgram(
        program,
        [qp.NoiseInstruction(1, qp.amplitude_damping(0, 1.0))],
    )
    density = qp.density_matrix(noisy)
    np.testing.assert_allclose(
        density.values,
        np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128),
        atol=1e-12,
    )

    plus = qp.h(qp.Program(1), 0)
    dephased = qp.density_matrix(
        qp.NoisyProgram(plus, [qp.NoiseInstruction(1, qp.phase_damping(0, 1.0))])
    )
    np.testing.assert_allclose(
        dephased.values,
        np.array([[0.5, 0.0], [0.0, 0.5]], dtype=np.complex128),
        atol=1e-12,
    )
    assert np.trace(dephased.values) == pytest.approx(1.0, abs=1e-12)


def test_density_matrix_backend_contract() -> None:
    program = qp.rz(qp.h(qp.Program(1), 0), 0.37, 0)
    assert qp.density_matrix(program).backend == "native-density"
    assert qp.density_matrix(program, "cpu").backend == "native-density"
    with pytest.raises(ValueError, match="density-matrix backend"):
        qp.density_matrix(program, "invalid")
    if not qp.cuda_available():
        with pytest.raises(RuntimeError, match="CUDA"):
            qp.density_matrix(program, "native-cuda")


def test_cuda_density_matrix_matches_cpu_for_unitary_and_noise_execution() -> None:
    if not qp.cuda_available():
        pytest.skip(qp.cuda_unavailable_reason())
    program = qp.h(qp.Program(3), 0)
    program = qp.rz(program, 0.41, 0)
    program = qp.ry(program, -0.27, 1)
    program = qp.rx(program, 0.39, 2)
    program = qp.cx(program, 0, 1)
    program = qp.cz(program, 1, 2)
    program = qp.swap(program, 0, 2)
    program = qp.rz(program, -0.17, 2)
    cpu = qp.density_matrix(program, "native-cpu")
    gpu = qp.density_matrix(program, "native-cuda")
    assert gpu.backend == "native-cuda-density"
    np.testing.assert_allclose(gpu.values, cpu.values, atol=3e-12, rtol=3e-12)

    phase = np.exp(0.37j)
    unitary = np.array([[0.0, phase], [phase.conjugate(), 0.0]], dtype=np.complex128)
    custom = qp.kraus_channel(
        1,
        [math.sqrt(0.6) * np.eye(2, dtype=np.complex128), math.sqrt(0.4) * unitary],
    )
    noisy = qp.NoisyProgram(
        program,
        [
            qp.NoiseInstruction(0, qp.bit_flip(0, 0.11)),
            qp.NoiseInstruction(2, qp.phase_flip(1, 0.07)),
            qp.NoiseInstruction(4, qp.depolarizing(2, 0.09)),
            qp.NoiseInstruction(5, qp.amplitude_damping(0, 0.13)),
            qp.NoiseInstruction(6, qp.phase_damping(1, 0.17)),
            qp.NoiseInstruction(8, qp.pauli_channel(2, 0.03, 0.04, 0.05)),
            qp.NoiseInstruction(8, custom),
        ],
    )
    noisy_cpu = qp.density_matrix(noisy, "native-cpu")
    noisy_gpu = qp.density_matrix(noisy, "native-cuda")
    np.testing.assert_allclose(noisy_gpu.values, noisy_cpu.values, atol=3e-12, rtol=3e-12)
    np.testing.assert_allclose(noisy_gpu.values, noisy_gpu.values.conj().T, atol=3e-12)
    assert np.trace(noisy_gpu.values) == pytest.approx(1.0, abs=3e-12)


def test_lindblad_rk4_preserves_static_state_and_models_decay() -> None:
    zero_hamiltonian = np.zeros((2, 2), dtype=np.complex128)
    initial_zero = qp.density_matrix(qp.Program(1))
    static = qp.lindblad_evolve(initial_zero, zero_hamiltonian, [], 0.01, 20)
    np.testing.assert_allclose(static.state.values, initial_zero.values, atol=1e-12)

    gamma = 0.7
    initial_one = qp.density_matrix(qp.x(qp.Program(1), 0))
    collapse = np.array(
        [[0.0, math.sqrt(gamma)], [0.0, 0.0]],
        dtype=np.complex128,
    )
    evolved = qp.lindblad_evolve(
        initial_one,
        zero_hamiltonian,
        [collapse],
        0.002,
        200,
    )
    expected_excited = math.exp(-gamma * 0.4)
    assert evolved.state.values[1, 1].real == pytest.approx(expected_excited, abs=2e-9)
    assert np.trace(evolved.state.values) == pytest.approx(1.0, abs=1e-12)


def test_openqasm_and_qir_exporters_cover_provider_interchange(tmp_path: Path) -> None:
    program = qp.Program(2)
    program = qp.h(program, 0)
    program = qp.rx(program, 0.25, 1)
    program = qp.cx(program, 0, 1)
    program = qp.swap(program, 0, 1)

    qasm = qp.to_openqasm3(program, measure_all=True)
    assert qasm.format == "openqasm3"
    assert qasm.text.startswith("OPENQASM 3.1;\ninclude \"stdgates.inc\";")
    assert "rx(0.25) q[1];" in qasm.text
    assert "c = measure q;" in qasm.text

    qir = qp.to_qir_base_profile(program, measure_all=True)
    assert qir.format == "qir-base-profile"
    assert '"qir_profiles"="base_profile"' in qir.text
    assert qir.text.count("@__quantum__qis__cnot__body") >= 5
    assert "@__quantum__qis__mz__body" in qir.text

    llvm_as = shutil.which("llvm-as")
    if llvm_as is not None:
        source = tmp_path / "program.ll"
        source.write_text(qir.text, encoding="utf-8")
        subprocess.run([llvm_as, str(source), "-o", str(tmp_path / "program.bc")], check=True)


def test_detector_models_sample_and_decode_deterministically() -> None:
    model = qp.repetition_code_detector_model(3, 2, 0.01, 0.02)
    assert model.detector_count == 4
    assert model.observable_count == 1
    assert len(model.errors) == 10
    assert len(model.fingerprint) == 64

    first = qp.sample_detector_model(model, 64, seed=1234)
    second = qp.sample_detector_model(model, 64, seed=1234)
    assert first.syndrome.shape == (64, 4)
    assert first.observables.shape == (64, 1)
    np.testing.assert_array_equal(first.syndrome, second.syndrome)
    np.testing.assert_array_equal(first.observables, second.observables)

    exact_model = qp.DetectorModel(
        2,
        1,
        [
            qp.DetectorError(0.10, [0], [0]),
            qp.DetectorError(0.05, [1]),
            qp.DetectorError(0.01, [0, 1]),
        ],
    )
    decoded = qp.decode_detector_model(exact_model, np.array([1, 0], dtype=np.int8))
    assert decoded.observables == [1]
    assert decoded.matched_errors == 1


def test_measurement_groups_and_shot_estimator_are_deterministic() -> None:
    program = qp.h(qp.Program(1), 0)
    observable = _observable(
        _term(1.0, (0, qp.Pauli.X)),
        _term(1.0, (0, qp.Pauli.Z)),
    )
    groups = qp.measurement_groups(observable)
    assert len(groups) == 2
    assert sorted(index for group in groups for index in group.term_indices) == [0, 1]

    first = qp.estimate_observable(program, observable, shots_per_group=4096, seed=19)
    second = qp.estimate_observable(program, observable, shots_per_group=4096, seed=19)
    assert first.value == second.value
    assert first.standard_error == second.standard_error
    assert first.group_count == 2
    assert first.total_shots == 8192
    assert first.backend == "native-cpu"
    assert first.value == pytest.approx(1.0, abs=0.08)
    assert first.standard_error > 0.0


def test_grad_jacobian_and_hessian_match_closed_form_rotations() -> None:
    program = qp.Program(2)
    program = qp.ry(program, 0.0, 0)
    program = qp.ry(program, 0.0, 1)
    slots = [qp.ParameterSlot(0), qp.ParameterSlot(1)]
    parameters = np.array([0.37, -0.29], dtype=np.float64)
    z0 = _observable(_term(1.0, (0, qp.Pauli.Z)))
    z1 = _observable(_term(1.0, (1, qp.Pauli.Z)))

    gradient = qp.grad(program, z0, slots, parameters)
    assert gradient.value == pytest.approx(math.cos(parameters[0]), abs=1e-12)
    np.testing.assert_allclose(gradient.gradient, [-math.sin(parameters[0]), 0.0], atol=1e-12)

    result = qp.jacobian(program, [z0, z1], slots, parameters)
    np.testing.assert_allclose(result.values, np.cos(parameters), atol=1e-12)
    np.testing.assert_allclose(
        result.jacobian,
        [[-math.sin(parameters[0]), 0.0], [0.0, -math.sin(parameters[1])]],
        atol=1e-12,
    )

    summed = _observable(
        _term(1.0, (0, qp.Pauli.Z)),
        _term(1.0, (1, qp.Pauli.Z)),
    )
    curvature = qp.hessian(program, summed, slots, parameters)
    assert curvature.value == pytest.approx(sum(np.cos(parameters)), abs=1e-12)
    np.testing.assert_allclose(curvature.gradient, -np.sin(parameters), atol=1e-12)
    np.testing.assert_allclose(
        curvature.hessian,
        [[-math.cos(parameters[0]), 0.0], [0.0, -math.cos(parameters[1])]],
        atol=1e-12,
    )
    assert curvature.method == "parameter-shift"
    assert curvature.parameter_count == 2


def test_optimizer_commutes_disjoint_gates_to_expose_reductions() -> None:
    program = qp.x(qp.Program(2), 0)
    program = qp.h(program, 1)
    program = qp.x(program, 0)
    level_one = qp.optimize(program, level=1)
    level_two = qp.optimize(program, level=2)
    assert level_one.optimized_operations == 3
    assert level_two.optimized_operations == 1
    assert level_two.program.operations[0].name == "h"
    assert "disjoint-commutation" in level_two.passes
    np.testing.assert_allclose(
        qp.statevector(level_two.program).values,
        qp.statevector(program).values,
        atol=1e-12,
    )

    rotations = qp.ry(qp.Program(2), 0.2, 0)
    rotations = qp.h(rotations, 1)
    rotations = qp.ry(rotations, 0.3, 0)
    optimized = qp.optimize(rotations, level=2)
    assert optimized.optimized_operations == 2
    assert {"rotation-merge", "disjoint-commutation"} <= set(optimized.passes)
    np.testing.assert_allclose(
        qp.statevector(optimized.program).values,
        qp.statevector(rotations).values,
        atol=1e-12,
    )


def test_distributed_surface_is_explicit_and_fail_closed() -> None:
    info = qp.distributed_info()
    assert info.world_size >= 1
    assert info.rank < info.world_size
    assert info.local_rank >= 0
    if not qp.mpi_compiled():
        with pytest.raises(RuntimeError, match="MPI support is not compiled"):
            qp.distributed_statevector(_bell_program())
        xx = _observable(_term(1.0, (0, qp.Pauli.X), (1, qp.Pauli.X)))
        with pytest.raises(RuntimeError, match="MPI support is not compiled"):
            qp.observable_plan(_bell_program(), [xx], backend="native-mpi")
        with pytest.raises(RuntimeError, match="MPI support is not compiled"):
            qp.expect_observable(_bell_program(), xx, backend="native-mpi")


def test_rich_pauli_propagation_handles_large_clifford_hamiltonians() -> None:
    qubits = 128
    program = qp.h(qp.Program(qubits), 0)
    for qubit in range(1, qubits):
        program = qp.cx(program, qubit - 1, qubit)

    all_x = _term(0.75, *((qubit, qp.Pauli.X) for qubit in range(qubits)))
    endpoint_zz = _term(0.25, (0, qp.Pauli.Z), (qubits - 1, qp.Pauli.Z))
    hamiltonian = _observable(all_x, endpoint_zz)

    result = qp.expect_observable(program, hamiltonian)
    assert result.backend == "native-cpu"
    assert result.active_qubits == qubits
    assert result.value == pytest.approx(1.0, abs=1e-12)
    assert qp.variance_observable(program, hamiltonian).value == pytest.approx(0.0, abs=1e-12)


def test_observable_plan_is_query_aware_and_uses_specialized_execution() -> None:
    bell = _bell_program()
    xx = _observable(_term(1.0, (0, qp.Pauli.X), (1, qp.Pauli.X)))
    plan = qp.observable_plan(bell, [xx])
    assert plan.backend == "native-cpu"
    assert plan.method == "pauli-propagation"
    assert plan.exact
    assert plan.active_qubits == 2
    assert plan.term_count == 1
    assert plan.measurement_group_count == 1
    assert plan.estimated_state_bytes == 0
    assert len(plan.query_fingerprint) == 64
    assert len(plan.cache_key) == 64

    program = qp.ry(qp.Program(1), 0.37, 0)
    z = _observable(_term(1.0, (0, qp.Pauli.Z)))
    dense = qp.observable_plan(program, [z])
    assert dense.backend == "native-cpu"
    assert dense.method == "statevector-observable"
    assert dense.estimated_state_bytes == 2 * np.dtype(np.complex128).itemsize


def test_custom_kraus_channels_are_validated_and_executable() -> None:
    gamma = 0.3
    k0 = np.array([[1.0, 0.0], [0.0, math.sqrt(1.0 - gamma)]], dtype=np.complex128)
    k1 = np.array([[0.0, math.sqrt(gamma)], [0.0, 0.0]], dtype=np.complex128)
    channel = qp.kraus_channel(0, [k0, k1])
    assert channel.code is qp.NoiseChannelCode.KRAUS
    assert channel.kraus_count == 2
    assert len(channel.kraus_operators) == 8

    program = qp.x(qp.Program(1), 0)
    noisy = qp.NoisyProgram(program, [qp.NoiseInstruction(1, channel)])
    result = qp.density_matrix(noisy)
    np.testing.assert_allclose(
        result.values,
        [[gamma, 0.0], [0.0, 1.0 - gamma]],
        atol=1e-12,
    )

    bad = np.array([[1.0, 0.0], [0.0, 0.5]], dtype=np.complex128)
    with pytest.raises(ValueError, match="trace-preserving"):
        qp.kraus_channel(0, [bad])


def test_cuda_pauli_reduction_matches_cpu_for_rich_observables() -> None:
    if not qp.cuda_available():
        pytest.skip(qp.cuda_unavailable_reason())
    program = qp.h(qp.Program(3), 0)
    program = qp.ry(program, 0.371, 1)
    program = qp.cx(program, 0, 1)
    program = qp.rz(program, -0.29, 2)
    program = qp.cx(program, 1, 2)
    left = _observable(
        _term(0.41, (0, qp.Pauli.X), (1, qp.Pauli.Y)),
        _term(-0.27, (1, qp.Pauli.Z), (2, qp.Pauli.X)),
        _term(0.13),
    )
    right = _observable(
        _term(0.36, (0, qp.Pauli.Y), (2, qp.Pauli.Y)),
        _term(0.22, (1, qp.Pauli.X)),
    )

    plan = qp.observable_plan(program, [left, right], backend="native-cuda")
    assert plan.backend == "native-cuda"
    assert plan.method == "cuda-pauli-reduction"

    cpu_expect = qp.expect_observable(program, left, backend="native-cpu")
    gpu_expect = qp.expect_observable(program, left, backend="native-cuda")
    cpu_variance = qp.variance_observable(program, left, backend="native-cpu")
    gpu_variance = qp.variance_observable(program, left, backend="native-cuda")
    cpu_covariance = qp.covariance(program, left, right, backend="native-cpu")
    gpu_covariance = qp.covariance(program, left, right, backend="native-cuda")
    assert gpu_expect.backend == "native-cuda"
    assert gpu_variance.backend == "native-cuda"
    assert gpu_covariance.backend == "native-cuda"
    assert gpu_expect.value == pytest.approx(cpu_expect.value, abs=2e-12)
    assert gpu_variance.value == pytest.approx(cpu_variance.value, abs=2e-12)
    assert gpu_covariance.value == pytest.approx(cpu_covariance.value, abs=2e-12)

    cpu_batch = qp.expect_observables(program, [left, right], backend="native-cpu")
    gpu_batch = qp.expect_observables(program, [left, right], backend="native-cuda")
    assert gpu_batch.backend == "native-cuda"
    np.testing.assert_allclose(gpu_batch.values, cpu_batch.values, atol=2e-12, rtol=2e-12)


def test_cuda_pauli_reduction_covers_every_four_qubit_pauli_string() -> None:
    if not qp.cuda_available():
        pytest.skip(qp.cuda_unavailable_reason())
    program = qp.Program(4)
    program = qp.ry(program, 0.31, 0)
    program = qp.rx(program, -0.47, 1)
    program = qp.rz(program, 0.29, 2)
    program = qp.h(program, 3)
    program = qp.cx(program, 0, 2)
    program = qp.cz(program, 1, 3)
    program = qp.ry(program, 0.19, 3)

    paulis = (qp.Pauli.I, qp.Pauli.X, qp.Pauli.Y, qp.Pauli.Z)
    observables = []
    for operators in itertools.product(paulis, repeat=4):
        factors = tuple(
            (qubit, pauli)
            for qubit, pauli in enumerate(operators)
            if pauli is not qp.Pauli.I
        )
        observables.append(_observable(_term(1.0, *factors)))

    cpu = qp.expect_observables(program, observables, backend="native-cpu")
    gpu = qp.expect_observables(program, observables, backend="native-cuda")
    assert gpu.backend == "native-cuda"
    np.testing.assert_allclose(gpu.values, cpu.values, atol=2e-12, rtol=2e-12)


def test_cuda_pauli_reduction_recursively_reduces_multiple_blocks() -> None:
    if not qp.cuda_available():
        pytest.skip(qp.cuda_unavailable_reason())
    program = qp.Program(10)
    for qubit in range(10):
        program = qp.ry(program, 0.037 * (qubit + 1), qubit)
    for qubit in range(9):
        program = qp.cx(program, qubit, qubit + 1)
    observable = _observable(
        _term(0.61, (0, qp.Pauli.X), (4, qp.Pauli.Y), (9, qp.Pauli.Z)),
        _term(-0.23, (2, qp.Pauli.Y), (7, qp.Pauli.X)),
    )
    cpu = qp.expect_observable(program, observable, backend="native-cpu")
    gpu = qp.expect_observable(program, observable, backend="native-cuda")
    assert gpu.backend == "native-cuda"
    assert gpu.value == pytest.approx(cpu.value, abs=2e-12)
