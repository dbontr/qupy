from __future__ import annotations

import math

import numpy as np
import pytest

import qupy as qp


def _z(qubit: int = 0) -> qp.Observable:
    return qp.observable_from_z(qp.Z(qubit))


def test_trajectory_expectation_matches_deterministic_noise() -> None:
    program = qp.x(qp.Program(1), 0)
    noisy = qp.NoisyProgram(
        program,
        [qp.NoiseInstruction(1, qp.bit_flip(0, 1.0))],
    )

    result = qp.trajectory_expectation(noisy, _z(), trajectories=32, seed=7)

    assert result.value == pytest.approx(1.0)
    assert result.standard_error == pytest.approx(0.0)
    assert result.trajectories == 32
    assert result.seed == 7
    assert result.state_bytes == 2 * np.dtype(np.complex128).itemsize
    assert result.exact is False
    assert result.backend == "native-cpu"
    assert result.method == "quantum-trajectory"


def test_trajectory_batch_matches_exact_density_with_reported_error() -> None:
    gamma = 0.25
    program = qp.x(qp.Program(1), 0)
    noisy = qp.NoisyProgram(
        program,
        [qp.NoiseInstruction(1, qp.amplitude_damping(0, gamma))],
    )

    result = qp.trajectory_expectations(
        noisy,
        [_z(), qp.observable([qp.pauli_term(2.0, [])])],
        trajectories=12000,
        seed=123456,
    )
    replay = qp.trajectory_expectations(
        noisy,
        [_z(), qp.observable([qp.pauli_term(2.0, [])])],
        trajectories=12000,
        seed=123456,
    )
    density = qp.density_matrix(noisy, backend="native-cpu")
    exact_z = float((density.values[0, 0] - density.values[1, 1]).real)

    assert result.values.shape == (2,)
    assert result.standard_errors.shape == (2,)
    assert result.values[0] == pytest.approx(exact_z, abs=0.035)
    assert result.values[1] == pytest.approx(2.0)
    assert 0.0 < result.standard_errors[0] < 0.025
    assert result.standard_errors[1] == pytest.approx(0.0)
    np.testing.assert_array_equal(result.values, replay.values)
    np.testing.assert_array_equal(result.standard_errors, replay.standard_errors)
    assert result.observable_count == 2


def test_trajectory_custom_kraus_and_insertion_order() -> None:
    reset_to_zero = qp.kraus_channel(
        0,
        [
            np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128),
            np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128),
        ],
    )
    program = qp.x(qp.Program(1), 0)
    reset = qp.NoisyProgram(program, [qp.NoiseInstruction(1, reset_to_zero)])
    before_gate = qp.NoisyProgram(
        program,
        [qp.NoiseInstruction(0, qp.bit_flip(0, 1.0))],
    )

    assert qp.trajectory_expectation(reset, _z(), 16, 5).value == pytest.approx(1.0)
    assert qp.trajectory_expectation(before_gate, _z(), 16, 5).value == pytest.approx(1.0)


def test_trajectory_state_memory_scales_as_statevector() -> None:
    qubits = 12
    noisy = qp.NoisyProgram(qp.Program(qubits), [])
    result = qp.trajectory_expectation(noisy, _z(), trajectories=2, seed=3)

    assert result.state_bytes == (1 << qubits) * np.dtype(np.complex128).itemsize
    density_bytes = (1 << (2 * qubits)) * np.dtype(np.complex128).itemsize
    assert result.state_bytes * (1 << qubits) == density_bytes


def test_trajectory_single_sample_and_validation() -> None:
    noisy = qp.NoisyProgram(qp.Program(1), [])
    single = qp.trajectory_expectation(noisy, _z(), trajectories=1, seed=9)

    assert single.value == pytest.approx(1.0)
    assert math.isnan(single.standard_error)

    with pytest.raises(ValueError, match="trajectory count"):
        qp.trajectory_expectation(noisy, _z(), trajectories=0, seed=1)
    with pytest.raises(ValueError, match="only native-cpu"):
        qp.trajectory_expectation(noisy, _z(), trajectories=8, seed=1, backend="native-cuda")
    with pytest.raises(ValueError, match="at least one observable"):
        qp.trajectory_expectations(noisy, [], trajectories=8, seed=1)
