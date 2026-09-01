import math

import numpy as np
import pytest

import qupy as qp


def bell_program() -> qp.Program:
    program = qp.Program(2)
    program = qp.h(program, 0)
    return qp.cx(program, 0, 1)


def test_bell_statevector() -> None:
    state = qp.statevector(bell_program())
    expected = np.array([1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)], dtype=np.complex128)
    np.testing.assert_allclose(state.values, expected, atol=1e-12)
    assert state.backend == "numpy-statevector"


def test_bell_samples_only_correlated_states() -> None:
    result = qp.sample(bell_program(), shots=512, seed=7)
    assert set(result.counts()) == {"00", "11"}
    assert result.shots == 512


def test_bell_z_expectation_is_zero() -> None:
    result = qp.expect(bell_program(), qp.Z(0))
    assert result.value == pytest.approx(0.0, abs=1e-12)

def test_x_and_rx_gate_conventions() -> None:
    x_program = qp.x(qp.Program(1), 0)
    np.testing.assert_allclose(qp.statevector(x_program).values, [0, 1], atol=1e-12)

    rx_program = qp.rx(qp.Program(1), math.pi, 0)
    np.testing.assert_allclose(qp.statevector(rx_program).values, [0, -1j], atol=1e-12)


def test_invalid_qubit_is_rejected_before_execution() -> None:
    with pytest.raises(ValueError, match="outside this program"):
        qp.h(qp.Program(2), 2)

    with pytest.raises(ValueError, match="same qubit twice"):
        qp.cx(qp.Program(2), 0, 0)


def test_invalid_execution_options_are_rejected() -> None:
    with pytest.raises(ValueError, match="shots must be at least 1"):
        qp.sample(qp.Program(1), shots=0)

    with pytest.raises(ValueError, match="unknown backend"):
        qp.statevector(qp.Program(1), backend="missing")  # type: ignore[arg-type]