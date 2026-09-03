from __future__ import annotations

import math

import pytest

import qupy as qp


def test_tensor_network_matches_dense_observable() -> None:
    program = qp.h(qp.Program(4), 0)
    program = qp.ry(program, 0.37, 1)
    program = qp.cx(program, 0, 3)
    program = qp.cz(program, 1, 2)
    program = qp.swap(program, 2, 3)
    observable = qp.observable(
        [
            qp.pauli_term(
                0.6,
                [qp.pauli(0, qp.Pauli.X), qp.pauli(3, qp.Pauli.Z)],
            ),
            qp.pauli_term(
                -0.2,
                [qp.pauli(1, qp.Pauli.Y), qp.pauli(2, qp.Pauli.X)],
            ),
            qp.pauli_term(0.3, []),
        ]
    )

    result = qp.tensor_network_expectation(program, observable)
    dense = qp.expect_observable(program, observable, backend="native-cpu")

    assert result.value == pytest.approx(dense.value, abs=2e-12)
    assert result.exact is True
    assert result.backend == "native-tn"
    assert result.method == "greedy-contraction"
    assert result.term_count == 3
    assert result.contractions > 0
    assert result.scalar_multiplications > 0.0


def test_tensor_network_handles_large_low_treewidth_program() -> None:
    qubits = 80
    program = qp.Program(qubits)
    final_angle = 0.0
    for qubit in range(qubits):
        angle = 0.013 * ((qubit % 7) + 1)
        program = qp.ry(program, angle, qubit)
        if qubit + 1 == qubits:
            final_angle = angle
    observable = qp.observable_from_z(qp.Z(qubits - 1))

    result = qp.tensor_network_expectation(
        program,
        observable,
        max_tensor_bytes=1 << 20,
    )

    assert result.value == pytest.approx(math.cos(final_angle), abs=2e-12)
    assert result.peak_tensor_rank <= 2
    assert result.peak_tensor_bytes <= 64


def test_tensor_network_reports_two_qubit_width_and_limit_validation() -> None:
    program = qp.cx(qp.Program(2), 0, 1)
    observable = qp.observable_from_z(qp.Z(0))

    result = qp.tensor_network_expectation(program, observable)
    assert result.peak_tensor_rank >= 4
    assert result.peak_tensor_bytes >= 256

    with pytest.raises(ValueError, match="max_tensor_bytes"):
        qp.tensor_network_expectation(program, observable, max_tensor_bytes=0)
