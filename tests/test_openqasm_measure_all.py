from __future__ import annotations

import pytest

import qupy as qp


def test_whole_register_measurement_lowers_to_indexed_circuit_measurements() -> None:
    circuit = qp.Circuit.from_openqasm3(
        """
        OPENQASM 3.1;
        qubit[3] data;
        bit[3] result;
        result = measure data;
        """
    )

    assert circuit.num_qubits == 3
    assert circuit.num_clbits == 3
    assert [instruction.name for instruction in circuit.instructions] == [
        "measure",
        "measure",
        "measure",
    ]
    assert [instruction.qubits for instruction in circuit.instructions] == [[0], [1], [2]]
    assert [instruction.classical_bits for instruction in circuit.instructions] == [
        [0],
        [1],
        [2],
    ]


def test_whole_register_measurement_rejects_mismatched_register_sizes() -> None:
    with pytest.raises(ValueError, match="equal quantum and classical register sizes"):
        qp.Circuit.from_openqasm3(
            """
            OPENQASM 3.1;
            qubit[2] q;
            bit[1] c;
            c = measure q;
            """
        )


def test_conditional_whole_register_measurement_fails_closed() -> None:
    with pytest.raises(ValueError, match="conditional whole-register measurement"):
        qp.Circuit.from_openqasm3(
            """
            OPENQASM 3.1;
            qubit[2] q;
            bit[2] c;
            if (c[0] == 1) { c = measure q; }
            """
        )
