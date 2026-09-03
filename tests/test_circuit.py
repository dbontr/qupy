from __future__ import annotations

import math

import numpy as np
import pytest

import qupy as qp


def test_unitary_circuit_lowers_to_existing_program_semantics() -> None:
    circuit = qp.Circuit(3)
    circuit = circuit.h(0).rx(0.25, 1).cx(0, 1).barrier([0, 1]).swap(1, 2)

    assert circuit.num_qubits == 3
    assert circuit.num_clbits == 0
    assert len(circuit.instructions) == 5
    assert circuit.instructions[0].code is qp.CircuitOperationCode.H
    assert circuit.instructions[3].code is qp.CircuitOperationCode.BARRIER
    assert circuit.fingerprint == circuit.fingerprint
    assert len(circuit.fingerprint) == 64

    lowered = circuit.to_program()
    reference = qp.h(qp.Program(3), 0)
    reference = qp.rx(reference, 0.25, 1)
    reference = qp.cx(reference, 0, 1)
    reference = qp.swap(reference, 1, 2)
    assert lowered.fingerprint == reference.fingerprint
    np.testing.assert_allclose(qp.statevector(lowered).values, qp.statevector(reference).values)

    restored = qp.Circuit.from_program(lowered, num_clbits=2)
    assert restored.num_clbits == 2
    assert restored.to_program().fingerprint == lowered.fingerprint


def test_dynamic_circuit_emits_openqasm_31() -> None:
    circuit = qp.Circuit(2, 1)
    circuit = circuit.h(0)
    circuit = circuit.cx(0, 1)
    circuit = circuit.measure(0, 0)
    circuit = circuit.x(1, qp.ClassicalCondition(0, True))
    circuit = circuit.reset(0, qp.ClassicalCondition(0, False))
    circuit = circuit.barrier([0, 1]).barrier()

    assert circuit.to_openqasm3() == (
        'OPENQASM 3.1;\n'
        'include "stdgates.inc";\n'
        'qubit[2] q;\n'
        'bit[1] c;\n'
        'h q[0];\n'
        'cx q[0], q[1];\n'
        'c[0] = measure q[0];\n'
        'if (c[0] == 1) {\n'
        '  x q[1];\n'
        '}\n'
        'if (c[0] == 0) {\n'
        '  reset q[0];\n'
        '}\n'
        'barrier q[0], q[1];\n'
        'barrier;\n'
    )
    assert circuit.instructions[2].name == "measure"
    assert circuit.instructions[2].classical_bits == [0]
    assert circuit.instructions[3].condition is not None
    assert circuit.instructions[3].condition.bit == 0
    assert circuit.instructions[3].condition.value is True


def test_openqasm_31_round_trip_preserves_circuit_identity() -> None:
    circuit = qp.Circuit(3, 2)
    circuit = circuit.h(0).x(1).y(2).z(0)
    circuit = circuit.rx(-0.125, 1).ry(3.25e-7, 2).rz(0.5, 0)
    circuit = circuit.cx(0, 1).cz(1, 2).swap(0, 2)
    circuit = circuit.measure(0, 0)
    circuit = circuit.measure(1, 1, qp.ClassicalCondition(0, True))
    circuit = circuit.x(2, qp.ClassicalCondition(1, False))
    circuit = circuit.reset(0, qp.ClassicalCondition(0, False))
    circuit = circuit.barrier([0, 2]).barrier()

    restored = qp.Circuit.from_openqasm3(circuit.to_openqasm3())

    assert restored.canonical_text == circuit.canonical_text
    assert restored.fingerprint == circuit.fingerprint
    assert restored.to_openqasm3() == circuit.to_openqasm3()


def test_openqasm_import_accepts_comments_and_named_registers() -> None:
    restored = qp.Circuit.from_openqasm3(
        """
        // register names are parser-local and do not affect Circuit identity
        OPENQASM 3;
        include "stdgates.inc";
        /* declarations */
        qubit[2] data;
        bit[1] flag;
        rx(-1.25e-2) data[0];
        flag[0] = measure data[0];
        if (flag[0] == 0) { z data[1]; }
        barrier data[0], data[1];
        """
    )

    assert restored.num_qubits == 2
    assert restored.num_clbits == 1
    assert len(restored.instructions) == 4
    assert restored.instructions[0].parameters[0] == pytest.approx(-0.0125)
    assert restored.instructions[2].condition is not None
    assert restored.instructions[2].condition.value is False


@pytest.mark.parametrize(
    "text",
    [
        "OPENQASM 2.0; qubit[1] q; h q[0];",
        'OPENQASM 3.1; include "other.inc"; qubit[1] q; h q[0];',
        "OPENQASM 3.1; qubit[1] q; qubit[1] r; h q[0];",
        "OPENQASM 3.1; qubit[1] q; h q[1];",
        "OPENQASM 3.1; qubit[1] q; t q[0];",
        "OPENQASM 3.1; qubit[1] q; rx(1e999) q[0];",
        "OPENQASM 3.1; qubit[1] q; bit[1] c; if (c[0] == 2) { x q[0]; }",
        "OPENQASM 3.1; qubit[1] q; bit[1] c; if (c[0] == 1) { x q[0]; z q[0]; }",
        "OPENQASM 3.1; qubit[1] q; bit[1] c; if (c[0] == 1) { barrier q[0]; }",
        "OPENQASM 3.1; qubit[1] q; gate custom a { x a; } custom q[0];",
        "OPENQASM 3.1; qubit[1] q; /* unterminated",
    ],
)
def test_openqasm_import_fails_closed(text: str) -> None:
    with pytest.raises(ValueError):
        qp.Circuit.from_openqasm3(text)


def test_openqasm_syntax_errors_report_source_location() -> None:
    with pytest.raises(ValueError, match=r"line 3, column"):
        qp.Circuit.from_openqasm3(
            "OPENQASM 3.1;\nqubit[1] q;\nh q[0]\n"
        )


def test_circuit_identity_is_versioned_and_parameter_exact() -> None:
    first = qp.Circuit(1, 1).ry(math.pi / 7.0, 0).measure(0, 0)
    second = qp.Circuit(1, 1).ry(math.pi / 7.0, 0).measure(0, 0)
    different = qp.Circuit(1, 1).ry(math.pi / 7.0, 0).measure(
        0, 0, qp.ClassicalCondition(0, True)
    )

    assert qp.circuit_ir_version() == 1
    assert first.canonical_text.startswith("qupy-circuit 1\nqubits 1\nclbits 1\n")
    assert first.canonical_text == second.canonical_text
    assert first.fingerprint == second.fingerprint
    assert first.fingerprint != different.fingerprint


def test_circuit_validation_fails_closed() -> None:
    with pytest.raises(ValueError, match="num_qubits"):
        qp.Circuit(0)
    with pytest.raises(ValueError, match="classical bit"):
        qp.Circuit(1, 1).measure(0, 1)
    with pytest.raises(ValueError, match="condition bit"):
        qp.Circuit(1, 1).x(0, qp.ClassicalCondition(1, True))
    with pytest.raises(ValueError, match="same qubit"):
        qp.Circuit(2).cx(0, 0)
    with pytest.raises(ValueError, match="barrier qubits"):
        qp.Circuit(2).barrier([0, 0])
    with pytest.raises(ValueError, match="finite"):
        qp.Circuit(1).rz(float("nan"), 0)


@pytest.mark.parametrize(
    "circuit",
    [
        qp.Circuit(1, 1).measure(0, 0),
        qp.Circuit(1).reset(0),
        qp.Circuit(1, 1).x(0, qp.ClassicalCondition(0, True)),
    ],
)
def test_hardware_control_does_not_silently_lower_to_unitary_program(circuit: qp.Circuit) -> None:
    with pytest.raises(ValueError, match="hardware control"):
        circuit.to_program()
