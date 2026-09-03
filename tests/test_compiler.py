from __future__ import annotations

import pytest

import qupy as qp


def _full_target(qubits: int = 3) -> qp.HardwareTarget:
    return qp.HardwareTarget(
        "full",
        qubits,
        [
            qp.CircuitOperationCode.H,
            qp.CircuitOperationCode.X,
            qp.CircuitOperationCode.Y,
            qp.CircuitOperationCode.Z,
            qp.CircuitOperationCode.RX,
            qp.CircuitOperationCode.RY,
            qp.CircuitOperationCode.RZ,
        ],
        [
            qp.CircuitOperationCode.CX,
            qp.CircuitOperationCode.CZ,
            qp.CircuitOperationCode.SWAP,
        ],
        measurement=True,
        mid_circuit_measurement=True,
        reset=True,
        dynamic_control=True,
    )


def test_compile_routes_long_range_interaction_and_tracks_layout() -> None:
    target = qp.HardwareTarget(
        "line",
        3,
        [qp.CircuitOperationCode.H],
        [qp.CircuitOperationCode.CX],
        [qp.Coupling(0, 1), qp.Coupling(1, 2)],
    )
    result = qp.compile(qp.Circuit(2).cx(0, 1), target, initial_layout=[0, 2], optimization_level=0)

    assert result.initial_layout == [0, 2]
    assert result.final_layout == [1, 2]
    assert result.inserted_swaps == 1
    assert result.routed_operations == 2
    assert result.compiled_operations == 4
    assert result.decompositions == 1
    assert all(instruction.code is not qp.CircuitOperationCode.SWAP for instruction in result.circuit.instructions)
    assert result.target_fingerprint == target.fingerprint


def test_compile_preserves_static_layout_for_conditional_routing() -> None:
    target = qp.HardwareTarget(
        "dynamic-line",
        3,
        [qp.CircuitOperationCode.X],
        [qp.CircuitOperationCode.CX, qp.CircuitOperationCode.SWAP],
        [qp.Coupling(0, 1), qp.Coupling(1, 2)],
        dynamic_control=True,
    )
    condition = qp.ClassicalCondition(0, True)
    result = qp.compile_circuit(
        qp.Circuit(2, 1).cx(0, 1, condition),
        target,
        initial_layout=[0, 2],
        optimization_level=0,
    )

    assert result.inserted_swaps == 2
    assert result.final_layout == [0, 2]
    assert [instruction.code for instruction in result.circuit.instructions] == [
        qp.CircuitOperationCode.SWAP,
        qp.CircuitOperationCode.CX,
        qp.CircuitOperationCode.SWAP,
    ]
    assert all(instruction.condition is not None for instruction in result.circuit.instructions)


def test_compile_translates_cx_to_native_cz_basis() -> None:
    target = qp.HardwareTarget(
        "cz-native",
        2,
        [qp.CircuitOperationCode.H],
        [qp.CircuitOperationCode.CZ],
    )
    result = qp.compile(qp.Circuit(2).cx(0, 1), target, initial_layout=[0, 1], optimization_level=0)

    assert [instruction.code for instruction in result.circuit.instructions] == [
        qp.CircuitOperationCode.H,
        qp.CircuitOperationCode.CZ,
        qp.CircuitOperationCode.H,
    ]
    assert result.decompositions == 1


def test_compile_distinguishes_terminal_and_mid_circuit_measurement() -> None:
    terminal_target = qp.HardwareTarget(
        "terminal",
        1,
        [qp.CircuitOperationCode.X],
        [],
        measurement=True,
    )
    qp.compile(qp.Circuit(1, 1).measure(0, 0), terminal_target, initial_layout=[0])

    mid = qp.Circuit(1, 1).measure(0, 0).x(0)
    with pytest.raises(ValueError, match="mid-circuit measurement"):
        qp.compile(mid, terminal_target, initial_layout=[0])

    mid_target = qp.HardwareTarget(
        "mid",
        1,
        [qp.CircuitOperationCode.X],
        [],
        measurement=True,
        mid_circuit_measurement=True,
    )
    qp.compile(mid, mid_target, initial_layout=[0])


def test_compile_schedules_classical_feed_forward() -> None:
    target = qp.HardwareTarget(
        "timed",
        2,
        [qp.CircuitOperationCode.H, qp.CircuitOperationCode.X],
        [qp.CircuitOperationCode.CX],
        measurement=True,
        mid_circuit_measurement=True,
        dynamic_control=True,
        durations=[
            qp.OperationDuration(qp.CircuitOperationCode.H, 10.0),
            qp.OperationDuration(qp.CircuitOperationCode.X, 20.0),
            qp.OperationDuration(qp.CircuitOperationCode.CX, 100.0),
            qp.OperationDuration(qp.CircuitOperationCode.MEASURE, 50.0),
        ],
    )
    circuit = qp.Circuit(2, 1).h(0).h(1).cx(0, 1).measure(0, 0)
    circuit = circuit.x(1, qp.ClassicalCondition(0, True))
    result = qp.compile(circuit, target, initial_layout=[0, 1], optimization_level=0)

    assert result.depth == 4
    assert result.duration_ns == pytest.approx(180.0)
    assert [entry.start_ns for entry in result.schedule] == pytest.approx(
        [0.0, 0.0, 10.0, 110.0, 160.0]
    )


def test_compile_optimization_and_fail_closed_validation() -> None:
    adjacent = qp.Circuit(1).h(0).h(0)
    assert qp.compile(adjacent, _full_target(1), initial_layout=[0], optimization_level=0).compiled_operations == 2
    assert qp.compile(adjacent, _full_target(1), initial_layout=[0], optimization_level=1).compiled_operations == 0

    with pytest.raises(ValueError, match="dynamic"):
        qp.compile(
            qp.Circuit(1, 1).x(0, qp.ClassicalCondition(0, True)),
            qp.HardwareTarget("static", 1, [qp.CircuitOperationCode.X], []),
            initial_layout=[0],
        )

    disconnected = qp.HardwareTarget(
        "disconnected",
        3,
        [qp.CircuitOperationCode.H],
        [qp.CircuitOperationCode.CX],
        [qp.Coupling(0, 1)],
    )
    with pytest.raises(ValueError, match="cannot connect"):
        qp.compile(
            qp.Circuit(2).cx(0, 1),
            disconnected,
            initial_layout=[0, 2],
            optimization_level=0,
        )
