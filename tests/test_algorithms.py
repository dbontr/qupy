from __future__ import annotations

import math

import numpy as np
import pytest

import qupy as qp


def _z_observable(qubit: int = 0) -> qp.Observable:
    return qp.Observable([qp.PauliTerm(1.0, [qp.PauliFactor(qubit, qp.Pauli.Z)])])


def test_hardware_efficient_ansatz_tracks_native_parameter_slots() -> None:
    template = qp.hardware_efficient_ansatz(
        3,
        2,
        rotations=("ry", "rz"),
        entanglement="ring",
    )

    assert template.parameter_count == 12
    assert len(template.program.operations) == 18
    assert len(set(template.parameter_names)) == template.parameter_count
    assert template.parameter_names[0] == "layer0.q0.ry"
    assert template.parameter_names[-1] == "layer1.q2.rz"

    values = np.linspace(-0.5, 0.6, template.parameter_count)
    bound = template.bind(values)
    for slot, value in zip(template.slots, values, strict=True):
        assert bound.operations[slot.operation_index].parameters[0] == pytest.approx(value)

    named = template.bind_named(
        {name: float(value) for name, value in zip(template.parameter_names, values, strict=True)}
    )
    assert named.fingerprint == bound.fingerprint


def test_variational_template_slots_feed_native_adjoint_gradient() -> None:
    template = qp.hardware_efficient_ansatz(
        1,
        1,
        rotations=("ry",),
        entanglement="none",
    )
    theta = 0.371
    result = qp.value_and_grad(
        template.program,
        _z_observable(),
        list(template.slots),
        np.asarray([theta], dtype=np.float64),
        backend="native-cpu",
        method=qp.GradientMethod.ADJOINT,
    )

    assert result.value == pytest.approx(math.cos(theta), abs=1e-12)
    np.testing.assert_allclose(result.gradient, [-math.sin(theta)], atol=1e-12)


def test_variational_template_validates_bindings_and_construction() -> None:
    template = qp.hardware_efficient_ansatz(2, 1, rotations=("rx",), entanglement="linear")
    with pytest.raises(ValueError, match="exactly 2"):
        template.bind([0.1])
    with pytest.raises(ValueError, match="finite"):
        template.bind([0.1, float("nan")])
    with pytest.raises(ValueError, match="missing"):
        template.bind_named({template.parameter_names[0]: 0.1})
    with pytest.raises(ValueError, match="positive"):
        qp.hardware_efficient_ansatz(0, 1)
    with pytest.raises(ValueError, match="positive"):
        qp.hardware_efficient_ansatz(2, 0)
    with pytest.raises(ValueError, match="rotations"):
        qp.hardware_efficient_ansatz(2, 1, rotations=("u3",))
    with pytest.raises(ValueError, match="entanglement"):
        qp.hardware_efficient_ansatz(2, 1, entanglement="all-to-all")


def test_qft_has_uniform_zero_state_and_inverse_round_trips_prepared_state() -> None:
    fourier = qp.qft(3)
    state = qp.statevector(fourier, backend="native-cpu").values
    np.testing.assert_allclose(np.abs(state), np.full(8, 1.0 / math.sqrt(8.0)), atol=1e-12)

    prepared = qp.Program(3)
    prepared = qp.x(prepared, 0)
    prepared = qp.ry(prepared, 0.43, 1)
    prepared = qp.x(prepared, 2)
    reference = qp.statevector(prepared, backend="native-cpu").values.copy()

    transformed = qp.append_qft(prepared)
    restored = qp.append_qft(transformed, inverse=True)
    actual = qp.statevector(restored, backend="native-cpu").values
    np.testing.assert_allclose(actual, reference, atol=2e-12)


def test_qft_selected_qubits_round_trip_without_touching_spectators() -> None:
    prepared = qp.Program(4)
    prepared = qp.x(prepared, 0)
    prepared = qp.ry(prepared, -0.29, 1)
    prepared = qp.x(prepared, 3)
    reference = qp.statevector(prepared, backend="native-cpu").values.copy()

    transformed = qp.append_qft(prepared, [1, 3])
    restored = qp.append_qft(transformed, [1, 3], inverse=True)
    actual = qp.statevector(restored, backend="native-cpu").values
    np.testing.assert_allclose(actual, reference, atol=2e-12)

    with pytest.raises(ValueError, match="at least one"):
        qp.append_qft(prepared, [])
    with pytest.raises(ValueError, match="unique"):
        qp.append_qft(prepared, [1, 1])
    with pytest.raises(ValueError, match="outside"):
        qp.append_qft(prepared, [4])


def test_single_qubit_pauli_evolution_matches_analytic_x_and_y() -> None:
    time = 0.4
    x_term = qp.PauliTerm(0.5, [qp.PauliFactor(0, qp.Pauli.X)])
    x_program = qp.append_pauli_evolution(qp.Program(1), x_term, time)
    x_state = qp.statevector(x_program, backend="native-cpu").values
    theta = time * 0.5
    np.testing.assert_allclose(
        x_state,
        [math.cos(theta), -1j * math.sin(theta)],
        atol=1e-12,
    )

    y_term = qp.PauliTerm(1.0, [qp.PauliFactor(0, qp.Pauli.Y)])
    y_program = qp.append_pauli_evolution(qp.Program(1), y_term, time)
    y_state = qp.statevector(y_program, backend="native-cpu").values
    np.testing.assert_allclose(
        y_state,
        [math.cos(time), math.sin(time)],
        atol=1e-12,
    )


def test_multi_qubit_pauli_evolution_matches_analytic_zz_phase() -> None:
    time = 0.27
    program = qp.h(qp.Program(2), 0)
    program = qp.h(program, 1)
    term = qp.PauliTerm(
        1.0,
        [qp.PauliFactor(0, qp.Pauli.Z), qp.PauliFactor(1, qp.Pauli.Z)],
    )
    evolved = qp.append_pauli_evolution(program, term, time)
    state = qp.statevector(evolved, backend="native-cpu").values
    expected = 0.5 * np.asarray(
        [
            np.exp(-1j * time),
            np.exp(1j * time),
            np.exp(1j * time),
            np.exp(-1j * time),
        ],
        dtype=np.complex128,
    )
    np.testing.assert_allclose(state, expected, atol=1e-12)


def test_pauli_evolution_validates_time_and_program_extent() -> None:
    term = qp.PauliTerm(1.0, [qp.PauliFactor(1, qp.Pauli.Z)])
    with pytest.raises(ValueError, match="outside"):
        qp.append_pauli_evolution(qp.Program(1), term, 0.1)
    with pytest.raises(ValueError, match="finite"):
        qp.append_pauli_evolution(qp.Program(2), term, float("inf"))


def test_hamiltonian_evolution_is_exact_for_commuting_pauli_terms() -> None:
    first = qp.PauliTerm(0.3, [qp.PauliFactor(0, qp.Pauli.Z)])
    second = qp.PauliTerm(
        -0.7,
        [qp.PauliFactor(0, qp.Pauli.Z), qp.PauliFactor(1, qp.Pauli.Z)],
    )
    hamiltonian = qp.Observable([first, second])
    prepared = qp.h(qp.Program(2), 0)
    prepared = qp.ry(prepared, 0.41, 1)
    time = 0.63

    reference = qp.append_pauli_evolution(prepared, first, time)
    reference = qp.append_pauli_evolution(reference, second, time)
    actual = qp.append_hamiltonian_evolution(
        prepared,
        hamiltonian,
        time,
        steps=4,
        order=1,
    )

    np.testing.assert_allclose(
        qp.statevector(actual, backend="native-cpu").values,
        qp.statevector(reference, backend="native-cpu").values,
        atol=2e-12,
    )


def test_second_order_hamiltonian_evolution_is_time_symmetric() -> None:
    hamiltonian = qp.Observable(
        [
            qp.PauliTerm(0.8, [qp.PauliFactor(0, qp.Pauli.X)]),
            qp.PauliTerm(-0.35, [qp.PauliFactor(0, qp.Pauli.Z)]),
        ]
    )
    prepared = qp.ry(qp.Program(1), 0.29, 0)
    reference = qp.statevector(prepared, backend="native-cpu").values.copy()

    evolved = qp.append_hamiltonian_evolution(
        prepared,
        hamiltonian,
        0.71,
        steps=3,
        order=2,
    )
    restored = qp.append_hamiltonian_evolution(
        evolved,
        hamiltonian,
        -0.71,
        steps=3,
        order=2,
    )
    np.testing.assert_allclose(
        qp.statevector(restored, backend="native-cpu").values,
        reference,
        atol=3e-12,
    )


def test_hamiltonian_evolution_validates_product_formula_contract() -> None:
    hamiltonian = _z_observable()
    with pytest.raises(ValueError, match="time must be finite"):
        qp.append_hamiltonian_evolution(qp.Program(1), hamiltonian, float("nan"))
    with pytest.raises(ValueError, match="steps must be positive"):
        qp.append_hamiltonian_evolution(qp.Program(1), hamiltonian, 0.1, steps=0)
    with pytest.raises(TypeError, match="steps must be an integer"):
        qp.append_hamiltonian_evolution(qp.Program(1), hamiltonian, 0.1, steps=True)
    with pytest.raises(ValueError, match="order must be 1 or 2"):
        qp.append_hamiltonian_evolution(qp.Program(1), hamiltonian, 0.1, order=3)
    with pytest.raises(TypeError, match="order must be an integer"):
        qp.append_hamiltonian_evolution(qp.Program(1), hamiltonian, 0.1, order=True)


def test_maxcut_hamiltonian_matches_weighted_cut_values_and_is_canonical() -> None:
    hamiltonian = qp.maxcut_hamiltonian(
        3,
        [(1, 2), (0, 1)],
        weights=[3.0, 2.0],
    )
    reordered = qp.maxcut_hamiltonian(
        3,
        [(0, 1), (2, 1)],
        weights=[2.0, 3.0],
    )
    assert hamiltonian.fingerprint == reordered.fingerprint

    uncut = qp.Program(3)
    fully_cut = qp.x(qp.Program(3), 1)
    assert qp.expect(uncut, hamiltonian, backend="native-cpu").value == pytest.approx(0.0)
    assert qp.expect(fully_cut, hamiltonian, backend="native-cpu").value == pytest.approx(5.0)


def test_qaoa_maxcut_single_edge_reaches_p1_optimum() -> None:
    hamiltonian = qp.maxcut_hamiltonian(2, [(0, 1)])
    program = qp.qaoa_maxcut_program(
        2,
        [(0, 1)],
        [math.pi / 2.0],
        [math.pi / 8.0],
    )
    energy = qp.expect(program, hamiltonian, backend="native-cpu").value
    assert energy == pytest.approx(1.0, abs=2e-12)


def test_maxcut_and_qaoa_validate_graph_and_layer_inputs() -> None:
    with pytest.raises(ValueError, match="distinct"):
        qp.maxcut_hamiltonian(2, [(0, 0)])
    with pytest.raises(ValueError, match="duplicate"):
        qp.maxcut_hamiltonian(2, [(0, 1), (1, 0)])
    with pytest.raises(ValueError, match="one value per edge"):
        qp.maxcut_hamiltonian(3, [(0, 1), (1, 2)], weights=[1.0])
    with pytest.raises(ValueError, match="non-negative"):
        qp.maxcut_hamiltonian(2, [(0, 1)], weights=[-1.0])
    with pytest.raises(ValueError, match="outside"):
        qp.maxcut_hamiltonian(2, [(0, 2)])

    with pytest.raises(ValueError, match="at least one layer"):
        qp.qaoa_maxcut_program(2, [(0, 1)], [], [])
    with pytest.raises(ValueError, match="same length"):
        qp.qaoa_maxcut_program(2, [(0, 1)], [0.1], [0.2, 0.3])
    with pytest.raises(ValueError, match="finite"):
        qp.qaoa_maxcut_program(2, [(0, 1)], [float("inf")], [0.2])
