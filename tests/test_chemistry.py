from __future__ import annotations

import math

import numpy as np
import pytest

import qupy as qp


def _number_operator(orbital: int, num_spin_orbitals: int) -> qp.Observable:
    return qp.jordan_wigner(
        num_spin_orbitals,
        [
            qp.fermion_term(
                1.0,
                [qp.fermion_creation(orbital), qp.fermion_annihilation(orbital)],
            )
        ],
    )


def test_jordan_wigner_number_operator_matches_occupation() -> None:
    number = _number_operator(0, 1)
    vacuum = qp.Program(1)
    occupied = qp.hartree_fock_state(1, 1)

    assert qp.expect(vacuum, number, backend="native-cpu").value == pytest.approx(0.0)
    assert qp.expect(occupied, number, backend="native-cpu").value == pytest.approx(1.0)


def test_jordan_wigner_hopping_matches_bonding_state() -> None:
    hopping = qp.jordan_wigner(
        2,
        [
            qp.fermion_term(
                1.0,
                [qp.fermion_creation(0), qp.fermion_annihilation(1)],
            ),
            qp.fermion_term(
                1.0,
                [qp.fermion_creation(1), qp.fermion_annihilation(0)],
            ),
        ],
    )

    bonding = qp.x(qp.Program(2), 1)
    bonding = qp.h(bonding, 0)
    bonding = qp.cx(bonding, 0, 1)
    energy = qp.expect(bonding, hopping, backend="native-cpu").value
    assert energy == pytest.approx(1.0, abs=1e-12)


def test_jordan_wigner_rejects_non_hermitian_operator() -> None:
    term = qp.fermion_term(
        1.0,
        [qp.fermion_creation(0), qp.fermion_annihilation(1)],
    )
    with pytest.raises(ValueError, match="must be Hermitian"):
        qp.jordan_wigner(2, [term])


def test_molecular_hamiltonian_matches_one_and_two_body_occupations() -> None:
    one_body = np.diag([1.25, -0.4]).astype(np.complex128)
    two_body = np.zeros((2, 2, 2, 2), dtype=np.complex128)
    two_body[0, 1, 0, 1] = 1.4
    hamiltonian = qp.molecular_hamiltonian(
        one_body,
        two_body,
        nuclear_repulsion=0.2,
    )

    vacuum = qp.hartree_fock_state(2, 0)
    one_electron = qp.hartree_fock_state(2, 1)
    two_electrons = qp.hartree_fock_state(2, 2)

    assert qp.expect(vacuum, hamiltonian, backend="native-cpu").value == pytest.approx(0.2)
    assert qp.expect(one_electron, hamiltonian, backend="native-cpu").value == pytest.approx(1.45)
    assert qp.expect(two_electrons, hamiltonian, backend="native-cpu").value == pytest.approx(1.75)


def test_molecular_hamiltonian_supports_complex_hermitian_hopping() -> None:
    one_body = np.array([[0.0, 1j], [-1j, 0.0]], dtype=np.complex128)
    hamiltonian = qp.molecular_hamiltonian(one_body)
    assert hamiltonian.terms
    assert all(math.isfinite(term.coefficient) for term in hamiltonian.terms)


def test_chemistry_observable_reuses_native_differentiation() -> None:
    hamiltonian = qp.molecular_hamiltonian(np.array([[1.0]], dtype=np.float64))
    template = qp.ry(qp.Program(1), 0.0, 0)
    theta = 0.37
    result = qp.value_and_grad(
        template,
        hamiltonian,
        [qp.ParameterSlot(0)],
        np.asarray([theta], dtype=np.float64),
        backend="native-cpu",
        method=qp.GradientMethod.ADJOINT,
    )

    assert result.value == pytest.approx(math.sin(theta / 2.0) ** 2, abs=1e-12)
    np.testing.assert_allclose(result.gradient, [0.5 * math.sin(theta)], atol=1e-12)


def test_hartree_fock_state_supports_explicit_occupations() -> None:
    state = qp.hartree_fock_state(4, 2, occupied_orbitals=[3, 1])
    assert [operation.qubits for operation in state.operations] == [[1], [3]]
    n1 = _number_operator(1, 4)
    n3 = _number_operator(3, 4)
    assert qp.expect(state, n1, backend="native-cpu").value == pytest.approx(1.0)
    assert qp.expect(state, n3, backend="native-cpu").value == pytest.approx(1.0)


def test_chemistry_inputs_fail_closed() -> None:
    with pytest.raises(ValueError, match="positive"):
        qp.jordan_wigner(0, [])
    with pytest.raises(ValueError, match="outside"):
        qp.jordan_wigner(
            1,
            [qp.fermion_term(1.0, [qp.fermion_creation(1)])],
        )
    with pytest.raises(ValueError, match="finite and non-negative"):
        qp.jordan_wigner(1, [], tolerance=-1.0)
    with pytest.raises(ValueError, match="finite"):
        qp.fermion_term(complex(float("nan"), 0.0), [])

    with pytest.raises(ValueError, match="non-empty square"):
        qp.molecular_hamiltonian(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="shape"):
        qp.molecular_hamiltonian(np.eye(2), np.zeros((2, 2, 2, 3)))
    with pytest.raises(ValueError, match="finite"):
        qp.molecular_hamiltonian(np.array([[float("nan")]]))

    with pytest.raises(ValueError, match="between 0"):
        qp.hartree_fock_state(2, 3)
    with pytest.raises(ValueError, match="exactly num_electrons"):
        qp.hartree_fock_state(3, 2, occupied_orbitals=[0])
    with pytest.raises(ValueError, match="unique"):
        qp.hartree_fock_state(3, 2, occupied_orbitals=[1, 1])
