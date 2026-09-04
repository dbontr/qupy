from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from . import _native

_PauliKey = tuple[tuple[int, str], ...]
_LOCAL_PAULI_PRODUCT: dict[tuple[str, str], tuple[complex, str | None]] = {
    ("X", "X"): (1.0, None),
    ("X", "Y"): (1j, "Z"),
    ("X", "Z"): (-1j, "Y"),
    ("Y", "X"): (-1j, "Z"),
    ("Y", "Y"): (1.0, None),
    ("Y", "Z"): (1j, "X"),
    ("Z", "X"): (1j, "Y"),
    ("Z", "Y"): (-1j, "X"),
    ("Z", "Z"): (1.0, None),
}
_NATIVE_PAULI = {"X": _native.Pauli.X, "Y": _native.Pauli.Y, "Z": _native.Pauli.Z}


@dataclass(frozen=True, slots=True)
class FermionLadder:
    """One fermionic creation or annihilation operator."""

    orbital: int
    creation: bool

    def __post_init__(self) -> None:
        if isinstance(self.orbital, bool) or not isinstance(self.orbital, int):
            raise TypeError("orbital must be an integer")
        if self.orbital < 0:
            raise ValueError("orbital must be non-negative")
        if not isinstance(self.creation, bool):
            raise TypeError("creation must be a boolean")


@dataclass(frozen=True, slots=True)
class FermionTerm:
    """Coefficient times an ordered product of fermionic ladder operators."""

    coefficient: complex
    operators: tuple[FermionLadder, ...]

    def __post_init__(self) -> None:
        coefficient = complex(self.coefficient)
        if not math.isfinite(coefficient.real) or not math.isfinite(coefficient.imag):
            raise ValueError("fermion coefficient must be finite")
        operators = tuple(self.operators)
        if any(not isinstance(operator, FermionLadder) for operator in operators):
            raise TypeError("operators must contain FermionLadder values")
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "operators", operators)


def fermion_creation(orbital: int) -> FermionLadder:
    """Construct a creation operator for one spin orbital."""
    return FermionLadder(orbital, True)


def fermion_annihilation(orbital: int) -> FermionLadder:
    """Construct an annihilation operator for one spin orbital."""
    return FermionLadder(orbital, False)


def fermion_term(
    coefficient: complex,
    operators: Sequence[FermionLadder],
) -> FermionTerm:
    """Construct an immutable fermionic term from an ordered operator product."""
    return FermionTerm(coefficient, tuple(operators))


def _tolerance(value: float) -> float:
    tolerance = float(value)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    return tolerance


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _multiply_pauli_keys(left: _PauliKey, right: _PauliKey) -> tuple[complex, _PauliKey]:
    factors = dict(left)
    phase = 1.0 + 0.0j
    for qubit, pauli in right:
        current = factors.get(qubit)
        if current is None:
            factors[qubit] = pauli
            continue
        local_phase, product = _LOCAL_PAULI_PRODUCT[(current, pauli)]
        phase *= local_phase
        if product is None:
            del factors[qubit]
        else:
            factors[qubit] = product
    return phase, tuple(sorted(factors.items()))


def _ladder_components(operator: FermionLadder) -> tuple[tuple[complex, _PauliKey], ...]:
    parity = tuple((qubit, "Z") for qubit in range(operator.orbital))
    x_key = tuple(sorted((*parity, (operator.orbital, "X"))))
    y_key = tuple(sorted((*parity, (operator.orbital, "Y"))))
    y_coefficient = -0.5j if operator.creation else 0.5j
    return ((0.5, x_key), (y_coefficient, y_key))


def _map_fermion_term(term: FermionTerm, tolerance: float) -> dict[_PauliKey, complex]:
    mapped: dict[_PauliKey, complex] = {(): term.coefficient}
    for operator in term.operators:
        next_terms: dict[_PauliKey, complex] = {}
        for left_key, left_coefficient in mapped.items():
            for right_coefficient, right_key in _ladder_components(operator):
                phase, product_key = _multiply_pauli_keys(left_key, right_key)
                coefficient = left_coefficient * right_coefficient * phase
                next_terms[product_key] = next_terms.get(product_key, 0.0j) + coefficient
        mapped = {
            key: coefficient
            for key, coefficient in next_terms.items()
            if abs(coefficient) > tolerance
        }
    return mapped


def jordan_wigner(
    num_spin_orbitals: int,
    terms: Sequence[FermionTerm],
    *,
    tolerance: float = 1e-12,
) -> _native.Observable:
    """Map a Hermitian fermionic operator to a real Pauli-sum Observable."""

    orbital_count = _positive_integer(num_spin_orbitals, "num_spin_orbitals")
    threshold = _tolerance(tolerance)
    accumulator: dict[_PauliKey, complex] = {}
    for index, term in enumerate(terms):
        if not isinstance(term, FermionTerm):
            raise TypeError(f"terms[{index}] must be a FermionTerm")
        for operator in term.operators:
            if operator.orbital >= orbital_count:
                raise ValueError(
                    f"terms[{index}] references orbital {operator.orbital} outside num_spin_orbitals"
                )
        for key, coefficient in _map_fermion_term(term, threshold).items():
            accumulator[key] = accumulator.get(key, 0.0j) + coefficient

    pauli_terms: list[_native.PauliTerm] = []
    for key in sorted(accumulator):
        coefficient = accumulator[key]
        if abs(coefficient) <= threshold:
            continue
        if abs(coefficient.imag) > threshold:
            raise ValueError(
                "Jordan-Wigner mapping produced a non-real Pauli coefficient; "
                "the fermionic operator must be Hermitian within tolerance"
            )
        real = float(coefficient.real)
        if abs(real) <= threshold:
            continue
        factors = [_native.PauliFactor(qubit, _NATIVE_PAULI[pauli]) for qubit, pauli in key]
        pauli_terms.append(_native.PauliTerm(real, factors))
    return _native.Observable(pauli_terms)


def _integral_array(values: npt.ArrayLike, *, name: str, ndim: int) -> npt.NDArray[np.complex128]:
    array = np.asarray(values, dtype=np.complex128)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if not np.all(np.isfinite(array.real)) or not np.all(np.isfinite(array.imag)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def molecular_hamiltonian(
    one_body_integrals: npt.ArrayLike,
    two_body_integrals: npt.ArrayLike | None = None,
    *,
    nuclear_repulsion: float = 0.0,
    tolerance: float = 1e-12,
) -> _native.Observable:
    """Map spin-orbital molecular integrals to a Jordan-Wigner Observable.

    The two-body convention is 0.5 * g[p,q,r,s] a†_p a†_q a_s a_r.
    """
    threshold = _tolerance(tolerance)
    one_body = _integral_array(one_body_integrals, name="one_body_integrals", ndim=2)
    if one_body.shape[0] == 0 or one_body.shape[0] != one_body.shape[1]:
        raise ValueError("one_body_integrals must be a non-empty square matrix")
    orbital_count = int(one_body.shape[0])

    try:
        nuclear_energy = float(nuclear_repulsion)
    except (TypeError, ValueError) as exc:
        raise TypeError("nuclear_repulsion must be a real number") from exc
    if not math.isfinite(nuclear_energy):
        raise ValueError("nuclear_repulsion must be finite")

    terms: list[FermionTerm] = []
    if abs(nuclear_energy) > threshold:
        terms.append(FermionTerm(nuclear_energy, ()))

    for row in np.argwhere(np.abs(one_body) > threshold):
        p, q = (int(value) for value in row)
        terms.append(
            FermionTerm(
                complex(one_body[p, q]),
                (fermion_creation(p), fermion_annihilation(q)),
            )
        )

    if two_body_integrals is not None:
        two_body = _integral_array(two_body_integrals, name="two_body_integrals", ndim=4)
        expected_shape = (orbital_count,) * 4
        if two_body.shape != expected_shape:
            raise ValueError(
                "two_body_integrals must have shape "
                f"({orbital_count}, {orbital_count}, {orbital_count}, {orbital_count})"
            )

        for row in np.argwhere(np.abs(two_body) > threshold):
            p, q, r, s = (int(value) for value in row)
            coefficient = 0.5 * complex(two_body[p, q, r, s])
            if abs(coefficient) <= threshold:
                continue
            terms.append(
                FermionTerm(
                    coefficient,
                    (
                        fermion_creation(p),
                        fermion_creation(q),
                        fermion_annihilation(s),
                        fermion_annihilation(r),
                    ),
                )
            )

    return jordan_wigner(orbital_count, terms, tolerance=threshold)


def hartree_fock_state(
    num_spin_orbitals: int,
    num_electrons: int,
    *,
    occupied_orbitals: Sequence[int] | None = None,
) -> _native.Program:
    """Prepare a computational-basis Hartree-Fock occupation state."""
    orbital_count = _positive_integer(num_spin_orbitals, "num_spin_orbitals")

    if isinstance(num_electrons, bool) or not isinstance(num_electrons, int):
        raise TypeError("num_electrons must be an integer")
    if num_electrons < 0 or num_electrons > orbital_count:
        raise ValueError("num_electrons must be between 0 and num_spin_orbitals")

    if occupied_orbitals is None:
        occupied = tuple(range(num_electrons))
    else:
        occupied = tuple(occupied_orbitals)
        if len(occupied) != num_electrons:
            raise ValueError("occupied_orbitals must contain exactly num_electrons entries")
        for orbital in occupied:
            if isinstance(orbital, bool) or not isinstance(orbital, int):
                raise TypeError("occupied_orbitals must contain integers")
            if orbital < 0 or orbital >= orbital_count:
                raise ValueError("occupied_orbitals contains an orbital outside num_spin_orbitals")
        if len(set(occupied)) != len(occupied):
            raise ValueError("occupied_orbitals must be unique")

    program = _native.Program(orbital_count)
    for orbital in sorted(occupied):
        program = _native.x(program, orbital)
    return program


__all__ = [
    "FermionLadder",
    "FermionTerm",
    "fermion_annihilation",
    "fermion_creation",
    "fermion_term",
    "hartree_fock_state",
    "jordan_wigner",
    "molecular_hamiltonian",
]
