from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import numpy.typing as npt

from . import _native
from .algorithms import append_pauli_evolution

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


@dataclass(frozen=True, slots=True)
class FermionicExcitation:
    """Canonical occupied-to-virtual single or double fermionic excitation."""

    occupied: tuple[int, ...]
    virtual: tuple[int, ...]

    def __post_init__(self) -> None:
        occupied = tuple(self.occupied)
        virtual = tuple(self.virtual)
        if len(occupied) not in (1, 2) or len(virtual) != len(occupied):
            raise ValueError("excitation rank must be one or two with matching orbital counts")
        for label, orbitals in (("occupied", occupied), ("virtual", virtual)):
            for orbital in orbitals:
                if isinstance(orbital, bool) or not isinstance(orbital, int):
                    raise TypeError(f"{label} orbitals must be integers")
                if orbital < 0:
                    raise ValueError(f"{label} orbitals must be non-negative")
            if len(set(orbitals)) != len(orbitals):
                raise ValueError(f"{label} orbitals must be unique")
        if set(occupied) & set(virtual):
            raise ValueError("occupied and virtual orbitals must be disjoint")
        object.__setattr__(self, "occupied", tuple(sorted(occupied)))
        object.__setattr__(self, "virtual", tuple(sorted(virtual)))

    @property
    def rank(self) -> int:
        return len(self.occupied)


@dataclass(frozen=True, slots=True)
class UccsdTemplate:
    """Factorized UCCSD program plus the linear map from amplitudes to gate slots."""

    program: _native.Program
    slots: tuple[_native.ParameterSlot, ...]
    slot_parameter_indices: tuple[int, ...]
    slot_scales: tuple[float, ...]
    parameter_names: tuple[str, ...]
    excitations: tuple[FermionicExcitation, ...]

    def __post_init__(self) -> None:
        slots = tuple(self.slots)
        indices = tuple(self.slot_parameter_indices)
        scales = tuple(float(scale) for scale in self.slot_scales)
        names = tuple(self.parameter_names)
        excitations = tuple(self.excitations)
        if len(slots) != len(indices) or len(slots) != len(scales):
            raise ValueError(
                "slots, slot_parameter_indices, and slot_scales must have equal length"
            )
        if len(names) != len(excitations):
            raise ValueError("parameter_names and excitations must have equal length")
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("parameter_names must be unique and non-empty")
        if any(index < 0 or index >= len(excitations) for index in indices):
            raise ValueError("slot_parameter_indices contains an invalid parameter index")
        if not all(math.isfinite(scale) for scale in scales):
            raise ValueError("slot_scales must be finite")
        object.__setattr__(self, "slots", slots)
        object.__setattr__(self, "slot_parameter_indices", indices)
        object.__setattr__(self, "slot_scales", scales)
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "excitations", excitations)

    @property
    def parameter_count(self) -> int:
        return len(self.excitations)

    @property
    def gate_parameter_count(self) -> int:
        return len(self.slots)

    def expanded_parameters(self, values: Sequence[float]) -> npt.NDArray[np.float64]:
        amplitudes = tuple(float(value) for value in values)
        if len(amplitudes) != self.parameter_count:
            raise ValueError(f"values must contain exactly {self.parameter_count} parameters")
        if not all(math.isfinite(value) for value in amplitudes):
            raise ValueError("parameter values must be finite")
        return np.asarray(
            [
                scale * amplitudes[index]
                for index, scale in zip(self.slot_parameter_indices, self.slot_scales, strict=True)
            ],
            dtype=np.float64,
        )

    def bind(self, values: Sequence[float]) -> _native.Program:
        expanded = self.expanded_parameters(values)
        if self.gate_parameter_count == 0:
            return self.program
        return self.program.bind(list(self.slots), expanded.tolist())

    def bind_named(self, values: Mapping[str, float]) -> _native.Program:
        expected = set(self.parameter_names)
        supplied = set(values)
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        if missing or extra:
            details: list[str] = []
            if missing:
                details.append(f"missing {missing}")
            if extra:
                details.append(f"unexpected {extra}")
            raise ValueError("named parameters do not match UCCSD template: " + ", ".join(details))
        return self.bind([values[name] for name in self.parameter_names])

    def compress_gradient(self, gradient: npt.ArrayLike) -> npt.NDArray[np.float64]:
        gate_gradient = np.asarray(gradient, dtype=np.float64)
        if gate_gradient.shape != (self.gate_parameter_count,):
            raise ValueError(f"gradient must have shape ({self.gate_parameter_count},)")
        if not np.all(np.isfinite(gate_gradient)):
            raise ValueError("gradient must contain only finite values")
        result = np.zeros(self.parameter_count, dtype=np.float64)
        for value, index, scale in zip(
            gate_gradient,
            self.slot_parameter_indices,
            self.slot_scales,
            strict=True,
        ):
            result[index] += scale * float(value)
        return result


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


def _reference_occupied_orbitals(
    orbital_count: int,
    num_electrons: int,
    occupied_orbitals: Sequence[int] | None,
) -> tuple[int, ...]:
    if isinstance(num_electrons, bool) or not isinstance(num_electrons, int):
        raise TypeError("num_electrons must be an integer")
    if num_electrons < 0 or num_electrons > orbital_count:
        raise ValueError("num_electrons must be between 0 and num_spin_orbitals")

    occupied = (
        tuple(range(num_electrons)) if occupied_orbitals is None else tuple(occupied_orbitals)
    )
    if len(occupied) != num_electrons:
        raise ValueError("occupied_orbitals must contain exactly num_electrons entries")
    for orbital in occupied:
        if isinstance(orbital, bool) or not isinstance(orbital, int):
            raise TypeError("occupied_orbitals must contain integers")
        if orbital < 0 or orbital >= orbital_count:
            raise ValueError("occupied_orbitals contains an orbital outside num_spin_orbitals")
    if len(set(occupied)) != len(occupied):
        raise ValueError("occupied_orbitals must be unique")
    return tuple(sorted(occupied))


def hartree_fock_state(
    num_spin_orbitals: int,
    num_electrons: int,
    *,
    occupied_orbitals: Sequence[int] | None = None,
) -> _native.Program:
    """Prepare a computational-basis Hartree-Fock occupation state."""
    orbital_count = _positive_integer(num_spin_orbitals, "num_spin_orbitals")
    occupied = _reference_occupied_orbitals(
        orbital_count,
        num_electrons,
        occupied_orbitals,
    )

    program = _native.Program(orbital_count)
    for orbital in occupied:
        program = _native.x(program, orbital)
    return program


def _virtual_orbitals(
    orbital_count: int,
    occupied: tuple[int, ...],
    virtual_orbitals: Sequence[int] | None,
) -> tuple[int, ...]:
    occupied_set = set(occupied)
    if virtual_orbitals is None:
        return tuple(orbital for orbital in range(orbital_count) if orbital not in occupied_set)

    virtual = tuple(virtual_orbitals)
    for orbital in virtual:
        if isinstance(orbital, bool) or not isinstance(orbital, int):
            raise TypeError("virtual_orbitals must contain integers")
        if orbital < 0 or orbital >= orbital_count:
            raise ValueError("virtual_orbitals contains an orbital outside num_spin_orbitals")
    if len(set(virtual)) != len(virtual):
        raise ValueError("virtual_orbitals must be unique")
    if occupied_set & set(virtual):
        raise ValueError("occupied_orbitals and virtual_orbitals must be disjoint")
    return tuple(sorted(virtual))


def uccsd_excitations(
    num_spin_orbitals: int,
    num_electrons: int,
    *,
    occupied_orbitals: Sequence[int] | None = None,
    virtual_orbitals: Sequence[int] | None = None,
) -> tuple[FermionicExcitation, ...]:
    """Enumerate canonical occupied-to-virtual UCCSD excitations."""
    orbital_count = _positive_integer(num_spin_orbitals, "num_spin_orbitals")
    occupied = _reference_occupied_orbitals(
        orbital_count,
        num_electrons,
        occupied_orbitals,
    )
    virtual = _virtual_orbitals(orbital_count, occupied, virtual_orbitals)

    excitations: list[FermionicExcitation] = []
    for source in occupied:
        for target in virtual:
            excitations.append(FermionicExcitation((source,), (target,)))
    for sources in combinations(occupied, 2):
        for targets in combinations(virtual, 2):
            excitations.append(FermionicExcitation(sources, targets))
    return tuple(excitations)


def fermionic_excitation_generator(
    num_spin_orbitals: int,
    excitation: FermionicExcitation,
    *,
    tolerance: float = 1e-12,
) -> _native.Observable:
    """Map i(A - A-dagger) for one excitation to a Hermitian Pauli generator."""
    orbital_count = _positive_integer(num_spin_orbitals, "num_spin_orbitals")
    if not isinstance(excitation, FermionicExcitation):
        raise TypeError("excitation must be a FermionicExcitation")
    for orbital in (*excitation.occupied, *excitation.virtual):
        if orbital >= orbital_count:
            raise ValueError("excitation references an orbital outside num_spin_orbitals")

    forward = tuple(fermion_creation(orbital) for orbital in excitation.virtual) + tuple(
        fermion_annihilation(orbital) for orbital in reversed(excitation.occupied)
    )
    adjoint = tuple(fermion_creation(orbital) for orbital in excitation.occupied) + tuple(
        fermion_annihilation(orbital) for orbital in reversed(excitation.virtual)
    )
    return jordan_wigner(
        orbital_count,
        [
            FermionTerm(1.0j, forward),
            FermionTerm(-1.0j, adjoint),
        ],
        tolerance=tolerance,
    )


def _pauli_terms_commute(left: _native.PauliTerm, right: _native.PauliTerm) -> bool:
    left_factors = {
        factor.qubit: factor.pauli for factor in left.factors if factor.pauli is not _native.Pauli.I
    }
    right_factors = {
        factor.qubit: factor.pauli
        for factor in right.factors
        if factor.pauli is not _native.Pauli.I
    }
    disagreements = sum(
        left_factors[qubit] is not right_factors[qubit]
        for qubit in left_factors.keys() & right_factors.keys()
    )
    return disagreements % 2 == 0


def _commuting_excitation_generator(generator: _native.Observable) -> bool:
    terms = tuple(generator.terms)
    return all(
        _pauli_terms_commute(left, right)
        for index, left in enumerate(terms)
        for right in terms[index + 1 :]
    )


def _excitation_parameter_name(excitation: FermionicExcitation) -> str:
    prefix = "single" if excitation.rank == 1 else "double"
    occupied = ",".join(str(orbital) for orbital in excitation.occupied)
    virtual = ",".join(str(orbital) for orbital in excitation.virtual)
    return f"{prefix}.{occupied}->{virtual}"


def uccsd_ansatz(
    num_spin_orbitals: int,
    num_electrons: int,
    *,
    occupied_orbitals: Sequence[int] | None = None,
    virtual_orbitals: Sequence[int] | None = None,
    tolerance: float = 1e-12,
) -> UccsdTemplate:
    """Build a deterministic first-order factorized spin-orbital UCCSD template."""
    orbital_count = _positive_integer(num_spin_orbitals, "num_spin_orbitals")
    threshold = _tolerance(tolerance)
    occupied = _reference_occupied_orbitals(
        orbital_count,
        num_electrons,
        occupied_orbitals,
    )
    excitations = uccsd_excitations(
        orbital_count,
        num_electrons,
        occupied_orbitals=occupied,
        virtual_orbitals=virtual_orbitals,
    )

    program = hartree_fock_state(
        orbital_count,
        num_electrons,
        occupied_orbitals=occupied,
    )
    slots: list[_native.ParameterSlot] = []
    slot_parameter_indices: list[int] = []
    slot_scales: list[float] = []
    names = tuple(_excitation_parameter_name(excitation) for excitation in excitations)

    for parameter_index, excitation in enumerate(excitations):
        generator = fermionic_excitation_generator(
            orbital_count,
            excitation,
            tolerance=threshold,
        )
        terms = tuple(generator.terms)
        if not terms:
            raise RuntimeError("UCCSD excitation generator mapped to no Pauli terms")
        if not _commuting_excitation_generator(generator):
            raise RuntimeError("UCCSD excitation generator mapped to non-commuting Pauli terms")

        for term in terms:
            operation_start = len(program.operations)
            program = append_pauli_evolution(program, term, 0.0)
            appended = program.operations[operation_start:]
            rz_indices = [
                operation_start + offset
                for offset, operation in enumerate(appended)
                if operation.code is _native.OperationCode.RZ
                and len(operation.parameters) == 1
                and operation.parameters[0] == 0.0
            ]
            if len(rz_indices) != 1:
                raise RuntimeError(
                    "UCCSD Pauli evolution did not expose exactly one RZ parameter slot"
                )
            slots.append(_native.ParameterSlot(rz_indices[0], 0))
            slot_parameter_indices.append(parameter_index)
            slot_scales.append(2.0 * float(term.coefficient))

    return UccsdTemplate(
        program=program,
        slots=tuple(slots),
        slot_parameter_indices=tuple(slot_parameter_indices),
        slot_scales=tuple(slot_scales),
        parameter_names=names,
        excitations=excitations,
    )


__all__ = [
    "FermionLadder",
    "FermionTerm",
    "FermionicExcitation",
    "UccsdTemplate",
    "fermion_annihilation",
    "fermion_creation",
    "fermion_term",
    "fermionic_excitation_generator",
    "hartree_fock_state",
    "jordan_wigner",
    "molecular_hamiltonian",
    "uccsd_ansatz",
    "uccsd_excitations",
]
