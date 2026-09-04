from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise

from . import _native

_ROTATION_AXES = {"rx", "ry", "rz"}
_ENTANGLEMENT_PATTERNS = {"linear", "ring", "none"}


@dataclass(frozen=True, slots=True)
class VariationalTemplate:
    """Native Program plus named parameter slots for variational workflows."""

    program: _native.Program
    slots: tuple[_native.ParameterSlot, ...]
    parameter_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.slots) != len(self.parameter_names):
            raise ValueError("slots and parameter_names must have the same length")
        if any(not name for name in self.parameter_names):
            raise ValueError("parameter names must be non-empty")
        if len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError("parameter names must be unique")

    @property
    def parameter_count(self) -> int:
        return len(self.slots)

    def bind(self, values: Sequence[float]) -> _native.Program:
        bound = [float(value) for value in values]
        if len(bound) != self.parameter_count:
            raise ValueError(
                f"values must contain exactly {self.parameter_count} parameters"
            )
        if not all(math.isfinite(value) for value in bound):
            raise ValueError("parameter values must be finite")
        return self.program.bind(list(self.slots), bound)

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
            raise ValueError("named parameters do not match template: " + ", ".join(details))
        return self.bind([values[name] for name in self.parameter_names])


def _append_rotation(program: _native.Program, axis: str, angle: float, qubit: int) -> _native.Program:
    if axis == "rx":
        return _native.rx(program, angle, qubit)
    if axis == "ry":
        return _native.ry(program, angle, qubit)
    if axis == "rz":
        return _native.rz(program, angle, qubit)
    raise ValueError(f"unsupported rotation axis {axis!r}")


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _finite_values(values: Sequence[float], name: str) -> tuple[float, ...]:
    converted = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in converted):
        raise ValueError(f"{name} must contain only finite values")
    return converted


def hardware_efficient_ansatz(
    num_qubits: int,
    layers: int,
    *,
    rotations: Sequence[str] = ("ry", "rz"),
    entanglement: str = "linear",
) -> VariationalTemplate:
    """Build a native hardware-efficient ansatz with tracked rotation slots."""
    if num_qubits <= 0:
        raise ValueError("num_qubits must be positive")
    if layers <= 0:
        raise ValueError("layers must be positive")
    axes = tuple(rotations)
    if not axes:
        raise ValueError("rotations must contain at least one axis")
    if any(axis not in _ROTATION_AXES for axis in axes):
        raise ValueError("rotations may contain only 'rx', 'ry', and 'rz'")
    if entanglement not in _ENTANGLEMENT_PATTERNS:
        raise ValueError("entanglement must be 'linear', 'ring', or 'none'")

    program = _native.Program(num_qubits)
    slots: list[_native.ParameterSlot] = []
    names: list[str] = []
    for layer in range(layers):
        for qubit in range(num_qubits):
            for axis in axes:
                program = _append_rotation(program, axis, 0.0, qubit)
                slots.append(_native.ParameterSlot(len(program.operations) - 1, 0))
                names.append(f"layer{layer}.q{qubit}.{axis}")

        if entanglement != "none":
            for qubit in range(num_qubits - 1):
                program = _native.cx(program, qubit, qubit + 1)
            if entanglement == "ring" and num_qubits > 2:
                program = _native.cx(program, num_qubits - 1, 0)

    return VariationalTemplate(program, tuple(slots), tuple(names))


def _controlled_phase(
    program: _native.Program,
    control: int,
    target: int,
    angle: float,
) -> _native.Program:
    """Synthesize controlled phase up to a state-independent global phase."""
    half = angle / 2.0
    program = _native.rz(program, half, control)
    program = _native.cx(program, control, target)
    program = _native.rz(program, -half, target)
    program = _native.cx(program, control, target)
    return _native.rz(program, half, target)


def _selected_qubits(program: _native.Program, qubits: Sequence[int] | None) -> tuple[int, ...]:
    selected = tuple(range(program.num_qubits)) if qubits is None else tuple(qubits)
    if not selected:
        raise ValueError("qubits must contain at least one qubit")
    if len(set(selected)) != len(selected):
        raise ValueError("qubits must be unique")
    if any(qubit < 0 or qubit >= program.num_qubits for qubit in selected):
        raise ValueError("qubits contains an index outside the program")
    return selected


def append_qft(
    program: _native.Program,
    qubits: Sequence[int] | None = None,
    *,
    inverse: bool = False,
    swaps: bool = True,
) -> _native.Program:
    """Append an exact QFT or inverse QFT to selected native Program qubits."""
    selected = _selected_qubits(program, qubits)
    size = len(selected)

    if inverse:
        if swaps:
            for first in range(size // 2):
                program = _native.swap(program, selected[first], selected[size - first - 1])
        for target_index in range(size - 1, -1, -1):
            target = selected[target_index]
            for control_index in range(size - 1, target_index, -1):
                angle = -math.pi / float(1 << (control_index - target_index))
                program = _controlled_phase(
                    program,
                    selected[control_index],
                    target,
                    angle,
                )
            program = _native.h(program, target)
        return program

    for target_index, target in enumerate(selected):
        program = _native.h(program, target)
        for control_index in range(target_index + 1, size):
            angle = math.pi / float(1 << (control_index - target_index))
            program = _controlled_phase(
                program,
                selected[control_index],
                target,
                angle,
            )
    if swaps:
        for first in range(size // 2):
            program = _native.swap(program, selected[first], selected[size - first - 1])
    return program


def qft(num_qubits: int, *, inverse: bool = False, swaps: bool = True) -> _native.Program:
    """Construct an exact QFT or inverse-QFT Program using QuPy native gates."""
    if num_qubits <= 0:
        raise ValueError("num_qubits must be positive")
    return append_qft(_native.Program(num_qubits), inverse=inverse, swaps=swaps)


def append_pauli_evolution(
    program: _native.Program,
    term: _native.PauliTerm,
    time: float,
) -> _native.Program:
    """Append exp(-i * time * coefficient * P) for one Pauli term."""
    evolution_time = float(time)
    if not math.isfinite(evolution_time):
        raise ValueError("time must be finite")
    coefficient = float(term.coefficient)
    if not math.isfinite(coefficient):
        raise ValueError("Pauli coefficient must be finite")

    factors = [factor for factor in term.factors if factor.pauli is not _native.Pauli.I]
    if not factors:
        return program
    qubits = [factor.qubit for factor in factors]
    if len(set(qubits)) != len(qubits):
        raise ValueError("Pauli term contains duplicate non-identity qubits")
    if any(qubit >= program.num_qubits for qubit in qubits):
        raise ValueError("Pauli term references a qubit outside the program")

    ordered = sorted(factors, key=lambda factor: factor.qubit)
    for factor in ordered:
        if factor.pauli is _native.Pauli.X:
            program = _native.h(program, factor.qubit)
        elif factor.pauli is _native.Pauli.Y:
            program = _native.rx(program, math.pi / 2.0, factor.qubit)
        elif factor.pauli is not _native.Pauli.Z:
            raise ValueError("unsupported Pauli factor")

    active_qubits = [factor.qubit for factor in ordered]
    edges = list(pairwise(active_qubits))
    for first, second in edges:
        program = _native.cx(program, first, second)
    program = _native.rz(
        program,
        2.0 * evolution_time * coefficient,
        active_qubits[-1],
    )
    for first, second in reversed(edges):
        program = _native.cx(program, first, second)

    for factor in reversed(ordered):
        if factor.pauli is _native.Pauli.X:
            program = _native.h(program, factor.qubit)
        elif factor.pauli is _native.Pauli.Y:
            program = _native.rx(program, -math.pi / 2.0, factor.qubit)

    return program


def append_hamiltonian_evolution(
    program: _native.Program,
    hamiltonian: _native.Observable,
    time: float,
    *,
    steps: int = 1,
    order: int = 1,
) -> _native.Program:
    """Append a first- or second-order product formula for a Pauli Hamiltonian."""
    evolution_time = float(time)
    if not math.isfinite(evolution_time):
        raise ValueError("time must be finite")
    step_count = _positive_integer(steps, "steps")
    if isinstance(order, bool) or not isinstance(order, int):
        raise TypeError("order must be an integer")
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")

    terms = tuple(hamiltonian.terms)
    if not terms or evolution_time == 0.0:
        return program

    step_time = evolution_time / float(step_count)
    if order == 1:
        for _ in range(step_count):
            for term in terms:
                program = append_pauli_evolution(program, term, step_time)
        return program

    half_step = step_time / 2.0
    for _ in range(step_count):
        for term in terms[:-1]:
            program = append_pauli_evolution(program, term, half_step)
        program = append_pauli_evolution(program, terms[-1], step_time)
        for term in reversed(terms[:-1]):
            program = append_pauli_evolution(program, term, half_step)
    return program


def _maxcut_edges(
    num_qubits: int,
    edges: Sequence[tuple[int, int]],
    weights: Sequence[float] | None,
) -> tuple[tuple[int, int, float], ...]:
    qubit_count = _positive_integer(num_qubits, "num_qubits")
    pairs = tuple(edges)
    edge_weights = (
        tuple(1.0 for _ in pairs)
        if weights is None
        else _finite_values(weights, "weights")
    )
    if len(edge_weights) != len(pairs):
        raise ValueError("weights must contain exactly one value per edge")

    normalized: list[tuple[int, int, float]] = []
    seen: set[tuple[int, int]] = set()
    for index, (first, second) in enumerate(pairs):
        if isinstance(first, bool) or not isinstance(first, int):
            raise TypeError(f"edges[{index}][0] must be an integer")
        if isinstance(second, bool) or not isinstance(second, int):
            raise TypeError(f"edges[{index}][1] must be an integer")
        if first < 0 or first >= qubit_count or second < 0 or second >= qubit_count:
            raise ValueError(f"edges[{index}] contains a qubit outside num_qubits")
        if first == second:
            raise ValueError(f"edges[{index}] must connect two distinct qubits")
        weight = edge_weights[index]
        if weight < 0.0:
            raise ValueError("weights must be non-negative")
        edge = (min(first, second), max(first, second))
        if edge in seen:
            raise ValueError(f"duplicate undirected edge {edge}")
        seen.add(edge)
        normalized.append((edge[0], edge[1], weight))
    normalized.sort(key=lambda item: (item[0], item[1]))
    return tuple(normalized)


def maxcut_hamiltonian(
    num_qubits: int,
    edges: Sequence[tuple[int, int]],
    *,
    weights: Sequence[float] | None = None,
) -> _native.Observable:
    """Construct the weighted MaxCut objective sum w_ij (I - Z_i Z_j) / 2."""
    normalized = _maxcut_edges(num_qubits, edges, weights)
    terms: list[_native.PauliTerm] = []
    constant = sum(weight for _, _, weight in normalized) / 2.0
    if constant != 0.0:
        terms.append(_native.PauliTerm(constant, []))
    for first, second, weight in normalized:
        if weight == 0.0:
            continue
        terms.append(
            _native.PauliTerm(
                -weight / 2.0,
                [
                    _native.PauliFactor(first, _native.Pauli.Z),
                    _native.PauliFactor(second, _native.Pauli.Z),
                ],
            )
        )
    return _native.Observable(terms)


def qaoa_maxcut_program(
    num_qubits: int,
    edges: Sequence[tuple[int, int]],
    gammas: Sequence[float],
    betas: Sequence[float],
    *,
    weights: Sequence[float] | None = None,
) -> _native.Program:
    """Construct a standard X-mixer QAOA Program for weighted MaxCut."""
    qubit_count = _positive_integer(num_qubits, "num_qubits")
    gamma_values = _finite_values(gammas, "gammas")
    beta_values = _finite_values(betas, "betas")
    if not gamma_values:
        raise ValueError("gammas and betas must contain at least one layer")
    if len(gamma_values) != len(beta_values):
        raise ValueError("gammas and betas must have the same length")

    cost = maxcut_hamiltonian(qubit_count, edges, weights=weights)
    program = _native.Program(qubit_count)
    for qubit in range(qubit_count):
        program = _native.h(program, qubit)

    for gamma, beta in zip(gamma_values, beta_values, strict=True):
        program = append_hamiltonian_evolution(program, cost, gamma, steps=1, order=1)
        for qubit in range(qubit_count):
            program = _native.rx(program, 2.0 * beta, qubit)
    return program


__all__ = [
    "VariationalTemplate",
    "append_hamiltonian_evolution",
    "append_pauli_evolution",
    "append_qft",
    "hardware_efficient_ansatz",
    "maxcut_hamiltonian",
    "qaoa_maxcut_program",
    "qft",
]
