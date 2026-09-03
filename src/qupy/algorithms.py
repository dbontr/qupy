from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

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


def qft(num_qubits: int, *, inverse: bool = False, swaps: bool = True) -> _native.Program:
    """Construct an exact QFT or inverse-QFT Program using QuPy native gates."""
    if num_qubits <= 0:
        raise ValueError("num_qubits must be positive")
    program = _native.Program(num_qubits)

    if inverse:
        if swaps:
            for first in range(num_qubits // 2):
                program = _native.swap(program, first, num_qubits - first - 1)
        for target in range(num_qubits - 1, -1, -1):
            for control in range(num_qubits - 1, target, -1):
                angle = -math.pi / float(1 << (control - target))
                program = _controlled_phase(program, control, target, angle)
            program = _native.h(program, target)
        return program

    for target in range(num_qubits):
        program = _native.h(program, target)
        for control in range(target + 1, num_qubits):
            angle = math.pi / float(1 << (control - target))
            program = _controlled_phase(program, control, target, angle)
    if swaps:
        for first in range(num_qubits // 2):
            program = _native.swap(program, first, num_qubits - first - 1)
    return program


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
    for first, second in zip(active_qubits, active_qubits[1:], strict=False):
        program = _native.cx(program, first, second)
    program = _native.rz(
        program,
        2.0 * evolution_time * coefficient,
        active_qubits[-1],
    )
    for first, second in reversed(list(zip(active_qubits, active_qubits[1:], strict=False))):
        program = _native.cx(program, first, second)

    for factor in reversed(ordered):
        if factor.pauli is _native.Pauli.X:
            program = _native.h(program, factor.qubit)
        elif factor.pauli is _native.Pauli.Y:
            program = _native.rx(program, -math.pi / 2.0, factor.qubit)

    return program


__all__ = [
    "VariationalTemplate",
    "append_pauli_evolution",
    "hardware_efficient_ansatz",
    "qft",
]
