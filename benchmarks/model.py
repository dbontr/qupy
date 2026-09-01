from __future__ import annotations

from dataclasses import dataclass
import math

_GATE_ARITY = {
    "h": 1,
    "x": 1,
    "y": 1,
    "z": 1,
    "rx": 1,
    "ry": 1,
    "rz": 1,
    "cx": 2,
    "cz": 2,
    "swap": 2,
}
_PARAMETERIZED = {"rx", "ry", "rz"}


@dataclass(frozen=True, slots=True)
class Gate:
    name: str
    qubits: tuple[int, ...]
    parameter: float | None = None

    def __post_init__(self) -> None:
        name = self.name.lower()
        object.__setattr__(self, "name", name)
        if name not in _GATE_ARITY:
            raise ValueError(f"unsupported benchmark gate: {name}")
        if len(self.qubits) != _GATE_ARITY[name]:
            raise ValueError(f"gate {name} requires {_GATE_ARITY[name]} qubits")
        if len(set(self.qubits)) != len(self.qubits):
            raise ValueError("benchmark gate cannot use the same qubit twice")
        if name in _PARAMETERIZED:
            if self.parameter is None or not math.isfinite(self.parameter):
                raise ValueError(f"gate {name} requires a finite parameter")
        elif self.parameter is not None:
            raise ValueError(f"gate {name} does not accept a parameter")


@dataclass(frozen=True, slots=True)
class Workload:
    name: str
    family: str
    num_qubits: int
    gates: tuple[Gate, ...]
    observable_qubit: int
    expected_value: float
    clifford: bool
    tolerance: float = 1e-9

    def __post_init__(self) -> None:
        if self.num_qubits < 1:
            raise ValueError("benchmark workload must contain at least one qubit")
        if not 0 <= self.observable_qubit < self.num_qubits:
            raise ValueError("benchmark observable is outside the workload")
        if not math.isfinite(self.expected_value):
            raise ValueError("benchmark expected value must be finite")
        if not math.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError("benchmark tolerance must be positive and finite")
        for gate in self.gates:
            for qubit in gate.qubits:
                if not 0 <= qubit < self.num_qubits:
                    raise ValueError(f"gate {gate.name} references qubit {qubit} outside workload")

    @property
    def operation_count(self) -> int:
        return len(self.gates)


def ghz_z_clifford(num_qubits: int) -> Workload:
    if num_qubits < 1:
        raise ValueError("GHZ workload requires at least one qubit")
    gates = [Gate("h", (0,))]
    gates.extend(Gate("cx", (qubit - 1, qubit)) for qubit in range(1, num_qubits))
    return Workload(
        name=f"ghz-z-{num_qubits}",
        family="clifford-ghz-z",
        num_qubits=num_qubits,
        gates=tuple(gates),
        observable_qubit=num_qubits - 1,
        expected_value=0.0,
        clifford=True,
    )


def local_nonclifford_z(num_qubits: int, angle: float = 0.37) -> Workload:
    if num_qubits < 2:
        raise ValueError("local non-Clifford workload requires at least two qubits")
    gates = (
        Gate("h", (0,)),
        Gate("ry", (0,), angle),
        Gate("ry", (num_qubits - 1,), 0.19),
    )
    return Workload(
        name=f"local-nonclifford-z-{num_qubits}",
        family="local-nonclifford-z",
        num_qubits=num_qubits,
        gates=gates,
        observable_qubit=0,
        expected_value=-math.sin(angle),
        clifford=False,
    )


def entangled_nonclifford_z(num_qubits: int, angle: float = 0.37) -> Workload:
    if num_qubits < 2:
        raise ValueError("entangled non-Clifford workload requires at least two qubits")
    gates = [Gate("h", (0,)), Gate("ry", (0,), angle)]
    gates.extend(Gate("cx", (qubit - 1, qubit)) for qubit in range(1, num_qubits))
    return Workload(
        name=f"entangled-nonclifford-z-{num_qubits}",
        family="entangled-nonclifford-z",
        num_qubits=num_qubits,
        gates=tuple(gates),
        observable_qubit=num_qubits - 1,
        expected_value=-math.sin(angle),
        clifford=False,
    )


def workloads_for_profile(profile: str) -> tuple[Workload, ...]:
    if profile == "smoke":
        return (
            ghz_z_clifford(6),
            local_nonclifford_z(6),
            entangled_nonclifford_z(6),
        )
    if profile == "standard":
        return (
            ghz_z_clifford(64),
            ghz_z_clifford(512),
            ghz_z_clifford(4096),
            local_nonclifford_z(12),
            local_nonclifford_z(16),
            local_nonclifford_z(20),
            entangled_nonclifford_z(12),
            entangled_nonclifford_z(16),
            entangled_nonclifford_z(20),
        )
    raise ValueError(f"unknown benchmark profile: {profile}")
