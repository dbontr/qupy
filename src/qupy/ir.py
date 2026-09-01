from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

OperationName = Literal["h", "x", "rx", "cx"]


@dataclass(frozen=True, slots=True)
class Operation:
    name: OperationName
    qubits: tuple[int, ...]
    parameters: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class Program:
    num_qubits: int
    operations: tuple[Operation, ...] = ()

    def __post_init__(self) -> None:
        if self.num_qubits < 1:
            raise ValueError("num_qubits must be at least 1")

    def append(self, operation: Operation) -> Program:
        for qubit in operation.qubits:
            if not 0 <= qubit < self.num_qubits:
                raise ValueError(f"qubit {qubit} is outside this program")
        if len(set(operation.qubits)) != len(operation.qubits):
            raise ValueError("an operation cannot use the same qubit twice")
        return replace(self, operations=(*self.operations, operation))