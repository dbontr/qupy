from __future__ import annotations

from dataclasses import dataclass

from .ir import Program


@dataclass(frozen=True, slots=True)
class Target:
    name: str
    operations: frozenset[str]
    result_modes: frozenset[str]
    max_qubits: int | None = None
    simulator: bool = False

    def validate(self, program: Program, result_mode: str) -> None:
        if self.max_qubits is not None and program.num_qubits > self.max_qubits:
            raise ValueError(f"target {self.name!r} supports at most {self.max_qubits} qubits")
        if result_mode not in self.result_modes:
            raise ValueError(f"target {self.name!r} does not support {result_mode!r} results")
        unsupported = {operation.name for operation in program.operations} - self.operations
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise ValueError(f"target {self.name!r} does not support operations: {names}")