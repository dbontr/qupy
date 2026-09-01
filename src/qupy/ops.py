from __future__ import annotations

from dataclasses import dataclass

from .ir import Operation, Program


@dataclass(frozen=True, slots=True)
class PauliZ:
    qubit: int


def h(program: Program, qubit: int) -> Program:
    return program.append(Operation("h", (qubit,)))


def x(program: Program, qubit: int) -> Program:
    return program.append(Operation("x", (qubit,)))


def rx(program: Program, angle: float, qubit: int) -> Program:
    return program.append(Operation("rx", (qubit,), (float(angle),)))


def cx(program: Program, control: int, target: int) -> Program:
    return program.append(Operation("cx", (control, target)))


def Z(qubit: int) -> PauliZ:
    return PauliZ(qubit)