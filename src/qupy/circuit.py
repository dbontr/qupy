from __future__ import annotations

from . import _native

Circuit = _native.Circuit
CircuitInstruction = _native.CircuitInstruction
CircuitOperationCode = _native.CircuitOperationCode
ClassicalCondition = _native.ClassicalCondition
circuit_ir_version = _native.circuit_ir_version

__all__ = [
    "Circuit",
    "CircuitInstruction",
    "CircuitOperationCode",
    "ClassicalCondition",
    "circuit_ir_version",
]
