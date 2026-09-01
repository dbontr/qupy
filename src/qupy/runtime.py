from __future__ import annotations

from typing import Literal

from .backends import numpy_statevector
from .ir import Program
from .ops import PauliZ
from .results import Expectation, Samples, StateVector

BackendName = Literal["auto", "numpy"]


def _require_backend(backend: BackendName) -> None:
    if backend not in {"auto", "numpy"}:
        raise ValueError(f"unknown backend: {backend!r}")


def statevector(program: Program, *, backend: BackendName = "auto") -> StateVector:
    _require_backend(backend)
    return numpy_statevector.statevector(program)


def sample(
    program: Program, *, shots: int = 1024, seed: int | None = None, backend: BackendName = "auto"
) -> Samples:
    _require_backend(backend)
    return numpy_statevector.sample(program, shots, seed)


def expect(program: Program, observable: PauliZ, *, backend: BackendName = "auto") -> Expectation:
    _require_backend(backend)
    return numpy_statevector.expectation(program, observable)