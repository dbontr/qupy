from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

from . import _native


class _NativeTensorNetworkResult(Protocol):
    @property
    def value(self) -> float: ...

    @property
    def term_count(self) -> int: ...

    @property
    def contractions(self) -> int: ...

    @property
    def peak_tensor_rank(self) -> int: ...

    @property
    def peak_tensor_bytes(self) -> int: ...

    @property
    def scalar_multiplications(self) -> float: ...

    @property
    def exact(self) -> bool: ...

    @property
    def backend(self) -> str: ...

    @property
    def method(self) -> str: ...


class _NativeTensorNetworkModule(Protocol):
    def tensor_network_expectation(
        self,
        program: _native.Program,
        observable: _native.Observable,
        max_tensor_bytes: int = 1 << 30,
    ) -> _NativeTensorNetworkResult: ...


@dataclass(frozen=True, slots=True)
class TensorNetworkEstimate:
    value: float
    term_count: int
    contractions: int
    peak_tensor_rank: int
    peak_tensor_bytes: int
    scalar_multiplications: float
    exact: bool
    backend: str
    method: str


def tensor_network_expectation(
    program: _native.Program,
    observable: _native.Observable,
    *,
    max_tensor_bytes: int = 1 << 30,
) -> TensorNetworkEstimate:
    native = cast(_NativeTensorNetworkModule, _native)
    result = native.tensor_network_expectation(
        program,
        observable,
        max_tensor_bytes,
    )
    return TensorNetworkEstimate(
        value=result.value,
        term_count=result.term_count,
        contractions=result.contractions,
        peak_tensor_rank=result.peak_tensor_rank,
        peak_tensor_bytes=result.peak_tensor_bytes,
        scalar_multiplications=result.scalar_multiplications,
        exact=result.exact,
        backend=result.backend,
        method=result.method,
    )


__all__ = ["TensorNetworkEstimate", "tensor_network_expectation"]
