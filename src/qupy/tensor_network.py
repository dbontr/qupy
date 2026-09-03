from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

from . import _native


class _NativeTensorNetworkPlan(Protocol):
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
    def max_tensor_bytes(self) -> int: ...

    @property
    def exact(self) -> bool: ...

    @property
    def backend(self) -> str: ...

    @property
    def method(self) -> str: ...

    @property
    def program_fingerprint(self) -> str: ...

    @property
    def observable_fingerprint(self) -> str: ...

    @property
    def plan_fingerprint(self) -> str: ...


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
    def tensor_network_plan(
        self,
        program: _native.Program,
        observable: _native.Observable,
        max_tensor_bytes: int = 1 << 30,
    ) -> _NativeTensorNetworkPlan: ...

    def tensor_network_expectation(
        self,
        program: _native.Program,
        observable: _native.Observable,
        max_tensor_bytes: int = 1 << 30,
    ) -> _NativeTensorNetworkResult: ...

    def tensor_network_observable_plan(
        self,
        program: _native.Program,
        observables: list[_native.Observable],
        max_tensor_bytes: int = 1 << 30,
    ) -> _native.ObservableExecutionPlan: ...

    def tensor_network_expect_observable(
        self,
        program: _native.Program,
        observable: _native.Observable,
        max_tensor_bytes: int = 1 << 30,
    ) -> _native.ObservableResult: ...

    def tensor_network_expect_observables(
        self,
        program: _native.Program,
        observables: list[_native.Observable],
        max_tensor_bytes: int = 1 << 30,
    ) -> _native.ObservableBatch: ...


@dataclass(frozen=True, slots=True)
class TensorNetworkPlan:
    term_count: int
    contractions: int
    peak_tensor_rank: int
    peak_tensor_bytes: int
    scalar_multiplications: float
    max_tensor_bytes: int
    exact: bool
    backend: str
    method: str
    program_fingerprint: str
    observable_fingerprint: str
    plan_fingerprint: str


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


def tensor_network_plan(
    program: _native.Program,
    observable: _native.Observable,
    *,
    max_tensor_bytes: int = 1 << 30,
) -> TensorNetworkPlan:
    native = cast(_NativeTensorNetworkModule, _native)
    result = native.tensor_network_plan(program, observable, max_tensor_bytes)
    return TensorNetworkPlan(
        term_count=result.term_count,
        contractions=result.contractions,
        peak_tensor_rank=result.peak_tensor_rank,
        peak_tensor_bytes=result.peak_tensor_bytes,
        scalar_multiplications=result.scalar_multiplications,
        max_tensor_bytes=result.max_tensor_bytes,
        exact=result.exact,
        backend=result.backend,
        method=result.method,
        program_fingerprint=result.program_fingerprint,
        observable_fingerprint=result.observable_fingerprint,
        plan_fingerprint=result.plan_fingerprint,
    )


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


def _observable_plan(
    program: _native.Program,
    observables: list[_native.Observable],
) -> _native.ObservableExecutionPlan:
    native = cast(_NativeTensorNetworkModule, _native)
    return native.tensor_network_observable_plan(program, observables)


def _expect_observable(
    program: _native.Program,
    observable: _native.Observable,
) -> _native.ObservableResult:
    native = cast(_NativeTensorNetworkModule, _native)
    return native.tensor_network_expect_observable(program, observable)


def _expect_observables(
    program: _native.Program,
    observables: list[_native.Observable],
) -> _native.ObservableBatch:
    native = cast(_NativeTensorNetworkModule, _native)
    return native.tensor_network_expect_observables(program, observables)


__all__ = [
    "TensorNetworkEstimate",
    "TensorNetworkPlan",
    "tensor_network_expectation",
    "tensor_network_plan",
]
