from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt

from . import _native


class _NativeDistributedTensorNetworkResult(Protocol):
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
    def world_size(self) -> int: ...

    @property
    def active_ranks(self) -> int: ...

    @property
    def exact(self) -> bool: ...

    @property
    def backend(self) -> str: ...

    @property
    def method(self) -> str: ...


class _NativeDistributedTrajectoryBatch(Protocol):
    @property
    def values(self) -> npt.NDArray[np.float64]: ...

    @property
    def standard_errors(self) -> npt.NDArray[np.float64]: ...

    @property
    def observable_count(self) -> int: ...

    @property
    def trajectories(self) -> int: ...

    @property
    def seed(self) -> int: ...

    @property
    def state_bytes_per_rank(self) -> int: ...

    @property
    def world_size(self) -> int: ...

    @property
    def active_ranks(self) -> int: ...

    @property
    def exact(self) -> bool: ...

    @property
    def backend(self) -> str: ...

    @property
    def method(self) -> str: ...


class _NativeMultiDeviceModule(Protocol):
    def distributed_tensor_network_expectation(
        self,
        program: _native.Program,
        observable: _native.Observable,
        max_tensor_bytes: int = 1 << 30,
    ) -> _NativeDistributedTensorNetworkResult: ...

    def distributed_trajectory_expectations(
        self,
        program: _native.NoisyProgram,
        observables: list[_native.Observable],
        trajectories: int,
        seed: int | None = None,
    ) -> _NativeDistributedTrajectoryBatch: ...


@dataclass(frozen=True, slots=True)
class DistributedTensorNetworkEstimate:
    value: float
    term_count: int
    contractions: int
    peak_tensor_rank: int
    peak_tensor_bytes: int
    scalar_multiplications: float
    world_size: int
    active_ranks: int
    exact: bool
    backend: str
    method: str


@dataclass(frozen=True, slots=True)
class DistributedTrajectoryResult:
    values: npt.NDArray[np.float64]
    standard_errors: npt.NDArray[np.float64]
    observable_count: int
    trajectories: int
    seed: int
    state_bytes_per_rank: int
    world_size: int
    active_ranks: int
    exact: bool
    backend: str
    method: str


@dataclass(frozen=True, slots=True)
class DistributedTrajectoryEstimate:
    value: float
    standard_error: float
    trajectories: int
    seed: int
    state_bytes_per_rank: int
    world_size: int
    active_ranks: int
    exact: bool
    backend: str
    method: str


def distributed_tensor_network_expectation(
    program: _native.Program,
    observable: _native.Observable,
    *,
    max_tensor_bytes: int = 1 << 30,
) -> DistributedTensorNetworkEstimate:
    native = cast(_NativeMultiDeviceModule, _native)
    result = native.distributed_tensor_network_expectation(
        program,
        observable,
        max_tensor_bytes,
    )
    return DistributedTensorNetworkEstimate(
        value=result.value,
        term_count=result.term_count,
        contractions=result.contractions,
        peak_tensor_rank=result.peak_tensor_rank,
        peak_tensor_bytes=result.peak_tensor_bytes,
        scalar_multiplications=result.scalar_multiplications,
        world_size=result.world_size,
        active_ranks=result.active_ranks,
        exact=result.exact,
        backend=result.backend,
        method=result.method,
    )


def distributed_trajectory_expectations(
    program: _native.NoisyProgram,
    observables: list[_native.Observable],
    trajectories: int = 1024,
    seed: int | None = None,
) -> DistributedTrajectoryResult:
    native = cast(_NativeMultiDeviceModule, _native)
    result = native.distributed_trajectory_expectations(
        program,
        observables,
        trajectories,
        seed,
    )
    return DistributedTrajectoryResult(
        values=np.asarray(result.values, dtype=np.float64).copy(),
        standard_errors=np.asarray(result.standard_errors, dtype=np.float64).copy(),
        observable_count=result.observable_count,
        trajectories=result.trajectories,
        seed=result.seed,
        state_bytes_per_rank=result.state_bytes_per_rank,
        world_size=result.world_size,
        active_ranks=result.active_ranks,
        exact=result.exact,
        backend=result.backend,
        method=result.method,
    )


def distributed_trajectory_expectation(
    program: _native.NoisyProgram,
    observable: _native.Observable,
    trajectories: int = 1024,
    seed: int | None = None,
) -> DistributedTrajectoryEstimate:
    result = distributed_trajectory_expectations(
        program,
        [observable],
        trajectories,
        seed,
    )
    return DistributedTrajectoryEstimate(
        value=float(result.values[0]),
        standard_error=float(result.standard_errors[0]),
        trajectories=result.trajectories,
        seed=result.seed,
        state_bytes_per_rank=result.state_bytes_per_rank,
        world_size=result.world_size,
        active_ranks=result.active_ranks,
        exact=result.exact,
        backend=result.backend,
        method=result.method,
    )


__all__ = [
    "DistributedTensorNetworkEstimate",
    "DistributedTrajectoryEstimate",
    "DistributedTrajectoryResult",
    "distributed_tensor_network_expectation",
    "distributed_trajectory_expectation",
    "distributed_trajectory_expectations",
]
