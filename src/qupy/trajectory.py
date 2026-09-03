from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt

from . import _native


class _NativeTrajectoryBatch(Protocol):
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
    def state_bytes(self) -> int: ...

    @property
    def exact(self) -> bool: ...

    @property
    def backend(self) -> str: ...

    @property
    def method(self) -> str: ...


class _NativeTrajectoryModule(Protocol):
    def trajectory_expectations(
        self,
        program: _native.NoisyProgram,
        observables: list[_native.Observable],
        trajectories: int,
        seed: int | None = None,
        backend: str = "auto",
    ) -> _NativeTrajectoryBatch: ...


@dataclass(frozen=True, slots=True)
class TrajectoryResult:
    values: npt.NDArray[np.float64]
    standard_errors: npt.NDArray[np.float64]
    observable_count: int
    trajectories: int
    seed: int
    state_bytes: int
    exact: bool
    backend: str
    method: str


@dataclass(frozen=True, slots=True)
class TrajectoryEstimate:
    value: float
    standard_error: float
    trajectories: int
    seed: int
    state_bytes: int
    exact: bool
    backend: str
    method: str


def trajectory_expectations(
    program: _native.NoisyProgram,
    observables: list[_native.Observable],
    trajectories: int = 1024,
    seed: int | None = None,
    backend: str = "auto",
) -> TrajectoryResult:
    native = cast(_NativeTrajectoryModule, _native)
    result = native.trajectory_expectations(
        program,
        observables,
        trajectories,
        seed,
        backend,
    )
    return TrajectoryResult(
        values=np.asarray(result.values, dtype=np.float64).copy(),
        standard_errors=np.asarray(result.standard_errors, dtype=np.float64).copy(),
        observable_count=result.observable_count,
        trajectories=result.trajectories,
        seed=result.seed,
        state_bytes=result.state_bytes,
        exact=result.exact,
        backend=result.backend,
        method=result.method,
    )


def trajectory_expectation(
    program: _native.NoisyProgram,
    observable: _native.Observable,
    trajectories: int = 1024,
    seed: int | None = None,
    backend: str = "auto",
) -> TrajectoryEstimate:
    result = trajectory_expectations(
        program,
        [observable],
        trajectories,
        seed,
        backend,
    )
    return TrajectoryEstimate(
        value=float(result.values[0]),
        standard_error=float(result.standard_errors[0]),
        trajectories=result.trajectories,
        seed=result.seed,
        state_bytes=result.state_bytes,
        exact=result.exact,
        backend=result.backend,
        method=result.method,
    )


__all__ = [
    "TrajectoryEstimate",
    "TrajectoryResult",
    "trajectory_expectation",
    "trajectory_expectations",
]
