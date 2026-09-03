from __future__ import annotations

from typing import Any, cast, overload

from . import _native
from ._native import (
    DensityMatrix,
    ExecutionPlan,
    Expectation,
    NoisyProgram,
    Observable,
    ObservableBatch,
    ObservableExecutionPlan,
    ObservableResult,
    PauliZ,
    PlannerCostModel,
    Program,
    ResultMode,
    StateVector,
    Variance,
)
from ._planner import resolve_planner_cost_model
from .tensor_network import (
    _expect_observable as _tensor_network_expect_observable,
    _expect_observables as _tensor_network_expect_observables,
    _observable_plan as _tensor_network_observable_plan,
)

_TENSOR_NETWORK_BACKEND = "native-tn"


def _reject_tensor_network_non_expectation(backend: str, operation: str) -> None:
    if backend == _TENSOR_NETWORK_BACKEND:
        raise ValueError(
            f"{operation} is not supported by native-tn; "
            "the tensor-network backend currently supports observable expectations only"
        )


def plan(
    program: Program,
    result_mode: ResultMode,
    backend: str = "auto",
    cost_model: PlannerCostModel | None = None,
) -> ExecutionPlan:
    return _native.plan(program, result_mode, backend, resolve_planner_cost_model(backend, cost_model))


def expectation_plan(
    program: Program,
    observable: PauliZ,
    backend: str = "auto",
    cost_model: PlannerCostModel | None = None,
) -> ExecutionPlan:
    return _native.expectation_plan(
        program, observable, backend, resolve_planner_cost_model(backend, cost_model)
    )


def variance_plan(
    program: Program,
    observable: PauliZ,
    backend: str = "auto",
    cost_model: PlannerCostModel | None = None,
) -> ExecutionPlan:
    _reject_tensor_network_non_expectation(backend, "variance planning")
    return _native.variance_plan(
        program, observable, backend, resolve_planner_cost_model(backend, cost_model)
    )


def statevector(
    program: Program,
    backend: str = "auto",
    cost_model: PlannerCostModel | None = None,
) -> StateVector:
    _reject_tensor_network_non_expectation(backend, "state-vector materialization")
    return _native.statevector(program, backend, resolve_planner_cost_model(backend, cost_model))


def observable_plan(
    program: Program,
    observables: list[Observable],
    backend: str = "auto",
    cost_model: PlannerCostModel | None = None,
) -> ObservableExecutionPlan:
    if backend == _TENSOR_NETWORK_BACKEND:
        return _tensor_network_observable_plan(program, observables)
    return _native.observable_plan(
        program, observables, backend, resolve_planner_cost_model(backend, cost_model)
    )


def expect_observable(
    program: Program,
    observable: Observable,
    backend: str = "auto",
    cost_model: PlannerCostModel | None = None,
) -> ObservableResult:
    if backend == _TENSOR_NETWORK_BACKEND:
        return _tensor_network_expect_observable(program, observable)
    function = cast(Any, _native.expect_observable)
    return cast(
        ObservableResult,
        function(program, observable, backend, resolve_planner_cost_model(backend, cost_model)),
    )


def variance_observable(
    program: Program,
    observable: Observable,
    backend: str = "auto",
    cost_model: PlannerCostModel | None = None,
) -> ObservableResult:
    _reject_tensor_network_non_expectation(backend, "observable variance")
    function = cast(Any, _native.variance_observable)
    return cast(
        ObservableResult,
        function(program, observable, backend, resolve_planner_cost_model(backend, cost_model)),
    )


def covariance(
    program: Program,
    left: Observable,
    right: Observable,
    backend: str = "auto",
    cost_model: PlannerCostModel | None = None,
) -> ObservableResult:
    _reject_tensor_network_non_expectation(backend, "observable covariance")
    function = cast(Any, _native.covariance)
    return cast(
        ObservableResult,
        function(program, left, right, backend, resolve_planner_cost_model(backend, cost_model)),
    )


def expect_observables(
    program: Program,
    observables: list[Observable],
    backend: str = "auto",
    cost_model: PlannerCostModel | None = None,
) -> ObservableBatch:
    if backend == _TENSOR_NETWORK_BACKEND:
        return _tensor_network_expect_observables(program, observables)
    function = cast(Any, _native.expect_observables)
    return cast(
        ObservableBatch,
        function(program, observables, backend, resolve_planner_cost_model(backend, cost_model)),
    )


@overload
def expect(
    program: Program,
    observable: PauliZ,
    backend: str = "auto",
    cost_model: PlannerCostModel | None = None,
) -> Expectation: ...


@overload
def expect(
    program: Program,
    observable: Observable,
    backend: str = "auto",
    cost_model: PlannerCostModel | None = None,
) -> ObservableResult: ...


def expect(
    program: Program,
    observable: PauliZ | Observable,
    backend: str = "auto",
    cost_model: PlannerCostModel | None = None,
) -> Expectation | ObservableResult:
    model = resolve_planner_cost_model(backend, cost_model)
    if isinstance(observable, PauliZ):
        _reject_tensor_network_non_expectation(
            backend,
            "PauliZ expectation through the state-vector planner",
        )
        return _native.expect(program, observable, backend, model)
    return expect_observable(program, observable, backend, cost_model)


@overload
def variance(
    program: Program,
    observable: PauliZ,
    backend: str = "auto",
    cost_model: PlannerCostModel | None = None,
) -> Variance: ...


@overload
def variance(
    program: Program,
    observable: Observable,
    backend: str = "auto",
    cost_model: PlannerCostModel | None = None,
) -> ObservableResult: ...


def variance(
    program: Program,
    observable: PauliZ | Observable,
    backend: str = "auto",
    cost_model: PlannerCostModel | None = None,
) -> Variance | ObservableResult:
    _reject_tensor_network_non_expectation(backend, "variance")
    model = resolve_planner_cost_model(backend, cost_model)
    if isinstance(observable, PauliZ):
        return _native.variance(program, observable, backend, model)
    function = cast(Any, _native.variance)
    return cast(ObservableResult, function(program, observable, backend, model))


@overload
def density_matrix(
    program: Program,
    backend: str = "auto",
    cost_model: PlannerCostModel | None = None,
) -> DensityMatrix: ...


@overload
def density_matrix(
    program: NoisyProgram,
    backend: str = "auto",
    cost_model: PlannerCostModel | None = None,
) -> DensityMatrix: ...


def density_matrix(
    program: Program | NoisyProgram,
    backend: str = "auto",
    cost_model: PlannerCostModel | None = None,
) -> DensityMatrix:
    _reject_tensor_network_non_expectation(backend, "density-matrix materialization")
    model = resolve_planner_cost_model(backend, cost_model)
    if isinstance(program, Program):
        return _native.density_matrix(program, backend, model)
    return _native.density_matrix(program, backend, model)
