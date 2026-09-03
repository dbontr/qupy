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
    return _native.variance_plan(
        program, observable, backend, resolve_planner_cost_model(backend, cost_model)
    )


def statevector(
    program: Program,
    backend: str = "auto",
    cost_model: PlannerCostModel | None = None,
) -> StateVector:
    return _native.statevector(program, backend, resolve_planner_cost_model(backend, cost_model))


def observable_plan(
    program: Program,
    observables: list[Observable],
    backend: str = "auto",
    cost_model: PlannerCostModel | None = None,
) -> ObservableExecutionPlan:
    return _native.observable_plan(
        program, observables, backend, resolve_planner_cost_model(backend, cost_model)
    )


def expect_observable(
    program: Program,
    observable: Observable,
    backend: str = "auto",
    cost_model: PlannerCostModel | None = None,
) -> ObservableResult:
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
        return _native.expect(program, observable, backend, model)
    function = cast(Any, _native.expect)
    return cast(ObservableResult, function(program, observable, backend, model))


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
    model = resolve_planner_cost_model(backend, cost_model)
    if isinstance(program, Program):
        return _native.density_matrix(program, backend, model)
    return _native.density_matrix(program, backend, model)
