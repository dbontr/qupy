from collections.abc import Callable, Sequence
from typing import Any

from ._native import GradientMethod, Observable, ParameterSlot, Program

def make_jax_expectation(
    program: Program,
    observable: Observable,
    slots: Sequence[ParameterSlot],
    *,
    backend: str = "auto",
    method: GradientMethod = GradientMethod.AUTO,
) -> Callable[[Any], Any]: ...


def make_torch_expectation(
    program: Program,
    observable: Observable,
    slots: Sequence[ParameterSlot],
    *,
    backend: str = "auto",
    method: GradientMethod = GradientMethod.AUTO,
) -> Callable[[Any], Any]: ...
