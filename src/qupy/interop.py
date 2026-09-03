from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from . import _native


_FLOAT_DTYPES = {np.dtype(np.float32), np.dtype(np.float64)}


def _require_optional(module_name: str, install_hint: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name or (
            exc.name is not None and exc.name.startswith(f"{module_name}.")
        ):
            raise ImportError(
                f"{module_name} is required for this QuPy adapter; install {install_hint}"
            ) from exc
        raise


def _slots_tuple(slots: Sequence[_native.ParameterSlot]) -> tuple[_native.ParameterSlot, ...]:
    result = tuple(slots)
    if not result:
        raise ValueError("slots must contain at least one parameter slot")
    return result


def _evaluate_value_and_gradient(
    program: _native.Program,
    observable: _native.Observable,
    slots: tuple[_native.ParameterSlot, ...],
    parameters: np.ndarray[Any, np.dtype[np.floating[Any]]],
    *,
    backend: str,
    method: _native.GradientMethod,
) -> tuple[float, np.ndarray[Any, np.dtype[np.float64]]]:
    values = np.asarray(parameters, dtype=np.float64)
    if values.ndim != 1 or values.shape[0] != len(slots):
        raise ValueError(f"parameters must have shape ({len(slots)},)")
    if not np.all(np.isfinite(values)):
        raise ValueError("parameters must contain only finite values")
    result = _native.value_and_grad(
        program,
        observable,
        list(slots),
        values,
        backend=backend,
        method=method,
    )
    gradient = np.asarray(result.gradient, dtype=np.float64)
    if gradient.shape != values.shape:
        raise RuntimeError("native QuPy gradient shape does not match the parameter vector")
    return float(result.value), gradient


def make_jax_expectation(
    program: _native.Program,
    observable: _native.Observable,
    slots: Sequence[_native.ParameterSlot],
    *,
    backend: str = "auto",
    method: _native.GradientMethod = _native.GradientMethod.AUTO,
) -> Callable[[Any], Any]:
    """Create a first-order differentiable JAX scalar expectation function."""
    jax = _require_optional("jax", 'the optional JAX dependency, for example "jax[cpu]"')
    jnp = _require_optional("jax.numpy", 'the optional JAX dependency, for example "jax[cpu]"')
    slot_tuple = _slots_tuple(slots)
    parameter_count = len(slot_tuple)

    def validate(parameters: Any) -> Any:
        array = jnp.asarray(parameters)
        if array.ndim != 1 or array.shape[0] != parameter_count:
            raise ValueError(f"parameters must have shape ({parameter_count},)")
        if np.dtype(array.dtype) not in _FLOAT_DTYPES:
            raise TypeError("JAX parameters must use float32 or float64")
        return array

    def host_value(parameters: Any) -> np.ndarray[Any, np.dtype[Any]]:
        array = np.asarray(parameters)
        value, _ = _evaluate_value_and_gradient(
            program,
            observable,
            slot_tuple,
            array,
            backend=backend,
            method=method,
        )
        return np.asarray(value, dtype=array.dtype)

    def host_value_and_gradient(
        parameters: Any,
    ) -> tuple[np.ndarray[Any, np.dtype[Any]], np.ndarray[Any, np.dtype[Any]]]:
        array = np.asarray(parameters)
        value, gradient = _evaluate_value_and_gradient(
            program,
            observable,
            slot_tuple,
            array,
            backend=backend,
            method=method,
        )
        return (
            np.asarray(value, dtype=array.dtype),
            np.asarray(gradient, dtype=array.dtype),
        )

    @jax.custom_jvp
    def expectation(parameters: Any) -> Any:
        array = validate(parameters)
        result_spec = jax.ShapeDtypeStruct((), array.dtype)
        return jax.pure_callback(
            host_value,
            result_spec,
            array,
            vmap_method="sequential",
        )

    @expectation.defjvp
    def expectation_jvp(primals: tuple[Any, ...], tangents: tuple[Any, ...]) -> tuple[Any, Any]:
        (parameters,) = primals
        (tangent,) = tangents
        array = validate(parameters)
        result_spec = jax.ShapeDtypeStruct((), array.dtype)
        gradient_spec = jax.ShapeDtypeStruct((parameter_count,), array.dtype)
        value, gradient = jax.pure_callback(
            host_value_and_gradient,
            (result_spec, gradient_spec),
            array,
            vmap_method="sequential",
        )
        return value, jnp.sum(gradient * tangent)

    return expectation


def make_torch_expectation(
    program: _native.Program,
    observable: _native.Observable,
    slots: Sequence[_native.ParameterSlot],
    *,
    backend: str = "auto",
    method: _native.GradientMethod = _native.GradientMethod.AUTO,
) -> Callable[[Any], Any]:
    """Create a first-order differentiable PyTorch scalar expectation function."""
    torch = _require_optional("torch", "the optional PyTorch dependency")
    slot_tuple = _slots_tuple(slots)
    parameter_count = len(slot_tuple)

    class _ExpectationFunction(torch.autograd.Function):  # type: ignore[misc]
        @staticmethod
        def forward(ctx: Any, parameters: Any) -> Any:
            if not isinstance(parameters, torch.Tensor):
                raise TypeError("PyTorch parameters must be a torch.Tensor")
            if parameters.ndim != 1 or parameters.shape[0] != parameter_count:
                raise ValueError(f"parameters must have shape ({parameter_count},)")
            if parameters.dtype not in (torch.float32, torch.float64):
                raise TypeError("PyTorch parameters must use float32 or float64")

            host_values = parameters.detach().to(device="cpu", dtype=torch.float64).numpy()
            value, gradient = _evaluate_value_and_gradient(
                program,
                observable,
                slot_tuple,
                host_values,
                backend=backend,
                method=method,
            )
            gradient_tensor = torch.as_tensor(
                gradient,
                dtype=parameters.dtype,
                device=parameters.device,
            )
            ctx.save_for_backward(gradient_tensor)
            return parameters.new_tensor(value)

        @staticmethod
        def backward(ctx: Any, grad_output: Any) -> tuple[Any]:
            (gradient,) = ctx.saved_tensors
            return (grad_output * gradient,)

    def expectation(parameters: Any) -> Any:
        return _ExpectationFunction.apply(parameters)

    return expectation


__all__ = ["make_jax_expectation", "make_torch_expectation"]
