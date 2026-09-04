from __future__ import annotations

import math

import numpy as np
import pytest

import qupy as qp
from qupy import interop


def _objective() -> tuple[qp.Program, qp.Observable, list[qp.ParameterSlot]]:
    program = qp.ry(qp.Program(1), 0.0, 0)
    observable = qp.Observable(
        [qp.PauliTerm(1.0, [qp.PauliFactor(0, qp.Pauli.Z)])]
    )
    return program, observable, [qp.ParameterSlot(0, 0)]


@pytest.mark.parametrize(
    ("factory_name", "missing_module"),
    [
        ("make_jax_expectation", "jax"),
        ("make_torch_expectation", "torch"),
    ],
)
def test_optional_framework_factories_report_missing_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    factory_name: str,
    missing_module: str,
) -> None:
    def missing_import(name: str) -> object:
        if name == missing_module:
            raise ModuleNotFoundError(f"No module named '{name}'", name=name)
        raise AssertionError(f"unexpected optional import {name!r}")

    monkeypatch.setattr(interop.importlib, "import_module", missing_import)
    program, observable, slots = _objective()
    factory = getattr(qp, factory_name)

    with pytest.raises(ImportError, match=f"{missing_module} is required"):
        factory(program, observable, slots)


def test_framework_adapter_does_not_mask_transitive_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_jax(name: str) -> object:
        if name == "jax":
            raise ModuleNotFoundError("No module named 'jaxlib'", name="jaxlib")
        raise AssertionError(f"unexpected optional import {name!r}")

    monkeypatch.setattr(interop.importlib, "import_module", broken_jax)
    program, observable, slots = _objective()

    with pytest.raises(ModuleNotFoundError) as raised:
        qp.make_jax_expectation(program, observable, slots)
    assert raised.value.name == "jaxlib"


def test_jax_expectation_supports_grad_jit_jvp_and_vmap() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    jax.config.update("jax_enable_x64", True)

    program, observable, slots = _objective()
    function = qp.make_jax_expectation(
        program,
        observable,
        slots,
        backend="native-cpu",
        method=qp.GradientMethod.ADJOINT,
    )
    theta = 0.371
    parameters = jnp.asarray([theta], dtype=jnp.float64)

    assert float(function(parameters)) == pytest.approx(math.cos(theta), abs=1e-12)
    gradient = jax.grad(function)(parameters)
    np.testing.assert_allclose(np.asarray(gradient), [-math.sin(theta)], atol=1e-12)

    jitted_value = jax.jit(function)(parameters)
    jitted_gradient = jax.jit(jax.grad(function))(parameters)
    assert float(jitted_value) == pytest.approx(math.cos(theta), abs=1e-12)
    np.testing.assert_allclose(np.asarray(jitted_gradient), [-math.sin(theta)], atol=1e-12)

    value, directional = jax.jvp(function, (parameters,), (jnp.ones_like(parameters),))
    assert float(value) == pytest.approx(math.cos(theta), abs=1e-12)
    assert float(directional) == pytest.approx(-math.sin(theta), abs=1e-12)

    batch = jnp.asarray([[0.1], [0.3], [-0.7]], dtype=jnp.float64)
    mapped = jax.vmap(function)(batch)
    np.testing.assert_allclose(
        np.asarray(mapped),
        np.cos(np.asarray(batch[:, 0])),
        atol=1e-12,
    )


def test_jax_expectation_validates_parameter_shape_and_dtype() -> None:
    jnp = pytest.importorskip("jax.numpy")
    program, observable, slots = _objective()
    function = qp.make_jax_expectation(program, observable, slots, backend="native-cpu")

    with pytest.raises(ValueError, match=r"shape \(1,\)"):
        function(jnp.asarray([[0.1]], dtype=jnp.float32))
    with pytest.raises(TypeError, match="float32 or float64"):
        function(jnp.asarray([1], dtype=jnp.int32))


def test_torch_expectation_supports_backward_and_gradcheck() -> None:
    torch = pytest.importorskip("torch")
    program, observable, slots = _objective()
    function = qp.make_torch_expectation(
        program,
        observable,
        slots,
        backend="native-cpu",
        method=qp.GradientMethod.ADJOINT,
    )
    theta = 0.371
    parameters = torch.tensor([theta], dtype=torch.float64, requires_grad=True)

    value = function(parameters)
    assert value.dtype == parameters.dtype
    assert value.device == parameters.device
    assert value.item() == pytest.approx(math.cos(theta), abs=1e-12)
    value.backward()
    assert parameters.grad is not None
    np.testing.assert_allclose(
        parameters.grad.detach().cpu().numpy(),
        [-math.sin(theta)],
        atol=1e-12,
    )

    gradcheck_parameters = torch.tensor([0.213], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        function,
        (gradcheck_parameters,),
        eps=1e-6,
        atol=1e-8,
        rtol=1e-6,
    )


def test_torch_expectation_preserves_float32_and_validates_inputs() -> None:
    torch = pytest.importorskip("torch")
    program, observable, slots = _objective()
    function = qp.make_torch_expectation(program, observable, slots, backend="native-cpu")

    parameters = torch.tensor([0.2], dtype=torch.float32, requires_grad=True)
    value = function(parameters)
    value.backward()
    assert value.dtype == torch.float32
    assert parameters.grad is not None
    assert parameters.grad.dtype == torch.float32
    assert parameters.grad.item() == pytest.approx(-math.sin(0.2), abs=2e-7)

    with pytest.raises(ValueError, match=r"shape \(1,\)"):
        function(torch.tensor([[0.2]], dtype=torch.float32))
    with pytest.raises(TypeError, match="float32 or float64"):
        function(torch.tensor([1], dtype=torch.int64))
