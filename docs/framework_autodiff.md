# JAX and PyTorch autodiff interoperability

QuPy can expose a parameterized quantum expectation as a scalar function understood by JAX or PyTorch while keeping QuPy's native execution and gradient engines authoritative.

The framework packages are optional. Installing `qupy-compute` does not install or import JAX or PyTorch. A framework is imported only when its adapter factory is called.

## Parameterized QuPy objective

Both adapters start from the same native objects:

```python
import qupy as qp

program = qp.ry(qp.Program(1), 0.0, 0)
observable = qp.Observable(
    [qp.PauliTerm(1.0, [qp.PauliFactor(0, qp.Pauli.Z)])]
)
slots = [qp.ParameterSlot(0, 0)]
```

The parameter vector is one-dimensional and must contain one `float32` or `float64` value per slot. QuPy evaluates parameters internally in `float64`; the adapter returns the scalar value and gradient in the framework input dtype.

## JAX

Create a JAX scalar objective with:

```python
import jax
import jax.numpy as jnp

objective = qp.make_jax_expectation(
    program,
    observable,
    slots,
    backend="native-cpu",
    method=qp.GradientMethod.ADJOINT,
)

parameters = jnp.array([0.37], dtype=jnp.float64)
value = objective(parameters)
gradient = jax.grad(objective)(parameters)
compiled_value = jax.jit(objective)(parameters)
compiled_gradient = jax.jit(jax.grad(objective))(parameters)
```

The adapter uses `jax.pure_callback` to execute QuPy on the host and a `jax.custom_jvp` rule backed by QuPy's native `value_and_grad`. First-order reverse-mode gradients, JVPs, `jax.jit`, and sequential `jax.vmap` are supported.

This does not compile QuPy into XLA. A JAX accelerator array is transferred to the host callback, QuPy executes using the selected QuPy backend, and the result returns to JAX. Selecting `backend="native-cuda"` asks QuPy to use its own CUDA backend; it does not share JAX device buffers or streams.

Higher-order JAX derivatives are not part of this contract because the native QuPy gradient returned by the host callback is intentionally opaque to JAX differentiation. Invalid values detected inside a staged callback can also surface as a JAX runtime callback error rather than the original Python exception type.

## PyTorch

Create a PyTorch scalar objective with:

```python
import torch

objective = qp.make_torch_expectation(
    program,
    observable,
    slots,
    backend="native-cpu",
    method=qp.GradientMethod.ADJOINT,
)

parameters = torch.tensor([0.37], dtype=torch.float64, requires_grad=True)
value = objective(parameters)
value.backward()
print(parameters.grad)
```

The adapter is a standard `torch.autograd.Function`. The forward pass asks QuPy for the native value and gradient once, stores that gradient on the input tensor's device and dtype, and the backward pass multiplies it by the incoming scalar cotangent.

CUDA PyTorch tensors are copied to host `float64` for QuPy evaluation and the result/gradient are copied back to the original PyTorch device. QuPy and PyTorch do not share device memory or streams in this interface.

Standard first-order PyTorch autograd is supported and checked with `torch.autograd.gradcheck`. Higher-order derivatives and `torch.func` transforms such as `vmap` are not part of the current contract.

## Backend and gradient semantics

The adapter does not implement an independent simulator or differentiation rule. It calls QuPy's native `value_and_grad`, so backend validation, observable semantics, parameter-slot validation, and gradient-method selection remain owned by QuPy.

`backend="auto"` follows the same native gradient execution rules available to `qp.value_and_grad`. Explicit backends remain explicit. Framework device placement is separate from QuPy backend selection.

## Optional dependency verification

QuPy's normal CI runs without JAX or PyTorch and verifies that the core package remains framework-independent. A separate framework-interoperability workflow installs pinned CPU releases of both frameworks and runs the adapter conformance suite.

The dedicated tests cover:

- JAX direct execution, `grad`, `jit`, `jvp`, and sequential `vmap`;
- PyTorch `backward` and numerical `gradcheck`;
- `float32` and `float64` result/gradient preservation;
- parameter shape and dtype validation.

This keeps optional ecosystem integration from becoming a mandatory runtime dependency or weakening QuPy's native ownership boundaries.
