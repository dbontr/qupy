# References

## Numerical and compiler architecture
- [NumPy Array API standard compatibility](https://numpy.org/devdocs/reference/array_api.html) — NumPy. Used for: classical array interoperability boundary.
- [JAX internals: the jaxpr language](https://docs.jax.dev/en/latest/jaxpr.html) — JAX. Used for: typed functional intermediate-representation design principles.

## Quantum execution and targets
- [Qiskit Target](https://quantum.cloud.ibm.com/docs/en/api/qiskit/2.3/qiskit.transpiler.Target) — IBM Quantum, Qiskit 2.3. Used for: separation of backend capability description from execution.
- [Executing Kernels](https://nvidia.github.io/cuda-quantum/latest/using/examples/executing_kernels.html) — NVIDIA CUDA-Q. Used for: explicit sample, expectation, and simulator-state execution semantics.
- [AerSimulator](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html) — Qiskit Aer. Used for: simulation-strategy taxonomy and automatic-method precedent.
- [NVIDIA cuQuantum SDK](https://docs.nvidia.com/cuda/cuquantum/latest/) — NVIDIA. Used for: GPU backend roadmap across state-vector, stabilizer, density-matrix, Pauli-propagation, and tensor-network methods.

## Interchange specifications
- [OpenQASM 3.1 Specification](https://openqasm.com/versions/3.1/index.html) — OpenQASM. Used for: planned circuit and dynamic-program interchange lowering.
- [Quantum Intermediate Representation](https://github.com/qir-alliance/qir-spec) — QIR Alliance. Used for: planned lower-level heterogeneous quantum compiler interchange.

## Tooling
- [setup-uv](https://github.com/astral-sh/setup-uv) — Astral, v9.0.0. Used for: pinned uv installation in GitHub Actions CI.