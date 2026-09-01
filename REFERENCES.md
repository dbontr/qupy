# References

## Native Python integration
- [nanobind documentation](https://nanobind.readthedocs.io/en/latest/) — nanobind 3.x. Used for: low-overhead C++/Python bindings and native type exposure.
- [nanobind ndarray](https://nanobind.readthedocs.io/en/latest/ndarray.html) — nanobind 3.x. Used for: ownership-safe zero-copy NumPy result views.
- [scikit-build-core getting started](https://scikit-build-core.readthedocs.io/en/latest/guide/getting_started.html) — scikit-build-core. Used for: CMake-backed Python packaging.

## Parallel numerical execution
- [OpenMP API specifications](https://www.openmp.org/specifications/) — OpenMP Architecture Review Board, OpenMP 6.0. Used for: portable CPU parallel execution of amplitude kernels.
- [NumPy Array API compatibility](https://numpy.org/devdocs/reference/array_api.html) — NumPy. Used for: classical array interoperability principles at the Python boundary.

## Program identity
- [FIPS 180-4 Secure Hash Standard](https://doi.org/10.6028/NIST.FIPS.180-4) — NIST. Used for: portable SHA-256 program and target fingerprints for execution-plan cache identity.

## Stabilizer and Pauli methods
- [Improved Simulation of Stabilizer Circuits](https://doi.org/10.1103/PhysRevA.70.052328) — Scott Aaronson and Daniel Gottesman, Physical Review A 70, 052328 (2004), arXiv:quant-ph/0406196. Used for: Clifford and Pauli conjugation principles behind exact backward Pauli propagation.

## Quantum execution architecture
- [Qiskit Target](https://quantum.cloud.ibm.com/docs/en/api/qiskit/2.3/qiskit.transpiler.Target) — IBM Quantum, Qiskit 2.3. Used for: separation of target capability data from execution.
- [AerSimulator](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html) — Qiskit Aer. Used for: result-aware simulator method and backend architecture comparisons.
- [qsim](https://github.com/quantumlib/qsim) — Google Quantum AI. Used for: native state-vector and gate-fusion optimization design comparison.
- [CircuitToEinsum](https://docs.nvidia.com/cuda/cuquantum/26.06.0/python/generated/cuquantum.tensornet.CircuitToEinsum.html) — NVIDIA cuQuantum 26.06. Used for: reverse-lightcone expectation reduction design comparison.

## Tooling
- [setup-uv](https://github.com/astral-sh/setup-uv) — Astral, v9.0.0. Used for: pinned uv installation in GitHub Actions CI.
- [actions/checkout](https://github.com/actions/checkout) — GitHub, v7.0.1. Used for: source checkout in CI with an immutable commit pin.
- [actions/setup-python](https://github.com/actions/setup-python) — GitHub, v7.0.0. Used for: Python matrix setup in CI with an immutable commit pin.
