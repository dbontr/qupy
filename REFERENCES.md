# References

## Native Python integration
- [nanobind documentation](https://nanobind.readthedocs.io/en/latest/) — nanobind 3.x. Used for: low-overhead C++/Python bindings and native type exposure.
- [nanobind ndarray](https://nanobind.readthedocs.io/en/latest/ndarray.html) — nanobind 3.x. Used for: ownership-safe zero-copy NumPy result views.
- [scikit-build-core getting started](https://scikit-build-core.readthedocs.io/en/latest/guide/getting_started.html) — scikit-build-core. Used for: CMake-backed Python packaging.

## Parallel numerical execution
- [OpenMP API specifications](https://www.openmp.org/specifications/) — OpenMP Architecture Review Board, OpenMP 6.0. Used for: portable CPU parallel execution of amplitude kernels.
- [NumPy Array API compatibility](https://numpy.org/devdocs/reference/array_api.html) — NumPy. Used for: classical array interoperability principles at the Python boundary.

## GPU execution
- [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/) — NVIDIA CUDA 13.x. Used for: runtime CUDA device/context management, module loading, memory operations, and kernel launch without a toolkit build dependency.
- [Parallel Thread Execution ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) — NVIDIA PTX ISA 9.3. Used for: embedded portable PTX gate, Pauli-expectation, shared-memory block-reduction, and recursive complex-reduction kernels.

## Program identity
- [FIPS 180-4 Secure Hash Standard](https://doi.org/10.6028/NIST.FIPS.180-4) — NIST. Used for: portable SHA-256 program and target fingerprints for execution-plan cache identity.

## Tensor-network and MPS execution
- [Efficient Classical Simulation of Slightly Entangled Quantum Computations](https://doi.org/10.1103/PhysRevLett.91.147902) — Guifre Vidal, Physical Review Letters 91, 147902 (2003). Used for: exact MPS simulation scaling with restricted entanglement and bond-dimension structure.
- [The density-matrix renormalization group in the age of matrix product states](https://doi.org/10.1016/j.aop.2010.09.012) — Ulrich Schollwock, Annals of Physics 326 (2011). Used for: MPS canonical forms, local tensor updates, and SVD-based bond factorization.
- [cuTensorNet overview](https://docs.nvidia.com/cuda/cuquantum/latest/cutensornet/overview.html) — NVIDIA cuQuantum. Used for: gate-split tensor-network execution and MPS contraction/decomposition architecture comparison.

## Stabilizer and Pauli methods
- [Improved Simulation of Stabilizer Circuits](https://doi.org/10.1103/PhysRevA.70.052328) — Scott Aaronson and Daniel Gottesman, Physical Review A 70, 052328 (2004), arXiv:quant-ph/0406196. Used for: Clifford tableau conjugation, stabilizer-state simulation principles, and exact backward Pauli propagation.

## Quantum execution architecture and benchmark adapters
- [Qiskit Target](https://quantum.cloud.ibm.com/docs/en/api/qiskit/2.3/qiskit.transpiler.Target) — IBM Quantum, Qiskit 2.3. Used for: separation of target capability data from execution.
- [AerSimulator](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html) — Qiskit Aer. Used for: result-aware simulator architecture and state-vector/stabilizer benchmark adapters.
- [qsim](https://github.com/quantumlib/qsim) — Google Quantum AI. Used for: native state-vector optimization comparison and the qsim expectation benchmark adapter.
- [Stim](https://github.com/quantumlib/Stim) — Google Quantum AI. Used for: independent stabilizer/tableau expectation benchmark comparisons.
- [CircuitToEinsum](https://docs.nvidia.com/cuda/cuquantum/26.06.0/python/generated/cuquantum.tensornet.CircuitToEinsum.html) — NVIDIA cuQuantum 26.06. Used for: reverse-lightcone expectation reduction design comparison.

## Tooling
- [setup-uv](https://github.com/astral-sh/setup-uv) — Astral, v9.0.0. Used for: pinned uv installation in GitHub Actions CI.
- [actions/checkout](https://github.com/actions/checkout) — GitHub, v7.0.1. Used for: source checkout in CI with an immutable commit pin.
- [actions/setup-python](https://github.com/actions/setup-python) — GitHub, v7.0.0. Used for: Python matrix setup in CI with an immutable commit pin.

## Quantum differentiation
- [Evaluating analytic gradients on quantum hardware](https://doi.org/10.1103/PhysRevA.99.032331) — Maria Schuld, Ville Bergholm, Christian Gogolin, Josh Izaac, and Nathan Killoran, Physical Review A 99, 032331 (2019). Used for: analytic parameter-shift differentiation of variational quantum circuits.
- [General parameter-shift rules for quantum gradients](https://arxiv.org/abs/2107.12390) — David Wierichs, Josh Izaac, Cody Wang, and Cedric Yen-Yu Lin (2021). Used for: higher-order parameter-shift structure and resource analysis.
- [Efficient calculation of gradients in classical simulations of variational quantum algorithms](https://arxiv.org/abs/2009.02823) — Tyson Jones and Julien Gacon (2020). Used for: reverse/adjoint state-vector gradient architecture.

## Open-system dynamics
- [Completely positive dynamical semigroups of N-level systems](https://doi.org/10.1063/1.522979) — Vittorio Gorini, Andrzej Kossakowski, and E. C. G. Sudarshan, Journal of Mathematical Physics 17 (1976). Used for: the GKSL generator and Markovian open-system evolution.
- [On the generators of quantum dynamical semigroups](https://doi.org/10.1007/BF01608499) — Goran Lindblad, Communications in Mathematical Physics 48 (1976). Used for: the Lindblad master-equation generator and completely positive dynamics.

## QPU interchange and distributed execution
- [OpenQASM 3.1 Specification](https://openqasm.com/versions/3.1/index.html) — OpenQASM contributors, specification 3.1. Used for: portable textual quantum-program interchange and gate/measurement syntax.
- [QIR Base Profile](https://github.com/qir-alliance/qir-spec/blob/main/specification/profiles/Base_Profile.md) — QIR Alliance. Used for: LLVM-based QIR entry-point, profile, qubit/result, and output-recording requirements.
- [MPI: A Message-Passing Interface Standard Version 5.0](https://www.mpi-forum.org/docs/mpi-5.0/mpi50-report/mpi50-report.htm) — MPI Forum, MPI 5.0 (2025). Used for: distributed state-vector point-to-point communication, communicator, and initialization semantics.

## Quantum error correction
- [Stim detector error model format](https://github.com/quantumlib/Stim/blob/main/doc/file_format_dem_detector_error_model.md) — Google Quantum AI, Stim. Used for: detector-event, independent-error-mechanism, and logical-frame-change semantics used by QuPy detector models.
