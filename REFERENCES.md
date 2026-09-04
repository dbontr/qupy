# References

## Native Python integration
- [nanobind documentation](https://nanobind.readthedocs.io/en/latest/) — nanobind 3.x. Used for: low-overhead C++/Python bindings and native type exposure.
- [nanobind ndarray](https://nanobind.readthedocs.io/en/latest/ndarray.html) — nanobind 3.x. Used for: ownership-safe zero-copy NumPy result views.
- [scikit-build-core getting started](https://scikit-build-core.readthedocs.io/en/latest/guide/getting_started.html) — scikit-build-core. Used for: CMake-backed Python packaging.
- [PEP 561 — Distributing and Packaging Type Information](https://peps.python.org/pep-0561/) — Python Packaging/Typing specification. Used for: shipping the `py.typed` marker alongside QuPy's inline annotations and extension-module stubs.
- [PyPI Trove classifiers](https://pypi.org/classifiers/) — Python Package Index. Used for: standardized alpha, platform, language, scientific-computing, and `Typing :: Typed` release metadata.

## Framework autodiff interoperability
- [JAX pure_callback](https://docs.jax.dev/en/latest/_autosummary/jax.pure_callback.html) — JAX. Used for: JIT-compatible host execution of QuPy objectives without making JAX a QuPy runtime dependency.
- [JAX custom_jvp](https://docs.jax.dev/en/latest/_autosummary/jax.custom_jvp.html) — JAX. Used for: first-order JVP and reverse-mode rules backed by QuPy native gradients.
- [Extending PyTorch](https://docs.pytorch.org/docs/stable/notes/extending.html) — PyTorch. Used for: standard `torch.autograd.Function` integration of external numerical code.
- [Extending torch.func with autograd.Function](https://docs.pytorch.org/docs/stable/notes/extending.func.html) — PyTorch. Used for: defining the current transform-composability boundary and avoiding unsupported `torch.func` claims.

## Parallel numerical execution
- [OpenMP API specifications](https://www.openmp.org/specifications/) — OpenMP Architecture Review Board, OpenMP 6.0. Used for: portable CPU parallel execution of amplitude kernels.
- [NumPy Array API compatibility](https://numpy.org/devdocs/reference/array_api.html) — NumPy. Used for: classical array interoperability principles at the Python boundary.

## GPU execution
- [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/) — NVIDIA CUDA 13.x. Used for: runtime CUDA device/context management, module loading, memory operations, and kernel launch without a toolkit build dependency.
- [Parallel Thread Execution ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) — NVIDIA PTX ISA 9.3. Used for: embedded portable PTX gate, density-matrix superoperator, Pauli-expectation, shared-memory block-reduction, and recursive complex-reduction kernels.

## Program identity
- [FIPS 180-4 Secure Hash Standard](https://doi.org/10.6028/NIST.FIPS.180-4) — NIST. Used for: portable SHA-256 program, circuit, and target fingerprints for semantic and execution identity.

## Tensor-network and MPS execution
- [Efficient Classical Simulation of Slightly Entangled Quantum Computations](https://doi.org/10.1103/PhysRevLett.91.147902) — Guifre Vidal, Physical Review Letters 91, 147902 (2003). Used for: exact MPS simulation scaling with restricted entanglement and bond-dimension structure.
- [The density-matrix renormalization group in the age of matrix product states](https://doi.org/10.1016/j.aop.2010.09.012) — Ulrich Schollwock, Annals of Physics 326 (2011). Used for: MPS canonical forms, local tensor updates, and SVD-based bond factorization.
- [Simulating quantum computation by contracting tensor networks](https://arxiv.org/abs/quant-ph/0511069) — Igor Markov and Yaoyun Shi (2005). Used for: exact circuit simulation by tensor contraction and the relationship between contraction complexity and graph/treewidth structure.
- [Constructing Optimal Contraction Trees for Tensor Network Quantum Circuit Simulation](https://arxiv.org/abs/2209.02895) — Cameron Ibrahim, Danylo Lykov, Zichang He, Yuri Alexeev, and Ilya Safro (2022). Used for: contraction-tree cost modeling and deterministic contraction-order architecture comparison.
- [cuTensorNet overview](https://docs.nvidia.com/cuda/cuquantum/latest/cutensornet/overview.html) — NVIDIA cuQuantum. Used for: gate-split tensor-network execution and MPS/general contraction architecture comparison.

## Stabilizer and Pauli methods
- [Improved Simulation of Stabilizer Circuits](https://doi.org/10.1103/PhysRevA.70.052328) — Scott Aaronson and Daniel Gottesman, Physical Review A 70, 052328 (2004), arXiv:quant-ph/0406196. Used for: Clifford tableau conjugation, stabilizer-state simulation principles, and exact backward Pauli propagation.

## Quantum execution architecture and benchmark adapters
- [Qiskit Target](https://quantum.cloud.ibm.com/docs/en/api/qiskit/2.3/qiskit.transpiler.Target) — IBM Quantum, Qiskit 2.3. Used for: target-owned operation, qubit, coupling, and instruction-property constraints for hardware compilation architecture comparison.
- [Transpiler stages](https://quantum.cloud.ibm.com/docs/en/guides/transpiler-stages) — IBM Quantum. Used for: layout, routing, basis-translation, optimization, and scheduling stage boundaries for target-aware compilation architecture comparison.
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

## Algorithm construction
- [An approximate Fourier transform useful in quantum factoring](https://arxiv.org/abs/quant-ph/0201067) — Don Coppersmith (1994/2002). Used for: the controlled-phase structure and bit-ordering conventions of the quantum Fourier transform circuit.
- [Hardware-efficient variational quantum eigensolver for small molecules and quantum magnets](https://doi.org/10.1038/nature23879) — Abhinav Kandala et al., Nature 549 (2017). Used for: layered hardware-efficient rotation-and-entanglement ansatz design.
- [Simulation of electronic structure Hamiltonians using quantum computers](https://arxiv.org/abs/1001.3855) — James D. Whitfield, Jacob Biamonte, and Alán Aspuru-Guzik (2011). Used for: Pauli-string basis changes, parity accumulation, and exponentiation patterns for Hamiltonian simulation.

## Open-system dynamics
- [Completely positive dynamical semigroups of N-level systems](https://doi.org/10.1063/1.522979) — Vittorio Gorini, Andrzej Kossakowski, and E. C. G. Sudarshan, Journal of Mathematical Physics 17 (1976). Used for: the GKSL generator and Markovian open-system evolution.
- [On the generators of quantum dynamical semigroups](https://doi.org/10.1007/BF01608499) — Goran Lindblad, Communications in Mathematical Physics 48 (1976). Used for: the Lindblad master-equation generator and completely positive dynamics.
- [Wave-function approach to dissipative processes in quantum optics](https://doi.org/10.1103/PhysRevLett.68.580) — Jean Dalibard, Yvan Castin, and Klaus Molmer, Physical Review Letters 68, 580 (1992). Used for: Monte Carlo wave-function/quantum-trajectory evolution, stochastic branch selection, normalization, and ensemble-equivalent open-system observables.

## QPU interchange and distributed execution
- [OpenQASM 3.1 Specification](https://openqasm.com/versions/3.1/index.html) — OpenQASM contributors, specification 3.1. Used for: supported-subset parsing and serialization of portable gate calls, measurement, reset, barriers, register declarations, and classical feed-forward syntax.
- [QIR Base Profile](https://github.com/qir-alliance/qir-spec/blob/main/specification/profiles/Base_Profile.md) — QIR Alliance. Used for: LLVM-based QIR entry-point, profile, qubit/result, and output-recording requirements.
- [MPI: A Message-Passing Interface Standard Version 5.0](https://www.mpi-forum.org/docs/mpi-5.0/mpi50-report/mpi50-report.htm) — MPI Forum, MPI 5.0 (2025). Used for: distributed state-vector point-to-point communication, collective broadcast and reduction, communicator topology, and initialization semantics.

## Provider adapters
- [Run your circuits with OpenQASM 3.0](https://docs.aws.amazon.com/braket/latest/developerguide/braket-openqasm.html) — Amazon Web Services. Used for: the Amazon Braket OpenQASM 3.0 provider transport profile and gate-model submission boundary.
- [Create and submit an example OpenQASM 3.0 quantum task](https://docs.aws.amazon.com/braket/latest/developerguide/braket-openqasm-create-submit-task.html) — Amazon Web Services. Used for: `braket.ir.openqasm.Program` construction and `device.run(..., shots=...)` submission semantics.
- [Tracking quantum tasks from the Amazon Braket SDK](https://docs.aws.amazon.com/braket/latest/developerguide/braket-monitor-tasks-sdk.html) — Amazon Web Services. Used for: task identifiers, queued/running/completed lifecycle polling, result retrieval, and cancellation semantics.
- [Testing a quantum task with the local simulator](https://docs.aws.amazon.com/braket/latest/developerguide/braket-send-to-local-simulator.html) — Amazon Web Services. Used for: credential-free `LocalSimulator` integration conformance.
- [Amazon Braket Python SDK](https://github.com/amazon-braket/amazon-braket-sdk-python) — Amazon Web Services. Used for: optional SDK integration and the pinned provider-interoperability test dependency.

## Quantum error correction
- [Stim detector error model format](https://github.com/quantumlib/Stim/blob/main/doc/file_format_dem_detector_error_model.md) — Google Quantum AI, Stim. Used for: detector-event, independent-error-mechanism, logical-frame-change, and decomposition-separator semantics used by QuPy detector models and QEC reference workloads.
- [Decoding across the quantum low-density parity-check code landscape](https://doi.org/10.1103/PhysRevResearch.2.043423) — Joschka Roffe, David R. White, Simon Burton, and Earl Campbell, Physical Review Research 2, 043423 (2020). Used for: belief-propagation plus ordered-statistics post-processing architecture for quantum error decoding.
- [Soft-decision decoding of linear block codes based on ordered statistics](https://doi.org/10.1109/18.412683) — Marc P. C. Fossorier and Shu Lin, IEEE Transactions on Information Theory 41(5), 1379-1396 (1995). Used for: reliability ordering and order-0 ordered-statistics reprocessing principles.
- [PyMatching documentation](https://pymatching.readthedocs.io/) — PyMatching 2.x. Used for: detector-error-model construction and batch-decoding interfaces in the optional QEC benchmark comparator.
- [Sparse Blossom: correcting a million errors per core second with minimum-weight matching](https://doi.org/10.22331/q-2025-01-20-1600) — Oscar Higgott and Craig Gidney, Quantum 9, 1600 (2025). Used for: sparse-blossom minimum-weight perfect matching as the independent surface-code benchmark baseline.
