# QuPy

QuPy is a native quantum numerical-computing library with a compact Python interface. The execution core is C++20; Python composes native programs, requests numerical results, and interoperates with the surrounding scientific-computing ecosystem.

QuPy is designed around one rule: **execution strategy may change, quantum semantics may not**. Dense CPU, CUDA, stabilizer, MPS, general tensor-network, MPI, noisy-density, trajectory, provider, and hardware-compilation paths share explicit immutable identities and fail closed when a requested capability is unavailable.

> **Status:** early alpha (`0.3.0a0`). The native core is usable and heavily tested, but the public API can still change before 1.0.

## Package identity

The Python distribution is named **`qupy-compute`** and the import package is **`qupy`**.

Do not assume a package published under the bare distribution name `qupy` is this repository. Until a QuPy release is published through this project, install from source.

```text
git clone https://github.com/dbontr/qupy.git
cd qupy
python -m pip install .
```

For development, the repository uses `uv`:

```text
uv sync
uv run pytest -q
```

QuPy requires Python 3.12+ and a C++20 compiler. NumPy is the only required Python runtime dependency. JAX and PyTorch integration is optional and loaded lazily.

## Quick start

```python
import qupy as qp

program = qp.Program(2)
program = qp.h(program, 0)
program = qp.cx(program, 0, 1)

samples = qp.sample(program, shots=1000, seed=7)
probabilities = qp.probabilities(program)
energy = qp.expect(program, qp.Z(0))
state = qp.statevector(program)

print(samples.counts())
print(probabilities.values)
print(energy.value)
print(state.values)
```

`Program`, execution plans, observables, and result storage are native C++ objects. Result arrays expose read-only NumPy views over native storage where possible.

## What QuPy already does

### Native numerical execution

- immutable, versioned `Program` / `Operation` execution IR with deterministic canonical text and SHA-256 identity;
- H, X, Y, Z, RX, RY, RZ, CX, CZ, and SWAP gates;
- exact dense state-vector simulation on CPU and explicit CUDA targets;
- exact probabilities, seeded sampling, expectation, variance, covariance, and observable batches;
- arbitrary real Pauli-sum observables and Hamiltonians;
- exact reverse-causal-cone reduction before observable execution;
- exact backward Pauli propagation for eligible Clifford observable cones;
- bit-packed stabilizer sampling for large Clifford programs;
- fused native kernels, workload-scaled OpenMP execution, and reusable workspaces;
- immutable native parameter binding and vectorized parameter batches.

### Specialized exact scale engines

- exact MPS execution with structural bond/routing/work estimates and adaptive MPS-to-dense continuation;
- exact general tensor-network expectation execution through explicit `native-tn`;
- deterministic tensor-network structural preflight with contraction count, peak rank, peak intermediate bytes, scalar work, and plan fingerprints;
- exact MPI-sharded state-vector execution and distributed Pauli/Hamiltonian reductions;
- MPI term-parallel general tensor-network expectation execution;
- MPI-distributed noisy quantum trajectories with collective failure propagation.

### GPU and open-system execution

- toolkit-free CUDA Driver API loading with embedded PTX JIT;
- GPU-resident state-vector execution and arbitrary-Pauli reduction;
- exact CPU/CUDA density-matrix execution;
- bit flip, phase flip, depolarizing, amplitude damping, phase damping, Pauli, and validated single-qubit Kraus channels;
- exact Monte Carlo quantum-trajectory expectation estimation;
- RK4 integration of the GKSL/Lindblad master equation.

### Differentiation and hybrid numerical workflows

- native `value_and_grad`, `grad`, `jacobian`, and `hessian`;
- adjoint, parameter-shift, and finite-difference gradient methods;
- deterministic `ParameterSlot` identity;
- first-order JAX integration backed by native QuPy gradients, including `grad`, `jit`, JVP, and sequential `vmap`;
- first-order PyTorch `autograd.Function` integration with native gradients and `gradcheck` conformance;
- semantics-preserving native circuit optimization.

### Hardware-facing execution

- separate immutable `Circuit` IR for measurement, reset, barriers, and single-bit classical feed-forward;
- deterministic `HardwareTarget` compilation with optimization, layout, routing, basis translation, dependency depth, and optional ASAP scheduling;
- OpenQASM 3.1 serialization **and import** for the supported hardware-capable subset;
- QIR Base Profile export for numerical programs;
- provider capability discovery, target discovery, compiled submission, polling, result retrieval, cancellation, and a stable plug-in C ABI;
- explicit separation between numerical execution identity and hardware-control semantics.

### Algorithms and QEC

- native `VariationalTemplate` with deterministic parameter names and tracked native slots;
- hardware-efficient variational ansätze with configurable RX/RY/RZ layers and linear/ring/none entanglement;
- composable exact QFT / inverse-QFT synthesis over arbitrary selected qubits;
- exact single-Pauli-string time evolution with explicit product-formula boundaries for noncommuting Hamiltonian sums;
- detector error models, deterministic syndrome sampling, repetition-code construction, and a bounded exact reference maximum-likelihood decoder.

## Architecture

```text
Python API
    |
    +-- algorithms / optional JAX + PyTorch adapters
    |
    v
nanobind extension
    |
    v
C++20 QuPy core
    |-- immutable Program execution IR
    |-- immutable hardware Circuit IR
    |-- HardwareTarget compiler
    |-- target capability validation
    |-- result/query-aware planner
    |-- CPU / CUDA / stabilizer / MPS / tensor-network engines
    |-- density-matrix / trajectory / Lindblad execution
    |-- MPI distributed execution
    |-- native differentiation and optimization
    `-- typed native result storage
         |
         `-- read-only NumPy views
```

The C++ core is independently buildable and directly tested with CTest. Python is not the simulator implementation.

## Execution backends

| Backend / method | Main role | Exactness / current boundary |
| --- | --- | --- |
| `native-cpu` | General dense execution and observables | Exact subject to floating-point roundoff |
| `native-cuda` | CUDA state vectors, Pauli reduction, density/noise | Exact; requires compatible NVIDIA driver |
| stabilizer | Large Clifford sampling | Exact; planner-selected for eligible workloads |
| `native-mps` / adaptive MPS | Low-entanglement exact execution | No user-visible bond truncation |
| `native-tn` | General tensor-network rich-observable expectations | Exact greedy contraction; explicit backend today |
| `native-mpi` | Distributed state vector and rich observables | Exact MPI sharding/reduction |
| distributed TN | Pauli-term-parallel tensor-network expectation | Exact MPI reduction |
| trajectories | Noisy wave-function Monte Carlo | Statistical estimator; seedable |
| density matrix | Exact open-system matrix evolution | CPU/CUDA; exponential state size |

QuPy does not silently reinterpret an explicit backend. Unsupported result modes or capabilities fail instead of falling back behind the caller's back.

## Evidence-gated automatic planning

`backend="auto"` is owned by the native planner. QuPy can load validated host-scoped cost artifacts and use them only for policies whose required evidence is present and valid.

```python
plan = qp.plan(program, qp.ResultMode.STATEVECTOR)
print(plan.backend)
print(plan.method)
print(plan.cache_key)
print(plan.predicted_ns)
```

Current promoted planner artifacts cover:

- schema v1: CPU timing evidence;
- schema v2: validated CPU/CUDA state-vector routing;
- schema v3: adaptive exact MPS policy;
- schema v4: query-aware CPU/CUDA rich-observable routing;
- schema v5: CPU/CUDA noisy-density routing.

Artifacts are bound to QuPy/workload versions and host fingerprints. CUDA evidence additionally binds the CUDA host/device fingerprint. Stale, malformed, wrong-host, or unvalidated artifacts fail closed. Explicit backends do not consult default planner discovery.

General tensor-network CPU-vs-TN routing now has its own repeated, counterbalanced, leave-one-workload-out calibration and promotion gates, but **`backend="auto"` does not select `native-tn` yet**. That evidence will enter automatic routing only after a native planner schema consumes it without weakening existing v1-v5 guarantees.

## Quantum numerical computing

An observable is a numerical query, not a backend-specific object.

```python
hamiltonian = qp.Observable(
    [
        qp.PauliTerm(
            0.5,
            [qp.PauliFactor(0, qp.Pauli.X), qp.PauliFactor(1, qp.Pauli.X)],
        ),
        qp.PauliTerm(
            0.25,
            [qp.PauliFactor(0, qp.Pauli.Z), qp.PauliFactor(1, qp.Pauli.Z)],
        ),
    ]
)

energy = qp.expect(program, hamiltonian)
plan = qp.observable_plan(program, [hamiltonian])
```

QuPy can evaluate the same semantic observable through dense CPU execution, GPU-resident Pauli reduction, exact Pauli propagation, adaptive MPS, general tensor networks, or MPI reductions when the requested path supports it.

## Variational workflows

```python
import numpy as np
import qupy as qp

template = qp.hardware_efficient_ansatz(
    4,
    3,
    rotations=("ry", "rz"),
    entanglement="ring",
)
parameters = np.zeros(template.parameter_count)

result = qp.value_and_grad(
    template.program,
    hamiltonian,
    list(template.slots),
    parameters,
)
```

The template only constructs a native program and records native parameter slots. Binding, execution, and differentiation stay in the established native ownership boundary.

### JAX

```python
objective = qp.make_jax_expectation(
    template.program,
    hamiltonian,
    template.slots,
)
# jax.grad(objective)(parameters)
```

### PyTorch

```python
objective = qp.make_torch_expectation(
    template.program,
    hamiltonian,
    template.slots,
)
# value = objective(parameters)
# value.backward()
```

JAX and PyTorch are optional dependencies. QuPy does not share framework device buffers/streams in the current adapter layer; framework arrays are bridged to native QuPy execution with explicit documented boundaries.

## QFT and Pauli evolution

```python
program = qp.Program(5)
program = qp.x(program, 0)
program = qp.append_qft(program, [0, 2, 4])
program = qp.append_qft(program, [0, 2, 4], inverse=True)

term = qp.PauliTerm(
    0.5,
    [qp.PauliFactor(0, qp.Pauli.X), qp.PauliFactor(2, qp.Pauli.Z)],
)
program = qp.append_pauli_evolution(program, term, 0.2)
```

`append_pauli_evolution` implements one exact Pauli exponential. Sequentially applying noncommuting Pauli exponentials is a caller-selected product formula and therefore has the corresponding approximation error; QuPy does not hide that distinction.

## Hardware circuits, OpenQASM, and providers

`Program` is the numerical unitary execution IR. `Circuit` is the hardware-facing IR for classical state and device-control semantics.

```python
circuit = qp.Circuit(2, num_clbits=1)
circuit = circuit.h(0).cx(0, 1)
circuit = circuit.measure(0, 0)
circuit = circuit.x(1, qp.ClassicalCondition(0, True))

text = circuit.to_openqasm3()
round_trip = qp.Circuit.from_openqasm3(text)
assert round_trip.fingerprint == circuit.fingerprint
```

Compilation is target-aware:

```python
target = qp.HardwareTarget(
    "example-qpu",
    3,
    [qp.CircuitOperationCode.H, qp.CircuitOperationCode.RZ],
    [qp.CircuitOperationCode.CZ],
    [qp.Coupling(0, 1), qp.Coupling(1, 2)],
    measurement=True,
    mid_circuit_measurement=True,
    reset=True,
    dynamic_control=True,
)

compiled = qp.compile(circuit, target, optimization_level=2)
```

Provider plug-ins keep remote credentials, queue policy, and vendor-specific behavior outside the numerical core.

## Open systems

Density matrices provide exact mixed-state evolution for supported local noise:

```python
channel = qp.amplitude_damping(0, 0.02)
noisy = qp.NoisyProgram(program, [qp.NoiseInstruction(0, channel)])
rho = qp.density_matrix(noisy)
```

For larger noisy systems where full density matrices are impractical, trajectory execution estimates observables by averaging independently sampled wave-function trajectories.

QuPy also exposes GKSL/Lindblad evolution for explicit density matrices.

## Distributed execution

MPI is optional at build time. When available, QuPy supports distributed state vectors and richer scale engines.

```python
info = qp.distributed_info()
print(info.available, info.world_size, info.rank)
```

Distributed tensor-network and trajectory APIs use collective failure propagation: a rank-local failure is surfaced collectively rather than leaving peers blocked in mismatched MPI collectives.

Multi-GPU CUDA execution is **not** implied by MPI support. The current distributed scale engines distribute work/rank ownership; physical GPU placement remains a separate execution concern.

## Testing and verification

The repository treats verification as part of the architecture, not as a release afterthought.

Standard CI covers:

- GCC and Clang warnings-as-errors native builds;
- Windows and macOS native builds;
- Python 3.12, 3.13, and 3.14;
- Ruff and strict mypy;
- ASan + UBSan;
- 2-rank and 4-rank MPI execution;
- wheel builds and native-runtime confirmation;
- benchmark-adapter compatibility;
- planner calibration/promotion validation.

A separate framework workflow installs pinned JAX and PyTorch CPU runtimes and executes the real autodiff adapter conformance suite.

Native-only development:

```text
cmake -S . -B build/native \
  -DQUPY_BUILD_PYTHON=OFF \
  -DQUPY_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/native
ctest --test-dir build/native --output-on-failure
```

Python development:

```text
uv sync
uv run pytest -q
uv run ruff check src tests benchmarks
uv run mypy src/qupy
```

## Documentation

The README is the overview. Detailed contracts live in focused documents:

- [Algorithm construction](docs/algorithms.md)
- [Distributed scale engines](docs/distributed_scale_engines.md)
- [JAX and PyTorch autodiff interoperability](docs/framework_autodiff.md)
- [Hardware compilation](docs/hardware_compilation.md)
- [OpenQASM 3 interchange](docs/openqasm3_interchange.md)
- [Provider execution](docs/provider_execution.md)
- [Quantum trajectories](docs/quantum_trajectories.md)
- [General tensor-network contraction](docs/tensor_network_contraction.md)
- [Research and implementation references](REFERENCES.md)

Benchmark and calibration tooling lives under [`benchmarks/`](benchmarks/). The repository's benchmark outputs are intended to justify specific planner decisions; QuPy does not use unvalidated headline benchmark numbers as automatic policy.

## Design rules

1. Keep quantum execution out of Python.
2. Keep the C++ core usable without Python.
3. Treat qubits as quantum resources, not NumPy arrays.
4. Keep semantic identity independent from execution strategy.
5. Make target capabilities and result semantics explicit.
6. Keep `backend="auto"` exact unless approximation is explicitly part of the requested method.
7. Fail closed on stale evidence, unsupported capabilities, and malformed interchange.
8. Add specialized engines only behind conformance tests against shared semantics.
9. Keep optional ecosystems optional; do not make JAX, PyTorch, MPI, CUDA, or vendor credentials mandatory for the core package.

## Current frontier

The largest remaining engineering work is not another isolated simulator backend. It is integration and scale:

- promote the validated general tensor-network CPU/TN calibration into a composable native planner schema without making CPU-only TN routing depend on unrelated CUDA/density evidence;
- expand contraction-path optimization only when it improves measured held-out routing quality;
- add multi-GPU / multi-node accelerator execution without conflating MPI rank distribution with GPU ownership;
- deepen first-party QPU provider adapters and lifecycle coverage while keeping credentials outside the core;
- broaden algorithm/application layers on top of the native execution and differentiation contracts;
- strengthen QEC with scalable production decoders beyond the bounded exact reference decoder;
- stabilize the public API, packaging, documentation, and compatibility policy toward a future 1.0.

The goal is a single quantum numerical-computing layer that can move from laptop simulation to accelerators, distributed execution, hybrid autodiff, and QPU submission without changing the program's meaning.
