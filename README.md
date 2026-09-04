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

QuPy requires Python 3.12+ and a C++20 compiler. NumPy is the only required Python runtime dependency. JAX, PyTorch, Amazon Braket, Qiskit, and Qiskit Aer integration are optional and loaded lazily.

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
- exact general tensor-network expectation execution and explicit parameter-shift differentiation through `native-tn`, with evidence-gated automatic CPU/TN expectation selection;
- deterministic tensor-network structural preflight with contraction count, peak rank, peak intermediate bytes, scalar work, and plan fingerprints;
- exact MPI-sharded host state-vector execution plus explicit rank-local CUDA shard execution with host-staged nonlocal-gate exchange, alongside distributed Pauli/Hamiltonian reductions;
- MPI term-parallel general tensor-network expectation execution;
- MPI-distributed noisy quantum trajectories with collective failure propagation.

### GPU and open-system execution

- toolkit-free CUDA Driver API loading with embedded PTX JIT;
- explicit CUDA device inventory and per-device runtime/context ownership, with `native-cuda:<device>` selecting a visible nonzero ordinal;
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
- OpenQASM 3.1 serialization **and import** for the supported hardware-capable subset, plus a provider-facing OpenQASM 3.0 transport profile;
- QIR Base Profile export for numerical programs;
- structural provider lifecycle for native C-ABI plug-ins and first-party Python adapters;
- provider capability discovery, target discovery, compiled submission, polling, result retrieval, cancellation, and explicit conformance testing;
- first-party Amazon Braket adapter with credential-free `LocalSimulator` interoperability and caller-owned AWS credentials for cloud devices;
- explicit separation between numerical execution identity and hardware-control semantics.

### Algorithms and QEC

- native `VariationalTemplate` with deterministic parameter names and tracked native slots;
- hardware-efficient variational ansätze with configurable RX/RY/RZ layers and linear/ring/none entanglement;
- composable exact QFT / inverse-QFT synthesis over arbitrary selected qubits;
- exact single-Pauli-string time evolution plus explicit first- and second-order Pauli-Hamiltonian product formulas;
- dependency-free Jordan-Wigner mapping, spin-orbital molecular Hamiltonians, Hartree-Fock references, and factorized UCCSD templates with native-gradient chain-rule mapping;
- canonical weighted MaxCut Hamiltonians and standard X-mixer QAOA construction;
- detector error models, deterministic syndrome sampling, repetition-code construction, a bounded exact reference maximum-likelihood decoder, and native sparse BP+OSD-0 decoding with reusable batch execution;
- reproducible surface-code evidence against PyMatching sparse-blossom and hypergraph-product QLDPC evidence against independent `ldpc` BP+OSD-0 comparators.

## Architecture

```text
Python API
    |
    +-- algorithms / chemistry / optional JAX + PyTorch adapters
    +-- provider adapters / optional Amazon Braket + Qiskit SDKs
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
    |-- detector-model QEC decoding
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
| `native-cuda` / `native-cuda:<device>` | CUDA state vectors, Pauli reduction, density/noise on an explicit visible device | Exact; requires compatible NVIDIA driver; device zero canonicalizes to `native-cuda` |
| stabilizer | Large Clifford sampling | Exact; planner-selected for eligible workloads |
| `native-mps` / adaptive MPS | Low-entanglement exact execution | No user-visible bond truncation |
| `native-tn` | General tensor-network rich-observable expectations and explicit expectation derivatives | Exact greedy contraction; auto expectation routing only with promoted in-domain CPU/TN evidence |
| `native-mpi` | Distributed state vector and rich observables | Exact MPI sharding/reduction |
| distributed TN | Pauli-term-parallel tensor-network expectation | Exact MPI reduction |
| trajectories | Noisy wave-function Monte Carlo | Statistical estimator; seedable |
| density matrix | Exact open-system matrix evolution | CPU/CUDA; exponential state size |

QuPy does not silently reinterpret an explicit backend. Unsupported result modes or capabilities fail instead of falling back behind the caller's back. `cuda_device_count()` reports CUDA-driver-visible devices, while `cuda_available(device)` reports whether QuPy can initialize that ordinal. `cuda`, `native-cuda`, and ordinal zero all canonicalize to `native-cuda`; `cuda:1` and `native-cuda:1` canonicalize to `native-cuda:1`, and similarly for higher ordinals. Each ordinal owns an independent lazily initialized CUDA primary context and QuPy workspace.

Automatic CUDA planning remains scoped to device zero because promoted planner artifacts are bound to that measured host/device fingerprint. Explicit nonzero CUDA devices execute without borrowing device-zero timing evidence.

## Evidence-gated automatic planning

`backend="auto"` is owned by evidence-gated planning. QuPy can load validated host-scoped cost artifacts and use them only for policies whose required evidence is present and valid.

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

General tensor-network CPU-vs-TN routing uses a separate native policy artifact rather than extending cumulative schemas v1-v5. It is produced from the repeated, counterbalanced, leave-one-workload-out calibration, bound to the exact QuPy core and planner host, and records the measured domain of every CPU/TN runtime-model feature. Automatic TN prediction is interpolation-only: workloads outside the promoted feature domain keep the established plan instead of extrapolating.

The TN policy composes conservatively with existing automatic planning. It may replace an ordinary full-cone dense-CPU rich-observable expectation when validated evidence predicts TN is faster, but it does not displace Pauli propagation or an existing validated CUDA/MPS/rich-observable adaptive decision. The current evidence compares CPU with TN; QuPy does not invent an unmeasured TN-vs-CUDA or TN-vs-MPS ordering.

```python
model = qp.install_tensor_network_cost_model("tn-policy.qptncost")
print(model.auto_validated)
print(qp.tensor_network_planner_cache_path())
```

See [General tensor-network contraction](docs/tensor_network_contraction.md) for collection, promotion, discovery precedence, interpolation gates, and exact automatic-routing boundaries.

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

## QFT and Hamiltonian evolution

```python
program = qp.Program(5)
program = qp.x(program, 0)
program = qp.append_qft(program, [0, 2, 4])
program = qp.append_qft(program, [0, 2, 4], inverse=True)

term = qp.PauliTerm(
    0.5,
    [qp.PauliFactor(0, qp.Pauli.X), qp.PauliFactor(2, qp.Pauli.Z)],
)
hamiltonian = qp.Observable([term])
program = qp.append_pauli_evolution(program, term, 0.2)
program = qp.append_hamiltonian_evolution(program, hamiltonian, 0.2, steps=4, order=2)
```

`append_pauli_evolution` implements one exact Pauli exponential. `append_hamiltonian_evolution` makes the product-formula approximation explicit through its order and positive step count. Commuting Pauli terms are exact up to floating-point roundoff and global phase; noncommuting sums retain the corresponding product-formula error.

## Electronic-structure chemistry

QuPy can translate second-quantized spin-orbital operators into its native Pauli-sum `Observable` model without adding a chemistry runtime dependency.

```python
import numpy as np
import qupy as qp

one_body = np.diag([-1.0, -0.4])
hamiltonian = qp.molecular_hamiltonian(one_body, nuclear_repulsion=0.7)
reference = qp.hartree_fock_state(2, 1)
energy = qp.expect(reference, hamiltonian)
```

`jordan_wigner()` accepts explicit ordered fermionic ladder terms and requires the mapped Pauli coefficients to be real within the requested tolerance. `molecular_hamiltonian()` uses the spin-orbital convention `sum h[p,q] a†_p a_q + 0.5 sum g[p,q,r,s] a†_p a†_q a_s a_r + E_nuc`. The returned observable uses the same CPU, CUDA, MPS, tensor-network, MPI, and differentiation paths as any other QuPy Hamiltonian. `hartree_fock_state()` prepares a computational-basis occupation state and supports explicit occupied-orbital selections.

Factorized spin-orbital UCCSD is built from the same primitives:

```python
template = qp.uccsd_ansatz(4, 2)
theta = np.zeros(template.parameter_count)
program = template.bind(theta)
energy = qp.expect(program, hamiltonian)
```

`uccsd_ansatz()` enumerates occupied-to-virtual singles followed by doubles in deterministic order. Each individual fermionic excitation is mapped as the Hermitian generator `i(A - A†)` and exponentiated exactly because its Jordan-Wigner Pauli terms commute; the ordered product across different excitations is the explicit first-order factorized UCCSD ansatz. `UccsdTemplate` also exposes the synthesized gate slots and the linear amplitude-to-gate map, so native gate-level gradients can be converted back to UCCSD amplitude gradients with `compress_gradient()` without adding a second symbolic parameter system.

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

Native provider plug-ins and first-party Python providers implement the same `ProviderBackend` lifecycle. Vendor credentials and vendor SDKs remain outside the numerical core.

Amazon Braket can be exercised without AWS credentials through its real local simulator:

```python
provider = qp.BraketProvider.local_simulator(num_qubits=4)
submission = qp.submit_circuit(
    provider,
    qp.Circuit(1, 1).h(0).measure(0, 0),
    100,
    initial_layout=[0],
)
print(provider.result_json(submission.job_id))
```

See [Provider execution](docs/provider_execution.md) for native ABI conformance, Amazon Braket cloud/local boundaries, OpenQASM provider transport, and credential ownership.

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

`distributed_cuda_statevector()` maps each MPI rank to an explicit CUDA ordinal and keeps that rank's shard in the selected device workspace for local-qubit gates. With `device=None`, the launcher-reported local rank selects the CUDA ordinal; callers may pass an explicit ordinal instead. Gates touching distributed qubits use the same exact MPI shard semantics as the host engine, with the shard staged through host memory for the exchange/update and then returned to the GPU. This is real rank-local GPU state ownership, but it is not CUDA-aware MPI or direct GPU-to-GPU transport.

## Quantum error correction

Detector models describe independent error mechanisms, detector flips, and logical-frame changes. The exact decoder remains the small-model maximum-likelihood reference; `BpOsdDecoder` is the reusable sparse path for larger detector models.

```python
model = qp.repetition_code_detector_model(5, 4, 0.01, 0.02)
samples = qp.sample_detector_model(model, shots=4096, seed=7)

decoder = qp.BpOsdDecoder(model, max_iterations=50, damping=0.1)
decoded = decoder.decode_batch(samples.syndrome)

print(decoded.observables)
```

BP+OSD-0 always verifies the returned correction against the requested syndrome when decoding succeeds. It is not a maximum-likelihood guarantee. The existing exact reference decoder is capped at 24 error mechanisms; the scalable path uses sparse belief propagation and deterministic GF(2) order-0 repair instead of subset enumeration.

The QEC evidence harnesses compare QuPy against independent decoders on shared workloads: PyMatching sparse-blossom on Stim-generated rotated surface-code detector samples, and `ldpc` 2.4.1 BP+OSD-0 on deterministic Hamming-seed hypergraph-product code-capacity samples. Both record logical-failure confidence intervals and raw decode timings without turning hosted-runner timing into a pass/fail claim.

See [Detector-model decoding](docs/qec_decoding.md) for algorithm, result, batch, benchmark, and failure contracts.

## Testing and verification

The repository treats verification as part of the architecture, not as a release afterthought.

Standard CI covers:

- GCC and Clang warnings-as-errors native builds;
- Windows and macOS native builds;
- Python 3.12, 3.13, and 3.14;
- Ruff and strict mypy;
- ASan + UBSan;
- 2-rank and 4-rank MPI execution;
- isolated wheel and sdist-built-wheel runtime verification;
- benchmark-adapter compatibility;
- planner calibration/promotion validation.

A separate framework workflow installs pinned JAX and PyTorch CPU runtimes and executes the real autodiff adapter conformance suite. A provider-interoperability workflow installs the pinned Amazon Braket SDK and runs QuPy submission, lifecycle, result normalization, and generic provider conformance against Braket `LocalSimulator`.

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
- [Pre-1.0 compatibility review](docs/compatibility_review_v0.md)
- [Detector-model decoding](docs/qec_decoding.md)
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
9. Keep optional ecosystems optional; do not make JAX, PyTorch, MPI, CUDA, vendor SDKs, or vendor credentials mandatory for the core package.

## Current frontier

The largest remaining engineering work is integration and scale:

- broaden tensor-network policy only with direct held-out evidence for TN-vs-CUDA/MPS decisions, and improve contraction paths only when measured routing quality improves;
- replace host-staged distributed-CUDA exchanges with measured CUDA-aware MPI or equivalent direct GPU transport where hardware evidence shows a benefit, and benchmark multi-GPU / multi-node scaling explicitly;
- extend first-party provider coverage beyond Amazon Braket and Qiskit, and deepen hardware-capability translation only where vendor evidence supports semantics that the portable target can represent safely;
- broaden chemistry beyond the dependency-free Jordan-Wigner/Hartree-Fock/factorized-UCCSD layer with evidence-backed active-space, excitation-selection, and molecular-workflow integrations, while reusing the native execution and differentiation contracts;
- broaden QEC evidence beyond Hamming-seed hypergraph products to circuit-level noise and additional LDPC families, and add higher-order or specialized decoders only when measured logical-error and latency gains justify them;
- keep the reviewed pre-1.0 compatibility baseline current while the remaining scale and provider evidence matures toward a future 1.0 commitment.

The goal is a single quantum numerical-computing layer that can move from laptop simulation to accelerators, distributed execution, hybrid autodiff, and QPU submission without changing the program's meaning.
