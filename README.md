# QuPy

QuPy is a native quantum numerical-computing library with a compact Python interface. C++20 implements the execution core, versioned program IR, target validation, planning, simulation, sampling, rich observables, differentiation, mixed-state dynamics, distributed execution, QEC primitives, and provider interchange.

Python is the user-facing language. It does not implement the quantum simulator. NumPy provides an interoperable array surface for native results.

> **Status:** early alpha. The native CPU core is usable and tested. The public API can still change.

## Quick start

```python
import qupy as qp

program = qp.Program(2)
program = qp.h(program, 0)
program = qp.cx(program, 0, 1)

samples = qp.sample(program, shots=1000, seed=7)
probabilities = qp.probabilities(program)
energy = qp.expect(program, qp.Z(0))
variance = qp.variance(program, qp.Z(0))
state = qp.statevector(program)

print(qp.core_language())     # C++20
print(qp.core_version())      # 0.3.0a0
print(samples.counts())
print(probabilities.values)
print(energy.value, variance.value)
print(state.values)
```

Native C++ objects back all program objects and execution results in this example.

## Native architecture

```text
Python API
    |
    v
nanobind extension
    |
    v
C++20 QuPy core
    |-- versioned immutable Program / Operation IR
    |-- deterministic program and target fingerprints
    |-- target capability validation
    |-- result-aware execution planner and cache identity
    |-- exact Pauli propagation for eligible observables
    |-- bit-packed stabilizer sampling for large Clifford programs
    |-- CUDA Driver API state-vector and Pauli-reduction execution with embedded PTX
    |-- gate kernels and state-vector runtime
    |-- sampling and expectation evaluation
    `-- typed result storage
         |
         `-- zero-copy NumPy views
```

The core library is independent of Python bindings. CTest validates the C++ implementation directly. The Python tests validate the bound API separately.

## Current native capabilities

- H, X, Y, Z, RX, RY, and RZ single-qubit gates
- CX, CZ, and SWAP two-qubit gates
- versioned immutable native program IR with deterministic canonical text
- SHA-256 program and target fingerprints for execution identity
- explicit target capabilities and result-aware execution plans with versioned structural workload fingerprints
- versioned cache keys that include program, target, result, method, and query identity
- exact dense state-vector simulation on CPU and explicit CUDA targets
- GPU-resident exact Pauli-string, Hamiltonian, expectation, variance, covariance, and observable-batch reduction
- exact adaptive MPS/dense observable execution behind validated host-scoped planner evidence
- exact bit-packed stabilizer sampling for large Clifford programs with polynomial state memory
- exact backward Pauli propagation for Clifford-compatible Pauli-Z expectation and variance cones
- arbitrary real Pauli-sum observables, Hamiltonians, expectation, variance, symmetrized covariance, and multi-observable batches
- commuting and qubit-wise measurement grouping with deterministic shot-based observable estimation
- exact polynomial Pauli propagation for arbitrary Pauli strings and Pauli sums on Clifford circuits
- query-aware observable execution plans with cache identity and planner-cost provenance
- native value/gradient, Jacobian, and Hessian evaluation with adjoint, parameter-shift, and finite-difference methods
- semantics-preserving circuit optimization with cancellation, rotation merging, and disjoint-gate commutation
- exact density-matrix simulation with built-in noise channels and validated custom single-qubit Kraus channels
- Runge-Kutta integration of the GKSL/Lindblad master equation
- optional MPI distributed state-vector execution and exact distributed Pauli/Hamiltonian reductions with fail-closed capability detection
- OpenQASM 3.1 and QIR Base Profile interchange plus a stable provider plug-in C ABI
- detector error models, deterministic syndrome sampling, and a reference maximum-likelihood decoder
- probabilities, Pauli-Z expectations, and Pauli-Z variances
- platform-independent deterministic seeded sampling
- read-only zero-copy NumPy views over native state, probability, and sample storage
- fused single-qubit native kernels
- compact branch-free CX, CZ, and SWAP pair traversal
- alias-table sampling for repeated shots
- exact reverse-causal-cone reduction before observable-method selection
- workload-scaled OpenMP teams for sufficiently large amplitude workloads, currently capped at the verified 16-thread kernel scaling limit
- reusable per-thread state workspaces for internal result execution
- immutable parameter binding with explicit native `ParameterSlot` objects
- vectorized exact expectation and sampling batches with reusable compiled templates
- toolkit-free CUDA Driver API loading with PTX JIT when a compatible NVIDIA driver is present
- compiler-optimized release builds
- validated host/version-scoped planner cost artifacts with native predicted-runtime introspection

## Parameter binding and batches

A parameter slot identifies an existing operation parameter without changing IR version 1. Binding returns a new immutable `Program`; it does not mutate the template.

```python
import math
import numpy as np
import qupy as qp

template = qp.ry(qp.Program(1), 0.0, 0)
slots = [qp.ParameterSlot(0)]
angles = np.array([[0.0], [math.pi / 2.0], [math.pi]], dtype=np.float64)

expectations = qp.expect_batch(template, qp.Z(0), slots, angles)
samples = qp.sample_batch(template, slots, angles, shots=128, seed=7)

print(expectations.values)  # [ 1.  0. -1.]
print(samples.values.shape) # (3, 128, 1)
```

The parameter table has shape `(batch_size, len(slots))`. Native expectation batches reuse causal-cone analysis, fusion structure, fixed matrices, and the internal state workspace across rows. Sampling batches reuse the compiled template and state workspace across rows. Result arrays are read-only NumPy views over native storage.

A seeded sampling batch uses one deterministic QuPy random stream in row-major batch order. A one-row batch therefore produces the same sample array as scalar `sample()` on the corresponding bound program with the same seed.

The native target advertises `parameter_batches=True`. Current parameterized gates have one parameter per operation, so `ParameterSlot(operation_index)` is sufficient for RX, RY, and RZ. The optional `parameter_index` remains explicit for forward-compatible slot identity.

## Execution model

C++ handles `backend="auto"`. The planner validates the requested result against a target before execution and returns an inspectable `ExecutionPlan`.

```python
plan = qp.plan(program, qp.ResultMode.SAMPLE)
print(plan.backend)              # native-cpu
print(plan.method)               # statevector
print(plan.exact)                # True
print(plan.program_fingerprint)  # SHA-256
print(plan.target_fingerprint)   # SHA-256
print(plan.cache_key)
```

Automatic execution preserves declared semantics. Execution strategies can change without moving backend policy into Python.

### CUDA state-vector and Pauli reduction

QuPy can execute explicit `native-cuda` state-vector, Pauli-Z expectation/variance, and arbitrary Pauli-sum observable plans when a compatible NVIDIA CUDA driver is present. The core loads the CUDA Driver API at runtime and JIT-loads embedded PTX, so QuPy does not require CUDA Toolkit headers, `nvcc`, or NVRTC to build. CPU-only systems keep the same build and package.

```python
if qp.cuda_available():
    plan = qp.plan(program, qp.ResultMode.STATEVECTOR, backend="native-cuda")
    state = qp.statevector(program, backend="native-cuda")
    print(qp.cuda_device_name())
    print(plan.method)  # cuda-statevector
```

The CUDA target supports the same H, X, Y, Z, RX, RY, RZ, CX, CZ, and SWAP operations as the CPU state-vector path. `STATEVECTOR`, Pauli-Z `EXPECTATION`, and Pauli-Z `VARIANCE` are target capabilities. Rich `Observable` queries use the same compiled CUDA gate sequence and then evaluate unique Pauli strings directly against the resident device state. Each Pauli kernel reduces complex contributions in shared memory, recursive device reductions collapse block partials, and the host receives one complex value per unique Pauli string rather than the full state vector. Hamiltonian, variance, covariance, and multi-observable queries deduplicate identical Pauli strings across the complete request.

Unsupported CUDA result modes still fail during target validation; QuPy does not silently execute them on the CPU. Device state memory is reused between calls. A public `StateVector` still performs one complete device-to-host transfer because returning amplitudes is the requested result, while `cuda-pauli-reduction` keeps observable reduction on the accelerator.

`backend="auto"` remains CPU-only when no planner cost model is supplied. A validated schema-v2 planner artifact can enable cost-based CPU/CUDA selection for `STATEVECTOR` when it matches both the native host and CUDA host fingerprints. Rich non-Clifford observable queries may reuse that validated backend decision as a conservative execution gate, but `ObservableExecutionPlan.predicted_ns` remains unset for `cuda-pauli-reduction` until a dedicated observable calibration model exists. Sanitizer builds keep the CUDA driver disabled and exercise the fail-closed path so third-party driver allocations do not enter ASan/LSan accounting.

### Exact MPS execution

QuPy exposes an explicit `native-mps` target for exact Pauli-Z expectation, variance, and state-vector results. It reuses the native compiled gate sequence, represents the state as a one-dimensional matrix product state, routes nonadjacent two-qubit gates with reversible SWAPs, and refactors each two-site update with an SVD.

The engine does not apply a user-visible bond cap or truncation tolerance. It removes only numerically null singular directions within a backward-error bound and fails instead of discarding larger state weight. A full state-vector request still requires exponential output storage; expectation and variance can remain compact when the circuit has limited entanglement.

```python
program = qp.ry(qp.Program(128), 0.371, 0)
for qubit in range(127):
    program = qp.cx(program, qubit, qubit + 1)

plan = qp.expectation_plan(program, qp.Z(127), backend="native-mps")
result = qp.expect(program, qp.Z(127), backend="native-mps")
print(plan.tensor_network_max_bond)          # 2
print(plan.tensor_network_routed_swaps)     # 0
print(plan.tensor_network_contraction_work)
```

`ExecutionPlan` reports structural MPS bond, routing, memory, and contraction-work estimates for measured planner validation. Explicit `native-mps` execution remains available without a planner artifact.

A validated schema-v3 planner artifact can enable exact MPS-aware selection for expectation and variance requests with `backend="auto"`. The current policy uses direct MPS when the structural estimate proves the bond remains small, selects dense state-vector execution for deterministic routing-heavy cases, and otherwise uses an adaptive exact path. The adaptive path can checkpoint an MPS calculation and continue from the exact checkpoint as a dense state vector when bond growth makes dense execution preferable. It never truncates retained singular values to satisfy the policy.

Schema-v3 promotion is evidence-gated. `benchmarks.mps_cost --profile policy` records two counterbalanced timing pairs for each workload: CPU versus adaptive execution and MPS versus adaptive execution. `benchmarks.mps_calibrate` recomputes regret from the raw pairs, requires at least three matching reports and 16 distinct workloads, rejects numerical disagreement above `2e-11`, and permits no workload with more than 10% median runtime regret. The resulting artifact remains bound to the QuPy core, workload schema, and native host fingerprint.

### Stabilizer sampling

For sampling-only Clifford programs at 24 qubits or larger, the native planner selects `stabilizer`. The engine conjugates a bit-packed stabilizer tableau through H, X, Y, Z, CX, CZ, and SWAP gates, reduces its computational-basis support to an affine GF(2) subspace, and samples that support without allocating a dense state vector. RX, RY, and RZ currently remain on the state-vector path even when a particular angle could represent a Clifford operation.

```python
program = qp.Program(24)
program = qp.h(program, 0)
for qubit in range(1, 24):
    program = qp.cx(program, qubit - 1, qubit)

plan = qp.plan(program, qp.ResultMode.SAMPLE)
print(plan.method)                 # stabilizer
print(plan.estimated_state_bytes)  # 408
```

The 24-qubit threshold keeps the established small-circuit seeded state-vector sampling contract unchanged. Stabilizer sampling has its own pinned cross-platform seed vector. Zero-parameter `sample_batch` calls reuse one stabilizer support across rows. Current planner cost artifacts predate the stabilizer class, so a valid state-vector/Pauli cost model may accompany a stabilizer plan but does not claim a stabilizer prediction.

### Validated planner cost evidence

Validated benchmark calibration can be promoted into a compact native planner artifact. Schema-v1 artifacts bind CPU cost evidence to the QuPy core version, workload schema, and native host fingerprint. Schema-v2 artifacts additionally bind a CUDA host fingerprint and validated CPU/CUDA state-vector return-cost evidence. Schema-v3 artifacts can preserve that CUDA evidence and add the validated adaptive-MPS observable policy. Invalid, stale, unvalidated, incomplete, or wrong-host artifacts fail closed.

```python
model = qp.load_planner_cost_model("planner.qpcost")
plan = qp.expectation_plan(program, qp.Z(0), cost_model=model)

print(plan.predicted_ns)
print(plan.cost_model_class)
print(plan.cost_model_fingerprint)
```

Schema-v1 models remain evidence-only: they do not change `backend`, `method`, or `cache_key`. A schema-v2 or schema-v3 artifact with validated CUDA evidence can affect only `ResultMode.STATEVECTOR` with `backend="auto"`. A schema-v3 artifact with validated MPS policy evidence can additionally select `native-adaptive-mps` for eligible expectation and variance requests. Pauli-propagation-eligible observables keep their specialized CPU method. The chosen plan retains its normal cache identity; cost-model fingerprints remain separate provenance. Execution without a supplied model is unchanged.

### Result-aware expectation planning

Expectation and variance planning first computes the exact reverse causal cone of the requested observable. The planner then selects an exact method for that reduced problem.

For a Clifford-compatible cone, QuPy propagates the Pauli observable backward to the initial computational-basis state. It does not allocate a state vector.

```python
program = qp.Program(100)
program = qp.h(program, 0)
program = qp.x(program, 99)

plan = qp.expectation_plan(program, qp.Z(0))
print(plan.method)                 # pauli-propagation
print(plan.active_qubits)          # 1
print(plan.estimated_state_bytes)  # 0
```

The gate on qubit 99 is outside the causal cone and is removed before method selection. A non-Clifford RX, RY, or RZ gate outside the cone is also irrelevant. If a retained gate is non-Clifford, QuPy falls back to the exact reduced state-vector method instead of approximating the result.

Entangling gates expand the causal cone when they can affect the observable. The reduction and Pauli method are exact and do not use truncation, sampling, or approximation. Native conformance includes a 4,096-qubit entangled Clifford cone that is evaluated with zero state-vector bytes.

## Quantum numerical computing

QuPy treats an observable query as a numerical result request rather than as a simulator-specific operation. `Observable` represents a real weighted sum of Pauli strings. Expectation, variance, covariance, multi-observable batches, commuting groups, qubit-wise measurement groups, and shot-based estimates use this shared representation.

```python
hamiltonian = qp.Observable([
    qp.PauliTerm(0.5, [qp.PauliFactor(0, qp.Pauli.X), qp.PauliFactor(1, qp.Pauli.X)]),
    qp.PauliTerm(0.25, [qp.PauliFactor(0, qp.Pauli.Z), qp.PauliFactor(1, qp.Pauli.Z)]),
])
energy = qp.expect(program, hamiltonian)
plan = qp.observable_plan(program, [hamiltonian])
```

For Clifford circuits, arbitrary Pauli strings and Pauli sums can use exact backward Pauli propagation with zero dense-state storage. Other observable requests use the reduced causal cone and the established state-vector planner. Explicit `native-cuda` execution keeps the reduced state and Pauli reductions on the GPU. A validated CPU/CUDA cost artifact can route a non-Clifford observable through the same device-reduction path without moving policy into Python.

### Differentiation and circuit optimization

`value_and_grad`, `grad`, `jacobian`, and `hessian` operate on native parameter slots. Automatic first derivatives select the native adjoint method when its state workspace fits the configured exact-memory bound and otherwise use parameter shift. Parameter shift and finite difference remain explicit choices. Hessians use exact parameter-shift identities for the supported RX, RY, and RZ parameterization.

```python
result = qp.value_and_grad(template, hamiltonian, slots, parameters)
jac = qp.jacobian(template, observables, slots, parameters)
hess = qp.hessian(template, hamiltonian, slots, parameters)
optimized = qp.optimize(template, level=2)
```

The optimizer preserves program semantics. Level 1 performs local inverse cancellation and same-axis rotation merging. Level 2 can commute operations on disjoint qubits to expose additional reductions.

### Mixed-state and open-system execution

`density_matrix` executes pure or noisy programs exactly in density-matrix form. Built-in channels include bit flip, phase flip, depolarizing, amplitude damping, phase damping, and general Pauli noise. `kraus_channel` accepts validated single-qubit operator-sum channels and rejects operators that do not satisfy the trace-preserving completeness relation. `lindblad_evolve` integrates the GKSL/Lindblad master equation with fourth-order Runge-Kutta steps.

### Distributed and accelerator execution

The CUDA target provides explicit native state-vector and GPU-resident arbitrary-Pauli reduction. Rich non-Clifford observable requests can execute their reduced state on CUDA and return only reduced observable values. Multi-GPU CUDA execution remains outside the current target; the distributed state-vector implementation is MPI-based.

When QuPy is built with MPI C++ support, `distributed_statevector` shards amplitudes across a power-of-two communicator and implements local and cross-rank H/X/Y/Z/RX/RY/RZ/CX/CZ/SWAP execution. Rich observables accept `backend="native-mpi"` for exact expectation, variance, covariance, and multi-observable reduction. Pauli terms reuse peer-shard exchanges by rank-flip pattern and combine only compact complex reductions with `MPI_Allreduce`; QuPy does not gather the complete distributed state to evaluate these observables. Parameter-shift differentiation can use the same backend. Automatic MPI routing remains disabled until host/topology calibration evidence is available. Builds without MPI expose the capability state and fail closed rather than silently substituting local execution.

### QPU interchange and provider ABI

`to_openqasm3` emits OpenQASM 3.1 source and `to_qir_base_profile` emits LLVM IR that targets the QIR Base Profile. A versioned C provider ABI supports capability discovery, submit, poll, result, cancel, and teardown operations. Vendor adapters remain outside the QuPy core so remote credentials and service policy do not enter the numerical runtime.

### Error-correction primitives

`DetectorModel` represents independent detector-error mechanisms and logical frame changes. QuPy can construct repetition-code models, sample deterministic syndrome streams, and run an exact reference maximum-likelihood decoder. The reference decoder is intentionally bounded to 24 mechanisms; production large-code decoding requires a specialized decoder implementation.
## Build and test

QuPy requires Python 3.12 or newer and a C++20 compiler. Python packaging uses scikit-build-core and nanobind.

```text
uv sync
uv run pytest -q
uv run ruff check src tests
uv run mypy src/qupy
```

The native core can also be built and tested without Python:

```text
cmake -S . -B build/native -DQUPY_BUILD_PYTHON=OFF -DQUPY_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/native
ctest --test-dir build/native --output-on-failure
```

If CMake finds an MPI C++ implementation, the native test build also creates `qupy_mpi_2` and `qupy_mpi_4`. These tests run the same exact observable reductions with two and four ranks and compare them with the native CPU results.

The distribution name is `qupy-compute`. The import package is `qupy`.

## Design rules

1. Keep quantum execution out of Python.
2. Keep the C++ core usable without Python.
3. Treat qubits as quantum resources, not NumPy arrays.
4. Make target capabilities and result semantics explicit.
5. Keep `backend="auto"` exact unless the caller explicitly permits approximation.
6. Use standard array protocols at the language boundary rather than inventing a tensor ecosystem.
7. Add specialized engines only behind conformance tests against shared semantics.

## Foundation contracts

QuPy separates semantic identity from execution strategy. IR version 1 has deterministic text and SHA-256 fingerprints. Targets publish explicit capabilities and independent fingerprints. Execution plans bind the program, target, requested result, method, and query-specific data into a versioned cache key.

Seeded sampling uses QuPy-defined random-number mapping rather than implementation-defined standard-library distributions. The conformance suite pins the resulting sequence so supported operating systems must agree for the same program and seed.

These contracts are the stable integration boundary for additional execution engines. A new engine can change performance or physical execution without changing the meaning of a QuPy program or silently changing an exact result request.

## Direction

Pauli propagation and stabilizer sampling are specialized exact methods selected by the native planner. The CUDA state-vector target uses the CUDA Driver API and PTX JIT without a toolkit build dependency. The MPS target provides exact tensor-network execution with inspectable bond, routing, and contraction-cost features, and schema-v3 evidence can enable its exact adaptive observable policy when explicitly supplied. Workload fingerprint version 1, held-out calibration, and validated planner artifacts provide host-scoped execution evidence with explicit provenance. Rich observables, gradients, mixed-state dynamics, provider interchange, detector models, and optional distributed execution share the same native-first semantics. Automatic routing remains evidence-gated: QuPy does not claim an execution path until the target is executable and its required planner evidence is valid.

See [REFERENCES.md](REFERENCES.md) for the specifications, libraries, and upstream systems that materially shape the implementation.

## License

MIT
