# QuPy

QuPy is a native quantum numerical-computing library with a compact Python interface. C++20 implements the execution core, versioned program IR, target validation, planning, simulation, sampling, probabilities, expectations, and variances.

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
    |-- CUDA Driver API state-vector execution with embedded PTX
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
- exact bit-packed stabilizer sampling for large Clifford programs with polynomial state memory
- exact backward Pauli propagation for Clifford-compatible Pauli-Z expectation and variance cones
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

### CUDA state-vector execution

QuPy can execute an explicit `native-cuda` state-vector plan when a compatible NVIDIA CUDA driver is present. The core loads the CUDA Driver API at runtime and JIT-loads embedded PTX, so QuPy does not require CUDA Toolkit headers, `nvcc`, or NVRTC to build. CPU-only systems keep the same build and package.

```python
if qp.cuda_available():
    plan = qp.plan(program, qp.ResultMode.STATEVECTOR, backend="native-cuda")
    state = qp.statevector(program, backend="native-cuda")
    print(qp.cuda_device_name())
    print(plan.method)  # cuda-statevector
```

The CUDA target currently exposes exact state-vector results for the same H, X, Y, Z, RX, RY, RZ, CX, CZ, and SWAP program operations as the CPU state-vector path. Unsupported CUDA result modes fail during target validation; QuPy does not silently execute them on the CPU. Device memory is reused between calls, and the public `StateVector` remains native host-owned storage after one device-to-host transfer.

`backend="auto"` remains CPU-only when no planner cost model is supplied. A validated schema-v2 planner artifact can enable cost-based CPU/CUDA selection for `STATEVECTOR` when it matches both the native host and CUDA host fingerprints. Schema-v2 promotion requires validated CPU and CUDA return-cost curves plus at least eight held-out backend decisions with no decision exceeding 10% runtime regret. Other result modes keep the established planner policy. Sanitizer builds keep the CUDA driver disabled and exercise the fail-closed path so third-party driver allocations do not enter ASan/LSan accounting.

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

`ExecutionPlan` reports the structural MPS bond, routing, memory, and contraction-work estimates needed for measured planner calibration. `backend="auto"` does not select MPS yet. `python -m benchmarks.mps_cost` collects paired CPU/MPS exact expectation evidence across low-bond, routing-heavy, and locally entangled workload families for that promotion gate.

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

Validated benchmark calibration can be promoted into a compact native planner artifact. Schema-v1 artifacts bind CPU cost evidence to the QuPy core version, workload schema, and native host fingerprint. Schema-v2 artifacts additionally bind a CUDA host fingerprint and validated CPU/CUDA state-vector return-cost evidence. Invalid, stale, unvalidated, or wrong-host artifacts fail closed.

```python
model = qp.load_planner_cost_model("planner.qpcost")
plan = qp.expectation_plan(program, qp.Z(0), cost_model=model)

print(plan.predicted_ns)
print(plan.cost_model_class)
print(plan.cost_model_fingerprint)
```

Schema-v1 models remain evidence-only: they do not change `backend`, `method`, or `cache_key`. A schema-v2 CUDA model can affect only `ResultMode.STATEVECTOR` with `backend="auto"` when that model is explicitly supplied. The planner compares native CPU and CUDA return-cost predictions using qubit count, compiled-step count, two-qubit operation fraction, and planned CPU thread count, then selects the lower predicted runtime. The chosen plan retains its normal cache identity; cost-model fingerprints remain separate provenance. Execution without a supplied model is unchanged.

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

Pauli propagation and stabilizer sampling are specialized exact methods selected by the native planner. The CUDA state-vector target uses the CUDA Driver API and PTX JIT without a toolkit build dependency. The explicit MPS target provides exact tensor-network expectation, variance, and state-vector execution with inspectable bond, routing, and contraction-cost features. Workload fingerprint version 1, held-out calibration, and validated planner artifacts provide host-scoped predicted-runtime evidence with explicit provenance. Schema-v2 evidence can select CPU or CUDA for exact state-vector return when explicitly supplied; automatic MPS selection, noise, richer observables, and physical-QPU targets remain gated on their own conformance and planner evidence.

See [REFERENCES.md](REFERENCES.md) for the specifications, libraries, and upstream systems that materially shape the implementation.

## License

MIT
