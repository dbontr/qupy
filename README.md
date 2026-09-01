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
- explicit target capabilities and result-aware execution plans
- versioned cache keys that include program, target, result, method, and query identity
- exact dense state-vector simulation
- exact backward Pauli propagation for Clifford-compatible Pauli-Z expectation and variance cones
- probabilities, Pauli-Z expectations, and Pauli-Z variances
- platform-independent deterministic seeded sampling
- read-only zero-copy NumPy views over native state, probability, and sample storage
- fused single-qubit native kernels
- compact branch-free CX, CZ, and SWAP pair traversal
- alias-table sampling for repeated shots
- exact reverse-causal-cone reduction before observable-method selection
- workload-scaled OpenMP teams for sufficiently large amplitude workloads
- compiler-optimized release builds

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

Pauli propagation is the first specialized exact method selected by the native planner. The next performance work is reproducible workload-class benchmarking, measured planner cost models, SIMD/cache tuning, parameter batching, and broader stabilizer execution. CUDA/cuQuantum, tensor-network, distributed, and physical-QPU engines remain additional planner targets behind the same conformance contracts.

See [REFERENCES.md](REFERENCES.md) for the specifications, libraries, and upstream systems that materially shape the implementation.

## License

MIT
