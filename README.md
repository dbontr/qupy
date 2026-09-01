# QuPy

QuPy is a native quantum numerical-computing library with a compact Python interface. C++20 implements the execution core, program IR, target validation, planner, simulator, sampling, and expectation evaluation.

Python is the user-facing language. It does not implement the quantum simulator. NumPy provides an interoperable array surface for native results.

> **Status:** early alpha. The native CPU core is usable and tested. The public API can still change.

## Quick start

```python
import qupy as qp

program = qp.Program(2)
program = qp.h(program, 0)
program = qp.cx(program, 0, 1)

samples = qp.sample(program, shots=1000, seed=7)
energy = qp.expect(program, qp.Z(0))
state = qp.statevector(program)

print(qp.core_language())     # C++20
print(samples.counts())
print(energy.value)
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
    |-- immutable Program / Operation IR
    |-- target capability validation
    |-- execution planner
    |-- gate kernels
    |-- state-vector runtime
    |-- sampling and expectation evaluation
    `-- typed result storage
         |
         `-- zero-copy NumPy views
```

The core library is independent of Python bindings. CTest validates the C++ implementation directly. The Python tests validate the bound API separately.

## Current native capabilities

- H, X, Y, Z, RX, RY, and RZ single-qubit gates
- CX, CZ, and SWAP two-qubit gates
- immutable native program IR
- explicit native target and execution plans
- exact dense state-vector simulation
- Pauli-Z expectation values
- deterministic seeded sampling
- read-only zero-copy NumPy views over native result storage
- OpenMP parallelism for sufficiently large amplitude workloads
- compiler-optimized release builds

## Execution model

C++ handles `backend="auto"`. The planner validates the requested result against a target before execution and returns an inspectable `ExecutionPlan`.

```python
plan = qp.plan(program, qp.ResultMode.SAMPLE)
print(plan.backend)   # native-cpu
print(plan.method)    # statevector
print(plan.exact)     # True
print(plan.threads)
```

Automatic execution must preserve declared semantics. Future accelerator and specialized-simulator strategies can share this planner without moving backend policy into Python.

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

## Direction

The native planner is the integration point for CUDA/cuQuantum, stabilizer, tensor-network, distributed, and physical-QPU execution. Those engines should compete on measured cost while the QuPy program and result contracts remain stable.

See [REFERENCES.md](REFERENCES.md) for the specifications, libraries, and upstream systems that materially shape the implementation.

## License

MIT
