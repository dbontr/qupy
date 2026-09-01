# QuPy

QuPy is a NumPy-style quantum numerical-computing layer for simulators, hybrid quantum-classical workflows, and physical QPUs.

The project aims for one compact Python programming model while preserving the differences that matter on quantum hardware. Classical parameters use the existing array ecosystem. Qubits, measurements, target constraints, and quantum result requests use explicit quantum semantics.

> **Status:** early alpha. The first reference backend is implemented; the public API and compiler IR will evolve.

## Quick start

```python
import qupy as qp

program = qp.Program(2)
program = qp.h(program, 0)
program = qp.cx(program, 0, 1)

samples = qp.sample(program, shots=1000, seed=7)
energy = qp.expect(program, qp.Z(0))
state = qp.statevector(program)

print(samples.counts())
print(energy.value)
print(state.values)
```

The same program object is designed to become portable across compatible CPU, GPU, specialized-simulator, and QPU targets. Unsupported semantics must fail during validation rather than degrade silently.
## Architecture direction

QuPy is structured as a compiler/runtime rather than a simulator wrapper:

1. A compact Python API builds or captures a quantum program.
2. A typed QuPy IR becomes the stable transformation boundary.
3. Compiler passes simplify, differentiate, batch, and validate the program.
4. A target describes operations, topology, timing, dynamic control, result modes, and execution limits.
5. A planner selects an exact execution strategy by default from the requested result and available targets.
6. A backend executes the lowered program and returns a typed result.

`backend="auto"` is intended to choose among dense state-vector, stabilizer, tensor-network, Pauli-propagation, distributed, or physical-QPU execution. Approximate execution will require an explicit accuracy policy.

## Current reference slice

Implemented now:

- immutable `Program` and `Operation` IR objects
- H, X, RX, and CX operations
- target capability validation
- NumPy state-vector execution
- deterministic seeded sampling
- Pauli-Z expectation values
- explicit simulator state-vector access
- typed sample, expectation, and state-vector results

The NumPy backend is the correctness oracle for future backend conformance tests.
## Development

QuPy currently requires Python 3.12 or newer. The distribution name is provisionally `qupy-compute`; the import package is `qupy`.

```text
uv sync
uv run pytest
uv run ruff check .
uv run mypy src/qupy
```

The core dependency set is intentionally small. CUDA, specialized simulators, autodiff frameworks, and QPU providers will be optional integrations.

## Roadmap

The next architecture milestones are:

1. stabilize the typed IR and result semantics
2. formalize target capability and accuracy-policy models
3. add a second backend and enforce backend conformance
4. add automatic simulator planning
5. add OpenQASM 3 and QIR lowering
6. add differentiation transforms and JAX interoperability
7. add physical-QPU provider adapters

See [REFERENCES.md](REFERENCES.md) for the external specifications and APIs that materially shape the current architecture.

## License

MIT