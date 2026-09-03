# Algorithm construction

QuPy's algorithm layer builds native `Program` objects rather than introducing a second execution model. Programs returned by these helpers use the same immutable IR, fingerprints, planner, simulators, observables, gradients, and accelerator backends as manually constructed programs.

## Variational templates

`hardware_efficient_ansatz()` creates a native rotation-and-entanglement ansatz and records every variational rotation as a native `ParameterSlot`.

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
bound = template.bind(parameters)
```

Parameter names are deterministic and unique, for example `layer0.q0.ry`. `bind_named()` accepts a mapping keyed by those names and rejects missing or extra parameters.

The tracked slots are the same native slots accepted by QuPy differentiation:

```python
result = qp.value_and_grad(
    template.program,
    hamiltonian,
    list(template.slots),
    parameters,
)
```

No Python-side simulator or differentiation rule is involved. The template owns construction metadata only; binding and differentiation remain native QuPy operations.

Entanglement modes are:

- `linear`: nearest-neighbor CX chain after every rotation layer;
- `ring`: the linear chain plus a final last-to-first CX for three or more qubits;
- `none`: rotations only.

Rotation axes may be any non-empty sequence of `rx`, `ry`, and `rz`.

## Quantum Fourier transform

`qft(n)` constructs an exact n-qubit QFT from H, RZ, CX, and SWAP gates already supported by QuPy's numerical IR.

```python
fourier = qp.qft(6)
inverse = qp.qft(6, inverse=True)
```

For composition with an existing program, use `append_qft()`:

```python
program = qp.x(qp.Program(5), 0)
program = qp.append_qft(program, [0, 2, 4])
program = qp.append_qft(program, [0, 2, 4], inverse=True)
```

The selected-qubit order defines the logical QFT register. Qubits outside that selection are untouched. Duplicate, empty, or out-of-range selections are rejected.

Controlled phase rotations are synthesized from RZ and CX. Each synthesized controlled-phase block differs from the ideal controlled-phase matrix only by a state-independent global phase. Forward and inverse synthesis use opposite angles, so those phases cancel in a QFT/inverse-QFT round trip.

`swaps=False` omits the final bit-order reversal. This is useful when the caller prefers to interpret the reversed logical order instead of paying for physical SWAP operations.

## Exact Pauli-string evolution

`append_pauli_evolution()` appends

`exp(-i * time * coefficient * P)`

for one `PauliTerm` P.

```python
term = qp.PauliTerm(
    0.5,
    [
        qp.PauliFactor(0, qp.Pauli.X),
        qp.PauliFactor(2, qp.Pauli.Y),
        qp.PauliFactor(4, qp.Pauli.Z),
    ],
)
program = qp.append_pauli_evolution(program, term, 0.2)
```

QuPy rotates X and Y factors into the Z basis, accumulates parity with a CX chain, applies one RZ rotation with angle `2 * time * coefficient`, then exactly uncomputes the parity and basis changes. Identity factors require no gates.

This operation is exact for a single Pauli string subject to floating-point roundoff. It is not itself a product-formula approximation. Simulating a Hamiltonian sum by sequentially applying several noncommuting Pauli exponentials introduces whatever product-formula/Trotter error follows from the caller's chosen ordering and step size; QuPy does not hide that approximation.

## Design boundary

The algorithm layer deliberately returns native primitives:

- no alternate circuit IR;
- no symbolic-expression engine;
- no Python simulation loop;
- no hidden approximate execution;
- no automatic Hamiltonian product formula.

Higher-level algorithms can build on these constructors while keeping execution, differentiation, planning, and exactness policy in the existing native ownership boundaries.
