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

`append_pauli_evolution()` appends `exp(-i * time * coefficient * P)` for one `PauliTerm` P.

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

QuPy rotates X and Y factors into the Z basis, accumulates parity with a CX chain, applies one RZ rotation with angle `2 * time * coefficient`, then exactly uncomputes the parity and basis changes. Identity factors require no gates. This operation is exact for a single Pauli string subject to floating-point roundoff. Identity-only terms contribute only a physically irrelevant global phase and therefore require no operation in QuPy's numerical program model.

## Pauli-Hamiltonian product formulas

`append_hamiltonian_evolution()` composes exact Pauli-string exponentials into an explicit product-formula approximation for a Hamiltonian represented by `Observable`.

```python
hamiltonian = qp.Observable(
    [
        qp.PauliTerm(0.7, [qp.PauliFactor(0, qp.Pauli.X)]),
        qp.PauliTerm(-0.2, [qp.PauliFactor(0, qp.Pauli.Z)]),
    ]
)
program = qp.append_hamiltonian_evolution(
    qp.Program(1), hamiltonian, 1.0, steps=8, order=2
)
```

`order=1` uses a Lie-Trotter step in the Hamiltonian term order. `order=2` uses a symmetric second-order composition. `steps` must be a positive integer and is never selected implicitly.

For mutually commuting Pauli terms, either order is exact up to floating-point roundoff and global phase. For noncommuting terms, the returned `Program` is an approximation whose error depends on commutators, total time, term ordering, formula order, and step count. QuPy does not relabel that approximation as exact execution; after construction, backend execution still follows the ordinary QuPy planning and exactness contracts for the resulting program.

## Weighted MaxCut

`maxcut_hamiltonian()` constructs the weighted MaxCut objective `C = sum w_ij (I - Z_i Z_j) / 2`.

```python
cost = qp.maxcut_hamiltonian(
    4,
    [(0, 1), (1, 2), (2, 3), (3, 0)],
    weights=[1.0, 2.0, 1.0, 2.0],
)
```

The expectation of `cost` on a computational-basis state is the weight of the represented cut. Edges are undirected, self-loops and duplicate undirected edges are rejected, weights must be finite and non-negative, and edge order is canonicalized so equivalent input orderings produce the same observable fingerprint. The identity component is retained so expectations report the actual cut objective rather than a constant-shifted objective.

## MaxCut QAOA

`qaoa_maxcut_program()` constructs the standard alternating QAOA circuit for the weighted MaxCut Hamiltonian with an X mixer.

```python
program = qp.qaoa_maxcut_program(
    4,
    [(0, 1), (1, 2), (2, 3), (3, 0)],
    gammas=[0.7, 0.4],
    betas=[0.3, 0.2],
)
energy = qp.expect(program, cost)
```

The constructor starts in `|+>^n`, applies one MaxCut cost layer for each `gamma`, then applies `exp(-i beta X)` independently to every qubit. MaxCut cost terms are all Z-type and commute, so each cost layer is exact up to global phase and floating-point roundoff rather than a Trotter approximation.

`gammas` and `betas` are explicit numeric layer parameters and must have the same non-zero length. QuPy does not introduce a second symbolic parameter-expression system solely for QAOA. Parameter searches can construct programs from candidate vectors and evaluate them through the same planner and backends, while native slot-based differentiation remains available for `VariationalTemplate` workflows whose parameters map directly to native gate slots.

For a single unweighted edge, the p=1 choice `gamma = pi/2`, `beta = pi/8` reaches the maximum cut expectation of 1; this analytic case is part of the conformance suite.

## Design boundary

The algorithm layer deliberately returns native primitives:

- no alternate circuit IR;
- no Python simulator;
- no hidden approximation or implicit product-formula step selection;
- no dependency on a separate optimization framework;
- no symbolic parameter-expression engine layered over native `Program` identity.

Higher-level applications can build on these constructors while keeping execution, differentiation, planning, observables, and exactness policy in the existing native ownership boundaries.
