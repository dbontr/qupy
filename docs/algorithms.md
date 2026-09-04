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

for one `PauliTerm` P, up to state-independent global phase.

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

QuPy rotates X and Y factors into the Z basis, accumulates parity with a CX chain, applies one RZ rotation with angle `2 * time * coefficient`, then exactly uncomputes the parity and basis changes. Identity factors require no gates. A term containing only identity factors contributes only a state-independent global phase and therefore adds no operation to the `Program`.

This operation is exact for a single Pauli string subject to floating-point roundoff and the omitted global phase. It is not itself a product-formula approximation.

## Hamiltonian product-formula evolution

`append_hamiltonian_evolution()` composes exact Pauli-string exponentials to approximate evolution under a real Pauli-sum `Observable`:

```python
hamiltonian = qp.Observable(
    [
        qp.PauliTerm(0.7, [qp.PauliFactor(0, qp.Pauli.X)]),
        qp.PauliTerm(-0.4, [qp.PauliFactor(0, qp.Pauli.Z)]),
    ]
)

program = qp.append_hamiltonian_evolution(
    program,
    hamiltonian,
    time=0.5,
    steps=8,
    order=2,
)
```

`hamiltonian_evolution()` is the corresponding constructor for a fresh native `Program`.

The supported formulas are explicit:

- `order=1`: first-order Lie-Trotter splitting. Each step applies every Hamiltonian term once, in `Observable.terms` order, with time `time / steps`.
- `order=2`: symmetric second-order Suzuki splitting. Each step applies all but the final term forward at half time, the final term at full time, and the preceding terms in reverse at half time.

The second-order construction combines the two adjacent half evolutions of the center term into one full evolution. It does not add redundant back-to-back half steps.

The approximation boundary is the noncommuting Hamiltonian sum. Each individual Pauli exponential is exact up to global phase and floating-point roundoff. If all non-identity terms commute, either supported formula reproduces the Hamiltonian evolution up to global phase and floating-point roundoff. For noncommuting terms, `steps` controls the time-slice size; increasing it generally reduces product-formula error while increasing circuit depth.

The term order is part of the requested product formula. QuPy preserves the `Observable.terms` order instead of sorting noncommuting terms or selecting an undocumented ordering heuristic. Negative time is supported. `steps` must be a positive integer, and `order` must be `1` or `2`.

QuPy does not choose a step count from a hidden accuracy target and does not report an error bound that it has not computed. Applications that need a specific simulation error must select and validate a step count for their Hamiltonian, time interval, and downstream observable.

## Design boundary

The algorithm layer deliberately returns native primitives:

- no alternate circuit IR;
- no symbolic-expression engine;
- no Python simulator;
- no hidden approximate backend selection;
- no hidden Hamiltonian ordering or step-size policy.

Higher-level algorithms can build on these constructors while keeping execution, differentiation, planning, and backend policy in the existing native ownership boundaries.
