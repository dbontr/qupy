# General tensor-network contraction

QuPy provides an exact circuit-expectation path based on explicit tensor-network contraction. It complements the existing one-dimensional MPS engine: MPS is specialized for chain-structured state evolution, while the general contraction engine builds the full bra/operator/ket network and contracts according to network structure rather than allocating a dense state vector.

## API

```python
import qupy as qp

program = qp.h(qp.Program(2), 0)
program = qp.cx(program, 0, 1)

observable = qp.observable(
    [
        qp.pauli_term(
            1.0,
            [qp.pauli(0, qp.Pauli.X), qp.pauli(1, qp.Pauli.X)],
        ),
        qp.pauli_term(
            0.5,
            [qp.pauli(0, qp.Pauli.Z), qp.pauli(1, qp.Pauli.Z)],
        ),
    ]
)

result = qp.tensor_network_expectation(program, observable)
print(result.value)
print(result.peak_tensor_rank)
print(result.peak_tensor_bytes)
print(result.scalar_multiplications)
```

The engine supports every operation in the numerical `Program` IR and arbitrary real Pauli-sum observables.

## Network construction

For each Pauli term, QuPy constructs a closed expectation network for

`<0| U^dagger P U |0>`.

Each qubit begins with an explicit `|0>` rank-one tensor. One-qubit gates become rank-two tensors and two-qubit gates become rank-four tensors. A conjugated copy of the circuit forms the bra side. Final ket and bra wire indices are connected by one Pauli matrix per qubit, including identity matrices on qubits omitted from the term.

No state vector is materialized. Every binary tensor index has dimension two.

## Contraction path

The first general engine uses a deterministic greedy pair-selection heuristic. At each step it considers tensor pairs sharing one or more indices and prefers the pair with the smallest contraction work rank, then the smallest output rank. Stable tensor identifiers break remaining ties.

This heuristic is deliberately inspectable and deterministic. General contraction-path optimization is combinatorial, and a future path optimizer can improve runtime without changing circuit or observable semantics.

Disconnected circuit components naturally contract to separate scalars and are multiplied only after their internal indices are eliminated.

## Exactness and memory guard

The engine performs no SVD truncation, bond cap, sampling, or approximate slicing. `TensorNetworkEstimate.exact` is therefore `True`. Numerical floating-point roundoff remains subject to the same machine-precision limits as the other native exact simulators.

`max_tensor_bytes` is a hard per-intermediate memory ceiling. QuPy checks every input and generated tensor before accepting it. If the selected contraction path would materialize a tensor larger than the limit, execution fails instead of silently approximating or allocating beyond the declared ceiling.

The default is 1 GiB:

```python
result = qp.tensor_network_expectation(
    program,
    observable,
    max_tensor_bytes=1 << 30,
)
```

## Provenance

`TensorNetworkEstimate` reports:

- final expectation value
- Pauli-term count
- number of tensor contractions
- peak intermediate tensor rank
- peak intermediate tensor bytes
- scalar multiplication count
- `exact=True`
- backend `native-tn`
- method `greedy-contraction`

These fields make the structural cost visible rather than hiding tensor-network behavior behind a single result scalar.

## When it helps

Dense state-vector execution scales with `2^n` amplitudes regardless of circuit connectivity. General tensor contraction instead scales primarily with the width induced by the network and contraction path. Low-treewidth or weakly connected circuits can therefore remain practical at qubit counts where a dense state vector is impossible.

The conformance suite includes an 80-qubit product circuit that contracts through rank-two intermediates while agreeing with the analytic expectation value. Small entangled circuits are cross-checked against QuPy's dense native observable engine.

## Relationship to MPS

Use explicit `native-mps` when the one-dimensional MPS representation is structurally favorable and its exact bond evolution is the desired execution model. Use `tensor_network_expectation()` when circuit topology is not naturally one-dimensional or when a general contraction path can eliminate local structure before large state tensors appear.

The current general-TN API is explicit rather than part of `backend="auto"`. Automatic routing requires measured planner evidence for the new cost surface before QuPy can claim it safely.

## Current boundary

- expectation values only; state-vector materialization remains owned by the dense and MPS paths
- binary qubit tensor indices only
- deterministic greedy pair contraction
- CPU execution
- Pauli terms are contracted independently
- no approximation or slicing

The next optimization layer is contraction-path planning and evidence-gated integration with QuPy's execution planner. Multi-device tensor contraction can then build on the same explicit network and path provenance.
