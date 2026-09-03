# General tensor-network contraction

QuPy provides exact circuit-expectation execution by explicit tensor-network contraction. It complements the one-dimensional MPS engine: MPS is specialized for chain-structured state evolution, while the general contraction engine builds a closed bra/operator/ket network and contracts according to network structure rather than allocating a dense state vector.

## First-class observable backend

Rich Pauli-sum observables can use the tensor-network engine through QuPy's normal observable API:

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

plan = qp.observable_plan(program, [observable], backend="native-tn")
result = qp.expect_observable(program, observable, backend="native-tn")
```

`qp.expect(program, observable, backend="native-tn")` and `qp.expect_observables(...)` use the same explicit backend. The backend currently owns rich observable expectations only. State vectors, density matrices, variances, and covariances fail explicitly instead of silently changing execution method.

`backend="auto"` does not select general tensor networks yet. Automatic selection requires measured, host-scoped planner evidence for the tensor-network cost surface before QuPy can safely add it to the existing evidence-gated policy.

## Structural preflight

`qp.tensor_network_plan()` performs a topology-only dry run of the deterministic contraction path before numerical contraction:

```python
plan = qp.tensor_network_plan(
    program,
    observable,
    max_tensor_bytes=1 << 30,
)

print(plan.peak_tensor_rank)
print(plan.peak_tensor_bytes)
print(plan.contractions)
print(plan.scalar_multiplications)
print(plan.plan_fingerprint)
```

The structural planner uses tensor indices, ranks, and stable tensor identifiers. It does not materialize numerical contraction intermediates. It reports:

- Pauli-term count
- contraction count
- peak intermediate tensor rank
- peak intermediate tensor bytes
- scalar multiplication count
- hard per-intermediate memory policy
- exactness
- backend and method
- program fingerprint
- observable fingerprint
- structural-plan fingerprint

The native conformance suite requires planner work, peak-rank, and peak-byte metrics to agree exactly with the numerical executor. This guards against planner/executor drift.

The normal `qp.observable_plan(..., backend="native-tn")` maps this structural preflight into QuPy's standard `ObservableExecutionPlan`. For this backend, `estimated_state_bytes` reports the peak tensor intermediate bytes because no dense state vector is materialized. `predicted_ns` remains unset until runtime calibration exists.

## Network construction

For each nonzero Pauli term, QuPy constructs a closed expectation network for

`<0| U^dagger P U |0>`.

Each qubit begins with an explicit `|0>` rank-one tensor. One-qubit gates become rank-two tensors and two-qubit gates become rank-four tensors. A conjugated copy of the circuit forms the bra side. Final ket and bra wire indices are connected by one Pauli matrix per qubit, including identity matrices on qubits omitted from the term.

No dense state vector is materialized. Every tensor index is binary.

## Contraction path

The current engine uses a deterministic greedy pair-selection heuristic. At each step it considers tensor pairs sharing one or more indices and prefers the pair with the smallest contraction work rank, then the smallest output rank. Stable tensor identifiers break remaining ties.

Disconnected circuit components naturally contract to separate scalars and are multiplied only after their internal indices are eliminated.

The heuristic is deliberately inspectable and deterministic. Better contraction-tree search can be added later without changing circuit or observable semantics.

## Exactness and memory guard

The engine performs no SVD truncation, bond cap, sampling, or approximate slicing. Results are exact subject to floating-point roundoff.

`max_tensor_bytes` is a hard per-intermediate memory ceiling. The structural planner rejects a path whose selected intermediate exceeds the limit before numerical contraction begins. Runtime checks remain in place as a second line of defense.

The detailed APIs accept a custom limit:

```python
result = qp.tensor_network_expectation(
    program,
    observable,
    max_tensor_bytes=1 << 28,
)
```

The normal `backend="native-tn"` observable path uses the standard 1 GiB tensor-intermediate policy. Use `tensor_network_plan()` and `tensor_network_expectation()` when a different explicit ceiling is required.

## Result provenance

`TensorNetworkEstimate` reports:

- final expectation value
- Pauli-term count
- contraction count
- peak intermediate tensor rank
- peak intermediate tensor bytes
- scalar multiplication count
- `exact=True`
- backend `native-tn`
- method `greedy-contraction`

The standard observable result reports backend `native-tn`; its companion `ObservableExecutionPlan` carries method `greedy-contraction-observable`, exactness, fingerprints, cache identity, term/group counts, and peak workspace bytes.

## When it helps

Dense state-vector execution scales with `2^n` amplitudes regardless of circuit connectivity. General tensor contraction instead scales primarily with the width induced by the network and contraction path. Low-treewidth or weakly connected circuits can therefore remain practical at qubit counts where a dense state vector is impossible.

The conformance suite includes an 80-qubit product circuit executed through the normal `expect_observable(..., backend="native-tn")` API while retaining rank-two intermediates. Small entangled circuits are cross-checked against QuPy's dense native observable engine.

## Relationship to MPS and distributed execution

Use explicit `native-mps` when a one-dimensional MPS representation is structurally favorable. Use explicit `native-tn` for rich observable expectations when general topology elimination is preferable.

QuPy also supports MPI term-parallel tensor-network execution through `distributed_tensor_network_expectation()`. That path distributes independent Pauli terms across ranks; it does not imply multi-GPU CUDA execution.

## Current boundary

- rich Pauli-sum expectation values
- binary qubit tensor indices
- deterministic greedy pair contraction
- CPU execution per contraction
- Pauli terms contracted independently
- no approximation or slicing
- explicit `native-tn`; no automatic planner selection yet

The next tensor-network tranche is evidence-backed path and runtime calibration, followed by automatic routing only if held-out measurements demonstrate a safe decision policy.
