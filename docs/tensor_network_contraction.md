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

`backend="auto"` does not select general tensor networks yet. Automatic selection remains gated on promoted host-scoped planner evidence; collecting calibration data does not itself change execution policy.

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

The normal `qp.observable_plan(..., backend="native-tn")` maps this structural preflight into QuPy's standard `ObservableExecutionPlan`. For this backend, `estimated_state_bytes` reports the peak tensor intermediate bytes because no dense state vector is materialized. `predicted_ns` remains unset until runtime evidence is promoted into the native planner.

## Runtime calibration

`benchmarks.tensor_network_cost` collects paired dense-CPU and exact tensor-network timings over expectation and multi-observable batch workloads. Each workload records the structural preflight used by the TN executor: contraction count, peak rank, peak intermediate bytes, scalar multiplications, term count, and structural-plan fingerprint.

Timing evidence is accepted only after dense CPU and TN results agree within `2e-11`. CPU and TN calls are counterbalanced within each workload, and the report stores the raw nanosecond samples rather than relying on claimed summary statistics.

Collect three policy reports on the same host:

```text
python -m benchmarks.tensor_network_cost --profile policy --warmups 2 --iterations 8 --output tn-policy-1.json
python -m benchmarks.tensor_network_cost --profile policy --warmups 2 --iterations 8 --output tn-policy-2.json
python -m benchmarks.tensor_network_cost --profile policy --warmups 2 --iterations 8 --output tn-policy-3.json
python -m benchmarks.tensor_network_calibrate tn-policy-1.json tn-policy-2.json tn-policy-3.json --output tn-calibration.json
```

Calibration recomputes medians from the raw timing arrays. It fits separate six-feature log-runtime models for the dense CPU baseline and tensor-network execution. Every prediction used for routing validation is leave-one-workload-out: the held-out workload contributes to neither candidate's fitted coefficients.

A calibration report is validated only when all of the following hold:

- at least three reports from one exact planner host;
- the same semantic workload fingerprint and tensor-network structural-plan fingerprint for each named workload in every report;
- at least 18 paired CPU/TN decision workloads;
- exact CPU/TN agreement no worse than `2e-11` for every workload;
- each model has held-out median error no worse than `1.5x` and maximum error no worse than `2.0x`;
- every leave-one-workload-out backend decision matches the measured faster backend;
- maximum decision regret is no worse than `1.10x`;
- dense CPU and TN are each the measured faster candidate on at least three held-out workloads, proving that the evidence covers a real routing boundary rather than a one-backend sweep.

The calibration artifact is evidence, not an execution-policy override. `backend="auto"` remains unchanged until a native planner schema explicitly consumes this evidence and preserves the same fail-closed validation contract.

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
- held-out CPU/TN routing calibration available, but not yet promoted into native auto policy

The next tensor-network tranche is native planner promotion of validated routing evidence, followed by broader contraction-path optimization only when it improves measured decision quality.
