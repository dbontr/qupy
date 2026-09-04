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

`qp.expect(program, observable, backend="native-tn")` and `qp.expect_observables(...)` use the same explicit backend. The backend owns rich observable expectations and expectation differentiation. State vectors, density matrices, variances, and covariances fail explicitly instead of silently changing execution method.

Automatic selection is available only when a validated host-scoped tensor-network policy artifact is installed or configured. Explicit backends never depend on that artifact.

## Tensor-network differentiation

The standard differentiation surface accepts the same explicit backend without creating a second tensor-network-specific user API:

```python
import numpy as np
import qupy as qp

program = qp.ry(qp.Program(2), 0.0, 0)
program = qp.rx(program, 0.0, 1)
program = qp.cx(program, 0, 1)
observable = qp.observable_from_z(qp.Z(0))
slots = [qp.ParameterSlot(0), qp.ParameterSlot(1)]
parameters = np.array([0.37, -0.21], dtype=np.float64)
result = qp.value_and_grad(
    program,
    observable,
    slots,
    parameters,
    backend="native-tn",
)
```

`qp.value_and_grad()`, `qp.grad()`, `qp.jacobian()`, and `qp.hessian()` evaluate their expectation queries through exact tensor-network contraction when `backend="native-tn"` is explicit. `GradientMethod.AUTO` selects analytic parameter shift for this backend. Explicit finite differences remain available with a positive finite `epsilon`; adjoint differentiation fails explicitly because the current adjoint implementation requires the native CPU state-vector backend.

This is expectation-based differentiation rather than reverse-mode contraction-tree differentiation: each shifted circuit is contracted independently under the standard 1 GiB tensor-intermediate policy. The reported backend remains `native-tn` and the method reports `parameter-shift` or `finite-difference` as appropriate.

Automatic tensor-network policy remains calibrated on expectation workloads, not on the multiplied evaluation cost of gradients, Jacobians, or Hessians. Therefore `backend="auto"` differentiation keeps the established differentiation planner instead of silently applying the expectation-only TN cost model. Gradient-aware automatic TN routing requires its own held-out timing evidence.

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

The normal `qp.observable_plan(..., backend="native-tn")` maps this structural preflight into QuPy's standard `ObservableExecutionPlan`. For this backend, `estimated_state_bytes` reports the peak tensor intermediate bytes because no dense state vector is materialized. An automatically selected plan additionally carries the promoted model class, model fingerprint, and predicted runtime.

## Runtime calibration and promotion

`benchmarks.tensor_network_cost` collects paired dense-CPU and exact tensor-network timings over expectation and multi-observable batch workloads. Each workload records the structural preflight used by the TN executor: contraction count, peak rank, peak intermediate bytes, scalar multiplications, term count, and structural-plan fingerprint.

Timing evidence is accepted only after dense CPU and TN results agree within `2e-11`. CPU and TN calls are counterbalanced within each workload, and the report stores the raw nanosecond samples rather than relying on claimed summary statistics.

Collect three policy reports on the same host, validate them, and promote the same evidence into a native artifact:

```text
python -m benchmarks.tensor_network_cost --profile policy --warmups 2 --iterations 8 --output tn-policy-1.json
python -m benchmarks.tensor_network_cost --profile policy --warmups 2 --iterations 8 --output tn-policy-2.json
python -m benchmarks.tensor_network_cost --profile policy --warmups 2 --iterations 8 --output tn-policy-3.json
python -m benchmarks.tensor_network_calibrate tn-policy-1.json tn-policy-2.json tn-policy-3.json --output tn-calibration.json
python -m benchmarks.tensor_network_promote tn-policy-1.json tn-policy-2.json tn-policy-3.json --output tn-policy.qptncost
```

Calibration recomputes medians from the raw timing arrays. It fits separate six-feature log-runtime models for the dense CPU baseline and tensor-network execution. Every prediction used for routing validation is leave-one-workload-out: the held-out workload contributes to neither candidate's fitted coefficients.

A policy is promotable only when all of the following hold:

- at least three reports from one exact planner host;
- the same semantic workload fingerprint and tensor-network structural-plan fingerprint for each named workload in every report;
- at least 18 paired CPU/TN decision workloads;
- exact CPU/TN agreement no worse than `2e-11` for every workload;
- each model has held-out median error no worse than `1.5x` and maximum error no worse than `2.0x`;
- every leave-one-workload-out backend decision matches the measured faster backend;
- maximum decision regret is no worse than `1.10x`;
- dense CPU and TN are each the measured faster candidate on at least three held-out workloads, proving that the evidence covers a real routing boundary rather than a one-backend sweep.

Promotion also records the observed minimum and maximum of every model feature. Runtime prediction is interpolation-only: if either the dense-CPU or tensor-network feature vector falls outside those measured bounds, the TN policy declines the workload instead of extrapolating.

## Installing and discovering policy evidence

Validate and install a promoted artifact once:

```python
import qupy as qp

model = qp.install_tensor_network_cost_model("tn-policy.qptncost")
print(model.auto_validated)
print(qp.tensor_network_planner_cache_path())
```

The installed artifact is scoped by QuPy core version and `planner_host_fingerprint()`. Automatic discovery uses this precedence:

1. an in-process override from `qp.set_default_tensor_network_cost_model(path)`;
2. `QUPY_TENSOR_NETWORK_COST_MODEL`;
3. the host-scoped installed cache artifact.

`QUPY_CACHE_DIR` overrides the cache root using the same convention as QuPy's existing planner cache. `qp.remove_tensor_network_cost_model()` removes only the installed TN artifact; it does not alter the existing `.qpcost` planner policy.

The TN policy format is intentionally independent of planner schemas v1-v5. General tensor-network evidence compares dense CPU with exact TN execution and therefore does not require unrelated CUDA, MPS, observable-CUDA, or noisy-density sections merely to load on a CPU-only host.

## Conservative automatic composition

TN policy is composed after QuPy's established automatic observable planner. It may replace a normal dense-CPU rich-observable expectation only when all of these conditions hold:

- `backend="auto"`;
- a valid TN artifact is discovered;
- the established planner chose `native-cpu`;
- the workload is a full-cone, non-Clifford rich-observable expectation in the calibrated feature domain;
- the existing planner is not already making a validated CUDA/MPS/rich-observable adaptive decision;
- the TN structural preflight satisfies the standard 1 GiB intermediate policy;
- the promoted TN model predicts lower runtime than the paired dense-CPU model.

Pauli propagation is never displaced by TN evidence. An existing accelerated automatic decision is also preserved because the current TN calibration compares only CPU and TN; QuPy does not infer an unmeasured TN-vs-CUDA or TN-vs-MPS ranking.

If the TN model predicts CPU, falls outside its calibration domain, or is absent, the established automatic plan is returned unchanged. Explicit `native-cpu`, `native-cuda`, `native-mps`, and `native-tn` requests do not load the TN artifact.

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

The standard observable result reports backend `native-tn`; its companion `ObservableExecutionPlan` carries method `greedy-contraction-observable`, exactness, fingerprints, cache identity, term/group counts, and peak workspace bytes. Automatic TN plans also bind cache identity to the TN structural-plan fingerprint and promoted policy fingerprint.

## When it helps

Dense state-vector execution scales with `2^n` amplitudes regardless of circuit connectivity. General tensor contraction instead scales primarily with the width induced by the network and contraction path. Low-treewidth or weakly connected circuits can therefore remain practical at qubit counts where a dense state vector is impossible.

The conformance suite includes an 80-qubit product circuit executed through the normal `expect_observable(..., backend="native-tn")` API while retaining rank-two intermediates. Small entangled circuits are cross-checked against QuPy's dense native observable engine.

Automatic routing is deliberately narrower than explicit TN capability: the policy may only interpolate inside the workload region represented by its promoted evidence. Large or otherwise novel low-width workloads remain available through explicit `native-tn` until matching timing evidence is collected.

## Relationship to MPS and distributed execution

Use explicit `native-mps` when a one-dimensional MPS representation is structurally favorable. Use explicit `native-tn` for rich observable expectations when general topology elimination is preferable.

QuPy also supports MPI term-parallel tensor-network execution through `distributed_tensor_network_expectation()`. That path distributes independent Pauli terms across ranks; it does not imply multi-GPU CUDA execution.

## Current boundary

- rich Pauli-sum expectation values
- explicit parameter-shift and finite-difference gradients, Jacobians, and parameter-shift Hessians through the standard differentiation API
- binary qubit tensor indices
- deterministic greedy pair contraction
- CPU execution per contraction
- Pauli terms contracted independently
- no approximation or slicing
- explicit `native-tn` across its full supported workload surface
- automatic CPU/TN selection only with promoted host-scoped evidence and only inside the recorded calibration domain
- existing CUDA/MPS adaptive decisions preserved until direct cross-backend TN evidence exists

The next tensor-network performance frontier is broader validated cross-backend policy and contraction-path optimization only when measured evidence shows improved decision quality.
