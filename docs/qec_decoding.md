# Detector-model decoding

QuPy uses detector error models to represent independent error mechanisms. Each `DetectorError` has a probability, a detector support, and an optional logical-observable support. Supports are binary: repeated detector or observable indices cancel in pairs.

QuPy provides two detector-model decoders with different contracts.

## Exact reference decoder

`decode_detector_model(model, syndrome)` is the small-model reference path. It enumerates every subset of error mechanisms and returns the maximum-likelihood assignment that reproduces the requested syndrome.

The reference decoder deliberately rejects models with more than 24 error mechanisms. Its exponential search is useful for conformance and small cases, not scale.

## Sparse BP+OSD-0 decoder

`BpOsdDecoder` is the scalable native path. Construct the decoder once when many syndromes share the same detector model.

```python
import numpy as np
import qupy as qp

model = qp.repetition_code_detector_model(
    distance=5,
    rounds=4,
    data_error_probability=0.01,
    measurement_error_probability=0.02,
)

decoder = qp.BpOsdDecoder(model, max_iterations=50, damping=0.1)
result = decoder.decode(np.zeros(model.detector_count, dtype=np.int8))

print(result.correction)
print(result.observables)
print(result.log_likelihood)
print(result.bp_converged, result.osd_used)
```

The decoder preprocesses the model once:

1. Error mechanisms with probability 0 are excluded because they cannot occur.
2. Error mechanisms with probability 1 are fixed into every correction and their detector parity is removed from the requested syndrome.
3. All remaining error mechanisms become variable nodes in a sparse detector/error Tanner graph.
4. Each active error starts with the prior log-likelihood ratio `log((1-p)/p)`.

Belief propagation uses binary parity-check sum-product updates in log-likelihood-ratio space. Check messages use the standard hyperbolic-tangent product form; variable messages add the prior and incoming check evidence. Messages are bounded to keep the numerical representation finite. `damping` applies to check-to-variable updates and must be in `[0, 1)`.

After each BP iteration, QuPy converts posterior log-likelihood ratios to a hard error assignment and checks the detector parity exactly. If the assignment reproduces the syndrome, decoding stops.

### OSD-0 fallback

If BP does not satisfy the syndrome within `max_iterations`, QuPy performs deterministic order-0 ordered-statistics repair:

1. Rank active error mechanisms by increasing absolute posterior log-likelihood ratio. Lower magnitude means lower reliability.
2. Select a linearly independent pivot set from the least reliable detector-error columns.
3. Keep the BP hard decisions for non-pivot variables.
4. Solve the pivot variables over GF(2) so the complete assignment reproduces the target syndrome.

This is order-0 post-processing: QuPy does not enumerate higher-order reliability flips around the repaired solution. If the requested syndrome is outside the span of all nonzero-probability error mechanisms, decoding fails instead of returning an inconsistent correction.

Every successful result is verified again against the original full detector model, including probability-1 mechanisms, before it is returned.

## Result contract

`BpOsdDecodeResult` exposes:

- `correction`: one bit per detector error mechanism in model order;
- `observables`: predicted logical-frame changes;
- `log_likelihood`: log probability of the returned independent-error assignment;
- `matched_errors`: number of selected error mechanisms;
- `iterations`: number of BP iterations completed;
- `bp_converged`: whether BP itself found a syndrome-consistent assignment;
- `osd_used`: whether OSD-0 repaired the BP assignment;
- `method`: `belief-propagation` or `belief-propagation-osd0`.

The returned correction is guaranteed to reproduce the requested syndrome when decoding succeeds. BP+OSD-0 is not a maximum-likelihood guarantee. Use the bounded exact decoder when an exact small-model reference is required.

For fixed model, syndrome, iteration limit, and damping, BP+OSD-0 is deterministic.

## Batch decoding

`decode_batch` accepts a two-dimensional array with shape `(shots, detector_count)` and executes the complete batch natively.

```python
samples = qp.sample_detector_model(model, shots=4096, seed=7)
batch = decoder.decode_batch(samples.syndrome)

predicted_logicals = batch.observables
```

Batch outputs are native-owned read-only NumPy views. The batch API removes per-shot Python dispatch overhead. The current implementation does not promise parallel shot execution.

## Surface-code decoder evidence

`python -m benchmarks.qec` measures QuPy BP+OSD-0 against PyMatching sparse-blossom minimum-weight perfect matching on identical detector samples from Stim-generated rotated surface-code memory circuits. Stim and PyMatching are benchmark-only dependencies and are not imported by the QuPy runtime package.

The harness converts the flattened Stim detector error model into a QuPy `DetectorModel`. It preserves the complete parity of each independent error mechanism. Stim `^` separators are treated as suggested graphlike decompositions, not as separate independent errors. The harness then checks every QuPy batch correction by reconstructing its detector syndrome from the imported model.

Each report includes:

- surface-code task, distance, rounds, physical error rate, shots, and fixed seed;
- detector, observable, error-mechanism, active-variable, and Tanner-edge counts;
- QuPy detector-model fingerprint and conversion time;
- decoder construction and raw batch-decode timing samples;
- median batch throughput without a hosted-CI timing threshold;
- logical-failure counts, rates, and 95% Wilson intervals for both decoders;
- QuPy BP convergence rate, OSD-0 usage rate, and iteration statistics;
- cross-decoder logical prediction agreement;
- exact QuPy correction-to-syndrome consistency.

The `smoke` profile is an integration check. The `standard` profile spans rotated X/Z memories, distances 3, 5, and 7, and two physical error rates. These Stim-generated circuits are reproducible reference workloads, not an exhaustive QEC corpus. Controlled repeated measurements with adequate shots are required before drawing decoder-quality or latency conclusions.

## Hypergraph-product QLDPC evidence

`python -m benchmarks.qec_ldpc` exercises QuPy on a second code family with a different graph structure and comparator. It constructs square Tillich-Zémor hypergraph-product CSS codes from full-rank binary Hamming parity-check matrices and evaluates independent code-capacity X noise. The external `ldpc` package is a benchmark-only dependency; it is not imported by QuPy's runtime package.

For a seed parity-check matrix `H` with shape `(r, n)`, the benchmark constructs

- `H_X = [H ⊗ I_n | I_r ⊗ H^T]`;
- `H_Z = [I_n ⊗ H | H^T ⊗ I_r]`.

The benchmark verifies `H_X H_Z^T = 0` over GF(2), computes a complete logical-Z basis for `ker(H_X) / row(H_Z)`, and verifies that basis against the X checks before sampling any errors. Each physical X error becomes one QuPy detector-error mechanism: its detector support is the corresponding `H_Z` column and its logical-frame support is the corresponding column of the logical-Z basis.

The seeded code-capacity sampler generates the same syndrome and logical truth used to score both decoders. QuPy uses native BP+OSD-0. The independent comparator uses `ldpc` 2.4.1 product-sum belief propagation with OSD-0. Both returned corrections are multiplied back through `H_Z`; the benchmark fails if either decoder returns a correction that does not reproduce the requested syndrome.

The smoke workload uses the rank-3 Hamming seed. It produces a 58-qubit CSS code with 21 X checks, 21 Z checks, and 16 encoded logical qubits. Its 58 independent physical-error mechanisms already exceed the exact reference decoder's 24-mechanism cap. The standard profile also includes the rank-4 seed, producing a 241-qubit code with 121 logical qubits, across several code-capacity error rates.

QLDPC reports include code dimensions, fixed seed and physical error rate, model fingerprint, sampling/setup/decode times, logical-failure counts and Wilson intervals, cross-decoder logical prediction agreement, both syndrome-consistency rates, and QuPy BP/OSD statistics. Hosted CI checks semantics and report contracts only. It does not impose a latency winner or logical-failure superiority threshold.

These workloads broaden evidence from topological surface-code circuits to finite-rate hypergraph-product codes, but they remain code-capacity Hamming-seed HGP cases. They do not substitute for circuit-level noise, biased/correlated noise, lifted-product or bivariate-bicycle families, or hardware-derived detector models.

## Scale boundary

The exact reference decoder grows exponentially with the number of error mechanisms and is capped at 24. BP+OSD-0 uses sparse message passing followed, when required, by a polynomial GF(2) solve. Its cost depends on Tanner-graph degree, BP iteration count, detector count, and the rank used by the OSD repair.

For large production workloads, decoder quality must be benchmarked against the target code family and noise model. Higher-order OSD, minimum-weight matching, union-find, or code-specific decoders can provide different accuracy/latency tradeoffs; QuPy does not select those methods implicitly. The repository now carries reproducible surface-code and hypergraph-product QLDPC comparison baselines. Broader code families and circuit-level noise evidence remain necessary before promoting additional decoder methods or making cross-family performance claims.

See [Research and implementation references](../REFERENCES.md) for the detector-model, hypergraph-product, BP+OSD, ordered-statistics, sparse-blossom, and independent LDPC comparator sources used by this implementation and benchmark architecture.
