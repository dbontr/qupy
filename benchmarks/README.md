# QuPy benchmark harness

The benchmark harness measures semantic workload classes rather than one undifferentiated circuit set. It is development infrastructure and is not imported by the `qupy` runtime package.

## Workload classes

- `clifford-ghz-z` measures exact Pauli or stabilizer-compatible observable evaluation on an entangled Clifford circuit.
- `local-nonclifford-z` places the relevant non-Clifford work in a one-qubit observable cone and adds unrelated non-Clifford work elsewhere. This exposes exact result-aware reduction.
- `entangled-nonclifford-z` keeps the non-Clifford operation inside an all-qubit entangled cone. This is the dense exact fallback class.

Every workload carries a closed-form expected Pauli-Z value. Each adapter must reproduce that value within the declared tolerance before any timing samples are accepted. Workloads default to `1e-9`. An adapter may declare a larger numerical validation tolerance when its upstream numerical contract requires it; the effective tolerance is written into every result. The qsim adapter uses `1e-5`, matching qsim's own expectation-value comparison tolerance.

## Timing contract

Circuit/program construction and adapter translation happen before timing. The timed region is the engine's user-level expectation execution call on the prepared workload. QuPy therefore includes its native planning, compilation, and execution work in each timed call. External adapters use their prepared circuit and simulator objects.

The JSON report records:

- workload family, qubits, operation count, and observable;
- engine, version, and selected method;
- expected value, measured value, effective tolerance, and semantic-validity status;
- warmup count and raw nanosecond timing samples;
- median, minimum, and maximum timing;
- host platform, Python version, and the native planner host fingerprint;
- QuPy planner evidence including the versioned structural workload fingerprint, original/active qubits and operations, gate-class counts, compiled steps, estimated state bytes, thread count, core version, and IR version;
- explicit skip reasons for unsupported workload/engine pairs.

Skipped work is never replaced by a different simulation method without being reported.

QuPy workload fingerprints are performance-structure identities, not exact program identities. Version 1 includes the result mode, original qubit count and gate histogram, active qubit count, and the result-aware active operation shape. Gate parameter values, target identity, selected execution method, host, and timing are deliberately excluded. Parameter sweeps over the same structure therefore share one workload fingerprint, while each benchmark result still records the selected method and host separately. `program_fingerprint` remains the exact semantic program identity for caching and reproducibility.

## Profiles

`smoke` uses six-qubit workloads and is intended for adapter compatibility checks.

`standard` includes 64, 512, and 4,096-qubit Clifford workloads plus 12, 16, and 20-qubit local and entangled non-Clifford workloads. Dense qsim and Aer state-vector adapters are capped at 24 qubits by the portable harness. Specialized engines can run larger workloads when their declared method supports them.

## Usage

Run QuPy only:

```text
python -m benchmarks.run --profile standard --engines qupy --warmups 2 --iterations 10 --output benchmark.json
```

Fit held-out QuPy planner cost models and emit a native artifact:

```text
python -m benchmarks.run --profile calibration --engines qupy --warmups 1 --iterations 3 --output calibration.json
python -m benchmarks.calibrate calibration.json --output cost-model.json --planner-output planner.qpcost
```

The calibration command requires every fitted cost class to pass its held-out error thresholds before it emits `planner.qpcost`. The native artifact is scoped to the QuPy core version, workload schema, and planner host fingerprint recorded by the benchmark process.

Validate the exact adaptive MPS observable policy with repeated paired reports:

```text
python -m benchmarks.mps_cost --profile policy --warmups 2 --iterations 20 --output mps-policy-1.json
python -m benchmarks.mps_cost --profile policy --warmups 2 --iterations 20 --output mps-policy-2.json
python -m benchmarks.mps_cost --profile policy --warmups 2 --iterations 20 --output mps-policy-3.json
python -m benchmarks.mps_calibrate mps-policy-1.json mps-policy-2.json mps-policy-3.json --base-artifact planner.qpcost --output mps-calibration.json --planner-output planner-v3.qpcost
```

The policy profile requires an even iteration count. For each workload, CPU and adaptive execution are counterbalanced as one pair, and MPS and adaptive execution are counterbalanced as a second pair. This prevents a slow baseline from contaminating the timing of the other decision candidate. Promotion recomputes medians and regret from raw samples, requires at least three reports and 16 distinct workloads, requires exact agreement within `2e-11`, and allows no workload above 10% median regret. A schema-v1 or schema-v2 base artifact can be promoted; existing validated CUDA evidence is preserved.

Run all compatibility adapters after installing their optional packages:

```text
python -m benchmarks.run --profile smoke --engines qupy,stim,qsim,aer-statevector,aer-stabilizer --warmups 1 --iterations 5 --require-engines
```

The repository CI compatibility job installs Stim 1.16.0, qsimcirq 0.22.1, and Qiskit Aer 0.17.2 only for benchmark-adapter verification. These packages are not QuPy runtime dependencies.

## Interpretation

Do not compare methods that answer different questions as if they were interchangeable. The report must be interpreted by workload family, result semantics, exactness, method, and effective numerical tolerance. CI verifies adapter correctness but does not enforce timing thresholds because hosted-runner performance is not stable enough for regression claims.
