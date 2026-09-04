# Provider execution

QuPy keeps vendor credentials, service policy, and remote transport outside the numerical runtime. Providers are loaded through the stable C provider ABI, while hardware discovery and compilation are layered above that ABI through the provider capability JSON document.

## Execution flow

The hardware path is:

1. Load a `ProviderPlugin`.
2. Read one provider capability snapshot.
3. Parse the optional versioned `hardware_target` object into a native `HardwareTarget`.
4. Compile the logical `Circuit` against that target.
5. Serialize the compiled physical circuit as OpenQASM 3.1.
6. Submit the payload through the existing provider ABI.
7. Poll, retrieve results, or cancel through `ProviderPlugin`.

`qp.submit_circuit()` performs steps 2 through 6. It returns a `ProviderSubmission` containing the provider job identifier, the complete `CompilationResult`, and the exact `ProviderProgram` that was submitted.

```python
import qupy as qp

provider = qp.ProviderPlugin("/path/to/provider-library")

circuit = qp.Circuit(2, 2)
circuit = circuit.h(0).cx(0, 1)
circuit = circuit.measure(0, 0).measure(1, 1)

submission = qp.submit_circuit(provider, circuit, 1000)
print(submission.job_id)
print(submission.compilation.target_fingerprint)
print(submission.program.text)
```

## Capability JSON

Provider ABI version 1 is unchanged. A provider that supports hardware discovery can add a `hardware_target` object to its existing capability JSON:

```json
{
  "formats": ["openqasm3"],
  "hardware_target": {
    "schema_version": 1,
    "name": "example-qpu",
    "num_qubits": 5,
    "one_qubit_operations": ["h", "x", "rz"],
    "two_qubit_operations": ["cz"],
    "couplings": [[0, 1], [1, 2], [2, 3], [3, 4]],
    "measurement": true,
    "mid_circuit_measurement": false,
    "reset": false,
    "dynamic_control": false,
    "durations_ns": {
      "h": 20.0,
      "x": 20.0,
      "rz": 5.0,
      "cz": 120.0,
      "measure": 500.0
    }
  }
}
```

The `hardware_target` schema is independent of the provider ABI version. That separation allows target metadata to evolve without changing the provider function table.

### Required target fields

- `schema_version`: currently `1`
- `name`: non-empty target identifier
- `num_qubits`: positive physical qubit count
- `one_qubit_operations`: native one-qubit unitary operation names
- `two_qubit_operations`: native two-qubit unitary operation names

### Optional target fields

- `couplings`: undirected physical coupling pairs; omission means all-to-all connectivity
- `measurement`: terminal measurement capability
- `mid_circuit_measurement`: measurement followed by later quantum or classically controlled work
- `reset`: reset capability
- `dynamic_control`: single-bit classical feed-forward capability
- `durations_ns`: per-operation durations used by the native ASAP scheduler

Unknown operation names, malformed fields, unsupported schema versions, invalid qubit indices, and contradictory capabilities fail closed.

## Explicit targets

A provider may omit `hardware_target`. In that case a caller can supply a trusted out-of-band target:

```python
target = qp.HardwareTarget(
    "lab-qpu",
    2,
    [qp.CircuitOperationCode.H, qp.CircuitOperationCode.X],
    [qp.CircuitOperationCode.CX],
    measurement=True,
)

submission = qp.submit_circuit(provider, circuit, 1000, target=target)
```

If the provider does advertise a target, an explicitly supplied target must have the same target fingerprint. QuPy rejects a mismatch rather than compiling against constraints that disagree with the provider's own capability snapshot.

## Precompiled submission

`qp.submit_compiled_circuit()` accepts an existing `CompilationResult`. This is useful when an application needs to inspect or archive compilation provenance before submission.

The provider must advertise `openqasm3` support. QuPy does not silently switch formats or bypass target validation.

## Provider conformance

`qp.check_provider_conformance()` exercises the portable discovery, compile, submit, poll, and result-retrieval contract against a provider plug-in. It is explicit because it submits a real provider job. Calling it can consume provider quota or incur provider-side cost; credentials, service policy, rate limits, and billing remain the caller's responsibility.

```python
import qupy as qp

provider = qp.ProviderPlugin("/path/to/provider-library")
report = qp.check_provider_conformance(
    provider,
    shots=1,
    max_polls=32,
    poll_interval_seconds=0.5,
)
print(report.to_json())
```

The checker requires advertised OpenQASM 3 support and either an advertised `hardware_target` or an explicit trusted target. It compiles and submits one single-qubit terminal-measurement circuit, then accepts repeated `queued` or `running` states while rejecting a regression from `running` back to `queued`. `failed`, `cancelled`, timeout, malformed result JSON, and capability/target inconsistencies fail closed.

For command-line provider adapters that advertise their target, the packaged wheel installs:

```text
qupy-provider-conformance /path/to/provider-library --shots 1 --max-polls 32 --poll-interval 0.5
```

The report contains provider/target identity, the observed lifecycle states, and SHA-256 identities for the submitted program, provider job identifier, and result JSON. It deliberately does not embed the raw program text, remote job identifier, or provider result payload.

This conformance check proves the portable QuPy provider contract only. It does not certify physical quantum fidelity, queue or latency service levels, billing behavior, credential lifecycle, security/compliance properties, provider-specific result semantics, or vendor SDK correctness. Those remain adapter/provider evidence. Cancellation remains part of ABI-v1 and is tested independently at the native ABI boundary; the successful remote conformance path does not create a cancellation race solely to test that operation.

## Provider ABI stability

The C provider ABI remains version 1 and continues to expose:

- capability discovery
- submit
- poll
- result retrieval
- cancellation
- provider teardown

The hardware execution bridge does not add credentials, HTTP clients, vendor SDKs, or service-specific policy to QuPy core. A vendor adapter owns those concerns and presents the stable QuPy provider ABI at the boundary.

## Current boundary

The provider capability target uses the same compact target model as the native hardware compiler: global operation support by arity, undirected connectivity, and optional per-operation durations. Providers that need directed gates, per-edge basis differences, or calibration snapshots can extend the capability schema in a future version while preserving provider ABI compatibility.
