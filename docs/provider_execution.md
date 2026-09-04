# Provider execution

QuPy keeps vendor credentials, service policy, and remote transport outside the numerical runtime. Providers implement one structural `ProviderBackend` lifecycle. Native shared libraries expose that lifecycle through the stable C provider ABI; first-party Python adapters can implement the same contract without moving vendor SDKs into the core package.

## Execution flow

The hardware path is:

1. Select a provider backend.
2. Read one provider capability snapshot.
3. Parse the optional versioned `hardware_target` object into a native `HardwareTarget`.
4. Compile the logical `Circuit` against that target.
5. Serialize the compiled physical circuit with QuPy's OpenQASM 3.0 provider transport profile.
6. Submit the exact `ProviderProgram` through the provider backend.
7. Poll, retrieve results, or cancel through the same backend.

`qp.submit_circuit()` performs steps 2 through 6. It returns a `ProviderSubmission` containing the provider job identifier, the complete `CompilationResult`, and the exact `ProviderProgram` submitted to the backend.

`Circuit.to_openqasm3()` remains the standalone OpenQASM 3.1 serializer. The generic provider bridge uses the syntax-compatible OpenQASM 3.0 profile because major provider APIs advertise OpenQASM 3.0. QuPy's current hardware-capable subset does not use 3.1-only syntax, so this generic bridge changes only the declared language version. A provider adapter can apply a documented vendor-specific lowering after the generic provider boundary when required by that provider's interface.

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

If the provider advertises a target, an explicitly supplied target must have the same target fingerprint. QuPy rejects a mismatch rather than compiling against constraints that disagree with the provider's capability snapshot.

## Amazon Braket

`BraketProvider` is a first-party Python adapter for Amazon Braket gate-model execution. The Amazon Braket SDK is optional and imported only when a Braket device is constructed or a task is submitted. It is not a `qupy-compute` runtime dependency. If the SDK itself is absent, construction raises `ImportError` with the package name; import failures from dependencies inside an installed SDK are preserved instead of being masked as a missing Braket installation.

Credential-free local execution uses the Braket `LocalSimulator`:

```python
import qupy as qp

provider = qp.BraketProvider.local_simulator(num_qubits=4)
circuit = qp.Circuit(2, 2).h(0).cx(0, 1)
circuit = circuit.measure(0, 0).measure(1, 1)

submission = qp.submit_circuit(
    provider,
    circuit,
    1000,
    initial_layout=[0, 1],
)
print(provider.poll(submission.job_id))
print(provider.result_json(submission.job_id))
```

The generic QuPy `ProviderProgram` remains the canonical OpenQASM 3.0 provider-boundary artifact and contains the standard `include "stdgates.inc";` prelude plus QuPy's `cx` spelling. Before constructing `braket.ir.openqasm.Program`, `BraketProvider` performs exactly two Braket dialect mappings: it removes that canonical include because the Braket interface exposes the supported standard gates directly, and it maps controlled-X calls from `cx` to Braket's `cnot` spelling. The adapter does not alter qubit declarations, gate parameters, other gate calls, measurements, or classical conditions. An unexpected prelude fails closed instead of being rewritten heuristically.

`braket_local_simulator_target()` advertises the QuPy gate subset exercised by the real SDK integration suite, all-to-all connectivity, and terminal measurement. It deliberately does not advertise reset, mid-circuit measurement, or dynamic control. The interoperability test submits H, X, Y, Z, RX, RY, RZ, CX/CNOT, CZ, and SWAP through the actual Braket `LocalSimulator`, so the advertised local gate set is tied to executable vendor evidence rather than inferred from documentation alone.

Cloud execution uses the caller's configured Amazon Braket SDK and AWS credential environment. QuPy does not copy, persist, refresh, or inspect AWS credentials. Device gate sets and topology differ across Braket hardware, so the AWS constructor does not invent a `HardwareTarget`; callers provide a trusted target until QuPy has direct conformance evidence for device-capability translation.

Amazon Braket quantum-task states map to the portable QuPy lifecycle as follows: `CREATED` and `QUEUED` map to queued; `RUNNING` and `CANCELLING` map to running; `COMPLETED` maps to succeeded; and `FAILED`/`CANCELLED` remain terminal failures. Unknown states fail closed. Result retrieval normalizes terminal measurement counts, probabilities, measured qubits, and shot count into deterministic JSON.

The current Braket adapter accepts the empty provider options object only. Vendor-specific task options are not silently forwarded through an unversioned JSON bag.

## Qiskit and Aer

`QiskitProvider` is a first-party adapter for Qiskit Backend-style jobs. Qiskit and Qiskit Aer are optional and lazily imported; neither package is a `qupy-compute` runtime dependency. If an SDK package itself is absent, adapter construction raises `ImportError` with the optional package requirement; import failures from dependencies inside an installed Qiskit stack are preserved rather than being masked as a missing SDK. Credential-free interoperability uses the real `AerSimulator`:

```python
import qupy as qp

provider = qp.QiskitProvider.aer_simulator(num_qubits=4)
circuit = qp.Circuit(2, 2).h(0).cx(0, 1)
circuit = circuit.measure(0, 0).measure(1, 1)

submission = qp.submit_circuit(
    provider,
    circuit,
    1000,
    initial_layout=[0, 1],
)
print(provider.poll(submission.job_id))
print(provider.result_json(submission.job_id))
```

The Qiskit adapter deliberately does not depend on Qiskit's optional OpenQASM importer. The exact QuPy `ProviderProgram` remains the submitted provenance artifact; the adapter parses that OpenQASM 3.0 text back through QuPy's own supported-subset parser and constructs a Qiskit `QuantumCircuit` directly. This keeps the accepted provider syntax identical to QuPy's tested hardware subset instead of delegating parsing or extension semantics to another package.

`qiskit_aer_target()` advertises H, X, Y, Z, RX, RY, RZ, CX, CZ, SWAP, all-to-all connectivity, and terminal measurement. Reset, mid-circuit measurement, and classical feed-forward remain unadvertised and are rejected by the adapter. The provider interoperability workflow submits a Bell circuit, every advertised unitary gate, and the generic QuPy provider-conformance circuit through Qiskit Aer itself.

Qiskit job states map to the portable lifecycle as follows: `INITIALIZING`, `QUEUED`, and `VALIDATING` map to queued; `RUNNING` maps to running; `DONE` maps to succeeded; `ERROR` maps to failed; and both `CANCELLED` and the alternate spelling `CANCELED` map to cancelled. Aer jobs are still treated as asynchronous jobs: callers poll until a terminal state rather than assuming local simulation completes before the first poll. Unknown status objects or names fail closed. Successful result retrieval normalizes measurement counts and shot count into deterministic JSON.

A caller can also wrap another Qiskit Backend-style object directly:

```python
provider = qp.QiskitProvider(backend, target=trusted_target)
```

QuPy does not create, copy, inspect, persist, or refresh IBM Quantum credentials. It also does not infer a hardware target from an arbitrary Qiskit backend yet. Cloud/device backends therefore require a caller-supplied trusted `HardwareTarget` until direct capability translation has its own executable conformance evidence. Vendor-specific run options are likewise not forwarded through an unversioned JSON bag; the current Qiskit adapter accepts the empty options object only.

## Precompiled submission

`qp.submit_compiled_circuit()` accepts an existing `CompilationResult`. This is useful when an application needs to inspect or archive compilation provenance before submission.

The provider must advertise `openqasm3` support. QuPy does not silently switch formats or bypass target validation.

## Provider conformance

`qp.check_provider_conformance()` exercises the portable discovery, compile, submit, poll, and result-retrieval contract against any `ProviderBackend`. It is explicit because it submits a real provider job. Calling it can consume provider quota or incur provider-side cost; credentials, service policy, rate limits, and billing remain the caller's responsibility.

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

The provider interoperability workflow applies this checker to both credential-free first-party integrations: Amazon Braket `LocalSimulator` and Qiskit `AerSimulator`. Those gates prove SDK import, adapter-specific circuit translation, local submission, lifecycle mapping, result normalization, generic QuPy submission, and provider-conformance composition without requiring cloud credentials or QPU quota. They do not certify physical hardware fidelity, cloud SLAs, billing, or credential lifecycle.

For native command-line provider adapters that advertise their target, the packaged wheel installs:

```text
qupy-provider-conformance /path/to/provider-library --shots 1 --max-polls 32 --poll-interval 0.5
```

The report contains provider/target identity, the observed lifecycle states, and SHA-256 identities for the submitted program, provider job identifier, and result JSON. It deliberately does not embed the raw program text, remote job identifier, or provider result payload.

Cancellation remains part of the provider lifecycle and is tested independently; the successful conformance path does not create a cancellation race solely to test that operation.

## Provider ABI stability

The C provider ABI remains version 1 and continues to expose:

- capability discovery
- submit
- poll
- result retrieval
- cancellation
- provider teardown

The structural `ProviderBackend` protocol does not alter that ABI. Native `ProviderPlugin` objects satisfy the same lifecycle used by Python adapters.

The hardware execution bridge does not add credentials, HTTP clients, vendor SDKs, or service-specific policy to QuPy core. A vendor adapter owns those concerns and presents the stable QuPy provider lifecycle at the boundary.

## Current boundary

The provider capability target uses the same compact target model as the native hardware compiler: global operation support by arity, undirected connectivity, and optional per-operation durations. Providers that need directed gates, per-edge basis differences, or calibration snapshots can extend the capability schema in a future version while preserving provider ABI compatibility. First-party local adapters currently provide executable evidence for Amazon Braket and Qiskit Aer; automatic capability translation for vendor cloud hardware remains intentionally outside the validated surface.
