# OpenQASM 3 interchange

QuPy supports bidirectional OpenQASM 3 interchange for the operation set represented by its hardware-capable `Circuit` IR.

## Round-trip

`Circuit.to_openqasm3()` serializes a circuit as OpenQASM 3.1. `Circuit.from_openqasm3()` parses the supported subset back into the immutable circuit model.

```python
import qupy as qp

circuit = qp.Circuit(2, 1)
circuit = circuit.h(0).cx(0, 1)
circuit = circuit.measure(0, 0)
circuit = circuit.x(1, qp.ClassicalCondition(0, True))

text = circuit.to_openqasm3()
restored = qp.Circuit.from_openqasm3(text)

assert restored.canonical_text == circuit.canonical_text
assert restored.fingerprint == circuit.fingerprint
```

Round-trip identity is semantic. Input whitespace, comments, and register names do not become part of QuPy's circuit identity. The immutable `Circuit` canonical text and SHA-256 fingerprint remain the identity boundary.

## Supported syntax

The importer accepts OpenQASM 3, 3.0, and 3.1 source using one quantum register and at most one classical bit register. Register identifiers do not need to be `q` and `c`.

Supported instructions are:

- `h`, `x`, `y`, `z`
- `rx`, `ry`, `rz` with one finite numeric literal
- `cx`, `cz`, `swap`
- indexed measurement assignment such as `c[0] = measure q[0];`
- top-level whole-register measurement such as `c = measure q;` when the quantum and classical register sizes are equal
- `reset`
- barriers with zero or more explicitly indexed qubits
- one supported instruction inside `if (c[index] == 0)` or `if (c[index] == 1)`

Whole-register measurement is lowered deterministically to one indexed `Circuit.measure()` instruction per qubit/classical-bit pair in ascending index order. Unequal register sizes are rejected rather than truncated or broadcast implicitly. This also makes the OpenQASM emitted by `to_openqasm3(program, measure_all=True)` directly importable as a `Circuit`.

Conditional whole-register measurement is intentionally rejected. The current `Circuit` IR stores a condition on each instruction, so expanding one conditional register measurement into multiple condition-bearing measurement instructions could re-evaluate a condition after an earlier measurement changes classical state. QuPy fails instead of changing that block-level semantic.

Both `//` line comments and `/* ... */` block comments are ignored. Decimal and scientific-notation numeric literals are accepted for rotation parameters.

`include "stdgates.inc";` is recognized. Other include files are rejected because QuPy does not resolve or execute external OpenQASM definitions.

## Deliberate boundary

`from_openqasm3()` is not a claim to implement the complete OpenQASM language. It is a strict parser for the subset QuPy can represent without changing circuit semantics.

The importer rejects unsupported constructs rather than dropping or rewriting them. This includes:

- additional quantum or classical registers
- custom gate definitions
- unsupported standard gates outside QuPy's current circuit operation set
- loops and other control-flow forms
- nested or multi-instruction conditional blocks
- conditional barriers, which the current Circuit IR does not represent
- conditional whole-register measurement
- symbolic or arithmetic parameter expressions
- calibration, delay, pulse, and timing constructs
- invalid or out-of-range qubit/classical references
- whole-register measurements with mismatched quantum/classical register sizes
- non-finite numeric parameters

Syntax failures include source line and column information. Existing `Circuit` validation remains responsible for semantic constraints such as distinct two-qubit operands and valid condition bits.

## Provider interchange

`Circuit.to_openqasm3()` remains the standalone OpenQASM 3.1 representation. `provider_program()` uses the same supported syntax with an OpenQASM 3.0 language header because provider interfaces such as Amazon Braket advertise OpenQASM 3.0. QuPy's current provider subset does not use syntax whose meaning differs between those language versions; only the declared language version changes at this generic boundary.

`ProviderSubmission.program` stores the exact text delivered from QuPy's generic provider layer to the selected provider backend. That text is the provider-boundary identity used by generic conformance and provenance checks. A provider adapter may perform a documented vendor-dialect lowering after this boundary when the vendor accepts equivalent OpenQASM semantics through a different prelude.

Amazon Braket is one such case. QuPy's generic provider text contains `include "stdgates.inc";`, while Braket's OpenQASM interface exposes the same supported standard gates without requiring that include. `BraketProvider` therefore removes exactly the canonical `stdgates.inc` prelude before constructing `braket.ir.openqasm.Program`; it does not modify circuit declarations, gate calls, parameters, measurements, or classical conditions. Unexpected provider preludes fail closed instead of being rewritten heuristically.

Both the generic 3.0 provider payload and standalone 3.1 text can be parsed back into `Circuit`, then inspected, compiled, fingerprinted, or re-serialized. The importer does not trust provider text. Unsupported or malformed input fails closed through the same parser boundary.

## Expanding the subset

New OpenQASM syntax should only be accepted when QuPy's `Circuit` IR can preserve its semantics. Adding a parser rule without a corresponding circuit representation would make round-trip behavior ambiguous and is intentionally avoided.
