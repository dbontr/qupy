# Hardware compilation

QuPy compiles `Circuit` objects against explicit `HardwareTarget` constraints before provider submission. Compilation is deterministic for the same circuit, target, initial layout, and optimization level.

## Target model

A hardware target declares:

- physical qubit count
- native one-qubit operations
- native two-qubit operations
- an undirected coupling graph, or all-to-all connectivity when no edges are supplied
- terminal measurement support
- mid-circuit measurement support
- reset support
- classical feed-forward support
- optional operation durations in nanoseconds

The target has versioned canonical text and a SHA-256 fingerprint. The fingerprint changes when any semantic compilation constraint changes.

```python
import qupy as qp

target = qp.HardwareTarget(
    "example-qpu",
    5,
    [qp.CircuitOperationCode.H, qp.CircuitOperationCode.RZ],
    [qp.CircuitOperationCode.CZ],
    [
        qp.Coupling(0, 1),
        qp.Coupling(1, 2),
        qp.Coupling(2, 3),
        qp.Coupling(3, 4),
    ],
    measurement=True,
    mid_circuit_measurement=True,
    reset=True,
    dynamic_control=True,
)
```

## Compiler stages

`qp.compile()` applies these stages:

1. Logical optimization removes adjacent inverse operations and merges same-axis rotations. Optimization level 2 can commute unconditional unitary operations across disjoint qubits to expose additional reductions.
2. Layout maps logical qubits to physical qubits. With no explicit layout, interacting logical qubits and highly connected physical qubits are ranked deterministically by degree.
3. Routing inserts shortest-path SWAP operations for interactions that are not adjacent on the target coupling graph.
4. Basis translation replaces unsupported entanglers when an exact decomposition exists. Current exact translations include CX through H-CZ-H, CZ through H-CX-H, and SWAP through three CX operations.
5. Depth analysis respects qubit dependencies and classical feed-forward dependencies.
6. Scheduling emits an ASAP schedule when the target supplies durations for every non-barrier compiled instruction.

Compilation fails instead of changing circuit semantics when the target cannot represent a required operation or control capability.

```python
circuit = qp.Circuit(3, 1)
circuit = circuit.h(0).cx(0, 2)
circuit = circuit.measure(0, 0)
circuit = circuit.x(1, qp.ClassicalCondition(0, True))

compiled = qp.compile(circuit, target, optimization_level=2)
print(compiled.initial_layout)
print(compiled.final_layout)
print(compiled.inserted_swaps)
print(compiled.depth)
print(compiled.circuit.to_openqasm3())
```

## Conditional routing

A classically conditioned interaction cannot permanently update the compiler's logical-to-physical map because the operation might not execute. For a conditioned long-range interaction, QuPy emits conditioned routing SWAPs, the conditioned operation, and conditioned reverse SWAPs. This restores the same physical mapping on both control-flow outcomes.

Unconditional routing can retain the changed placement and records that placement in `final_layout`.

## Measurement capability

Terminal measurement and mid-circuit measurement are separate target capabilities. A target can accept a final measurement suffix without claiming that it can measure and then continue quantum execution.

A measurement requires mid-circuit capability when a later instruction performs quantum work, reset, or classically controlled work. Consecutive terminal measurements and barriers do not require mid-circuit capability.

## Compilation report

`CompilationResult` contains:

- compiled physical `Circuit`
- initial and final logical-to-physical layouts
- original, optimized, routed, and compiled operation counts
- inserted routing SWAP count
- decomposition count
- logical dependency depth
- optional scheduled duration
- optional instruction schedule
- target fingerprint

The compiled circuit uses the hardware target's physical qubit width so physical indices remain explicit for provider interchange.

## Current boundary

The first hardware compiler intentionally keeps a compact target model: operation support is global by arity and the coupling graph is undirected. Operation durations are per operation code rather than per qubit or edge. Provider-specific calibration can extend this boundary when a concrete backend requires directed gates, per-edge basis differences, or time-dependent calibration data.
