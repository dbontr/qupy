from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Callable

import qupy as qp

from .model import Gate, Workload


class AdapterUnavailable(RuntimeError):
    pass


class WorkloadUnsupported(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PreparedCase:
    engine: str
    engine_version: str
    method: str
    execute: Callable[[], float]
    metadata: dict[str, object]
    validation_tolerance: float | None = None


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"


def _append_qupy_gate(program: qp.Program, gate: Gate) -> qp.Program:
    if gate.name == "h":
        return qp.h(program, gate.qubits[0])
    if gate.name == "x":
        return qp.x(program, gate.qubits[0])
    if gate.name == "y":
        return qp.y(program, gate.qubits[0])
    if gate.name == "z":
        return qp.z(program, gate.qubits[0])
    if gate.name == "rx":
        assert gate.parameter is not None
        return qp.rx(program, gate.parameter, gate.qubits[0])
    if gate.name == "ry":
        assert gate.parameter is not None
        return qp.ry(program, gate.parameter, gate.qubits[0])
    if gate.name == "rz":
        assert gate.parameter is not None
        return qp.rz(program, gate.parameter, gate.qubits[0])
    if gate.name == "cx":
        return qp.cx(program, gate.qubits[0], gate.qubits[1])
    if gate.name == "cz":
        return qp.cz(program, gate.qubits[0], gate.qubits[1])
    if gate.name == "swap":
        return qp.swap(program, gate.qubits[0], gate.qubits[1])
    raise ValueError(f"unsupported QuPy benchmark gate: {gate.name}")


def prepare_qupy(workload: Workload) -> PreparedCase:
    program = qp.Program(workload.num_qubits)
    for gate in workload.gates:
        program = _append_qupy_gate(program, gate)
    plan = qp.expectation_plan(program, qp.Z(workload.observable_qubit))
    metadata: dict[str, object] = {
        "backend": plan.backend,
        "active_qubits": plan.active_qubits,
        "compiled_steps": plan.compiled_steps,
        "estimated_state_bytes": plan.estimated_state_bytes,
        "exact": plan.exact,
        "threads": plan.threads,
        "core_version": qp.core_version(),
        "ir_version": qp.ir_version(),
    }
    return PreparedCase(
        engine="qupy",
        engine_version=qp.core_version(),
        method=plan.method,
        execute=lambda: float(qp.expect(program, qp.Z(workload.observable_qubit)).value),
        metadata=metadata,
    )


def prepare_stim(workload: Workload) -> PreparedCase:
    if not workload.clifford:
        raise WorkloadUnsupported("Stim adapter accepts Clifford workloads only")
    try:
        stim = import_module("stim")
    except ImportError as error:
        raise AdapterUnavailable("Stim is not installed") from error

    circuit = stim.Circuit()
    gate_names = {
        "h": "H",
        "x": "X",
        "y": "Y",
        "z": "Z",
        "cx": "CX",
        "cz": "CZ",
        "swap": "SWAP",
    }
    for gate in workload.gates:
        try:
            stim_name = gate_names[gate.name]
        except KeyError as error:
            raise WorkloadUnsupported(f"Stim adapter does not support {gate.name}") from error
        circuit.append(stim_name, list(gate.qubits))
    observable_text = ["I"] * workload.num_qubits
    observable_text[workload.observable_qubit] = "Z"
    observable = stim.PauliString("".join(observable_text))

    def execute() -> float:
        simulator = stim.TableauSimulator()
        simulator.do(circuit)
        return float(simulator.peek_observable_expectation(observable))

    return PreparedCase(
        engine="stim",
        engine_version=_package_version("stim"),
        method="tableau",
        execute=execute,
        metadata={"exact": True},
    )


def _cirq_operation(cirq: object, qubits: list[object], gate: Gate) -> object:
    q0 = qubits[gate.qubits[0]]
    if gate.name == "h":
        return cirq.H(q0)
    if gate.name == "x":
        return cirq.X(q0)
    if gate.name == "y":
        return cirq.Y(q0)
    if gate.name == "z":
        return cirq.Z(q0)
    if gate.name == "rx":
        assert gate.parameter is not None
        return cirq.rx(gate.parameter)(q0)
    if gate.name == "ry":
        assert gate.parameter is not None
        return cirq.ry(gate.parameter)(q0)
    if gate.name == "rz":
        assert gate.parameter is not None
        return cirq.rz(gate.parameter)(q0)
    q1 = qubits[gate.qubits[1]]
    if gate.name == "cx":
        return cirq.CNOT(q0, q1)
    if gate.name == "cz":
        return cirq.CZ(q0, q1)
    if gate.name == "swap":
        return cirq.SWAP(q0, q1)
    raise WorkloadUnsupported(f"qsim adapter does not support {gate.name}")


def prepare_qsim(workload: Workload) -> PreparedCase:
    if workload.num_qubits > 24:
        raise WorkloadUnsupported("qsim adapter is capped at 24 qubits in the portable harness")
    try:
        cirq = import_module("cirq")
        qsimcirq = import_module("qsimcirq")
    except ImportError as error:
        raise AdapterUnavailable("qsimcirq and Cirq are not installed") from error

    qubits = list(cirq.LineQubit.range(workload.num_qubits))
    circuit = cirq.Circuit(_cirq_operation(cirq, qubits, gate) for gate in workload.gates)
    observable = cirq.Z(qubits[workload.observable_qubit])
    simulator = qsimcirq.QSimSimulator()

    def execute() -> float:
        value = simulator.simulate_expectation_values(circuit, observables=[observable])[0]
        return float(complex(value).real)

    return PreparedCase(
        engine="qsim",
        engine_version=_package_version("qsimcirq"),
        method="qsim-statevector",
        execute=execute,
        metadata={"exact": True, "portable_qubit_cap": 24},
        validation_tolerance=1e-5,
    )


def _append_aer_gate(circuit: object, gate: Gate) -> None:
    q0 = gate.qubits[0]
    if gate.name in {"h", "x", "y", "z"}:
        getattr(circuit, gate.name)(q0)
        return
    if gate.name in {"rx", "ry", "rz"}:
        assert gate.parameter is not None
        getattr(circuit, gate.name)(gate.parameter, q0)
        return
    q1 = gate.qubits[1]
    if gate.name == "cx":
        circuit.cx(q0, q1)
        return
    if gate.name == "cz":
        circuit.cz(q0, q1)
        return
    if gate.name == "swap":
        circuit.swap(q0, q1)
        return
    raise WorkloadUnsupported(f"Aer adapter does not support {gate.name}")


def prepare_aer(workload: Workload, method: str) -> PreparedCase:
    if method == "stabilizer" and not workload.clifford:
        raise WorkloadUnsupported("Aer stabilizer adapter accepts Clifford workloads only")
    if method == "statevector" and workload.num_qubits > 24:
        raise WorkloadUnsupported("Aer statevector adapter is capped at 24 qubits")
    try:
        qiskit = import_module("qiskit")
        quantum_info = import_module("qiskit.quantum_info")
        qiskit_aer = import_module("qiskit_aer")
    except ImportError as error:
        raise AdapterUnavailable("Qiskit Aer is not installed") from error

    circuit = qiskit.QuantumCircuit(workload.num_qubits)
    for gate in workload.gates:
        _append_aer_gate(circuit, gate)
    circuit.save_expectation_value(
        quantum_info.Pauli("Z"),
        [workload.observable_qubit],
        label="expectation",
    )
    simulator = qiskit_aer.AerSimulator(method=method)

    def execute() -> float:
        result = simulator.run(circuit).result()
        value = result.data(0)["expectation"]
        return float(complex(value).real)

    engine_name = f"aer-{method}"
    return PreparedCase(
        engine=engine_name,
        engine_version=_package_version("qiskit-aer"),
        method=method,
        execute=execute,
        metadata={
            "exact": True,
            "portable_qubit_cap": 24 if method == "statevector" else None,
        },
    )


_ADAPTERS: dict[str, Callable[[Workload], PreparedCase]] = {
    "qupy": prepare_qupy,
    "stim": prepare_stim,
    "qsim": prepare_qsim,
    "aer-statevector": lambda workload: prepare_aer(workload, "statevector"),
    "aer-stabilizer": lambda workload: prepare_aer(workload, "stabilizer"),
}


def adapter_names() -> tuple[str, ...]:
    return tuple(_ADAPTERS)


def prepare(engine: str, workload: Workload) -> PreparedCase:
    try:
        factory = _ADAPTERS[engine]
    except KeyError as error:
        raise ValueError(f"unknown benchmark engine: {engine}") from error
    return factory(workload)
