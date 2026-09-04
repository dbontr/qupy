from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from importlib import import_module
from types import ModuleType
from typing import Protocol, cast

from . import _native
from .circuit import Circuit, CircuitOperationCode
from .compiler import HardwareTarget
from .provider import _hardware_target_payload


class _QiskitJob(Protocol):
    def job_id(self) -> str: ...

    def status(self) -> object: ...

    def result(self) -> object: ...

    def cancel(self) -> object: ...


class _QiskitBackend(Protocol):
    def run(self, run_input: object, *, shots: int) -> _QiskitJob: ...


class _QiskitCircuit(Protocol):
    def h(self, qubit: int) -> object: ...

    def x(self, qubit: int) -> object: ...

    def y(self, qubit: int) -> object: ...

    def z(self, qubit: int) -> object: ...

    def rx(self, angle: float, qubit: int) -> object: ...

    def ry(self, angle: float, qubit: int) -> object: ...

    def rz(self, angle: float, qubit: int) -> object: ...

    def cx(self, first: int, second: int) -> object: ...

    def cz(self, first: int, second: int) -> object: ...

    def swap(self, first: int, second: int) -> object: ...

    def measure(self, qubit: int, clbit: int) -> object: ...

    def barrier(self, *qubits: int) -> object: ...


_STATE_MAP = {
    "INITIALIZING": _native.ProviderJobState.QUEUED,
    "QUEUED": _native.ProviderJobState.QUEUED,
    "VALIDATING": _native.ProviderJobState.QUEUED,
    "RUNNING": _native.ProviderJobState.RUNNING,
    "DONE": _native.ProviderJobState.SUCCEEDED,
    "ERROR": _native.ProviderJobState.FAILED,
    "CANCELLED": _native.ProviderJobState.CANCELLED,
    "CANCELED": _native.ProviderJobState.CANCELLED,
}


def _qiskit_module(name: str) -> ModuleType:
    try:
        return import_module(name)
    except ModuleNotFoundError as exc:
        missing = exc.name
        if missing is not None and (name == missing or name.startswith(f"{missing}.")):
            raise ImportError(
                "Qiskit provider support requires the optional qiskit and qiskit-aer packages"
            ) from exc
        raise


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _options(options_json: str) -> dict[str, object]:
    if not isinstance(options_json, str):
        raise TypeError("options_json must be a string")
    try:
        decoded = json.loads(options_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Qiskit provider options are invalid JSON: {exc.msg}") from exc
    if not isinstance(decoded, dict):
        raise TypeError("Qiskit provider options must be a JSON object")
    payload = cast(dict[str, object], decoded)
    if payload:
        raise ValueError(
            "Qiskit provider options are not yet part of QuPy's stable provider contract"
        )
    return payload


def _backend_name(backend: object) -> str:
    raw = getattr(backend, "name", None)
    if callable(raw):
        raw = raw()
    if isinstance(raw, str) and raw:
        return raw
    return type(backend).__name__


def _status_name(job: _QiskitJob) -> str:
    status = job.status()
    if isinstance(status, str):
        return status.upper()
    name = getattr(status, "name", None)
    if isinstance(name, str):
        return name.upper()
    value = getattr(status, "value", None)
    if isinstance(value, str):
        return value.upper()
    raise RuntimeError("Qiskit returned an unsupported non-string job status")


def _measurement_counts(result: object) -> dict[str, int]:
    getter = getattr(result, "get_counts", None)
    if not callable(getter):
        raise TypeError("Qiskit result does not provide get_counts()")
    raw = getter()
    if not isinstance(raw, Mapping):
        raise TypeError("Qiskit result counts must be a mapping")
    counts: dict[str, int] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise TypeError("Qiskit measurement count keys must be strings")
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("Qiskit measurement counts must be integers")
        if value < 0:
            raise ValueError("Qiskit measurement counts must be non-negative")
        counts[key] = value
    if sum(counts.values()) <= 0:
        raise RuntimeError("Qiskit result contains no measurement shots")
    return counts


def qiskit_aer_target(num_qubits: int = 24) -> HardwareTarget:
    """Return the conservative QuPy target used for Qiskit Aer integration."""
    qubits = _positive_integer(num_qubits, "num_qubits")
    return HardwareTarget(
        "qiskit-aer-simulator",
        qubits,
        [
            CircuitOperationCode.H,
            CircuitOperationCode.X,
            CircuitOperationCode.Y,
            CircuitOperationCode.Z,
            CircuitOperationCode.RX,
            CircuitOperationCode.RY,
            CircuitOperationCode.RZ,
        ],
        [
            CircuitOperationCode.CX,
            CircuitOperationCode.CZ,
            CircuitOperationCode.SWAP,
        ],
        measurement=True,
        mid_circuit_measurement=False,
        reset=False,
        dynamic_control=False,
    )


def _new_quantum_circuit(num_qubits: int, num_clbits: int) -> _QiskitCircuit:
    module = _qiskit_module("qiskit")
    factory = cast(Callable[..., object], module.QuantumCircuit)
    return cast(_QiskitCircuit, factory(num_qubits, num_clbits))


def _append_instructions(target: _QiskitCircuit, circuit: Circuit) -> _QiskitCircuit:
    for instruction in circuit.instructions:
        if instruction.condition is not None:
            raise ValueError("Qiskit provider adapter does not support classical feed-forward")
        code = instruction.code
        qubits = instruction.qubits
        parameters = instruction.parameters
        if code is CircuitOperationCode.H:
            target.h(qubits[0])
        elif code is CircuitOperationCode.X:
            target.x(qubits[0])
        elif code is CircuitOperationCode.Y:
            target.y(qubits[0])
        elif code is CircuitOperationCode.Z:
            target.z(qubits[0])
        elif code is CircuitOperationCode.RX:
            target.rx(parameters[0], qubits[0])
        elif code is CircuitOperationCode.RY:
            target.ry(parameters[0], qubits[0])
        elif code is CircuitOperationCode.RZ:
            target.rz(parameters[0], qubits[0])
        elif code is CircuitOperationCode.CX:
            target.cx(qubits[0], qubits[1])
        elif code is CircuitOperationCode.CZ:
            target.cz(qubits[0], qubits[1])
        elif code is CircuitOperationCode.SWAP:
            target.swap(qubits[0], qubits[1])
        elif code is CircuitOperationCode.MEASURE:
            target.measure(qubits[0], instruction.classical_bits[0])
        elif code is CircuitOperationCode.BARRIER:
            target.barrier(*qubits)
        elif code is CircuitOperationCode.RESET:
            raise ValueError("Qiskit provider adapter does not advertise reset support")
        else:
            raise ValueError(f"unsupported Qiskit provider circuit operation {code!r}")
    return target


def _qiskit_circuit(program: _native.ProviderProgram) -> _QiskitCircuit:
    if program.format != "openqasm3":
        raise ValueError("Qiskit provider requires an openqasm3 provider program")
    circuit = Circuit.from_openqasm3(program.text)
    if circuit.num_qubits != program.num_qubits:
        raise ValueError("provider program qubit metadata does not match its OpenQASM circuit")
    target = _new_quantum_circuit(circuit.num_qubits, circuit.num_clbits)
    return _append_instructions(target, circuit)


class QiskitProvider:
    """First-party QuPy provider adapter for Qiskit Backend-style execution."""

    __slots__ = ("_backend", "_jobs", "_name", "_target")

    def __init__(
        self,
        backend: object,
        *,
        target: HardwareTarget | None = None,
        name: str | None = None,
    ) -> None:
        if backend is None:
            raise TypeError("backend must be a Qiskit backend object")
        if name is not None and (not isinstance(name, str) or not name):
            raise ValueError("name must be a non-empty string when supplied")
        self._backend = cast(_QiskitBackend, backend)
        self._name = name if name is not None else f"qiskit:{_backend_name(backend)}"
        self._target = target
        self._jobs: dict[str, _QiskitJob] = {}

    @classmethod
    def aer_simulator(
        cls,
        *,
        method: str = "automatic",
        num_qubits: int = 24,
    ) -> QiskitProvider:
        if not isinstance(method, str) or not method:
            raise ValueError("method must be a non-empty string")
        module = _qiskit_module("qiskit_aer")
        factory = cast(Callable[..., object], module.AerSimulator)
        backend = factory(method=method)
        return cls(
            backend,
            target=qiskit_aer_target(num_qubits),
            name=f"qiskit-aer:{method}",
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def target(self) -> HardwareTarget | None:
        return self._target

    def capabilities_json(self) -> str:
        payload: dict[str, object] = {"formats": ["openqasm3"]}
        if self._target is not None:
            payload["hardware_target"] = _hardware_target_payload(self._target)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def submit(
        self,
        program: _native.ProviderProgram,
        shots: int,
        options_json: str = "{}",
    ) -> str:
        shot_count = _positive_integer(shots, "shots")
        _options(options_json)
        if self._target is not None and program.num_qubits > self._target.num_qubits:
            raise ValueError("provider program exceeds the configured Qiskit target")
        qiskit_circuit = _qiskit_circuit(program)
        job = self._backend.run(qiskit_circuit, shots=shot_count)
        job_id = job.job_id()
        if not isinstance(job_id, str) or not job_id:
            raise RuntimeError("Qiskit returned an invalid job identifier")
        self._jobs[job_id] = job
        return job_id

    def _job(self, job_id: str) -> _QiskitJob:
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("job_id must be a non-empty string")
        try:
            return self._jobs[job_id]
        except KeyError:
            raise KeyError("Qiskit job is not owned by this provider instance") from None

    def poll(self, job_id: str) -> _native.ProviderJobState:
        state_name = _status_name(self._job(job_id))
        try:
            return _STATE_MAP[state_name]
        except KeyError:
            raise RuntimeError(f"unsupported Qiskit job status {state_name!r}") from None

    def result_json(self, job_id: str) -> str:
        job = self._job(job_id)
        if _status_name(job) != "DONE":
            raise RuntimeError("Qiskit result is unavailable before job completion")
        counts = _measurement_counts(job.result())
        payload: dict[str, object] = {
            "measurement_counts": counts,
            "shots": sum(counts.values()),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def cancel(self, job_id: str) -> None:
        self._job(job_id).cancel()


__all__ = ["QiskitProvider", "qiskit_aer_target"]
