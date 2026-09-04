from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from importlib import import_module
from types import ModuleType
from typing import Protocol, cast

from . import _native
from .circuit import CircuitOperationCode
from .compiler import HardwareTarget
from .provider import _hardware_target_payload


class _BraketTask(Protocol):
    @property
    def id(self) -> str: ...

    def state(self) -> object: ...

    def result(self) -> object: ...

    def cancel(self) -> object: ...


class _BraketDevice(Protocol):
    def run(self, task_specification: object, *, shots: int) -> _BraketTask: ...


_STATE_MAP = {
    "CREATED": _native.ProviderJobState.QUEUED,
    "QUEUED": _native.ProviderJobState.QUEUED,
    "RUNNING": _native.ProviderJobState.RUNNING,
    "CANCELLING": _native.ProviderJobState.RUNNING,
    "COMPLETED": _native.ProviderJobState.SUCCEEDED,
    "FAILED": _native.ProviderJobState.FAILED,
    "CANCELLED": _native.ProviderJobState.CANCELLED,
}
_PROVIDER_HEADER = "OPENQASM 3.0;\n"
_STDGATES_INCLUDE = 'include "stdgates.inc";\n'


def _braket_module(name: str) -> ModuleType:
    try:
        return import_module(name)
    except ImportError as exc:
        raise RuntimeError(
            "Amazon Braket support requires the optional amazon-braket-sdk package"
        ) from exc


def _braket_openqasm_source(source: str) -> str:
    if not source.startswith(_PROVIDER_HEADER):
        raise ValueError("Amazon Braket adapter requires the OpenQASM 3.0 transport profile")
    body = source[len(_PROVIDER_HEADER) :]
    if not body.startswith(_STDGATES_INCLUDE):
        raise ValueError("Amazon Braket adapter expected QuPy's stdgates.inc provider prelude")
    lowered = _PROVIDER_HEADER + body[len(_STDGATES_INCLUDE) :]
    lowered = lowered.replace("\ncx ", "\ncnot ")
    lowered = lowered.replace(") cx ", ") cnot ")
    return lowered


def _openqasm_program(source: str) -> object:
    module = _braket_module("braket.ir.openqasm")
    factory = cast(Callable[..., object], module.Program)
    return factory(source=source)


def _state_name(task: _BraketTask) -> str:
    state = task.state()
    if isinstance(state, str):
        return state.upper()
    value = getattr(state, "value", None)
    if isinstance(value, str):
        return value.upper()
    raise RuntimeError("Amazon Braket returned a non-string quantum task state")


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
        raise ValueError(f"Amazon Braket options are invalid JSON: {exc.msg}") from exc
    if not isinstance(decoded, dict):
        raise TypeError("Amazon Braket options must be a JSON object")
    payload = cast(dict[str, object], decoded)
    if payload:
        raise ValueError(
            "Amazon Braket provider options are not yet part of QuPy's stable provider contract"
        )
    return payload


def _measurement_counts(result: object) -> dict[str, int]:
    raw = getattr(result, "measurement_counts", None)
    if not isinstance(raw, Mapping):
        raise TypeError("Amazon Braket result measurement_counts must be a mapping")
    counts: dict[str, int] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise TypeError("Amazon Braket measurement count keys must be strings")
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("Amazon Braket measurement counts must be integers")
        if value < 0:
            raise ValueError("Amazon Braket measurement counts must be non-negative")
        counts[key] = value
    if sum(counts.values()) <= 0:
        raise RuntimeError("Amazon Braket result contains no measurement shots")
    return counts


def _measurement_probabilities(result: object) -> dict[str, float]:
    raw = getattr(result, "measurement_probabilities", None)
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError("Amazon Braket measurement_probabilities must be a mapping")
    probabilities: dict[str, float] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise TypeError("Amazon Braket probability keys must be strings")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("Amazon Braket probabilities must be numeric")
        probability = float(value)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("Amazon Braket probabilities must be finite and in [0, 1]")
        probabilities[key] = probability
    return probabilities


def _measured_qubits(result: object) -> list[int]:
    raw = getattr(result, "measured_qubits", None)
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        raise TypeError("Amazon Braket measured_qubits must be a sequence")
    qubits: list[int] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("Amazon Braket measured qubits must be integers")
        if value < 0:
            raise ValueError("Amazon Braket measured qubits must be non-negative")
        qubits.append(value)
    return qubits


def braket_local_simulator_target(num_qubits: int = 24) -> HardwareTarget:
    """Return the conservative QuPy gate target used for Braket LocalSimulator integration."""
    qubits = _positive_integer(num_qubits, "num_qubits")
    return HardwareTarget(
        "amazon-braket-local-simulator",
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


class BraketProvider:
    """First-party QuPy provider adapter for Amazon Braket gate-model devices."""

    __slots__ = ("_device", "_name", "_target", "_tasks")

    def __init__(
        self,
        device: object,
        *,
        target: HardwareTarget | None = None,
        name: str | None = None,
    ) -> None:
        if device is None:
            raise TypeError("device must be an Amazon Braket device object")
        if name is not None and (not isinstance(name, str) or not name):
            raise ValueError("name must be a non-empty string when supplied")
        self._device = cast(_BraketDevice, device)
        device_name = getattr(device, "name", None)
        if name is not None:
            self._name = name
        elif isinstance(device_name, str) and device_name:
            self._name = f"amazon-braket:{device_name}"
        else:
            self._name = "amazon-braket"
        self._target = target
        self._tasks: dict[str, _BraketTask] = {}

    @classmethod
    def local_simulator(
        cls,
        *,
        backend: str = "default",
        num_qubits: int = 24,
    ) -> BraketProvider:
        if not isinstance(backend, str) or not backend:
            raise ValueError("backend must be a non-empty string")
        module = _braket_module("braket.devices")
        factory = cast(Callable[..., object], module.LocalSimulator)
        device = factory() if backend == "default" else factory(backend=backend)
        return cls(
            device,
            target=braket_local_simulator_target(num_qubits),
            name=f"amazon-braket:local:{backend}",
        )

    @classmethod
    def aws_device(
        cls,
        device_arn: str,
        *,
        target: HardwareTarget | None = None,
    ) -> BraketProvider:
        if not isinstance(device_arn, str) or not device_arn:
            raise ValueError("device_arn must be a non-empty string")
        module = _braket_module("braket.aws")
        factory = cast(Callable[..., object], module.AwsDevice)
        return cls(factory(device_arn), target=target)

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
        if program.format != "openqasm3":
            raise ValueError("Amazon Braket requires an openqasm3 provider program")
        if self._target is not None and program.num_qubits > self._target.num_qubits:
            raise ValueError("provider program exceeds the configured Amazon Braket target")

        source = _braket_openqasm_source(program.text)
        task = self._device.run(_openqasm_program(source), shots=shot_count)
        job_id = task.id
        if not isinstance(job_id, str) or not job_id:
            raise RuntimeError("Amazon Braket returned an invalid quantum task identifier")
        self._tasks[job_id] = task
        return job_id

    def _task(self, job_id: str) -> _BraketTask:
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("job_id must be a non-empty string")
        try:
            return self._tasks[job_id]
        except KeyError:
            raise KeyError(
                "Amazon Braket task is not owned by this provider instance"
            ) from None

    def poll(self, job_id: str) -> _native.ProviderJobState:
        state_name = _state_name(self._task(job_id))
        try:
            return _STATE_MAP[state_name]
        except KeyError:
            raise RuntimeError(f"unsupported Amazon Braket quantum task state {state_name!r}") from None

    def result_json(self, job_id: str) -> str:
        task = self._task(job_id)
        if _state_name(task) != "COMPLETED":
            raise RuntimeError("Amazon Braket result is unavailable before task completion")
        result = task.result()
        counts = _measurement_counts(result)
        payload: dict[str, object] = {
            "measurement_counts": counts,
            "measurement_probabilities": _measurement_probabilities(result),
            "measured_qubits": _measured_qubits(result),
            "shots": sum(counts.values()),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def cancel(self, job_id: str) -> None:
        self._task(job_id).cancel()


__all__ = ["BraketProvider", "braket_local_simulator_target"]
