from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from . import _native
from .circuit import Circuit, CircuitOperationCode
from .compiler import CompilationResult, Coupling, HardwareTarget, OperationDuration, compile_circuit

_HARDWARE_TARGET_SCHEMA = 1
_OPERATION_BY_NAME = {
    "h": CircuitOperationCode.H,
    "x": CircuitOperationCode.X,
    "y": CircuitOperationCode.Y,
    "z": CircuitOperationCode.Z,
    "rx": CircuitOperationCode.RX,
    "ry": CircuitOperationCode.RY,
    "rz": CircuitOperationCode.RZ,
    "cx": CircuitOperationCode.CX,
    "cz": CircuitOperationCode.CZ,
    "swap": CircuitOperationCode.SWAP,
    "measure": CircuitOperationCode.MEASURE,
    "reset": CircuitOperationCode.RESET,
}


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    formats: tuple[str, ...]
    hardware_target: HardwareTarget | None


@dataclass(frozen=True, slots=True)
class ProviderSubmission:
    job_id: str
    compilation: CompilationResult
    program: _native.ProviderProgram


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return cast(dict[str, object], value)


def _sequence(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    return cast(list[object], value)


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _boolean(value: object, name: str, default: bool = False) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _operation(name: object, field: str) -> CircuitOperationCode:
    operation_name = _string(name, field)
    try:
        return _OPERATION_BY_NAME[operation_name]
    except KeyError:
        raise ValueError(f"{field} contains unsupported operation {operation_name!r}") from None


def _operation_list(value: object, field: str) -> list[CircuitOperationCode]:
    return [_operation(item, field) for item in _sequence(value, field)]


def _couplings(value: object | None) -> list[Coupling]:
    if value is None:
        return []
    result: list[Coupling] = []
    for index, item in enumerate(_sequence(value, "hardware_target.couplings")):
        pair = _sequence(item, f"hardware_target.couplings[{index}]")
        if len(pair) != 2:
            raise ValueError(f"hardware_target.couplings[{index}] must contain two qubits")
        result.append(
            Coupling(
                _integer(pair[0], f"hardware_target.couplings[{index}][0]"),
                _integer(pair[1], f"hardware_target.couplings[{index}][1]"),
            )
        )
    return result


def _durations(value: object | None) -> list[OperationDuration]:
    if value is None:
        return []
    mapping = _mapping(value, "hardware_target.durations_ns")
    result: list[OperationDuration] = []
    for name, duration in mapping.items():
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            raise ValueError(f"hardware_target.durations_ns.{name} must be a number")
        result.append(
            OperationDuration(
                _operation(name, "hardware_target.durations_ns"),
                float(duration),
            )
        )
    return result


def _hardware_target(value: object) -> HardwareTarget:
    target = _mapping(value, "hardware_target")
    schema = _integer(target.get("schema_version"), "hardware_target.schema_version")
    if schema != _HARDWARE_TARGET_SCHEMA:
        raise ValueError(
            f"unsupported hardware_target schema_version {schema}; expected {_HARDWARE_TARGET_SCHEMA}"
        )
    return HardwareTarget(
        _string(target.get("name"), "hardware_target.name"),
        _integer(target.get("num_qubits"), "hardware_target.num_qubits"),
        _operation_list(
            target.get("one_qubit_operations"),
            "hardware_target.one_qubit_operations",
        ),
        _operation_list(
            target.get("two_qubit_operations"),
            "hardware_target.two_qubit_operations",
        ),
        _couplings(target.get("couplings")),
        measurement=_boolean(target.get("measurement"), "hardware_target.measurement"),
        mid_circuit_measurement=_boolean(
            target.get("mid_circuit_measurement"),
            "hardware_target.mid_circuit_measurement",
        ),
        reset=_boolean(target.get("reset"), "hardware_target.reset"),
        dynamic_control=_boolean(
            target.get("dynamic_control"),
            "hardware_target.dynamic_control",
        ),
        durations=_durations(target.get("durations_ns")),
    )


def provider_capabilities(plugin: _native.ProviderPlugin) -> ProviderCapabilities:
    try:
        raw = cast(object, json.loads(plugin.capabilities_json()))
    except json.JSONDecodeError as exc:
        raise ValueError(f"provider returned invalid capability JSON: {exc.msg}") from exc
    payload = _mapping(raw, "provider capabilities")
    formats = tuple(
        _string(item, "provider capabilities.formats")
        for item in _sequence(payload.get("formats"), "provider capabilities.formats")
    )
    target_payload = payload.get("hardware_target")
    target = None if target_payload is None else _hardware_target(target_payload)
    return ProviderCapabilities(formats=formats, hardware_target=target)


def _measures_all(circuit: Circuit) -> bool:
    measured: set[int] = set()
    for instruction in reversed(circuit.instructions):
        if instruction.code is CircuitOperationCode.BARRIER:
            continue
        if (
            instruction.code is CircuitOperationCode.MEASURE
            and instruction.condition is None
        ):
            measured.add(instruction.qubits[0])
            continue
        break
    return len(measured) == circuit.num_qubits


def provider_program(circuit: Circuit) -> _native.ProviderProgram:
    factory = cast(
        Callable[[str, str, int, bool], _native.ProviderProgram],
        getattr(_native, "_make_provider_program"),
    )
    return factory(
        "openqasm3",
        circuit.to_openqasm3(),
        circuit.num_qubits,
        _measures_all(circuit),
    )


def submit_compiled_circuit(
    plugin: _native.ProviderPlugin,
    compilation: CompilationResult,
    shots: int,
    options_json: str = "{}",
) -> ProviderSubmission:
    if shots <= 0:
        raise ValueError("shots must be positive")
    capabilities = provider_capabilities(plugin)
    if "openqasm3" not in capabilities.formats:
        raise ValueError("provider does not advertise openqasm3 program support")
    program = provider_program(compilation.circuit)
    job_id = plugin.submit(program, shots, options_json)
    return ProviderSubmission(job_id=job_id, compilation=compilation, program=program)


def submit_circuit(
    plugin: _native.ProviderPlugin,
    circuit: Circuit,
    shots: int,
    *,
    target: HardwareTarget | None = None,
    initial_layout: list[int] | None = None,
    optimization_level: int = 1,
    options_json: str = "{}",
) -> ProviderSubmission:
    capabilities = provider_capabilities(plugin)
    selected_target = target if target is not None else capabilities.hardware_target
    if selected_target is None:
        raise ValueError(
            "provider does not advertise a hardware_target; pass target= explicitly"
        )
    compilation = compile_circuit(
        circuit,
        selected_target,
        [] if initial_layout is None else initial_layout,
        optimization_level,
    )
    return submit_compiled_circuit(plugin, compilation, shots, options_json)


__all__ = [
    "ProviderCapabilities",
    "ProviderSubmission",
    "provider_capabilities",
    "provider_program",
    "submit_circuit",
    "submit_compiled_circuit",
]
