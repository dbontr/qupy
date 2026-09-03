from __future__ import annotations

import json
from typing import cast

import pytest

import qupy as qp


def _capabilities(*, hardware: bool = True, formats: list[str] | None = None) -> str:
    payload: dict[str, object] = {"formats": ["openqasm3"] if formats is None else formats}
    if hardware:
        payload["hardware_target"] = {
            "schema_version": 1,
            "name": "fixture-qpu",
            "num_qubits": 3,
            "one_qubit_operations": ["h", "x", "rz"],
            "two_qubit_operations": ["cz"],
            "couplings": [[0, 1], [1, 2]],
            "measurement": True,
            "mid_circuit_measurement": False,
            "reset": False,
            "dynamic_control": False,
            "durations_ns": {"h": 10.0, "cz": 100.0, "measure": 50.0},
        }
    return json.dumps(payload)


class _FakePlugin:
    def __init__(self, capabilities: str) -> None:
        self._capabilities = capabilities
        self.capability_calls = 0
        self.submissions: list[tuple[qp.ProviderProgram, int, str]] = []

    def capabilities_json(self) -> str:
        self.capability_calls += 1
        return self._capabilities

    def submit(
        self,
        program: qp.ProviderProgram,
        shots: int,
        options_json: str = "{}",
    ) -> str:
        self.submissions.append((program, shots, options_json))
        return "fixture-job-1"


def _plugin(fake: _FakePlugin) -> qp.ProviderPlugin:
    return cast(qp.ProviderPlugin, fake)


def test_provider_capabilities_parses_versioned_hardware_target() -> None:
    fake = _FakePlugin(_capabilities())
    capabilities = qp.provider_capabilities(_plugin(fake))

    assert capabilities.formats == ("openqasm3",)
    assert capabilities.hardware_target is not None
    target = capabilities.hardware_target
    assert target.name == "fixture-qpu"
    assert target.num_qubits == 3
    assert target.measurement
    assert not target.mid_circuit_measurement
    assert target.adjacent(0, 1)
    assert not target.adjacent(0, 2)
    assert target.duration_ns(qp.CircuitOperationCode.CZ) == pytest.approx(100.0)
    assert len(target.fingerprint) == 64
    assert fake.capability_calls == 1


def test_submit_circuit_compiles_serializes_and_submits_once() -> None:
    fake = _FakePlugin(_capabilities())
    circuit = qp.Circuit(2, 2).h(0).cx(0, 1).measure(0, 0).measure(1, 1)

    submission = qp.submit_circuit(
        _plugin(fake),
        circuit,
        128,
        initial_layout=[0, 2],
        optimization_level=0,
        options_json='{"priority":"normal"}',
    )

    assert submission.job_id == "fixture-job-1"
    assert submission.shots == 128
    assert submission.options_json == '{"priority":"normal"}'
    assert fake.capability_calls == 1
    assert len(fake.submissions) == 1
    program, shots, options = fake.submissions[0]
    assert shots == submission.shots
    assert options == submission.options_json
    assert program.format == "openqasm3"
    assert program.text == submission.compilation.circuit.to_openqasm3()
    assert program.text.startswith("OPENQASM 3.1;")
    assert "qubit[3] q;" in program.text
    assert "cz q[" in program.text
    assert "measure q[" in program.text
    assert "swap " not in program.text
    assert submission.compilation.inserted_swaps == 1
    assert submission.compilation.decompositions > 0
    assert not program.measures_all


def test_submit_compiled_circuit_rejects_missing_format_and_bad_shots() -> None:
    target = qp.HardwareTarget(
        "explicit",
        1,
        [qp.CircuitOperationCode.H],
        [],
    )
    compilation = qp.compile(qp.Circuit(1).h(0), target, initial_layout=[0])

    no_qasm = _FakePlugin(_capabilities(hardware=False, formats=["qir-base-profile"]))
    with pytest.raises(ValueError, match="openqasm3"):
        qp.submit_compiled_circuit(_plugin(no_qasm), compilation, 32)
    assert not no_qasm.submissions

    valid = _FakePlugin(_capabilities(hardware=False))
    with pytest.raises(ValueError, match="shots must be positive"):
        qp.submit_compiled_circuit(_plugin(valid), compilation, 0)
    with pytest.raises(ValueError, match="shots must be an integer"):
        qp.submit_compiled_circuit(_plugin(valid), compilation, True)
    assert not valid.submissions


def test_submit_compiled_circuit_rejects_provider_target_mismatch() -> None:
    fake = _FakePlugin(_capabilities())
    conflicting = qp.HardwareTarget(
        "other-qpu",
        2,
        [qp.CircuitOperationCode.H],
        [qp.CircuitOperationCode.CX],
    )
    compilation = qp.compile(
        qp.Circuit(2).h(0).cx(0, 1),
        conflicting,
        initial_layout=[0, 1],
    )

    with pytest.raises(ValueError, match="compiled target does not match"):
        qp.submit_compiled_circuit(_plugin(fake), compilation, 64)
    assert fake.capability_calls == 1
    assert not fake.submissions


def test_submit_circuit_fails_closed_on_target_mismatch() -> None:
    fake = _FakePlugin(_capabilities())
    conflicting = qp.HardwareTarget(
        "other-qpu",
        2,
        [qp.CircuitOperationCode.H],
        [qp.CircuitOperationCode.CX],
    )

    with pytest.raises(ValueError, match="does not match"):
        qp.submit_circuit(
            _plugin(fake),
            qp.Circuit(1).h(0),
            16,
            target=conflicting,
        )
    assert fake.capability_calls == 1
    assert not fake.submissions


def test_submit_circuit_accepts_explicit_target_when_provider_has_none() -> None:
    fake = _FakePlugin(_capabilities(hardware=False))
    target = qp.HardwareTarget(
        "out-of-band",
        1,
        [qp.CircuitOperationCode.H],
        [],
    )

    submission = qp.submit_circuit(
        _plugin(fake),
        qp.Circuit(1).h(0),
        8,
        target=target,
        initial_layout=[0],
    )
    assert submission.job_id == "fixture-job-1"
    assert submission.compilation.target_fingerprint == target.fingerprint


def test_provider_capabilities_rejects_invalid_contracts() -> None:
    with pytest.raises(ValueError, match="invalid capability JSON"):
        qp.provider_capabilities(_plugin(_FakePlugin("{")))

    wrong_schema = json.loads(_capabilities())
    assert isinstance(wrong_schema, dict)
    hardware = wrong_schema["hardware_target"]
    assert isinstance(hardware, dict)
    hardware["schema_version"] = 2
    with pytest.raises(ValueError, match="schema_version"):
        qp.provider_capabilities(_plugin(_FakePlugin(json.dumps(wrong_schema))))

    negative_qubits = json.loads(_capabilities())
    assert isinstance(negative_qubits, dict)
    hardware = negative_qubits["hardware_target"]
    assert isinstance(hardware, dict)
    hardware["num_qubits"] = -1
    with pytest.raises(ValueError, match="num_qubits must be positive"):
        qp.provider_capabilities(_plugin(_FakePlugin(json.dumps(negative_qubits))))
