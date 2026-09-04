from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

import qupy as qp
import qupy.braket_provider as braket_provider


@dataclass
class _FakeResult:
    measurement_counts: dict[str, int]
    measurement_probabilities: dict[str, float]
    measured_qubits: list[int]


class _FakeTask:
    def __init__(self, states: list[str], result: _FakeResult | None = None) -> None:
        self.id = "braket-task-1"
        self._states = states
        self._state_calls = 0
        self._result = result or _FakeResult(
            {"0": 3, "1": 1},
            {"0": 0.75, "1": 0.25},
            [0],
        )
        self.cancelled = False

    def state(self) -> str:
        index = min(self._state_calls, len(self._states) - 1)
        self._state_calls += 1
        return self._states[index]

    def result(self) -> _FakeResult:
        return self._result

    def cancel(self) -> None:
        self.cancelled = True


class _FakeDevice:
    name = "fixture-device"

    def __init__(self, task: _FakeTask) -> None:
        self.task = task
        self.calls: list[tuple[object, int]] = []

    def run(self, task_specification: object, *, shots: int) -> _FakeTask:
        self.calls.append((task_specification, shots))
        return self.task


def _target() -> qp.HardwareTarget:
    return qp.HardwareTarget(
        "fixture-braket-target",
        3,
        [
            qp.CircuitOperationCode.H,
            qp.CircuitOperationCode.X,
            qp.CircuitOperationCode.RZ,
        ],
        [qp.CircuitOperationCode.CX],
        measurement=True,
    )


def test_braket_capabilities_round_trip_the_configured_target() -> None:
    provider = qp.BraketProvider(_FakeDevice(_FakeTask(["COMPLETED"])), target=_target())

    capabilities = qp.provider_capabilities(provider)

    assert provider.name == "amazon-braket:fixture-device"
    assert provider.target is not None
    assert capabilities.formats == ("openqasm3",)
    assert capabilities.hardware_target is not None
    assert capabilities.hardware_target.fingerprint == _target().fingerprint


def test_braket_submission_lowers_only_the_standard_gate_include(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _FakeTask(["QUEUED", "RUNNING", "COMPLETED"])
    device = _FakeDevice(task)
    provider = qp.BraketProvider(device, target=_target())
    captured_sources: list[str] = []

    def fake_program(source: str) -> object:
        captured_sources.append(source)
        return {"source": source}

    monkeypatch.setattr(braket_provider, "_openqasm_program", fake_program)

    circuit = qp.Circuit(2, 2).h(0).cx(0, 1).measure(0, 0).measure(1, 1)
    submission = qp.submit_circuit(
        provider,
        circuit,
        4,
        initial_layout=[0, 1],
        optimization_level=0,
    )

    assert submission.program.format == "openqasm3"
    assert submission.program.text.startswith(
        'OPENQASM 3.0;\ninclude "stdgates.inc";\n'
    )
    assert circuit.to_openqasm3().startswith("OPENQASM 3.1;\n")
    braket_source = submission.program.text.replace('include "stdgates.inc";\n', "", 1)
    assert captured_sources == [braket_source]
    assert device.calls == [({"source": braket_source}, 4)]
    assert 'include "stdgates.inc";' not in braket_source
    assert braket_source.splitlines()[0] == "OPENQASM 3.0;"
    assert provider.poll(submission.job_id) is qp.ProviderJobState.QUEUED
    assert provider.poll(submission.job_id) is qp.ProviderJobState.RUNNING
    assert provider.poll(submission.job_id) is qp.ProviderJobState.SUCCEEDED

    result = json.loads(provider.result_json(submission.job_id))
    assert result == {
        "measured_qubits": [0],
        "measurement_counts": {"0": 3, "1": 1},
        "measurement_probabilities": {"0": 0.75, "1": 0.25},
        "shots": 4,
    }

    provider.cancel(submission.job_id)
    assert task.cancelled


def test_braket_transport_rejects_noncanonical_provider_prelude() -> None:
    with pytest.raises(ValueError, match="OpenQASM 3.0"):
        braket_provider._braket_openqasm_source("OPENQASM 3.1;\n")
    with pytest.raises(ValueError, match="stdgates"):
        braket_provider._braket_openqasm_source("OPENQASM 3.0;\nqubit[1] q;\n")


def test_braket_provider_maps_cloud_lifecycle_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(braket_provider, "_openqasm_program", lambda source: source)
    program = qp.provider_program(qp.Circuit(1, 1).measure(0, 0))

    expected = {
        "CREATED": qp.ProviderJobState.QUEUED,
        "QUEUED": qp.ProviderJobState.QUEUED,
        "RUNNING": qp.ProviderJobState.RUNNING,
        "CANCELLING": qp.ProviderJobState.RUNNING,
        "COMPLETED": qp.ProviderJobState.SUCCEEDED,
        "FAILED": qp.ProviderJobState.FAILED,
        "CANCELLED": qp.ProviderJobState.CANCELLED,
    }
    for state, mapped in expected.items():
        provider = qp.BraketProvider(_FakeDevice(_FakeTask([state])), target=_target())
        job_id = provider.submit(program, 1)
        assert provider.poll(job_id) is mapped

    provider = qp.BraketProvider(_FakeDevice(_FakeTask(["MYSTERY"])), target=_target())
    job_id = provider.submit(program, 1)
    with pytest.raises(RuntimeError, match="unsupported Amazon Braket"):
        provider.poll(job_id)

    with pytest.raises(KeyError, match="not owned"):
        provider.poll("unknown-task")


def test_braket_provider_validates_options_and_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(braket_provider, "_openqasm_program", lambda source: source)
    task = _FakeTask(["RUNNING"])
    provider = qp.BraketProvider(_FakeDevice(task), target=_target())
    program = qp.provider_program(qp.Circuit(1, 1).measure(0, 0))

    with pytest.raises(ValueError, match="not yet part"):
        provider.submit(program, 1, '{"priority":"normal"}')
    with pytest.raises(ValueError, match="invalid JSON"):
        provider.submit(program, 1, "{")

    job_id = provider.submit(program, 1)
    with pytest.raises(RuntimeError, match="before task completion"):
        provider.result_json(job_id)


def test_braket_provider_satisfies_generic_conformance_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(braket_provider, "_openqasm_program", lambda source: source)
    provider = qp.BraketProvider(
        _FakeDevice(_FakeTask(["QUEUED", "RUNNING", "COMPLETED"])),
        target=_target(),
    )

    report = qp.check_provider_conformance(
        provider,
        shots=4,
        max_polls=3,
        poll_interval_seconds=0.0,
    )

    assert report.provider_name == "amazon-braket:fixture-device"
    assert report.target_fingerprint == _target().fingerprint
    assert report.poll_states == ("queued", "running", "succeeded")
    assert report.result_json_type == "object"


def test_braket_local_simulator_target_is_conservative_and_fully_connected() -> None:
    target = qp.braket_local_simulator_target(5)

    assert target.num_qubits == 5
    assert target.measurement
    assert not target.mid_circuit_measurement
    assert not target.reset
    assert not target.dynamic_control
    assert target.adjacent(0, 4)
    assert target.supports(qp.CircuitOperationCode.CX, [0, 4])

    with pytest.raises(ValueError, match="positive"):
        qp.braket_local_simulator_target(0)


def test_braket_local_simulator_real_sdk_integration() -> None:
    pytest.importorskip("braket")

    provider = qp.BraketProvider.local_simulator(num_qubits=4)
    circuit = qp.Circuit(2, 2).h(0).cx(0, 1).measure(0, 0).measure(1, 1)
    submission = qp.submit_circuit(
        provider,
        circuit,
        64,
        initial_layout=[0, 1],
        optimization_level=0,
    )

    assert provider.poll(submission.job_id) is qp.ProviderJobState.SUCCEEDED
    result = json.loads(provider.result_json(submission.job_id))
    assert result["shots"] == 64
    assert sum(result["measurement_counts"].values()) == 64
    assert set(result["measurement_counts"]) <= {"00", "11"}

    report = qp.check_provider_conformance(
        provider,
        shots=4,
        max_polls=1,
        poll_interval_seconds=0.0,
    )
    assert report.poll_states == ("succeeded",)
    assert report.target_name == "amazon-braket-local-simulator"
