from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum

import pytest

import qupy as qp
from qupy import qiskit_provider


class _FakeStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"


@dataclass
class _FakeResult:
    counts: dict[str, int]

    def get_counts(self) -> dict[str, int]:
        return self.counts


class _FakeJob:
    def __init__(
        self,
        states: list[object],
        result: _FakeResult | None = None,
        *,
        job_id: str = "qiskit-job-1",
    ) -> None:
        self._states = states
        self._state_calls = 0
        self._result = result or _FakeResult({"0": 3, "1": 1})
        self._job_id = job_id
        self.cancelled = False

    def job_id(self) -> str:
        return self._job_id

    def status(self) -> object:
        index = min(self._state_calls, len(self._states) - 1)
        self._state_calls += 1
        return self._states[index]

    def result(self) -> _FakeResult:
        return self._result

    def cancel(self) -> bool:
        self.cancelled = True
        return True


class _FakeBackend:
    name = "fixture-backend"

    def __init__(self, job: _FakeJob) -> None:
        self.job = job
        self.calls: list[tuple[object, int]] = []

    def run(self, run_input: object, *, shots: int) -> _FakeJob:
        self.calls.append((run_input, shots))
        return self.job


class _FakeQuantumCircuit:
    def __init__(self, num_qubits: int, num_clbits: int) -> None:
        self.num_qubits = num_qubits
        self.num_clbits = num_clbits
        self.operations: list[tuple[object, ...]] = []

    def h(self, qubit: int) -> None:
        self.operations.append(("h", qubit))

    def x(self, qubit: int) -> None:
        self.operations.append(("x", qubit))

    def y(self, qubit: int) -> None:
        self.operations.append(("y", qubit))

    def z(self, qubit: int) -> None:
        self.operations.append(("z", qubit))

    def rx(self, angle: float, qubit: int) -> None:
        self.operations.append(("rx", angle, qubit))

    def ry(self, angle: float, qubit: int) -> None:
        self.operations.append(("ry", angle, qubit))

    def rz(self, angle: float, qubit: int) -> None:
        self.operations.append(("rz", angle, qubit))

    def cx(self, first: int, second: int) -> None:
        self.operations.append(("cx", first, second))

    def cz(self, first: int, second: int) -> None:
        self.operations.append(("cz", first, second))

    def swap(self, first: int, second: int) -> None:
        self.operations.append(("swap", first, second))

    def measure(self, qubit: int, clbit: int) -> None:
        self.operations.append(("measure", qubit, clbit))

    def barrier(self, *qubits: int) -> None:
        self.operations.append(("barrier", *qubits))


def _target() -> qp.HardwareTarget:
    return qp.HardwareTarget(
        "fixture-qiskit-target",
        3,
        [
            qp.CircuitOperationCode.H,
            qp.CircuitOperationCode.X,
            qp.CircuitOperationCode.RZ,
        ],
        [qp.CircuitOperationCode.CX],
        measurement=True,
    )


def test_qiskit_capabilities_round_trip_the_configured_target() -> None:
    provider = qp.QiskitProvider(_FakeBackend(_FakeJob(["DONE"])), target=_target())

    capabilities = qp.provider_capabilities(provider)

    assert provider.name == "qiskit:fixture-backend"
    assert provider.target is not None
    assert capabilities.formats == ("openqasm3",)
    assert capabilities.hardware_target is not None
    assert capabilities.hardware_target.fingerprint == _target().fingerprint


def test_qiskit_submission_builds_supported_circuit_and_maps_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _FakeJob(["QUEUED", _FakeStatus.RUNNING, _FakeStatus.DONE])
    backend = _FakeBackend(job)
    provider = qp.QiskitProvider(backend, target=_target())
    constructed: list[_FakeQuantumCircuit] = []

    def fake_quantum_circuit(num_qubits: int, num_clbits: int) -> _FakeQuantumCircuit:
        circuit = _FakeQuantumCircuit(num_qubits, num_clbits)
        constructed.append(circuit)
        return circuit

    monkeypatch.setattr(qiskit_provider, "_new_quantum_circuit", fake_quantum_circuit)

    circuit = qp.Circuit(2, 2).h(0).cx(0, 1).barrier([0, 1]).measure(0, 0).measure(1, 1)
    submission = qp.submit_circuit(
        provider,
        circuit,
        4,
        initial_layout=[0, 1],
        optimization_level=0,
    )

    assert len(constructed) == 1
    qiskit_circuit = constructed[0]
    assert qiskit_circuit.num_qubits == 3
    assert qiskit_circuit.num_clbits == 2
    assert qiskit_circuit.operations == [
        ("h", 0),
        ("cx", 0, 1),
        ("barrier", 0, 1),
        ("measure", 0, 0),
        ("measure", 1, 1),
    ]
    assert backend.calls == [(qiskit_circuit, 4)]
    assert provider.poll(submission.job_id) is qp.ProviderJobState.QUEUED
    assert provider.poll(submission.job_id) is qp.ProviderJobState.RUNNING
    assert provider.poll(submission.job_id) is qp.ProviderJobState.SUCCEEDED

    result = json.loads(provider.result_json(submission.job_id))
    assert result == {"measurement_counts": {"0": 3, "1": 1}, "shots": 4}

    provider.cancel(submission.job_id)
    assert job.cancelled


def test_qiskit_provider_maps_job_states_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        qiskit_provider,
        "_new_quantum_circuit",
        lambda num_qubits, num_clbits: _FakeQuantumCircuit(num_qubits, num_clbits),
    )
    program = qp.provider_program(qp.Circuit(1, 1).measure(0, 0))

    expected = {
        "INITIALIZING": qp.ProviderJobState.QUEUED,
        "QUEUED": qp.ProviderJobState.QUEUED,
        "VALIDATING": qp.ProviderJobState.QUEUED,
        "RUNNING": qp.ProviderJobState.RUNNING,
        "DONE": qp.ProviderJobState.SUCCEEDED,
        "ERROR": qp.ProviderJobState.FAILED,
        "CANCELLED": qp.ProviderJobState.CANCELLED,
        "CANCELED": qp.ProviderJobState.CANCELLED,
    }
    for state, mapped in expected.items():
        provider = qp.QiskitProvider(_FakeBackend(_FakeJob([state])), target=_target())
        job_id = provider.submit(program, 1)
        assert provider.poll(job_id) is mapped

    provider = qp.QiskitProvider(_FakeBackend(_FakeJob(["MYSTERY"])), target=_target())
    job_id = provider.submit(program, 1)
    with pytest.raises(RuntimeError, match="unsupported Qiskit"):
        provider.poll(job_id)

    with pytest.raises(KeyError, match="not owned"):
        provider.poll("unknown-job")


def test_qiskit_provider_validates_options_completion_and_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        qiskit_provider,
        "_new_quantum_circuit",
        lambda num_qubits, num_clbits: _FakeQuantumCircuit(num_qubits, num_clbits),
    )
    program = qp.provider_program(qp.Circuit(1, 1).measure(0, 0))

    pending = qp.QiskitProvider(_FakeBackend(_FakeJob(["RUNNING"])), target=_target())
    with pytest.raises(ValueError, match="not yet part"):
        pending.submit(program, 1, '{"seed":7}')
    with pytest.raises(ValueError, match="invalid JSON"):
        pending.submit(program, 1, "{")

    job_id = pending.submit(program, 1)
    with pytest.raises(RuntimeError, match="before job completion"):
        pending.result_json(job_id)

    malformed = qp.QiskitProvider(
        _FakeBackend(_FakeJob(["DONE"], _FakeResult({"0": -1}))), target=_target()
    )
    malformed_id = malformed.submit(program, 1)
    with pytest.raises(ValueError, match="non-negative"):
        malformed.result_json(malformed_id)


def test_qiskit_provider_rejects_unadvertised_control_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        qiskit_provider,
        "_new_quantum_circuit",
        lambda num_qubits, num_clbits: _FakeQuantumCircuit(num_qubits, num_clbits),
    )
    provider = qp.QiskitProvider(_FakeBackend(_FakeJob(["DONE"])), target=None)

    conditioned = qp.Circuit(1, 1).x(0, qp.ClassicalCondition(0, True))
    with pytest.raises(ValueError, match="classical feed-forward"):
        provider.submit(qp.provider_program(conditioned), 1)

    reset = qp.Circuit(1, 1).reset(0).measure(0, 0)
    with pytest.raises(ValueError, match="reset support"):
        provider.submit(qp.provider_program(reset), 1)


def test_qiskit_provider_satisfies_generic_conformance_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        qiskit_provider,
        "_new_quantum_circuit",
        lambda num_qubits, num_clbits: _FakeQuantumCircuit(num_qubits, num_clbits),
    )
    provider = qp.QiskitProvider(
        _FakeBackend(_FakeJob(["QUEUED", "RUNNING", "DONE"])),
        target=_target(),
    )

    report = qp.check_provider_conformance(
        provider,
        shots=4,
        max_polls=3,
        poll_interval_seconds=0.0,
    )

    assert report.provider_name == "qiskit:fixture-backend"
    assert report.target_fingerprint == _target().fingerprint
    assert report.poll_states == ("queued", "running", "succeeded")
    assert report.result_json_type == "object"


def test_qiskit_aer_target_is_conservative_and_fully_connected() -> None:
    target = qp.qiskit_aer_target(5)

    assert target.num_qubits == 5
    assert target.measurement
    assert not target.mid_circuit_measurement
    assert not target.reset
    assert not target.dynamic_control
    assert target.adjacent(0, 4)
    assert target.supports(qp.CircuitOperationCode.CX, [0, 4])

    with pytest.raises(ValueError, match="positive"):
        qp.qiskit_aer_target(0)


def test_qiskit_aer_real_sdk_integration() -> None:
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")

    provider = qp.QiskitProvider.aer_simulator(num_qubits=4)
    bell = qp.Circuit(2, 2).h(0).cx(0, 1).measure(0, 0).measure(1, 1)
    submission = qp.submit_circuit(
        provider,
        bell,
        64,
        initial_layout=[0, 1],
        optimization_level=0,
    )

    assert provider.poll(submission.job_id) is qp.ProviderJobState.SUCCEEDED
    result = json.loads(provider.result_json(submission.job_id))
    assert result["shots"] == 64
    assert sum(result["measurement_counts"].values()) == 64
    assert set(result["measurement_counts"]) <= {"00", "11"}

    all_gates = qp.Circuit(2, 2)
    all_gates = all_gates.h(0).x(0).y(0).z(0)
    all_gates = all_gates.rx(0.11, 0).ry(-0.23, 0).rz(0.37, 0)
    all_gates = all_gates.cx(0, 1).cz(0, 1).swap(0, 1)
    all_gates = all_gates.measure(0, 0).measure(1, 1)
    gate_submission = qp.submit_circuit(
        provider,
        all_gates,
        8,
        initial_layout=[0, 1],
        optimization_level=0,
    )
    assert provider.poll(gate_submission.job_id) is qp.ProviderJobState.SUCCEEDED
    gate_result = json.loads(provider.result_json(gate_submission.job_id))
    assert gate_result["shots"] == 8
    assert sum(gate_result["measurement_counts"].values()) == 8

    report = qp.check_provider_conformance(
        provider,
        shots=4,
        max_polls=2,
        poll_interval_seconds=0.0,
    )
    assert report.poll_states[-1] == "succeeded"
    assert report.target_name == "qiskit-aer-simulator"
