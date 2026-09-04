from __future__ import annotations

import json
import time
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


class _SymbolicParameter:
    pass


@dataclass
class _FakeGate:
    params: list[object]


class _FakeBackendTarget:
    def __init__(
        self,
        num_qubits: int | None,
        qargs: dict[str, set[tuple[int, ...] | None]],
        *,
        parameters: dict[str, list[object]] | None = None,
        angle_bounds: set[str] | None = None,
    ) -> None:
        self.num_qubits = num_qubits
        self.operation_names = tuple(qargs)
        self._qargs = qargs
        self._parameters = {} if parameters is None else parameters
        self._angle_bounds = set() if angle_bounds is None else angle_bounds

    def qargs_for_operation_name(self, operation: str) -> set[tuple[int, ...] | None]:
        return self._qargs[operation]

    def operation_from_name(self, operation: str) -> _FakeGate:
        return _FakeGate(list(self._parameters.get(operation, [])))

    def gate_has_angle_bounds(self, operation: str) -> bool:
        return operation in self._angle_bounds


class _FakeDiscoverableBackend(_FakeBackend):
    def __init__(self, target: _FakeBackendTarget, *, num_qubits: int) -> None:
        super().__init__(_FakeJob(["DONE"]))
        self.target = target
        self.num_qubits = num_qubits


_TERMINAL_STATES = {
    qp.ProviderJobState.SUCCEEDED,
    qp.ProviderJobState.FAILED,
    qp.ProviderJobState.CANCELLED,
}


def _poll_to_terminal(
    provider: qp.QiskitProvider,
    job_id: str,
    *,
    max_polls: int = 100,
) -> qp.ProviderJobState:
    for _ in range(max_polls):
        state = provider.poll(job_id)
        if state in _TERMINAL_STATES:
            return state
        time.sleep(0.01)
    raise AssertionError(f"Qiskit job {job_id!r} did not reach a terminal state")


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


def test_qiskit_optional_sdk_failure_is_precise(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_qiskit(name: str) -> object:
        raise ModuleNotFoundError(f"No module named '{name}'", name=name)

    monkeypatch.setattr(qiskit_provider, "import_module", missing_qiskit)

    with pytest.raises(ImportError, match="qiskit-aer"):
        qp.QiskitProvider.aer_simulator()


def test_qiskit_does_not_mask_transitive_import_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_qiskit(name: str) -> object:
        raise ModuleNotFoundError("No module named 'rustworkx'", name="rustworkx")

    monkeypatch.setattr(qiskit_provider, "import_module", broken_qiskit)

    with pytest.raises(ModuleNotFoundError) as raised:
        qp.QiskitProvider.aer_simulator()
    assert raised.value.name == "rustworkx"


def test_qiskit_backend_target_conservatively_translates_backendv2_metadata() -> None:
    all_qubits = {(qubit,) for qubit in range(4)}
    target = _FakeBackendTarget(
        4,
        {
            "h": all_qubits,
            "x": {None},
            "y": {(0,), (1,)},
            "rx": all_qubits,
            "ry": all_qubits,
            "rz": all_qubits,
            "cx": {(0, 1), (1, 0), (1, 2)},
            "cz": {(0, 1), (1, 2), (2, 3)},
            "swap": {(0, 1)},
            "measure": all_qubits,
            "reset": {(0,), (1,)},
        },
        parameters={
            "rx": [_SymbolicParameter()],
            "ry": [0.5],
            "rz": [_SymbolicParameter()],
        },
        angle_bounds={"rx"},
    )
    backend = _FakeDiscoverableBackend(target, num_qubits=4)

    translated = qp.qiskit_backend_target(backend)

    assert translated.name == "qiskit:fixture-backend"
    assert translated.num_qubits == 4
    assert translated.one_qubit_operations == [
        qp.CircuitOperationCode.H,
        qp.CircuitOperationCode.X,
        qp.CircuitOperationCode.RZ,
    ]
    assert translated.two_qubit_operations == [qp.CircuitOperationCode.CZ]
    assert [(edge.first, edge.second) for edge in translated.couplings] == [
        (0, 1),
        (1, 2),
        (2, 3),
    ]
    assert translated.measurement
    assert not translated.reset
    assert not translated.mid_circuit_measurement
    assert not translated.dynamic_control


def test_qiskit_backend_target_preserves_only_bidirectional_cx_edges() -> None:
    target = _FakeBackendTarget(
        3,
        {
            "cx": {(0, 1), (1, 0), (1, 2)},
            "measure": {None},
        },
    )
    translated = qp.qiskit_backend_target(_FakeDiscoverableBackend(target, num_qubits=3))

    assert translated.two_qubit_operations == [qp.CircuitOperationCode.CX]
    assert [(edge.first, edge.second) for edge in translated.couplings] == [(0, 1)]
    assert translated.supports(qp.CircuitOperationCode.CX, [0, 1])
    assert not translated.supports(qp.CircuitOperationCode.CX, [1, 2])


def test_qiskit_backend_target_global_two_qubit_operation_is_all_to_all() -> None:
    target = _FakeBackendTarget(
        3,
        {
            "x": {None},
            "cz": {None},
            "measure": {None},
        },
    )
    translated = qp.qiskit_backend_target(_FakeDiscoverableBackend(target, num_qubits=3))

    assert translated.two_qubit_operations == [qp.CircuitOperationCode.CZ]
    assert translated.couplings == []
    assert translated.supports(qp.CircuitOperationCode.CZ, [0, 2])


def test_qiskit_provider_auto_advertises_discovered_backend_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _FakeBackendTarget(
        2,
        {
            "h": {None},
            "cz": {None},
            "measure": {None},
        },
    )
    backend = _FakeDiscoverableBackend(target, num_qubits=2)
    provider = qp.QiskitProvider(backend)
    constructed: list[_FakeQuantumCircuit] = []

    def fake_quantum_circuit(num_qubits: int, num_clbits: int) -> _FakeQuantumCircuit:
        circuit = _FakeQuantumCircuit(num_qubits, num_clbits)
        constructed.append(circuit)
        return circuit

    monkeypatch.setattr(qiskit_provider, "_new_quantum_circuit", fake_quantum_circuit)

    capabilities = qp.provider_capabilities(provider)
    assert provider.target is not None
    assert capabilities.hardware_target is not None
    assert capabilities.hardware_target.fingerprint == provider.target.fingerprint

    circuit = qp.Circuit(2, 2).h(0).cz(0, 1).measure(0, 0).measure(1, 1)
    submission = qp.submit_circuit(provider, circuit, 4, initial_layout=[0, 1])
    assert submission.job_id == "qiskit-job-1"
    assert len(constructed) == 1
    assert backend.calls == [(constructed[0], 4)]


def test_qiskit_backend_target_fails_closed_on_malformed_metadata() -> None:
    missing_qubits = _FakeBackend(_FakeJob(["DONE"]))
    with pytest.raises(TypeError, match="num_qubits"):
        qp.qiskit_backend_target(missing_qubits)

    missing_target = _FakeBackend(_FakeJob(["DONE"]))
    missing_target.num_qubits = 2  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="target metadata"):
        qp.qiskit_backend_target(missing_target)

    mismatch = _FakeDiscoverableBackend(
        _FakeBackendTarget(3, {"x": {None}}),
        num_qubits=2,
    )
    with pytest.raises(ValueError, match="disagrees"):
        qp.qiskit_backend_target(mismatch)

    out_of_range = _FakeDiscoverableBackend(
        _FakeBackendTarget(2, {"x": {(0,), (2,)}}),
        num_qubits=2,
    )
    with pytest.raises(ValueError, match="exceed"):
        qp.qiskit_backend_target(out_of_range)


def test_qiskit_backend_target_matches_real_generic_backendv2() -> None:
    fake_provider = pytest.importorskip("qiskit.providers.fake_provider")
    backend = fake_provider.GenericBackendV2(
        num_qubits=3,
        basis_gates=["x", "rz", "cz"],
        coupling_map=[[0, 1], [1, 0], [1, 2], [2, 1]],
        seed=7,
    )

    translated = qp.qiskit_backend_target(backend)
    provider = qp.QiskitProvider(backend)

    assert translated.num_qubits == 3
    assert translated.one_qubit_operations == [
        qp.CircuitOperationCode.X,
        qp.CircuitOperationCode.RZ,
    ]
    assert translated.two_qubit_operations == [qp.CircuitOperationCode.CZ]
    assert [(edge.first, edge.second) for edge in translated.couplings] == [
        (0, 1),
        (1, 2),
    ]
    assert translated.measurement
    assert translated.reset
    assert provider.target is not None
    assert provider.target.fingerprint == translated.fingerprint


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

    assert _poll_to_terminal(provider, submission.job_id) is qp.ProviderJobState.SUCCEEDED
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
    assert _poll_to_terminal(provider, gate_submission.job_id) is qp.ProviderJobState.SUCCEEDED
    gate_result = json.loads(provider.result_json(gate_submission.job_id))
    assert gate_result["shots"] == 8
    assert sum(gate_result["measurement_counts"].values()) == 8

    report = qp.check_provider_conformance(
        provider,
        shots=4,
        max_polls=100,
        poll_interval_seconds=0.01,
    )
    assert report.poll_states[-1] == "succeeded"
    assert report.target_name == "qiskit-aer-simulator"
