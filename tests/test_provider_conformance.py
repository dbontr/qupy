from __future__ import annotations

import json
from typing import cast

import pytest

import qupy as qp


def _capabilities(*, hardware: bool = True, measurement: bool = True) -> str:
    payload: dict[str, object] = {"formats": ["openqasm3"]}
    if hardware:
        durations = {"h": 10.0, "cz": 100.0}
        if measurement:
            durations["measure"] = 50.0
        payload["hardware_target"] = {
            "schema_version": 1,
            "name": "conformance-qpu",
            "num_qubits": 3,
            "one_qubit_operations": ["h", "x", "rz"],
            "two_qubit_operations": ["cz"],
            "couplings": [[0, 1], [1, 2]],
            "measurement": measurement,
            "mid_circuit_measurement": False,
            "reset": False,
            "dynamic_control": False,
            "durations_ns": durations,
        }
    return json.dumps(payload)


class _LifecyclePlugin:
    def __init__(
        self,
        states: list[qp.ProviderJobState],
        *,
        capabilities: str | None = None,
        result_json: str = '{"shots":3,"counts":{"0":3}}',
    ) -> None:
        self.name = "conformance-fixture"
        self._states = states
        self._capabilities = _capabilities() if capabilities is None else capabilities
        self._result_json = result_json
        self.poll_calls = 0
        self.submissions: list[tuple[qp.ProviderProgram, int, str]] = []

    def capabilities_json(self) -> str:
        return self._capabilities

    def submit(
        self,
        program: qp.ProviderProgram,
        shots: int,
        options_json: str = "{}",
    ) -> str:
        self.submissions.append((program, shots, options_json))
        return "secret-provider-job-id"

    def poll(self, job_id: str) -> qp.ProviderJobState:
        assert job_id == "secret-provider-job-id"
        index = min(self.poll_calls, len(self._states) - 1)
        self.poll_calls += 1
        return self._states[index]

    def result_json(self, job_id: str) -> str:
        assert job_id == "secret-provider-job-id"
        return self._result_json

    def cancel(self, job_id: str) -> None:
        assert job_id == "secret-provider-job-id"


def _plugin(fake: _LifecyclePlugin) -> qp.ProviderPlugin:
    return cast(qp.ProviderPlugin, fake)


def test_provider_conformance_exercises_success_lifecycle_without_raw_identifiers() -> None:
    fake = _LifecyclePlugin(
        [
            qp.ProviderJobState.QUEUED,
            qp.ProviderJobState.RUNNING,
            qp.ProviderJobState.SUCCEEDED,
        ]
    )

    report = qp.check_provider_conformance(
        _plugin(fake),
        shots=3,
        max_polls=4,
        poll_interval_seconds=0.0,
    )

    assert report.provider_name == "conformance-fixture"
    assert report.formats == ("openqasm3",)
    assert report.target_name == "conformance-qpu"
    assert len(report.target_fingerprint) == 64
    assert report.shots == 3
    assert report.poll_states == ("queued", "running", "succeeded")
    assert report.result_json_type == "object"
    assert len(report.program_sha256) == 64
    assert len(report.job_id_sha256) == 64
    assert len(report.result_json_sha256) == 64
    assert "secret-provider-job-id" not in report.to_json()

    assert len(fake.submissions) == 1
    program, shots, options = fake.submissions[0]
    assert shots == 3
    assert options == "{}"
    assert program.format == "openqasm3"
    assert "measure q[" in program.text


def test_provider_conformance_accepts_repeated_nonterminal_states() -> None:
    fake = _LifecyclePlugin(
        [
            qp.ProviderJobState.QUEUED,
            qp.ProviderJobState.QUEUED,
            qp.ProviderJobState.RUNNING,
            qp.ProviderJobState.RUNNING,
            qp.ProviderJobState.SUCCEEDED,
        ]
    )

    report = qp.check_provider_conformance(
        _plugin(fake),
        max_polls=5,
        poll_interval_seconds=0.0,
    )
    assert report.poll_states[-1] == "succeeded"
    assert fake.poll_calls == 5


def test_provider_conformance_rejects_state_regression_and_terminal_failure() -> None:
    regressing = _LifecyclePlugin(
        [qp.ProviderJobState.RUNNING, qp.ProviderJobState.QUEUED]
    )
    with pytest.raises(RuntimeError, match="regressed"):
        qp.check_provider_conformance(
            _plugin(regressing),
            max_polls=2,
            poll_interval_seconds=0.0,
        )

    failed = _LifecyclePlugin([qp.ProviderJobState.FAILED])
    with pytest.raises(RuntimeError, match="failed state"):
        qp.check_provider_conformance(_plugin(failed), max_polls=1)

    cancelled = _LifecyclePlugin([qp.ProviderJobState.CANCELLED])
    with pytest.raises(RuntimeError, match="cancelled state"):
        qp.check_provider_conformance(_plugin(cancelled), max_polls=1)


def test_provider_conformance_rejects_timeout_and_invalid_result_json() -> None:
    pending = _LifecyclePlugin([qp.ProviderJobState.QUEUED])
    with pytest.raises(TimeoutError, match="within 2 polls"):
        qp.check_provider_conformance(
            _plugin(pending),
            max_polls=2,
            poll_interval_seconds=0.0,
        )

    invalid_result = _LifecyclePlugin(
        [qp.ProviderJobState.SUCCEEDED],
        result_json="{",
    )
    with pytest.raises(ValueError, match="invalid result JSON"):
        qp.check_provider_conformance(_plugin(invalid_result), max_polls=1)


def test_provider_conformance_supports_explicit_out_of_band_target() -> None:
    fake = _LifecyclePlugin(
        [qp.ProviderJobState.SUCCEEDED],
        capabilities=_capabilities(hardware=False),
    )
    target = qp.HardwareTarget(
        "out-of-band-qpu",
        1,
        [qp.CircuitOperationCode.H],
        [],
        measurement=True,
    )

    report = qp.check_provider_conformance(
        _plugin(fake),
        target=target,
        max_polls=1,
    )
    assert report.target_name == target.name
    assert report.target_fingerprint == target.fingerprint


def test_provider_conformance_validates_local_contract_before_submission() -> None:
    no_hardware = _LifecyclePlugin(
        [qp.ProviderJobState.SUCCEEDED],
        capabilities=_capabilities(hardware=False),
    )
    with pytest.raises(ValueError, match="hardware_target or target"):
        qp.check_provider_conformance(_plugin(no_hardware))
    assert not no_hardware.submissions

    no_measurement = _LifecyclePlugin(
        [qp.ProviderJobState.SUCCEEDED],
        capabilities=_capabilities(measurement=False),
    )
    with pytest.raises(ValueError, match="measurement support"):
        qp.check_provider_conformance(_plugin(no_measurement))
    assert not no_measurement.submissions

    no_qasm_payload = json.loads(_capabilities())
    assert isinstance(no_qasm_payload, dict)
    no_qasm_payload["formats"] = ["qir-base-profile"]
    no_qasm = _LifecyclePlugin(
        [qp.ProviderJobState.SUCCEEDED],
        capabilities=json.dumps(no_qasm_payload),
    )
    with pytest.raises(ValueError, match="openqasm3"):
        qp.check_provider_conformance(_plugin(no_qasm))
    assert not no_qasm.submissions

    valid = _LifecyclePlugin([qp.ProviderJobState.SUCCEEDED])
    with pytest.raises(TypeError, match="shots must be an integer"):
        qp.check_provider_conformance(_plugin(valid), shots=True)
    with pytest.raises(ValueError, match="max_polls must be positive"):
        qp.check_provider_conformance(_plugin(valid), max_polls=0)
    with pytest.raises(ValueError, match="poll_interval_seconds"):
        qp.check_provider_conformance(_plugin(valid), poll_interval_seconds=float("inf"))
    assert not valid.submissions
