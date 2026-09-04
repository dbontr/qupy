from __future__ import annotations

from pathlib import Path

import pytest

import qupy as qp


def _artifact(
    path: Path,
    *,
    cpu_bias: float,
    tensor_bias: float,
    active_qubits: tuple[float, float] = (0.0, 100.0),
    host: str | None = None,
    include_domains: bool = True,
) -> Path:
    host_fingerprint = qp.planner_host_fingerprint() if host is None else host
    cpu_domain = (
        (1.0, 1.0),
        active_qubits,
        (0.0, 100.0),
        (0.0, 1.0),
        (0.0, 100.0),
        (0.0, 100.0),
    )
    tensor_domain = (
        (1.0, 1.0),
        (0.0, 100.0),
        (0.0, 100.0),
        (0.0, 100.0),
        (0.0, 100.0),
        (0.0, 100.0),
    )
    lines = [
        "qupy-tensor-network-cost 1",
        f"engine {qp.core_version()}",
        "workload 1",
        f"host {host_fingerprint}",
        "policy 1",
        "reports 3",
        "decision 18 0 1 9 9",
        f"model tensor-network-baseline-cpu 6 {cpu_bias} 0 0 0 0 0 1 1",
        f"model tensor-network-return-cpu 6 {tensor_bias} 0 0 0 0 0 1 1",
    ]
    if include_domains:
        cpu_bounds = " ".join(f"{minimum} {maximum}" for minimum, maximum in cpu_domain)
        tensor_bounds = " ".join(
            f"{minimum} {maximum}" for minimum, maximum in tensor_domain
        )
        lines.extend(
            [
                f"domain tensor-network-baseline-cpu 6 {cpu_bounds}",
                f"domain tensor-network-return-cpu 6 {tensor_bounds}",
            ]
        )
    lines.append("validated 1")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _program(qubits: int) -> qp.Program:
    program = qp.Program(qubits)
    for qubit in range(qubits):
        program = qp.ry(program, 0.017 * (qubit + 1), qubit)
    for qubit in range(qubits - 1):
        program = qp.cx(program, qubit, qubit + 1)
    return program


def _observable(qubit: int) -> qp.Observable:
    return qp.Observable([qp.PauliTerm(1.0, [qp.PauliFactor(qubit, qp.Pauli.Z)])])


def _isolate_planner_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("QUPY_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("QUPY_PLANNER_COST_MODEL", raising=False)
    monkeypatch.delenv("QUPY_TENSOR_NETWORK_COST_MODEL", raising=False)
    qp.set_default_planner_cost_model(None)
    qp.set_default_tensor_network_cost_model(None)


def test_tensor_network_model_loads_and_rejects_wrong_host_or_incomplete_domain(
    tmp_path: Path,
) -> None:
    valid = qp.load_tensor_network_cost_model(
        _artifact(tmp_path / "valid.qptncost", cpu_bias=10.0, tensor_bias=9.0)
    )
    assert valid.auto_validated
    assert valid.report_count == 3
    assert valid.decision_samples == 18
    assert valid.decision_mistakes == 0
    assert valid.decision_max_regret == pytest.approx(1.0)
    assert valid.cpu_wins == 9
    assert valid.tensor_network_wins == 9
    assert valid.predict_cpu_ns(8, 20, 7, 15, 1, 1) > 0.0
    assert valid.predict_tensor_network_ns(40, 4, 256, 1000.0, 1) > 0.0

    with pytest.raises(ValueError, match="host does not match"):
        qp.load_tensor_network_cost_model(
            _artifact(
                tmp_path / "wrong-host.qptncost",
                cpu_bias=10.0,
                tensor_bias=9.0,
                host="f" * 64,
            )
        )
    with pytest.raises(ValueError, match="incomplete"):
        qp.load_tensor_network_cost_model(
            _artifact(
                tmp_path / "missing-domain.qptncost",
                cpu_bias=10.0,
                tensor_bias=9.0,
                include_domains=False,
            )
        )


def test_auto_routes_validated_full_cone_observable_to_tensor_network(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _isolate_planner_cache(monkeypatch, tmp_path)
    policy = _artifact(tmp_path / "tn.qptncost", cpu_bias=12.0, tensor_bias=1.0)
    qp.set_default_tensor_network_cost_model(policy)
    try:
        program = _program(8)
        observable = _observable(7)
        selected = qp.observable_plan(program, [observable])
        assert selected.backend == "native-tn"
        assert selected.cost_model_class == "tensor-network-return-cpu"
        assert selected.predicted_ns == pytest.approx(2.718281828459045)
        assert len(selected.cost_model_fingerprint) == 64

        automatic = qp.expect_observable(program, observable)
        reference = qp.expect_observable(program, observable, backend="native-cpu")
        assert automatic.backend == "native-tn"
        assert automatic.value == pytest.approx(reference.value, abs=2e-11)
    finally:
        qp.set_default_tensor_network_cost_model(None)


def test_auto_keeps_cpu_when_tensor_network_model_predicts_cpu_win(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _isolate_planner_cache(monkeypatch, tmp_path)
    policy = _artifact(tmp_path / "cpu.qptncost", cpu_bias=1.0, tensor_bias=12.0)
    qp.set_default_tensor_network_cost_model(policy)
    try:
        program = _program(8)
        observable = _observable(7)
        assert qp.observable_plan(program, [observable]).backend == "native-cpu"
        assert qp.expect_observable(program, observable).backend == "native-cpu"
    finally:
        qp.set_default_tensor_network_cost_model(None)


def test_auto_fails_closed_outside_calibration_domain_and_on_pauli_propagation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _isolate_planner_cache(monkeypatch, tmp_path)
    policy = _artifact(
        tmp_path / "bounded.qptncost",
        cpu_bias=12.0,
        tensor_bias=1.0,
        active_qubits=(8.0, 8.0),
    )
    model = qp.set_default_tensor_network_cost_model(policy)
    assert model is not None
    try:
        with pytest.raises(ValueError, match="outside the calibrated domain"):
            model.predict_cpu_ns(9, 20, 8, 17, 1, 1)

        program = _program(9)
        observable = _observable(8)
        assert qp.observable_plan(program, [observable]).backend == "native-cpu"

        clifford = qp.Program(8)
        clifford = qp.h(clifford, 0)
        for qubit in range(7):
            clifford = qp.cx(clifford, qubit, qubit + 1)
        fast = qp.observable_plan(clifford, [_observable(7)])
        assert fast.backend == "native-cpu"
        assert fast.method == "pauli-propagation"
    finally:
        qp.set_default_tensor_network_cost_model(None)


def test_tensor_network_policy_cache_install_discovery_and_removal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _isolate_planner_cache(monkeypatch, tmp_path)
    source = _artifact(tmp_path / "source.qptncost", cpu_bias=10.0, tensor_bias=9.0)

    installed = qp.install_tensor_network_cost_model(source)
    destination = qp.tensor_network_planner_cache_path()
    assert destination.is_file()
    assert installed.artifact_fingerprint == qp.default_tensor_network_cost_model().artifact_fingerprint
    assert qp.remove_tensor_network_cost_model()
    assert not destination.exists()
    assert qp.default_tensor_network_cost_model() is None
    assert not qp.remove_tensor_network_cost_model()


def test_explicit_backends_do_not_resolve_tensor_network_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _isolate_planner_cache(monkeypatch, tmp_path)
    monkeypatch.setenv(
        "QUPY_TENSOR_NETWORK_COST_MODEL",
        str(tmp_path / "does-not-exist.qptncost"),
    )
    program = _program(8)
    observable = _observable(7)

    assert qp.observable_plan(program, [observable], backend="native-cpu").backend == "native-cpu"
    assert qp.expect_observable(program, observable, backend="native-tn").backend == "native-tn"
