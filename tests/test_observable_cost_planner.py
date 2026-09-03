from __future__ import annotations

from pathlib import Path

import pytest

import qupy as qp


def _write_v4(path: Path, *, cuda_observable_faster: bool = True) -> None:
    if not qp.cuda_available():
        pytest.skip("schema-v4 observable policy requires a matching CUDA host")
    cpu_bias, cuda_bias = ((10.0, 1.0) if cuda_observable_faster else (1.0, 10.0))
    lines = [
        "qupy-planner-cost 4",
        f"engine {qp.core_version()}",
        "workload 1",
        f"host {qp.planner_host_fingerprint()}",
        f"cuda-host {qp.planner_cuda_host_fingerprint()}",
        "validated 1",
        "model pauli-propagation 2 5 0.85 1 1",
        "model statevector-parallel 3 4.5 0.55 0.012 1 1",
        "model statevector-serial 3 4.5 0.55 0.012 1 1",
        "model statevector-return-cpu 5 1 0 0 0 0 1 1",
        "model statevector-return-cuda 4 20 0 0 0 1 1",
    ]
    lines.extend(
        [
            f"model observable-return-cpu 11 {cpu_bias} 0 0 0 0 0 0 0 0 0 0 1 1",
            f"model observable-return-cuda 7 {cuda_bias} 0 0 0 0 0 0 1 1",
            "decision statevector-auto 8 0 1",
            "policy adaptive-mps 1",
            "decision observable-auto 16 0 1",
            "policy rich-observable 1",
            "decision rich-observable-auto 12 0 1",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _workload() -> tuple[qp.Program, qp.Observable]:
    program = qp.Program(6)
    for qubit in range(6):
        program = qp.ry(program, 0.071 * (qubit + 1), qubit)
    for qubit in range(5):
        program = qp.cx(program, qubit, qubit + 1)
    observable = qp.Observable(
        [
            qp.PauliTerm(0.7, [qp.PauliFactor(0, qp.Pauli.X), qp.PauliFactor(5, qp.Pauli.Y)]),
            qp.PauliTerm(-0.2, [qp.PauliFactor(2, qp.Pauli.Z)]),
        ]
    )
    return program, observable

def test_schema_v4_observable_policy_overrides_statevector_proxy(tmp_path: Path) -> None:
    artifact = tmp_path / "observable-policy.qpcost"
    _write_v4(artifact, cuda_observable_faster=True)
    model = qp.load_planner_cost_model(str(artifact))
    program, observable = _workload()

    assert model.schema_version == 4
    assert model.observable_auto_validated
    assert model.observable_policy_version == 1
    assert model.mps_auto_validated
    assert model.cuda_auto_validated

    state_plan = qp.plan(program, qp.ResultMode.STATEVECTOR, cost_model=model)
    assert state_plan.backend == "native-cpu"
    plan = qp.observable_plan(program, [observable], cost_model=model)
    assert plan.backend == "native-cuda"
    assert plan.method == "cuda-pauli-reduction"
    assert plan.cost_model_class == "observable-return-cuda"
    assert plan.cost_model_fingerprint == model.artifact_fingerprint
    assert plan.predicted_ns is not None

    selected = qp.expect_observable(program, observable, cost_model=model)
    cpu = qp.expect_observable(program, observable, backend="native-cpu")
    assert selected.backend == "native-cuda"
    assert selected.value == pytest.approx(cpu.value, abs=3e-12)

def test_schema_v4_can_select_cpu_for_rich_observable(tmp_path: Path) -> None:
    artifact = tmp_path / "observable-policy.qpcost"
    _write_v4(artifact, cuda_observable_faster=False)
    model = qp.load_planner_cost_model(str(artifact))
    program, observable = _workload()

    plan = qp.observable_plan(program, [observable], cost_model=model)
    assert plan.backend == "native-cpu"
    assert plan.method == "statevector-observable"
    assert plan.cost_model_class == "observable-return-cpu"
    selected = qp.expect_observable(program, observable, cost_model=model)
    explicit = qp.expect_observable(program, observable, backend="native-cpu")
    assert selected.backend == "native-cpu"
    assert selected.value == pytest.approx(explicit.value, abs=3e-12)


def test_schema_v4_rejects_incomplete_observable_evidence(tmp_path: Path) -> None:
    artifact = tmp_path / "observable-policy.qpcost"
    _write_v4(artifact)
    text = artifact.read_text(encoding="utf-8")
    artifact.write_text(
        text.replace("decision rich-observable-auto 12 0 1\n", ""),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="incomplete"):
        qp.load_planner_cost_model(str(artifact))


def test_schema_v4_rejects_negative_additive_cuda_coefficients(tmp_path: Path) -> None:
    artifact = tmp_path / "observable-policy.qpcost"
    _write_v4(artifact)
    text = artifact.read_text(encoding="utf-8")
    artifact.write_text(
        text.replace(
            "model observable-return-cuda 7 1.0 0 0 0 0 0 0 1 1",
            "model observable-return-cuda 7 -1 0 0 0 0 0 0 1 1",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="additive CUDA coefficients"):
        qp.load_planner_cost_model(str(artifact))


def test_schema_v4_auto_skips_cuda_above_device_capacity(tmp_path: Path) -> None:
    if not qp.cuda_available():
        pytest.skip("schema-v4 observable policy requires a matching CUDA host")
    max_qubits = qp.cuda_target().max_qubits
    if max_qubits is None:
        pytest.skip("CUDA target does not expose a finite capacity")

    qubits = max_qubits + 1
    program = qp.ry(qp.Program(qubits), 0.371, 0)
    for qubit in range(qubits - 1):
        program = qp.cx(program, qubit, qubit + 1)
    observable = qp.Observable(
        [qp.PauliTerm(1.0, [qp.PauliFactor(qubits - 1, qp.Pauli.Z)])]
    )
    artifact = tmp_path / "observable-policy.qpcost"
    _write_v4(artifact)
    model = qp.load_planner_cost_model(str(artifact))

    plan = qp.observable_plan(program, [observable], cost_model=model)
    assert plan.backend == "native-cpu"
    assert plan.cost_model_class == "observable-return-cpu"
