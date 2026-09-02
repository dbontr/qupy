from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import qupy as qp


def _base_rows() -> tuple[str, ...]:
    return (
        "model pauli-propagation 2 5 0.85 1.1 1.2",
        "model statevector-parallel 3 4.5 0.55 0.012 1.1 1.2",
        "model statevector-serial 3 4.5 0.55 0.012 1.1 1.2",
    )


def _write_v1(path: Path) -> None:
    lines = (
        "qupy-planner-cost 1",
        f"engine {qp.core_version()}",
        "workload 1",
        f"host {qp.planner_host_fingerprint()}",
        "validated 1",
        *_base_rows(),
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_v2(path: Path, *, cuda_faster: bool, decision: str = "8 0 1.0") -> None:
    cpu_coefficients = "0 0 0 0 0" if not cuda_faster else "10 0 0 0 0"
    cuda_coefficients = "10 0 0 0" if not cuda_faster else "0 0 0 0"
    cuda_host = qp.planner_cuda_host_fingerprint() if qp.cuda_available() else "0" * 64
    lines = (
        "qupy-planner-cost 2",
        f"engine {qp.core_version()}",
        "workload 1",
        f"host {qp.planner_host_fingerprint()}",
        f"cuda-host {cuda_host}",
        "validated 1",
        *_base_rows(),
        f"model statevector-return-cpu 5 {cpu_coefficients} 1.1 1.2",
        f"model statevector-return-cuda 4 {cuda_coefficients} 1.1 1.2",
        f"decision statevector-auto {decision}",
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _program(qubits: int = 3) -> qp.Program:
    program = qp.Program(qubits)
    program = qp.h(program, 0)
    for qubit in range(1, qubits):
        program = qp.cx(program, qubit - 1, qubit)
    return qp.ry(program, 0.37, qubits - 1)


def test_v1_artifact_does_not_predict_full_statevector_return(tmp_path: Path) -> None:
    artifact = tmp_path / "planner-v1.qpcost"
    _write_v1(artifact)
    model = qp.load_planner_cost_model(str(artifact))
    assert model.schema_version == 1
    assert model.cuda_host_fingerprint == ""
    assert not model.cuda_auto_validated

    execution_plan = qp.plan(_program(), qp.ResultMode.STATEVECTOR, cost_model=model)
    assert execution_plan.backend == "native-cpu"
    assert execution_plan.method == "statevector"
    assert execution_plan.predicted_ns is None
    assert execution_plan.cost_model_class == ""
    assert execution_plan.cost_model_cuda_host_fingerprint == ""


def test_v2_artifact_fails_closed_without_matching_cuda(tmp_path: Path) -> None:
    artifact = tmp_path / "planner-v2.qpcost"
    _write_v2(artifact, cuda_faster=True)
    if qp.cuda_available():
        model = qp.load_planner_cost_model(str(artifact))
        assert model.cuda_auto_validated
        return
    with pytest.raises(ValueError, match="requires CUDA"):
        qp.load_planner_cost_model(str(artifact))
    with pytest.raises(RuntimeError):
        qp.planner_cuda_host_fingerprint()


def test_v2_artifact_selects_validated_cpu_or_cuda(tmp_path: Path) -> None:
    if not qp.cuda_available():
        pytest.skip("CUDA selection requires a matching GPU host")
    program = _program(4)

    cpu_artifact = tmp_path / "cpu.qpcost"
    _write_v2(cpu_artifact, cuda_faster=False)
    cpu_model = qp.load_planner_cost_model(str(cpu_artifact))
    cpu_plan = qp.plan(program, qp.ResultMode.STATEVECTOR, cost_model=cpu_model)
    assert cpu_plan.backend == "native-cpu"
    assert cpu_plan.method == "statevector"
    assert cpu_plan.predicted_ns == pytest.approx(1.0)
    assert cpu_plan.cost_model_class == "statevector-return-cpu"
    assert cpu_plan.cost_model_cuda_host_fingerprint == qp.planner_cuda_host_fingerprint()

    cuda_artifact = tmp_path / "cuda.qpcost"
    _write_v2(cuda_artifact, cuda_faster=True)
    cuda_model = qp.load_planner_cost_model(str(cuda_artifact))
    cuda_plan = qp.plan(program, qp.ResultMode.STATEVECTOR, cost_model=cuda_model)
    assert cuda_plan.backend == "native-cuda"
    assert cuda_plan.method == "cuda-statevector"
    assert cuda_plan.predicted_ns == pytest.approx(1.0)
    assert cuda_plan.cost_model_class == "statevector-return-cuda"
    selected = qp.statevector(program, cost_model=cuda_model)
    explicit = qp.statevector(program, backend="native-cuda")
    np.testing.assert_allclose(selected.values, explicit.values, atol=2e-12, rtol=2e-12)
    assert selected.backend == "native-cuda"


def test_v2_artifact_rejects_weak_decision_evidence(tmp_path: Path) -> None:
    if not qp.cuda_available():
        pytest.skip("CUDA artifact validation requires a matching GPU host")
    cases = (
        ("7 0 1.0", "decision evidence"),
        ("8 1 1.0", "decision evidence"),
        ("8 0 1.11", "decision evidence"),
    )
    for index, (decision, message) in enumerate(cases):
        artifact = tmp_path / f"weak-{index}.qpcost"
        _write_v2(artifact, cuda_faster=True, decision=decision)
        with pytest.raises(ValueError, match=message):
            qp.load_planner_cost_model(str(artifact))


def test_v2_artifact_rejects_wrong_cuda_host(tmp_path: Path) -> None:
    if not qp.cuda_available():
        pytest.skip("CUDA artifact validation requires a matching GPU host")
    artifact = tmp_path / "wrong-cuda-host.qpcost"
    _write_v2(artifact, cuda_faster=True)
    payload = artifact.read_text(encoding="utf-8").replace(
        qp.planner_cuda_host_fingerprint(), "f" * 64
    )
    artifact.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="CUDA host does not match"):
        qp.load_planner_cost_model(str(artifact))
