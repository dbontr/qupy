from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import qupy as qp


def _base_lines(schema: int) -> list[str]:
    if not qp.cuda_available():
        pytest.skip("density planner policy requires a matching CUDA host")
    lines = [
        f"qupy-planner-cost {schema}",
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
        "model observable-return-cpu 11 10 0 0 0 0 0 0 0 0 0 0 1 1",
        "model observable-return-cuda 7 1 0 0 0 0 0 0 1 1",
        "decision statevector-auto 8 0 1",
        "policy adaptive-mps 1",
        "decision observable-auto 16 0 1",
        "policy rich-observable 1",
        "decision rich-observable-auto 12 0 1",
    ]
    return lines


def _write_v4(path: Path) -> None:
    path.write_text("\n".join(_base_lines(4)) + "\n", encoding="utf-8")


def _write_v5(path: Path, *, cuda_density_faster: bool = True) -> None:
    if cuda_density_faster:
        cpu_bias, cuda_bias, speedup_bias = 0.0, 10.0, 1.0
    else:
        cpu_bias, cuda_bias, speedup_bias = 20.0, 1.0, -1.0
    lines = _base_lines(5)
    lines.extend(
        [
            f"model density-return-cpu 6 {cpu_bias} 0 0 0 0 0 1 1",
            f"model density-return-cuda 7 {cuda_bias} 0 0 0 0 0 0 1 1",
            f"model density-speedup 6 {speedup_bias} 0 0 0 0 0 1 1",
            "policy noisy-density 1",
            "decision noisy-density-auto 24 0 1",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _noisy_program(qubits: int = 4) -> qp.NoisyProgram:
    program = qp.Program(qubits)
    for qubit in range(qubits):
        program = qp.ry(program, 0.031 * (qubit + 1), qubit)
    for qubit in range(qubits - 1):
        program = qp.cx(program, qubit, qubit + 1)
    return qp.NoisyProgram(
        program,
        [
            qp.NoiseInstruction(len(program.operations), qp.amplitude_damping(0, 0.08)),
            qp.NoiseInstruction(len(program.operations), qp.phase_damping(qubits - 1, 0.06)),
        ],
    )


def test_schema_v4_remains_density_inactive(tmp_path: Path) -> None:
    artifact = tmp_path / "planner-v4.qpcost"
    _write_v4(artifact)
    model = qp.load_planner_cost_model(str(artifact))
    assert model.schema_version == 4
    assert model.cuda_auto_validated
    assert model.mps_auto_validated
    assert model.observable_auto_validated
    assert not model.density_auto_validated
    assert model.density_policy_version == 0


def test_schema_v5_routes_noisy_density_to_cuda(tmp_path: Path) -> None:
    artifact = tmp_path / "planner-v5.qpcost"
    _write_v5(artifact, cuda_density_faster=True)
    model = qp.load_planner_cost_model(str(artifact))
    noisy = _noisy_program()

    assert model.schema_version == 5
    assert model.density_auto_validated
    assert model.density_policy_version == 1
    selected = qp.density_matrix(noisy, cost_model=model)
    explicit = qp.density_matrix(noisy, "native-cuda")
    assert selected.backend == "native-cuda-density"
    np.testing.assert_allclose(selected.values, explicit.values, atol=3e-12, rtol=3e-12)


def test_schema_v5_can_keep_noisy_density_on_cpu(tmp_path: Path) -> None:
    artifact = tmp_path / "planner-v5.qpcost"
    _write_v5(artifact, cuda_density_faster=False)
    model = qp.load_planner_cost_model(str(artifact))
    noisy = _noisy_program()

    selected = qp.density_matrix(noisy, cost_model=model)
    explicit = qp.density_matrix(noisy, "native-cpu")
    assert selected.backend == "native-density"
    np.testing.assert_allclose(selected.values, explicit.values, atol=3e-12, rtol=3e-12)


def test_schema_v5_does_not_route_pure_density(tmp_path: Path) -> None:
    artifact = tmp_path / "planner-v5.qpcost"
    _write_v5(artifact, cuda_density_faster=True)
    model = qp.load_planner_cost_model(str(artifact))
    program = qp.ry(qp.Program(4), 0.37, 0)

    selected = qp.density_matrix(program, cost_model=model)
    assert selected.backend == "native-density"


def test_schema_v5_rejects_incomplete_density_evidence(tmp_path: Path) -> None:
    artifact = tmp_path / "planner-v5.qpcost"
    _write_v5(artifact)
    text = artifact.read_text(encoding="utf-8")
    artifact.write_text(
        text.replace("decision noisy-density-auto 24 0 1\n", ""),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="incomplete"):
        qp.load_planner_cost_model(str(artifact))


def test_schema_v5_rejects_negative_additive_density_cuda_coefficients(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "planner-v5.qpcost"
    _write_v5(artifact)
    text = artifact.read_text(encoding="utf-8")
    artifact.write_text(
        text.replace(
            "model density-return-cuda 7 10.0 0 0 0 0 0 0 1 1",
            "model density-return-cuda 7 -1 0 0 0 0 0 0 1 1",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="additive CUDA coefficients"):
        qp.load_planner_cost_model(str(artifact))


def test_schema_v5_empty_noisy_program_stays_cpu(tmp_path: Path) -> None:
    artifact = tmp_path / "planner-v5.qpcost"
    _write_v5(artifact)
    model = qp.load_planner_cost_model(str(artifact))
    noisy = qp.NoisyProgram(qp.ry(qp.Program(2), 0.2, 0), [])
    assert qp.density_matrix(noisy, cost_model=model).backend == "native-density"


def test_schema_v5_requires_paired_speedup_evidence(tmp_path: Path) -> None:
    artifact = tmp_path / "planner-v5.qpcost"
    _write_v5(artifact)
    text = artifact.read_text(encoding="utf-8")
    speedup = next(line for line in text.splitlines() if line.startswith("model density-speedup "))
    artifact.write_text(text.replace(speedup + "\n", ""), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete|invalid model class"):
        qp.load_planner_cost_model(str(artifact))
