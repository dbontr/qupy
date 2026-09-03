from __future__ import annotations

from pathlib import Path

import pytest

import qupy as qp


def _write_v1(path: Path) -> None:
    lines = (
        "qupy-planner-cost 1",
        f"engine {qp.core_version()}",
        "workload 1",
        f"host {qp.planner_host_fingerprint()}",
        "validated 1",
        "model pauli-propagation 2 5 0.85 1.1 1.2",
        "model statevector-parallel 3 4.5 0.55 0.012 1.1 1.2",
        "model statevector-serial 3 4.5 0.55 0.012 1.1 1.2",
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _program() -> qp.Program:
    return qp.ry(qp.h(qp.Program(2), 0), 0.37, 1)


def test_auto_execution_uses_environment_planner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    artifact = tmp_path / "planner.qpcost"
    _write_v1(artifact)
    monkeypatch.setenv("QUPY_PLANNER_COST_MODEL", str(artifact))
    qp.set_default_planner_cost_model(None)

    execution_plan = qp.expectation_plan(_program(), qp.Z(1))
    assert execution_plan.predicted_ns is not None
    assert execution_plan.cost_model_fingerprint
    assert qp.default_planner_cost_model() is not None


def test_explicit_backend_does_not_resolve_default_planner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("QUPY_PLANNER_COST_MODEL", str(tmp_path / "missing.qpcost"))
    qp.set_default_planner_cost_model(None)

    result = qp.statevector(_program(), backend="native-cpu")
    assert result.backend == "native-cpu"


def test_installed_planner_is_discovered_from_host_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.qpcost"
    _write_v1(source)
    monkeypatch.delenv("QUPY_PLANNER_COST_MODEL", raising=False)
    monkeypatch.setenv("QUPY_CACHE_DIR", str(tmp_path / "cache"))
    qp.set_default_planner_cost_model(None)

    installed = qp.install_planner_cost_model(source)
    assert qp.planner_cache_path().is_file()
    qp.set_default_planner_cost_model(None)
    discovered = qp.default_planner_cost_model()
    assert discovered is not None
    assert discovered.artifact_fingerprint == installed.artifact_fingerprint

    assert qp.remove_planner_cost_model()
    assert not qp.remove_planner_cost_model()
    assert qp.default_planner_cost_model() is None


def test_explicit_model_overrides_invalid_discovery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact = tmp_path / "planner.qpcost"
    _write_v1(artifact)
    model = qp.load_planner_cost_model(str(artifact))
    monkeypatch.setenv("QUPY_PLANNER_COST_MODEL", str(tmp_path / "missing.qpcost"))
    qp.set_default_planner_cost_model(None)

    execution_plan = qp.expectation_plan(_program(), qp.Z(1), cost_model=model)
    assert execution_plan.cost_model_fingerprint == model.artifact_fingerprint
