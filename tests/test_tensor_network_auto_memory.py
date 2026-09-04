from __future__ import annotations

from pathlib import Path

import qupy as qp
from qupy import _native


def _artifact(path: Path) -> Path:
    host = qp.planner_host_fingerprint()
    wide_domain = " ".join("0 100" for _ in range(6))
    text = "\n".join(
        [
            "qupy-tensor-network-cost 1",
            f"engine {qp.core_version()}",
            "workload 1",
            f"host {host}",
            "policy 1",
            "reports 3",
            "decision 18 0 1 9 9",
            "model tensor-network-baseline-cpu 6 12 0 0 0 0 0 1 1",
            "model tensor-network-return-cpu 6 1 0 0 0 0 0 1 1",
            f"domain tensor-network-baseline-cpu 6 1 1 {wide_domain.removeprefix('0 100 ')}",
            f"domain tensor-network-return-cpu 6 1 1 {wide_domain.removeprefix('0 100 ')}",
            "validated 1",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")
    return path


def test_tensor_network_auto_preflight_memory_limit_declines_candidate(tmp_path: Path) -> None:
    model = qp.load_tensor_network_cost_model(_artifact(tmp_path / "policy.qptncost"))
    program = qp.ry(qp.Program(2), 0.17, 0)
    program = qp.cx(program, 0, 1)
    observable = qp.Observable(
        [qp.PauliTerm(1.0, [qp.PauliFactor(1, qp.Pauli.Z)])]
    )

    plan = _native.tensor_network_auto_observable_plan(
        program,
        [observable],
        model,
        8,
    )

    assert plan.backend == "native-cpu"
    assert plan.cost_model_fingerprint == ""
