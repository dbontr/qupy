from __future__ import annotations

import pytest

import qupy as qp
from benchmarks import mps_cost


def test_measure_pair_counterbalances_backend_order(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_expect(
        program: qp.Program, observable: qp.PauliZ, *, backend: str
    ) -> None:
        del program, observable
        calls.append(backend)

    monkeypatch.setattr(mps_cost.qp, "expect", fake_expect)
    backends = ("native-cpu", "native-adaptive-mps")
    samples = mps_cost._measure_pair(qp.Program(1), qp.Z(0), backends, 2, 4)

    assert calls == [
        "native-cpu",
        "native-adaptive-mps",
        "native-adaptive-mps",
        "native-cpu",
        "native-cpu",
        "native-adaptive-mps",
        "native-adaptive-mps",
        "native-cpu",
        "native-cpu",
        "native-adaptive-mps",
        "native-adaptive-mps",
        "native-cpu",
    ]
    assert len(samples["native-cpu"]) == 4
    assert len(samples["native-adaptive-mps"]) == 4


def test_policy_profile_requires_even_iteration_count() -> None:
    with pytest.raises(ValueError, match="even iteration count"):
        mps_cost.run_report(profile="policy", warmups=0, iterations=3)
