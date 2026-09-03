from __future__ import annotations

import pytest

import qupy as qp
from benchmarks.density_cost import policy_profile, run_report, smoke_profile


def test_density_policy_profile_has_broad_unique_coverage() -> None:
    workloads = policy_profile()
    assert len(workloads) == 36
    assert len({workload.name for workload in workloads}) == len(workloads)
    assert {workload.qubits for workload in workloads} == set(range(4, 10))
    assert {workload.variant for workload in workloads} == {"chain", "ladder"}
    assert {workload.noise_profile for workload in workloads} == {
        "sparse",
        "mixed",
        "dense",
    }
    assert len(smoke_profile()) == 2


def test_density_policy_requires_even_positive_timing_count() -> None:
    with pytest.raises(ValueError, match="positive even iteration count"):
        run_report(profile="policy", warmups=1, iterations=3)


def test_density_smoke_report_matches_cpu_and_cuda() -> None:
    if not qp.cuda_available():
        pytest.skip(qp.cuda_unavailable_reason())
    report = run_report(profile="smoke", warmups=0, iterations=2)
    assert report["schema_version"] == 1
    assert report["policy_version"] == 1
    assert len(report["policy_evidence"]) == 2
    assert all(
        row["max_abs_error"] <= 2e-11 for row in report["validations"]
    )


def test_density_workload_identity_includes_noise_semantics() -> None:
    from benchmarks.density_cost import DensityWorkload, _identity, _work

    workload = DensityWorkload("identity", 2, "chain", "sparse")
    program = qp.h(qp.Program(2), 0)
    first = qp.NoisyProgram(
        program, [qp.NoiseInstruction(1, qp.amplitude_damping(0, 0.1))]
    )
    second = qp.NoisyProgram(
        program, [qp.NoiseInstruction(1, qp.amplitude_damping(0, 0.2))]
    )
    assert _work(first) == _work(second)
    assert _identity(workload, first, _work(first)) != _identity(
        workload, second, _work(second)
    )
