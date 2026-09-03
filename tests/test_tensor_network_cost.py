from __future__ import annotations

from benchmarks.tensor_network_cost import run_report


def test_tensor_network_smoke_report_is_paired_and_exact() -> None:
    report = run_report(profile="smoke", warmups=0, iterations=2)

    assert report["schema_version"] == 1
    assert report["policy_version"] == 1
    assert report["profile"] == "smoke"
    assert len(report["validations"]) == 2
    assert len(report["policy_evidence"]) == 2
    assert all(row["max_abs_error"] <= 2e-11 for row in report["validations"])

    for row in report["policy_evidence"]:
        assert len(row["fingerprint"]) == 64
        assert len(row["tn_plan_fingerprint"]) == 64
        assert row["term_count"] > 0
        assert row["tn_contractions"] > 0
        assert row["tn_peak_tensor_rank"] > 0
        assert row["tn_peak_tensor_bytes"] > 0
        assert row["tn_scalar_multiplications"] > 0
        assert len(row["cpu_timings_ns"]) == 2
        assert len(row["tn_timings_ns"]) == 2
        assert all(sample > 0 for sample in row["cpu_timings_ns"])
        assert all(sample > 0 for sample in row["tn_timings_ns"])
