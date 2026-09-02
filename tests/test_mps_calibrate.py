from __future__ import annotations

import json
from pathlib import Path

import pytest

import qupy as qp
from benchmarks.mps_calibrate import calibrate_files


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


def _write_report(
    path: Path,
    *,
    scale: int = 1,
    workload_count: int = 16,
    slow_workload: int | None = None,
    platform_name: str = "test-host",
) -> None:
    policy_evidence = []
    validations = []
    for index in range(workload_count):
        fingerprint = f"{index + 1:064x}"
        cpu_base = (1000 + index * 11) * scale
        mps_base = (800 + index * 7) * scale
        ratio = 1.15 if index == slow_workload else 1.05
        comparisons = []
        for backend, baseline in (("native-cpu", cpu_base), ("native-mps", mps_base)):
            baseline_samples = [baseline - scale, baseline + scale] * 2
            adaptive = round(baseline * ratio)
            adaptive_samples = [adaptive - scale, adaptive + scale] * 2
            comparisons.append(
                {
                    "baseline_backend": backend,
                    "baseline_timings_ns": baseline_samples,
                    "adaptive_timings_ns": adaptive_samples,
                    "baseline_median_ns": baseline,
                    "adaptive_median_ns": adaptive,
                    "regret": 1.0,
                }
            )
        policy_evidence.append(
            {
                "workload": f"policy-{index}",
                "workload_fingerprint": fingerprint,
                "comparisons": comparisons,
                "max_regret": 1.0,
            }
        )
        validations.append(
            {
                "workload": f"policy-{index}",
                "cpu_mps_abs_error": 0.0,
                "cpu_adaptive_abs_error": 0.0,
                "max_abs_error": 0.0,
            }
        )
    payload = {
        "schema_version": 1,
        "engine_version": qp.core_version(),
        "workload_version": 1,
        "profile": "policy",
        "policy_version": 1,
        "warmups": 1,
        "iterations": 4,
        "host": {
            "planner_host_fingerprint": qp.planner_host_fingerprint(),
            "platform": platform_name,
        },
        "validations": validations,
        "policy_evidence": policy_evidence,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _reports(tmp_path: Path, **kwargs: object) -> tuple[Path, Path, Path]:
    paths = tuple(tmp_path / f"report-{index}.json" for index in range(3))
    for index, path in enumerate(paths, start=1):
        _write_report(path, scale=index, **kwargs)
    return paths


def test_paired_policy_calibration_validates_and_promotes_schema_v3(tmp_path: Path) -> None:
    reports = _reports(tmp_path)
    calibration = calibrate_files(reports)
    assert calibration.report_count == 3
    assert calibration.decision.samples == 16
    assert calibration.decision.mistakes == 0
    assert calibration.decision.max_regret == pytest.approx(1.05, abs=0.002)
    assert calibration.validated

    base = tmp_path / "base.qpcost"
    promoted = tmp_path / "promoted.qpcost"
    _write_v1(base)
    promoted.write_text(calibration.to_planner_text(base), encoding="utf-8")
    model = qp.load_planner_cost_model(str(promoted))
    assert model.schema_version == 3
    assert model.mps_policy_version == 1
    assert model.mps_auto_validated
    assert not model.cuda_auto_validated


def test_policy_calibration_recomputes_regret_from_raw_pairs(tmp_path: Path) -> None:
    reports = _reports(tmp_path, slow_workload=7)
    calibration = calibrate_files(reports)
    assert calibration.decision.samples == 16
    assert calibration.decision.mistakes == 1
    assert calibration.decision.max_regret == pytest.approx(1.15, abs=0.002)
    assert not calibration.validated

    base = tmp_path / "base.qpcost"
    _write_v1(base)
    with pytest.raises(ValueError, match="validated policy evidence"):
        calibration.to_planner_text(base)


def test_policy_calibration_requires_three_matching_reports(tmp_path: Path) -> None:
    reports = _reports(tmp_path)
    with pytest.raises(ValueError, match="at least 3 reports"):
        calibrate_files(reports[:2])

    mismatched = tmp_path / "mismatched.json"
    _write_report(mismatched, scale=4, platform_name="different-host-metadata")
    with pytest.raises(ValueError, match="different hosts"):
        calibrate_files((reports[0], reports[1], mismatched))


def test_policy_calibration_requires_complete_even_pairs(tmp_path: Path) -> None:
    reports = list(_reports(tmp_path))
    payload = json.loads(reports[2].read_text(encoding="utf-8"))
    payload["policy_evidence"][0]["comparisons"][0]["adaptive_timings_ns"].pop()
    reports[2].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="positive even sample count"):
        calibrate_files(tuple(reports))


def test_policy_calibration_requires_complete_exactness_coverage(tmp_path: Path) -> None:
    reports = list(_reports(tmp_path))
    payload = json.loads(reports[2].read_text(encoding="utf-8"))
    payload["validations"].pop()
    reports[2].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="validation does not cover"):
        calibrate_files(tuple(reports))


def test_policy_calibration_rejects_wrong_policy_version_and_sample_count(
    tmp_path: Path,
) -> None:
    reports = list(_reports(tmp_path))
    payload = json.loads(reports[2].read_text(encoding="utf-8"))
    payload["policy_version"] = 2
    reports[2].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported policy version"):
        calibrate_files(tuple(reports))

    _write_report(reports[2], scale=3)
    payload = json.loads(reports[2].read_text(encoding="utf-8"))
    payload["iterations"] = 6
    reports[2].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="timing counts do not match"):
        calibrate_files(tuple(reports))


def test_schema_v2_promotion_preserves_validated_cuda_evidence(tmp_path: Path) -> None:
    if not qp.cuda_available():
        pytest.skip("CUDA preservation requires a matching GPU host")
    reports = _reports(tmp_path)
    calibration = calibrate_files(reports)
    base = tmp_path / "base-v2.qpcost"
    cuda_host = qp.planner_cuda_host_fingerprint()
    lines = (
        "qupy-planner-cost 2",
        f"engine {qp.core_version()}",
        "workload 1",
        f"host {qp.planner_host_fingerprint()}",
        f"cuda-host {cuda_host}",
        "validated 1",
        *_base_rows(),
        "model statevector-return-cpu 5 10 0 0 0 0 1.1 1.2",
        "model statevector-return-cuda 4 0 0 0 0 1.1 1.2",
        "decision statevector-auto 8 0 1.0",
    )
    base.write_text("\n".join(lines) + "\n", encoding="utf-8")

    promoted = tmp_path / "promoted-v3.qpcost"
    promoted.write_text(calibration.to_planner_text(base), encoding="utf-8")
    model = qp.load_planner_cost_model(str(promoted))
    assert model.schema_version == 3
    assert model.cuda_auto_validated
    assert model.cuda_host_fingerprint == cuda_host
    assert model.mps_auto_validated
