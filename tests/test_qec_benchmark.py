from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import qupy as qp
from benchmarks.qec import (
    QecWorkload,
    _wilson_interval,
    qec_workloads_for_profile,
    reconstruct_syndromes,
    run_qec_workload,
)


def test_qec_benchmark_profiles_cover_surface_memory_bases() -> None:
    smoke = qec_workloads_for_profile("smoke")
    standard = qec_workloads_for_profile("standard")

    assert len(smoke) == 1
    assert smoke[0].task == "surface_code:rotated_memory_x"
    assert len(standard) == 12
    assert {workload.task for workload in standard} == {
        "surface_code:rotated_memory_x",
        "surface_code:rotated_memory_z",
    }
    assert {workload.distance for workload in standard} == {3, 5, 7}
    assert {workload.physical_error_rate for workload in standard} == {0.001, 0.005}


def test_qec_benchmark_workload_validation_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        QecWorkload("bad", "surface_code:unknown", 3, 3, 0.001, 10, 1)
    with pytest.raises(ValueError, match="odd integer"):
        QecWorkload("bad", "surface_code:rotated_memory_x", 4, 3, 0.001, 10, 1)
    with pytest.raises(ValueError, match="error rate"):
        QecWorkload("bad", "surface_code:rotated_memory_x", 3, 3, 0.0, 10, 1)
    with pytest.raises(ValueError, match="shots"):
        QecWorkload("bad", "surface_code:rotated_memory_x", 3, 3, 0.001, 0, 1)


def test_reconstruct_syndromes_uses_error_mechanism_parity() -> None:
    model = qp.DetectorModel(
        3,
        1,
        [
            qp.DetectorError(0.1, [0, 1]),
            qp.DetectorError(0.2, [1, 2], [0]),
            qp.DetectorError(0.3, [0, 2]),
        ],
    )
    corrections = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=np.int8,
    )

    reconstructed = reconstruct_syndromes(model, corrections)
    np.testing.assert_array_equal(
        reconstructed,
        np.array(
            [
                [0, 0, 0],
                [1, 1, 0],
                [1, 1, 0],
                [0, 0, 0],
            ],
            dtype=np.int8,
        ),
    )

    with pytest.raises(ValueError, match="two-dimensional"):
        reconstruct_syndromes(model, [0, 1, 0])
    with pytest.raises(ValueError, match="columns"):
        reconstruct_syndromes(model, np.zeros((2, 2), dtype=np.int8))
    with pytest.raises(ValueError, match="zero or one"):
        reconstruct_syndromes(model, [[0, 1, 2]])


def test_wilson_interval_is_bounded_and_contains_observed_rate() -> None:
    for failures, shots in ((0, 100), (1, 100), (50, 100), (100, 100)):
        lower, upper = _wilson_interval(failures, shots)
        observed = failures / shots
        assert 0.0 <= lower <= observed <= upper <= 1.0


def test_qec_benchmark_runs_real_optional_surface_decoder_pair() -> None:
    pytest.importorskip("stim")
    pytest.importorskip("pymatching")

    workload = replace(qec_workloads_for_profile("smoke")[0], shots=32)
    result = run_qec_workload(workload, warmups=0, iterations=1, max_bp_iterations=20)

    assert result.detector_count > 0
    assert result.observable_count == 1
    assert result.error_count > 24
    assert result.active_error_count > 0
    assert result.edge_count > 0
    assert result.syndrome_consistency_rate == 1.0
    assert 0.0 <= result.qupy.logical_failure_rate <= 1.0
    assert 0.0 <= result.pymatching.logical_failure_rate <= 1.0
    assert 0.0 <= result.prediction_agreement_rate <= 1.0
    assert 0.0 <= result.bp_convergence_rate <= 1.0
    assert 0.0 <= result.osd_usage_rate <= 1.0
    assert result.qupy.median_ns > 0
    assert result.pymatching.median_ns > 0
