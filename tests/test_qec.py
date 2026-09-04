from __future__ import annotations

import math

import numpy as np
import pytest

import qupy as qp


def _small_model() -> qp.DetectorModel:
    return qp.DetectorModel(
        2,
        1,
        [
            qp.DetectorError(0.10, [0], [0]),
            qp.DetectorError(0.05, [1]),
            qp.DetectorError(0.01, [0, 1], [0]),
        ],
    )


def _large_model() -> qp.DetectorModel:
    errors: list[qp.DetectorError] = []
    for detector in range(32):
        observables = [0] if detector == 31 else []
        errors.append(qp.DetectorError(0.01 + 0.0001 * detector, [detector], observables))
    for detector in range(32):
        errors.append(
            qp.DetectorError(
                0.005 + 0.00005 * detector,
                [detector, (detector + 1) % 32],
            )
        )
    return qp.DetectorModel(32, 1, errors)


def _syndrome(model: qp.DetectorModel, correction: np.ndarray) -> np.ndarray:
    result = np.zeros(model.detector_count, dtype=np.int8)
    for selected, error in zip(correction, model.errors, strict=True):
        if selected:
            for detector in error.detectors:
                result[detector] ^= 1
    return result


def test_bp_osd_matches_exact_reference_on_small_model() -> None:
    model = _small_model()
    syndrome = np.array([1, 0], dtype=np.int8)

    exact = qp.decode_detector_model(model, syndrome)
    scalable = qp.decode_detector_model_bp_osd(model, syndrome)

    np.testing.assert_array_equal(_syndrome(model, scalable.correction), syndrome)
    np.testing.assert_array_equal(scalable.observables, exact.observables)
    assert scalable.log_likelihood == pytest.approx(exact.log_likelihood, abs=1e-12)
    assert scalable.matched_errors == exact.matched_errors
    assert scalable.method in {"belief-propagation", "belief-propagation-osd0"}
    assert not scalable.correction.flags.writeable
    assert not scalable.observables.flags.writeable


def test_bp_osd_zero_iterations_forces_order_zero_repair() -> None:
    decoder = qp.BpOsdDecoder(_small_model(), max_iterations=0)
    syndrome = np.array([1, 1], dtype=np.int8)
    result = decoder.decode(syndrome)

    assert result.osd_used
    assert not result.bp_converged
    assert result.iterations == 0
    assert result.method == "belief-propagation-osd0"
    np.testing.assert_array_equal(_syndrome(decoder.model, result.correction), syndrome)


def test_bp_osd_scales_beyond_reference_decoder_limit_and_batches() -> None:
    model = _large_model()
    known = np.zeros(len(model.errors), dtype=np.int8)
    known[[3, 17, 31]] = 1
    syndrome = _syndrome(model, known)

    with pytest.raises((ValueError, RuntimeError), match="at most 24"):
        qp.decode_detector_model(model, syndrome)

    decoder = qp.BpOsdDecoder(model, max_iterations=30, damping=0.1)
    result = decoder.decode(syndrome)
    np.testing.assert_array_equal(_syndrome(model, result.correction), syndrome)
    assert decoder.edge_count > 0
    assert decoder.active_error_count == 64
    assert math.isfinite(result.log_likelihood)

    syndromes = np.stack(
        [
            np.zeros(model.detector_count, dtype=np.int8),
            syndrome,
            syndrome,
        ]
    )
    batch = decoder.decode_batch(syndromes)

    assert batch.shots == 3
    assert batch.error_count == 64
    assert batch.observable_count == 1
    assert batch.corrections.shape == (3, 64)
    assert batch.observables.shape == (3, 1)
    assert batch.log_likelihoods.shape == (3,)
    assert batch.matched_errors.shape == (3,)
    assert batch.iterations.shape == (3,)
    assert batch.bp_converged.shape == (3,)
    assert batch.osd_used.shape == (3,)
    assert not batch.corrections.flags.writeable
    assert not batch.observables.flags.writeable
    np.testing.assert_array_equal(batch.corrections[1], result.correction)
    np.testing.assert_array_equal(batch.corrections[2], result.correction)
    for shot in range(batch.shots):
        np.testing.assert_array_equal(
            _syndrome(model, batch.corrections[shot]),
            syndromes[shot],
        )


def test_bp_osd_handles_deterministic_error_probabilities() -> None:
    model = qp.DetectorModel(
        2,
        1,
        [
            qp.DetectorError(1.0, [0], [0]),
            qp.DetectorError(0.2, [1]),
            qp.DetectorError(0.0, [0, 1], [0]),
        ],
    )
    result = qp.BpOsdDecoder(model, damping=0.25).decode([1, 1])

    np.testing.assert_array_equal(result.correction, [1, 1, 0])
    np.testing.assert_array_equal(result.observables, [1])
    np.testing.assert_array_equal(_syndrome(model, result.correction), [1, 1])


def test_bp_osd_rejects_impossible_syndrome_and_invalid_inputs() -> None:
    impossible = qp.DetectorModel(1, 0, [qp.DetectorError(0.0, [0])])
    with pytest.raises(ValueError, match="not representable"):
        qp.decode_detector_model_bp_osd(impossible, [1], max_iterations=0)

    with pytest.raises(ValueError, match="damping"):
        qp.BpOsdDecoder(_small_model(), damping=1.0)

    decoder = qp.BpOsdDecoder(_small_model())
    with pytest.raises(ValueError, match="one-dimensional"):
        decoder.decode([[0, 1]])
    with pytest.raises(ValueError, match="two-dimensional"):
        decoder.decode_batch([0, 1])
    with pytest.raises(ValueError, match="columns must match"):
        decoder.decode_batch(np.zeros((2, 3), dtype=np.int8))
    for invalid in ([0, 2], [0.5, 1.0], [0, 256]):
        with pytest.raises(ValueError, match="zero or one"):
            decoder.decode(invalid)
    with pytest.raises(ValueError, match="zero or one"):
        decoder.decode_batch([[0, 1], [1, 0.5]])
