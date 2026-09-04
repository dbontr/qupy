from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from benchmarks.qec_ldpc import (
    LdpcWorkload,
    _binary_product,
    _gf2_nullspace,
    gf2_rank,
    hamming_check_matrix,
    hypergraph_product_hamming_code,
    ldpc_workloads_for_profile,
    run_ldpc_workload,
)


def test_hamming_seed_and_hypergraph_product_dimensions() -> None:
    check = hamming_check_matrix(3)
    assert check.shape == (3, 7)
    assert gf2_rank(check) == 3
    assert _gf2_nullspace(check).shape == (4, 7)

    code = hypergraph_product_hamming_code(3)
    assert code.qubit_count == 58
    assert code.hx.shape == (21, 58)
    assert code.hz.shape == (21, 58)
    assert gf2_rank(code.hx) == 21
    assert gf2_rank(code.hz) == 21
    assert code.logical_qubits == 16
    assert not bool(np.any(_binary_product(code.hx, code.hz.T)))
    assert not bool(np.any(_binary_product(code.hx, code.logical_z.T)))


def test_hypergraph_product_hamming_family_scales_rate() -> None:
    rank3 = hypergraph_product_hamming_code(3)
    rank4 = hypergraph_product_hamming_code(4)

    assert rank4.qubit_count == 241
    assert rank4.logical_qubits == 121
    assert rank4.qubit_count > rank3.qubit_count
    assert rank4.logical_qubits / rank4.qubit_count > rank3.logical_qubits / rank3.qubit_count


def test_qldpc_workload_profiles_and_validation() -> None:
    smoke = ldpc_workloads_for_profile("smoke")
    standard = ldpc_workloads_for_profile("standard")

    assert len(smoke) == 1
    assert smoke[0].hamming_rank == 3
    assert len(standard) == 6
    assert {workload.hamming_rank for workload in standard} == {3, 4}
    assert {workload.physical_error_rate for workload in standard} == {0.01, 0.03, 0.05}

    with pytest.raises(TypeError, match="hamming_rank"):
        LdpcWorkload("bad", True, 0.02, 10, 1)
    with pytest.raises(ValueError, match="at least three"):
        LdpcWorkload("bad", 2, 0.02, 10, 1)
    with pytest.raises(ValueError, match="error rate"):
        LdpcWorkload("bad", 3, 0.0, 10, 1)
    with pytest.raises(ValueError, match="shots"):
        LdpcWorkload("bad", 3, 0.02, 0, 1)
    with pytest.raises(ValueError, match="seed"):
        LdpcWorkload("bad", 3, 0.02, 10, -1)


def test_qldpc_benchmark_runs_real_optional_decoder_pair() -> None:
    pytest.importorskip("ldpc")

    workload = replace(ldpc_workloads_for_profile("smoke")[0], shots=24)
    result = run_ldpc_workload(
        workload,
        warmups=0,
        iterations=1,
        max_bp_iterations=20,
    )

    assert result.family == "hypergraph-product-hamming"
    assert result.qubit_count == 58
    assert result.error_count == 58
    assert result.logical_qubits == 16
    assert result.error_count > 24
    assert result.active_error_count == 58
    assert result.edge_count > 0
    assert result.qupy_syndrome_consistency_rate == 1.0
    assert result.ldpc_syndrome_consistency_rate == 1.0
    assert 0.0 <= result.qupy.logical_failure_rate <= 1.0
    assert 0.0 <= result.ldpc.logical_failure_rate <= 1.0
    assert 0.0 <= result.prediction_agreement_rate <= 1.0
    assert 0.0 <= result.bp_convergence_rate <= 1.0
    assert 0.0 <= result.osd_usage_rate <= 1.0
    assert result.qupy.median_ns > 0
    assert result.ldpc.median_ns > 0
