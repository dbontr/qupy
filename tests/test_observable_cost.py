from __future__ import annotations

import pytest

import qupy as qp
from benchmarks.observable_cost import _program, _query_data, policy_profile


def _workload(query: str, qubits: int):
    return next(
        workload
        for workload in policy_profile()
        if workload.query == query and workload.qubits == qubits
    )


def test_cuda_work_counts_deduplicated_pauli_masks() -> None:
    workload = _workload("batch", 14)
    _, cpu_work, cuda_work = _query_data(workload)

    assert cpu_work.term_evaluations == 24
    assert cuda_work.term_evaluations == 12
    assert cuda_work.term_evaluations < cpu_work.term_evaluations


def test_cuda_work_count_matches_native_query_size() -> None:
    if not qp.cuda_available():
        pytest.skip("native CUDA runtime is unavailable")

    for query, qubits in (
        ("expectation", 12),
        ("batch", 14),
        ("variance", 15),
        ("covariance", 16),
    ):
        workload = _workload(query, qubits)
        program = _program(workload)
        values, _, cuda_work = _query_data(workload)
        if query == "expectation":
            actual = qp.expect_observable(program, values[0], backend="native-cuda").evaluations
        elif query == "variance":
            actual = qp.variance_observable(program, values[0], backend="native-cuda").evaluations
        elif query == "covariance":
            actual = qp.covariance(program, values[0], values[1], backend="native-cuda").evaluations
        else:
            combined = qp.Observable([term for value in values for term in value.terms])
            actual = qp.expect_observable(program, combined, backend="native-cuda").evaluations
        assert cuda_work.term_evaluations == actual
