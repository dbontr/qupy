from __future__ import annotations

import pytest

import qupy as qp


def _observable() -> qp.Observable:
    return qp.observable_from_z(qp.Z(0))


def _noisy_program() -> qp.NoisyProgram:
    program = qp.x(qp.Program(1), 0)
    return qp.NoisyProgram(program, [(1, qp.bit_flip(0, 0.25))])


def test_distributed_scale_bindings_require_multiple_ranks() -> None:
    program = qp.h(qp.Program(1), 0)
    observable = _observable()
    noisy = _noisy_program()

    if qp.mpi_compiled():
        info = qp.distributed_info()
        if info.world_size > 1:
            pytest.skip("single-process Python smoke test requires an MPI world of size one")
        message = "at least two MPI ranks"
    else:
        message = "without MPI support"

    with pytest.raises(RuntimeError, match=message):
        qp.distributed_tensor_network_expectation(program, observable)

    with pytest.raises(RuntimeError, match=message):
        qp.distributed_trajectory_expectations(noisy, [observable], trajectories=8, seed=11)

    with pytest.raises(RuntimeError, match=message):
        qp.distributed_trajectory_expectation(noisy, observable, trajectories=8, seed=11)
