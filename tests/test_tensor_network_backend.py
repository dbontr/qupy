from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

import qupy as qp
from qupy.tensor_network import tensor_network_expectation, tensor_network_plan


def _rich_observable() -> qp.Observable:
    return qp.observable(
        [
            qp.pauli_term(0.7, [qp.pauli(0, qp.Pauli.X), qp.pauli(2, qp.Pauli.Z)]),
            qp.pauli_term(-0.2, [qp.pauli(1, qp.Pauli.Y)]),
            qp.pauli_term(0.1, []),
        ]
    )


def _program() -> qp.Program:
    program = qp.h(qp.Program(3), 0)
    program = qp.ry(program, 0.31, 1)
    program = qp.cx(program, 0, 2)
    return qp.cz(program, 1, 2)


def test_tensor_network_plan_matches_execution_provenance() -> None:
    program = _program()
    observable = _rich_observable()

    plan = tensor_network_plan(program, observable)
    result = tensor_network_expectation(program, observable)

    assert plan.term_count == result.term_count
    assert plan.contractions == result.contractions
    assert plan.peak_tensor_rank == result.peak_tensor_rank
    assert plan.peak_tensor_bytes == result.peak_tensor_bytes
    assert plan.scalar_multiplications == pytest.approx(result.scalar_multiplications)
    assert plan.exact is True
    assert plan.backend == "native-tn"
    assert plan.method == "greedy-contraction"
    assert plan.program_fingerprint == program.fingerprint
    assert plan.observable_fingerprint == observable.fingerprint
    assert len(plan.plan_fingerprint) == 64


def test_native_tn_uses_normal_observable_plan_and_execution_surface() -> None:
    program = _program()
    observable = _rich_observable()
    detailed = tensor_network_plan(program, observable)

    plan = qp.observable_plan(program, [observable], backend="native-tn")
    result = qp.expect_observable(program, observable, backend="native-tn")
    through_expect = qp.expect(program, observable, backend="native-tn")
    dense = qp.expect_observable(program, observable, backend="native-cpu")

    assert plan.backend == "native-tn"
    assert plan.method == "greedy-contraction-observable"
    assert plan.exact is True
    assert plan.active_qubits == program.num_qubits
    assert plan.estimated_state_bytes == detailed.peak_tensor_bytes
    assert plan.predicted_ns is None
    assert plan.cost_model_class == ""
    assert plan.cost_model_fingerprint == ""
    assert result.backend == "native-tn"
    assert result.value == pytest.approx(dense.value, abs=2e-12)
    assert through_expect.backend == "native-tn"
    assert through_expect.value == pytest.approx(dense.value, abs=2e-12)


def test_native_tn_batches_rich_observables() -> None:
    program = _program()
    first = _rich_observable()
    second = qp.observable_from_z(qp.Z(1))

    batch = qp.expect_observables(program, [first, second], backend="native-tn")
    dense = qp.expect_observables(program, [first, second], backend="native-cpu")

    assert batch.backend == "native-tn"
    assert batch.observable_count == 2
    assert batch.active_qubits == program.num_qubits
    assert batch.values == pytest.approx(dense.values, abs=2e-12)


def test_native_tn_handles_large_low_width_program_through_normal_api() -> None:
    qubits = 80
    program = qp.Program(qubits)
    final_angle = 0.0
    for qubit in range(qubits):
        final_angle = 0.013 * ((qubit % 7) + 1)
        program = qp.ry(program, final_angle, qubit)
    observable = qp.observable_from_z(qp.Z(qubits - 1))

    result = qp.expect_observable(program, observable, backend="native-tn")
    plan = qp.observable_plan(program, [observable], backend="native-tn")

    assert result.value == pytest.approx(math.cos(final_angle), abs=2e-12)
    assert result.backend == "native-tn"
    assert plan.estimated_state_bytes <= 64


def test_native_tn_uses_unified_gradient_surface() -> None:
    program = qp.ry(qp.Program(2), 0.0, 0)
    program = qp.rx(program, 0.0, 1)
    program = qp.cx(program, 0, 1)
    slots = [qp.ParameterSlot(0), qp.ParameterSlot(1)]
    parameters = np.array([0.37, -0.21], dtype=np.float64)
    observable = qp.observable(
        [
            qp.pauli_term(0.7, [qp.pauli(0, qp.Pauli.Z)]),
            qp.pauli_term(0.4, [qp.pauli(0, qp.Pauli.X), qp.pauli(1, qp.Pauli.X)]),
        ]
    )

    dense = qp.value_and_grad(
        program,
        observable,
        slots,
        parameters,
        backend="native-cpu",
        method=qp.GradientMethod.PARAMETER_SHIFT,
    )
    tensor = qp.value_and_grad(program, observable, slots, parameters, backend="native-tn")
    alias = qp.grad(program, observable, slots, parameters, backend="native-tn")

    assert tensor.backend == "native-tn"
    assert tensor.method == "parameter-shift"
    assert tensor.value == pytest.approx(dense.value, abs=2e-12)
    np.testing.assert_allclose(tensor.gradient, dense.gradient, atol=2e-12, rtol=0.0)
    assert alias.value == pytest.approx(tensor.value, abs=2e-12)
    np.testing.assert_allclose(alias.gradient, tensor.gradient, atol=2e-12, rtol=0.0)
    with pytest.raises(ValueError, match="native CPU state-vector backend"):
        qp.value_and_grad(
            program,
            observable,
            slots,
            parameters,
            backend="native-tn",
            method=qp.GradientMethod.ADJOINT,
        )


def test_native_tn_unified_jacobian_and_hessian_match_dense() -> None:
    program = qp.ry(qp.Program(2), 0.0, 0)
    program = qp.rx(program, 0.0, 1)
    program = qp.cx(program, 0, 1)
    slots = [qp.ParameterSlot(0), qp.ParameterSlot(1)]
    parameters = np.array([0.37, -0.21], dtype=np.float64)
    first = qp.observable_from_z(qp.Z(0))
    second = qp.observable(
        [
            qp.pauli_term(0.5, [qp.pauli(0, qp.Pauli.X), qp.pauli(1, qp.Pauli.X)]),
            qp.pauli_term(-0.2, [qp.pauli(1, qp.Pauli.Z)]),
        ]
    )

    dense_jacobian = qp.jacobian(
        program,
        [first, second],
        slots,
        parameters,
        backend="native-cpu",
        method=qp.GradientMethod.PARAMETER_SHIFT,
    )
    tensor_jacobian = qp.jacobian(program, [first, second], slots, parameters, backend="native-tn")
    assert tensor_jacobian.backend == "native-tn"
    assert tensor_jacobian.method == "parameter-shift"
    np.testing.assert_allclose(tensor_jacobian.values, dense_jacobian.values, atol=2e-12, rtol=0.0)
    np.testing.assert_allclose(
        tensor_jacobian.jacobian, dense_jacobian.jacobian, atol=2e-12, rtol=0.0
    )

    dense_hessian = qp.hessian(program, second, slots, parameters, backend="native-cpu")
    tensor_hessian = qp.hessian(program, second, slots, parameters, backend="native-tn")
    assert tensor_hessian.backend == "native-tn"
    assert tensor_hessian.method == "parameter-shift"
    assert tensor_hessian.value == pytest.approx(dense_hessian.value, abs=2e-12)
    np.testing.assert_allclose(
        tensor_hessian.gradient, dense_hessian.gradient, atol=2e-12, rtol=0.0
    )
    np.testing.assert_allclose(tensor_hessian.hessian, dense_hessian.hessian, atol=3e-12, rtol=0.0)


def test_auto_routing_does_not_select_uncalibrated_tensor_network() -> None:
    plan = qp.observable_plan(_program(), [_rich_observable()], backend="auto")
    assert plan.backend != "native-tn"


@pytest.mark.parametrize(
    "operation",
    [
        lambda program, observable: qp.variance_observable(
            program, observable, backend="native-tn"
        ),
        lambda program, observable: qp.variance(program, observable, backend="native-tn"),
        lambda program, observable: qp.covariance(
            program, observable, observable, backend="native-tn"
        ),
        lambda program, observable: qp.statevector(program, backend="native-tn"),
        lambda program, observable: qp.density_matrix(program, backend="native-tn"),
    ],
)
def test_native_tn_fails_closed_for_unsupported_result_modes(
    operation: Callable[[qp.Program, qp.Observable], object],
) -> None:
    with pytest.raises(
        ValueError,
        match="currently supports observable expectations and expectation differentiation only",
    ):
        operation(_program(), _rich_observable())
