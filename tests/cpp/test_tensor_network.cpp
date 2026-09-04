#include "qupy/tensor_network.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Function>
void require_invalid(Function&& function, const std::string& message) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(message);
}

template <typename Function>
void require_length(Function&& function, const std::string& message) {
    try {
        function();
    } catch (const std::length_error&) {
        return;
    }
    throw std::runtime_error(message);
}

qupy::Observable pauli_sum() {
    return qupy::Observable({
        qupy::PauliTerm(1.0, {{0U, qupy::Pauli::X}, {1U, qupy::Pauli::X}}),
        qupy::PauliTerm(0.5, {{0U, qupy::Pauli::Z}, {1U, qupy::Pauli::Z}}),
        qupy::PauliTerm(-0.25, {}),
    });
}

void require_plan_matches_result(
    const qupy::TensorNetworkPlan& plan,
    const qupy::TensorNetworkResult& result
) {
    require(plan.term_count == result.term_count, "tensor plan term count drifted from execution");
    require(plan.contractions == result.contractions, "tensor plan contraction count drifted from execution");
    require(plan.peak_tensor_rank == result.peak_tensor_rank, "tensor plan peak rank drifted from execution");
    require(plan.peak_tensor_bytes == result.peak_tensor_bytes, "tensor plan peak bytes drifted from execution");
    require(
        std::abs(plan.scalar_multiplications - result.scalar_multiplications) < 1e-12,
        "tensor plan work drifted from execution"
    );
}

void test_bell_pauli_sum() {
    qupy::Program bell(2U);
    bell = qupy::h(bell, 0U);
    bell = qupy::cx(bell, 0U, 1U);
    const qupy::Observable observable = pauli_sum();

    const qupy::TensorNetworkPlan plan = qupy::tensor_network_plan(bell, observable);
    const qupy::TensorNetworkResult result = qupy::tensor_network_expectation(bell, observable);
    require_plan_matches_result(plan, result);
    require(std::abs(result.value - 1.25) < 1e-12, "Bell tensor-network expectation mismatch");
    require(result.term_count == 3U, "tensor-network term count mismatch");
    require(result.contractions > 0U, "tensor-network contraction count is empty");
    require(result.peak_tensor_rank >= 4U, "two-qubit gate did not appear in tensor rank");
    require(result.peak_tensor_bytes >= 16U * sizeof(qupy::Complex), "tensor peak bytes mismatch");
    require(result.scalar_multiplications > 0.0, "tensor-network work counter is empty");
    require(result.exact, "tensor-network result was not marked exact");
    require(result.backend == "native-tn", "tensor-network backend mismatch");
    require(result.method == "greedy-contraction", "tensor-network method mismatch");
    require(plan.max_tensor_bytes == qupy::kTensorNetworkDefaultMaxBytes, "tensor plan memory policy mismatch");
    require(plan.program_fingerprint == bell.fingerprint(), "tensor plan program identity mismatch");
    require(plan.observable_fingerprint == observable.fingerprint(), "tensor plan observable identity mismatch");
    require(plan.plan_fingerprint.size() == 64U, "tensor plan fingerprint is not SHA-256 sized");
}

void test_matches_dense_rich_observable() {
    qupy::Program program(5U);
    program = qupy::h(program, 0U);
    program = qupy::rx(program, 0.31, 1U);
    program = qupy::ry(program, -0.47, 2U);
    program = qupy::rz(program, 0.19, 3U);
    program = qupy::cx(program, 0U, 4U);
    program = qupy::cz(program, 1U, 3U);
    program = qupy::swap(program, 2U, 4U);
    program = qupy::ry(program, 0.27, 4U);
    program = qupy::cx(program, 4U, 1U);

    const qupy::Observable observable({
        qupy::PauliTerm(0.37, {{0U, qupy::Pauli::X}, {4U, qupy::Pauli::Z}}),
        qupy::PauliTerm(-0.21, {{1U, qupy::Pauli::Y}, {3U, qupy::Pauli::X}}),
        qupy::PauliTerm(0.18, {{2U, qupy::Pauli::Z}}),
        qupy::PauliTerm(0.09, {}),
    });
    const double dense = qupy::expect_observable(program, observable, "native-cpu").value;
    const qupy::TensorNetworkPlan plan = qupy::tensor_network_plan(program, observable);
    const qupy::TensorNetworkResult tensor = qupy::tensor_network_expectation(program, observable);
    require_plan_matches_result(plan, tensor);
    require(std::abs(tensor.value - dense) < 2e-12, "general tensor network disagrees with dense execution");

    const qupy::ObservableResult adapter = qupy::tensor_network_expect_observable(program, observable);
    require(std::abs(adapter.value - dense) < 2e-12, "tensor observable adapter disagrees with dense execution");
    require(adapter.backend == "native-tn", "tensor observable adapter backend mismatch");
    require(adapter.active_qubits == program.num_qubits(), "tensor observable adapter qubit provenance mismatch");
    require(adapter.evaluations == observable.terms().size(), "tensor observable adapter evaluation count mismatch");

    const qupy::ObservableExecutionPlan observable_plan = qupy::tensor_network_observable_plan(
        program, {observable}
    );
    require(observable_plan.backend == "native-tn", "tensor observable plan backend mismatch");
    require(observable_plan.method == "greedy-contraction-observable", "tensor observable plan method mismatch");
    require(observable_plan.exact, "tensor observable plan must be exact");
    require(observable_plan.estimated_state_bytes == plan.peak_tensor_bytes, "tensor observable plan workspace mismatch");
    require(observable_plan.predicted_ns == std::nullopt, "uncalibrated tensor plan reported runtime prediction");
}

void test_large_product_circuit_without_statevector() {
    constexpr std::size_t qubits = 80U;
    qupy::Program program(qubits);
    double final_angle = 0.0;
    for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
        const double angle = 0.013 * static_cast<double>((qubit % 7U) + 1U);
        program = qupy::ry(program, angle, qubit);
        if (qubit + 1U == qubits) {
            final_angle = angle;
        }
    }
    const qupy::Observable observable({
        qupy::PauliTerm(1.0, {{qubits - 1U, qupy::Pauli::Z}}),
    });
    const qupy::TensorNetworkPlan plan = qupy::tensor_network_plan(
        program, observable, 1U << 20U
    );
    const qupy::TensorNetworkResult result = qupy::tensor_network_expectation(
        program, observable, 1U << 20U
    );
    require_plan_matches_result(plan, result);

    require(
        std::abs(result.value - std::cos(final_angle)) < 2e-12,
        "large product-circuit tensor expectation mismatch"
    );
    require(result.peak_tensor_rank <= 2U, "product circuit created unnecessary tensor width");
    require(result.peak_tensor_bytes <= 4U * sizeof(qupy::Complex), "product tensor memory is too large");

    const qupy::ObservableResult adapter = qupy::tensor_network_expect_observable(
        program, observable, 1U << 20U
    );
    require(
        std::abs(adapter.value - std::cos(final_angle)) < 2e-12,
        "large product-circuit tensor adapter mismatch"
    );
}

void test_batch_adapter() {
    qupy::Program program(3U);
    program = qupy::h(program, 0U);
    program = qupy::cx(program, 0U, 1U);
    program = qupy::ry(program, 0.27, 2U);
    const qupy::Observable first({
        qupy::PauliTerm(1.0, {{0U, qupy::Pauli::X}, {1U, qupy::Pauli::X}}),
    });
    const qupy::Observable second({
        qupy::PauliTerm(0.5, {{2U, qupy::Pauli::Z}}),
        qupy::PauliTerm(-0.25, {}),
    });
    const qupy::ObservableBatch batch = qupy::tensor_network_expect_observables(
        program, {first, second}
    );
    require(batch.backend == "native-tn", "tensor batch backend mismatch");
    require(batch.observable_count == 2U, "tensor batch observable count mismatch");
    require(batch.values.size() == 2U, "tensor batch result shape mismatch");
    require(
        std::abs(batch.values[0] - qupy::expect_observable(program, first, "native-cpu").value) < 2e-12,
        "tensor batch first value mismatch"
    );
    require(
        std::abs(batch.values[1] - qupy::expect_observable(program, second, "native-cpu").value) < 2e-12,
        "tensor batch second value mismatch"
    );
}

void test_unified_gradient_uses_tensor_backend() {
    qupy::Program program(2U);
    program = qupy::ry(program, 0.0, 0U);
    program = qupy::rx(program, 0.0, 1U);
    program = qupy::cx(program, 0U, 1U);
    const std::vector<qupy::ParameterSlot> slots = {{0U, 0U}, {1U, 0U}};
    const std::vector<double> parameters = {0.37, -0.21};
    const qupy::Observable observable({
        qupy::PauliTerm(0.7, {{0U, qupy::Pauli::Z}}),
        qupy::PauliTerm(0.4, {{0U, qupy::Pauli::X}, {1U, qupy::Pauli::X}}),
    });

    const qupy::GradientResult dense = qupy::value_and_grad(
        program, observable, slots, parameters, "native-cpu", qupy::GradientMethod::ParameterShift
    );
    const qupy::GradientResult tensor = qupy::value_and_grad(
        program, observable, slots, parameters, "native-tn", qupy::GradientMethod::Auto
    );
    require(tensor.backend == "native-tn", "unified gradient did not retain tensor backend");
    require(tensor.method == "parameter-shift", "tensor auto gradient did not select parameter shift");
    require(std::abs(tensor.value - dense.value) < 2e-12, "tensor gradient value disagrees with dense execution");
    require(tensor.gradient.size() == dense.gradient.size(), "tensor gradient shape mismatch");
    for (std::size_t index = 0U; index < tensor.gradient.size(); ++index) {
        require(
            std::abs(tensor.gradient[index] - dense.gradient[index]) < 2e-12,
            "tensor gradient disagrees with dense parameter shift"
        );
    }
    require_invalid(
        [&] {
            static_cast<void>(qupy::value_and_grad(
                program, observable, slots, parameters, "native-tn", qupy::GradientMethod::Adjoint
            ));
        },
        "tensor-network adjoint differentiation was accepted"
    );
}

void test_unified_jacobian_and_hessian_use_tensor_backend() {
    qupy::Program program(2U);
    program = qupy::ry(program, 0.0, 0U);
    program = qupy::rx(program, 0.0, 1U);
    program = qupy::cx(program, 0U, 1U);
    const std::vector<qupy::ParameterSlot> slots = {{0U, 0U}, {1U, 0U}};
    const std::vector<double> parameters = {0.37, -0.21};
    const qupy::Observable first({qupy::PauliTerm(1.0, {{0U, qupy::Pauli::Z}})});
    const qupy::Observable second({
        qupy::PauliTerm(0.5, {{0U, qupy::Pauli::X}, {1U, qupy::Pauli::X}}),
        qupy::PauliTerm(-0.2, {{1U, qupy::Pauli::Z}}),
    });

    const qupy::JacobianResult dense_jacobian = qupy::jacobian(
        program, {first, second}, slots, parameters, "native-cpu",
        qupy::GradientMethod::ParameterShift
    );
    const qupy::JacobianResult tensor_jacobian = qupy::jacobian(
        program, {first, second}, slots, parameters, "native-tn", qupy::GradientMethod::Auto
    );
    require(tensor_jacobian.backend == "native-tn", "tensor Jacobian backend mismatch");
    require(tensor_jacobian.method == "parameter-shift", "tensor Jacobian method mismatch");
    require(tensor_jacobian.values.size() == dense_jacobian.values.size(), "tensor Jacobian value shape mismatch");
    require(tensor_jacobian.jacobian.size() == dense_jacobian.jacobian.size(), "tensor Jacobian shape mismatch");
    for (std::size_t index = 0U; index < tensor_jacobian.values.size(); ++index) {
        require(std::abs(tensor_jacobian.values[index] - dense_jacobian.values[index]) < 2e-12, "tensor Jacobian value mismatch");
    }
    for (std::size_t index = 0U; index < tensor_jacobian.jacobian.size(); ++index) {
        require(std::abs(tensor_jacobian.jacobian[index] - dense_jacobian.jacobian[index]) < 2e-12, "tensor Jacobian derivative mismatch");
    }

    const qupy::HessianResult dense_hessian = qupy::hessian(
        program, second, slots, parameters, "native-cpu"
    );
    const qupy::HessianResult tensor_hessian = qupy::hessian(
        program, second, slots, parameters, "native-tn"
    );
    require(tensor_hessian.backend == "native-tn", "tensor Hessian backend mismatch");
    require(tensor_hessian.method == "parameter-shift", "tensor Hessian method mismatch");
    require(tensor_hessian.hessian.size() == dense_hessian.hessian.size(), "tensor Hessian shape mismatch");
    for (std::size_t index = 0U; index < tensor_hessian.hessian.size(); ++index) {
        require(std::abs(tensor_hessian.hessian[index] - dense_hessian.hessian[index]) < 3e-12, "tensor Hessian mismatch");
    }
}

void test_memory_guard_and_validation() {
    qupy::Program entangled(2U);
    entangled = qupy::cx(entangled, 0U, 1U);
    const qupy::Observable z({
        qupy::PauliTerm(1.0, {{0U, qupy::Pauli::Z}}),
    });

    require_length(
        [&] { static_cast<void>(qupy::tensor_network_plan(entangled, z, 128U)); },
        "tensor planner did not reject an oversized gate tensor"
    );
    require_length(
        [&] { static_cast<void>(qupy::tensor_network_expectation(entangled, z, 128U)); },
        "tensor memory guard did not reject an oversized gate tensor"
    );
    require_invalid(
        [&] { static_cast<void>(qupy::tensor_network_plan(entangled, z, 0U)); },
        "zero tensor planner memory limit was accepted"
    );
    require_invalid(
        [&] { static_cast<void>(qupy::tensor_network_expectation(entangled, z, 0U)); },
        "zero tensor memory limit was accepted"
    );

    const qupy::Observable outside({
        qupy::PauliTerm(1.0, {{2U, qupy::Pauli::Z}}),
    });
    require_invalid(
        [&] { static_cast<void>(qupy::tensor_network_plan(entangled, outside)); },
        "out-of-range tensor planner observable was accepted"
    );
    require_invalid(
        [&] { static_cast<void>(qupy::tensor_network_expectation(entangled, outside)); },
        "out-of-range tensor observable was accepted"
    );
}

void test_zero_observable_short_circuit() {
    const qupy::Observable zero({qupy::PauliTerm(0.0, {})});
    const qupy::TensorNetworkPlan plan = qupy::tensor_network_plan(qupy::Program(3U), zero);
    const qupy::TensorNetworkResult result = qupy::tensor_network_expectation(
        qupy::Program(3U), zero
    );
    require_plan_matches_result(plan, result);
    require(result.value == 0.0, "zero tensor observable changed value");
    require(result.contractions == 0U, "zero tensor observable performed contractions");
    require(result.peak_tensor_bytes == 0U, "zero tensor observable allocated tensors");
}

}  // namespace

int main() {
    test_bell_pauli_sum();
    test_matches_dense_rich_observable();
    test_large_product_circuit_without_statevector();
    test_batch_adapter();
    test_unified_gradient_uses_tensor_backend();
    test_unified_jacobian_and_hessian_use_tensor_backend();
    test_memory_guard_and_validation();
    test_zero_observable_short_circuit();
    return 0;
}
