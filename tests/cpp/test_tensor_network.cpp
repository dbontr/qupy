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

void test_bell_pauli_sum() {
    qupy::Program bell(2U);
    bell = qupy::h(bell, 0U);
    bell = qupy::cx(bell, 0U, 1U);

    const qupy::TensorNetworkResult result = qupy::tensor_network_expectation(
        bell, pauli_sum()
    );
    require(std::abs(result.value - 1.25) < 1e-12, "Bell tensor-network expectation mismatch");
    require(result.term_count == 3U, "tensor-network term count mismatch");
    require(result.contractions > 0U, "tensor-network contraction count is empty");
    require(result.peak_tensor_rank >= 4U, "two-qubit gate did not appear in tensor rank");
    require(result.peak_tensor_bytes >= 16U * sizeof(qupy::Complex), "tensor peak bytes mismatch");
    require(result.scalar_multiplications > 0.0, "tensor-network work counter is empty");
    require(result.exact, "tensor-network result was not marked exact");
    require(result.backend == "native-tn", "tensor-network backend mismatch");
    require(result.method == "greedy-contraction", "tensor-network method mismatch");
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
    const double tensor = qupy::tensor_network_expectation(program, observable).value;
    require(std::abs(tensor - dense) < 2e-12, "general tensor network disagrees with dense execution");
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
    const qupy::TensorNetworkResult result = qupy::tensor_network_expectation(
        program, observable, 1U << 20U
    );

    require(
        std::abs(result.value - std::cos(final_angle)) < 2e-12,
        "large product-circuit tensor expectation mismatch"
    );
    require(result.peak_tensor_rank <= 2U, "product circuit created unnecessary tensor width");
    require(result.peak_tensor_bytes <= 4U * sizeof(qupy::Complex), "product tensor memory is too large");
}

void test_memory_guard_and_validation() {
    qupy::Program entangled(2U);
    entangled = qupy::cx(entangled, 0U, 1U);
    const qupy::Observable z({
        qupy::PauliTerm(1.0, {{0U, qupy::Pauli::Z}}),
    });

    require_length(
        [&] { static_cast<void>(qupy::tensor_network_expectation(entangled, z, 128U)); },
        "tensor memory guard did not reject an oversized gate tensor"
    );
    require_invalid(
        [&] { static_cast<void>(qupy::tensor_network_expectation(entangled, z, 0U)); },
        "zero tensor memory limit was accepted"
    );

    const qupy::Observable outside({
        qupy::PauliTerm(1.0, {{2U, qupy::Pauli::Z}}),
    });
    require_invalid(
        [&] { static_cast<void>(qupy::tensor_network_expectation(entangled, outside)); },
        "out-of-range tensor observable was accepted"
    );
}

void test_zero_observable_short_circuit() {
    const qupy::Observable zero({qupy::PauliTerm(0.0, {})});
    const qupy::TensorNetworkResult result = qupy::tensor_network_expectation(
        qupy::Program(3U), zero
    );
    require(result.value == 0.0, "zero tensor observable changed value");
    require(result.contractions == 0U, "zero tensor observable performed contractions");
    require(result.peak_tensor_bytes == 0U, "zero tensor observable allocated tensors");
}

}  // namespace

int main() {
    test_bell_pauli_sum();
    test_matches_dense_rich_observable();
    test_large_product_circuit_without_statevector();
    test_memory_guard_and_validation();
    test_zero_observable_short_circuit();
    return 0;
}
