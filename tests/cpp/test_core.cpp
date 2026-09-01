#include "qupy/core.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

constexpr double kTolerance = 1e-12;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(qupy::Complex actual, qupy::Complex expected, const std::string& message) {
    require(std::abs(actual - expected) <= kTolerance, message);
}

void test_bell_state() {
    qupy::Program program(2);
    program = qupy::h(program, 0);
    program = qupy::cx(program, 0, 1);

    const qupy::StateVector result = qupy::statevector(program);
    const double amplitude = 1.0 / std::sqrt(2.0);
    require(result.values.size() == 4, "Bell state dimension is wrong");
    require_close(result.values[0], amplitude, "Bell |00> amplitude is wrong");
    require_close(result.values[1], 0.0, "Bell |01> amplitude is wrong");
    require_close(result.values[2], 0.0, "Bell |10> amplitude is wrong");
    require_close(result.values[3], amplitude, "Bell |11> amplitude is wrong");
    require(result.backend == "native-cpu", "Bell state did not use the native backend");
}

void test_rotation_and_pauli_gates() {
    qupy::Program program(1);
    program = qupy::y(program, 0);
    const auto y_state = qupy::statevector(program);
    require_close(y_state.values[0], 0.0, "Y |0> zero amplitude is wrong");
    require_close(y_state.values[1], {0.0, 1.0}, "Y |0> one amplitude is wrong");

    program = qupy::Program(1);
    program = qupy::ry(program, std::acos(-1.0), 0);
    const auto ry_state = qupy::statevector(program);
    require_close(ry_state.values[0], 0.0, "RY(pi) zero amplitude is wrong");
    require_close(ry_state.values[1], 1.0, "RY(pi) one amplitude is wrong");

    program = qupy::Program(1);
    program = qupy::x(program, 0);
    program = qupy::rz(program, std::acos(-1.0), 0);
    const auto rz_state = qupy::statevector(program);
    require_close(rz_state.values[0], 0.0, "RZ(pi) zero amplitude is wrong");
    require_close(rz_state.values[1], {0.0, 1.0}, "RZ(pi) phase is wrong");
}

void test_two_qubit_gates() {
    qupy::Program program(2);
    program = qupy::x(program, 0);
    program = qupy::swap(program, 0, 1);
    auto state = qupy::statevector(program);
    require_close(state.values[2], 1.0, "SWAP did not move the excitation");

    program = qupy::Program(2);
    program = qupy::x(program, 0);
    program = qupy::x(program, 1);
    program = qupy::cz(program, 0, 1);
    state = qupy::statevector(program);
    require_close(state.values[3], -1.0, "CZ did not apply the |11> phase");
}

void test_results_and_planner() {
    qupy::Program program(2);
    program = qupy::h(program, 0);
    program = qupy::cx(program, 0, 1);

    const auto expectation = qupy::expectation(program, qupy::pauli_z(0));
    require(std::abs(expectation.value) <= kTolerance, "Bell Z expectation is not zero");

    const auto samples = qupy::sample(program, 256, 7U);
    const auto counts = samples.counts();
    require(counts.size() == 2, "Bell sampling returned impossible states");
    require(counts.contains("00"), "Bell sampling did not return |00>");
    require(counts.contains("11"), "Bell sampling did not return |11>");

    const auto deterministic = qupy::sample(program, 16, 7U);
    const std::vector<std::int8_t> expected = {
        0, 0, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 0, 0, 0, 0,
        1, 1, 0, 0, 1, 1, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0,
    };
    require(deterministic.values == expected, "seeded sampling sequence changed");

    const auto execution_plan = qupy::plan(program, qupy::ResultMode::Sample);
    require(execution_plan.exact, "native plan must be exact");
    require(execution_plan.threads >= 1, "native plan reported no threads");
    require(execution_plan.original_operations == 2, "plan operation count is wrong");
    require(execution_plan.compiled_steps == 2, "Bell plan step count is wrong");
    require(execution_plan.active_qubits == 2, "Bell plan active qubits are wrong");
    require(execution_plan.estimated_state_bytes == 64, "Bell plan memory is wrong");
}

void test_semantic_identity() {
    qupy::Program first(2);
    first = qupy::h(first, 0);
    first = qupy::cx(first, 0, 1);

    qupy::Program second(2);
    second = qupy::h(second, 0);
    second = qupy::cx(second, 0, 1);

    require(first.canonical_text() == second.canonical_text(), "canonical IR is not deterministic");
    require(first.fingerprint() == second.fingerprint(), "program fingerprint is not deterministic");
    require(first.fingerprint().size() == 64, "program fingerprint width is wrong");
    require(
        first.fingerprint() == "ab7840ba9d0cd5353fe9e66c9100b195a8f5ad566f13e82f8d775e350f7e8009",
        "program SHA-256 fingerprint is wrong"
    );

    const qupy::Program changed = qupy::x(second, 1);
    require(first.fingerprint() != changed.fingerprint(), "different programs share a fingerprint");

    const qupy::Target target = qupy::native_target();
    require(target.state_access, "native target must report state access");
    require(!target.dynamic_control, "native target must not report dynamic control");
    require(target.fingerprint() == qupy::native_target().fingerprint(), "target fingerprint changed");

    const auto sample_plan = qupy::plan(first, qupy::ResultMode::Sample);
    const auto state_plan = qupy::plan(first, qupy::ResultMode::StateVector);
    require(sample_plan.program_fingerprint == first.fingerprint(), "plan program fingerprint is wrong");
    require(sample_plan.target_fingerprint == target.fingerprint(), "plan target fingerprint is wrong");
    require(sample_plan.result_mode == qupy::ResultMode::Sample, "plan result mode is wrong");
    require(sample_plan.cache_key != state_plan.cache_key, "different result modes share a cache key");

    const auto z0 = qupy::expectation_plan(first, qupy::pauli_z(0));
    const auto z1 = qupy::expectation_plan(first, qupy::pauli_z(1));
    require(z0.cache_key != z1.cache_key, "different observables share a cache key");
}

void test_compiler_fusion() {
    qupy::Program program(2);
    program = qupy::h(program, 0);
    program = qupy::rx(program, 0.2, 0);
    program = qupy::rz(program, -0.4, 0);
    program = qupy::x(program, 1);
    program = qupy::ry(program, 0.3, 1);
    const auto execution_plan = qupy::plan(program, qupy::ResultMode::StateVector);
    require(execution_plan.original_operations == 5, "fusion input count is wrong");
    require(execution_plan.compiled_steps == 2, "single-qubit fusion did not reduce steps");
}

void test_expectation_lightcone() {
    qupy::Program program(100);
    program = qupy::h(program, 0);
    program = qupy::x(program, 98);
    program = qupy::ry(program, 0.7, 99);
    const auto execution_plan = qupy::expectation_plan(program, qupy::pauli_z(0));
    require(execution_plan.method == "statevector-lightcone", "lightcone method not selected");
    require(execution_plan.active_qubits == 1, "lightcone retained unrelated qubits");
    require(execution_plan.estimated_state_bytes == 32, "lightcone memory estimate is wrong");
    const auto result = qupy::expectation(program, qupy::pauli_z(0));
    require(std::abs(result.value) <= kTolerance, "lightcone expectation is wrong");
    require(result.active_qubits == 1, "expectation metadata has wrong active qubit count");
}

void test_probabilities_and_variance() {
    qupy::Program program(2);
    program = qupy::h(program, 0);
    program = qupy::cx(program, 0, 1);

    const auto probabilities = qupy::probabilities(program);
    require(probabilities.values.size() == 4, "probability dimension is wrong");
    require(std::abs(probabilities.values[0] - 0.5) <= kTolerance, "P(00) is wrong");
    require(std::abs(probabilities.values[1]) <= kTolerance, "P(01) is wrong");
    require(std::abs(probabilities.values[2]) <= kTolerance, "P(10) is wrong");
    require(std::abs(probabilities.values[3] - 0.5) <= kTolerance, "P(11) is wrong");

    const auto bell_variance = qupy::variance(program, qupy::pauli_z(0));
    require(std::abs(bell_variance.value - 1.0) <= kTolerance, "Bell Z variance is wrong");
    require(bell_variance.active_qubits == 2, "variance active-qubit metadata is wrong");

    qupy::Program basis(1);
    basis = qupy::x(basis, 0);
    const auto basis_variance = qupy::variance(basis, qupy::pauli_z(0));
    require(std::abs(basis_variance.value) <= kTolerance, "basis-state variance is not zero");

    const auto expectation_key = qupy::expectation_plan(program, qupy::pauli_z(0)).cache_key;
    const auto variance_key = qupy::variance_plan(program, qupy::pauli_z(0)).cache_key;
    require(expectation_key != variance_key, "expectation and variance share a cache key");
}

void test_validation() {
    bool rejected = false;
    try {
        static_cast<void>(qupy::h(qupy::Program(2), 2));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "invalid qubit index was not rejected");

    rejected = false;
    try {
        static_cast<void>(qupy::cx(qupy::Program(2), 0, 0));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "duplicate two-qubit operand was not rejected");

    rejected = false;
    try {
        static_cast<void>(qupy::plan(qupy::Program(1), qupy::ResultMode::Expectation));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "observable-dependent generic plan was not rejected");
}

}  // namespace

int main() {
    try {
        require(std::string(qupy::core_language()) == "C++20", "core language is wrong");
        require(std::string(qupy::core_version()) == "0.3.0a0", "core version is wrong");
        require(qupy::ir_version() == 1U, "IR version is wrong");
        test_bell_state();
        test_rotation_and_pauli_gates();
        test_two_qubit_gates();
        test_results_and_planner();
        test_semantic_identity();
        test_compiler_fusion();
        test_expectation_lightcone();
        test_probabilities_and_variance();
        test_validation();
        std::cout << "QuPy native core tests: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "QuPy native core tests: FAIL: " << error.what() << '\n';
        return 1;
    }
}
