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

    const auto execution_plan = qupy::plan(program, qupy::ResultMode::Sample);
    require(execution_plan.exact, "native plan must be exact");
    require(execution_plan.threads >= 1, "native plan reported no threads");
    require(execution_plan.original_operations == 2, "plan operation count is wrong");
    require(execution_plan.compiled_steps == 2, "Bell plan step count is wrong");
    require(execution_plan.active_qubits == 2, "Bell plan active qubits are wrong");
    require(execution_plan.estimated_state_bytes == 64, "Bell plan memory is wrong");
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
}

}  // namespace

int main() {
    try {
        require(std::string(qupy::core_language()) == "C++20", "core language is wrong");
        test_bell_state();
        test_rotation_and_pauli_gates();
        test_two_qubit_gates();
        test_results_and_planner();
        test_compiler_fusion();
        test_expectation_lightcone();
        test_validation();
        std::cout << "QuPy native core tests: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "QuPy native core tests: FAIL: " << error.what() << '\n';
        return 1;
    }
}
