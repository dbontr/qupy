#include "qupy/trajectory.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
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

qupy::Observable z_observable(std::size_t qubit) {
    return qupy::Observable({
        qupy::PauliTerm(1.0, {{qubit, qupy::Pauli::Z}}),
    });
}

void test_deterministic_unitary_and_noise() {
    qupy::Program program(1U);
    program = qupy::x(program, 0U);
    const qupy::NoisyProgram deterministic(
        program,
        {{1U, qupy::bit_flip(0U, 1.0)}}
    );
    const qupy::TrajectoryBatch result = qupy::trajectory_expectations(
        deterministic, {z_observable(0U)}, 32U, 7U
    );

    require(result.observable_count == 1U, "trajectory observable count mismatch");
    require(result.trajectories == 32U, "trajectory count mismatch");
    require(result.seed == 7U, "trajectory seed mismatch");
    require(result.state_bytes == 2U * sizeof(qupy::Complex), "trajectory state byte count mismatch");
    require(!result.exact, "trajectory estimate was marked exact");
    require(result.backend == "native-cpu", "trajectory backend mismatch");
    require(result.method == "quantum-trajectory", "trajectory method mismatch");
    require(std::abs(result.values.front() - 1.0) < 1e-12, "deterministic trajectory value mismatch");
    require(result.standard_errors.front() == 0.0, "deterministic trajectory error was nonzero");
}

void test_noise_insertion_order() {
    qupy::Program program(1U);
    program = qupy::x(program, 0U);
    const qupy::NoisyProgram before_gate(
        program,
        {{0U, qupy::bit_flip(0U, 1.0)}}
    );
    const auto result = qupy::trajectory_expectations(
        before_gate, {z_observable(0U)}, 8U, 11U
    );
    require(std::abs(result.values.front() - 1.0) < 1e-12, "noise insertion point ordering mismatch");
}

void test_amplitude_damping_converges_and_replays() {
    qupy::Program excited(1U);
    excited = qupy::x(excited, 0U);
    const double gamma = 0.25;
    const qupy::NoisyProgram noisy(
        excited,
        {{1U, qupy::amplitude_damping(0U, gamma)}}
    );
    const qupy::TrajectoryBatch first = qupy::trajectory_expectations(
        noisy, {z_observable(0U)}, 20000U, 123456U
    );
    const qupy::TrajectoryBatch replay = qupy::trajectory_expectations(
        noisy, {z_observable(0U)}, 20000U, 123456U
    );

    const double expected = 2.0 * gamma - 1.0;
    require(std::abs(first.values.front() - expected) < 0.03, "trajectory damping estimate did not converge");
    require(first.standard_errors.front() > 0.0, "stochastic trajectory error was not reported");
    require(first.standard_errors.front() < 0.02, "trajectory standard error is unexpectedly large");
    require(first.values == replay.values, "fixed-seed trajectory replay changed values");
    require(first.standard_errors == replay.standard_errors, "fixed-seed trajectory replay changed errors");
}

void test_custom_kraus_and_multiple_observables() {
    const qupy::NoiseChannel reset_to_zero = qupy::kraus_channel(
        0U,
        {
            {1.0, 0.0, 0.0, 0.0},
            {0.0, 1.0, 0.0, 0.0},
        }
    );
    qupy::Program excited(1U);
    excited = qupy::x(excited, 0U);
    const qupy::NoisyProgram noisy(excited, {{1U, reset_to_zero}});
    const qupy::Observable identity({qupy::PauliTerm(2.0, {})});
    const auto result = qupy::trajectory_expectations(
        noisy, {z_observable(0U), identity}, 16U, 5U, "cpu"
    );

    require(result.values.size() == 2U, "trajectory batch shape mismatch");
    require(std::abs(result.values[0] - 1.0) < 1e-12, "custom Kraus trajectory mismatch");
    require(std::abs(result.values[1] - 2.0) < 1e-12, "trajectory identity observable mismatch");
    require(result.standard_errors[0] == 0.0, "deterministic custom Kraus error was nonzero");
    require(result.standard_errors[1] == 0.0, "constant observable error was nonzero");
}

void test_single_trajectory_and_validation() {
    const qupy::NoisyProgram clean(qupy::Program(1U), {});
    const auto single = qupy::trajectory_expectations(
        clean, {z_observable(0U)}, 1U, 9U, "native-cpu"
    );
    require(std::abs(single.values.front() - 1.0) < 1e-12, "single trajectory value mismatch");
    require(std::isnan(single.standard_errors.front()), "single trajectory error should be undefined");

    require_invalid(
        [&] { static_cast<void>(qupy::trajectory_expectations(clean, {z_observable(0U)}, 0U, 1U)); },
        "zero trajectory count was accepted"
    );
    require_invalid(
        [&] { static_cast<void>(qupy::trajectory_expectations(clean, {}, 8U, 1U)); },
        "empty trajectory observable set was accepted"
    );
    require_invalid(
        [&] {
            static_cast<void>(qupy::trajectory_expectations(
                clean, {z_observable(0U)}, 8U, 1U, "native-cuda"
            ));
        },
        "unsupported trajectory backend was accepted"
    );
    require_invalid(
        [&] { static_cast<void>(qupy::trajectory_expectations(clean, {z_observable(1U)}, 8U, 1U)); },
        "out-of-range trajectory observable was accepted"
    );
}

}  // namespace

int main() {
    test_deterministic_unitary_and_noise();
    test_noise_insertion_order();
    test_amplitude_damping_converges_and_replays();
    test_custom_kraus_and_multiple_observables();
    test_single_trajectory_and_validation();
    return 0;
}
