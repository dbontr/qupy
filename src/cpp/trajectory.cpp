#include "qupy/trajectory.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qupy {
namespace {

using Matrix2 = std::array<Complex, 4>;
constexpr double kHermitianTolerance = 1e-12;

[[nodiscard]] std::size_t checked_dimension(std::size_t num_qubits) {
    if (num_qubits >= std::numeric_limits<std::size_t>::digits) {
        throw std::length_error("trajectory qubit count exceeds native address space");
    }
    return std::size_t{1} << num_qubits;
}

[[nodiscard]] Matrix2 gate_matrix(const Operation& operation) {
    const double inv_sqrt_two = 1.0 / std::sqrt(2.0);
    switch (operation.code) {
    case OperationCode::H:
        return {inv_sqrt_two, inv_sqrt_two, inv_sqrt_two, -inv_sqrt_two};
    case OperationCode::X:
        return {0.0, 1.0, 1.0, 0.0};
    case OperationCode::Y:
        return {0.0, Complex{0.0, -1.0}, Complex{0.0, 1.0}, 0.0};
    case OperationCode::Z:
        return {1.0, 0.0, 0.0, -1.0};
    case OperationCode::RX: {
        const double half = operation.parameters.at(0) / 2.0;
        const Complex sine{0.0, -std::sin(half)};
        return {std::cos(half), sine, sine, std::cos(half)};
    }
    case OperationCode::RY: {
        const double half = operation.parameters.at(0) / 2.0;
        return {std::cos(half), -std::sin(half), std::sin(half), std::cos(half)};
    }
    case OperationCode::RZ: {
        const double half = operation.parameters.at(0) / 2.0;
        return {std::polar(1.0, -half), 0.0, 0.0, std::polar(1.0, half)};
    }
    case OperationCode::CX:
    case OperationCode::CZ:
    case OperationCode::SWAP:
        break;
    }
    throw std::invalid_argument("operation is not a single-qubit gate");
}

void apply_single_state(
    std::vector<Complex>& state,
    const Matrix2& matrix,
    std::size_t qubit
) {
    const std::size_t bit = std::size_t{1} << qubit;
    const std::size_t span = bit << 1U;
    for (std::size_t base = 0; base < state.size(); base += span) {
        for (std::size_t offset = 0; offset < bit; ++offset) {
            const std::size_t zero = base + offset;
            const std::size_t one = zero + bit;
            const Complex a = state[zero];
            const Complex b = state[one];
            state[zero] = matrix[0] * a + matrix[1] * b;
            state[one] = matrix[2] * a + matrix[3] * b;
        }
    }
}

void apply_two_state(std::vector<Complex>& state, const Operation& operation) {
    const std::size_t first_bit = std::size_t{1} << operation.qubits.at(0);
    const std::size_t second_bit = std::size_t{1} << operation.qubits.at(1);
    if (operation.code == OperationCode::CZ) {
        for (std::size_t index = 0; index < state.size(); ++index) {
            if ((index & first_bit) != 0U && (index & second_bit) != 0U) {
                state[index] = -state[index];
            }
        }
        return;
    }
    if (operation.code == OperationCode::CX) {
        for (std::size_t index = 0; index < state.size(); ++index) {
            if ((index & first_bit) != 0U && (index & second_bit) == 0U) {
                std::swap(state[index], state[index | second_bit]);
            }
        }
        return;
    }
    if (operation.code == OperationCode::SWAP) {
        for (std::size_t index = 0; index < state.size(); ++index) {
            const bool first = (index & first_bit) != 0U;
            const bool second = (index & second_bit) != 0U;
            if (first && !second) {
                const std::size_t other = (index ^ first_bit) | second_bit;
                std::swap(state[index], state[other]);
            }
        }
        return;
    }
    throw std::invalid_argument("operation is not a supported two-qubit gate");
}

void apply_operation_state(std::vector<Complex>& state, const Operation& operation) {
    if (operation.qubits.size() == 1U) {
        apply_single_state(state, gate_matrix(operation), operation.qubits.front());
    } else {
        apply_two_state(state, operation);
    }
}

[[nodiscard]] Matrix2 scaled(const Matrix2& matrix, double factor) {
    Matrix2 result = matrix;
    for (Complex& value : result) {
        value *= factor;
    }
    return result;
}

[[nodiscard]] std::vector<Matrix2> noise_kraus_operators(const NoiseChannel& channel) {
    const Matrix2 identity{1.0, 0.0, 0.0, 1.0};
    const Matrix2 x_matrix{0.0, 1.0, 1.0, 0.0};
    const Matrix2 y_matrix{0.0, Complex{0.0, -1.0}, Complex{0.0, 1.0}, 0.0};
    const Matrix2 z_matrix{1.0, 0.0, 0.0, -1.0};
    switch (channel.code) {
    case NoiseChannelCode::BitFlip: {
        const double probability = channel.parameters.at(0);
        return {
            scaled(identity, std::sqrt(1.0 - probability)),
            scaled(x_matrix, std::sqrt(probability)),
        };
    }
    case NoiseChannelCode::PhaseFlip: {
        const double probability = channel.parameters.at(0);
        return {
            scaled(identity, std::sqrt(1.0 - probability)),
            scaled(z_matrix, std::sqrt(probability)),
        };
    }
    case NoiseChannelCode::Depolarizing: {
        const double probability = channel.parameters.at(0);
        const double error_scale = std::sqrt(probability / 3.0);
        return {
            scaled(identity, std::sqrt(1.0 - probability)),
            scaled(x_matrix, error_scale),
            scaled(y_matrix, error_scale),
            scaled(z_matrix, error_scale),
        };
    }
    case NoiseChannelCode::AmplitudeDamping: {
        const double gamma = channel.parameters.at(0);
        return {
            Matrix2{1.0, 0.0, 0.0, std::sqrt(1.0 - gamma)},
            Matrix2{0.0, std::sqrt(gamma), 0.0, 0.0},
        };
    }
    case NoiseChannelCode::PhaseDamping: {
        const double gamma = channel.parameters.at(0);
        return {
            Matrix2{1.0, 0.0, 0.0, std::sqrt(1.0 - gamma)},
            Matrix2{0.0, 0.0, 0.0, std::sqrt(gamma)},
        };
    }
    case NoiseChannelCode::Pauli: {
        const double probability_x = channel.parameters.at(0);
        const double probability_y = channel.parameters.at(1);
        const double probability_z = channel.parameters.at(2);
        const double identity_probability =
            1.0 - probability_x - probability_y - probability_z;
        return {
            scaled(identity, std::sqrt(identity_probability)),
            scaled(x_matrix, std::sqrt(probability_x)),
            scaled(y_matrix, std::sqrt(probability_y)),
            scaled(z_matrix, std::sqrt(probability_z)),
        };
    }
    case NoiseChannelCode::Kraus: {
        if (channel.kraus_count == 0U ||
            channel.kraus_operators.size() != channel.kraus_count * 4U) {
            throw std::invalid_argument("custom Kraus channel has invalid operator storage");
        }
        std::vector<Matrix2> operators;
        operators.reserve(channel.kraus_count);
        for (std::size_t index = 0U; index < channel.kraus_count; ++index) {
            Matrix2 matrix{};
            std::copy_n(
                channel.kraus_operators.begin() +
                    static_cast<std::ptrdiff_t>(index * 4U),
                4U,
                matrix.begin()
            );
            operators.push_back(matrix);
        }
        return operators;
    }
    }
    throw std::invalid_argument("unknown noise channel");
}

[[nodiscard]] double squared_norm(const std::vector<Complex>& values) {
    return std::accumulate(
        values.begin(),
        values.end(),
        0.0,
        [](double total, const Complex& value) { return total + std::norm(value); }
    );
}

[[nodiscard]] double unit_interval(std::mt19937_64& generator) {
    constexpr double scale = 1.0 / 9007199254740992.0;
    return static_cast<double>(generator() >> 11U) * scale;
}

void apply_noise_trajectory(
    std::vector<Complex>& state,
    const NoiseChannel& channel,
    std::mt19937_64& generator
) {
    const std::vector<Matrix2> operators = noise_kraus_operators(channel);
    std::vector<double> probabilities;
    probabilities.reserve(operators.size());
    std::vector<Complex> scratch;
    scratch.reserve(state.size());
    double total_probability = 0.0;
    for (const Matrix2& matrix : operators) {
        scratch = state;
        apply_single_state(scratch, matrix, channel.qubit);
        double probability = squared_norm(scratch);
        if (probability < 0.0 && probability > -1e-15) {
            probability = 0.0;
        }
        if (!std::isfinite(probability) || probability < 0.0) {
            throw std::runtime_error("trajectory Kraus branch produced an invalid probability");
        }
        probabilities.push_back(probability);
        total_probability += probability;
    }
    if (!std::isfinite(total_probability) || total_probability <= 0.0) {
        throw std::runtime_error("trajectory Kraus probabilities have invalid total weight");
    }

    const double threshold = unit_interval(generator) * total_probability;
    double cumulative = 0.0;
    std::size_t selected = operators.size();
    std::size_t last_nonzero = operators.size();
    for (std::size_t index = 0U; index < probabilities.size(); ++index) {
        const double probability = probabilities[index];
        if (probability <= 0.0) {
            continue;
        }
        last_nonzero = index;
        cumulative += probability;
        if (threshold < cumulative) {
            selected = index;
            break;
        }
    }
    if (selected == operators.size()) {
        selected = last_nonzero;
    }
    if (selected == operators.size() || probabilities[selected] <= 0.0) {
        throw std::runtime_error("trajectory failed to select a nonzero Kraus branch");
    }

    apply_single_state(state, operators[selected], channel.qubit);
    const double inverse_norm = 1.0 / std::sqrt(probabilities[selected]);
    for (Complex& amplitude : state) {
        amplitude *= inverse_norm;
    }
}

[[nodiscard]] Complex pauli_phase(Pauli pauli, bool bit) {
    switch (pauli) {
    case Pauli::I:
    case Pauli::X:
        return 1.0;
    case Pauli::Y:
        return bit ? Complex{0.0, -1.0} : Complex{0.0, 1.0};
    case Pauli::Z:
        return bit ? -1.0 : 1.0;
    }
    throw std::invalid_argument("unknown Pauli operator");
}

void validate_observable(const Observable& value, std::size_t num_qubits) {
    for (const PauliTerm& term : value.terms()) {
        for (const PauliFactor& factor : term.factors()) {
            if (factor.qubit >= num_qubits) {
                throw std::invalid_argument("observable qubit is outside this program");
            }
        }
    }
}

[[nodiscard]] double observable_expectation_from_state(
    const std::vector<Complex>& state,
    const Observable& observable
) {
    Complex expectation{0.0, 0.0};
    for (const PauliTerm& term : observable.terms()) {
        if (term.coefficient() == 0.0) {
            continue;
        }
        for (std::size_t basis = 0U; basis < state.size(); ++basis) {
            std::size_t mapped = basis;
            Complex phase{term.coefficient(), 0.0};
            for (const PauliFactor& factor : term.factors()) {
                const std::size_t bit_mask = std::size_t{1} << factor.qubit;
                const bool bit = (basis & bit_mask) != 0U;
                phase *= pauli_phase(factor.pauli, bit);
                if (factor.pauli == Pauli::X || factor.pauli == Pauli::Y) {
                    mapped ^= bit_mask;
                }
            }
            expectation += std::conj(state[mapped]) * phase * state[basis];
        }
    }
    if (std::abs(expectation.imag()) > kHermitianTolerance) {
        throw std::domain_error("trajectory observable expectation acquired an imaginary part");
    }
    return expectation.real();
}

[[nodiscard]] std::uint64_t random_seed() {
    std::random_device device;
    return (static_cast<std::uint64_t>(device()) << 32U) ^
        static_cast<std::uint64_t>(device());
}

[[nodiscard]] std::vector<const NoiseInstruction*> ordered_noise(const NoisyProgram& noisy) {
    std::vector<const NoiseInstruction*> result;
    result.reserve(noisy.noise().size());
    for (const NoiseInstruction& instruction : noisy.noise()) {
        result.push_back(&instruction);
    }
    std::stable_sort(
        result.begin(),
        result.end(),
        [](const NoiseInstruction* left, const NoiseInstruction* right) {
            return left->after_operation < right->after_operation;
        }
    );
    return result;
}

void execute_trajectory(
    const NoisyProgram& noisy,
    const std::vector<const NoiseInstruction*>& noise,
    std::mt19937_64& generator,
    std::vector<Complex>& state
) {
    const Program& program = noisy.program();
    std::fill(state.begin(), state.end(), Complex{0.0, 0.0});
    state.front() = 1.0;
    std::size_t noise_index = 0U;
    const auto apply_at = [&](std::size_t point) {
        while (noise_index < noise.size() && noise[noise_index]->after_operation == point) {
            apply_noise_trajectory(state, noise[noise_index]->channel, generator);
            ++noise_index;
        }
    };
    apply_at(0U);
    for (std::size_t index = 0U; index < program.operations().size(); ++index) {
        apply_operation_state(state, program.operations()[index]);
        apply_at(index + 1U);
    }
    if (noise_index != noise.size()) {
        throw std::logic_error("trajectory noise schedule was not fully consumed");
    }
}

}  // namespace

TrajectoryBatch trajectory_expectations(
    const NoisyProgram& noisy,
    const std::vector<Observable>& observables,
    std::size_t trajectories,
    std::optional<std::uint64_t> seed,
    const std::string& backend
) {
    if (backend != "auto" && backend != "cpu" && backend != "native-cpu") {
        throw std::invalid_argument("quantum trajectories currently support only native-cpu");
    }
    if (trajectories == 0U) {
        throw std::invalid_argument("trajectory count must be positive");
    }
    if (observables.empty()) {
        throw std::invalid_argument("trajectory estimation requires at least one observable");
    }
    const Program& program = noisy.program();
    for (const Observable& observable : observables) {
        validate_observable(observable, program.num_qubits());
    }

    const std::size_t dimension = checked_dimension(program.num_qubits());
    if (dimension > std::numeric_limits<std::size_t>::max() / sizeof(Complex)) {
        throw std::length_error("trajectory state exceeds native address space");
    }
    const std::size_t state_bytes = dimension * sizeof(Complex);
    const std::uint64_t actual_seed = seed.value_or(random_seed());
    std::mt19937_64 generator(actual_seed);
    const std::vector<const NoiseInstruction*> noise = ordered_noise(noisy);

    std::vector<double> means(observables.size(), 0.0);
    std::vector<double> m2(observables.size(), 0.0);
    std::vector<Complex> state(dimension, Complex{0.0, 0.0});
    for (std::size_t trajectory = 0U; trajectory < trajectories; ++trajectory) {
        execute_trajectory(noisy, noise, generator, state);
        const double count = static_cast<double>(trajectory + 1U);
        for (std::size_t index = 0U; index < observables.size(); ++index) {
            const double sample = observable_expectation_from_state(state, observables[index]);
            const double delta = sample - means[index];
            means[index] += delta / count;
            const double delta_after = sample - means[index];
            m2[index] += delta * delta_after;
        }
    }

    std::vector<double> standard_errors(observables.size());
    if (trajectories == 1U) {
        std::fill(
            standard_errors.begin(),
            standard_errors.end(),
            std::numeric_limits<double>::quiet_NaN()
        );
    } else {
        const double denominator = static_cast<double>(trajectories) *
            static_cast<double>(trajectories - 1U);
        for (std::size_t index = 0U; index < observables.size(); ++index) {
            standard_errors[index] = std::sqrt(std::max(0.0, m2[index]) / denominator);
        }
    }

    return {
        std::move(means),
        std::move(standard_errors),
        observables.size(),
        trajectories,
        actual_seed,
        state_bytes,
        false,
        "native-cpu",
        "quantum-trajectory",
    };
}

}  // namespace qupy
