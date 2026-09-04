#include "qupy/advanced.hpp"
#include "qupy/tensor_network.hpp"

#include "cuda_driver.hpp"
#include "distributed.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace qupy {
namespace {

using Matrix2 = std::array<Complex, 4>;
constexpr double kHermitianTolerance = 1e-12;
constexpr std::size_t kAdjointMemoryLimitBytes = std::size_t{1} << 30U;
constexpr std::size_t kReferenceDecoderErrorLimit = 24U;

[[nodiscard]] const char* pauli_name(Pauli pauli) {
    switch (pauli) {
    case Pauli::I: return "I";
    case Pauli::X: return "X";
    case Pauli::Y: return "Y";
    case Pauli::Z: return "Z";
    }
    throw std::invalid_argument("unknown Pauli operator");
}
[[nodiscard]] std::string fingerprint_text(std::string_view text) {
    static constexpr std::array<std::uint32_t, 8> initial = {
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
    };
    static constexpr std::array<std::uint32_t, 64> round = {
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U,
        0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
        0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U,
        0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
        0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
        0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
        0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU,
        0x5b9cca4fU, 0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
    };
    if (text.size() > std::numeric_limits<std::uint64_t>::max() / 8U) {
        throw std::length_error("fingerprint input is too large");
    }
    const std::uint64_t bit_length = static_cast<std::uint64_t>(text.size()) * 8U;
    std::vector<std::uint8_t> message(text.begin(), text.end());
    message.push_back(0x80U);
    while ((message.size() % 64U) != 56U) {
        message.push_back(0U);
    }
    for (int shift = 56; shift >= 0; shift -= 8) {
        message.push_back(static_cast<std::uint8_t>(bit_length >> shift));
    }
    auto state = initial;
    for (std::size_t offset = 0; offset < message.size(); offset += 64U) {
        std::array<std::uint32_t, 64> schedule{};
        for (std::size_t index = 0; index < 16U; ++index) {
            const std::size_t base = offset + index * 4U;
            schedule[index] =
                (static_cast<std::uint32_t>(message[base]) << 24U) |
                (static_cast<std::uint32_t>(message[base + 1U]) << 16U) |
                (static_cast<std::uint32_t>(message[base + 2U]) << 8U) |
                static_cast<std::uint32_t>(message[base + 3U]);
        }
        for (std::size_t index = 16U; index < schedule.size(); ++index) {
            const std::uint32_t s0 = std::rotr(schedule[index - 15U], 7) ^
                                     std::rotr(schedule[index - 15U], 18) ^
                                     (schedule[index - 15U] >> 3U);
            const std::uint32_t s1 = std::rotr(schedule[index - 2U], 17) ^
                                     std::rotr(schedule[index - 2U], 19) ^
                                     (schedule[index - 2U] >> 10U);
            schedule[index] = schedule[index - 16U] + s0 + schedule[index - 7U] + s1;
        }
        std::uint32_t a = state[0];
        std::uint32_t b = state[1];
        std::uint32_t c = state[2];
        std::uint32_t d = state[3];
        std::uint32_t e = state[4];
        std::uint32_t f = state[5];
        std::uint32_t g = state[6];
        std::uint32_t h = state[7];
        for (std::size_t index = 0; index < schedule.size(); ++index) {
            const std::uint32_t sigma1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
            const std::uint32_t choose = (e & f) ^ ((~e) & g);
            const std::uint32_t temp1 = h + sigma1 + choose + round[index] + schedule[index];
            const std::uint32_t sigma0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
            const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t temp2 = sigma0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
        state[4] += e;
        state[5] += f;
        state[6] += g;
        state[7] += h;
    }
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << std::hex << std::setfill('0');
    for (const std::uint32_t word : state) {
        output << std::setw(8) << word;
    }
    return output.str();
}

[[nodiscard]] std::size_t checked_dimension(std::size_t num_qubits) {
    if (num_qubits >= std::numeric_limits<std::size_t>::digits) {
        throw std::length_error("qubit count exceeds native address space");
    }
    return std::size_t{1} << num_qubits;
}
void validate_probability(double value, const char* name) {
    if (!std::isfinite(value) || value < 0.0 || value > 1.0) {
        throw std::invalid_argument(std::string(name) + " must be finite and in [0, 1]");
    }
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
[[nodiscard]] Matrix2 dagger(const Matrix2& matrix) {
    return {
        std::conj(matrix[0]), std::conj(matrix[2]),
        std::conj(matrix[1]), std::conj(matrix[3]),
    };
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

void apply_adjoint_operation_state(std::vector<Complex>& state, const Operation& operation) {
    if (operation.qubits.size() == 1U) {
        apply_single_state(state, dagger(gate_matrix(operation)), operation.qubits.front());
    } else {
        apply_two_state(state, operation);
    }
}
[[nodiscard]] Matrix2 derivative_matrix(const Operation& operation) {
    const double half = operation.parameters.at(0) / 2.0;
    switch (operation.code) {
    case OperationCode::RX: {
        const double diagonal = -0.5 * std::sin(half);
        const Complex off_diagonal{0.0, -0.5 * std::cos(half)};
        return {diagonal, off_diagonal, off_diagonal, diagonal};
    }
    case OperationCode::RY:
        return {
            -0.5 * std::sin(half), -0.5 * std::cos(half),
            0.5 * std::cos(half), -0.5 * std::sin(half),
        };
    case OperationCode::RZ:
        return {
            Complex{0.0, -0.5} * std::polar(1.0, -half), 0.0,
            0.0, Complex{0.0, 0.5} * std::polar(1.0, half),
        };
    default:
        break;
    }
    throw std::invalid_argument("adjoint gradients require RX, RY, or RZ parameter slots");
}

[[nodiscard]] bool is_rotation(OperationCode code) noexcept {
    return code == OperationCode::RX || code == OperationCode::RY || code == OperationCode::RZ;
}

void validate_slots(const Program& program, const std::vector<ParameterSlot>& slots) {
    std::set<std::pair<std::size_t, std::size_t>> identities;
    for (const ParameterSlot& slot : slots) {
        if (slot.operation_index >= program.operations().size()) {
            throw std::invalid_argument("parameter slot operation index is outside the program");
        }
        const Operation& operation = program.operations()[slot.operation_index];
        if (!is_rotation(operation.code) || slot.parameter_index != 0U) {
            throw std::invalid_argument("gradient slots must select RX, RY, or RZ parameters");
        }
        if (!identities.insert({slot.operation_index, slot.parameter_index}).second) {
            throw std::invalid_argument("gradient parameter slots must be unique");
        }
    }
}

[[nodiscard]] bool term_commutes(const PauliTerm& left, const PauliTerm& right) {
    std::map<std::size_t, Pauli> left_by_qubit;
    for (const PauliFactor& factor : left.factors()) {
        left_by_qubit[factor.qubit] = factor.pauli;
    }
    std::size_t anti_commuting_sites = 0U;
    for (const PauliFactor& factor : right.factors()) {
        const auto found = left_by_qubit.find(factor.qubit);
        if (found == left_by_qubit.end() || found->second == Pauli::I ||
            factor.pauli == Pauli::I || found->second == factor.pauli) {
            continue;
        }
        ++anti_commuting_sites;
    }
    return (anti_commuting_sites % 2U) == 0U;
}

struct ObservablePauliFrame {
    std::vector<std::uint8_t> x;
    std::vector<std::uint8_t> z;
    bool negative = false;
};

void frame_h(ObservablePauliFrame& frame, std::size_t qubit) {
    if (frame.x[qubit] != 0U && frame.z[qubit] != 0U) frame.negative = !frame.negative;
    std::swap(frame.x[qubit], frame.z[qubit]);
}

void frame_x(ObservablePauliFrame& frame, std::size_t qubit) {
    if (frame.z[qubit] != 0U) frame.negative = !frame.negative;
}

void frame_y(ObservablePauliFrame& frame, std::size_t qubit) {
    if ((frame.x[qubit] != 0U) != (frame.z[qubit] != 0U)) frame.negative = !frame.negative;
}

void frame_z(ObservablePauliFrame& frame, std::size_t qubit) {
    if (frame.x[qubit] != 0U) frame.negative = !frame.negative;
}

void frame_cx(ObservablePauliFrame& frame, std::size_t control, std::size_t target) {
    const bool phase_flip = frame.x[control] != 0U && frame.z[target] != 0U &&
        ((frame.x[target] != 0U) == (frame.z[control] != 0U));
    if (phase_flip) frame.negative = !frame.negative;
    frame.x[target] ^= frame.x[control];
    frame.z[control] ^= frame.z[target];
}

void frame_cz(ObservablePauliFrame& frame, std::size_t control, std::size_t target) {
    frame_h(frame, target);
    frame_cx(frame, control, target);
    frame_h(frame, target);
}

[[nodiscard]] bool rich_pauli_propagation_supported(const Program& program) {
    return std::all_of(program.operations().begin(), program.operations().end(), [](const Operation& op) {
        return op.code != OperationCode::RX && op.code != OperationCode::RY &&
            op.code != OperationCode::RZ;
    });
}

[[nodiscard]] double propagated_pauli_expectation(
    const Program& program,
    const std::vector<PauliFactor>& factors
) {
    ObservablePauliFrame frame{
        std::vector<std::uint8_t>(program.num_qubits(), 0U),
        std::vector<std::uint8_t>(program.num_qubits(), 0U),
        false,
    };
    for (const PauliFactor& factor : factors) {
        switch (factor.pauli) {
        case Pauli::I: break;
        case Pauli::X: frame.x[factor.qubit] = 1U; break;
        case Pauli::Y:
            frame.x[factor.qubit] = 1U;
            frame.z[factor.qubit] = 1U;
            break;
        case Pauli::Z: frame.z[factor.qubit] = 1U; break;
        }
    }
    for (std::size_t index = program.operations().size(); index > 0U; --index) {
        const Operation& operation = program.operations()[index - 1U];
        switch (operation.code) {
        case OperationCode::H: frame_h(frame, operation.qubits[0]); break;
        case OperationCode::X: frame_x(frame, operation.qubits[0]); break;
        case OperationCode::Y: frame_y(frame, operation.qubits[0]); break;
        case OperationCode::Z: frame_z(frame, operation.qubits[0]); break;
        case OperationCode::CX: frame_cx(frame, operation.qubits[0], operation.qubits[1]); break;
        case OperationCode::CZ: frame_cz(frame, operation.qubits[0], operation.qubits[1]); break;
        case OperationCode::SWAP:
            std::swap(frame.x[operation.qubits[0]], frame.x[operation.qubits[1]]);
            std::swap(frame.z[operation.qubits[0]], frame.z[operation.qubits[1]]);
            break;
        case OperationCode::RX:
        case OperationCode::RY:
        case OperationCode::RZ:
            throw std::logic_error("non-Clifford operation reached rich Pauli propagation");
        }
    }
    if (std::any_of(frame.x.begin(), frame.x.end(), [](std::uint8_t bit) { return bit != 0U; })) {
        return 0.0;
    }
    return frame.negative ? -1.0 : 1.0;
}

[[nodiscard]] double propagated_observable_expectation(
    const Program& program,
    const Observable& value
) {
    return std::accumulate(
        value.terms().begin(), value.terms().end(), 0.0,
        [&](double total, const PauliTerm& term) {
            return total + term.coefficient() * propagated_pauli_expectation(program, term.factors());
        }
    );
}

struct PauliProduct {
    Complex phase;
    std::vector<PauliFactor> factors;
};

[[nodiscard]] std::pair<Pauli, Complex> multiply_pauli(Pauli left, Pauli right) {
    if (left == Pauli::I) return {right, 1.0};
    if (right == Pauli::I) return {left, 1.0};
    if (left == right) return {Pauli::I, 1.0};
    if (left == Pauli::X && right == Pauli::Y) return {Pauli::Z, Complex{0.0, 1.0}};
    if (left == Pauli::Y && right == Pauli::X) return {Pauli::Z, Complex{0.0, -1.0}};
    if (left == Pauli::Y && right == Pauli::Z) return {Pauli::X, Complex{0.0, 1.0}};
    if (left == Pauli::Z && right == Pauli::Y) return {Pauli::X, Complex{0.0, -1.0}};
    if (left == Pauli::Z && right == Pauli::X) return {Pauli::Y, Complex{0.0, 1.0}};
    if (left == Pauli::X && right == Pauli::Z) return {Pauli::Y, Complex{0.0, -1.0}};
    throw std::logic_error("invalid Pauli product");
}

[[nodiscard]] PauliProduct multiply_pauli_terms(const PauliTerm& left, const PauliTerm& right) {
    std::map<std::size_t, Pauli> factors;
    for (const PauliFactor& factor : left.factors()) factors.emplace(factor.qubit, factor.pauli);
    Complex phase{1.0, 0.0};
    for (const PauliFactor& factor : right.factors()) {
        const auto found = factors.find(factor.qubit);
        if (found == factors.end()) {
            factors.emplace(factor.qubit, factor.pauli);
            continue;
        }
        const auto [product, local_phase] = multiply_pauli(found->second, factor.pauli);
        phase *= local_phase;
        if (product == Pauli::I) factors.erase(found);
        else found->second = product;
    }
    std::vector<PauliFactor> result;
    result.reserve(factors.size());
    for (const auto& [qubit, pauli_value] : factors) result.push_back({qubit, pauli_value});
    return {phase, std::move(result)};
}

[[nodiscard]] detail::CudaPauliMask cuda_pauli_mask(
    const std::vector<PauliFactor>& factors
) {
    detail::CudaPauliMask result{0U, 0U, 0U};
    for (const PauliFactor& factor : factors) {
        if (factor.qubit >= std::numeric_limits<std::uint64_t>::digits) {
            throw std::length_error("CUDA Pauli mask exceeds native mask width");
        }
        const std::uint64_t bit = std::uint64_t{1} << factor.qubit;
        switch (factor.pauli) {
        case Pauli::I: break;
        case Pauli::X: result.flip_mask |= bit; break;
        case Pauli::Y:
            result.flip_mask |= bit;
            result.sign_mask |= bit;
            result.y_phase = (result.y_phase + 1U) & 3U;
            break;
        case Pauli::Z: result.sign_mask |= bit; break;
        }
    }
    return result;
}

class CudaPauliQuery {
public:
    [[nodiscard]] std::size_t add(const std::vector<PauliFactor>& factors) {
        const detail::CudaPauliMask mask = cuda_pauli_mask(factors);
        const std::array<std::uint64_t, 3> key = {
            mask.flip_mask, mask.sign_mask, static_cast<std::uint64_t>(mask.y_phase),
        };
        const auto found = indices_.find(key);
        if (found != indices_.end()) {
            return found->second;
        }
        const std::size_t index = masks_.size();
        indices_.emplace(key, index);
        masks_.push_back(mask);
        return index;
    }

    [[nodiscard]] std::vector<Complex> evaluate(
        const Program& program,
        std::size_t device
    ) const {
        if (masks_.empty()) {
            if (!detail::cuda_available(device)) {
                throw std::runtime_error(detail::cuda_unavailable_reason(device));
            }
            return {};
        }
        return detail::cuda_pauli_expectations(program, masks_, device);
    }

    [[nodiscard]] std::size_t size() const noexcept { return masks_.size(); }

private:
    std::map<std::array<std::uint64_t, 3>, std::size_t> indices_;
    std::vector<detail::CudaPauliMask> masks_;
};

struct CudaLinearTerm {
    std::size_t index;
    Complex coefficient;
};

[[nodiscard]] std::vector<CudaLinearTerm> index_observable(
    CudaPauliQuery& query,
    const Observable& value
) {
    std::vector<CudaLinearTerm> result;
    result.reserve(value.terms().size());
    for (const PauliTerm& term : value.terms()) {
        result.push_back({query.add(term.factors()), Complex{term.coefficient(), 0.0}});
    }
    return result;
}

[[nodiscard]] Complex propagated_observable_product(
    const Program& program,
    const Observable& left,
    const Observable& right
) {
    Complex result{0.0, 0.0};
    for (const PauliTerm& left_term : left.terms()) {
        for (const PauliTerm& right_term : right.terms()) {
            const PauliProduct product = multiply_pauli_terms(left_term, right_term);
            result += left_term.coefficient() * right_term.coefficient() * product.phase *
                propagated_pauli_expectation(program, product.factors);
        }
    }
    return result;
}

[[nodiscard]] std::vector<CudaLinearTerm> index_observable_product(
    CudaPauliQuery& query,
    const Observable& left,
    const Observable& right
) {
    std::vector<CudaLinearTerm> result;
    if (left.terms().size() > 0U &&
        right.terms().size() > std::numeric_limits<std::size_t>::max() / left.terms().size()) {
        throw std::length_error("observable product term count exceeds native range");
    }
    result.reserve(left.terms().size() * right.terms().size());
    for (const PauliTerm& left_term : left.terms()) {
        for (const PauliTerm& right_term : right.terms()) {
            PauliProduct product = multiply_pauli_terms(left_term, right_term);
            result.push_back({
                query.add(product.factors),
                left_term.coefficient() * right_term.coefficient() * product.phase,
            });
        }
    }
    return result;
}

[[nodiscard]] Complex evaluate_cuda_linear(
    const std::vector<Complex>& values,
    const std::vector<CudaLinearTerm>& terms
) {
    Complex result{0.0, 0.0};
    for (const CudaLinearTerm& term : terms) {
        if (term.index >= values.size()) {
            throw std::logic_error("CUDA Pauli query returned an incomplete result");
        }
        result += term.coefficient * values[term.index];
    }
    return result;
}

[[nodiscard]] double hermitian_cuda_value(Complex value, std::string_view label) {
    if (std::abs(value.imag()) > kHermitianTolerance) {
        throw std::domain_error(std::string(label) + " acquired an invalid imaginary part");
    }
    return value.real();
}

class MpiPauliQuery {
public:
    [[nodiscard]] std::size_t add(const std::vector<PauliFactor>& factors) {
        std::vector<std::pair<std::size_t, std::uint8_t>> key;
        key.reserve(factors.size());
        for (const PauliFactor& factor : factors) {
            key.emplace_back(factor.qubit, static_cast<std::uint8_t>(factor.pauli));
        }
        const auto found = indices_.find(key);
        if (found != indices_.end()) {
            return found->second;
        }
        const std::size_t index = factors_.size();
        indices_.emplace(std::move(key), index);
        factors_.push_back(factors);
        return index;
    }

    [[nodiscard]] std::vector<Complex> evaluate(const Program& program) const {
        return detail::distributed_pauli_expectations(program, factors_);
    }

    [[nodiscard]] std::size_t size() const noexcept { return factors_.size(); }

private:
    std::map<std::vector<std::pair<std::size_t, std::uint8_t>>, std::size_t> indices_;
    std::vector<std::vector<PauliFactor>> factors_;
};

struct MpiLinearTerm {
    std::size_t index;
    Complex coefficient;
};

[[nodiscard]] std::vector<MpiLinearTerm> index_mpi_observable(
    MpiPauliQuery& query,
    const Observable& value
) {
    std::vector<MpiLinearTerm> result;
    result.reserve(value.terms().size());
    for (const PauliTerm& term : value.terms()) {
        result.push_back({query.add(term.factors()), Complex{term.coefficient(), 0.0}});
    }
    return result;
}

[[nodiscard]] std::vector<MpiLinearTerm> index_mpi_observable_product(
    MpiPauliQuery& query,
    const Observable& left,
    const Observable& right
) {
    std::vector<MpiLinearTerm> result;
    if (left.terms().size() > 0U &&
        right.terms().size() > std::numeric_limits<std::size_t>::max() / left.terms().size()) {
        throw std::length_error("observable product term count exceeds native range");
    }
    result.reserve(left.terms().size() * right.terms().size());
    for (const PauliTerm& left_term : left.terms()) {
        for (const PauliTerm& right_term : right.terms()) {
            PauliProduct product = multiply_pauli_terms(left_term, right_term);
            result.push_back({
                query.add(product.factors),
                left_term.coefficient() * right_term.coefficient() * product.phase,
            });
        }
    }
    return result;
}

[[nodiscard]] Complex evaluate_mpi_linear(
    const std::vector<Complex>& values,
    const std::vector<MpiLinearTerm>& terms
) {
    Complex result{0.0, 0.0};
    for (const MpiLinearTerm& term : terms) {
        if (term.index >= values.size()) {
            throw std::logic_error("MPI Pauli query returned an incomplete result");
        }
        result += term.coefficient * values[term.index];
    }
    return result;
}

[[nodiscard]] double hermitian_mpi_value(Complex value, std::string_view label) {
    if (std::abs(value.imag()) > kHermitianTolerance) {
        throw std::domain_error(std::string(label) + " acquired an invalid imaginary part");
    }
    return value.real();
}

[[nodiscard]] bool explicit_mpi_backend(const std::string& backend) {
    return backend == "mpi" || backend == "native-mpi";
}

[[nodiscard]] bool explicit_cuda_backend(const std::string& backend) {
    return detail::cuda_backend_device(backend).has_value();
}

[[nodiscard]] bool explicit_default_cuda_backend(const std::string& backend) {
    const std::optional<std::size_t> device = detail::cuda_backend_device(backend);
    return device.has_value() && *device == 0U;
}

[[nodiscard]] std::size_t cuda_device_for_execution(const std::string& backend) noexcept {
    return detail::cuda_backend_device(backend).value_or(0U);
}

[[nodiscard]] std::string cuda_backend_for_execution(const std::string& backend) {
    return detail::cuda_backend_name(cuda_device_for_execution(backend));
}

[[nodiscard]] std::size_t checked_term_sum(std::size_t left, std::size_t right) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        throw std::length_error("observable term evaluation count exceeds native range");
    }
    return left + right;
}

struct ObservableCostWork {
    std::size_t term_evaluations;
    std::size_t state_passes;
};

[[nodiscard]] bool has_observable_cost_work(const ObservableCostWork& work) noexcept {
    return work.term_evaluations != 0U || work.state_passes != 0U;
}

[[nodiscard]] bool cuda_observable_candidate_supported(const Program& program) noexcept {
    if (!detail::cuda_available()) {
        return false;
    }
    try {
        cuda_target().validate(program, ResultMode::StateVector);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

[[nodiscard]] bool needs_cuda_observable_query(
    const Program& program,
    const std::string& backend,
    const PlannerCostModel* cost_model
) noexcept {
    return explicit_cuda_backend(backend) ||
        (backend == "auto" && cost_model != nullptr && cost_model->observable_auto_validated() &&
         cuda_observable_candidate_supported(program));
}

[[nodiscard]] bool use_cuda_observable(
    const Program& program,
    const ObservableCostWork& cpu_work,
    const ObservableCostWork& cuda_work,
    const std::string& backend,
    const PlannerCostModel* cost_model
) {
    if (explicit_cuda_backend(backend)) {
        return true;
    }
    if (backend != "auto" || cost_model == nullptr ||
        !has_observable_cost_work(cpu_work) || !has_observable_cost_work(cuda_work)) {
        return false;
    }
    if (cost_model->observable_auto_validated()) {
        if (!cuda_observable_candidate_supported(program)) {
            return false;
        }
        const ExecutionPlan cpu = plan(program, ResultMode::StateVector, "native-cpu");
        const ExecutionPlan cuda = plan(program, ResultMode::StateVector, "native-cuda");
        return cost_model->predict_observable_ns(
                   cuda, cuda_work.term_evaluations, cuda_work.state_passes
               ) <
               cost_model->predict_observable_ns(
                   cpu, cpu_work.term_evaluations, cpu_work.state_passes
               );
    }
    const ExecutionPlan candidate = plan(program, ResultMode::StateVector, "auto", cost_model);
    return detail::cuda_backend_device(candidate.backend).has_value() &&
        candidate.method == "cuda-statevector";
}

[[nodiscard]] bool use_rich_pauli_propagation(const Program& program, const std::string& backend) {
    return (backend == "auto" || backend == "native-cpu" || backend == "cpu") &&
        rich_pauli_propagation_supported(program);
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
void apply_observable_to_state(
    const std::vector<Complex>& state,
    const Observable& value,
    std::vector<Complex>& output
) {
    output.assign(state.size(), Complex{0.0, 0.0});
    for (const PauliTerm& term : value.terms()) {
        if (term.coefficient() == 0.0) {
            continue;
        }
        for (std::size_t basis = 0; basis < state.size(); ++basis) {
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
            output[mapped] += phase * state[basis];
        }
    }
}

[[nodiscard]] double observable_expectation_from_state(
    const std::vector<Complex>& state,
    const Observable& value
) {
    std::vector<Complex> applied;
    apply_observable_to_state(state, value, applied);
    Complex expectation_value{0.0, 0.0};
    for (std::size_t index = 0; index < state.size(); ++index) {
        expectation_value += std::conj(state[index]) * applied[index];
    }
    if (std::abs(expectation_value.imag()) > kHermitianTolerance) {
        throw std::domain_error("observable expectation is not real; observable is not Hermitian");
    }
    return expectation_value.real();
}

[[nodiscard]] double squared_norm(const std::vector<Complex>& values) {
    return std::accumulate(
        values.begin(), values.end(), 0.0,
        [](double total, const Complex& value) { return total + std::norm(value); }
    );
}
struct ReducedObservableSet {
    Program program;
    std::vector<Observable> observables;
    std::size_t active_qubits;
};

[[nodiscard]] ReducedObservableSet reduce_observables(
    const Program& program,
    const std::vector<Observable>& observables
) {
    for (const Observable& value : observables) {
        validate_observable(value, program.num_qubits());
    }
    std::vector<bool> active(program.num_qubits(), false);
    for (const Observable& value : observables) {
        for (const PauliTerm& term : value.terms()) {
            for (const PauliFactor& factor : term.factors()) {
                if (factor.pauli != Pauli::I) {
                    active[factor.qubit] = true;
                }
            }
        }
    }
    const auto& operations = program.operations();
    std::vector<bool> retained(operations.size(), false);
    for (std::size_t index = operations.size(); index > 0U; --index) {
        const Operation& operation = operations[index - 1U];
        bool touches = false;
        for (const std::size_t qubit : operation.qubits) {
            touches = touches || active[qubit];
        }
        if (!touches) {
            continue;
        }
        retained[index - 1U] = true;
        for (const std::size_t qubit : operation.qubits) {
            active[qubit] = true;
        }
    }
    const std::size_t missing = std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> mapping(program.num_qubits(), missing);
    std::size_t active_count = 0U;
    for (std::size_t qubit = 0; qubit < active.size(); ++qubit) {
        if (active[qubit]) {
            mapping[qubit] = active_count++;
        }
    }
    Program reduced_program(std::max<std::size_t>(1U, active_count));
    for (std::size_t index = 0; index < operations.size(); ++index) {
        if (!retained[index]) {
            continue;
        }
        Operation operation = operations[index];
        for (std::size_t& qubit : operation.qubits) {
            qubit = mapping[qubit];
        }
        reduced_program = reduced_program.appended(std::move(operation));
    }
    std::vector<Observable> reduced_observables;
    reduced_observables.reserve(observables.size());
    for (const Observable& value : observables) {
        std::vector<PauliTerm> terms;
        terms.reserve(value.terms().size());
        for (const PauliTerm& term : value.terms()) {
            std::vector<PauliFactor> factors;
            factors.reserve(term.factors().size());
            for (const PauliFactor& factor : term.factors()) {
                if (factor.pauli != Pauli::I) {
                    factors.push_back({mapping[factor.qubit], factor.pauli});
                }
            }
            terms.emplace_back(term.coefficient(), std::move(factors));
        }
        reduced_observables.emplace_back(std::move(terms));
    }
    return {std::move(reduced_program), std::move(reduced_observables), active_count};
}
[[nodiscard]] GradientResult adjoint_gradient(
    const Program& template_program,
    const Observable& value,
    const std::vector<ParameterSlot>& slots,
    const std::vector<double>& parameter_values
) {
    Program program = template_program.bound(slots, parameter_values);
    validate_observable(value, program.num_qubits());
    const std::size_t dimension = checked_dimension(program.num_qubits());
    const std::size_t states = program.operations().size() + 1U;
    if (dimension > kAdjointMemoryLimitBytes / sizeof(Complex) / states) {
        throw std::length_error("adjoint differentiation workspace exceeds 1 GiB");
    }
    std::vector<std::vector<Complex>> forward;
    forward.reserve(states);
    std::vector<Complex> state(dimension, Complex{0.0, 0.0});
    state.front() = 1.0;
    forward.push_back(state);
    for (const Operation& operation : program.operations()) {
        apply_operation_state(state, operation);
        forward.push_back(state);
    }
    const double value_result = observable_expectation_from_state(state, value);
    std::vector<Complex> lambda;
    apply_observable_to_state(state, value, lambda);
    std::vector<double> gradient(slots.size(), 0.0);
    std::unordered_map<std::size_t, std::size_t> slot_by_operation;
    for (std::size_t index = 0; index < slots.size(); ++index) {
        slot_by_operation.emplace(slots[index].operation_index, index);
    }
    for (std::size_t index = program.operations().size(); index > 0U; --index) {
        const std::size_t operation_index = index - 1U;
        const Operation& operation = program.operations()[operation_index];
        const auto slot = slot_by_operation.find(operation_index);
        if (slot != slot_by_operation.end()) {
            std::vector<Complex> derivative_state = forward[operation_index];
            apply_single_state(
                derivative_state, derivative_matrix(operation), operation.qubits.front()
            );
            Complex inner{0.0, 0.0};
            for (std::size_t basis = 0; basis < dimension; ++basis) {
                inner += std::conj(lambda[basis]) * derivative_state[basis];
            }
            gradient[slot->second] = 2.0 * inner.real();
        }
        apply_adjoint_operation_state(lambda, operation);
    }
    return {value_result, std::move(gradient), "adjoint", "native-cpu", 1U};
}
void apply_single_density_matrix(
    std::vector<Complex>& rho,
    std::size_t dimension,
    const Matrix2& matrix,
    std::size_t qubit
) {
    const std::size_t bit = std::size_t{1} << qubit;
    const std::size_t span = bit << 1U;
    for (std::size_t column = 0; column < dimension; ++column) {
        for (std::size_t base = 0; base < dimension; base += span) {
            for (std::size_t offset = 0; offset < bit; ++offset) {
                const std::size_t zero = base + offset;
                const std::size_t one = zero + bit;
                const Complex a = rho[zero * dimension + column];
                const Complex b = rho[one * dimension + column];
                rho[zero * dimension + column] = matrix[0] * a + matrix[1] * b;
                rho[one * dimension + column] = matrix[2] * a + matrix[3] * b;
            }
        }
    }
    for (std::size_t row = 0; row < dimension; ++row) {
        for (std::size_t base = 0; base < dimension; base += span) {
            for (std::size_t offset = 0; offset < bit; ++offset) {
                const std::size_t zero = base + offset;
                const std::size_t one = zero + bit;
                const Complex a = rho[row * dimension + zero];
                const Complex b = rho[row * dimension + one];
                rho[row * dimension + zero] =
                    a * std::conj(matrix[0]) + b * std::conj(matrix[1]);
                rho[row * dimension + one] =
                    a * std::conj(matrix[2]) + b * std::conj(matrix[3]);
            }
        }
    }
}
[[nodiscard]] std::pair<std::size_t, Complex> map_two_qubit_basis(
    std::size_t basis,
    const Operation& operation
) {
    const std::size_t first_bit = std::size_t{1} << operation.qubits.at(0);
    const std::size_t second_bit = std::size_t{1} << operation.qubits.at(1);
    switch (operation.code) {
    case OperationCode::CX:
        return {(basis & first_bit) != 0U ? basis ^ second_bit : basis, 1.0};
    case OperationCode::CZ:
        return {basis, ((basis & first_bit) != 0U && (basis & second_bit) != 0U) ? -1.0 : 1.0};
    case OperationCode::SWAP: {
        const bool first = (basis & first_bit) != 0U;
        const bool second = (basis & second_bit) != 0U;
        return {first == second ? basis : basis ^ first_bit ^ second_bit, 1.0};
    }
    default:
        break;
    }
    throw std::invalid_argument("operation is not a supported two-qubit gate");
}

void apply_two_density_matrix(
    std::vector<Complex>& rho,
    std::size_t dimension,
    const Operation& operation
) {
    std::vector<Complex> transformed(rho.size(), Complex{0.0, 0.0});
    for (std::size_t row = 0; row < dimension; ++row) {
        const auto [mapped_row, row_phase] = map_two_qubit_basis(row, operation);
        for (std::size_t column = 0; column < dimension; ++column) {
            const auto [mapped_column, column_phase] = map_two_qubit_basis(column, operation);
            transformed[mapped_row * dimension + mapped_column] +=
                row_phase * std::conj(column_phase) * rho[row * dimension + column];
        }
    }
    rho.swap(transformed);
}
void apply_operation_density(
    std::vector<Complex>& rho,
    std::size_t dimension,
    const Operation& operation
) {
    if (operation.qubits.size() == 1U) {
        apply_single_density_matrix(rho, dimension, gate_matrix(operation), operation.qubits.front());
    } else {
        apply_two_density_matrix(rho, dimension, operation);
    }
}

void apply_kraus_channel(
    std::vector<Complex>& rho,
    std::size_t dimension,
    std::size_t qubit,
    const std::vector<Matrix2>& kraus
) {
    std::vector<Complex> result(rho.size(), Complex{0.0, 0.0});
    for (const Matrix2& matrix : kraus) {
        std::vector<Complex> term = rho;
        apply_single_density_matrix(term, dimension, matrix, qubit);
        for (std::size_t index = 0; index < result.size(); ++index) {
            result[index] += term[index];
        }
    }
    rho.swap(result);
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
        const double p = channel.parameters.at(0);
        return {scaled(identity, std::sqrt(1.0 - p)), scaled(x_matrix, std::sqrt(p))};
    }
    case NoiseChannelCode::PhaseFlip: {
        const double p = channel.parameters.at(0);
        return {scaled(identity, std::sqrt(1.0 - p)), scaled(z_matrix, std::sqrt(p))};
    }
    case NoiseChannelCode::Depolarizing: {
        const double p = channel.parameters.at(0);
        const double error_scale = std::sqrt(p / 3.0);
        return {
            scaled(identity, std::sqrt(1.0 - p)), scaled(x_matrix, error_scale),
            scaled(y_matrix, error_scale), scaled(z_matrix, error_scale),
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
        const double px = channel.parameters.at(0);
        const double py = channel.parameters.at(1);
        const double pz = channel.parameters.at(2);
        const double identity_probability = 1.0 - px - py - pz;
        return {
            scaled(identity, std::sqrt(identity_probability)), scaled(x_matrix, std::sqrt(px)),
            scaled(y_matrix, std::sqrt(py)), scaled(z_matrix, std::sqrt(pz)),
        };
    }
    case NoiseChannelCode::Kraus: {
        if (channel.kraus_count == 0U || channel.kraus_operators.size() != channel.kraus_count * 4U) {
            throw std::invalid_argument("custom Kraus channel has invalid operator storage");
        }
        std::vector<Matrix2> operators;
        operators.reserve(channel.kraus_count);
        for (std::size_t index = 0U; index < channel.kraus_count; ++index) {
            Matrix2 matrix{};
            std::copy_n(
                channel.kraus_operators.begin() + static_cast<std::ptrdiff_t>(index * 4U),
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

void apply_noise_channel(
    std::vector<Complex>& rho,
    std::size_t dimension,
    const NoiseChannel& channel
) {
    apply_kraus_channel(rho, dimension, channel.qubit, noise_kraus_operators(channel));
}

[[nodiscard]] std::array<Complex, 16> density_superoperator(
    const std::vector<Matrix2>& operators
) {
    std::array<Complex, 16> result{};
    for (const Matrix2& matrix : operators) {
        for (std::size_t row = 0U; row < 2U; ++row) {
            for (std::size_t column = 0U; column < 2U; ++column) {
                const std::size_t output = column + 2U * row;
                for (std::size_t input_row = 0U; input_row < 2U; ++input_row) {
                    for (std::size_t input_column = 0U; input_column < 2U; ++input_column) {
                        const std::size_t input = input_column + 2U * input_row;
                        result[output * 4U + input] +=
                            matrix[row * 2U + input_row] *
                            std::conj(matrix[column * 2U + input_column]);
                    }
                }
            }
        }
    }
    return result;
}

[[nodiscard]] detail::CudaDensityStepKind cuda_density_gate_kind(OperationCode code) {
    switch (code) {
    case OperationCode::CX: return detail::CudaDensityStepKind::CX;
    case OperationCode::CZ: return detail::CudaDensityStepKind::CZ;
    case OperationCode::SWAP: return detail::CudaDensityStepKind::SWAP;
    default: break;
    }
    throw std::invalid_argument("operation is not a CUDA density two-qubit gate");
}

void append_cuda_density_operation(
    std::vector<detail::CudaDensityStep>& steps,
    const Operation& operation,
    std::size_t num_qubits
) {
    if (operation.qubits.size() == 1U) {
        const Matrix2 matrix = gate_matrix(operation);
        std::array<Complex, 16> column_matrix{};
        std::array<Complex, 16> row_matrix{};
        for (std::size_t index = 0U; index < matrix.size(); ++index) {
            column_matrix[index] = std::conj(matrix[index]);
            row_matrix[index] = matrix[index];
        }
        const std::size_t qubit = operation.qubits.front();
        steps.push_back({detail::CudaDensityStepKind::Single, column_matrix, qubit, 0U});
        steps.push_back({detail::CudaDensityStepKind::Single, row_matrix, qubit + num_qubits, 0U});
        return;
    }
    const detail::CudaDensityStepKind kind = cuda_density_gate_kind(operation.code);
    const std::array<Complex, 16> matrix{};
    steps.push_back({kind, matrix, operation.qubits.at(0), operation.qubits.at(1)});
    steps.push_back({
        kind,
        matrix,
        operation.qubits.at(0) + num_qubits,
        operation.qubits.at(1) + num_qubits,
    });
}

void append_cuda_density_noise(
    std::vector<detail::CudaDensityStep>& steps,
    const NoiseChannel& channel,
    std::size_t num_qubits
) {
    steps.push_back({
        detail::CudaDensityStepKind::Matrix4,
        density_superoperator(noise_kraus_operators(channel)),
        channel.qubit,
        channel.qubit + num_qubits,
    });
}

[[nodiscard]] std::vector<detail::CudaDensityStep> cuda_density_steps(const Program& program) {
    std::vector<detail::CudaDensityStep> steps;
    steps.reserve(program.operations().size() * 2U);
    for (const Operation& operation : program.operations()) {
        append_cuda_density_operation(steps, operation, program.num_qubits());
    }
    return steps;
}

[[nodiscard]] std::vector<detail::CudaDensityStep> cuda_density_steps(
    const NoisyProgram& noisy
) {
    const Program& program = noisy.program();
    std::vector<detail::CudaDensityStep> steps;
    steps.reserve(program.operations().size() * 2U + noisy.noise().size());
    std::size_t noise_index = 0U;
    const auto append_noise_at = [&](std::size_t point) {
        while (noise_index < noisy.noise().size() &&
               noisy.noise()[noise_index].after_operation == point) {
            append_cuda_density_noise(steps, noisy.noise()[noise_index].channel, program.num_qubits());
            ++noise_index;
        }
    };
    append_noise_at(0U);
    for (std::size_t index = 0U; index < program.operations().size(); ++index) {
        append_cuda_density_operation(steps, program.operations()[index], program.num_qubits());
        append_noise_at(index + 1U);
    }
    return steps;
}

[[nodiscard]] std::vector<Complex> matrix_multiply(
    const std::vector<Complex>& left,
    const std::vector<Complex>& right,
    std::size_t dimension
) {
    std::vector<Complex> result(dimension * dimension, Complex{0.0, 0.0});
    for (std::size_t row = 0; row < dimension; ++row) {
        for (std::size_t inner = 0; inner < dimension; ++inner) {
            const Complex value = left[row * dimension + inner];
            if (value == Complex{0.0, 0.0}) {
                continue;
            }
            for (std::size_t column = 0; column < dimension; ++column) {
                result[row * dimension + column] += value * right[inner * dimension + column];
            }
        }
    }
    return result;
}
[[nodiscard]] std::vector<Complex> matrix_dagger(
    const std::vector<Complex>& matrix,
    std::size_t dimension
) {
    std::vector<Complex> result(matrix.size());
    for (std::size_t row = 0; row < dimension; ++row) {
        for (std::size_t column = 0; column < dimension; ++column) {
            result[column * dimension + row] = std::conj(matrix[row * dimension + column]);
        }
    }
    return result;
}

[[nodiscard]] std::vector<Complex> lindblad_derivative(
    const std::vector<Complex>& rho,
    const std::vector<Complex>& hamiltonian,
    const std::vector<std::vector<Complex>>& collapse_operators,
    std::size_t dimension
) {
    const std::vector<Complex> h_rho = matrix_multiply(hamiltonian, rho, dimension);
    const std::vector<Complex> rho_h = matrix_multiply(rho, hamiltonian, dimension);
    std::vector<Complex> derivative(rho.size());
    for (std::size_t index = 0; index < rho.size(); ++index) {
        derivative[index] = Complex{0.0, -1.0} * (h_rho[index] - rho_h[index]);
    }
    for (const auto& collapse : collapse_operators) {
        const std::vector<Complex> collapse_dagger = matrix_dagger(collapse, dimension);
        const std::vector<Complex> c_rho = matrix_multiply(collapse, rho, dimension);
        const std::vector<Complex> jump = matrix_multiply(c_rho, collapse_dagger, dimension);
        const std::vector<Complex> cdag_c = matrix_multiply(collapse_dagger, collapse, dimension);
        const std::vector<Complex> loss_left = matrix_multiply(cdag_c, rho, dimension);
        const std::vector<Complex> loss_right = matrix_multiply(rho, cdag_c, dimension);
        for (std::size_t index = 0; index < derivative.size(); ++index) {
            derivative[index] += jump[index] - 0.5 * (loss_left[index] + loss_right[index]);
        }
    }
    return derivative;
}
[[nodiscard]] std::vector<Complex> add_scaled(
    const std::vector<Complex>& base,
    const std::vector<Complex>& delta,
    double scale
) {
    std::vector<Complex> result(base.size());
    for (std::size_t index = 0; index < base.size(); ++index) {
        result[index] = base[index] + scale * delta[index];
    }
    return result;
}

void restore_density_invariants(std::vector<Complex>& rho, std::size_t dimension) {
    for (std::size_t row = 0; row < dimension; ++row) {
        rho[row * dimension + row] = Complex{rho[row * dimension + row].real(), 0.0};
        for (std::size_t column = row + 1U; column < dimension; ++column) {
            const Complex average =
                0.5 * (rho[row * dimension + column] + std::conj(rho[column * dimension + row]));
            rho[row * dimension + column] = average;
            rho[column * dimension + row] = std::conj(average);
        }
    }
    double trace = 0.0;
    for (std::size_t index = 0; index < dimension; ++index) {
        trace += rho[index * dimension + index].real();
    }
    if (!std::isfinite(trace) || std::abs(trace) < 1e-15) {
        throw std::domain_error("Lindblad integration produced an invalid density matrix trace");
    }
    for (Complex& value : rho) {
        value /= trace;
    }
}

[[nodiscard]] std::string format_angle(double value) {
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
    return output.str();
}
[[nodiscard]] std::string qasm_operation(const Operation& operation) {
    std::ostringstream output;
    output.imbue(std::locale::classic());
    const auto q = [&](std::size_t index) { return "q[" + std::to_string(index) + "]"; };
    switch (operation.code) {
    case OperationCode::H: output << "h " << q(operation.qubits[0]); break;
    case OperationCode::X: output << "x " << q(operation.qubits[0]); break;
    case OperationCode::Y: output << "y " << q(operation.qubits[0]); break;
    case OperationCode::Z: output << "z " << q(operation.qubits[0]); break;
    case OperationCode::RX:
        output << "rx(" << format_angle(operation.parameters[0]) << ") " << q(operation.qubits[0]);
        break;
    case OperationCode::RY:
        output << "ry(" << format_angle(operation.parameters[0]) << ") " << q(operation.qubits[0]);
        break;
    case OperationCode::RZ:
        output << "rz(" << format_angle(operation.parameters[0]) << ") " << q(operation.qubits[0]);
        break;
    case OperationCode::CX:
        output << "cx " << q(operation.qubits[0]) << ", " << q(operation.qubits[1]);
        break;
    case OperationCode::CZ:
        output << "cz " << q(operation.qubits[0]) << ", " << q(operation.qubits[1]);
        break;
    case OperationCode::SWAP:
        output << "swap " << q(operation.qubits[0]) << ", " << q(operation.qubits[1]);
        break;
    }
    output << ';';
    return output.str();
}

[[nodiscard]] std::string qir_qubit(std::size_t qubit) {
    return qubit == 0U ? "ptr null" : "ptr inttoptr (i64 " + std::to_string(qubit) + " to ptr)";
}
[[nodiscard]] std::vector<std::string> qir_calls(const Operation& operation) {
    const auto one = [&](const char* name) {
        return std::vector<std::string>{
            "  call void @__quantum__qis__" + std::string(name) + "__body(" +
            qir_qubit(operation.qubits[0]) + ")"
        };
    };
    switch (operation.code) {
    case OperationCode::H: return one("h");
    case OperationCode::X: return one("x");
    case OperationCode::Y: return one("y");
    case OperationCode::Z: return one("z");
    case OperationCode::RX:
    case OperationCode::RY:
    case OperationCode::RZ: {
        const char* name = operation.code == OperationCode::RX ? "rx" :
            (operation.code == OperationCode::RY ? "ry" : "rz");
        return {"  call void @__quantum__qis__" + std::string(name) + "__body(double " +
                format_angle(operation.parameters[0]) + ", " + qir_qubit(operation.qubits[0]) + ")"};
    }
    case OperationCode::CX:
    case OperationCode::CZ: {
        const char* name = operation.code == OperationCode::CX ? "cnot" : "cz";
        return {"  call void @__quantum__qis__" + std::string(name) + "__body(" +
                qir_qubit(operation.qubits[0]) + ", " + qir_qubit(operation.qubits[1]) + ")"};
    }
    case OperationCode::SWAP: {
        const std::string first = qir_qubit(operation.qubits[0]);
        const std::string second = qir_qubit(operation.qubits[1]);
        return {
            "  call void @__quantum__qis__cnot__body(" + first + ", " + second + ")",
            "  call void @__quantum__qis__cnot__body(" + second + ", " + first + ")",
            "  call void @__quantum__qis__cnot__body(" + first + ", " + second + ")",
        };
    }
    }
    throw std::invalid_argument("unsupported operation for QIR export");
}
[[nodiscard]] std::optional<std::string> environment_value(const char* name) {
#ifdef _WIN32
    char* value = nullptr;
    std::size_t length = 0U;
    if (_dupenv_s(&value, &length, name) != 0 || value == nullptr) {
        return std::nullopt;
    }
    std::string result(value);
    std::free(value);
    if (result.empty()) {
        return std::nullopt;
    }
    return result;
#else
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return std::nullopt;
    }
    return std::string(value);
#endif
}

[[nodiscard]] std::optional<std::size_t> environment_size(const char* name) {
    const auto value = environment_value(name);
    if (!value.has_value()) {
        return std::nullopt;
    }
    try {
        const unsigned long long parsed = std::stoull(*value);
        if (parsed > std::numeric_limits<std::size_t>::max()) {
            return std::nullopt;
        }
        return static_cast<std::size_t>(parsed);
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

}  // namespace

PauliTerm::PauliTerm(double coefficient, std::vector<PauliFactor> factors)
    : coefficient_(coefficient), factors_(std::move(factors)) {
    if (!std::isfinite(coefficient_)) {
        throw std::invalid_argument("Pauli-term coefficient must be finite");
    }
    factors_.erase(
        std::remove_if(factors_.begin(), factors_.end(), [](const PauliFactor& factor) {
            return factor.pauli == Pauli::I;
        }),
        factors_.end()
    );
    std::sort(factors_.begin(), factors_.end(), [](const PauliFactor& left, const PauliFactor& right) {
        return left.qubit < right.qubit;
    });
    for (std::size_t index = 1; index < factors_.size(); ++index) {
        if (factors_[index - 1U].qubit == factors_[index].qubit) {
            throw std::invalid_argument("a Pauli term can contain only one operator per qubit");
        }
    }
}

double PauliTerm::coefficient() const noexcept {
    return coefficient_;
}

const std::vector<PauliFactor>& PauliTerm::factors() const noexcept {
    return factors_;
}

std::string PauliTerm::canonical_text() const {
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << std::setprecision(std::numeric_limits<double>::max_digits10) << coefficient_;
    for (const PauliFactor& factor : factors_) {
        output << ' ' << pauli_name(factor.pauli) << factor.qubit;
    }
    return output.str();
}

Observable::Observable(std::vector<PauliTerm> terms) : terms_(std::move(terms)) {}

const std::vector<PauliTerm>& Observable::terms() const noexcept {
    return terms_;
}

std::string Observable::canonical_text() const {
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << "qupy-observable 1\n";
    for (const PauliTerm& term : terms_) {
        output << "term " << term.canonical_text() << '\n';
    }
    return output.str();
}
std::string Observable::fingerprint() const {
    return fingerprint_text(canonical_text());
}

NoisyProgram::NoisyProgram(Program program, std::vector<NoiseInstruction> noise)
    : program_(std::move(program)), noise_(std::move(noise)) {
    for (const NoiseInstruction& instruction : noise_) {
        if (instruction.after_operation > program_.operations().size()) {
            throw std::invalid_argument("noise insertion point is outside the program");
        }
        if (instruction.channel.qubit >= program_.num_qubits()) {
            throw std::invalid_argument("noise channel qubit is outside the program");
        }
    }
    std::stable_sort(noise_.begin(), noise_.end(), [](const auto& left, const auto& right) {
        return left.after_operation < right.after_operation;
    });
}

const Program& NoisyProgram::program() const noexcept {
    return program_;
}

const std::vector<NoiseInstruction>& NoisyProgram::noise() const noexcept {
    return noise_;
}

DetectorModel::DetectorModel(
    std::size_t detector_count,
    std::size_t observable_count,
    std::vector<DetectorError> errors
) : detector_count_(detector_count), observable_count_(observable_count), errors_(std::move(errors)) {
    for (const DetectorError& error : errors_) {
        validate_probability(error.probability, "detector error probability");
        for (const std::size_t detector : error.detectors) {
            if (detector >= detector_count_) {
                throw std::invalid_argument("detector error references an unknown detector");
            }
        }
        for (const std::size_t observable_index : error.observables) {
            if (observable_index >= observable_count_) {
                throw std::invalid_argument("detector error references an unknown logical observable");
            }
        }
    }
}

std::size_t DetectorModel::detector_count() const noexcept {
    return detector_count_;
}

std::size_t DetectorModel::observable_count() const noexcept {
    return observable_count_;
}

const std::vector<DetectorError>& DetectorModel::errors() const noexcept {
    return errors_;
}

std::string DetectorModel::canonical_text() const {
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << "qupy-detector-model 1\n";
    output << "detectors " << detector_count_ << '\n';
    output << "observables " << observable_count_ << '\n';
    output << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (const DetectorError& error : errors_) {
        output << "error " << error.probability << " D";
        for (std::size_t index = 0; index < error.detectors.size(); ++index) {
            output << (index == 0U ? "" : ",") << error.detectors[index];
        }
        output << " L";
        for (std::size_t index = 0; index < error.observables.size(); ++index) {
            output << (index == 0U ? "" : ",") << error.observables[index];
        }
        output << '\n';
    }
    return output.str();
}
std::string DetectorModel::fingerprint() const {
    return fingerprint_text(canonical_text());
}

PauliTerm pauli_term(double coefficient, std::vector<PauliFactor> factors) {
    return PauliTerm(coefficient, std::move(factors));
}

Observable observable(std::vector<PauliTerm> terms) {
    return Observable(std::move(terms));
}

Observable observable(PauliZ value) {
    return Observable({PauliTerm(1.0, {{value.qubit, Pauli::Z}})});
}

ObservableExecutionPlan observable_plan(
    const Program& program,
    const std::vector<Observable>& values,
    const std::string& backend,
    const PlannerCostModel* cost_model
) {
    ReducedObservableSet reduced = reduce_observables(program, values);
    std::size_t term_count = 0U;
    std::size_t group_count = 0U;
    std::ostringstream query_text;
    query_text.imbue(std::locale::classic());
    query_text << "qupy-observable-query 1\n";
    for (const Observable& value : values) {
        term_count = checked_term_sum(term_count, value.terms().size());
        group_count = checked_term_sum(group_count, measurement_groups(value).size());
        query_text << "observable " << value.fingerprint() << '\n';
    }
    const std::string query_fingerprint = fingerprint_text(query_text.str());
    if (explicit_mpi_backend(backend)) {
        if (!mpi_compiled()) {
            throw std::runtime_error(
                "MPI support is not compiled; rebuild QuPy with an MPI C++ implementation available"
            );
        }
        if (reduced.active_qubits >= std::numeric_limits<std::size_t>::digits) {
            throw std::length_error("distributed statevector exceeds native address space");
        }
        const std::size_t dimension = std::size_t{1} << reduced.active_qubits;
        if (dimension > std::numeric_limits<std::size_t>::max() / sizeof(Complex)) {
            throw std::length_error("distributed statevector exceeds native address space");
        }
        const DistributedInfo topology = distributed_info();
        if (topology.world_size == 0U ||
            (topology.world_size & (topology.world_size - 1U)) != 0U) {
            throw std::invalid_argument("distributed statevector requires a power-of-two MPI world size");
        }
        if (topology.world_size > dimension) {
            throw std::invalid_argument("MPI world size exceeds the state-vector dimension");
        }
        const std::size_t state_bytes = dimension * sizeof(Complex);
        const std::size_t local_state_bytes = state_bytes / topology.world_size;
        const std::string cache_key = fingerprint_text(
            "qupy-observable-cache 1\nprogram " + program.fingerprint() +
            "\nquery " + query_fingerprint +
            "\nbackend native-mpi\nmethod mpi-pauli-reduction\nworld-size " +
            std::to_string(topology.world_size) + "\nruntime " + topology.runtime + "\n"
        );
        return {
            "native-mpi", "mpi-pauli-reduction", true, reduced.active_qubits, values.size(),
            term_count, group_count, local_state_bytes, program.fingerprint(), query_fingerprint,
            cache_key, std::nullopt, {}, {},
        };
    }
    if (use_rich_pauli_propagation(reduced.program, backend)) {
        const std::string cache_key = fingerprint_text(
            "qupy-observable-cache 1\nprogram " + program.fingerprint() +
            "\nquery " + query_fingerprint + "\nbackend native-cpu\nmethod pauli-propagation\n"
        );
        return {
            "native-cpu", "pauli-propagation", true, reduced.active_qubits, values.size(),
            term_count, group_count, 0U, program.fingerprint(), query_fingerprint, cache_key,
            std::nullopt, {}, {},
        };
    }
    const bool observable_policy = cost_model != nullptr &&
        cost_model->observable_auto_validated() && term_count > 0U;
    const std::string initial_backend = observable_policy &&
        (backend == "auto" || backend == "cpu") ? "native-cpu" : backend;
    ExecutionPlan base = plan(
        reduced.program, ResultMode::StateVector, initial_backend, cost_model
    );
    std::optional<double> predicted =
        detail::cuda_backend_device(base.backend).has_value() ? std::nullopt : base.predicted_ns;
    std::string cost_class = base.cost_model_class;
    std::string cost_fingerprint = base.cost_model_fingerprint;
    if (observable_policy &&
        (backend == "auto" || backend == "native-cpu" || backend == "cpu" ||
         explicit_default_cuda_backend(backend))) {
        const ExecutionPlan cpu = plan(reduced.program, ResultMode::StateVector, "native-cpu");
        const double cpu_prediction = cost_model->predict_observable_ns(cpu, term_count, 0U);
        base = cpu;
        predicted = cpu_prediction;
        cost_class = "observable-return-cpu";
        if (explicit_cuda_backend(backend) ||
            (backend == "auto" && cuda_observable_candidate_supported(reduced.program))) {
            const ExecutionPlan cuda = plan(
                reduced.program, ResultMode::StateVector, "native-cuda"
            );
            CudaPauliQuery cuda_query;
            for (const Observable& value : reduced.observables) {
                static_cast<void>(index_observable(cuda_query, value));
            }
            const double cuda_prediction = cost_model->predict_observable_ns(
                cuda, cuda_query.size(), 0U
            );
            if (explicit_cuda_backend(backend) || cuda_prediction < cpu_prediction) {
                base = cuda;
                predicted = cuda_prediction;
                cost_class = "observable-return-cuda";
            }
        }
        cost_fingerprint = cost_model->artifact_fingerprint();
    }
    const bool cuda_reduction =
        detail::cuda_backend_device(base.backend).has_value() &&
        base.method == "cuda-statevector";
    const std::string method = cuda_reduction
        ? "cuda-pauli-reduction" : base.method + "-observable";
    const std::string cache_key = fingerprint_text(
        "qupy-observable-cache 1\nprogram " + program.fingerprint() +
        "\nquery " + query_fingerprint + "\nbase " + base.cache_key + "\nmethod " + method + "\n"
    );
    return {
        base.backend, method, base.exact, reduced.active_qubits, values.size(), term_count,
        group_count, base.estimated_state_bytes, program.fingerprint(), query_fingerprint,
        cache_key, predicted, cost_class, cost_fingerprint,
    };
}
std::vector<std::vector<std::size_t>> commuting_groups(const Observable& value) {
    std::vector<std::vector<std::size_t>> groups;
    for (std::size_t term_index = 0; term_index < value.terms().size(); ++term_index) {
        bool placed = false;
        for (auto& group : groups) {
            bool commutes = true;
            for (const std::size_t other_index : group) {
                if (!term_commutes(value.terms()[term_index], value.terms()[other_index])) {
                    commutes = false;
                    break;
                }
            }
            if (commutes) {
                group.push_back(term_index);
                placed = true;
                break;
            }
        }
        if (!placed) {
            groups.push_back({term_index});
        }
    }
    return groups;
}

std::vector<MeasurementGroup> measurement_groups(const Observable& value) {
    std::vector<MeasurementGroup> groups;
    for (std::size_t term_index = 0; term_index < value.terms().size(); ++term_index) {
        const PauliTerm& term = value.terms()[term_index];
        if (term.coefficient() == 0.0 || term.factors().empty()) {
            continue;
        }
        bool placed = false;
        for (MeasurementGroup& group : groups) {
            bool compatible = true;
            for (const PauliFactor& factor : term.factors()) {
                const auto existing = std::find_if(
                    group.basis.begin(), group.basis.end(),
                    [&](const PauliFactor& candidate) { return candidate.qubit == factor.qubit; }
                );
                if (existing != group.basis.end() && existing->pauli != factor.pauli) {
                    compatible = false;
                    break;
                }
            }
            if (!compatible) {
                continue;
            }
            group.term_indices.push_back(term_index);
            for (const PauliFactor& factor : term.factors()) {
                const auto existing = std::find_if(
                    group.basis.begin(), group.basis.end(),
                    [&](const PauliFactor& candidate) { return candidate.qubit == factor.qubit; }
                );
                if (existing == group.basis.end()) {
                    group.basis.push_back(factor);
                }
            }
            std::sort(group.basis.begin(), group.basis.end(), [](const auto& left, const auto& right) {
                return left.qubit < right.qubit;
            });
            placed = true;
            break;
        }
        if (!placed) {
            groups.push_back({{term_index}, term.factors()});
        }
    }
    return groups;
}

ShotEstimate estimate_observable(
    const Program& program,
    const Observable& value,
    std::size_t shots_per_group,
    std::optional<std::uint64_t> seed,
    const std::string& backend
) {
    if (shots_per_group < 2U) {
        throw std::invalid_argument("observable estimation requires at least two shots per group");
    }
    validate_observable(value, program.num_qubits());
    const std::vector<MeasurementGroup> groups = measurement_groups(value);
    double result = 0.0;
    for (const PauliTerm& term : value.terms()) {
        if (term.factors().empty()) {
            result += term.coefficient();
        }
    }
    if (groups.empty()) {
        return {result, 0.0, shots_per_group, 0U, 0U, "constant"};
    }
    if (groups.size() > std::numeric_limits<std::size_t>::max() / shots_per_group) {
        throw std::length_error("observable-estimation shot count exceeds native range");
    }
    double variance_of_estimate = 0.0;
    std::string executed_backend;
    const double pi = std::acos(-1.0);
    for (std::size_t group_index = 0; group_index < groups.size(); ++group_index) {
        const MeasurementGroup& group = groups[group_index];
        Program measurement_program = program;
        for (const PauliFactor& factor : group.basis) {
            if (factor.pauli == Pauli::X) {
                measurement_program = h(measurement_program, factor.qubit);
            } else if (factor.pauli == Pauli::Y) {
                measurement_program = rz(measurement_program, -pi / 2.0, factor.qubit);
                measurement_program = h(measurement_program, factor.qubit);
            }
        }
        std::optional<std::uint64_t> group_seed = std::nullopt;
        if (seed.has_value()) {
            std::uint64_t mixed = *seed + 0x9e3779b97f4a7c15ULL * (group_index + 1U);
            mixed = (mixed ^ (mixed >> 30U)) * 0xbf58476d1ce4e5b9ULL;
            mixed = (mixed ^ (mixed >> 27U)) * 0x94d049bb133111ebULL;
            group_seed = mixed ^ (mixed >> 31U);
        }
        const Samples samples = sample(measurement_program, shots_per_group, group_seed, backend);
        if (executed_backend.empty()) {
            executed_backend = samples.backend;
        } else if (executed_backend != samples.backend) {
            executed_backend = "mixed";
        }
        double sum = 0.0;
        double sum_squares = 0.0;
        for (std::size_t shot = 0; shot < shots_per_group; ++shot) {
            double shot_value = 0.0;
            for (const std::size_t term_index : group.term_indices) {
                const PauliTerm& term = value.terms()[term_index];
                bool odd = false;
                for (const PauliFactor& factor : term.factors()) {
                    const std::size_t sample_index = shot * program.num_qubits() + factor.qubit;
                    odd = odd != (samples.values[sample_index] != 0);
                }
                shot_value += term.coefficient() * (odd ? -1.0 : 1.0);
            }
            sum += shot_value;
            sum_squares += shot_value * shot_value;
        }
        const double count = static_cast<double>(shots_per_group);
        const double mean = sum / count;
        const double centered_sum = std::max(0.0, sum_squares - count * mean * mean);
        const double sample_variance = centered_sum / static_cast<double>(shots_per_group - 1U);
        result += mean;
        variance_of_estimate += sample_variance / count;
    }
    return {
        result,
        std::sqrt(variance_of_estimate),
        shots_per_group,
        shots_per_group * groups.size(),
        groups.size(),
        std::move(executed_backend),
    };
}

ObservableResult expect_observable(
    const Program& program,
    const Observable& value,
    const std::string& backend,
    const PlannerCostModel* cost_model
) {
    ReducedObservableSet reduced = reduce_observables(program, {value});
    if (use_rich_pauli_propagation(reduced.program, backend)) {
        return {
            propagated_observable_expectation(reduced.program, reduced.observables.front()),
            "native-cpu",
            reduced.active_qubits,
            reduced.observables.front().terms().size(),
        };
    }
    if (explicit_mpi_backend(backend)) {
        MpiPauliQuery query;
        const std::vector<MpiLinearTerm> indexed = index_mpi_observable(
            query, reduced.observables.front()
        );
        const std::vector<Complex> values = query.evaluate(reduced.program);
        return {
            hermitian_mpi_value(evaluate_mpi_linear(values, indexed), "observable expectation"),
            "native-mpi",
            reduced.active_qubits,
            query.size(),
        };
    }
    const std::size_t expectation_terms = reduced.observables.front().terms().size();
    CudaPauliQuery cuda_query;
    std::vector<CudaLinearTerm> cuda_indexed;
    bool cuda_query_ready = false;
    if (needs_cuda_observable_query(reduced.program, backend, cost_model)) {
        cuda_indexed = index_observable(cuda_query, reduced.observables.front());
        cuda_query_ready = true;
    }
    const std::size_t cuda_evaluations = cuda_query_ready ? cuda_query.size() : expectation_terms;
    if (use_cuda_observable(
            reduced.program,
            {expectation_terms, 0U},
            {cuda_evaluations, 0U},
            backend,
            cost_model
        )) {
        if (!cuda_query_ready) {
            cuda_indexed = index_observable(cuda_query, reduced.observables.front());
        }
        const std::vector<Complex> values = cuda_query.evaluate(reduced.program, cuda_device_for_execution(backend));
        return {
            hermitian_cuda_value(
                evaluate_cuda_linear(values, cuda_indexed), "observable expectation"
            ),
            cuda_backend_for_execution(backend),
            reduced.active_qubits,
            cuda_query.size(),
        };
    }
    StateVector state = statevector(reduced.program, backend, cost_model);
    return {
        observable_expectation_from_state(state.values, reduced.observables.front()),
        state.backend,
        reduced.active_qubits,
        1U,
    };
}

ObservableResult variance_observable(
    const Program& program,
    const Observable& value,
    const std::string& backend,
    const PlannerCostModel* cost_model
) {
    ReducedObservableSet reduced = reduce_observables(program, {value});
    if (use_rich_pauli_propagation(reduced.program, backend)) {
        const Observable& observable_value = reduced.observables.front();
        const double mean = propagated_observable_expectation(reduced.program, observable_value);
        const Complex second = propagated_observable_product(
            reduced.program, observable_value, observable_value
        );
        if (std::abs(second.imag()) > kHermitianTolerance) {
            throw std::domain_error("observable variance acquired an invalid imaginary part");
        }
        double result = second.real() - mean * mean;
        if (result < 0.0 && result > -1e-12) result = 0.0;
        return {result, "native-cpu", reduced.active_qubits, observable_value.terms().size()};
    }
    if (explicit_mpi_backend(backend)) {
        const Observable& observable_value = reduced.observables.front();
        MpiPauliQuery query;
        const std::vector<MpiLinearTerm> indexed = index_mpi_observable(query, observable_value);
        const std::vector<MpiLinearTerm> squared = index_mpi_observable_product(
            query, observable_value, observable_value
        );
        const std::vector<Complex> values = query.evaluate(reduced.program);
        const double mean = hermitian_mpi_value(
            evaluate_mpi_linear(values, indexed), "observable expectation"
        );
        const double second = hermitian_mpi_value(
            evaluate_mpi_linear(values, squared), "observable variance"
        );
        double result = second - mean * mean;
        if (result < 0.0 && result > -1e-12) result = 0.0;
        return {result, "native-mpi", reduced.active_qubits, query.size()};
    }
    const Observable& observable_value = reduced.observables.front();
    const std::size_t variance_terms = observable_value.terms().size();
    CudaPauliQuery cuda_query;
    std::vector<CudaLinearTerm> cuda_indexed;
    std::vector<CudaLinearTerm> cuda_squared;
    bool cuda_query_ready = false;
    if (needs_cuda_observable_query(reduced.program, backend, cost_model)) {
        cuda_indexed = index_observable(cuda_query, observable_value);
        cuda_squared = index_observable_product(cuda_query, observable_value, observable_value);
        cuda_query_ready = true;
    }
    const std::size_t cuda_evaluations = cuda_query_ready ? cuda_query.size() : variance_terms;
    if (use_cuda_observable(
            reduced.program,
            {checked_term_sum(variance_terms, variance_terms), 1U},
            {cuda_evaluations, 0U},
            backend,
            cost_model
        )) {
        if (!cuda_query_ready) {
            cuda_indexed = index_observable(cuda_query, observable_value);
            cuda_squared = index_observable_product(cuda_query, observable_value, observable_value);
        }
        const std::vector<Complex> values = cuda_query.evaluate(reduced.program, cuda_device_for_execution(backend));
        const double mean = hermitian_cuda_value(
            evaluate_cuda_linear(values, cuda_indexed), "observable expectation"
        );
        const double second = hermitian_cuda_value(
            evaluate_cuda_linear(values, cuda_squared), "observable variance"
        );
        double result = second - mean * mean;
        if (result < 0.0 && result > -1e-12) result = 0.0;
        return {
            result,
            cuda_backend_for_execution(backend),
            reduced.active_qubits,
            cuda_query.size(),
        };
    }
    StateVector state = statevector(reduced.program, backend, cost_model);
    std::vector<Complex> applied;
    apply_observable_to_state(state.values, reduced.observables.front(), applied);
    const double mean = observable_expectation_from_state(state.values, reduced.observables.front());
    double result = squared_norm(applied) - mean * mean;
    if (result < 0.0 && result > -1e-12) result = 0.0;
    return {result, state.backend, reduced.active_qubits, 1U};
}

ObservableResult covariance_observable(
    const Program& program,
    const Observable& left,
    const Observable& right,
    const std::string& backend,
    const PlannerCostModel* cost_model
) {
    ReducedObservableSet reduced = reduce_observables(program, {left, right});
    if (use_rich_pauli_propagation(reduced.program, backend)) {
        const double left_mean = propagated_observable_expectation(
            reduced.program, reduced.observables[0]
        );
        const double right_mean = propagated_observable_expectation(
            reduced.program, reduced.observables[1]
        );
        const Complex product = propagated_observable_product(
            reduced.program, reduced.observables[0], reduced.observables[1]
        );
        return {
            product.real() - left_mean * right_mean,
            "native-cpu",
            reduced.active_qubits,
            reduced.observables[0].terms().size() + reduced.observables[1].terms().size(),
        };
    }
    if (explicit_mpi_backend(backend)) {
        MpiPauliQuery query;
        const std::vector<MpiLinearTerm> left_indexed = index_mpi_observable(
            query, reduced.observables[0]
        );
        const std::vector<MpiLinearTerm> right_indexed = index_mpi_observable(
            query, reduced.observables[1]
        );
        const std::vector<MpiLinearTerm> product_indexed = index_mpi_observable_product(
            query, reduced.observables[0], reduced.observables[1]
        );
        const std::vector<Complex> values = query.evaluate(reduced.program);
        const double left_mean = hermitian_mpi_value(
            evaluate_mpi_linear(values, left_indexed), "left observable expectation"
        );
        const double right_mean = hermitian_mpi_value(
            evaluate_mpi_linear(values, right_indexed), "right observable expectation"
        );
        const Complex product = evaluate_mpi_linear(values, product_indexed);
        return {
            product.real() - left_mean * right_mean,
            "native-mpi",
            reduced.active_qubits,
            query.size(),
        };
    }
    const std::size_t left_terms = reduced.observables[0].terms().size();
    const std::size_t right_terms = reduced.observables[1].terms().size();
    const std::size_t covariance_mean_terms = checked_term_sum(left_terms, right_terms);
    CudaPauliQuery cuda_query;
    std::vector<CudaLinearTerm> cuda_left;
    std::vector<CudaLinearTerm> cuda_right;
    std::vector<CudaLinearTerm> cuda_product;
    bool cuda_query_ready = false;
    if (needs_cuda_observable_query(reduced.program, backend, cost_model)) {
        cuda_left = index_observable(cuda_query, reduced.observables[0]);
        cuda_right = index_observable(cuda_query, reduced.observables[1]);
        cuda_product = index_observable_product(
            cuda_query, reduced.observables[0], reduced.observables[1]
        );
        cuda_query_ready = true;
    }
    const std::size_t cuda_evaluations = cuda_query_ready ? cuda_query.size() : covariance_mean_terms;
    if (use_cuda_observable(
            reduced.program,
            {checked_term_sum(covariance_mean_terms, covariance_mean_terms), 1U},
            {cuda_evaluations, 0U},
            backend,
            cost_model
        )) {
        if (!cuda_query_ready) {
            cuda_left = index_observable(cuda_query, reduced.observables[0]);
            cuda_right = index_observable(cuda_query, reduced.observables[1]);
            cuda_product = index_observable_product(
                cuda_query, reduced.observables[0], reduced.observables[1]
            );
        }
        const std::vector<Complex> values = cuda_query.evaluate(reduced.program, cuda_device_for_execution(backend));
        const double left_mean = hermitian_cuda_value(
            evaluate_cuda_linear(values, cuda_left), "left observable expectation"
        );
        const double right_mean = hermitian_cuda_value(
            evaluate_cuda_linear(values, cuda_right), "right observable expectation"
        );
        const Complex product = evaluate_cuda_linear(values, cuda_product);
        return {
            product.real() - left_mean * right_mean,
            cuda_backend_for_execution(backend),
            reduced.active_qubits,
            cuda_query.size(),
        };
    }
    StateVector state = statevector(reduced.program, backend, cost_model);
    std::vector<Complex> left_state;
    std::vector<Complex> right_state;
    apply_observable_to_state(state.values, reduced.observables[0], left_state);
    apply_observable_to_state(state.values, reduced.observables[1], right_state);
    Complex overlap{0.0, 0.0};
    for (std::size_t index = 0; index < state.values.size(); ++index) {
        overlap += std::conj(left_state[index]) * right_state[index];
    }
    const double left_mean = observable_expectation_from_state(state.values, reduced.observables[0]);
    const double right_mean = observable_expectation_from_state(state.values, reduced.observables[1]);
    return {overlap.real() - left_mean * right_mean, state.backend, reduced.active_qubits, 1U};
}

ObservableBatch expect_observables(
    const Program& program,
    const std::vector<Observable>& values,
    const std::string& backend,
    const PlannerCostModel* cost_model
) {
    if (values.empty()) {
        return {{}, 0U, backend == "auto" ? "native-cpu" : backend, 0U};
    }
    ReducedObservableSet reduced = reduce_observables(program, values);
    std::vector<double> results;
    results.reserve(values.size());
    if (use_rich_pauli_propagation(reduced.program, backend)) {
        for (const Observable& value : reduced.observables) {
            results.push_back(propagated_observable_expectation(reduced.program, value));
        }
        return {std::move(results), values.size(), "native-cpu", reduced.active_qubits};
    }
    if (explicit_mpi_backend(backend)) {
        MpiPauliQuery query;
        std::vector<std::vector<MpiLinearTerm>> indexed;
        indexed.reserve(reduced.observables.size());
        for (const Observable& value : reduced.observables) {
            indexed.push_back(index_mpi_observable(query, value));
        }
        const std::vector<Complex> mpi_values = query.evaluate(reduced.program);
        for (const auto& terms : indexed) {
            results.push_back(hermitian_mpi_value(
                evaluate_mpi_linear(mpi_values, terms), "observable expectation"
            ));
        }
        return {std::move(results), values.size(), "native-mpi", reduced.active_qubits};
    }
    std::size_t batch_evaluations = 0U;
    for (const Observable& value : reduced.observables) {
        batch_evaluations = checked_term_sum(batch_evaluations, value.terms().size());
    }
    CudaPauliQuery cuda_query;
    std::vector<std::vector<CudaLinearTerm>> cuda_indexed;
    bool cuda_query_ready = false;
    if (needs_cuda_observable_query(reduced.program, backend, cost_model)) {
        cuda_indexed.reserve(reduced.observables.size());
        for (const Observable& value : reduced.observables) {
            cuda_indexed.push_back(index_observable(cuda_query, value));
        }
        cuda_query_ready = true;
    }
    const std::size_t cuda_evaluations = cuda_query_ready ? cuda_query.size() : batch_evaluations;
    if (use_cuda_observable(
            reduced.program,
            {batch_evaluations, 0U},
            {cuda_evaluations, 0U},
            backend,
            cost_model
        )) {
        if (!cuda_query_ready) {
            cuda_indexed.reserve(reduced.observables.size());
            for (const Observable& value : reduced.observables) {
                cuda_indexed.push_back(index_observable(cuda_query, value));
            }
        }
        const std::vector<Complex> gpu_values = cuda_query.evaluate(reduced.program, cuda_device_for_execution(backend));
        for (const auto& terms : cuda_indexed) {
            results.push_back(hermitian_cuda_value(
                evaluate_cuda_linear(gpu_values, terms), "observable expectation"
            ));
        }
        return {
            std::move(results),
            values.size(),
            cuda_backend_for_execution(backend),
            reduced.active_qubits,
        };
    }
    StateVector state = statevector(reduced.program, backend, cost_model);
    for (const Observable& value : reduced.observables) {
        results.push_back(observable_expectation_from_state(state.values, value));
    }
    return {std::move(results), values.size(), state.backend, reduced.active_qubits};
}

GradientResult value_and_grad(
    const Program& program,
    const Observable& value,
    const std::vector<ParameterSlot>& slots,
    const std::vector<double>& parameter_values,
    const std::string& backend,
    GradientMethod method,
    double epsilon
) {
    if (backend == "native-tn") {
        return tensor_network_value_and_grad(
            program, value, slots, parameter_values, method, epsilon
        );
    }
    if (slots.size() != parameter_values.size()) {
        throw std::invalid_argument("parameter values must match parameter slots");
    }
    validate_slots(program, slots);
    validate_observable(value, program.num_qubits());
    if (slots.empty()) {
        const ObservableResult result = expect_observable(program, value, backend);
        return {result.value, {}, "none", result.backend, 1U};
    }
    GradientMethod selected = method;
    if (selected == GradientMethod::Auto) {
        bool adjoint_fits = false;
        if ((backend == "auto" || backend == "native-cpu" || backend == "cpu") &&
            program.num_qubits() < std::numeric_limits<std::size_t>::digits) {
            const std::size_t dimension = std::size_t{1} << program.num_qubits();
            const std::size_t states = program.operations().size() + 1U;
            adjoint_fits = states != 0U &&
                dimension <= kAdjointMemoryLimitBytes / sizeof(Complex) / states;
        }
        selected = adjoint_fits ? GradientMethod::Adjoint : GradientMethod::ParameterShift;
    }
    if (selected == GradientMethod::Adjoint) {
        if (backend != "auto" && backend != "native-cpu" && backend != "cpu") {
            throw std::invalid_argument("adjoint differentiation requires the native CPU backend");
        }
        return adjoint_gradient(program, value, slots, parameter_values);
    }
    if (selected == GradientMethod::FiniteDifference &&
        (!std::isfinite(epsilon) || epsilon <= 0.0)) {
        throw std::invalid_argument("finite-difference epsilon must be finite and positive");
    }
    Program bound = program.bound(slots, parameter_values);
    const ObservableResult base = expect_observable(bound, value, backend);
    std::vector<double> gradient(slots.size(), 0.0);
    const double shift = selected == GradientMethod::ParameterShift
        ? std::acos(-1.0) / 2.0 : epsilon;
    for (std::size_t index = 0; index < slots.size(); ++index) {
        std::vector<double> plus = parameter_values;
        std::vector<double> minus = parameter_values;
        plus[index] += shift;
        minus[index] -= shift;
        const double plus_value = expect_observable(program.bound(slots, plus), value, backend).value;
        const double minus_value = expect_observable(program.bound(slots, minus), value, backend).value;
        gradient[index] = selected == GradientMethod::ParameterShift
            ? 0.5 * (plus_value - minus_value)
            : (plus_value - minus_value) / (2.0 * shift);
    }
    return {
        base.value,
        std::move(gradient),
        selected == GradientMethod::ParameterShift ? "parameter-shift" : "finite-difference",
        base.backend,
        1U + 2U * slots.size(),
    };
}

GradientResult grad(
    const Program& program,
    const Observable& value,
    const std::vector<ParameterSlot>& slots,
    const std::vector<double>& parameter_values,
    const std::string& backend,
    GradientMethod method,
    double epsilon
) {
    return value_and_grad(program, value, slots, parameter_values, backend, method, epsilon);
}

JacobianResult jacobian(
    const Program& program,
    const std::vector<Observable>& values,
    const std::vector<ParameterSlot>& slots,
    const std::vector<double>& parameter_values,
    const std::string& backend,
    GradientMethod method,
    double epsilon
) {
    if (backend == "native-tn") {
        return tensor_network_jacobian(
            program, values, slots, parameter_values, method, epsilon
        );
    }
    if (slots.size() != parameter_values.size()) {
        throw std::invalid_argument("parameter values must match parameter slots");
    }
    validate_slots(program, slots);
    for (const Observable& value : values) {
        validate_observable(value, program.num_qubits());
    }
    if (values.empty()) {
        return {{}, {}, 0U, slots.size(), "none", backend == "auto" ? "native-cpu" : backend, 0U};
    }
    std::vector<double> result_values;
    std::vector<double> derivatives;
    result_values.reserve(values.size());
    derivatives.reserve(values.size() * slots.size());
    std::string selected_method;
    std::string executed_backend;
    std::size_t evaluations = 0U;
    for (const Observable& value : values) {
        GradientResult result = value_and_grad(
            program, value, slots, parameter_values, backend, method, epsilon
        );
        result_values.push_back(result.value);
        derivatives.insert(derivatives.end(), result.gradient.begin(), result.gradient.end());
        evaluations += result.evaluations;
        if (selected_method.empty()) {
            selected_method = result.method;
            executed_backend = result.backend;
        } else {
            if (selected_method != result.method) selected_method = "mixed";
            if (executed_backend != result.backend) executed_backend = "mixed";
        }
    }
    return {
        std::move(result_values), std::move(derivatives), values.size(), slots.size(),
        std::move(selected_method), std::move(executed_backend), evaluations,
    };
}

HessianResult hessian(
    const Program& program,
    const Observable& value,
    const std::vector<ParameterSlot>& slots,
    const std::vector<double>& parameter_values,
    const std::string& backend
) {
    if (backend == "native-tn") {
        return tensor_network_hessian(program, value, slots, parameter_values);
    }
    if (slots.size() != parameter_values.size()) {
        throw std::invalid_argument("parameter values must match parameter slots");
    }
    validate_slots(program, slots);
    validate_observable(value, program.num_qubits());
    if (slots.empty()) {
        const ObservableResult result = expect_observable(program, value, backend);
        return {result.value, {}, {}, 0U, "none", result.backend, 1U};
    }
    GradientResult first = value_and_grad(
        program, value, slots, parameter_values, backend, GradientMethod::ParameterShift
    );
    const std::size_t count = slots.size();
    std::vector<double> matrix(count * count, 0.0);
    const double pi = std::acos(-1.0);
    std::size_t evaluations = first.evaluations;
    const auto evaluate = [&](const std::vector<double>& parameters) {
        return expect_observable(program.bound(slots, parameters), value, backend).value;
    };
    for (std::size_t row = 0; row < count; ++row) {
        std::vector<double> plus = parameter_values;
        std::vector<double> minus = parameter_values;
        plus[row] += pi;
        minus[row] -= pi;
        matrix[row * count + row] =
            (evaluate(plus) - 2.0 * first.value + evaluate(minus)) / 4.0;
        evaluations += 2U;
        for (std::size_t column = row + 1U; column < count; ++column) {
            double mixed = 0.0;
            for (int row_sign : {-1, 1}) {
                for (int column_sign : {-1, 1}) {
                    std::vector<double> shifted = parameter_values;
                    shifted[row] += static_cast<double>(row_sign) * pi / 2.0;
                    shifted[column] += static_cast<double>(column_sign) * pi / 2.0;
                    mixed += static_cast<double>(row_sign * column_sign) * evaluate(shifted);
                    ++evaluations;
                }
            }
            mixed /= 4.0;
            matrix[row * count + column] = mixed;
            matrix[column * count + row] = mixed;
        }
    }
    return {
        first.value, std::move(first.gradient), std::move(matrix), count,
        "parameter-shift", std::move(first.backend), evaluations,
    };
}
OptimizationReport optimize(const Program& program, std::uint32_t level) {
    if (level == 0U || level > 2U) {
        throw std::invalid_argument("optimization level must be 1 or 2");
    }
    std::vector<Operation> optimized;
    optimized.reserve(program.operations().size());
    bool cancelled = false;
    bool merged = false;
    bool commuted = false;
    const auto same_qubits = [](const Operation& left, const Operation& right) {
        if (left.qubits == right.qubits) return true;
        const bool symmetric = left.code == OperationCode::CZ || left.code == OperationCode::SWAP;
        return symmetric && left.qubits.size() == 2U && right.qubits.size() == 2U &&
            left.qubits[0] == right.qubits[1] && left.qubits[1] == right.qubits[0];
    };
    const auto disjoint = [](const Operation& left, const Operation& right) {
        return std::none_of(left.qubits.begin(), left.qubits.end(), [&](std::size_t qubit) {
            return std::find(right.qubits.begin(), right.qubits.end(), qubit) != right.qubits.end();
        });
    };
    for (const Operation& operation : program.operations()) {
        std::optional<std::size_t> candidate;
        if (!optimized.empty()) {
            std::size_t index = optimized.size();
            while (index > 0U) {
                --index;
                const Operation& prior = optimized[index];
                if (prior.code == operation.code && same_qubits(prior, operation)) {
                    candidate = index;
                    break;
                }
                if (level == 1U || !disjoint(prior, operation)) break;
            }
        }
        if (!candidate.has_value()) {
            optimized.push_back(operation);
            continue;
        }
        if (*candidate + 1U != optimized.size()) commuted = true;
        Operation& prior = optimized[*candidate];
        if (is_rotation(operation.code)) {
            const double angle = prior.parameters[0] + operation.parameters[0];
            if (angle == 0.0) {
                optimized.erase(optimized.begin() + static_cast<std::ptrdiff_t>(*candidate));
                cancelled = true;
            } else {
                prior.parameters[0] = angle;
                merged = true;
            }
        } else {
            optimized.erase(optimized.begin() + static_cast<std::ptrdiff_t>(*candidate));
            cancelled = true;
        }
    }
    Program result(program.num_qubits());
    for (Operation operation : optimized) result = result.appended(std::move(operation));
    std::vector<std::string> passes;
    if (cancelled) passes.emplace_back("inverse-cancellation");
    if (merged) passes.emplace_back("rotation-merge");
    if (commuted) passes.emplace_back("disjoint-commutation");
    return {
        std::move(result), program.operations().size(), optimized.size(), std::move(passes),
    };
}
NoiseChannel bit_flip(std::size_t qubit, double probability) {
    validate_probability(probability, "bit-flip probability");
    return {NoiseChannelCode::BitFlip, qubit, {probability}, {}, 0U};
}

NoiseChannel phase_flip(std::size_t qubit, double probability) {
    validate_probability(probability, "phase-flip probability");
    return {NoiseChannelCode::PhaseFlip, qubit, {probability}, {}, 0U};
}

NoiseChannel depolarizing(std::size_t qubit, double probability) {
    validate_probability(probability, "depolarizing probability");
    return {NoiseChannelCode::Depolarizing, qubit, {probability}, {}, 0U};
}
NoiseChannel amplitude_damping(std::size_t qubit, double gamma) {
    validate_probability(gamma, "amplitude-damping gamma");
    return {NoiseChannelCode::AmplitudeDamping, qubit, {gamma}, {}, 0U};
}

NoiseChannel phase_damping(std::size_t qubit, double gamma) {
    validate_probability(gamma, "phase-damping gamma");
    return {NoiseChannelCode::PhaseDamping, qubit, {gamma}, {}, 0U};
}

NoiseChannel pauli_channel(
    std::size_t qubit,
    double probability_x,
    double probability_y,
    double probability_z
) {
    validate_probability(probability_x, "Pauli-X probability");
    validate_probability(probability_y, "Pauli-Y probability");
    validate_probability(probability_z, "Pauli-Z probability");
    if (probability_x + probability_y + probability_z > 1.0 + 1e-15) {
        throw std::invalid_argument("Pauli-channel probabilities must sum to at most 1");
    }
    return {
        NoiseChannelCode::Pauli,
        qubit,
        {probability_x, probability_y, probability_z},
        {},
        0U,
    };
}

NoiseChannel kraus_channel(
    std::size_t qubit,
    const std::vector<std::vector<Complex>>& operators
) {
    if (operators.empty()) {
        throw std::invalid_argument("Kraus channel requires at least one operator");
    }
    Matrix2 completeness{};
    std::vector<Complex> flattened;
    flattened.reserve(operators.size() * 4U);
    for (const std::vector<Complex>& values : operators) {
        if (values.size() != 4U) {
            throw std::invalid_argument("single-qubit Kraus operators must have shape 2x2");
        }
        for (std::size_t row = 0U; row < 2U; ++row) {
            for (std::size_t column = 0U; column < 2U; ++column) {
                Complex value{0.0, 0.0};
                for (std::size_t inner = 0U; inner < 2U; ++inner) {
                    value += std::conj(values[inner * 2U + row]) * values[inner * 2U + column];
                }
                completeness[row * 2U + column] += value;
            }
        }
        flattened.insert(flattened.end(), values.begin(), values.end());
    }
    const Matrix2 identity{1.0, 0.0, 0.0, 1.0};
    for (std::size_t index = 0U; index < completeness.size(); ++index) {
        if (std::abs(completeness[index] - identity[index]) > 1e-10) {
            throw std::invalid_argument("Kraus operators must satisfy the trace-preserving completeness relation");
        }
    }
    return {NoiseChannelCode::Kraus, qubit, {}, std::move(flattened), operators.size()};
}
[[nodiscard]] bool cuda_density_backend(const std::string& backend) {
    return detail::cuda_backend_device(backend).has_value();
}

[[nodiscard]] std::string cuda_density_backend_for_execution(const std::string& backend) {
    const std::size_t device = cuda_device_for_execution(backend);
    return device == 0U
        ? "native-cuda-density"
        : "native-cuda-density:" + std::to_string(device);
}

void validate_density_backend(const std::string& backend) {
    if (backend != "auto" && backend != "cpu" && backend != "native-cpu" &&
        !cuda_density_backend(backend)) {
        throw std::invalid_argument("unknown density-matrix backend: " + backend);
    }
}

[[nodiscard]] std::size_t density_dimension(const Program& program) {
    const std::size_t dimension = checked_dimension(program.num_qubits());
    if (dimension > std::numeric_limits<std::size_t>::max() / dimension) {
        throw std::length_error("density matrix exceeds native address space");
    }
    return dimension;
}

DensityMatrix density_matrix(
    const Program& program,
    const std::string& backend,
    const PlannerCostModel* cost_model
) {
    static_cast<void>(cost_model);
    validate_density_backend(backend);
    const std::size_t dimension = density_dimension(program);
    if (cuda_density_backend(backend)) {
        const std::size_t device = cuda_device_for_execution(backend);
        if (!detail::cuda_available(device)) {
            throw std::runtime_error(detail::cuda_unavailable_reason(device));
        }
        std::vector<Complex> values = detail::cuda_density_matrix(
            program.num_qubits(), cuda_density_steps(program), device
        );
        if (values.size() != dimension * dimension) {
            throw std::logic_error("CUDA density matrix returned an invalid dimension");
        }
        return {
            std::move(values), dimension, cuda_density_backend_for_execution(backend)
        };
    }

    StateVector state = statevector(program, "native-cpu");
    std::vector<Complex> values(dimension * dimension);
    for (std::size_t row = 0; row < dimension; ++row) {
        for (std::size_t column = 0; column < dimension; ++column) {
            values[row * dimension + column] = state.values[row] * std::conj(state.values[column]);
        }
    }
    return {std::move(values), dimension, "native-density"};
}

struct DensityCostWork {
    std::size_t single_qubit_operations = 0U;
    std::size_t two_qubit_operations = 0U;
    std::size_t noise_events = 0U;
    std::size_t kraus_evaluations = 0U;
};

[[nodiscard]] DensityCostWork density_cost_work(const NoisyProgram& noisy) {
    DensityCostWork work;
    for (const Operation& operation : noisy.program().operations()) {
        if (operation.qubits.size() == 1U) {
            ++work.single_qubit_operations;
        } else if (operation.qubits.size() == 2U) {
            ++work.two_qubit_operations;
        }
    }
    work.noise_events = noisy.noise().size();
    for (const NoiseInstruction& instruction : noisy.noise()) {
        work.kraus_evaluations += noise_kraus_operators(instruction.channel).size();
    }
    return work;
}

[[nodiscard]] bool cuda_density_fits(std::size_t density_values) noexcept {
    if (!detail::cuda_available()) {
        return false;
    }
    const std::size_t memory = detail::cuda_total_memory_bytes();
    const std::size_t reserve = sizeof(std::array<Complex, 16>);
    return memory > reserve && density_values <= (memory - reserve) / sizeof(Complex);
}

DensityMatrix density_matrix(
    const NoisyProgram& noisy,
    const std::string& backend,
    const PlannerCostModel* cost_model
) {
    validate_density_backend(backend);
    const Program& program = noisy.program();
    const std::size_t dimension = density_dimension(program);
    const std::size_t density_values = dimension * dimension;
    const std::size_t cuda_device = cuda_device_for_execution(backend);
    bool use_cuda = cuda_density_backend(backend);
    if (
        backend == "auto" && cost_model != nullptr && cost_model->density_auto_validated() &&
        cuda_density_fits(density_values)
    ) {
        const DensityCostWork work = density_cost_work(noisy);
        if (work.noise_events != 0U) {
            const double speedup = cost_model->predict_density_speedup(
                program.num_qubits(), work.single_qubit_operations,
                work.two_qubit_operations, work.noise_events, work.kraus_evaluations
            );
            use_cuda = speedup > 1.0;
        }
    }
    if (use_cuda) {
        if (!detail::cuda_available(cuda_device)) {
            throw std::runtime_error(detail::cuda_unavailable_reason(cuda_device));
        }
        std::vector<Complex> values = detail::cuda_density_matrix(
            program.num_qubits(), cuda_density_steps(noisy), cuda_device
        );
        if (values.size() != dimension * dimension) {
            throw std::logic_error("CUDA density matrix returned an invalid dimension");
        }
        return {
            std::move(values), dimension, cuda_density_backend_for_execution(backend)
        };
    }

    std::vector<Complex> rho(dimension * dimension, Complex{0.0, 0.0});
    rho.front() = 1.0;
    std::size_t noise_index = 0U;
    const auto apply_at = [&](std::size_t point, std::size_t& cursor, std::vector<Complex>& state) {
        while (cursor < noisy.noise().size() && noisy.noise()[cursor].after_operation == point) {
            apply_noise_channel(state, dimension, noisy.noise()[cursor].channel);
            ++cursor;
        }
    };
    apply_at(0U, noise_index, rho);
    for (std::size_t index = 0; index < program.operations().size(); ++index) {
        apply_operation_density(rho, dimension, program.operations()[index]);
        apply_at(index + 1U, noise_index, rho);
    }
    return {std::move(rho), dimension, "native-density"};
}

LindbladResult lindblad_evolve(
    const DensityMatrix& initial,
    const std::vector<Complex>& hamiltonian,
    const std::vector<std::vector<Complex>>& collapse_operators,
    double dt,
    std::size_t steps
) {
    if (!std::isfinite(dt) || dt <= 0.0) {
        throw std::invalid_argument("Lindblad time step must be finite and positive");
    }
    if (steps == 0U) {
        throw std::invalid_argument("Lindblad step count must be positive");
    }
    const std::size_t dimension = initial.dimension;
    if (dimension == 0U || initial.values.size() != dimension * dimension ||
        hamiltonian.size() != dimension * dimension) {
        throw std::invalid_argument("Lindblad matrices must match the density-matrix dimension");
    }
    for (const auto& collapse : collapse_operators) {
        if (collapse.size() != dimension * dimension) {
            throw std::invalid_argument("collapse operator must match the density-matrix dimension");
        }
    }
    std::vector<Complex> rho = initial.values;
    for (std::size_t step = 0; step < steps; ++step) {
        const std::vector<Complex> k1 = lindblad_derivative(
            rho, hamiltonian, collapse_operators, dimension
        );
        const std::vector<Complex> k2 = lindblad_derivative(
            add_scaled(rho, k1, dt * 0.5), hamiltonian, collapse_operators, dimension
        );
        const std::vector<Complex> k3 = lindblad_derivative(
            add_scaled(rho, k2, dt * 0.5), hamiltonian, collapse_operators, dimension
        );
        const std::vector<Complex> k4 = lindblad_derivative(
            add_scaled(rho, k3, dt), hamiltonian, collapse_operators, dimension
        );
        for (std::size_t index = 0; index < rho.size(); ++index) {
            rho[index] += dt * (k1[index] + 2.0 * k2[index] + 2.0 * k3[index] + k4[index]) / 6.0;
        }
        restore_density_invariants(rho, dimension);
    }
    return {{std::move(rho), dimension, "native-lindblad-rk4"}, dt, steps};
}

ProviderProgram to_openqasm3(const Program& program, bool measure_all) {
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << "OPENQASM 3.1;\n";
    output << "include \"stdgates.inc\";\n";
    output << "qubit[" << program.num_qubits() << "] q;\n";
    if (measure_all) {
        output << "bit[" << program.num_qubits() << "] c;\n";
    }
    for (const Operation& operation : program.operations()) {
        output << qasm_operation(operation) << '\n';
    }
    if (measure_all) {
        output << "c = measure q;\n";
    }
    return {"openqasm3", output.str(), program.num_qubits(), measure_all};
}
ProviderProgram to_qir_base_profile(const Program& program, bool measure_all) {
    std::ostringstream output;
    output.imbue(std::locale::classic());
    if (measure_all) {
        output << "@tuple_label = internal constant [8 x i8] c\"results\\00\"\n";
        for (std::size_t qubit = 0; qubit < program.num_qubits(); ++qubit) {
            const std::string label = "r" + std::to_string(qubit);
            output << "@result_" << qubit << " = internal constant [" << label.size() + 1U
                   << " x i8] c\"" << label << "\\00\"\n";
        }
        output << '\n';
    }
    output << "define i64 @qupy_entry() #0 {\n";
    output << "entry:\n";
    output << "  call void @__quantum__rt__initialize(ptr null)\n";
    output << "  br label %body\n\n";
    output << "body:\n";
    for (const Operation& operation : program.operations()) {
        for (const std::string& call : qir_calls(operation)) {
            output << call << '\n';
        }
    }
    output << "  br label %measurements\n\n";
    output << "measurements:\n";
    if (measure_all) {
        for (std::size_t qubit = 0; qubit < program.num_qubits(); ++qubit) {
            const std::string result = qubit == 0U
                ? "ptr null" : "ptr inttoptr (i64 " + std::to_string(qubit) + " to ptr)";
            output << "  call void @__quantum__qis__mz__body(" << qir_qubit(qubit)
                   << ", " << result << ")\n";
        }
    }
    output << "  br label %output\n\n";
    output << "output:\n";
    if (measure_all) {
        output << "  call void @__quantum__rt__tuple_record_output(i64 " << program.num_qubits()
               << ", ptr @tuple_label)\n";
        for (std::size_t qubit = 0; qubit < program.num_qubits(); ++qubit) {
            const std::string result = qubit == 0U
                ? "ptr null" : "ptr inttoptr (i64 " + std::to_string(qubit) + " to ptr)";
            output << "  call void @__quantum__rt__result_record_output(" << result
                   << ", ptr @result_" << qubit << ")\n";
        }
    }
    output << "  ret i64 0\n}\n\n";
    output << "declare void @__quantum__qis__h__body(ptr)\n";
    output << "declare void @__quantum__qis__x__body(ptr)\n";
    output << "declare void @__quantum__qis__y__body(ptr)\n";
    output << "declare void @__quantum__qis__z__body(ptr)\n";
    output << "declare void @__quantum__qis__rx__body(double, ptr)\n";
    output << "declare void @__quantum__qis__ry__body(double, ptr)\n";
    output << "declare void @__quantum__qis__rz__body(double, ptr)\n";
    output << "declare void @__quantum__qis__cnot__body(ptr, ptr)\n";
    output << "declare void @__quantum__qis__cz__body(ptr, ptr)\n";
    output << "declare void @__quantum__qis__mz__body(ptr, ptr writeonly) #1\n";
    output << "declare void @__quantum__rt__initialize(ptr)\n";
    output << "declare void @__quantum__rt__tuple_record_output(i64, ptr)\n";
    output << "declare void @__quantum__rt__result_record_output(ptr, ptr)\n\n";
    output << "attributes #0 = { \"entry_point\" \"qir_profiles\"=\"base_profile\" "
              "\"output_labeling_schema\"=\"qupy_v1\" \"required_num_qubits\"=\""
           << program.num_qubits() << "\" \"required_num_results\"=\""
           << (measure_all ? program.num_qubits() : 0U) << "\" }\n";
    output << "attributes #1 = { \"irreversible\" }\n\n";
    output << "!llvm.module.flags = !{!0, !1, !2, !3}\n";
    output << "!0 = !{i32 1, !\"qir_major_version\", i32 2}\n";
    output << "!1 = !{i32 7, !\"qir_minor_version\", i32 0}\n";
    output << "!2 = !{i32 1, !\"dynamic_qubit_management\", i1 false}\n";
    output << "!3 = !{i32 1, !\"dynamic_result_management\", i1 false}\n";
    return {"qir-base-profile", output.str(), program.num_qubits(), measure_all};
}
DetectorModel repetition_code_detector_model(
    std::size_t distance,
    std::size_t rounds,
    double data_error_probability,
    double measurement_error_probability
) {
    if (distance < 2U) {
        throw std::invalid_argument("repetition-code distance must be at least 2");
    }
    if (rounds == 0U) {
        throw std::invalid_argument("repetition-code rounds must be positive");
    }
    validate_probability(data_error_probability, "data error probability");
    validate_probability(measurement_error_probability, "measurement error probability");
    const std::size_t checks = distance - 1U;
    const std::size_t detector_count = checks * rounds;
    const auto detector = [checks](std::size_t round, std::size_t check) {
        return round * checks + check;
    };
    std::vector<DetectorError> errors;
    errors.reserve(rounds * (distance + checks));
    for (std::size_t round = 0; round < rounds; ++round) {
        for (std::size_t data = 0; data < distance; ++data) {
            std::vector<std::size_t> detectors;
            if (data > 0U) {
                detectors.push_back(detector(round, data - 1U));
            }
            if (data < checks) {
                detectors.push_back(detector(round, data));
            }
            std::vector<std::size_t> logical;
            if (data == 0U || data + 1U == distance) {
                logical.push_back(0U);
            }
            errors.push_back({data_error_probability, std::move(detectors), std::move(logical)});
        }
        for (std::size_t check = 0; check < checks; ++check) {
            std::vector<std::size_t> detectors{detector(round, check)};
            if (round + 1U < rounds) {
                detectors.push_back(detector(round + 1U, check));
            }
            errors.push_back({measurement_error_probability, std::move(detectors), {}});
        }
    }
    return DetectorModel(detector_count, 1U, std::move(errors));
}

DetectorSamples sample_detector_model(
    const DetectorModel& model,
    std::size_t shots,
    std::optional<std::uint64_t> seed
) {
    if (shots == 0U) {
        throw std::invalid_argument("detector-model shots must be positive");
    }
    const std::uint64_t actual_seed = seed.value_or(
        (static_cast<std::uint64_t>(std::random_device{}()) << 32U) ^ std::random_device{}()
    );
    std::mt19937_64 generator(actual_seed);
    const auto uniform = [&generator]() {
        return static_cast<double>(generator() >> 11U) * (1.0 / 9007199254740992.0);
    };
    std::vector<std::int8_t> syndrome(shots * model.detector_count(), 0);
    std::vector<std::int8_t> observables(shots * model.observable_count(), 0);
    for (std::size_t shot = 0; shot < shots; ++shot) {
        for (const DetectorError& error : model.errors()) {
            if (uniform() >= error.probability) {
                continue;
            }
            for (const std::size_t detector_index : error.detectors) {
                syndrome[shot * model.detector_count() + detector_index] ^= 1;
            }
            for (const std::size_t observable_index : error.observables) {
                observables[shot * model.observable_count() + observable_index] ^= 1;
            }
        }
    }
    return {
        std::move(syndrome), std::move(observables), shots,
        model.detector_count(), model.observable_count(),
    };
}
DecodeResult decode_detector_model(
    const DetectorModel& model,
    const std::vector<std::int8_t>& syndrome
) {
    if (syndrome.size() != model.detector_count()) {
        throw std::invalid_argument("syndrome length must match detector count");
    }
    for (const std::int8_t bit : syndrome) {
        if (bit != 0 && bit != 1) {
            throw std::invalid_argument("syndrome values must be zero or one");
        }
    }
    if (model.errors().size() > kReferenceDecoderErrorLimit) {
        throw std::length_error("reference decoder supports at most 24 error mechanisms");
    }
    const std::uint64_t combinations = std::uint64_t{1} << model.errors().size();
    double best_log_likelihood = -std::numeric_limits<double>::infinity();
    std::uint64_t best_mask = 0U;
    bool found = false;
    for (std::uint64_t mask = 0U; mask < combinations; ++mask) {
        std::vector<std::int8_t> candidate(model.detector_count(), 0);
        double log_likelihood = 0.0;
        bool possible = true;
        for (std::size_t error_index = 0; error_index < model.errors().size(); ++error_index) {
            const DetectorError& error = model.errors()[error_index];
            const bool selected = (mask & (std::uint64_t{1} << error_index)) != 0U;
            const double probability = selected ? error.probability : 1.0 - error.probability;
            if (probability <= 0.0) {
                possible = false;
                break;
            }
            log_likelihood += std::log(probability);
            if (selected) {
                for (const std::size_t detector_index : error.detectors) {
                    candidate[detector_index] ^= 1;
                }
            }
        }
        if (possible && candidate == syndrome && (!found || log_likelihood > best_log_likelihood)) {
            found = true;
            best_log_likelihood = log_likelihood;
            best_mask = mask;
        }
    }
    if (!found) {
        throw std::domain_error("detector model has no error configuration matching the syndrome");
    }
    std::vector<std::int8_t> logical(model.observable_count(), 0);
    for (std::size_t error_index = 0; error_index < model.errors().size(); ++error_index) {
        if ((best_mask & (std::uint64_t{1} << error_index)) == 0U) {
            continue;
        }
        for (const std::size_t observable_index : model.errors()[error_index].observables) {
            logical[observable_index] ^= 1;
        }
    }
    return {
        std::move(logical),
        best_log_likelihood,
        static_cast<std::size_t>(std::popcount(best_mask)),
    };
}

DistributedInfo distributed_info() {
    const auto first = [](std::initializer_list<const char*> names, std::size_t fallback) {
        for (const char* name : names) {
            if (const auto value = environment_size(name); value.has_value()) {
                return *value;
            }
        }
        return fallback;
    };
    const std::size_t world_size = first(
        {"OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "MV2_COMM_WORLD_SIZE", "WORLD_SIZE"}, 1U
    );
    const std::size_t rank = first(
        {"OMPI_COMM_WORLD_RANK", "PMI_RANK", "MV2_COMM_WORLD_RANK", "RANK"}, 0U
    );
    const std::size_t local_rank = first(
        {"OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID", "LOCAL_RANK"}, 0U
    );
    std::string runtime = "none";
    if (environment_value("OMPI_COMM_WORLD_SIZE").has_value()) runtime = "openmpi";
    else if (environment_value("MV2_COMM_WORLD_SIZE").has_value()) runtime = "mvapich";
    else if (environment_value("PMI_SIZE").has_value()) runtime = "pmi";
    else if (environment_value("WORLD_SIZE").has_value()) runtime = "launcher";
    return {world_size > 1U, world_size, rank, local_rank, std::move(runtime)};
}

}  // namespace qupy
