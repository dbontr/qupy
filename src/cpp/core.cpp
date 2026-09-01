#include "qupy/core.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

#ifdef QUPY_HAS_OPENMP
#include <omp.h>
#endif

namespace qupy {
namespace {

#ifdef QUPY_HAS_OPENMP
constexpr std::size_t kParallelThreshold = 1U << 16U;
#endif

constexpr std::uint32_t kIrVersion = 1U;
constexpr std::string_view kCoreVersion = "0.3.0a0";

struct OperationSpec {
    std::size_t qubits;
    std::size_t parameters;
};

[[nodiscard]] OperationSpec operation_spec(OperationCode code) {
    switch (code) {
    case OperationCode::H:
    case OperationCode::X:
    case OperationCode::Y:
    case OperationCode::Z:
        return {1, 0};
    case OperationCode::RX:
    case OperationCode::RY:
    case OperationCode::RZ:
        return {1, 1};
    case OperationCode::CX:
    case OperationCode::CZ:
    case OperationCode::SWAP:
        return {2, 0};
    }
    throw std::invalid_argument("unknown operation code");
}

[[nodiscard]] const char* operation_name(OperationCode code) {
    switch (code) {
    case OperationCode::H: return "h";
    case OperationCode::X: return "x";
    case OperationCode::Y: return "y";
    case OperationCode::Z: return "z";
    case OperationCode::RX: return "rx";
    case OperationCode::RY: return "ry";
    case OperationCode::RZ: return "rz";
    case OperationCode::CX: return "cx";
    case OperationCode::CZ: return "cz";
    case OperationCode::SWAP: return "swap";
    }
    throw std::invalid_argument("unknown operation code");
}

[[nodiscard]] const char* result_mode_name(ResultMode mode) {
    switch (mode) {
    case ResultMode::Sample: return "sample";
    case ResultMode::Expectation: return "expectation";
    case ResultMode::Probabilities: return "probabilities";
    case ResultMode::Variance: return "variance";
    case ResultMode::StateVector: return "statevector";
    }
    throw std::invalid_argument("unknown result mode");
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
    std::vector<std::uint8_t> message;
    message.reserve(text.size() + 72U);
    for (const char character : text) {
        message.push_back(static_cast<std::uint8_t>(static_cast<unsigned char>(character)));
    }
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

[[nodiscard]] std::size_t state_dimension(std::size_t num_qubits) {
    if (num_qubits >= std::numeric_limits<std::size_t>::digits) {
        throw std::length_error("qubit count exceeds native address space");
    }
    return std::size_t{1} << num_qubits;
}

[[nodiscard]] std::size_t state_memory_bytes(std::size_t num_qubits) {
    const std::size_t dimension = state_dimension(num_qubits);
    if (dimension > std::numeric_limits<std::size_t>::max() / sizeof(Complex)) {
        throw std::length_error("state vector exceeds native address space");
    }
    return dimension * sizeof(Complex);
}

using Matrix2 = std::array<Complex, 4>;

enum class CompiledKind : std::uint8_t {
    Single,
    CX,
    CZ,
    SWAP,
};

struct CompiledStep {
    CompiledKind kind;
    Matrix2 matrix;
    std::size_t first;
    std::size_t second;
};

[[nodiscard]] Matrix2 identity_matrix() {
    return {1.0, 0.0, 0.0, 1.0};
}

[[nodiscard]] Matrix2 multiply(const Matrix2& left, const Matrix2& right) {
    return {
        left[0] * right[0] + left[1] * right[2],
        left[0] * right[1] + left[1] * right[3],
        left[2] * right[0] + left[3] * right[2],
        left[2] * right[1] + left[3] * right[3],
    };
}

[[nodiscard]] bool is_single_qubit(OperationCode code) {
    return operation_spec(code).qubits == 1U;
}

[[nodiscard]] Matrix2 single_matrix(const Operation& operation) {
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
        const double half = operation.parameters.front() / 2.0;
        const double cosine = std::cos(half);
        const Complex sine{0.0, -std::sin(half)};
        return {cosine, sine, sine, cosine};
    }
    case OperationCode::RY: {
        const double half = operation.parameters.front() / 2.0;
        const double cosine = std::cos(half);
        const double sine = std::sin(half);
        return {cosine, -sine, sine, cosine};
    }
    case OperationCode::RZ: {
        const double half = operation.parameters.front() / 2.0;
        return {std::polar(1.0, -half), 0.0, 0.0, std::polar(1.0, half)};
    }
    case OperationCode::CX:
    case OperationCode::CZ:
    case OperationCode::SWAP:
        break;
    }
    throw std::invalid_argument("operation is not a single-qubit gate");
}

[[nodiscard]] CompiledKind compiled_kind(OperationCode code) {
    switch (code) {
    case OperationCode::CX: return CompiledKind::CX;
    case OperationCode::CZ: return CompiledKind::CZ;
    case OperationCode::SWAP: return CompiledKind::SWAP;
    default: break;
    }
    throw std::invalid_argument("operation is not a two-qubit gate");
}

[[nodiscard]] std::vector<CompiledStep> compile_program(const Program& program) {
    const Matrix2 identity = identity_matrix();
    std::vector<Matrix2> pending(program.num_qubits(), identity);
    std::vector<bool> has_pending(program.num_qubits(), false);
    std::vector<CompiledStep> steps;
    steps.reserve(program.operations().size());

    const auto flush = [&](std::size_t qubit) {
        if (!has_pending[qubit]) {
            return;
        }
        steps.push_back({CompiledKind::Single, pending[qubit], qubit, 0U});
        pending[qubit] = identity;
        has_pending[qubit] = false;
    };

    for (const Operation& operation : program.operations()) {
        if (is_single_qubit(operation.code)) {
            const std::size_t qubit = operation.qubits.front();
            pending[qubit] = multiply(single_matrix(operation), pending[qubit]);
            has_pending[qubit] = true;
            continue;
        }

        const std::size_t first = operation.qubits[0];
        const std::size_t second = operation.qubits[1];
        flush(first);
        flush(second);
        steps.push_back({compiled_kind(operation.code), identity, first, second});
    }

    for (std::size_t qubit = 0; qubit < program.num_qubits(); ++qubit) {
        flush(qubit);
    }
    return steps;
}

struct PreparedProgram {
    ExecutionPlan execution_plan;
    std::vector<CompiledStep> steps;
};

struct PreparedExpectation {
    Program program;
    std::size_t observable_qubit;
    ExecutionPlan execution_plan;
    std::vector<CompiledStep> steps;
};

void validate_backend(const std::string& backend) {
    if (backend != "auto" && backend != "native" && backend != "native-cpu") {
        throw std::invalid_argument("unknown backend: " + backend);
    }
}

[[nodiscard]] ExecutionPlan make_plan(
    const Program& source_program,
    const Target& target,
    ResultMode result_mode,
    std::size_t original_operations,
    std::size_t compiled_steps,
    std::size_t active_qubits,
    const std::string& method,
    std::string_view qualifier = {}
) {
    const std::string program_fingerprint = source_program.fingerprint();
    const std::string target_fingerprint = target.fingerprint();
    std::ostringstream cache_key;
    cache_key.imbue(std::locale::classic());
    cache_key << "qupy-cache/1/" << kCoreVersion << '/' << program_fingerprint << '/'
              << target_fingerprint << '/' << result_mode_name(result_mode) << '/' << method;
    if (!qualifier.empty()) {
        cache_key << '/' << qualifier;
    }

    return {
        target.name,
        method,
        true,
        parallel_threads(),
        original_operations,
        compiled_steps,
        active_qubits,
        state_memory_bytes(active_qubits),
        result_mode,
        program_fingerprint,
        target_fingerprint,
        cache_key.str(),
    };
}

[[nodiscard]] std::size_t insert_zero_bit(std::size_t value, std::size_t position) {
    const std::size_t low_mask = position == 0U
        ? 0U
        : (std::size_t{1} << position) - 1U;
    const std::size_t low = value & low_mask;
    const std::size_t high = value & ~low_mask;
    return low | (high << 1U);
}

[[nodiscard]] std::size_t expand_two_zero_bits(
    std::size_t value,
    std::size_t first,
    std::size_t second
) {
    const std::size_t low = std::min(first, second);
    const std::size_t high = std::max(first, second);
    return insert_zero_bit(insert_zero_bit(value, low), high);
}

void apply_single(
    std::vector<Complex>& state,
    const Matrix2& matrix,
    std::size_t qubit
) {
    const std::size_t step = std::size_t{1} << qubit;
    const std::size_t block = step << 1U;
    const auto dimension = static_cast<std::ptrdiff_t>(state.size());
    const auto stride = static_cast<std::ptrdiff_t>(block);

#ifdef QUPY_HAS_OPENMP
#pragma omp parallel for schedule(static) if(state.size() >= kParallelThreshold)
#endif
    for (std::ptrdiff_t base = 0; base < dimension; base += stride) {
        const auto first = static_cast<std::size_t>(base);
        for (std::size_t offset = 0; offset < step; ++offset) {
            const std::size_t zero = first + offset;
            const std::size_t one = zero + step;
            const Complex a = state[zero];
            const Complex b = state[one];
            state[zero] = matrix[0] * a + matrix[1] * b;
            state[one] = matrix[2] * a + matrix[3] * b;
        }
    }
}

void apply_cx(
    std::vector<Complex>& state,
    std::size_t control,
    std::size_t target
) {
    const std::size_t control_mask = std::size_t{1} << control;
    const std::size_t target_mask = std::size_t{1} << target;
    const auto pairs = static_cast<std::ptrdiff_t>(state.size() >> 2U);

#ifdef QUPY_HAS_OPENMP
#pragma omp parallel for schedule(static) if(state.size() >= kParallelThreshold)
#endif
    for (std::ptrdiff_t raw = 0; raw < pairs; ++raw) {
        const std::size_t base = expand_two_zero_bits(
            static_cast<std::size_t>(raw), control, target
        );
        const std::size_t zero_target = base | control_mask;
        std::swap(state[zero_target], state[zero_target | target_mask]);
    }
}

void apply_cz(
    std::vector<Complex>& state,
    std::size_t control,
    std::size_t target
) {
    const std::size_t control_mask = std::size_t{1} << control;
    const std::size_t target_mask = std::size_t{1} << target;
    const auto pairs = static_cast<std::ptrdiff_t>(state.size() >> 2U);

#ifdef QUPY_HAS_OPENMP
#pragma omp parallel for schedule(static) if(state.size() >= kParallelThreshold)
#endif
    for (std::ptrdiff_t raw = 0; raw < pairs; ++raw) {
        const std::size_t base = expand_two_zero_bits(
            static_cast<std::size_t>(raw), control, target
        );
        const std::size_t both_one = base | control_mask | target_mask;
        state[both_one] = -state[both_one];
    }
}

void apply_swap(
    std::vector<Complex>& state,
    std::size_t first,
    std::size_t second
) {
    const std::size_t first_mask = std::size_t{1} << first;
    const std::size_t second_mask = std::size_t{1} << second;
    const auto pairs = static_cast<std::ptrdiff_t>(state.size() >> 2U);

#ifdef QUPY_HAS_OPENMP
#pragma omp parallel for schedule(static) if(state.size() >= kParallelThreshold)
#endif
    for (std::ptrdiff_t raw = 0; raw < pairs; ++raw) {
        const std::size_t base = expand_two_zero_bits(
            static_cast<std::size_t>(raw), first, second
        );
        const std::size_t first_one = base | first_mask;
        const std::size_t second_one = base | second_mask;
        std::swap(state[first_one], state[second_one]);
    }
}

[[nodiscard]] PreparedProgram prepare_program(
    const Program& program,
    ResultMode result_mode,
    const std::string& backend
) {
    validate_backend(backend);
    const Target target = native_target();
    target.validate(program, result_mode);
    std::vector<CompiledStep> steps = compile_program(program);
    ExecutionPlan execution_plan = make_plan(
        program,
        target,
        result_mode,
        program.operations().size(),
        steps.size(),
        program.num_qubits(),
        "statevector"
    );
    return {std::move(execution_plan), std::move(steps)};
}

[[nodiscard]] PreparedExpectation prepare_expectation(
    const Program& program,
    PauliZ observable,
    ResultMode result_mode,
    const std::string& backend
) {
    if (observable.qubit >= program.num_qubits()) {
        throw std::invalid_argument("observable qubit is outside this program");
    }
    if (result_mode != ResultMode::Expectation && result_mode != ResultMode::Variance) {
        throw std::invalid_argument("observable plan requires expectation or variance mode");
    }
    validate_backend(backend);
    const Target target = native_target();
    target.validate(program, result_mode);

    const auto& operations = program.operations();
    std::vector<bool> active(program.num_qubits(), false);
    std::vector<bool> retained(operations.size(), false);
    active[observable.qubit] = true;

    for (std::size_t index = operations.size(); index > 0U; --index) {
        const Operation& operation = operations[index - 1U];
        if (is_single_qubit(operation.code)) {
            if (active[operation.qubits[0]]) {
                retained[index - 1U] = true;
            }
            continue;
        }

        const std::size_t first = operation.qubits[0];
        const std::size_t second = operation.qubits[1];
        if (active[first] || active[second]) {
            retained[index - 1U] = true;
            active[first] = true;
            active[second] = true;
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

    Program reduced(active_count);
    for (std::size_t index = 0; index < operations.size(); ++index) {
        if (!retained[index]) {
            continue;
        }
        Operation operation = operations[index];
        for (std::size_t& qubit : operation.qubits) {
            qubit = mapping[qubit];
        }
        reduced = reduced.appended(std::move(operation));
    }

    std::vector<CompiledStep> steps = compile_program(reduced);
    const std::string method = active_count < program.num_qubits()
        ? "statevector-lightcone"
        : "statevector";
    const std::string qualifier = "z:" + std::to_string(observable.qubit);
    ExecutionPlan execution_plan = make_plan(
        program,
        target,
        result_mode,
        operations.size(),
        steps.size(),
        active_count,
        method,
        qualifier
    );
    return {
        std::move(reduced),
        mapping[observable.qubit],
        std::move(execution_plan),
        std::move(steps),
    };
}

[[nodiscard]] StateVector run_statevector(
    std::size_t num_qubits,
    const std::vector<CompiledStep>& steps,
    const ExecutionPlan& execution_plan
) {
    std::vector<Complex> state(state_dimension(num_qubits), Complex{0.0, 0.0});
    state.front() = Complex{1.0, 0.0};

    for (const CompiledStep& step : steps) {
        switch (step.kind) {
        case CompiledKind::Single:
            apply_single(state, step.matrix, step.first);
            break;
        case CompiledKind::CX:
            apply_cx(state, step.first, step.second);
            break;
        case CompiledKind::CZ:
            apply_cz(state, step.first, step.second);
            break;
        case CompiledKind::SWAP:
            apply_swap(state, step.first, step.second);
            break;
        }
    }

    return {std::move(state), execution_plan.backend};
}

[[nodiscard]] std::vector<double> state_probabilities(const std::vector<Complex>& state) {
    std::vector<double> probabilities(state.size());
#ifdef QUPY_HAS_OPENMP
#pragma omp parallel for schedule(static) if(state.size() >= kParallelThreshold)
#endif
    for (std::ptrdiff_t raw = 0; raw < static_cast<std::ptrdiff_t>(state.size()); ++raw) {
        const auto index = static_cast<std::size_t>(raw);
        probabilities[index] = std::norm(state[index]);
    }
    return probabilities;
}

[[nodiscard]] double pauli_z_value(
    const std::vector<Complex>& state,
    std::size_t qubit
) {
    const std::size_t mask = std::size_t{1} << qubit;
    double value = 0.0;
#ifdef QUPY_HAS_OPENMP
#pragma omp parallel for reduction(+ : value) schedule(static) if(state.size() >= kParallelThreshold)
#endif
    for (std::ptrdiff_t raw = 0; raw < static_cast<std::ptrdiff_t>(state.size()); ++raw) {
        const auto index = static_cast<std::size_t>(raw);
        const double sign = (index & mask) == 0U ? 1.0 : -1.0;
        value += sign * std::norm(state[index]);
    }
    return value;
}

[[nodiscard]] std::uint64_t unbiased_index(
    std::mt19937_64& generator,
    std::uint64_t bound
) {
    if (bound == 0U) {
        throw std::invalid_argument("random bound must be positive");
    }
    const std::uint64_t threshold = (std::uint64_t{0} - bound) % bound;
    std::uint64_t value = 0U;
    do {
        value = generator();
    } while (value < threshold);
    return value % bound;
}

[[nodiscard]] double unit_interval(std::mt19937_64& generator) {
    constexpr double scale = 1.0 / 9007199254740992.0;
    return static_cast<double>(generator() >> 11U) * scale;
}

class AliasSampler {
public:
    explicit AliasSampler(const std::vector<double>& weights)
        : accept_(weights.size(), 1.0), alias_(weights.size(), 0U) {
        if (weights.empty()) {
            throw std::invalid_argument("sampling distribution is empty");
        }
        double total = 0.0;
        for (const double weight : weights) {
            if (!std::isfinite(weight) || weight < 0.0) {
                throw std::invalid_argument("sampling distribution contains invalid weights");
            }
            total += weight;
        }
        if (!(total > 0.0)) {
            throw std::invalid_argument("sampling distribution has zero mass");
        }

        const std::size_t count = weights.size();
        std::vector<double> scaled(count);
        std::vector<std::size_t> small;
        std::vector<std::size_t> large;
        small.reserve(count);
        large.reserve(count);
        const double scale = static_cast<double>(count) / total;
        for (std::size_t index = 0; index < count; ++index) {
            scaled[index] = weights[index] * scale;
            (scaled[index] < 1.0 ? small : large).push_back(index);
        }
        while (!small.empty() && !large.empty()) {
            const std::size_t low = small.back();
            small.pop_back();
            const std::size_t high = large.back();
            large.pop_back();
            accept_[low] = scaled[low];
            alias_[low] = high;
            scaled[high] = (scaled[high] + scaled[low]) - 1.0;
            (scaled[high] < 1.0 ? small : large).push_back(high);
        }
        for (const std::size_t index : large) {
            accept_[index] = 1.0;
            alias_[index] = index;
        }
        for (const std::size_t index : small) {
            accept_[index] = 1.0;
            alias_[index] = index;
        }
    }

    [[nodiscard]] std::size_t draw(std::mt19937_64& generator) const {
        const std::size_t index = static_cast<std::size_t>(
            unbiased_index(generator, static_cast<std::uint64_t>(accept_.size()))
        );
        return unit_interval(generator) < accept_[index] ? index : alias_[index];
    }

private:
    std::vector<double> accept_;
    std::vector<std::size_t> alias_;
};

}  // namespace

std::string Operation::name() const {
    return operation_name(code);
}

Program::Program(std::size_t num_qubits) : num_qubits_(num_qubits) {
    if (num_qubits_ == 0U) {
        throw std::invalid_argument("num_qubits must be at least 1");
    }
}

std::size_t Program::num_qubits() const noexcept {
    return num_qubits_;
}

const std::vector<Operation>& Program::operations() const noexcept {
    return operations_;
}

std::string Program::canonical_text() const {
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << "qupy-ir " << kIrVersion << '\n';
    output << "qubits " << num_qubits_ << '\n';

    for (const Operation& operation : operations_) {
        output << "op " << operation_name(operation.code) << " q";
        for (std::size_t index = 0; index < operation.qubits.size(); ++index) {
            if (index != 0U) {
                output << ',';
            }
            output << operation.qubits[index];
        }
        output << " p";
        if (operation.parameters.empty()) {
            output << '-';
        } else {
            for (std::size_t index = 0; index < operation.parameters.size(); ++index) {
                if (index != 0U) {
                    output << ',';
                }
                const auto bits = std::bit_cast<std::uint64_t>(operation.parameters[index]);
                output << std::hex << std::setfill('0') << std::setw(16) << bits << std::dec;
            }
        }
        output << '\n';
    }
    return output.str();
}

std::string Program::fingerprint() const {
    return fingerprint_text(canonical_text());
}

Program Program::appended(Operation operation) const {
    const OperationSpec spec = operation_spec(operation.code);
    if (operation.qubits.size() != spec.qubits) {
        throw std::invalid_argument("operation has an invalid qubit count");
    }
    if (operation.parameters.size() != spec.parameters) {
        throw std::invalid_argument("operation has an invalid parameter count");
    }

    for (const std::size_t qubit : operation.qubits) {
        if (qubit >= num_qubits_) {
            throw std::invalid_argument("qubit is outside this program");
        }
    }
    if (operation.qubits.size() == 2U && operation.qubits[0] == operation.qubits[1]) {
        throw std::invalid_argument("an operation cannot use the same qubit twice");
    }
    for (const double parameter : operation.parameters) {
        if (!std::isfinite(parameter)) {
            throw std::invalid_argument("operation parameters must be finite");
        }
    }

    Program next = *this;
    next.operations_.push_back(std::move(operation));
    return next;
}

bool Target::supports(OperationCode code) const {
    return std::find(operations.begin(), operations.end(), code) != operations.end();
}

bool Target::supports(ResultMode mode) const {
    return std::find(result_modes.begin(), result_modes.end(), mode) != result_modes.end();
}

std::string Target::fingerprint() const {
    std::vector<unsigned int> operation_codes;
    operation_codes.reserve(operations.size());
    for (const OperationCode code : operations) {
        operation_codes.push_back(static_cast<unsigned int>(code));
    }
    std::sort(operation_codes.begin(), operation_codes.end());

    std::vector<unsigned int> modes;
    modes.reserve(result_modes.size());
    for (const ResultMode mode : result_modes) {
        modes.push_back(static_cast<unsigned int>(mode));
    }
    std::sort(modes.begin(), modes.end());

    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << "qupy-target 1\nname " << name << '\n' << "ops";
    for (const unsigned int code : operation_codes) {
        output << ' ' << code;
    }
    output << "\nresults";
    for (const unsigned int mode : modes) {
        output << ' ' << mode;
    }
    output << "\nmax " << (max_qubits.has_value() ? std::to_string(*max_qubits) : "none");
    output << "\nflags " << simulator << state_access << mid_circuit_measurement
           << reset << dynamic_control << parameter_batches << '\n';
    return fingerprint_text(output.str());
}

void Target::validate(const Program& program, ResultMode mode) const {
    if (max_qubits.has_value() && program.num_qubits() > *max_qubits) {
        throw std::invalid_argument("target qubit limit exceeded");
    }
    if (!supports(mode)) {
        throw std::invalid_argument("target does not support the requested result mode");
    }
    for (const Operation& operation : program.operations()) {
        if (!supports(operation.code)) {
            throw std::invalid_argument("target does not support operation " + operation.name());
        }
    }
}

Target native_target() {
    return {
        "native-cpu",
        {
            OperationCode::H,
            OperationCode::X,
            OperationCode::Y,
            OperationCode::Z,
            OperationCode::RX,
            OperationCode::RY,
            OperationCode::RZ,
            OperationCode::CX,
            OperationCode::CZ,
            OperationCode::SWAP,
        },
        {
            ResultMode::Sample,
            ResultMode::Expectation,
            ResultMode::Probabilities,
            ResultMode::Variance,
            ResultMode::StateVector,
        },
        std::nullopt,
        true,
        true,
        false,
        false,
        false,
        false,
    };
}

ExecutionPlan plan(
    const Program& program,
    ResultMode result_mode,
    const std::string& backend
) {
    if (result_mode == ResultMode::Expectation || result_mode == ResultMode::Variance) {
        throw std::invalid_argument(
            "observable result mode requires expectation_plan or variance_plan"
        );
    }
    return prepare_program(program, result_mode, backend).execution_plan;
}

ExecutionPlan expectation_plan(
    const Program& program,
    PauliZ observable,
    const std::string& backend
) {
    return prepare_expectation(
        program, observable, ResultMode::Expectation, backend
    ).execution_plan;
}

ExecutionPlan variance_plan(
    const Program& program,
    PauliZ observable,
    const std::string& backend
) {
    return prepare_expectation(
        program, observable, ResultMode::Variance, backend
    ).execution_plan;
}

Program h(const Program& program, std::size_t qubit) {
    return program.appended({OperationCode::H, {qubit}, {}});
}

Program x(const Program& program, std::size_t qubit) {
    return program.appended({OperationCode::X, {qubit}, {}});
}

Program y(const Program& program, std::size_t qubit) {
    return program.appended({OperationCode::Y, {qubit}, {}});
}

Program z(const Program& program, std::size_t qubit) {
    return program.appended({OperationCode::Z, {qubit}, {}});
}

Program rx(const Program& program, double angle, std::size_t qubit) {
    return program.appended({OperationCode::RX, {qubit}, {angle}});
}

Program ry(const Program& program, double angle, std::size_t qubit) {
    return program.appended({OperationCode::RY, {qubit}, {angle}});
}

Program rz(const Program& program, double angle, std::size_t qubit) {
    return program.appended({OperationCode::RZ, {qubit}, {angle}});
}

Program cx(const Program& program, std::size_t control, std::size_t target) {
    return program.appended({OperationCode::CX, {control, target}, {}});
}

Program cz(const Program& program, std::size_t control, std::size_t target) {
    return program.appended({OperationCode::CZ, {control, target}, {}});
}

Program swap(const Program& program, std::size_t first, std::size_t second) {
    return program.appended({OperationCode::SWAP, {first, second}, {}});
}

PauliZ pauli_z(std::size_t qubit) {
    return {qubit};
}

StateVector statevector(const Program& program, const std::string& backend) {
    PreparedProgram prepared = prepare_program(program, ResultMode::StateVector, backend);
    return run_statevector(program.num_qubits(), prepared.steps, prepared.execution_plan);
}

Probabilities probabilities(const Program& program, const std::string& backend) {
    PreparedProgram prepared = prepare_program(program, ResultMode::Probabilities, backend);
    StateVector state = run_statevector(
        program.num_qubits(), prepared.steps, prepared.execution_plan
    );
    return {state_probabilities(state.values), prepared.execution_plan.backend};
}

Samples sample(
    const Program& program,
    std::size_t shots,
    std::optional<std::uint64_t> seed,
    const std::string& backend
) {
    if (shots == 0U) {
        throw std::invalid_argument("shots must be at least 1");
    }
    PreparedProgram prepared = prepare_program(program, ResultMode::Sample, backend);
    StateVector state = run_statevector(program.num_qubits(), prepared.steps, prepared.execution_plan);
    const std::vector<double> probabilities = state_probabilities(state.values);

    std::mt19937_64 generator;
    if (seed.has_value()) {
        generator.seed(*seed);
    } else {
        std::random_device source;
        generator.seed((static_cast<std::uint64_t>(source()) << 32U) ^ source());
    }
    const AliasSampler distribution(probabilities);

    std::vector<std::int8_t> values(shots * program.num_qubits());
    for (std::size_t shot = 0; shot < shots; ++shot) {
        const std::size_t basis = distribution.draw(generator);
        for (std::size_t column = 0; column < program.num_qubits(); ++column) {
            const std::size_t qubit = program.num_qubits() - column - 1U;
            values[shot * program.num_qubits() + column] =
                static_cast<std::int8_t>((basis >> qubit) & 1U);
        }
    }

    return {
        std::move(values),
        shots,
        program.num_qubits(),
        prepared.execution_plan.backend,
    };
}

Expectation expectation(
    const Program& program,
    PauliZ observable,
    const std::string& backend
) {
    PreparedExpectation prepared = prepare_expectation(
        program, observable, ResultMode::Expectation, backend
    );
    StateVector state = run_statevector(
        prepared.program.num_qubits(), prepared.steps, prepared.execution_plan
    );
    const double value = pauli_z_value(state.values, prepared.observable_qubit);

    return {
        value,
        prepared.execution_plan.backend,
        prepared.execution_plan.active_qubits,
        prepared.execution_plan.compiled_steps,
        prepared.execution_plan.estimated_state_bytes,
    };
}

Variance variance(
    const Program& program,
    PauliZ observable,
    const std::string& backend
) {
    PreparedExpectation prepared = prepare_expectation(
        program, observable, ResultMode::Variance, backend
    );
    StateVector state = run_statevector(
        prepared.program.num_qubits(), prepared.steps, prepared.execution_plan
    );
    const double expectation_value = pauli_z_value(state.values, prepared.observable_qubit);
    const double variance_value = std::max(0.0, 1.0 - expectation_value * expectation_value);
    return {
        variance_value,
        prepared.execution_plan.backend,
        prepared.execution_plan.active_qubits,
        prepared.execution_plan.compiled_steps,
        prepared.execution_plan.estimated_state_bytes,
    };
}

std::map<std::string, std::size_t> Samples::counts() const {
    std::map<std::string, std::size_t> result;
    for (std::size_t shot = 0; shot < shots; ++shot) {
        std::string key;
        key.reserve(num_qubits);
        const std::size_t row = shot * num_qubits;
        for (std::size_t column = 0; column < num_qubits; ++column) {
            key.push_back(values[row + column] == 0 ? '0' : '1');
        }
        ++result[key];
    }
    return result;
}

const char* core_language() noexcept {
    return "C++20";
}

const char* core_version() noexcept {
    return kCoreVersion.data();
}

std::uint32_t ir_version() noexcept {
    return kIrVersion;
}

std::size_t parallel_threads() noexcept {
#ifdef QUPY_HAS_OPENMP
    return static_cast<std::size_t>(omp_get_max_threads());
#else
    return 1U;
#endif
}

}  // namespace qupy
