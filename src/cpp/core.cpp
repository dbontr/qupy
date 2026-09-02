#include "qupy/core.hpp"

#include "cuda_driver.hpp"
#include "stabilizer.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <locale>
#include <random>
#include <set>
#include <thread>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

#ifdef QUPY_HAS_OPENMP
#include <omp.h>
#endif
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
#include <cpuid.h>
#endif
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

namespace qupy {
namespace {

#ifdef QUPY_HAS_OPENMP
constexpr std::size_t kParallelMinimumState = 1U << 16U;
constexpr std::size_t kAmplitudesPerThread = 1U << 13U;
constexpr int kMaximumOpenMpTeam = 16;

[[nodiscard]] int parallel_team_size(std::size_t state_size) noexcept {
    if (state_size < kParallelMinimumState) {
        return 1;
    }
    const int runtime_max_threads = omp_get_max_threads();
    omp_set_num_threads(runtime_max_threads);
    const int max_threads = std::min(runtime_max_threads, kMaximumOpenMpTeam);
    if (max_threads <= 1) {
        return 1;
    }
    const std::size_t useful_threads = std::max<std::size_t>(
        1U, state_size / kAmplitudesPerThread
    );
    return static_cast<int>(std::min<std::size_t>(
        static_cast<std::size_t>(max_threads), useful_threads
    ));
}
#endif

[[nodiscard]] std::size_t planned_threads(std::size_t state_bytes) noexcept {
#ifdef QUPY_HAS_OPENMP
    if (state_bytes == 0U) {
        return 1U;
    }
    return static_cast<std::size_t>(parallel_team_size(state_bytes / sizeof(Complex)));
#else
    static_cast<void>(state_bytes);
    return 1U;
#endif
}

constexpr std::uint32_t kIrVersion = 1U;
constexpr std::uint32_t kWorkloadVersion = 1U;
constexpr std::size_t kStabilizerSamplingMinQubits = 24U;
constexpr std::uint32_t kPlannerCostSchemaVersion = 1U;
constexpr double kPlannerPromotionMaxHoldoutMedianFactor = 1.5;
constexpr double kPlannerPromotionMaxHoldoutFactor = 2.0;
constexpr std::string_view kCoreVersion = "0.3.0a0";

[[nodiscard]] std::string cpu_identity() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int registers[4]{};
    __cpuid(registers, static_cast<int>(0x80000000U));
    const unsigned int max_leaf = static_cast<unsigned int>(registers[0]);
    if (max_leaf >= 0x80000004U) {
        std::array<char, 49> brand{};
        for (unsigned int index = 0U; index < 3U; ++index) {
            __cpuid(registers, static_cast<int>(0x80000002U + index));
            std::memcpy(brand.data() + index * 16U, registers, 16U);
        }
        return std::string(brand.data());
    }
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    const unsigned int max_leaf = __get_cpuid_max(0x80000000U, nullptr);
    if (max_leaf >= 0x80000004U) {
        std::array<char, 49> brand{};
        for (unsigned int index = 0U; index < 3U; ++index) {
            unsigned int eax = 0U;
            unsigned int ebx = 0U;
            unsigned int ecx = 0U;
            unsigned int edx = 0U;
            __cpuid(0x80000002U + index, eax, ebx, ecx, edx);
            const std::array<unsigned int, 4> values{eax, ebx, ecx, edx};
            std::memcpy(brand.data() + index * 16U, values.data(), 16U);
        }
        return std::string(brand.data());
    }
#endif
#ifdef __APPLE__
    for (const char* key : {"machdep.cpu.brand_string", "hw.model"}) {
        std::size_t size = 0U;
        if (sysctlbyname(key, nullptr, &size, nullptr, 0) == 0 && size > 1U) {
            std::vector<char> buffer(size);
            if (sysctlbyname(key, buffer.data(), &size, nullptr, 0) == 0) {
                return std::string(buffer.data());
            }
        }
    }
#elif defined(_WIN32)
    char* identifier = nullptr;
    std::size_t length = 0U;
    if (_dupenv_s(&identifier, &length, "PROCESSOR_IDENTIFIER") == 0 && identifier != nullptr) {
        const std::string result(identifier);
        std::free(identifier);
        if (!result.empty()) {
            return result;
        }
    }
#elif defined(__linux__)
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (line.rfind("model name", 0U) == 0U || line.rfind("Hardware", 0U) == 0U) {
            const std::size_t separator = line.find(':');
            if (separator != std::string::npos) {
                std::string identity = line.substr(separator + 1U);
                identity.erase(
                    identity.begin(),
                    std::find_if(identity.begin(), identity.end(), [](unsigned char value) {
                        return !std::isspace(value);
                    })
                );
                if (!identity.empty()) {
                    return identity;
                }
            }
        }
    }
#endif
    return "unknown";
}

[[nodiscard]] std::string planner_host_text() {
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << "qupy-planner-host 1\n";
#if defined(_WIN32)
    output << "os windows\n";
#elif defined(__APPLE__)
    output << "os macos\n";
#elif defined(__linux__)
    output << "os linux\n";
#else
    output << "os other\n";
#endif
#if defined(__x86_64__) || defined(_M_X64)
    output << "arch x86_64\n";
#elif defined(__aarch64__) || defined(_M_ARM64)
    output << "arch aarch64\n";
#else
    output << "arch other\n";
#endif
    output << "pointer-bits " << sizeof(void*) * 8U << '\n';
    output << "cpu " << cpu_identity() << '\n';
#if defined(__clang__)
    output << "compiler clang-" << __clang_major__ << '.' << __clang_minor__ << '\n';
#elif defined(_MSC_VER)
    output << "compiler msvc-" << _MSC_VER << '\n';
#elif defined(__GNUC__)
    output << "compiler gcc-" << __GNUC__ << '.' << __GNUC_MINOR__ << '\n';
#else
    output << "compiler other\n";
#endif
    output << "logical-processors " << std::thread::hardware_concurrency() << '\n';
#ifdef QUPY_HAS_OPENMP
    output << "openmp-max " << omp_get_max_threads() << '\n';
#else
    output << "openmp-max 1\n";
#endif
    output << "kernel-team-ceiling 16\n";
    return output.str();
}

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

struct WorkloadFeatures {
    std::size_t active_operations;
    std::size_t single_qubit_operations;
    std::size_t two_qubit_operations;
    std::size_t parameterized_operations;
    std::size_t non_clifford_operations;
    std::string fingerprint;
};

[[nodiscard]] bool is_non_clifford(OperationCode code) noexcept {
    return code == OperationCode::RX || code == OperationCode::RY || code == OperationCode::RZ;
}

[[nodiscard]] WorkloadFeatures workload_features(
    const Program& source_program,
    ResultMode result_mode,
    std::size_t active_qubits,
    const std::vector<Operation>& active_operations
) {
    std::size_t single_qubit_operations = 0U;
    std::size_t two_qubit_operations = 0U;
    std::size_t parameterized_operations = 0U;
    std::size_t non_clifford_operations = 0U;
    for (const Operation& operation : active_operations) {
        if (is_single_qubit(operation.code)) {
            ++single_qubit_operations;
        } else {
            ++two_qubit_operations;
        }
        if (!operation.parameters.empty()) {
            ++parameterized_operations;
        }
        if (is_non_clifford(operation.code)) {
            ++non_clifford_operations;
        }
    }

    constexpr std::array<OperationCode, 10> operation_codes = {
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
    };
    std::array<std::size_t, operation_codes.size()> source_counts{};
    for (const Operation& operation : source_program.operations()) {
        const auto raw_code = static_cast<std::size_t>(operation.code);
        if (raw_code >= source_counts.size()) {
            throw std::logic_error("operation code is outside workload fingerprint schema");
        }
        ++source_counts[raw_code];
    }

    std::ostringstream canonical;
    canonical.imbue(std::locale::classic());
    canonical << "qupy-workload " << kWorkloadVersion << '\n';
    canonical << "result " << result_mode_name(result_mode) << '\n';
    canonical << "original-qubits " << source_program.num_qubits() << '\n';
    canonical << "original-operations " << source_program.operations().size() << '\n';
    canonical << "active-qubits " << active_qubits << '\n';
    canonical << "active-operations " << active_operations.size() << '\n';
    for (const OperationCode code : operation_codes) {
        canonical << "source-count " << operation_name(code) << ' '
                  << source_counts[static_cast<std::size_t>(code)] << '\n';
    }
    for (const Operation& operation : active_operations) {
        canonical << "active-op " << operation_name(operation.code) << " q";
        for (std::size_t index = 0; index < operation.qubits.size(); ++index) {
            if (index != 0U) {
                canonical << ',';
            }
            canonical << operation.qubits[index];
        }
        canonical << " p" << operation.parameters.size() << '\n';
    }

    return {
        active_operations.size(),
        single_qubit_operations,
        two_qubit_operations,
        parameterized_operations,
        non_clifford_operations,
        fingerprint_text(canonical.str()),
    };
}

[[nodiscard]] Matrix2 single_matrix(
    const Operation& operation,
    std::optional<double> parameter_override = std::nullopt
) {
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
        const double half = parameter_override.value_or(operation.parameters.front()) / 2.0;
        const double cosine = std::cos(half);
        const Complex sine{0.0, -std::sin(half)};
        return {cosine, sine, sine, cosine};
    }
    case OperationCode::RY: {
        const double half = parameter_override.value_or(operation.parameters.front()) / 2.0;
        const double cosine = std::cos(half);
        const double sine = std::sin(half);
        return {cosine, -sine, sine, cosine};
    }
    case OperationCode::RZ: {
        const double half = parameter_override.value_or(operation.parameters.front()) / 2.0;
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

template <typename ParameterResolver>
[[nodiscard]] std::vector<CompiledStep> compile_operations_with(
    std::size_t num_qubits,
    const std::vector<Operation>& operations,
    ParameterResolver&& parameter_resolver
) {
    const Matrix2 identity = identity_matrix();
    std::vector<Matrix2> pending(num_qubits, identity);
    std::vector<bool> has_pending(num_qubits, false);
    std::vector<CompiledStep> steps;
    steps.reserve(operations.size());

    const auto flush = [&](std::size_t qubit) {
        if (!has_pending[qubit]) {
            return;
        }
        steps.push_back({CompiledKind::Single, pending[qubit], qubit, 0U});
        pending[qubit] = identity;
        has_pending[qubit] = false;
    };

    for (std::size_t index = 0; index < operations.size(); ++index) {
        const Operation& operation = operations[index];
        if (is_single_qubit(operation.code)) {
            const std::size_t qubit = operation.qubits.front();
            pending[qubit] = multiply(
                single_matrix(operation, parameter_resolver(index, operation)),
                pending[qubit]
            );
            has_pending[qubit] = true;
            continue;
        }

        const std::size_t first = operation.qubits[0];
        const std::size_t second = operation.qubits[1];
        flush(first);
        flush(second);
        steps.push_back({compiled_kind(operation.code), identity, first, second});
    }

    for (std::size_t qubit = 0; qubit < num_qubits; ++qubit) {
        flush(qubit);
    }
    return steps;
}

[[nodiscard]] std::vector<CompiledStep> compile_operations(
    std::size_t num_qubits,
    const std::vector<Operation>& operations
) {
    return compile_operations_with(
        num_qubits,
        operations,
        [](std::size_t, const Operation&) { return std::optional<double>{}; }
    );
}

[[nodiscard]] std::vector<CompiledStep> compile_program(const Program& program) {
    return compile_operations(program.num_qubits(), program.operations());
}

struct PreparedProgram {
    ExecutionPlan execution_plan;
    std::vector<CompiledStep> steps;
};

struct PreparedExpectation {
    std::size_t observable_qubit;
    ExecutionPlan execution_plan;
    std::vector<CompiledStep> steps;
    std::vector<Operation> pauli_operations;
    bool pauli_propagation;
};

struct ReducedExpectation {
    std::size_t active_qubits;
    std::size_t observable_qubit;
    std::vector<Operation> operations;
    std::vector<std::size_t> source_operation_indices;
};

struct ParameterLayout {
    std::vector<std::size_t> column_by_operation;
    std::size_t parameter_count;
};

constexpr std::size_t kMissingParameterColumn = std::numeric_limits<std::size_t>::max();

struct BatchMatrixFactor {
    Matrix2 fixed_matrix;
    Operation operation;
    std::size_t parameter_column;
    bool parameterized;
};

struct BatchCompiledStep {
    CompiledKind kind;
    std::size_t first;
    std::size_t second;
    std::vector<BatchMatrixFactor> factors;
};

[[nodiscard]] ParameterLayout parameter_layout(
    const Program& program,
    const std::vector<ParameterSlot>& slots
) {
    std::vector<std::size_t> columns(program.operations().size(), kMissingParameterColumn);
    for (std::size_t column = 0; column < slots.size(); ++column) {
        const ParameterSlot slot = slots[column];
        if (slot.operation_index >= program.operations().size()) {
            throw std::invalid_argument("parameter slot operation is outside this program");
        }
        const Operation& operation = program.operations()[slot.operation_index];
        if (slot.parameter_index >= operation.parameters.size()) {
            throw std::invalid_argument("parameter slot does not reference an operation parameter");
        }
        if (operation.parameters.size() != 1U || slot.parameter_index != 0U) {
            throw std::invalid_argument(
                "native parameter batches currently support one parameter per operation"
            );
        }
        if (columns[slot.operation_index] != kMissingParameterColumn) {
            throw std::invalid_argument("parameter slots must reference distinct operations");
        }
        columns[slot.operation_index] = column;
    }
    return {std::move(columns), slots.size()};
}

void validate_parameter_batch_values(
    const ParameterLayout& layout,
    const std::vector<double>& values,
    std::size_t batch_size
) {
    if (batch_size == 0U) {
        throw std::invalid_argument("parameter batch must contain at least one row");
    }
    if (layout.parameter_count != 0U &&
        batch_size > std::numeric_limits<std::size_t>::max() / layout.parameter_count) {
        throw std::length_error("parameter batch shape exceeds native address space");
    }
    const std::size_t expected = batch_size * layout.parameter_count;
    if (values.size() != expected) {
        throw std::invalid_argument("parameter batch shape does not match its slots");
    }
    if (!std::all_of(values.begin(), values.end(), [](double value) { return std::isfinite(value); })) {
        throw std::invalid_argument("parameter batch values must be finite");
    }
}

[[nodiscard]] bool has_relevant_parameter_slot(
    const ReducedExpectation& reduced,
    const ParameterLayout& layout
) {
    return std::any_of(
        reduced.source_operation_indices.begin(),
        reduced.source_operation_indices.end(),
        [&](std::size_t source_index) {
            return layout.column_by_operation[source_index] != kMissingParameterColumn;
        }
    );
}

[[nodiscard]] std::vector<BatchCompiledStep> compile_parameterized_operations(
    std::size_t num_qubits,
    const std::vector<Operation>& operations,
    const std::vector<std::size_t>& source_operation_indices,
    const ParameterLayout& layout
) {
    if (source_operation_indices.size() != operations.size()) {
        throw std::logic_error("parameterized compiler source index mismatch");
    }

    const Matrix2 identity = identity_matrix();
    std::vector<std::vector<BatchMatrixFactor>> pending(num_qubits);
    std::vector<BatchCompiledStep> steps;
    steps.reserve(operations.size());

    const auto flush = [&](std::size_t qubit) {
        if (pending[qubit].empty()) {
            return;
        }
        std::vector<BatchMatrixFactor> factors;
        factors.swap(pending[qubit]);
        steps.push_back({CompiledKind::Single, qubit, 0U, std::move(factors)});
    };

    for (std::size_t index = 0; index < operations.size(); ++index) {
        const Operation& operation = operations[index];
        if (is_single_qubit(operation.code)) {
            const std::size_t qubit = operation.qubits.front();
            const std::size_t source_index = source_operation_indices[index];
            if (source_index >= layout.column_by_operation.size()) {
                throw std::logic_error("parameterized compiler source operation is invalid");
            }
            const std::size_t column = layout.column_by_operation[source_index];
            if (column == kMissingParameterColumn) {
                const Matrix2 matrix = single_matrix(operation);
                if (!pending[qubit].empty() && !pending[qubit].back().parameterized) {
                    pending[qubit].back().fixed_matrix = multiply(
                        matrix, pending[qubit].back().fixed_matrix
                    );
                } else {
                    pending[qubit].push_back({matrix, operation, column, false});
                }
            } else {
                pending[qubit].push_back({identity, operation, column, true});
            }
            continue;
        }

        const std::size_t first = operation.qubits[0];
        const std::size_t second = operation.qubits[1];
        flush(first);
        flush(second);
        steps.push_back({compiled_kind(operation.code), first, second, {}});
    }

    for (std::size_t qubit = 0; qubit < num_qubits; ++qubit) {
        flush(qubit);
    }
    return steps;
}

void materialize_parameterized_steps(
    const std::vector<BatchCompiledStep>& batch_steps,
    const double* row,
    std::vector<CompiledStep>& steps
) {
    const Matrix2 identity = identity_matrix();
    steps.clear();
    if (steps.capacity() < batch_steps.size()) {
        steps.reserve(batch_steps.size());
    }

    for (const BatchCompiledStep& step : batch_steps) {
        if (step.kind != CompiledKind::Single) {
            steps.push_back({step.kind, identity, step.first, step.second});
            continue;
        }

        Matrix2 matrix = identity;
        for (const BatchMatrixFactor& factor : step.factors) {
            const Matrix2 current = factor.parameterized
                ? single_matrix(factor.operation, row[factor.parameter_column])
                : factor.fixed_matrix;
            matrix = multiply(current, matrix);
        }
        steps.push_back({CompiledKind::Single, matrix, step.first, 0U});
    }
}

[[nodiscard]] std::vector<std::size_t> operation_indices(const Program& program) {
    std::vector<std::size_t> indices(program.operations().size());
    for (std::size_t index = 0; index < indices.size(); ++index) {
        indices[index] = index;
    }
    return indices;
}
[[nodiscard]] bool is_cuda_backend(const std::string& backend) {
    return backend == "cuda" || backend == "native-cuda";
}

void validate_backend(const std::string& backend) {
    if (backend != "auto" && backend != "native" && backend != "native-cpu" &&
        !is_cuda_backend(backend)) {
        throw std::invalid_argument("unknown backend: " + backend);
    }
}

[[nodiscard]] bool cost_model_supports_method(std::string_view method) {
    return method == "pauli-propagation" || method == "statevector" ||
           method == "statevector-lightcone";
}

[[nodiscard]] std::string plan_cost_class(const ExecutionPlan& plan) {
    if (plan.method == "pauli-propagation") {
        return "pauli-propagation";
    }
    if (plan.method == "statevector" || plan.method == "statevector-lightcone") {
        return plan.threads == 1U ? "statevector-serial" : "statevector-parallel";
    }
    throw std::invalid_argument("planner cost model does not support method " + plan.method);
}

[[nodiscard]] std::vector<double> plan_cost_features(const ExecutionPlan& plan) {
    const std::string cost_class = plan_cost_class(plan);
    if (cost_class == "pauli-propagation") {
        return {1.0, std::log(static_cast<double>(std::max<std::size_t>(plan.active_operations, 1U)))};
    }
    const double log_work =
        std::log(static_cast<double>(std::max<std::size_t>(plan.compiled_steps, 1U))) +
        static_cast<double>(plan.active_qubits) * std::log(2.0) -
        std::log(static_cast<double>(std::max<std::size_t>(plan.threads, 1U)));
    return {1.0, log_work, log_work * log_work};
}

[[nodiscard]] ExecutionPlan make_plan(
    const Program& source_program,
    const Target& target,
    ResultMode result_mode,
    std::size_t original_operations,
    const std::vector<Operation>& workload_operations,
    std::size_t compiled_steps,
    std::size_t active_qubits,
    std::size_t estimated_state_bytes,
    const std::string& method,
    std::string_view qualifier,
    bool collect_workload_fingerprint,
    const PlannerCostModel* cost_model
) {
    const bool use_cost_model = cost_model != nullptr && cost_model_supports_method(method);
    const WorkloadFeatures features = (collect_workload_fingerprint || use_cost_model)
        ? workload_features(source_program, result_mode, active_qubits, workload_operations)
        : WorkloadFeatures{workload_operations.size(), 0U, 0U, 0U, 0U, {}};
    const std::string program_fingerprint = source_program.fingerprint();
    const std::string target_fingerprint = target.fingerprint();
    std::ostringstream cache_key;
    cache_key.imbue(std::locale::classic());
    cache_key << "qupy-cache/1/" << kCoreVersion << '/' << program_fingerprint << '/'
              << target_fingerprint << '/' << result_mode_name(result_mode) << '/' << method;
    if (!qualifier.empty()) {
        cache_key << '/' << qualifier;
    }

    ExecutionPlan execution_plan{
        target.name, method, true,
        (method == "stabilizer" || method == "cuda-statevector")
            ? 1U : planned_threads(estimated_state_bytes),
        source_program.num_qubits(), original_operations, active_qubits,
        features.active_operations, features.single_qubit_operations,
        features.two_qubit_operations, features.parameterized_operations,
        features.non_clifford_operations, compiled_steps, estimated_state_bytes,
        result_mode, kWorkloadVersion, features.fingerprint, program_fingerprint,
        target_fingerprint, cache_key.str(), std::nullopt, {}, {}, {},
    };
    if (use_cost_model) {
        execution_plan.predicted_ns = cost_model->predict_ns(execution_plan);
        execution_plan.cost_model_class = plan_cost_class(execution_plan);
        execution_plan.cost_model_fingerprint = cost_model->artifact_fingerprint();
        execution_plan.cost_model_host_fingerprint = cost_model->host_fingerprint();
    }
    return execution_plan;
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
    const int threads = parallel_team_size(state.size());
#pragma omp parallel for schedule(static) if(threads > 1) num_threads(threads)
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
    const int threads = parallel_team_size(state.size());
#pragma omp parallel for schedule(static) if(threads > 1) num_threads(threads)
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
    const int threads = parallel_team_size(state.size());
#pragma omp parallel for schedule(static) if(threads > 1) num_threads(threads)
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
    const int threads = parallel_team_size(state.size());
#pragma omp parallel for schedule(static) if(threads > 1) num_threads(threads)
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

struct PauliFrame {
    std::vector<std::uint8_t> x;
    std::vector<std::uint8_t> z;
    bool negative = false;
};

void conjugate_h(PauliFrame& pauli, std::size_t qubit) {
    if (pauli.x[qubit] != 0U && pauli.z[qubit] != 0U) {
        pauli.negative = !pauli.negative;
    }
    std::swap(pauli.x[qubit], pauli.z[qubit]);
}

void conjugate_x(PauliFrame& pauli, std::size_t qubit) {
    if (pauli.z[qubit] != 0U) {
        pauli.negative = !pauli.negative;
    }
}

void conjugate_y(PauliFrame& pauli, std::size_t qubit) {
    if ((pauli.x[qubit] != 0U) != (pauli.z[qubit] != 0U)) {
        pauli.negative = !pauli.negative;
    }
}

void conjugate_z(PauliFrame& pauli, std::size_t qubit) {
    if (pauli.x[qubit] != 0U) {
        pauli.negative = !pauli.negative;
    }
}

void conjugate_cx(PauliFrame& pauli, std::size_t control, std::size_t target) {
    const bool phase_flip = pauli.x[control] != 0U &&
                            pauli.z[target] != 0U &&
                            ((pauli.x[target] != 0U) == (pauli.z[control] != 0U));
    if (phase_flip) {
        pauli.negative = !pauli.negative;
    }
    pauli.x[target] ^= pauli.x[control];
    pauli.z[control] ^= pauli.z[target];
}

void conjugate_cz(PauliFrame& pauli, std::size_t control, std::size_t target) {
    conjugate_h(pauli, target);
    conjugate_cx(pauli, control, target);
    conjugate_h(pauli, target);
}

void conjugate_swap(PauliFrame& pauli, std::size_t first, std::size_t second) {
    std::swap(pauli.x[first], pauli.x[second]);
    std::swap(pauli.z[first], pauli.z[second]);
}

[[nodiscard]] bool supports_pauli_propagation(const std::vector<Operation>& operations) {
    return std::all_of(
        operations.begin(),
        operations.end(),
        [](const Operation& operation) {
            switch (operation.code) {
            case OperationCode::H:
            case OperationCode::X:
            case OperationCode::Y:
            case OperationCode::Z:
            case OperationCode::CX:
            case OperationCode::CZ:
            case OperationCode::SWAP:
                return true;
            case OperationCode::RX:
            case OperationCode::RY:
            case OperationCode::RZ:
                return false;
            }
            return false;
        }
    );
}

[[nodiscard]] double pauli_z_propagated_value(
    std::size_t num_qubits,
    const std::vector<Operation>& operations,
    std::size_t observable_qubit
) {
    PauliFrame pauli{
        std::vector<std::uint8_t>(num_qubits, 0U),
        std::vector<std::uint8_t>(num_qubits, 0U),
        false,
    };
    pauli.z[observable_qubit] = 1U;

    for (std::size_t index = operations.size(); index > 0U; --index) {
        const Operation& operation = operations[index - 1U];
        switch (operation.code) {
        case OperationCode::H:
            conjugate_h(pauli, operation.qubits[0]);
            break;
        case OperationCode::X:
            conjugate_x(pauli, operation.qubits[0]);
            break;
        case OperationCode::Y:
            conjugate_y(pauli, operation.qubits[0]);
            break;
        case OperationCode::Z:
            conjugate_z(pauli, operation.qubits[0]);
            break;
        case OperationCode::CX:
            conjugate_cx(pauli, operation.qubits[0], operation.qubits[1]);
            break;
        case OperationCode::CZ:
            conjugate_cz(pauli, operation.qubits[0], operation.qubits[1]);
            break;
        case OperationCode::SWAP:
            conjugate_swap(pauli, operation.qubits[0], operation.qubits[1]);
            break;
        case OperationCode::RX:
        case OperationCode::RY:
        case OperationCode::RZ:
            throw std::logic_error("non-Clifford operation reached Pauli propagation");
        }
    }

    if (std::any_of(
            pauli.x.begin(),
            pauli.x.end(),
            [](std::uint8_t bit) { return bit != 0U; }
        )) {
        return 0.0;
    }
    return pauli.negative ? -1.0 : 1.0;
}

[[nodiscard]] PreparedProgram prepare_program(
    const Program& program,
    ResultMode result_mode,
    const std::string& backend,
    bool collect_workload_fingerprint = false,
    const PlannerCostModel* cost_model = nullptr
) {
    validate_backend(backend);
    const bool cuda_backend = is_cuda_backend(backend);
    const Target target = cuda_backend ? cuda_target() : native_target();
    target.validate(program, result_mode);
    if (cuda_backend) {
        std::vector<CompiledStep> steps = compile_program(program);
        ExecutionPlan execution_plan = make_plan(
            program, target, result_mode, program.operations().size(), program.operations(),
            steps.size(), program.num_qubits(), state_memory_bytes(program.num_qubits()),
            "cuda-statevector", {}, collect_workload_fingerprint, cost_model
        );
        return {std::move(execution_plan), std::move(steps)};
    }
    if (
        result_mode == ResultMode::Sample &&
        program.num_qubits() >= kStabilizerSamplingMinQubits &&
        detail::supports_stabilizer(program)
    ) {
        ExecutionPlan execution_plan = make_plan(
            program,
            target,
            result_mode,
            program.operations().size(),
            program.operations(),
            program.operations().size(),
            program.num_qubits(),
            detail::stabilizer_state_bytes(program.num_qubits()),
            "stabilizer",
            {},
            collect_workload_fingerprint,
            cost_model
        );
        return {std::move(execution_plan), {}};
    }
    std::vector<CompiledStep> steps = compile_program(program);
    ExecutionPlan execution_plan = make_plan(
        program,
        target,
        result_mode,
        program.operations().size(),
        program.operations(),
        steps.size(),
        program.num_qubits(),
        state_memory_bytes(program.num_qubits()),
        "statevector",
        {},
        collect_workload_fingerprint,
        cost_model
    );
    return {std::move(execution_plan), std::move(steps)};
}

[[nodiscard]] ReducedExpectation reduce_expectation(
    const Program& program,
    PauliZ observable
) {
    if (observable.qubit >= program.num_qubits()) {
        throw std::invalid_argument("observable qubit is outside this program");
    }

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

    std::vector<Operation> reduced_operations;
    std::vector<std::size_t> source_operation_indices;
    reduced_operations.reserve(operations.size());
    source_operation_indices.reserve(operations.size());
    for (std::size_t index = 0; index < operations.size(); ++index) {
        if (!retained[index]) {
            continue;
        }
        Operation operation = operations[index];
        for (std::size_t& qubit : operation.qubits) {
            qubit = mapping[qubit];
        }
        reduced_operations.push_back(std::move(operation));
        source_operation_indices.push_back(index);
    }

    return {
        active_count,
        mapping[observable.qubit],
        std::move(reduced_operations),
        std::move(source_operation_indices),
    };
}

[[nodiscard]] PreparedExpectation prepare_expectation(
    const Program& program,
    PauliZ observable,
    ResultMode result_mode,
    const std::string& backend,
    bool collect_workload_fingerprint = false,
    const PlannerCostModel* cost_model = nullptr
) {
    if (observable.qubit >= program.num_qubits()) {
        throw std::invalid_argument("observable qubit is outside this program");
    }
    if (result_mode != ResultMode::Expectation && result_mode != ResultMode::Variance) {
        throw std::invalid_argument("observable plan requires expectation or variance mode");
    }
    validate_backend(backend);
    const Target target = is_cuda_backend(backend) ? cuda_target() : native_target();
    target.validate(program, result_mode);

    ReducedExpectation reduced = reduce_expectation(program, observable);
    const bool pauli_propagation = supports_pauli_propagation(reduced.operations);
    std::vector<CompiledStep> steps;
    std::vector<Operation> pauli_operations;
    std::string method;
    std::size_t compiled_steps = 0U;
    std::size_t estimated_state_bytes = 0U;
    if (pauli_propagation) {
        method = "pauli-propagation";
        compiled_steps = reduced.operations.size();
    } else {
        steps = compile_operations(reduced.active_qubits, reduced.operations);
        method = reduced.active_qubits < program.num_qubits()
            ? "statevector-lightcone"
            : "statevector";
        compiled_steps = steps.size();
        estimated_state_bytes = state_memory_bytes(reduced.active_qubits);
    }

    const std::string qualifier = "z:" + std::to_string(observable.qubit);
    ExecutionPlan execution_plan = make_plan(
        program,
        target,
        result_mode,
        program.operations().size(),
        reduced.operations,
        compiled_steps,
        reduced.active_qubits,
        estimated_state_bytes,
        method,
        qualifier,
        collect_workload_fingerprint,
        cost_model
    );
    if (pauli_propagation) {
        pauli_operations = std::move(reduced.operations);
    }
    return {
        reduced.observable_qubit,
        std::move(execution_plan),
        std::move(steps),
        std::move(pauli_operations),
        pauli_propagation,
    };
}

void evolve_statevector(
    std::vector<Complex>& state,
    std::size_t num_qubits,
    const std::vector<CompiledStep>& steps
) {
    const std::size_t dimension = state_dimension(num_qubits);
    if (state.size() != dimension) {
        state.assign(dimension, Complex{0.0, 0.0});
    } else {
#ifdef QUPY_HAS_OPENMP
        const int threads = parallel_team_size(state.size());
#pragma omp parallel for schedule(static) if(threads > 1) num_threads(threads)
        for (std::ptrdiff_t raw = 0; raw < static_cast<std::ptrdiff_t>(state.size()); ++raw) {
            state[static_cast<std::size_t>(raw)] = Complex{0.0, 0.0};
        }
#else
        std::fill(state.begin(), state.end(), Complex{0.0, 0.0});
#endif
    }
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
}

[[nodiscard]] std::vector<detail::CudaStep> cuda_steps(
    const std::vector<CompiledStep>& steps
) {
    std::vector<detail::CudaStep> result;
    result.reserve(steps.size());
    for (const CompiledStep& step : steps) {
        detail::CudaStepKind kind = detail::CudaStepKind::Single;
        switch (step.kind) {
        case CompiledKind::Single: kind = detail::CudaStepKind::Single; break;
        case CompiledKind::CX: kind = detail::CudaStepKind::CX; break;
        case CompiledKind::CZ: kind = detail::CudaStepKind::CZ; break;
        case CompiledKind::SWAP: kind = detail::CudaStepKind::SWAP; break;
        }
        result.push_back({kind, step.matrix, step.first, step.second});
    }
    return result;
}

[[nodiscard]] StateVector run_statevector(
    std::size_t num_qubits,
    const std::vector<CompiledStep>& steps,
    const ExecutionPlan& execution_plan
) {
    std::vector<Complex> state;
    evolve_statevector(state, num_qubits, steps);
    return {std::move(state), execution_plan.backend};
}

[[nodiscard]] const std::vector<Complex>& run_statevector_workspace(
    std::size_t num_qubits,
    const std::vector<CompiledStep>& steps
) {
    // Internal execution has no user callbacks. A per-thread buffer avoids shared mutable state.
    thread_local std::vector<Complex> state;
    evolve_statevector(state, num_qubits, steps);
    return state;
}

[[nodiscard]] std::vector<double> state_probabilities(const std::vector<Complex>& state) {
    std::vector<double> probabilities(state.size());
#ifdef QUPY_HAS_OPENMP
    const int threads = parallel_team_size(state.size());
#pragma omp parallel for schedule(static) if(threads > 1) num_threads(threads)
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
    const int threads = parallel_team_size(state.size());
#pragma omp parallel for reduction(+ : value) schedule(static) if(threads > 1) num_threads(threads)
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

[[nodiscard]] std::size_t checked_product(
    std::size_t left,
    std::size_t right,
    const char* message
) {
    if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::length_error(message);
    }
    return left * right;
}

[[nodiscard]] std::mt19937_64 sampling_generator(std::optional<std::uint64_t> seed) {
    std::mt19937_64 generator;
    if (seed.has_value()) {
        generator.seed(*seed);
    } else {
        std::random_device source;
        generator.seed((static_cast<std::uint64_t>(source()) << 32U) ^ source());
    }
    return generator;
}

void draw_samples(
    const AliasSampler& distribution,
    std::mt19937_64& generator,
    std::size_t num_qubits,
    std::size_t shots,
    std::int8_t* values
) {
    for (std::size_t shot = 0; shot < shots; ++shot) {
        const std::size_t basis = distribution.draw(generator);
        for (std::size_t column = 0; column < num_qubits; ++column) {
            const std::size_t qubit = num_qubits - column - 1U;
            values[shot * num_qubits + column] =
                static_cast<std::int8_t>((basis >> qubit) & 1U);
        }
    }
}

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

Program Program::bound(
    const std::vector<ParameterSlot>& slots,
    const std::vector<double>& values
) const {
    if (slots.size() != values.size()) {
        throw std::invalid_argument("parameter slot and value counts must match");
    }

    Program next = *this;
    for (std::size_t index = 0; index < slots.size(); ++index) {
        const ParameterSlot slot = slots[index];
        if (slot.operation_index >= next.operations_.size()) {
            throw std::invalid_argument("parameter slot operation is outside this program");
        }
        Operation& operation = next.operations_[slot.operation_index];
        if (slot.parameter_index >= operation.parameters.size()) {
            throw std::invalid_argument("parameter slot does not reference an operation parameter");
        }
        if (!std::isfinite(values[index])) {
            throw std::invalid_argument("bound parameter values must be finite");
        }
        for (std::size_t prior = 0; prior < index; ++prior) {
            if (slots[prior].operation_index == slot.operation_index &&
                slots[prior].parameter_index == slot.parameter_index) {
                throw std::invalid_argument("parameter slots must be unique");
            }
        }
        operation.parameters[slot.parameter_index] = values[index];
    }
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


std::uint32_t PlannerCostModel::schema_version() const noexcept { return schema_version_; }
std::uint32_t PlannerCostModel::workload_version() const noexcept { return workload_version_; }
const std::string& PlannerCostModel::engine_version() const noexcept { return engine_version_; }
const std::string& PlannerCostModel::host_fingerprint() const noexcept { return host_fingerprint_; }
const std::string& PlannerCostModel::artifact_fingerprint() const noexcept {
    return artifact_fingerprint_;
}

std::vector<std::string> PlannerCostModel::cost_classes() const {
    std::vector<std::string> result;
    result.reserve(curves_.size());
    for (const Curve& curve : curves_) {
        result.push_back(curve.cost_class);
    }
    std::sort(result.begin(), result.end());
    return result;
}

double PlannerCostModel::predict_ns(const ExecutionPlan& execution_plan) const {
    if (execution_plan.workload_version != workload_version_) {
        throw std::invalid_argument("execution plan workload version does not match cost model");
    }
    const std::string cost_class = plan_cost_class(execution_plan);
    const auto curve = std::find_if(
        curves_.begin(), curves_.end(),
        [&](const Curve& item) { return item.cost_class == cost_class; }
    );
    if (curve == curves_.end()) {
        throw std::invalid_argument("cost model does not contain class " + cost_class);
    }
    const std::vector<double> features = plan_cost_features(execution_plan);
    if (features.size() != curve->coefficients.size()) {
        throw std::logic_error("cost model coefficient count does not match plan features");
    }
    double log_runtime = 0.0;
    for (std::size_t index = 0; index < features.size(); ++index) {
        log_runtime += features[index] * curve->coefficients[index];
    }
    const double prediction = std::exp(log_runtime);
    if (!std::isfinite(prediction) || prediction <= 0.0) {
        throw std::overflow_error("cost model produced an invalid runtime prediction");
    }
    return prediction;
}

std::string planner_host_fingerprint() {
    return fingerprint_text(planner_host_text());
}

void strip_trailing_carriage_return(std::string& line) {
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
}

PlannerCostModel load_planner_cost_model(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::invalid_argument("cannot open planner cost artifact: " + path);
    }
    const std::string text{
        std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()
    };
    std::istringstream lines(text);
    std::string line;
    if (!std::getline(lines, line)) {
        throw std::invalid_argument("planner cost artifact has an unsupported schema");
    }
    strip_trailing_carriage_return(line);
    if (line != "qupy-planner-cost 1") {
        throw std::invalid_argument("planner cost artifact has an unsupported schema");
    }

    PlannerCostModel model;
    model.schema_version_ = kPlannerCostSchemaVersion;
    bool has_engine = false;
    bool has_workload = false;
    bool has_host = false;
    bool validated = false;
    std::set<std::string> classes;
    while (std::getline(lines, line)) {
        strip_trailing_carriage_return(line);
        if (line.empty()) {
            continue;
        }
        std::istringstream fields(line);
        std::string key;
        fields >> key;
        if (key == "engine") {
            if (has_engine || !(fields >> model.engine_version_)) {
                throw std::invalid_argument("planner cost artifact has invalid engine metadata");
            }
            has_engine = true;
        } else if (key == "workload") {
            if (has_workload || !(fields >> model.workload_version_)) {
                throw std::invalid_argument("planner cost artifact has invalid workload metadata");
            }
            has_workload = true;
        } else if (key == "host") {
            if (has_host || !(fields >> model.host_fingerprint_)) {
                throw std::invalid_argument("planner cost artifact has invalid host metadata");
            }
            has_host = true;
        } else if (key == "validated") {
            int value = 0;
            if (!(fields >> value) || value != 1) {
                throw std::invalid_argument("planner cost artifact is not validated");
            }
            validated = true;
        } else if (key == "model") {
            PlannerCostModel::Curve curve;
            std::size_t coefficient_count = 0U;
            if (!(fields >> curve.cost_class >> coefficient_count)) {
                throw std::invalid_argument("planner cost artifact has a malformed model row");
            }
            const std::size_t expected = curve.cost_class == "pauli-propagation" ? 2U :
                ((curve.cost_class == "statevector-serial" ||
                  curve.cost_class == "statevector-parallel") ? 3U : 0U);
            if (expected == 0U || coefficient_count != expected || !classes.insert(curve.cost_class).second) {
                throw std::invalid_argument("planner cost artifact has an invalid model class");
            }
            curve.coefficients.resize(coefficient_count);
            for (double& coefficient : curve.coefficients) {
                if (!(fields >> coefficient) || !std::isfinite(coefficient)) {
                    throw std::invalid_argument("planner cost artifact has an invalid coefficient");
                }
            }
            if (!(fields >> curve.holdout_median_factor >> curve.holdout_max_factor) ||
                !std::isfinite(curve.holdout_median_factor) ||
                !std::isfinite(curve.holdout_max_factor) ||
                curve.holdout_median_factor < 1.0 || curve.holdout_max_factor < 1.0 ||
                curve.holdout_median_factor > kPlannerPromotionMaxHoldoutMedianFactor ||
                curve.holdout_max_factor > kPlannerPromotionMaxHoldoutFactor) {
                throw std::invalid_argument("planner cost artifact has invalid holdout metrics");
            }
            model.curves_.push_back(std::move(curve));
        } else {
            throw std::invalid_argument("planner cost artifact contains an unknown field");
        }
        std::string extra;
        if (fields >> extra) {
            throw std::invalid_argument("planner cost artifact row contains unexpected data");
        }
    }
    const std::set<std::string> expected_classes = {
        "pauli-propagation", "statevector-parallel", "statevector-serial"
    };
    if (!has_engine || !has_workload || !has_host || !validated || classes != expected_classes) {
        throw std::invalid_argument("planner cost artifact is incomplete");
    }
    if (model.engine_version_ != kCoreVersion) {
        throw std::invalid_argument("planner cost artifact engine version does not match this runtime");
    }
    if (model.workload_version_ != kWorkloadVersion) {
        throw std::invalid_argument("planner cost artifact workload version does not match this runtime");
    }
    if (model.host_fingerprint_ != planner_host_fingerprint()) {
        throw std::invalid_argument("planner cost artifact host does not match this runtime");
    }
    model.artifact_fingerprint_ = fingerprint_text(text);
    return model;
}


bool cuda_available() noexcept { return detail::cuda_available(); }
std::string cuda_unavailable_reason() { return detail::cuda_unavailable_reason(); }
std::string cuda_device_name() { return detail::cuda_device_name(); }

Target cuda_target() {
    if (!detail::cuda_available()) {
        throw std::runtime_error(detail::cuda_unavailable_reason());
    }
    std::size_t max_qubits = 0U;
    std::size_t bytes = sizeof(Complex);
    const std::size_t memory = detail::cuda_total_memory_bytes();
    while (bytes <= memory / 2U) { bytes *= 2U; ++max_qubits; }
    return {"native-cuda", {OperationCode::H, OperationCode::X, OperationCode::Y, OperationCode::Z,
        OperationCode::RX, OperationCode::RY, OperationCode::RZ, OperationCode::CX,
        OperationCode::CZ, OperationCode::SWAP}, {ResultMode::StateVector}, max_qubits,
        true, true, false, false, false, false};
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
        true,
    };
}

ExecutionPlan plan(
    const Program& program,
    ResultMode result_mode,
    const std::string& backend,
    const PlannerCostModel* cost_model
) {
    if (result_mode == ResultMode::Expectation || result_mode == ResultMode::Variance) {
        throw std::invalid_argument(
            "observable result mode requires expectation_plan or variance_plan"
        );
    }
    return prepare_program(program, result_mode, backend, true, cost_model).execution_plan;
}

ExecutionPlan expectation_plan(
    const Program& program,
    PauliZ observable,
    const std::string& backend,
    const PlannerCostModel* cost_model
) {
    return prepare_expectation(
        program, observable, ResultMode::Expectation, backend, true, cost_model
    ).execution_plan;
}

ExecutionPlan variance_plan(
    const Program& program,
    PauliZ observable,
    const std::string& backend,
    const PlannerCostModel* cost_model
) {
    return prepare_expectation(
        program, observable, ResultMode::Variance, backend, true, cost_model
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
    if (prepared.execution_plan.method == "cuda-statevector") {
        return {
            detail::cuda_statevector(program.num_qubits(), cuda_steps(prepared.steps)),
            prepared.execution_plan.backend,
        };
    }
    return run_statevector(program.num_qubits(), prepared.steps, prepared.execution_plan);
}

Probabilities probabilities(const Program& program, const std::string& backend) {
    PreparedProgram prepared = prepare_program(program, ResultMode::Probabilities, backend);
    const std::vector<Complex>& state = run_statevector_workspace(
        program.num_qubits(), prepared.steps
    );
    return {state_probabilities(state), prepared.execution_plan.backend};
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
    std::mt19937_64 generator = sampling_generator(seed);
    if (prepared.execution_plan.method == "stabilizer") {
        const detail::StabilizerSupport support = detail::build_stabilizer_support(program);
        return {
            detail::draw_stabilizer_samples(support, shots, generator),
            shots,
            program.num_qubits(),
            prepared.execution_plan.backend,
        };
    }

    const std::vector<Complex>& state = run_statevector_workspace(
        program.num_qubits(), prepared.steps
    );
    const AliasSampler distribution(state_probabilities(state));
    const std::size_t value_count = checked_product(
        shots, program.num_qubits(), "sample result shape exceeds native address space"
    );
    std::vector<std::int8_t> values(value_count);
    draw_samples(distribution, generator, program.num_qubits(), shots, values.data());

    return {
        std::move(values),
        shots,
        program.num_qubits(),
        prepared.execution_plan.backend,
    };
}

SamplesBatch sample_batch(
    const Program& program,
    const std::vector<ParameterSlot>& slots,
    const std::vector<double>& parameter_values,
    std::size_t batch_size,
    std::size_t shots,
    std::optional<std::uint64_t> seed,
    const std::string& backend
) {
    if (shots == 0U) {
        throw std::invalid_argument("shots must be at least 1");
    }
    validate_backend(backend);
    const Target target = native_target();
    target.validate(program, ResultMode::Sample);
    if (!target.parameter_batches) {
        throw std::invalid_argument("target does not support parameter batches");
    }

    const ParameterLayout layout = parameter_layout(program, slots);
    validate_parameter_batch_values(layout, parameter_values, batch_size);
    if (
        layout.parameter_count == 0U &&
        program.num_qubits() >= kStabilizerSamplingMinQubits &&
        detail::supports_stabilizer(program)
    ) {
        const std::size_t total_shots = checked_product(
            batch_size, shots, "sample batch shot count exceeds native address space"
        );
        std::mt19937_64 generator = sampling_generator(seed);
        const detail::StabilizerSupport support = detail::build_stabilizer_support(program);
        std::vector<std::int8_t> values = detail::draw_stabilizer_samples(
            support, total_shots, generator
        );
        return {
            std::move(values),
            batch_size,
            shots,
            program.num_qubits(),
            0U,
            target.name,
            program.operations().size(),
            detail::stabilizer_state_bytes(program.num_qubits()),
        };
    }
    const std::vector<BatchCompiledStep> batch_steps = compile_parameterized_operations(
        program.num_qubits(), program.operations(), operation_indices(program), layout
    );
    const std::size_t row_width = checked_product(
        shots, program.num_qubits(), "sample batch row shape exceeds native address space"
    );
    const std::size_t value_count = checked_product(
        batch_size, row_width, "sample batch result shape exceeds native address space"
    );
    std::vector<std::int8_t> values(value_count);
    std::vector<CompiledStep> steps;
    steps.reserve(batch_steps.size());
    std::mt19937_64 generator = sampling_generator(seed);

    for (std::size_t row_index = 0; row_index < batch_size; ++row_index) {
        if (row_index == 0U || layout.parameter_count != 0U) {
            const double* row = layout.parameter_count == 0U
                ? nullptr
                : parameter_values.data() + row_index * layout.parameter_count;
            materialize_parameterized_steps(batch_steps, row, steps);
        }
        const std::vector<Complex>& state = run_statevector_workspace(
            program.num_qubits(), steps
        );
        const AliasSampler distribution(state_probabilities(state));
        draw_samples(
            distribution,
            generator,
            program.num_qubits(),
            shots,
            values.data() + row_index * row_width
        );
    }

    return {
        std::move(values),
        batch_size,
        shots,
        program.num_qubits(),
        layout.parameter_count,
        target.name,
        batch_steps.size(),
        state_memory_bytes(program.num_qubits()),
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
    double value = 0.0;
    if (prepared.pauli_propagation) {
        value = pauli_z_propagated_value(
            prepared.execution_plan.active_qubits,
            prepared.pauli_operations,
            prepared.observable_qubit
        );
    } else {
        const std::vector<Complex>& state = run_statevector_workspace(
            prepared.execution_plan.active_qubits, prepared.steps
        );
        value = pauli_z_value(state, prepared.observable_qubit);
    }

    return {
        value,
        prepared.execution_plan.backend,
        prepared.execution_plan.active_qubits,
        prepared.execution_plan.compiled_steps,
        prepared.execution_plan.estimated_state_bytes,
    };
}

ExpectationBatch expectation_batch(
    const Program& program,
    PauliZ observable,
    const std::vector<ParameterSlot>& slots,
    const std::vector<double>& parameter_values,
    std::size_t batch_size,
    const std::string& backend
) {
    validate_backend(backend);
    const Target target = native_target();
    target.validate(program, ResultMode::Expectation);
    if (!target.parameter_batches) {
        throw std::invalid_argument("target does not support parameter batches");
    }

    const ParameterLayout layout = parameter_layout(program, slots);
    validate_parameter_batch_values(layout, parameter_values, batch_size);
    ReducedExpectation reduced = reduce_expectation(program, observable);
    std::vector<double> values(batch_size);

    if (!has_relevant_parameter_slot(reduced, layout)) {
        const Expectation scalar = expectation(program, observable, backend);
        std::fill(values.begin(), values.end(), scalar.value);
        return {
            std::move(values),
            batch_size,
            layout.parameter_count,
            scalar.backend,
            scalar.active_qubits,
            scalar.compiled_steps,
            scalar.estimated_state_bytes,
        };
    }

    const std::size_t parameter_count = layout.parameter_count;
    const std::size_t estimated_state_bytes = state_memory_bytes(reduced.active_qubits);
    const std::vector<BatchCompiledStep> batch_steps = compile_parameterized_operations(
        reduced.active_qubits,
        reduced.operations,
        reduced.source_operation_indices,
        layout
    );
    std::vector<CompiledStep> steps;
    steps.reserve(batch_steps.size());
    for (std::size_t row_index = 0; row_index < batch_size; ++row_index) {
        const double* row = parameter_values.data() + row_index * parameter_count;
        materialize_parameterized_steps(batch_steps, row, steps);
        const std::vector<Complex>& state = run_statevector_workspace(
            reduced.active_qubits, steps
        );
        values[row_index] = pauli_z_value(state, reduced.observable_qubit);
    }

    return {
        std::move(values),
        batch_size,
        parameter_count,
        target.name,
        reduced.active_qubits,
        batch_steps.size(),
        estimated_state_bytes,
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
    double expectation_value = 0.0;
    if (prepared.pauli_propagation) {
        expectation_value = pauli_z_propagated_value(
            prepared.execution_plan.active_qubits,
            prepared.pauli_operations,
            prepared.observable_qubit
        );
    } else {
        const std::vector<Complex>& state = run_statevector_workspace(
            prepared.execution_plan.active_qubits, prepared.steps
        );
        expectation_value = pauli_z_value(state, prepared.observable_qubit);
    }
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

std::map<std::string, std::size_t> SamplesBatch::counts(std::size_t batch_index) const {
    if (batch_index >= batch_size) {
        throw std::out_of_range("sample batch index is outside this result");
    }
    std::map<std::string, std::size_t> result;
    const std::size_t batch_offset = batch_index * shots * num_qubits;
    for (std::size_t shot = 0; shot < shots; ++shot) {
        std::string key;
        key.reserve(num_qubits);
        const std::size_t row = batch_offset + shot * num_qubits;
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
