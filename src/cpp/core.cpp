#include "qupy/core.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>

#ifdef QUPY_HAS_OPENMP
#include <omp.h>
#endif

namespace qupy {
namespace {

#ifdef QUPY_HAS_OPENMP
constexpr std::size_t kParallelThreshold = 1U << 16U;
#endif

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

[[nodiscard]] std::size_t state_dimension(std::size_t num_qubits) {
    if (num_qubits >= std::numeric_limits<std::size_t>::digits) {
        throw std::length_error("qubit count exceeds native address space");
    }
    return std::size_t{1} << num_qubits;
}

using Matrix2 = std::array<Complex, 4>;

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
    const auto dimension = static_cast<std::ptrdiff_t>(state.size());

#ifdef QUPY_HAS_OPENMP
#pragma omp parallel for schedule(static) if(state.size() >= kParallelThreshold)
#endif
    for (std::ptrdiff_t raw = 0; raw < dimension; ++raw) {
        const auto index = static_cast<std::size_t>(raw);
        if ((index & control_mask) != 0U && (index & target_mask) == 0U) {
            std::swap(state[index], state[index | target_mask]);
        }
    }
}

void apply_cz(
    std::vector<Complex>& state,
    std::size_t control,
    std::size_t target
) {
    const std::size_t control_mask = std::size_t{1} << control;
    const std::size_t target_mask = std::size_t{1} << target;
    const auto dimension = static_cast<std::ptrdiff_t>(state.size());

#ifdef QUPY_HAS_OPENMP
#pragma omp parallel for schedule(static) if(state.size() >= kParallelThreshold)
#endif
    for (std::ptrdiff_t raw = 0; raw < dimension; ++raw) {
        const auto index = static_cast<std::size_t>(raw);
        if ((index & control_mask) != 0U && (index & target_mask) != 0U) {
            state[index] = -state[index];
        }
    }
}

void apply_swap(
    std::vector<Complex>& state,
    std::size_t first,
    std::size_t second
) {
    const std::size_t first_mask = std::size_t{1} << first;
    const std::size_t second_mask = std::size_t{1} << second;
    const auto dimension = static_cast<std::ptrdiff_t>(state.size());

#ifdef QUPY_HAS_OPENMP
#pragma omp parallel for schedule(static) if(state.size() >= kParallelThreshold)
#endif
    for (std::ptrdiff_t raw = 0; raw < dimension; ++raw) {
        const auto index = static_cast<std::size_t>(raw);
        if ((index & first_mask) != 0U && (index & second_mask) == 0U) {
            const std::size_t paired = (index ^ first_mask) ^ second_mask;
            std::swap(state[index], state[paired]);
        }
    }
}

[[nodiscard]] StateVector run_statevector(
    const Program& program,
    const ExecutionPlan& execution_plan
) {
    std::vector<Complex> state(state_dimension(program.num_qubits()), Complex{0.0, 0.0});
    state.front() = Complex{1.0, 0.0};
    const double inv_sqrt_two = 1.0 / std::sqrt(2.0);

    for (const Operation& operation : program.operations()) {
        const std::size_t qubit = operation.qubits.front();
        switch (operation.code) {
        case OperationCode::H:
            apply_single(state, {inv_sqrt_two, inv_sqrt_two, inv_sqrt_two, -inv_sqrt_two}, qubit);
            break;
        case OperationCode::X:
            apply_single(state, {0.0, 1.0, 1.0, 0.0}, qubit);
            break;
        case OperationCode::Y:
            apply_single(state, {0.0, Complex{0.0, -1.0}, Complex{0.0, 1.0}, 0.0}, qubit);
            break;
        case OperationCode::Z:
            apply_single(state, {1.0, 0.0, 0.0, -1.0}, qubit);
            break;
        case OperationCode::RX: {
            const double half = operation.parameters.front() / 2.0;
            const double cosine = std::cos(half);
            const Complex sine{0.0, -std::sin(half)};
            apply_single(state, {cosine, sine, sine, cosine}, qubit);
            break;
        }
        case OperationCode::RY: {
            const double half = operation.parameters.front() / 2.0;
            const double cosine = std::cos(half);
            const double sine = std::sin(half);
            apply_single(state, {cosine, -sine, sine, cosine}, qubit);
            break;
        }
        case OperationCode::RZ: {
            const double half = operation.parameters.front() / 2.0;
            const Complex negative = std::polar(1.0, -half);
            const Complex positive = std::polar(1.0, half);
            apply_single(state, {negative, 0.0, 0.0, positive}, qubit);
            break;
        }
        case OperationCode::CX:
            apply_cx(state, operation.qubits[0], operation.qubits[1]);
            break;
        case OperationCode::CZ:
            apply_cz(state, operation.qubits[0], operation.qubits[1]);
            break;
        case OperationCode::SWAP:
            apply_swap(state, operation.qubits[0], operation.qubits[1]);
            break;
        }
    }

    return {std::move(state), execution_plan.backend};
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
        {ResultMode::Sample, ResultMode::Expectation, ResultMode::StateVector},
        std::nullopt,
        true,
    };
}

ExecutionPlan plan(
    const Program& program,
    ResultMode result_mode,
    const std::string& backend
) {
    if (backend != "auto" && backend != "native" && backend != "native-cpu") {
        throw std::invalid_argument("unknown backend: " + backend);
    }
    const Target target = native_target();
    target.validate(program, result_mode);
    return {target.name, "statevector", true, parallel_threads()};
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
    const ExecutionPlan execution_plan = plan(program, ResultMode::StateVector, backend);
    return run_statevector(program, execution_plan);
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
    const ExecutionPlan execution_plan = plan(program, ResultMode::Sample, backend);
    StateVector state = run_statevector(program, execution_plan);
    std::vector<double> probabilities(state.values.size());
    for (std::size_t index = 0; index < state.values.size(); ++index) {
        probabilities[index] = std::norm(state.values[index]);
    }

    std::mt19937_64 generator;
    if (seed.has_value()) {
        generator.seed(*seed);
    } else {
        std::random_device source;
        generator.seed((static_cast<std::uint64_t>(source()) << 32U) ^ source());
    }
    std::discrete_distribution<std::size_t> distribution(
        probabilities.begin(),
        probabilities.end()
    );

    std::vector<std::int8_t> values(shots * program.num_qubits());
    for (std::size_t shot = 0; shot < shots; ++shot) {
        const std::size_t basis = distribution(generator);
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
        execution_plan.backend,
    };
}

Expectation expectation(
    const Program& program,
    PauliZ observable,
    const std::string& backend
) {
    if (observable.qubit >= program.num_qubits()) {
        throw std::invalid_argument("observable qubit is outside this program");
    }
    const ExecutionPlan execution_plan = plan(program, ResultMode::Expectation, backend);
    StateVector state = run_statevector(program, execution_plan);
    const std::size_t mask = std::size_t{1} << observable.qubit;
    const auto dimension = static_cast<std::ptrdiff_t>(state.values.size());
    double value = 0.0;

#ifdef QUPY_HAS_OPENMP
#pragma omp parallel for reduction(+ : value) schedule(static) if(state.values.size() >= kParallelThreshold)
#endif
    for (std::ptrdiff_t raw = 0; raw < dimension; ++raw) {
        const auto index = static_cast<std::size_t>(raw);
        const double sign = (index & mask) == 0U ? 1.0 : -1.0;
        value += sign * std::norm(state.values[index]);
    }

    return {value, execution_plan.backend};
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

std::size_t parallel_threads() noexcept {
#ifdef QUPY_HAS_OPENMP
    return static_cast<std::size_t>(omp_get_max_threads());
#else
    return 1U;
#endif
}

}  // namespace qupy
