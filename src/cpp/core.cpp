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
    ResultMode result_mode,
    std::size_t original_operations,
    std::size_t compiled_steps,
    std::size_t active_qubits,
    const std::string& method
) {
    static_cast<void>(result_mode);
    return {
        "native-cpu",
        method,
        true,
        parallel_threads(),
        original_operations,
        compiled_steps,
        active_qubits,
        state_memory_bytes(active_qubits),
    };
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
    const std::string& backend
) {
    if (observable.qubit >= program.num_qubits()) {
        throw std::invalid_argument("observable qubit is outside this program");
    }
    validate_backend(backend);
    native_target().validate(program, ResultMode::Expectation);

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
    ExecutionPlan execution_plan = make_plan(
        ResultMode::Expectation,
        operations.size(),
        steps.size(),
        active_count,
        method
    );
    return {
        std::move(reduced),
        mapping[observable.qubit],
        std::move(execution_plan),
        std::move(steps),
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
            if (scaled[index] < 1.0) {
                small.push_back(index);
            } else {
                large.push_back(index);
            }
        }

        while (!small.empty() && !large.empty()) {
            const std::size_t low = small.back();
            small.pop_back();
            const std::size_t high = large.back();
            large.pop_back();
            accept_[low] = scaled[low];
            alias_[low] = high;
            scaled[high] = (scaled[high] + scaled[low]) - 1.0;
            if (scaled[high] < 1.0) {
                small.push_back(high);
            } else {
                large.push_back(high);
            }
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

    template <typename Generator>
    [[nodiscard]] std::size_t draw(Generator& generator) const {
        std::uniform_int_distribution<std::size_t> column(0U, accept_.size() - 1U);
        std::uniform_real_distribution<double> coin(0.0, 1.0);
        const std::size_t index = column(generator);
        return coin(generator) < accept_[index] ? index : alias_[index];
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
    return prepare_program(program, result_mode, backend).execution_plan;
}

ExecutionPlan expectation_plan(
    const Program& program,
    PauliZ observable,
    const std::string& backend
) {
    return prepare_expectation(program, observable, backend).execution_plan;
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
    std::vector<double> probabilities(state.values.size());

#ifdef QUPY_HAS_OPENMP
#pragma omp parallel for schedule(static) if(state.values.size() >= kParallelThreshold)
#endif
    for (std::ptrdiff_t raw = 0; raw < static_cast<std::ptrdiff_t>(state.values.size()); ++raw) {
        const auto index = static_cast<std::size_t>(raw);
        probabilities[index] = std::norm(state.values[index]);
    }

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
    PreparedExpectation prepared = prepare_expectation(program, observable, backend);
    StateVector state = run_statevector(
        prepared.program.num_qubits(), prepared.steps, prepared.execution_plan
    );
    const std::size_t mask = std::size_t{1} << prepared.observable_qubit;
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

    return {
        value,
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

std::size_t parallel_threads() noexcept {
#ifdef QUPY_HAS_OPENMP
    return static_cast<std::size_t>(omp_get_max_threads());
#else
    return 1U;
#endif
}

}  // namespace qupy
