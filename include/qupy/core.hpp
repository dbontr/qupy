#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace qupy {

using Complex = std::complex<double>;

enum class OperationCode : std::uint8_t {
    H,
    X,
    Y,
    Z,
    RX,
    RY,
    RZ,
    CX,
    CZ,
    SWAP,
};

enum class ResultMode : std::uint8_t {
    Sample,
    Expectation,
    Probabilities,
    Variance,
    StateVector,
};

struct Operation {
    OperationCode code;
    std::vector<std::size_t> qubits;
    std::vector<double> parameters;

    [[nodiscard]] std::string name() const;
};

class Program {
public:
    explicit Program(std::size_t num_qubits);

    [[nodiscard]] std::size_t num_qubits() const noexcept;
    [[nodiscard]] const std::vector<Operation>& operations() const noexcept;
    [[nodiscard]] std::string canonical_text() const;
    [[nodiscard]] std::string fingerprint() const;
    [[nodiscard]] Program appended(Operation operation) const;

private:
    std::size_t num_qubits_;
    std::vector<Operation> operations_;
};

struct PauliZ {
    std::size_t qubit;
};

struct Target {
    std::string name;
    std::vector<OperationCode> operations;
    std::vector<ResultMode> result_modes;
    std::optional<std::size_t> max_qubits;
    bool simulator = false;
    bool state_access = false;
    bool mid_circuit_measurement = false;
    bool reset = false;
    bool dynamic_control = false;
    bool parameter_batches = false;

    [[nodiscard]] bool supports(OperationCode code) const;
    [[nodiscard]] bool supports(ResultMode mode) const;
    [[nodiscard]] std::string fingerprint() const;
    void validate(const Program& program, ResultMode mode) const;
};

struct ExecutionPlan {
    std::string backend;
    std::string method;
    bool exact;
    std::size_t threads;
    std::size_t original_operations;
    std::size_t compiled_steps;
    std::size_t active_qubits;
    std::size_t estimated_state_bytes;
    ResultMode result_mode;
    std::string program_fingerprint;
    std::string target_fingerprint;
    std::string cache_key;
};

struct StateVector {
    std::vector<Complex> values;
    std::string backend;
};

struct Probabilities {
    std::vector<double> values;
    std::string backend;
};

struct Samples {
    std::vector<std::int8_t> values;
    std::size_t shots;
    std::size_t num_qubits;
    std::string backend;

    [[nodiscard]] std::map<std::string, std::size_t> counts() const;
};

struct Expectation {
    double value;
    std::string backend;
    std::size_t active_qubits;
    std::size_t compiled_steps;
    std::size_t estimated_state_bytes;
};

struct Variance {
    double value;
    std::string backend;
    std::size_t active_qubits;
    std::size_t compiled_steps;
    std::size_t estimated_state_bytes;
};

[[nodiscard]] Target native_target();
[[nodiscard]] ExecutionPlan plan(
    const Program& program,
    ResultMode result_mode,
    const std::string& backend = "auto"
);
[[nodiscard]] ExecutionPlan expectation_plan(
    const Program& program,
    PauliZ observable,
    const std::string& backend = "auto"
);
[[nodiscard]] ExecutionPlan variance_plan(
    const Program& program,
    PauliZ observable,
    const std::string& backend = "auto"
);

[[nodiscard]] Program h(const Program& program, std::size_t qubit);
[[nodiscard]] Program x(const Program& program, std::size_t qubit);
[[nodiscard]] Program y(const Program& program, std::size_t qubit);
[[nodiscard]] Program z(const Program& program, std::size_t qubit);
[[nodiscard]] Program rx(const Program& program, double angle, std::size_t qubit);
[[nodiscard]] Program ry(const Program& program, double angle, std::size_t qubit);
[[nodiscard]] Program rz(const Program& program, double angle, std::size_t qubit);
[[nodiscard]] Program cx(const Program& program, std::size_t control, std::size_t target);
[[nodiscard]] Program cz(const Program& program, std::size_t control, std::size_t target);
[[nodiscard]] Program swap(const Program& program, std::size_t first, std::size_t second);
[[nodiscard]] PauliZ pauli_z(std::size_t qubit);

[[nodiscard]] StateVector statevector(
    const Program& program,
    const std::string& backend = "auto"
);
[[nodiscard]] Probabilities probabilities(
    const Program& program,
    const std::string& backend = "auto"
);
[[nodiscard]] Samples sample(
    const Program& program,
    std::size_t shots = 1024,
    std::optional<std::uint64_t> seed = std::nullopt,
    const std::string& backend = "auto"
);
[[nodiscard]] Expectation expectation(
    const Program& program,
    PauliZ observable,
    const std::string& backend = "auto"
);
[[nodiscard]] Variance variance(
    const Program& program,
    PauliZ observable,
    const std::string& backend = "auto"
);

[[nodiscard]] const char* core_language() noexcept;
[[nodiscard]] const char* core_version() noexcept;
[[nodiscard]] std::uint32_t ir_version() noexcept;
[[nodiscard]] std::size_t parallel_threads() noexcept;

}  // namespace qupy
