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

    [[nodiscard]] bool supports(OperationCode code) const;
    [[nodiscard]] bool supports(ResultMode mode) const;
    void validate(const Program& program, ResultMode mode) const;
};

struct ExecutionPlan {
    std::string backend;
    std::string method;
    bool exact;
    std::size_t threads;
};

struct StateVector {
    std::vector<Complex> values;
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
};

[[nodiscard]] Target native_target();
[[nodiscard]] ExecutionPlan plan(
    const Program& program,
    ResultMode result_mode,
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

[[nodiscard]] const char* core_language() noexcept;
[[nodiscard]] std::size_t parallel_threads() noexcept;

}  // namespace qupy
