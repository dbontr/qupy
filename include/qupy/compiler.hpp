#pragma once

#include "qupy/circuit.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace qupy {

struct Coupling {
    Coupling(std::size_t first_, std::size_t second_) : first(first_), second(second_) {}

    std::size_t first;
    std::size_t second;
};

struct OperationDuration {
    OperationDuration(CircuitOperationCode code_, double nanoseconds_)
        : code(code_), nanoseconds(nanoseconds_) {}

    CircuitOperationCode code;
    double nanoseconds;
};

class HardwareTarget {
public:
    HardwareTarget(
        std::string name,
        std::size_t num_qubits,
        std::vector<CircuitOperationCode> one_qubit_operations,
        std::vector<CircuitOperationCode> two_qubit_operations,
        std::vector<Coupling> couplings = {},
        bool measurement = false,
        bool mid_circuit_measurement = false,
        bool reset = false,
        bool dynamic_control = false,
        std::vector<OperationDuration> durations = {}
    );

    [[nodiscard]] const std::string& name() const noexcept;
    [[nodiscard]] std::size_t num_qubits() const noexcept;
    [[nodiscard]] const std::vector<CircuitOperationCode>& one_qubit_operations() const noexcept;
    [[nodiscard]] const std::vector<CircuitOperationCode>& two_qubit_operations() const noexcept;
    [[nodiscard]] const std::vector<Coupling>& couplings() const noexcept;
    [[nodiscard]] bool measurement() const noexcept;
    [[nodiscard]] bool mid_circuit_measurement() const noexcept;
    [[nodiscard]] bool reset() const noexcept;
    [[nodiscard]] bool dynamic_control() const noexcept;
    [[nodiscard]] const std::vector<OperationDuration>& durations() const noexcept;
    [[nodiscard]] bool adjacent(std::size_t first, std::size_t second) const noexcept;
    [[nodiscard]] bool supports(
        CircuitOperationCode code,
        const std::vector<std::size_t>& qubits
    ) const noexcept;
    [[nodiscard]] std::optional<double> duration_ns(CircuitOperationCode code) const noexcept;
    [[nodiscard]] std::string canonical_text() const;
    [[nodiscard]] std::string fingerprint() const;

private:
    std::string name_;
    std::size_t num_qubits_;
    std::vector<CircuitOperationCode> one_qubit_operations_;
    std::vector<CircuitOperationCode> two_qubit_operations_;
    std::vector<Coupling> couplings_;
    bool measurement_;
    bool mid_circuit_measurement_;
    bool reset_;
    bool dynamic_control_;
    std::vector<OperationDuration> durations_;
};

struct ScheduledInstruction {
    std::size_t instruction_index;
    double start_ns;
    double duration_ns;
};

struct CompilationResult {
    Circuit circuit;
    std::vector<std::size_t> initial_layout;
    std::vector<std::size_t> final_layout;
    std::size_t original_operations;
    std::size_t optimized_operations;
    std::size_t routed_operations;
    std::size_t compiled_operations;
    std::size_t inserted_swaps;
    std::size_t decompositions;
    std::size_t depth;
    std::optional<double> duration_ns;
    std::vector<ScheduledInstruction> schedule;
    std::string target_fingerprint;
};

[[nodiscard]] CompilationResult compile_circuit(
    const Circuit& circuit,
    const HardwareTarget& target,
    const std::vector<std::size_t>& initial_layout = {},
    std::uint32_t optimization_level = 1U
);

}  // namespace qupy
