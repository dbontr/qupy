#pragma once

#include "qupy/core.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace qupy {

enum class CircuitOperationCode : std::uint8_t {
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
    Measure,
    Reset,
    Barrier,
};

struct ClassicalCondition {
    ClassicalCondition(std::size_t bit_, bool value_) : bit(bit_), value(value_) {}

    std::size_t bit;
    bool value;
};

struct CircuitInstruction {
    CircuitOperationCode code;
    std::vector<std::size_t> qubits;
    std::vector<double> parameters;
    std::vector<std::size_t> classical_bits;
    std::optional<ClassicalCondition> condition;

    [[nodiscard]] std::string name() const;
};

class Circuit {
public:
    explicit Circuit(std::size_t num_qubits, std::size_t num_clbits = 0U);

    [[nodiscard]] std::size_t num_qubits() const noexcept;
    [[nodiscard]] std::size_t num_clbits() const noexcept;
    [[nodiscard]] const std::vector<CircuitInstruction>& instructions() const noexcept;
    [[nodiscard]] std::string canonical_text() const;
    [[nodiscard]] std::string fingerprint() const;
    [[nodiscard]] std::string to_openqasm3() const;
    [[nodiscard]] Program to_program() const;
    [[nodiscard]] static Circuit from_program(
        const Program& program,
        std::size_t num_clbits = 0U
    );

    [[nodiscard]] Circuit h(
        std::size_t qubit,
        std::optional<ClassicalCondition> condition = std::nullopt
    ) const;
    [[nodiscard]] Circuit x(
        std::size_t qubit,
        std::optional<ClassicalCondition> condition = std::nullopt
    ) const;
    [[nodiscard]] Circuit y(
        std::size_t qubit,
        std::optional<ClassicalCondition> condition = std::nullopt
    ) const;
    [[nodiscard]] Circuit z(
        std::size_t qubit,
        std::optional<ClassicalCondition> condition = std::nullopt
    ) const;
    [[nodiscard]] Circuit rx(
        double angle,
        std::size_t qubit,
        std::optional<ClassicalCondition> condition = std::nullopt
    ) const;
    [[nodiscard]] Circuit ry(
        double angle,
        std::size_t qubit,
        std::optional<ClassicalCondition> condition = std::nullopt
    ) const;
    [[nodiscard]] Circuit rz(
        double angle,
        std::size_t qubit,
        std::optional<ClassicalCondition> condition = std::nullopt
    ) const;
    [[nodiscard]] Circuit cx(
        std::size_t control,
        std::size_t target,
        std::optional<ClassicalCondition> condition = std::nullopt
    ) const;
    [[nodiscard]] Circuit cz(
        std::size_t control,
        std::size_t target,
        std::optional<ClassicalCondition> condition = std::nullopt
    ) const;
    [[nodiscard]] Circuit swap(
        std::size_t first,
        std::size_t second,
        std::optional<ClassicalCondition> condition = std::nullopt
    ) const;
    [[nodiscard]] Circuit measure(
        std::size_t qubit,
        std::size_t classical_bit,
        std::optional<ClassicalCondition> condition = std::nullopt
    ) const;
    [[nodiscard]] Circuit reset(
        std::size_t qubit,
        std::optional<ClassicalCondition> condition = std::nullopt
    ) const;
    [[nodiscard]] Circuit barrier(const std::vector<std::size_t>& qubits = {}) const;

private:
    [[nodiscard]] Circuit appended(CircuitInstruction instruction) const;

    std::size_t num_qubits_;
    std::size_t num_clbits_;
    std::vector<CircuitInstruction> instructions_;
};

[[nodiscard]] std::uint32_t circuit_ir_version() noexcept;

}  // namespace qupy
