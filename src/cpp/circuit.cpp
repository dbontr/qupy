#include "qupy/circuit.hpp"

#include "qupy/detail/fingerprint.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace qupy {
namespace {

constexpr std::uint32_t kCircuitIrVersion = 1U;

struct CircuitOperationSpec {
    std::size_t qubits;
    std::size_t parameters;
    std::size_t classical_bits;
};

[[nodiscard]] CircuitOperationSpec circuit_operation_spec(CircuitOperationCode code) {
    switch (code) {
    case CircuitOperationCode::H:
    case CircuitOperationCode::X:
    case CircuitOperationCode::Y:
    case CircuitOperationCode::Z:
    case CircuitOperationCode::Reset:
        return {1U, 0U, 0U};
    case CircuitOperationCode::RX:
    case CircuitOperationCode::RY:
    case CircuitOperationCode::RZ:
        return {1U, 1U, 0U};
    case CircuitOperationCode::CX:
    case CircuitOperationCode::CZ:
    case CircuitOperationCode::SWAP:
        return {2U, 0U, 0U};
    case CircuitOperationCode::Measure:
        return {1U, 0U, 1U};
    case CircuitOperationCode::Barrier:
        return {0U, 0U, 0U};
    }
    throw std::invalid_argument("unknown circuit operation code");
}

[[nodiscard]] const char* circuit_operation_name(CircuitOperationCode code) {
    switch (code) {
    case CircuitOperationCode::H: return "h";
    case CircuitOperationCode::X: return "x";
    case CircuitOperationCode::Y: return "y";
    case CircuitOperationCode::Z: return "z";
    case CircuitOperationCode::RX: return "rx";
    case CircuitOperationCode::RY: return "ry";
    case CircuitOperationCode::RZ: return "rz";
    case CircuitOperationCode::CX: return "cx";
    case CircuitOperationCode::CZ: return "cz";
    case CircuitOperationCode::SWAP: return "swap";
    case CircuitOperationCode::Measure: return "measure";
    case CircuitOperationCode::Reset: return "reset";
    case CircuitOperationCode::Barrier: return "barrier";
    }
    throw std::invalid_argument("unknown circuit operation code");
}

[[nodiscard]] CircuitOperationCode circuit_code(OperationCode code) {
    switch (code) {
    case OperationCode::H: return CircuitOperationCode::H;
    case OperationCode::X: return CircuitOperationCode::X;
    case OperationCode::Y: return CircuitOperationCode::Y;
    case OperationCode::Z: return CircuitOperationCode::Z;
    case OperationCode::RX: return CircuitOperationCode::RX;
    case OperationCode::RY: return CircuitOperationCode::RY;
    case OperationCode::RZ: return CircuitOperationCode::RZ;
    case OperationCode::CX: return CircuitOperationCode::CX;
    case OperationCode::CZ: return CircuitOperationCode::CZ;
    case OperationCode::SWAP: return CircuitOperationCode::SWAP;
    }
    throw std::invalid_argument("unknown program operation code");
}

[[nodiscard]] std::string qasm_instruction(const CircuitInstruction& instruction) {
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << std::setprecision(std::numeric_limits<double>::max_digits10);

    if (instruction.code == CircuitOperationCode::Measure) {
        output << "c[" << instruction.classical_bits.front() << "] = measure q["
               << instruction.qubits.front() << "];";
        return output.str();
    }
    if (instruction.code == CircuitOperationCode::Reset) {
        output << "reset q[" << instruction.qubits.front() << "];";
        return output.str();
    }
    if (instruction.code == CircuitOperationCode::Barrier) {
        output << "barrier";
        for (std::size_t index = 0; index < instruction.qubits.size(); ++index) {
            output << (index == 0U ? " " : ", ") << "q[" << instruction.qubits[index] << ']';
        }
        output << ';';
        return output.str();
    }

    output << circuit_operation_name(instruction.code);
    if (!instruction.parameters.empty()) {
        output << '(';
        for (std::size_t index = 0; index < instruction.parameters.size(); ++index) {
            if (index != 0U) {
                output << ", ";
            }
            output << instruction.parameters[index];
        }
        output << ')';
    }
    output << ' ';
    for (std::size_t index = 0; index < instruction.qubits.size(); ++index) {
        if (index != 0U) {
            output << ", ";
        }
        output << "q[" << instruction.qubits[index] << ']';
    }
    output << ';';
    return output.str();
}

[[nodiscard]] CircuitInstruction gate_instruction(
    CircuitOperationCode code,
    std::vector<std::size_t> qubits,
    std::vector<double> parameters,
    std::optional<ClassicalCondition> condition
) {
    return {code, std::move(qubits), std::move(parameters), {}, condition};
}

}  // namespace

std::string CircuitInstruction::name() const {
    return circuit_operation_name(code);
}

Circuit::Circuit(std::size_t num_qubits, std::size_t num_clbits)
    : num_qubits_(num_qubits), num_clbits_(num_clbits) {
    if (num_qubits_ == 0U) {
        throw std::invalid_argument("num_qubits must be at least 1");
    }
}

std::size_t Circuit::num_qubits() const noexcept { return num_qubits_; }
std::size_t Circuit::num_clbits() const noexcept { return num_clbits_; }
const std::vector<CircuitInstruction>& Circuit::instructions() const noexcept {
    return instructions_;
}

Circuit Circuit::appended(CircuitInstruction instruction) const {
    const CircuitOperationSpec spec = circuit_operation_spec(instruction.code);
    if (instruction.code == CircuitOperationCode::Barrier) {
        if (!instruction.parameters.empty() || !instruction.classical_bits.empty() ||
            instruction.condition.has_value()) {
            throw std::invalid_argument(
                "barrier cannot have parameters, classical bits, or a condition"
            );
        }
    } else {
        if (instruction.qubits.size() != spec.qubits) {
            throw std::invalid_argument("circuit operation has an invalid qubit count");
        }
        if (instruction.parameters.size() != spec.parameters) {
            throw std::invalid_argument("circuit operation has an invalid parameter count");
        }
        if (instruction.classical_bits.size() != spec.classical_bits) {
            throw std::invalid_argument("circuit operation has an invalid classical-bit count");
        }
    }

    for (const std::size_t qubit : instruction.qubits) {
        if (qubit >= num_qubits_) {
            throw std::invalid_argument("qubit is outside this circuit");
        }
    }
    if (instruction.code == CircuitOperationCode::Barrier) {
        std::vector<std::size_t> sorted = instruction.qubits;
        std::sort(sorted.begin(), sorted.end());
        if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
            throw std::invalid_argument("barrier qubits must be unique");
        }
    } else if (instruction.qubits.size() == 2U && instruction.qubits[0] == instruction.qubits[1]) {
        throw std::invalid_argument("a circuit operation cannot use the same qubit twice");
    }
    for (const double parameter : instruction.parameters) {
        if (!std::isfinite(parameter)) {
            throw std::invalid_argument("circuit operation parameters must be finite");
        }
    }
    for (const std::size_t bit : instruction.classical_bits) {
        if (bit >= num_clbits_) {
            throw std::invalid_argument("classical bit is outside this circuit");
        }
    }
    if (instruction.condition.has_value() && instruction.condition->bit >= num_clbits_) {
        throw std::invalid_argument("condition bit is outside this circuit");
    }

    Circuit next = *this;
    next.instructions_.push_back(std::move(instruction));
    return next;
}

std::string Circuit::canonical_text() const {
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << "qupy-circuit " << kCircuitIrVersion << '\n';
    output << "qubits " << num_qubits_ << '\n';
    output << "clbits " << num_clbits_ << '\n';
    for (const CircuitInstruction& instruction : instructions_) {
        output << "op " << circuit_operation_name(instruction.code) << " q";
        if (instruction.qubits.empty()) {
            output << '-';
        } else {
            for (std::size_t index = 0; index < instruction.qubits.size(); ++index) {
                if (index != 0U) {
                    output << ',';
                }
                output << instruction.qubits[index];
            }
        }
        output << " p";
        if (instruction.parameters.empty()) {
            output << '-';
        } else {
            for (std::size_t index = 0; index < instruction.parameters.size(); ++index) {
                if (index != 0U) {
                    output << ',';
                }
                const auto bits = std::bit_cast<std::uint64_t>(instruction.parameters[index]);
                output << std::hex << std::setfill('0') << std::setw(16) << bits << std::dec;
            }
        }
        output << " c";
        if (instruction.classical_bits.empty()) {
            output << '-';
        } else {
            for (std::size_t index = 0; index < instruction.classical_bits.size(); ++index) {
                if (index != 0U) {
                    output << ',';
                }
                output << instruction.classical_bits[index];
            }
        }
        output << " if";
        if (instruction.condition.has_value()) {
            output << instruction.condition->bit << '=' << (instruction.condition->value ? 1 : 0);
        } else {
            output << '-';
        }
        output << '\n';
    }
    return output.str();
}

std::string Circuit::fingerprint() const {
    return detail::fingerprint_text(canonical_text());
}

std::string Circuit::to_openqasm3() const {
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << "OPENQASM 3.1;\n";
    output << "include \"stdgates.inc\";\n";
    output << "qubit[" << num_qubits_ << "] q;\n";
    if (num_clbits_ != 0U) {
        output << "bit[" << num_clbits_ << "] c;\n";
    }
    for (const CircuitInstruction& instruction : instructions_) {
        const std::string statement = qasm_instruction(instruction);
        if (instruction.condition.has_value()) {
            output << "if (c[" << instruction.condition->bit << "] == "
                   << (instruction.condition->value ? 1 : 0) << ") {\n";
            output << "  " << statement << '\n';
            output << "}\n";
        } else {
            output << statement << '\n';
        }
    }
    return output.str();
}

Program Circuit::to_program() const {
    Program program(num_qubits_);
    for (const CircuitInstruction& instruction : instructions_) {
        if (instruction.condition.has_value() || instruction.code == CircuitOperationCode::Measure ||
            instruction.code == CircuitOperationCode::Reset) {
            throw std::invalid_argument(
                "circuit contains hardware control and cannot be lowered to Program"
            );
        }
        switch (instruction.code) {
        case CircuitOperationCode::H:
            program = qupy::h(program, instruction.qubits[0]);
            break;
        case CircuitOperationCode::X:
            program = qupy::x(program, instruction.qubits[0]);
            break;
        case CircuitOperationCode::Y:
            program = qupy::y(program, instruction.qubits[0]);
            break;
        case CircuitOperationCode::Z:
            program = qupy::z(program, instruction.qubits[0]);
            break;
        case CircuitOperationCode::RX:
            program = qupy::rx(program, instruction.parameters[0], instruction.qubits[0]);
            break;
        case CircuitOperationCode::RY:
            program = qupy::ry(program, instruction.parameters[0], instruction.qubits[0]);
            break;
        case CircuitOperationCode::RZ:
            program = qupy::rz(program, instruction.parameters[0], instruction.qubits[0]);
            break;
        case CircuitOperationCode::CX:
            program = qupy::cx(program, instruction.qubits[0], instruction.qubits[1]);
            break;
        case CircuitOperationCode::CZ:
            program = qupy::cz(program, instruction.qubits[0], instruction.qubits[1]);
            break;
        case CircuitOperationCode::SWAP:
            program = qupy::swap(program, instruction.qubits[0], instruction.qubits[1]);
            break;
        case CircuitOperationCode::Barrier:
            break;
        case CircuitOperationCode::Measure:
        case CircuitOperationCode::Reset:
            throw std::logic_error("non-unitary circuit instruction escaped lowering guard");
        }
    }
    return program;
}

Circuit Circuit::from_program(const Program& program, std::size_t num_clbits) {
    Circuit circuit(program.num_qubits(), num_clbits);
    for (const Operation& operation : program.operations()) {
        circuit = circuit.appended(
            {circuit_code(operation.code), operation.qubits, operation.parameters, {}, std::nullopt}
        );
    }
    return circuit;
}

Circuit Circuit::h(std::size_t qubit, std::optional<ClassicalCondition> condition) const {
    return appended(gate_instruction(CircuitOperationCode::H, {qubit}, {}, condition));
}

Circuit Circuit::x(std::size_t qubit, std::optional<ClassicalCondition> condition) const {
    return appended(gate_instruction(CircuitOperationCode::X, {qubit}, {}, condition));
}

Circuit Circuit::y(std::size_t qubit, std::optional<ClassicalCondition> condition) const {
    return appended(gate_instruction(CircuitOperationCode::Y, {qubit}, {}, condition));
}

Circuit Circuit::z(std::size_t qubit, std::optional<ClassicalCondition> condition) const {
    return appended(gate_instruction(CircuitOperationCode::Z, {qubit}, {}, condition));
}

Circuit Circuit::rx(
    double angle,
    std::size_t qubit,
    std::optional<ClassicalCondition> condition
) const {
    return appended(gate_instruction(CircuitOperationCode::RX, {qubit}, {angle}, condition));
}

Circuit Circuit::ry(
    double angle,
    std::size_t qubit,
    std::optional<ClassicalCondition> condition
) const {
    return appended(gate_instruction(CircuitOperationCode::RY, {qubit}, {angle}, condition));
}

Circuit Circuit::rz(
    double angle,
    std::size_t qubit,
    std::optional<ClassicalCondition> condition
) const {
    return appended(gate_instruction(CircuitOperationCode::RZ, {qubit}, {angle}, condition));
}

Circuit Circuit::cx(
    std::size_t control,
    std::size_t target,
    std::optional<ClassicalCondition> condition
) const {
    return appended(gate_instruction(CircuitOperationCode::CX, {control, target}, {}, condition));
}

Circuit Circuit::cz(
    std::size_t control,
    std::size_t target,
    std::optional<ClassicalCondition> condition
) const {
    return appended(gate_instruction(CircuitOperationCode::CZ, {control, target}, {}, condition));
}

Circuit Circuit::swap(
    std::size_t first,
    std::size_t second,
    std::optional<ClassicalCondition> condition
) const {
    return appended(gate_instruction(CircuitOperationCode::SWAP, {first, second}, {}, condition));
}

Circuit Circuit::measure(
    std::size_t qubit,
    std::size_t classical_bit,
    std::optional<ClassicalCondition> condition
) const {
    return appended({CircuitOperationCode::Measure, {qubit}, {}, {classical_bit}, condition});
}

Circuit Circuit::reset(
    std::size_t qubit,
    std::optional<ClassicalCondition> condition
) const {
    return appended({CircuitOperationCode::Reset, {qubit}, {}, {}, condition});
}

Circuit Circuit::barrier(const std::vector<std::size_t>& qubits) const {
    return appended({CircuitOperationCode::Barrier, qubits, {}, {}, std::nullopt});
}

std::uint32_t circuit_ir_version() noexcept {
    return kCircuitIrVersion;
}

}  // namespace qupy
