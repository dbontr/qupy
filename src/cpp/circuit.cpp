#include "qupy/circuit.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string_view>
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
    return fingerprint_text(canonical_text());
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
