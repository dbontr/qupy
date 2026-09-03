#include "qupy/circuit.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Function>
void require_invalid(Function&& function, const std::string& message) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(message);
}

void test_unitary_lowering_round_trip() {
    qupy::Circuit circuit(3);
    circuit = circuit.h(0);
    circuit = circuit.rx(0.25, 1);
    circuit = circuit.cx(0, 1);
    circuit = circuit.barrier({0, 1});
    circuit = circuit.swap(1, 2);

    const qupy::Program program = circuit.to_program();
    require(program.operations().size() == 4U, "barrier was not removed during lowering");
    require(program.operations()[0].code == qupy::OperationCode::H, "H lowering is wrong");
    require(program.operations()[1].code == qupy::OperationCode::RX, "RX lowering is wrong");
    require(program.operations()[2].code == qupy::OperationCode::CX, "CX lowering is wrong");
    require(program.operations()[3].code == qupy::OperationCode::SWAP, "SWAP lowering is wrong");

    const qupy::Circuit restored = qupy::Circuit::from_program(program, 2);
    require(restored.num_qubits() == 3U, "restored circuit qubit count is wrong");
    require(restored.num_clbits() == 2U, "restored circuit classical-bit count is wrong");
    require(restored.to_program().fingerprint() == program.fingerprint(), "unitary round trip changed Program semantics");
}

void test_dynamic_openqasm() {
    qupy::Circuit circuit(2, 1);
    circuit = circuit.h(0);
    circuit = circuit.cx(0, 1);
    circuit = circuit.measure(0, 0);
    circuit = circuit.x(1, qupy::ClassicalCondition{0, true});
    circuit = circuit.reset(0, qupy::ClassicalCondition{0, false});
    circuit = circuit.barrier({0, 1});
    circuit = circuit.barrier();

    const std::string expected =
        "OPENQASM 3.1;\n"
        "include \"stdgates.inc\";\n"
        "qubit[2] q;\n"
        "bit[1] c;\n"
        "h q[0];\n"
        "cx q[0], q[1];\n"
        "c[0] = measure q[0];\n"
        "if (c[0] == 1) {\n"
        "  x q[1];\n"
        "}\n"
        "if (c[0] == 0) {\n"
        "  reset q[0];\n"
        "}\n"
        "barrier q[0], q[1];\n"
        "barrier;\n";
    require(circuit.to_openqasm3() == expected, "dynamic OpenQASM output is wrong");
    require(circuit.instructions()[2].code == qupy::CircuitOperationCode::Measure, "measurement code is wrong");
    require(circuit.instructions()[3].condition.has_value(), "conditional gate lost its condition");
    require(circuit.instructions()[3].condition->value, "conditional gate polarity is wrong");
}

void test_identity_and_validation() {
    const qupy::Circuit first = qupy::Circuit(1, 1).ry(0.375, 0).measure(0, 0);
    const qupy::Circuit second = qupy::Circuit(1, 1).ry(0.375, 0).measure(0, 0);
    require(first.canonical_text() == second.canonical_text(), "canonical circuit text is unstable");
    require(first.fingerprint() == second.fingerprint(), "circuit fingerprint is unstable");
    require(first.fingerprint().size() == 64U, "circuit fingerprint is not SHA-256 sized");
    require(qupy::circuit_ir_version() == 1U, "unexpected circuit IR version");

    require_invalid([] { static_cast<void>(qupy::Circuit(0)); }, "zero-qubit circuit was accepted");
    require_invalid(
        [] { static_cast<void>(qupy::Circuit(1, 1).measure(0, 1)); },
        "out-of-range measurement bit was accepted"
    );
    require_invalid(
        [] { static_cast<void>(qupy::Circuit(1, 1).x(0, qupy::ClassicalCondition{1, true})); },
        "out-of-range condition bit was accepted"
    );
    require_invalid(
        [] { static_cast<void>(qupy::Circuit(2).cx(0, 0)); },
        "duplicate two-qubit operand was accepted"
    );
    require_invalid(
        [] { static_cast<void>(qupy::Circuit(2).barrier({0, 0})); },
        "duplicate barrier operand was accepted"
    );
    require_invalid(
        [] {
            static_cast<void>(
                qupy::Circuit(1).rz(std::numeric_limits<double>::quiet_NaN(), 0)
            );
        },
        "non-finite rotation was accepted"
    );
}

void test_dynamic_lowering_fails_closed() {
    require_invalid(
        [] { static_cast<void>(qupy::Circuit(1, 1).measure(0, 0).to_program()); },
        "measurement circuit lowered to Program"
    );
    require_invalid(
        [] { static_cast<void>(qupy::Circuit(1).reset(0).to_program()); },
        "reset circuit lowered to Program"
    );
    require_invalid(
        [] {
            static_cast<void>(
                qupy::Circuit(1, 1).x(0, qupy::ClassicalCondition{0, true}).to_program()
            );
        },
        "conditional circuit lowered to Program"
    );
}

}  // namespace

int main() {
    test_unitary_lowering_round_trip();
    test_dynamic_openqasm();
    test_identity_and_validation();
    test_dynamic_lowering_fails_closed();
    return 0;
}
