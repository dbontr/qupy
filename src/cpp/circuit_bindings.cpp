#include "qupy/circuit.hpp"

#include "qupy/advanced.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <string>
#include <utility>

namespace nb = nanobind;
using namespace nb::literals;

void bind_circuit(nb::module_& module) {
    nb::enum_<qupy::CircuitOperationCode>(module, "CircuitOperationCode")
        .value("H", qupy::CircuitOperationCode::H)
        .value("X", qupy::CircuitOperationCode::X)
        .value("Y", qupy::CircuitOperationCode::Y)
        .value("Z", qupy::CircuitOperationCode::Z)
        .value("RX", qupy::CircuitOperationCode::RX)
        .value("RY", qupy::CircuitOperationCode::RY)
        .value("RZ", qupy::CircuitOperationCode::RZ)
        .value("CX", qupy::CircuitOperationCode::CX)
        .value("CZ", qupy::CircuitOperationCode::CZ)
        .value("SWAP", qupy::CircuitOperationCode::SWAP)
        .value("MEASURE", qupy::CircuitOperationCode::Measure)
        .value("RESET", qupy::CircuitOperationCode::Reset)
        .value("BARRIER", qupy::CircuitOperationCode::Barrier);

    nb::class_<qupy::ClassicalCondition>(module, "ClassicalCondition")
        .def(nb::init<std::size_t, bool>(), "bit"_a, "value"_a)
        .def_ro("bit", &qupy::ClassicalCondition::bit)
        .def_ro("value", &qupy::ClassicalCondition::value);

    nb::class_<qupy::CircuitInstruction>(module, "CircuitInstruction")
        .def_prop_ro("name", &qupy::CircuitInstruction::name)
        .def_ro("code", &qupy::CircuitInstruction::code)
        .def_ro("qubits", &qupy::CircuitInstruction::qubits)
        .def_ro("parameters", &qupy::CircuitInstruction::parameters)
        .def_ro("classical_bits", &qupy::CircuitInstruction::classical_bits)
        .def_ro("condition", &qupy::CircuitInstruction::condition);

    nb::class_<qupy::Circuit>(module, "Circuit")
        .def(nb::init<std::size_t, std::size_t>(), "num_qubits"_a, "num_clbits"_a = 0U)
        .def_prop_ro("num_qubits", &qupy::Circuit::num_qubits)
        .def_prop_ro("num_clbits", &qupy::Circuit::num_clbits)
        .def_prop_ro(
            "instructions",
            [](const qupy::Circuit& circuit) { return circuit.instructions(); }
        )
        .def_prop_ro("canonical_text", &qupy::Circuit::canonical_text)
        .def_prop_ro("fingerprint", &qupy::Circuit::fingerprint)
        .def("to_openqasm3", &qupy::Circuit::to_openqasm3)
        .def("to_program", &qupy::Circuit::to_program)
        .def_static(
            "from_program",
            &qupy::Circuit::from_program,
            "program"_a,
            "num_clbits"_a = 0U
        )
        .def_static(
            "from_openqasm3",
            &qupy::Circuit::from_openqasm3,
            "text"_a
        )
        .def("h", &qupy::Circuit::h, "qubit"_a, "condition"_a = nb::none())
        .def("x", &qupy::Circuit::x, "qubit"_a, "condition"_a = nb::none())
        .def("y", &qupy::Circuit::y, "qubit"_a, "condition"_a = nb::none())
        .def("z", &qupy::Circuit::z, "qubit"_a, "condition"_a = nb::none())
        .def(
            "rx",
            &qupy::Circuit::rx,
            "angle"_a,
            "qubit"_a,
            "condition"_a = nb::none()
        )
        .def(
            "ry",
            &qupy::Circuit::ry,
            "angle"_a,
            "qubit"_a,
            "condition"_a = nb::none()
        )
        .def(
            "rz",
            &qupy::Circuit::rz,
            "angle"_a,
            "qubit"_a,
            "condition"_a = nb::none()
        )
        .def(
            "cx",
            &qupy::Circuit::cx,
            "control"_a,
            "target"_a,
            "condition"_a = nb::none()
        )
        .def(
            "cz",
            &qupy::Circuit::cz,
            "control"_a,
            "target"_a,
            "condition"_a = nb::none()
        )
        .def(
            "swap",
            &qupy::Circuit::swap,
            "first"_a,
            "second"_a,
            "condition"_a = nb::none()
        )
        .def(
            "measure",
            &qupy::Circuit::measure,
            "qubit"_a,
            "classical_bit"_a,
            "condition"_a = nb::none()
        )
        .def("reset", &qupy::Circuit::reset, "qubit"_a, "condition"_a = nb::none())
        .def("barrier", &qupy::Circuit::barrier, "qubits"_a = std::vector<std::size_t>{});

    module.def("circuit_ir_version", &qupy::circuit_ir_version);
    module.def(
        "_make_provider_program",
        [](std::string format, std::string text, std::size_t num_qubits, bool measures_all) {
            return qupy::ProviderProgram{
                std::move(format), std::move(text), num_qubits, measures_all
            };
        },
        "format"_a,
        "text"_a,
        "num_qubits"_a,
        "measures_all"_a
    );
}
