#include "qupy/compiler.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;

void bind_compiler(nb::module_& module) {
    nb::class_<qupy::Coupling>(module, "Coupling")
        .def(nb::init<std::size_t, std::size_t>(), "first"_a, "second"_a)
        .def_ro("first", &qupy::Coupling::first)
        .def_ro("second", &qupy::Coupling::second);

    nb::class_<qupy::OperationDuration>(module, "OperationDuration")
        .def(
            nb::init<qupy::CircuitOperationCode, double>(),
            "code"_a,
            "nanoseconds"_a
        )
        .def_ro("code", &qupy::OperationDuration::code)
        .def_ro("nanoseconds", &qupy::OperationDuration::nanoseconds);

    nb::class_<qupy::HardwareTarget>(module, "HardwareTarget")
        .def(
            nb::init<
                std::string,
                std::size_t,
                std::vector<qupy::CircuitOperationCode>,
                std::vector<qupy::CircuitOperationCode>,
                std::vector<qupy::Coupling>,
                bool,
                bool,
                bool,
                bool,
                std::vector<qupy::OperationDuration>
            >(),
            "name"_a,
            "num_qubits"_a,
            "one_qubit_operations"_a,
            "two_qubit_operations"_a,
            "couplings"_a = std::vector<qupy::Coupling>{},
            "measurement"_a = false,
            "mid_circuit_measurement"_a = false,
            "reset"_a = false,
            "dynamic_control"_a = false,
            "durations"_a = std::vector<qupy::OperationDuration>{}
        )
        .def_prop_ro("name", &qupy::HardwareTarget::name)
        .def_prop_ro("num_qubits", &qupy::HardwareTarget::num_qubits)
        .def_prop_ro(
            "one_qubit_operations",
            &qupy::HardwareTarget::one_qubit_operations
        )
        .def_prop_ro(
            "two_qubit_operations",
            &qupy::HardwareTarget::two_qubit_operations
        )
        .def_prop_ro("couplings", &qupy::HardwareTarget::couplings)
        .def_prop_ro("measurement", &qupy::HardwareTarget::measurement)
        .def_prop_ro(
            "mid_circuit_measurement",
            &qupy::HardwareTarget::mid_circuit_measurement
        )
        .def_prop_ro("reset", &qupy::HardwareTarget::reset)
        .def_prop_ro("dynamic_control", &qupy::HardwareTarget::dynamic_control)
        .def_prop_ro("durations", &qupy::HardwareTarget::durations)
        .def("adjacent", &qupy::HardwareTarget::adjacent, "first"_a, "second"_a)
        .def("supports", &qupy::HardwareTarget::supports, "code"_a, "qubits"_a)
        .def("duration_ns", &qupy::HardwareTarget::duration_ns, "code"_a)
        .def_prop_ro("canonical_text", &qupy::HardwareTarget::canonical_text)
        .def_prop_ro("fingerprint", &qupy::HardwareTarget::fingerprint);

    nb::class_<qupy::ScheduledInstruction>(module, "ScheduledInstruction")
        .def_ro("instruction_index", &qupy::ScheduledInstruction::instruction_index)
        .def_ro("start_ns", &qupy::ScheduledInstruction::start_ns)
        .def_ro("duration_ns", &qupy::ScheduledInstruction::duration_ns);

    nb::class_<qupy::CompilationResult>(module, "CompilationResult")
        .def_ro("circuit", &qupy::CompilationResult::circuit)
        .def_ro("initial_layout", &qupy::CompilationResult::initial_layout)
        .def_ro("final_layout", &qupy::CompilationResult::final_layout)
        .def_ro("original_operations", &qupy::CompilationResult::original_operations)
        .def_ro("optimized_operations", &qupy::CompilationResult::optimized_operations)
        .def_ro("routed_operations", &qupy::CompilationResult::routed_operations)
        .def_ro("compiled_operations", &qupy::CompilationResult::compiled_operations)
        .def_ro("inserted_swaps", &qupy::CompilationResult::inserted_swaps)
        .def_ro("decompositions", &qupy::CompilationResult::decompositions)
        .def_ro("depth", &qupy::CompilationResult::depth)
        .def_ro("duration_ns", &qupy::CompilationResult::duration_ns)
        .def_ro("schedule", &qupy::CompilationResult::schedule)
        .def_ro("target_fingerprint", &qupy::CompilationResult::target_fingerprint);

    module.def(
        "compile_circuit",
        &qupy::compile_circuit,
        "circuit"_a,
        "target"_a,
        "initial_layout"_a = std::vector<std::size_t>{},
        "optimization_level"_a = 1U
    );
}
