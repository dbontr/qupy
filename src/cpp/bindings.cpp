#include "qupy/core.hpp"

#include <cstdint>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using ComplexArray = nb::ndarray<
    nb::numpy,
    const qupy::Complex,
    nb::ndim<1>,
    nb::c_contig
>;
using SampleArray = nb::ndarray<
    nb::numpy,
    const std::int8_t,
    nb::ndim<2>,
    nb::c_contig
>;

ComplexArray state_values(qupy::StateVector& result) {
    return ComplexArray(
        result.values.data(),
        {result.values.size()},
        nb::find(&result)
    );
}

SampleArray sample_values(qupy::Samples& result) {
    return SampleArray(
        result.values.data(),
        {result.shots, result.num_qubits},
        nb::find(&result)
    );
}

}  // namespace

NB_MODULE(_native, module) {
    module.doc() = "QuPy native C++20 quantum runtime";

    nb::enum_<qupy::OperationCode>(module, "OperationCode")
        .value("H", qupy::OperationCode::H)
        .value("X", qupy::OperationCode::X)
        .value("Y", qupy::OperationCode::Y)
        .value("Z", qupy::OperationCode::Z)
        .value("RX", qupy::OperationCode::RX)
        .value("RY", qupy::OperationCode::RY)
        .value("RZ", qupy::OperationCode::RZ)
        .value("CX", qupy::OperationCode::CX)
        .value("CZ", qupy::OperationCode::CZ)
        .value("SWAP", qupy::OperationCode::SWAP);

    nb::enum_<qupy::ResultMode>(module, "ResultMode")
        .value("SAMPLE", qupy::ResultMode::Sample)
        .value("EXPECTATION", qupy::ResultMode::Expectation)
        .value("STATEVECTOR", qupy::ResultMode::StateVector);

    nb::class_<qupy::Operation>(module, "Operation")
        .def_prop_ro("name", &qupy::Operation::name)
        .def_ro("code", &qupy::Operation::code)
        .def_ro("qubits", &qupy::Operation::qubits)
        .def_ro("parameters", &qupy::Operation::parameters);

    nb::class_<qupy::Program>(module, "Program")
        .def(nb::init<std::size_t>(), "num_qubits"_a)
        .def_prop_ro("num_qubits", &qupy::Program::num_qubits)
        .def_prop_ro(
            "operations",
            [](const qupy::Program& program) { return program.operations(); }
        );

    nb::class_<qupy::PauliZ>(module, "PauliZ")
        .def_ro("qubit", &qupy::PauliZ::qubit);

    nb::class_<qupy::Target>(module, "Target")
        .def_ro("name", &qupy::Target::name)
        .def_ro("operations", &qupy::Target::operations)
        .def_ro("result_modes", &qupy::Target::result_modes)
        .def_ro("max_qubits", &qupy::Target::max_qubits)
        .def_ro("simulator", &qupy::Target::simulator)
        .def("supports_operation", nb::overload_cast<qupy::OperationCode>(&qupy::Target::supports, nb::const_))
        .def("supports_result", nb::overload_cast<qupy::ResultMode>(&qupy::Target::supports, nb::const_));

    nb::class_<qupy::ExecutionPlan>(module, "ExecutionPlan")
        .def_ro("backend", &qupy::ExecutionPlan::backend)
        .def_ro("method", &qupy::ExecutionPlan::method)
        .def_ro("exact", &qupy::ExecutionPlan::exact)
        .def_ro("threads", &qupy::ExecutionPlan::threads);

    nb::class_<qupy::StateVector>(module, "StateVector")
        .def_prop_ro("values", &state_values)
        .def_ro("backend", &qupy::StateVector::backend);

    nb::class_<qupy::Samples>(module, "Samples")
        .def_prop_ro("values", &sample_values)
        .def_ro("shots", &qupy::Samples::shots)
        .def_ro("backend", &qupy::Samples::backend)
        .def("counts", &qupy::Samples::counts);

    nb::class_<qupy::Expectation>(module, "Expectation")
        .def_ro("value", &qupy::Expectation::value)
        .def_ro("backend", &qupy::Expectation::backend);

    module.def("h", &qupy::h, "program"_a, "qubit"_a);
    module.def("x", &qupy::x, "program"_a, "qubit"_a);
    module.def("y", &qupy::y, "program"_a, "qubit"_a);
    module.def("z", &qupy::z, "program"_a, "qubit"_a);
    module.def("rx", &qupy::rx, "program"_a, "angle"_a, "qubit"_a);
    module.def("ry", &qupy::ry, "program"_a, "angle"_a, "qubit"_a);
    module.def("rz", &qupy::rz, "program"_a, "angle"_a, "qubit"_a);
    module.def("cx", &qupy::cx, "program"_a, "control"_a, "target"_a);
    module.def("cz", &qupy::cz, "program"_a, "control"_a, "target"_a);
    module.def("swap", &qupy::swap, "program"_a, "first"_a, "second"_a);
    module.def("Z", &qupy::pauli_z, "qubit"_a);

    module.def("native_target", &qupy::native_target);
    module.def(
        "plan",
        &qupy::plan,
        "program"_a,
        "result_mode"_a,
        "backend"_a = "auto"
    );
    module.def(
        "statevector",
        &qupy::statevector,
        "program"_a,
        "backend"_a = "auto",
        nb::call_guard<nb::gil_scoped_release>()
    );
    module.def(
        "sample",
        &qupy::sample,
        "program"_a,
        "shots"_a = 1024,
        "seed"_a = std::nullopt,
        "backend"_a = "auto",
        nb::call_guard<nb::gil_scoped_release>()
    );
    module.def(
        "expect",
        &qupy::expectation,
        "program"_a,
        "observable"_a,
        "backend"_a = "auto",
        nb::call_guard<nb::gil_scoped_release>()
    );

    module.def("core_language", &qupy::core_language);
    module.def("parallel_threads", &qupy::parallel_threads);
}
