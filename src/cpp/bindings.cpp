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
using DoubleArray = nb::ndarray<
    nb::numpy,
    const double,
    nb::ndim<1>,
    nb::c_contig
>;
using ParameterArray = nb::ndarray<
    nb::numpy,
    const double,
    nb::ndim<2>,
    nb::c_contig
>;
using SampleArray = nb::ndarray<
    nb::numpy,
    const std::int8_t,
    nb::ndim<2>,
    nb::c_contig
>;
using SampleBatchArray = nb::ndarray<
    nb::numpy,
    const std::int8_t,
    nb::ndim<3>,
    nb::c_contig
>;

ComplexArray state_values(qupy::StateVector& result) {
    return ComplexArray(
        result.values.data(),
        {result.values.size()},
        nb::find(&result)
    );
}

DoubleArray probability_values(qupy::Probabilities& result) {
    return DoubleArray(
        result.values.data(),
        {result.values.size()},
        nb::find(&result)
    );
}

DoubleArray expectation_batch_values(qupy::ExpectationBatch& result) {
    return DoubleArray(
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

SampleBatchArray sample_batch_values(qupy::SamplesBatch& result) {
    return SampleBatchArray(
        result.values.data(),
        {result.batch_size, result.shots, result.num_qubits},
        nb::find(&result)
    );
}

std::vector<double> parameter_table_values(
    ParameterArray parameter_values,
    std::size_t parameter_count
) {
    if (parameter_values.shape(1) != parameter_count) {
        throw nb::value_error("parameter table columns must match parameter slots");
    }
    std::vector<double> flattened;
    if (parameter_values.size() != 0U) {
        flattened.assign(
            parameter_values.data(),
            parameter_values.data() + parameter_values.size()
        );
    }
    return flattened;
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
        .value("PROBABILITIES", qupy::ResultMode::Probabilities)
        .value("VARIANCE", qupy::ResultMode::Variance)
        .value("STATEVECTOR", qupy::ResultMode::StateVector);

    nb::class_<qupy::Operation>(module, "Operation")
        .def_prop_ro("name", &qupy::Operation::name)
        .def_ro("code", &qupy::Operation::code)
        .def_ro("qubits", &qupy::Operation::qubits)
        .def_ro("parameters", &qupy::Operation::parameters);

    nb::class_<qupy::ParameterSlot>(module, "ParameterSlot")
        .def(
            nb::init<std::size_t, std::size_t>(),
            "operation_index"_a,
            "parameter_index"_a = 0U
        )
        .def_ro("operation_index", &qupy::ParameterSlot::operation_index)
        .def_ro("parameter_index", &qupy::ParameterSlot::parameter_index);

    nb::class_<qupy::Program>(module, "Program")
        .def(nb::init<std::size_t>(), "num_qubits"_a)
        .def_prop_ro("num_qubits", &qupy::Program::num_qubits)
        .def_prop_ro(
            "operations",
            [](const qupy::Program& program) { return program.operations(); }
        )
        .def_prop_ro("canonical_text", &qupy::Program::canonical_text)
        .def_prop_ro("fingerprint", &qupy::Program::fingerprint)
        .def("bind", &qupy::Program::bound, "slots"_a, "values"_a);

    nb::class_<qupy::PauliZ>(module, "PauliZ")
        .def_ro("qubit", &qupy::PauliZ::qubit);

    nb::class_<qupy::Target>(module, "Target")
        .def_ro("name", &qupy::Target::name)
        .def_ro("operations", &qupy::Target::operations)
        .def_ro("result_modes", &qupy::Target::result_modes)
        .def_ro("max_qubits", &qupy::Target::max_qubits)
        .def_ro("simulator", &qupy::Target::simulator)
        .def_ro("state_access", &qupy::Target::state_access)
        .def_ro("mid_circuit_measurement", &qupy::Target::mid_circuit_measurement)
        .def_ro("reset", &qupy::Target::reset)
        .def_ro("dynamic_control", &qupy::Target::dynamic_control)
        .def_ro("parameter_batches", &qupy::Target::parameter_batches)
        .def_prop_ro("fingerprint", &qupy::Target::fingerprint)
        .def("supports_operation", nb::overload_cast<qupy::OperationCode>(&qupy::Target::supports, nb::const_))
        .def("supports_result", nb::overload_cast<qupy::ResultMode>(&qupy::Target::supports, nb::const_));

    nb::class_<qupy::PlannerCostModel>(module, "PlannerCostModel")
        .def_prop_ro("schema_version", &qupy::PlannerCostModel::schema_version)
        .def_prop_ro("workload_version", &qupy::PlannerCostModel::workload_version)
        .def_prop_ro("engine_version", &qupy::PlannerCostModel::engine_version)
        .def_prop_ro("host_fingerprint", &qupy::PlannerCostModel::host_fingerprint)
        .def_prop_ro("cuda_host_fingerprint", &qupy::PlannerCostModel::cuda_host_fingerprint)
        .def_prop_ro("artifact_fingerprint", &qupy::PlannerCostModel::artifact_fingerprint)
        .def_prop_ro("cost_classes", &qupy::PlannerCostModel::cost_classes)
        .def_prop_ro("cuda_auto_validated", &qupy::PlannerCostModel::cuda_auto_validated)
        .def("predict_ns", &qupy::PlannerCostModel::predict_ns, "plan"_a);

    nb::class_<qupy::ExecutionPlan>(module, "ExecutionPlan")
        .def_ro("backend", &qupy::ExecutionPlan::backend)
        .def_ro("method", &qupy::ExecutionPlan::method)
        .def_ro("exact", &qupy::ExecutionPlan::exact)
        .def_ro("threads", &qupy::ExecutionPlan::threads)
        .def_ro("original_qubits", &qupy::ExecutionPlan::original_qubits)
        .def_ro("original_operations", &qupy::ExecutionPlan::original_operations)
        .def_ro("active_qubits", &qupy::ExecutionPlan::active_qubits)
        .def_ro("active_operations", &qupy::ExecutionPlan::active_operations)
        .def_ro("single_qubit_operations", &qupy::ExecutionPlan::single_qubit_operations)
        .def_ro("two_qubit_operations", &qupy::ExecutionPlan::two_qubit_operations)
        .def_ro("parameterized_operations", &qupy::ExecutionPlan::parameterized_operations)
        .def_ro("non_clifford_operations", &qupy::ExecutionPlan::non_clifford_operations)
        .def_ro("compiled_steps", &qupy::ExecutionPlan::compiled_steps)
        .def_ro("estimated_state_bytes", &qupy::ExecutionPlan::estimated_state_bytes)
        .def_ro("result_mode", &qupy::ExecutionPlan::result_mode)
        .def_ro("workload_version", &qupy::ExecutionPlan::workload_version)
        .def_ro("workload_fingerprint", &qupy::ExecutionPlan::workload_fingerprint)
        .def_ro("program_fingerprint", &qupy::ExecutionPlan::program_fingerprint)
        .def_ro("target_fingerprint", &qupy::ExecutionPlan::target_fingerprint)
        .def_ro("cache_key", &qupy::ExecutionPlan::cache_key)
        .def_ro("predicted_ns", &qupy::ExecutionPlan::predicted_ns)
        .def_ro("cost_model_class", &qupy::ExecutionPlan::cost_model_class)
        .def_ro("cost_model_fingerprint", &qupy::ExecutionPlan::cost_model_fingerprint)
        .def_ro("cost_model_host_fingerprint", &qupy::ExecutionPlan::cost_model_host_fingerprint)
        .def_ro("cost_model_cuda_host_fingerprint", &qupy::ExecutionPlan::cost_model_cuda_host_fingerprint)
        .def_ro("tensor_network_max_bond", &qupy::ExecutionPlan::tensor_network_max_bond)
        .def_ro("tensor_network_routed_swaps", &qupy::ExecutionPlan::tensor_network_routed_swaps)
        .def_ro(
            "tensor_network_contraction_work",
            &qupy::ExecutionPlan::tensor_network_contraction_work
        );

    nb::class_<qupy::StateVector>(module, "StateVector")
        .def_prop_ro("values", &state_values)
        .def_ro("backend", &qupy::StateVector::backend);

    nb::class_<qupy::Probabilities>(module, "Probabilities")
        .def_prop_ro("values", &probability_values)
        .def_ro("backend", &qupy::Probabilities::backend);

    nb::class_<qupy::Samples>(module, "Samples")
        .def_prop_ro("values", &sample_values)
        .def_ro("shots", &qupy::Samples::shots)
        .def_ro("backend", &qupy::Samples::backend)
        .def("counts", &qupy::Samples::counts);

    nb::class_<qupy::SamplesBatch>(module, "SamplesBatch")
        .def_prop_ro("values", &sample_batch_values)
        .def_ro("batch_size", &qupy::SamplesBatch::batch_size)
        .def_ro("shots", &qupy::SamplesBatch::shots)
        .def_ro("num_qubits", &qupy::SamplesBatch::num_qubits)
        .def_ro("parameter_count", &qupy::SamplesBatch::parameter_count)
        .def_ro("backend", &qupy::SamplesBatch::backend)
        .def_ro("compiled_steps", &qupy::SamplesBatch::compiled_steps)
        .def_ro("estimated_state_bytes", &qupy::SamplesBatch::estimated_state_bytes)
        .def("counts", &qupy::SamplesBatch::counts, "batch_index"_a);

    nb::class_<qupy::Expectation>(module, "Expectation")
        .def_ro("value", &qupy::Expectation::value)
        .def_ro("backend", &qupy::Expectation::backend)
        .def_ro("active_qubits", &qupy::Expectation::active_qubits)
        .def_ro("compiled_steps", &qupy::Expectation::compiled_steps)
        .def_ro("estimated_state_bytes", &qupy::Expectation::estimated_state_bytes);

    nb::class_<qupy::ExpectationBatch>(module, "ExpectationBatch")
        .def_prop_ro("values", &expectation_batch_values)
        .def_ro("batch_size", &qupy::ExpectationBatch::batch_size)
        .def_ro("parameter_count", &qupy::ExpectationBatch::parameter_count)
        .def_ro("backend", &qupy::ExpectationBatch::backend)
        .def_ro("active_qubits", &qupy::ExpectationBatch::active_qubits)
        .def_ro("compiled_steps", &qupy::ExpectationBatch::compiled_steps)
        .def_ro("estimated_state_bytes", &qupy::ExpectationBatch::estimated_state_bytes);

    nb::class_<qupy::Variance>(module, "Variance")
        .def_ro("value", &qupy::Variance::value)
        .def_ro("backend", &qupy::Variance::backend)
        .def_ro("active_qubits", &qupy::Variance::active_qubits)
        .def_ro("compiled_steps", &qupy::Variance::compiled_steps)
        .def_ro("estimated_state_bytes", &qupy::Variance::estimated_state_bytes);

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
    module.def("cuda_available", &qupy::cuda_available);
    module.def("cuda_unavailable_reason", &qupy::cuda_unavailable_reason);
    module.def("cuda_device_name", &qupy::cuda_device_name);
    module.def("cuda_target", &qupy::cuda_target);
    module.def("mps_target", &qupy::mps_target);
    module.def("planner_host_fingerprint", &qupy::planner_host_fingerprint);
    module.def("planner_cuda_host_fingerprint", &qupy::planner_cuda_host_fingerprint);
    module.def("load_planner_cost_model", &qupy::load_planner_cost_model, "path"_a);
    module.def(
        "plan",
        [](const qupy::Program& program, qupy::ResultMode result_mode,
           const std::string& backend, nb::object cost_model) {
            const auto* model = cost_model.is_none()
                ? nullptr : &nb::cast<const qupy::PlannerCostModel&>(cost_model);
            return qupy::plan(program, result_mode, backend, model);
        },
        "program"_a, "result_mode"_a, "backend"_a = "auto", "cost_model"_a = nb::none()
    );
    module.def(
        "expectation_plan",
        [](const qupy::Program& program, qupy::PauliZ observable,
           const std::string& backend, nb::object cost_model) {
            const auto* model = cost_model.is_none()
                ? nullptr : &nb::cast<const qupy::PlannerCostModel&>(cost_model);
            return qupy::expectation_plan(program, observable, backend, model);
        },
        "program"_a, "observable"_a, "backend"_a = "auto", "cost_model"_a = nb::none()
    );
    module.def(
        "variance_plan",
        [](const qupy::Program& program, qupy::PauliZ observable,
           const std::string& backend, nb::object cost_model) {
            const auto* model = cost_model.is_none()
                ? nullptr : &nb::cast<const qupy::PlannerCostModel&>(cost_model);
            return qupy::variance_plan(program, observable, backend, model);
        },
        "program"_a, "observable"_a, "backend"_a = "auto", "cost_model"_a = nb::none()
    );
    module.def(
        "statevector",
        [](const qupy::Program& program, const std::string& backend, nb::object cost_model) {
            const auto* model = cost_model.is_none()
                ? nullptr : &nb::cast<const qupy::PlannerCostModel&>(cost_model);
            nb::gil_scoped_release release;
            return qupy::statevector(program, backend, model);
        },
        "program"_a, "backend"_a = "auto", "cost_model"_a = nb::none()
    );
    module.def(
        "probabilities",
        &qupy::probabilities,
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
        "sample_batch",
        [](
            const qupy::Program& program,
            const std::vector<qupy::ParameterSlot>& slots,
            ParameterArray parameter_values,
            std::size_t shots,
            std::optional<std::uint64_t> seed,
            const std::string& backend
        ) {
            const std::size_t rows = parameter_values.shape(0);
            std::vector<double> flattened = parameter_table_values(
                parameter_values, slots.size()
            );
            nb::gil_scoped_release release;
            return qupy::sample_batch(
                program, slots, flattened, rows, shots, seed, backend
            );
        },
        "program"_a,
        "slots"_a,
        "parameter_values"_a,
        "shots"_a = 1024,
        "seed"_a = std::nullopt,
        "backend"_a = "auto"
    );
    module.def(
        "expect",
        &qupy::expectation,
        "program"_a,
        "observable"_a,
        "backend"_a = "auto",
        nb::call_guard<nb::gil_scoped_release>()
    );
    module.def(
        "expect_batch",
        [](
            const qupy::Program& program,
            qupy::PauliZ observable,
            const std::vector<qupy::ParameterSlot>& slots,
            ParameterArray parameter_values,
            const std::string& backend
        ) {
            const std::size_t rows = parameter_values.shape(0);
            std::vector<double> flattened = parameter_table_values(
                parameter_values, slots.size()
            );
            nb::gil_scoped_release release;
            return qupy::expectation_batch(
                program, observable, slots, flattened, rows, backend
            );
        },
        "program"_a,
        "observable"_a,
        "slots"_a,
        "parameter_values"_a,
        "backend"_a = "auto"
    );
    module.def(
        "variance",
        &qupy::variance,
        "program"_a,
        "observable"_a,
        "backend"_a = "auto",
        nb::call_guard<nb::gil_scoped_release>()
    );

    module.def("core_language", &qupy::core_language);
    module.def("core_version", &qupy::core_version);
    module.def("ir_version", &qupy::ir_version);
    module.def("parallel_threads", &qupy::parallel_threads);
}
