#include "qupy/tensor_network.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;

void bind_multi_device(nb::module_& module);

void bind_tensor_network(nb::module_& module) {
    nb::class_<qupy::TensorNetworkCostModel>(module, "TensorNetworkCostModel")
        .def_prop_ro("schema_version", &qupy::TensorNetworkCostModel::schema_version)
        .def_prop_ro("policy_version", &qupy::TensorNetworkCostModel::policy_version)
        .def_prop_ro("workload_version", &qupy::TensorNetworkCostModel::workload_version)
        .def_prop_ro("engine_version", &qupy::TensorNetworkCostModel::engine_version)
        .def_prop_ro("host_fingerprint", &qupy::TensorNetworkCostModel::host_fingerprint)
        .def_prop_ro("artifact_fingerprint", &qupy::TensorNetworkCostModel::artifact_fingerprint)
        .def_prop_ro("report_count", &qupy::TensorNetworkCostModel::report_count)
        .def_prop_ro("decision_samples", &qupy::TensorNetworkCostModel::decision_samples)
        .def_prop_ro("decision_mistakes", &qupy::TensorNetworkCostModel::decision_mistakes)
        .def_prop_ro("decision_max_regret", &qupy::TensorNetworkCostModel::decision_max_regret)
        .def_prop_ro("cpu_wins", &qupy::TensorNetworkCostModel::cpu_wins)
        .def_prop_ro("tensor_network_wins", &qupy::TensorNetworkCostModel::tensor_network_wins)
        .def_prop_ro("auto_validated", &qupy::TensorNetworkCostModel::auto_validated)
        .def(
            "predict_cpu_ns",
            &qupy::TensorNetworkCostModel::predict_cpu_ns,
            "active_qubits"_a,
            "compiled_steps"_a,
            "two_qubit_operations"_a,
            "operation_count"_a,
            "term_count"_a,
            "threads"_a
        )
        .def(
            "predict_tensor_network_ns",
            &qupy::TensorNetworkCostModel::predict_tensor_network_ns,
            "contractions"_a,
            "peak_tensor_rank"_a,
            "peak_tensor_bytes"_a,
            "scalar_multiplications"_a,
            "term_count"_a
        );

    nb::class_<qupy::TensorNetworkPlan>(module, "TensorNetworkPlan")
        .def_ro("term_count", &qupy::TensorNetworkPlan::term_count)
        .def_ro("contractions", &qupy::TensorNetworkPlan::contractions)
        .def_ro("peak_tensor_rank", &qupy::TensorNetworkPlan::peak_tensor_rank)
        .def_ro("peak_tensor_bytes", &qupy::TensorNetworkPlan::peak_tensor_bytes)
        .def_ro("scalar_multiplications", &qupy::TensorNetworkPlan::scalar_multiplications)
        .def_ro("max_tensor_bytes", &qupy::TensorNetworkPlan::max_tensor_bytes)
        .def_ro("exact", &qupy::TensorNetworkPlan::exact)
        .def_ro("backend", &qupy::TensorNetworkPlan::backend)
        .def_ro("method", &qupy::TensorNetworkPlan::method)
        .def_ro("program_fingerprint", &qupy::TensorNetworkPlan::program_fingerprint)
        .def_ro("observable_fingerprint", &qupy::TensorNetworkPlan::observable_fingerprint)
        .def_ro("plan_fingerprint", &qupy::TensorNetworkPlan::plan_fingerprint);

    nb::class_<qupy::TensorNetworkResult>(module, "TensorNetworkResult")
        .def_ro("value", &qupy::TensorNetworkResult::value)
        .def_ro("term_count", &qupy::TensorNetworkResult::term_count)
        .def_ro("contractions", &qupy::TensorNetworkResult::contractions)
        .def_ro("peak_tensor_rank", &qupy::TensorNetworkResult::peak_tensor_rank)
        .def_ro("peak_tensor_bytes", &qupy::TensorNetworkResult::peak_tensor_bytes)
        .def_ro("scalar_multiplications", &qupy::TensorNetworkResult::scalar_multiplications)
        .def_ro("exact", &qupy::TensorNetworkResult::exact)
        .def_ro("backend", &qupy::TensorNetworkResult::backend)
        .def_ro("method", &qupy::TensorNetworkResult::method);

    module.def(
        "load_tensor_network_cost_model",
        &qupy::load_tensor_network_cost_model,
        "path"_a,
        nb::call_guard<nb::gil_scoped_release>()
    );
    module.def(
        "tensor_network_plan",
        &qupy::tensor_network_plan,
        "program"_a,
        "observable"_a,
        "max_tensor_bytes"_a = qupy::kTensorNetworkDefaultMaxBytes,
        nb::call_guard<nb::gil_scoped_release>()
    );
    module.def(
        "tensor_network_expectation",
        &qupy::tensor_network_expectation,
        "program"_a,
        "observable"_a,
        "max_tensor_bytes"_a = qupy::kTensorNetworkDefaultMaxBytes,
        nb::call_guard<nb::gil_scoped_release>()
    );
    module.def(
        "tensor_network_observable_plan",
        &qupy::tensor_network_observable_plan,
        "program"_a,
        "observables"_a,
        "max_tensor_bytes"_a = qupy::kTensorNetworkDefaultMaxBytes,
        nb::call_guard<nb::gil_scoped_release>()
    );
    module.def(
        "tensor_network_auto_observable_plan",
        &qupy::tensor_network_auto_observable_plan,
        "program"_a,
        "observables"_a,
        "cost_model"_a,
        "max_tensor_bytes"_a = qupy::kTensorNetworkDefaultMaxBytes,
        nb::call_guard<nb::gil_scoped_release>()
    );
    module.def(
        "tensor_network_expect_observable",
        &qupy::tensor_network_expect_observable,
        "program"_a,
        "observable"_a,
        "max_tensor_bytes"_a = qupy::kTensorNetworkDefaultMaxBytes,
        nb::call_guard<nb::gil_scoped_release>()
    );
    module.def(
        "tensor_network_expect_observables",
        &qupy::tensor_network_expect_observables,
        "program"_a,
        "observables"_a,
        "max_tensor_bytes"_a = qupy::kTensorNetworkDefaultMaxBytes,
        nb::call_guard<nb::gil_scoped_release>()
    );

    bind_multi_device(module);
}