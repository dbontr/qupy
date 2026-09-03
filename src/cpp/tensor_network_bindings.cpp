#include "qupy/tensor_network.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;

void bind_multi_device(nb::module_& module);

void bind_tensor_network(nb::module_& module) {
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
    module.def(
        "tensor_network_value_and_grad",
        &qupy::tensor_network_value_and_grad,
        "program"_a,
        "observable"_a,
        "slots"_a,
        "parameter_values"_a,
        "method"_a = qupy::GradientMethod::Auto,
        "epsilon"_a = 1e-7,
        "max_tensor_bytes"_a = qupy::kTensorNetworkDefaultMaxBytes,
        nb::call_guard<nb::gil_scoped_release>()
    );
    module.def(
        "tensor_network_jacobian",
        &qupy::tensor_network_jacobian,
        "program"_a,
        "observables"_a,
        "slots"_a,
        "parameter_values"_a,
        "method"_a = qupy::GradientMethod::Auto,
        "epsilon"_a = 1e-7,
        "max_tensor_bytes"_a = qupy::kTensorNetworkDefaultMaxBytes,
        nb::call_guard<nb::gil_scoped_release>()
    );
    module.def(
        "tensor_network_hessian",
        &qupy::tensor_network_hessian,
        "program"_a,
        "observable"_a,
        "slots"_a,
        "parameter_values"_a,
        "max_tensor_bytes"_a = qupy::kTensorNetworkDefaultMaxBytes,
        nb::call_guard<nb::gil_scoped_release>()
    );

    bind_multi_device(module);
}
