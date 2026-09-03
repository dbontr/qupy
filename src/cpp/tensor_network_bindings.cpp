#include "qupy/tensor_network.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;
using namespace nb::literals;

void bind_tensor_network(nb::module_& module) {
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
        "tensor_network_expectation",
        &qupy::tensor_network_expectation,
        "program"_a,
        "observable"_a,
        "max_tensor_bytes"_a = std::size_t{1} << 30U,
        nb::call_guard<nb::gil_scoped_release>()
    );
}
