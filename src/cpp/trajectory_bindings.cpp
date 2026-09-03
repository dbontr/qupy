#include "qupy/trajectory.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;

void bind_tensor_network(nb::module_& module);

namespace {

using DoubleArray = nb::ndarray<
    nb::numpy,
    const double,
    nb::ndim<1>,
    nb::c_contig
>;

DoubleArray trajectory_values(qupy::TrajectoryBatch& result) {
    return DoubleArray(
        result.values.data(),
        {result.values.size()},
        nb::find(&result)
    );
}

DoubleArray trajectory_standard_errors(qupy::TrajectoryBatch& result) {
    return DoubleArray(
        result.standard_errors.data(),
        {result.standard_errors.size()},
        nb::find(&result)
    );
}

}  // namespace

void bind_trajectory(nb::module_& module) {
    nb::class_<qupy::TrajectoryBatch>(module, "TrajectoryBatch")
        .def_prop_ro("values", &trajectory_values)
        .def_prop_ro("standard_errors", &trajectory_standard_errors)
        .def_ro("observable_count", &qupy::TrajectoryBatch::observable_count)
        .def_ro("trajectories", &qupy::TrajectoryBatch::trajectories)
        .def_ro("seed", &qupy::TrajectoryBatch::seed)
        .def_ro("state_bytes", &qupy::TrajectoryBatch::state_bytes)
        .def_ro("exact", &qupy::TrajectoryBatch::exact)
        .def_ro("backend", &qupy::TrajectoryBatch::backend)
        .def_ro("method", &qupy::TrajectoryBatch::method);

    module.def(
        "trajectory_expectations",
        &qupy::trajectory_expectations,
        "program"_a,
        "observables"_a,
        "trajectories"_a,
        "seed"_a = std::nullopt,
        "backend"_a = "auto",
        nb::call_guard<nb::gil_scoped_release>()
    );

    bind_tensor_network(module);
}
