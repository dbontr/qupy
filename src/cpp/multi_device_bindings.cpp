#include "qupy/multi_device.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using DoubleArray = nb::ndarray<
    nb::numpy,
    const double,
    nb::ndim<1>,
    nb::c_contig
>;

DoubleArray distributed_trajectory_values(qupy::DistributedTrajectoryBatch& result) {
    return DoubleArray(
        result.values.data(),
        {result.values.size()},
        nb::find(&result)
    );
}

DoubleArray distributed_trajectory_standard_errors(
    qupy::DistributedTrajectoryBatch& result
) {
    return DoubleArray(
        result.standard_errors.data(),
        {result.standard_errors.size()},
        nb::find(&result)
    );
}

}  // namespace

void bind_multi_device(nb::module_& module) {
    nb::class_<qupy::DistributedTensorNetworkResult>(module, "DistributedTensorNetworkResult")
        .def_ro("value", &qupy::DistributedTensorNetworkResult::value)
        .def_ro("term_count", &qupy::DistributedTensorNetworkResult::term_count)
        .def_ro("contractions", &qupy::DistributedTensorNetworkResult::contractions)
        .def_ro("peak_tensor_rank", &qupy::DistributedTensorNetworkResult::peak_tensor_rank)
        .def_ro("peak_tensor_bytes", &qupy::DistributedTensorNetworkResult::peak_tensor_bytes)
        .def_ro(
            "scalar_multiplications",
            &qupy::DistributedTensorNetworkResult::scalar_multiplications
        )
        .def_ro("world_size", &qupy::DistributedTensorNetworkResult::world_size)
        .def_ro("active_ranks", &qupy::DistributedTensorNetworkResult::active_ranks)
        .def_ro("exact", &qupy::DistributedTensorNetworkResult::exact)
        .def_ro("backend", &qupy::DistributedTensorNetworkResult::backend)
        .def_ro("method", &qupy::DistributedTensorNetworkResult::method);

    nb::class_<qupy::DistributedTrajectoryBatch>(module, "DistributedTrajectoryBatch")
        .def_prop_ro("values", &distributed_trajectory_values)
        .def_prop_ro("standard_errors", &distributed_trajectory_standard_errors)
        .def_ro("observable_count", &qupy::DistributedTrajectoryBatch::observable_count)
        .def_ro("trajectories", &qupy::DistributedTrajectoryBatch::trajectories)
        .def_ro("seed", &qupy::DistributedTrajectoryBatch::seed)
        .def_ro("state_bytes_per_rank", &qupy::DistributedTrajectoryBatch::state_bytes_per_rank)
        .def_ro("world_size", &qupy::DistributedTrajectoryBatch::world_size)
        .def_ro("active_ranks", &qupy::DistributedTrajectoryBatch::active_ranks)
        .def_ro("exact", &qupy::DistributedTrajectoryBatch::exact)
        .def_ro("backend", &qupy::DistributedTrajectoryBatch::backend)
        .def_ro("method", &qupy::DistributedTrajectoryBatch::method);

    module.def(
        "distributed_tensor_network_expectation",
        &qupy::distributed_tensor_network_expectation,
        "program"_a,
        "observable"_a,
        "max_tensor_bytes"_a = std::size_t{1} << 30U,
        nb::call_guard<nb::gil_scoped_release>()
    );
    module.def(
        "distributed_trajectory_expectations",
        &qupy::distributed_trajectory_expectations,
        "program"_a,
        "observables"_a,
        "trajectories"_a,
        "seed"_a = std::nullopt,
        nb::call_guard<nb::gil_scoped_release>()
    );
}
