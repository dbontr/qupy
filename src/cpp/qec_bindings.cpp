#include "qupy/qec.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;

void bind_tensor_network(nb::module_& module);

namespace {

using Int8Vector = nb::ndarray<
    nb::numpy,
    const std::int8_t,
    nb::ndim<1>,
    nb::c_contig
>;
using Int8Matrix = nb::ndarray<
    nb::numpy,
    const std::int8_t,
    nb::ndim<2>,
    nb::c_contig
>;
using Int8OutputVector = nb::ndarray<
    nb::numpy,
    const std::int8_t,
    nb::ndim<1>,
    nb::c_contig
>;
using Int8OutputMatrix = nb::ndarray<
    nb::numpy,
    const std::int8_t,
    nb::ndim<2>,
    nb::c_contig
>;
using DoubleOutputVector = nb::ndarray<
    nb::numpy,
    const double,
    nb::ndim<1>,
    nb::c_contig
>;
using UInt64OutputVector = nb::ndarray<
    nb::numpy,
    const std::uint64_t,
    nb::ndim<1>,
    nb::c_contig
>;

std::vector<std::int8_t> copy_vector(Int8Vector values) {
    return values.size() == 0U
        ? std::vector<std::int8_t>{}
        : std::vector<std::int8_t>(values.data(), values.data() + values.size());
}

std::vector<std::int8_t> copy_matrix(Int8Matrix values) {
    return values.size() == 0U
        ? std::vector<std::int8_t>{}
        : std::vector<std::int8_t>(values.data(), values.data() + values.size());
}

Int8OutputVector correction_values(qupy::BpOsdDecodeResult& result) {
    return Int8OutputVector(
        result.correction.data(),
        {result.correction.size()},
        nb::find(&result)
    );
}

Int8OutputVector observable_values(qupy::BpOsdDecodeResult& result) {
    return Int8OutputVector(
        result.observables.data(),
        {result.observables.size()},
        nb::find(&result)
    );
}

Int8OutputMatrix batch_corrections(qupy::BpOsdDecodeBatch& result) {
    return Int8OutputMatrix(
        result.corrections.data(),
        {result.shots, result.error_count},
        nb::find(&result)
    );
}

Int8OutputMatrix batch_observables(qupy::BpOsdDecodeBatch& result) {
    return Int8OutputMatrix(
        result.observables.data(),
        {result.shots, result.observable_count},
        nb::find(&result)
    );
}

DoubleOutputVector batch_log_likelihoods(qupy::BpOsdDecodeBatch& result) {
    return DoubleOutputVector(
        result.log_likelihoods.data(),
        {result.shots},
        nb::find(&result)
    );
}

UInt64OutputVector batch_matched_errors(qupy::BpOsdDecodeBatch& result) {
    return UInt64OutputVector(
        result.matched_errors.data(),
        {result.shots},
        nb::find(&result)
    );
}

UInt64OutputVector batch_iterations(qupy::BpOsdDecodeBatch& result) {
    return UInt64OutputVector(
        result.iterations.data(),
        {result.shots},
        nb::find(&result)
    );
}

Int8OutputVector batch_bp_converged(qupy::BpOsdDecodeBatch& result) {
    return Int8OutputVector(
        result.bp_converged.data(),
        {result.shots},
        nb::find(&result)
    );
}

Int8OutputVector batch_osd_used(qupy::BpOsdDecodeBatch& result) {
    return Int8OutputVector(
        result.osd_used.data(),
        {result.shots},
        nb::find(&result)
    );
}

}  // namespace

void bind_qec(nb::module_& module) {
    nb::class_<qupy::BpOsdDecodeResult>(module, "BpOsdDecodeResult")
        .def_prop_ro("correction", &correction_values)
        .def_prop_ro("observables", &observable_values)
        .def_ro("log_likelihood", &qupy::BpOsdDecodeResult::log_likelihood)
        .def_ro("matched_errors", &qupy::BpOsdDecodeResult::matched_errors)
        .def_ro("iterations", &qupy::BpOsdDecodeResult::iterations)
        .def_ro("bp_converged", &qupy::BpOsdDecodeResult::bp_converged)
        .def_ro("osd_used", &qupy::BpOsdDecodeResult::osd_used);

    nb::class_<qupy::BpOsdDecodeBatch>(module, "BpOsdDecodeBatch")
        .def_prop_ro("corrections", &batch_corrections)
        .def_prop_ro("observables", &batch_observables)
        .def_prop_ro("log_likelihoods", &batch_log_likelihoods)
        .def_prop_ro("matched_errors", &batch_matched_errors)
        .def_prop_ro("iterations", &batch_iterations)
        .def_prop_ro("bp_converged", &batch_bp_converged)
        .def_prop_ro("osd_used", &batch_osd_used)
        .def_ro("shots", &qupy::BpOsdDecodeBatch::shots)
        .def_ro("error_count", &qupy::BpOsdDecodeBatch::error_count)
        .def_ro("observable_count", &qupy::BpOsdDecodeBatch::observable_count);

    nb::class_<qupy::BpOsdDecoder>(module, "BpOsdDecoder")
        .def(
            nb::init<qupy::DetectorModel, std::size_t, double>(),
            "model"_a,
            "max_iterations"_a = 50U,
            "damping"_a = 0.0
        )
        .def_prop_ro("model", &qupy::BpOsdDecoder::model)
        .def_prop_ro("max_iterations", &qupy::BpOsdDecoder::max_iterations)
        .def_prop_ro("damping", &qupy::BpOsdDecoder::damping)
        .def_prop_ro("edge_count", &qupy::BpOsdDecoder::edge_count)
        .def_prop_ro("active_error_count", &qupy::BpOsdDecoder::active_error_count)
        .def(
            "decode",
            [](const qupy::BpOsdDecoder& decoder, Int8Vector syndrome) {
                const std::vector<std::int8_t> copied = copy_vector(syndrome);
                nb::gil_scoped_release release;
                return decoder.decode(copied);
            },
            "syndrome"_a
        )
        .def(
            "decode_batch",
            [](const qupy::BpOsdDecoder& decoder, Int8Matrix syndromes) {
                if (syndromes.shape(1) != decoder.model().detector_count()) {
                    throw nb::value_error(
                        "syndrome batch columns must match detector count"
                    );
                }
                const std::size_t shots = syndromes.shape(0);
                const std::vector<std::int8_t> copied = copy_matrix(syndromes);
                nb::gil_scoped_release release;
                return decoder.decode_batch(copied, shots);
            },
            "syndromes"_a
        );

    module.def(
        "decode_detector_model_bp_osd",
        [](
            const qupy::DetectorModel& model,
            Int8Vector syndrome,
            std::size_t max_iterations,
            double damping
        ) {
            const std::vector<std::int8_t> copied = copy_vector(syndrome);
            nb::gil_scoped_release release;
            return qupy::decode_detector_model_bp_osd(
                model,
                copied,
                max_iterations,
                damping
            );
        },
        "model"_a,
        "syndrome"_a,
        "max_iterations"_a = 50U,
        "damping"_a = 0.0
    );

    bind_tensor_network(module);
}
