#include "qupy/advanced.hpp"

#include <cstdint>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using ComplexMatrix = nb::ndarray<
    nb::numpy,
    const qupy::Complex,
    nb::ndim<2>,
    nb::c_contig
>;
using DoubleVector = nb::ndarray<
    nb::numpy,
    const double,
    nb::ndim<1>,
    nb::c_contig
>;
using Int8Vector = nb::ndarray<
    nb::numpy,
    const std::int8_t,
    nb::ndim<1>,
    nb::c_contig
>;
using ComplexOutputVector = nb::ndarray<
    nb::numpy,
    const qupy::Complex,
    nb::ndim<1>,
    nb::c_contig
>;
using ComplexOutputMatrix = nb::ndarray<
    nb::numpy,
    const qupy::Complex,
    nb::ndim<2>,
    nb::c_contig
>;
using DoubleOutputVector = nb::ndarray<
    nb::numpy,
    const double,
    nb::ndim<1>,
    nb::c_contig
>;
using DoubleOutputMatrix = nb::ndarray<
    nb::numpy,
    const double,
    nb::ndim<2>,
    nb::c_contig
>;
using Int8OutputMatrix = nb::ndarray<
    nb::numpy,
    const std::int8_t,
    nb::ndim<2>,
    nb::c_contig
>;

DoubleOutputVector observable_batch_values(qupy::ObservableBatch& result) {
    return DoubleOutputVector(
        result.values.data(),
        {result.values.size()},
        nb::find(&result)
    );
}

DoubleOutputVector gradient_values(qupy::GradientResult& result) {
    return DoubleOutputVector(
        result.gradient.data(),
        {result.gradient.size()},
        nb::find(&result)
    );
}

DoubleOutputVector jacobian_result_values(qupy::JacobianResult& result) {
    return DoubleOutputVector(
        result.values.data(),
        {result.observable_count},
        nb::find(&result)
    );
}

DoubleOutputMatrix jacobian_values(qupy::JacobianResult& result) {
    return DoubleOutputMatrix(
        result.jacobian.data(),
        {result.observable_count, result.parameter_count},
        nb::find(&result)
    );
}

DoubleOutputVector hessian_gradient_values(qupy::HessianResult& result) {
    return DoubleOutputVector(
        result.gradient.data(),
        {result.parameter_count},
        nb::find(&result)
    );
}

DoubleOutputMatrix hessian_values(qupy::HessianResult& result) {
    return DoubleOutputMatrix(
        result.hessian.data(),
        {result.parameter_count, result.parameter_count},
        nb::find(&result)
    );
}

ComplexOutputVector distributed_values(qupy::DistributedStateVector& result) {
    return ComplexOutputVector(
        result.local_values.data(),
        {result.local_values.size()},
        nb::find(&result)
    );
}

ComplexOutputMatrix density_values(qupy::DensityMatrix& result) {
    return ComplexOutputMatrix(
        result.values.data(),
        {result.dimension, result.dimension},
        nb::find(&result)
    );
}

Int8OutputMatrix detector_syndrome(qupy::DetectorSamples& result) {
    return Int8OutputMatrix(
        result.syndrome.data(),
        {result.shots, result.detector_count},
        nb::find(&result)
    );
}

Int8OutputMatrix detector_observables(qupy::DetectorSamples& result) {
    return Int8OutputMatrix(
        result.observables.data(),
        {result.shots, result.observable_count},
        nb::find(&result)
    );
}

std::vector<double> copy_double_vector(DoubleVector values) {
    return values.size() == 0U
        ? std::vector<double>{}
        : std::vector<double>(values.data(), values.data() + values.size());
}

std::vector<std::int8_t> copy_int8_vector(Int8Vector values) {
    return values.size() == 0U
        ? std::vector<std::int8_t>{}
        : std::vector<std::int8_t>(values.data(), values.data() + values.size());
}
std::vector<qupy::Complex> copy_square_matrix(ComplexMatrix values, std::size_t expected) {
    if (values.shape(0) != expected || values.shape(1) != expected) {
        throw nb::value_error("matrix shape must match the density-matrix dimension");
    }
    return values.size() == 0U
        ? std::vector<qupy::Complex>{}
        : std::vector<qupy::Complex>(values.data(), values.data() + values.size());
}

std::vector<std::vector<qupy::Complex>> copy_collapse_operators(
    const nb::list& values,
    std::size_t expected
) {
    std::vector<std::vector<qupy::Complex>> result;
    result.reserve(values.size());
    for (nb::handle item : values) {
        ComplexMatrix matrix = nb::cast<ComplexMatrix>(item);
        result.push_back(copy_square_matrix(matrix, expected));
    }
    return result;
}

}  // namespace

void bind_advanced(nb::module_& module) {
    nb::enum_<qupy::Pauli>(module, "Pauli")
        .value("I", qupy::Pauli::I)
        .value("X", qupy::Pauli::X)
        .value("Y", qupy::Pauli::Y)
        .value("Z", qupy::Pauli::Z);

    nb::enum_<qupy::GradientMethod>(module, "GradientMethod")
        .value("AUTO", qupy::GradientMethod::Auto)
        .value("ADJOINT", qupy::GradientMethod::Adjoint)
        .value("PARAMETER_SHIFT", qupy::GradientMethod::ParameterShift)
        .value("FINITE_DIFFERENCE", qupy::GradientMethod::FiniteDifference);
    nb::enum_<qupy::ProviderJobState>(module, "ProviderJobState")
        .value("QUEUED", qupy::ProviderJobState::Queued)
        .value("RUNNING", qupy::ProviderJobState::Running)
        .value("SUCCEEDED", qupy::ProviderJobState::Succeeded)
        .value("FAILED", qupy::ProviderJobState::Failed)
        .value("CANCELLED", qupy::ProviderJobState::Cancelled);

    nb::enum_<qupy::NoiseChannelCode>(module, "NoiseChannelCode")
        .value("BIT_FLIP", qupy::NoiseChannelCode::BitFlip)
        .value("PHASE_FLIP", qupy::NoiseChannelCode::PhaseFlip)
        .value("DEPOLARIZING", qupy::NoiseChannelCode::Depolarizing)
        .value("AMPLITUDE_DAMPING", qupy::NoiseChannelCode::AmplitudeDamping)
        .value("PHASE_DAMPING", qupy::NoiseChannelCode::PhaseDamping)
        .value("PAULI", qupy::NoiseChannelCode::Pauli)
        .value("KRAUS", qupy::NoiseChannelCode::Kraus);

    nb::class_<qupy::PauliFactor>(module, "PauliFactor")
        .def(nb::init<std::size_t, qupy::Pauli>(), "qubit"_a, "pauli"_a)
        .def_ro("qubit", &qupy::PauliFactor::qubit)
        .def_ro("pauli", &qupy::PauliFactor::pauli);

    nb::class_<qupy::PauliTerm>(module, "PauliTerm")
        .def(
            nb::init<double, std::vector<qupy::PauliFactor>>(),
            "coefficient"_a,
            "factors"_a
        )
        .def_prop_ro("coefficient", &qupy::PauliTerm::coefficient)
        .def_prop_ro("factors", &qupy::PauliTerm::factors)
        .def_prop_ro("canonical_text", &qupy::PauliTerm::canonical_text);

    nb::class_<qupy::Observable>(module, "Observable")
        .def(nb::init<std::vector<qupy::PauliTerm>>(), "terms"_a)
        .def_prop_ro("terms", &qupy::Observable::terms)
        .def_prop_ro("canonical_text", &qupy::Observable::canonical_text)
        .def_prop_ro("fingerprint", &qupy::Observable::fingerprint);

    nb::class_<qupy::ObservableExecutionPlan>(module, "ObservableExecutionPlan")
        .def_ro("backend", &qupy::ObservableExecutionPlan::backend)
        .def_ro("method", &qupy::ObservableExecutionPlan::method)
        .def_ro("exact", &qupy::ObservableExecutionPlan::exact)
        .def_ro("active_qubits", &qupy::ObservableExecutionPlan::active_qubits)
        .def_ro("observable_count", &qupy::ObservableExecutionPlan::observable_count)
        .def_ro("term_count", &qupy::ObservableExecutionPlan::term_count)
        .def_ro("measurement_group_count", &qupy::ObservableExecutionPlan::measurement_group_count)
        .def_ro("estimated_state_bytes", &qupy::ObservableExecutionPlan::estimated_state_bytes)
        .def_ro("program_fingerprint", &qupy::ObservableExecutionPlan::program_fingerprint)
        .def_ro("query_fingerprint", &qupy::ObservableExecutionPlan::query_fingerprint)
        .def_ro("cache_key", &qupy::ObservableExecutionPlan::cache_key)
        .def_ro("predicted_ns", &qupy::ObservableExecutionPlan::predicted_ns)
        .def_ro("cost_model_class", &qupy::ObservableExecutionPlan::cost_model_class)
        .def_ro("cost_model_fingerprint", &qupy::ObservableExecutionPlan::cost_model_fingerprint);

    nb::class_<qupy::ObservableResult>(module, "ObservableResult")
        .def_ro("value", &qupy::ObservableResult::value)
        .def_ro("backend", &qupy::ObservableResult::backend)
        .def_ro("active_qubits", &qupy::ObservableResult::active_qubits)
        .def_ro("evaluations", &qupy::ObservableResult::evaluations);

    nb::class_<qupy::ObservableBatch>(module, "ObservableBatch")
        .def_prop_ro("values", &observable_batch_values)
        .def_ro("observable_count", &qupy::ObservableBatch::observable_count)
        .def_ro("backend", &qupy::ObservableBatch::backend)
        .def_ro("active_qubits", &qupy::ObservableBatch::active_qubits);

    nb::class_<qupy::MeasurementGroup>(module, "MeasurementGroup")
        .def_ro("term_indices", &qupy::MeasurementGroup::term_indices)
        .def_ro("basis", &qupy::MeasurementGroup::basis);

    nb::class_<qupy::ShotEstimate>(module, "ShotEstimate")
        .def_ro("value", &qupy::ShotEstimate::value)
        .def_ro("standard_error", &qupy::ShotEstimate::standard_error)
        .def_ro("shots_per_group", &qupy::ShotEstimate::shots_per_group)
        .def_ro("total_shots", &qupy::ShotEstimate::total_shots)
        .def_ro("group_count", &qupy::ShotEstimate::group_count)
        .def_ro("backend", &qupy::ShotEstimate::backend);

    nb::class_<qupy::GradientResult>(module, "GradientResult")
        .def_ro("value", &qupy::GradientResult::value)
        .def_prop_ro("gradient", &gradient_values)
        .def_ro("method", &qupy::GradientResult::method)
        .def_ro("backend", &qupy::GradientResult::backend)
        .def_ro("evaluations", &qupy::GradientResult::evaluations);

    nb::class_<qupy::JacobianResult>(module, "JacobianResult")
        .def_prop_ro("values", &jacobian_result_values)
        .def_prop_ro("jacobian", &jacobian_values)
        .def_ro("observable_count", &qupy::JacobianResult::observable_count)
        .def_ro("parameter_count", &qupy::JacobianResult::parameter_count)
        .def_ro("method", &qupy::JacobianResult::method)
        .def_ro("backend", &qupy::JacobianResult::backend)
        .def_ro("evaluations", &qupy::JacobianResult::evaluations);

    nb::class_<qupy::HessianResult>(module, "HessianResult")
        .def_ro("value", &qupy::HessianResult::value)
        .def_prop_ro("gradient", &hessian_gradient_values)
        .def_prop_ro("hessian", &hessian_values)
        .def_ro("parameter_count", &qupy::HessianResult::parameter_count)
        .def_ro("method", &qupy::HessianResult::method)
        .def_ro("backend", &qupy::HessianResult::backend)
        .def_ro("evaluations", &qupy::HessianResult::evaluations);

    nb::class_<qupy::OptimizationReport>(module, "OptimizationReport")
        .def_ro("program", &qupy::OptimizationReport::program)
        .def_ro("original_operations", &qupy::OptimizationReport::original_operations)
        .def_ro("optimized_operations", &qupy::OptimizationReport::optimized_operations)
        .def_ro("passes", &qupy::OptimizationReport::passes);
    nb::class_<qupy::NoiseChannel>(module, "NoiseChannel")
        .def_ro("code", &qupy::NoiseChannel::code)
        .def_ro("qubit", &qupy::NoiseChannel::qubit)
        .def_ro("parameters", &qupy::NoiseChannel::parameters)
        .def_ro("kraus_operators", &qupy::NoiseChannel::kraus_operators)
        .def_ro("kraus_count", &qupy::NoiseChannel::kraus_count);

    nb::class_<qupy::NoiseInstruction>(module, "NoiseInstruction")
        .def(
            nb::init<std::size_t, qupy::NoiseChannel>(),
            "after_operation"_a,
            "channel"_a
        )
        .def_ro("after_operation", &qupy::NoiseInstruction::after_operation)
        .def_ro("channel", &qupy::NoiseInstruction::channel);

    nb::class_<qupy::NoisyProgram>(module, "NoisyProgram")
        .def(
            nb::init<qupy::Program, std::vector<qupy::NoiseInstruction>>(),
            "program"_a,
            "noise"_a
        )
        .def_prop_ro("program", &qupy::NoisyProgram::program)
        .def_prop_ro("noise", &qupy::NoisyProgram::noise);

    nb::class_<qupy::DensityMatrix>(module, "DensityMatrix")
        .def_prop_ro("values", &density_values)
        .def_ro("dimension", &qupy::DensityMatrix::dimension)
        .def_ro("backend", &qupy::DensityMatrix::backend);

    nb::class_<qupy::LindbladResult>(module, "LindbladResult")
        .def_ro("state", &qupy::LindbladResult::state)
        .def_ro("dt", &qupy::LindbladResult::dt)
        .def_ro("steps", &qupy::LindbladResult::steps);
    nb::class_<qupy::ProviderProgram>(module, "ProviderProgram")
        .def_ro("format", &qupy::ProviderProgram::format)
        .def_ro("text", &qupy::ProviderProgram::text)
        .def_ro("num_qubits", &qupy::ProviderProgram::num_qubits)
        .def_ro("measures_all", &qupy::ProviderProgram::measures_all);

    nb::class_<qupy::ProviderPlugin>(module, "ProviderPlugin")
        .def(nb::init<std::string>(), "path"_a)
        .def_prop_ro("name", &qupy::ProviderPlugin::name)
        .def(
            "capabilities_json",
            &qupy::ProviderPlugin::capabilities_json,
            nb::call_guard<nb::gil_scoped_release>()
        )
        .def(
            "submit",
            &qupy::ProviderPlugin::submit,
            "program"_a,
            "shots"_a,
            "options_json"_a = "{}",
            nb::call_guard<nb::gil_scoped_release>()
        )
        .def("poll", &qupy::ProviderPlugin::poll, "job_id"_a, nb::call_guard<nb::gil_scoped_release>())
        .def(
            "result_json",
            &qupy::ProviderPlugin::result_json,
            "job_id"_a,
            nb::call_guard<nb::gil_scoped_release>()
        )
        .def("cancel", &qupy::ProviderPlugin::cancel, "job_id"_a, nb::call_guard<nb::gil_scoped_release>());

    nb::class_<qupy::DetectorError>(module, "DetectorError")
        .def(
            nb::init<double, std::vector<std::size_t>, std::vector<std::size_t>>(),
            "probability"_a,
            "detectors"_a,
            "observables"_a = std::vector<std::size_t>{}
        )
        .def_ro("probability", &qupy::DetectorError::probability)
        .def_ro("detectors", &qupy::DetectorError::detectors)
        .def_ro("observables", &qupy::DetectorError::observables);

    nb::class_<qupy::DetectorModel>(module, "DetectorModel")
        .def(
            nb::init<std::size_t, std::size_t, std::vector<qupy::DetectorError>>(),
            "detector_count"_a,
            "observable_count"_a,
            "errors"_a
        )
        .def_prop_ro("detector_count", &qupy::DetectorModel::detector_count)
        .def_prop_ro("observable_count", &qupy::DetectorModel::observable_count)
        .def_prop_ro("errors", &qupy::DetectorModel::errors)
        .def_prop_ro("canonical_text", &qupy::DetectorModel::canonical_text)
        .def_prop_ro("fingerprint", &qupy::DetectorModel::fingerprint);
    nb::class_<qupy::DetectorSamples>(module, "DetectorSamples")
        .def_prop_ro("syndrome", &detector_syndrome)
        .def_prop_ro("observables", &detector_observables)
        .def_ro("shots", &qupy::DetectorSamples::shots)
        .def_ro("detector_count", &qupy::DetectorSamples::detector_count)
        .def_ro("observable_count", &qupy::DetectorSamples::observable_count);

    nb::class_<qupy::DecodeResult>(module, "DecodeResult")
        .def_ro("observables", &qupy::DecodeResult::observables)
        .def_ro("log_likelihood", &qupy::DecodeResult::log_likelihood)
        .def_ro("matched_errors", &qupy::DecodeResult::matched_errors);

    nb::class_<qupy::DistributedInfo>(module, "DistributedInfo")
        .def_ro("available", &qupy::DistributedInfo::available)
        .def_ro("world_size", &qupy::DistributedInfo::world_size)
        .def_ro("rank", &qupy::DistributedInfo::rank)
        .def_ro("local_rank", &qupy::DistributedInfo::local_rank)
        .def_ro("runtime", &qupy::DistributedInfo::runtime);

    nb::class_<qupy::DistributedStateVector>(module, "DistributedStateVector")
        .def_prop_ro("local_values", &distributed_values)
        .def_ro("global_size", &qupy::DistributedStateVector::global_size)
        .def_ro("global_offset", &qupy::DistributedStateVector::global_offset)
        .def_ro("rank", &qupy::DistributedStateVector::rank)
        .def_ro("world_size", &qupy::DistributedStateVector::world_size)
        .def_ro("backend", &qupy::DistributedStateVector::backend);

    module.def(
        "pauli",
        [](std::size_t qubit, qupy::Pauli value) { return qupy::PauliFactor{qubit, value}; },
        "qubit"_a,
        "value"_a
    );
    module.def("pauli_term", &qupy::pauli_term, "coefficient"_a, "factors"_a);
    module.def(
        "observable",
        nb::overload_cast<std::vector<qupy::PauliTerm>>(&qupy::observable),
        "terms"_a
    );
    module.def(
        "observable_from_z",
        nb::overload_cast<qupy::PauliZ>(&qupy::observable),
        "observable"_a
    );
    module.def(
        "observable_plan",
        [](
            const qupy::Program& program,
            const std::vector<qupy::Observable>& observables,
            const std::string& backend,
            nb::object cost_model
        ) {
            const auto* model = cost_model.is_none()
                ? nullptr : &nb::cast<const qupy::PlannerCostModel&>(cost_model);
            return qupy::observable_plan(program, observables, backend, model);
        },
        "program"_a, "observables"_a, "backend"_a = "auto", "cost_model"_a = nb::none()
    );
    module.def("commuting_groups", &qupy::commuting_groups, "observable"_a);
    module.def("measurement_groups", &qupy::measurement_groups, "observable"_a);
    module.def(
        "estimate_observable",
        &qupy::estimate_observable,
        "program"_a,
        "observable"_a,
        "shots_per_group"_a = 1024U,
        "seed"_a = std::nullopt,
        "backend"_a = "auto",
        nb::call_guard<nb::gil_scoped_release>()
    );
    const auto cost_model_ptr = [](nb::object cost_model) -> const qupy::PlannerCostModel* {
        return cost_model.is_none()
            ? nullptr : &nb::cast<const qupy::PlannerCostModel&>(cost_model);
    };
    module.def(
        "expect_observable",
        [cost_model_ptr](const qupy::Program& program, const qupy::Observable& observable,
                         const std::string& backend, nb::object cost_model) {
            const auto* model = cost_model_ptr(cost_model);
            nb::gil_scoped_release release;
            return qupy::expect_observable(program, observable, backend, model);
        },
        "program"_a, "observable"_a, "backend"_a = "auto", "cost_model"_a = nb::none()
    );
    module.def(
        "expect",
        [cost_model_ptr](const qupy::Program& program, const qupy::Observable& observable,
                         const std::string& backend, nb::object cost_model) {
            const auto* model = cost_model_ptr(cost_model);
            nb::gil_scoped_release release;
            return qupy::expect_observable(program, observable, backend, model);
        },
        "program"_a, "observable"_a, "backend"_a = "auto", "cost_model"_a = nb::none()
    );
    module.def(
        "variance_observable",
        [cost_model_ptr](const qupy::Program& program, const qupy::Observable& observable,
                         const std::string& backend, nb::object cost_model) {
            const auto* model = cost_model_ptr(cost_model);
            nb::gil_scoped_release release;
            return qupy::variance_observable(program, observable, backend, model);
        },
        "program"_a, "observable"_a, "backend"_a = "auto", "cost_model"_a = nb::none()
    );
    module.def(
        "variance",
        [cost_model_ptr](const qupy::Program& program, const qupy::Observable& observable,
                         const std::string& backend, nb::object cost_model) {
            const auto* model = cost_model_ptr(cost_model);
            nb::gil_scoped_release release;
            return qupy::variance_observable(program, observable, backend, model);
        },
        "program"_a, "observable"_a, "backend"_a = "auto", "cost_model"_a = nb::none()
    );
    module.def(
        "covariance",
        [cost_model_ptr](const qupy::Program& program, const qupy::Observable& left,
                         const qupy::Observable& right, const std::string& backend,
                         nb::object cost_model) {
            const auto* model = cost_model_ptr(cost_model);
            nb::gil_scoped_release release;
            return qupy::covariance_observable(program, left, right, backend, model);
        },
        "program"_a, "left"_a, "right"_a, "backend"_a = "auto", "cost_model"_a = nb::none()
    );
    module.def(
        "expect_observables",
        [cost_model_ptr](const qupy::Program& program,
                         const std::vector<qupy::Observable>& observables,
                         const std::string& backend, nb::object cost_model) {
            const auto* model = cost_model_ptr(cost_model);
            nb::gil_scoped_release release;
            return qupy::expect_observables(program, observables, backend, model);
        },
        "program"_a, "observables"_a, "backend"_a = "auto", "cost_model"_a = nb::none()
    );
    module.def(
        "value_and_grad",
        [](
            const qupy::Program& program,
            const qupy::Observable& observable,
            const std::vector<qupy::ParameterSlot>& slots,
            DoubleVector parameter_values,
            const std::string& backend,
            qupy::GradientMethod method,
            double epsilon
        ) {
            std::vector<double> values = copy_double_vector(parameter_values);
            nb::gil_scoped_release release;
            return qupy::value_and_grad(
                program, observable, slots, values, backend, method, epsilon
            );
        },
        "program"_a,
        "observable"_a,
        "slots"_a,
        "parameter_values"_a,
        "backend"_a = "auto",
        "method"_a = qupy::GradientMethod::Auto,
        "epsilon"_a = 1e-7
    );
    module.def(
        "grad",
        [](
            const qupy::Program& program,
            const qupy::Observable& observable,
            const std::vector<qupy::ParameterSlot>& slots,
            DoubleVector parameter_values,
            const std::string& backend,
            qupy::GradientMethod method,
            double epsilon
        ) {
            std::vector<double> values = copy_double_vector(parameter_values);
            nb::gil_scoped_release release;
            return qupy::grad(program, observable, slots, values, backend, method, epsilon);
        },
        "program"_a, "observable"_a, "slots"_a, "parameter_values"_a,
        "backend"_a = "auto", "method"_a = qupy::GradientMethod::Auto, "epsilon"_a = 1e-7
    );
    module.def(
        "jacobian",
        [](
            const qupy::Program& program,
            const std::vector<qupy::Observable>& observables,
            const std::vector<qupy::ParameterSlot>& slots,
            DoubleVector parameter_values,
            const std::string& backend,
            qupy::GradientMethod method,
            double epsilon
        ) {
            std::vector<double> values = copy_double_vector(parameter_values);
            nb::gil_scoped_release release;
            return qupy::jacobian(program, observables, slots, values, backend, method, epsilon);
        },
        "program"_a, "observables"_a, "slots"_a, "parameter_values"_a,
        "backend"_a = "auto", "method"_a = qupy::GradientMethod::Auto, "epsilon"_a = 1e-7
    );
    module.def(
        "hessian",
        [](
            const qupy::Program& program,
            const qupy::Observable& observable,
            const std::vector<qupy::ParameterSlot>& slots,
            DoubleVector parameter_values,
            const std::string& backend
        ) {
            std::vector<double> values = copy_double_vector(parameter_values);
            nb::gil_scoped_release release;
            return qupy::hessian(program, observable, slots, values, backend);
        },
        "program"_a, "observable"_a, "slots"_a, "parameter_values"_a,
        "backend"_a = "auto"
    );
    module.def("optimize", &qupy::optimize, "program"_a, "level"_a = 2U);

    module.def("bit_flip", &qupy::bit_flip, "qubit"_a, "probability"_a);
    module.def("phase_flip", &qupy::phase_flip, "qubit"_a, "probability"_a);
    module.def("depolarizing", &qupy::depolarizing, "qubit"_a, "probability"_a);
    module.def("amplitude_damping", &qupy::amplitude_damping, "qubit"_a, "gamma"_a);
    module.def("phase_damping", &qupy::phase_damping, "qubit"_a, "gamma"_a);
    module.def(
        "pauli_channel",
        &qupy::pauli_channel,
        "qubit"_a,
        "probability_x"_a,
        "probability_y"_a,
        "probability_z"_a
    );
    module.def(
        "kraus_channel",
        [](std::size_t qubit, const nb::list& operators) {
            std::vector<std::vector<qupy::Complex>> values;
            values.reserve(operators.size());
            for (nb::handle item : operators) {
                values.push_back(copy_square_matrix(nb::cast<ComplexMatrix>(item), 2U));
            }
            return qupy::kraus_channel(qubit, values);
        },
        "qubit"_a,
        "operators"_a
    );
    module.def(
        "density_matrix",
        [](const qupy::Program& program, const std::string& backend, nb::object cost_model) {
            const auto* model = cost_model.is_none()
                ? nullptr : &nb::cast<const qupy::PlannerCostModel&>(cost_model);
            nb::gil_scoped_release release;
            return qupy::density_matrix(program, backend, model);
        },
        "program"_a,
        "backend"_a = "auto",
        "cost_model"_a = nb::none()
    );
    module.def(
        "density_matrix",
        [](const qupy::NoisyProgram& program, const std::string& backend, nb::object cost_model) {
            const auto* model = cost_model.is_none()
                ? nullptr : &nb::cast<const qupy::PlannerCostModel&>(cost_model);
            nb::gil_scoped_release release;
            return qupy::density_matrix(program, backend, model);
        },
        "program"_a,
        "backend"_a = "auto",
        "cost_model"_a = nb::none()
    );
    module.def(
        "lindblad_evolve",
        [](
            const qupy::DensityMatrix& initial,
            ComplexMatrix hamiltonian,
            const nb::list& collapse_operators,
            double dt,
            std::size_t steps
        ) {
            std::vector<qupy::Complex> h = copy_square_matrix(
                hamiltonian, initial.dimension
            );
            auto collapse = copy_collapse_operators(collapse_operators, initial.dimension);
            nb::gil_scoped_release release;
            return qupy::lindblad_evolve(initial, h, collapse, dt, steps);
        },
        "initial"_a,
        "hamiltonian"_a,
        "collapse_operators"_a,
        "dt"_a,
        "steps"_a
    );

    module.def("to_openqasm3", &qupy::to_openqasm3, "program"_a, "measure_all"_a = false);
    module.def(
        "to_qir_base_profile",
        &qupy::to_qir_base_profile,
        "program"_a,
        "measure_all"_a = true
    );
    module.def(
        "repetition_code_detector_model",
        &qupy::repetition_code_detector_model,
        "distance"_a,
        "rounds"_a,
        "data_error_probability"_a,
        "measurement_error_probability"_a
    );
    module.def(
        "sample_detector_model",
        &qupy::sample_detector_model,
        "model"_a,
        "shots"_a,
        "seed"_a = std::nullopt,
        nb::call_guard<nb::gil_scoped_release>()
    );
    module.def(
        "decode_detector_model",
        [](
            const qupy::DetectorModel& model,
            Int8Vector syndrome
        ) {
            std::vector<std::int8_t> values = copy_int8_vector(syndrome);
            nb::gil_scoped_release release;
            return qupy::decode_detector_model(model, values);
        },
        "model"_a,
        "syndrome"_a
    );
    module.def("distributed_info", &qupy::distributed_info);
    module.def("mpi_compiled", &qupy::mpi_compiled);
    module.def(
        "distributed_statevector",
        &qupy::distributed_statevector,
        "program"_a,
        nb::call_guard<nb::gil_scoped_release>()
    );
}
