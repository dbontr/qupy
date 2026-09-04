#pragma once

#include "qupy/core.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace qupy {

enum class Pauli : std::uint8_t {
    I,
    X,
    Y,
    Z,
};

struct PauliFactor {
    std::size_t qubit;
    Pauli pauli;
};

class PauliTerm {
public:
    PauliTerm(double coefficient, std::vector<PauliFactor> factors);

    [[nodiscard]] double coefficient() const noexcept;
    [[nodiscard]] const std::vector<PauliFactor>& factors() const noexcept;
    [[nodiscard]] std::string canonical_text() const;

private:
    double coefficient_;
    std::vector<PauliFactor> factors_;
};

class Observable {
public:
    explicit Observable(std::vector<PauliTerm> terms);

    [[nodiscard]] const std::vector<PauliTerm>& terms() const noexcept;
    [[nodiscard]] std::string canonical_text() const;
    [[nodiscard]] std::string fingerprint() const;

private:
    std::vector<PauliTerm> terms_;
};

struct ObservableExecutionPlan {
    std::string backend;
    std::string method;
    bool exact;
    std::size_t active_qubits;
    std::size_t observable_count;
    std::size_t term_count;
    std::size_t measurement_group_count;
    std::size_t estimated_state_bytes;
    std::string program_fingerprint;
    std::string query_fingerprint;
    std::string cache_key;
    std::optional<double> predicted_ns;
    std::string cost_model_class;
    std::string cost_model_fingerprint;
};
struct ObservableResult {
    double value;
    std::string backend;
    std::size_t active_qubits;
    std::size_t evaluations;
};

struct ObservableBatch {
    std::vector<double> values;
    std::size_t observable_count;
    std::string backend;
    std::size_t active_qubits;
};

struct MeasurementGroup {
    std::vector<std::size_t> term_indices;
    std::vector<PauliFactor> basis;
};

struct ShotEstimate {
    double value;
    double standard_error;
    std::size_t shots_per_group;
    std::size_t total_shots;
    std::size_t group_count;
    std::string backend;
};
enum class GradientMethod : std::uint8_t {
    Auto,
    Adjoint,
    ParameterShift,
    FiniteDifference,
};

struct GradientResult {
    double value;
    std::vector<double> gradient;
    std::string method;
    std::string backend;
    std::size_t evaluations;
};

struct JacobianResult {
    std::vector<double> values;
    std::vector<double> jacobian;
    std::size_t observable_count;
    std::size_t parameter_count;
    std::string method;
    std::string backend;
    std::size_t evaluations;
};

struct HessianResult {
    double value;
    std::vector<double> gradient;
    std::vector<double> hessian;
    std::size_t parameter_count;
    std::string method;
    std::string backend;
    std::size_t evaluations;
};

struct OptimizationReport {
    Program program;
    std::size_t original_operations;
    std::size_t optimized_operations;
    std::vector<std::string> passes;
};

enum class NoiseChannelCode : std::uint8_t {
    BitFlip,
    PhaseFlip,
    Depolarizing,
    AmplitudeDamping,
    PhaseDamping,
    Pauli,
    Kraus,
};

struct NoiseChannel {
    NoiseChannelCode code;
    std::size_t qubit;
    std::vector<double> parameters;
    std::vector<Complex> kraus_operators;
    std::size_t kraus_count = 0U;
};
struct NoiseInstruction {
    std::size_t after_operation;
    NoiseChannel channel;
};

class NoisyProgram {
public:
    NoisyProgram(Program program, std::vector<NoiseInstruction> noise);

    [[nodiscard]] const Program& program() const noexcept;
    [[nodiscard]] const std::vector<NoiseInstruction>& noise() const noexcept;

private:
    Program program_;
    std::vector<NoiseInstruction> noise_;
};

struct DensityMatrix {
    std::vector<Complex> values;
    std::size_t dimension;
    std::string backend;
};

struct LindbladResult {
    DensityMatrix state;
    double dt;
    std::size_t steps;
};

struct ProviderProgram {
    std::string format;
    std::string text;
    std::size_t num_qubits;
    bool measures_all;
};

enum class ProviderJobState : std::uint8_t {
    Queued,
    Running,
    Succeeded,
    Failed,
    Cancelled,
};

class ProviderPlugin {
public:
    explicit ProviderPlugin(std::string path);

    [[nodiscard]] const std::string& name() const noexcept;
    [[nodiscard]] std::string capabilities_json() const;
    [[nodiscard]] std::string submit(
        const ProviderProgram& program,
        std::uint64_t shots,
        const std::string& options_json = "{}"
    ) const;
    [[nodiscard]] ProviderJobState poll(const std::string& job_id) const;
    [[nodiscard]] std::string result_json(const std::string& job_id) const;
    void cancel(const std::string& job_id) const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

struct DetectorError {
    double probability;
    std::vector<std::size_t> detectors;
    std::vector<std::size_t> observables;
};

class DetectorModel {
public:
    DetectorModel(
        std::size_t detector_count,
        std::size_t observable_count,
        std::vector<DetectorError> errors
    );

    [[nodiscard]] std::size_t detector_count() const noexcept;
    [[nodiscard]] std::size_t observable_count() const noexcept;
    [[nodiscard]] const std::vector<DetectorError>& errors() const noexcept;
    [[nodiscard]] std::string canonical_text() const;
    [[nodiscard]] std::string fingerprint() const;

private:
    std::size_t detector_count_;
    std::size_t observable_count_;
    std::vector<DetectorError> errors_;
};

struct DetectorSamples {
    std::vector<std::int8_t> syndrome;
    std::vector<std::int8_t> observables;
    std::size_t shots;
    std::size_t detector_count;
    std::size_t observable_count;
};
struct DecodeResult {
    std::vector<std::int8_t> observables;
    double log_likelihood;
    std::size_t matched_errors;
};

struct DistributedInfo {
    bool available;
    std::size_t world_size;
    std::size_t rank;
    std::size_t local_rank;
    std::string runtime;
};

struct DistributedStateVector {
    std::vector<Complex> local_values;
    std::size_t global_size;
    std::size_t global_offset;
    std::size_t rank;
    std::size_t world_size;
    std::string backend;
};

[[nodiscard]] PauliTerm pauli_term(
    double coefficient,
    std::vector<PauliFactor> factors
);
[[nodiscard]] Observable observable(std::vector<PauliTerm> terms);
[[nodiscard]] Observable observable(PauliZ value);
[[nodiscard]] ObservableExecutionPlan observable_plan(
    const Program& program,
    const std::vector<Observable>& values,
    const std::string& backend = "auto",
    const PlannerCostModel* cost_model = nullptr
);
[[nodiscard]] std::vector<std::vector<std::size_t>> commuting_groups(
    const Observable& value
);
[[nodiscard]] std::vector<MeasurementGroup> measurement_groups(
    const Observable& value
);
[[nodiscard]] ShotEstimate estimate_observable(
    const Program& program,
    const Observable& value,
    std::size_t shots_per_group = 1024U,
    std::optional<std::uint64_t> seed = std::nullopt,
    const std::string& backend = "auto"
);
[[nodiscard]] ObservableResult expect_observable(
    const Program& program,
    const Observable& value,
    const std::string& backend = "auto",
    const PlannerCostModel* cost_model = nullptr
);
[[nodiscard]] ObservableResult variance_observable(
    const Program& program,
    const Observable& value,
    const std::string& backend = "auto",
    const PlannerCostModel* cost_model = nullptr
);
[[nodiscard]] ObservableResult covariance_observable(
    const Program& program,
    const Observable& left,
    const Observable& right,
    const std::string& backend = "auto",
    const PlannerCostModel* cost_model = nullptr
);
[[nodiscard]] ObservableBatch expect_observables(
    const Program& program,
    const std::vector<Observable>& values,
    const std::string& backend = "auto",
    const PlannerCostModel* cost_model = nullptr
);
[[nodiscard]] GradientResult value_and_grad(
    const Program& program,
    const Observable& value,
    const std::vector<ParameterSlot>& slots,
    const std::vector<double>& parameter_values,
    const std::string& backend = "auto",
    GradientMethod method = GradientMethod::Auto,
    double epsilon = 1e-7
);
[[nodiscard]] GradientResult grad(
    const Program& program,
    const Observable& value,
    const std::vector<ParameterSlot>& slots,
    const std::vector<double>& parameter_values,
    const std::string& backend = "auto",
    GradientMethod method = GradientMethod::Auto,
    double epsilon = 1e-7
);
[[nodiscard]] JacobianResult jacobian(
    const Program& program,
    const std::vector<Observable>& values,
    const std::vector<ParameterSlot>& slots,
    const std::vector<double>& parameter_values,
    const std::string& backend = "auto",
    GradientMethod method = GradientMethod::Auto,
    double epsilon = 1e-7
);
[[nodiscard]] HessianResult hessian(
    const Program& program,
    const Observable& value,
    const std::vector<ParameterSlot>& slots,
    const std::vector<double>& parameter_values,
    const std::string& backend = "auto"
);
[[nodiscard]] OptimizationReport optimize(const Program& program, std::uint32_t level = 2U);

[[nodiscard]] NoiseChannel bit_flip(std::size_t qubit, double probability);
[[nodiscard]] NoiseChannel phase_flip(std::size_t qubit, double probability);
[[nodiscard]] NoiseChannel depolarizing(std::size_t qubit, double probability);
[[nodiscard]] NoiseChannel amplitude_damping(std::size_t qubit, double gamma);
[[nodiscard]] NoiseChannel phase_damping(std::size_t qubit, double gamma);
[[nodiscard]] NoiseChannel pauli_channel(
    std::size_t qubit,
    double probability_x,
    double probability_y,
    double probability_z
);
[[nodiscard]] NoiseChannel kraus_channel(
    std::size_t qubit,
    const std::vector<std::vector<Complex>>& operators
);
[[nodiscard]] DensityMatrix density_matrix(
    const Program& program,
    const std::string& backend = "auto",
    const PlannerCostModel* cost_model = nullptr
);
[[nodiscard]] DensityMatrix density_matrix(
    const NoisyProgram& program,
    const std::string& backend = "auto",
    const PlannerCostModel* cost_model = nullptr
);
[[nodiscard]] LindbladResult lindblad_evolve(
    const DensityMatrix& initial,
    const std::vector<Complex>& hamiltonian,
    const std::vector<std::vector<Complex>>& collapse_operators,
    double dt,
    std::size_t steps
);

[[nodiscard]] ProviderProgram to_openqasm3(
    const Program& program,
    bool measure_all = false
);
[[nodiscard]] ProviderProgram to_qir_base_profile(
    const Program& program,
    bool measure_all = true
);

[[nodiscard]] DetectorModel repetition_code_detector_model(
    std::size_t distance,
    std::size_t rounds,
    double data_error_probability,
    double measurement_error_probability
);
[[nodiscard]] DetectorSamples sample_detector_model(
    const DetectorModel& model,
    std::size_t shots,
    std::optional<std::uint64_t> seed = std::nullopt
);
[[nodiscard]] DecodeResult decode_detector_model(
    const DetectorModel& model,
    const std::vector<std::int8_t>& syndrome
);
[[nodiscard]] DistributedInfo distributed_info();
[[nodiscard]] bool mpi_compiled() noexcept;
[[nodiscard]] DistributedStateVector distributed_statevector(const Program& program);
[[nodiscard]] DistributedStateVector distributed_cuda_statevector(
    const Program& program,
    std::optional<std::size_t> device = std::nullopt
);

}  // namespace qupy
