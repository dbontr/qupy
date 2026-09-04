#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace qupy {

using Complex = std::complex<double>;

enum class OperationCode : std::uint8_t {
    H,
    X,
    Y,
    Z,
    RX,
    RY,
    RZ,
    CX,
    CZ,
    SWAP,
};

enum class ResultMode : std::uint8_t {
    Sample,
    Expectation,
    Probabilities,
    Variance,
    StateVector,
};

struct Operation {
    OperationCode code;
    std::vector<std::size_t> qubits;
    std::vector<double> parameters;

    [[nodiscard]] std::string name() const;
};

struct ParameterSlot {
    ParameterSlot(std::size_t operation_index_, std::size_t parameter_index_ = 0U)
        : operation_index(operation_index_), parameter_index(parameter_index_) {}

    std::size_t operation_index;
    std::size_t parameter_index;
};

class Program {
public:
    explicit Program(std::size_t num_qubits);

    [[nodiscard]] std::size_t num_qubits() const noexcept;
    [[nodiscard]] const std::vector<Operation>& operations() const noexcept;
    [[nodiscard]] std::string canonical_text() const;
    [[nodiscard]] std::string fingerprint() const;
    [[nodiscard]] Program appended(Operation operation) const;
    [[nodiscard]] Program bound(
        const std::vector<ParameterSlot>& slots,
        const std::vector<double>& values
    ) const;

private:
    std::size_t num_qubits_;
    std::vector<Operation> operations_;
};

struct PauliZ {
    std::size_t qubit;
};

struct Target {
    std::string name;
    std::vector<OperationCode> operations;
    std::vector<ResultMode> result_modes;
    std::optional<std::size_t> max_qubits;
    bool simulator = false;
    bool state_access = false;
    bool mid_circuit_measurement = false;
    bool reset = false;
    bool dynamic_control = false;
    bool parameter_batches = false;

    [[nodiscard]] bool supports(OperationCode code) const;
    [[nodiscard]] bool supports(ResultMode mode) const;
    [[nodiscard]] std::string fingerprint() const;
    void validate(const Program& program, ResultMode mode) const;
};

struct ExecutionPlan {
    std::string backend;
    std::string method;
    bool exact;
    std::size_t threads;
    std::size_t original_qubits;
    std::size_t original_operations;
    std::size_t active_qubits;
    std::size_t active_operations;
    std::size_t single_qubit_operations;
    std::size_t two_qubit_operations;
    std::size_t parameterized_operations;
    std::size_t non_clifford_operations;
    std::size_t compiled_steps;
    std::size_t estimated_state_bytes;
    ResultMode result_mode;
    std::uint32_t workload_version;
    std::string workload_fingerprint;
    std::string program_fingerprint;
    std::string target_fingerprint;
    std::string cache_key;
    std::optional<double> predicted_ns;
    std::string cost_model_class;
    std::string cost_model_fingerprint;
    std::string cost_model_host_fingerprint;
    std::string cost_model_cuda_host_fingerprint;
    std::size_t tensor_network_max_bond = 0U;
    std::size_t tensor_network_routed_swaps = 0U;
    double tensor_network_contraction_work = 0.0;
};

class PlannerCostModel {
public:
    [[nodiscard]] std::uint32_t schema_version() const noexcept;
    [[nodiscard]] std::uint32_t workload_version() const noexcept;
    [[nodiscard]] const std::string& engine_version() const noexcept;
    [[nodiscard]] const std::string& host_fingerprint() const noexcept;
    [[nodiscard]] const std::string& cuda_host_fingerprint() const noexcept;
    [[nodiscard]] const std::string& artifact_fingerprint() const noexcept;
    [[nodiscard]] std::vector<std::string> cost_classes() const;
    [[nodiscard]] bool has_cost_class(const std::string& cost_class) const;
    [[nodiscard]] bool cuda_auto_validated() const noexcept;
    [[nodiscard]] bool mps_auto_validated() const noexcept;
    [[nodiscard]] bool observable_auto_validated() const noexcept;
    [[nodiscard]] bool density_auto_validated() const noexcept;
    [[nodiscard]] std::uint32_t mps_policy_version() const noexcept;
    [[nodiscard]] std::uint32_t observable_policy_version() const noexcept;
    [[nodiscard]] std::uint32_t density_policy_version() const noexcept;
    [[nodiscard]] double predict_ns(const ExecutionPlan& plan) const;
    [[nodiscard]] double predict_observable_ns(
        const ExecutionPlan& state_plan,
        std::size_t term_evaluations,
        std::size_t state_passes
    ) const;
    [[nodiscard]] double predict_density_ns(
        const std::string& backend,
        std::size_t qubits,
        std::size_t single_qubit_operations,
        std::size_t two_qubit_operations,
        std::size_t noise_events,
        std::size_t kraus_evaluations
    ) const;    [[nodiscard]] double predict_density_speedup(
        std::size_t qubits,
        std::size_t single_qubit_operations,
        std::size_t two_qubit_operations,
        std::size_t noise_events,
        std::size_t kraus_evaluations
    ) const;

private:
    struct Curve {
        std::string cost_class;
        std::vector<double> coefficients;
        double holdout_median_factor;
        double holdout_max_factor;
    };

    PlannerCostModel() = default;
    friend PlannerCostModel load_planner_cost_model(const std::string& path);
    std::uint32_t schema_version_ = 0U;
    std::uint32_t workload_version_ = 0U;
    std::string engine_version_;
    std::string host_fingerprint_;
    std::string cuda_host_fingerprint_;
    std::string artifact_fingerprint_;
    std::size_t cuda_decision_samples_ = 0U;
    std::size_t cuda_decision_mistakes_ = 0U;
    double cuda_decision_max_regret_ = 0.0;
    std::uint32_t mps_policy_version_ = 0U;
    std::size_t mps_decision_samples_ = 0U;
    std::size_t mps_decision_mistakes_ = 0U;
    double mps_decision_max_regret_ = 0.0;
    std::uint32_t observable_policy_version_ = 0U;
    std::size_t observable_decision_samples_ = 0U;
    std::size_t observable_decision_mistakes_ = 0U;
    double observable_decision_max_regret_ = 0.0;
    std::uint32_t density_policy_version_ = 0U;
    std::size_t density_decision_samples_ = 0U;
    std::size_t density_decision_mistakes_ = 0U;
    double density_decision_max_regret_ = 0.0;
    std::vector<Curve> curves_;
};

struct StateVector {
    std::vector<Complex> values;
    std::string backend;
};

struct Probabilities {
    std::vector<double> values;
    std::string backend;
};

struct Samples {
    std::vector<std::int8_t> values;
    std::size_t shots;
    std::size_t num_qubits;
    std::string backend;

    [[nodiscard]] std::map<std::string, std::size_t> counts() const;
};

struct SamplesBatch {
    std::vector<std::int8_t> values;
    std::size_t batch_size;
    std::size_t shots;
    std::size_t num_qubits;
    std::size_t parameter_count;
    std::string backend;
    std::size_t compiled_steps;
    std::size_t estimated_state_bytes;

    [[nodiscard]] std::map<std::string, std::size_t> counts(std::size_t batch_index) const;
};

struct Expectation {
    double value;
    std::string backend;
    std::size_t active_qubits;
    std::size_t compiled_steps;
    std::size_t estimated_state_bytes;
};

struct ExpectationBatch {
    std::vector<double> values;
    std::size_t batch_size;
    std::size_t parameter_count;
    std::string backend;
    std::size_t active_qubits;
    std::size_t compiled_steps;
    std::size_t estimated_state_bytes;
};

struct Variance {
    double value;
    std::string backend;
    std::size_t active_qubits;
    std::size_t compiled_steps;
    std::size_t estimated_state_bytes;
};

[[nodiscard]] Target native_target();
[[nodiscard]] std::size_t cuda_device_count() noexcept;
[[nodiscard]] bool cuda_available(std::size_t device = 0U) noexcept;
[[nodiscard]] std::string cuda_unavailable_reason(std::size_t device = 0U);
[[nodiscard]] std::string cuda_device_name(std::size_t device = 0U);
[[nodiscard]] Target cuda_target(std::size_t device = 0U);
[[nodiscard]] Target mps_target();
[[nodiscard]] Target adaptive_mps_target();
[[nodiscard]] std::string planner_host_fingerprint();
[[nodiscard]] std::string planner_cuda_host_fingerprint();
[[nodiscard]] PlannerCostModel load_planner_cost_model(const std::string& path);
[[nodiscard]] ExecutionPlan plan(
    const Program& program,
    ResultMode result_mode,
    const std::string& backend = "auto",
    const PlannerCostModel* cost_model = nullptr
);
[[nodiscard]] ExecutionPlan expectation_plan(
    const Program& program,
    PauliZ observable,
    const std::string& backend = "auto",
    const PlannerCostModel* cost_model = nullptr
);
[[nodiscard]] ExecutionPlan variance_plan(
    const Program& program,
    PauliZ observable,
    const std::string& backend = "auto",
    const PlannerCostModel* cost_model = nullptr
);

[[nodiscard]] Program h(const Program& program, std::size_t qubit);
[[nodiscard]] Program x(const Program& program, std::size_t qubit);
[[nodiscard]] Program y(const Program& program, std::size_t qubit);
[[nodiscard]] Program z(const Program& program, std::size_t qubit);
[[nodiscard]] Program rx(const Program& program, double angle, std::size_t qubit);
[[nodiscard]] Program ry(const Program& program, double angle, std::size_t qubit);
[[nodiscard]] Program rz(const Program& program, double angle, std::size_t qubit);
[[nodiscard]] Program cx(const Program& program, std::size_t control, std::size_t target);
[[nodiscard]] Program cz(const Program& program, std::size_t control, std::size_t target);
[[nodiscard]] Program swap(const Program& program, std::size_t first, std::size_t second);
[[nodiscard]] PauliZ pauli_z(std::size_t qubit);

[[nodiscard]] StateVector statevector(
    const Program& program,
    const std::string& backend = "auto",
    const PlannerCostModel* cost_model = nullptr
);
[[nodiscard]] Probabilities probabilities(
    const Program& program,
    const std::string& backend = "auto"
);
[[nodiscard]] Samples sample(
    const Program& program,
    std::size_t shots = 1024,
    std::optional<std::uint64_t> seed = std::nullopt,
    const std::string& backend = "auto"
);
[[nodiscard]] SamplesBatch sample_batch(
    const Program& program,
    const std::vector<ParameterSlot>& slots,
    const std::vector<double>& parameter_values,
    std::size_t batch_size,
    std::size_t shots = 1024,
    std::optional<std::uint64_t> seed = std::nullopt,
    const std::string& backend = "auto"
);
[[nodiscard]] Expectation expectation(
    const Program& program,
    PauliZ observable,
    const std::string& backend = "auto",
    const PlannerCostModel* cost_model = nullptr
);
[[nodiscard]] ExpectationBatch expectation_batch(
    const Program& program,
    PauliZ observable,
    const std::vector<ParameterSlot>& slots,
    const std::vector<double>& parameter_values,
    std::size_t batch_size,
    const std::string& backend = "auto"
);
[[nodiscard]] Variance variance(
    const Program& program,
    PauliZ observable,
    const std::string& backend = "auto",
    const PlannerCostModel* cost_model = nullptr
);

[[nodiscard]] const char* core_language() noexcept;
[[nodiscard]] const char* core_version() noexcept;
[[nodiscard]] std::uint32_t ir_version() noexcept;
[[nodiscard]] std::size_t parallel_threads() noexcept;

}  // namespace qupy
