#pragma once

#include "qupy/advanced.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace qupy {

inline constexpr std::size_t kTensorNetworkDefaultMaxBytes = std::size_t{1} << 30U;

struct TensorNetworkPlan {
    std::size_t term_count;
    std::size_t contractions;
    std::size_t peak_tensor_rank;
    std::size_t peak_tensor_bytes;
    double scalar_multiplications;
    std::size_t max_tensor_bytes;
    bool exact;
    std::string backend;
    std::string method;
    std::string program_fingerprint;
    std::string observable_fingerprint;
    std::string plan_fingerprint;
};

struct TensorNetworkResult {
    double value;
    std::size_t term_count;
    std::size_t contractions;
    std::size_t peak_tensor_rank;
    std::size_t peak_tensor_bytes;
    double scalar_multiplications;
    bool exact;
    std::string backend;
    std::string method;
};

class TensorNetworkCostModel {
public:
    [[nodiscard]] std::uint32_t schema_version() const noexcept;
    [[nodiscard]] std::uint32_t policy_version() const noexcept;
    [[nodiscard]] std::uint32_t workload_version() const noexcept;
    [[nodiscard]] const std::string& engine_version() const noexcept;
    [[nodiscard]] const std::string& host_fingerprint() const noexcept;
    [[nodiscard]] const std::string& artifact_fingerprint() const noexcept;
    [[nodiscard]] std::size_t report_count() const noexcept;
    [[nodiscard]] std::size_t decision_samples() const noexcept;
    [[nodiscard]] std::size_t decision_mistakes() const noexcept;
    [[nodiscard]] double decision_max_regret() const noexcept;
    [[nodiscard]] std::size_t cpu_wins() const noexcept;
    [[nodiscard]] std::size_t tensor_network_wins() const noexcept;
    [[nodiscard]] bool auto_validated() const noexcept;

    [[nodiscard]] double predict_cpu_ns(
        std::size_t active_qubits,
        std::size_t compiled_steps,
        std::size_t two_qubit_operations,
        std::size_t operation_count,
        std::size_t term_count,
        std::size_t threads
    ) const;

    [[nodiscard]] double predict_tensor_network_ns(
        std::size_t contractions,
        std::size_t peak_tensor_rank,
        std::size_t peak_tensor_bytes,
        double scalar_multiplications,
        std::size_t term_count
    ) const;

private:
    std::uint32_t schema_version_ = 0U;
    std::uint32_t policy_version_ = 0U;
    std::uint32_t workload_version_ = 0U;
    std::string engine_version_;
    std::string host_fingerprint_;
    std::string artifact_fingerprint_;
    std::size_t report_count_ = 0U;
    std::size_t decision_samples_ = 0U;
    std::size_t decision_mistakes_ = 0U;
    double decision_max_regret_ = 0.0;
    std::size_t cpu_wins_ = 0U;
    std::size_t tensor_network_wins_ = 0U;
    std::array<double, 6> cpu_coefficients_{};
    std::array<double, 6> tensor_network_coefficients_{};
    double cpu_holdout_median_factor_ = 0.0;
    double cpu_holdout_max_factor_ = 0.0;
    double tensor_network_holdout_median_factor_ = 0.0;
    double tensor_network_holdout_max_factor_ = 0.0;
    bool validated_ = false;

    friend TensorNetworkCostModel load_tensor_network_cost_model(const std::string& path);
};

[[nodiscard]] TensorNetworkCostModel load_tensor_network_cost_model(const std::string& path);

[[nodiscard]] TensorNetworkPlan tensor_network_plan(
    const Program& program,
    const Observable& observable,
    std::size_t max_tensor_bytes = kTensorNetworkDefaultMaxBytes
);

[[nodiscard]] TensorNetworkResult tensor_network_expectation(
    const Program& program,
    const Observable& observable,
    std::size_t max_tensor_bytes = kTensorNetworkDefaultMaxBytes
);

[[nodiscard]] ObservableExecutionPlan tensor_network_observable_plan(
    const Program& program,
    const std::vector<Observable>& observables,
    std::size_t max_tensor_bytes = kTensorNetworkDefaultMaxBytes
);

[[nodiscard]] ObservableExecutionPlan tensor_network_auto_observable_plan(
    const Program& program,
    const std::vector<Observable>& observables,
    const TensorNetworkCostModel& cost_model,
    std::size_t max_tensor_bytes = kTensorNetworkDefaultMaxBytes
);

[[nodiscard]] ObservableResult tensor_network_expect_observable(
    const Program& program,
    const Observable& observable,
    std::size_t max_tensor_bytes = kTensorNetworkDefaultMaxBytes
);

[[nodiscard]] ObservableBatch tensor_network_expect_observables(
    const Program& program,
    const std::vector<Observable>& observables,
    std::size_t max_tensor_bytes = kTensorNetworkDefaultMaxBytes
);

}  // namespace qupy
