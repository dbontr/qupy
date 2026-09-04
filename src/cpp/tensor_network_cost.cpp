#include "qupy/tensor_network.hpp"

#include "qupy/detail/fingerprint.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace qupy {
namespace {

constexpr std::uint32_t kTensorNetworkCostSchemaVersion = 1U;
constexpr std::uint32_t kTensorNetworkPolicyVersion = 1U;
constexpr std::uint32_t kTensorNetworkWorkloadVersion = 1U;
constexpr std::size_t kMinimumReports = 3U;
constexpr std::size_t kMinimumDecisionSamples = 18U;
constexpr std::size_t kMinimumBackendWins = 3U;
constexpr double kMaximumDecisionRegret = 1.10;
constexpr double kMaximumHoldoutMedianFactor = 1.5;
constexpr double kMaximumHoldoutFactor = 2.0;
constexpr std::size_t kFeatureCount = 6U;
using Features = std::array<double, kFeatureCount>;

void strip_trailing_carriage_return(std::string& line) {
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
}

void require_no_extra(std::istringstream& fields) {
    std::string extra;
    if (fields >> extra) {
        throw std::invalid_argument(
            "tensor-network cost artifact row contains unexpected data"
        );
    }
}

void validate_holdout(double median_factor, double max_factor) {
    if (!std::isfinite(median_factor) || !std::isfinite(max_factor) ||
        median_factor < 1.0 || max_factor < 1.0 ||
        median_factor > kMaximumHoldoutMedianFactor ||
        max_factor > kMaximumHoldoutFactor) {
        throw std::invalid_argument(
            "tensor-network cost artifact has invalid holdout metrics"
        );
    }
}

[[nodiscard]] Features cpu_features(
    std::size_t active_qubits,
    std::size_t compiled_steps,
    std::size_t two_qubit_operations,
    std::size_t operation_count,
    std::size_t term_count,
    std::size_t threads
) {
    const double operation_denominator = static_cast<double>(
        std::max<std::size_t>(operation_count, 1U)
    );
    return {
        1.0,
        static_cast<double>(active_qubits),
        std::log1p(static_cast<double>(compiled_steps)),
        static_cast<double>(two_qubit_operations) / operation_denominator,
        std::log1p(static_cast<double>(term_count)),
        std::log(static_cast<double>(threads)),
    };
}

[[nodiscard]] Features tensor_network_features(
    std::size_t contractions,
    std::size_t peak_tensor_rank,
    std::size_t peak_tensor_bytes,
    double scalar_multiplications,
    std::size_t term_count
) {
    return {
        1.0,
        std::log1p(scalar_multiplications),
        std::log1p(static_cast<double>(contractions)),
        static_cast<double>(peak_tensor_rank),
        std::log1p(static_cast<double>(peak_tensor_bytes)),
        std::log1p(static_cast<double>(term_count)),
    };
}

[[nodiscard]] bool features_in_domain(
    const Features& features,
    const Features& minimum,
    const Features& maximum
) noexcept {
    for (std::size_t index = 0U; index < features.size(); ++index) {
        if (!std::isfinite(features[index]) ||
            features[index] < minimum[index] ||
            features[index] > maximum[index]) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] double evaluate_log_model(
    const Features& coefficients,
    const Features& features
) {
    double score = 0.0;
    for (std::size_t index = 0U; index < coefficients.size(); ++index) {
        score += coefficients[index] * features[index];
    }
    const double prediction = std::exp(score);
    if (!std::isfinite(prediction) || prediction <= 0.0) {
        throw std::overflow_error(
            "tensor-network cost model produced an invalid runtime prediction"
        );
    }
    return prediction;
}

[[nodiscard]] std::size_t checked_sum(
    std::size_t left,
    std::size_t right,
    const char* label
) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        throw std::length_error(std::string(label) + " exceeds native range");
    }
    return left + right;
}

}  // namespace

std::uint32_t TensorNetworkCostModel::schema_version() const noexcept {
    return schema_version_;
}

std::uint32_t TensorNetworkCostModel::policy_version() const noexcept {
    return policy_version_;
}

std::uint32_t TensorNetworkCostModel::workload_version() const noexcept {
    return workload_version_;
}

const std::string& TensorNetworkCostModel::engine_version() const noexcept {
    return engine_version_;
}

const std::string& TensorNetworkCostModel::host_fingerprint() const noexcept {
    return host_fingerprint_;
}

const std::string& TensorNetworkCostModel::artifact_fingerprint() const noexcept {
    return artifact_fingerprint_;
}

std::size_t TensorNetworkCostModel::report_count() const noexcept {
    return report_count_;
}

std::size_t TensorNetworkCostModel::decision_samples() const noexcept {
    return decision_samples_;
}

std::size_t TensorNetworkCostModel::decision_mistakes() const noexcept {
    return decision_mistakes_;
}

double TensorNetworkCostModel::decision_max_regret() const noexcept {
    return decision_max_regret_;
}

std::size_t TensorNetworkCostModel::cpu_wins() const noexcept {
    return cpu_wins_;
}

std::size_t TensorNetworkCostModel::tensor_network_wins() const noexcept {
    return tensor_network_wins_;
}

bool TensorNetworkCostModel::auto_validated() const noexcept {
    return validated_ && has_cpu_domain_ && has_tensor_network_domain_ &&
           schema_version_ == kTensorNetworkCostSchemaVersion &&
           policy_version_ == kTensorNetworkPolicyVersion &&
           workload_version_ == kTensorNetworkWorkloadVersion &&
           report_count_ >= kMinimumReports &&
           decision_samples_ >= kMinimumDecisionSamples &&
           decision_mistakes_ == 0U &&
           decision_max_regret_ >= 1.0 &&
           decision_max_regret_ <= kMaximumDecisionRegret &&
           cpu_wins_ >= kMinimumBackendWins &&
           tensor_network_wins_ >= kMinimumBackendWins &&
           cpu_wins_ + tensor_network_wins_ == decision_samples_ &&
           cpu_holdout_median_factor_ >= 1.0 &&
           cpu_holdout_median_factor_ <= kMaximumHoldoutMedianFactor &&
           cpu_holdout_max_factor_ >= 1.0 &&
           cpu_holdout_max_factor_ <= kMaximumHoldoutFactor &&
           tensor_network_holdout_median_factor_ >= 1.0 &&
           tensor_network_holdout_median_factor_ <= kMaximumHoldoutMedianFactor &&
           tensor_network_holdout_max_factor_ >= 1.0 &&
           tensor_network_holdout_max_factor_ <= kMaximumHoldoutFactor;
}

bool TensorNetworkCostModel::cpu_in_domain(
    std::size_t active_qubits,
    std::size_t compiled_steps,
    std::size_t two_qubit_operations,
    std::size_t operation_count,
    std::size_t term_count,
    std::size_t threads
) const noexcept {
    if (!auto_validated() || term_count == 0U || threads == 0U ||
        two_qubit_operations > operation_count) {
        return false;
    }
    return features_in_domain(
        cpu_features(
            active_qubits,
            compiled_steps,
            two_qubit_operations,
            operation_count,
            term_count,
            threads
        ),
        cpu_feature_min_,
        cpu_feature_max_
    );
}

bool TensorNetworkCostModel::tensor_network_in_domain(
    std::size_t contractions,
    std::size_t peak_tensor_rank,
    std::size_t peak_tensor_bytes,
    double scalar_multiplications,
    std::size_t term_count
) const noexcept {
    if (!auto_validated() || term_count == 0U ||
        !std::isfinite(scalar_multiplications) || scalar_multiplications < 0.0) {
        return false;
    }
    return features_in_domain(
        tensor_network_features(
            contractions,
            peak_tensor_rank,
            peak_tensor_bytes,
            scalar_multiplications,
            term_count
        ),
        tensor_network_feature_min_,
        tensor_network_feature_max_
    );
}

double TensorNetworkCostModel::predict_cpu_ns(
    std::size_t active_qubits,
    std::size_t compiled_steps,
    std::size_t two_qubit_operations,
    std::size_t operation_count,
    std::size_t term_count,
    std::size_t threads
) const {
    if (!cpu_in_domain(
            active_qubits,
            compiled_steps,
            two_qubit_operations,
            operation_count,
            term_count,
            threads
        )) {
        throw std::invalid_argument(
            "tensor-network CPU prediction is outside the calibrated domain"
        );
    }
    return evaluate_log_model(
        cpu_coefficients_,
        cpu_features(
            active_qubits,
            compiled_steps,
            two_qubit_operations,
            operation_count,
            term_count,
            threads
        )
    );
}

double TensorNetworkCostModel::predict_tensor_network_ns(
    std::size_t contractions,
    std::size_t peak_tensor_rank,
    std::size_t peak_tensor_bytes,
    double scalar_multiplications,
    std::size_t term_count
) const {
    if (!tensor_network_in_domain(
            contractions,
            peak_tensor_rank,
            peak_tensor_bytes,
            scalar_multiplications,
            term_count
        )) {
        throw std::invalid_argument(
            "tensor-network prediction is outside the calibrated domain"
        );
    }
    return evaluate_log_model(
        tensor_network_coefficients_,
        tensor_network_features(
            contractions,
            peak_tensor_rank,
            peak_tensor_bytes,
            scalar_multiplications,
            term_count
        )
    );
}

TensorNetworkCostModel load_tensor_network_cost_model(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::invalid_argument(
            "cannot open tensor-network cost artifact: " + path
        );
    }
    const std::string text{
        std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()
    };
    std::istringstream lines(text);
    std::string line;
    if (!std::getline(lines, line)) {
        throw std::invalid_argument(
            "tensor-network cost artifact has an unsupported schema"
        );
    }
    strip_trailing_carriage_return(line);
    if (line != "qupy-tensor-network-cost 1") {
        throw std::invalid_argument(
            "tensor-network cost artifact has an unsupported schema"
        );
    }

    TensorNetworkCostModel model;
    model.schema_version_ = kTensorNetworkCostSchemaVersion;
    bool has_engine = false;
    bool has_workload = false;
    bool has_host = false;
    bool has_policy = false;
    bool has_reports = false;
    bool has_decision = false;
    bool has_validated = false;
    bool has_cpu_model = false;
    bool has_tensor_network_model = false;

    while (std::getline(lines, line)) {
        strip_trailing_carriage_return(line);
        if (line.empty()) {
            continue;
        }
        std::istringstream fields(line);
        std::string key;
        fields >> key;
        if (key == "engine") {
            if (has_engine || !(fields >> model.engine_version_)) {
                throw std::invalid_argument(
                    "tensor-network cost artifact has invalid engine metadata"
                );
            }
            has_engine = true;
        } else if (key == "workload") {
            if (has_workload || !(fields >> model.workload_version_)) {
                throw std::invalid_argument(
                    "tensor-network cost artifact has invalid workload metadata"
                );
            }
            has_workload = true;
        } else if (key == "host") {
            if (has_host || !(fields >> model.host_fingerprint_)) {
                throw std::invalid_argument(
                    "tensor-network cost artifact has invalid host metadata"
                );
            }
            has_host = true;
        } else if (key == "policy") {
            if (has_policy || !(fields >> model.policy_version_) ||
                model.policy_version_ != kTensorNetworkPolicyVersion) {
                throw std::invalid_argument(
                    "tensor-network cost artifact has invalid policy metadata"
                );
            }
            has_policy = true;
        } else if (key == "reports") {
            if (has_reports || !(fields >> model.report_count_) ||
                model.report_count_ < kMinimumReports) {
                throw std::invalid_argument(
                    "tensor-network cost artifact has insufficient report evidence"
                );
            }
            has_reports = true;
        } else if (key == "decision") {
            if (has_decision ||
                !(fields >> model.decision_samples_ >> model.decision_mistakes_ >>
                  model.decision_max_regret_ >> model.cpu_wins_ >>
                  model.tensor_network_wins_) ||
                model.decision_samples_ < kMinimumDecisionSamples ||
                model.decision_mistakes_ != 0U ||
                !std::isfinite(model.decision_max_regret_) ||
                model.decision_max_regret_ < 1.0 ||
                model.decision_max_regret_ > kMaximumDecisionRegret ||
                model.cpu_wins_ < kMinimumBackendWins ||
                model.tensor_network_wins_ < kMinimumBackendWins ||
                model.cpu_wins_ + model.tensor_network_wins_ != model.decision_samples_) {
                throw std::invalid_argument(
                    "tensor-network cost artifact has invalid decision evidence"
                );
            }
            has_decision = true;
        } else if (key == "model") {
            std::string cost_class;
            std::size_t coefficient_count = 0U;
            if (!(fields >> cost_class >> coefficient_count) ||
                coefficient_count != kFeatureCount) {
                throw std::invalid_argument(
                    "tensor-network cost artifact has an invalid model class"
                );
            }
            Features* coefficients = nullptr;
            double* holdout_median = nullptr;
            double* holdout_max = nullptr;
            if (cost_class == "tensor-network-baseline-cpu" && !has_cpu_model) {
                coefficients = &model.cpu_coefficients_;
                holdout_median = &model.cpu_holdout_median_factor_;
                holdout_max = &model.cpu_holdout_max_factor_;
                has_cpu_model = true;
            } else if (cost_class == "tensor-network-return-cpu" &&
                       !has_tensor_network_model) {
                coefficients = &model.tensor_network_coefficients_;
                holdout_median = &model.tensor_network_holdout_median_factor_;
                holdout_max = &model.tensor_network_holdout_max_factor_;
                has_tensor_network_model = true;
            } else {
                throw std::invalid_argument(
                    "tensor-network cost artifact has an invalid model class"
                );
            }
            for (double& coefficient : *coefficients) {
                if (!(fields >> coefficient) || !std::isfinite(coefficient)) {
                    throw std::invalid_argument(
                        "tensor-network cost artifact has an invalid coefficient"
                    );
                }
            }
            if (!(fields >> *holdout_median >> *holdout_max)) {
                throw std::invalid_argument(
                    "tensor-network cost artifact has malformed holdout metrics"
                );
            }
            validate_holdout(*holdout_median, *holdout_max);
        } else if (key == "domain") {
            std::string cost_class;
            std::size_t feature_count = 0U;
            Features* minimum = nullptr;
            Features* maximum = nullptr;
            bool* present = nullptr;
            if (!(fields >> cost_class >> feature_count) ||
                feature_count != kFeatureCount) {
                throw std::invalid_argument(
                    "tensor-network cost artifact has an invalid domain class"
                );
            }
            if (cost_class == "tensor-network-baseline-cpu" &&
                !model.has_cpu_domain_) {
                minimum = &model.cpu_feature_min_;
                maximum = &model.cpu_feature_max_;
                present = &model.has_cpu_domain_;
            } else if (cost_class == "tensor-network-return-cpu" &&
                       !model.has_tensor_network_domain_) {
                minimum = &model.tensor_network_feature_min_;
                maximum = &model.tensor_network_feature_max_;
                present = &model.has_tensor_network_domain_;
            } else {
                throw std::invalid_argument(
                    "tensor-network cost artifact has an invalid domain class"
                );
            }
            for (std::size_t index = 0U; index < kFeatureCount; ++index) {
                if (!(fields >> (*minimum)[index] >> (*maximum)[index]) ||
                    !std::isfinite((*minimum)[index]) ||
                    !std::isfinite((*maximum)[index]) ||
                    (*minimum)[index] > (*maximum)[index]) {
                    throw std::invalid_argument(
                        "tensor-network cost artifact has invalid feature bounds"
                    );
                }
            }
            *present = true;
        } else if (key == "validated") {
            int value = 0;
            if (has_validated || !(fields >> value) || value != 1) {
                throw std::invalid_argument(
                    "tensor-network cost artifact is not validated"
                );
            }
            model.validated_ = true;
            has_validated = true;
        } else {
            throw std::invalid_argument(
                "tensor-network cost artifact contains an unknown field"
            );
        }
        require_no_extra(fields);
    }

    if (!has_engine || !has_workload || !has_host || !has_policy ||
        !has_reports || !has_decision || !has_cpu_model ||
        !has_tensor_network_model || !model.has_cpu_domain_ ||
        !model.has_tensor_network_domain_ || !has_validated) {
        throw std::invalid_argument("tensor-network cost artifact is incomplete");
    }
    if (model.engine_version_ != core_version()) {
        throw std::invalid_argument(
            "tensor-network cost artifact engine version does not match this runtime"
        );
    }
    if (model.workload_version_ != kTensorNetworkWorkloadVersion) {
        throw std::invalid_argument(
            "tensor-network cost artifact workload version does not match this runtime"
        );
    }
    if (model.host_fingerprint_ != planner_host_fingerprint()) {
        throw std::invalid_argument(
            "tensor-network cost artifact host does not match this runtime"
        );
    }
    if (!model.auto_validated()) {
        throw std::invalid_argument(
            "tensor-network cost artifact did not meet automatic routing gates"
        );
    }
    model.artifact_fingerprint_ = detail::fingerprint_text(text);
    return model;
}

ObservableExecutionPlan tensor_network_auto_observable_plan(
    const Program& program,
    const std::vector<Observable>& observables,
    const TensorNetworkCostModel& cost_model,
    std::size_t max_tensor_bytes
) {
    if (!cost_model.auto_validated()) {
        throw std::invalid_argument("tensor-network cost model is not validated");
    }
    if (max_tensor_bytes == 0U) {
        throw std::invalid_argument("max_tensor_bytes must be positive");
    }

    ObservableExecutionPlan cpu_query = observable_plan(
        program,
        observables,
        "native-cpu",
        nullptr
    );
    std::size_t term_count = 0U;
    for (const Observable& observable : observables) {
        term_count = checked_sum(
            term_count,
            observable.terms().size(),
            "observable term count"
        );
    }

    const bool calibrated_dense_cpu =
        !observables.empty() && term_count > 0U &&
        cpu_query.backend == "native-cpu" &&
        cpu_query.method != "pauli-propagation" &&
        cpu_query.active_qubits == program.num_qubits();
    if (!calibrated_dense_cpu) {
        return cpu_query;
    }

    const ExecutionPlan cpu_state = plan(
        program,
        ResultMode::StateVector,
        "native-cpu",
        nullptr
    );
    if (cpu_state.backend != "native-cpu") {
        throw std::logic_error(
            "explicit native CPU tensor-network baseline planning changed backend"
        );
    }

    std::size_t two_qubit_operations = 0U;
    for (const Operation& operation : program.operations()) {
        if (operation.qubits.size() == 2U) {
            ++two_qubit_operations;
        }
    }

    std::size_t contractions = 0U;
    std::size_t peak_tensor_rank = 0U;
    std::size_t peak_tensor_bytes = 0U;
    double scalar_multiplications = 0.0;
    std::string tensor_plan_identity;
    try {
        for (const Observable& observable : observables) {
            const TensorNetworkPlan tensor_plan = tensor_network_plan(
                program,
                observable,
                max_tensor_bytes
            );
            contractions = checked_sum(
                contractions,
                tensor_plan.contractions,
                "tensor-network contraction count"
            );
            peak_tensor_rank = std::max(
                peak_tensor_rank,
                tensor_plan.peak_tensor_rank
            );
            peak_tensor_bytes = std::max(
                peak_tensor_bytes,
                tensor_plan.peak_tensor_bytes
            );
            scalar_multiplications += tensor_plan.scalar_multiplications;
            if (!std::isfinite(scalar_multiplications)) {
                throw std::overflow_error(
                    "tensor-network scalar multiplication work exceeds native range"
                );
            }
            tensor_plan_identity += tensor_plan.plan_fingerprint + "\n";
        }
    } catch (const std::length_error&) {
        return cpu_query;
    }

    const std::size_t operation_count = program.operations().size();
    if (!cost_model.cpu_in_domain(
            cpu_query.active_qubits,
            cpu_state.compiled_steps,
            two_qubit_operations,
            operation_count,
            term_count,
            cpu_state.threads
        ) ||
        !cost_model.tensor_network_in_domain(
            contractions,
            peak_tensor_rank,
            peak_tensor_bytes,
            scalar_multiplications,
            term_count
        )) {
        return cpu_query;
    }

    const double cpu_prediction = cost_model.predict_cpu_ns(
        cpu_query.active_qubits,
        cpu_state.compiled_steps,
        two_qubit_operations,
        operation_count,
        term_count,
        cpu_state.threads
    );
    const double tensor_prediction = cost_model.predict_tensor_network_ns(
        contractions,
        peak_tensor_rank,
        peak_tensor_bytes,
        scalar_multiplications,
        term_count
    );

    ObservableExecutionPlan selected = cpu_query;
    if (tensor_prediction < cpu_prediction) {
        selected = tensor_network_observable_plan(
            program,
            observables,
            max_tensor_bytes
        );
        selected.predicted_ns = tensor_prediction;
        selected.cost_model_class = "tensor-network-return-cpu";
    } else {
        selected.predicted_ns = cpu_prediction;
        selected.cost_model_class = "tensor-network-baseline-cpu";
    }
    selected.cost_model_fingerprint = cost_model.artifact_fingerprint();
    selected.cache_key = detail::fingerprint_text(
        "qupy-tensor-network-auto-observable 1\nbase " + selected.cache_key +
        "\nmodel " + cost_model.artifact_fingerprint() +
        "\ntensor-plan " + detail::fingerprint_text(tensor_plan_identity) + "\n"
    );
    return selected;
}

}  // namespace qupy
