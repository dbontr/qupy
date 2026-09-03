#pragma once

#include "qupy/advanced.hpp"

#include <cstddef>
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

[[nodiscard]] GradientResult tensor_network_value_and_grad(
    const Program& program,
    const Observable& observable,
    const std::vector<ParameterSlot>& slots,
    const std::vector<double>& parameter_values,
    GradientMethod method = GradientMethod::Auto,
    double epsilon = 1e-7,
    std::size_t max_tensor_bytes = kTensorNetworkDefaultMaxBytes
);

[[nodiscard]] JacobianResult tensor_network_jacobian(
    const Program& program,
    const std::vector<Observable>& observables,
    const std::vector<ParameterSlot>& slots,
    const std::vector<double>& parameter_values,
    GradientMethod method = GradientMethod::Auto,
    double epsilon = 1e-7,
    std::size_t max_tensor_bytes = kTensorNetworkDefaultMaxBytes
);

[[nodiscard]] HessianResult tensor_network_hessian(
    const Program& program,
    const Observable& observable,
    const std::vector<ParameterSlot>& slots,
    const std::vector<double>& parameter_values,
    std::size_t max_tensor_bytes = kTensorNetworkDefaultMaxBytes
);

}  // namespace qupy
