#include "qupy/tensor_network.hpp"

#include "qupy/detail/fingerprint.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qupy {
namespace {

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

[[nodiscard]] std::string observable_query_fingerprint(
    const std::vector<Observable>& observables
) {
    std::string text = "qupy-observable-query 1\n";
    for (const Observable& observable : observables) {
        text += "observable " + observable.fingerprint() + "\n";
    }
    return detail::fingerprint_text(text);
}

}  // namespace

ObservableExecutionPlan tensor_network_observable_plan(
    const Program& program,
    const std::vector<Observable>& observables,
    std::size_t max_tensor_bytes
) {
    if (max_tensor_bytes == 0U) {
        throw std::invalid_argument("max_tensor_bytes must be positive");
    }

    std::size_t term_count = 0U;
    std::size_t measurement_group_count = 0U;
    std::size_t peak_tensor_bytes = 0U;
    std::string plan_identity = "qupy-tensor-network-observable-plan 1\n";
    plan_identity += "program " + program.fingerprint() + "\n";
    plan_identity += "max-tensor-bytes " + std::to_string(max_tensor_bytes) + "\n";

    for (const Observable& observable : observables) {
        term_count = checked_sum(
            term_count,
            observable.terms().size(),
            "observable term count"
        );
        measurement_group_count = checked_sum(
            measurement_group_count,
            measurement_groups(observable).size(),
            "observable measurement-group count"
        );
        const TensorNetworkPlan plan = tensor_network_plan(
            program,
            observable,
            max_tensor_bytes
        );
        peak_tensor_bytes = std::max(peak_tensor_bytes, plan.peak_tensor_bytes);
        plan_identity += "plan " + plan.plan_fingerprint + "\n";
    }

    const std::string query_fingerprint = observable_query_fingerprint(observables);
    const std::string cache_key = detail::fingerprint_text(
        plan_identity + "query " + query_fingerprint +
        "\nbackend native-tn\nmethod greedy-contraction-observable\n"
    );

    return {
        "native-tn",
        "greedy-contraction-observable",
        true,
        program.num_qubits(),
        observables.size(),
        term_count,
        measurement_group_count,
        peak_tensor_bytes,
        program.fingerprint(),
        query_fingerprint,
        cache_key,
        std::nullopt,
        {},
        {},
    };
}

ObservableResult tensor_network_expect_observable(
    const Program& program,
    const Observable& observable,
    std::size_t max_tensor_bytes
) {
    static_cast<void>(tensor_network_plan(program, observable, max_tensor_bytes));
    const TensorNetworkResult result = tensor_network_expectation(
        program,
        observable,
        max_tensor_bytes
    );
    return {
        result.value,
        "native-tn",
        program.num_qubits(),
        result.term_count,
    };
}

ObservableBatch tensor_network_expect_observables(
    const Program& program,
    const std::vector<Observable>& observables,
    std::size_t max_tensor_bytes
) {
    static_cast<void>(tensor_network_observable_plan(
        program,
        observables,
        max_tensor_bytes
    ));

    std::vector<double> values;
    values.reserve(observables.size());
    for (const Observable& observable : observables) {
        values.push_back(
            tensor_network_expectation(program, observable, max_tensor_bytes).value
        );
    }
    return {
        std::move(values),
        observables.size(),
        "native-tn",
        observables.empty() ? 0U : program.num_qubits(),
    };
}

}  // namespace qupy
