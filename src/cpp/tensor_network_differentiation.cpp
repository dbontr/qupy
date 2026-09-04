#include "qupy/tensor_network.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qupy {
namespace {

[[nodiscard]] bool is_rotation(OperationCode code) noexcept {
    return code == OperationCode::RX || code == OperationCode::RY || code == OperationCode::RZ;
}

void validate_slots(
    const Program& program,
    const std::vector<ParameterSlot>& slots,
    const std::vector<double>& parameter_values
) {
    if (slots.size() != parameter_values.size()) {
        throw std::invalid_argument("parameter values must match parameter slots");
    }
    std::set<std::pair<std::size_t, std::size_t>> identities;
    for (const ParameterSlot& slot : slots) {
        if (slot.operation_index >= program.operations().size()) {
            throw std::invalid_argument("parameter slot operation index is outside the program");
        }
        const Operation& operation = program.operations()[slot.operation_index];
        if (!is_rotation(operation.code) || slot.parameter_index != 0U) {
            throw std::invalid_argument("gradient slots must select RX, RY, or RZ parameters");
        }
        if (!identities.insert({slot.operation_index, slot.parameter_index}).second) {
            throw std::invalid_argument("gradient parameter slots must be unique");
        }
    }
    for (const double value : parameter_values) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("bound parameter values must be finite");
        }
    }
}

void validate_observable(const Observable& observable, std::size_t num_qubits) {
    for (const PauliTerm& term : observable.terms()) {
        std::set<std::size_t> qubits;
        for (const PauliFactor& factor : term.factors()) {
            if (factor.qubit >= num_qubits) {
                throw std::invalid_argument("observable qubit is outside this program");
            }
            if (!qubits.insert(factor.qubit).second) {
                throw std::invalid_argument("Pauli term contains duplicate qubit factors");
            }
        }
    }
}

[[nodiscard]] GradientMethod select_method(GradientMethod method, double epsilon) {
    if (method == GradientMethod::Adjoint) {
        throw std::invalid_argument(
            "adjoint differentiation requires the native CPU state-vector backend"
        );
    }
    const GradientMethod selected = method == GradientMethod::Auto
        ? GradientMethod::ParameterShift : method;
    if (selected == GradientMethod::FiniteDifference &&
        (!std::isfinite(epsilon) || epsilon <= 0.0)) {
        throw std::invalid_argument("finite-difference epsilon must be finite and positive");
    }
    return selected;
}

[[nodiscard]] double evaluate(
    const Program& program,
    const Observable& observable,
    std::size_t max_tensor_bytes
) {
    return tensor_network_expectation(program, observable, max_tensor_bytes).value;
}

}  // namespace

GradientResult tensor_network_value_and_grad(
    const Program& program,
    const Observable& observable,
    const std::vector<ParameterSlot>& slots,
    const std::vector<double>& parameter_values,
    GradientMethod method,
    double epsilon,
    std::size_t max_tensor_bytes
) {
    validate_slots(program, slots, parameter_values);
    validate_observable(observable, program.num_qubits());
    static_cast<void>(tensor_network_plan(program, observable, max_tensor_bytes));

    if (slots.empty()) {
        const double value = evaluate(program, observable, max_tensor_bytes);
        return {value, {}, "none", "native-tn", 1U};
    }

    const GradientMethod selected = select_method(method, epsilon);
    const double shift = selected == GradientMethod::ParameterShift
        ? std::acos(-1.0) / 2.0 : epsilon;
    const Program bound = program.bound(slots, parameter_values);
    const double base = evaluate(bound, observable, max_tensor_bytes);
    std::vector<double> gradient(slots.size(), 0.0);
    for (std::size_t index = 0U; index < slots.size(); ++index) {
        std::vector<double> plus = parameter_values;
        std::vector<double> minus = parameter_values;
        plus[index] += shift;
        minus[index] -= shift;
        const double plus_value = evaluate(
            program.bound(slots, plus), observable, max_tensor_bytes
        );
        const double minus_value = evaluate(
            program.bound(slots, minus), observable, max_tensor_bytes
        );
        gradient[index] = selected == GradientMethod::ParameterShift
            ? 0.5 * (plus_value - minus_value)
            : (plus_value - minus_value) / (2.0 * shift);
    }

    return {
        base,
        std::move(gradient),
        selected == GradientMethod::ParameterShift ? "parameter-shift" : "finite-difference",
        "native-tn",
        1U + 2U * slots.size(),
    };
}

JacobianResult tensor_network_jacobian(
    const Program& program,
    const std::vector<Observable>& observables,
    const std::vector<ParameterSlot>& slots,
    const std::vector<double>& parameter_values,
    GradientMethod method,
    double epsilon,
    std::size_t max_tensor_bytes
) {
    validate_slots(program, slots, parameter_values);
    for (const Observable& observable : observables) {
        validate_observable(observable, program.num_qubits());
    }
    if (observables.empty()) {
        return {{}, {}, 0U, slots.size(), "none", "native-tn", 0U};
    }

    std::vector<double> values;
    std::vector<double> derivatives;
    values.reserve(observables.size());
    if (slots.size() != 0U &&
        observables.size() > std::numeric_limits<std::size_t>::max() / slots.size()) {
        throw std::length_error("tensor-network Jacobian shape exceeds native range");
    }
    derivatives.reserve(observables.size() * slots.size());
    std::string selected_method;
    std::size_t evaluations = 0U;
    for (const Observable& observable : observables) {
        GradientResult result = tensor_network_value_and_grad(
            program,
            observable,
            slots,
            parameter_values,
            method,
            epsilon,
            max_tensor_bytes
        );
        values.push_back(result.value);
        derivatives.insert(
            derivatives.end(),
            result.gradient.begin(),
            result.gradient.end()
        );
        evaluations += result.evaluations;
        if (selected_method.empty()) {
            selected_method = result.method;
        } else if (selected_method != result.method) {
            selected_method = "mixed";
        }
    }

    return {
        std::move(values),
        std::move(derivatives),
        observables.size(),
        slots.size(),
        std::move(selected_method),
        "native-tn",
        evaluations,
    };
}

HessianResult tensor_network_hessian(
    const Program& program,
    const Observable& observable,
    const std::vector<ParameterSlot>& slots,
    const std::vector<double>& parameter_values,
    std::size_t max_tensor_bytes
) {
    validate_slots(program, slots, parameter_values);
    validate_observable(observable, program.num_qubits());
    static_cast<void>(tensor_network_plan(program, observable, max_tensor_bytes));
    if (slots.empty()) {
        const double value = evaluate(program, observable, max_tensor_bytes);
        return {value, {}, {}, 0U, "none", "native-tn", 1U};
    }

    GradientResult first = tensor_network_value_and_grad(
        program,
        observable,
        slots,
        parameter_values,
        GradientMethod::ParameterShift,
        1e-7,
        max_tensor_bytes
    );
    const std::size_t count = slots.size();
    if (count > std::numeric_limits<std::size_t>::max() / count) {
        throw std::length_error("tensor-network Hessian shape exceeds native range");
    }
    std::vector<double> matrix(count * count, 0.0);
    const double pi = std::acos(-1.0);
    std::size_t evaluations = first.evaluations;
    const auto shifted_value = [&](const std::vector<double>& parameters) {
        return evaluate(
            program.bound(slots, parameters),
            observable,
            max_tensor_bytes
        );
    };

    for (std::size_t row = 0U; row < count; ++row) {
        std::vector<double> plus = parameter_values;
        std::vector<double> minus = parameter_values;
        plus[row] += pi;
        minus[row] -= pi;
        matrix[row * count + row] =
            (shifted_value(plus) - 2.0 * first.value + shifted_value(minus)) / 4.0;
        evaluations += 2U;

        for (std::size_t column = row + 1U; column < count; ++column) {
            double mixed = 0.0;
            for (const int row_sign : {-1, 1}) {
                for (const int column_sign : {-1, 1}) {
                    std::vector<double> shifted = parameter_values;
                    shifted[row] += static_cast<double>(row_sign) * pi / 2.0;
                    shifted[column] += static_cast<double>(column_sign) * pi / 2.0;
                    mixed += static_cast<double>(row_sign * column_sign) * shifted_value(shifted);
                    ++evaluations;
                }
            }
            mixed /= 4.0;
            matrix[row * count + column] = mixed;
            matrix[column * count + row] = mixed;
        }
    }

    return {
        first.value,
        std::move(first.gradient),
        std::move(matrix),
        count,
        "parameter-shift",
        "native-tn",
        evaluations,
    };
}

}  // namespace qupy
