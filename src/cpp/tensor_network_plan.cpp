#include "qupy/tensor_network.hpp"

#include "qupy/detail/fingerprint.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qupy {
namespace {

struct TensorShape {
    std::vector<std::size_t> indices;
    std::size_t id;
};

struct PlanStats {
    std::size_t contractions = 0U;
    std::size_t peak_tensor_rank = 0U;
    std::size_t peak_tensor_bytes = 0U;
    double scalar_multiplications = 0.0;
};

struct PairChoice {
    std::size_t first = 0U;
    std::size_t second = 0U;
    std::size_t work_rank = std::numeric_limits<std::size_t>::max();
    std::size_t output_rank = std::numeric_limits<std::size_t>::max();
    bool found = false;
};

[[nodiscard]] std::size_t checked_value_count(std::size_t rank) {
    if (rank >= std::numeric_limits<std::size_t>::digits) {
        throw std::length_error("tensor rank exceeds native address space");
    }
    return std::size_t{1} << rank;
}

[[nodiscard]] std::size_t tensor_bytes(std::size_t rank) {
    const std::size_t values = checked_value_count(rank);
    if (values > std::numeric_limits<std::size_t>::max() / sizeof(Complex)) {
        throw std::length_error("tensor storage exceeds native address space");
    }
    return values * sizeof(Complex);
}

void observe_shape(
    const TensorShape& shape,
    std::size_t max_tensor_bytes,
    PlanStats& stats
) {
    const std::size_t bytes = tensor_bytes(shape.indices.size());
    if (bytes > max_tensor_bytes) {
        throw std::length_error("tensor contraction exceeds max_tensor_bytes");
    }
    stats.peak_tensor_rank = std::max(stats.peak_tensor_rank, shape.indices.size());
    stats.peak_tensor_bytes = std::max(stats.peak_tensor_bytes, bytes);
}

void append_shape(
    std::vector<TensorShape>& shapes,
    TensorShape shape,
    std::size_t max_tensor_bytes,
    PlanStats& stats
) {
    observe_shape(shape, max_tensor_bytes, stats);
    shapes.push_back(std::move(shape));
}

[[nodiscard]] std::vector<std::size_t> build_circuit_side(
    const Program& program,
    std::size_t& next_index,
    std::size_t& next_tensor_id,
    std::vector<TensorShape>& shapes,
    std::size_t max_tensor_bytes,
    PlanStats& stats
) {
    std::vector<std::size_t> current(program.num_qubits());
    for (std::size_t qubit = 0U; qubit < program.num_qubits(); ++qubit) {
        const std::size_t index = next_index++;
        current[qubit] = index;
        append_shape(
            shapes,
            TensorShape{{index}, next_tensor_id++},
            max_tensor_bytes,
            stats
        );
    }

    for (const Operation& operation : program.operations()) {
        if (operation.qubits.size() == 1U) {
            const std::size_t qubit = operation.qubits.front();
            const std::size_t input = current[qubit];
            const std::size_t output = next_index++;
            append_shape(
                shapes,
                TensorShape{{input, output}, next_tensor_id++},
                max_tensor_bytes,
                stats
            );
            current[qubit] = output;
            continue;
        }
        if (operation.qubits.size() != 2U) {
            throw std::invalid_argument("tensor network supports one- and two-qubit operations");
        }
        const std::size_t first = operation.qubits[0];
        const std::size_t second = operation.qubits[1];
        const std::size_t input_first = current[first];
        const std::size_t input_second = current[second];
        const std::size_t output_first = next_index++;
        const std::size_t output_second = next_index++;
        append_shape(
            shapes,
            TensorShape{
                {input_first, input_second, output_first, output_second},
                next_tensor_id++,
            },
            max_tensor_bytes,
            stats
        );
        current[first] = output_first;
        current[second] = output_second;
    }
    return current;
}

void validate_term(const PauliTerm& term, std::size_t num_qubits) {
    std::vector<bool> occupied(num_qubits, false);
    for (const PauliFactor& factor : term.factors()) {
        if (factor.qubit >= num_qubits) {
            throw std::invalid_argument("observable qubit is outside this program");
        }
        if (occupied[factor.qubit]) {
            throw std::invalid_argument("Pauli term contains duplicate qubit factors");
        }
        occupied[factor.qubit] = true;
    }
}

[[nodiscard]] std::vector<TensorShape> build_expectation_shapes(
    const Program& program,
    const PauliTerm& term,
    std::size_t max_tensor_bytes,
    PlanStats& stats
) {
    validate_term(term, program.num_qubits());
    std::vector<TensorShape> shapes;
    shapes.reserve(
        2U * (program.num_qubits() + program.operations().size()) + program.num_qubits()
    );
    std::size_t next_index = 0U;
    std::size_t next_tensor_id = 0U;
    const std::vector<std::size_t> ket = build_circuit_side(
        program, next_index, next_tensor_id, shapes, max_tensor_bytes, stats
    );
    const std::vector<std::size_t> bra = build_circuit_side(
        program, next_index, next_tensor_id, shapes, max_tensor_bytes, stats
    );
    for (std::size_t qubit = 0U; qubit < program.num_qubits(); ++qubit) {
        append_shape(
            shapes,
            TensorShape{{ket[qubit], bra[qubit]}, next_tensor_id++},
            max_tensor_bytes,
            stats
        );
    }
    return shapes;
}

[[nodiscard]] bool contains_index(
    const std::vector<std::size_t>& indices,
    std::size_t value
) {
    return std::find(indices.begin(), indices.end(), value) != indices.end();
}

[[nodiscard]] std::vector<std::size_t> shared_indices(
    const TensorShape& first,
    const TensorShape& second
) {
    std::vector<std::size_t> shared;
    for (const std::size_t index : first.indices) {
        if (contains_index(second.indices, index)) {
            shared.push_back(index);
        }
    }
    return shared;
}

[[nodiscard]] PairChoice choose_pair(const std::vector<TensorShape>& shapes) {
    PairChoice choice;
    for (std::size_t first = 0U; first < shapes.size(); ++first) {
        for (std::size_t second = first + 1U; second < shapes.size(); ++second) {
            const std::size_t shared = shared_indices(shapes[first], shapes[second]).size();
            if (shared == 0U) {
                continue;
            }
            const std::size_t output_rank = shapes[first].indices.size() +
                shapes[second].indices.size() - 2U * shared;
            const std::size_t work_rank = output_rank + shared;
            const bool better = !choice.found || work_rank < choice.work_rank ||
                (work_rank == choice.work_rank && output_rank < choice.output_rank) ||
                (work_rank == choice.work_rank && output_rank == choice.output_rank &&
                 std::pair{shapes[first].id, shapes[second].id} <
                     std::pair{shapes[choice.first].id, shapes[choice.second].id});
            if (better) {
                choice = {first, second, work_rank, output_rank, true};
            }
        }
    }
    return choice;
}

[[nodiscard]] TensorShape combine_shapes(
    const TensorShape& first,
    const TensorShape& second,
    std::size_t next_tensor_id,
    std::size_t max_tensor_bytes,
    PlanStats& stats
) {
    const std::vector<std::size_t> shared = shared_indices(first, second);
    if (shared.empty()) {
        throw std::logic_error("tensor pair does not share a contraction index");
    }
    std::vector<std::size_t> output_indices;
    output_indices.reserve(first.indices.size() + second.indices.size() - 2U * shared.size());
    for (const std::size_t index : first.indices) {
        if (!contains_index(shared, index)) {
            output_indices.push_back(index);
        }
    }
    for (const std::size_t index : second.indices) {
        if (!contains_index(shared, index)) {
            output_indices.push_back(index);
        }
    }

    TensorShape result{std::move(output_indices), next_tensor_id};
    observe_shape(result, max_tensor_bytes, stats);
    const std::size_t work_rank = result.indices.size() + shared.size();
    stats.scalar_multiplications += static_cast<double>(checked_value_count(work_rank));
    ++stats.contractions;
    return result;
}

void absorb_scalar(
    std::vector<TensorShape>& shapes,
    std::size_t scalar_index,
    PlanStats& stats
) {
    if (!shapes[scalar_index].indices.empty()) {
        throw std::logic_error("tensor scalar absorption received a non-scalar tensor");
    }
    if (shapes.size() < 2U) {
        return;
    }
    const std::size_t target = scalar_index == 0U ? 1U : 0U;
    stats.scalar_multiplications += static_cast<double>(
        checked_value_count(shapes[target].indices.size())
    );
    ++stats.contractions;
    shapes.erase(shapes.begin() + static_cast<std::ptrdiff_t>(scalar_index));
}

void plan_network(
    std::vector<TensorShape> shapes,
    std::size_t max_tensor_bytes,
    PlanStats& stats
) {
    if (shapes.empty()) {
        return;
    }
    std::size_t next_tensor_id = shapes.size();
    while (shapes.size() > 1U) {
        const PairChoice choice = choose_pair(shapes);
        if (!choice.found) {
            const auto scalar = std::find_if(
                shapes.begin(), shapes.end(), [](const TensorShape& shape) {
                    return shape.indices.empty();
                }
            );
            if (scalar == shapes.end()) {
                throw std::logic_error("closed tensor network became disconnected with open indices");
            }
            absorb_scalar(
                shapes,
                static_cast<std::size_t>(scalar - shapes.begin()),
                stats
            );
            continue;
        }
        TensorShape combined = combine_shapes(
            shapes[choice.first],
            shapes[choice.second],
            next_tensor_id++,
            max_tensor_bytes,
            stats
        );
        shapes.erase(shapes.begin() + static_cast<std::ptrdiff_t>(choice.second));
        shapes.erase(shapes.begin() + static_cast<std::ptrdiff_t>(choice.first));
        shapes.push_back(std::move(combined));
    }
    if (!shapes.front().indices.empty()) {
        throw std::logic_error("tensor contraction plan did not produce a scalar");
    }
}

}  // namespace

TensorNetworkPlan tensor_network_plan(
    const Program& program,
    const Observable& observable,
    std::size_t max_tensor_bytes
) {
    if (max_tensor_bytes == 0U) {
        throw std::invalid_argument("max_tensor_bytes must be positive");
    }

    PlanStats total;
    for (const PauliTerm& term : observable.terms()) {
        if (term.coefficient() == 0.0) {
            continue;
        }
        PlanStats term_stats;
        std::vector<TensorShape> shapes = build_expectation_shapes(
            program, term, max_tensor_bytes, term_stats
        );
        plan_network(std::move(shapes), max_tensor_bytes, term_stats);
        total.contractions += term_stats.contractions;
        total.peak_tensor_rank = std::max(
            total.peak_tensor_rank,
            term_stats.peak_tensor_rank
        );
        total.peak_tensor_bytes = std::max(
            total.peak_tensor_bytes,
            term_stats.peak_tensor_bytes
        );
        total.scalar_multiplications += term_stats.scalar_multiplications;
    }

    const std::string program_fingerprint = program.fingerprint();
    const std::string observable_fingerprint = observable.fingerprint();
    const std::string plan_fingerprint = detail::fingerprint_text(
        "qupy-tensor-network-plan 1\nprogram " + program_fingerprint +
        "\nobservable " + observable_fingerprint +
        "\nmax-tensor-bytes " + std::to_string(max_tensor_bytes) +
        "\nmethod greedy-contraction\n"
    );

    return {
        observable.terms().size(),
        total.contractions,
        total.peak_tensor_rank,
        total.peak_tensor_bytes,
        total.scalar_multiplications,
        max_tensor_bytes,
        true,
        "native-tn",
        "greedy-contraction",
        program_fingerprint,
        observable_fingerprint,
        plan_fingerprint,
    };
}

}  // namespace qupy
