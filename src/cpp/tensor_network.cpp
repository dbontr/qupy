#include "qupy/tensor_network.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qupy {
namespace {

using Matrix2 = std::array<Complex, 4>;
constexpr double kHermitianTolerance = 1e-10;

struct Tensor {
    std::vector<std::size_t> indices;
    std::vector<Complex> values;
    std::size_t id;
};

struct ContractionStats {
    std::size_t contractions = 0U;
    std::size_t peak_tensor_rank = 0U;
    std::size_t peak_tensor_bytes = 0U;
    double scalar_multiplications = 0.0;
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

void observe_tensor(
    const Tensor& tensor,
    std::size_t max_tensor_bytes,
    ContractionStats& stats
) {
    const std::size_t bytes = tensor_bytes(tensor.indices.size());
    if (tensor.values.size() != checked_value_count(tensor.indices.size())) {
        throw std::logic_error("tensor value storage does not match tensor rank");
    }
    if (bytes > max_tensor_bytes) {
        throw std::length_error("tensor contraction exceeds max_tensor_bytes");
    }
    stats.peak_tensor_rank = std::max(stats.peak_tensor_rank, tensor.indices.size());
    stats.peak_tensor_bytes = std::max(stats.peak_tensor_bytes, bytes);
}

[[nodiscard]] Matrix2 gate_matrix(const Operation& operation) {
    const double inv_sqrt_two = 1.0 / std::sqrt(2.0);
    switch (operation.code) {
    case OperationCode::H:
        return {inv_sqrt_two, inv_sqrt_two, inv_sqrt_two, -inv_sqrt_two};
    case OperationCode::X:
        return {0.0, 1.0, 1.0, 0.0};
    case OperationCode::Y:
        return {0.0, Complex{0.0, -1.0}, Complex{0.0, 1.0}, 0.0};
    case OperationCode::Z:
        return {1.0, 0.0, 0.0, -1.0};
    case OperationCode::RX: {
        const double half = operation.parameters.at(0) / 2.0;
        const Complex sine{0.0, -std::sin(half)};
        return {std::cos(half), sine, sine, std::cos(half)};
    }
    case OperationCode::RY: {
        const double half = operation.parameters.at(0) / 2.0;
        return {std::cos(half), -std::sin(half), std::sin(half), std::cos(half)};
    }
    case OperationCode::RZ: {
        const double half = operation.parameters.at(0) / 2.0;
        return {std::polar(1.0, -half), 0.0, 0.0, std::polar(1.0, half)};
    }
    case OperationCode::CX:
    case OperationCode::CZ:
    case OperationCode::SWAP:
        break;
    }
    throw std::invalid_argument("operation is not a single-qubit gate");
}

[[nodiscard]] Matrix2 pauli_matrix(Pauli pauli) {
    switch (pauli) {
    case Pauli::I: return {1.0, 0.0, 0.0, 1.0};
    case Pauli::X: return {0.0, 1.0, 1.0, 0.0};
    case Pauli::Y: return {0.0, Complex{0.0, -1.0}, Complex{0.0, 1.0}, 0.0};
    case Pauli::Z: return {1.0, 0.0, 0.0, -1.0};
    }
    throw std::invalid_argument("unknown Pauli operator");
}

[[nodiscard]] std::vector<Complex> single_gate_tensor(
    const Operation& operation,
    bool conjugate
) {
    const Matrix2 matrix = gate_matrix(operation);
    std::vector<Complex> values(4U);
    for (std::size_t input = 0U; input < 2U; ++input) {
        for (std::size_t output = 0U; output < 2U; ++output) {
            const Complex value = matrix[output * 2U + input];
            values[input + 2U * output] = conjugate ? std::conj(value) : value;
        }
    }
    return values;
}

[[nodiscard]] std::vector<Complex> two_gate_tensor(
    const Operation& operation,
    bool conjugate
) {
    std::vector<Complex> values(16U, Complex{0.0, 0.0});
    for (std::size_t first = 0U; first < 2U; ++first) {
        for (std::size_t second = 0U; second < 2U; ++second) {
            std::size_t output_first = first;
            std::size_t output_second = second;
            Complex value{1.0, 0.0};
            switch (operation.code) {
            case OperationCode::CX:
                output_second ^= first;
                break;
            case OperationCode::CZ:
                if (first != 0U && second != 0U) {
                    value = -1.0;
                }
                break;
            case OperationCode::SWAP:
                output_first = second;
                output_second = first;
                break;
            default:
                throw std::invalid_argument("operation is not a two-qubit gate");
            }
            if (conjugate) {
                value = std::conj(value);
            }
            const std::size_t offset = first + 2U * second +
                4U * output_first + 8U * output_second;
            values[offset] = value;
        }
    }
    return values;
}

void append_tensor(
    std::vector<Tensor>& tensors,
    Tensor tensor,
    std::size_t max_tensor_bytes,
    ContractionStats& stats
) {
    observe_tensor(tensor, max_tensor_bytes, stats);
    tensors.push_back(std::move(tensor));
}

[[nodiscard]] std::vector<std::size_t> build_circuit_side(
    const Program& program,
    bool conjugate,
    std::size_t& next_index,
    std::size_t& next_tensor_id,
    std::vector<Tensor>& tensors,
    std::size_t max_tensor_bytes,
    ContractionStats& stats
) {
    std::vector<std::size_t> current(program.num_qubits());
    for (std::size_t qubit = 0U; qubit < program.num_qubits(); ++qubit) {
        const std::size_t index = next_index++;
        current[qubit] = index;
        append_tensor(
            tensors,
            Tensor{{index}, {Complex{1.0, 0.0}, Complex{0.0, 0.0}}, next_tensor_id++},
            max_tensor_bytes,
            stats
        );
    }

    for (const Operation& operation : program.operations()) {
        if (operation.qubits.size() == 1U) {
            const std::size_t qubit = operation.qubits.front();
            const std::size_t input = current[qubit];
            const std::size_t output = next_index++;
            append_tensor(
                tensors,
                Tensor{{input, output}, single_gate_tensor(operation, conjugate), next_tensor_id++},
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
        append_tensor(
            tensors,
            Tensor{
                {input_first, input_second, output_first, output_second},
                two_gate_tensor(operation, conjugate),
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

[[nodiscard]] std::vector<Pauli> term_paulis(
    const PauliTerm& term,
    std::size_t num_qubits
) {
    std::vector<Pauli> values(num_qubits, Pauli::I);
    std::vector<bool> occupied(num_qubits, false);
    for (const PauliFactor& factor : term.factors()) {
        if (factor.qubit >= num_qubits) {
            throw std::invalid_argument("observable qubit is outside this program");
        }
        if (occupied[factor.qubit]) {
            throw std::invalid_argument("Pauli term contains duplicate qubit factors");
        }
        occupied[factor.qubit] = true;
        values[factor.qubit] = factor.pauli;
    }
    return values;
}

[[nodiscard]] std::vector<Tensor> build_expectation_network(
    const Program& program,
    const PauliTerm& term,
    std::size_t max_tensor_bytes,
    ContractionStats& stats
) {
    std::vector<Tensor> tensors;
    tensors.reserve(
        2U * (program.num_qubits() + program.operations().size()) + program.num_qubits()
    );
    std::size_t next_index = 0U;
    std::size_t next_tensor_id = 0U;
    const std::vector<std::size_t> ket = build_circuit_side(
        program, false, next_index, next_tensor_id, tensors, max_tensor_bytes, stats
    );
    const std::vector<std::size_t> bra = build_circuit_side(
        program, true, next_index, next_tensor_id, tensors, max_tensor_bytes, stats
    );
    const std::vector<Pauli> paulis = term_paulis(term, program.num_qubits());
    for (std::size_t qubit = 0U; qubit < program.num_qubits(); ++qubit) {
        const Matrix2 matrix = pauli_matrix(paulis[qubit]);
        std::vector<Complex> values(4U);
        for (std::size_t ket_bit = 0U; ket_bit < 2U; ++ket_bit) {
            for (std::size_t bra_bit = 0U; bra_bit < 2U; ++bra_bit) {
                values[ket_bit + 2U * bra_bit] = matrix[bra_bit * 2U + ket_bit];
            }
        }
        append_tensor(
            tensors,
            Tensor{{ket[qubit], bra[qubit]}, std::move(values), next_tensor_id++},
            max_tensor_bytes,
            stats
        );
    }
    return tensors;
}

[[nodiscard]] bool contains_index(
    const std::vector<std::size_t>& indices,
    std::size_t value
) {
    return std::find(indices.begin(), indices.end(), value) != indices.end();
}

[[nodiscard]] std::vector<std::size_t> shared_indices(
    const Tensor& first,
    const Tensor& second
) {
    std::vector<std::size_t> shared;
    for (const std::size_t index : first.indices) {
        if (contains_index(second.indices, index)) {
            shared.push_back(index);
        }
    }
    return shared;
}

[[nodiscard]] std::size_t index_position(
    const std::vector<std::size_t>& indices,
    std::size_t value
) {
    const auto found = std::find(indices.begin(), indices.end(), value);
    if (found == indices.end()) {
        throw std::logic_error("tensor contraction index mapping is incomplete");
    }
    return static_cast<std::size_t>(found - indices.begin());
}

struct PairChoice {
    std::size_t first = 0U;
    std::size_t second = 0U;
    std::size_t shared = 0U;
    std::size_t work_rank = std::numeric_limits<std::size_t>::max();
    std::size_t output_rank = std::numeric_limits<std::size_t>::max();
    bool found = false;
};

[[nodiscard]] PairChoice choose_pair(const std::vector<Tensor>& tensors) {
    PairChoice choice;
    for (std::size_t first = 0U; first < tensors.size(); ++first) {
        for (std::size_t second = first + 1U; second < tensors.size(); ++second) {
            const std::size_t shared = shared_indices(tensors[first], tensors[second]).size();
            if (shared == 0U) {
                continue;
            }
            const std::size_t output_rank = tensors[first].indices.size() +
                tensors[second].indices.size() - 2U * shared;
            const std::size_t work_rank = output_rank + shared;
            const bool better = !choice.found || work_rank < choice.work_rank ||
                (work_rank == choice.work_rank && output_rank < choice.output_rank) ||
                (work_rank == choice.work_rank && output_rank == choice.output_rank &&
                 std::pair{tensors[first].id, tensors[second].id} <
                     std::pair{tensors[choice.first].id, tensors[choice.second].id});
            if (better) {
                choice = {first, second, shared, work_rank, output_rank, true};
            }
        }
    }
    return choice;
}

[[nodiscard]] Tensor contract_pair(
    const Tensor& first,
    const Tensor& second,
    std::size_t next_tensor_id,
    std::size_t max_tensor_bytes,
    ContractionStats& stats
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
    std::vector<std::size_t> all_indices = output_indices;
    all_indices.insert(all_indices.end(), shared.begin(), shared.end());

    const std::size_t output_count = checked_value_count(output_indices.size());
    const std::size_t assignment_count = checked_value_count(all_indices.size());
    Tensor result{output_indices, std::vector<Complex>(output_count, Complex{0.0, 0.0}), next_tensor_id};
    observe_tensor(result, max_tensor_bytes, stats);

    std::vector<std::size_t> first_positions;
    first_positions.reserve(first.indices.size());
    for (const std::size_t index : first.indices) {
        first_positions.push_back(index_position(all_indices, index));
    }
    std::vector<std::size_t> second_positions;
    second_positions.reserve(second.indices.size());
    for (const std::size_t index : second.indices) {
        second_positions.push_back(index_position(all_indices, index));
    }

    const std::size_t output_mask = output_count - 1U;
    for (std::size_t assignment = 0U; assignment < assignment_count; ++assignment) {
        std::size_t first_offset = 0U;
        for (std::size_t position = 0U; position < first_positions.size(); ++position) {
            first_offset |= ((assignment >> first_positions[position]) & 1U) << position;
        }
        std::size_t second_offset = 0U;
        for (std::size_t position = 0U; position < second_positions.size(); ++position) {
            second_offset |= ((assignment >> second_positions[position]) & 1U) << position;
        }
        const std::size_t output_offset = assignment & output_mask;
        result.values[output_offset] += first.values[first_offset] * second.values[second_offset];
    }
    ++stats.contractions;
    stats.scalar_multiplications += static_cast<double>(assignment_count);
    return result;
}

void absorb_scalar(
    std::vector<Tensor>& tensors,
    std::size_t scalar_index,
    ContractionStats& stats
) {
    if (!tensors[scalar_index].indices.empty() || tensors[scalar_index].values.size() != 1U) {
        throw std::logic_error("tensor scalar absorption received a non-scalar tensor");
    }
    if (tensors.size() < 2U) {
        return;
    }
    const Complex scalar = tensors[scalar_index].values.front();
    const std::size_t target = scalar_index == 0U ? 1U : 0U;
    for (Complex& value : tensors[target].values) {
        value *= scalar;
    }
    stats.scalar_multiplications += static_cast<double>(tensors[target].values.size());
    ++stats.contractions;
    tensors.erase(tensors.begin() + static_cast<std::ptrdiff_t>(scalar_index));
}

[[nodiscard]] Complex contract_network(
    std::vector<Tensor> tensors,
    std::size_t max_tensor_bytes,
    ContractionStats& stats
) {
    if (tensors.empty()) {
        return Complex{1.0, 0.0};
    }
    std::size_t next_tensor_id = tensors.size();
    while (tensors.size() > 1U) {
        const PairChoice choice = choose_pair(tensors);
        if (!choice.found) {
            const auto scalar = std::find_if(
                tensors.begin(), tensors.end(), [](const Tensor& tensor) {
                    return tensor.indices.empty();
                }
            );
            if (scalar == tensors.end()) {
                throw std::logic_error("closed tensor network became disconnected with open indices");
            }
            absorb_scalar(
                tensors,
                static_cast<std::size_t>(scalar - tensors.begin()),
                stats
            );
            continue;
        }
        Tensor combined = contract_pair(
            tensors[choice.first],
            tensors[choice.second],
            next_tensor_id++,
            max_tensor_bytes,
            stats
        );
        tensors.erase(tensors.begin() + static_cast<std::ptrdiff_t>(choice.second));
        tensors.erase(tensors.begin() + static_cast<std::ptrdiff_t>(choice.first));
        tensors.push_back(std::move(combined));
    }
    if (!tensors.front().indices.empty() || tensors.front().values.size() != 1U) {
        throw std::logic_error("tensor contraction did not produce a scalar");
    }
    return tensors.front().values.front();
}

}  // namespace

TensorNetworkResult tensor_network_expectation(
    const Program& program,
    const Observable& observable,
    std::size_t max_tensor_bytes
) {
    if (max_tensor_bytes == 0U) {
        throw std::invalid_argument("max_tensor_bytes must be positive");
    }
    ContractionStats total;
    Complex expectation{0.0, 0.0};
    for (const PauliTerm& term : observable.terms()) {
        if (term.coefficient() == 0.0) {
            continue;
        }
        ContractionStats term_stats;
        std::vector<Tensor> network = build_expectation_network(
            program, term, max_tensor_bytes, term_stats
        );
        const Complex value = contract_network(std::move(network), max_tensor_bytes, term_stats);
        expectation += term.coefficient() * value;
        total.contractions += term_stats.contractions;
        total.peak_tensor_rank = std::max(total.peak_tensor_rank, term_stats.peak_tensor_rank);
        total.peak_tensor_bytes = std::max(total.peak_tensor_bytes, term_stats.peak_tensor_bytes);
        total.scalar_multiplications += term_stats.scalar_multiplications;
    }
    const double scale = std::max(1.0, std::abs(expectation.real()));
    if (std::abs(expectation.imag()) > kHermitianTolerance * scale) {
        throw std::domain_error("tensor-network observable expectation acquired an imaginary part");
    }
    return {
        expectation.real(),
        observable.terms().size(),
        total.contractions,
        total.peak_tensor_rank,
        total.peak_tensor_bytes,
        total.scalar_multiplications,
        true,
        "native-tn",
        "greedy-contraction",
    };
}

}  // namespace qupy
