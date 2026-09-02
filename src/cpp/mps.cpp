#include "mps.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <utility>

namespace qupy::detail {
namespace {

[[nodiscard]] std::size_t checked_product(
    std::size_t left,
    std::size_t right,
    const char* message
) {
    if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::length_error(message);
    }
    return left * right;
}

[[nodiscard]] std::size_t checked_sum(
    std::size_t left,
    std::size_t right,
    const char* message
) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        throw std::length_error(message);
    }
    return left + right;
}

struct Matrix {
    std::size_t rows;
    std::size_t columns;
    std::vector<Complex> values;

    Matrix(std::size_t rows_, std::size_t columns_)
        : rows(rows_),
          columns(columns_),
          values(checked_product(rows_, columns_, "MPS matrix exceeds native address space")) {}

    [[nodiscard]] Complex& at(std::size_t row, std::size_t column) noexcept {
        return values[row * columns + column];
    }

    [[nodiscard]] const Complex& at(std::size_t row, std::size_t column) const noexcept {
        return values[row * columns + column];
    }
};

[[nodiscard]] Matrix conjugate_transpose(const Matrix& input) {
    Matrix result(input.columns, input.rows);
    for (std::size_t row = 0U; row < input.rows; ++row) {
        for (std::size_t column = 0U; column < input.columns; ++column) {
            result.at(column, row) = std::conj(input.at(row, column));
        }
    }
    return result;
}

struct SvdResult {
    Matrix u;
    std::vector<double> singular_values;
    Matrix vh;
    double discarded_weight;
};

[[nodiscard]] Matrix identity_matrix(std::size_t size) {
    Matrix result(size, size);
    for (std::size_t index = 0U; index < size; ++index) {
        result.at(index, index) = Complex{1.0, 0.0};
    }
    return result;
}

[[nodiscard]] double column_norm_squared(const Matrix& matrix, std::size_t column) {
    double result = 0.0;
    for (std::size_t row = 0U; row < matrix.rows; ++row) {
        result += std::norm(matrix.at(row, column));
    }
    return result;
}

[[nodiscard]] Complex column_inner_product(
    const Matrix& matrix,
    std::size_t first,
    std::size_t second
) {
    Complex result{0.0, 0.0};
    for (std::size_t row = 0U; row < matrix.rows; ++row) {
        result += std::conj(matrix.at(row, first)) * matrix.at(row, second);
    }
    return result;
}

void rotate_columns(
    Matrix& matrix,
    std::size_t first,
    std::size_t second,
    double cosine,
    double sine,
    Complex phase
) {
    for (std::size_t row = 0U; row < matrix.rows; ++row) {
        const Complex first_value = matrix.at(row, first);
        const Complex second_value = matrix.at(row, second);
        matrix.at(row, first) = cosine * phase * first_value - sine * second_value;
        matrix.at(row, second) = sine * phase * first_value + cosine * second_value;
    }
}

[[nodiscard]] SvdResult jacobi_svd_tall(const Matrix& input) {
    if (input.rows < input.columns || input.columns == 0U) {
        throw std::invalid_argument("MPS Jacobi SVD requires a nonempty tall matrix");
    }
    Matrix work = input;
    Matrix right = identity_matrix(input.columns);
    constexpr double kOrthogonalityScale = 32.0;
    const double epsilon = std::numeric_limits<double>::epsilon();
    const std::size_t max_sweeps = 64U + 2U * input.columns;

    for (std::size_t sweep = 0U; sweep < max_sweeps; ++sweep) {
        bool changed = false;
        for (std::size_t first = 0U; first + 1U < input.columns; ++first) {
            for (std::size_t second = first + 1U; second < input.columns; ++second) {
                const double alpha = column_norm_squared(work, first);
                const double beta = column_norm_squared(work, second);
                if (alpha == 0.0 || beta == 0.0) {
                    continue;
                }
                const Complex gamma = column_inner_product(work, first, second);
                const double gamma_abs = std::abs(gamma);
                const double threshold =
                    kOrthogonalityScale * epsilon * std::max(alpha, beta);
                if (gamma_abs <= threshold) {
                    continue;
                }
                changed = true;
                const double tau = (beta - alpha) / (2.0 * gamma_abs);
                const double tangent = tau == 0.0
                    ? 1.0
                    : std::copysign(
                        1.0 / (std::abs(tau) + std::sqrt(1.0 + tau * tau)),
                        tau
                    );
                const double cosine = 1.0 / std::sqrt(1.0 + tangent * tangent);
                const double sine = tangent * cosine;
                const Complex phase = gamma / gamma_abs;
                rotate_columns(work, first, second, cosine, sine, phase);
                rotate_columns(right, first, second, cosine, sine, phase);
            }
        }
        if (!changed) {
            break;
        }
        if (sweep + 1U == max_sweeps) {
            throw std::runtime_error("MPS Jacobi SVD did not converge");
        }
    }

    std::vector<double> singular(input.columns);
    for (std::size_t column = 0U; column < input.columns; ++column) {
        singular[column] = std::sqrt(std::max(0.0, column_norm_squared(work, column)));
    }
    std::vector<std::size_t> order(input.columns);
    std::iota(order.begin(), order.end(), 0U);
    std::sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right_index) {
        return singular[left] > singular[right_index];
    });

    const double largest = singular[order.front()];
    if (!(largest > 0.0) || !std::isfinite(largest)) {
        throw std::runtime_error("MPS factorization produced an invalid zero state");
    }
    const double rank_tolerance = 64.0 * epsilon *
        static_cast<double>(std::max(input.rows, input.columns)) * largest;
    std::size_t rank = 0U;
    double discarded_weight = 0.0;
    for (const std::size_t index : order) {
        if (singular[index] > rank_tolerance) {
            ++rank;
        } else {
            discarded_weight += singular[index] * singular[index];
        }
    }
    rank = std::max<std::size_t>(rank, 1U);

    Matrix u(input.rows, rank);
    Matrix vh(rank, input.columns);
    std::vector<double> kept(rank);
    for (std::size_t output = 0U; output < rank; ++output) {
        const std::size_t source = order[output];
        const double sigma = singular[source];
        kept[output] = sigma;
        for (std::size_t row = 0U; row < input.rows; ++row) {
            u.at(row, output) = work.at(row, source) / sigma;
        }
        for (std::size_t column = 0U; column < input.columns; ++column) {
            vh.at(output, column) = std::conj(right.at(column, source));
        }
    }

    const double input_norm_squared = std::accumulate(
        input.values.begin(),
        input.values.end(),
        0.0,
        [](double total, const Complex& value) { return total + std::norm(value); }
    );
    const double backward_error = 128.0 * epsilon *
        static_cast<double>(std::max(input.rows, input.columns)) *
        std::sqrt(input_norm_squared);
    if (std::sqrt(discarded_weight) > backward_error) {
        throw std::runtime_error("MPS factorization would discard non-numerical state weight");
    }
    return {std::move(u), std::move(kept), std::move(vh), discarded_weight};
}

[[nodiscard]] SvdResult jacobi_svd(const Matrix& input) {
    if (input.rows >= input.columns) {
        return jacobi_svd_tall(input);
    }
    const Matrix transposed = conjugate_transpose(input);
    SvdResult swapped = jacobi_svd_tall(transposed);
    Matrix u = conjugate_transpose(swapped.vh);
    Matrix vh = conjugate_transpose(swapped.u);
    return {
        std::move(u),
        std::move(swapped.singular_values),
        std::move(vh),
        swapped.discarded_weight,
    };
}

struct SiteTensor {
    std::size_t left;
    std::size_t right;
    std::vector<Complex> values;

    SiteTensor(std::size_t left_, std::size_t right_)
        : left(left_),
          right(right_),
          values(
              checked_product(
                  checked_product(left_, 2U, "MPS tensor exceeds native address space"),
                  right_,
                  "MPS tensor exceeds native address space"
              )
          ) {}

    [[nodiscard]] Complex& at(std::size_t l, std::size_t physical, std::size_t r) noexcept {
        return values[(l * 2U + physical) * right + r];
    }

    [[nodiscard]] const Complex& at(
        std::size_t l,
        std::size_t physical,
        std::size_t r
    ) const noexcept {
        return values[(l * 2U + physical) * right + r];
    }
};

class MpsState {
public:
    explicit MpsState(std::size_t num_qubits) : sites_(num_qubits, SiteTensor(1U, 1U)) {
        if (num_qubits == 0U) {
            throw std::invalid_argument("MPS execution requires at least one qubit");
        }
        for (SiteTensor& site : sites_) {
            site.at(0U, 0U, 0U) = Complex{1.0, 0.0};
        }
        update_stats();
    }

    void apply(const MpsStep& step) {
        if (step.first >= sites_.size() ||
            (step.kind != MpsStepKind::Single && step.second >= sites_.size())) {
            throw std::invalid_argument("MPS step references a qubit outside the state");
        }
        if (step.kind == MpsStepKind::Single) {
            apply_single(step.first, step.matrix);
            return;
        }
        apply_two_qubit(step.kind, step.first, step.second);
    }

    [[nodiscard]] double expectation_z(std::size_t qubit) const {
        if (qubit >= sites_.size()) {
            throw std::invalid_argument("MPS observable is outside the state");
        }
        const Complex numerator = contract_transfer(qubit);
        const Complex denominator = contract_transfer(std::nullopt);
        if (std::abs(denominator) == 0.0) {
            throw std::runtime_error("MPS state has zero norm");
        }
        const Complex value = numerator / denominator;
        if (std::abs(value.imag()) > 1e-10) {
            throw std::runtime_error("MPS Pauli-Z expectation acquired an invalid imaginary part");
        }
        return std::clamp(value.real(), -1.0, 1.0);
    }

    [[nodiscard]] std::vector<Complex> statevector() const {
        if (sites_.size() >= std::numeric_limits<std::size_t>::digits) {
            throw std::length_error("MPS state vector exceeds native address space");
        }
        const std::size_t dimension = std::size_t{1} << sites_.size();
        std::vector<Complex> result(dimension);
        std::vector<Complex> current(1U, Complex{1.0, 0.0});
        std::vector<Complex> next;
        for (std::size_t basis = 0U; basis < dimension; ++basis) {
            current.assign(1U, Complex{1.0, 0.0});
            for (std::size_t qubit = 0U; qubit < sites_.size(); ++qubit) {
                const SiteTensor& site = sites_[qubit];
                if (current.size() != site.left) {
                    throw std::logic_error("MPS bond dimensions are inconsistent");
                }
                next.assign(site.right, Complex{0.0, 0.0});
                const std::size_t physical = (basis >> qubit) & std::size_t{1};
                for (std::size_t left = 0U; left < site.left; ++left) {
                    for (std::size_t right = 0U; right < site.right; ++right) {
                        next[right] += current[left] * site.at(left, physical, right);
                    }
                }
                current.swap(next);
            }
            if (current.size() != 1U) {
                throw std::logic_error("MPS terminal bond is not one-dimensional");
            }
            result[basis] = current.front();
        }
        return result;
    }

    [[nodiscard]] std::size_t state_bytes() const noexcept { return state_bytes_; }
    [[nodiscard]] std::size_t max_bond() const noexcept { return max_bond_; }
    [[nodiscard]] double discarded_weight() const noexcept { return discarded_weight_; }

private:
    void apply_single(std::size_t qubit, const std::array<Complex, 4>& matrix) {
        SiteTensor& site = sites_[qubit];
        for (std::size_t left = 0U; left < site.left; ++left) {
            for (std::size_t right = 0U; right < site.right; ++right) {
                const Complex zero = site.at(left, 0U, right);
                const Complex one = site.at(left, 1U, right);
                site.at(left, 0U, right) = matrix[0] * zero + matrix[1] * one;
                site.at(left, 1U, right) = matrix[2] * zero + matrix[3] * one;
            }
        }
    }

    void apply_two_qubit(MpsStepKind kind, std::size_t first, std::size_t second) {
        if (first == second) {
            throw std::invalid_argument("MPS two-qubit step uses the same qubit twice");
        }
        const std::size_t low = std::min(first, second);
        const std::size_t high = std::max(first, second);
        for (std::size_t position = high - 1U; position > low; --position) {
            apply_adjacent(MpsStepKind::SWAP, position, true);
        }
        const bool first_is_left = first == low;
        apply_adjacent(kind, low, first_is_left);
        for (std::size_t position = low + 1U; position < high; ++position) {
            apply_adjacent(MpsStepKind::SWAP, position, true);
        }
    }

    void apply_adjacent(MpsStepKind kind, std::size_t left_site, bool first_is_left) {
        SiteTensor& left = sites_[left_site];
        SiteTensor& right = sites_[left_site + 1U];
        if (left.right != right.left) {
            throw std::logic_error("MPS adjacent bond dimensions are inconsistent");
        }
        const std::size_t outer_left = left.left;
        const std::size_t center = left.right;
        const std::size_t outer_right = right.right;
        Matrix theta(
            checked_product(outer_left, 2U, "MPS two-site matrix is too large"),
            checked_product(outer_right, 2U, "MPS two-site matrix is too large")
        );
        for (std::size_t l = 0U; l < outer_left; ++l) {
            for (std::size_t s0 = 0U; s0 < 2U; ++s0) {
                for (std::size_t s1 = 0U; s1 < 2U; ++s1) {
                    for (std::size_t r = 0U; r < outer_right; ++r) {
                        Complex value{0.0, 0.0};
                        for (std::size_t bond = 0U; bond < center; ++bond) {
                            value += left.at(l, s0, bond) * right.at(bond, s1, r);
                        }
                        theta.at(l * 2U + s0, s1 * outer_right + r) = value;
                    }
                }
            }
        }

        Matrix transformed(theta.rows, theta.columns);
        for (std::size_t l = 0U; l < outer_left; ++l) {
            for (std::size_t s0 = 0U; s0 < 2U; ++s0) {
                for (std::size_t s1 = 0U; s1 < 2U; ++s1) {
                    std::size_t output0 = s0;
                    std::size_t output1 = s1;
                    double phase = 1.0;
                    switch (kind) {
                    case MpsStepKind::CX:
                        if (first_is_left) {
                            output1 = s1 ^ s0;
                        } else {
                            output0 = s0 ^ s1;
                        }
                        break;
                    case MpsStepKind::CZ:
                        if (s0 != 0U && s1 != 0U) {
                            phase = -1.0;
                        }
                        break;
                    case MpsStepKind::SWAP:
                        output0 = s1;
                        output1 = s0;
                        break;
                    case MpsStepKind::Single:
                        throw std::logic_error("single-qubit step reached MPS two-site execution");
                    }
                    for (std::size_t r = 0U; r < outer_right; ++r) {
                        transformed.at(
                            l * 2U + output0,
                            output1 * outer_right + r
                        ) += phase * theta.at(l * 2U + s0, s1 * outer_right + r);
                    }
                }
            }
        }

        SvdResult svd = jacobi_svd(transformed);
        const std::size_t rank = svd.singular_values.size();
        SiteTensor next_left(outer_left, rank);
        SiteTensor next_right(rank, outer_right);
        for (std::size_t l = 0U; l < outer_left; ++l) {
            for (std::size_t physical = 0U; physical < 2U; ++physical) {
                const std::size_t row = l * 2U + physical;
                for (std::size_t bond = 0U; bond < rank; ++bond) {
                    next_left.at(l, physical, bond) = svd.u.at(row, bond);
                }
            }
        }
        for (std::size_t bond = 0U; bond < rank; ++bond) {
            for (std::size_t physical = 0U; physical < 2U; ++physical) {
                for (std::size_t r = 0U; r < outer_right; ++r) {
                    const std::size_t column = physical * outer_right + r;
                    next_right.at(bond, physical, r) =
                        svd.singular_values[bond] * svd.vh.at(bond, column);
                }
            }
        }
        sites_[left_site] = std::move(next_left);
        sites_[left_site + 1U] = std::move(next_right);
        discarded_weight_ += svd.discarded_weight;
        update_stats();
    }

    [[nodiscard]] Complex contract_transfer(std::optional<std::size_t> observable) const {
        std::vector<Complex> environment(1U, Complex{1.0, 0.0});
        std::size_t environment_dim = 1U;
        for (std::size_t qubit = 0U; qubit < sites_.size(); ++qubit) {
            const SiteTensor& site = sites_[qubit];
            if (site.left != environment_dim) {
                throw std::logic_error("MPS transfer bond dimensions are inconsistent");
            }
            const std::size_t next_count = checked_product(
                site.right, site.right, "MPS transfer matrix exceeds native address space"
            );
            std::vector<Complex> next(next_count, Complex{0.0, 0.0});
            for (std::size_t left_bra = 0U; left_bra < site.left; ++left_bra) {
                for (std::size_t left_ket = 0U; left_ket < site.left; ++left_ket) {
                    const Complex weight = environment[left_bra * site.left + left_ket];
                    for (std::size_t physical = 0U; physical < 2U; ++physical) {
                        const double factor =
                            observable.has_value() && *observable == qubit && physical == 1U
                            ? -1.0 : 1.0;
                        for (std::size_t right_bra = 0U; right_bra < site.right; ++right_bra) {
                            const Complex bra = std::conj(site.at(left_bra, physical, right_bra));
                            for (std::size_t right_ket = 0U; right_ket < site.right; ++right_ket) {
                                next[right_bra * site.right + right_ket] +=
                                    weight * bra * site.at(left_ket, physical, right_ket) * factor;
                            }
                        }
                    }
                }
            }
            environment.swap(next);
            environment_dim = site.right;
        }
        if (environment.size() != 1U) {
            throw std::logic_error("MPS transfer contraction did not reach a scalar");
        }
        return environment.front();
    }

    void update_stats() {
        std::size_t bytes = 0U;
        std::size_t maximum = 1U;
        for (const SiteTensor& site : sites_) {
            const std::size_t current_bytes = checked_product(
                site.values.size(), sizeof(Complex), "MPS state exceeds native address space"
            );
            bytes = checked_sum(bytes, current_bytes, "MPS state exceeds native address space");
            maximum = std::max({maximum, site.left, site.right});
        }
        state_bytes_ = std::max(state_bytes_, bytes);
        max_bond_ = std::max(max_bond_, maximum);
    }

    std::vector<SiteTensor> sites_;
    std::size_t state_bytes_ = 0U;
    std::size_t max_bond_ = 1U;
    double discarded_weight_ = 0.0;
};

[[nodiscard]] std::size_t saturated_product(std::size_t left, std::size_t right) noexcept {
    if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left) {
        return std::numeric_limits<std::size_t>::max();
    }
    return left * right;
}

[[nodiscard]] std::size_t saturated_sum(std::size_t left, std::size_t right) noexcept {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        return std::numeric_limits<std::size_t>::max();
    }
    return left + right;
}

[[nodiscard]] double single_qubit_work(std::size_t left, std::size_t right) noexcept {
    return 8.0 * static_cast<double>(left) * static_cast<double>(right);
}

[[nodiscard]] double contraction_work(
    std::size_t left,
    std::size_t center,
    std::size_t right
) noexcept {
    const double rows = 2.0 * static_cast<double>(left);
    const double columns = 2.0 * static_cast<double>(right);
    const double contract = 4.0 * static_cast<double>(left) *
        static_cast<double>(center) * static_cast<double>(right);
    const double factor = rows * columns * std::min(rows, columns);
    return contract + factor;
}

void update_estimated_memory(
    const std::vector<std::size_t>& bonds,
    std::size_t num_qubits,
    std::size_t& maximum_bytes,
    std::size_t& maximum_bond
) {
    std::size_t bytes = 0U;
    for (std::size_t site = 0U; site < num_qubits; ++site) {
        const std::size_t left = site == 0U ? 1U : bonds[site - 1U];
        const std::size_t right = site + 1U == num_qubits ? 1U : bonds[site];
        const std::size_t values = saturated_product(saturated_product(left, 2U), right);
        bytes = saturated_sum(bytes, saturated_product(values, sizeof(Complex)));
        maximum_bond = std::max({maximum_bond, left, right});
    }
    maximum_bytes = std::max(maximum_bytes, bytes);
}

void estimate_adjacent(
    std::vector<std::size_t>& bonds,
    std::size_t num_qubits,
    std::size_t left_site,
    std::size_t& maximum_bytes,
    std::size_t& maximum_bond,
    double& work
) {
    const std::size_t left = left_site == 0U ? 1U : bonds[left_site - 1U];
    const std::size_t center = bonds[left_site];
    const std::size_t right = left_site + 2U == num_qubits ? 1U : bonds[left_site + 1U];
    const std::size_t new_bond = std::min(
        saturated_product(left, 2U),
        saturated_product(right, 2U)
    );
    bonds[left_site] = std::max<std::size_t>(new_bond, 1U);
    work += contraction_work(left, center, right);
    update_estimated_memory(bonds, num_qubits, maximum_bytes, maximum_bond);
}

void estimate_two_qubit(
    std::vector<std::size_t>& bonds,
    std::size_t num_qubits,
    std::size_t first,
    std::size_t second,
    std::size_t& maximum_bytes,
    std::size_t& maximum_bond,
    std::size_t& routed_swaps,
    double& work
) {
    const std::size_t low = std::min(first, second);
    const std::size_t high = std::max(first, second);
    if (low == high || high >= num_qubits) {
        throw std::invalid_argument("MPS estimate received an invalid two-qubit step");
    }
    for (std::size_t position = high - 1U; position > low; --position) {
        estimate_adjacent(bonds, num_qubits, position, maximum_bytes, maximum_bond, work);
        ++routed_swaps;
    }
    estimate_adjacent(bonds, num_qubits, low, maximum_bytes, maximum_bond, work);
    for (std::size_t position = low + 1U; position < high; ++position) {
        estimate_adjacent(bonds, num_qubits, position, maximum_bytes, maximum_bond, work);
        ++routed_swaps;
    }
}

[[nodiscard]] MpsState execute_mps(
    std::size_t num_qubits,
    const std::vector<MpsStep>& steps
) {
    MpsState state(num_qubits);
    for (const MpsStep& step : steps) {
        state.apply(step);
    }
    return state;
}

}  // namespace

MpsEstimate mps_estimate(
    std::size_t num_qubits,
    const std::vector<MpsStep>& steps
) {
    if (num_qubits == 0U) {
        throw std::invalid_argument("MPS estimate requires at least one qubit");
    }
    std::vector<std::size_t> bonds(num_qubits - 1U, 1U);
    std::size_t maximum_bytes = 0U;
    std::size_t maximum_bond = 1U;
    std::size_t routed_swaps = 0U;
    double work = 0.0;
    update_estimated_memory(bonds, num_qubits, maximum_bytes, maximum_bond);
    for (const MpsStep& step : steps) {
        if (step.first >= num_qubits ||
            (step.kind != MpsStepKind::Single && step.second >= num_qubits)) {
            throw std::invalid_argument("MPS estimate step references an invalid qubit");
        }
        if (step.kind == MpsStepKind::Single) {
            const std::size_t left = step.first == 0U ? 1U : bonds[step.first - 1U];
            const std::size_t right = step.first + 1U == num_qubits ? 1U : bonds[step.first];
            work += single_qubit_work(left, right);
            continue;
        }
        estimate_two_qubit(
            bonds, num_qubits, step.first, step.second,
            maximum_bytes, maximum_bond, routed_swaps, work
        );
    }
    return {maximum_bytes, maximum_bond, routed_swaps, work};
}

MpsStateResult mps_statevector(
    std::size_t num_qubits,
    const std::vector<MpsStep>& steps
) {
    MpsState state = execute_mps(num_qubits, steps);
    return {
        state.statevector(),
        state.state_bytes(),
        state.max_bond(),
        state.discarded_weight(),
    };
}

MpsExpectationResult mps_pauli_z_expectation(
    std::size_t num_qubits,
    const std::vector<MpsStep>& steps,
    std::size_t observable_qubit
) {
    MpsState state = execute_mps(num_qubits, steps);
    return {
        state.expectation_z(observable_qubit),
        state.state_bytes(),
        state.max_bond(),
        state.discarded_weight(),
    };
}

}  // namespace qupy::detail
