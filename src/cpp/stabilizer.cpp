#include "stabilizer.hpp"

#include <algorithm>
#include <bit>
#include <limits>
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

class StabilizerTableau {
public:
    explicit StabilizerTableau(std::size_t num_qubits)
        : num_qubits_(num_qubits),
          word_count_(num_qubits / 64U + static_cast<std::size_t>(num_qubits % 64U != 0U)),
          x_(checked_product(num_qubits, word_count_, "stabilizer X tableau is too large"), 0U),
          z_(checked_product(num_qubits, word_count_, "stabilizer Z tableau is too large"), 0U),
          negative_(num_qubits, 0U) {
        for (std::size_t qubit = 0U; qubit < num_qubits_; ++qubit) {
            set_bit(z_, qubit, qubit, true);
        }
    }

    [[nodiscard]] std::size_t word_count() const noexcept { return word_count_; }

    void apply(const Operation& operation) {
        switch (operation.code) {
        case OperationCode::H:
            apply_h(operation.qubits[0]);
            return;
        case OperationCode::X:
            apply_x(operation.qubits[0]);
            return;
        case OperationCode::Y:
            apply_y(operation.qubits[0]);
            return;
        case OperationCode::Z:
            apply_z(operation.qubits[0]);
            return;
        case OperationCode::CX:
            apply_cx(operation.qubits[0], operation.qubits[1]);
            return;
        case OperationCode::CZ:
            apply_h(operation.qubits[1]);
            apply_cx(operation.qubits[0], operation.qubits[1]);
            apply_h(operation.qubits[1]);
            return;
        case OperationCode::SWAP:
            swap_columns(operation.qubits[0], operation.qubits[1]);
            return;
        case OperationCode::RX:
        case OperationCode::RY:
        case OperationCode::RZ:
            throw std::invalid_argument("stabilizer execution received a non-Clifford rotation");
        }
        throw std::logic_error("unknown operation reached stabilizer execution");
    }

    [[nodiscard]] std::size_t reduce_x() {
        std::size_t pivot_row = 0U;
        for (std::size_t qubit = 0U; qubit < num_qubits_ && pivot_row < num_qubits_; ++qubit) {
            std::size_t selected = pivot_row;
            while (selected < num_qubits_ && !x_bit(selected, qubit)) {
                ++selected;
            }
            if (selected == num_qubits_) {
                continue;
            }
            swap_rows(pivot_row, selected);
            for (std::size_t row = 0U; row < num_qubits_; ++row) {
                if (row != pivot_row && x_bit(row, qubit)) {
                    multiply_rows(row, pivot_row);
                }
            }
            ++pivot_row;
        }
        return pivot_row;
    }

    [[nodiscard]] std::vector<std::uint64_t> solve_base(std::size_t x_rank) {
        std::vector<std::size_t> pivot_columns;
        pivot_columns.reserve(num_qubits_ - x_rank);
        std::size_t pivot_row = x_rank;
        for (std::size_t qubit = 0U; qubit < num_qubits_ && pivot_row < num_qubits_; ++qubit) {
            std::size_t selected = pivot_row;
            while (selected < num_qubits_ && !z_bit(selected, qubit)) {
                ++selected;
            }
            if (selected == num_qubits_) {
                continue;
            }
            swap_rows(pivot_row, selected);
            for (std::size_t row = pivot_row + 1U; row < num_qubits_; ++row) {
                if (z_bit(row, qubit)) {
                    multiply_rows(row, pivot_row);
                }
            }
            pivot_columns.push_back(qubit);
            ++pivot_row;
        }
        if (pivot_row != num_qubits_) {
            throw std::logic_error("stabilizer constraints lost rank");
        }

        std::vector<std::uint64_t> base(word_count_, 0U);
        for (std::size_t offset = pivot_columns.size(); offset > 0U; --offset) {
            const std::size_t row = x_rank + offset - 1U;
            const std::size_t pivot = pivot_columns[offset - 1U];
            const bool parity = dot_parity(z_row(row), base.data());
            const bool value = (negative_[row] != 0U) != parity;
            set_word_bit(base, pivot, value);
        }
        return base;
    }

    [[nodiscard]] std::vector<std::uint64_t> x_generators(std::size_t rank) const {
        std::vector<std::uint64_t> generators(
            checked_product(rank, word_count_, "stabilizer support basis is too large"),
            0U
        );
        for (std::size_t row = 0U; row < rank; ++row) {
            std::copy_n(x_row(row), word_count_, generators.data() + row * word_count_);
        }
        return generators;
    }

private:
    [[nodiscard]] const std::uint64_t* x_row(std::size_t row) const noexcept {
        return x_.data() + row * word_count_;
    }

    [[nodiscard]] const std::uint64_t* z_row(std::size_t row) const noexcept {
        return z_.data() + row * word_count_;
    }

    [[nodiscard]] std::uint64_t* x_row(std::size_t row) noexcept {
        return x_.data() + row * word_count_;
    }

    [[nodiscard]] std::uint64_t* z_row(std::size_t row) noexcept {
        return z_.data() + row * word_count_;
    }

    [[nodiscard]] bool x_bit(std::size_t row, std::size_t qubit) const noexcept {
        return row_bit(x_, row, qubit);
    }

    [[nodiscard]] bool z_bit(std::size_t row, std::size_t qubit) const noexcept {
        return row_bit(z_, row, qubit);
    }

    [[nodiscard]] bool row_bit(
        const std::vector<std::uint64_t>& storage,
        std::size_t row,
        std::size_t qubit
    ) const noexcept {
        const std::size_t word = qubit / 64U;
        const std::uint64_t mask = std::uint64_t{1} << (qubit % 64U);
        return (storage[row * word_count_ + word] & mask) != 0U;
    }

    void set_bit(
        std::vector<std::uint64_t>& storage,
        std::size_t row,
        std::size_t qubit,
        bool value
    ) noexcept {
        const std::size_t word = qubit / 64U;
        const std::uint64_t mask = std::uint64_t{1} << (qubit % 64U);
        std::uint64_t& slot = storage[row * word_count_ + word];
        if (value) {
            slot |= mask;
        } else {
            slot &= ~mask;
        }
    }

    static void set_word_bit(
        std::vector<std::uint64_t>& storage,
        std::size_t qubit,
        bool value
    ) noexcept {
        const std::size_t word = qubit / 64U;
        const std::uint64_t mask = std::uint64_t{1} << (qubit % 64U);
        if (value) {
            storage[word] |= mask;
        } else {
            storage[word] &= ~mask;
        }
    }

    [[nodiscard]] bool dot_parity(
        const std::uint64_t* left,
        const std::uint64_t* right
    ) const noexcept {
        unsigned int parity = 0U;
        for (std::size_t word = 0U; word < word_count_; ++word) {
            parity ^= static_cast<unsigned int>(std::popcount(left[word] & right[word]) & 1U);
        }
        return parity != 0U;
    }

    [[nodiscard]] unsigned int dot_mod4(
        const std::uint64_t* left,
        const std::uint64_t* right
    ) const noexcept {
        unsigned int value = 0U;
        for (std::size_t word = 0U; word < word_count_; ++word) {
            value = (value + static_cast<unsigned int>(std::popcount(left[word] & right[word]))) & 3U;
        }
        return value;
    }

    [[nodiscard]] unsigned int xor_dot_mod4(
        const std::uint64_t* x_left,
        const std::uint64_t* x_right,
        const std::uint64_t* z_left,
        const std::uint64_t* z_right
    ) const noexcept {
        unsigned int value = 0U;
        for (std::size_t word = 0U; word < word_count_; ++word) {
            value = (
                value
                + static_cast<unsigned int>(
                    std::popcount((x_left[word] ^ x_right[word]) & (z_left[word] ^ z_right[word]))
                )
            ) & 3U;
        }
        return value;
    }

    void multiply_rows(std::size_t target, std::size_t source) {
        const std::uint64_t* target_x = x_row(target);
        const std::uint64_t* target_z = z_row(target);
        const std::uint64_t* source_x = x_row(source);
        const std::uint64_t* source_z = z_row(source);

        int exponent = static_cast<int>(2U * negative_[target] + 2U * negative_[source]);
        exponent += static_cast<int>(dot_mod4(target_x, target_z));
        exponent += static_cast<int>(dot_mod4(source_x, source_z));
        exponent += static_cast<int>(2U * dot_mod4(target_z, source_x));
        exponent -= static_cast<int>(
            xor_dot_mod4(target_x, source_x, target_z, source_z)
        );
        exponent %= 4;
        if (exponent < 0) {
            exponent += 4;
        }
        if ((exponent & 1) != 0) {
            throw std::logic_error("stabilizer row multiplication encountered anticommuting rows");
        }
        negative_[target] = static_cast<std::uint8_t>(exponent == 2 ? 1U : 0U);

        std::uint64_t* mutable_x = x_row(target);
        std::uint64_t* mutable_z = z_row(target);
        for (std::size_t word = 0U; word < word_count_; ++word) {
            mutable_x[word] ^= source_x[word];
            mutable_z[word] ^= source_z[word];
        }
    }

    void swap_rows(std::size_t first, std::size_t second) noexcept {
        if (first == second) {
            return;
        }
        for (std::size_t word = 0U; word < word_count_; ++word) {
            std::swap(x_[first * word_count_ + word], x_[second * word_count_ + word]);
            std::swap(z_[first * word_count_ + word], z_[second * word_count_ + word]);
        }
        std::swap(negative_[first], negative_[second]);
    }

    void swap_columns(std::size_t first, std::size_t second) noexcept {
        if (first == second) {
            return;
        }
        for (std::size_t row = 0U; row < num_qubits_; ++row) {
            const bool first_x = x_bit(row, first);
            const bool second_x = x_bit(row, second);
            const bool first_z = z_bit(row, first);
            const bool second_z = z_bit(row, second);
            set_bit(x_, row, first, second_x);
            set_bit(x_, row, second, first_x);
            set_bit(z_, row, first, second_z);
            set_bit(z_, row, second, first_z);
        }
    }

    void apply_h(std::size_t qubit) noexcept {
        for (std::size_t row = 0U; row < num_qubits_; ++row) {
            const bool x = x_bit(row, qubit);
            const bool z = z_bit(row, qubit);
            negative_[row] ^= static_cast<std::uint8_t>(x && z);
            set_bit(x_, row, qubit, z);
            set_bit(z_, row, qubit, x);
        }
    }

    void apply_x(std::size_t qubit) noexcept {
        for (std::size_t row = 0U; row < num_qubits_; ++row) {
            negative_[row] ^= static_cast<std::uint8_t>(z_bit(row, qubit));
        }
    }

    void apply_y(std::size_t qubit) noexcept {
        for (std::size_t row = 0U; row < num_qubits_; ++row) {
            negative_[row] ^= static_cast<std::uint8_t>(x_bit(row, qubit) != z_bit(row, qubit));
        }
    }

    void apply_z(std::size_t qubit) noexcept {
        for (std::size_t row = 0U; row < num_qubits_; ++row) {
            negative_[row] ^= static_cast<std::uint8_t>(x_bit(row, qubit));
        }
    }

    void apply_cx(std::size_t control, std::size_t target) noexcept {
        for (std::size_t row = 0U; row < num_qubits_; ++row) {
            const bool x_control = x_bit(row, control);
            const bool x_target = x_bit(row, target);
            const bool z_control = z_bit(row, control);
            const bool z_target = z_bit(row, target);
            negative_[row] ^= static_cast<std::uint8_t>(
                x_control && z_target && (x_target == z_control)
            );
            set_bit(x_, row, target, x_target != x_control);
            set_bit(z_, row, control, z_control != z_target);
        }
    }

    std::size_t num_qubits_;
    std::size_t word_count_;
    std::vector<std::uint64_t> x_;
    std::vector<std::uint64_t> z_;
    std::vector<std::uint8_t> negative_;
};

class RandomBitStream {
public:
    explicit RandomBitStream(std::mt19937_64& generator) : generator_(generator) {}

    [[nodiscard]] bool draw() {
        if (remaining_ == 0U) {
            word_ = generator_();
            remaining_ = 64U;
        }
        const bool result = (word_ & 1U) != 0U;
        word_ >>= 1U;
        --remaining_;
        return result;
    }

private:
    std::mt19937_64& generator_;
    std::uint64_t word_ = 0U;
    unsigned int remaining_ = 0U;
};

}  // namespace

bool supports_stabilizer(const Program& program) noexcept {
    return std::all_of(
        program.operations().begin(),
        program.operations().end(),
        [](const Operation& operation) {
            switch (operation.code) {
            case OperationCode::H:
            case OperationCode::X:
            case OperationCode::Y:
            case OperationCode::Z:
            case OperationCode::CX:
            case OperationCode::CZ:
            case OperationCode::SWAP:
                return true;
            case OperationCode::RX:
            case OperationCode::RY:
            case OperationCode::RZ:
                return false;
            }
            return false;
        }
    );
}

std::size_t stabilizer_state_bytes(std::size_t num_qubits) {
    const std::size_t words = num_qubits / 64U + static_cast<std::size_t>(num_qubits % 64U != 0U);
    const std::size_t cells = checked_product(
        num_qubits, words, "stabilizer tableau exceeds native address space"
    );
    const std::size_t planes = checked_product(
        checked_product(cells, 2U, "stabilizer tableau exceeds native address space"),
        sizeof(std::uint64_t),
        "stabilizer tableau exceeds native address space"
    );
    const std::size_t phases = checked_product(
        num_qubits, sizeof(std::uint8_t), "stabilizer tableau is too large"
    );
    return checked_sum(planes, phases, "stabilizer tableau is too large");
}

StabilizerSupport build_stabilizer_support(const Program& program) {
    if (!supports_stabilizer(program)) {
        throw std::invalid_argument("program is not supported by stabilizer execution");
    }
    StabilizerTableau tableau(program.num_qubits());
    for (const Operation& operation : program.operations()) {
        tableau.apply(operation);
    }
    const std::size_t rank = tableau.reduce_x();
    std::vector<std::uint64_t> base = tableau.solve_base(rank);
    std::vector<std::uint64_t> generators = tableau.x_generators(rank);
    return {
        program.num_qubits(),
        tableau.word_count(),
        rank,
        std::move(base),
        std::move(generators),
    };
}

std::vector<std::int8_t> draw_stabilizer_samples(
    const StabilizerSupport& support,
    std::size_t shots,
    std::mt19937_64& generator
) {
    if (shots == 0U) {
        throw std::invalid_argument("shots must be at least 1");
    }
    if (support.base.size() != support.word_count) {
        throw std::invalid_argument("stabilizer support base has invalid storage");
    }
    const std::size_t generator_words = checked_product(
        support.rank, support.word_count, "stabilizer support basis exceeds native address space"
    );
    if (support.generators.size() != generator_words) {
        throw std::invalid_argument("stabilizer support basis has invalid storage");
    }

    const std::size_t value_count = checked_product(
        shots, support.num_qubits, "sample result shape exceeds native address space"
    );
    std::vector<std::int8_t> values(value_count);
    std::vector<std::uint64_t> basis(support.word_count, 0U);
    RandomBitStream random_bits(generator);

    for (std::size_t shot = 0U; shot < shots; ++shot) {
        std::copy(support.base.begin(), support.base.end(), basis.begin());
        for (std::size_t row = 0U; row < support.rank; ++row) {
            if (!random_bits.draw()) {
                continue;
            }
            const std::uint64_t* generator_row =
                support.generators.data() + row * support.word_count;
            for (std::size_t word = 0U; word < support.word_count; ++word) {
                basis[word] ^= generator_row[word];
            }
        }
        for (std::size_t column = 0U; column < support.num_qubits; ++column) {
            const std::size_t qubit = support.num_qubits - column - 1U;
            const std::uint64_t mask = std::uint64_t{1} << (qubit % 64U);
            values[shot * support.num_qubits + column] = static_cast<std::int8_t>(
                (basis[qubit / 64U] & mask) != 0U
            );
        }
    }
    return values;
}

}  // namespace qupy::detail
