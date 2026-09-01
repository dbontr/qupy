#pragma once

#include "qupy/core.hpp"

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace qupy::detail {

struct StabilizerSupport {
    std::size_t num_qubits;
    std::size_t word_count;
    std::size_t rank;
    std::vector<std::uint64_t> base;
    std::vector<std::uint64_t> generators;
};

[[nodiscard]] bool supports_stabilizer(const Program& program) noexcept;
[[nodiscard]] std::size_t stabilizer_state_bytes(std::size_t num_qubits);
[[nodiscard]] StabilizerSupport build_stabilizer_support(const Program& program);
[[nodiscard]] std::vector<std::int8_t> draw_stabilizer_samples(
    const StabilizerSupport& support,
    std::size_t shots,
    std::mt19937_64& generator
);

}  // namespace qupy::detail
