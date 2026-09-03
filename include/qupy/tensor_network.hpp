#pragma once

#include "qupy/advanced.hpp"

#include <cstddef>
#include <string>

namespace qupy {

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

[[nodiscard]] TensorNetworkResult tensor_network_expectation(
    const Program& program,
    const Observable& observable,
    std::size_t max_tensor_bytes = std::size_t{1} << 30U
);

}  // namespace qupy
