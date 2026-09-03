#pragma once

#include "qupy/advanced.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace qupy {

struct TrajectoryBatch {
    std::vector<double> values;
    std::vector<double> standard_errors;
    std::size_t observable_count;
    std::size_t trajectories;
    std::uint64_t seed;
    std::size_t state_bytes;
    bool exact;
    std::string backend;
    std::string method;
};

[[nodiscard]] TrajectoryBatch trajectory_expectations(
    const NoisyProgram& noisy,
    const std::vector<Observable>& observables,
    std::size_t trajectories,
    std::optional<std::uint64_t> seed = std::nullopt,
    const std::string& backend = "auto"
);

}  // namespace qupy
