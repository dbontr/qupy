#pragma once

#include "qupy/tensor_network.hpp"
#include "qupy/trajectory.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace qupy {

struct DistributedTensorNetworkResult {
    double value;
    std::size_t term_count;
    std::size_t contractions;
    std::size_t peak_tensor_rank;
    std::size_t peak_tensor_bytes;
    double scalar_multiplications;
    std::size_t world_size;
    std::size_t active_ranks;
    bool exact;
    std::string backend;
    std::string method;
};

struct DistributedTrajectoryBatch {
    std::vector<double> values;
    std::vector<double> standard_errors;
    std::size_t observable_count;
    std::size_t trajectories;
    std::uint64_t seed;
    std::size_t state_bytes_per_rank;
    std::size_t world_size;
    std::size_t active_ranks;
    bool exact;
    std::string backend;
    std::string method;
};

[[nodiscard]] DistributedTensorNetworkResult distributed_tensor_network_expectation(
    const Program& program,
    const Observable& observable,
    std::size_t max_tensor_bytes = std::size_t{1} << 30U
);

[[nodiscard]] DistributedTrajectoryBatch distributed_trajectory_expectations(
    const NoisyProgram& noisy,
    const std::vector<Observable>& observables,
    std::size_t trajectories,
    std::optional<std::uint64_t> seed = std::nullopt
);

}  // namespace qupy
