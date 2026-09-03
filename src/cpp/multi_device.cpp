#include "qupy/multi_device.hpp"

#include "qupy/advanced.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef QUPY_HAS_MPI
#include <mpi.h>
#endif

namespace qupy {
namespace {

void validate_observables(
    const std::vector<Observable>& observables,
    std::size_t num_qubits
) {
    if (observables.empty()) {
        throw std::invalid_argument("distributed trajectory estimation requires at least one observable");
    }
    for (const Observable& observable : observables) {
        for (const PauliTerm& term : observable.terms()) {
            for (const PauliFactor& factor : term.factors()) {
                if (factor.qubit >= num_qubits) {
                    throw std::invalid_argument("observable qubit is outside this program");
                }
            }
        }
    }
}

[[nodiscard]] std::uint64_t splitmix64(std::uint64_t value) noexcept {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

[[nodiscard]] std::uint64_t random_seed() {
    std::random_device device;
    return (static_cast<std::uint64_t>(device()) << 32U) ^
        static_cast<std::uint64_t>(device());
}

#ifdef QUPY_HAS_MPI

void check_mpi(int status, const char* operation) {
    if (status != MPI_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed");
    }
}

[[nodiscard]] std::size_t checked_size(std::uint64_t value, const char* label) {
    if (value > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::length_error(std::string(label) + " exceeds native address space");
    }
    return static_cast<std::size_t>(value);
}

[[nodiscard]] std::uint64_t allreduce_u64(
    std::uint64_t value,
    MPI_Op operation
) {
    std::uint64_t result = 0U;
    check_mpi(
        MPI_Allreduce(&value, &result, 1, MPI_UINT64_T, operation, MPI_COMM_WORLD),
        "MPI_Allreduce"
    );
    return result;
}

[[nodiscard]] double allreduce_double(double value, MPI_Op operation) {
    double result = 0.0;
    check_mpi(
        MPI_Allreduce(&value, &result, 1, MPI_DOUBLE, operation, MPI_COMM_WORLD),
        "MPI_Allreduce"
    );
    return result;
}

void allreduce_doubles(
    const std::vector<double>& local,
    std::vector<double>& global
) {
    if (local.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::length_error("MPI observable count exceeds collective range");
    }
    global.resize(local.size());
    if (local.empty()) {
        return;
    }
    check_mpi(
        MPI_Allreduce(
            local.data(), global.data(), static_cast<int>(local.size()),
            MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD
        ),
        "MPI_Allreduce"
    );
}

[[nodiscard]] std::uint64_t broadcast_seed(
    std::optional<std::uint64_t> requested,
    std::size_t rank
) {
    std::uint64_t seed = requested.value_or(rank == 0U ? random_seed() : 0U);
    check_mpi(
        MPI_Bcast(&seed, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD),
        "MPI_Bcast"
    );
    return seed;
}

void require_multi_device(const DistributedInfo& info) {
    if (info.world_size < 2U) {
        throw std::runtime_error("multi-device execution requires at least two MPI ranks");
    }
}

#endif

}  // namespace

DistributedTensorNetworkResult distributed_tensor_network_expectation(
    const Program& program,
    const Observable& observable,
    std::size_t max_tensor_bytes
) {
    if (max_tensor_bytes == 0U) {
        throw std::invalid_argument("max_tensor_bytes must be positive");
    }
#ifndef QUPY_HAS_MPI
    static_cast<void>(program);
    static_cast<void>(observable);
    throw std::runtime_error("QuPy was built without MPI support");
#else
    const DistributedInfo info = distributed_info();
    require_multi_device(info);

    std::vector<PauliTerm> local_terms;
    local_terms.reserve((observable.terms().size() + info.world_size - 1U) / info.world_size);
    bool local_active = false;
    for (std::size_t index = info.rank; index < observable.terms().size(); index += info.world_size) {
        const PauliTerm& term = observable.terms()[index];
        local_terms.push_back(term);
        local_active = local_active || term.coefficient() != 0.0;
    }

    TensorNetworkResult local{
        0.0, local_terms.size(), 0U, 0U, 0U, 0.0,
        true, "native-tn", "greedy-contraction",
    };
    std::uint64_t local_failed = 0U;
    if (!local_terms.empty()) {
        try {
            local = tensor_network_expectation(
                program, Observable(std::move(local_terms)), max_tensor_bytes
            );
        } catch (const std::exception&) {
            local_failed = 1U;
        }
    }

    if (allreduce_u64(local_failed, MPI_MAX) != 0U) {
        throw std::runtime_error(
            "distributed tensor-network contraction failed on one or more MPI ranks"
        );
    }

    const double value = allreduce_double(local.value, MPI_SUM);
    const std::size_t contractions = checked_size(
        allreduce_u64(static_cast<std::uint64_t>(local.contractions), MPI_SUM),
        "distributed contraction count"
    );
    const std::size_t peak_rank = checked_size(
        allreduce_u64(static_cast<std::uint64_t>(local.peak_tensor_rank), MPI_MAX),
        "distributed peak tensor rank"
    );
    const std::size_t peak_bytes = checked_size(
        allreduce_u64(static_cast<std::uint64_t>(local.peak_tensor_bytes), MPI_MAX),
        "distributed peak tensor bytes"
    );
    const double scalar_multiplications = allreduce_double(
        local.scalar_multiplications, MPI_SUM
    );
    const std::size_t active_ranks = checked_size(
        allreduce_u64(local_active ? 1U : 0U, MPI_SUM),
        "distributed active rank count"
    );

    return {
        value,
        observable.terms().size(),
        contractions,
        peak_rank,
        peak_bytes,
        scalar_multiplications,
        info.world_size,
        active_ranks,
        true,
        "native-mpi-tn",
        "mpi-term-parallel-greedy-contraction",
    };
#endif
}

DistributedTrajectoryBatch distributed_trajectory_expectations(
    const NoisyProgram& noisy,
    const std::vector<Observable>& observables,
    std::size_t trajectories,
    std::optional<std::uint64_t> seed
) {
    if (trajectories == 0U) {
        throw std::invalid_argument("trajectory count must be positive");
    }
    validate_observables(observables, noisy.program().num_qubits());
#ifndef QUPY_HAS_MPI
    static_cast<void>(seed);
    throw std::runtime_error("QuPy was built without MPI support");
#else
    const DistributedInfo info = distributed_info();
    require_multi_device(info);
    const std::uint64_t global_seed = broadcast_seed(seed, info.rank);

    const std::size_t base = trajectories / info.world_size;
    const std::size_t remainder = trajectories % info.world_size;
    const std::size_t local_count = base + (info.rank < remainder ? 1U : 0U);
    const bool local_active = local_count != 0U;
    const std::uint64_t rank_seed = splitmix64(
        global_seed ^ (0xd1b54a32d192ed03ULL * static_cast<std::uint64_t>(info.rank + 1U))
    );

    std::vector<double> local_sums(observables.size(), 0.0);
    std::vector<double> local_sum_squares(observables.size(), 0.0);
    std::size_t local_state_bytes = 0U;
    std::uint64_t local_failed = 0U;
    if (local_active) {
        try {
            const TrajectoryBatch local = trajectory_expectations(
                noisy, observables, local_count, rank_seed, "native-cpu"
            );
            local_state_bytes = local.state_bytes;
            const double count = static_cast<double>(local_count);
            for (std::size_t index = 0U; index < observables.size(); ++index) {
                const double mean = local.values[index];
                double m2 = 0.0;
                if (local_count > 1U) {
                    const double error = local.standard_errors[index];
                    m2 = error * error * count * static_cast<double>(local_count - 1U);
                }
                local_sums[index] = count * mean;
                local_sum_squares[index] = m2 + count * mean * mean;
            }
        } catch (const std::exception&) {
            local_failed = 1U;
        }
    }

    if (allreduce_u64(local_failed, MPI_MAX) != 0U) {
        throw std::runtime_error(
            "distributed trajectory execution failed on one or more MPI ranks"
        );
    }

    std::vector<double> sums;
    std::vector<double> sum_squares;
    allreduce_doubles(local_sums, sums);
    allreduce_doubles(local_sum_squares, sum_squares);
    const double total_count = static_cast<double>(trajectories);
    std::vector<double> values(observables.size(), 0.0);
    std::vector<double> standard_errors(observables.size(), 0.0);
    for (std::size_t index = 0U; index < observables.size(); ++index) {
        const double mean = sums[index] / total_count;
        values[index] = mean;
        if (trajectories == 1U) {
            standard_errors[index] = std::numeric_limits<double>::quiet_NaN();
            continue;
        }
        const double m2 = std::max(
            0.0,
            sum_squares[index] - total_count * mean * mean
        );
        standard_errors[index] = std::sqrt(
            m2 / (total_count * static_cast<double>(trajectories - 1U))
        );
    }

    const std::size_t state_bytes_per_rank = checked_size(
        allreduce_u64(static_cast<std::uint64_t>(local_state_bytes), MPI_MAX),
        "distributed trajectory state bytes"
    );
    const std::size_t active_ranks = checked_size(
        allreduce_u64(local_active ? 1U : 0U, MPI_SUM),
        "distributed active rank count"
    );

    return {
        std::move(values),
        std::move(standard_errors),
        observables.size(),
        trajectories,
        global_seed,
        state_bytes_per_rank,
        info.world_size,
        active_ranks,
        false,
        "native-mpi-trajectory",
        "mpi-trajectory-ensemble",
    };
#endif
}

}  // namespace qupy
