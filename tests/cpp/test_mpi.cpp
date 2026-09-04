#include "qupy/advanced.hpp"
#include "qupy/multi_device.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(double actual, double expected, const char* message) {
    if (std::abs(actual - expected) > 5e-12) {
        throw std::runtime_error(message);
    }
}

void require_complex_close(
    qupy::Complex actual,
    qupy::Complex expected,
    const char* message
) {
    if (std::abs(actual - expected) > 5e-12) {
        throw std::runtime_error(message);
    }
}

qupy::Program distributed_program() {
    qupy::Program program(4U);
    program = qupy::ry(program, 0.0, 0U);
    program = qupy::h(program, 2U);
    program = qupy::cx(program, 0U, 2U);
    program = qupy::ry(program, -0.31, 3U);
    program = qupy::cx(program, 2U, 1U);
    program = qupy::cz(program, 1U, 3U);
    program = qupy::swap(program, 0U, 3U);
    return program;
}

qupy::Observable left_observable() {
    return qupy::Observable({
        qupy::PauliTerm(0.41, {{0U, qupy::Pauli::X}, {2U, qupy::Pauli::Y}}),
        qupy::PauliTerm(-0.27, {{1U, qupy::Pauli::Z}, {3U, qupy::Pauli::X}}),
        qupy::PauliTerm(0.13, {}),
    });
}

qupy::Observable right_observable() {
    return qupy::Observable({
        qupy::PauliTerm(0.36, {{0U, qupy::Pauli::Y}, {3U, qupy::Pauli::Y}}),
        qupy::PauliTerm(0.22, {{2U, qupy::Pauli::X}}),
    });
}

void test_observable_reductions() {
    const qupy::Program program = distributed_program();
    const qupy::Observable left = left_observable();
    const qupy::Observable right = right_observable();
    const auto plan = qupy::observable_plan(program, {left, right}, "native-mpi");
    require(plan.backend == "native-mpi", "MPI observable plan backend mismatch");
    require(plan.method == "mpi-pauli-reduction", "MPI observable plan method mismatch");
    require(plan.exact, "MPI observable plan must be exact");
    const qupy::DistributedInfo topology = qupy::distributed_info();
    require(
        plan.estimated_state_bytes == 16U * sizeof(qupy::Complex) / topology.world_size,
        "MPI observable plan memory is not per-rank"
    );

    const auto cpu_expect = qupy::expect_observable(program, left, "native-cpu");
    const auto mpi_expect = qupy::expect_observable(program, left, "native-mpi");
    require_close(mpi_expect.value, cpu_expect.value, "MPI expectation mismatch");

    const auto cpu_variance = qupy::variance_observable(program, left, "native-cpu");
    const auto mpi_variance = qupy::variance_observable(program, left, "native-mpi");
    require_close(mpi_variance.value, cpu_variance.value, "MPI variance mismatch");

    const auto cpu_covariance = qupy::covariance_observable(
        program, left, right, "native-cpu"
    );
    const auto mpi_covariance = qupy::covariance_observable(
        program, left, right, "native-mpi"
    );
    require_close(mpi_covariance.value, cpu_covariance.value, "MPI covariance mismatch");
}

void test_observable_batch_and_gradient() {
    const qupy::Program program = distributed_program();
    const qupy::Observable left = left_observable();
    const qupy::Observable right = right_observable();
    const auto cpu_batch = qupy::expect_observables(program, {left, right}, "native-cpu");
    const auto mpi_batch = qupy::expect_observables(program, {left, right}, "native-mpi");
    require(mpi_batch.backend == "native-mpi", "MPI observable batch backend mismatch");
    require(mpi_batch.values.size() == cpu_batch.values.size(), "MPI observable batch shape mismatch");
    for (std::size_t index = 0U; index < cpu_batch.values.size(); ++index) {
        require_close(mpi_batch.values[index], cpu_batch.values[index], "MPI batch mismatch");
    }

    const std::vector<qupy::ParameterSlot> slots{{0U, 0U}};
    const std::vector<double> parameters{0.37};
    const auto cpu_gradient = qupy::value_and_grad(
        program, left, slots, parameters, "native-cpu", qupy::GradientMethod::ParameterShift
    );
    const auto mpi_gradient = qupy::value_and_grad(
        program, left, slots, parameters, "native-mpi", qupy::GradientMethod::ParameterShift
    );
    require_close(mpi_gradient.value, cpu_gradient.value, "MPI gradient value mismatch");
    require(mpi_gradient.gradient.size() == 1U, "MPI gradient shape mismatch");
    require_close(
        mpi_gradient.gradient.front(), cpu_gradient.gradient.front(), "MPI gradient mismatch"
    );
}

void test_distributed_cuda_statevector() {
    const qupy::Program program = distributed_program();
    try {
        const qupy::DistributedStateVector distributed =
            qupy::distributed_cuda_statevector(program);
        const qupy::StateVector cpu = qupy::statevector(program, "native-cpu");
        require(
            distributed.global_size == cpu.values.size(),
            "distributed CUDA global size mismatch"
        );
        require(
            distributed.local_values.size() == distributed.global_size / distributed.world_size,
            "distributed CUDA local shard size mismatch"
        );
        require(
            distributed.backend.rfind("native-mpi-cuda:", 0U) == 0U,
            "distributed CUDA backend identity mismatch"
        );
        for (std::size_t index = 0U; index < distributed.local_values.size(); ++index) {
            require_complex_close(
                distributed.local_values[index],
                cpu.values[distributed.global_offset + index],
                "distributed CUDA state shard mismatch"
            );
        }
    } catch (const std::runtime_error& error) {
        require(
            std::string(error.what()).find(
                "distributed CUDA execution requires a usable mapped CUDA device"
            ) != std::string::npos,
            "distributed CUDA failed after device readiness validation"
        );
    }
}

void test_distributed_tensor_network() {
    const qupy::DistributedInfo info = qupy::distributed_info();
    const qupy::Program program = distributed_program();
    const qupy::Observable observable = left_observable();
    const auto local = qupy::tensor_network_expectation(program, observable);
    const auto distributed = qupy::distributed_tensor_network_expectation(program, observable);

    require_close(distributed.value, local.value, "distributed tensor-network value mismatch");
    require(distributed.term_count == observable.terms().size(), "distributed tensor term count mismatch");
    require(distributed.contractions == local.contractions, "distributed tensor contraction count mismatch");
    require(distributed.peak_tensor_rank == local.peak_tensor_rank, "distributed tensor peak rank mismatch");
    require(distributed.peak_tensor_bytes == local.peak_tensor_bytes, "distributed tensor peak bytes mismatch");
    require(
        std::abs(distributed.scalar_multiplications - local.scalar_multiplications) < 1e-12,
        "distributed tensor work mismatch"
    );
    require(distributed.world_size == info.world_size, "distributed tensor world size mismatch");
    require(
        distributed.active_ranks == std::min(info.world_size, observable.terms().size()),
        "distributed tensor work did not span expected MPI ranks"
    );
    require(distributed.exact, "distributed tensor-network result must be exact");
    require(distributed.backend == "native-mpi-tn", "distributed tensor backend mismatch");
    require(
        distributed.method == "mpi-term-parallel-greedy-contraction",
        "distributed tensor method mismatch"
    );

    bool collective_failure = false;
    try {
        static_cast<void>(qupy::distributed_tensor_network_expectation(program, observable, 128U));
    } catch (const std::runtime_error& error) {
        collective_failure = std::string(error.what()).find("failed on one or more MPI ranks") !=
            std::string::npos;
    }
    require(collective_failure, "rank-local tensor failure was not propagated collectively");
}

qupy::Observable z_observable() {
    return qupy::Observable({
        qupy::PauliTerm(1.0, {{0U, qupy::Pauli::Z}}),
    });
}

void test_distributed_trajectories() {
    const qupy::DistributedInfo info = qupy::distributed_info();
    qupy::Program excited(1U);
    excited = qupy::x(excited, 0U);
    const qupy::Observable z = z_observable();

    const qupy::NoisyProgram deterministic(
        excited,
        {{1U, qupy::bit_flip(0U, 1.0)}}
    );
    const auto deterministic_result = qupy::distributed_trajectory_expectations(
        deterministic, {z}, 64U, 17U
    );
    require_close(deterministic_result.values.front(), 1.0, "distributed deterministic trajectory mismatch");
    require_close(deterministic_result.standard_errors.front(), 0.0, "distributed deterministic trajectory error mismatch");
    require(deterministic_result.world_size == info.world_size, "distributed trajectory world size mismatch");
    require(deterministic_result.active_ranks == info.world_size, "trajectory ensemble did not use every MPI rank");
    require(deterministic_result.state_bytes_per_rank == 2U * sizeof(qupy::Complex), "trajectory per-rank state bytes mismatch");
    require(!deterministic_result.exact, "distributed trajectory result was marked exact");
    require(deterministic_result.backend == "native-mpi-trajectory", "distributed trajectory backend mismatch");
    require(deterministic_result.method == "mpi-trajectory-ensemble", "distributed trajectory method mismatch");

    const auto sparse = qupy::distributed_trajectory_expectations(
        deterministic, {z}, 1U, 17U
    );
    require_close(sparse.values.front(), 1.0, "sparse distributed trajectory mismatch");
    require(std::isnan(sparse.standard_errors.front()), "single distributed trajectory error must be NaN");
    require(sparse.active_ranks == 1U, "inactive trajectory ranks were reported as active");
    require(sparse.world_size == info.world_size, "sparse distributed trajectory world size mismatch");

    const double gamma = 0.25;
    const qupy::NoisyProgram damping(
        excited,
        {{1U, qupy::amplitude_damping(0U, gamma)}}
    );
    const auto first = qupy::distributed_trajectory_expectations(
        damping, {z}, 4096U, 123456U
    );
    const auto replay = qupy::distributed_trajectory_expectations(
        damping, {z}, 4096U, 123456U
    );
    require(
        std::abs(first.values.front() - (2.0 * gamma - 1.0)) < 0.05,
        "distributed trajectory ensemble did not converge"
    );
    require(first.standard_errors.front() > 0.0, "distributed stochastic trajectory error missing");
    require(first.standard_errors.front() < 0.03, "distributed trajectory standard error too large");
    require(first.values == replay.values, "distributed fixed-seed replay changed values");
    require(
        first.standard_errors == replay.standard_errors,
        "distributed fixed-seed replay changed standard errors"
    );
    require(first.seed == 123456U, "distributed trajectory seed mismatch");
}

}  // namespace

int main() {
    try {
        const qupy::DistributedInfo info = qupy::distributed_info();
        require(qupy::mpi_compiled(), "MPI test requires MPI support");
        require(info.world_size == 2U || info.world_size == 4U, "unexpected MPI world size");
        require(info.rank < info.world_size, "MPI rank is outside the world");
        test_observable_reductions();
        test_observable_batch_and_gradient();
        test_distributed_cuda_statevector();
        test_distributed_tensor_network();
        test_distributed_trajectories();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
