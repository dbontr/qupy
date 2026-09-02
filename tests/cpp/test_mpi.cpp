#include "qupy/advanced.hpp"

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

}  // namespace

int main() {
    try {
        const qupy::DistributedInfo info = qupy::distributed_info();
        require(qupy::mpi_compiled(), "MPI test requires MPI support");
        require(info.world_size == 2U || info.world_size == 4U, "unexpected MPI world size");
        require(info.rank < info.world_size, "MPI rank is outside the world");
        test_observable_reductions();
        test_observable_batch_and_gradient();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
