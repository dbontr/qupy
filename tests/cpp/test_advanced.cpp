#include "qupy/advanced.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#ifndef QUPY_TEST_PROVIDER_PATH
#error QUPY_TEST_PROVIDER_PATH must be defined
#endif

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_observable_and_gradient() {
    qupy::Program program(1U);
    program = qupy::ry(program, 0.0, 0U);
    const qupy::Observable z({
        qupy::PauliTerm(1.0, {{0U, qupy::Pauli::Z}}),
    });
    const double theta = 0.42;
    const auto result = qupy::value_and_grad(
        program,
        z,
        {{0U, 0U}},
        {theta},
        "native-cpu",
        qupy::GradientMethod::Adjoint
    );
    require(std::abs(result.value - std::cos(theta)) < 1e-12, "adjoint value mismatch");
    require(result.gradient.size() == 1U, "adjoint gradient shape mismatch");
    require(
        std::abs(result.gradient.front() + std::sin(theta)) < 1e-12,
        "adjoint gradient mismatch"
    );
}

void test_observable_plan_and_optimizer() {
    qupy::Program bell(2U);
    bell = qupy::h(bell, 0U);
    bell = qupy::cx(bell, 0U, 1U);
    const qupy::Observable xx({
        qupy::PauliTerm(1.0, {{0U, qupy::Pauli::X}, {1U, qupy::Pauli::X}}),
    });
    const auto plan = qupy::observable_plan(bell, {xx});
    require(plan.backend == "native-cpu", "observable plan backend mismatch");
    require(plan.method == "pauli-propagation", "observable plan method mismatch");
    require(plan.estimated_state_bytes == 0U, "Pauli plan allocated a dense state");
    require(
        std::abs(qupy::expect_observable(bell, xx).value - 1.0) < 1e-12,
        "rich Pauli propagation mismatch"
    );

    qupy::Program circuit(2U);
    circuit = qupy::x(circuit, 0U);
    circuit = qupy::h(circuit, 1U);
    circuit = qupy::x(circuit, 0U);
    const auto optimized = qupy::optimize(circuit, 2U);
    require(optimized.optimized_operations == 1U, "disjoint cancellation mismatch");
    require(optimized.program.operations().front().code == qupy::OperationCode::H, "optimizer result mismatch");
}

void test_cuda_pauli_reduction() {
    if (!qupy::cuda_available()) {
        return;
    }
    qupy::Program program(3U);
    program = qupy::h(program, 0U);
    program = qupy::ry(program, 0.371, 1U);
    program = qupy::cx(program, 0U, 1U);
    program = qupy::rz(program, -0.29, 2U);
    program = qupy::cx(program, 1U, 2U);
    const qupy::Observable left({
        qupy::PauliTerm(0.41, {{0U, qupy::Pauli::X}, {1U, qupy::Pauli::Y}}),
        qupy::PauliTerm(-0.27, {{1U, qupy::Pauli::Z}, {2U, qupy::Pauli::X}}),
        qupy::PauliTerm(0.13, {}),
    });
    const qupy::Observable right({
        qupy::PauliTerm(0.36, {{0U, qupy::Pauli::Y}, {2U, qupy::Pauli::Y}}),
        qupy::PauliTerm(0.22, {{1U, qupy::Pauli::X}}),
    });

    const auto plan = qupy::observable_plan(program, {left, right}, "native-cuda");
    require(plan.backend == "native-cuda", "CUDA observable plan backend mismatch");
    require(plan.method == "cuda-pauli-reduction", "CUDA observable plan method mismatch");

    const auto cpu_expect = qupy::expect_observable(program, left, "native-cpu");
    const auto gpu_expect = qupy::expect_observable(program, left, "native-cuda");
    const auto cpu_variance = qupy::variance_observable(program, left, "native-cpu");
    const auto gpu_variance = qupy::variance_observable(program, left, "native-cuda");
    const auto cpu_covariance = qupy::covariance_observable(program, left, right, "native-cpu");
    const auto gpu_covariance = qupy::covariance_observable(program, left, right, "native-cuda");
    require(std::abs(gpu_expect.value - cpu_expect.value) < 2e-12, "CUDA observable expectation mismatch");
    require(std::abs(gpu_variance.value - cpu_variance.value) < 2e-12, "CUDA observable variance mismatch");
    require(std::abs(gpu_covariance.value - cpu_covariance.value) < 2e-12, "CUDA observable covariance mismatch");

    const auto cpu_batch = qupy::expect_observables(program, {left, right}, "native-cpu");
    const auto gpu_batch = qupy::expect_observables(program, {left, right}, "native-cuda");
    require(gpu_batch.backend == "native-cuda", "CUDA observable batch backend mismatch");
    require(gpu_batch.values.size() == cpu_batch.values.size(), "CUDA observable batch shape mismatch");
    for (std::size_t index = 0U; index < cpu_batch.values.size(); ++index) {
        require(std::abs(gpu_batch.values[index] - cpu_batch.values[index]) < 2e-12, "CUDA observable batch mismatch");
    }
}

void test_custom_kraus_channel() {
    const double gamma = 0.25;
    const double survive = std::sqrt(1.0 - gamma);
    const double decay = std::sqrt(gamma);
    const auto channel = qupy::kraus_channel(
        0U,
        {{1.0, 0.0, 0.0, survive}, {0.0, decay, 0.0, 0.0}}
    );
    require(channel.kraus_count == 2U, "custom Kraus count mismatch");
    qupy::Program excited(1U);
    excited = qupy::x(excited, 0U);
    const qupy::NoisyProgram noisy(excited, {{1U, channel}});
    const auto rho = qupy::density_matrix(noisy);
    require(std::abs(rho.values[0].real() - gamma) < 1e-12, "Kraus decay probability mismatch");
    require(std::abs(rho.values[3].real() - (1.0 - gamma)) < 1e-12, "Kraus survival probability mismatch");
}
void test_qec_interchange_and_distributed_capability() {
    const qupy::DetectorModel model(
        2U,
        1U,
        {
            {0.1, {0U}, {0U}},
            {0.2, {0U, 1U}, {}},
        }
    );
    const auto samples = qupy::sample_detector_model(model, 16U, 9U);
    require(samples.shots == 16U, "detector sample count mismatch");
    require(samples.syndrome.size() == 32U, "detector syndrome shape mismatch");
    const auto decoded = qupy::decode_detector_model(model, {0, 0});
    require(decoded.observables.size() == 1U, "decoder output shape mismatch");
    require(decoded.observables.front() == 0, "zero syndrome decoder mismatch");

    qupy::Program bell(2U);
    bell = qupy::h(bell, 0U);
    bell = qupy::cx(bell, 0U, 1U);
    require(
        qupy::to_openqasm3(bell, true).text.find("OPENQASM 3.1") != std::string::npos,
        "OpenQASM version mismatch"
    );
    require(
        qupy::to_qir_base_profile(bell, true).text.find("base_profile") != std::string::npos,
        "QIR profile mismatch"
    );
    const auto info = qupy::distributed_info();
    require(info.world_size >= 1U, "distributed world size mismatch");
    require(info.rank < info.world_size, "distributed rank mismatch");
}
void test_provider_plugin() {
    qupy::ProviderPlugin provider(QUPY_TEST_PROVIDER_PATH);
    require(provider.name() == "qupy-test-provider", "provider name mismatch");
    require(
        provider.capabilities_json().find("openqasm3") != std::string::npos,
        "provider capabilities mismatch"
    );

    qupy::Program program(2U);
    program = qupy::h(program, 0U);
    program = qupy::cx(program, 0U, 1U);
    const qupy::ProviderProgram payload = qupy::to_openqasm3(program, true);
    const std::string job_id = provider.submit(payload, 32U);
    require(job_id == "fixture-job-1", "provider job id mismatch");
    require(
        provider.poll(job_id) == qupy::ProviderJobState::Succeeded,
        "provider job state mismatch"
    );
    const std::string result = provider.result_json(job_id);
    require(result.find("\"shots\":32") != std::string::npos, "provider result mismatch");
    provider.cancel(job_id);
}

}  // namespace

int main() {
    try {
        test_observable_and_gradient();
        test_observable_plan_and_optimizer();
        test_cuda_pauli_reduction();
        test_custom_kraus_channel();
        test_qec_interchange_and_distributed_capability();
        test_provider_plugin();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
