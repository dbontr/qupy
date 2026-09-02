#include "qupy/core.hpp"
#include "stabilizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

constexpr double kTolerance = 1e-12;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(qupy::Complex actual, qupy::Complex expected, const std::string& message) {
    require(std::abs(actual - expected) <= kTolerance, message);
}

[[nodiscard]] double dense_z_value(const qupy::Program& program, std::size_t qubit) {
    const qupy::StateVector state = qupy::statevector(program);
    const std::size_t mask = std::size_t{1} << qubit;
    double value = 0.0;
    for (std::size_t index = 0; index < state.values.size(); ++index) {
        value += ((index & mask) == 0U ? 1.0 : -1.0) * std::norm(state.values[index]);
    }
    return value;
}

[[nodiscard]] qupy::Program append_clifford_gate(qupy::Program program, std::size_t gate) {
    switch (gate) {
    case 0U: return qupy::h(program, 0);
    case 1U: return qupy::h(program, 1);
    case 2U: return qupy::x(program, 0);
    case 3U: return qupy::x(program, 1);
    case 4U: return qupy::y(program, 0);
    case 5U: return qupy::y(program, 1);
    case 6U: return qupy::z(program, 0);
    case 7U: return qupy::z(program, 1);
    case 8U: return qupy::cx(program, 0, 1);
    case 9U: return qupy::cx(program, 1, 0);
    case 10U: return qupy::cz(program, 0, 1);
    case 11U: return qupy::swap(program, 0, 1);
    default: throw std::logic_error("unknown Clifford test gate");
    }
}

void test_bell_state() {
    qupy::Program program(2);
    program = qupy::h(program, 0);
    program = qupy::cx(program, 0, 1);

    const qupy::StateVector result = qupy::statevector(program);
    const double amplitude = 1.0 / std::sqrt(2.0);
    require(result.values.size() == 4, "Bell state dimension is wrong");
    require_close(result.values[0], amplitude, "Bell |00> amplitude is wrong");
    require_close(result.values[1], 0.0, "Bell |01> amplitude is wrong");
    require_close(result.values[2], 0.0, "Bell |10> amplitude is wrong");
    require_close(result.values[3], amplitude, "Bell |11> amplitude is wrong");
    require(result.backend == "native-cpu", "Bell state did not use the native backend");
}

void test_rotation_and_pauli_gates() {
    qupy::Program program(1);
    program = qupy::y(program, 0);
    const auto y_state = qupy::statevector(program);
    require_close(y_state.values[0], 0.0, "Y |0> zero amplitude is wrong");
    require_close(y_state.values[1], {0.0, 1.0}, "Y |0> one amplitude is wrong");

    program = qupy::Program(1);
    program = qupy::ry(program, std::acos(-1.0), 0);
    const auto ry_state = qupy::statevector(program);
    require_close(ry_state.values[0], 0.0, "RY(pi) zero amplitude is wrong");
    require_close(ry_state.values[1], 1.0, "RY(pi) one amplitude is wrong");

    program = qupy::Program(1);
    program = qupy::x(program, 0);
    program = qupy::rz(program, std::acos(-1.0), 0);
    const auto rz_state = qupy::statevector(program);
    require_close(rz_state.values[0], 0.0, "RZ(pi) zero amplitude is wrong");
    require_close(rz_state.values[1], {0.0, 1.0}, "RZ(pi) phase is wrong");
}

void test_two_qubit_gates() {
    qupy::Program program(2);
    program = qupy::x(program, 0);
    program = qupy::swap(program, 0, 1);
    auto state = qupy::statevector(program);
    require_close(state.values[2], 1.0, "SWAP did not move the excitation");

    program = qupy::Program(2);
    program = qupy::x(program, 0);
    program = qupy::x(program, 1);
    program = qupy::cz(program, 0, 1);
    state = qupy::statevector(program);
    require_close(state.values[3], -1.0, "CZ did not apply the |11> phase");
}

void test_results_and_planner() {
    qupy::Program program(2);
    program = qupy::h(program, 0);
    program = qupy::cx(program, 0, 1);

    const auto expectation = qupy::expectation(program, qupy::pauli_z(0));
    require(std::abs(expectation.value) <= kTolerance, "Bell Z expectation is not zero");

    const auto samples = qupy::sample(program, 256, 7U);
    const auto counts = samples.counts();
    require(counts.size() == 2, "Bell sampling returned impossible states");
    require(counts.contains("00"), "Bell sampling did not return |00>");
    require(counts.contains("11"), "Bell sampling did not return |11>");

    const auto deterministic = qupy::sample(program, 16, 7U);
    const std::vector<std::int8_t> expected = {
        0, 0, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 0, 0, 0, 0,
        1, 1, 0, 0, 1, 1, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0,
    };
    require(deterministic.values == expected, "seeded sampling sequence changed");

    const auto execution_plan = qupy::plan(program, qupy::ResultMode::Sample);
    require(execution_plan.exact, "native plan must be exact");
    require(execution_plan.threads == 1, "small native plan must remain serial");
    require(execution_plan.original_operations == 2, "plan operation count is wrong");
    require(execution_plan.compiled_steps == 2, "Bell plan step count is wrong");
    require(execution_plan.active_qubits == 2, "Bell plan active qubits are wrong");
    require(execution_plan.estimated_state_bytes == 64, "Bell plan memory is wrong");

    const auto parallel_plan = qupy::plan(qupy::Program(16), qupy::ResultMode::StateVector);
    const std::size_t expected_threads = std::min<std::size_t>(qupy::parallel_threads(), 8U);
    require(parallel_plan.threads == expected_threads, "16-qubit plan thread count is wrong");

    const auto large_parallel_plan = qupy::plan(
        qupy::Program(20), qupy::ResultMode::StateVector
    );
    const std::size_t expected_large_threads = std::min<std::size_t>(
        qupy::parallel_threads(), 16U
    );
    require(
        large_parallel_plan.threads == expected_large_threads,
        "large native plan exceeded the verified OpenMP team ceiling"
    );
}

void test_native_planner_cost_artifact() {
    const std::filesystem::path artifact = std::filesystem::temp_directory_path() /
        ("qupy-planner-cost-" + qupy::planner_host_fingerprint().substr(0, 12) + ".qpcost");
    {
        std::ofstream output(artifact);
        require(static_cast<bool>(output), "planner cost fixture could not be created");
        output << "qupy-planner-cost 1\n";
        output << "engine " << qupy::core_version() << "\n";
        output << "workload 1\n";
        output << "host " << qupy::planner_host_fingerprint() << "\n";
        output << "validated 1\n";
        output << "model pauli-propagation 2 5 0.85 1.1 1.2\n";
        output << "model statevector-parallel 3 4.5 0.55 0.012 1.1 1.2\n";
        output << "model statevector-serial 3 4.5 0.55 0.012 1.1 1.2\n";
    }

    const qupy::PlannerCostModel model = qupy::load_planner_cost_model(artifact.string());
    require(model.schema_version() == 1U, "planner cost schema version is wrong");
    require(model.workload_version() == 1U, "planner workload version is wrong");
    require(model.engine_version() == qupy::core_version(), "planner engine version is wrong");
    require(
        model.host_fingerprint() == qupy::planner_host_fingerprint(),
        "planner host fingerprint is wrong"
    );
    require(model.artifact_fingerprint().size() == 64U, "planner artifact fingerprint is wrong");

    qupy::Program program(1);
    program = qupy::ry(program, 0.37, 0);
    const auto base = qupy::expectation_plan(program, qupy::pauli_z(0));
    const auto costed = qupy::expectation_plan(program, qupy::pauli_z(0), "auto", &model);
    require(base.method == costed.method, "cost evidence changed planner method selection");
    require(base.backend == costed.backend, "cost evidence changed planner backend selection");
    require(base.cache_key == costed.cache_key, "cost evidence changed execution cache identity");
    require(!base.predicted_ns.has_value(), "uncalibrated plan reported a predicted cost");
    require(costed.predicted_ns.has_value(), "calibrated plan did not report a predicted cost");
    require(*costed.predicted_ns > 0.0, "calibrated plan predicted a non-positive cost");
    require(
        costed.cost_model_fingerprint == model.artifact_fingerprint(),
        "plan cost-model provenance is wrong"
    );

    qupy::Program stabilizer_program(24U);
    stabilizer_program = qupy::h(stabilizer_program, 0U);
    for (std::size_t qubit = 1U; qubit < 24U; ++qubit) {
        stabilizer_program = qupy::cx(stabilizer_program, qubit - 1U, qubit);
    }
    const auto stabilizer_plan = qupy::plan(
        stabilizer_program, qupy::ResultMode::Sample, "auto", &model
    );
    require(stabilizer_plan.method == "stabilizer", "cost evidence blocked stabilizer planning");
    require(
        !stabilizer_plan.predicted_ns.has_value(),
        "state-vector cost model predicted an unsupported stabilizer plan"
    );
    require(
        stabilizer_plan.cost_model_fingerprint.empty(),
        "unsupported stabilizer plan claimed cost-model provenance"
    );
    const auto v1_statevector_plan = qupy::plan(
        program, qupy::ResultMode::StateVector, "auto", &model
    );
    require(
        !v1_statevector_plan.predicted_ns.has_value(),
        "legacy expectation cost model predicted a full state-vector return"
    );
    if (qupy::cuda_available()) {
        const auto cuda_plan = qupy::plan(
            program, qupy::ResultMode::StateVector, "native-cuda", &model
        );
        require(cuda_plan.method == "cuda-statevector", "cost evidence blocked CUDA planning");
        require(!cuda_plan.predicted_ns.has_value(), "CPU cost model predicted a CUDA plan");
        require(
            cuda_plan.cost_model_fingerprint.empty(),
            "unsupported CUDA plan claimed cost-model provenance"
        );
    }
    require(std::filesystem::remove(artifact), "planner cost fixture was not removed");
}

void test_cuda_planner_cost_artifact() {
    const std::filesystem::path artifact = std::filesystem::temp_directory_path() /
        "qupy-cuda-planner-cost.qpcost";
    const std::string cuda_host = qupy::cuda_available()
        ? qupy::planner_cuda_host_fingerprint() : std::string(64U, '0');
    {
        std::ofstream output(artifact);
        require(static_cast<bool>(output), "CUDA planner fixture could not be created");
        output << "qupy-planner-cost 2\n";
        output << "engine " << qupy::core_version() << "\n";
        output << "workload 1\n";
        output << "host " << qupy::planner_host_fingerprint() << "\n";
        output << "cuda-host " << cuda_host << "\n";
        output << "validated 1\n";
        output << "model pauli-propagation 2 5 0.85 1.1 1.2\n";
        output << "model statevector-parallel 3 4.5 0.55 0.012 1.1 1.2\n";
        output << "model statevector-serial 3 4.5 0.55 0.012 1.1 1.2\n";
        output << "model statevector-return-cpu 5 10 0 0 0 0 1.1 1.2\n";
        output << "model statevector-return-cuda 4 0 0 0 0 1.1 1.2\n";
        output << "decision statevector-auto 8 0 1.0\n";
    }

    if (!qupy::cuda_available()) {
        bool rejected = false;
        try {
            static_cast<void>(qupy::load_planner_cost_model(artifact.string()));
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "CUDA planner artifact did not fail closed without CUDA");
        require(std::filesystem::remove(artifact), "CUDA planner fixture was not removed");
        return;
    }

    const auto model = qupy::load_planner_cost_model(artifact.string());
    require(model.schema_version() == 2U, "CUDA planner schema version is wrong");
    require(model.cuda_auto_validated(), "CUDA planner decision evidence was not accepted");
    require(
        model.cuda_host_fingerprint() == qupy::planner_cuda_host_fingerprint(),
        "CUDA planner host fingerprint is wrong"
    );

    qupy::Program program(3U);
    program = qupy::h(program, 0U);
    program = qupy::cx(program, 0U, 1U);
    program = qupy::ry(program, 0.37, 2U);
    const auto selected = qupy::plan(
        program, qupy::ResultMode::StateVector, "auto", &model
    );
    require(selected.backend == "native-cuda", "CUDA planner did not select the cheaper target");
    require(selected.method == "cuda-statevector", "CUDA planner selected the wrong method");
    require(selected.predicted_ns.has_value(), "CUDA planner omitted its selected prediction");
    require(
        selected.cost_model_class == "statevector-return-cuda",
        "CUDA planner reported the wrong cost class"
    );
    const auto automatic = qupy::statevector(program, "auto", &model);
    const auto explicit_cuda = qupy::statevector(program, "native-cuda");
    require(automatic.backend == "native-cuda", "CUDA execution ignored planner selection");
    for (std::size_t index = 0U; index < automatic.values.size(); ++index) {
        require_close(
            automatic.values[index], explicit_cuda.values[index],
            "automatic CUDA statevector diverged from explicit CUDA"
        );
    }
    require(std::filesystem::remove(artifact), "CUDA planner fixture was not removed");
}


void test_adaptive_mps_planner_artifact() {
    const std::filesystem::path artifact = std::filesystem::temp_directory_path() /
        "qupy-adaptive-mps-policy.qpcost";
    {
        std::ofstream output(artifact);
        require(static_cast<bool>(output), "adaptive MPS policy fixture could not be created");
        output << "qupy-planner-cost 3\n";
        output << "engine " << qupy::core_version() << "\n";
        output << "workload 1\n";
        output << "host " << qupy::planner_host_fingerprint() << "\n";
        output << "validated 1\n";
        output << "model pauli-propagation 2 5 0.85 1.1 1.2\n";
        output << "model statevector-parallel 3 4.5 0.55 0.012 1.1 1.2\n";
        output << "model statevector-serial 3 4.5 0.55 0.012 1.1 1.2\n";
        output << "policy adaptive-mps 1\n";
        output << "decision observable-auto 29 0 1.0\n";
    }

    const auto model = qupy::load_planner_cost_model(artifact.string());
    require(model.schema_version() == 3U, "adaptive MPS policy schema version is wrong");
    require(model.mps_auto_validated(), "adaptive MPS policy evidence was not accepted");
    require(model.mps_policy_version() == 1U, "adaptive MPS policy version is wrong");
    require(!model.cuda_auto_validated(), "MPS-only artifact unexpectedly enabled CUDA selection");

    const auto target = qupy::adaptive_mps_target();
    require(target.name == "native-adaptive-mps", "adaptive MPS target name is wrong");
    require(target.supports(qupy::ResultMode::Expectation), "adaptive MPS target lacks expectation");
    require(target.supports(qupy::ResultMode::Variance), "adaptive MPS target lacks variance");
    require(!target.supports(qupy::ResultMode::StateVector), "adaptive MPS target exposes statevector");
    require(!target.parameter_batches, "adaptive MPS target exposes parameter batches");

    qupy::Program program(15U);
    program = qupy::ry(program, 0.371, 0U);
    for (std::size_t qubit = 0U; qubit + 1U < 15U; ++qubit) {
        program = qupy::cx(program, qubit, qubit + 1U);
    }
    const auto observable = qupy::pauli_z(14U);
    const auto baseline = qupy::expectation_plan(program, observable);
    const auto selected = qupy::expectation_plan(program, observable, "auto", &model);
    require(baseline.backend == "native-cpu", "default observable backend changed");
    require(selected.backend == "native-adaptive-mps", "validated policy did not select adaptive MPS");
    require(selected.method == "mps", "safe adaptive MPS plan did not resolve directly to MPS");
    require(selected.cost_model_class == "adaptive-mps-policy", "adaptive MPS policy provenance is wrong");
    require(!selected.predicted_ns.has_value(), "adaptive MPS policy reported a static prediction");
    const auto cpu = qupy::expectation(program, observable, "native-cpu");
    const auto automatic = qupy::expectation(program, observable, "auto", &model);
    require(automatic.backend == "native-adaptive-mps", "adaptive MPS execution ignored policy");
    require(
        std::abs(automatic.value - cpu.value) <= 5e-12,
        "adaptive MPS expectation diverged from dense CPU"
    );
    const auto cpu_variance = qupy::variance(program, observable, "native-cpu");
    const auto automatic_variance = qupy::variance(program, observable, "auto", &model);
    require(
        automatic_variance.backend == "native-adaptive-mps",
        "adaptive MPS variance execution ignored policy"
    );
    require(
        std::abs(automatic_variance.value - cpu_variance.value) <= 5e-12,
        "adaptive MPS variance diverged from dense CPU"
    );

    qupy::Program fallback_program(15U);
    for (std::size_t qubit = 0U; qubit < 15U; ++qubit) {
        fallback_program = qupy::ry(fallback_program, 0.019 * static_cast<double>(qubit + 1U), qubit);
    }
    for (std::size_t layer = 0U; layer < 7U; ++layer) {
        const std::size_t parity = layer % 2U;
        for (std::size_t qubit = parity; qubit + 1U < 15U; qubit += 2U) {
            fallback_program = qupy::cz(fallback_program, qubit, qubit + 1U);
        }
        for (std::size_t qubit = 0U; qubit < 15U; ++qubit) {
            fallback_program = qupy::rz(
                fallback_program,
                0.007 * static_cast<double>(layer + qubit + 1U),
                qubit
            );
        }
    }
    for (std::size_t qubit = 0U; qubit + 1U < 15U; ++qubit) {
        fallback_program = qupy::cx(fallback_program, qubit, qubit + 1U);
    }
    const auto fallback_observable = qupy::pauli_z(14U);
    const auto fallback_plan = qupy::expectation_plan(
        fallback_program, fallback_observable, "native-adaptive-mps"
    );
    require(fallback_plan.method == "adaptive-mps", "bond-growth plan did not use adaptive MPS");
    require(
        fallback_plan.tensor_network_max_bond > 16U,
        "bond-growth plan did not cross the adaptive checkpoint limit"
    );
    const auto fallback_cpu = qupy::expectation(
        fallback_program, fallback_observable, "native-cpu"
    );
    const auto fallback_adaptive = qupy::expectation(
        fallback_program, fallback_observable, "native-adaptive-mps"
    );
    require(
        std::abs(fallback_adaptive.value - fallback_cpu.value) <= 5e-12,
        "adaptive MPS checkpoint continuation diverged from dense CPU"
    );

    bool rejected = false;
    try {
        static_cast<void>(qupy::statevector(program, "native-adaptive-mps"));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "adaptive MPS target did not reject statevector return");
    require(std::filesystem::remove(artifact), "adaptive MPS policy fixture was not removed");
}

void test_parameter_binding_and_batches() {
    const double pi = std::acos(-1.0);
    qupy::Program templated(1);
    templated = qupy::ry(templated, 0.0, 0);
    const std::string template_fingerprint = templated.fingerprint();
    const std::vector<qupy::ParameterSlot> slots{{0U, 0U}};

    const qupy::Program bound = templated.bound(slots, {pi});
    require(templated.fingerprint() == template_fingerprint, "parameter binding mutated its template");
    require(bound.fingerprint() != template_fingerprint, "parameter binding did not change identity");
    require(
        std::abs(qupy::expectation(bound, qupy::pauli_z(0)).value + 1.0) <= kTolerance,
        "bound program expectation is wrong"
    );

    const qupy::ExpectationBatch batch = qupy::expectation_batch(
        templated,
        qupy::pauli_z(0),
        slots,
        {0.0, pi / 2.0, pi},
        3U
    );
    require(batch.values.size() == 3U, "parameter batch result size is wrong");
    require(std::abs(batch.values[0] - 1.0) <= kTolerance, "batch row 0 is wrong");
    require(std::abs(batch.values[1]) <= kTolerance, "batch row 1 is wrong");
    require(std::abs(batch.values[2] + 1.0) <= kTolerance, "batch row 2 is wrong");
    require(batch.batch_size == 3U, "parameter batch row count is wrong");
    require(batch.parameter_count == 1U, "parameter batch column count is wrong");
    require(batch.active_qubits == 1U, "parameter batch active qubits are wrong");
    require(batch.estimated_state_bytes == 32U, "parameter batch state estimate is wrong");
    require(qupy::native_target().parameter_batches, "native target does not advertise batches");

    qupy::Program fused(1);
    fused = qupy::rx(fused, 0.0, 0);
    fused = qupy::ry(fused, 0.0, 0);
    const std::vector<qupy::ParameterSlot> fused_slots{{0U, 0U}, {1U, 0U}};
    const std::vector<double> fused_values{0.1, 0.2, -0.3, 0.4};
    const qupy::ExpectationBatch fused_batch = qupy::expectation_batch(
        fused, qupy::pauli_z(0), fused_slots, fused_values, 2U
    );
    for (std::size_t row = 0; row < 2U; ++row) {
        const qupy::Program scalar_program = fused.bound(
            fused_slots,
            {fused_values[row * 2U], fused_values[row * 2U + 1U]}
        );
        const double scalar = qupy::expectation(scalar_program, qupy::pauli_z(0)).value;
        require(
            std::abs(fused_batch.values[row] - scalar) <= kTolerance,
            "multi-slot batch diverged from scalar binding"
        );
    }

    qupy::Program irrelevant(2);
    irrelevant = qupy::ry(irrelevant, 0.37, 0);
    irrelevant = qupy::ry(irrelevant, 0.0, 1);
    const qupy::ExpectationBatch irrelevant_batch = qupy::expectation_batch(
        irrelevant,
        qupy::pauli_z(0),
        {{1U, 0U}},
        {0.0, 1.0, 2.0},
        3U
    );
    const double expected_irrelevant = std::cos(0.37);
    require(irrelevant_batch.active_qubits == 1U, "irrelevant batch slot expanded causal cone");
    require(
        std::all_of(
            irrelevant_batch.values.begin(),
            irrelevant_batch.values.end(),
            [&](double value) { return std::abs(value - expected_irrelevant) <= kTolerance; }
        ),
        "irrelevant parameter slot changed an observable"
    );

    qupy::Program sampled(2);
    sampled = qupy::ry(sampled, 0.0, 0);
    sampled = qupy::cx(sampled, 0, 1);
    const qupy::SamplesBatch sampled_batch = qupy::sample_batch(
        sampled, slots, {0.0, pi}, 2U, 32U, 7U
    );
    require(sampled_batch.batch_size == 2U, "sample batch row count is wrong");
    require(sampled_batch.shots == 32U, "sample batch shot count is wrong");
    require(sampled_batch.num_qubits == 2U, "sample batch qubit count is wrong");
    require(sampled_batch.parameter_count == 1U, "sample batch parameter count is wrong");
    require(sampled_batch.values.size() == 128U, "sample batch result size is wrong");
    require(sampled_batch.counts(0).at("00") == 32U, "sample batch row 0 is wrong");
    require(sampled_batch.counts(1).at("11") == 32U, "sample batch row 1 is wrong");

    const qupy::SamplesBatch one_row = qupy::sample_batch(
        sampled, slots, {pi / 2.0}, 1U, 32U, 19U
    );
    const qupy::Samples scalar_samples = qupy::sample(
        sampled.bound(slots, {pi / 2.0}), 32U, 19U
    );
    require(
        one_row.values == scalar_samples.values,
        "one-row sample batch changed deterministic seeded sampling"
    );

    bool rejected = false;
    try {
        static_cast<void>(templated.bound({{0U, 0U}, {0U, 0U}}, {0.1, 0.2}));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "duplicate parameter slots were not rejected");

    rejected = false;
    try {
        static_cast<void>(qupy::expectation_batch(
            templated, qupy::pauli_z(0), slots, {0.0, 1.0}, 3U
        ));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "invalid parameter batch shape was not rejected");

    rejected = false;
    try {
        static_cast<void>(qupy::expectation_batch(
            templated,
            qupy::pauli_z(0),
            slots,
            {std::numeric_limits<double>::quiet_NaN()},
            1U
        ));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "non-finite parameter batch value was not rejected");
}

[[nodiscard]] qupy::Program ghz_program(std::size_t num_qubits) {
    qupy::Program program(num_qubits);
    program = qupy::h(program, 0U);
    for (std::size_t qubit = 1U; qubit < num_qubits; ++qubit) {
        program = qupy::cx(program, qubit - 1U, qubit);
    }
    return program;
}

void require_stabilizer_support_matches_dense(
    const qupy::Program& program,
    const std::string& label
) {
    require(program.num_qubits() <= 10U, "support conformance fixture is too large");
    const qupy::detail::StabilizerSupport support =
        qupy::detail::build_stabilizer_support(program);
    require(support.word_count == 1U, "small support fixture unexpectedly spans multiple words");
    require(support.rank < 63U, "support fixture rank is too large to enumerate");

    const std::size_t dimension = std::size_t{1} << program.num_qubits();
    const std::size_t support_size = std::size_t{1} << support.rank;
    std::vector<bool> expected_support(dimension, false);
    for (std::size_t mask = 0U; mask < support_size; ++mask) {
        std::uint64_t basis = support.base[0];
        for (std::size_t row = 0U; row < support.rank; ++row) {
            if (((mask >> row) & 1U) != 0U) {
                basis ^= support.generators[row];
            }
        }
        require(basis < dimension, label + " produced an out-of-range support state");
        require(!expected_support[static_cast<std::size_t>(basis)], label + " support basis is dependent");
        expected_support[static_cast<std::size_t>(basis)] = true;
    }

    const qupy::StateVector dense = qupy::statevector(program);
    const double expected_probability = 1.0 / static_cast<double>(support_size);
    for (std::size_t basis = 0U; basis < dimension; ++basis) {
        const double probability = std::norm(dense.values[basis]);
        if (expected_support[basis]) {
            require(
                std::abs(probability - expected_probability) <= kTolerance,
                label + " support probability disagrees with dense statevector"
            );
        } else {
            require(
                probability <= kTolerance,
                label + " omitted a nonzero dense-state basis value"
            );
        }
    }
}

void test_stabilizer_support_matches_dense_statevector() {
    qupy::Program shifted(4);
    shifted = qupy::x(shifted, 0U);
    shifted = qupy::y(shifted, 1U);
    shifted = qupy::z(shifted, 1U);
    shifted = qupy::h(shifted, 2U);
    shifted = qupy::cx(shifted, 2U, 3U);
    shifted = qupy::cz(shifted, 0U, 3U);
    shifted = qupy::swap(shifted, 1U, 2U);
    require_stabilizer_support_matches_dense(shifted, "shifted Clifford fixture");

    qupy::Program mixed(5);
    mixed = qupy::h(mixed, 0U);
    mixed = qupy::h(mixed, 2U);
    mixed = qupy::x(mixed, 4U);
    mixed = qupy::cx(mixed, 0U, 1U);
    mixed = qupy::cz(mixed, 2U, 3U);
    mixed = qupy::swap(mixed, 1U, 4U);
    mixed = qupy::y(mixed, 3U);
    mixed = qupy::h(mixed, 4U);
    mixed = qupy::cx(mixed, 4U, 0U);
    mixed = qupy::z(mixed, 2U);
    require_stabilizer_support_matches_dense(mixed, "mixed Clifford fixture");

    const qupy::Program ghz = ghz_program(6U);
    require_stabilizer_support_matches_dense(ghz, "GHZ Clifford fixture");

    for (std::size_t fixture = 0U; fixture < 64U; ++fixture) {
        qupy::Program generated(2U);
        for (std::size_t step = 0U; step < 24U; ++step) {
            const std::size_t gate = (fixture * 5U + step * 7U + step * step) % 12U;
            generated = append_clifford_gate(std::move(generated), gate);
        }
        require_stabilizer_support_matches_dense(
            generated, "generated Clifford fixture " + std::to_string(fixture)
        );
    }

    qupy::Program non_clifford(2);
    non_clifford = qupy::ry(non_clifford, 0.37, 0U);
    require(
        !qupy::detail::supports_stabilizer(non_clifford),
        "non-Clifford rotation was accepted by stabilizer support"
    );
    bool rejected = false;
    try {
        static_cast<void>(qupy::detail::build_stabilizer_support(non_clifford));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "non-Clifford stabilizer construction was not rejected");
}

void test_stabilizer_sampling_planner_and_execution() {
    const qupy::Program below_threshold = ghz_program(23U);
    const auto below_plan = qupy::plan(below_threshold, qupy::ResultMode::Sample);
    require(below_plan.method == "statevector", "small Clifford sampling changed methods");

    const qupy::Program large = ghz_program(24U);
    const auto large_plan = qupy::plan(large, qupy::ResultMode::Sample);
    require(large_plan.method == "stabilizer", "large Clifford sampling missed stabilizer execution");
    require(large_plan.threads == 1U, "stabilizer execution must currently be serial");
    require(large_plan.estimated_state_bytes == 408U, "24-qubit stabilizer memory estimate is wrong");
    require(
        large_plan.estimated_state_bytes < (std::size_t{1} << 24U) * sizeof(qupy::Complex),
        "stabilizer plan did not reduce exponential state memory"
    );

    qupy::Program non_clifford(24U);
    non_clifford = qupy::ry(non_clifford, 0.37, 0U);
    require(
        qupy::plan(non_clifford, qupy::ResultMode::Sample).method == "statevector",
        "non-Clifford sampling incorrectly selected stabilizer execution"
    );

    const qupy::Samples first = qupy::sample(large, 32U, 7U);
    const qupy::Samples second = qupy::sample(large, 32U, 7U);
    require(first.values == second.values, "stabilizer seeded sampling is not deterministic");
    const auto counts = first.counts();
    require(counts.size() == 2U, "GHZ stabilizer sampling returned impossible basis states");
    const std::string zeros(24U, '0');
    const std::string ones(24U, '1');
    require(counts.contains(zeros), "GHZ stabilizer sampling did not return all-zero state");
    require(counts.contains(ones), "GHZ stabilizer sampling did not return all-one state");
    const std::string expected_bits = "11100101100110110110011011010111";
    for (std::size_t shot = 0U; shot < expected_bits.size(); ++shot) {
        const std::int8_t expected = static_cast<std::int8_t>(expected_bits[shot] - '0');
        for (std::size_t qubit = 0U; qubit < 24U; ++qubit) {
            require(
                first.values[shot * 24U + qubit] == expected,
                "stabilizer seeded sampling sequence changed"
            );
        }
    }

    const qupy::SamplesBatch batch = qupy::sample_batch(
        large, {}, {}, 2U, 16U, 7U
    );
    require(batch.batch_size == 2U, "stabilizer sample batch row count is wrong");
    require(batch.estimated_state_bytes == 408U, "stabilizer sample batch memory is wrong");
    require(
        batch.values == first.values,
        "stabilizer batch did not preserve the scalar row-major random stream"
    );
    for (std::size_t row = 0U; row < batch.batch_size; ++row) {
        const auto row_counts = batch.counts(row);
        require(row_counts.size() == 2U, "stabilizer sample batch returned impossible states");
        require(row_counts.contains(zeros), "stabilizer sample batch omitted all-zero state");
        require(row_counts.contains(ones), "stabilizer sample batch omitted all-one state");
    }
}

void test_internal_state_workspace_resets_between_calls() {
    const double pi = std::acos(-1.0);
    qupy::Program large(3);
    large = qupy::ry(large, pi, 0);
    large = qupy::cx(large, 0, 1);
    large = qupy::cx(large, 1, 2);

    const auto first = qupy::expectation(large, qupy::pauli_z(2));
    require(std::abs(first.value + 1.0) <= kTolerance, "large workspace expectation is wrong");
    const auto large_probabilities = qupy::probabilities(large);
    require_close(large_probabilities.values.back(), 1.0, "large workspace state is wrong");

    qupy::Program small(1);
    small = qupy::ry(small, 0.0, 0);
    const auto middle = qupy::expectation(small, qupy::pauli_z(0));
    require(std::abs(middle.value - 1.0) <= kTolerance, "workspace did not reset when shrinking");

    const auto repeated = qupy::expectation(large, qupy::pauli_z(2));
    require(std::abs(repeated.value + 1.0) <= kTolerance, "workspace did not reset when growing");
}

void test_semantic_identity() {
    qupy::Program first(2);
    first = qupy::h(first, 0);
    first = qupy::cx(first, 0, 1);

    qupy::Program second(2);
    second = qupy::h(second, 0);
    second = qupy::cx(second, 0, 1);

    require(first.canonical_text() == second.canonical_text(), "canonical IR is not deterministic");
    require(first.fingerprint() == second.fingerprint(), "program fingerprint is not deterministic");
    require(first.fingerprint().size() == 64, "program fingerprint width is wrong");
    require(
        first.fingerprint() == "ab7840ba9d0cd5353fe9e66c9100b195a8f5ad566f13e82f8d775e350f7e8009",
        "program SHA-256 fingerprint is wrong"
    );

    const qupy::Program changed = qupy::x(second, 1);
    require(first.fingerprint() != changed.fingerprint(), "different programs share a fingerprint");

    const qupy::Target target = qupy::native_target();
    require(target.state_access, "native target must report state access");
    require(!target.dynamic_control, "native target must not report dynamic control");
    require(target.fingerprint() == qupy::native_target().fingerprint(), "target fingerprint changed");

    const auto sample_plan = qupy::plan(first, qupy::ResultMode::Sample);
    const auto state_plan = qupy::plan(first, qupy::ResultMode::StateVector);
    require(sample_plan.program_fingerprint == first.fingerprint(), "plan program fingerprint is wrong");
    require(sample_plan.target_fingerprint == target.fingerprint(), "plan target fingerprint is wrong");
    require(sample_plan.result_mode == qupy::ResultMode::Sample, "plan result mode is wrong");
    require(sample_plan.workload_version == 1U, "workload fingerprint version is wrong");
    require(sample_plan.original_qubits == 2U, "workload original qubit count is wrong");
    require(sample_plan.active_operations == 2U, "workload active operation count is wrong");
    require(sample_plan.single_qubit_operations == 1U, "workload single-qubit count is wrong");
    require(sample_plan.two_qubit_operations == 1U, "workload two-qubit count is wrong");
    require(sample_plan.parameterized_operations == 0U, "workload parameter count is wrong");
    require(sample_plan.non_clifford_operations == 0U, "workload non-Clifford count is wrong");
    require(
        sample_plan.workload_fingerprint ==
            "6146406a5bd9baf7b57435a3815bc8427a2a6990b1c183fd10db46cebb841b0d",
        "workload SHA-256 fingerprint is wrong"
    );
    require(sample_plan.workload_fingerprint != state_plan.workload_fingerprint,
        "different result modes share a workload fingerprint");
    require(sample_plan.cache_key != state_plan.cache_key, "different result modes share a cache key");

    const auto z0 = qupy::expectation_plan(first, qupy::pauli_z(0));
    const auto z1 = qupy::expectation_plan(first, qupy::pauli_z(1));
    require(z0.cache_key != z1.cache_key, "different observables share a cache key");
}

void test_compiler_fusion() {
    qupy::Program program(2);
    program = qupy::h(program, 0);
    program = qupy::rx(program, 0.2, 0);
    program = qupy::rz(program, -0.4, 0);
    program = qupy::x(program, 1);
    program = qupy::ry(program, 0.3, 1);
    const auto execution_plan = qupy::plan(program, qupy::ResultMode::StateVector);
    require(execution_plan.original_operations == 5, "fusion input count is wrong");
    require(execution_plan.compiled_steps == 2, "single-qubit fusion did not reduce steps");
}

void test_clifford_expectation_uses_pauli_propagation() {
    qupy::Program program(100);
    program = qupy::h(program, 0);
    program = qupy::x(program, 98);
    program = qupy::ry(program, 0.7, 99);

    const auto execution_plan = qupy::expectation_plan(program, qupy::pauli_z(0));
    require(execution_plan.method == "pauli-propagation", "Pauli propagation was not selected");
    require(execution_plan.active_qubits == 1, "Pauli propagation retained unrelated qubits");
    require(execution_plan.compiled_steps == 1, "Pauli propagation work estimate is wrong");
    require(execution_plan.estimated_state_bytes == 0, "Pauli propagation allocated state memory");

    const auto result = qupy::expectation(program, qupy::pauli_z(0));
    require(std::abs(result.value) <= kTolerance, "Pauli-propagated expectation is wrong");
    require(result.active_qubits == 1, "expectation metadata has wrong active qubit count");
    require(result.estimated_state_bytes == 0, "expectation metadata reports state memory");

    qupy::Program sign_program(2);
    sign_program = qupy::x(sign_program, 0);
    sign_program = qupy::swap(sign_program, 0, 1);
    const auto signed_result = qupy::expectation(sign_program, qupy::pauli_z(1));
    require(std::abs(signed_result.value + 1.0) <= kTolerance, "Pauli sign propagation is wrong");
}

void test_non_clifford_expectation_falls_back_to_statevector_lightcone() {
    qupy::Program program(100);
    program = qupy::h(program, 0);
    program = qupy::ry(program, 0.7, 0);
    program = qupy::x(program, 99);

    const auto execution_plan = qupy::expectation_plan(program, qupy::pauli_z(0));
    require(execution_plan.method == "statevector-lightcone", "non-Clifford cone did not fall back");
    require(execution_plan.active_qubits == 1, "fallback retained unrelated qubits");
    require(execution_plan.estimated_state_bytes == 32, "fallback memory estimate is wrong");

    const auto result = qupy::expectation(program, qupy::pauli_z(0));
    require(
        std::abs(result.value + std::sin(0.7)) <= kTolerance,
        "non-Clifford fallback expectation is wrong"
    );
}

void test_pauli_propagation_matches_dense_statevector() {
    constexpr std::size_t gate_count = 12U;
    for (std::size_t first = 0; first < gate_count; ++first) {
        for (std::size_t second = 0; second < gate_count; ++second) {
            for (std::size_t third = 0; third < gate_count; ++third) {
                qupy::Program program(2);
                program = append_clifford_gate(std::move(program), first);
                program = append_clifford_gate(std::move(program), second);
                program = append_clifford_gate(std::move(program), third);
                for (std::size_t qubit = 0; qubit < 2U; ++qubit) {
                    const auto execution_plan = qupy::expectation_plan(
                        program, qupy::pauli_z(qubit)
                    );
                    require(
                        execution_plan.method == "pauli-propagation",
                        "Clifford circuit did not use Pauli propagation"
                    );
                    require(
                        execution_plan.estimated_state_bytes == 0,
                        "Pauli propagation reported state-vector memory"
                    );
                    const double propagated = qupy::expectation(
                        program, qupy::pauli_z(qubit)
                    ).value;
                    const double dense = dense_z_value(program, qubit);
                    require(
                        std::abs(propagated - dense) <= kTolerance,
                        "Pauli propagation disagrees with dense state-vector execution"
                    );
                }
            }
        }
    }
}

void test_large_clifford_cone_avoids_statevector_allocation() {
    constexpr std::size_t qubits = 4096U;
    qupy::Program program(qubits);
    program = qupy::h(program, 0);
    for (std::size_t qubit = 1; qubit < qubits; ++qubit) {
        program = qupy::cx(program, qubit - 1U, qubit);
    }

    const auto execution_plan = qupy::expectation_plan(
        program, qupy::pauli_z(qubits - 1U)
    );
    require(execution_plan.method == "pauli-propagation", "large Clifford cone used statevector");
    require(execution_plan.active_qubits == qubits, "large Clifford cone lost dependencies");
    require(execution_plan.estimated_state_bytes == 0, "large Clifford cone requested state memory");

    const auto result = qupy::expectation(program, qupy::pauli_z(qubits - 1U));
    require(std::abs(result.value) <= kTolerance, "large Clifford expectation is wrong");
    require(result.estimated_state_bytes == 0, "large Clifford result reports state memory");
}

void test_probabilities_and_variance() {
    qupy::Program program(2);
    program = qupy::h(program, 0);
    program = qupy::cx(program, 0, 1);

    const auto probabilities = qupy::probabilities(program);
    require(probabilities.values.size() == 4, "probability dimension is wrong");
    require(std::abs(probabilities.values[0] - 0.5) <= kTolerance, "P(00) is wrong");
    require(std::abs(probabilities.values[1]) <= kTolerance, "P(01) is wrong");
    require(std::abs(probabilities.values[2]) <= kTolerance, "P(10) is wrong");
    require(std::abs(probabilities.values[3] - 0.5) <= kTolerance, "P(11) is wrong");

    const auto bell_variance = qupy::variance(program, qupy::pauli_z(0));
    require(std::abs(bell_variance.value - 1.0) <= kTolerance, "Bell Z variance is wrong");
    require(bell_variance.active_qubits == 2, "variance active-qubit metadata is wrong");
    require(bell_variance.estimated_state_bytes == 0, "Clifford variance allocated state memory");

    qupy::Program basis(1);
    basis = qupy::x(basis, 0);
    const auto basis_variance = qupy::variance(basis, qupy::pauli_z(0));
    require(std::abs(basis_variance.value) <= kTolerance, "basis-state variance is not zero");

    const auto expectation_key = qupy::expectation_plan(program, qupy::pauli_z(0)).cache_key;
    const auto variance_key = qupy::variance_plan(program, qupy::pauli_z(0)).cache_key;
    require(expectation_key != variance_key, "expectation and variance share a cache key");
}

void test_cuda_statevector_backend() {
    qupy::Program program(3U);
    program = qupy::h(program, 0U);
    program = qupy::x(program, 1U);
    program = qupy::y(program, 2U);
    program = qupy::z(program, 0U);
    program = qupy::rx(program, -0.21, 1U);
    program = qupy::ry(program, 0.37, 2U);
    program = qupy::rz(program, 0.19, 0U);
    program = qupy::cx(program, 0U, 2U);
    program = qupy::cz(program, 1U, 2U);
    program = qupy::swap(program, 0U, 1U);

    if (!qupy::cuda_available()) {
        require(!qupy::cuda_unavailable_reason().empty(), "missing CUDA failure reason");
        bool rejected = false;
        try {
            static_cast<void>(qupy::statevector(program, "native-cuda"));
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        require(rejected, "unavailable CUDA backend did not fail closed");
        return;
    }

    const auto target = qupy::cuda_target();
    require(target.name == "native-cuda", "CUDA target name is wrong");
    require(target.simulator && target.state_access, "CUDA target capabilities are wrong");
    require(!target.parameter_batches, "CUDA target exposed unsupported parameter batches");
    require(target.supports(qupy::ResultMode::StateVector), "CUDA target lacks statevector result");
    require(target.supports(qupy::ResultMode::Expectation), "CUDA target lacks expectation result");
    require(target.supports(qupy::ResultMode::Variance), "CUDA target lacks variance result");
    require(!target.supports(qupy::ResultMode::Sample), "CUDA target exposed sampling");

    const auto plan = qupy::plan(program, qupy::ResultMode::StateVector, "native-cuda");
    require(plan.backend == "native-cuda", "CUDA plan backend is wrong");
    require(plan.method == "cuda-statevector", "CUDA plan method is wrong");
    require(plan.threads == 1U, "CUDA plan reported CPU threads");
    require(!plan.predicted_ns.has_value(), "CPU cost model leaked into CUDA plan");

    const auto cpu = qupy::statevector(program, "native-cpu");
    const auto gpu = qupy::statevector(program, "native-cuda");
    require(cpu.values.size() == gpu.values.size(), "CUDA state dimension is wrong");
    for (std::size_t index = 0U; index < cpu.values.size(); ++index) {
        require_close(gpu.values[index], cpu.values[index], "CUDA statevector diverged from CPU");
    }

    const qupy::PauliZ observable{2U};
    const auto expectation_plan = qupy::expectation_plan(program, observable, "native-cuda");
    const auto variance_plan = qupy::variance_plan(program, observable, "native-cuda");
    require(expectation_plan.method == "cuda-pauli-reduction", "CUDA expectation plan is wrong");
    require(variance_plan.method == "cuda-pauli-reduction", "CUDA variance plan is wrong");
    const auto cpu_expectation = qupy::expectation(program, observable, "native-cpu");
    const auto gpu_expectation = qupy::expectation(program, observable, "native-cuda");
    const auto cpu_variance = qupy::variance(program, observable, "native-cpu");
    const auto gpu_variance = qupy::variance(program, observable, "native-cuda");
    require_close(gpu_expectation.value, cpu_expectation.value, "CUDA expectation diverged from CPU");
    require_close(gpu_variance.value, cpu_variance.value, "CUDA variance diverged from CPU");

    bool rejected = false;
    try {
        static_cast<void>(qupy::sample(program, 8U, 7U, "native-cuda"));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "CUDA backend silently accepted unsupported sampling");
}

void test_mps_backend() {
    qupy::Program program(3U);
    program = qupy::h(program, 0U);
    program = qupy::ry(program, 0.37, 1U);
    program = qupy::cx(program, 0U, 2U);
    program = qupy::cz(program, 1U, 2U);
    program = qupy::swap(program, 0U, 1U);

    const qupy::Target target = qupy::mps_target();
    require(target.name == "native-mps", "MPS target name is wrong");
    require(target.supports(qupy::ResultMode::StateVector), "MPS target lacks state-vector support");
    require(target.supports(qupy::ResultMode::Expectation), "MPS target lacks expectation support");
    require(target.supports(qupy::ResultMode::Variance), "MPS target lacks variance support");
    require(!target.supports(qupy::ResultMode::Sample), "MPS target unexpectedly supports sampling");

    const auto plan = qupy::plan(program, qupy::ResultMode::StateVector, "native-mps");
    require(plan.backend == "native-mps", "MPS plan backend is wrong");
    require(plan.method == "mps-statevector", "MPS plan method is wrong");
    require(plan.threads == 1U, "MPS plan reported worker threads");
    require(plan.tensor_network_max_bond >= 2U, "MPS plan bond estimate is missing");
    require(plan.tensor_network_routed_swaps >= 2U, "MPS plan routing estimate is missing");
    require(plan.tensor_network_contraction_work > 0.0, "MPS contraction estimate is missing");

    const auto cpu = qupy::statevector(program, "native-cpu");
    const auto mps = qupy::statevector(program, "native-mps");
    require(cpu.values.size() == mps.values.size(), "MPS state dimension is wrong");
    for (std::size_t index = 0U; index < cpu.values.size(); ++index) {
        require_close(mps.values[index], cpu.values[index], "MPS statevector diverged from CPU");
    }

    const auto cpu_expectation = qupy::expectation(program, qupy::pauli_z(2U), "native-cpu");
    const auto mps_expectation = qupy::expectation(program, qupy::pauli_z(2U), "native-mps");
    require(
        std::abs(cpu_expectation.value - mps_expectation.value) <= kTolerance,
        "MPS expectation diverged from CPU"
    );
    const auto cpu_variance = qupy::variance(program, qupy::pauli_z(2U), "native-cpu");
    const auto mps_variance = qupy::variance(program, qupy::pauli_z(2U), "native-mps");
    require(
        std::abs(cpu_variance.value - mps_variance.value) <= kTolerance,
        "MPS variance diverged from CPU"
    );

    const auto automatic = qupy::plan(program, qupy::ResultMode::StateVector);
    require(automatic.backend != "native-mps", "default planner selected uncalibrated MPS execution");

    bool rejected = false;
    try {
        static_cast<void>(qupy::sample(program, 8U, 7U, "native-mps"));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "MPS backend silently accepted unsupported sampling");
}

void test_mps_randomized_conformance() {
    std::mt19937_64 generator(0x51A7E5ULL);
    std::uniform_real_distribution<double> angle(-3.0, 3.0);
    constexpr double tolerance = 5e-10;
    for (std::size_t trial = 0U; trial < 48U; ++trial) {
        const std::size_t qubits = 1U + static_cast<std::size_t>(generator() % 7U);
        qupy::Program program(qubits);
        const std::size_t gates = 16U + static_cast<std::size_t>(generator() % 25U);
        for (std::size_t gate = 0U; gate < gates; ++gate) {
            const std::size_t first = static_cast<std::size_t>(generator() % qubits);
            const std::size_t kind = static_cast<std::size_t>(
                generator() % (qubits > 1U ? 10U : 7U)
            );
            switch (kind) {
            case 0U: program = qupy::h(program, first); break;
            case 1U: program = qupy::x(program, first); break;
            case 2U: program = qupy::y(program, first); break;
            case 3U: program = qupy::z(program, first); break;
            case 4U: program = qupy::rx(program, angle(generator), first); break;
            case 5U: program = qupy::ry(program, angle(generator), first); break;
            case 6U: program = qupy::rz(program, angle(generator), first); break;
            default: {
                std::size_t second = static_cast<std::size_t>(generator() % (qubits - 1U));
                if (second >= first) {
                    ++second;
                }
                if (kind == 7U) {
                    program = qupy::cx(program, first, second);
                } else if (kind == 8U) {
                    program = qupy::cz(program, first, second);
                } else {
                    program = qupy::swap(program, first, second);
                }
                break;
            }
            }
        }

        const qupy::StateVector cpu = qupy::statevector(program, "native-cpu");
        const qupy::StateVector mps = qupy::statevector(program, "native-mps");
        require(cpu.values.size() == mps.values.size(), "MPS randomized state dimension is wrong");
        for (std::size_t index = 0U; index < cpu.values.size(); ++index) {
            require(
                std::abs(cpu.values[index] - mps.values[index]) <= tolerance,
                "MPS randomized statevector diverged from CPU"
            );
        }
        const std::size_t observable = static_cast<std::size_t>(generator() % qubits);
        const qupy::PauliZ z = qupy::pauli_z(observable);
        const qupy::Expectation cpu_expectation = qupy::expectation(program, z, "native-cpu");
        const qupy::Expectation mps_expectation = qupy::expectation(program, z, "native-mps");
        require(
            std::abs(cpu_expectation.value - mps_expectation.value) <= tolerance,
            "MPS randomized expectation diverged from CPU"
        );
        const qupy::Variance cpu_variance = qupy::variance(program, z, "native-cpu");
        const qupy::Variance mps_variance = qupy::variance(program, z, "native-mps");
        require(
            std::abs(cpu_variance.value - mps_variance.value) <= tolerance,
            "MPS randomized variance diverged from CPU"
        );
    }
}

void test_validation() {
    bool rejected = false;
    try {
        static_cast<void>(qupy::h(qupy::Program(2), 2));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "invalid qubit index was not rejected");

    rejected = false;
    try {
        static_cast<void>(qupy::cx(qupy::Program(2), 0, 0));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "duplicate two-qubit operand was not rejected");

    rejected = false;
    try {
        static_cast<void>(qupy::plan(qupy::Program(1), qupy::ResultMode::Expectation));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "observable-dependent generic plan was not rejected");
}

}  // namespace

int main() {
    try {
        require(std::string(qupy::core_language()) == "C++20", "core language is wrong");
        require(std::string(qupy::core_version()) == "0.3.0a0", "core version is wrong");
        require(qupy::ir_version() == 1U, "IR version is wrong");
        test_bell_state();
        test_rotation_and_pauli_gates();
        test_two_qubit_gates();
        test_results_and_planner();
        test_native_planner_cost_artifact();
        test_cuda_planner_cost_artifact();
        test_adaptive_mps_planner_artifact();
        test_parameter_binding_and_batches();
        test_stabilizer_support_matches_dense_statevector();
        test_stabilizer_sampling_planner_and_execution();
        test_internal_state_workspace_resets_between_calls();
        test_semantic_identity();
        test_compiler_fusion();
        test_clifford_expectation_uses_pauli_propagation();
        test_non_clifford_expectation_falls_back_to_statevector_lightcone();
        test_pauli_propagation_matches_dense_statevector();
        test_large_clifford_cone_avoids_statevector_allocation();
        test_probabilities_and_variance();
        test_cuda_statevector_backend();
        test_mps_backend();
        test_mps_randomized_conformance();
        test_validation();
        std::cout << "QuPy native core tests: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "QuPy native core tests: FAIL: " << error.what() << '\n';
        return 1;
    }
}
