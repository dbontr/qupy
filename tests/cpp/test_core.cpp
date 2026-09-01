#include "qupy/core.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
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
        test_parameter_binding_and_batches();
        test_internal_state_workspace_resets_between_calls();
        test_semantic_identity();
        test_compiler_fusion();
        test_clifford_expectation_uses_pauli_propagation();
        test_non_clifford_expectation_falls_back_to_statevector_lightcone();
        test_pauli_propagation_matches_dense_statevector();
        test_large_clifford_cone_avoids_statevector_allocation();
        test_probabilities_and_variance();
        test_validation();
        std::cout << "QuPy native core tests: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "QuPy native core tests: FAIL: " << error.what() << '\n';
        return 1;
    }
}
