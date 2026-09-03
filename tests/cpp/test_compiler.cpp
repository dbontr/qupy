#include "qupy/compiler.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Code = qupy::CircuitOperationCode;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Function>
void require_invalid(Function&& function, const std::string& message) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(message);
}

qupy::HardwareTarget full_target(std::size_t qubits = 3U) {
    return qupy::HardwareTarget(
        "full",
        qubits,
        {Code::H, Code::X, Code::Y, Code::Z, Code::RX, Code::RY, Code::RZ},
        {Code::CX, Code::CZ, Code::SWAP},
        {},
        true,
        true,
        true,
        true
    );
}

void test_direct_compile_and_identity() {
    qupy::Circuit circuit(2);
    circuit = circuit.h(0).cx(0, 1);
    const qupy::HardwareTarget target = full_target();
    const qupy::CompilationResult result = qupy::compile_circuit(circuit, target);

    require(result.initial_layout == std::vector<std::size_t>({0, 1}), "default layout is unstable");
    require(result.final_layout == result.initial_layout, "direct compilation changed layout");
    require(result.original_operations == 2U, "original operation count is wrong");
    require(result.optimized_operations == 2U, "optimized operation count is wrong");
    require(result.routed_operations == 2U, "routed operation count is wrong");
    require(result.compiled_operations == 2U, "compiled operation count is wrong");
    require(result.inserted_swaps == 0U, "direct compilation inserted swaps");
    require(result.decompositions == 0U, "direct compilation decomposed operations");
    require(result.depth == 2U, "direct circuit depth is wrong");
    require(!result.duration_ns.has_value(), "schedule was claimed without complete timing data");
    require(result.schedule.empty(), "schedule entries were emitted without complete timing data");
    require(result.circuit.num_qubits() == 3U, "compiled circuit does not use target width");
    require(result.target_fingerprint == target.fingerprint(), "target fingerprint was not preserved");
    require(target.fingerprint().size() == 64U, "target fingerprint is not SHA-256 sized");
    require(target.canonical_text().find("qupy-hardware-target 1") == 0U, "target identity is not versioned");
}

void test_line_routing_and_mapping() {
    const qupy::HardwareTarget target(
        "line",
        3,
        {Code::H},
        {Code::CX},
        {qupy::Coupling(0, 1), qupy::Coupling(1, 2)}
    );
    const qupy::Circuit circuit = qupy::Circuit(2).cx(0, 1);
    const qupy::CompilationResult result = qupy::compile_circuit(
        circuit, target, {0, 2}, 0
    );

    require(result.inserted_swaps == 1U, "long-range interaction did not insert a routing swap");
    require(result.final_layout == std::vector<std::size_t>({1, 2}), "routing layout update is wrong");
    require(result.routed_operations == 2U, "routed operation count is wrong");
    require(result.compiled_operations == 4U, "SWAP-to-CX translation count is wrong");
    require(result.decompositions == 1U, "routing SWAP decomposition count is wrong");
    require(result.circuit.instructions().back().code == Code::CX, "logical CX is missing after routing");
    for (const auto& instruction : result.circuit.instructions()) {
        require(instruction.code != Code::SWAP, "unsupported SWAP survived basis translation");
        if (instruction.qubits.size() == 2U) {
            require(target.adjacent(instruction.qubits[0], instruction.qubits[1]), "compiled interaction violates coupling graph");
        }
    }
}

void test_conditional_routing_restores_static_layout() {
    const qupy::HardwareTarget target(
        "dynamic-line",
        3,
        {Code::X},
        {Code::CX, Code::SWAP},
        {qupy::Coupling(0, 1), qupy::Coupling(1, 2)},
        false,
        false,
        false,
        true
    );
    const qupy::ClassicalCondition condition(0, true);
    const qupy::Circuit circuit = qupy::Circuit(2, 1).cx(0, 1, condition);
    const qupy::CompilationResult result = qupy::compile_circuit(
        circuit, target, {0, 2}, 0
    );

    require(result.inserted_swaps == 2U, "conditional routing must swap forward and back");
    require(result.final_layout == result.initial_layout, "conditional routing changed static layout");
    require(result.compiled_operations == 3U, "conditional route operation count is wrong");
    for (const auto& instruction : result.circuit.instructions()) {
        require(instruction.condition.has_value(), "conditional routing emitted an unconditional operation");
        require(instruction.condition->bit == 0U && instruction.condition->value, "routing condition changed");
    }
}

void test_basis_translation() {
    const qupy::HardwareTarget cz_target(
        "cz-native",
        2,
        {Code::H},
        {Code::CZ}
    );
    const qupy::CompilationResult cx_result = qupy::compile_circuit(
        qupy::Circuit(2).cx(0, 1), cz_target, {0, 1}, 0
    );
    require(cx_result.compiled_operations == 3U, "CX did not lower to H-CZ-H");
    require(cx_result.decompositions == 1U, "CX decomposition count is wrong");
    require(cx_result.circuit.instructions()[0].code == Code::H, "CX translation prefix is wrong");
    require(cx_result.circuit.instructions()[1].code == Code::CZ, "CX translation entangler is wrong");
    require(cx_result.circuit.instructions()[2].code == Code::H, "CX translation suffix is wrong");

    const qupy::CompilationResult swap_result = qupy::compile_circuit(
        qupy::Circuit(2).swap(0, 1), cz_target, {0, 1}, 0
    );
    require(swap_result.compiled_operations == 9U, "SWAP did not lower through three translated CX operations");
    for (const auto& instruction : swap_result.circuit.instructions()) {
        require(instruction.code == Code::H || instruction.code == Code::CZ, "unsupported operation survived SWAP translation");
    }
}

void test_measurement_reset_and_dynamic_capabilities() {
    const qupy::HardwareTarget terminal(
        "terminal-measure",
        2,
        {Code::H, Code::X},
        {Code::CX},
        {},
        true,
        false,
        false,
        false
    );
    qupy::Circuit terminal_circuit(2, 2);
    terminal_circuit = terminal_circuit.h(0).measure(0, 0).measure(1, 1);
    static_cast<void>(qupy::compile_circuit(terminal_circuit, terminal, {0, 1}, 0));

    const qupy::Circuit mid = qupy::Circuit(1, 1).measure(0, 0).x(0);
    require_invalid(
        [&] { static_cast<void>(qupy::compile_circuit(mid, terminal, {0}, 0)); },
        "mid-circuit measurement was accepted without capability"
    );
    const qupy::HardwareTarget mid_target(
        "mid-measure",
        1,
        {Code::X},
        {},
        {},
        true,
        true,
        false,
        false
    );
    static_cast<void>(qupy::compile_circuit(mid, mid_target, {0}, 0));

    require_invalid(
        [] {
            static_cast<void>(qupy::HardwareTarget(
                "invalid-mid", 1, {Code::X}, {}, {}, false, true
            ));
        },
        "mid-circuit measurement was allowed without measurement"
    );

    const qupy::Circuit reset_circuit = qupy::Circuit(1).reset(0);
    require_invalid(
        [&] { static_cast<void>(qupy::compile_circuit(reset_circuit, mid_target, {0}, 0)); },
        "reset was accepted without reset capability"
    );
    const qupy::HardwareTarget reset_target(
        "reset", 1, {Code::X}, {}, {}, false, false, true
    );
    static_cast<void>(qupy::compile_circuit(reset_circuit, reset_target, {0}, 0));

    const qupy::Circuit conditional = qupy::Circuit(1, 1).x(
        0, qupy::ClassicalCondition(0, true)
    );
    require_invalid(
        [&] { static_cast<void>(qupy::compile_circuit(conditional, reset_target, {0}, 0)); },
        "classical feed-forward was accepted without dynamic-control capability"
    );
}

void test_optimizer_levels() {
    qupy::Circuit adjacent(1);
    adjacent = adjacent.h(0).h(0);
    const qupy::CompilationResult level0 = qupy::compile_circuit(adjacent, full_target(1), {0}, 0);
    const qupy::CompilationResult level1 = qupy::compile_circuit(adjacent, full_target(1), {0}, 1);
    require(level0.compiled_operations == 2U, "optimization level 0 changed the circuit");
    require(level1.optimized_operations == 0U, "level 1 did not cancel adjacent inverse gates");

    qupy::Circuit commuting(2);
    commuting = commuting.h(0).x(1).h(0);
    const qupy::CompilationResult conservative = qupy::compile_circuit(
        commuting, full_target(2), {0, 1}, 1
    );
    const qupy::CompilationResult aggressive = qupy::compile_circuit(
        commuting, full_target(2), {0, 1}, 2
    );
    require(conservative.optimized_operations == 3U, "level 1 commuted gates unexpectedly");
    require(aggressive.optimized_operations == 1U, "level 2 did not expose disjoint cancellation");
    require(aggressive.circuit.instructions()[0].code == Code::X, "level 2 kept the wrong operation");
}

void test_schedule_and_classical_dependency() {
    const qupy::HardwareTarget target(
        "timed",
        2,
        {Code::H, Code::X},
        {Code::CX},
        {},
        true,
        true,
        false,
        true,
        {
            qupy::OperationDuration(Code::H, 10.0),
            qupy::OperationDuration(Code::X, 20.0),
            qupy::OperationDuration(Code::CX, 100.0),
            qupy::OperationDuration(Code::Measure, 50.0),
        }
    );
    qupy::Circuit circuit(2, 1);
    circuit = circuit.h(0).h(1).cx(0, 1).measure(0, 0);
    circuit = circuit.x(1, qupy::ClassicalCondition(0, true));
    const qupy::CompilationResult result = qupy::compile_circuit(circuit, target, {0, 1}, 0);

    require(result.depth == 4U, "classical dependency depth is wrong");
    require(result.duration_ns.has_value(), "complete timing data did not produce a schedule");
    require(std::abs(*result.duration_ns - 180.0) < 1e-12, "scheduled duration is wrong");
    require(result.schedule.size() == 5U, "schedule entry count is wrong");
    require(std::abs(result.schedule[0].start_ns) < 1e-12, "first H did not start at zero");
    require(std::abs(result.schedule[1].start_ns) < 1e-12, "parallel H did not start at zero");
    require(std::abs(result.schedule[2].start_ns - 10.0) < 1e-12, "CX start time is wrong");
    require(std::abs(result.schedule[3].start_ns - 110.0) < 1e-12, "measurement start time is wrong");
    require(std::abs(result.schedule[4].start_ns - 160.0) < 1e-12, "condition did not wait for measurement result");
}

void test_validation_and_fingerprints() {
    const qupy::HardwareTarget first(
        "line", 3, {Code::H}, {Code::CX}, {qupy::Coupling(0, 1), qupy::Coupling(1, 2)}
    );
    const qupy::HardwareTarget same(
        "line", 3, {Code::H}, {Code::CX}, {qupy::Coupling(1, 2), qupy::Coupling(1, 0)}
    );
    const qupy::HardwareTarget different(
        "line", 3, {Code::H}, {Code::CX}, {qupy::Coupling(0, 1)}
    );
    require(first.fingerprint() == same.fingerprint(), "normalized target fingerprint is unstable");
    require(first.fingerprint() != different.fingerprint(), "coupling change did not change fingerprint");

    require_invalid(
        [] { static_cast<void>(qupy::HardwareTarget("bad", 0, {Code::H}, {})); },
        "zero-qubit target was accepted"
    );
    require_invalid(
        [] {
            static_cast<void>(qupy::HardwareTarget(
                "bad-duration",
                1,
                {Code::H},
                {},
                {},
                false,
                false,
                false,
                false,
                {qupy::OperationDuration(Code::RZ, 10.0)}
            ));
        },
        "duration for an unsupported operation was accepted"
    );
    require_invalid(
        [&] {
            static_cast<void>(qupy::compile_circuit(
                qupy::Circuit(2).cx(0, 1), first, {0, 0}, 0
            ));
        },
        "duplicate physical layout was accepted"
    );
    const qupy::HardwareTarget disconnected(
        "disconnected", 3, {Code::H}, {Code::CX}, {qupy::Coupling(0, 1)}
    );
    require_invalid(
        [&] {
            static_cast<void>(qupy::compile_circuit(
                qupy::Circuit(2).cx(0, 1), disconnected, {0, 2}, 0
            ));
        },
        "disconnected interaction was compiled"
    );
    require_invalid(
        [] {
            static_cast<void>(qupy::compile_circuit(
                qupy::Circuit(2), full_target(1), {}, 0
            ));
        },
        "circuit larger than target was compiled"
    );
}

}  // namespace

int main() {
    test_direct_compile_and_identity();
    test_line_routing_and_mapping();
    test_conditional_routing_restores_static_layout();
    test_basis_translation();
    test_measurement_reset_and_dynamic_capabilities();
    test_optimizer_levels();
    test_schedule_and_classical_dependency();
    test_validation_and_fingerprints();
    return 0;
}
