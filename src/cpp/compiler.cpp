#include "qupy/compiler.hpp"

#include "qupy/detail/fingerprint.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <locale>
#include <optional>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qupy {
namespace {

constexpr std::uint32_t kHardwareTargetVersion = 1U;

[[nodiscard]] const char* operation_name(CircuitOperationCode code) {
    switch (code) {
    case CircuitOperationCode::H: return "h";
    case CircuitOperationCode::X: return "x";
    case CircuitOperationCode::Y: return "y";
    case CircuitOperationCode::Z: return "z";
    case CircuitOperationCode::RX: return "rx";
    case CircuitOperationCode::RY: return "ry";
    case CircuitOperationCode::RZ: return "rz";
    case CircuitOperationCode::CX: return "cx";
    case CircuitOperationCode::CZ: return "cz";
    case CircuitOperationCode::SWAP: return "swap";
    case CircuitOperationCode::Measure: return "measure";
    case CircuitOperationCode::Reset: return "reset";
    case CircuitOperationCode::Barrier: return "barrier";
    }
    throw std::invalid_argument("unknown circuit operation code");
}

[[nodiscard]] bool is_one_qubit_unitary(CircuitOperationCode code) noexcept {
    switch (code) {
    case CircuitOperationCode::H:
    case CircuitOperationCode::X:
    case CircuitOperationCode::Y:
    case CircuitOperationCode::Z:
    case CircuitOperationCode::RX:
    case CircuitOperationCode::RY:
    case CircuitOperationCode::RZ:
        return true;
    default:
        return false;
    }
}

[[nodiscard]] bool is_two_qubit_unitary(CircuitOperationCode code) noexcept {
    return code == CircuitOperationCode::CX || code == CircuitOperationCode::CZ ||
           code == CircuitOperationCode::SWAP;
}

[[nodiscard]] bool is_unitary(CircuitOperationCode code) noexcept {
    return is_one_qubit_unitary(code) || is_two_qubit_unitary(code);
}

[[nodiscard]] bool is_rotation(CircuitOperationCode code) noexcept {
    return code == CircuitOperationCode::RX || code == CircuitOperationCode::RY ||
           code == CircuitOperationCode::RZ;
}

[[nodiscard]] bool is_self_inverse(CircuitOperationCode code) noexcept {
    switch (code) {
    case CircuitOperationCode::H:
    case CircuitOperationCode::X:
    case CircuitOperationCode::Y:
    case CircuitOperationCode::Z:
    case CircuitOperationCode::CX:
    case CircuitOperationCode::CZ:
    case CircuitOperationCode::SWAP:
        return true;
    default:
        return false;
    }
}

[[nodiscard]] bool code_less(CircuitOperationCode left, CircuitOperationCode right) noexcept {
    return static_cast<std::uint8_t>(left) < static_cast<std::uint8_t>(right);
}

[[nodiscard]] bool contains_code(
    const std::vector<CircuitOperationCode>& values,
    CircuitOperationCode code
) noexcept {
    return std::binary_search(values.begin(), values.end(), code, code_less);
}

void sort_codes(std::vector<CircuitOperationCode>& values) {
    std::sort(values.begin(), values.end(), code_less);
    values.erase(std::unique(values.begin(), values.end()), values.end());
}

[[nodiscard]] bool same_condition(
    const std::optional<ClassicalCondition>& left,
    const std::optional<ClassicalCondition>& right
) noexcept {
    if (left.has_value() != right.has_value()) {
        return false;
    }
    return !left.has_value() ||
           (left->bit == right->bit && left->value == right->value);
}

[[nodiscard]] bool intersects(
    const std::vector<std::size_t>& left,
    const std::vector<std::size_t>& right
) noexcept {
    for (const std::size_t first : left) {
        for (const std::size_t second : right) {
            if (first == second) {
                return true;
            }
        }
    }
    return false;
}

[[nodiscard]] bool can_cancel(
    const CircuitInstruction& left,
    const CircuitInstruction& right
) noexcept {
    return left.code == right.code && is_self_inverse(left.code) &&
           left.qubits == right.qubits && left.parameters.empty() && right.parameters.empty() &&
           left.classical_bits.empty() && right.classical_bits.empty() &&
           same_condition(left.condition, right.condition);
}

[[nodiscard]] bool can_merge_rotation(
    const CircuitInstruction& left,
    const CircuitInstruction& right
) noexcept {
    return left.code == right.code && is_rotation(left.code) && left.qubits == right.qubits &&
           left.parameters.size() == 1U && right.parameters.size() == 1U &&
           left.classical_bits.empty() && right.classical_bits.empty() &&
           same_condition(left.condition, right.condition);
}

[[nodiscard]] bool can_commute_past(
    const CircuitInstruction& moving,
    const CircuitInstruction& intermediate
) noexcept {
    return !moving.condition.has_value() && !intermediate.condition.has_value() &&
           is_unitary(moving.code) && is_unitary(intermediate.code) &&
           !intersects(moving.qubits, intermediate.qubits);
}

[[nodiscard]] Circuit append_instruction(Circuit circuit, const CircuitInstruction& instruction) {
    switch (instruction.code) {
    case CircuitOperationCode::H:
        return circuit.h(instruction.qubits[0], instruction.condition);
    case CircuitOperationCode::X:
        return circuit.x(instruction.qubits[0], instruction.condition);
    case CircuitOperationCode::Y:
        return circuit.y(instruction.qubits[0], instruction.condition);
    case CircuitOperationCode::Z:
        return circuit.z(instruction.qubits[0], instruction.condition);
    case CircuitOperationCode::RX:
        return circuit.rx(instruction.parameters[0], instruction.qubits[0], instruction.condition);
    case CircuitOperationCode::RY:
        return circuit.ry(instruction.parameters[0], instruction.qubits[0], instruction.condition);
    case CircuitOperationCode::RZ:
        return circuit.rz(instruction.parameters[0], instruction.qubits[0], instruction.condition);
    case CircuitOperationCode::CX:
        return circuit.cx(instruction.qubits[0], instruction.qubits[1], instruction.condition);
    case CircuitOperationCode::CZ:
        return circuit.cz(instruction.qubits[0], instruction.qubits[1], instruction.condition);
    case CircuitOperationCode::SWAP:
        return circuit.swap(instruction.qubits[0], instruction.qubits[1], instruction.condition);
    case CircuitOperationCode::Measure:
        return circuit.measure(
            instruction.qubits[0], instruction.classical_bits[0], instruction.condition
        );
    case CircuitOperationCode::Reset:
        return circuit.reset(instruction.qubits[0], instruction.condition);
    case CircuitOperationCode::Barrier:
        return circuit.barrier(instruction.qubits);
    }
    throw std::invalid_argument("unknown circuit operation code");
}

[[nodiscard]] Circuit circuit_from_instructions(
    std::size_t num_qubits,
    std::size_t num_clbits,
    const std::vector<CircuitInstruction>& instructions
) {
    Circuit result(num_qubits, num_clbits);
    for (const CircuitInstruction& instruction : instructions) {
        result = append_instruction(std::move(result), instruction);
    }
    return result;
}

[[nodiscard]] Circuit optimize_logical(const Circuit& circuit, std::uint32_t level) {
    if (level > 2U) {
        throw std::invalid_argument("optimization_level must be 0, 1, or 2");
    }
    if (level == 0U) {
        return circuit;
    }

    std::vector<CircuitInstruction> optimized;
    optimized.reserve(circuit.instructions().size());
    for (const CircuitInstruction& instruction : circuit.instructions()) {
        bool consumed = false;
        if (!optimized.empty()) {
            CircuitInstruction& previous = optimized.back();
            if (can_cancel(previous, instruction)) {
                optimized.pop_back();
                consumed = true;
            } else if (can_merge_rotation(previous, instruction)) {
                previous.parameters[0] += instruction.parameters[0];
                if (previous.parameters[0] == 0.0) {
                    optimized.pop_back();
                }
                consumed = true;
            }
        }
        if (consumed) {
            continue;
        }

        if (level >= 2U && !instruction.condition.has_value() && is_unitary(instruction.code)) {
            for (std::size_t offset = optimized.size(); offset > 0U; --offset) {
                const std::size_t candidate_index = offset - 1U;
                CircuitInstruction& candidate = optimized[candidate_index];
                if (can_cancel(candidate, instruction)) {
                    optimized.erase(optimized.begin() + static_cast<std::ptrdiff_t>(candidate_index));
                    consumed = true;
                    break;
                }
                if (can_merge_rotation(candidate, instruction)) {
                    candidate.parameters[0] += instruction.parameters[0];
                    if (candidate.parameters[0] == 0.0) {
                        optimized.erase(
                            optimized.begin() + static_cast<std::ptrdiff_t>(candidate_index)
                        );
                    }
                    consumed = true;
                    break;
                }
                if (!can_commute_past(instruction, candidate)) {
                    break;
                }
            }
        }
        if (!consumed) {
            optimized.push_back(instruction);
        }
    }
    return circuit_from_instructions(circuit.num_qubits(), circuit.num_clbits(), optimized);
}

[[nodiscard]] bool requires_mid_circuit_measurement(const Circuit& circuit) noexcept {
    const auto& instructions = circuit.instructions();
    for (std::size_t index = 0; index < instructions.size(); ++index) {
        if (instructions[index].code != CircuitOperationCode::Measure) {
            continue;
        }
        if (instructions[index].condition.has_value()) {
            return true;
        }
        for (std::size_t later = index + 1U; later < instructions.size(); ++later) {
            const CircuitInstruction& instruction = instructions[later];
            if (instruction.code == CircuitOperationCode::Barrier) {
                continue;
            }
            if (instruction.code == CircuitOperationCode::Measure &&
                !instruction.condition.has_value()) {
                continue;
            }
            return true;
        }
    }
    return false;
}

[[nodiscard]] std::vector<std::size_t> default_layout(
    const Circuit& circuit,
    const HardwareTarget& target
) {
    std::vector<std::size_t> logical_degree(circuit.num_qubits(), 0U);
    for (const CircuitInstruction& instruction : circuit.instructions()) {
        if (is_two_qubit_unitary(instruction.code)) {
            ++logical_degree[instruction.qubits[0]];
            ++logical_degree[instruction.qubits[1]];
        }
    }

    std::vector<std::size_t> physical_degree(target.num_qubits(), 0U);
    if (target.couplings().empty()) {
        const std::size_t degree = target.num_qubits() - 1U;
        std::fill(physical_degree.begin(), physical_degree.end(), degree);
    } else {
        for (const Coupling& coupling : target.couplings()) {
            ++physical_degree[coupling.first];
            ++physical_degree[coupling.second];
        }
    }

    std::vector<std::size_t> logical_order(circuit.num_qubits());
    for (std::size_t index = 0; index < logical_order.size(); ++index) {
        logical_order[index] = index;
    }
    std::sort(logical_order.begin(), logical_order.end(), [&](std::size_t left, std::size_t right) {
        if (logical_degree[left] != logical_degree[right]) {
            return logical_degree[left] > logical_degree[right];
        }
        return left < right;
    });

    std::vector<std::size_t> physical_order(target.num_qubits());
    for (std::size_t index = 0; index < physical_order.size(); ++index) {
        physical_order[index] = index;
    }
    std::sort(
        physical_order.begin(),
        physical_order.end(),
        [&](std::size_t left, std::size_t right) {
            if (physical_degree[left] != physical_degree[right]) {
                return physical_degree[left] > physical_degree[right];
            }
            return left < right;
        }
    );

    std::vector<std::size_t> layout(circuit.num_qubits());
    for (std::size_t index = 0; index < logical_order.size(); ++index) {
        layout[logical_order[index]] = physical_order[index];
    }
    return layout;
}

void validate_layout(
    const Circuit& circuit,
    const HardwareTarget& target,
    const std::vector<std::size_t>& layout
) {
    if (layout.size() != circuit.num_qubits()) {
        throw std::invalid_argument("initial_layout must contain one physical qubit per circuit qubit");
    }
    std::vector<bool> used(target.num_qubits(), false);
    for (const std::size_t physical : layout) {
        if (physical >= target.num_qubits()) {
            throw std::invalid_argument("initial_layout contains a qubit outside the hardware target");
        }
        if (used[physical]) {
            throw std::invalid_argument("initial_layout physical qubits must be unique");
        }
        used[physical] = true;
    }
}

[[nodiscard]] std::vector<std::size_t> shortest_path(
    const HardwareTarget& target,
    std::size_t start,
    std::size_t goal
) {
    if (start == goal) {
        return {start};
    }
    if (target.couplings().empty()) {
        return {start, goal};
    }

    const std::size_t missing = target.num_qubits();
    std::vector<std::size_t> parent(target.num_qubits(), missing);
    std::queue<std::size_t> pending;
    parent[start] = start;
    pending.push(start);
    while (!pending.empty() && parent[goal] == missing) {
        const std::size_t current = pending.front();
        pending.pop();
        for (const Coupling& coupling : target.couplings()) {
            std::optional<std::size_t> neighbor;
            if (coupling.first == current) {
                neighbor = coupling.second;
            } else if (coupling.second == current) {
                neighbor = coupling.first;
            }
            if (neighbor.has_value() && parent[*neighbor] == missing) {
                parent[*neighbor] = current;
                pending.push(*neighbor);
            }
        }
    }
    if (parent[goal] == missing) {
        return {};
    }

    std::vector<std::size_t> path;
    for (std::size_t current = goal; current != start; current = parent[current]) {
        path.push_back(current);
    }
    path.push_back(start);
    std::reverse(path.begin(), path.end());
    return path;
}

struct RoutingResult {
    Circuit circuit;
    std::vector<std::size_t> final_layout;
    std::size_t inserted_swaps;
};

[[nodiscard]] RoutingResult route_circuit(
    const Circuit& source,
    const HardwareTarget& target,
    const std::vector<std::size_t>& initial_layout
) {
    Circuit routed(target.num_qubits(), source.num_clbits());
    std::vector<std::size_t> layout = initial_layout;
    std::vector<std::optional<std::size_t>> occupant(target.num_qubits());
    for (std::size_t logical = 0; logical < layout.size(); ++logical) {
        occupant[layout[logical]] = logical;
    }
    std::size_t inserted_swaps = 0U;

    const auto append_routing_swap = [](
        Circuit current,
        std::size_t first,
        std::size_t second,
        std::optional<ClassicalCondition> condition
    ) { return current.swap(first, second, condition); };

    for (const CircuitInstruction& instruction : source.instructions()) {
        if (instruction.code == CircuitOperationCode::Barrier) {
            std::vector<std::size_t> mapped;
            mapped.reserve(instruction.qubits.size());
            for (const std::size_t logical : instruction.qubits) {
                mapped.push_back(layout[logical]);
            }
            routed = routed.barrier(mapped);
            continue;
        }

        if (!is_two_qubit_unitary(instruction.code)) {
            CircuitInstruction mapped = instruction;
            for (std::size_t& logical : mapped.qubits) {
                logical = layout[logical];
            }
            routed = append_instruction(std::move(routed), mapped);
            continue;
        }

        const std::size_t logical_first = instruction.qubits[0];
        const std::size_t logical_second = instruction.qubits[1];
        const std::size_t first = layout[logical_first];
        const std::size_t second = layout[logical_second];
        if (target.adjacent(first, second)) {
            CircuitInstruction mapped = instruction;
            mapped.qubits = {first, second};
            routed = append_instruction(std::move(routed), mapped);
            continue;
        }

        const std::vector<std::size_t> path = shortest_path(target, first, second);
        if (path.size() < 2U) {
            throw std::invalid_argument("hardware coupling graph cannot connect a required interaction");
        }

        if (instruction.condition.has_value()) {
            for (std::size_t index = 0; index + 2U < path.size(); ++index) {
                routed = append_routing_swap(
                    std::move(routed), path[index], path[index + 1U], instruction.condition
                );
                ++inserted_swaps;
            }
            CircuitInstruction mapped = instruction;
            mapped.qubits = {path[path.size() - 2U], second};
            routed = append_instruction(std::move(routed), mapped);
            for (std::size_t index = path.size() - 2U; index > 0U; --index) {
                routed = append_routing_swap(
                    std::move(routed), path[index - 1U], path[index], instruction.condition
                );
                ++inserted_swaps;
            }
            continue;
        }

        for (std::size_t index = 0; index + 2U < path.size(); ++index) {
            const std::size_t swap_first = path[index];
            const std::size_t swap_second = path[index + 1U];
            routed = append_routing_swap(std::move(routed), swap_first, swap_second, std::nullopt);
            ++inserted_swaps;
            std::swap(occupant[swap_first], occupant[swap_second]);
            if (occupant[swap_first].has_value()) {
                layout[*occupant[swap_first]] = swap_first;
            }
            if (occupant[swap_second].has_value()) {
                layout[*occupant[swap_second]] = swap_second;
            }
        }
        CircuitInstruction mapped = instruction;
        mapped.qubits = {layout[logical_first], layout[logical_second]};
        routed = append_instruction(std::move(routed), mapped);
    }

    return {std::move(routed), std::move(layout), inserted_swaps};
}

[[nodiscard]] bool can_emit_cx(
    const HardwareTarget& target,
    std::size_t control,
    std::size_t target_qubit
) noexcept {
    return target.supports(CircuitOperationCode::CX, {control, target_qubit}) ||
           (target.supports(CircuitOperationCode::H, {target_qubit}) &&
            target.supports(CircuitOperationCode::CZ, {control, target_qubit}));
}

[[nodiscard]] Circuit emit_cx(
    Circuit circuit,
    const HardwareTarget& target,
    std::size_t control,
    std::size_t target_qubit,
    std::optional<ClassicalCondition> condition,
    std::size_t& decompositions
) {
    if (target.supports(CircuitOperationCode::CX, {control, target_qubit})) {
        return circuit.cx(control, target_qubit, condition);
    }
    if (target.supports(CircuitOperationCode::H, {target_qubit}) &&
        target.supports(CircuitOperationCode::CZ, {control, target_qubit})) {
        ++decompositions;
        circuit = circuit.h(target_qubit, condition);
        circuit = circuit.cz(control, target_qubit, condition);
        return circuit.h(target_qubit, condition);
    }
    throw std::invalid_argument("hardware target cannot implement CX on a required coupling");
}

[[nodiscard]] Circuit translate_instruction(
    Circuit circuit,
    const HardwareTarget& target,
    const CircuitInstruction& instruction,
    std::size_t& decompositions
) {
    if (instruction.condition.has_value() && !target.dynamic_control()) {
        throw std::invalid_argument("hardware target does not support classical feed-forward control");
    }
    if (instruction.code == CircuitOperationCode::Barrier) {
        return circuit.barrier(instruction.qubits);
    }
    if (target.supports(instruction.code, instruction.qubits)) {
        return append_instruction(std::move(circuit), instruction);
    }

    if (instruction.code == CircuitOperationCode::CX) {
        const std::size_t control = instruction.qubits[0];
        const std::size_t target_qubit = instruction.qubits[1];
        if (target.supports(CircuitOperationCode::H, {target_qubit}) &&
            target.supports(CircuitOperationCode::CZ, {control, target_qubit})) {
            ++decompositions;
            circuit = circuit.h(target_qubit, instruction.condition);
            circuit = circuit.cz(control, target_qubit, instruction.condition);
            return circuit.h(target_qubit, instruction.condition);
        }
    } else if (instruction.code == CircuitOperationCode::CZ) {
        const std::size_t control = instruction.qubits[0];
        const std::size_t target_qubit = instruction.qubits[1];
        if (target.supports(CircuitOperationCode::H, {target_qubit}) &&
            target.supports(CircuitOperationCode::CX, {control, target_qubit})) {
            ++decompositions;
            circuit = circuit.h(target_qubit, instruction.condition);
            circuit = circuit.cx(control, target_qubit, instruction.condition);
            return circuit.h(target_qubit, instruction.condition);
        }
    } else if (instruction.code == CircuitOperationCode::SWAP) {
        const std::size_t first = instruction.qubits[0];
        const std::size_t second = instruction.qubits[1];
        if (can_emit_cx(target, first, second) && can_emit_cx(target, second, first)) {
            ++decompositions;
            circuit = emit_cx(
                std::move(circuit), target, first, second, instruction.condition, decompositions
            );
            circuit = emit_cx(
                std::move(circuit), target, second, first, instruction.condition, decompositions
            );
            return emit_cx(
                std::move(circuit), target, first, second, instruction.condition, decompositions
            );
        }
    }

    if (instruction.code == CircuitOperationCode::Measure) {
        throw std::invalid_argument("hardware target does not support measurement");
    }
    if (instruction.code == CircuitOperationCode::Reset) {
        throw std::invalid_argument("hardware target does not support reset");
    }
    throw std::invalid_argument(
        std::string("hardware target cannot implement operation ") + operation_name(instruction.code)
    );
}

struct TranslationResult {
    Circuit circuit;
    std::size_t decompositions;
};

[[nodiscard]] TranslationResult translate_circuit(
    const Circuit& routed,
    const HardwareTarget& target
) {
    Circuit translated(target.num_qubits(), routed.num_clbits());
    std::size_t decompositions = 0U;
    for (const CircuitInstruction& instruction : routed.instructions()) {
        translated = translate_instruction(
            std::move(translated), target, instruction, decompositions
        );
    }
    return {std::move(translated), decompositions};
}

[[nodiscard]] std::size_t circuit_depth(const Circuit& circuit) {
    std::vector<std::size_t> qubit_ready(circuit.num_qubits(), 0U);
    std::vector<std::size_t> classical_ready(circuit.num_clbits(), 0U);
    std::size_t depth = 0U;

    for (const CircuitInstruction& instruction : circuit.instructions()) {
        if (instruction.code == CircuitOperationCode::Barrier) {
            std::size_t ready = 0U;
            for (const std::size_t qubit : instruction.qubits) {
                ready = std::max(ready, qubit_ready[qubit]);
            }
            for (const std::size_t qubit : instruction.qubits) {
                qubit_ready[qubit] = ready;
            }
            continue;
        }

        std::size_t ready = 0U;
        for (const std::size_t qubit : instruction.qubits) {
            ready = std::max(ready, qubit_ready[qubit]);
        }
        if (instruction.condition.has_value()) {
            ready = std::max(ready, classical_ready[instruction.condition->bit]);
        }
        const std::size_t end = ready + 1U;
        for (const std::size_t qubit : instruction.qubits) {
            qubit_ready[qubit] = end;
        }
        if (instruction.code == CircuitOperationCode::Measure) {
            classical_ready[instruction.classical_bits[0]] = end;
        }
        depth = std::max(depth, end);
    }
    return depth;
}

struct ScheduleResult {
    std::optional<double> duration_ns;
    std::vector<ScheduledInstruction> schedule;
};

[[nodiscard]] ScheduleResult schedule_circuit(
    const Circuit& circuit,
    const HardwareTarget& target
) {
    std::vector<double> qubit_ready(circuit.num_qubits(), 0.0);
    std::vector<double> classical_ready(circuit.num_clbits(), 0.0);
    std::vector<ScheduledInstruction> schedule;
    schedule.reserve(circuit.instructions().size());
    double total = 0.0;

    for (std::size_t index = 0; index < circuit.instructions().size(); ++index) {
        const CircuitInstruction& instruction = circuit.instructions()[index];
        if (instruction.code == CircuitOperationCode::Barrier) {
            double ready = 0.0;
            for (const std::size_t qubit : instruction.qubits) {
                ready = std::max(ready, qubit_ready[qubit]);
            }
            for (const std::size_t qubit : instruction.qubits) {
                qubit_ready[qubit] = ready;
            }
            total = std::max(total, ready);
            continue;
        }

        const std::optional<double> duration = target.duration_ns(instruction.code);
        if (!duration.has_value()) {
            return {std::nullopt, {}};
        }
        double start = 0.0;
        for (const std::size_t qubit : instruction.qubits) {
            start = std::max(start, qubit_ready[qubit]);
        }
        if (instruction.condition.has_value()) {
            start = std::max(start, classical_ready[instruction.condition->bit]);
        }
        const double end = start + *duration;
        for (const std::size_t qubit : instruction.qubits) {
            qubit_ready[qubit] = end;
        }
        if (instruction.code == CircuitOperationCode::Measure) {
            classical_ready[instruction.classical_bits[0]] = end;
        }
        schedule.push_back({index, start, *duration});
        total = std::max(total, end);
    }
    return {total, std::move(schedule)};
}

[[nodiscard]] bool duration_code_supported(
    CircuitOperationCode code,
    const std::vector<CircuitOperationCode>& one_qubit_operations,
    const std::vector<CircuitOperationCode>& two_qubit_operations,
    bool measurement,
    bool reset
) noexcept {
    if (is_one_qubit_unitary(code)) {
        return contains_code(one_qubit_operations, code);
    }
    if (is_two_qubit_unitary(code)) {
        return contains_code(two_qubit_operations, code);
    }
    if (code == CircuitOperationCode::Measure) {
        return measurement;
    }
    if (code == CircuitOperationCode::Reset) {
        return reset;
    }
    return false;
}

}  // namespace

HardwareTarget::HardwareTarget(
    std::string name,
    std::size_t num_qubits,
    std::vector<CircuitOperationCode> one_qubit_operations,
    std::vector<CircuitOperationCode> two_qubit_operations,
    std::vector<Coupling> couplings,
    bool measurement,
    bool mid_circuit_measurement,
    bool reset,
    bool dynamic_control,
    std::vector<OperationDuration> durations
)
    : name_(std::move(name)),
      num_qubits_(num_qubits),
      one_qubit_operations_(std::move(one_qubit_operations)),
      two_qubit_operations_(std::move(two_qubit_operations)),
      couplings_(std::move(couplings)),
      measurement_(measurement),
      mid_circuit_measurement_(mid_circuit_measurement),
      reset_(reset),
      dynamic_control_(dynamic_control),
      durations_(std::move(durations)) {
    if (name_.empty() || name_.find_first_of("\r\n") != std::string::npos) {
        throw std::invalid_argument("hardware target name must be non-empty and single-line");
    }
    if (num_qubits_ == 0U) {
        throw std::invalid_argument("hardware target must contain at least one qubit");
    }
    if (mid_circuit_measurement_ && !measurement_) {
        throw std::invalid_argument("mid-circuit measurement requires measurement support");
    }
    for (const CircuitOperationCode code : one_qubit_operations_) {
        if (!is_one_qubit_unitary(code)) {
            throw std::invalid_argument(
                "one_qubit_operations contains a non-unitary or wrong-arity operation"
            );
        }
    }
    for (const CircuitOperationCode code : two_qubit_operations_) {
        if (!is_two_qubit_unitary(code)) {
            throw std::invalid_argument(
                "two_qubit_operations contains a non-unitary or wrong-arity operation"
            );
        }
    }
    sort_codes(one_qubit_operations_);
    sort_codes(two_qubit_operations_);

    for (Coupling& coupling : couplings_) {
        if (coupling.first >= num_qubits_ || coupling.second >= num_qubits_) {
            throw std::invalid_argument("coupling contains a qubit outside the hardware target");
        }
        if (coupling.first == coupling.second) {
            throw std::invalid_argument("coupling endpoints must be distinct");
        }
        if (coupling.second < coupling.first) {
            std::swap(coupling.first, coupling.second);
        }
    }
    std::sort(couplings_.begin(), couplings_.end(), [](const Coupling& left, const Coupling& right) {
        return std::pair{left.first, left.second} < std::pair{right.first, right.second};
    });
    couplings_.erase(
        std::unique(
            couplings_.begin(),
            couplings_.end(),
            [](const Coupling& left, const Coupling& right) {
                return left.first == right.first && left.second == right.second;
            }
        ),
        couplings_.end()
    );

    std::sort(
        durations_.begin(),
        durations_.end(),
        [](const OperationDuration& left, const OperationDuration& right) {
            return code_less(left.code, right.code);
        }
    );
    for (std::size_t index = 0; index < durations_.size(); ++index) {
        const OperationDuration& duration = durations_[index];
        if (!duration_code_supported(
                duration.code,
                one_qubit_operations_,
                two_qubit_operations_,
                measurement_,
                reset_
            )) {
            throw std::invalid_argument(
                "operation duration references an operation not supported by the hardware target"
            );
        }
        if (!std::isfinite(duration.nanoseconds) || duration.nanoseconds <= 0.0) {
            throw std::invalid_argument("operation durations must be finite and positive");
        }
        if (index != 0U && durations_[index - 1U].code == duration.code) {
            throw std::invalid_argument(
                "operation durations must contain at most one value per operation"
            );
        }
    }
}

const std::string& HardwareTarget::name() const noexcept { return name_; }
std::size_t HardwareTarget::num_qubits() const noexcept { return num_qubits_; }
const std::vector<CircuitOperationCode>& HardwareTarget::one_qubit_operations() const noexcept {
    return one_qubit_operations_;
}
const std::vector<CircuitOperationCode>& HardwareTarget::two_qubit_operations() const noexcept {
    return two_qubit_operations_;
}
const std::vector<Coupling>& HardwareTarget::couplings() const noexcept { return couplings_; }
bool HardwareTarget::measurement() const noexcept { return measurement_; }
bool HardwareTarget::mid_circuit_measurement() const noexcept { return mid_circuit_measurement_; }
bool HardwareTarget::reset() const noexcept { return reset_; }
bool HardwareTarget::dynamic_control() const noexcept { return dynamic_control_; }
const std::vector<OperationDuration>& HardwareTarget::durations() const noexcept { return durations_; }

bool HardwareTarget::adjacent(std::size_t first, std::size_t second) const noexcept {
    if (first >= num_qubits_ || second >= num_qubits_ || first == second) {
        return false;
    }
    if (couplings_.empty()) {
        return true;
    }
    if (second < first) {
        std::swap(first, second);
    }
    return std::any_of(couplings_.begin(), couplings_.end(), [&](const Coupling& coupling) {
        return coupling.first == first && coupling.second == second;
    });
}

bool HardwareTarget::supports(
    CircuitOperationCode code,
    const std::vector<std::size_t>& qubits
) const noexcept {
    for (const std::size_t qubit : qubits) {
        if (qubit >= num_qubits_) {
            return false;
        }
    }
    if (code == CircuitOperationCode::Barrier) {
        return true;
    }
    if (code == CircuitOperationCode::Measure) {
        return measurement_ && qubits.size() == 1U;
    }
    if (code == CircuitOperationCode::Reset) {
        return reset_ && qubits.size() == 1U;
    }
    if (is_one_qubit_unitary(code)) {
        return qubits.size() == 1U && contains_code(one_qubit_operations_, code);
    }
    if (is_two_qubit_unitary(code)) {
        return qubits.size() == 2U && contains_code(two_qubit_operations_, code) &&
               adjacent(qubits[0], qubits[1]);
    }
    return false;
}

std::optional<double> HardwareTarget::duration_ns(CircuitOperationCode code) const noexcept {
    const auto found = std::lower_bound(
        durations_.begin(),
        durations_.end(),
        code,
        [](const OperationDuration& duration, CircuitOperationCode value) {
            return code_less(duration.code, value);
        }
    );
    if (found == durations_.end() || found->code != code) {
        return std::nullopt;
    }
    return found->nanoseconds;
}

std::string HardwareTarget::canonical_text() const {
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << "qupy-hardware-target " << kHardwareTargetVersion << '\n';
    output << "name " << name_ << '\n';
    output << "qubits " << num_qubits_ << '\n';
    output << "one";
    for (const CircuitOperationCode code : one_qubit_operations_) {
        output << ' ' << operation_name(code);
    }
    output << '\n';
    output << "two";
    for (const CircuitOperationCode code : two_qubit_operations_) {
        output << ' ' << operation_name(code);
    }
    output << '\n';
    output << "couplings";
    if (couplings_.empty()) {
        output << " all-to-all";
    }
    output << '\n';
    for (const Coupling& coupling : couplings_) {
        output << "edge " << coupling.first << ' ' << coupling.second << '\n';
    }
    output << "measurement " << (measurement_ ? 1 : 0) << '\n';
    output << "mid-circuit-measurement " << (mid_circuit_measurement_ ? 1 : 0) << '\n';
    output << "reset " << (reset_ ? 1 : 0) << '\n';
    output << "dynamic " << (dynamic_control_ ? 1 : 0) << '\n';
    for (const OperationDuration& duration : durations_) {
        const auto bits = std::bit_cast<std::uint64_t>(duration.nanoseconds);
        output << "duration " << operation_name(duration.code) << ' ' << std::hex
               << std::setfill('0') << std::setw(16) << bits << std::dec << '\n';
    }
    return output.str();
}

std::string HardwareTarget::fingerprint() const {
    return detail::fingerprint_text(canonical_text());
}

CompilationResult compile_circuit(
    const Circuit& circuit,
    const HardwareTarget& target,
    const std::vector<std::size_t>& requested_layout,
    std::uint32_t optimization_level
) {
    if (circuit.num_qubits() > target.num_qubits()) {
        throw std::invalid_argument("circuit requires more qubits than the hardware target provides");
    }
    if (requires_mid_circuit_measurement(circuit) && !target.mid_circuit_measurement()) {
        throw std::invalid_argument("hardware target does not support mid-circuit measurement");
    }

    const Circuit optimized = optimize_logical(circuit, optimization_level);
    std::vector<std::size_t> layout = requested_layout.empty()
        ? default_layout(optimized, target)
        : requested_layout;
    validate_layout(optimized, target, layout);
    const std::vector<std::size_t> initial_layout = layout;

    RoutingResult routed = route_circuit(optimized, target, layout);
    const std::size_t routed_operations = routed.circuit.instructions().size();
    TranslationResult translated = translate_circuit(routed.circuit, target);
    const std::size_t compiled_operations = translated.circuit.instructions().size();
    const std::size_t depth = circuit_depth(translated.circuit);
    ScheduleResult schedule = schedule_circuit(translated.circuit, target);
    const std::string target_fingerprint = target.fingerprint();
    Circuit compiled = std::move(translated.circuit);

    return {
        std::move(compiled),
        initial_layout,
        std::move(routed.final_layout),
        circuit.instructions().size(),
        optimized.instructions().size(),
        routed_operations,
        compiled_operations,
        routed.inserted_swaps,
        translated.decompositions,
        depth,
        schedule.duration_ns,
        std::move(schedule.schedule),
        target_fingerprint,
    };
}

}  // namespace qupy
