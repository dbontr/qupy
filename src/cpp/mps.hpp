#pragma once

#include "qupy/core.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace qupy::detail {

enum class MpsStepKind {
    Single,
    CX,
    CZ,
    SWAP,
};

struct MpsStep {
    MpsStepKind kind;
    std::array<Complex, 4> matrix;
    std::size_t first;
    std::size_t second;
};

struct MpsEstimate {
    std::size_t state_bytes;
    std::size_t max_bond;
    std::size_t routed_swaps;
    double contraction_work;
};

struct MpsStateResult {
    std::vector<Complex> values;
    std::size_t state_bytes;
    std::size_t max_bond;
    double discarded_weight;
};

struct MpsExpectationResult {
    double value;
    std::size_t state_bytes;
    std::size_t max_bond;
    double discarded_weight;
};

[[nodiscard]] MpsEstimate mps_estimate(
    std::size_t num_qubits,
    const std::vector<MpsStep>& steps
);

[[nodiscard]] MpsStateResult mps_statevector(
    std::size_t num_qubits,
    const std::vector<MpsStep>& steps
);

[[nodiscard]] MpsExpectationResult mps_pauli_z_expectation(
    std::size_t num_qubits,
    const std::vector<MpsStep>& steps,
    std::size_t observable_qubit
);

}  // namespace qupy::detail
