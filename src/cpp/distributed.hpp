#pragma once

#include "qupy/advanced.hpp"

#include <vector>

namespace qupy::detail {

[[nodiscard]] std::vector<Complex> distributed_pauli_expectations(
    const Program& program,
    const std::vector<std::vector<PauliFactor>>& terms
);

}  // namespace qupy::detail
