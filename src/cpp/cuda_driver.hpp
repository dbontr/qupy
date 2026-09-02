#pragma once

#include "qupy/core.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace qupy::detail {

enum class CudaStepKind : std::uint8_t { Single, CX, CZ, SWAP };

struct CudaStep {
    CudaStepKind kind;
    std::array<Complex, 4> matrix;
    std::size_t first;
    std::size_t second;
};

[[nodiscard]] bool cuda_available() noexcept;
[[nodiscard]] std::string cuda_unavailable_reason();
[[nodiscard]] std::string cuda_device_name();
[[nodiscard]] int cuda_driver_version();
[[nodiscard]] std::size_t cuda_total_memory_bytes();
[[nodiscard]] std::vector<Complex> cuda_statevector(
    std::size_t num_qubits,
    const std::vector<CudaStep>& steps
);

}  // namespace qupy::detail
