#pragma once

#include "qupy/core.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
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

enum class CudaDensityStepKind : std::uint8_t { Single, CX, CZ, SWAP, Matrix4 };

struct CudaDensityStep {
    CudaDensityStepKind kind;
    std::array<Complex, 16> matrix;
    std::size_t first;
    std::size_t second;
};

struct CudaPauliMask {
    std::uint64_t flip_mask;
    std::uint64_t sign_mask;
    std::uint32_t y_phase;
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
[[nodiscard]] std::vector<Complex> cuda_density_matrix(
    std::size_t num_qubits,
    const std::vector<CudaDensityStep>& steps
);
[[nodiscard]] std::vector<Complex> cuda_pauli_expectations(
    std::size_t num_qubits,
    const std::vector<CudaStep>& steps,
    const std::vector<CudaPauliMask>& terms
);
[[nodiscard]] std::vector<Complex> cuda_pauli_expectations(
    const Program& program,
    const std::vector<CudaPauliMask>& terms
);

}  // namespace qupy::detail
