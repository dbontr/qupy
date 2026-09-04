#pragma once

#include "qupy/core.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
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

[[nodiscard]] std::optional<std::size_t> cuda_backend_device(
    const std::string& backend
) noexcept;
[[nodiscard]] std::string cuda_backend_name(std::size_t device);
[[nodiscard]] std::size_t cuda_device_count() noexcept;
[[nodiscard]] bool cuda_available(std::size_t device = 0U) noexcept;
[[nodiscard]] std::string cuda_unavailable_reason(std::size_t device = 0U);
[[nodiscard]] std::string cuda_device_name(std::size_t device = 0U);
[[nodiscard]] int cuda_driver_version(std::size_t device = 0U);
[[nodiscard]] std::size_t cuda_total_memory_bytes(std::size_t device = 0U);
[[nodiscard]] std::vector<Complex> cuda_statevector(
    std::size_t num_qubits,
    const std::vector<CudaStep>& steps,
    std::size_t device = 0U
);
[[nodiscard]] std::vector<Complex> cuda_density_matrix(
    std::size_t num_qubits,
    const std::vector<CudaDensityStep>& steps,
    std::size_t device = 0U
);
[[nodiscard]] std::vector<Complex> cuda_pauli_expectations(
    std::size_t num_qubits,
    const std::vector<CudaStep>& steps,
    const std::vector<CudaPauliMask>& terms,
    std::size_t device = 0U
);
[[nodiscard]] std::vector<Complex> cuda_pauli_expectations(
    const Program& program,
    const std::vector<CudaPauliMask>& terms,
    std::size_t device = 0U
);

}  // namespace qupy::detail
