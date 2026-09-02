#include "qupy/advanced.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef QUPY_HAS_MPI
#include <mpi.h>
#endif

namespace qupy {
namespace {

#ifdef QUPY_HAS_MPI

using Matrix2 = std::array<Complex, 4>;

[[nodiscard]] Matrix2 operation_matrix(const Operation& operation) {
    const double inv_sqrt_two = 1.0 / std::sqrt(2.0);
    switch (operation.code) {
    case OperationCode::H:
        return {inv_sqrt_two, inv_sqrt_two, inv_sqrt_two, -inv_sqrt_two};
    case OperationCode::X:
        return {0.0, 1.0, 1.0, 0.0};
    case OperationCode::Y:
        return {0.0, Complex{0.0, -1.0}, Complex{0.0, 1.0}, 0.0};
    case OperationCode::Z:
        return {1.0, 0.0, 0.0, -1.0};
    case OperationCode::RX: {
        const double half = operation.parameters.at(0) / 2.0;
        const Complex sine{0.0, -std::sin(half)};
        return {std::cos(half), sine, sine, std::cos(half)};
    }
    case OperationCode::RY: {
        const double half = operation.parameters.at(0) / 2.0;
        return {std::cos(half), -std::sin(half), std::sin(half), std::cos(half)};
    }
    case OperationCode::RZ: {
        const double half = operation.parameters.at(0) / 2.0;
        return {std::polar(1.0, -half), 0.0, 0.0, std::polar(1.0, half)};
    }
    case OperationCode::CX:
    case OperationCode::CZ:
    case OperationCode::SWAP:
        break;
    }
    throw std::invalid_argument("operation is not a single-qubit gate");
}

[[nodiscard]] bool is_power_of_two(std::size_t value) noexcept {
    return value != 0U && (value & (value - 1U)) == 0U;
}

[[nodiscard]] std::size_t integer_log2(std::size_t value) {
    std::size_t result = 0U;
    while (value > 1U) {
        value >>= 1U;
        ++result;
    }
    return result;
}
void apply_local_single(
    std::vector<Complex>& state,
    const Matrix2& matrix,
    std::size_t qubit
) {
    const std::size_t bit = std::size_t{1} << qubit;
    const std::size_t span = bit << 1U;
    for (std::size_t base = 0; base < state.size(); base += span) {
        for (std::size_t offset = 0; offset < bit; ++offset) {
            const std::size_t zero = base + offset;
            const std::size_t one = zero + bit;
            const Complex a = state[zero];
            const Complex b = state[one];
            state[zero] = matrix[0] * a + matrix[1] * b;
            state[one] = matrix[2] * a + matrix[3] * b;
        }
    }
}

void apply_local_x(std::vector<Complex>& state, std::size_t qubit) {
    const std::size_t bit = std::size_t{1} << qubit;
    for (std::size_t index = 0; index < state.size(); ++index) {
        if ((index & bit) == 0U) {
            std::swap(state[index], state[index | bit]);
        }
    }
}

void apply_local_swap(
    std::vector<Complex>& state,
    std::size_t first,
    std::size_t second
) {
    if (first == second) {
        return;
    }
    const std::size_t first_bit = std::size_t{1} << first;
    const std::size_t second_bit = std::size_t{1} << second;
    for (std::size_t index = 0; index < state.size(); ++index) {
        const bool first_value = (index & first_bit) != 0U;
        const bool second_value = (index & second_bit) != 0U;
        if (first_value && !second_value) {
            std::swap(state[index], state[index ^ first_bit ^ second_bit]);
        }
    }
}
void apply_local_cx(
    std::vector<Complex>& state,
    std::size_t control,
    std::size_t target
) {
    const std::size_t control_bit = std::size_t{1} << control;
    const std::size_t target_bit = std::size_t{1} << target;
    for (std::size_t index = 0; index < state.size(); ++index) {
        if ((index & control_bit) != 0U && (index & target_bit) == 0U) {
            std::swap(state[index], state[index | target_bit]);
        }
    }
}

[[nodiscard]] bool qubit_value(
    std::size_t qubit,
    std::size_t local_qubits,
    std::size_t rank,
    std::size_t local_index
) {
    if (qubit < local_qubits) {
        return (local_index & (std::size_t{1} << qubit)) != 0U;
    }
    return (rank & (std::size_t{1} << (qubit - local_qubits))) != 0U;
}

void apply_distributed_cz(
    std::vector<Complex>& state,
    std::size_t control,
    std::size_t target,
    std::size_t local_qubits,
    std::size_t rank
) {
    for (std::size_t index = 0; index < state.size(); ++index) {
        if (qubit_value(control, local_qubits, rank, index) &&
            qubit_value(target, local_qubits, rank, index)) {
            state[index] = -state[index];
        }
    }
}
void mpi_check(int status, const char* operation) {
    if (status == MPI_SUCCESS) {
        return;
    }
    std::array<char, MPI_MAX_ERROR_STRING> buffer{};
    int length = 0;
    MPI_Error_string(status, buffer.data(), &length);
    throw std::runtime_error(
        std::string(operation) + " failed: " +
        std::string(buffer.data(), static_cast<std::size_t>(length))
    );
}

[[nodiscard]] std::vector<Complex> exchange_shard(
    const std::vector<Complex>& state,
    int partner
) {
    std::vector<Complex> peer(state.size());
    std::size_t offset = 0U;
    while (offset < state.size()) {
        const std::size_t remaining = state.size() - offset;
        const int count = static_cast<int>(std::min<std::size_t>(
            remaining,
            static_cast<std::size_t>(std::numeric_limits<int>::max())
        ));
        mpi_check(
            MPI_Sendrecv(
                state.data() + offset,
                count,
                MPI_CXX_DOUBLE_COMPLEX,
                partner,
                701,
                peer.data() + offset,
                count,
                MPI_CXX_DOUBLE_COMPLEX,
                partner,
                701,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE
            ),
            "MPI_Sendrecv"
        );
        offset += static_cast<std::size_t>(count);
    }
    return peer;
}
void apply_cross_rank_single(
    std::vector<Complex>& state,
    const Matrix2& matrix,
    std::size_t qubit,
    std::size_t local_qubits,
    std::size_t rank
) {
    const std::size_t rank_bit = std::size_t{1} << (qubit - local_qubits);
    const int partner = static_cast<int>(rank ^ rank_bit);
    const std::vector<Complex> peer = exchange_shard(state, partner);
    const bool one = (rank & rank_bit) != 0U;
    for (std::size_t index = 0; index < state.size(); ++index) {
        const Complex own = state[index];
        const Complex other = peer[index];
        state[index] = one
            ? matrix[2] * other + matrix[3] * own
            : matrix[0] * own + matrix[1] * other;
    }
}

void apply_distributed_cx(
    std::vector<Complex>& state,
    std::size_t control,
    std::size_t target,
    std::size_t local_qubits,
    std::size_t rank
) {
    const bool control_local = control < local_qubits;
    const bool target_local = target < local_qubits;
    if (control_local && target_local) {
        apply_local_cx(state, control, target);
        return;
    }
    if (!control_local && target_local) {
        if (qubit_value(control, local_qubits, rank, 0U)) {
            apply_local_x(state, target);
        }
        return;
    }
    const std::size_t target_rank_bit = std::size_t{1} << (target - local_qubits);
    const int partner = static_cast<int>(rank ^ target_rank_bit);
    if (!control_local) {
        if (!qubit_value(control, local_qubits, rank, 0U)) {
            return;
        }
        state = exchange_shard(state, partner);
        return;
    }
    const std::vector<Complex> peer = exchange_shard(state, partner);
    const std::size_t control_bit = std::size_t{1} << control;
    for (std::size_t index = 0; index < state.size(); ++index) {
        if ((index & control_bit) != 0U) {
            state[index] = peer[index];
        }
    }
}

void apply_distributed_swap(
    std::vector<Complex>& state,
    std::size_t first,
    std::size_t second,
    std::size_t local_qubits,
    std::size_t rank
) {
    const bool first_local = first < local_qubits;
    const bool second_local = second < local_qubits;
    if (first_local && second_local) {
        apply_local_swap(state, first, second);
        return;
    }
    if (!first_local && !second_local) {
        const std::size_t first_rank_bit = std::size_t{1} << (first - local_qubits);
        const std::size_t second_rank_bit = std::size_t{1} << (second - local_qubits);
        const bool first_value = (rank & first_rank_bit) != 0U;
        const bool second_value = (rank & second_rank_bit) != 0U;
        if (first_value != second_value) {
            state = exchange_shard(
                state,
                static_cast<int>(rank ^ first_rank_bit ^ second_rank_bit)
            );
        }
        return;
    }
    const std::size_t local_qubit = first_local ? first : second;
    const std::size_t distributed_qubit = first_local ? second : first;
    const std::size_t rank_bit = std::size_t{1} << (distributed_qubit - local_qubits);
    const bool distributed_value = (rank & rank_bit) != 0U;
    const int partner = static_cast<int>(rank ^ rank_bit);
    const std::vector<Complex> peer = exchange_shard(state, partner);
    const std::size_t local_bit = std::size_t{1} << local_qubit;
    for (std::size_t index = 0; index < state.size(); ++index) {
        const bool local_value = (index & local_bit) != 0U;
        if (local_value != distributed_value) {
            state[index] = peer[index ^ local_bit];
        }
    }
}

void apply_distributed_operation(
    std::vector<Complex>& state,
    const Operation& operation,
    std::size_t local_qubits,
    std::size_t rank
) {
    if (operation.qubits.size() == 1U) {
        const std::size_t qubit = operation.qubits[0];
        const Matrix2 matrix = operation_matrix(operation);
        if (qubit < local_qubits) {
            apply_local_single(state, matrix, qubit);
        } else {
            apply_cross_rank_single(state, matrix, qubit, local_qubits, rank);
        }
        return;
    }
    switch (operation.code) {
    case OperationCode::CX:
        apply_distributed_cx(
            state, operation.qubits[0], operation.qubits[1], local_qubits, rank
        );
        return;
    case OperationCode::CZ:
        apply_distributed_cz(
            state, operation.qubits[0], operation.qubits[1], local_qubits, rank
        );
        return;
    case OperationCode::SWAP:
        apply_distributed_swap(
            state, operation.qubits[0], operation.qubits[1], local_qubits, rank
        );
        return;
    default:
        throw std::invalid_argument("distributed statevector received an invalid operation");
    }
}
class MpiLifecycle {
public:
    MpiLifecycle() {
        int initialized = 0;
        mpi_check(MPI_Initialized(&initialized), "MPI_Initialized");
        if (initialized == 0) {
            int provided = MPI_THREAD_SINGLE;
            mpi_check(
                MPI_Init_thread(nullptr, nullptr, MPI_THREAD_FUNNELED, &provided),
                "MPI_Init_thread"
            );
            if (provided < MPI_THREAD_FUNNELED) {
                MPI_Finalize();
                throw std::runtime_error("MPI runtime does not provide MPI_THREAD_FUNNELED");
            }
            owns_mpi_ = true;
        }
        mpi_check(
            MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN),
            "MPI_Comm_set_errhandler"
        );
    }

    ~MpiLifecycle() {
        if (!owns_mpi_) {
            return;
        }
        int finalized = 0;
        if (MPI_Finalized(&finalized) == MPI_SUCCESS && finalized == 0) {
            MPI_Finalize();
        }
    }

private:
    bool owns_mpi_ = false;
};

MpiLifecycle& mpi_lifecycle() {
    static MpiLifecycle lifecycle;
    return lifecycle;
}

#endif

}  // namespace

bool mpi_compiled() noexcept {
#ifdef QUPY_HAS_MPI
    return true;
#else
    return false;
#endif
}
DistributedStateVector distributed_statevector(const Program& program) {
#ifndef QUPY_HAS_MPI
    static_cast<void>(program);
    throw std::runtime_error(
        "MPI support is not compiled; rebuild QuPy with an MPI C++ implementation available"
    );
#else
    static_cast<void>(mpi_lifecycle());
    int finalized = 0;
    mpi_check(MPI_Finalized(&finalized), "MPI_Finalized");
    if (finalized != 0) {
        throw std::runtime_error("MPI has already been finalized");
    }
    int raw_world_size = 0;
    int raw_rank = 0;
    mpi_check(MPI_Comm_size(MPI_COMM_WORLD, &raw_world_size), "MPI_Comm_size");
    mpi_check(MPI_Comm_rank(MPI_COMM_WORLD, &raw_rank), "MPI_Comm_rank");
    if (raw_world_size <= 0 || raw_rank < 0) {
        throw std::runtime_error("MPI returned an invalid communicator topology");
    }
    const std::size_t world_size = static_cast<std::size_t>(raw_world_size);
    const std::size_t rank = static_cast<std::size_t>(raw_rank);
    if (!is_power_of_two(world_size)) {
        throw std::invalid_argument("distributed statevector requires a power-of-two MPI world size");
    }
    if (program.num_qubits() >= std::numeric_limits<std::size_t>::digits) {
        throw std::length_error("distributed statevector exceeds native address space");
    }
    const std::size_t global_size = std::size_t{1} << program.num_qubits();
    if (world_size > global_size) {
        throw std::invalid_argument("MPI world size exceeds the state-vector dimension");
    }
    const std::size_t rank_qubits = integer_log2(world_size);
    const std::size_t local_qubits = program.num_qubits() - rank_qubits;
    const std::size_t local_size = global_size / world_size;
    std::vector<Complex> state(local_size, Complex{0.0, 0.0});
    if (rank == 0U) {
        state.front() = 1.0;
    }
    for (const Operation& operation : program.operations()) {
        apply_distributed_operation(state, operation, local_qubits, rank);
    }
    return {
        std::move(state),
        global_size,
        rank * local_size,
        rank,
        world_size,
        "mpi-statevector",
    };
#endif
}

}  // namespace qupy
