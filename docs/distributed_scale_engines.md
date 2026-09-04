# Distributed scale engines

QuPy can distribute two independent-workload simulation paths across an MPI world: general tensor-network expectation terms and noisy quantum trajectories. These APIs complement the existing MPI-sharded state-vector and Pauli-reduction paths.

## Execution model

MPI is optional. QuPy builds without an MPI implementation and keeps the same Python package surface. Distributed calls fail explicitly when MPI support is unavailable. The independent-work tensor-network and trajectory APIs require at least two ranks; state-vector shard APIs also operate in a one-rank MPI world, where they provide no distribution benefit.

The MPI launcher owns process placement. A rank can represent a process on the same host or a process on another node. QuPy reports MPI participation; it does not claim a physical accelerator mapping that the launcher did not establish.

CUDA device ownership is separately explicit within each process. `cuda_device_count()` reports CUDA-driver-visible ordinals, and an explicit backend such as `native-cuda:1` selects a per-device runtime/context without changing process-global device state. `distributed_cuda_statevector()` uses that ownership model directly for MPI shards.

## Rank-local CUDA state-vector sharding

`distributed_cuda_statevector()` stores each rank's state-vector shard in one selected CUDA device workspace. Local-qubit gates execute with the ordinary native CUDA kernels while the device runtime remains exclusively owned by the distributed shard session.

```python
import qupy as qp

program = qp.h(qp.Program(4), 0)
program = qp.cx(program, 0, 3)

shard = qp.distributed_cuda_statevector(program)
print(shard.rank, shard.global_offset, shard.backend)
```

With `device=None`, QuPy maps the launcher's reported local MPI rank to the same CUDA ordinal. It does not wrap or modulo the ordinal: if any rank maps to an unavailable device, the request fails collectively. Passing `device=<ordinal>` overrides that mapping for every process that makes the call.

A gate whose qubits all belong to the local shard executes without moving the shard back to host memory. A gate touching a distributed qubit stages the current shard to host memory, performs the same exact MPI exchange/update used by `distributed_statevector()`, then uploads the updated shard to the same CUDA device. The result is therefore exact subject to floating-point roundoff and has genuine rank-local GPU state ownership, but cross-rank transport is currently host-staged rather than CUDA-aware or peer-to-peer.

The returned `DistributedStateVector.local_values` is a host copy of the final local shard because Python owns the result after execution. Its backend identity is `native-mpi-cuda:<device>` and therefore reports the local ordinal used by that rank.

Initialization, local CUDA gate execution, and host/device transfer failures are reduced collectively before ranks advance to the next distributed boundary. This prevents a rank-local CUDA failure from leaving peers blocked in a later MPI exchange.

## Tensor-network expectation distribution

`distributed_tensor_network_expectation()` partitions Pauli terms across ranks by term index. Each active rank contracts its assigned terms with the exact general tensor-network engine, using the same deterministic greedy contraction path and the same per-intermediate `max_tensor_bytes` ceiling as local execution.

```python
import qupy as qp

program = qp.h(qp.Program(2), 0)
program = qp.cx(program, 0, 1)
observable = qp.observable(
    [
        qp.pauli_term(1.0, [qp.pauli(0, qp.Pauli.X), qp.pauli(1, qp.Pauli.X)]),
        qp.pauli_term(0.5, [qp.pauli(0, qp.Pauli.Z), qp.pauli(1, qp.Pauli.Z)]),
    ]
)

result = qp.distributed_tensor_network_expectation(program, observable)
print(result.value)
print(result.world_size, result.active_ranks)
```

The result is exact subject to floating-point roundoff. Rank-local values and work counters are reduced across the MPI world. Peak tensor rank and peak tensor bytes report the largest rank-local intermediate. `active_ranks` counts ranks that received at least one nonzero Pauli term.

This method is useful when an observable contains enough independent terms to keep multiple ranks busy. A single Pauli term cannot be accelerated by term partitioning; use the local general tensor-network engine for that case unless another distributed method is more appropriate.

## Trajectory ensemble distribution

`distributed_trajectory_expectations()` partitions a requested trajectory count across ranks. Each active rank executes its local stochastic ensemble with the native CPU trajectory engine. QuPy combines rank-local first and second moments to produce the global mean and standard error.

```python
import qupy as qp

program = qp.x(qp.Program(1), 0)
noisy = qp.NoisyProgram(
    program,
    [qp.NoiseInstruction(1, qp.amplitude_damping(0, 0.2))],
)
observable = qp.observable_from_z(qp.Z(0))

result = qp.distributed_trajectory_expectation(
    noisy,
    observable,
    trajectories=8192,
    seed=7,
)
print(result.value, result.standard_error)
print(result.world_size, result.active_ranks)
```

A supplied seed is broadcast from rank zero and deterministically mixed with each rank identity. Repeating the same request with the same MPI world size and seed produces the same result. Changing the world size changes the rank partition and therefore can change the sampled sequence while preserving the same statistical estimator.

Trajectory estimates remain statistical. `exact` is always `False`. `state_bytes_per_rank` reports the largest pure-state trajectory buffer used by an active rank; it does not include MPI implementation buffers or process overhead.

## Failure behavior

Distributed scale APIs use collective failure propagation after rank-local simulation work. If one rank cannot complete its assigned tensor contraction, trajectory batch, CUDA initialization, local CUDA gate, or CUDA shard transfer, all ranks fail the distributed request instead of returning a partial aggregate.

Validation that is independent of rank placement occurs before distributed work. Invalid observable qubits, zero trajectory counts, zero tensor-memory limits, unsupported MPI topologies, and unusable distributed CUDA device mappings fail explicitly.

## Relationship to other distributed paths

QuPy has four distinct MPI execution structures:

- host-sharded state vectors divide one dense state across ranks and exchange host shards for nonlocal gates;
- CUDA-sharded state vectors keep each rank's shard on an explicit local GPU for local gates and host-stage only distributed-qubit boundaries;
- distributed Pauli reduction evaluates observables against the host-sharded state-vector path;
- distributed scale engines divide independent tensor-network terms or trajectory samples across ranks.

The appropriate structure depends on the workload. The distributed CUDA path establishes real rank-to-GPU state ownership without claiming direct GPU networking. CUDA-aware MPI, GPUDirect-style transport, distributed CUDA observable reduction, and measured multi-node GPU scaling remain separate optimization/evidence steps rather than implied capabilities.
