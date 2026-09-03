# Distributed scale engines

QuPy can distribute two independent-workload simulation paths across an MPI world: general tensor-network expectation terms and noisy quantum trajectories. These APIs complement the existing MPI-sharded state-vector and Pauli-reduction paths.

## Execution model

MPI is optional. QuPy builds without an MPI implementation and keeps the same Python package surface. Distributed execution fails explicitly when MPI support is unavailable or when the active MPI world has fewer than two ranks.

The MPI launcher owns process placement. A rank can represent a process on the same host or a process on another node. QuPy reports MPI participation; it does not claim a physical accelerator mapping that the launcher did not establish.

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

Distributed scale APIs use collective failure propagation after rank-local simulation work. If one rank cannot complete its assigned tensor contraction or trajectory batch, all ranks fail the distributed request instead of returning a partial aggregate.

Validation that is independent of rank placement occurs before distributed work. Invalid observable qubits, zero trajectory counts, and zero tensor-memory limits fail explicitly.

## Relationship to other distributed paths

QuPy has three distinct MPI execution structures:

- sharded state vectors divide one dense state across ranks and exchange shards for nonlocal gates;
- distributed Pauli reduction evaluates observables against that sharded state;
- distributed scale engines divide independent tensor-network terms or trajectory samples across ranks.

The appropriate structure depends on the workload. The independent-work APIs in this document do not replace state-vector sharding, and they do not imply multi-GPU CUDA execution. GPU-aware rank placement and direct multi-GPU kernels require separate device-specific evidence and runtime support.
