# Quantum trajectories

QuPy provides Monte Carlo quantum trajectories for noisy circuits when a full density matrix is unnecessarily expensive. The trajectory engine is statistical and does not replace the exact `density_matrix()` API.

## Why trajectories

For `n` qubits, a dense state vector contains `2^n` complex amplitudes while a dense density matrix contains `4^n` complex values. A trajectory therefore keeps one state-vector-sized quantum state and repeats the noisy evolution to estimate ensemble observables.

`TrajectoryResult.state_bytes` reports the quantum-state allocation for one trajectory. It does not include small observable/statistics workspaces.

## Discrete Kraus unraveling

For each `NoiseInstruction`, QuPy obtains the same Kraus operators used by the exact density-matrix channel. Given a normalized state `|psi>`, each branch receives probability

`p_i = ||K_i |psi>||^2`.

QuPy samples one branch using the trajectory random stream and updates the state to

`K_i |psi> / sqrt(p_i)`.

The procedure is applied at the exact `after_operation` insertion points carried by `NoisyProgram`. Multiple channels at the same insertion point preserve their input order.

The ensemble average of trajectory observables estimates the corresponding open-system expectation value. `exact` is therefore always `False` for trajectory results, even when a particular channel happens to be deterministic.

## Supported noise

The trajectory engine uses every existing single-qubit QuPy noise channel:

- bit flip
- phase flip
- depolarizing
- amplitude damping
- phase damping
- Pauli channels
- custom trace-preserving Kraus channels

The Kraus definitions are the same as the exact density-matrix implementation. Custom Kraus validation remains owned by `kraus_channel()`.

## Python API

```python
import qupy as qp

program = qp.x(qp.Program(1), 0)
noisy = qp.NoisyProgram(
    program,
    [qp.NoiseInstruction(1, qp.amplitude_damping(0, 0.25))],
)
z = qp.observable_from_z(qp.Z(0))

estimate = qp.trajectory_expectation(
    noisy,
    z,
    trajectories=20_000,
    seed=1234,
)

print(estimate.value)
print(estimate.standard_error)
print(estimate.state_bytes)
print(estimate.seed)
```

Use `trajectory_expectations()` to evaluate several observables from the same trajectory ensemble:

```python
result = qp.trajectory_expectations(
    noisy,
    [z, qp.observable([qp.pauli_term(1.0, [])])],
    trajectories=20_000,
    seed=1234,
)
```

The returned NumPy arrays contain one mean and one standard error per observable.

## Statistical contract

QuPy updates means and second moments online using Welford accumulation. For more than one trajectory, the reported standard error is the sample standard deviation divided by the square root of the trajectory count. With one trajectory, the standard error is `NaN` because sampling uncertainty cannot be estimated from a single realization.

The actual seed is always returned. Supplying a seed replays the same MT19937-64 random stream and uses an explicit 53-bit mapping from engine output to the unit interval. Omitting the seed generates one and returns it so a run can be repeated.

## Backend contract

The first trajectory engine is native CPU only. `backend="auto"`, `"cpu"`, and `"native-cpu"` select the same implementation. Other backends fail closed instead of silently transferring work or changing stochastic semantics.

This boundary is intentional. GPU and distributed trajectory execution can be added later with an explicit reproducibility policy and independent validation.

## Relationship to exact density matrices

Use `density_matrix()` when the full mixed state is required or an exact finite-size result is practical. Use trajectories when only ensemble observables are needed and `4^n` density storage is the limiting resource.

Conformance tests compare seeded trajectory estimates with the existing exact density solver and include deterministic channels where both paths must agree to numerical precision.

## Current limits

- The engine estimates final observables; it does not return every individual trajectory state.
- Noise channels are the existing single-qubit `NoiseChannel` model.
- Trajectories execute sequentially in this first implementation. Parallel trajectory scheduling can be added without changing the public statistical contract.
- CUDA and MPI trajectory backends are not claimed yet.
