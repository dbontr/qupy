# Pre-1.0 compatibility review v0

This review records the first complete compatibility baseline for the current QuPy alpha surface. It does not declare QuPy 1.0 or freeze every alpha-era design choice. It makes the reviewed contract explicit so later incompatible changes are deliberate and versioned.

The machine-readable record is `api/compatibility_v0.toml`. The executable review is `tests/test_compatibility_review.py`. Native runtime/stub agreement remains enforced separately by `tools/verify_native_stub.py`.

## Reviewed surfaces

1. **Top-level names and signatures.** `api/public_api_v0.txt` pins exported top-level names. `src/qupy/_native.pyi` is structurally checked against a stub generated from the built nanobind extension. Strict mypy checks the Python implementation.
2. **Result shape and ownership.** The compatibility suite pins representative state-vector, probability, sample, density-matrix, gradient, Jacobian, and Hessian shapes, NumPy dtypes, and read-only result-backed views.
3. **Exceptions and validation.** The suite pins representative `ValueError`, `TypeError`, and fail-closed runtime boundaries rather than accepting arbitrary exception drift.
4. **Backend names and explicit execution.** `native-cpu`, `native-cuda`, `native-mps`, `native-tn`, and `native-mpi` are the reviewed explicit backend names. Hardware- or build-dependent backends must fail explicitly when unavailable.
5. **Versioned boundaries.** Numerical Program IR and hardware Circuit IR remain version 1. Planner cost artifacts currently accept schemas 1 through 5 with workload/policy versions recorded in the compatibility manifest; tensor-network cost artifacts remain schema/policy/workload version 1. Provider C ABI and the provider `hardware_target` capability schema remain version 1. Standalone Circuit interchange uses OpenQASM 3.1, the generic provider boundary uses OpenQASM 3.0, and QIR output remains the Base Profile.
6. **Optional dependencies.** JAX, PyTorch, Amazon Braket, Qiskit, and Qiskit Aer remain outside the base runtime. The isolated-wheel verifier requires those modules to remain absent and requires their adapters to fail with explicit dependency errors when invoked without them. CUDA and MPI continue to fail closed when their native runtime/build support is unavailable.
7. **Release artifacts.** Wheels execute in clean virtual environments on the supported Python/OS matrix. The source distribution is rebuilt to a wheel and subjected to the same isolated runtime checks.
8. **Deprecation status.** This baseline contains no deprecated public symbol. `deprecated_public_symbols` is intentionally empty. A future removal must first update the compatibility record and provide a deprecation or migration path unless correctness, security, or architecture requires immediate removal.

## What this baseline does not claim

This review is a compatibility commitment for the current alpha boundary, not evidence that all future 1.0 scale goals are complete. It does not certify physical QPU fidelity, vendor service behavior, multi-GPU scale-out, every tensor-network topology, or every QEC family.

The package therefore remains `0.3.0a0`. A future 1.0 candidate should promote a reviewed successor of this baseline after the remaining scale and provider evidence is sufficient.

## Change discipline

An incompatible pre-1.0 change must update every affected checked contract in the same change set: public manifest or native stub, `api/compatibility_v0.toml` or its successor, executable compatibility tests, release notes or durable documentation, and package version when required by the pre-1.0 policy.
