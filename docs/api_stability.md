# API and compatibility policy

QuPy tracks compatibility as an explicit engineering contract. The project is still pre-1.0, so incompatible changes remain possible, but they must be deliberate, reviewable, and tied to a version change. Accidental removal or renaming of a public top-level symbol is a test failure.

## Public Python surface

The primary public Python surface is `qupy.__all__`. The checked baseline is stored in [`api/public_api_v0.txt`](../api/public_api_v0.txt). The manifest is intentionally simple: one exported top-level name per line, in the same order as `qupy.__all__`.

A change to the public surface must do one of the following:

- preserve the current manifest exactly;
- add a new public symbol and update the manifest in the same change; or
- intentionally remove or rename a symbol, update the manifest, document the incompatibility, and use an appropriate package-version transition.

Names beginning with `_` are implementation details unless another QuPy specification explicitly defines them as interchange artifacts. Importing a private module does not create a compatibility promise.

The manifest protects name availability. It does not by itself prove behavioral compatibility. Tests for each subsystem remain the semantic contract for argument validation, result shape, exactness, failure behavior, ownership, and backend selection.

## Version identity

A packaged QuPy installation has three version identities that must agree:

1. the `qupy-compute` distribution version;
2. `qupy.__version__`;
3. the native C++ core version returned by `qupy.core_version()`.

QuPy checks the Python/native pair during import when distribution metadata is available. The test suite also compares all three identities with the version declared in `pyproject.toml`. A wheel containing a native extension from a different QuPy release must therefore fail closed instead of running as a mixed installation.

Standalone C++ builds do not have Python distribution metadata. Their core version remains the native build identity.

## Release artifact verification

Building an artifact is not sufficient release evidence. CI installs each built platform wheel into a fresh virtual environment with source-tree Python paths removed, then imports and executes QuPy from that environment.

The isolated verification requires:

- `qupy` and its native extension to resolve from the fresh environment rather than the repository checkout;
- the distribution, Python package, and native core versions to agree;
- the native extension to report C++20 and the expected program/circuit IR versions;
- the installed package to contain its `py.typed` marker; and
- a Bell-state statevector and probability calculation to execute correctly through the installed native wheel.

The Linux Python 3.12 release lane additionally rebuilds the generated source distribution into a wheel in an isolated build environment and applies the same installation/runtime checks to that rebuilt wheel. This catches source-distribution omissions that a direct source-tree wheel build can miss.

The verifier lives at `tools/verify_wheel.py` and intentionally has no QuPy import in its driver process. The QuPy import occurs only inside the isolated child environment. That environment installs only the base wheel and its required dependencies, verifies that JAX, PyTorch, Amazon Braket, Qiskit, and Qiskit Aer remain absent after importing QuPy, and checks that invoking each optional adapter fails with the documented dependency error rather than importing an optional ecosystem implicitly.

The Linux Python 3.12 CI lane also generates a reference stub from the built nanobind extension and compares it structurally with the checked `src/qupy/_native.pyi`. The verifier requires complete public declaration and class-member coverage and matching callable argument structure while deliberately ignoring nanobind's runtime metaclass and callable-wrapper implementation details. This catches native API drift without treating ordinary static type checking as sufficient evidence or suppressing whole classes of nanobind-specific false positives.

## Pre-1.0 policy

Before 1.0, QuPy may make incompatible API changes when they materially improve correctness or the long-term design. Such changes must not be silent. They require:

- an intentional public-manifest update;
- release notes or equivalent durable documentation that names the incompatibility;
- a version change consistent with the project's pre-1.0 release policy; and
- migration guidance when a direct replacement exists.

Deprecation is preferred when an old interface can be retained without correctness, security, or architectural harm.

## 1.0 policy

For 1.x releases, QuPy intends to follow semantic versioning for the documented public Python API:

- patch releases preserve public behavior apart from compatible bug fixes;
- minor releases may add public interfaces and compatible capabilities;
- removal, incompatible signature changes, or intentional semantic breaks require a new major version unless an interface must be disabled to fix a critical security or correctness defect.

The 1.0 release should promote a reviewed successor of the current pre-1.0 manifest rather than treating the current alpha surface as automatically frozen.

## Separately versioned contracts

Several durable formats evolve independently from the package version. Their own version fields remain authoritative:

- numerical `Program` IR: `ir_version()`;
- hardware `Circuit` IR: `circuit_ir_version()`;
- planner cost artifacts: artifact schema and policy versions;
- tensor-network planner artifacts: their own schema and policy version;
- provider C ABI: currently version 1;
- provider `hardware_target` capability schema: currently version 1;
- supported OpenQASM and QIR interchange subsets: governed by their documented format/profile boundaries.

A package release may support more than one compatible artifact version. QuPy must reject unsupported or malformed versions rather than reinterpret them silently.

## Compatibility review v0

The first complete pre-1.0 review is recorded in `api/compatibility_v0.toml` and explained in `docs/compatibility_review_v0.md`. `tests/test_compatibility_review.py` makes the reviewed boundary executable alongside the top-level API manifest, native stub/runtime conformance, optional-dependency checks, and isolated release-artifact gates.

The v0 review covers:

1. exported top-level names and native signatures;
2. representative result types, array shapes, NumPy dtypes, mutability, and ownership;
3. representative exception classes and fail-closed validation boundaries;
4. explicit backend names and unavailable-backend behavior;
5. program, circuit, provider, and interchange version boundaries;
6. optional-dependency behavior when JAX, PyTorch, Amazon Braket, Qiskit, Qiskit Aer, CUDA, or MPI support is absent;
7. wheel import/runtime behavior on every supported Python and operating-system target plus the source-distribution round trip; and
8. deprecation status and the migration discipline required before a reviewed public symbol is removed.

This closes the compatibility-review checklist for the current alpha baseline. It does not declare QuPy 1.0-ready by itself. The package remains alpha while the remaining scale and provider evidence matures, and future incompatible pre-1.0 changes must update the reviewed compatibility record deliberately.
