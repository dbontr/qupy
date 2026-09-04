from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import numpy as np
import pytest

import qupy as qp

_ROOT = Path(__file__).resolve().parents[1]
_CONTRACT_PATH = _ROOT / "api" / "compatibility_v0.toml"


def _contract() -> dict[str, object]:
    with _CONTRACT_PATH.open("rb") as file:
        return tomllib.load(file)


def _observable() -> qp.Observable:
    return qp.observable_from_z(qp.Z(0))


def _assert_read_only_array(
    value: np.ndarray,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype[np.generic],
) -> None:
    assert value.shape == shape
    assert value.dtype == dtype
    assert not value.flags.writeable


def test_checked_compatibility_contract_is_current() -> None:
    contract = _contract()
    assert contract["schema_version"] == 1
    assert contract["status"] == "reviewed-pre-1.0"
    assert contract["public_api_manifest"] == "api/public_api_v0.txt"
    assert contract["native_stub"] == "src/qupy/_native.pyi"
    assert contract["deprecated_public_symbols"] == []
    assert contract["explicit_backends"] == [
        "native-cpu",
        "native-cuda",
        "native-mps",
        "native-tn",
        "native-mpi",
    ]
    assert contract["parameterized_backends"] == ["native-cuda:<device>"]
    assert contract["optional_python_modules"] == [
        "jax",
        "torch",
        "braket",
        "qiskit",
        "qiskit_aer",
    ]


def test_versioned_public_boundaries_match_contract() -> None:
    versions = _contract()["versions"]
    assert isinstance(versions, dict)
    assert qp.ir_version() == versions["program_ir"]
    assert qp.circuit_ir_version() == versions["circuit_ir"]

    core_source = (_ROOT / "src" / "cpp" / "core.cpp").read_text(encoding="utf-8")
    planner_schemas = {int(value) for value in re.findall(r"qupy-planner-cost (\d+)", core_source)}
    assert min(planner_schemas) == versions["planner_cost_schema_min"]
    assert max(planner_schemas) == versions["planner_cost_schema_max"]
    planner_constants = {
        name: int(value)
        for name, value in re.findall(
            r"constexpr std::uint32_t (k(?:Workload|AdaptiveMpsPolicy|ObservablePolicy|DensityPolicy)Version) = (\d+)U;",
            core_source,
        )
    }
    assert planner_constants["kWorkloadVersion"] == versions["planner_workload"]
    assert planner_constants["kAdaptiveMpsPolicyVersion"] == versions["planner_mps_policy"]
    assert planner_constants["kObservablePolicyVersion"] == versions["planner_observable_policy"]
    assert planner_constants["kDensityPolicyVersion"] == versions["planner_density_policy"]

    tensor_source = (_ROOT / "src" / "cpp" / "tensor_network_cost.cpp").read_text(encoding="utf-8")
    tensor_constants = {
        name: int(value)
        for name, value in re.findall(
            r"constexpr std::uint32_t (kTensorNetwork(?:CostSchema|Policy|Workload)Version) = (\d+)U;",
            tensor_source,
        )
    }
    assert (
        tensor_constants["kTensorNetworkCostSchemaVersion"]
        == versions["tensor_network_cost_schema"]
    )
    assert tensor_constants["kTensorNetworkPolicyVersion"] == versions["tensor_network_policy"]
    assert tensor_constants["kTensorNetworkWorkloadVersion"] == versions["tensor_network_workload"]

    provider_header = (_ROOT / "include" / "qupy" / "provider_abi.h").read_text(encoding="utf-8")
    match = re.search(r"#define QUPY_PROVIDER_ABI_VERSION (\d+)u", provider_header)
    assert match is not None
    assert int(match.group(1)) == versions["provider_abi"]

    class CapabilityProvider:
        def capabilities_json(self) -> str:
            return json.dumps(
                {
                    "formats": ["openqasm3"],
                    "hardware_target": {
                        "schema_version": versions["provider_hardware_target_schema"],
                        "name": "compatibility-fixture",
                        "num_qubits": 2,
                        "one_qubit_operations": ["h", "rz"],
                        "two_qubit_operations": ["cz"],
                    },
                }
            )

    capabilities = qp.provider_capabilities(CapabilityProvider())  # type: ignore[arg-type]
    assert capabilities.hardware_target is not None
    assert capabilities.hardware_target.name == "compatibility-fixture"

    circuit = qp.Circuit(1, 1).h(0).measure(0, 0)
    assert circuit.to_openqasm3().startswith(f"OPENQASM {versions['standalone_openqasm']};")
    provider_program = qp.provider_program(circuit)
    assert provider_program.text.startswith(f"OPENQASM {versions['provider_openqasm']};")

    qir = qp.to_qir_base_profile(qp.h(qp.Program(1), 0), measure_all=True)
    assert f'"qir_profiles"="{versions["qir_profile"]}"' in qir.text


def test_core_result_array_shape_dtype_and_ownership_contracts() -> None:
    template = qp.ry(qp.Program(1), 0.0, 0)
    slots = [qp.ParameterSlot(0)]
    parameters = np.array([0.37], dtype=np.float64)
    bound = template.bind(slots, parameters.tolist())
    observable = _observable()

    state = qp.statevector(bound, backend="native-cpu")
    _assert_read_only_array(state.values, shape=(2,), dtype=np.dtype(np.complex128))

    probabilities = qp.probabilities(bound, backend="native-cpu")
    _assert_read_only_array(probabilities.values, shape=(2,), dtype=np.dtype(np.float64))

    samples = qp.sample(bound, shots=4, seed=7, backend="native-cpu")
    _assert_read_only_array(samples.values, shape=(4, 1), dtype=np.dtype(np.int8))

    density = qp.density_matrix(bound, backend="native-cpu")
    _assert_read_only_array(density.values, shape=(2, 2), dtype=np.dtype(np.complex128))

    gradient = qp.value_and_grad(template, observable, slots, parameters, backend="native-cpu")
    _assert_read_only_array(gradient.gradient, shape=(1,), dtype=np.dtype(np.float64))

    jacobian = qp.jacobian(template, [observable], slots, parameters, backend="native-cpu")
    _assert_read_only_array(jacobian.values, shape=(1,), dtype=np.dtype(np.float64))
    _assert_read_only_array(jacobian.jacobian, shape=(1, 1), dtype=np.dtype(np.float64))

    hessian = qp.hessian(template, observable, slots, parameters, backend="native-cpu")
    _assert_read_only_array(hessian.gradient, shape=(1,), dtype=np.dtype(np.float64))
    _assert_read_only_array(hessian.hessian, shape=(1, 1), dtype=np.dtype(np.float64))


def test_explicit_backend_names_and_fail_closed_behavior() -> None:
    contract = _contract()
    assert contract["explicit_backends"] == [
        "native-cpu",
        "native-cuda",
        "native-mps",
        "native-tn",
        "native-mpi",
    ]
    assert contract["parameterized_backends"] == ["native-cuda:<device>"]

    program = qp.ry(qp.Program(1), 0.37, 0)
    observable = _observable()
    assert qp.statevector(program, backend="native-cpu").backend == "native-cpu"
    assert qp.statevector(program, backend="native-mps").backend == "native-mps"
    assert qp.expect_observable(program, observable, backend="native-tn").backend == "native-tn"

    if qp.cuda_available():
        assert qp.statevector(program, backend="native-cuda").backend == "native-cuda"
    else:
        with pytest.raises(RuntimeError):
            qp.statevector(program, backend="native-cuda")

    if qp.mpi_compiled():
        assert (
            qp.expect_observable(program, observable, backend="native-mpi").backend == "native-mpi"
        )
    else:
        with pytest.raises(RuntimeError, match="MPI support is not compiled"):
            qp.expect_observable(program, observable, backend="native-mpi")


def test_exception_classes_are_part_of_the_reviewed_boundary() -> None:
    with pytest.raises(ValueError, match="outside this program"):
        qp.h(qp.Program(1), 1)
    with pytest.raises(ValueError, match="shots must be at least 1"):
        qp.sample(qp.Program(1), shots=0)
    with pytest.raises(ValueError, match="unknown backend"):
        qp.statevector(qp.Program(1), backend="missing")

    initial = qp.density_matrix(qp.Program(1))
    hamiltonian = np.zeros((2, 2), dtype=np.complex128)
    with pytest.raises(TypeError):
        qp.lindblad_evolve(initial, hamiltonian, dt=0.01, steps=1)


def test_optional_dependency_and_deprecation_contract_is_declared() -> None:
    contract = _contract()
    optional_modules = contract["optional_python_modules"]
    assert isinstance(optional_modules, list)
    wheel_verifier = (_ROOT / "tools" / "verify_wheel.py").read_text(encoding="utf-8")
    for module in optional_modules:
        assert isinstance(module, str)
        assert f'"{module}"' in wheel_verifier

    deprecated = contract["deprecated_public_symbols"]
    assert deprecated == []
    assert set(deprecated).isdisjoint(qp.__all__)
