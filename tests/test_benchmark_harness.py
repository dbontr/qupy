import json
import math

import pytest

from benchmarks.harness import run_case, run_suite
from benchmarks.model import (
    Gate,
    entangled_nonclifford_z,
    ghz_z_clifford,
    local_nonclifford_z,
    workloads_for_profile,
)


def test_benchmark_model_rejects_invalid_gates_and_profiles() -> None:
    with pytest.raises(ValueError, match="requires 2 qubits"):
        Gate("cx", (0,))
    with pytest.raises(ValueError, match="requires a finite parameter"):
        Gate("ry", (0,))
    with pytest.raises(ValueError, match="unknown benchmark profile"):
        workloads_for_profile("missing")


def test_smoke_profile_covers_distinct_execution_classes() -> None:
    workloads = workloads_for_profile("smoke")
    assert [workload.family for workload in workloads] == [
        "clifford-ghz-z",
        "local-nonclifford-z",
        "entangled-nonclifford-z",
    ]
    assert workloads[0].clifford
    assert not workloads[1].clifford
    assert not workloads[2].clifford


def test_qupy_benchmark_records_specialized_pauli_plan() -> None:
    result = run_case(ghz_z_clifford(64), "qupy", warmups=0, iterations=1)
    assert not result.skipped
    assert result.valid
    assert result.result_value == pytest.approx(0.0, abs=1e-12)
    assert result.method == "pauli-propagation"
    assert result.metadata["workload_version"] == 1
    assert len(str(result.metadata["workload_fingerprint"])) == 64
    assert result.metadata["original_qubits"] == 64
    assert result.metadata["original_operations"] == 64
    assert result.metadata["active_qubits"] == 64
    assert result.metadata["active_operations"] == 64
    assert result.metadata["single_qubit_operations"] == 1
    assert result.metadata["two_qubit_operations"] == 63
    assert result.metadata["parameterized_operations"] == 0
    assert result.metadata["non_clifford_operations"] == 0
    assert result.metadata["estimated_state_bytes"] == 0
    assert len(result.timings_ns) == 1
    assert result.timings_ns[0] > 0


def test_qupy_benchmark_records_exact_local_lightcone() -> None:
    workload = local_nonclifford_z(64)
    result = run_case(workload, "qupy", warmups=0, iterations=1)
    assert not result.skipped
    assert result.valid
    assert result.result_value == pytest.approx(-math.sin(0.37), abs=1e-12)
    assert result.method == "statevector-lightcone"
    assert result.metadata["original_qubits"] == 64
    assert result.metadata["original_operations"] == 3
    assert result.metadata["active_qubits"] == 1
    assert result.metadata["active_operations"] == 2
    assert result.metadata["single_qubit_operations"] == 2
    assert result.metadata["two_qubit_operations"] == 0
    assert result.metadata["parameterized_operations"] == 1
    assert result.metadata["non_clifford_operations"] == 1
    assert result.metadata["estimated_state_bytes"] == 32


def test_qupy_benchmark_records_entangled_dense_fallback() -> None:
    workload = entangled_nonclifford_z(8)
    result = run_case(workload, "qupy", warmups=0, iterations=1)
    assert not result.skipped
    assert result.valid
    assert result.result_value == pytest.approx(-math.sin(0.37), abs=1e-12)
    assert result.method == "statevector"
    assert result.metadata["active_qubits"] == 8
    assert result.metadata["estimated_state_bytes"] == 4096


def test_report_is_machine_readable_and_stable_shape() -> None:
    report = run_suite(
        (ghz_z_clifford(6),),
        ("qupy",),
        warmups=0,
        iterations=2,
    )
    decoded = json.loads(report.to_json())
    assert decoded["schema_version"] == 1
    assert decoded["host"]["python"]
    assert len(decoded["results"]) == 1
    result = decoded["results"][0]
    assert result["engine"] == "qupy"
    assert result["valid"] is True
    assert len(result["timings_ns"]) == 2
