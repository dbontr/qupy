import json
import math
from dataclasses import replace

import pytest

import qupy as qp
from benchmarks.cost_model import CostObservation, fit_cost_models, load_observations


def _statevector_observation(index: int, active_qubits: int) -> CostObservation:
    compiled_steps = active_qubits + 1
    threads = 1 if active_qubits < 16 else min(16, max(1, (1 << active_qubits) // 8192))
    cost_class = "statevector-serial" if threads == 1 else "statevector-parallel"
    observation = CostObservation(
        workload_fingerprint=f"{index:064x}",
        workload=f"statevector-{active_qubits}",
        engine_version="synthetic",
        method="statevector",
        cost_class=cost_class,
        median_ns=1.0,
        active_qubits=active_qubits,
        active_operations=compiled_steps,
        compiled_steps=compiled_steps,
        threads=threads,
    )
    log_work = observation.log_work
    runtime = math.exp(4.5 + 0.55 * log_work + 0.012 * log_work * log_work)
    return CostObservation(
        workload_fingerprint=observation.workload_fingerprint,
        workload=observation.workload,
        engine_version=observation.engine_version,
        method=observation.method,
        cost_class=observation.cost_class,
        median_ns=runtime,
        active_qubits=observation.active_qubits,
        active_operations=observation.active_operations,
        compiled_steps=observation.compiled_steps,
        threads=observation.threads,
    )


def _pauli_observation(index: int, operations: int) -> CostObservation:
    return CostObservation(
        workload_fingerprint=f"{index:064x}",
        workload=f"pauli-{operations}",
        engine_version="synthetic",
        method="pauli-propagation",
        cost_class="pauli-propagation",
        median_ns=math.exp(5.0 + 0.85 * math.log(operations)),
        active_qubits=operations,
        active_operations=operations,
        compiled_steps=operations,
        threads=1,
    )


def test_cost_model_recovers_synthetic_execution_curves() -> None:
    observations = [
        *(_statevector_observation(index, qubits) for index, qubits in enumerate(range(10, 22))),
        *(
            _pauli_observation(100 + index, operations)
            for index, operations in enumerate((32, 64, 128, 256, 512, 1024, 2048, 4096))
        ),
    ]
    report = fit_cost_models(
        observations,
        host={"machine": "synthetic"},
        max_holdout_factor=1.001,
        max_holdout_median_factor=1.001,
    )

    assert report.schema_version == 1
    assert report.workload_version == 1
    assert report.engine_version == "synthetic"
    assert report.validated
    assert report.observations == len(observations)
    assert {model.cost_class for model in report.models} == {
        "pauli-propagation",
        "statevector-parallel",
        "statevector-serial",
    }
    for model in report.models:
        assert model.validated
        assert model.holdout_error.max_factor == pytest.approx(1.0, abs=1e-9)
        assert set(model.training_fingerprints).isdisjoint(model.holdout_fingerprints)


def test_cost_model_aggregates_repeated_workload_fingerprints() -> None:
    observation = _statevector_observation(1, 12)
    repeated = CostObservation(
        workload_fingerprint=observation.workload_fingerprint,
        workload=observation.workload,
        engine_version=observation.engine_version,
        method=observation.method,
        cost_class=observation.cost_class,
        median_ns=observation.median_ns * 1.1,
        active_qubits=observation.active_qubits,
        active_operations=observation.active_operations,
        compiled_steps=observation.compiled_steps,
        threads=observation.threads,
    )
    observations = [
        observation,
        repeated,
        *(_statevector_observation(index, qubits) for index, qubits in enumerate(range(11, 16), 2)),
    ]
    report = fit_cost_models(
        observations,
        host={"machine": "synthetic"},
        max_holdout_factor=2.0,
        max_holdout_median_factor=2.0,
    )
    assert report.observations == 6


def test_cost_model_rejects_mixed_engine_versions() -> None:
    observations = [
        *(_statevector_observation(index, qubits) for index, qubits in enumerate(range(10, 22))),
        *(
            _pauli_observation(100 + index, operations)
            for index, operations in enumerate((32, 64, 128, 256, 512, 1024, 2048, 4096))
        ),
    ]
    observations[-1] = replace(observations[-1], engine_version="different")
    with pytest.raises(ValueError, match="engine versions"):
        fit_cost_models(observations, host={"machine": "synthetic"})


def test_cost_model_loader_rejects_mixed_hosts(tmp_path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    base = {"schema_version": 1, "results": []}
    first.write_text(
        json.dumps({**base, "host": {"machine": "alpha"}}),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps({**base, "host": {"machine": "beta"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="different hosts"):
        load_observations((first, second))


def test_validated_report_emits_native_planner_artifact() -> None:
    observations = [
        *(
            replace(_statevector_observation(index, qubits), engine_version=qp.core_version())
            for index, qubits in enumerate(range(10, 22))
        ),
        *(
            replace(_pauli_observation(100 + index, operations), engine_version=qp.core_version())
            for index, operations in enumerate((32, 64, 128, 256, 512, 1024, 2048, 4096))
        ),
    ]
    report = fit_cost_models(
        observations,
        host={"planner_host_fingerprint": qp.planner_host_fingerprint()},
        max_holdout_factor=1.001,
        max_holdout_median_factor=1.001,
    )
    payload = report.to_planner_text()
    assert payload.startswith("qupy-planner-cost 1\n")
    assert f"engine {qp.core_version()}\n" in payload
    assert f"host {qp.planner_host_fingerprint()}\n" in payload
    assert payload.count("\nmodel ") == 3

    weak_model = replace(
        report.models[0],
        holdout_error=replace(report.models[0].holdout_error, max_factor=2.1),
    )
    weak_report = replace(report, models=(weak_model, *report.models[1:]))
    with pytest.raises(ValueError, match="fixed promotion thresholds"):
        weak_report.to_planner_text()
