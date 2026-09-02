from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import qupy as qp

_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class MpsWorkload:
    name: str
    qubits: int
    variant: str
    layers: int = 1


def calibration_profile() -> tuple[MpsWorkload, ...]:
    chain = tuple(MpsWorkload(f"mps-chain-{q}", q, "chain") for q in range(8, 25, 2))
    star = tuple(MpsWorkload(f"mps-star-{q}", q, "star") for q in range(8, 21, 2))
    ladder = tuple(
        MpsWorkload(f"mps-ladder{layers}-{q}", q, "ladder", layers)
        for layers in (2, 4, 6)
        for q in range(8, 17, 2)
    )
    long_pairs = tuple(
        MpsWorkload(f"mps-long{layers}-{q}", q, "long-pairs", layers)
        for layers in (1, 2)
        for q in range(8, 17, 2)
    )
    ring = tuple(MpsWorkload(f"mps-ring-{q}", q, "ring") for q in range(8, 19, 2))
    cut_chain = tuple(
        MpsWorkload(f"mps-cut{sources}-{q}", q, "cut-chain", sources)
        for q in range(8, 15, 2)
        for sources in sorted({1, max(2, (q // 2) // 2), q // 2})
    )
    return chain + star + ladder + long_pairs + ring + cut_chain


def policy_profile() -> tuple[MpsWorkload, ...]:
    chain = tuple(MpsWorkload(f"policy-chain-{q}", q, "chain") for q in (15, 17, 19, 21))
    star = tuple(MpsWorkload(f"policy-star-{q}", q, "star") for q in (15, 17, 19))
    ladder = tuple(
        MpsWorkload(f"policy-ladder{layers}-{q}", q, "ladder", layers)
        for layers in (3, 5, 7)
        for q in (15, 17)
    )
    ring = tuple(MpsWorkload(f"policy-ring-{q}", q, "ring") for q in (15, 17))
    cut_chain = (
        MpsWorkload("policy-cut1-15", 15, "cut-chain", 1),
        MpsWorkload("policy-cut3-15", 15, "cut-chain", 3),
        MpsWorkload("policy-cut7-15", 15, "cut-chain", 7),
        MpsWorkload("policy-cut1-17", 17, "cut-chain", 1),
        MpsWorkload("policy-cut3-17", 17, "cut-chain", 3),
        MpsWorkload("policy-cut4-17", 17, "cut-chain", 4),
    )
    return chain + star + ladder + ring + cut_chain


def smoke_profile() -> tuple[MpsWorkload, ...]:
    return (
        MpsWorkload("mps-chain-10", 10, "chain"),
        MpsWorkload("mps-star-8", 8, "star"),
        MpsWorkload("mps-ladder2-10", 10, "ladder", 2),
    )


def _program(workload: MpsWorkload) -> qp.Program:
    program = qp.Program(workload.qubits)
    if workload.variant == "chain":
        program = qp.ry(program, 0.371, 0)
        for qubit in range(workload.qubits - 1):
            program = qp.cx(program, qubit, qubit + 1)
        return program
    if workload.variant == "star":
        program = qp.ry(program, 0.371, 0)
        for qubit in range(1, workload.qubits):
            program = qp.cx(program, 0, qubit)
        return program
    if workload.variant == "ladder":
        for qubit in range(workload.qubits):
            program = qp.ry(program, 0.019 * (qubit + 1), qubit)
        for layer in range(workload.layers):
            parity = layer % 2
            for qubit in range(parity, workload.qubits - 1, 2):
                program = qp.cz(program, qubit, qubit + 1)
            for qubit in range(workload.qubits):
                program = qp.rz(program, 0.007 * (layer + qubit + 1), qubit)
        for qubit in range(workload.qubits - 1):
            program = qp.cx(program, qubit, qubit + 1)
        return program
    if workload.variant == "long-pairs":
        for layer in range(workload.layers):
            for qubit in range(workload.qubits):
                program = qp.ry(program, 0.013 * (1 + layer + qubit), qubit)
            for first in range(workload.qubits // 2):
                second = workload.qubits - first - 1
                if (layer + first) % 2 == 0:
                    program = qp.cx(program, first, second)
                else:
                    program = qp.cz(program, first, second)
        return program
    if workload.variant == "ring":
        for qubit in range(workload.qubits):
            program = qp.ry(program, 0.011 * (qubit + 1), qubit)
        for qubit in range(workload.qubits - 1):
            program = qp.cx(program, qubit, qubit + 1)
        program = qp.cz(program, workload.qubits - 1, 0)
        return program
    if workload.variant == "cut-chain":
        half = workload.qubits // 2
        for qubit in range(min(workload.layers, half)):
            program = qp.ry(program, 0.037 * (qubit + 1), qubit)
        for qubit in range(half):
            program = qp.cx(program, qubit, qubit + half)
        for qubit in range(half, workload.qubits - 1):
            program = qp.cx(program, qubit, qubit + 1)
        return program
    raise ValueError(f"unknown MPS workload variant: {workload.variant}")


def _measure_pair(
    program: qp.Program,
    observable: qp.PauliZ,
    backends: tuple[str, str],
    warmups: int,
    iterations: int,
) -> dict[str, list[int]]:
    if backends[0] == backends[1]:
        raise ValueError("paired MPS benchmark requires distinct backends")
    samples = {backend: [] for backend in backends}
    for warmup in range(warmups):
        order = backends if warmup % 2 == 0 else backends[::-1]
        for backend in order:
            qp.expect(program, observable, backend=backend)
    for iteration in range(iterations):
        order = backends if iteration % 2 == 0 else backends[::-1]
        for backend in order:
            start = time.perf_counter_ns()
            qp.expect(program, observable, backend=backend)
            samples[backend].append(time.perf_counter_ns() - start)
    return samples


def _result(
    workload: MpsWorkload,
    program: qp.Program,
    observable: qp.PauliZ,
    backend: str,
    samples: list[int],
) -> dict[str, Any]:
    plan = qp.expectation_plan(program, observable, backend=backend)
    return {
        "workload": workload.name,
        "variant": workload.variant,
        "layers": workload.layers,
        "workload_fingerprint": plan.workload_fingerprint,
        "backend": plan.backend,
        "method": plan.method,
        "qubits": workload.qubits,
        "active_qubits": plan.active_qubits,
        "operations": plan.original_operations,
        "active_operations": plan.active_operations,
        "single_qubit_operations": plan.single_qubit_operations,
        "two_qubit_operations": plan.two_qubit_operations,
        "parameterized_operations": plan.parameterized_operations,
        "non_clifford_operations": plan.non_clifford_operations,
        "compiled_steps": plan.compiled_steps,
        "threads": plan.threads,
        "estimated_state_bytes": plan.estimated_state_bytes,
        "tensor_network_max_bond": plan.tensor_network_max_bond,
        "tensor_network_routed_swaps": plan.tensor_network_routed_swaps,
        "tensor_network_contraction_work": plan.tensor_network_contraction_work,
        "timings_ns": samples,
        "median_ns": statistics.median(samples),
    }


def _pair_evidence(
    baseline_backend: str,
    baseline_samples: list[int],
    adaptive_samples: list[int],
) -> dict[str, Any]:
    baseline_median = statistics.median(baseline_samples)
    adaptive_median = statistics.median(adaptive_samples)
    regret = max(1.0, adaptive_median / baseline_median)
    return {
        "baseline_backend": baseline_backend,
        "baseline_timings_ns": baseline_samples,
        "adaptive_timings_ns": adaptive_samples,
        "baseline_median_ns": baseline_median,
        "adaptive_median_ns": adaptive_median,
        "regret": regret,
    }


def _host() -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "planner_host_fingerprint": qp.planner_host_fingerprint(),
    }


def run_report(*, profile: str = "calibration", warmups: int, iterations: int) -> dict[str, Any]:
    if warmups < 0 or iterations < 1:
        raise ValueError("warmups must be non-negative and iterations must be positive")
    if profile == "policy" and iterations % 2 != 0:
        raise ValueError("MPS policy evidence requires an even iteration count")
    if profile == "smoke":
        workloads = smoke_profile()
    elif profile == "calibration":
        workloads = calibration_profile()
    elif profile == "policy":
        workloads = policy_profile()
    else:
        raise ValueError(f"unknown MPS benchmark profile: {profile}")

    results: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    policy_evidence: list[dict[str, Any]] = []
    for workload in workloads:
        program = _program(workload)
        observable = qp.Z(workload.qubits - 1)
        cpu_plan = qp.expectation_plan(program, observable, backend="native-cpu")
        mps_plan = qp.expectation_plan(program, observable, backend="native-mps")
        adaptive_plan = qp.expectation_plan(
            program, observable, backend="native-adaptive-mps"
        )
        fingerprints = {
            cpu_plan.workload_fingerprint,
            mps_plan.workload_fingerprint,
            adaptive_plan.workload_fingerprint,
        }
        if len(fingerprints) != 1:
            raise RuntimeError("CPU, MPS, and adaptive workload fingerprints diverged")
        cpu_value = qp.expect(program, observable, backend="native-cpu").value
        mps_value = qp.expect(program, observable, backend="native-mps").value
        adaptive_value = qp.expect(
            program, observable, backend="native-adaptive-mps"
        ).value
        mps_error = abs(cpu_value - mps_value)
        adaptive_error = abs(cpu_value - adaptive_value)
        error = max(mps_error, adaptive_error)
        if error > 2e-11:
            raise RuntimeError(f"MPS validation failed for {workload.name}: {error}")
        validations.append(
            {
                "workload": workload.name,
                "cpu_mps_abs_error": mps_error,
                "cpu_adaptive_abs_error": adaptive_error,
                "max_abs_error": error,
            }
        )
        adaptive_backend = "native-adaptive-mps"
        if profile == "policy":
            cpu_pair = _measure_pair(
                program, observable, ("native-cpu", adaptive_backend), warmups, iterations
            )
            mps_pair = _measure_pair(
                program, observable, ("native-mps", adaptive_backend), warmups, iterations
            )
            cpu_evidence = _pair_evidence(
                "native-cpu", cpu_pair["native-cpu"], cpu_pair[adaptive_backend]
            )
            mps_evidence = _pair_evidence(
                "native-mps", mps_pair["native-mps"], mps_pair[adaptive_backend]
            )
            policy_evidence.append(
                {
                    "workload": workload.name,
                    "workload_fingerprint": cpu_plan.workload_fingerprint,
                    "comparisons": [cpu_evidence, mps_evidence],
                    "max_regret": max(cpu_evidence["regret"], mps_evidence["regret"]),
                }
            )
            adaptive_samples = (
                cpu_pair[adaptive_backend]
                if adaptive_plan.method in {"statevector", "statevector-lightcone"}
                else mps_pair[adaptive_backend]
            )
            results.extend(
                (
                    _result(
                        workload, program, observable, "native-cpu", cpu_pair["native-cpu"]
                    ),
                    _result(
                        workload, program, observable, "native-mps", mps_pair["native-mps"]
                    ),
                    _result(
                        workload, program, observable, adaptive_backend, adaptive_samples
                    ),
                )
            )
        else:
            samples_by_backend = _measure_pair(
                program, observable, ("native-cpu", "native-mps"), warmups, iterations
            )
            for backend in ("native-cpu", "native-mps"):
                results.append(
                    _result(
                        workload, program, observable, backend, samples_by_backend[backend]
                    )
                )

    report: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "engine_version": qp.core_version(),
        "workload_version": 1,
        "host": _host(),
        "profile": profile,
        "warmups": warmups,
        "iterations": iterations,
        "validations": validations,
        "results": results,
    }
    if profile == "policy":
        report["policy_version"] = 1
        report["policy_evidence"] = policy_evidence
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect paired QuPy CPU/MPS expectation costs")
    parser.add_argument(
        "--profile", choices=("smoke", "calibration", "policy"), default="calibration"
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_report(profile=args.profile, warmups=args.warmups, iterations=args.iterations)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
