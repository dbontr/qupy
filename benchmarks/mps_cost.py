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


def calibration_profile() -> tuple[MpsWorkload, ...]:
    chain = tuple(MpsWorkload(f"mps-chain-{q}", q, "chain") for q in range(10, 25, 2))
    star = tuple(MpsWorkload(f"mps-star-{q}", q, "star") for q in range(8, 19, 2))
    ladder = tuple(MpsWorkload(f"mps-ladder-{q}", q, "ladder") for q in range(10, 19, 2))
    return chain + star + ladder


def smoke_profile() -> tuple[MpsWorkload, ...]:
    return (
        MpsWorkload("mps-chain-10", 10, "chain"),
        MpsWorkload("mps-star-8", 8, "star"),
        MpsWorkload("mps-ladder-10", 10, "ladder"),
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
        for layer in range(4):
            parity = layer % 2
            for qubit in range(parity, workload.qubits - 1, 2):
                program = qp.cz(program, qubit, qubit + 1)
            for qubit in range(workload.qubits):
                program = qp.rz(program, 0.007 * (layer + qubit + 1), qubit)
        for qubit in range(workload.qubits - 1):
            program = qp.cx(program, qubit, qubit + 1)
        return program
    raise ValueError(f"unknown MPS workload variant: {workload.variant}")


def _measure(
    program: qp.Program,
    observable: qp.PauliZ,
    backend: str,
    warmups: int,
    iterations: int,
) -> list[int]:
    for _ in range(warmups):
        qp.expect(program, observable, backend=backend)
    samples: list[int] = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        qp.expect(program, observable, backend=backend)
        samples.append(time.perf_counter_ns() - start)
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
        "workload_fingerprint": plan.workload_fingerprint,
        "backend": plan.backend,
        "method": plan.method,
        "qubits": workload.qubits,
        "active_qubits": plan.active_qubits,
        "operations": plan.original_operations,
        "active_operations": plan.active_operations,
        "single_qubit_operations": plan.single_qubit_operations,
        "two_qubit_operations": plan.two_qubit_operations,
        "compiled_steps": plan.compiled_steps,
        "threads": plan.threads,
        "estimated_state_bytes": plan.estimated_state_bytes,
        "tensor_network_max_bond": plan.tensor_network_max_bond,
        "tensor_network_routed_swaps": plan.tensor_network_routed_swaps,
        "tensor_network_contraction_work": plan.tensor_network_contraction_work,
        "timings_ns": samples,
        "median_ns": statistics.median(samples),
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
    if profile == "smoke":
        workloads = smoke_profile()
    elif profile == "calibration":
        workloads = calibration_profile()
    else:
        raise ValueError(f"unknown MPS benchmark profile: {profile}")

    results: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    for workload in workloads:
        program = _program(workload)
        observable = qp.Z(workload.qubits - 1)
        cpu_plan = qp.expectation_plan(program, observable, backend="native-cpu")
        mps_plan = qp.expectation_plan(program, observable, backend="native-mps")
        if cpu_plan.workload_fingerprint != mps_plan.workload_fingerprint:
            raise RuntimeError("CPU and MPS workload fingerprints diverged")
        cpu_value = qp.expect(program, observable, backend="native-cpu").value
        mps_value = qp.expect(program, observable, backend="native-mps").value
        error = abs(cpu_value - mps_value)
        if error > 2e-11:
            raise RuntimeError(f"MPS validation failed for {workload.name}: {error}")
        validations.append({"workload": workload.name, "max_abs_error": error})
        for backend in ("native-cpu", "native-mps"):
            samples = _measure(program, observable, backend, warmups, iterations)
            results.append(_result(workload, program, observable, backend, samples))

    return {
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect paired QuPy CPU/MPS expectation costs")
    parser.add_argument("--profile", choices=("smoke", "calibration"), default="calibration")
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
