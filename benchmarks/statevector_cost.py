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

import numpy as np

import qupy as qp

_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class StatevectorWorkload:
    name: str
    qubits: int
    variant: str


def calibration_profile() -> tuple[StatevectorWorkload, ...]:
    qubits = (14, 16, 18, 19, 20, 21, 22, 23, 24)
    return tuple(
        StatevectorWorkload(f"statevector-{variant}-{count}", count, variant)
        for count in qubits
        for variant in ("chain", "brick")
    )


def _program(workload: StatevectorWorkload) -> qp.Program:
    program = qp.Program(workload.qubits)
    if workload.variant == "chain":
        program = qp.h(program, 0)
        for qubit in range(1, workload.qubits):
            program = qp.cx(program, qubit - 1, qubit)
        for qubit in range(workload.qubits):
            program = qp.ry(program, 0.013 * (qubit + 1), qubit)
        for qubit in range(0, workload.qubits - 1, 2):
            program = qp.cz(program, qubit, qubit + 1)
        return program
    if workload.variant == "brick":
        for qubit in range(workload.qubits):
            program = qp.rx(program, 0.009 * (qubit + 1), qubit)
        for parity in (0, 1):
            for qubit in range(parity, workload.qubits - 1, 2):
                program = qp.cx(program, qubit, qubit + 1)
        for qubit in range(workload.qubits):
            program = qp.rz(program, -0.007 * (qubit + 1), qubit)
        return program
    raise ValueError(f"unknown state-vector workload variant: {workload.variant}")


def _measure(program: qp.Program, backend: str, warmups: int, iterations: int) -> list[int]:
    for _ in range(warmups):
        qp.statevector(program, backend=backend)
    samples: list[int] = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        qp.statevector(program, backend=backend)
        samples.append(time.perf_counter_ns() - start)
    return samples


def _result(
    workload: StatevectorWorkload,
    program: qp.Program,
    backend: str,
    samples: list[int],
) -> dict[str, Any]:
    plan = qp.plan(program, qp.ResultMode.STATEVECTOR, backend=backend)
    return {
        "workload": workload.name,
        "variant": workload.variant,
        "workload_fingerprint": plan.workload_fingerprint,
        "backend": plan.backend,
        "method": plan.method,
        "qubits": workload.qubits,
        "operations": plan.original_operations,
        "single_qubit_operations": plan.single_qubit_operations,
        "two_qubit_operations": plan.two_qubit_operations,
        "compiled_steps": plan.compiled_steps,
        "threads": plan.threads,
        "estimated_state_bytes": plan.estimated_state_bytes,
        "timings_ns": samples,
        "median_ns": statistics.median(samples),
    }


def _host() -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "planner_host_fingerprint": qp.planner_host_fingerprint(),
        "planner_cuda_host_fingerprint": qp.planner_cuda_host_fingerprint(),
        "cuda_device": qp.cuda_device_name(),
    }


def run_report(*, warmups: int, iterations: int) -> dict[str, Any]:
    if not qp.cuda_available():
        raise RuntimeError(qp.cuda_unavailable_reason() or "CUDA is not available")
    if warmups < 0 or iterations < 1:
        raise ValueError("warmups must be non-negative and iterations must be positive")

    results: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    for workload in calibration_profile():
        program = _program(workload)
        cpu_plan = qp.plan(program, qp.ResultMode.STATEVECTOR, backend="native-cpu")
        cuda_plan = qp.plan(program, qp.ResultMode.STATEVECTOR, backend="native-cuda")
        if cpu_plan.workload_fingerprint != cuda_plan.workload_fingerprint:
            raise RuntimeError("CPU and CUDA workload fingerprints diverged")
        cpu_state = qp.statevector(program, backend="native-cpu").values
        cuda_state = qp.statevector(program, backend="native-cuda").values
        max_error = float(np.max(np.abs(cpu_state - cuda_state)))
        if max_error > 2e-12:
            raise RuntimeError(f"state-vector validation failed for {workload.name}: {max_error}")
        validations.append({"workload": workload.name, "max_abs_error": max_error})
        for backend in ("native-cpu", "native-cuda"):
            samples = _measure(program, backend, warmups, iterations)
            results.append(_result(workload, program, backend, samples))

    return {
        "schema_version": _SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "engine_version": qp.core_version(),
        "workload_version": 1,
        "host": _host(),
        "warmups": warmups,
        "iterations": iterations,
        "validations": validations,
        "results": results,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect paired QuPy CPU/CUDA state-vector costs")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_report(warmups=args.warmups, iterations=args.iterations)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
