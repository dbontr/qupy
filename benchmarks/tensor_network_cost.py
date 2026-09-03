from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import qupy as qp

_SCHEMA_VERSION = 1
_POLICY_VERSION = 1
_MAX_SEMANTIC_ERROR = 2e-11


@dataclass(frozen=True, slots=True)
class TensorNetworkWorkload:
    name: str
    qubits: int
    variant: str
    query: str
    terms: int


@dataclass(frozen=True, slots=True)
class TensorNetworkWork:
    term_count: int
    contractions: int
    peak_tensor_rank: int
    peak_tensor_bytes: int
    scalar_multiplications: float
    plan_fingerprint: str


def policy_profile() -> tuple[TensorNetworkWorkload, ...]:
    workloads: list[TensorNetworkWorkload] = []
    variants = ("chain", "ring", "star", "ladder")
    term_counts = (1, 2, 3, 4, 6, 8)
    for index, qubits in enumerate(range(8, 17)):
        for query_offset, query in enumerate(("expectation", "batch")):
            variant = variants[(index + query_offset) % len(variants)]
            terms = term_counts[(index * 2 + query_offset) % len(term_counts)]
            workloads.append(
                TensorNetworkWorkload(
                    f"tensor-network-{query}-{variant}-{qubits}",
                    qubits,
                    variant,
                    query,
                    terms,
                )
            )
    return tuple(workloads)


def smoke_profile() -> tuple[TensorNetworkWorkload, ...]:
    return (
        TensorNetworkWorkload("tensor-network-expectation-chain-8", 8, "chain", "expectation", 2),
        TensorNetworkWorkload("tensor-network-batch-ring-9", 9, "ring", "batch", 3),
    )


def _program(workload: TensorNetworkWorkload) -> qp.Program:
    program = qp.Program(workload.qubits)
    for qubit in range(workload.qubits):
        program = qp.ry(program, 0.013 * (qubit + 1), qubit)

    if workload.variant == "chain":
        for qubit in range(workload.qubits - 1):
            program = qp.cx(program, qubit, qubit + 1)
    elif workload.variant == "ring":
        for qubit in range(workload.qubits - 1):
            program = qp.cx(program, qubit, qubit + 1)
        program = qp.cz(program, workload.qubits - 1, 0)
    elif workload.variant == "star":
        for qubit in range(1, workload.qubits):
            program = qp.cx(program, 0, qubit)
    elif workload.variant == "ladder":
        for layer in range(2):
            for qubit in range(layer % 2, workload.qubits - 1, 2):
                program = qp.cz(program, qubit, qubit + 1)
            for qubit in range(workload.qubits):
                program = qp.rz(program, 0.007 * (layer + qubit + 1), qubit)
        for qubit in range(workload.qubits - 1):
            program = qp.cx(program, qubit, qubit + 1)
    else:
        raise ValueError(f"unknown tensor-network workload variant: {workload.variant}")
    return program


def _observable(qubits: int, terms: int, phase: int = 0) -> qp.Observable:
    paulis = (qp.Pauli.X, qp.Pauli.Y, qp.Pauli.Z)
    values: list[qp.PauliTerm] = []
    middle_span = max(qubits - 2, 1)
    for index in range(terms):
        middle = 1 + ((index * 3 + phase) % middle_span)
        factors = [
            qp.PauliFactor(0, paulis[(index + phase) % 3]),
            qp.PauliFactor(qubits - 1, paulis[(index + phase + 1) % 3]),
        ]
        if middle not in {0, qubits - 1}:
            factors.append(qp.PauliFactor(middle, paulis[(index + phase + 2) % 3]))
        coefficient = (-1.0 if index % 2 else 1.0) / float(index + 1)
        values.append(qp.PauliTerm(coefficient, factors))
    return qp.Observable(values)


def _observables(workload: TensorNetworkWorkload) -> list[qp.Observable]:
    left = _observable(workload.qubits, workload.terms)
    if workload.query == "expectation":
        return [left]
    if workload.query == "batch":
        return [left, _observable(workload.qubits, max(1, workload.terms // 2), 1)]
    raise ValueError(f"unknown tensor-network query: {workload.query}")


def _run_query(
    program: qp.Program,
    observables: list[qp.Observable],
    backend: str,
) -> tuple[float, ...]:
    if len(observables) == 1:
        return (qp.expect_observable(program, observables[0], backend=backend).value,)
    result = qp.expect_observables(program, observables, backend=backend)
    return tuple(float(value) for value in result.values)


def _tensor_network_work(
    program: qp.Program,
    observables: list[qp.Observable],
) -> TensorNetworkWork:
    plans = [qp.tensor_network_plan(program, observable) for observable in observables]
    identity = hashlib.sha256()
    for plan in plans:
        identity.update(plan.plan_fingerprint.encode("ascii"))
        identity.update(b"\n")
    return TensorNetworkWork(
        term_count=sum(plan.term_count for plan in plans),
        contractions=sum(plan.contractions for plan in plans),
        peak_tensor_rank=max((plan.peak_tensor_rank for plan in plans), default=0),
        peak_tensor_bytes=max((plan.peak_tensor_bytes for plan in plans), default=0),
        scalar_multiplications=sum(plan.scalar_multiplications for plan in plans),
        plan_fingerprint=identity.hexdigest(),
    )


def _measure_pair(
    program: qp.Program,
    observables: list[qp.Observable],
    *,
    warmups: int,
    iterations: int,
) -> dict[str, list[int]]:
    backends = ("native-cpu", "native-tn")
    samples = {backend: [] for backend in backends}
    for warmup in range(warmups):
        order = backends if warmup % 2 == 0 else backends[::-1]
        for backend in order:
            _run_query(program, observables, backend)
    for iteration in range(iterations):
        order = backends if iteration % 2 == 0 else backends[::-1]
        for backend in order:
            start = time.perf_counter_ns()
            _run_query(program, observables, backend)
            samples[backend].append(time.perf_counter_ns() - start)
    return samples


def _host() -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "planner_host_fingerprint": qp.planner_host_fingerprint(),
    }


def _workload_identity(
    program: qp.Program,
    observables: list[qp.Observable],
    work: TensorNetworkWork,
    query: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(program.fingerprint.encode("ascii"))
    digest.update(b"\n")
    for observable in observables:
        digest.update(observable.fingerprint.encode("ascii"))
        digest.update(b"\n")
    digest.update(work.plan_fingerprint.encode("ascii"))
    digest.update(b"\n")
    digest.update(query.encode("ascii"))
    return digest.hexdigest()


def run_report(*, profile: str, warmups: int, iterations: int) -> dict[str, Any]:
    if warmups < 0 or iterations < 2 or iterations % 2 != 0:
        raise ValueError(
            "tensor-network policy timing requires non-negative warmups and a positive even iteration count"
        )
    workloads = smoke_profile() if profile == "smoke" else policy_profile()
    evidence: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []

    for workload in workloads:
        program = _program(workload)
        observables = _observables(workload)
        work = _tensor_network_work(program, observables)
        cpu_state_plan = qp.plan(program, qp.ResultMode.STATEVECTOR, backend="native-cpu")
        cpu_query_plan = qp.observable_plan(program, observables, backend="native-cpu")
        tn_query_plan = qp.observable_plan(program, observables, backend="native-tn")
        if cpu_query_plan.active_qubits != workload.qubits:
            raise RuntimeError(
                f"tensor-network calibration workload lost its full CPU causal cone: {workload.name}"
            )
        if tn_query_plan.estimated_state_bytes != work.peak_tensor_bytes:
            raise RuntimeError("tensor-network observable plan disagrees with structural preflight")

        cpu_values = _run_query(program, observables, "native-cpu")
        tn_values = _run_query(program, observables, "native-tn")
        semantic_error = max(
            (abs(left - right) for left, right in zip(cpu_values, tn_values, strict=True)),
            default=0.0,
        )
        if semantic_error > _MAX_SEMANTIC_ERROR:
            raise RuntimeError(
                f"tensor-network validation failed for {workload.name}: {semantic_error}"
            )
        validations.append({"workload": workload.name, "max_abs_error": semantic_error})

        samples = _measure_pair(
            program,
            observables,
            warmups=warmups,
            iterations=iterations,
        )
        operation_count = len(program.operations)
        two_qubit_operations = sum(
            1 for operation in program.operations if len(operation.qubits) == 2
        )
        evidence.append(
            {
                "workload": workload.name,
                "fingerprint": _workload_identity(program, observables, work, workload.query),
                "query": workload.query,
                "qubits": workload.qubits,
                "variant": workload.variant,
                "observable_count": len(observables),
                "term_count": work.term_count,
                "operation_count": operation_count,
                "two_qubit_operations": two_qubit_operations,
                "compiled_steps": cpu_state_plan.compiled_steps,
                "threads": cpu_state_plan.threads,
                "cpu_active_qubits": cpu_query_plan.active_qubits,
                "cpu_estimated_state_bytes": cpu_query_plan.estimated_state_bytes,
                "tn_contractions": work.contractions,
                "tn_peak_tensor_rank": work.peak_tensor_rank,
                "tn_peak_tensor_bytes": work.peak_tensor_bytes,
                "tn_scalar_multiplications": work.scalar_multiplications,
                "tn_plan_fingerprint": work.plan_fingerprint,
                "cpu_timings_ns": samples["native-cpu"],
                "tn_timings_ns": samples["native-tn"],
                "cpu_median_ns": statistics.median(samples["native-cpu"]),
                "tn_median_ns": statistics.median(samples["native-tn"]),
            }
        )

    return {
        "schema_version": _SCHEMA_VERSION,
        "policy_version": _POLICY_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "engine_version": qp.core_version(),
        "workload_version": 1,
        "host": _host(),
        "profile": profile,
        "warmups": warmups,
        "iterations": iterations,
        "validations": validations,
        "policy_evidence": evidence,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect paired dense-CPU and exact tensor-network observable costs"
    )
    parser.add_argument("--profile", choices=("smoke", "policy"), default="policy")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=4)
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
