from __future__ import annotations

import argparse
import hashlib
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
_POLICY_VERSION = 1


@dataclass(frozen=True, slots=True)
class ObservableWorkload:
    name: str
    qubits: int
    variant: str
    query: str
    terms: int


@dataclass(frozen=True, slots=True)
class QueryWork:
    term_evaluations: int
    state_passes: int


def policy_profile() -> tuple[ObservableWorkload, ...]:
    workloads: list[ObservableWorkload] = []
    variants = ("chain", "ladder", "ring", "star")
    queries = ("expectation", "batch", "variance", "covariance")
    for index, qubits in enumerate(range(10, 19)):
        for offset, query in enumerate(queries):
            variant = variants[(index + offset) % len(variants)]
            terms = (1, 2, 3, 5, 8, 12, 16)[(index + 2 * offset) % 7]
            workloads.append(ObservableWorkload(f"observable-{query}-{variant}-{qubits}", qubits, variant, query, terms))
    return tuple(workloads)


def smoke_profile() -> tuple[ObservableWorkload, ...]:
    return (
        ObservableWorkload("observable-expectation-chain-12", 12, "chain", "expectation", 3),
        ObservableWorkload("observable-variance-ladder-14", 14, "ladder", "variance", 4),
    )


def _program(workload: ObservableWorkload) -> qp.Program:
    program = qp.Program(workload.qubits)
    for qubit in range(workload.qubits):
        program = qp.ry(program, 0.017 * (qubit + 1), qubit)
    if workload.variant == "chain":
        for qubit in range(workload.qubits - 1):
            program = qp.cx(program, qubit, qubit + 1)
    elif workload.variant == "star":
        for qubit in range(1, workload.qubits):
            program = qp.cx(program, 0, qubit)
    elif workload.variant == "ring":
        for qubit in range(workload.qubits - 1):
            program = qp.cx(program, qubit, qubit + 1)
        program = qp.cz(program, workload.qubits - 1, 0)
    elif workload.variant == "ladder":
        for layer in range(3):
            for qubit in range(layer % 2, workload.qubits - 1, 2):
                program = qp.cz(program, qubit, qubit + 1)
            for qubit in range(workload.qubits):
                program = qp.rz(program, 0.009 * (layer + qubit + 1), qubit)
        for qubit in range(workload.qubits - 1):
            program = qp.cx(program, qubit, qubit + 1)
    else:
        raise ValueError(f"unknown observable workload variant: {workload.variant}")
    return program


def _observable(qubits: int, terms: int, phase: int = 0) -> qp.Observable:
    paulis = (qp.Pauli.X, qp.Pauli.Y, qp.Pauli.Z)
    values: list[qp.PauliTerm] = []
    for index in range(terms):
        middle = 1 + ((index + phase) % max(qubits - 2, 1))
        factors = [
            qp.PauliFactor(0, paulis[(index + phase) % 3]),
            qp.PauliFactor(qubits - 1, paulis[(index + phase + 1) % 3]),
        ]
        if middle not in {0, qubits - 1}:
            factors.append(qp.PauliFactor(middle, paulis[(index + phase + 2) % 3]))
        coefficient = (-1.0 if index % 2 else 1.0) / (index + 1)
        values.append(qp.PauliTerm(coefficient, factors))
    return qp.Observable(values)


def _multiply_pauli(left: qp.Pauli, right: qp.Pauli) -> qp.Pauli:
    if left == qp.Pauli.I:
        return right
    if right == qp.Pauli.I:
        return left
    if left == right:
        return qp.Pauli.I
    if {left, right} == {qp.Pauli.X, qp.Pauli.Y}:
        return qp.Pauli.Z
    if {left, right} == {qp.Pauli.Y, qp.Pauli.Z}:
        return qp.Pauli.X
    return qp.Pauli.Y


def _product_factors(left: qp.PauliTerm, right: qp.PauliTerm) -> list[qp.PauliFactor]:
    factors = {factor.qubit: factor.pauli for factor in left.factors}
    for factor in right.factors:
        product = _multiply_pauli(factors.get(factor.qubit, qp.Pauli.I), factor.pauli)
        if product == qp.Pauli.I:
            factors.pop(factor.qubit, None)
        else:
            factors[factor.qubit] = product
    return [qp.PauliFactor(qubit, pauli) for qubit, pauli in sorted(factors.items())]


def _cuda_mask_key(factors: list[qp.PauliFactor]) -> tuple[int, int, int]:
    flip_mask = 0
    sign_mask = 0
    y_phase = 0
    for factor in factors:
        bit = 1 << factor.qubit
        if factor.pauli == qp.Pauli.X:
            flip_mask |= bit
        elif factor.pauli == qp.Pauli.Y:
            flip_mask |= bit
            sign_mask |= bit
            y_phase = (y_phase + 1) & 3
        elif factor.pauli == qp.Pauli.Z:
            sign_mask |= bit
    return flip_mask, sign_mask, y_phase


def _cuda_query_evaluations(query: str, values: list[qp.Observable]) -> int:
    masks = {_cuda_mask_key(term.factors) for value in values for term in value.terms}
    if query == "variance":
        value = values[0]
        masks.update(
            _cuda_mask_key(_product_factors(left, right))
            for left in value.terms
            for right in value.terms
        )
    elif query == "covariance":
        masks.update(
            _cuda_mask_key(_product_factors(left, right))
            for left in values[0].terms
            for right in values[1].terms
        )
    return len(masks)


def _query_data(workload: ObservableWorkload) -> tuple[list[qp.Observable], QueryWork, QueryWork]:
    left = _observable(workload.qubits, workload.terms)
    if workload.query == "expectation":
        values = [left]
        cpu_work = QueryWork(workload.terms, 0)
    elif workload.query == "batch":
        right_terms = max(1, workload.terms // 2)
        values = [left, _observable(workload.qubits, right_terms, 1)]
        cpu_work = QueryWork(workload.terms + right_terms, 0)
    elif workload.query == "variance":
        values = [left]
        cpu_work = QueryWork(2 * workload.terms, 1)
    elif workload.query == "covariance":
        right_terms = max(1, workload.terms // 2)
        values = [left, _observable(workload.qubits, right_terms, 2)]
        cpu_work = QueryWork(2 * (workload.terms + right_terms), 1)
    else:
        raise ValueError(f"unknown observable query: {workload.query}")
    return values, cpu_work, QueryWork(_cuda_query_evaluations(workload.query, values), 0)


def _run_query(program: qp.Program, workload: ObservableWorkload, values: list[qp.Observable], backend: str) -> tuple[float, ...]:
    if workload.query == "expectation":
        return (qp.expect_observable(program, values[0], backend=backend).value,)
    if workload.query == "batch":
        return tuple(float(value) for value in qp.expect_observables(program, values, backend=backend).values)
    if workload.query == "variance":
        return (qp.variance_observable(program, values[0], backend=backend).value,)
    if workload.query == "covariance":
        return (qp.covariance(program, values[0], values[1], backend=backend).value,)
    raise ValueError(f"unknown observable query: {workload.query}")


def _measure_pair(program: qp.Program, workload: ObservableWorkload, values: list[qp.Observable], warmups: int, iterations: int) -> dict[str, list[int]]:
    backends = ("native-cpu", "native-cuda")
    samples = {backend: [] for backend in backends}
    for warmup in range(warmups):
        for backend in (backends if warmup % 2 == 0 else backends[::-1]):
            _run_query(program, workload, values, backend)
    for iteration in range(iterations):
        for backend in (backends if iteration % 2 == 0 else backends[::-1]):
            start = time.perf_counter_ns()
            _run_query(program, workload, values, backend)
            samples[backend].append(time.perf_counter_ns() - start)
    return samples


def _host() -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "planner_host_fingerprint": qp.planner_host_fingerprint(),
        "planner_cuda_host_fingerprint": qp.planner_cuda_host_fingerprint(),
        "cuda_device": qp.cuda_device_name(),
    }


def run_report(*, profile: str, warmups: int, iterations: int) -> dict[str, Any]:
    if not qp.cuda_available():
        raise RuntimeError(qp.cuda_unavailable_reason())
    if warmups < 0 or iterations < 2 or iterations % 2 != 0:
        raise ValueError("observable policy timing requires non-negative warmups and positive even iterations")
    workloads = smoke_profile() if profile == "smoke" else policy_profile()
    evidence: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    for workload in workloads:
        program = _program(workload)
        observables, cpu_work, cuda_work = _query_data(workload)
        cpu_plan = qp.plan(program, qp.ResultMode.STATEVECTOR, backend="native-cpu")
        cuda_plan = qp.plan(program, qp.ResultMode.STATEVECTOR, backend="native-cuda")
        query_plan = qp.observable_plan(program, observables, backend="native-cpu")
        if cpu_plan.workload_fingerprint != cuda_plan.workload_fingerprint:
            raise RuntimeError("CPU and CUDA observable workload fingerprints diverged")
        if query_plan.active_qubits != workload.qubits:
            raise RuntimeError("observable calibration workload did not preserve its full causal cone")
        cpu_value = _run_query(program, workload, observables, "native-cpu")
        cuda_value = _run_query(program, workload, observables, "native-cuda")
        error = max((abs(a - b) for a, b in zip(cpu_value, cuda_value, strict=True)), default=0.0)
        if error > 2e-11:
            raise RuntimeError(f"observable validation failed for {workload.name}: {error}")
        validations.append({"workload": workload.name, "max_abs_error": error})
        samples = _measure_pair(program, workload, observables, warmups, iterations)
        identity = hashlib.sha256(f"{cpu_plan.workload_fingerprint}|{query_plan.query_fingerprint}|{workload.query}".encode()).hexdigest()
        evidence.append(
            {
                "workload": workload.name,
                "fingerprint": identity,
                "query": workload.query,
                "qubits": workload.qubits,
                "variant": workload.variant,
                "terms": workload.terms,
                "cpu_term_evaluations": cpu_work.term_evaluations,
                "cpu_state_passes": cpu_work.state_passes,
                "cuda_term_evaluations": cuda_work.term_evaluations,
                "cuda_state_passes": cuda_work.state_passes,
                "single_qubit_operations": cpu_plan.single_qubit_operations,
                "two_qubit_operations": cpu_plan.two_qubit_operations,
                "compiled_steps": cpu_plan.compiled_steps,
                "threads": cpu_plan.threads,
                "cpu_timings_ns": samples["native-cpu"],
                "cuda_timings_ns": samples["native-cuda"],
                "cpu_median_ns": statistics.median(samples["native-cpu"]),
                "cuda_median_ns": statistics.median(samples["native-cuda"]),
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
    parser = argparse.ArgumentParser(description="Collect paired CPU/CUDA rich-observable costs")
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
