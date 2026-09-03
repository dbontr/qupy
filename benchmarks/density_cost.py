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

import numpy as np

import qupy as qp

_SCHEMA_VERSION = 1
_POLICY_VERSION = 1
_MAX_SEMANTIC_ERROR = 2e-11


@dataclass(frozen=True, slots=True)
class DensityWorkload:
    name: str
    qubits: int
    variant: str
    noise_profile: str


def policy_profile() -> tuple[DensityWorkload, ...]:
    workloads: list[DensityWorkload] = []
    variants = ("chain", "ladder")
    profiles = ("sparse", "mixed", "dense")
    for qubits in range(4, 10):
        for variant in variants:
            for profile in profiles:
                workloads.append(
                    DensityWorkload(
                        f"density-{profile}-{variant}-{qubits}",
                        qubits,
                        variant,
                        profile,
                    )
                )
    return tuple(workloads)


def smoke_profile() -> tuple[DensityWorkload, ...]:
    return (
        DensityWorkload("density-sparse-chain-5", 5, "chain", "sparse"),
        DensityWorkload("density-mixed-ladder-7", 7, "ladder", "mixed"),
    )


def _program(workload: DensityWorkload) -> qp.Program:
    program = qp.Program(workload.qubits)
    for qubit in range(workload.qubits):
        program = qp.ry(program, 0.017 * (qubit + 1), qubit)
        if (qubit + (workload.variant == "ladder")) % 2 == 0:
            program = qp.rz(program, -0.013 * (qubit + 1), qubit)
    for qubit in range(workload.qubits - 1):
        program = qp.cx(program, qubit, qubit + 1)
    if workload.variant == "ladder":
        for qubit in range(0, workload.qubits - 1, 2):
            program = qp.cz(program, qubit, qubit + 1)
        for qubit in range(workload.qubits):
            program = qp.rx(program, 0.009 * (qubit + 1), qubit)
    elif workload.variant != "chain":
        raise ValueError(f"unknown density workload variant: {workload.variant}")
    return program


def _custom_channel(qubit: int) -> qp.NoiseChannel:
    phase = np.exp(0.31j)
    unitary = np.array(
        [[0.0, phase], [phase.conjugate(), 0.0]], dtype=np.complex128
    )
    return qp.kraus_channel(
        qubit,
        [math.sqrt(0.6) * np.eye(2, dtype=np.complex128), math.sqrt(0.4) * unitary],
    )


def _noisy_program(workload: DensityWorkload, program: qp.Program) -> qp.NoisyProgram:
    operations = len(program.operations)
    last = workload.qubits - 1
    if workload.noise_profile == "sparse":
        noise = [qp.NoiseInstruction(operations, qp.amplitude_damping(0, 0.08))]
    elif workload.noise_profile == "mixed":
        noise = [
            qp.NoiseInstruction(max(1, operations // 2), qp.depolarizing(0, 0.06)),
            qp.NoiseInstruction(operations, qp.phase_damping(last, 0.09)),
            qp.NoiseInstruction(operations, _custom_channel(workload.qubits // 2)),
        ]
    elif workload.noise_profile == "dense":
        count = min(operations, workload.qubits + 2)
        noise = [
            qp.NoiseInstruction(index + 1, qp.bit_flip(index % workload.qubits, 0.03))
            for index in range(count)
        ]
    else:
        raise ValueError(f"unknown density noise profile: {workload.noise_profile}")
    return qp.NoisyProgram(program, noise)


def _kraus_count(channel: qp.NoiseChannel) -> int:
    if channel.code in {qp.NoiseChannelCode.DEPOLARIZING, qp.NoiseChannelCode.PAULI}:
        return 4
    if channel.code is qp.NoiseChannelCode.KRAUS:
        return channel.kraus_count
    return 2


def _work(noisy: qp.NoisyProgram) -> dict[str, int]:
    operations = noisy.program.operations
    single = sum(len(operation.qubits) == 1 for operation in operations)
    two = sum(len(operation.qubits) == 2 for operation in operations)
    kraus = sum(_kraus_count(instruction.channel) for instruction in noisy.noise)
    return {
        "single_qubit_operations": single,
        "two_qubit_operations": two,
        "noise_events": len(noisy.noise),
        "kraus_evaluations": kraus,
    }


def _measure_pair(
    noisy: qp.NoisyProgram,
    warmups: int,
    iterations: int,
) -> dict[str, list[int]]:
    backends = ("native-cpu", "native-cuda")
    samples = {backend: [] for backend in backends}
    for warmup in range(warmups):
        order = backends if warmup % 2 == 0 else backends[::-1]
        for backend in order:
            qp.density_matrix(noisy, backend)
    for iteration in range(iterations):
        order = backends if iteration % 2 == 0 else backends[::-1]
        for backend in order:
            start = time.perf_counter_ns()
            qp.density_matrix(noisy, backend)
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


def _complex_token(value: complex) -> str:
    return f"{float(value.real).hex()}:{float(value.imag).hex()}"


def _identity(workload: DensityWorkload, noisy: qp.NoisyProgram, work: dict[str, int]) -> str:
    lines = [
        "qupy-density-workload 1",
        f"program {noisy.program.fingerprint}",
        f"name {workload.name}",
        f"noise-events {work['noise_events']}",
        f"kraus-evaluations {work['kraus_evaluations']}",
    ]
    for instruction in noisy.noise:
        channel = instruction.channel
        fields = [
            str(instruction.after_operation), channel.code.name, str(channel.qubit),
            *(float(value).hex() for value in channel.parameters),
        ]
        if channel.code is qp.NoiseChannelCode.KRAUS:
            fields.append(str(channel.kraus_count))
            fields.extend(_complex_token(value) for value in channel.kraus_operators)
        lines.append("noise " + " ".join(fields))
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()


def run_report(*, profile: str, warmups: int, iterations: int) -> dict[str, Any]:
    if warmups < 0 or iterations < 2 or iterations % 2 != 0:
        raise ValueError(
            "density policy timing requires non-negative warmups and a positive even iteration count"
        )
    if not qp.cuda_available():
        raise RuntimeError(qp.cuda_unavailable_reason())
    workloads = smoke_profile() if profile == "smoke" else policy_profile()
    evidence: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    for workload in workloads:
        program = _program(workload)
        noisy = _noisy_program(workload, program)
        work = _work(noisy)
        cpu = qp.density_matrix(noisy, "native-cpu")
        cuda = qp.density_matrix(noisy, "native-cuda")
        if cpu.values.shape != cuda.values.shape:
            raise RuntimeError(f"density shape mismatch for {workload.name}")
        error = float(np.max(np.abs(cpu.values - cuda.values)))
        if error > _MAX_SEMANTIC_ERROR:
            raise RuntimeError(f"density validation failed for {workload.name}: {error}")
        validations.append({"workload": workload.name, "max_abs_error": error})
        samples = _measure_pair(noisy, warmups, iterations)
        evidence.append(
            {
                "workload": workload.name,
                "fingerprint": _identity(workload, noisy, work),
                "qubits": workload.qubits,
                "variant": workload.variant,
                "noise_profile": workload.noise_profile,
                **work,
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
    parser = argparse.ArgumentParser(description="Collect paired CPU/CUDA noisy-density costs")
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
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
