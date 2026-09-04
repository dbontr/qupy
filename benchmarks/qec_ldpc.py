from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path

import numpy as np
import numpy.typing as npt

import qupy as qp
from benchmarks.qec import (
    DecoderMetrics,
    _decoder_metrics,
    _logical_failures,
    _observable_matrix,
    _package_version,
    _timed_batch,
    reconstruct_syndromes,
)

_SCHEMA_VERSION = 1
_FAMILY = "hypergraph-product-hamming"


@dataclass(frozen=True, slots=True)
class LdpcWorkload:
    name: str
    hamming_rank: int
    physical_error_rate: float
    shots: int
    seed: int

    def __post_init__(self) -> None:
        if isinstance(self.hamming_rank, bool) or not isinstance(self.hamming_rank, int):
            raise TypeError("hamming_rank must be an integer")
        if self.hamming_rank < 3:
            raise ValueError("hamming_rank must be at least three")
        if not math.isfinite(self.physical_error_rate) or not 0.0 < self.physical_error_rate < 1.0:
            raise ValueError("physical error rate must be finite and between zero and one")
        if self.shots < 1:
            raise ValueError("QLDPC benchmark shots must be at least one")
        if self.seed < 0:
            raise ValueError("QLDPC benchmark seed cannot be negative")


@dataclass(frozen=True, slots=True)
class HypergraphProductCode:
    seed_check: npt.NDArray[np.int8]
    hx: npt.NDArray[np.int8]
    hz: npt.NDArray[np.int8]
    logical_z: npt.NDArray[np.int8]

    @property
    def qubit_count(self) -> int:
        return int(self.hx.shape[1])

    @property
    def logical_qubits(self) -> int:
        return int(self.logical_z.shape[0])


@dataclass(frozen=True, slots=True)
class LdpcBenchmarkResult:
    workload: str
    family: str
    hamming_rank: int
    seed_code_length: int
    physical_error_rate: float
    shots: int
    seed: int
    qubit_count: int
    x_check_count: int
    z_check_count: int
    logical_qubits: int
    error_count: int
    qupy_model_fingerprint: str
    qupy_model_conversion_ns: int
    sample_ns: int
    qupy: DecoderMetrics
    ldpc: DecoderMetrics
    prediction_agreement_rate: float
    qupy_syndrome_consistency_rate: float
    ldpc_syndrome_consistency_rate: float
    bp_convergence_rate: float
    osd_usage_rate: float
    mean_bp_iterations: float
    max_bp_iterations: int
    active_error_count: int
    edge_count: int


@dataclass(frozen=True, slots=True)
class LdpcBenchmarkReport:
    schema_version: int
    kind: str
    generated_at: str
    host: dict[str, str]
    warmups: int
    iterations: int
    max_bp_iterations: int
    damping: float
    results: tuple[LdpcBenchmarkResult, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def _binary_matrix(values: npt.ArrayLike, name: str) -> npt.NDArray[np.int8]:
    matrix = np.asarray(values)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if not bool(np.all(np.isin(matrix, (0, 1)))):
        raise ValueError(f"{name} values must be zero or one")
    return np.ascontiguousarray(matrix, dtype=np.int8)


def _gf2_rref(values: npt.ArrayLike) -> tuple[npt.NDArray[np.int8], tuple[int, ...]]:
    matrix = _binary_matrix(values, "GF(2) matrix").copy()
    rows, columns = matrix.shape
    pivot_row = 0
    pivots: list[int] = []
    for column in range(columns):
        candidates = np.flatnonzero(matrix[pivot_row:, column])
        if candidates.size == 0:
            continue
        selected = pivot_row + int(candidates[0])
        if selected != pivot_row:
            matrix[[pivot_row, selected]] = matrix[[selected, pivot_row]]
        for row in range(rows):
            if row != pivot_row and matrix[row, column]:
                matrix[row] ^= matrix[pivot_row]
        pivots.append(column)
        pivot_row += 1
        if pivot_row == rows:
            break
    return matrix, tuple(pivots)


def gf2_rank(values: npt.ArrayLike) -> int:
    return len(_gf2_rref(values)[1])


def _gf2_row_basis(values: npt.ArrayLike) -> npt.NDArray[np.int8]:
    reduced, pivots = _gf2_rref(values)
    return np.ascontiguousarray(reduced[: len(pivots)], dtype=np.int8)


def _gf2_nullspace(values: npt.ArrayLike) -> npt.NDArray[np.int8]:
    matrix = _binary_matrix(values, "GF(2) constraint matrix")
    reduced, pivots = _gf2_rref(matrix)
    pivot_set = set(pivots)
    free_columns = [column for column in range(matrix.shape[1]) if column not in pivot_set]
    basis = np.zeros((len(free_columns), matrix.shape[1]), dtype=np.int8)
    for basis_row, free_column in enumerate(free_columns):
        basis[basis_row, free_column] = 1
        for row, pivot_column in enumerate(pivots):
            basis[basis_row, pivot_column] = reduced[row, free_column]
    return basis


def _quotient_basis(
    kernel_generators: npt.ArrayLike,
    subspace_generators: npt.ArrayLike,
) -> npt.NDArray[np.int8]:
    kernel = _binary_matrix(kernel_generators, "kernel generators")
    subspace = _gf2_row_basis(subspace_generators)
    if kernel.shape[1] != subspace.shape[1]:
        raise ValueError("kernel and subspace generators must have the same width")

    span = subspace.copy()
    span_rank = span.shape[0]
    quotient: list[npt.NDArray[np.int8]] = []
    for candidate in kernel:
        combined = np.vstack((span, candidate))
        rank = gf2_rank(combined)
        if rank == span_rank:
            continue
        quotient.append(candidate.copy())
        span = _gf2_row_basis(combined)
        span_rank = rank
    if not quotient:
        return np.zeros((0, kernel.shape[1]), dtype=np.int8)
    return np.ascontiguousarray(np.stack(quotient), dtype=np.int8)


def hamming_check_matrix(rank: int) -> npt.NDArray[np.int8]:
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise TypeError("rank must be an integer")
    if rank < 2:
        raise ValueError("rank must be at least two")
    columns = (1 << rank) - 1
    matrix = np.zeros((rank, columns), dtype=np.int8)
    for column in range(1, columns + 1):
        for bit in range(rank):
            matrix[bit, column - 1] = (column >> bit) & 1
    if gf2_rank(matrix) != rank:
        raise RuntimeError("constructed Hamming check matrix is not full row rank")
    return matrix


def hypergraph_product_hamming_code(rank: int) -> HypergraphProductCode:
    check = hamming_check_matrix(rank)
    checks, bits = check.shape
    hx = np.concatenate(
        (
            np.kron(check, np.eye(bits, dtype=np.int8)),
            np.kron(np.eye(checks, dtype=np.int8), check.T),
        ),
        axis=1,
    ).astype(np.int8, copy=False)
    hz = np.concatenate(
        (
            np.kron(np.eye(bits, dtype=np.int8), check),
            np.kron(check.T, np.eye(checks, dtype=np.int8)),
        ),
        axis=1,
    ).astype(np.int8, copy=False)
    if bool(np.any((hx.astype(np.uint64) @ hz.T.astype(np.uint64)) & 1)):
        raise RuntimeError("hypergraph-product CSS checks do not commute")

    logical_z = _quotient_basis(_gf2_nullspace(hx), hz)
    expected_logicals = hx.shape[1] - gf2_rank(hx) - gf2_rank(hz)
    if logical_z.shape[0] != expected_logicals:
        raise RuntimeError("failed to construct a complete logical-Z quotient basis")
    if logical_z.shape[0] == 0:
        raise RuntimeError("hypergraph-product benchmark code has no logical qubits")
    if bool(np.any((hx.astype(np.uint64) @ logical_z.T.astype(np.uint64)) & 1)):
        raise RuntimeError("logical-Z basis does not commute with X checks")
    return HypergraphProductCode(check, hx, hz, logical_z)


def ldpc_workloads_for_profile(profile: str) -> tuple[LdpcWorkload, ...]:
    if profile == "smoke":
        return (
            LdpcWorkload(
                name="hgp-hamming-r3-p0.02",
                hamming_rank=3,
                physical_error_rate=0.02,
                shots=128,
                seed=1703,
            ),
        )
    if profile == "standard":
        workloads: list[LdpcWorkload] = []
        seed = 2700
        for rank in (3, 4):
            for error_rate in (0.01, 0.03, 0.05):
                workloads.append(
                    LdpcWorkload(
                        name=f"hgp-hamming-r{rank}-p{error_rate:g}",
                        hamming_rank=rank,
                        physical_error_rate=error_rate,
                        shots=5000,
                        seed=seed,
                    )
                )
                seed += 1
        return tuple(workloads)
    raise ValueError(f"unknown QLDPC benchmark profile: {profile}")


def _host_metadata() -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "qupy_core": qp.core_version(),
        "ldpc": _package_version("ldpc"),
    }


def _binary_product(left: npt.ArrayLike, right: npt.ArrayLike) -> npt.NDArray[np.int8]:
    first = np.asarray(left, dtype=np.uint64)
    second = np.asarray(right, dtype=np.uint64)
    return np.ascontiguousarray((first @ second) & 1, dtype=np.int8)


def _detector_model(
    code: HypergraphProductCode,
    probability: float,
) -> qp.DetectorModel:
    errors: list[qp.DetectorError] = []
    for qubit in range(code.qubit_count):
        detectors = np.flatnonzero(code.hz[:, qubit]).astype(int).tolist()
        observables = np.flatnonzero(code.logical_z[:, qubit]).astype(int).tolist()
        errors.append(qp.DetectorError(probability, detectors, observables))
    return qp.DetectorModel(code.hz.shape[0], code.logical_qubits, errors)


def _sample_code_capacity(
    code: HypergraphProductCode,
    *,
    probability: float,
    shots: int,
    seed: int,
) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.int8], npt.NDArray[np.int8]]:
    rng = np.random.default_rng(seed)
    errors = np.ascontiguousarray(
        rng.random((shots, code.qubit_count)) < probability,
        dtype=np.int8,
    )
    syndromes = _binary_product(errors, code.hz.T)
    observables = _binary_product(errors, code.logical_z.T)
    return errors, syndromes, observables


def _ldpc_decode_batch(decoder: object, syndromes: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
    corrections: list[npt.NDArray[np.int8]] = []
    for syndrome in syndromes:
        correction = np.asarray(decoder.decode(syndrome), dtype=np.int8)
        if correction.ndim != 1:
            raise RuntimeError("LDPC comparator returned a non-vector correction")
        corrections.append(np.ascontiguousarray(correction, dtype=np.int8))
    if not corrections:
        return np.zeros((0, 0), dtype=np.int8)
    return np.ascontiguousarray(np.stack(corrections), dtype=np.int8)


def run_ldpc_workload(
    workload: LdpcWorkload,
    *,
    warmups: int = 1,
    iterations: int = 5,
    max_bp_iterations: int = 50,
    damping: float = 0.0,
) -> LdpcBenchmarkResult:
    if warmups < 0:
        raise ValueError("warmups cannot be negative")
    if iterations < 1:
        raise ValueError("iterations must be at least one")
    if max_bp_iterations < 0:
        raise ValueError("max BP iterations cannot be negative")
    if not math.isfinite(damping) or not 0.0 <= damping < 1.0:
        raise ValueError("damping must be finite and in [0, 1)")

    try:
        ldpc_module = import_module("ldpc.bposd_decoder")
        decoder_type = ldpc_module.BpOsdDecoder
    except (ImportError, AttributeError) as error:
        raise RuntimeError("QLDPC benchmarks require the optional ldpc package") from error

    code = hypergraph_product_hamming_code(workload.hamming_rank)
    started = time.perf_counter_ns()
    model = _detector_model(code, workload.physical_error_rate)
    model_conversion_ns = time.perf_counter_ns() - started
    if len(model.errors) != code.qubit_count:
        raise RuntimeError("QLDPC detector model changed the physical-error count")

    started = time.perf_counter_ns()
    _, syndromes, actual_observables = _sample_code_capacity(
        code,
        probability=workload.physical_error_rate,
        shots=workload.shots,
        seed=workload.seed,
    )
    sample_ns = time.perf_counter_ns() - started

    started = time.perf_counter_ns()
    qupy_decoder = qp.BpOsdDecoder(
        model,
        max_iterations=max_bp_iterations,
        damping=damping,
    )
    qupy_setup_ns = time.perf_counter_ns() - started

    started = time.perf_counter_ns()
    reference_decoder = decoder_type(
        code.hz,
        error_rate=workload.physical_error_rate,
        max_iter=max_bp_iterations,
        bp_method="product_sum",
        osd_method="osd0",
        osd_order=0,
    )
    ldpc_setup_ns = time.perf_counter_ns() - started

    qupy_result_object, qupy_timings = _timed_batch(
        lambda: qupy_decoder.decode_batch(syndromes),
        warmups=warmups,
        iterations=iterations,
    )
    qupy_result = qupy_result_object
    qupy_observables = _observable_matrix(
        qupy_result.observables,
        shots=workload.shots,
        observable_count=code.logical_qubits,
        name="QuPy QLDPC predicted observables",
    )

    ldpc_result_object, ldpc_timings = _timed_batch(
        lambda: _ldpc_decode_batch(reference_decoder, syndromes),
        warmups=warmups,
        iterations=iterations,
    )
    ldpc_corrections = np.asarray(ldpc_result_object, dtype=np.int8)
    if ldpc_corrections.shape != (workload.shots, code.qubit_count):
        raise RuntimeError("LDPC comparator corrections have an unexpected shape")
    ldpc_observables = _binary_product(ldpc_corrections, code.logical_z.T)

    qupy_reconstructed = reconstruct_syndromes(model, qupy_result.corrections)
    qupy_consistency = float(np.mean(np.all(qupy_reconstructed == syndromes, axis=1)))
    if qupy_consistency != 1.0:
        raise AssertionError("QuPy returned a QLDPC correction that does not reproduce its syndrome")

    ldpc_reconstructed = _binary_product(ldpc_corrections, code.hz.T)
    ldpc_consistency = float(np.mean(np.all(ldpc_reconstructed == syndromes, axis=1)))
    if ldpc_consistency != 1.0:
        raise AssertionError("LDPC comparator returned a correction that does not reproduce its syndrome")

    qupy_failures = _logical_failures(qupy_observables, actual_observables)
    ldpc_failures = _logical_failures(ldpc_observables, actual_observables)
    prediction_agreement_rate = float(
        np.mean(np.all(qupy_observables == ldpc_observables, axis=1))
    )

    bp_converged = np.asarray(qupy_result.bp_converged, dtype=np.int8)
    osd_used = np.asarray(qupy_result.osd_used, dtype=np.int8)
    bp_iterations = np.asarray(qupy_result.iterations, dtype=np.uint64)

    return LdpcBenchmarkResult(
        workload=workload.name,
        family=_FAMILY,
        hamming_rank=workload.hamming_rank,
        seed_code_length=code.seed_check.shape[1],
        physical_error_rate=workload.physical_error_rate,
        shots=workload.shots,
        seed=workload.seed,
        qubit_count=code.qubit_count,
        x_check_count=code.hx.shape[0],
        z_check_count=code.hz.shape[0],
        logical_qubits=code.logical_qubits,
        error_count=len(model.errors),
        qupy_model_fingerprint=model.fingerprint,
        qupy_model_conversion_ns=model_conversion_ns,
        sample_ns=sample_ns,
        qupy=_decoder_metrics(
            engine="qupy",
            engine_version=qp.core_version(),
            method="belief-propagation-osd0",
            setup_ns=qupy_setup_ns,
            timings_ns=qupy_timings,
            failures=qupy_failures,
            shots=workload.shots,
        ),
        ldpc=_decoder_metrics(
            engine="ldpc",
            engine_version=_package_version("ldpc"),
            method="bp-osd0-product-sum",
            setup_ns=ldpc_setup_ns,
            timings_ns=ldpc_timings,
            failures=ldpc_failures,
            shots=workload.shots,
        ),
        prediction_agreement_rate=prediction_agreement_rate,
        qupy_syndrome_consistency_rate=qupy_consistency,
        ldpc_syndrome_consistency_rate=ldpc_consistency,
        bp_convergence_rate=float(np.mean(bp_converged)),
        osd_usage_rate=float(np.mean(osd_used)),
        mean_bp_iterations=float(np.mean(bp_iterations)),
        max_bp_iterations=int(np.max(bp_iterations, initial=0)),
        active_error_count=qupy_decoder.active_error_count,
        edge_count=qupy_decoder.edge_count,
    )


def run_ldpc_suite(
    workloads: tuple[LdpcWorkload, ...],
    *,
    warmups: int = 1,
    iterations: int = 5,
    max_bp_iterations: int = 50,
    damping: float = 0.0,
) -> LdpcBenchmarkReport:
    results = tuple(
        run_ldpc_workload(
            workload,
            warmups=warmups,
            iterations=iterations,
            max_bp_iterations=max_bp_iterations,
            damping=damping,
        )
        for workload in workloads
    )
    return LdpcBenchmarkReport(
        schema_version=_SCHEMA_VERSION,
        kind="qec-ldpc-decoder-evidence",
        generated_at=datetime.now(UTC).isoformat(),
        host=_host_metadata(),
        warmups=warmups,
        iterations=iterations,
        max_bp_iterations=max_bp_iterations,
        damping=damping,
        results=results,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare QuPy BP+OSD-0 with LDPC BP+OSD-0 on hypergraph-product CSS codes"
    )
    parser.add_argument("--profile", choices=("smoke", "standard"), default="smoke")
    parser.add_argument("--shots", type=int, default=None, help="override shots for every workload")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--max-bp-iterations", type=int, default=50)
    parser.add_argument("--damping", type=float, default=0.0)
    parser.add_argument("--output", default="-", help="JSON output path or - for stdout")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    workloads = ldpc_workloads_for_profile(args.profile)
    if args.shots is not None:
        if args.shots < 1:
            raise SystemExit("shots must be at least one")
        workloads = tuple(replace(workload, shots=args.shots) for workload in workloads)
    report = run_ldpc_suite(
        workloads,
        warmups=args.warmups,
        iterations=args.iterations,
        max_bp_iterations=args.max_bp_iterations,
        damping=args.damping,
    )
    payload = report.to_json() + "\n"
    if args.output == "-":
        sys.stdout.write(payload)
    else:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
