from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import numpy.typing as npt

import qupy as qp

_SCHEMA_VERSION = 1
_SUPPORTED_TASKS = (
    "surface_code:rotated_memory_x",
    "surface_code:rotated_memory_z",
)


@dataclass(frozen=True, slots=True)
class QecWorkload:
    name: str
    task: str
    distance: int
    rounds: int
    physical_error_rate: float
    shots: int
    seed: int

    def __post_init__(self) -> None:
        if self.task not in _SUPPORTED_TASKS:
            raise ValueError(f"unsupported QEC benchmark task: {self.task}")
        if self.distance < 3 or self.distance % 2 == 0:
            raise ValueError("surface-code distance must be an odd integer of at least three")
        if self.rounds < 1:
            raise ValueError("surface-code rounds must be at least one")
        if not math.isfinite(self.physical_error_rate) or not 0.0 < self.physical_error_rate < 1.0:
            raise ValueError("physical error rate must be finite and between zero and one")
        if self.shots < 1:
            raise ValueError("QEC benchmark shots must be at least one")
        if self.seed < 0:
            raise ValueError("QEC benchmark seed cannot be negative")


@dataclass(frozen=True, slots=True)
class DecoderMetrics:
    engine: str
    engine_version: str
    method: str
    setup_ns: int
    timings_ns: tuple[int, ...]
    median_ns: float
    min_ns: int
    max_ns: int
    median_shots_per_second: float
    logical_failures: int
    logical_failure_rate: float
    logical_failure_interval_95: tuple[float, float]


@dataclass(frozen=True, slots=True)
class QecBenchmarkResult:
    workload: str
    task: str
    distance: int
    rounds: int
    physical_error_rate: float
    shots: int
    seed: int
    detector_count: int
    observable_count: int
    error_count: int
    qupy_model_fingerprint: str
    qupy_model_conversion_ns: int
    sample_ns: int
    qupy: DecoderMetrics
    pymatching: DecoderMetrics
    prediction_agreement_rate: float
    syndrome_consistency_rate: float
    bp_convergence_rate: float
    osd_usage_rate: float
    mean_bp_iterations: float
    max_bp_iterations: int
    active_error_count: int
    edge_count: int


@dataclass(frozen=True, slots=True)
class QecBenchmarkReport:
    schema_version: int
    kind: str
    generated_at: str
    host: dict[str, str]
    warmups: int
    iterations: int
    max_bp_iterations: int
    damping: float
    results: tuple[QecBenchmarkResult, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"


def qec_workloads_for_profile(profile: str) -> tuple[QecWorkload, ...]:
    if profile == "smoke":
        return (
            QecWorkload(
                name="rotated-memory-x-d3-p0.002",
                task="surface_code:rotated_memory_x",
                distance=3,
                rounds=3,
                physical_error_rate=0.002,
                shots=256,
                seed=113,
            ),
        )
    if profile == "standard":
        workloads: list[QecWorkload] = []
        seed = 1000
        for task, task_name in (
            ("surface_code:rotated_memory_x", "rotated-memory-x"),
            ("surface_code:rotated_memory_z", "rotated-memory-z"),
        ):
            for error_rate in (0.001, 0.005):
                for distance in (3, 5, 7):
                    workloads.append(
                        QecWorkload(
                            name=f"{task_name}-d{distance}-p{error_rate:g}",
                            task=task,
                            distance=distance,
                            rounds=distance,
                            physical_error_rate=error_rate,
                            shots=5000,
                            seed=seed,
                        )
                    )
                    seed += 1
        return tuple(workloads)
    raise ValueError(f"unknown QEC benchmark profile: {profile}")


def _host_metadata() -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "qupy_core": qp.core_version(),
        "stim": _package_version("stim"),
        "pymatching": _package_version("pymatching"),
    }


def _parity_support(values: list[int]) -> list[int]:
    odd: set[int] = set()
    for value in values:
        if value in odd:
            odd.remove(value)
        else:
            odd.add(value)
    return sorted(odd)


def _qupy_model_from_stim(dem: object) -> qp.DetectorModel:
    flattened = dem.flattened()
    errors: list[qp.DetectorError] = []
    for instruction in flattened:
        instruction_type = instruction.type
        if instruction_type in {"detector", "logical_observable"}:
            continue
        if instruction_type != "error":
            raise ValueError(f"unsupported flattened Stim DEM instruction: {instruction_type}")

        arguments = instruction.args_copy()
        if len(arguments) != 1:
            raise ValueError("Stim error instruction must contain exactly one probability")
        probability = float(arguments[0])
        detectors: list[int] = []
        observables: list[int] = []
        for target in instruction.targets_copy():
            if target.is_separator():
                continue
            if target.is_relative_detector_id():
                detectors.append(int(target.val))
                continue
            if target.is_logical_observable_id():
                observables.append(int(target.val))
                continue
            raise ValueError(f"unsupported Stim DEM target: {target}")
        errors.append(
            qp.DetectorError(
                probability,
                _parity_support(detectors),
                _parity_support(observables),
            )
        )

    model = qp.DetectorModel(
        int(flattened.num_detectors),
        int(flattened.num_observables),
        errors,
    )
    if len(model.errors) != int(flattened.num_errors):
        raise RuntimeError("Stim-to-QuPy detector-model conversion changed the error count")
    return model


def reconstruct_syndromes(
    model: qp.DetectorModel,
    corrections: npt.ArrayLike,
) -> npt.NDArray[np.int8]:
    values = np.asarray(corrections)
    if values.ndim != 2:
        raise ValueError("corrections must be a two-dimensional array")
    if values.shape[1] != len(model.errors):
        raise ValueError("correction columns must match detector-model errors")
    if not bool(np.all(np.isin(values, (0, 1)))):
        raise ValueError("correction values must be zero or one")

    reconstructed = np.zeros((values.shape[0], model.detector_count), dtype=np.int8)
    bits = np.ascontiguousarray(values, dtype=np.int8)
    for error_index, error in enumerate(model.errors):
        selected = bits[:, error_index]
        for detector in error.detectors:
            reconstructed[:, detector] ^= selected
    return reconstructed


def _observable_matrix(
    values: npt.ArrayLike,
    *,
    shots: int,
    observable_count: int,
    name: str,
) -> npt.NDArray[np.int8]:
    array = np.asarray(values)
    if observable_count == 1 and array.ndim == 1:
        array = array[:, np.newaxis]
    if array.shape != (shots, observable_count):
        raise ValueError(
            f"{name} must have shape ({shots}, {observable_count}), got {array.shape}"
        )
    if not bool(np.all(np.isin(array, (0, 1)))):
        raise ValueError(f"{name} values must be zero or one")
    return np.ascontiguousarray(array, dtype=np.int8)


def _logical_failures(
    predicted: npt.NDArray[np.int8],
    actual: npt.NDArray[np.int8],
) -> int:
    return int(np.count_nonzero(np.any(predicted != actual, axis=1)))


def _wilson_interval(failures: int, shots: int) -> tuple[float, float]:
    if not 0 <= failures <= shots or shots < 1:
        raise ValueError("Wilson interval requires 0 <= failures <= shots and shots >= 1")
    z = 1.959963984540054
    observed = failures / shots
    z2 = z * z
    denominator = 1.0 + z2 / shots
    center = (observed + z2 / (2.0 * shots)) / denominator
    radius = (
        z
        * math.sqrt(observed * (1.0 - observed) / shots + z2 / (4.0 * shots * shots))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _timed_batch(
    execute: object,
    *,
    warmups: int,
    iterations: int,
) -> tuple[object, tuple[int, ...]]:
    if warmups < 0:
        raise ValueError("warmups cannot be negative")
    if iterations < 1:
        raise ValueError("iterations must be at least one")
    for _ in range(warmups):
        execute()
    timings: list[int] = []
    result: object | None = None
    for _ in range(iterations):
        started = time.perf_counter_ns()
        result = execute()
        timings.append(time.perf_counter_ns() - started)
    if result is None:
        raise RuntimeError("decoder timing produced no result")
    return result, tuple(timings)


def _decoder_metrics(
    *,
    engine: str,
    engine_version: str,
    method: str,
    setup_ns: int,
    timings_ns: tuple[int, ...],
    failures: int,
    shots: int,
) -> DecoderMetrics:
    median_ns = float(statistics.median(timings_ns))
    return DecoderMetrics(
        engine=engine,
        engine_version=engine_version,
        method=method,
        setup_ns=setup_ns,
        timings_ns=timings_ns,
        median_ns=median_ns,
        min_ns=min(timings_ns),
        max_ns=max(timings_ns),
        median_shots_per_second=shots * 1_000_000_000.0 / median_ns,
        logical_failures=failures,
        logical_failure_rate=failures / shots,
        logical_failure_interval_95=_wilson_interval(failures, shots),
    )


def _generated_surface_code(stim: object, workload: QecWorkload) -> object:
    probability = workload.physical_error_rate
    return stim.Circuit.generated(
        workload.task,
        distance=workload.distance,
        rounds=workload.rounds,
        after_clifford_depolarization=probability,
        before_round_data_depolarization=probability,
        before_measure_flip_probability=probability,
        after_reset_flip_probability=probability,
    )


def run_qec_workload(
    workload: QecWorkload,
    *,
    warmups: int = 1,
    iterations: int = 5,
    max_bp_iterations: int = 50,
    damping: float = 0.1,
) -> QecBenchmarkResult:
    if warmups < 0:
        raise ValueError("warmups cannot be negative")
    if iterations < 1:
        raise ValueError("iterations must be at least one")
    if max_bp_iterations < 0:
        raise ValueError("max BP iterations cannot be negative")
    if not math.isfinite(damping) or not 0.0 <= damping < 1.0:
        raise ValueError("damping must be finite and in [0, 1)")

    try:
        stim = import_module("stim")
        pymatching = import_module("pymatching")
    except ImportError as error:
        raise RuntimeError(
            "QEC benchmarks require the optional stim and pymatching packages"
        ) from error

    circuit = _generated_surface_code(stim, workload)
    dem = circuit.detector_error_model(decompose_errors=True)

    started = time.perf_counter_ns()
    model = _qupy_model_from_stim(dem)
    model_conversion_ns = time.perf_counter_ns() - started
    if model.detector_count != int(dem.num_detectors):
        raise RuntimeError("Stim-to-QuPy conversion changed the detector count")
    if model.observable_count != int(dem.num_observables):
        raise RuntimeError("Stim-to-QuPy conversion changed the observable count")

    sampler = circuit.compile_detector_sampler(seed=workload.seed)
    started = time.perf_counter_ns()
    sampled = sampler.sample(shots=workload.shots, separate_observables=True)
    sample_ns = time.perf_counter_ns() - started
    syndromes = np.ascontiguousarray(sampled[0], dtype=np.int8)
    actual_observables = _observable_matrix(
        sampled[1],
        shots=workload.shots,
        observable_count=model.observable_count,
        name="Stim observable samples",
    )
    if syndromes.shape != (workload.shots, model.detector_count):
        raise RuntimeError("Stim syndrome samples do not match the converted detector model")

    started = time.perf_counter_ns()
    qupy_decoder = qp.BpOsdDecoder(
        model,
        max_iterations=max_bp_iterations,
        damping=damping,
    )
    qupy_setup_ns = time.perf_counter_ns() - started

    started = time.perf_counter_ns()
    matching = pymatching.Matching.from_detector_error_model(dem)
    pymatching_setup_ns = time.perf_counter_ns() - started

    qupy_result_object, qupy_timings = _timed_batch(
        lambda: qupy_decoder.decode_batch(syndromes),
        warmups=warmups,
        iterations=iterations,
    )
    qupy_result = qupy_result_object
    qupy_observables = _observable_matrix(
        qupy_result.observables,
        shots=workload.shots,
        observable_count=model.observable_count,
        name="QuPy predicted observables",
    )

    pymatching_result_object, pymatching_timings = _timed_batch(
        lambda: matching.decode_batch(syndromes),
        warmups=warmups,
        iterations=iterations,
    )
    pymatching_observables = _observable_matrix(
        pymatching_result_object,
        shots=workload.shots,
        observable_count=model.observable_count,
        name="PyMatching predicted observables",
    )

    reconstructed = reconstruct_syndromes(model, qupy_result.corrections)
    syndrome_consistency = np.all(reconstructed == syndromes, axis=1)
    syndrome_consistency_rate = float(np.mean(syndrome_consistency))
    if syndrome_consistency_rate != 1.0:
        raise AssertionError("QuPy returned a correction that does not reproduce its syndrome")

    qupy_failures = _logical_failures(qupy_observables, actual_observables)
    pymatching_failures = _logical_failures(pymatching_observables, actual_observables)
    prediction_agreement_rate = float(
        np.mean(np.all(qupy_observables == pymatching_observables, axis=1))
    )

    bp_converged = np.asarray(qupy_result.bp_converged, dtype=np.int8)
    osd_used = np.asarray(qupy_result.osd_used, dtype=np.int8)
    bp_iterations = np.asarray(qupy_result.iterations, dtype=np.uint64)

    return QecBenchmarkResult(
        workload=workload.name,
        task=workload.task,
        distance=workload.distance,
        rounds=workload.rounds,
        physical_error_rate=workload.physical_error_rate,
        shots=workload.shots,
        seed=workload.seed,
        detector_count=model.detector_count,
        observable_count=model.observable_count,
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
        pymatching=_decoder_metrics(
            engine="pymatching",
            engine_version=_package_version("pymatching"),
            method="sparse-blossom-mwpm",
            setup_ns=pymatching_setup_ns,
            timings_ns=pymatching_timings,
            failures=pymatching_failures,
            shots=workload.shots,
        ),
        prediction_agreement_rate=prediction_agreement_rate,
        syndrome_consistency_rate=syndrome_consistency_rate,
        bp_convergence_rate=float(np.mean(bp_converged)),
        osd_usage_rate=float(np.mean(osd_used)),
        mean_bp_iterations=float(np.mean(bp_iterations)),
        max_bp_iterations=int(np.max(bp_iterations, initial=0)),
        active_error_count=qupy_decoder.active_error_count,
        edge_count=qupy_decoder.edge_count,
    )


def run_qec_suite(
    workloads: tuple[QecWorkload, ...],
    *,
    warmups: int = 1,
    iterations: int = 5,
    max_bp_iterations: int = 50,
    damping: float = 0.1,
) -> QecBenchmarkReport:
    results = tuple(
        run_qec_workload(
            workload,
            warmups=warmups,
            iterations=iterations,
            max_bp_iterations=max_bp_iterations,
            damping=damping,
        )
        for workload in workloads
    )
    return QecBenchmarkReport(
        schema_version=_SCHEMA_VERSION,
        kind="qec-decoder-evidence",
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
        description="Compare QuPy BP+OSD-0 with PyMatching on Stim surface-code workloads"
    )
    parser.add_argument("--profile", choices=("smoke", "standard"), default="smoke")
    parser.add_argument("--shots", type=int, default=None, help="override shots for every workload")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--max-bp-iterations", type=int, default=50)
    parser.add_argument("--damping", type=float, default=0.1)
    parser.add_argument("--output", default="-", help="JSON output path or - for stdout")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    workloads = qec_workloads_for_profile(args.profile)
    if args.shots is not None:
        if args.shots < 1:
            raise SystemExit("shots must be at least one")
        workloads = tuple(replace(workload, shots=args.shots) for workload in workloads)
    report = run_qec_suite(
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
