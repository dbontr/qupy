from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import platform
import statistics
import time
from typing import Iterable

from .adapters import AdapterUnavailable, WorkloadUnsupported, prepare
from .model import Workload

_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    workload: str
    family: str
    num_qubits: int
    operation_count: int
    observable_qubit: int
    engine: str
    engine_version: str | None
    method: str | None
    expected_value: float
    result_value: float | None
    tolerance: float
    valid: bool | None
    warmups: int
    iterations: int
    timings_ns: tuple[int, ...]
    median_ns: float | None
    min_ns: int | None
    max_ns: int | None
    metadata: dict[str, object]
    skipped: bool
    skip_reason: str | None

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["timings_ns"] = list(self.timings_ns)
        return result


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    schema_version: int
    generated_at: str
    host: dict[str, str]
    results: tuple[BenchmarkResult, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "host": self.host,
            "results": [result.to_dict() for result in self.results],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def _host_metadata() -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
    }


def _skipped_result(workload: Workload, engine: str, reason: str) -> BenchmarkResult:
    return BenchmarkResult(
        workload=workload.name,
        family=workload.family,
        num_qubits=workload.num_qubits,
        operation_count=workload.operation_count,
        observable_qubit=workload.observable_qubit,
        engine=engine,
        engine_version=None,
        method=None,
        expected_value=workload.expected_value,
        result_value=None,
        tolerance=workload.tolerance,
        valid=None,
        warmups=0,
        iterations=0,
        timings_ns=(),
        median_ns=None,
        min_ns=None,
        max_ns=None,
        metadata={},
        skipped=True,
        skip_reason=reason,
    )


def run_case(
    workload: Workload,
    engine: str,
    *,
    warmups: int,
    iterations: int,
    require_engine: bool = False,
) -> BenchmarkResult:
    if warmups < 0:
        raise ValueError("warmups cannot be negative")
    if iterations < 1:
        raise ValueError("iterations must be at least one")
    try:
        prepared = prepare(engine, workload)
    except AdapterUnavailable as error:
        if require_engine:
            raise
        return _skipped_result(workload, engine, str(error))
    except WorkloadUnsupported as error:
        return _skipped_result(workload, engine, str(error))

    validation_value = prepared.execute()
    valid = abs(validation_value - workload.expected_value) <= workload.tolerance
    if not valid:
        raise AssertionError(
            f"{engine} returned {validation_value} for {workload.name}; "
            f"expected {workload.expected_value} +/- {workload.tolerance}"
        )

    for _ in range(warmups):
        prepared.execute()

    timings: list[int] = []
    last_value = validation_value
    for _ in range(iterations):
        started = time.perf_counter_ns()
        last_value = prepared.execute()
        timings.append(time.perf_counter_ns() - started)

    return BenchmarkResult(
        workload=workload.name,
        family=workload.family,
        num_qubits=workload.num_qubits,
        operation_count=workload.operation_count,
        observable_qubit=workload.observable_qubit,
        engine=prepared.engine,
        engine_version=prepared.engine_version,
        method=prepared.method,
        expected_value=workload.expected_value,
        result_value=last_value,
        tolerance=workload.tolerance,
        valid=True,
        warmups=warmups,
        iterations=iterations,
        timings_ns=tuple(timings),
        median_ns=float(statistics.median(timings)),
        min_ns=min(timings),
        max_ns=max(timings),
        metadata=prepared.metadata,
        skipped=False,
        skip_reason=None,
    )


def run_suite(
    workloads: Iterable[Workload],
    engines: Iterable[str],
    *,
    warmups: int = 1,
    iterations: int = 5,
    require_engines: bool = False,
) -> BenchmarkReport:
    workload_list = tuple(workloads)
    engine_list = tuple(engines)
    results = tuple(
        run_case(
            workload,
            engine,
            warmups=warmups,
            iterations=iterations,
            require_engine=require_engines,
        )
        for workload in workload_list
        for engine in engine_list
    )
    if require_engines:
        for engine in engine_list:
            if not any(result.engine == engine and not result.skipped for result in results):
                raise RuntimeError(f"benchmark engine {engine} ran no compatible workloads")
    return BenchmarkReport(
        schema_version=_SCHEMA_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        host=_host_metadata(),
        results=results,
    )
