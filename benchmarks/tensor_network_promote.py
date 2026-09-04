from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

from benchmarks.tensor_network_calibrate import (
    Observation,
    TensorNetworkCalibration,
    _aggregate,
    _features,
    _load_report,
    calibrate_files,
)


def _format_float(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("tensor-network planner artifact requires finite values")
    return format(value, ".17g")


def _calibration_rows(paths: tuple[Path, ...]) -> list[Observation]:
    rows: list[Observation] = []
    for path in paths:
        _, report_rows, _ = _load_report(path)
        rows.extend(report_rows)
    return _aggregate(rows)


def _feature_domains(
    rows: list[Observation],
) -> dict[str, tuple[tuple[float, float], ...]]:
    features_by_class: dict[str, list[tuple[float, ...]]] = defaultdict(list)
    for row in rows:
        features_by_class[row.cost_class].append(_features(row))

    domains: dict[str, tuple[tuple[float, float], ...]] = {}
    for cost_class in ("tensor-network-baseline-cpu", "tensor-network-return-cpu"):
        samples = features_by_class[cost_class]
        if not samples:
            raise ValueError(f"tensor-network promotion is missing {cost_class} evidence")
        width = len(samples[0])
        if any(len(sample) != width for sample in samples):
            raise ValueError("tensor-network promotion feature widths are inconsistent")
        domains[cost_class] = tuple(
            (min(sample[index] for sample in samples), max(sample[index] for sample in samples))
            for index in range(width)
        )
    return domains


def _render_artifact(
    calibration: TensorNetworkCalibration,
    domains: dict[str, tuple[tuple[float, float], ...]],
) -> str:
    if not calibration.validated:
        raise ValueError("tensor-network routing evidence did not meet promotion thresholds")
    host = calibration.host.get("planner_host_fingerprint", "")
    if len(host) != 64:
        raise ValueError("tensor-network calibration host fingerprint is invalid")

    lines = [
        "qupy-tensor-network-cost 1",
        f"engine {calibration.engine_version}",
        f"workload {calibration.workload_version}",
        f"host {host}",
        "policy 1",
        f"reports {calibration.report_count}",
        (
            "decision "
            f"{calibration.decision.samples} {calibration.decision.mistakes} "
            f"{_format_float(calibration.decision.max_regret)} "
            f"{calibration.decision.cpu_wins} {calibration.decision.tn_wins}"
        ),
    ]
    for model in calibration.models:
        coefficient_text = " ".join(_format_float(value) for value in model.coefficients)
        lines.append(
            f"model {model.cost_class} {len(model.coefficients)} {coefficient_text} "
            f"{_format_float(model.holdout_error.median_factor)} "
            f"{_format_float(model.holdout_error.max_factor)}"
        )
        bounds = domains.get(model.cost_class)
        if bounds is None or len(bounds) != len(model.coefficients):
            raise ValueError(f"tensor-network promotion has invalid {model.cost_class} domain")
        domain_text = " ".join(
            f"{_format_float(minimum)} {_format_float(maximum)}"
            for minimum, maximum in bounds
        )
        lines.append(f"domain {model.cost_class} {len(bounds)} {domain_text}")
    lines.append("validated 1")
    return "\n".join(lines) + "\n"


def promote_files(paths: tuple[Path, ...]) -> str:
    calibration = calibrate_files(paths)
    rows = _calibration_rows(paths)
    return _render_artifact(calibration, _feature_domains(rows))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Promote validated tensor-network routing evidence to a native artifact"
    )
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifact = promote_files(tuple(args.reports))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(artifact, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())