from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .cost_model import calibrate_files


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit and validate QuPy host-specific execution cost models"
    )
    parser.add_argument(
        "reports",
        nargs="+",
        type=Path,
        help="benchmark JSON reports from one host",
    )
    parser.add_argument("--output", default="-", help="cost-model JSON output path or - for stdout")
    parser.add_argument(
        "--max-holdout-factor",
        type=float,
        default=2.0,
        help="maximum allowed worst held-out multiplicative error",
    )
    parser.add_argument(
        "--max-holdout-median-factor",
        type=float,
        default=1.5,
        help="maximum allowed median held-out multiplicative error",
    )
    parser.add_argument(
        "--allow-unvalidated",
        action="store_true",
        help="write an exploratory model even when held-out validation misses thresholds",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = calibrate_files(
        args.reports,
        max_holdout_factor=args.max_holdout_factor,
        max_holdout_median_factor=args.max_holdout_median_factor,
    )
    payload = report.to_json() + "\n"
    if args.output == "-":
        sys.stdout.write(payload)
    else:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")

    for model in report.models:
        sys.stderr.write(
            f"{model.cost_class}: holdout median {model.holdout_error.median_factor:.3f}x, "
            f"max {model.holdout_error.max_factor:.3f}x, "
            f"validated={model.validated}\n"
        )
    if not report.validated and not args.allow_unvalidated:
        sys.stderr.write("cost-model validation failed; planner promotion is blocked\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
