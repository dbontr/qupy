from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .adapters import adapter_names
from .harness import run_suite
from .model import workloads_for_profile


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run QuPy workload-class benchmarks")
    parser.add_argument("--profile", choices=("smoke", "standard"), default="smoke")
    parser.add_argument(
        "--engines",
        default="qupy",
        help=f"comma-separated engines ({', '.join(adapter_names())})",
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", default="-", help="JSON output path or - for stdout")
    parser.add_argument(
        "--require-engines",
        action="store_true",
        help="fail if a requested optional engine is unavailable or runs no compatible workload",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    engines = tuple(part.strip() for part in args.engines.split(",") if part.strip())
    if not engines:
        raise SystemExit("at least one benchmark engine is required")
    unknown = sorted(set(engines) - set(adapter_names()))
    if unknown:
        raise SystemExit(f"unknown benchmark engines: {', '.join(unknown)}")

    report = run_suite(
        workloads_for_profile(args.profile),
        engines,
        warmups=args.warmups,
        iterations=args.iterations,
        require_engines=args.require_engines,
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
