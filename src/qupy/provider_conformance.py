from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass

from . import _native
from .circuit import Circuit
from .compiler import HardwareTarget
from .provider import provider_capabilities, submit_circuit


@dataclass(frozen=True, slots=True)
class ProviderConformanceReport:
    provider_name: str
    formats: tuple[str, ...]
    target_name: str
    target_fingerprint: str
    shots: int
    poll_states: tuple[str, ...]
    program_sha256: str
    job_id_sha256: str
    result_json_sha256: str
    result_json_type: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


_JOB_STATE_NAMES = {
    _native.ProviderJobState.QUEUED: "queued",
    _native.ProviderJobState.RUNNING: "running",
    _native.ProviderJobState.SUCCEEDED: "succeeded",
    _native.ProviderJobState.FAILED: "failed",
    _native.ProviderJobState.CANCELLED: "cancelled",
}
_NONTERMINAL_STATE_RANK = {
    _native.ProviderJobState.QUEUED: 0,
    _native.ProviderJobState.RUNNING: 1,
}


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_finite_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_type(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (int, float)):
        return "number"
    raise TypeError(f"unsupported decoded JSON type: {type(value).__name__}")


def check_provider_conformance(
    plugin: _native.ProviderPlugin,
    *,
    target: HardwareTarget | None = None,
    shots: int = 1,
    max_polls: int = 32,
    poll_interval_seconds: float = 0.5,
    options_json: str = "{}",
) -> ProviderConformanceReport:
    """Exercise the portable provider discovery, submission, and success lifecycle.

    This function submits a real provider job. Callers are responsible for any
    provider-side cost, credentials, service policy, and rate limits.
    """

    shot_count = _positive_integer(shots, "shots")
    poll_limit = _positive_integer(max_polls, "max_polls")
    poll_interval = _nonnegative_finite_float(
        poll_interval_seconds, "poll_interval_seconds"
    )

    capabilities = provider_capabilities(plugin)
    if "openqasm3" not in capabilities.formats:
        raise ValueError("provider conformance requires advertised openqasm3 support")

    advertised_target = capabilities.hardware_target
    if target is not None and advertised_target is not None:
        if target.fingerprint != advertised_target.fingerprint:
            raise ValueError(
                "explicit target does not match the provider-advertised hardware_target"
            )
    selected_target = target if target is not None else advertised_target
    if selected_target is None:
        raise ValueError(
            "provider conformance requires an advertised hardware_target or target= explicitly"
        )
    if not selected_target.measurement:
        raise ValueError("provider conformance requires terminal measurement support")

    circuit = Circuit(1, 1).measure(0, 0)
    submission = submit_circuit(
        plugin,
        circuit,
        shot_count,
        target=selected_target,
        initial_layout=[0],
        optimization_level=0,
        options_json=options_json,
    )

    state_names: list[str] = []
    highest_nonterminal_rank = -1
    succeeded = False
    for poll_index in range(poll_limit):
        state = plugin.poll(submission.job_id)
        try:
            state_name = _JOB_STATE_NAMES[state]
        except KeyError:
            raise RuntimeError("provider returned an unsupported job state") from None
        state_names.append(state_name)

        if state == _native.ProviderJobState.SUCCEEDED:
            succeeded = True
            break
        if state == _native.ProviderJobState.FAILED:
            raise RuntimeError("provider conformance job entered the failed state")
        if state == _native.ProviderJobState.CANCELLED:
            raise RuntimeError("provider conformance job entered the cancelled state")

        rank = _NONTERMINAL_STATE_RANK[state]
        if rank < highest_nonterminal_rank:
            raise RuntimeError(
                "provider job state regressed from running back to queued"
            )
        highest_nonterminal_rank = max(highest_nonterminal_rank, rank)
        if poll_interval > 0.0 and poll_index + 1 < poll_limit:
            time.sleep(poll_interval)

    if not succeeded:
        raise TimeoutError(
            f"provider conformance job did not succeed within {poll_limit} polls"
        )

    result_text = plugin.result_json(submission.job_id)
    try:
        decoded_result = json.loads(result_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"provider returned invalid result JSON: {exc.msg}") from exc

    return ProviderConformanceReport(
        provider_name=plugin.name,
        formats=capabilities.formats,
        target_name=selected_target.name,
        target_fingerprint=selected_target.fingerprint,
        shots=shot_count,
        poll_states=tuple(state_names),
        program_sha256=_sha256(submission.program.text),
        job_id_sha256=_sha256(submission.job_id),
        result_json_sha256=_sha256(result_text),
        result_json_type=_json_type(decoded_result),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exercise a QuPy provider plug-in against the portable execution contract"
    )
    parser.add_argument("plugin", help="path to a provider shared library")
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--max-polls", type=int, default=32)
    parser.add_argument("--poll-interval", type=float, default=0.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plugin = _native.ProviderPlugin(args.plugin)
        report = check_provider_conformance(
            plugin,
            shots=args.shots,
            max_polls=args.max_polls,
            poll_interval_seconds=args.poll_interval,
        )
    except Exception as exc:
        print(f"provider conformance failed: {exc}", file=sys.stderr)
        return 1
    print(report.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
