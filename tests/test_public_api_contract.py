from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
import tomllib

import qupy as qp

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_API_BASELINE = _REPOSITORY_ROOT / "api" / "public_api_v0.txt"


def _public_api_baseline() -> tuple[str, ...]:
    lines = _API_BASELINE.read_text(encoding="utf-8").splitlines()
    return tuple(line.strip() for line in lines if line.strip() and not line.startswith("#"))


def test_top_level_public_api_matches_checked_baseline() -> None:
    expected = _public_api_baseline()

    assert tuple(qp.__all__) == expected
    assert len(expected) == len(set(expected))
    assert all(name == "__version__" or not name.startswith("_") for name in expected)
    assert all(hasattr(qp, name) for name in expected)


def test_distribution_python_and_native_versions_agree() -> None:
    with (_REPOSITORY_ROOT / "pyproject.toml").open("rb") as file:
        project = tomllib.load(file)["project"]

    declared = project["version"]
    installed = version("qupy-compute")

    assert declared == installed
    assert qp.__version__ == installed
    assert qp.core_version() == installed
    assert qp.__version__ != "0+unknown"
