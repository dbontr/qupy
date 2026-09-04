from __future__ import annotations

from pathlib import Path

import pytest

from tools.verify_wheel import find_sdist, find_wheel


def test_release_artifact_discovery_requires_exactly_one_match(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="exactly one QuPy wheel"):
        find_wheel(tmp_path)
    with pytest.raises(RuntimeError, match="exactly one QuPy sdist"):
        find_sdist(tmp_path)

    wheel = tmp_path / "qupy_compute-0.3.0a0-cp312-cp312-manylinux.whl"
    sdist = tmp_path / "qupy_compute-0.3.0a0.tar.gz"
    wheel.touch()
    sdist.touch()

    assert find_wheel(tmp_path) == wheel.resolve()
    assert find_sdist(tmp_path) == sdist.resolve()

    second_wheel = tmp_path / "qupy_compute-0.3.0a0-cp313-cp313-manylinux.whl"
    second_wheel.touch()
    with pytest.raises(RuntimeError, match="found 2"):
        find_wheel(tmp_path)
