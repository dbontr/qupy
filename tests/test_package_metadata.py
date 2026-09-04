from __future__ import annotations

from importlib.metadata import metadata, version
from importlib.resources import files

import qupy as qp


def test_distribution_version_matches_public_package_version() -> None:
    distribution_version = version("qupy-compute")
    assert qp.__version__ == distribution_version
    assert distribution_version == "0.3.0a0"


def test_release_metadata_points_to_canonical_repository() -> None:
    project_urls = metadata("qupy-compute").get_all("Project-URL") or []
    assert "Homepage, https://github.com/dbontr/qupy" in project_urls
    assert "Repository, https://github.com/dbontr/qupy" in project_urls
    assert "Documentation, https://github.com/dbontr/qupy/tree/main/docs" in project_urls
    assert "Issues, https://github.com/dbontr/qupy/issues" in project_urls


def test_pep561_marker_is_packaged_with_qupy() -> None:
    assert files("qupy").joinpath("py.typed").is_file()
