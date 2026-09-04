from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import venv
from pathlib import Path

_WHEEL_GLOB = "qupy_compute-*.whl"
_SDIST_GLOB = "qupy_compute-*.tar.gz"

_VERIFY_CODE = r'''
import importlib.machinery
import importlib.metadata
import importlib.resources
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import qupy as qp
import qupy._native as native

prefix = Path(sys.prefix).resolve()
package_path = Path(qp.__file__).resolve()
native_path = Path(native.__file__).resolve()

assert package_path.is_relative_to(prefix), (package_path, prefix)
assert native_path.is_relative_to(prefix), (native_path, prefix)
assert any(str(native_path).endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES)

distribution = importlib.metadata.distribution("qupy-compute")
distribution_version = distribution.version
assert distribution_version == qp.__version__ == qp.core_version()
assert qp.core_language() == "C++20"
assert qp.ir_version() == 1
assert qp.circuit_ir_version() == 1

console_scripts = {
    entry.name: entry.value
    for entry in distribution.entry_points
    if entry.group == "console_scripts"
}
assert console_scripts["qupy-provider-conformance"] == "qupy.provider_conformance:main"

marker = importlib.resources.files("qupy").joinpath("py.typed")
assert marker.is_file(), marker

for optional_module in ("jax", "torch", "braket"):
    assert importlib.util.find_spec(optional_module) is None, optional_module
    assert optional_module not in sys.modules, optional_module

parameterized = qp.ry(qp.Program(1), 0.0, 0)
parameterized_observable = qp.observable_from_z(qp.Z(0))
parameterized_slots = [qp.ParameterSlot(0, 0)]
for factory, dependency in (
    (qp.make_jax_expectation, "jax"),
    (qp.make_torch_expectation, "torch"),
):
    try:
        factory(parameterized, parameterized_observable, parameterized_slots)
    except ImportError as exc:
        assert dependency in str(exc), (dependency, exc)
    else:
        raise AssertionError(f"{dependency} adapter did not fail without its optional dependency")

try:
    qp.BraketProvider.local_simulator()
except ImportError as exc:
    assert "amazon-braket-sdk" in str(exc), exc
else:
    raise AssertionError("Amazon Braket adapter did not fail without its optional dependency")

program = qp.Program(2)
program = qp.h(program, 0)
program = qp.cx(program, 0, 1)
probabilities = np.asarray(qp.probabilities(program).values)
expected = np.asarray([0.5, 0.0, 0.0, 0.5])
np.testing.assert_allclose(probabilities, expected, rtol=0.0, atol=1e-12)

state = np.asarray(qp.statevector(program).values)
expected_state = np.asarray([1.0 / math.sqrt(2.0), 0.0, 0.0, 1.0 / math.sqrt(2.0)])
np.testing.assert_allclose(state, expected_state, rtol=0.0, atol=1e-12)

print(json.dumps({
    "distribution_version": distribution_version,
    "package": str(package_path),
    "native_extension": str(native_path),
    "prefix": str(prefix),
    "py_typed": str(marker),
}, sort_keys=True))
'''


def _single_artifact(directory: Path, pattern: str, kind: str) -> Path:
    artifacts = sorted(directory.glob(pattern))
    if len(artifacts) != 1:
        names = ", ".join(path.name for path in artifacts) or "none"
        raise RuntimeError(
            f"expected exactly one QuPy {kind} in {directory}, found {len(artifacts)}: {names}"
        )
    return artifacts[0].resolve()


def find_wheel(directory: Path) -> Path:
    return _single_artifact(directory, _WHEEL_GLOB, "wheel")


def find_sdist(directory: Path) -> Path:
    return _single_artifact(directory, _SDIST_GLOB, "sdist")


def _venv_python(environment: Path) -> Path:
    if os.name == "nt":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def _venv_script(environment: Path, name: str) -> Path:
    if os.name == "nt":
        return environment / "Scripts" / f"{name}.exe"
    return environment / "bin" / name


def _clean_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    return environment


def _create_environment(path: Path) -> Path:
    venv.EnvBuilder(with_pip=True, clear=True).create(path)
    python = _venv_python(path)
    if not python.is_file():
        raise RuntimeError(f"virtual environment did not create an interpreter at {python}")
    return python


def _run(command: list[str], *, cwd: Path, environment: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def install_and_verify(wheel: Path, workspace: Path) -> None:
    environment = _clean_environment()
    install_environment = workspace / "install-env"
    python = _create_environment(install_environment)
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-input",
            str(wheel),
        ],
        cwd=workspace,
        environment=environment,
    )
    _run(
        [str(python), "-I", "-c", _VERIFY_CODE],
        cwd=workspace,
        environment=environment,
    )
    conformance_cli = _venv_script(install_environment, "qupy-provider-conformance")
    if not conformance_cli.is_file():
        raise RuntimeError(f"wheel did not install the provider conformance CLI at {conformance_cli}")
    _run(
        [str(conformance_cli), "--help"],
        cwd=workspace,
        environment=environment,
    )


def rebuild_sdist(sdist: Path, workspace: Path) -> Path:
    environment = _clean_environment()
    python = _create_environment(workspace / "build-env")
    output = workspace / "rebuilt-wheel"
    output.mkdir()
    _run(
        [
            str(python),
            "-m",
            "pip",
            "wheel",
            "--no-input",
            "--no-deps",
            "--wheel-dir",
            str(output),
            str(sdist),
        ],
        cwd=workspace,
        environment=environment,
    )
    return find_wheel(output)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Install and execute a built QuPy wheel in a clean virtual environment"
    )
    parser.add_argument(
        "--dist-dir",
        type=Path,
        default=Path("dist"),
        help="directory containing the QuPy wheel and source distribution",
    )
    parser.add_argument(
        "--rebuild-sdist",
        action="store_true",
        help="also rebuild the source distribution into a wheel and verify that wheel",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    distribution_directory = args.dist_dir.resolve()
    wheel = find_wheel(distribution_directory)

    with tempfile.TemporaryDirectory(prefix="qupy-wheel-") as temporary:
        root = Path(temporary)
        direct_workspace = root / "direct"
        direct_workspace.mkdir()
        install_and_verify(wheel, direct_workspace)

        if args.rebuild_sdist:
            sdist = find_sdist(distribution_directory)
            rebuild_workspace = root / "sdist"
            rebuild_workspace.mkdir()
            rebuilt_wheel = rebuild_sdist(sdist, rebuild_workspace)
            install_workspace = root / "sdist-install"
            install_workspace.mkdir()
            install_and_verify(rebuilt_wheel, install_workspace)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
