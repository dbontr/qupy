from __future__ import annotations

import os
import sys
import tempfile
import threading
from os import PathLike
from pathlib import Path

from ._native import (
    PlannerCostModel,
    core_version,
    load_planner_cost_model,
    planner_host_fingerprint,
)

_ENV_MODEL = "QUPY_PLANNER_COST_MODEL"
_ENV_CACHE = "QUPY_CACHE_DIR"
_lock = threading.RLock()
_configured_path: Path | None = None
_cached_path: Path | None = None
_cached_stamp: tuple[int, int] | None = None
_cached_model: PlannerCostModel | None = None


def _cache_root() -> Path:
    override = os.environ.get(_ENV_CACHE)
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        return Path(local).expanduser() if local else Path.home() / "AppData" / "Local"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches"
    xdg = os.environ.get("XDG_CACHE_HOME")
    return Path(xdg).expanduser() if xdg else Path.home() / ".cache"


def planner_cache_path() -> Path:
    """Return the host-scoped path used for an installed planner artifact."""
    host = planner_host_fingerprint()
    return _cache_root() / "qupy" / "planner" / core_version() / f"{host}.qpcost"


def _discovered_path() -> Path | None:
    if _configured_path is not None:
        return _configured_path
    environment = os.environ.get(_ENV_MODEL)
    if environment:
        return Path(environment).expanduser()
    cached = planner_cache_path()
    return cached if cached.is_file() else None


def _clear_memory_cache() -> None:
    global _cached_path, _cached_stamp, _cached_model
    _cached_path = None
    _cached_stamp = None
    _cached_model = None


def default_planner_cost_model() -> PlannerCostModel | None:
    """Load the configured or installed validated planner model, if one exists."""
    global _cached_path, _cached_stamp, _cached_model
    with _lock:
        path = _discovered_path()
        if path is None:
            _clear_memory_cache()
            return None
        try:
            stat = path.stat()
        except FileNotFoundError:
            _clear_memory_cache()
            if _configured_path is not None or os.environ.get(_ENV_MODEL):
                raise ValueError(f"planner cost artifact does not exist: {path}") from None
            return None
        stamp = (stat.st_mtime_ns, stat.st_size)
        if _cached_path == path and _cached_stamp == stamp and _cached_model is not None:
            return _cached_model
        model = load_planner_cost_model(os.fspath(path))
        _cached_path = path
        _cached_stamp = stamp
        _cached_model = model
        return model


def set_default_planner_cost_model(path: str | PathLike[str] | None) -> PlannerCostModel | None:
    """Set an in-process planner artifact override, or restore automatic discovery."""
    global _configured_path
    with _lock:
        _configured_path = None if path is None else Path(path).expanduser()
        _clear_memory_cache()
        if _configured_path is None:
            return None
        return default_planner_cost_model()


def install_planner_cost_model(path: str | PathLike[str]) -> PlannerCostModel:
    """Validate and atomically install a planner artifact for automatic reuse."""
    source = Path(path).expanduser()
    load_planner_cost_model(os.fspath(source))
    destination = planner_cache_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(source.read_bytes())
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    model = load_planner_cost_model(os.fspath(destination))
    with _lock:
        _clear_memory_cache()
    return model


def remove_planner_cost_model() -> bool:
    """Remove the installed host-scoped planner artifact."""
    destination = planner_cache_path()
    with _lock:
        existed = destination.exists()
        destination.unlink(missing_ok=True)
        _clear_memory_cache()
    return existed


def resolve_planner_cost_model(
    backend: str,
    cost_model: PlannerCostModel | None,
) -> PlannerCostModel | None:
    if cost_model is not None or backend != "auto":
        return cost_model
    return default_planner_cost_model()
