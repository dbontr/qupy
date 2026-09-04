from __future__ import annotations

from typing import Protocol, cast

import numpy as np
import numpy.typing as npt

from . import _native


class _NativeBpOsdDecodeResult(Protocol):
    @property
    def correction(self) -> npt.NDArray[np.int8]: ...

    @property
    def observables(self) -> npt.NDArray[np.int8]: ...

    @property
    def log_likelihood(self) -> float: ...

    @property
    def matched_errors(self) -> int: ...

    @property
    def iterations(self) -> int: ...

    @property
    def bp_converged(self) -> bool: ...

    @property
    def osd_used(self) -> bool: ...


class _NativeBpOsdDecodeBatch(Protocol):
    @property
    def corrections(self) -> npt.NDArray[np.int8]: ...

    @property
    def observables(self) -> npt.NDArray[np.int8]: ...

    @property
    def log_likelihoods(self) -> npt.NDArray[np.float64]: ...

    @property
    def matched_errors(self) -> npt.NDArray[np.uint64]: ...

    @property
    def iterations(self) -> npt.NDArray[np.uint64]: ...

    @property
    def bp_converged(self) -> npt.NDArray[np.int8]: ...

    @property
    def osd_used(self) -> npt.NDArray[np.int8]: ...

    @property
    def shots(self) -> int: ...

    @property
    def error_count(self) -> int: ...

    @property
    def observable_count(self) -> int: ...


class _NativeBpOsdDecoder(Protocol):
    @property
    def model(self) -> _native.DetectorModel: ...

    @property
    def max_iterations(self) -> int: ...

    @property
    def damping(self) -> float: ...

    @property
    def edge_count(self) -> int: ...

    @property
    def active_error_count(self) -> int: ...

    def decode(self, syndrome: npt.NDArray[np.int8]) -> _NativeBpOsdDecodeResult: ...

    def decode_batch(
        self,
        syndromes: npt.NDArray[np.int8],
    ) -> _NativeBpOsdDecodeBatch: ...


class _NativeQecModule(Protocol):
    def BpOsdDecoder(
        self,
        model: _native.DetectorModel,
        max_iterations: int = 50,
        damping: float = 0.0,
    ) -> _NativeBpOsdDecoder: ...


def _binary_int8_array(
    values: npt.ArrayLike,
    *,
    ndim: int,
    name: str,
) -> npt.NDArray[np.int8]:
    array = np.asarray(values)
    if array.ndim != ndim:
        dimension = "one-dimensional" if ndim == 1 else "two-dimensional"
        raise ValueError(f"{name} must be a {dimension} array")
    if not bool(np.all(np.isin(array, (0, 1)))):
        raise ValueError(f"{name} values must be zero or one")
    return np.ascontiguousarray(array, dtype=np.int8)


class BpOsdDecodeResult:
    """One syndrome-consistent BP+OSD-0 detector-model decoding result."""

    __slots__ = ("_native",)

    def __init__(self, native: _NativeBpOsdDecodeResult) -> None:
        self._native = native

    @property
    def correction(self) -> npt.NDArray[np.int8]:
        return self._native.correction

    @property
    def observables(self) -> npt.NDArray[np.int8]:
        return self._native.observables

    @property
    def log_likelihood(self) -> float:
        return self._native.log_likelihood

    @property
    def matched_errors(self) -> int:
        return self._native.matched_errors

    @property
    def iterations(self) -> int:
        return self._native.iterations

    @property
    def bp_converged(self) -> bool:
        return self._native.bp_converged

    @property
    def osd_used(self) -> bool:
        return self._native.osd_used

    @property
    def method(self) -> str:
        return "belief-propagation-osd0" if self.osd_used else "belief-propagation"


class BpOsdDecodeBatch:
    """Native batch of BP+OSD-0 detector-model decoding results."""

    __slots__ = ("_native",)

    def __init__(self, native: _NativeBpOsdDecodeBatch) -> None:
        self._native = native

    @property
    def corrections(self) -> npt.NDArray[np.int8]:
        return self._native.corrections

    @property
    def observables(self) -> npt.NDArray[np.int8]:
        return self._native.observables

    @property
    def log_likelihoods(self) -> npt.NDArray[np.float64]:
        return self._native.log_likelihoods

    @property
    def matched_errors(self) -> npt.NDArray[np.uint64]:
        return self._native.matched_errors

    @property
    def iterations(self) -> npt.NDArray[np.uint64]:
        return self._native.iterations

    @property
    def bp_converged(self) -> npt.NDArray[np.int8]:
        return self._native.bp_converged

    @property
    def osd_used(self) -> npt.NDArray[np.int8]:
        return self._native.osd_used

    @property
    def shots(self) -> int:
        return self._native.shots

    @property
    def error_count(self) -> int:
        return self._native.error_count

    @property
    def observable_count(self) -> int:
        return self._native.observable_count


class BpOsdDecoder:
    """Reusable native sparse BP decoder with deterministic OSD-0 fallback."""

    __slots__ = ("_native",)

    def __init__(
        self,
        model: _native.DetectorModel,
        *,
        max_iterations: int = 50,
        damping: float = 0.0,
    ) -> None:
        native = cast(_NativeQecModule, _native)
        self._native = native.BpOsdDecoder(model, max_iterations, damping)

    @property
    def model(self) -> _native.DetectorModel:
        return self._native.model

    @property
    def max_iterations(self) -> int:
        return self._native.max_iterations

    @property
    def damping(self) -> float:
        return self._native.damping

    @property
    def edge_count(self) -> int:
        return self._native.edge_count

    @property
    def active_error_count(self) -> int:
        return self._native.active_error_count

    def decode(self, syndrome: npt.ArrayLike) -> BpOsdDecodeResult:
        values = _binary_int8_array(syndrome, ndim=1, name="syndrome")
        return BpOsdDecodeResult(self._native.decode(values))

    def decode_batch(self, syndromes: npt.ArrayLike) -> BpOsdDecodeBatch:
        values = _binary_int8_array(syndromes, ndim=2, name="syndromes")
        return BpOsdDecodeBatch(self._native.decode_batch(values))


def decode_detector_model_bp_osd(
    model: _native.DetectorModel,
    syndrome: npt.ArrayLike,
    *,
    max_iterations: int = 50,
    damping: float = 0.0,
) -> BpOsdDecodeResult:
    """Decode one detector syndrome with sparse BP and OSD-0 fallback."""
    return BpOsdDecoder(
        model,
        max_iterations=max_iterations,
        damping=damping,
    ).decode(syndrome)


__all__ = [
    "BpOsdDecodeBatch",
    "BpOsdDecodeResult",
    "BpOsdDecoder",
    "decode_detector_model_bp_osd",
]
