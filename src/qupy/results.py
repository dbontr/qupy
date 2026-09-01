from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

ComplexArray = npt.NDArray[np.complex128]
IntArray = npt.NDArray[np.int8]


@dataclass(frozen=True, slots=True)
class StateVector:
    values: ComplexArray
    backend: str


@dataclass(frozen=True, slots=True)
class Samples:
    values: IntArray
    backend: str

    @property
    def shots(self) -> int:
        return int(self.values.shape[0])

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.values:
            key = "".join(str(int(bit)) for bit in row)
            counts[key] = counts.get(key, 0) + 1
        return counts


@dataclass(frozen=True, slots=True)
class Expectation:
    value: float
    backend: str