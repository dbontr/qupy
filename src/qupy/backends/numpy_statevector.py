from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from ..ir import Program
from ..ops import PauliZ
from ..results import Expectation, Samples, StateVector
from ..targets import Target

ComplexArray = npt.NDArray[np.complex128]

TARGET = Target(
    name="numpy-statevector",
    operations=frozenset({"h", "x", "rx", "cx"}),
    result_modes=frozenset({"sample", "expectation", "statevector"}),
    simulator=True,
)


def _apply_single(state: ComplexArray, matrix: ComplexArray, qubit: int) -> None:
    indices = np.arange(state.size, dtype=np.int64)
    zero = indices[(indices & (1 << qubit)) == 0]
    one = zero | (1 << qubit)
    a = state[zero].copy()
    b = state[one].copy()
    state[zero] = matrix[0, 0] * a + matrix[0, 1] * b
    state[one] = matrix[1, 0] * a + matrix[1, 1] * b

def _apply_cx(state: ComplexArray, control: int, target: int) -> None:
    indices = np.arange(state.size, dtype=np.int64)
    selected = indices[
        (((indices >> control) & 1) == 1) & (((indices >> target) & 1) == 0)
    ]
    paired = selected | (1 << target)
    scratch = state[selected].copy()
    state[selected] = state[paired]
    state[paired] = scratch


def statevector(program: Program) -> StateVector:
    TARGET.validate(program, "statevector")
    state = np.zeros(1 << program.num_qubits, dtype=np.complex128)
    state[0] = 1.0
    root2 = math.sqrt(2.0)
    h_matrix = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / root2
    x_matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

    for operation in program.operations:
        if operation.name == "h":
            _apply_single(state, h_matrix, operation.qubits[0])
        elif operation.name == "x":
            _apply_single(state, x_matrix, operation.qubits[0])
        elif operation.name == "rx":
            half = operation.parameters[0] / 2.0
            c, s = math.cos(half), math.sin(half)
            matrix = np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)
            _apply_single(state, matrix, operation.qubits[0])
        elif operation.name == "cx":
            _apply_cx(state, operation.qubits[0], operation.qubits[1])

    return StateVector(state, TARGET.name)


def sample(program: Program, shots: int, seed: int | None = None) -> Samples:
    if shots < 1:
        raise ValueError("shots must be at least 1")
    TARGET.validate(program, "sample")
    state = statevector(program).values
    probabilities = np.abs(state) ** 2
    rng = np.random.default_rng(seed)
    basis_states = rng.choice(state.size, size=shots, p=probabilities)
    bits = np.empty((shots, program.num_qubits), dtype=np.int8)
    for column, qubit in enumerate(reversed(range(program.num_qubits))):
        bits[:, column] = (basis_states >> qubit) & 1
    return Samples(bits, TARGET.name)


def expectation(program: Program, observable: PauliZ) -> Expectation:
    if not 0 <= observable.qubit < program.num_qubits:
        raise ValueError(f"qubit {observable.qubit} is outside this program")
    TARGET.validate(program, "expectation")
    state = statevector(program).values
    indices = np.arange(state.size, dtype=np.int64)
    signs = 1.0 - 2.0 * ((indices >> observable.qubit) & 1)
    value = float(np.sum((np.abs(state) ** 2) * signs))
    return Expectation(value, TARGET.name)