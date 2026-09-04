from collections.abc import Mapping, Sequence

from ._native import Observable, ParameterSlot, PauliTerm, Program

class VariationalTemplate:
    program: Program
    slots: tuple[ParameterSlot, ...]
    parameter_names: tuple[str, ...]
    @property
    def parameter_count(self) -> int: ...
    def bind(self, values: Sequence[float]) -> Program: ...
    def bind_named(self, values: Mapping[str, float]) -> Program: ...


def hardware_efficient_ansatz(
    num_qubits: int,
    layers: int,
    *,
    rotations: Sequence[str] = ("ry", "rz"),
    entanglement: str = "linear",
) -> VariationalTemplate: ...


def append_qft(
    program: Program,
    qubits: Sequence[int] | None = None,
    *,
    inverse: bool = False,
    swaps: bool = True,
) -> Program: ...


def qft(num_qubits: int, *, inverse: bool = False, swaps: bool = True) -> Program: ...


def append_pauli_evolution(program: Program, term: PauliTerm, time: float) -> Program: ...


def append_hamiltonian_evolution(
    program: Program,
    hamiltonian: Observable,
    time: float,
    *,
    steps: int = 1,
    order: int = 1,
) -> Program: ...


def maxcut_hamiltonian(
    num_qubits: int,
    edges: Sequence[tuple[int, int]],
    *,
    weights: Sequence[float] | None = None,
) -> Observable: ...


def qaoa_maxcut_program(
    num_qubits: int,
    edges: Sequence[tuple[int, int]],
    gammas: Sequence[float],
    betas: Sequence[float],
    *,
    weights: Sequence[float] | None = None,
) -> Program: ...
