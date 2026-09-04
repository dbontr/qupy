from typing import Protocol

from ._native import ProviderJobState, ProviderProgram
from .circuit import Circuit
from .compiler import CompilationResult, HardwareTarget

class ProviderBackend(Protocol):
    @property
    def name(self) -> str: ...
    def capabilities_json(self) -> str: ...
    def submit(
        self,
        program: ProviderProgram,
        shots: int,
        options_json: str = "{}",
    ) -> str: ...
    def poll(self, job_id: str) -> ProviderJobState: ...
    def result_json(self, job_id: str) -> str: ...
    def cancel(self, job_id: str) -> None: ...


class ProviderCapabilities:
    @property
    def formats(self) -> tuple[str, ...]: ...
    @property
    def hardware_target(self) -> HardwareTarget | None: ...


class ProviderSubmission:
    @property
    def job_id(self) -> str: ...
    @property
    def compilation(self) -> CompilationResult: ...
    @property
    def program(self) -> ProviderProgram: ...
    @property
    def shots(self) -> int: ...
    @property
    def options_json(self) -> str: ...


def _hardware_target_payload(target: HardwareTarget) -> dict[str, object]: ...

def provider_capabilities(plugin: ProviderBackend) -> ProviderCapabilities: ...

def provider_program(circuit: Circuit) -> ProviderProgram: ...

def submit_compiled_circuit(
    plugin: ProviderBackend,
    compilation: CompilationResult,
    shots: int,
    options_json: str = "{}",
) -> ProviderSubmission: ...

def submit_circuit(
    plugin: ProviderBackend,
    circuit: Circuit,
    shots: int,
    *,
    target: HardwareTarget | None = None,
    initial_layout: list[int] | None = None,
    optimization_level: int = 1,
    options_json: str = "{}",
) -> ProviderSubmission: ...
