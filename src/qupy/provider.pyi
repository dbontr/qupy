from ._native import ProviderPlugin, ProviderProgram
from .circuit import Circuit
from .compiler import CompilationResult, HardwareTarget

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


def provider_capabilities(plugin: ProviderPlugin) -> ProviderCapabilities: ...

def provider_program(circuit: Circuit) -> ProviderProgram: ...

def submit_compiled_circuit(
    plugin: ProviderPlugin,
    compilation: CompilationResult,
    shots: int,
    options_json: str = "{}",
) -> ProviderSubmission: ...

def submit_circuit(
    plugin: ProviderPlugin,
    circuit: Circuit,
    shots: int,
    *,
    target: HardwareTarget | None = None,
    initial_layout: list[int] | None = None,
    optimization_level: int = 1,
    options_json: str = "{}",
) -> ProviderSubmission: ...
