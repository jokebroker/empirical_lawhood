"""Closed issued-config decoder and provider for the uniform electron gas analytic unit."""

from dataclasses import dataclass

from empirical_lawhood.adapters.composition.discovery import executable
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution, ExecutableCapabilityBinding, ExecutablePlatformPort

from .extension_bundle import CAPABILITY, COMPONENTS
from .native_quickstart import UniformElectronGasAnalyticReferenceConfig
from .runtime_provider import UniformElectronGasAnalyticSourceProvider

BINDING = executable(CAPABILITY, COMPONENTS, (UniformElectronGasAnalyticReferenceConfig,))


@dataclass(frozen=True, slots=True)
class UniformElectronGasAnalyticSourceFactory:
    binding: ExecutableCapabilityBinding = BINDING

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> UniformElectronGasAnalyticSourceProvider:
        if len(records) != 1 or not isinstance(records[0], UniformElectronGasAnalyticReferenceConfig):
            raise ValueError(
                "uniform electron gas analytic native factory requires its exact issued config"
            )
        if platform_ports:
            raise ValueError(
                "uniform electron gas analytic source cannot accept a physical platform port"
            )
        return UniformElectronGasAnalyticSourceProvider(registry, records[0])


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    "executable-contribution.uniform-electron-gas-analytic-reference.native-panel",
    "1.0.0",
    (BINDING,),
)
EXECUTABLE_BINDING_FACTORIES = (UniformElectronGasAnalyticSourceFactory(),)
EXECUTABLE_RECORD_TYPES = (UniformElectronGasAnalyticReferenceConfig,)
