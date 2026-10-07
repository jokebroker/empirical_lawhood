"""Code-owned reconstruction of the native prefix provider; construction has no effects."""

from dataclasses import dataclass, replace

from empirical_lawhood.adapters.composition.discovery import executable
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution, ExecutableCapabilityBinding, ExecutablePlatformPort
from empirical_lawhood.runtime.providers import ExternalInputPayload
from .extension_bundle import CAPABILITY, COMPONENTS
from .panel import ReactorPrefixConfig
from .provider import ReactorPrefixProvider


BINDING = replace(
    executable(CAPABILITY, COMPONENTS, (ReactorPrefixConfig,)),
    required_platform_port_keys=("tbs-reactor-source-input",),
)


@dataclass(frozen=True, slots=True)
class ReactorPrefixFactory:
    binding: ExecutableCapabilityBinding = BINDING

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> ReactorPrefixProvider:
        if len(records) != 1 or not isinstance(records[0], ReactorPrefixConfig):
            raise ValueError("reactor factory requires its exact prefix configuration")
        if tuple(
            p.port_key for p in platform_ports
        ) != self.binding.required_platform_port_keys or not isinstance(
            platform_ports[0].port, ExternalInputPayload
        ):
            raise ValueError("reactor factory requires its authenticated source input port")
        return ReactorPrefixProvider(registry, records[0], platform_ports[0].port)


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    "executable-contribution.terminal-bench-science.reactor-prefix", "1.0.0", (BINDING,)
)
EXECUTABLE_BINDING_FACTORIES = (ReactorPrefixFactory(),)
EXECUTABLE_RECORD_TYPES = (ReactorPrefixConfig,)
