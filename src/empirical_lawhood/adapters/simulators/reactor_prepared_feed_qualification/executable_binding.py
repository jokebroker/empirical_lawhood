"""Static reconstruction of the exact local qualification configuration."""

from dataclasses import dataclass, replace
from typing import cast
from empirical_lawhood.adapters.methods.reactor_causal_response.provider import DependencyCustodyReader
from empirical_lawhood.adapters.methods.reactor_causal_response.resource_contract import EmpiricalResourceGuard
from empirical_lawhood.adapters.composition.discovery import executable
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution, ExecutableCapabilityBinding, ExecutablePlatformPort
from empirical_lawhood.runtime.providers import ExternalInputPayload
from .config import FeedNativeConfig
from .provider import FeedNativeProvider
from .extension_bundle import CAPABILITY, COMPONENTS

BINDING = replace(
    executable(CAPABILITY, COMPONENTS, (FeedNativeConfig,)),
    required_platform_port_keys=(
        "reactor-feed-native-dependency-reader",
        "reactor-feed-native-resource-guard",
        "reactor-feed-native-source-input",
    ),
)


@dataclass(frozen=True, slots=True)
class FeedNativeFactory:
    binding: ExecutableCapabilityBinding = BINDING

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> FeedNativeProvider:
        if len(records) != 1 or not isinstance(records[0], FeedNativeConfig):
            raise ValueError("local factory requires its exact configuration")
        ports = {p.port_key: p.port for p in platform_ports}
        if tuple(sorted(ports)) != self.binding.required_platform_port_keys or len(ports) != len(
            platform_ports
        ):
            raise ValueError("local factory port roster differs")
        if not callable(
            getattr(ports["reactor-feed-native-dependency-reader"], "read_dependency", None)
        ):
            raise TypeError(
                "local factory port contract differs: reactor-feed-native-dependency-reader"
            )
        if not callable(getattr(ports["reactor-feed-native-resource-guard"], "task", None)):
            raise TypeError(
                "local factory port contract differs: reactor-feed-native-resource-guard"
            )
        if not isinstance(ports["reactor-feed-native-source-input"], ExternalInputPayload):
            raise TypeError("local factory port contract differs: reactor-feed-native-source-input")
        return FeedNativeProvider(
            registry,
            records[0],
            ports["reactor-feed-native-source-input"],
            cast(DependencyCustodyReader, ports["reactor-feed-native-dependency-reader"]),
            cast(EmpiricalResourceGuard, ports["reactor-feed-native-resource-guard"]),
        )


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    "executable-contribution.terminal-bench-science-feed.simulators", "1.0.0", (BINDING,)
)
EXECUTABLE_BINDING_FACTORIES = (FeedNativeFactory(),)
EXECUTABLE_RECORD_TYPES = (FeedNativeConfig,)
