"""Installed exact factory; caller configuration never selects code."""

from typing import Any


from dataclasses import dataclass, replace
from empirical_lawhood.adapters.composition.closed_provider_ports import closed_ports, input_payloads
from empirical_lawhood.adapters.composition.discovery import executable
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution
from empirical_lawhood.adapters.methods.preparation_applicability.config import PreparationApplicabilityNativeConfig
from .extension_bundle import CAPABILITY, COMPONENTS

BINDING = replace(
    executable(CAPABILITY, COMPONENTS, (PreparationApplicabilityNativeConfig,)),
    required_platform_port_keys=tuple(
        sorted(("pa-native-inputs", "pa-native-custody", "pa-native-limits"))
    ),
)


@dataclass(frozen=True, slots=True)
class Factory:
    binding = BINDING

    def build_provider(self, *, registry: Any, records: Any, platform_ports: Any) -> Any:
        from .provider import PreparationApplicabilityNativeRunner, PreparationApplicabilityNativeProvider
        config, ports = closed_ports(
            self.binding, records, PreparationApplicabilityNativeConfig, platform_ports
        )
        runner = PreparationApplicabilityNativeRunner(
            config, ports["pa-native-custody"], ports["pa-native-limits"]
        )
        return PreparationApplicabilityNativeProvider(
            registry, runner, input_payloads(ports["pa-native-inputs"])
        )


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    "executable-contribution.preparation-applicability.native", "1.0.0", (BINDING,)
)
EXECUTABLE_BINDING_FACTORIES = (Factory(),)
EXECUTABLE_RECORD_TYPES = (PreparationApplicabilityNativeConfig,)
