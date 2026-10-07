"""Installed exact factory; caller configuration never selects code."""

from typing import Any


from dataclasses import dataclass, replace
from empirical_lawhood.adapters.composition.closed_provider_ports import closed_ports, input_payloads
from empirical_lawhood.adapters.composition.discovery import executable
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution
from empirical_lawhood.adapters.methods.constructed_preparation_applicability.config import ConstructedPreparationNativeConfig
from .extension_bundle import CAPABILITY, COMPONENTS

BINDING = replace(
    executable(CAPABILITY, COMPONENTS, (ConstructedPreparationNativeConfig,)),
    required_platform_port_keys=tuple(
        sorted(("bp-native-inputs", "bp-native-custody", "bp-native-limits"))
    ),
)


@dataclass(frozen=True, slots=True)
class Factory:
    binding = BINDING

    def build_provider(self, *, registry: Any, records: Any, platform_ports: Any) -> Any:
        from .provider import ConstructedPreparationNativeRunner, ConstructedPreparationNativeProvider
        config, ports = closed_ports(
            self.binding, records, ConstructedPreparationNativeConfig, platform_ports
        )
        runner = ConstructedPreparationNativeRunner(
            config, ports["bp-native-custody"], ports["bp-native-limits"]
        )
        return ConstructedPreparationNativeProvider(
            registry, runner, input_payloads(ports["bp-native-inputs"])
        )


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    "executable-contribution.constructed-preparation-applicability.native", "1.0.0", (BINDING,)
)
EXECUTABLE_BINDING_FACTORIES = (Factory(),)
EXECUTABLE_RECORD_TYPES = (ConstructedPreparationNativeConfig,)
