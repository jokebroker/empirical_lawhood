"""Installed exact factory; caller configuration never selects code."""

from typing import Any


from dataclasses import dataclass, replace
from empirical_lawhood.adapters.composition.closed_provider_ports import closed_ports, input_payloads
from empirical_lawhood.adapters.composition.discovery import executable
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution
from empirical_lawhood.adapters.methods.constructed_preparation_applicability.config import ConstructedPreparationStage
from .extension_bundle import CAPABILITY, COMPONENTS

BINDING = replace(
    executable(CAPABILITY, COMPONENTS, (ConstructedPreparationStage,)),
    required_platform_port_keys=tuple(
        sorted(
            (
                "bp-method-inputs",
                "bp-method-custody",
                "bp-method-limits",
                "candidate-payload-publisher",
                "candidate-payload-reader",
            )
        )
    ),
)


@dataclass(frozen=True, slots=True)
class Factory:
    binding = BINDING

    def build_provider(self, *, registry: Any, records: Any, platform_ports: Any) -> Any:
        from .provider import ConstructedPreparationMethodRunner, ConstructedPreparationMethodProvider
        config, ports = closed_ports(self.binding, records, ConstructedPreparationStage, platform_ports)
        if (
            ports["candidate-payload-publisher"] is not ports["bp-method-custody"]
            or ports["candidate-payload-reader"] is not ports["bp-method-custody"]
        ):
            raise ValueError("method factory substitutes the shared candidate custody ports")
        runner = ConstructedPreparationMethodRunner(
            config, ports["bp-method-custody"], ports["bp-method-limits"]
        )
        return ConstructedPreparationMethodProvider(
            registry, runner, input_payloads(ports["bp-method-inputs"])
        )


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    "executable-contribution.constructed-preparation-applicability.method", "1.0.0", (BINDING,)
)
EXECUTABLE_BINDING_FACTORIES = (Factory(),)
EXECUTABLE_RECORD_TYPES = (ConstructedPreparationStage,)
