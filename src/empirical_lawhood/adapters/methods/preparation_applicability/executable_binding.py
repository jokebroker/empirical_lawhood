"""Installed exact factory; caller configuration never selects code."""

from typing import Any


from dataclasses import dataclass, replace
from empirical_lawhood.adapters.composition.closed_provider_ports import closed_ports, input_payloads
from empirical_lawhood.adapters.composition.discovery import executable
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution
from empirical_lawhood.adapters.methods.preparation_applicability.config import PreparationApplicabilityStage
from .extension_bundle import CAPABILITY, COMPONENTS

BINDING = replace(
    executable(CAPABILITY, COMPONENTS, (PreparationApplicabilityStage,)),
    required_platform_port_keys=tuple(
        sorted(
            (
                "pa-method-inputs",
                "pa-method-custody",
                "pa-method-limits",
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
        from .provider import PreparationApplicabilityMethodRunner, PreparationApplicabilityMethodProvider
        config, ports = closed_ports(self.binding, records, PreparationApplicabilityStage, platform_ports)
        if (
            ports["candidate-payload-publisher"] is not ports["pa-method-custody"]
            or ports["candidate-payload-reader"] is not ports["pa-method-custody"]
        ):
            raise ValueError("method factory substitutes the shared candidate custody ports")
        runner = PreparationApplicabilityMethodRunner(
            config, ports["pa-method-custody"], ports["pa-method-limits"]
        )
        return PreparationApplicabilityMethodProvider(
            registry, runner, input_payloads(ports["pa-method-inputs"])
        )


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    "executable-contribution.preparation-applicability.method", "1.0.0", (BINDING,)
)
EXECUTABLE_BINDING_FACTORIES = (Factory(),)
EXECUTABLE_RECORD_TYPES = (PreparationApplicabilityStage,)
