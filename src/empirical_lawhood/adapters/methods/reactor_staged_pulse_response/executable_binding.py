"""Installed closed factory; configuration never chooses executable code."""

from dataclasses import dataclass, replace
from typing import Any, cast
from empirical_lawhood.adapters.composition.closed_provider_ports import closed_ports, input_payloads
from empirical_lawhood.adapters.composition.discovery import executable
from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.config import ClassicalStage
from empirical_lawhood.adapters.simulators.reactor_staged_pulse_response.config import ClassicalNativeConfig
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution, ExecutableCapabilityBinding, ExecutablePlatformPort
from .extension_bundle import CAPABILITY, COMPONENTS
from .provider import ClassicalMethodRunner, ClassicalMethodProvider

BINDING = replace(
    executable(CAPABILITY, COMPONENTS, (ClassicalStage,)),
    required_platform_port_keys=tuple(
        sorted(
            (
                "candidate-payload-publisher",
                "candidate-payload-reader",
                "classical-control",
                "classical-custody",
                "classical-method-inputs",
                "classical-limits",
                "classical-native-config",
            )
        )
    ),
)


@dataclass(frozen=True, slots=True)
class ClassicalMethodFactory:
    binding: ExecutableCapabilityBinding = BINDING

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> ClassicalMethodProvider:
        config, ports = closed_ports(self.binding, records, ClassicalStage, platform_ports)
        other = ports["classical-native-config"]
        if not isinstance(other, ClassicalNativeConfig) or other.design != config.design:
            raise ValueError("provider changes its exact stage/native design")
        runner = ClassicalMethodRunner(
            config,
            other,
            cast(Any, ports["classical-custody"]),
            cast(Any, ports["classical-limits"]),
            cast(Any, ports["candidate-payload-publisher"]),
            cast(Any, ports["candidate-payload-reader"]),
            cast(Any, ports["classical-control"]),
        )
        return ClassicalMethodProvider(
            registry, runner, input_payloads(ports["classical-method-inputs"])
        )


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    "executable-contribution.reactor-staged-pulse-response.method", "1.0.0", (BINDING,)
)
EXECUTABLE_BINDING_FACTORIES = (ClassicalMethodFactory(),)
EXECUTABLE_RECORD_TYPES = (ClassicalStage,)
