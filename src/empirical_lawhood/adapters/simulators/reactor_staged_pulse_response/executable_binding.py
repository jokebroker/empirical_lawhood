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
from .provider import ClassicalNativeRunner, ClassicalNativeProvider

BINDING = replace(
    executable(CAPABILITY, COMPONENTS, (ClassicalNativeConfig,)),
    required_platform_port_keys=tuple(
        sorted(
            (
                "classical-native-reader",
                "classical-native-control",
                "classical-native-custody",
                "classical-native-inputs",
                "classical-native-limits",
                "classical-stage",
            )
        )
    ),
)


@dataclass(frozen=True, slots=True)
class ClassicalNativeFactory:
    binding: ExecutableCapabilityBinding = BINDING

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> ClassicalNativeProvider:
        config, ports = closed_ports(self.binding, records, ClassicalNativeConfig, platform_ports)
        other = ports["classical-stage"]
        if not isinstance(other, ClassicalStage) or other.design != config.design:
            raise ValueError("provider changes its exact stage/native design")
        runner = ClassicalNativeRunner(
            config,
            other,
            cast(Any, ports["classical-native-custody"]),
            cast(Any, ports["classical-native-limits"]),
            cast(Any, ports["classical-native-control"]),
            cast(Any, ports["classical-native-reader"]),
        )
        return ClassicalNativeProvider(
            registry, runner, input_payloads(ports["classical-native-inputs"])
        )


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    "executable-contribution.reactor-staged-pulse-response.native", "1.0.0", (BINDING,)
)
EXECUTABLE_BINDING_FACTORIES = (ClassicalNativeFactory(),)
EXECUTABLE_RECORD_TYPES = (ClassicalNativeConfig,)
