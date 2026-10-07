"""Installed native factory over the pinned source and declared phase inputs."""

from dataclasses import dataclass, replace
from typing import cast, Any
from empirical_lawhood.adapters.composition.discovery import executable
from empirical_lawhood.adapters.methods.reactor_causal_response.provider import DependencyCustodyReader
from empirical_lawhood.adapters.methods.reactor_causal_response.resource_contract import EmpiricalResourceGuard
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.phase import FrontierPhase
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution, ExecutableCapabilityBinding, ExecutablePlatformPort
from empirical_lawhood.runtime.providers import ExternalInputPayload
from .config import FrontierNativeConfig
from .extension_bundle import CAPABILITY, COMPONENTS
from .provider import FrontierNativeRunner, FrontierNativeProvider

BINDING = replace(
    executable(CAPABILITY, COMPONENTS, (FrontierNativeConfig,)),
    required_platform_port_keys=tuple(
        sorted(
            (
                "frontier-native-custody",
                "frontier-native-limits",
                "frontier-native-control",
                "frontier-native-inputs",
                "frontier-phase",
            )
        )
    ),
)


@dataclass(frozen=True, slots=True)
class FrontierNativeFactory:
    binding: ExecutableCapabilityBinding = BINDING

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> FrontierNativeProvider:
        if len(records) != 1 or not isinstance(records[0], FrontierNativeConfig):
            raise ValueError("native factory requires its exact issued configuration")
        ports = {p.port_key: p.port for p in platform_ports}
        if tuple(sorted(ports)) != self.binding.required_platform_port_keys or len(ports) != len(
            platform_ports
        ):
            raise ValueError("native factory changed its closed port roster")
        phase = ports["frontier-phase"]
        if not isinstance(phase, FrontierPhase) or phase.design != records[0].design:
            raise ValueError("native factory lost its exact issued phase")
        inputs = ports["frontier-native-inputs"]
        if not isinstance(inputs, tuple) or any(
            not isinstance(p, ExternalInputPayload) for p in inputs
        ):
            raise TypeError("native factory requires its exact upstream payloads")
        runner = FrontierNativeRunner(
            records[0],
            phase,
            cast(DependencyCustodyReader, ports["frontier-native-custody"]),
            cast(EmpiricalResourceGuard, ports["frontier-native-limits"]),
            cast(Any, ports["frontier-native-control"]),
        )
        return FrontierNativeProvider(registry, runner, inputs)


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    "executable-contribution.reactor-finite-control-frontier.native", "1.0.0", (BINDING,)
)
EXECUTABLE_BINDING_FACTORIES = (FrontierNativeFactory(),)
EXECUTABLE_RECORD_TYPES = (FrontierNativeConfig,)
