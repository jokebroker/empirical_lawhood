"""Installed static frontier factory; configuration never names executable code."""

from dataclasses import dataclass, replace
from typing import cast, Any
from empirical_lawhood.adapters.composition.discovery import executable
from empirical_lawhood.adapters.methods.law_assessment import (
    CandidatePayloadPublisher,
    CandidatePayloadReader,
)
from empirical_lawhood.adapters.methods.reactor_causal_response.provider import DependencyCustodyReader
from empirical_lawhood.adapters.methods.reactor_causal_response.resource_contract import EmpiricalResourceGuard
from empirical_lawhood.adapters.simulators.reactor_finite_control_frontier.config import FrontierNativeConfig
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution, ExecutableCapabilityBinding, ExecutablePlatformPort
from empirical_lawhood.runtime.providers import ExternalInputPayload
from .phase import FrontierPhase
from .extension_bundle import CAPABILITY, COMPONENTS
from .provider import FrontierMethodRunner, FrontierMethodProvider

BINDING = replace(
    executable(CAPABILITY, COMPONENTS, (FrontierPhase,)),
    required_platform_port_keys=tuple(
        sorted(
            (
                "frontier-custody",
                "frontier-limits",
                "frontier-control",
                "frontier-native-config",
                "frontier-method-inputs",
                "candidate-payload-reader",
                "candidate-payload-publisher",
            )
        )
    ),
)


@dataclass(frozen=True, slots=True)
class FrontierMethodFactory:
    binding: ExecutableCapabilityBinding = BINDING

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> FrontierMethodProvider:
        if len(records) != 1 or not isinstance(records[0], FrontierPhase):
            raise ValueError("method factory requires its exact issued phase")
        ports = {p.port_key: p.port for p in platform_ports}
        if tuple(sorted(ports)) != self.binding.required_platform_port_keys or len(ports) != len(
            platform_ports
        ):
            raise ValueError("method factory changed its closed port roster")
        native = ports["frontier-native-config"]
        if not isinstance(native, FrontierNativeConfig) or native.design != records[0].design:
            raise ValueError("method factory lost its pinned native configuration")
        runner = FrontierMethodRunner(
            records[0],
            native,
            cast(DependencyCustodyReader, ports["frontier-custody"]),
            cast(EmpiricalResourceGuard, ports["frontier-limits"]),
            cast(CandidatePayloadPublisher, ports["candidate-payload-publisher"]),
            cast(CandidatePayloadReader, ports["candidate-payload-reader"]),
            cast(Any, ports["frontier-control"]),
        )
        inputs = ports["frontier-method-inputs"]
        if not isinstance(inputs, tuple) or any(
            not isinstance(p, ExternalInputPayload) for p in inputs
        ):
            raise TypeError("method factory requires exact upstream source payloads")
        return FrontierMethodProvider(registry, runner, inputs)


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    "executable-contribution.reactor-finite-control-frontier.method", "1.0.0", (BINDING,)
)
EXECUTABLE_BINDING_FACTORIES = (FrontierMethodFactory(),)
EXECUTABLE_RECORD_TYPES = (FrontierPhase,)
