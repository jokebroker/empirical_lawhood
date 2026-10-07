"""Installed finite-action campaign product, separate from the baseline compiler."""

from dataclasses import dataclass, replace
from typing import cast

from empirical_lawhood.adapters.methods.law_assessment import (
    CandidatePayloadPublisher,
    CandidatePayloadReader,
)
from empirical_lawhood.adapters.composition.discovery import executable
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution, ExecutableCapabilityBinding, ExecutablePlatformPort
from .design import ReactorScienceDesign
from .extension_bundle import CAPABILITY, COMPONENTS
from .provider import FiniteChainProvider, DependencyCustodyReader


BINDING = replace(
    executable(CAPABILITY, COMPONENTS, (ReactorScienceDesign,)),
    required_platform_port_keys=(
        "candidate-payload-publisher",
        "candidate-payload-reader",
        "dependency-custody-reader",
    ),
)


@dataclass(frozen=True, slots=True)
class FiniteChainFactory:
    binding: ExecutableCapabilityBinding = BINDING

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> FiniteChainProvider:
        if len(records) != 1 or not isinstance(records[0], ReactorScienceDesign):
            raise ValueError("finite chain factory requires its exact pre-acquisition design")
        ports = {p.port_key: p.port for p in platform_ports}
        if (
            len(platform_ports) != 3
            or tuple(sorted(ports)) != self.binding.required_platform_port_keys
        ):
            raise ValueError("finite chain factory requires exact payload and custody ports")
        if (
            not callable(
                getattr(ports["candidate-payload-publisher"], "publish_candidate_payload", None)
            )
            or not callable(
                getattr(ports["candidate-payload-reader"], "read_candidate_payload", None)
            )
            or not callable(getattr(ports["dependency-custody-reader"], "read_dependency", None))
        ):
            raise TypeError("finite chain payload ports do not satisfy their protocols")
        return FiniteChainProvider(
            registry,
            records[0],
            cast(CandidatePayloadPublisher, ports["candidate-payload-publisher"]),
            cast(CandidatePayloadReader, ports["candidate-payload-reader"]),
            cast(DependencyCustodyReader, ports["dependency-custody-reader"]),
        )


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    "executable-contribution.terminal-bench-science.finite-chain", "1.0.0", (BINDING,)
)
EXECUTABLE_BINDING_FACTORIES = (FiniteChainFactory(),)
EXECUTABLE_RECORD_TYPES = (ReactorScienceDesign,)
