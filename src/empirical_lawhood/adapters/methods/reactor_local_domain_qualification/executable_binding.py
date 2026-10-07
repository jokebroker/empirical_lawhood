"""Static reconstruction of the exact local qualification configuration."""

from dataclasses import dataclass, replace
from typing import cast
from empirical_lawhood.adapters.methods.law_assessment import (
    CandidatePayloadPublisher,
    CandidatePayloadReader,
)
from empirical_lawhood.adapters.methods.reactor_causal_response.provider import DependencyCustodyReader
from empirical_lawhood.adapters.methods.reactor_causal_response.resource_contract import EmpiricalResourceGuard
from empirical_lawhood.adapters.composition.discovery import executable
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution, ExecutableCapabilityBinding, ExecutablePlatformPort
from .config import LocalQualificationDesign
from .provider import LocalMethodProvider
from .extension_bundle import CAPABILITY, COMPONENTS

BINDING = replace(
    executable(CAPABILITY, COMPONENTS, (LocalQualificationDesign,)),
    required_platform_port_keys=(
        "candidate-payload-publisher",
        "candidate-payload-reader",
        "dependency-custody-reader",
        "reactor-local-resource-guard",
    ),
)


@dataclass(frozen=True, slots=True)
class LocalMethodFactory:
    binding: ExecutableCapabilityBinding = BINDING

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> LocalMethodProvider:
        if len(records) != 1 or not isinstance(records[0], LocalQualificationDesign):
            raise ValueError("local factory requires its exact configuration")
        ports = {p.port_key: p.port for p in platform_ports}
        if tuple(sorted(ports)) != self.binding.required_platform_port_keys or len(ports) != len(
            platform_ports
        ):
            raise ValueError("local factory port roster differs")
        if not callable(
            getattr(ports["candidate-payload-publisher"], "publish_candidate_payload", None)
        ):
            raise TypeError("local factory port contract differs: candidate-payload-publisher")
        if not callable(getattr(ports["candidate-payload-reader"], "read_candidate_payload", None)):
            raise TypeError("local factory port contract differs: candidate-payload-reader")
        if not callable(getattr(ports["dependency-custody-reader"], "read_dependency", None)):
            raise TypeError("local factory port contract differs: dependency-custody-reader")
        if not callable(getattr(ports["reactor-local-resource-guard"], "task", None)):
            raise TypeError("local factory port contract differs: reactor-local-resource-guard")
        return LocalMethodProvider(
            registry,
            records[0],
            cast(DependencyCustodyReader, ports["dependency-custody-reader"]),
            cast(CandidatePayloadPublisher, ports["candidate-payload-publisher"]),
            cast(CandidatePayloadReader, ports["candidate-payload-reader"]),
            cast(EmpiricalResourceGuard, ports["reactor-local-resource-guard"]),
        )


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    "executable-contribution.terminal-bench-science-local.methods", "1.0.0", (BINDING,)
)
EXECUTABLE_BINDING_FACTORIES = (LocalMethodFactory(),)
EXECUTABLE_RECORD_TYPES = (LocalQualificationDesign,)
