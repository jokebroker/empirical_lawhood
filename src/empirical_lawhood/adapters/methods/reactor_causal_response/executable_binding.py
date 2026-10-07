"""Installed finite-action campaign product, separate from the baseline compiler."""

from dataclasses import dataclass, replace
from typing import cast
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.adapters.methods.reactor_causal_response.resource_contract import EmpiricalResourceGuard

from empirical_lawhood.adapters.methods.law_assessment import (
    CandidatePayloadPublisher,
    CandidatePayloadReader,
)
from empirical_lawhood.adapters.composition.discovery import executable
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution, ExecutableCapabilityBinding, ExecutablePlatformPort
from .config import EmpiricalRecipe
from .extension_bundle import CAPABILITY, COMPONENTS
from .provider import EmpiricalProvider, DependencyCustodyReader
from .native_benchmark import NativeVerifierPort
from empirical_lawhood.runtime.providers import ExternalInputPayload
from empirical_lawhood.adapters.simulators.reactor_causal_response.campaign import ControlCustodyPort


BINDING = replace(
    executable(CAPABILITY, COMPONENTS, (EmpiricalRecipe,)),
    required_platform_port_keys=(
        "candidate-payload-publisher",
        "candidate-payload-reader",
        "dependency-custody-reader",
        "reactor-causal-response-source-input",
        "reactor-empirical-control-custody",
        "reactor-empirical-native-verifier",
        "reactor-empirical-resource-guard",
    ),
)


@dataclass(frozen=True, slots=True)
class EmpiricalFactory:
    binding: ExecutableCapabilityBinding = BINDING

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> EmpiricalProvider:
        if len(records) != 1 or not isinstance(records[0], EmpiricalRecipe):
            raise ValueError("finite chain factory requires its exact pre-acquisition design")
        ports = {p.port_key: p.port for p in platform_ports}
        if (
            len(platform_ports) != 7
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
        if not callable(
            getattr(ports["reactor-empirical-resource-guard"], "task", None)
        ) or not callable(
            getattr(ports["reactor-empirical-control-custody"], "open_control_store", None)
        ):
            raise TypeError("empirical resource/control ports differ")
        if not isinstance(ports["reactor-causal-response-source-input"], ExternalInputPayload) or (
            not callable(getattr(ports["reactor-empirical-native-verifier"], "verify", None))
            or not isinstance(
                getattr(ports["reactor-empirical-native-verifier"], "environment", None),
                ObjectIdentity,
            )
        ):
            raise TypeError("empirical source/verifier ports differ")
        return EmpiricalProvider(
            registry,
            records[0],
            cast(CandidatePayloadPublisher, ports["candidate-payload-publisher"]),
            cast(CandidatePayloadReader, ports["candidate-payload-reader"]),
            cast(DependencyCustodyReader, ports["dependency-custody-reader"]),
            cast(ControlCustodyPort, ports["reactor-empirical-control-custody"]),
            cast(NativeVerifierPort, ports["reactor-empirical-native-verifier"]),
            ports["reactor-causal-response-source-input"],
            limits=cast(EmpiricalResourceGuard, ports["reactor-empirical-resource-guard"]),
        )


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    "executable-contribution.reactor-causal-response.numerical-method",
    "1.0.0",
    (BINDING,),
)
EXECUTABLE_BINDING_FACTORIES = (EmpiricalFactory(),)
EXECUTABLE_RECORD_TYPES = (EmpiricalRecipe,)
