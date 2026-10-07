"""Static factory binding for the classical reactor method adapter."""

from dataclasses import dataclass, replace
from typing import cast
from empirical_lawhood.adapters.composition.discovery import executable
from empirical_lawhood.adapters.methods.law_assessment import (
    CandidatePayloadPublisher,
    CandidatePayloadReader,
)
from empirical_lawhood.adapters.methods.reactor_causal_response.provider import DependencyCustodyReader
from empirical_lawhood.adapters.methods.reactor_causal_response.resource_contract import EmpiricalResourceGuard
from empirical_lawhood.adapters.methods.reactor_selected_action_response.provider import ClassicalControlCustody
from empirical_lawhood.adapters.simulators.reactor_selected_action_response.config import ClassicalNativeConfig
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution, ExecutableCapabilityBinding, ExecutablePlatformPort
from .config import ClassicalDesign
from .extension_bundle import CAPABILITY, COMPONENTS
from .provider import ClassicalMethodRunner, ClassicalMethodProvider

BINDING = replace(
    executable(CAPABILITY, COMPONENTS, (ClassicalDesign,)),
    required_platform_port_keys=tuple(
        sorted(
            (
                "classical-custody",
                "classical-limits",
                "classical-control",
                "candidate-payload-reader",
                "candidate-payload-publisher",
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
        if len(records) != 1 or not isinstance(records[0], ClassicalDesign):
            raise ValueError("classical method factory requires its exact issued configuration")
        ports = {p.port_key: p.port for p in platform_ports}
        if tuple(sorted(ports)) != self.binding.required_platform_port_keys or len(ports) != len(
            platform_ports
        ):
            raise ValueError("classical method factory port roster changed")
        for key, method in (
            ("classical-custody", "read_dependency"),
            ("classical-limits", "task"),
            ("classical-control", "open_prepared_store"),
            ("candidate-payload-reader", "read_candidate_payload"),
        ):
            if not callable(getattr(ports[key], method, None)):
                raise TypeError(
                    "classical factory lacks its installed custody, limits or control port"
                )
        native = ports["classical-native-config"]
        if not isinstance(native, ClassicalNativeConfig) or native.design != records[0]:
            raise ValueError("classical method factory changed its bound native preparation domain")
        if not callable(
            getattr(ports["candidate-payload-publisher"], "publish_candidate_payload", None)
        ):
            raise TypeError("classical method factory lacks publication")
        runner = ClassicalMethodRunner(
            records[0],
            native,
            cast(DependencyCustodyReader, ports["classical-custody"]),
            cast(EmpiricalResourceGuard, ports["classical-limits"]),
            cast(CandidatePayloadPublisher, ports["candidate-payload-publisher"]),
            cast(CandidatePayloadReader, ports["candidate-payload-reader"]),
            cast(ClassicalControlCustody, ports["classical-control"]),
        )
        return ClassicalMethodProvider(registry, runner)


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    "executable-contribution.reactor-selected-action-response.method", "1.0.0", (BINDING,)
)
EXECUTABLE_BINDING_FACTORIES = (ClassicalMethodFactory(),)
EXECUTABLE_RECORD_TYPES = (ClassicalDesign,)
