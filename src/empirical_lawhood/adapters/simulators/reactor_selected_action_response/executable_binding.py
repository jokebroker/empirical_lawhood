"""Static factory binding for the classical reactor native adapter."""

from dataclasses import dataclass, replace
from typing import cast
from empirical_lawhood.adapters.composition.discovery import executable
from empirical_lawhood.adapters.methods.law_assessment import CandidatePayloadReader
from empirical_lawhood.adapters.methods.reactor_causal_response.provider import DependencyCustodyReader
from empirical_lawhood.adapters.methods.reactor_causal_response.resource_contract import EmpiricalResourceGuard
from empirical_lawhood.adapters.methods.reactor_selected_action_response.provider import ClassicalControlCustody
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution, ExecutableCapabilityBinding, ExecutablePlatformPort
from empirical_lawhood.runtime.providers import ExternalInputPayload
from .config import ClassicalNativeConfig
from .extension_bundle import CAPABILITY, COMPONENTS
from .provider import ClassicalNativeRunner, ClassicalNativeProvider

BINDING = replace(
    executable(CAPABILITY, COMPONENTS, (ClassicalNativeConfig,)),
    required_platform_port_keys=tuple(
        sorted(
            (
                "classical-native-custody",
                "classical-native-limits",
                "classical-native-control",
                "classical-native-law-reader",
                "classical-source-input",
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
        if len(records) != 1 or not isinstance(records[0], ClassicalNativeConfig):
            raise ValueError("classical native factory requires its exact issued configuration")
        ports = {p.port_key: p.port for p in platform_ports}
        if tuple(sorted(ports)) != self.binding.required_platform_port_keys or len(ports) != len(
            platform_ports
        ):
            raise ValueError("classical native factory port roster changed")
        for key, method in (
            ("classical-native-custody", "read_dependency"),
            ("classical-native-limits", "task"),
            ("classical-native-control", "open_prepared_store"),
            ("classical-native-law-reader", "read_candidate_payload"),
        ):
            if not callable(getattr(ports[key], method, None)):
                raise TypeError(
                    "classical factory lacks its installed custody, limits or control port"
                )
        runner = ClassicalNativeRunner(
            records[0],
            cast(DependencyCustodyReader, ports["classical-native-custody"]),
            cast(EmpiricalResourceGuard, ports["classical-native-limits"]),
            cast(ClassicalControlCustody, ports["classical-native-control"]),
            cast(CandidatePayloadReader, ports["classical-native-law-reader"]),
        )
        source = ports["classical-source-input"]
        if not isinstance(source, ExternalInputPayload):
            raise TypeError("classical native source has no declared input payload")
        return ClassicalNativeProvider(registry, runner, source)


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    "executable-contribution.reactor-selected-action-response.native", "1.0.0", (BINDING,)
)
EXECUTABLE_BINDING_FACTORIES = (ClassicalNativeFactory(),)
EXECUTABLE_RECORD_TYPES = (ClassicalNativeConfig,)
