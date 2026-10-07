"""Static reconstruction through authenticated config/source/custody ports."""

from dataclasses import dataclass, replace
from typing import cast
from empirical_lawhood.adapters.methods.reactor_causal_response.resource_contract import EmpiricalResourceGuard
from empirical_lawhood.adapters.composition.discovery import executable
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution, ExecutableCapabilityBinding, ExecutablePlatformPort
from empirical_lawhood.runtime.providers import ExternalInputPayload
from empirical_lawhood.adapters.methods.law_assessment import CandidatePayloadReader
from .campaign import ControlCustodyPort
from empirical_lawhood.adapters.methods.reactor_causal_response.provider import DependencyCustodyReader
from .config import EmpiricalNativeConfig
from .extension_bundle import CAPABILITY, COMPONENTS
from .provider import EmpiricalNativeProvider

BINDING = replace(
    executable(CAPABILITY, COMPONENTS, (EmpiricalNativeConfig,)),
    required_platform_port_keys=(
        "reactor-empirical-native-control-custody",
        "reactor-empirical-native-dependency-reader",
        "reactor-empirical-native-payload-reader",
        "reactor-empirical-native-resource-guard",
        "reactor-empirical-native-source-input",
    ),
)


@dataclass(frozen=True, slots=True)
class EmpiricalNativeFactory:
    binding: ExecutableCapabilityBinding = BINDING

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> EmpiricalNativeProvider:
        if len(records) != 1 or not isinstance(records[0], EmpiricalNativeConfig):
            raise ValueError("native empirical factory requires its exact configuration")
        ports = {p.port_key: p.port for p in platform_ports}
        if tuple(sorted(ports)) != self.binding.required_platform_port_keys or len(ports) != len(
            platform_ports
        ):
            raise ValueError("native empirical factory port roster differs")
        source = ports["reactor-empirical-native-source-input"]
        if (
            not callable(
                getattr(
                    ports["reactor-empirical-native-payload-reader"], "read_candidate_payload", None
                )
            )
            or not isinstance(source, ExternalInputPayload)
            or not callable(
                getattr(
                    ports["reactor-empirical-native-dependency-reader"], "read_dependency", None
                )
            )
        ):
            raise TypeError("native empirical factory port contract differs")
        if not callable(
            getattr(ports["reactor-empirical-native-resource-guard"], "task", None)
        ) or not callable(
            getattr(ports["reactor-empirical-native-control-custody"], "open_control_store", None)
        ):
            raise TypeError("empirical resource/control ports differ")
        return EmpiricalNativeProvider(
            registry,
            records[0],
            source,
            cast(DependencyCustodyReader, ports["reactor-empirical-native-dependency-reader"]),
            cast(ControlCustodyPort, ports["reactor-empirical-native-control-custody"]),
            cast(CandidatePayloadReader, ports["reactor-empirical-native-payload-reader"]),
            limits=cast(EmpiricalResourceGuard, ports["reactor-empirical-native-resource-guard"]),
        )


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    "executable-contribution.reactor-causal-response.native-acquisition",
    "1.0.0",
    (BINDING,),
)
EXECUTABLE_BINDING_FACTORIES = (EmpiricalNativeFactory(),)
EXECUTABLE_RECORD_TYPES = (EmpiricalNativeConfig,)
