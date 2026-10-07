"Static reconstruction of the regime-response study's native preparation/assay runner."

from dataclasses import dataclass, replace
from typing import cast

from empirical_lawhood.adapters.methods.reactor_causal_response.provider import DependencyCustodyReader
from empirical_lawhood.adapters.methods.reactor_causal_response.resource_contract import EmpiricalResourceGuard
from empirical_lawhood.adapters.methods.reactor_regime_response.prior import RegimePriorArtifact
from empirical_lawhood.adapters.methods.law_assessment import CandidatePayloadReader
from empirical_lawhood.adapters.methods.reactor_regime_response.continuation import RegimeCContinuation, RegimeRetainedInputsPort
from empirical_lawhood.adapters.composition.discovery import executable
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution, ExecutableCapabilityBinding, ExecutablePlatformPort
from empirical_lawhood.runtime.providers import ExternalInputPayload

from .config import ReactorRegimeNativeConfig
from .extension_bundle import CAPABILITY, COMPONENTS
from .provider import RegimeNativeProvider, RegimeControlCustodyPort

BINDING = replace(
    executable(CAPABILITY, COMPONENTS, (ReactorRegimeNativeConfig,)),
    required_platform_port_keys=(
        "reactor-regime-native-control-custody",
        "reactor-regime-native-dependency-reader",
        "reactor-regime-native-law-reader",
        "reactor-regime-native-prior-artifacts",
        "reactor-regime-native-resource-guard",
        "reactor-regime-native-retained-inputs",
        "reactor-regime-native-source-input",
    ),
)


@dataclass(frozen=True, slots=True)
class RegimeNativeFactory:
    binding: ExecutableCapabilityBinding = BINDING

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> RegimeNativeProvider:
        if len(records) != 1 or not isinstance(records[0], ReactorRegimeNativeConfig):
            raise ValueError("reactor native factory requires its exact config")
        ports = {port.port_key: port.port for port in platform_ports}
        if tuple(sorted(ports)) != self.binding.required_platform_port_keys or len(ports) != len(
            platform_ports
        ):
            raise ValueError("reactor native factory port roster differs")
        if not callable(
            getattr(ports["reactor-regime-native-dependency-reader"], "read_dependency", None)
        ) or not callable(getattr(ports["reactor-regime-native-resource-guard"], "task", None)):
            raise TypeError("reactor native factory lacks custody or task limits")
        if not isinstance(ports["reactor-regime-native-source-input"], ExternalInputPayload):
            raise TypeError("reactor native factory source input differs")
        control = ports["reactor-regime-native-control-custody"]
        law_reader = ports["reactor-regime-native-law-reader"]
        if (control is None) != (law_reader is None):
            raise TypeError("reactor native control custody/law reader must be paired")
        if control is not None and (
            not callable(getattr(control, "open_control_store", None))
            or not callable(getattr(control, "open_prepared_store", None))
            or not callable(getattr(control, "freeze_clock", None))
            or not isinstance(getattr(control, "authority", None), ObjectIdentity)
            or not isinstance(getattr(control, "resources", None), ObjectIdentity)
            or not isinstance(getattr(control, 'issued_study', None), ObjectIdentity)
            or not callable(getattr(law_reader, "read_candidate_payload", None))
        ):
            raise TypeError("reactor native finite controller ports differ")
        prior = ports["reactor-regime-native-prior-artifacts"]
        retained = ports["reactor-regime-native-retained-inputs"]
        if (not callable(getattr(retained, "create_source", None))
                or not hasattr(retained, "continuation")
                or retained.continuation is not None
                and not isinstance(retained.continuation, RegimeCContinuation)):
            raise TypeError("reactor native retained-input port lacks its declaration")
        if not isinstance(prior, tuple) or any(
            not isinstance(value, RegimePriorArtifact) for value in prior
        ):
            raise TypeError("reactor native factory prior-phase custody differs")
        return RegimeNativeProvider(
            registry,
            records[0],
            ports["reactor-regime-native-source-input"],
            cast(DependencyCustodyReader, ports["reactor-regime-native-dependency-reader"]),
            cast(EmpiricalResourceGuard, ports["reactor-regime-native-resource-guard"]),
            prior,
            cast(RegimeControlCustodyPort | None, control),
            cast(CandidatePayloadReader | None, law_reader),
            cast(RegimeRetainedInputsPort, retained),
        )


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    "executable-contribution.terminal-bench-science-regime.simulators",
    "1.0.0",
    (BINDING,),
)
EXECUTABLE_BINDING_FACTORIES = (RegimeNativeFactory(),)
EXECUTABLE_RECORD_TYPES = (ReactorRegimeNativeConfig,)
