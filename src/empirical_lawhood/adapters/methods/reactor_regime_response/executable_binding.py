"Static reconstruction of the regime-response study fit/seal method provider."

from dataclasses import dataclass, replace
from typing import cast

from empirical_lawhood.adapters.methods.reactor_causal_response.provider import DependencyCustodyReader
from empirical_lawhood.adapters.methods.law_assessment import (
    CandidatePayloadPublisher,
    CandidatePayloadReader,
)
from empirical_lawhood.adapters.methods.reactor_causal_response.resource_contract import EmpiricalResourceGuard
from empirical_lawhood.adapters.methods.reactor_causal_response.campaign_records import EmpiricalStudySource
from empirical_lawhood.adapters.composition.discovery import executable
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution, ExecutableCapabilityBinding, ExecutablePlatformPort

from .config import ReactorRegimeResponseDesign
from .extension_bundle import CAPABILITY, COMPONENTS
from .provider import RegimeDMethodBinding, RegimeDMethodCustodyPort, RegimeMethodProvider
from .prior import RegimePriorArtifact
from .continuation import RegimeCContinuation, RegimeRetainedInputsPort
from .preassay_readout import RegimeReferenceComparators

BINDING = replace(
    executable(CAPABILITY, COMPONENTS, (ReactorRegimeResponseDesign,)),
    required_platform_port_keys=(
        "candidate-payload-publisher",
        "candidate-payload-reader",
        "reactor-regime-method-control-custody",
        "reactor-regime-method-d-binding",
        "reactor-regime-method-dependency-reader",
        "reactor-regime-method-prior-artifacts",
        "reactor-regime-method-reference-comparators",
        "reactor-regime-method-resource-guard",
        "reactor-regime-method-retained-inputs",
        "reactor-regime-method-source",
    ),
)


@dataclass(frozen=True, slots=True)
class RegimeMethodFactory:
    binding: ExecutableCapabilityBinding = BINDING

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> RegimeMethodProvider:
        if len(records) != 1 or not isinstance(
            records[0], ReactorRegimeResponseDesign
        ):
            raise ValueError("reactor method factory requires its exact design")
        ports = {port.port_key: port.port for port in platform_ports}
        if tuple(sorted(ports)) != self.binding.required_platform_port_keys or len(
            ports
        ) != len(platform_ports):
            raise ValueError("reactor method factory port roster differs")
        if not callable(
            getattr(
                ports["reactor-regime-method-dependency-reader"],
                "read_dependency",
                None,
            )
        ) or not callable(
            getattr(ports["reactor-regime-method-resource-guard"], "task", None)
        ):
            raise TypeError("reactor method factory lacks custody or task limits")
        if not callable(
            getattr(
                ports["candidate-payload-publisher"], "publish_candidate_payload", None
            )
        ):
            raise TypeError("reactor method factory lacks candidate publication")
        if not callable(
            getattr(ports["candidate-payload-reader"], "read_candidate_payload", None)
        ):
            raise TypeError("reactor method factory lacks candidate payload custody")
        priors = ports["reactor-regime-method-prior-artifacts"]
        if not isinstance(priors, tuple) or any(
            not isinstance(value, RegimePriorArtifact) for value in priors
        ):
            raise TypeError("reactor method factory prior B package custody differs")
        d_binding = ports["reactor-regime-method-d-binding"]
        d_source = ports["reactor-regime-method-source"]
        d_custody = ports["reactor-regime-method-control-custody"]
        references = ports["reactor-regime-method-reference-comparators"]
        retained = ports["reactor-regime-method-retained-inputs"]
        if (
            not callable(getattr(retained, "create_source", None))
            or not hasattr(retained, "continuation")
            or retained.continuation is not None
            and not isinstance(retained.continuation, RegimeCContinuation)
        ):
            raise TypeError("reactor method retained-input port lacks its declaration")
        if references is not None and not isinstance(
            references, RegimeReferenceComparators
        ):
            raise TypeError("retained comparator port lacks pinned canonical operands")
        if d_binding is not None and not isinstance(d_binding, RegimeDMethodBinding):
            raise TypeError("reactor method D authority binding differs")
        if d_source is not None and not isinstance(d_source, EmpiricalStudySource):
            raise TypeError("reactor method D source differs")
        if d_custody is not None and (
            not callable(getattr(d_custody, "open_prepared_store", None))
            or not callable(getattr(d_custody, "event_clock", None))
        ):
            raise TypeError("reactor method D prepared-event custody differs")
        return RegimeMethodProvider(
            registry,
            records[0],
            cast(
                DependencyCustodyReader,
                ports["reactor-regime-method-dependency-reader"],
            ),
            cast(EmpiricalResourceGuard, ports["reactor-regime-method-resource-guard"]),
            cast(CandidatePayloadPublisher, ports["candidate-payload-publisher"]),
            cast(CandidatePayloadReader, ports["candidate-payload-reader"]),
            cast(tuple[RegimePriorArtifact, ...], priors),
            d_binding,
            d_source,
            cast(RegimeDMethodCustodyPort | None, d_custody),
            references,
            cast(RegimeRetainedInputsPort, retained),
        )


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    "executable-contribution.terminal-bench-science-regime.methods",
    "1.0.0",
    (BINDING,),
)
EXECUTABLE_BINDING_FACTORIES = (RegimeMethodFactory(),)
EXECUTABLE_RECORD_TYPES = (ReactorRegimeResponseDesign,)
