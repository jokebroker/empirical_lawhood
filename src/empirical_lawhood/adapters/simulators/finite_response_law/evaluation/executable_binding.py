"Installed finite response-law evaluation factories; reconstruction is nonexecuting and grants no authority."

from dataclasses import dataclass, replace
from typing import cast

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.source_qualification import QualifiedSourceUseExperiment
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution, ExecutableCapabilityBinding, ExecutablePlatformPort
from empirical_lawhood.runtime.plans import ProtocolTemplate
from empirical_lawhood.runtime.source_qualification import QualifiedSourceUseSubstrateBinding
from empirical_lawhood.runtime.source_resolution import ContentAddressedInputResolver
from empirical_lawhood.runtime.providers import CampaignRuntimeProvider
from empirical_lawhood.adapters.composition.discovery import executable
from empirical_lawhood.adapters.methods.law_assessment import CandidatePayloadReader
from empirical_lawhood.adapters.methods.finite_response_law.native_provider import FiniteResponseLawNativeMethodProvider
from empirical_lawhood.adapters.methods.finite_response_law.evaluation_completion import FiniteResponseLawEvaluationCompletionProvider
from empirical_lawhood.adapters.methods.finite_response_law.assigned_native_records import FiniteResponseLawAssignedEvaluationProjectionConfig, FiniteResponseLawAssignedEvaluationCompletionConfig
from empirical_lawhood.adapters.methods.finite_response_law.control_provider import FiniteResponseLawControlProvider
from empirical_lawhood.adapters.methods.finite_response_law.control_records import FiniteResponseLawControlConfig
from empirical_lawhood.adapters.methods.finite_response_law.evaluation_results import FiniteResponseLawEvaluationRevealConfig
from empirical_lawhood.adapters.methods.finite_response_law.evaluation_provider import FiniteResponseLawEvaluationRevealProvider
from empirical_lawhood.adapters.methods.finite_response_law.control_ports import CONTROL_RUNTIME_PORT, SOURCE_CONTROL_PORT, SEALED_CONTROL_PORT, REVEAL_CONTROL_PORT, FiniteResponseLawControlRuntimePort
from empirical_lawhood.adapters.methods.finite_response_law.method_provider import CANDIDATE_PAYLOAD_PORT, INPUT_RESOLVER_PORT
from ..assigned_contracts import FiniteResponseLawAssignedEvaluationConfig
from ..evaluation_provider import FiniteResponseLawEvaluationSourceProvider
from ..protocol import validate_source_records
from .protocol import evaluation_protocol_steps
from .discovery import NAMESPACE, SOURCE_CAPABILITY, SOURCE_COMPONENTS, PROJECTION_CAPABILITY, PROJECTION_COMPONENTS, EVALUATION_CAPABILITY, EVALUATION_COMPONENTS, CONTROL_CAPABILITY, CONTROL_COMPONENTS, REVEAL_CAPABILITY, REVEAL_COMPONENTS

SOURCE_BINDING = replace(
    executable(
        SOURCE_CAPABILITY,
        SOURCE_COMPONENTS,
        (
            FiniteResponseLawAssignedEvaluationConfig,
            QualifiedSourceUseExperiment,
            QualifiedSourceUseSubstrateBinding,
        ),
    ),
    required_platform_port_keys=(SOURCE_CONTROL_PORT,),
)
PROJECTION_BINDING = executable(
    PROJECTION_CAPABILITY, PROJECTION_COMPONENTS, (FiniteResponseLawAssignedEvaluationProjectionConfig,)
)
EVALUATION_BINDING = replace(
    executable(EVALUATION_CAPABILITY, EVALUATION_COMPONENTS, (FiniteResponseLawAssignedEvaluationCompletionConfig,)),
    required_platform_port_keys=(SEALED_CONTROL_PORT,),
)
CONTROL_BINDING = replace(
    executable(CONTROL_CAPABILITY, CONTROL_COMPONENTS, (FiniteResponseLawControlConfig,)),
    required_platform_port_keys=tuple(
        sorted((CONTROL_RUNTIME_PORT, CANDIDATE_PAYLOAD_PORT, INPUT_RESOLVER_PORT))
    ),
)
REVEAL_BINDING = replace(
    executable(REVEAL_CAPABILITY, REVEAL_COMPONENTS, (FiniteResponseLawEvaluationRevealConfig,)),
    required_platform_port_keys=(REVEAL_CONTROL_PORT,),
)


@dataclass(frozen=True, slots=True)
class FiniteResponseLawEvaluationSourceFactory:
    binding: ExecutableCapabilityBinding = SOURCE_BINDING

    def _records(
        self, records: tuple[CanonicalRecord, ...]
    ) -> tuple[
        FiniteResponseLawAssignedEvaluationConfig,
        QualifiedSourceUseExperiment,
        QualifiedSourceUseSubstrateBinding,
    ]:
        by_type = {type(r): r for r in records}
        required = {
            FiniteResponseLawAssignedEvaluationConfig,
            QualifiedSourceUseExperiment,
            QualifiedSourceUseSubstrateBinding,
        }
        if (
            self.binding != SOURCE_BINDING
            or len(by_type) != len(records)
            or not required <= set(by_type)
        ):
            raise ValueError("Finite response-law evaluation source requires exact current decoded records")
        source = cast(FiniteResponseLawAssignedEvaluationConfig, by_type[FiniteResponseLawAssignedEvaluationConfig])
        carrier = cast(QualifiedSourceUseExperiment, by_type[QualifiedSourceUseExperiment])
        association = cast(
            QualifiedSourceUseSubstrateBinding, by_type[QualifiedSourceUseSubstrateBinding]
        )
        validate_source_records(
            source,
            carrier,
            association,
            ObjectIdentity.from_record(SOURCE_BINDING.binding_id, SOURCE_BINDING),
        )
        return source, carrier, association

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> FiniteResponseLawEvaluationSourceProvider:
        if len(records) != 3 or tuple(p.port_key for p in platform_ports) != (SOURCE_CONTROL_PORT,):
            raise ValueError("Finite response-law evaluation source requires its durable-control runtime port")
        source, carrier, _ = self._records(records)
        runtime = cast(FiniteResponseLawControlRuntimePort, platform_ports[0].port)
        provider = FiniteResponseLawEvaluationSourceProvider(registry, SOURCE_CAPABILITY, source, carrier, None)
        provider.bind_control_store(runtime.prepared_store)
        return provider

    def expand_parameterised_protocol(
        self, *, records: tuple[CanonicalRecord, ...], template: ProtocolTemplate
    ) -> ProtocolTemplate:
        by_type = {type(r): r for r in records}
        if len(records) != 7 or set(by_type) != {
            FiniteResponseLawAssignedEvaluationConfig,
            QualifiedSourceUseExperiment,
            QualifiedSourceUseSubstrateBinding,
            FiniteResponseLawAssignedEvaluationProjectionConfig,
            FiniteResponseLawAssignedEvaluationCompletionConfig,
            FiniteResponseLawControlConfig,
            FiniteResponseLawEvaluationRevealConfig,
        }:
            raise ValueError("Finite response-law evaluation expansion requires the complete source and causal-control records")
        source, carrier, _ = self._records(records)
        return replace(
            template,
            template_id=f"{template.template_id}.{carrier.fingerprint()[:16]}",
            steps=evaluation_protocol_steps(
                source,
                carrier,
                cast(FiniteResponseLawAssignedEvaluationProjectionConfig, by_type[FiniteResponseLawAssignedEvaluationProjectionConfig]),
                cast(FiniteResponseLawAssignedEvaluationCompletionConfig, by_type[FiniteResponseLawAssignedEvaluationCompletionConfig]),
                cast(FiniteResponseLawControlConfig, by_type[FiniteResponseLawControlConfig]),
                cast(FiniteResponseLawEvaluationRevealConfig, by_type[FiniteResponseLawEvaluationRevealConfig]),
            ),
            requires_model_set=True,
            requests_controller=True,
            nonactuating=False,
        )


@dataclass(frozen=True, slots=True)
class FiniteResponseLawEvaluationMethodFactory:
    binding: ExecutableCapabilityBinding

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> CampaignRuntimeProvider:
        if len(records) != 1:
            raise ValueError("Finite response-law evaluation method requires exactly its issued config")
        config = records[0]
        if self.binding == REVEAL_BINDING and type(config) is FiniteResponseLawEvaluationRevealConfig:
            if tuple(p.port_key for p in platform_ports) != (REVEAL_CONTROL_PORT,):
                raise ValueError("Finite response-law evaluation reveal lacks its authenticated runtime")
            return FiniteResponseLawEvaluationRevealProvider(
                registry,
                REVEAL_CAPABILITY,
                config,
                cast(FiniteResponseLawControlRuntimePort, platform_ports[0].port),
            )
        if self.binding == CONTROL_BINDING and type(config) is FiniteResponseLawControlConfig:
            if (
                tuple(p.port_key for p in platform_ports)
                != CONTROL_BINDING.required_platform_port_keys
            ):
                raise ValueError(
                    "Finite response-law evaluation control method lacks its exact authenticated runtime/payload/input ports"
                )
            ports = {p.port_key: p.port for p in platform_ports}
            return FiniteResponseLawControlProvider(
                registry,
                CONTROL_CAPABILITY,
                config,
                cast(FiniteResponseLawControlRuntimePort, ports[CONTROL_RUNTIME_PORT]),
                cast(CandidatePayloadReader, ports[CANDIDATE_PAYLOAD_PORT]),
                cast(ContentAddressedInputResolver, ports[INPUT_RESOLVER_PORT]),
            )
        if self.binding == EVALUATION_BINDING and type(config) is FiniteResponseLawAssignedEvaluationCompletionConfig:
            if tuple(p.port_key for p in platform_ports) != (SEALED_CONTROL_PORT,):
                raise ValueError("Finite response-law evaluation sealed closure lacks its actual durable runtime")
            provider = FiniteResponseLawEvaluationCompletionProvider(registry, EVALUATION_CAPABILITY, config)
            provider.bind_control_runtime(cast(FiniteResponseLawControlRuntimePort, platform_ports[0].port))
            return provider
        if platform_ports:
            raise ValueError("Finite response-law evaluation measurement reducer has no additional platform ports")
        if self.binding == PROJECTION_BINDING and type(config) is FiniteResponseLawAssignedEvaluationProjectionConfig:
            return FiniteResponseLawNativeMethodProvider(registry, PROJECTION_CAPABILITY, config)
        raise ValueError("Finite response-law evaluation method changes its closed current executable binding")


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    f"executable-contribution.{NAMESPACE}",
    "1.0.0",
    tuple(
        sorted(
            (
                SOURCE_BINDING,
                PROJECTION_BINDING,
                EVALUATION_BINDING,
                CONTROL_BINDING,
                REVEAL_BINDING,
            ),
            key=lambda b: b.binding_id,
        )
    ),
)
EXECUTABLE_BINDING_FACTORIES = (
    FiniteResponseLawEvaluationSourceFactory(),
    FiniteResponseLawEvaluationMethodFactory(PROJECTION_BINDING),
    FiniteResponseLawEvaluationMethodFactory(EVALUATION_BINDING),
    FiniteResponseLawEvaluationMethodFactory(CONTROL_BINDING),
    FiniteResponseLawEvaluationMethodFactory(REVEAL_BINDING),
)
_RECORD_TYPES: tuple[type[CanonicalRecord], ...] = (
    FiniteResponseLawAssignedEvaluationConfig,
    QualifiedSourceUseExperiment,
    QualifiedSourceUseSubstrateBinding,
    FiniteResponseLawAssignedEvaluationProjectionConfig,
    FiniteResponseLawAssignedEvaluationCompletionConfig,
    FiniteResponseLawControlConfig,
    FiniteResponseLawEvaluationRevealConfig,
)
EXECUTABLE_RECORD_TYPES = tuple(sorted(_RECORD_TYPES, key=lambda t: t.SCHEMA))
