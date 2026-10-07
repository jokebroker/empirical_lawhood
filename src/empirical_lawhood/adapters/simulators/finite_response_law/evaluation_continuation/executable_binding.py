"""Installed continuation factories over exact original-source compatibility."""

from dataclasses import dataclass, replace
from typing import cast

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.source_qualification import ProspectiveRetainedSourceUse
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution, ExecutableCapabilityBinding, ExecutablePlatformPort
from empirical_lawhood.runtime.plans import ProtocolTemplate
from empirical_lawhood.runtime.source_qualification import ProspectiveRetainedSourceUseBinding
from empirical_lawhood.runtime.source_resolution import ContentAddressedInputResolver
from empirical_lawhood.adapters.composition.discovery import executable
from empirical_lawhood.adapters.methods.finite_response_law.control_ports import SOURCE_CONTROL_PORT, FiniteResponseLawControlRuntimePort
from empirical_lawhood.adapters.methods.finite_response_law.control_records import FiniteResponseLawControlConfig
from empirical_lawhood.adapters.methods.finite_response_law.assigned_native_records import FiniteResponseLawAssignedEvaluationProjectionConfig, FiniteResponseLawAssignedEvaluationCompletionConfig
from empirical_lawhood.adapters.methods.finite_response_law.evaluation_results import FiniteResponseLawEvaluationRevealConfig
from ..assigned_contracts import FiniteResponseLawAssignedEvaluationConfig
from ..evaluation_retention import FiniteResponseLawEvaluationRetention
from ..evaluation.protocol import evaluation_protocol_steps
from ..roster import native_declarations, substrate_binding, Q_CLOCK, Q_RECEIVERS
from .discovery import NAMESPACE, SOURCE_CAPABILITY, SOURCE_COMPONENTS, IMPORT_CAPABILITY, IMPORT_COMPONENTS
from .provider import FiniteResponseLawContinuationSourceProvider, FiniteResponseLawPrefixImportProvider

IMPORT_RESOLVER_PORT = "finite-response-law.retained-prefix-input-resolver"
SOURCE_RECORDS = (
    FiniteResponseLawAssignedEvaluationConfig,
    ProspectiveRetainedSourceUse,
    ProspectiveRetainedSourceUseBinding,
)
SOURCE_BINDING = replace(
    executable(
        SOURCE_CAPABILITY,
        tuple(c for i, c in enumerate(SOURCE_COMPONENTS) if i != 3),
        SOURCE_RECORDS,
    ),
    required_platform_port_keys=(SOURCE_CONTROL_PORT,),
)
IMPORT_BINDING = replace(
    executable(IMPORT_CAPABILITY, IMPORT_COMPONENTS, (FiniteResponseLawEvaluationRetention,)),
    required_platform_port_keys=(IMPORT_RESOLVER_PORT,),
)


@dataclass(frozen=True, slots=True)
class FiniteResponseLawContinuationSourceFactory:
    binding: ExecutableCapabilityBinding = SOURCE_BINDING

    def _records(
        self, records: tuple[CanonicalRecord, ...]
    ) -> tuple[FiniteResponseLawAssignedEvaluationConfig, ProspectiveRetainedSourceUse]:
        by_type = {type(r): r for r in records}
        if len(by_type) != len(records) or not set(SOURCE_RECORDS) <= by_type.keys():
            raise ValueError("Finite response-law evaluation continuation requires its exact versioned source records")
        source = cast(FiniteResponseLawAssignedEvaluationConfig, by_type[FiniteResponseLawAssignedEvaluationConfig])
        carrier = cast(ProspectiveRetainedSourceUse, by_type[ProspectiveRetainedSourceUse])
        association = cast(
            ProspectiveRetainedSourceUseBinding, by_type[ProspectiveRetainedSourceUseBinding]
        )
        original = native_declarations(source)
        imported = {p.segment_id for p in carrier.retained_predecessors}
        segments = tuple(s for s in original.segments if s.segment_id not in imported)
        views = tuple(
            replace(v, segment_ids=tuple(k for k in v.segment_ids if k not in imported))
            for v in original.views
        )
        if (
            (carrier.physical_units, carrier.segments, carrier.views)
            != (original.units, segments, views)
            or carrier.source_capability
            != ObjectIdentity.from_record(SOURCE_CAPABILITY.capability_key, SOURCE_CAPABILITY)
            or carrier.source_config != ObjectIdentity.from_record(source.spec_id, source)
            or carrier.receiver_ids != Q_RECEIVERS
            or carrier.clock_ids != (Q_CLOCK,)
            or association
            != substrate_binding(
                source,
                carrier,
                ObjectIdentity.from_record(SOURCE_BINDING.binding_id, SOURCE_BINDING),
            )
        ):
            raise ValueError(
                "Finite response-law evaluation continuation changes original roots, native census or compatibility"
            )
        return source, carrier

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> FiniteResponseLawContinuationSourceProvider:
        if len(records) != 3 or tuple(p.port_key for p in platform_ports) != (SOURCE_CONTROL_PORT,):
            raise ValueError("Finite response-law evaluation continuation source lacks its exact installed runtime")
        source, carrier = self._records(records)
        return FiniteResponseLawContinuationSourceProvider(
            registry, source, carrier, cast(FiniteResponseLawControlRuntimePort, platform_ports[0].port)
        )

    def expand_parameterised_protocol(
        self, *, records: tuple[CanonicalRecord, ...], template: ProtocolTemplate
    ) -> ProtocolTemplate:
        required = {
            *SOURCE_RECORDS,
            FiniteResponseLawEvaluationRetention,
            FiniteResponseLawAssignedEvaluationProjectionConfig,
            FiniteResponseLawAssignedEvaluationCompletionConfig,
            FiniteResponseLawControlConfig,
            FiniteResponseLawEvaluationRevealConfig,
        }
        by_type = {type(r): r for r in records}
        if len(records) != 8 or set(by_type) != required:
            raise ValueError("Finite response-law evaluation continuation changes its full composition record census")
        source, carrier = self._records(records)
        retention = cast(FiniteResponseLawEvaluationRetention, by_type[FiniteResponseLawEvaluationRetention])
        if retention.source != source or retention.prefixes != carrier.retained_predecessors:
            raise ValueError("Finite response-law evaluation continuation substitutes retained origins")
        return replace(
            template,
            template_id=f"{template.template_id}.{carrier.fingerprint()[:16]}",
            steps=evaluation_protocol_steps(
                retention.source,
                carrier,
                cast(FiniteResponseLawAssignedEvaluationProjectionConfig, by_type[FiniteResponseLawAssignedEvaluationProjectionConfig]),
                cast(FiniteResponseLawAssignedEvaluationCompletionConfig, by_type[FiniteResponseLawAssignedEvaluationCompletionConfig]),
                cast(FiniteResponseLawControlConfig, by_type[FiniteResponseLawControlConfig]),
                cast(FiniteResponseLawEvaluationRevealConfig, by_type[FiniteResponseLawEvaluationRevealConfig]),
                retention,
            ),
            requires_model_set=True,
            requests_controller=True,
            nonactuating=False,
        )


@dataclass(frozen=True, slots=True)
class FiniteResponseLawPrefixImportFactory:
    binding: ExecutableCapabilityBinding = IMPORT_BINDING

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> FiniteResponseLawPrefixImportProvider:
        if (
            len(records) != 1
            or type(records[0]) is not FiniteResponseLawEvaluationRetention
            or tuple(p.port_key for p in platform_ports) != (IMPORT_RESOLVER_PORT,)
        ):
            raise ValueError("Finite response-law evaluation prefix import lacks exact retention/input bindings")
        return FiniteResponseLawPrefixImportProvider(
            registry, records[0], cast(ContentAddressedInputResolver, platform_ports[0].port)
        )


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    f"executable-contribution.{NAMESPACE}",
    "1.0.0",
    tuple(sorted((SOURCE_BINDING, IMPORT_BINDING), key=lambda b: b.binding_id)),
)
EXECUTABLE_BINDING_FACTORIES = (FiniteResponseLawContinuationSourceFactory(), FiniteResponseLawPrefixImportFactory())
EXECUTABLE_RECORD_TYPES = tuple(
    sorted(
        (*(r for r in SOURCE_RECORDS if r is not FiniteResponseLawAssignedEvaluationConfig), FiniteResponseLawEvaluationRetention),
        key=lambda t: t.SCHEMA,
    )
)
