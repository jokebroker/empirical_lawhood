"causal response prediction factory and exact expansion through the generic qualification topology."

from dataclasses import dataclass, replace
from math import ceil
from typing import cast
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.source_qualification import FreshSourceQualificationExperiment
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution, ExecutableCapabilityBinding, ExecutablePlatformPort, ExecutableRecordTypeBinding
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.plans import (
    BarrierKind,
    ProtocolStepTemplate,
    ProtocolTemplate,
    ScientificStage,
)
from empirical_lawhood.runtime.source_qualification import FreshSourceQualificationSubstrateBinding, derive_source_qualification_topology
from empirical_lawhood.adapters.composition.discovery import executable
from empirical_lawhood.adapters.simulators.response_geometry_prospective.executable_binding import ASSAY_SOURCE_BINDING as SHARED_QUALIFICATION_CODEC_OWNER
from empirical_lawhood.adapters.composition.protocol_helpers import capability_config_ref as _config_ref, protocol_outputs as _outputs
from empirical_lawhood.adapters.methods.causal_response.extension_bundle import PROJECTION_CAPABILITY, EVALUATION_CAPABILITY
from empirical_lawhood.adapters.methods.causal_response.records import CausalResponseProjectionConfig, CausalResponseEvaluationConfig, CausalResponseViewObservation, CausalResponseEvaluation, CausalResponseCommittedPrediction
from empirical_lawhood.adapters.simulators.prepared_response.source_outputs import PreparedNativeTaskResult
from empirical_lawhood.adapters.simulators.prepared_response.native_pair import NATIVE_PAIR_SCHEMA
from .contracts import CausalResponseNativeConfig, native_invocations
from .extension_bundle import SOURCE_CAPABILITY, SOURCE_COMPONENTS, SOURCE_RECORDS
from .provider import CausalResponseSourceProvider
from .roster import REFERENCE_CLOCK, NATIVE_RECEIVERS, native_declarations, substrate_binding

_BASE = executable(SOURCE_CAPABILITY, SOURCE_COMPONENTS, (CausalResponseNativeConfig,))
_BASE = replace(
    _BASE,
    issued_decoder_registrations=tuple(
        replace(d, maximum_payload_bytes=16 * 1024**2)
        for d in _BASE.issued_decoder_registrations
    ),
)
_SHARED_SCHEMAS = (
    FreshSourceQualificationExperiment.SCHEMA,
    FreshSourceQualificationSubstrateBinding.SCHEMA,
)
_SHARED_COMPONENTS = tuple(
    component
    for component in SHARED_QUALIFICATION_CODEC_OWNER.discovery_components
    if component.input_schema_ids in tuple((schema,) for schema in _SHARED_SCHEMAS)
    and component.output_schema_ids == component.input_schema_ids
)
SOURCE_BINDING = replace(
    _BASE,
    discovery_components=tuple(
        sorted(
            (*_BASE.discovery_components, *_SHARED_COMPONENTS),
            key=lambda value: value.registration_id,
        )
    ),
    accepted_config_types=tuple(
        sorted(
            (
                ExecutableRecordTypeBinding(value.SCHEMA, value.VERSION)
                for value in SOURCE_RECORDS
            ),
            key=lambda value: value.type_id,
        )
    ),
    required_issued_payload_schemas=tuple(
        sorted(value.SCHEMA for value in SOURCE_RECORDS)
    ),
    codec_registration_identities=tuple(
        sorted(
            (
                *_BASE.codec_registration_identities,
                *(
                    ObjectIdentity.from_record(value.registration_id, value)
                    for value in _SHARED_COMPONENTS
                ),
            ),
            key=lambda value: value.object_id,
        )
    ),
    issued_decoder_registrations=tuple(
        sorted(
            (
                *_BASE.issued_decoder_registrations,
                *(
                    value
                    for value in SHARED_QUALIFICATION_CODEC_OWNER.issued_decoder_registrations
                    if value.payload_schema in _SHARED_SCHEMAS
                ),
            ),
            key=lambda value: value.registration_id,
        )
    ),
)


def validate_source_records(
    spec: CausalResponseNativeConfig,
    carrier: FreshSourceQualificationExperiment,
    association: FreshSourceQualificationSubstrateBinding,
) -> None:
    binding = ObjectIdentity.from_record(SOURCE_BINDING.binding_id, SOURCE_BINDING)
    source_owner = ObjectIdentity.from_record(
        SOURCE_CAPABILITY.capability_key, SOURCE_CAPABILITY
    )
    declarations = native_declarations(spec)
    if (
        spec.recipe.native_implementation != source_owner
        or carrier.source_capability != source_owner
        or carrier.source_config != ObjectIdentity.from_record(spec.spec_id, spec)
        or (carrier.physical_units, carrier.segments, carrier.views)
        != (declarations.units, declarations.segments, declarations.views)
        or carrier.receiver_ids != NATIVE_RECEIVERS
        or carrier.clock_ids != (REFERENCE_CLOCK,)
        or not carrier.extension_set_id.endswith(".source-qualification")
        or carrier.evaluator_task_id
        != f"{carrier.extension_set_id.removesuffix('.source-qualification')}.evaluate"
        or association != substrate_binding(spec, carrier, binding)
    ):
        raise ValueError(
            "causal response prediction source records change native units/actions/views/clock/provider lineage"
        )


@dataclass(frozen=True, slots=True)
class CausalResponseSourceFactory:
    binding: ExecutableCapabilityBinding = SOURCE_BINDING

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> CausalResponseSourceProvider:
        by_type = {type(value): value for value in records}
        if (
            self.binding != SOURCE_BINDING
            or platform_ports
            or len(records) != 3
            or set(by_type) != set(SOURCE_RECORDS)
        ):
            raise ValueError(
                "causal response prediction source factory requires its three exact issued records"
            )
        spec = cast(CausalResponseNativeConfig, by_type[CausalResponseNativeConfig])
        carrier = cast(
            FreshSourceQualificationExperiment, by_type[FreshSourceQualificationExperiment]
        )
        association = cast(
            FreshSourceQualificationSubstrateBinding,
            by_type[FreshSourceQualificationSubstrateBinding],
        )
        validate_source_records(spec, carrier, association)
        return CausalResponseSourceProvider(registry, SOURCE_CAPABILITY, spec)

    def expand_parameterised_protocol(
        self, *, records: tuple[CanonicalRecord, ...], template: ProtocolTemplate
    ) -> ProtocolTemplate:
        by_type = {type(value): value for value in records}
        if (
            self.binding != SOURCE_BINDING
            or len(records) != 5
            or set(by_type)
            != {*SOURCE_RECORDS, CausalResponseProjectionConfig, CausalResponseEvaluationConfig}
        ):
            raise ValueError(
                "causal response prediction expansion requires its exact source/carrier/method records"
            )
        spec = cast(CausalResponseNativeConfig, by_type[CausalResponseNativeConfig])
        carrier = cast(
            FreshSourceQualificationExperiment, by_type[FreshSourceQualificationExperiment]
        )
        association = cast(
            FreshSourceQualificationSubstrateBinding,
            by_type[FreshSourceQualificationSubstrateBinding],
        )
        projection = cast(CausalResponseProjectionConfig, by_type[CausalResponseProjectionConfig])
        evaluation = cast(CausalResponseEvaluationConfig, by_type[CausalResponseEvaluationConfig])
        validate_source_records(spec, carrier, association)
        if (
            projection.native_spec != spec
            or evaluation.projection != projection
            or carrier.projection_config
            != ObjectIdentity.from_record(projection.config_id, projection)
            or carrier.evaluator_config
            != ObjectIdentity.from_record(evaluation.config_id, evaluation)
            or carrier.projection_capability
            != ObjectIdentity.from_record(
                PROJECTION_CAPABILITY.capability_key, PROJECTION_CAPABILITY
            )
            or carrier.evaluator_capability
            != ObjectIdentity.from_record(
                EVALUATION_CAPABILITY.capability_key, EVALUATION_CAPABILITY
            )
        ):
            raise ValueError(
                "causal response prediction expansion changes its exact projection/evaluation owners"
            )
        source_id = ObjectIdentity.from_record(spec.spec_id, spec)
        owners = {
            ObjectIdentity.from_record(manifest.capability_key, manifest): (
                manifest,
                config,
                identity,
            )
            for manifest, config, identity in (
                (SOURCE_CAPABILITY, spec, source_id),
                (PROJECTION_CAPABILITY, projection, carrier.projection_config),
                (EVALUATION_CAPABILITY, evaluation, carrier.evaluator_config),
            )
        }
        outputs = {
            ScientificStage.PREPARE: _outputs(
                (
                    ("native-result", PreparedNativeTaskResult.SCHEMA),
                    ("committed-prediction", CausalResponseCommittedPrediction.SCHEMA),
                    ("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
                ),
                ("native-observations", NATIVE_PAIR_SCHEMA),
            ),
            ScientificStage.TRANSFORM: _outputs(
                (
                    ("view-report", CausalResponseViewObservation.SCHEMA),
                    ("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
                )
            ),
            ScientificStage.EVALUATE: _outputs(
                (
                    ("evaluation", CausalResponseEvaluation.SCHEMA),
                    ("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
                    ("scientific-adjudication", ScientificAdjudicationRecord.SCHEMA),
                )
            ),
        }
        native = {task.task_id: task for task in native_invocations(spec)}
        stages = derive_source_qualification_topology(carrier).stages
        by_id = {stage.task_id: stage for stage in stages}
        steps = []
        for stage in stages:
            manifest, config, identity = owners[stage.owner]
            if stage.config != identity:
                raise ValueError("causal response prediction topology changes its frozen stage config")
            scan = len(config.canonical_bytes()) * (
                2
                if stage.stage is ScientificStage.PREPARE
                and not stage.dependency_task_ids
                else 1
            )
            scan += sum(
                owners[by_id[key].owner][0].resource_ceiling.output_bytes
                for key in stage.dependency_task_ids
            )
            if scan > manifest.resource_ceiling.source_scan_bytes:
                raise ValueError(
                    "causal response prediction complete predecessor roster exceeds its installed scan ceiling"
                )
            # Conservative inherited numerical cost; complete route/resource admission is separate.
            wall = (
                ceil(2 + native[stage.task_id].maximum_native_updates * 0.004)
                if stage.stage is ScientificStage.PREPARE
                else manifest.resource_ceiling.wall_time_seconds
            )
            steps.append(
                ProtocolStepTemplate(
                    stage.task_id,
                    stage.stage,
                    manifest.capability_key,
                    manifest.capability_version,
                    _config_ref(config, identity, manifest),
                    stage.dependency_task_ids,
                    outputs[stage.stage],
                    manifest.permissions,
                    OutcomeAccess.EVALUATION_REVEALED
                    if stage.stage is ScientificStage.EVALUATE
                    else OutcomeAccess.EVALUATION_SEALED,
                    VisibilityCeiling.OUTCOME_VISIBLE
                    if stage.stage is ScientificStage.EVALUATE
                    else VisibilityCeiling.DEVELOPMENT_ONLY,
                    replace(
                        manifest.resource_ceiling,
                        source_scan_bytes=scan,
                        wall_time_seconds=wall,
                    ),
                    (),
                    BarrierKind.REVEAL
                    if stage.stage is ScientificStage.EVALUATE
                    else BarrierKind.FREEZE
                    if stage.stage is ScientificStage.TRANSFORM
                    else BarrierKind.NONE,
                    1,
                    (
                        f"{carrier.evaluator_task_id.removesuffix('.evaluate')}.single-terminal",
                    )
                    if stage.stage is ScientificStage.EVALUATE
                    else (f"custody.{stage.task_id}",),
                )
            )
        if len(steps) != 3073:
            raise ValueError(
                "causal response prediction expansion changes its complete production task census"
            )
        return replace(
            template,
            template_id=f"{template.template_id}.{carrier.fingerprint()[:16]}",
            steps=tuple(sorted(steps, key=lambda value: value.step_id)),
            requires_model_set=False,
            requests_controller=False,
            nonactuating=True,
        )


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    "executable-contribution.causal-response-prediction.source",
    "1.0.0",
    (SOURCE_BINDING,),
)
EXECUTABLE_BINDING_FACTORIES = (CausalResponseSourceFactory(),)
EXECUTABLE_RECORD_TYPES = (CausalResponseNativeConfig,)
