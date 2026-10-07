"source qualification source factory and exact source-qualification protocol expansion."

from dataclasses import dataclass, replace
from hashlib import sha256
from math import ceil
from typing import cast

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.source_qualification import FreshSourceQualificationExperiment
from empirical_lawhood.planning.native_source import NativeCrossfitConfig, NativeCrossfitExperiment, NativeSourceProfile
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.capabilities import (
    CapabilityManifest,
    CapabilityRegistry,
)
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution, ExecutableCapabilityBinding, ExecutablePlatformPort, ExecutableRecordTypeBinding
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.plans import (
    BarrierKind,
    OutputTemplate,
    ProtocolStepTemplate,
    ProtocolTemplate,
    ScientificStage,
)
from empirical_lawhood.runtime.source_qualification import FreshSourceQualificationSubstrateBinding, derive_source_qualification_topology
from empirical_lawhood.runtime.response_experiment_ports import ResponseSubstrateBinding
from empirical_lawhood.adapters.composition.discovery import executable
from empirical_lawhood.adapters.simulators.response_geometry_prospective.executable_binding import ASSAY_SOURCE_BINDING as SHARED_QUALIFICATION_CODEC_OWNER
from empirical_lawhood.adapters.composition.protocol_helpers import capability_config_ref as _config_ref, protocol_outputs as _outputs
from empirical_lawhood.adapters.methods.prepared_response.extension_bundle import CALIBRATION_CALIBRATION_CAPABILITY, CALIBRATION_POLICY_CAPABILITY, CALIBRATION_PROJECTION_CAPABILITY, DEVELOPMENT_FIT_CAPABILITY, DEVELOPMENT_PROJECTION_CAPABILITY, DEVELOPMENT_SELECTION_CAPABILITY, SOURCE_QUALIFICATION_PROJECTION_CAPABILITY, SOURCE_QUALIFICATION_EVALUATION_CAPABILITY
from empirical_lawhood.adapters.methods.prepared_response.calibration_records import PreparedResponseCalibrationCalibratedLibrary, PreparedResponseCalibrationCalibrationConfig, PreparedResponseCalibrationProjectionConfig, PreparedResponseCalibrationViewProjection
from empirical_lawhood.adapters.methods.prepared_response.calibration_provider import PreparedResponseCalibrationCalibrationTask
from empirical_lawhood.adapters.methods.prepared_response.policy_decision import PreparedParentDecision, PreparedPolicyDecisionConfig
from empirical_lawhood.adapters.methods.prepared_response.development_fit import PreparedResponseDevelopmentFitConfig, PreparedResponseDevelopmentModelFit
from empirical_lawhood.adapters.methods.prepared_response.development_projection import PreparedResponseDevelopmentProjectionConfig, PreparedResponseDevelopmentViewProjection
from empirical_lawhood.adapters.methods.prepared_response.models import STRUCTURES
from empirical_lawhood.adapters.methods.prepared_response.development_selection import PreparedResponseDevelopmentNominalLibrary, PreparedResponseDevelopmentSelectionConfig
from empirical_lawhood.adapters.methods.prepared_response.development_benchmarks import PreparedResponseDevelopmentConstantGainReport
from empirical_lawhood.adapters.methods.prepared_response.development_provider import PreparedResponseDevelopmentSelectionTask
from empirical_lawhood.adapters.control.backbone_linked_campaign.executable_binding import (
    NATIVE_CROSSFIT_PLAN_EXECUTABLE_BINDING,
    NATIVE_RESPONSE_LAW_QUALIFICATION_PLAN_EXECUTABLE_BINDING,
    PARAMETERISED_RESPONSE_PLAN_EXECUTABLE_BINDING,
)
from empirical_lawhood.adapters.methods.prepared_response.qualification_records import PreparedResponseSourceQualificationProjectionConfig, PreparedResponseSourceQualificationViewObservation
from empirical_lawhood.adapters.methods.prepared_response.qualification import PreparedResponseSourceQualificationEvaluationConfig, PreparedResponseSourceQualificationEvaluation
from .contracts import PreparedNativeSpec, PreparedRoot
from .extension_bundle import CALIBRATION_SOURCE_CAPABILITY, CALIBRATION_SOURCE_COMPONENTS, DEVELOPMENT_SOURCE_CAPABILITY, DEVELOPMENT_SOURCE_COMPONENTS, SOURCE_QUALIFICATION_SOURCE_CAPABILITY, SOURCE_QUALIFICATION_SOURCE_COMPONENTS, SOURCE_QUALIFICATION_SOURCE_RECORDS
from .policy_native import CALIBRATION_POLICIES, PreparedResponseCalibrationNativeInvocation, PreparedResponseCalibrationNativeTaskResult, prepared_response_calibration_native_invocations
from .policy_provider import PreparedResponseCalibrationSourceProvider
from .native_tasks import prepared_static_native_invocations
from .native_pair import NATIVE_PAIR_SCHEMA
from .source_outputs import PreparedNativeTaskResult
from .provider import PreparedStaticSourceProvider
from .qualification_roster import SOURCE_QUALIFICATION_CLOCK, SOURCE_QUALIFICATION_RECEIVERS, prepared_response_source_qualification_native_declarations, prepared_response_source_qualification_substrate


_BASE = executable(SOURCE_QUALIFICATION_SOURCE_CAPABILITY, SOURCE_QUALIFICATION_SOURCE_COMPONENTS, (PreparedNativeSpec,))
_D_SOURCE_BASE = executable(
    DEVELOPMENT_SOURCE_CAPABILITY, DEVELOPMENT_SOURCE_COMPONENTS, (PreparedNativeSpec,)
)
_Q_SOURCE_DECODER_COMPONENTS = tuple(
    component
    for component in _BASE.discovery_components
    if component.registration_id.endswith(".decoder.0")
)
DEVELOPMENT_SOURCE_BINDING = replace(
    _D_SOURCE_BASE,
    discovery_components=tuple(
        sorted(
            (
                *(
                    component
                    for component in _D_SOURCE_BASE.discovery_components
                    if not component.registration_id.endswith(".decoder.0")
                ),
                *_Q_SOURCE_DECODER_COMPONENTS,
            ),
            key=lambda value: value.registration_id,
        )
    ),
    codec_registration_identities=_BASE.codec_registration_identities,
    issued_decoder_registrations=_BASE.issued_decoder_registrations,
)
_D_CARRIER_RECORD_TYPES = (
    NativeSourceProfile,
    NativeCrossfitConfig,
    NativeCrossfitExperiment,
    ResponseSubstrateBinding,
)
_D_CARRIER_SCHEMAS = tuple(value.SCHEMA for value in _D_CARRIER_RECORD_TYPES)
_D_CARRIER_CODEC_OWNERS = (
    NATIVE_CROSSFIT_PLAN_EXECUTABLE_BINDING,
    NATIVE_RESPONSE_LAW_QUALIFICATION_PLAN_EXECUTABLE_BINDING,
    PARAMETERISED_RESPONSE_PLAN_EXECUTABLE_BINDING,
)
_D_CARRIER_COMPONENTS = tuple(
    sorted(
        {
            component.registration_id: component
            for binding in _D_CARRIER_CODEC_OWNERS
            for component in binding.discovery_components
            if component.input_schema_ids
            in tuple((schema,) for schema in _D_CARRIER_SCHEMAS)
            and component.output_schema_ids == component.input_schema_ids
        }.values(),
        key=lambda value: value.registration_id,
    )
)
_D_CARRIER_DECODERS = tuple(
    sorted(
        {
            decoder.payload_schema: decoder
            for binding in _D_CARRIER_CODEC_OWNERS
            for decoder in binding.issued_decoder_registrations
            if decoder.payload_schema in _D_CARRIER_SCHEMAS
        }.values(),
        key=lambda value: value.registration_id,
    )
)
_CP_PROFILE_RECORD_TYPES = (NativeSourceProfile,)
_CP_PROFILE_SCHEMAS = tuple(value.SCHEMA for value in _CP_PROFILE_RECORD_TYPES)
_CP_PROFILE_COMPONENTS = tuple(
    sorted(
        {
            component.registration_id: component
            for binding in _D_CARRIER_CODEC_OWNERS
            for component in binding.discovery_components
            if component.input_schema_ids
            in tuple((schema,) for schema in _CP_PROFILE_SCHEMAS)
            and component.output_schema_ids == component.input_schema_ids
        }.values(),
        key=lambda value: value.registration_id,
    )
)
_CP_PROFILE_DECODERS = tuple(
    sorted(
        {
            decoder.payload_schema: decoder
            for binding in _D_CARRIER_CODEC_OWNERS
            for decoder in binding.issued_decoder_registrations
            if decoder.payload_schema in _CP_PROFILE_SCHEMAS
        }.values(),
        key=lambda value: value.registration_id,
    )
)
DEVELOPMENT_SOURCE_BINDING = replace(
    DEVELOPMENT_SOURCE_BINDING,
    discovery_components=tuple(
        sorted(
            (*DEVELOPMENT_SOURCE_BINDING.discovery_components, *_D_CARRIER_COMPONENTS),
            key=lambda value: value.registration_id,
        )
    ),
    accepted_config_types=tuple(
        sorted(
            (
                ExecutableRecordTypeBinding(
                    PreparedNativeSpec.SCHEMA, PreparedNativeSpec.VERSION
                ),
                *(
                    ExecutableRecordTypeBinding(value.SCHEMA, value.VERSION)
                    for value in _D_CARRIER_RECORD_TYPES
                ),
            ),
            key=lambda value: value.type_id,
        )
    ),
    required_issued_payload_schemas=tuple(
        sorted((PreparedNativeSpec.SCHEMA, *_D_CARRIER_SCHEMAS))
    ),
    codec_registration_identities=tuple(
        sorted(
            (
                *DEVELOPMENT_SOURCE_BINDING.codec_registration_identities,
                *(
                    ObjectIdentity.from_record(value.registration_id, value)
                    for value in _D_CARRIER_COMPONENTS
                ),
            ),
            key=lambda value: value.object_id,
        )
    ),
    issued_decoder_registrations=tuple(
        sorted(
            (*DEVELOPMENT_SOURCE_BINDING.issued_decoder_registrations, *_D_CARRIER_DECODERS),
            key=lambda value: value.registration_id,
        )
    ),
)
_CP_SOURCE_BASE = executable(
    CALIBRATION_SOURCE_CAPABILITY, CALIBRATION_SOURCE_COMPONENTS, (PreparedNativeSpec,)
)
CALIBRATION_SOURCE_BINDING = replace(
    _CP_SOURCE_BASE,
    discovery_components=tuple(
        sorted(
            (
                *(
                    component
                    for component in _CP_SOURCE_BASE.discovery_components
                    if not component.registration_id.endswith(".decoder.0")
                ),
                *_Q_SOURCE_DECODER_COMPONENTS,
                *_CP_PROFILE_COMPONENTS,
            ),
            key=lambda value: value.registration_id,
        )
    ),
    accepted_config_types=tuple(
        sorted(
            (
                ExecutableRecordTypeBinding(
                    PreparedNativeSpec.SCHEMA, PreparedNativeSpec.VERSION
                ),
                *(
                    ExecutableRecordTypeBinding(value.SCHEMA, value.VERSION)
                    for value in _CP_PROFILE_RECORD_TYPES
                ),
            ),
            key=lambda value: value.type_id,
        )
    ),
    required_issued_payload_schemas=tuple(
        sorted((PreparedNativeSpec.SCHEMA, *_CP_PROFILE_SCHEMAS))
    ),
    codec_registration_identities=tuple(
        sorted(
            (
                *_BASE.codec_registration_identities,
                *(
                    ObjectIdentity.from_record(value.registration_id, value)
                    for value in _CP_PROFILE_COMPONENTS
                ),
            ),
            key=lambda value: value.object_id,
        )
    ),
    issued_decoder_registrations=tuple(
        sorted(
            (*_BASE.issued_decoder_registrations, *_CP_PROFILE_DECODERS),
            key=lambda value: value.registration_id,
        )
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
SOURCE_QUALIFICATION_SOURCE_BINDING = replace(
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
                for value in SOURCE_QUALIFICATION_SOURCE_RECORDS
            ),
            key=lambda value: value.type_id,
        )
    ),
    required_issued_payload_schemas=tuple(
        sorted(value.SCHEMA for value in SOURCE_QUALIFICATION_SOURCE_RECORDS)
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


def validate_prepared_response_source_qualification_source_records(
    spec: PreparedNativeSpec,
    carrier: FreshSourceQualificationExperiment,
    association: FreshSourceQualificationSubstrateBinding,
) -> None:
    binding = ObjectIdentity.from_record(SOURCE_QUALIFICATION_SOURCE_BINDING.binding_id, SOURCE_QUALIFICATION_SOURCE_BINDING)
    source_owner = ObjectIdentity.from_record(
        SOURCE_QUALIFICATION_SOURCE_CAPABILITY.capability_key, SOURCE_QUALIFICATION_SOURCE_CAPABILITY
    )
    declarations = prepared_response_source_qualification_native_declarations(spec)
    if (
        spec.native_implementation != source_owner
        or carrier.source_capability != source_owner
        or carrier.source_config != ObjectIdentity.from_record(spec.spec_id, spec)
        or (carrier.physical_units, carrier.segments, carrier.views)
        != (declarations.units, declarations.segments, declarations.views)
        or carrier.receiver_ids != SOURCE_QUALIFICATION_RECEIVERS
        or carrier.clock_ids != (SOURCE_QUALIFICATION_CLOCK,)
        or not carrier.extension_set_id.endswith(".source-qualification")
        or carrier.evaluator_task_id
        != f"{carrier.extension_set_id.removesuffix('.source-qualification')}.evaluate"
        or association != prepared_response_source_qualification_substrate(spec, carrier, binding)
    ):
        raise ValueError(
            "prepared source qualification source records change native units/actions/views/clock/provider lineage"
        )


@dataclass(frozen=True, slots=True)
class PreparedResponseSourceQualificationSourceFactory:
    binding: ExecutableCapabilityBinding = SOURCE_QUALIFICATION_SOURCE_BINDING

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> PreparedStaticSourceProvider:
        by_type = {type(value): value for value in records}
        if (
            self.binding != SOURCE_QUALIFICATION_SOURCE_BINDING
            or platform_ports
            or len(records) != 3
            or set(by_type) != set(SOURCE_QUALIFICATION_SOURCE_RECORDS)
        ):
            raise ValueError(
                "prepared source qualification source factory requires its three exact issued records"
            )
        spec = cast(PreparedNativeSpec, by_type[PreparedNativeSpec])
        carrier = cast(
            FreshSourceQualificationExperiment, by_type[FreshSourceQualificationExperiment]
        )
        association = cast(
            FreshSourceQualificationSubstrateBinding,
            by_type[FreshSourceQualificationSubstrateBinding],
        )
        validate_prepared_response_source_qualification_source_records(spec, carrier, association)
        return PreparedStaticSourceProvider(registry, SOURCE_QUALIFICATION_SOURCE_CAPABILITY, spec)

    def expand_parameterised_protocol(
        self, *, records: tuple[CanonicalRecord, ...], template: ProtocolTemplate
    ) -> ProtocolTemplate:
        by_type = {type(value): value for value in records}
        if (
            self.binding != SOURCE_QUALIFICATION_SOURCE_BINDING
            or len(records) != 5
            or set(by_type)
            != {
                *SOURCE_QUALIFICATION_SOURCE_RECORDS,
                PreparedResponseSourceQualificationProjectionConfig,
                PreparedResponseSourceQualificationEvaluationConfig,
            }
        ):
            raise ValueError(
                "prepared source qualification expansion requires its exact source/carrier/method records"
            )
        spec = cast(PreparedNativeSpec, by_type[PreparedNativeSpec])
        carrier = cast(
            FreshSourceQualificationExperiment, by_type[FreshSourceQualificationExperiment]
        )
        association = cast(
            FreshSourceQualificationSubstrateBinding,
            by_type[FreshSourceQualificationSubstrateBinding],
        )
        projection = cast(
            PreparedResponseSourceQualificationProjectionConfig, by_type[PreparedResponseSourceQualificationProjectionConfig]
        )
        evaluation = cast(
            PreparedResponseSourceQualificationEvaluationConfig, by_type[PreparedResponseSourceQualificationEvaluationConfig]
        )
        validate_prepared_response_source_qualification_source_records(spec, carrier, association)
        if (
            projection.native_spec != spec
            or evaluation.projection != projection
            or carrier.projection_config
            != ObjectIdentity.from_record(projection.config_id, projection)
            or carrier.evaluator_config
            != ObjectIdentity.from_record(evaluation.config_id, evaluation)
            or carrier.projection_capability
            != ObjectIdentity.from_record(
                SOURCE_QUALIFICATION_PROJECTION_CAPABILITY.capability_key, SOURCE_QUALIFICATION_PROJECTION_CAPABILITY
            )
            or carrier.evaluator_capability
            != ObjectIdentity.from_record(
                SOURCE_QUALIFICATION_EVALUATION_CAPABILITY.capability_key, SOURCE_QUALIFICATION_EVALUATION_CAPABILITY
            )
        ):
            raise ValueError(
                "prepared source qualification expansion changes its exact projection/evaluation owners"
            )
        source_id = ObjectIdentity.from_record(spec.spec_id, spec)
        owners = {
            ObjectIdentity.from_record(manifest.capability_key, manifest): (
                manifest,
                config,
                identity,
            )
            for manifest, config, identity in (
                (SOURCE_QUALIFICATION_SOURCE_CAPABILITY, spec, source_id),
                (SOURCE_QUALIFICATION_PROJECTION_CAPABILITY, projection, carrier.projection_config),
                (SOURCE_QUALIFICATION_EVALUATION_CAPABILITY, evaluation, carrier.evaluator_config),
            )
        }
        outputs = {
            ScientificStage.PREPARE: _outputs(
                (
                    ("native-result", PreparedNativeTaskResult.SCHEMA),
                    ("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
                ),
                ("native-observations", NATIVE_PAIR_SCHEMA),
            ),
            ScientificStage.TRANSFORM: _outputs(
                (
                    ("view-report", PreparedResponseSourceQualificationViewObservation.SCHEMA),
                    ("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
                )
            ),
            ScientificStage.EVALUATE: _outputs(
                (
                    ("evaluation", PreparedResponseSourceQualificationEvaluation.SCHEMA),
                    ("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
                    ("scientific-adjudication", ScientificAdjudicationRecord.SCHEMA),
                )
            ),
        }
        native = {
            task.task_id: task for task in prepared_static_native_invocations(spec)
        }
        stages = derive_source_qualification_topology(carrier).stages
        by_id = {stage.task_id: stage for stage in stages}
        steps = []
        for stage in stages:
            manifest, config, identity = owners[stage.owner]
            if stage.config != identity:
                raise ValueError("prepared source qualification topology changes its frozen stage config")
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
                    "prepared source qualification complete predecessor roster exceeds its installed scan ceiling"
                )
            # A provisional planning ceiling, not a measured canary forecast.
            # Full-pipeline admission still requires the declared production canary.
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
        if len(steps) != 2977:
            raise ValueError(
                "prepared source qualification expansion changes its complete production task census"
            )
        return replace(
            template,
            template_id=f"{template.template_id}.{carrier.fingerprint()[:16]}",
            steps=tuple(sorted(steps, key=lambda value: value.step_id)),
            requires_model_set=False,
            requests_controller=False,
            nonactuating=True,
        )


@dataclass(frozen=True, slots=True)
class PreparedResponseDevelopmentSourceFactory:
    binding: ExecutableCapabilityBinding = DEVELOPMENT_SOURCE_BINDING

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> PreparedStaticSourceProvider:
        by_type = {type(value): value for value in records}
        core = {PreparedNativeSpec}
        full = {PreparedNativeSpec, *_D_CARRIER_RECORD_TYPES}
        if (
            self.binding != DEVELOPMENT_SOURCE_BINDING
            or platform_ports
            or len(by_type) != len(records)
            or set(by_type) not in (core, full)
        ):
            raise ValueError(
                "prepared dependent refinement source factory requires its exact dependent refinement issued records"
            )
        spec = cast(PreparedNativeSpec, by_type[PreparedNativeSpec])
        if spec.stage != 'development':
            raise ValueError("prepared dependent refinement source factory requires the dependent refinement native stage")
        if set(by_type) == full:
            profile = cast(NativeSourceProfile, by_type[NativeSourceProfile])
            qualification_config = cast(
                NativeCrossfitConfig, by_type[NativeCrossfitConfig]
            )
            extension = cast(
                NativeCrossfitExperiment,
                by_type[NativeCrossfitExperiment],
            )
            substrate = cast(ResponseSubstrateBinding, by_type[ResponseSubstrateBinding])
            if (
                profile.source_config != ObjectIdentity.from_record(spec.spec_id, spec)
                or qualification_config.source_pipeline_profile
                != ObjectIdentity.from_record(profile.profile_id, profile)
                or extension.identification_config != qualification_config
                or substrate.native_config
                != ObjectIdentity.from_record(spec.spec_id, spec)
                or substrate.installed_executable_binding
                != ObjectIdentity.from_record(
                    DEVELOPMENT_SOURCE_BINDING.binding_id, DEVELOPMENT_SOURCE_BINDING
                )
            ):
                raise ValueError(
                    "prepared dependent refinement source factory changes carrier or substrate lineage"
                )
        return PreparedStaticSourceProvider(registry, DEVELOPMENT_SOURCE_CAPABILITY, spec)

    def expand_parameterised_protocol(
        self, *, records: tuple[CanonicalRecord, ...], template: ProtocolTemplate
    ) -> ProtocolTemplate:
        by_type = {type(value): value for value in records}
        core_types = {
            PreparedNativeSpec,
            PreparedResponseDevelopmentProjectionConfig,
            PreparedResponseDevelopmentFitConfig,
            PreparedResponseDevelopmentSelectionConfig,
        }
        full_types = {
            *core_types,
            NativeSourceProfile,
            NativeCrossfitConfig,
            NativeCrossfitExperiment,
            ResponseSubstrateBinding,
        }
        if len(by_type) != len(records) or set(by_type) not in (core_types, full_types):
            raise ValueError(
                "prepared dependent refinement expansion requires its exact source/projection/fit records"
            )
        spec = cast(PreparedNativeSpec, by_type[PreparedNativeSpec])
        projection = cast(
            PreparedResponseDevelopmentProjectionConfig, by_type[PreparedResponseDevelopmentProjectionConfig]
        )
        fit = cast(PreparedResponseDevelopmentFitConfig, by_type[PreparedResponseDevelopmentFitConfig])
        selection = cast(
            PreparedResponseDevelopmentSelectionConfig, by_type[PreparedResponseDevelopmentSelectionConfig]
        )
        if (
            spec.stage != 'development'
            or projection.native_spec != spec
            or fit.projection != projection
            or selection.fit != fit
        ):
            raise ValueError(
                "prepared dependent refinement expansion changes its source or method lineage"
            )
        if set(by_type) == full_types:
            profile = cast(NativeSourceProfile, by_type[NativeSourceProfile])
            qualification_config = cast(
                NativeCrossfitConfig, by_type[NativeCrossfitConfig]
            )
            extension = cast(
                NativeCrossfitExperiment,
                by_type[NativeCrossfitExperiment],
            )
            substrate = cast(ResponseSubstrateBinding, by_type[ResponseSubstrateBinding])
            if (
                profile.source_config != ObjectIdentity.from_record(spec.spec_id, spec)
                or profile.physical_independent_unit_ids
                != tuple(root.physical_unit_id for root in spec.roots)
                or qualification_config.source_pipeline_profile
                != ObjectIdentity.from_record(profile.profile_id, profile)
                or qualification_config.projection_config
                != ObjectIdentity.from_record(projection.config_id, projection)
                or qualification_config.method_config != ObjectIdentity.from_record(fit.config_id, fit)
                or qualification_config.crossfit_plan != fit.crossfit_plan
                or extension.identification_config != qualification_config
                or substrate.native_config
                != ObjectIdentity.from_record(spec.spec_id, spec)
                or substrate.installed_executable_binding
                != ObjectIdentity.from_record(
                    DEVELOPMENT_SOURCE_BINDING.binding_id, DEVELOPMENT_SOURCE_BINDING
                )
            ):
                raise ValueError(
                    "prepared dependent refinement expansion changes its strict carrier or substrate"
                )
        native = prepared_static_native_invocations(spec)
        native_by_root = {
            root: tuple(task.task_id for task in native if task.root == root)
            for root in spec.roots
        }
        definitions: list[
            tuple[
                str,
                ScientificStage,
                CanonicalRecord,
                ObjectIdentity,
                CapabilityManifest,
                tuple[str, ...],
                tuple[OutputTemplate, ...],
                BarrierKind,
                int,
            ]
        ] = []
        source_identity = ObjectIdentity.from_record(spec.spec_id, spec)
        source_outputs = _outputs(
            (
                ("native-result", PreparedNativeTaskResult.SCHEMA),
                ("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
            ),
            ("native-observations", NATIVE_PAIR_SCHEMA),
        )
        for task in native:
            definitions.append(
                (
                    task.task_id,
                    ScientificStage.PREPARE,
                    spec,
                    source_identity,
                    DEVELOPMENT_SOURCE_CAPABILITY,
                    task.dependency_task_ids,
                    source_outputs,
                    BarrierKind.NONE,
                    ceil(2 + task.maximum_native_updates * 0.004),
                )
            )
        projection_identity = ObjectIdentity.from_record(
            projection.config_id, projection
        )
        projection_outputs = _outputs(
            (
                ("view-report", PreparedResponseDevelopmentViewProjection.SCHEMA),
                ("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
            )
        )
        for root in spec.roots:
            for refinement in (1, 2):
                definitions.append(
                    (
                        f"{root.root_id}.project.r{refinement}",
                        ScientificStage.TRANSFORM,
                        projection,
                        projection_identity,
                        DEVELOPMENT_PROJECTION_CAPABILITY,
                        native_by_root[root],
                        projection_outputs,
                        BarrierKind.FREEZE,
                        DEVELOPMENT_PROJECTION_CAPABILITY.resource_ceiling.wall_time_seconds,
                    )
                )
        fit_identity = ObjectIdentity.from_record(fit.config_id, fit)
        fit_outputs = _outputs(
            (
                ("model-fit", PreparedResponseDevelopmentModelFit.SCHEMA),
                ("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
            )
        )
        for context_name in ("assembling", "prepared"):
            dependencies = tuple(
                f"{root.root_id}.project.r{refinement}"
                for root in spec.roots
                if root.context == context_name
                for refinement in (1, 2)
            )
            for structure in STRUCTURES:
                definitions.append(
                    (
                        f"prepared-response.dependent-refinement.fit.{context_name}.{structure}",
                        ScientificStage.DEVELOP,
                        fit,
                        fit_identity,
                        DEVELOPMENT_FIT_CAPABILITY,
                        dependencies,
                        fit_outputs,
                        BarrierKind.FREEZE,
                        DEVELOPMENT_FIT_CAPABILITY.resource_ceiling.wall_time_seconds,
                    )
                )
        selection_identity = ObjectIdentity.from_record(selection.config_id, selection)
        selection_dependencies = tuple(
            sorted(
                (
                    *(
                        f"{root.root_id}.project.r{refinement}"
                        for root in spec.roots
                        for refinement in (1, 2)
                    ),
                    *(
                        f"prepared-response.dependent-refinement.fit.{context_name}.{structure}"
                        for context_name in ("assembling", "prepared")
                        for structure in STRUCTURES
                    ),
                )
            )
        )
        definitions.append(
            (
                PreparedResponseDevelopmentSelectionTask.TASK_ID,
                ScientificStage.EVALUATE,
                selection,
                selection_identity,
                DEVELOPMENT_SELECTION_CAPABILITY,
                selection_dependencies,
                _outputs(
                    (
                        ("nominal-library", PreparedResponseDevelopmentNominalLibrary.SCHEMA),
                        (
                            "constant-gain-benchmark",
                            PreparedResponseDevelopmentConstantGainReport.SCHEMA,
                        ),
                        ("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
                        (
                            "scientific-adjudication",
                            ScientificAdjudicationRecord.SCHEMA,
                        ),
                    )
                ),
                BarrierKind.REVEAL,
                DEVELOPMENT_SELECTION_CAPABILITY.resource_ceiling.wall_time_seconds,
            )
        )
        output_caps = {
            DEVELOPMENT_SOURCE_CAPABILITY.capability_key: DEVELOPMENT_SOURCE_CAPABILITY.resource_ceiling.output_bytes,
            DEVELOPMENT_PROJECTION_CAPABILITY.capability_key: (
                DEVELOPMENT_PROJECTION_CAPABILITY.resource_ceiling.output_bytes
            ),
            DEVELOPMENT_FIT_CAPABILITY.capability_key: DEVELOPMENT_FIT_CAPABILITY.resource_ceiling.output_bytes,
        }
        steps = []
        stage_capability = {
            DEVELOPMENT_SOURCE_CAPABILITY.capability_key: DEVELOPMENT_SOURCE_CAPABILITY,
            DEVELOPMENT_PROJECTION_CAPABILITY.capability_key: DEVELOPMENT_PROJECTION_CAPABILITY,
            DEVELOPMENT_FIT_CAPABILITY.capability_key: DEVELOPMENT_FIT_CAPABILITY,
            DEVELOPMENT_SELECTION_CAPABILITY.capability_key: DEVELOPMENT_SELECTION_CAPABILITY,
        }
        owner_by_task = {
            task_id: capability for task_id, _, _, _, capability, *_ in definitions
        }
        for (
            task_id,
            stage,
            config,
            identity,
            capability,
            dependencies,
            outputs,
            barrier,
            wall,
        ) in definitions:
            manifest = stage_capability[capability.capability_key]
            scan = len(config.canonical_bytes()) + sum(
                output_caps[owner_by_task[dependency].capability_key]
                for dependency in dependencies
            )
            if stage is ScientificStage.PREPARE and not dependencies:
                scan += len(config.canonical_bytes())
            if scan > manifest.resource_ceiling.source_scan_bytes:
                raise ValueError(
                    "prepared dependent refinement predecessor roster exceeds its installed scan ceiling"
                )
            steps.append(
                ProtocolStepTemplate(
                    task_id,
                    stage,
                    manifest.capability_key,
                    manifest.capability_version,
                    _config_ref(config, identity, manifest),
                    dependencies,
                    outputs,
                    manifest.permissions,
                    OutcomeAccess.EVALUATION_REVEALED
                    if stage is ScientificStage.EVALUATE
                    else OutcomeAccess.DEVELOPMENT_VISIBLE,
                    VisibilityCeiling.OUTCOME_VISIBLE
                    if stage is ScientificStage.EVALUATE
                    else VisibilityCeiling.DEVELOPMENT_ONLY,
                    replace(
                        manifest.resource_ceiling,
                        source_scan_bytes=scan,
                        wall_time_seconds=wall,
                    ),
                    (),
                    barrier,
                    1,
                    ("prepared-response.dependent-refinement.single-terminal",)
                    if stage is ScientificStage.EVALUATE
                    else (f"dependent-refinement.co-resident.{sha256(task_id.encode()).hexdigest()[:20]}",),
                )
            )
        if len(steps) != 8713:
            raise ValueError(
                "prepared dependent refinement expansion changes its acquisition/projection/fit/selection census"
            )
        return replace(
            template,
            template_id=f"{template.template_id}.{fit.fingerprint()[:16]}",
            steps=tuple(sorted(steps, key=lambda value: value.step_id)),
            requires_model_set=False,
            requests_controller=False,
            nonactuating=True,
        )


def _validate_fresh_calibration_source_profile(
    spec: PreparedNativeSpec, profile: NativeSourceProfile
) -> None:
    invocations = prepared_response_calibration_native_invocations(spec)
    if (
        profile.source_config != ObjectIdentity.from_record(spec.spec_id, spec)
        or spec.native_implementation
        != ObjectIdentity.from_record(
            CALIBRATION_SOURCE_CAPABILITY.capability_key, CALIBRATION_SOURCE_CAPABILITY
        )
        or profile.source_selection.capability_key
        != CALIBRATION_SOURCE_CAPABILITY.capability_key
        or profile.source_selection.capability_version
        != CALIBRATION_SOURCE_CAPABILITY.capability_version
        or profile.source_selection.implementation_sha256
        != CALIBRATION_SOURCE_CAPABILITY.implementation_sha256
        or profile.observation_schema != NATIVE_PAIR_SCHEMA
        or profile.physical_independent_unit_ids
        != tuple(root.physical_unit_id for root in spec.roots)
        or profile.maximum_native_segment_count != len(invocations)
        or profile.maximum_native_update_count
        != sum(value.maximum_native_updates for value in invocations)
        or not profile.resource_ceiling.contains(CALIBRATION_SOURCE_CAPABILITY.resource_ceiling)
    ):
        raise ValueError(
            "prepared fresh calibration source profile changes its source, roster or work bounds"
        )


@dataclass(frozen=True, slots=True)
class PreparedResponseCalibrationSourceFactory:
    binding: ExecutableCapabilityBinding = CALIBRATION_SOURCE_BINDING

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> PreparedResponseCalibrationSourceProvider:
        by_type = {type(value): value for value in records}
        core = {PreparedNativeSpec}
        full = {PreparedNativeSpec, *_CP_PROFILE_RECORD_TYPES}
        if (
            self.binding != CALIBRATION_SOURCE_BINDING
            or platform_ports
            or len(by_type) != len(records)
            or set(by_type) not in (core, full)
        ):
            raise ValueError(
                "prepared fresh calibration source factory requires its exact issued records"
            )
        spec = cast(PreparedNativeSpec, by_type[PreparedNativeSpec])
        if spec.stage != 'calibration':
            raise ValueError(
                "prepared fresh calibration source factory requires the fresh calibration native stage"
            )
        if set(by_type) == full:
            _validate_fresh_calibration_source_profile(
                spec, cast(NativeSourceProfile, by_type[NativeSourceProfile])
            )
        return PreparedResponseCalibrationSourceProvider(registry, CALIBRATION_SOURCE_CAPABILITY, spec)

    def expand_parameterised_protocol(
        self, *, records: tuple[CanonicalRecord, ...], template: ProtocolTemplate
    ) -> ProtocolTemplate:
        by_type = {type(value): value for value in records}
        core_types = {
            PreparedNativeSpec,
            PreparedPolicyDecisionConfig,
            PreparedResponseCalibrationProjectionConfig,
            PreparedResponseCalibrationCalibrationConfig,
        }
        full_types = {*core_types, *_CP_PROFILE_RECORD_TYPES}
        if len(by_type) != len(records) or set(by_type) not in (core_types, full_types):
            raise ValueError(
                "prepared fresh calibration expansion requires its exact source/policy/projection/calibration records"
            )
        spec = cast(PreparedNativeSpec, by_type[PreparedNativeSpec])
        policy = cast(
            PreparedPolicyDecisionConfig, by_type[PreparedPolicyDecisionConfig]
        )
        projection = cast(
            PreparedResponseCalibrationProjectionConfig, by_type[PreparedResponseCalibrationProjectionConfig]
        )
        calibration = cast(
            PreparedResponseCalibrationCalibrationConfig, by_type[PreparedResponseCalibrationCalibrationConfig]
        )
        if (
            spec.stage != 'calibration'
            or policy.native_spec != spec
            or projection.native_spec != spec
            or policy != projection.policy_config
            or calibration.projection != projection
        ):
            raise ValueError("prepared fresh calibration expansion changes its source qualification/dependent refinement/source lineage")
        if set(by_type) == full_types:
            profile = cast(NativeSourceProfile, by_type[NativeSourceProfile])
            _validate_fresh_calibration_source_profile(spec, profile)
            if profile.prerequisite_qualification != ObjectIdentity.from_record(
                "prepared-response.dependent-refinement.nominal-library", policy.development_library
            ):
                raise ValueError(
                    "prepared fresh calibration source profile changes its frozen dependent refinement prerequisite"
                )

        source_identity = ObjectIdentity.from_record(spec.spec_id, spec)
        policy_identity = ObjectIdentity.from_record(policy.config_id, policy)
        projection_identity = ObjectIdentity.from_record(
            projection.config_id, projection
        )
        calibration_identity = ObjectIdentity.from_record(
            calibration.config_id, calibration
        )
        source_outputs = _outputs(
            (
                ("native-result", PreparedResponseCalibrationNativeTaskResult.SCHEMA),
                ("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
            ),
            ("native-observations", NATIVE_PAIR_SCHEMA),
        )
        policy_outputs = _outputs(
            (
                ("parent-decision", PreparedParentDecision.SCHEMA),
                ("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
            )
        )
        projection_outputs = _outputs(
            (
                ("view-report", PreparedResponseCalibrationViewProjection.SCHEMA),
                ("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
            )
        )
        definitions: list[
            tuple[
                str,
                ScientificStage,
                CanonicalRecord,
                ObjectIdentity,
                CapabilityManifest,
                tuple[str, ...],
                tuple[OutputTemplate, ...],
                BarrierKind,
                int,
            ]
        ] = []
        native = prepared_response_calibration_native_invocations(spec)
        for invocation in native:
            definitions.append(
                (
                    invocation.task_id,
                    ScientificStage.PREPARE
                    if invocation.phase == "prefix"
                    else ScientificStage.ACQUIRE,
                    spec,
                    source_identity,
                    CALIBRATION_SOURCE_CAPABILITY,
                    invocation.dependency_task_ids,
                    source_outputs,
                    BarrierKind.NONE,
                    ceil(2 + invocation.maximum_native_updates * 0.004),
                )
            )
        native_by_root: dict[PreparedRoot, list[PreparedResponseCalibrationNativeInvocation]] = {
            root: [] for root in spec.roots
        }
        for invocation in native:
            native_by_root[invocation.root].append(invocation)
        for root, root_native in native_by_root.items():
            prefix_id = next(
                value.task_id for value in root_native if value.phase == "prefix"
            )
            for policy_id in CALIBRATION_POLICIES:
                decision_id = f"{root.root_id}.policy.{policy_id}.decision"
                definitions.append(
                    (
                        decision_id,
                        ScientificStage.DEVELOP,
                        policy,
                        policy_identity,
                        CALIBRATION_POLICY_CAPABILITY,
                        (prefix_id,),
                        policy_outputs,
                        BarrierKind.FREEZE,
                        CALIBRATION_POLICY_CAPABILITY.resource_ceiling.wall_time_seconds,
                    )
                )
                source_dependencies = tuple(
                    sorted(
                        (
                            *(
                                value.task_id
                                for value in root_native
                                if value.phase == "prefix"
                                or value.policy_id == policy_id
                            ),
                            decision_id,
                        )
                    )
                )
                for refinement in (1, 2):
                    definitions.append(
                        (
                            f"{root.root_id}.policy.{policy_id}.project.r{refinement}",
                            ScientificStage.TRANSFORM,
                            projection,
                            projection_identity,
                            CALIBRATION_PROJECTION_CAPABILITY,
                            source_dependencies,
                            projection_outputs,
                            BarrierKind.FREEZE,
                            CALIBRATION_PROJECTION_CAPABILITY.resource_ceiling.wall_time_seconds,
                        )
                    )
        calibration_dependencies = tuple(
            f"{root.root_id}.policy.{policy_id}.project.r{refinement}"
            for root in spec.roots
            for policy_id in CALIBRATION_POLICIES
            for refinement in (1, 2)
        )
        definitions.append(
            (
                PreparedResponseCalibrationCalibrationTask.TASK_ID,
                ScientificStage.EVALUATE,
                calibration,
                calibration_identity,
                CALIBRATION_CALIBRATION_CAPABILITY,
                tuple(sorted(calibration_dependencies)),
                _outputs(
                    (
                        ("calibrated-library", PreparedResponseCalibrationCalibratedLibrary.SCHEMA),
                        ("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
                        (
                            "scientific-adjudication",
                            ScientificAdjudicationRecord.SCHEMA,
                        ),
                    )
                ),
                BarrierKind.REVEAL,
                CALIBRATION_CALIBRATION_CAPABILITY.resource_ceiling.wall_time_seconds,
            )
        )

        output_caps = {
            CALIBRATION_SOURCE_CAPABILITY.capability_key: CALIBRATION_SOURCE_CAPABILITY.resource_ceiling.output_bytes,
            CALIBRATION_POLICY_CAPABILITY.capability_key: CALIBRATION_POLICY_CAPABILITY.resource_ceiling.output_bytes,
            CALIBRATION_PROJECTION_CAPABILITY.capability_key: (
                CALIBRATION_PROJECTION_CAPABILITY.resource_ceiling.output_bytes
            ),
        }
        owner_by_task = {
            task_id: capability for task_id, _, _, _, capability, *_ in definitions
        }
        config_sizes = {
            identity: len(config.canonical_bytes())
            for config, identity in (
                (spec, source_identity),
                (policy, policy_identity),
                (projection, projection_identity),
                (calibration, calibration_identity),
            )
        }
        steps = []
        for (
            task_id,
            stage,
            config,
            identity,
            manifest,
            dependencies,
            outputs,
            barrier,
            wall,
        ) in definitions:
            scan = config_sizes[identity] + sum(
                output_caps[owner_by_task[dependency].capability_key]
                for dependency in dependencies
            )
            if stage is ScientificStage.PREPARE and not dependencies:
                scan += config_sizes[identity]
            if scan > manifest.resource_ceiling.source_scan_bytes:
                raise ValueError(
                    "prepared fresh calibration predecessor roster exceeds its installed scan ceiling"
                )
            steps.append(
                ProtocolStepTemplate(
                    task_id,
                    stage,
                    manifest.capability_key,
                    manifest.capability_version,
                    _config_ref(config, identity, manifest),
                    dependencies,
                    outputs,
                    manifest.permissions,
                    OutcomeAccess.EVALUATION_REVEALED
                    if stage is ScientificStage.EVALUATE
                    else OutcomeAccess.EVALUATION_SEALED,
                    VisibilityCeiling.OUTCOME_VISIBLE
                    if stage is ScientificStage.EVALUATE
                    else VisibilityCeiling.DEVELOPMENT_ONLY,
                    replace(
                        manifest.resource_ceiling,
                        source_scan_bytes=scan,
                        wall_time_seconds=wall,
                    ),
                    (),
                    barrier,
                    1,
                    ("prepared-response.fresh-response-calibration.single-terminal",)
                    if stage is ScientificStage.EVALUATE
                    else (f"fresh-response-calibration.co-resident.{sha256(task_id.encode()).hexdigest()[:20]}",),
                )
            )
        if len(steps) != 10177:
            raise ValueError(
                "prepared fresh calibration expansion changes its native/policy/view/calibration census"
            )
        return replace(
            template,
            template_id=f"{template.template_id}.{calibration.fingerprint()[:16]}",
            steps=tuple(sorted(steps, key=lambda value: value.step_id)),
            requires_model_set=False,
            requests_controller=False,
            nonactuating=True,
        )


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    "executable-contribution.prepared-response.source-qualification-dependent-refinement-fresh-calibration.source",
    "1.0.0",
    tuple(
        sorted(
            (SOURCE_QUALIFICATION_SOURCE_BINDING, DEVELOPMENT_SOURCE_BINDING, CALIBRATION_SOURCE_BINDING),
            key=lambda value: value.binding_id,
        )
    ),
)
EXECUTABLE_BINDING_FACTORIES = (
    PreparedResponseSourceQualificationSourceFactory(),
    PreparedResponseDevelopmentSourceFactory(),
    PreparedResponseCalibrationSourceFactory(),
)
# The shared qualification carrier and substrate record classes already have
# canonical codec owners in the generated registry. This descriptor contributes
# only the new source type while its binding retains all three required schemas.
EXECUTABLE_RECORD_TYPES = (PreparedNativeSpec,)
