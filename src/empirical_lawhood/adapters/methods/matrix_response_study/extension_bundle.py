"""Static Six-matrix response method-config discovery; no law or verdict is produced here."""

from __future__ import annotations

from hashlib import sha256

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.evidence_profiles import ExperimentObjectiveProfile
from empirical_lawhood.runtime.extension_bundles import ExtensionBundleContribution, ExtensionComponentKind, ExtensionComponentRegistration, ExtensionContributionKind, ExtensionProfileKind, ExtensionProfileRegistration
from empirical_lawhood.runtime.capabilities import (
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
)
from empirical_lawhood.runtime.candidate_composition import CandidateCapabilityRegistration
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.adapters.simulators.six_matrix_response.reactive_entrance import REACTIVE_ENTRANCE_HDF5_SCHEMA, SixMatrixResponseReactiveEntrancePrecursorResult
from empirical_lawhood.adapters.simulators.six_matrix_response.prospective_reactive_source import SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HDF5_SCHEMA, SixMatrixResponseProspectiveReactiveSourceLawHistoryResult
from empirical_lawhood.adapters.composition.matrix_response_study.causal_intersection_residence_design import MatrixResponseCausalIntersectionResidenceStudyConfig

from .contracts import MatrixResponseLawMethodConfig, MatrixResponsePairedPanelReducerConfig, MatrixResponseRoleEquivarianceConfig, MatrixResponseStructuralFacePlan
from .reactive_entrance_provider import REACTIVE_ENTRANCE_FINALIZATION_CAPABILITY_KEY, REACTIVE_ENTRANCE_METHOD_CAPABILITY_VERSION, REACTIVE_ENTRANCE_PROJECTION_CAPABILITY_KEY
from .reactive_entrance_source import MatrixResponseReactiveEntranceAggregate, MatrixResponseReactiveEntranceMethodConfig, MatrixResponseReactiveEntranceProjection
from .prospective_reactive_source_provider import PROSPECTIVE_REACTIVE_SOURCE_LAW_FINALIZATION_CAPABILITY_KEY, PROSPECTIVE_REACTIVE_SOURCE_LAW_METHOD_CAPABILITY_VERSION, PROSPECTIVE_REACTIVE_SOURCE_LAW_PROJECTION_CAPABILITY_KEY
from .prospective_reactive_source_law import MatrixResponseProspectiveReactiveSourceLawAggregate, MatrixResponseProspectiveReactiveSourceLawMethodConfig, MatrixResponseProspectiveReactiveSourceLawProjection
from .history_conditioned_geometry import HistoryViewTrace
from .observation_order import MatrixObservationOrderEvaluationConfig, MatrixObservationOrderEvaluation, MatrixObservationOrderProjectionConfig, MatrixObservationOrderViewReport
from .observation_order_provider import OBSERVATION_ORDER_EVALUATION_CAPABILITY_KEY, OBSERVATION_ORDER_METHOD_CAPABILITY_VERSION, OBSERVATION_ORDER_PROJECTION_CAPABILITY_KEY
from empirical_lawhood.adapters.simulators.six_matrix_response.observation_order import MatrixObservationOrderPairedHistoryResult, OBSERVATION_ORDER_HDF5_SCHEMA


_IMPLEMENTATION_SHA256 = sha256(b"matrix-response-study:method-config-reconstruction").hexdigest()
REACTIVE_ENTRANCE_METHOD_IMPLEMENTATION_SHA256 = sha256(
    b"matrix-response-reactive-entrance-causal-projection-and-single-terminal"
).hexdigest()
PROSPECTIVE_REACTIVE_SOURCE_LAW_METHOD_IMPLEMENTATION_SHA256 = sha256(
    b"matrix-response-prospective-reactive-source-law-causal-source-projection-and-terminal"
).hexdigest()
OBSERVATION_ORDER_METHOD_IMPLEMENTATION_SHA256 = sha256(
    b"matrix-observation-order-geometry-interval-transition-adjudication"
).hexdigest()


OBSERVATION_ORDER_PROJECTION_CAPABILITY = CapabilityManifest(
    capability_key=OBSERVATION_ORDER_PROJECTION_CAPABILITY_KEY,
    capability_version=OBSERVATION_ORDER_METHOD_CAPABILITY_VERSION,
    kind=CapabilityKind.TRANSFORM,
    config_schema=MatrixObservationOrderProjectionConfig.SCHEMA,
    config_schema_sha256=sha256(MatrixObservationOrderProjectionConfig.SCHEMA.encode()).hexdigest(),
    input_schema_ids=tuple(
        sorted(
            (
                OBSERVATION_ORDER_HDF5_SCHEMA,
                MatrixObservationOrderPairedHistoryResult.SCHEMA,
                LinkedCampaignStageEnvelope.SCHEMA,
            )
        )
    ),
    output_schema_ids=tuple(
        sorted((HistoryViewTrace.SCHEMA, LinkedCampaignStageEnvelope.SCHEMA))
    ),
    permissions=(
        CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
        CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
    ),
    maximum_evidence_ceiling=EvidenceCeiling.ORDER_RELATION,
    maximum_outcome_access=OutcomeAccess.EVALUATION_SEALED,
    resource_ceiling=ResourceBudget(
        cpu_cores=1,
        memory_bytes=2 * 1024**3,
        gpu_devices=0,
        wall_time_seconds=7_200,
        source_scan_bytes=16 * 1024**2,
        output_bytes=512 * 1024,
    ),
    deterministic=True,
    seed_required=False,
    language_id="python",
    runtime_id="matrix-observation-order-method",
    requires_clean_commit=True,
    requires_active_mount=True,
    requires_network=False,
    conformance_check_ids=(
        "dense-frozen-receiver-lattice",
        "one-pure-projection-per-view",
        "primitive-overlap-preserved",
    ),
    implementation_sha256=OBSERVATION_ORDER_METHOD_IMPLEMENTATION_SHA256,
)

OBSERVATION_ORDER_EVALUATION_CAPABILITY = CapabilityManifest(
    capability_key=OBSERVATION_ORDER_EVALUATION_CAPABILITY_KEY,
    capability_version=OBSERVATION_ORDER_METHOD_CAPABILITY_VERSION,
    kind=CapabilityKind.EVALUATOR,
    config_schema=MatrixObservationOrderEvaluationConfig.SCHEMA,
    config_schema_sha256=sha256(MatrixObservationOrderEvaluationConfig.SCHEMA.encode()).hexdigest(),
    input_schema_ids=tuple(
        sorted((HistoryViewTrace.SCHEMA, LinkedCampaignStageEnvelope.SCHEMA))
    ),
    output_schema_ids=tuple(
        sorted(
            (
                MatrixObservationOrderEvaluation.SCHEMA,
                MatrixObservationOrderViewReport.SCHEMA,
                LinkedCampaignStageEnvelope.SCHEMA,
                ScientificAdjudicationRecord.SCHEMA,
            )
        )
    ),
    permissions=tuple(
        sorted(
            (
                CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                CapabilityPermission.READ_OUTCOME_VISIBLE,
                CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
            )
        )
    ),
    maximum_evidence_ceiling=EvidenceCeiling.ORDER_RELATION,
    maximum_outcome_access=OutcomeAccess.EVALUATION_REVEALED,
    resource_ceiling=ResourceBudget(
        cpu_cores=1,
        memory_bytes=8 * 1024**3,
        gpu_devices=0,
        wall_time_seconds=21_600,
        source_scan_bytes=5 * 1024**3,
        output_bytes=32 * 1024**2,
    ),
    deterministic=True,
    seed_required=False,
    language_id="python",
    runtime_id="matrix-observation-order-method",
    requires_clean_commit=True,
    requires_active_mount=True,
    requires_network=False,
    conformance_check_ids=(
        "family-stratified-history-bootstrap",
        "grouped-interval-censored-npmle",
        "noncompensating-two-view-terminal",
        "single-immutable-adjudication",
    ),
    implementation_sha256=OBSERVATION_ORDER_METHOD_IMPLEMENTATION_SHA256,
)


PROSPECTIVE_REACTIVE_SOURCE_LAW_PROJECTION_CAPABILITY = CapabilityManifest(
    capability_key=PROSPECTIVE_REACTIVE_SOURCE_LAW_PROJECTION_CAPABILITY_KEY,
    capability_version=PROSPECTIVE_REACTIVE_SOURCE_LAW_METHOD_CAPABILITY_VERSION,
    kind=CapabilityKind.TRANSFORM,
    config_schema=MatrixResponseProspectiveReactiveSourceLawMethodConfig.SCHEMA,
    config_schema_sha256=sha256(MatrixResponseProspectiveReactiveSourceLawMethodConfig.SCHEMA.encode()).hexdigest(),
    input_schema_ids=tuple(sorted((SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HDF5_SCHEMA, SixMatrixResponseProspectiveReactiveSourceLawHistoryResult.SCHEMA, LinkedCampaignStageEnvelope.SCHEMA))),
    output_schema_ids=tuple(sorted((MatrixResponseProspectiveReactiveSourceLawProjection.SCHEMA, LinkedCampaignStageEnvelope.SCHEMA))),
    permissions=(CapabilityPermission.READ_EXTERNAL_ARTIFACTS, CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS),
    maximum_evidence_ceiling=EvidenceCeiling.ORDER_RELATION,
    maximum_outcome_access=OutcomeAccess.EVALUATION_SEALED,
    resource_ceiling=ResourceBudget(cpu_cores=1, memory_bytes=8 * 1024**3, gpu_devices=0, wall_time_seconds=14_400, source_scan_bytes=1024**3, output_bytes=16 * 1024**2),
    deterministic=True, seed_required=False, language_id="python", runtime_id="matrix-response-prospective-reactive-source-law-method",
    requires_clean_commit=True, requires_active_mount=True, requires_network=False,
    conformance_check_ids=("exact-reactive-entrance-outcomes", "past-only-prefix-recomputation", "routing-digest-verification"),
    implementation_sha256=PROSPECTIVE_REACTIVE_SOURCE_LAW_METHOD_IMPLEMENTATION_SHA256,
)

PROSPECTIVE_REACTIVE_SOURCE_LAW_FINALIZATION_CAPABILITY = CapabilityManifest(
    capability_key=PROSPECTIVE_REACTIVE_SOURCE_LAW_FINALIZATION_CAPABILITY_KEY,
    capability_version=PROSPECTIVE_REACTIVE_SOURCE_LAW_METHOD_CAPABILITY_VERSION,
    kind=CapabilityKind.EVALUATOR,
    config_schema=MatrixResponseProspectiveReactiveSourceLawMethodConfig.SCHEMA,
    config_schema_sha256=sha256(MatrixResponseProspectiveReactiveSourceLawMethodConfig.SCHEMA.encode()).hexdigest(),
    input_schema_ids=tuple(sorted((MatrixResponseProspectiveReactiveSourceLawProjection.SCHEMA, LinkedCampaignStageEnvelope.SCHEMA))),
    output_schema_ids=tuple(sorted((MatrixResponseProspectiveReactiveSourceLawAggregate.SCHEMA, LinkedCampaignStageEnvelope.SCHEMA, ScientificAdjudicationRecord.SCHEMA))),
    permissions=tuple(sorted((CapabilityPermission.READ_EXTERNAL_ARTIFACTS, CapabilityPermission.READ_OUTCOME_VISIBLE, CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS))),
    maximum_evidence_ceiling=EvidenceCeiling.ORDER_RELATION,
    maximum_outcome_access=OutcomeAccess.EVALUATION_REVEALED,
    resource_ceiling=ResourceBudget(cpu_cores=1, memory_bytes=8 * 1024**3, gpu_devices=0, wall_time_seconds=14_400, source_scan_bytes=4 * 1024**3, output_bytes=32 * 1024**2),
    deterministic=True, seed_required=False, language_id="python", runtime_id="matrix-response-prospective-reactive-source-law-method",
    requires_clean_commit=True, requires_active_mount=True, requires_network=False,
    conformance_check_ids=("complete-256-plus-64-roster", "history-clustered-intervals", "single-frozen-terminal"),
    implementation_sha256=PROSPECTIVE_REACTIVE_SOURCE_LAW_METHOD_IMPLEMENTATION_SHA256,
)


REACTIVE_ENTRANCE_PROJECTION_CAPABILITY = CapabilityManifest(
    capability_key=REACTIVE_ENTRANCE_PROJECTION_CAPABILITY_KEY,
    capability_version=REACTIVE_ENTRANCE_METHOD_CAPABILITY_VERSION,
    kind=CapabilityKind.TRANSFORM,
    config_schema=MatrixResponseReactiveEntranceMethodConfig.SCHEMA,
    config_schema_sha256=sha256(MatrixResponseReactiveEntranceMethodConfig.SCHEMA.encode()).hexdigest(),
    input_schema_ids=tuple(
        sorted(
            (
                REACTIVE_ENTRANCE_HDF5_SCHEMA,
                SixMatrixResponseReactiveEntrancePrecursorResult.SCHEMA,
                LinkedCampaignStageEnvelope.SCHEMA,
            )
        )
    ),
    output_schema_ids=tuple(
        sorted((MatrixResponseReactiveEntranceProjection.SCHEMA, LinkedCampaignStageEnvelope.SCHEMA))
    ),
    permissions=(
        CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
        CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
    ),
    maximum_evidence_ceiling=EvidenceCeiling.ORDER_RELATION,
    maximum_outcome_access=OutcomeAccess.EVALUATION_SEALED,
    resource_ceiling=ResourceBudget(
        cpu_cores=1,
        memory_bytes=4 * 1024**3,
        gpu_devices=0,
        wall_time_seconds=7_200,
        source_scan_bytes=32 * 1024**2,
        output_bytes=4 * 1024**2,
    ),
    deterministic=True,
    seed_required=False,
    language_id="python",
    runtime_id="matrix-response-reactive-entrance-method",
    requires_clean_commit=True,
    requires_active_mount=True,
    requires_network=False,
    conformance_check_ids=(
        "causal-competing-risk-entry",
        "complete-causal-intersection-residence-preparation-scheduling-response-geometric-conjunction",
        "one-projection-per-issued-view",
        "persisted-hdf5-custody-before-method",
    ),
    implementation_sha256=REACTIVE_ENTRANCE_METHOD_IMPLEMENTATION_SHA256,
)

REACTIVE_ENTRANCE_FINALIZATION_CAPABILITY = CapabilityManifest(
    capability_key=REACTIVE_ENTRANCE_FINALIZATION_CAPABILITY_KEY,
    capability_version=REACTIVE_ENTRANCE_METHOD_CAPABILITY_VERSION,
    kind=CapabilityKind.EVALUATOR,
    config_schema=MatrixResponseReactiveEntranceMethodConfig.SCHEMA,
    config_schema_sha256=sha256(MatrixResponseReactiveEntranceMethodConfig.SCHEMA.encode()).hexdigest(),
    input_schema_ids=tuple(
        sorted(
            (
                MatrixResponseReactiveEntranceAggregate.SCHEMA,
                MatrixResponseReactiveEntranceProjection.SCHEMA,
                LinkedCampaignStageEnvelope.SCHEMA,
            )
        )
    ),
    output_schema_ids=tuple(
        sorted(
            (
                MatrixResponseReactiveEntranceAggregate.SCHEMA,
                LinkedCampaignStageEnvelope.SCHEMA,
                ScientificAdjudicationRecord.SCHEMA,
            )
        )
    ),
    permissions=tuple(
        sorted(
            (
                CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                CapabilityPermission.READ_OUTCOME_VISIBLE,
                CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
            )
        )
    ),
    maximum_evidence_ceiling=EvidenceCeiling.ORDER_RELATION,
    maximum_outcome_access=OutcomeAccess.EVALUATION_REVEALED,
    resource_ceiling=ResourceBudget(
        cpu_cores=1,
        memory_bytes=4 * 1024**3,
        gpu_devices=0,
        wall_time_seconds=3_600,
        source_scan_bytes=4 * 1024**3,
        output_bytes=16 * 1024**2,
    ),
    deterministic=True,
    seed_required=False,
    language_id="python",
    runtime_id="matrix-response-reactive-entrance-method",
    requires_clean_commit=True,
    requires_active_mount=True,
    requires_network=False,
    conformance_check_ids=(
        "exact-512-plus-128-roster",
        "five-way-reactive-entrance-terminal",
        "no-law-qualification",
        "single-frozen-terminal",
    ),
    implementation_sha256=REACTIVE_ENTRANCE_METHOD_IMPLEMENTATION_SHA256,
)


def _component(
    record_type: type[CanonicalRecord],
    suffix: str,
    kind: ExtensionComponentKind,
) -> ExtensionComponentRegistration:
    return ExtensionComponentRegistration(
        registration_id=f"component.matrix-response-study.{suffix}",
        component_key=f"matrix-response-study-{suffix}",
        component_version="1.0.0",
        kind=kind,
        input_schema_ids=(record_type.SCHEMA,),
        output_schema_ids=(record_type.SCHEMA,),
        implementation_sha256=_IMPLEMENTATION_SHA256,
    )


MATRIX_RESPONSE_LAW_CONFIG_DECODER = _component(
    MatrixResponseLawMethodConfig,
    "law-method-config-decoder",
    ExtensionComponentKind.CONFIG_DECODER,
)
MATRIX_RESPONSE_LAW_CONFIG_RECONSTRUCTOR = _component(
    MatrixResponseLawMethodConfig,
    "law-method-config-reconstructor",
    ExtensionComponentKind.RUNTIME_PROVIDER,
)
MATRIX_RESPONSE_REDUCER_CONFIG_DECODER = _component(
    MatrixResponsePairedPanelReducerConfig,
    "paired-panel-reducer-config-decoder",
    ExtensionComponentKind.CONFIG_DECODER,
)
MATRIX_RESPONSE_REDUCER_CONFIG_RECONSTRUCTOR = _component(
    MatrixResponsePairedPanelReducerConfig,
    "paired-panel-reducer-config-reconstructor",
    ExtensionComponentKind.RUNTIME_PROVIDER,
)
MATRIX_RESPONSE_ROLE_CONFIG_DECODER = _component(
    MatrixResponseRoleEquivarianceConfig,
    "role-equivariance-config-decoder",
    ExtensionComponentKind.CONFIG_DECODER,
)
MATRIX_RESPONSE_ROLE_CONFIG_RECONSTRUCTOR = _component(
    MatrixResponseRoleEquivarianceConfig,
    "role-equivariance-config-reconstructor",
    ExtensionComponentKind.RUNTIME_PROVIDER,
)
MATRIX_RESPONSE_STRUCTURAL_CONFIG_DECODER = _component(
    MatrixResponseStructuralFacePlan,
    "structural-face-plan-decoder",
    ExtensionComponentKind.CONFIG_DECODER,
)
MATRIX_RESPONSE_STRUCTURAL_CONFIG_RECONSTRUCTOR = _component(
    MatrixResponseStructuralFacePlan,
    "structural-face-plan-reconstructor",
    ExtensionComponentKind.RUNTIME_PROVIDER,
)
MATRIX_RESPONSE_CAUSAL_INTERSECTION_RESIDENCE_CONFIG_DECODER = _component(
    MatrixResponseCausalIntersectionResidenceStudyConfig,
    "causal-intersection-residence-study-config-decoder",
    ExtensionComponentKind.CONFIG_DECODER,
)
MATRIX_RESPONSE_CAUSAL_INTERSECTION_RESIDENCE_CONFIG_RECONSTRUCTOR = _component(
    MatrixResponseCausalIntersectionResidenceStudyConfig,
    "causal-intersection-residence-study-config-reconstructor",
    ExtensionComponentKind.RUNTIME_PROVIDER,
)
REACTIVE_ENTRANCE_METHOD_CONFIG_DECODER = ExtensionComponentRegistration(
    registration_id="component.matrix-response-reactive-entrance.method-config-decoder",
    component_key="matrix-response-reactive-entrance-method-config-decoder",
    component_version="1.0.0",
    kind=ExtensionComponentKind.CONFIG_DECODER,
    input_schema_ids=(MatrixResponseReactiveEntranceMethodConfig.SCHEMA,),
    output_schema_ids=(MatrixResponseReactiveEntranceMethodConfig.SCHEMA,),
    implementation_sha256=REACTIVE_ENTRANCE_METHOD_IMPLEMENTATION_SHA256,
)
REACTIVE_ENTRANCE_PROJECTION_PROVIDER = ExtensionComponentRegistration(
    registration_id="component.matrix-response-reactive-entrance.projection-provider",
    component_key="matrix-response-reactive-entrance-projection-provider",
    component_version="1.0.0",
    kind=ExtensionComponentKind.RUNTIME_PROVIDER,
    input_schema_ids=REACTIVE_ENTRANCE_PROJECTION_CAPABILITY.input_schema_ids,
    output_schema_ids=REACTIVE_ENTRANCE_PROJECTION_CAPABILITY.output_schema_ids,
    implementation_sha256=REACTIVE_ENTRANCE_METHOD_IMPLEMENTATION_SHA256,
)
REACTIVE_ENTRANCE_FINALIZATION_PROVIDER = ExtensionComponentRegistration(
    registration_id="component.matrix-response-reactive-entrance.finalization-provider",
    component_key="matrix-response-reactive-entrance-finalization-provider",
    component_version="1.0.0",
    kind=ExtensionComponentKind.RUNTIME_PROVIDER,
    input_schema_ids=REACTIVE_ENTRANCE_FINALIZATION_CAPABILITY.input_schema_ids,
    output_schema_ids=REACTIVE_ENTRANCE_FINALIZATION_CAPABILITY.output_schema_ids,
    implementation_sha256=REACTIVE_ENTRANCE_METHOD_IMPLEMENTATION_SHA256,
)
PROSPECTIVE_REACTIVE_SOURCE_LAW_METHOD_CONFIG_DECODER = ExtensionComponentRegistration(
    registration_id="component.matrix-response-prospective-reactive-source-law.method-config-decoder",
    component_key="matrix-response-prospective-reactive-source-law-method-config-decoder",
    component_version="1.0.0",
    kind=ExtensionComponentKind.CONFIG_DECODER,
    input_schema_ids=(MatrixResponseProspectiveReactiveSourceLawMethodConfig.SCHEMA,),
    output_schema_ids=(MatrixResponseProspectiveReactiveSourceLawMethodConfig.SCHEMA,),
    implementation_sha256=PROSPECTIVE_REACTIVE_SOURCE_LAW_METHOD_IMPLEMENTATION_SHA256,
)
PROSPECTIVE_REACTIVE_SOURCE_LAW_PROJECTION_PROVIDER = ExtensionComponentRegistration(
    registration_id="component.matrix-response-prospective-reactive-source-law.projection-provider",
    component_key="matrix-response-prospective-reactive-source-law-projection-provider",
    component_version="1.0.0",
    kind=ExtensionComponentKind.RUNTIME_PROVIDER,
    input_schema_ids=PROSPECTIVE_REACTIVE_SOURCE_LAW_PROJECTION_CAPABILITY.input_schema_ids,
    output_schema_ids=PROSPECTIVE_REACTIVE_SOURCE_LAW_PROJECTION_CAPABILITY.output_schema_ids,
    implementation_sha256=PROSPECTIVE_REACTIVE_SOURCE_LAW_METHOD_IMPLEMENTATION_SHA256,
)
PROSPECTIVE_REACTIVE_SOURCE_LAW_FINALIZATION_PROVIDER = ExtensionComponentRegistration(
    registration_id="component.matrix-response-prospective-reactive-source-law.finalization-provider",
    component_key="matrix-response-prospective-reactive-source-law-finalization-provider",
    component_version="1.0.0",
    kind=ExtensionComponentKind.RUNTIME_PROVIDER,
    input_schema_ids=PROSPECTIVE_REACTIVE_SOURCE_LAW_FINALIZATION_CAPABILITY.input_schema_ids,
    output_schema_ids=PROSPECTIVE_REACTIVE_SOURCE_LAW_FINALIZATION_CAPABILITY.output_schema_ids,
    implementation_sha256=PROSPECTIVE_REACTIVE_SOURCE_LAW_METHOD_IMPLEMENTATION_SHA256,
)
OBSERVATION_ORDER_PROJECTION_CONFIG_DECODER = ExtensionComponentRegistration(
    registration_id="component.matrix-observation-order.projection-config-decoder",
    component_key="matrix-observation-order-projection-config-decoder",
    component_version="1.0.0",
    kind=ExtensionComponentKind.CONFIG_DECODER,
    input_schema_ids=(MatrixObservationOrderProjectionConfig.SCHEMA,),
    output_schema_ids=(MatrixObservationOrderProjectionConfig.SCHEMA,),
    implementation_sha256=OBSERVATION_ORDER_METHOD_IMPLEMENTATION_SHA256,
)
OBSERVATION_ORDER_EVALUATION_CONFIG_DECODER = ExtensionComponentRegistration(
    registration_id="component.matrix-observation-order.evaluation-config-decoder",
    component_key="matrix-observation-order-evaluation-config-decoder",
    component_version="1.0.0",
    kind=ExtensionComponentKind.CONFIG_DECODER,
    input_schema_ids=(MatrixObservationOrderEvaluationConfig.SCHEMA,),
    output_schema_ids=(MatrixObservationOrderEvaluationConfig.SCHEMA,),
    implementation_sha256=OBSERVATION_ORDER_METHOD_IMPLEMENTATION_SHA256,
)
OBSERVATION_ORDER_PROJECTION_PROVIDER = ExtensionComponentRegistration(
    registration_id="component.matrix-observation-order.projection-provider",
    component_key="matrix-observation-order-projection-provider",
    component_version="1.0.0",
    kind=ExtensionComponentKind.RUNTIME_PROVIDER,
    input_schema_ids=OBSERVATION_ORDER_PROJECTION_CAPABILITY.input_schema_ids,
    output_schema_ids=OBSERVATION_ORDER_PROJECTION_CAPABILITY.output_schema_ids,
    implementation_sha256=OBSERVATION_ORDER_METHOD_IMPLEMENTATION_SHA256,
)
OBSERVATION_ORDER_EVALUATION_PROVIDER = ExtensionComponentRegistration(
    registration_id="component.matrix-observation-order.evaluation-provider",
    component_key="matrix-observation-order-evaluation-provider",
    component_version="1.0.0",
    kind=ExtensionComponentKind.RUNTIME_PROVIDER,
    input_schema_ids=OBSERVATION_ORDER_EVALUATION_CAPABILITY.input_schema_ids,
    output_schema_ids=OBSERVATION_ORDER_EVALUATION_CAPABILITY.output_schema_ids,
    implementation_sha256=OBSERVATION_ORDER_METHOD_IMPLEMENTATION_SHA256,
)
OBSERVATION_ORDER_PROJECTION_CANDIDATE_REGISTRATION = CandidateCapabilityRegistration(
    OBSERVATION_ORDER_PROJECTION_CAPABILITY,
    OBSERVATION_ORDER_PROJECTION_PROVIDER.component_key,
    OBSERVATION_ORDER_PROJECTION_PROVIDER.component_version,
    "application/vnd.empirical-lawhood.canonical+json",
    4 * 1024**2,
)
OBSERVATION_ORDER_EVALUATION_CANDIDATE_REGISTRATION = CandidateCapabilityRegistration(
    OBSERVATION_ORDER_EVALUATION_CAPABILITY,
    OBSERVATION_ORDER_EVALUATION_PROVIDER.component_key,
    OBSERVATION_ORDER_EVALUATION_PROVIDER.component_version,
    "application/vnd.empirical-lawhood.canonical+json",
    4 * 1024**2,
)
OBSERVATION_ORDER_OBSERVATION_METHOD_PROFILE_REGISTRATION = ExtensionProfileRegistration(
    "profile-registration.matrix-observation-order-observation-methods",
    "matrix-observation-order-observation-methods",
    "1.0.0",
    ExtensionProfileKind.OBJECTIVE_TERMINAL,
    ExperimentObjectiveProfile.SCHEMA,
    sha256(ExperimentObjectiveProfile.SCHEMA.encode()).hexdigest(),
    OBSERVATION_ORDER_METHOD_IMPLEMENTATION_SHA256,
)


_DECODERS = tuple(
    sorted(
        (
            MATRIX_RESPONSE_LAW_CONFIG_DECODER,
            OBSERVATION_ORDER_EVALUATION_CONFIG_DECODER,
            OBSERVATION_ORDER_PROJECTION_CONFIG_DECODER,
            PROSPECTIVE_REACTIVE_SOURCE_LAW_METHOD_CONFIG_DECODER,
            REACTIVE_ENTRANCE_METHOD_CONFIG_DECODER,
            MATRIX_RESPONSE_REDUCER_CONFIG_DECODER,
            MATRIX_RESPONSE_ROLE_CONFIG_DECODER,
            MATRIX_RESPONSE_CAUSAL_INTERSECTION_RESIDENCE_CONFIG_DECODER,
            MATRIX_RESPONSE_STRUCTURAL_CONFIG_DECODER,
        ),
        key=lambda value: value.registration_id,
    )
)
_RECONSTRUCTORS = tuple(
    sorted(
        (
            MATRIX_RESPONSE_LAW_CONFIG_RECONSTRUCTOR,
            OBSERVATION_ORDER_EVALUATION_PROVIDER,
            OBSERVATION_ORDER_PROJECTION_PROVIDER,
            PROSPECTIVE_REACTIVE_SOURCE_LAW_FINALIZATION_PROVIDER,
            PROSPECTIVE_REACTIVE_SOURCE_LAW_PROJECTION_PROVIDER,
            REACTIVE_ENTRANCE_FINALIZATION_PROVIDER,
            REACTIVE_ENTRANCE_PROJECTION_PROVIDER,
            MATRIX_RESPONSE_REDUCER_CONFIG_RECONSTRUCTOR,
            MATRIX_RESPONSE_ROLE_CONFIG_RECONSTRUCTOR,
            MATRIX_RESPONSE_CAUSAL_INTERSECTION_RESIDENCE_CONFIG_RECONSTRUCTOR,
            MATRIX_RESPONSE_STRUCTURAL_CONFIG_RECONSTRUCTOR,
        ),
        key=lambda value: value.registration_id,
    )
)


EXTENSION_BUNDLE_CONTRIBUTION = ExtensionBundleContribution(
    contribution_id="extension-contribution.matrix-response-study-method-config",
    contribution_version="1.0.0",
    kind=ExtensionContributionKind.METHOD,
    evidence_profile_registries=(),
    capability_manifests=tuple(
        sorted(
            (
                PROSPECTIVE_REACTIVE_SOURCE_LAW_FINALIZATION_CAPABILITY,
                PROSPECTIVE_REACTIVE_SOURCE_LAW_PROJECTION_CAPABILITY,
                REACTIVE_ENTRANCE_FINALIZATION_CAPABILITY,
                REACTIVE_ENTRANCE_PROJECTION_CAPABILITY,
                OBSERVATION_ORDER_EVALUATION_CAPABILITY,
                OBSERVATION_ORDER_PROJECTION_CAPABILITY,
            ),
            key=lambda value: value.registry_id,
        )
    ),
    candidate_capability_registrations=tuple(
        sorted(
            (
                OBSERVATION_ORDER_EVALUATION_CANDIDATE_REGISTRATION,
                OBSERVATION_ORDER_PROJECTION_CANDIDATE_REGISTRATION,
            ),
            key=lambda value: value.registration_id,
        )
    ),
    dataset_capability_registries=(),
    profile_registrations=(OBSERVATION_ORDER_OBSERVATION_METHOD_PROFILE_REGISTRATION,),
    method_registrations=(),
    config_decoders=tuple(
        ObjectIdentity.from_record(value.registration_id, value) for value in _DECODERS
    ),
    artifact_validators=(),
    runtime_providers=tuple(
        ObjectIdentity.from_record(value.registration_id, value) for value in _RECONSTRUCTORS
    ),
    study_authors=(),
    grants_authority=False,
    embeds_scientific_payload=False,
)


__all__ = [
    "OBSERVATION_ORDER_EVALUATION_CAPABILITY",
    "OBSERVATION_ORDER_EVALUATION_CONFIG_DECODER",
    "OBSERVATION_ORDER_EVALUATION_PROVIDER",
    "OBSERVATION_ORDER_METHOD_IMPLEMENTATION_SHA256",
    "OBSERVATION_ORDER_PROJECTION_CAPABILITY",
    "OBSERVATION_ORDER_PROJECTION_CONFIG_DECODER",
    "OBSERVATION_ORDER_PROJECTION_PROVIDER",
    "MATRIX_RESPONSE_LAW_CONFIG_DECODER",
    "MATRIX_RESPONSE_LAW_CONFIG_RECONSTRUCTOR",
    "MATRIX_RESPONSE_REDUCER_CONFIG_DECODER",
    "MATRIX_RESPONSE_REDUCER_CONFIG_RECONSTRUCTOR",
    "REACTIVE_ENTRANCE_FINALIZATION_CAPABILITY",
    "REACTIVE_ENTRANCE_FINALIZATION_PROVIDER",
    "REACTIVE_ENTRANCE_METHOD_CONFIG_DECODER",
    "REACTIVE_ENTRANCE_METHOD_IMPLEMENTATION_SHA256",
    "REACTIVE_ENTRANCE_PROJECTION_CAPABILITY",
    "REACTIVE_ENTRANCE_PROJECTION_PROVIDER",
    "PROSPECTIVE_REACTIVE_SOURCE_LAW_FINALIZATION_CAPABILITY",
    "PROSPECTIVE_REACTIVE_SOURCE_LAW_FINALIZATION_PROVIDER",
    "PROSPECTIVE_REACTIVE_SOURCE_LAW_METHOD_CONFIG_DECODER",
    "PROSPECTIVE_REACTIVE_SOURCE_LAW_METHOD_IMPLEMENTATION_SHA256",
    "PROSPECTIVE_REACTIVE_SOURCE_LAW_PROJECTION_CAPABILITY",
    "PROSPECTIVE_REACTIVE_SOURCE_LAW_PROJECTION_PROVIDER",
    "MATRIX_RESPONSE_ROLE_CONFIG_DECODER",
    "MATRIX_RESPONSE_ROLE_CONFIG_RECONSTRUCTOR",
    "MATRIX_RESPONSE_CAUSAL_INTERSECTION_RESIDENCE_CONFIG_DECODER",
    "MATRIX_RESPONSE_CAUSAL_INTERSECTION_RESIDENCE_CONFIG_RECONSTRUCTOR",
    "MATRIX_RESPONSE_STRUCTURAL_CONFIG_DECODER",
    "MATRIX_RESPONSE_STRUCTURAL_CONFIG_RECONSTRUCTOR",
    "EXTENSION_BUNDLE_CONTRIBUTION",
]
