'Static discovery for method-owned structural transport and prospective structural recurrence.'

from __future__ import annotations

from hashlib import sha256

from empirical_lawhood.adapters.methods.structural_bootstrap_inputs import StructuralBootstrapInputCensus
from empirical_lawhood.adapters.methods.prospective_structural_recurrence import StructuralRecurrenceFrozenLawTransportForecasts, PROSPECTIVE_STRUCTURAL_RECURRENCE_IMPLEMENTATION_SHA256, PROSPECTIVE_STRUCTURAL_RECURRENCE_METHOD_KEY, PROSPECTIVE_STRUCTURAL_RECURRENCE_METHOD_VERSION, StructuralRecurrenceMetricDynamicalForecast, ProspectiveStructuralRecurrenceAdjudication, ProspectiveStructuralRecurrenceFaceTerminal, ProspectiveStructuralRecurrenceMethodSpec, ProspectiveStructuralRecurrencePlan, ProspectiveStructuralRecurrenceRosterIssue, ProspectiveStructuralRecurrenceTargetObservation, ProspectiveStructuralRecurrenceTargetTerminal
from empirical_lawhood.adapters.methods.structural_recurrence_targets import StructuralRecurrenceStageEvidence
from empirical_lawhood.adapters.methods.structural_recurrence import StructuralObservation, StructuralPredictionInput
from empirical_lawhood.adapters.methods.interval_property_comparison import INTERVAL_PROPERTY_COMPARISON_IMPLEMENTATION_SHA256, INTERVAL_PROPERTY_COMPARISON_METHOD_KEY, INTERVAL_PROPERTY_COMPARISON_METHOD_VERSION, IntervalPropertyComparisonMethodSpec, IntervalPropertyComparisonOperands, IntervalPropertyComparisonResult
from empirical_lawhood.adapters.methods.transformed_property_comparison import TRANSFORMED_PROPERTY_COMPARISON_IMPLEMENTATION_SHA256, TRANSFORMED_PROPERTY_COMPARISON_METHOD_KEY, TRANSFORMED_PROPERTY_COMPARISON_METHOD_VERSION, TransformedPropertyComparisonMethodSpec, TransformedPropertyComparisonOperands, TransformedPropertyComparisonResult
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.laws import ResponseLaw
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.contrasting_objectives import StructuralFaceHandoffProfile
from empirical_lawhood.planning.metatheory_campaign import MetatheoryCampaignProfile, MetatheoryCampaignStageRole
from empirical_lawhood.runtime.candidate_composition import CandidateCapabilityRegistration
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.capabilities import (
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
)
from empirical_lawhood.runtime.extension_bundles import ExtensionBundleContribution, ExtensionComponentKind, ExtensionComponentRegistration, ExtensionContributionKind, ExtensionProfileKind, ExtensionProfileRegistration
from empirical_lawhood.runtime.response_experiment import ResponseStageTerminal

from .campaign import StructuralDevelopmentBridgeConfig, StructuralDevelopmentInputs, StructuralPredictionFreezeReceipt, StructuralRecurrenceTransportConfig, StructuralReporterConfig, StructuralTargetBridgeConfig
from .provider import STRUCTURAL_TRANSPORT_MEDIA_TYPE
from .metatheory_conformance import EXECUTABLE_METATHEORY_ACQUISITION_KEY, EXECUTABLE_METATHEORY_CONFORMANCE_IMPLEMENTATION_SHA256, EXECUTABLE_METATHEORY_CONFORMANCE_VERSION, EXECUTABLE_METATHEORY_DEVELOPMENT_KEY, EXECUTABLE_METATHEORY_EVALUATOR_KEY, EXECUTABLE_METATHEORY_PREDICTION_KEY, EXECUTABLE_METATHEORY_QUALIFICATION_KEY, EXECUTABLE_METATHEORY_REPORTER_KEY, METATHEORY_CONFORMANCE_TERMINAL_TYPE_BY_ROLE, MetatheoryConformanceAcquisitionConfig, MetatheoryConformanceDevelopmentConfig, MetatheoryConformanceEvaluatorConfig, MetatheoryConformanceFixture, MetatheoryConformancePredictionConfig, MetatheoryConformanceQualificationConfig, MetatheoryConformanceReporterConfig
from .source_free_property_transport_contracts import source_free_property_transport_capability_registry
from .source_free_property_transport_records import SOURCE_FREE_PROPERTY_TRANSPORT_CONFIG_TYPE_BY_ROLE


STRUCTURAL_TRANSPORT_PROVIDER_IMPLEMENTATION_SHA256 = sha256(
    b"empirical-lawhood/backbone-structural-transport-provider"
).hexdigest()
_MAXIMUM_CONFIG_BYTES = 4 * 1024 * 1024
_RESOURCE = ResourceBudget(
    cpu_cores=2,
    memory_bytes=2 * 1024**3,
    gpu_devices=0,
    wall_time_seconds=600,
    source_scan_bytes=256 * 1024**2,
    output_bytes=256 * 1024**2,
)
_PERMISSIONS = tuple(
    sorted(
        (
            CapabilityPermission.READ_DEVELOPMENT,
            CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
            CapabilityPermission.READ_OUTCOME_VISIBLE,
            CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
        )
    )
)


INTERVAL_PROPERTY_COMPARISON_MANIFEST = CapabilityManifest(
    capability_key=INTERVAL_PROPERTY_COMPARISON_METHOD_KEY,
    capability_version=INTERVAL_PROPERTY_COMPARISON_METHOD_VERSION,
    kind=CapabilityKind.TRANSPORT_TESTER,
    config_schema=IntervalPropertyComparisonMethodSpec.SCHEMA,
    config_schema_sha256=sha256(IntervalPropertyComparisonMethodSpec.SCHEMA.encode()).hexdigest(),
    input_schema_ids=(IntervalPropertyComparisonOperands.SCHEMA,),
    output_schema_ids=(IntervalPropertyComparisonResult.SCHEMA,),
    permissions=_PERMISSIONS,
    maximum_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
    maximum_outcome_access=OutcomeAccess.EVALUATION_REVEALED,
    resource_ceiling=_RESOURCE,
    deterministic=True,
    seed_required=False,
    language_id="python",
    runtime_id='cpython-3.11-interval-property-comparison',
    requires_clean_commit=True,
    requires_active_mount=False,
    requires_network=False,
    conformance_check_ids=(
        "eight-kind-method-owned-falsifiers",
        "no-caller-preservation-verdict",
        "source-target-face-identity",
    ),
    implementation_sha256=INTERVAL_PROPERTY_COMPARISON_IMPLEMENTATION_SHA256,
)
TRANSFORMED_PROPERTY_COMPARISON_MANIFEST = CapabilityManifest(
    capability_key=TRANSFORMED_PROPERTY_COMPARISON_METHOD_KEY,
    capability_version=TRANSFORMED_PROPERTY_COMPARISON_METHOD_VERSION,
    kind=CapabilityKind.TRANSPORT_TESTER,
    config_schema=TransformedPropertyComparisonMethodSpec.SCHEMA,
    config_schema_sha256=sha256(TransformedPropertyComparisonMethodSpec.SCHEMA.encode()).hexdigest(),
    input_schema_ids=(TransformedPropertyComparisonOperands.SCHEMA,),
    output_schema_ids=(TransformedPropertyComparisonResult.SCHEMA,),
    permissions=_PERMISSIONS,
    maximum_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
    maximum_outcome_access=OutcomeAccess.EVALUATION_REVEALED,
    resource_ceiling=_RESOURCE,
    deterministic=True,
    seed_required=False,
    language_id="python",
    runtime_id='cpython-3.11-transformed-property-comparison',
    requires_clean_commit=True,
    requires_active_mount=False,
    requires_network=False,
    conformance_check_ids=tuple(
        sorted(
            (
                "eight-distinct-kind-local-algorithms",
                "typed-unit-frame-direction-map-applied",
                "boundary-uncertainty-and-falsifier-owned",
                "no-caller-preservation-verdict",
            )
        )
    ),
    implementation_sha256=TRANSFORMED_PROPERTY_COMPARISON_IMPLEMENTATION_SHA256,
)

_PROSPECTIVE_STRUCTURAL_RECURRENCE_INPUT_SCHEMAS = tuple(
    sorted(
        (
            StructuralBootstrapInputCensus.SCHEMA,
            ProspectiveStructuralRecurrencePlan.SCHEMA,
            ProspectiveStructuralRecurrenceRosterIssue.SCHEMA,
            StructuralRecurrenceMetricDynamicalForecast.SCHEMA,
            StructuralRecurrenceStageEvidence.SCHEMA,
            IntervalPropertyComparisonResult.SCHEMA,
            StructuralObservation.SCHEMA,
            StructuralPredictionInput.SCHEMA,
        )
    )
)
_PROSPECTIVE_STRUCTURAL_RECURRENCE_OUTPUT_SCHEMAS = tuple(
    sorted(
        (
            ProspectiveStructuralRecurrenceAdjudication.SCHEMA,
            ProspectiveStructuralRecurrenceFaceTerminal.SCHEMA,
            ProspectiveStructuralRecurrenceRosterIssue.SCHEMA,
            ProspectiveStructuralRecurrenceTargetObservation.SCHEMA,
            ProspectiveStructuralRecurrenceTargetTerminal.SCHEMA,
        )
    )
)
PROSPECTIVE_STRUCTURAL_RECURRENCE_MANIFEST = CapabilityManifest(
    capability_key=PROSPECTIVE_STRUCTURAL_RECURRENCE_METHOD_KEY,
    capability_version=PROSPECTIVE_STRUCTURAL_RECURRENCE_METHOD_VERSION,
    kind=CapabilityKind.HYPOTHESIS_ADJUDICATOR,
    config_schema=ProspectiveStructuralRecurrenceMethodSpec.SCHEMA,
    config_schema_sha256=sha256(ProspectiveStructuralRecurrenceMethodSpec.SCHEMA.encode()).hexdigest(),
    input_schema_ids=_PROSPECTIVE_STRUCTURAL_RECURRENCE_INPUT_SCHEMAS,
    output_schema_ids=_PROSPECTIVE_STRUCTURAL_RECURRENCE_OUTPUT_SCHEMAS,
    permissions=_PERMISSIONS,
    maximum_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
    maximum_outcome_access=OutcomeAccess.EVALUATION_REVEALED,
    resource_ceiling=_RESOURCE,
    deterministic=True,
    seed_required=False,
    language_id="python",
    runtime_id='cpython-3.11-structural-recurrence-prospective',
    requires_clean_commit=True,
    requires_active_mount=False,
    requires_network=False,
    conformance_check_ids=(
        "arbitrary-target-roster",
        'facewise-measurement-through-law-qualification',
        "no-cross-target-pooling",
        "prospective-issue-before-target-outcomes",
    ),
    implementation_sha256=PROSPECTIVE_STRUCTURAL_RECURRENCE_IMPLEMENTATION_SHA256,
)


STRUCTURAL_DEVELOPMENT_BRIDGE_KEY = "method.structural-development-bridge"
STRUCTURAL_RECURRENCE_TRANSPORT_KEY = 'method.structural-recurrence-transport'
STRUCTURAL_TARGET_BRIDGE_KEY = "method.structural-target-bridge"
STRUCTURAL_REPORTER_KEY = "method.structural-reporter"
STRUCTURAL_CAMPAIGN_IMPLEMENTATION_SHA256 = sha256(
    b"empirical-lawhood/backbone-structural-transport-campaign"
).hexdigest()

STRUCTURAL_DEVELOPMENT_BRIDGE_MANIFEST = CapabilityManifest(
    capability_key=STRUCTURAL_DEVELOPMENT_BRIDGE_KEY,
    capability_version="1.0.0",
    kind=CapabilityKind.ANALYSIS,
    config_schema=StructuralDevelopmentBridgeConfig.SCHEMA,
    config_schema_sha256=sha256(StructuralDevelopmentBridgeConfig.SCHEMA.encode()).hexdigest(),
    input_schema_ids=(StructuralDevelopmentInputs.SCHEMA,),
    output_schema_ids=tuple(
        sorted(
            (
                StructuralBootstrapInputCensus.SCHEMA,
                ProspectiveStructuralRecurrencePlan.SCHEMA,
                StructuralRecurrenceStageEvidence.SCHEMA,
                StructuralPredictionInput.SCHEMA,
            )
        )
    ),
    permissions=_PERMISSIONS,
    maximum_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
    maximum_outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
    resource_ceiling=_RESOURCE,
    deterministic=True,
    seed_required=False,
    language_id="python",
    runtime_id="cpython-3.11-structural-development-bridge",
    requires_clean_commit=True,
    requires_active_mount=False,
    requires_network=False,
    conformance_check_ids=(
        "authenticated-development-inputs",
        "development-products-only",
        "zero-target-outcome-access",
    ),
    implementation_sha256=STRUCTURAL_CAMPAIGN_IMPLEMENTATION_SHA256,
)

_STRUCTURAL_RECURRENCE_TRANSPORT_INPUT_SCHEMAS = tuple(
    sorted(
        (
            StructuralBootstrapInputCensus.SCHEMA,
            ResponseStageTerminal.SCHEMA,
            ResponseLaw.SCHEMA,
            ProspectiveStructuralRecurrenceMethodSpec.SCHEMA,
            ProspectiveStructuralRecurrencePlan.SCHEMA,
            StructuralRecurrenceFrozenLawTransportForecasts.SCHEMA,
            ProspectiveStructuralRecurrenceRosterIssue.SCHEMA,
            StructuralRecurrenceStageEvidence.SCHEMA,
            IntervalPropertyComparisonResult.SCHEMA,
            StructuralObservation.SCHEMA,
            StructuralPredictionInput.SCHEMA,
        )
    )
)
_STRUCTURAL_RECURRENCE_TRANSPORT_OUTPUT_SCHEMAS = tuple(
    sorted((*_PROSPECTIVE_STRUCTURAL_RECURRENCE_OUTPUT_SCHEMAS, StructuralPredictionFreezeReceipt.SCHEMA))
)
STRUCTURAL_RECURRENCE_TRANSPORT_MANIFEST = CapabilityManifest(
    capability_key=STRUCTURAL_RECURRENCE_TRANSPORT_KEY,
    capability_version="1.0.0",
    kind=CapabilityKind.HYPOTHESIS_ADJUDICATOR,
    config_schema=StructuralRecurrenceTransportConfig.SCHEMA,
    config_schema_sha256=sha256(StructuralRecurrenceTransportConfig.SCHEMA.encode()).hexdigest(),
    input_schema_ids=_STRUCTURAL_RECURRENCE_TRANSPORT_INPUT_SCHEMAS,
    output_schema_ids=_STRUCTURAL_RECURRENCE_TRANSPORT_OUTPUT_SCHEMAS,
    permissions=_PERMISSIONS,
    maximum_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
    maximum_outcome_access=OutcomeAccess.EVALUATION_REVEALED,
    resource_ceiling=_RESOURCE,
    deterministic=True,
    seed_required=False,
    language_id="python",
    runtime_id='cpython-3.11-structural-recurrence-transport',
    requires_clean_commit=True,
    requires_active_mount=False,
    requires_network=False,
    conformance_check_ids=(
        "complete-five-parent-law-transport",
        'facewise-structural-recurrence-adjudication',
        "no-cross-target-pooling",
        "prediction-freeze-before-target",
    ),
    implementation_sha256=STRUCTURAL_CAMPAIGN_IMPLEMENTATION_SHA256,
)

STRUCTURAL_TARGET_BRIDGE_MANIFEST = CapabilityManifest(
    capability_key=STRUCTURAL_TARGET_BRIDGE_KEY,
    capability_version="1.0.0",
    kind=CapabilityKind.EVALUATOR,
    config_schema=StructuralTargetBridgeConfig.SCHEMA,
    config_schema_sha256=sha256(StructuralTargetBridgeConfig.SCHEMA.encode()).hexdigest(),
    input_schema_ids=(StructuralPredictionFreezeReceipt.SCHEMA,),
    output_schema_ids=tuple(
        sorted(
            (
                StructuralRecurrenceStageEvidence.SCHEMA,
                StructuralObservation.SCHEMA,
                IntervalPropertyComparisonOperands.SCHEMA,
            )
        )
    ),
    permissions=_PERMISSIONS,
    maximum_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
    maximum_outcome_access=OutcomeAccess.EVALUATION_REVEALED,
    resource_ceiling=_RESOURCE,
    deterministic=True,
    seed_required=False,
    language_id="python",
    runtime_id="cpython-3.11-structural-target-bridge",
    requires_clean_commit=True,
    requires_active_mount=False,
    requires_network=False,
    conformance_check_ids=(
        "eight-kind-typed-operands",
        "prediction-freeze-before-target-contact",
        "single-target-source-effect",
    ),
    implementation_sha256=STRUCTURAL_CAMPAIGN_IMPLEMENTATION_SHA256,
)

STRUCTURAL_REPORTER_MANIFEST = CapabilityManifest(
    capability_key=STRUCTURAL_REPORTER_KEY,
    capability_version="1.0.0",
    kind=CapabilityKind.REPORTER,
    config_schema=StructuralReporterConfig.SCHEMA,
    config_schema_sha256=sha256(StructuralReporterConfig.SCHEMA.encode()).hexdigest(),
    input_schema_ids=(ProspectiveStructuralRecurrenceAdjudication.SCHEMA,),
    output_schema_ids=(ScientificAdjudicationRecord.SCHEMA,),
    permissions=_PERMISSIONS,
    maximum_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
    maximum_outcome_access=OutcomeAccess.EVALUATION_REVEALED,
    resource_ceiling=_RESOURCE,
    deterministic=True,
    seed_required=False,
    language_id="python",
    runtime_id="cpython-3.11-structural-reporter",
    requires_clean_commit=True,
    requires_active_mount=False,
    requires_network=False,
    conformance_check_ids=('sole-structural-recurrence-adjudication-input',),
    implementation_sha256=STRUCTURAL_CAMPAIGN_IMPLEMENTATION_SHA256,
)


def _metatheory_manifest(
    *,
    capability_key: str,
    kind: CapabilityKind,
    config_type: type[CanonicalRecord],
    input_schema_ids: tuple[str, ...],
    output_schema_ids: tuple[str, ...],
    maximum_outcome_access: OutcomeAccess,
    permissions: tuple[CapabilityPermission, ...],
) -> CapabilityManifest:
    return CapabilityManifest(
        capability_key=capability_key,
        capability_version=EXECUTABLE_METATHEORY_CONFORMANCE_VERSION,
        kind=kind,
        config_schema=config_type.SCHEMA,
        config_schema_sha256=sha256(config_type.SCHEMA.encode()).hexdigest(),
        input_schema_ids=tuple(sorted(input_schema_ids)),
        output_schema_ids=tuple(sorted(output_schema_ids)),
        permissions=permissions,
        maximum_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        maximum_outcome_access=maximum_outcome_access,
        resource_ceiling=_RESOURCE,
        deterministic=True,
        seed_required=False,
        language_id="python",
        runtime_id=f"cpython-3.11-{capability_key}",
        requires_clean_commit=True,
        requires_active_mount=False,
        requires_network=False,
        conformance_check_ids=(
            "closed-stage-role-roster",
            "contract-conformance-ceiling-only",
            "freeze-authority-reveal-ordering",
            "plumbing-only-scientific-adjudication",
        ),
        implementation_sha256=(EXECUTABLE_METATHEORY_CONFORMANCE_IMPLEMENTATION_SHA256),
    )


def _metatheory_terminal_schemas(
    *roles: MetatheoryCampaignStageRole,
) -> tuple[str, ...]:
    return tuple(
        sorted(METATHEORY_CONFORMANCE_TERMINAL_TYPE_BY_ROLE[value].SCHEMA for value in roles)
    )


EXECUTABLE_METATHEORY_ACQUISITION_MANIFEST = _metatheory_manifest(
    capability_key=EXECUTABLE_METATHEORY_ACQUISITION_KEY,
    kind=CapabilityKind.SOURCE,
    config_type=MetatheoryConformanceAcquisitionConfig,
    input_schema_ids=(
        MetatheoryConformanceFixture.SCHEMA,
        *_metatheory_terminal_schemas(
            MetatheoryCampaignStageRole.PREDICTION_ISSUE,
        ),
    ),
    output_schema_ids=_metatheory_terminal_schemas(
        MetatheoryCampaignStageRole.SOURCE_PIPELINE,
        MetatheoryCampaignStageRole.TARGET_ACQUISITION,
    ),
    maximum_outcome_access=OutcomeAccess.EVALUATION_SEALED,
    permissions=(
        CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
        CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
    ),
)
EXECUTABLE_METATHEORY_DEVELOPMENT_MANIFEST = _metatheory_manifest(
    capability_key=EXECUTABLE_METATHEORY_DEVELOPMENT_KEY,
    kind=CapabilityKind.ANALYSIS,
    config_type=MetatheoryConformanceDevelopmentConfig,
    input_schema_ids=_metatheory_terminal_schemas(
        MetatheoryCampaignStageRole.SCIENTIFIC_SOURCE_QUALIFICATION,
        MetatheoryCampaignStageRole.REVEAL_AND_ADJUDICATION,
    ),
    output_schema_ids=_metatheory_terminal_schemas(
        MetatheoryCampaignStageRole.COORDINATE_CONSTRUCTION,
        MetatheoryCampaignStageRole.OBSTRUCTION_CLOSEOUT,
    ),
    maximum_outcome_access=OutcomeAccess.EVALUATION_REVEALED,
    permissions=(
        CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
        CapabilityPermission.READ_OUTCOME_VISIBLE,
        CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
    ),
)
EXECUTABLE_METATHEORY_QUALIFICATION_MANIFEST = _metatheory_manifest(
    capability_key=EXECUTABLE_METATHEORY_QUALIFICATION_KEY,
    kind=CapabilityKind.NUMERICAL_QUALIFIER,
    config_type=MetatheoryConformanceQualificationConfig,
    input_schema_ids=_metatheory_terminal_schemas(
        MetatheoryCampaignStageRole.SOURCE_PIPELINE,
        MetatheoryCampaignStageRole.COORDINATE_CONSTRUCTION,
        MetatheoryCampaignStageRole.COORDINATE_EVALUATION,
        MetatheoryCampaignStageRole.ATLAS_QUALIFICATION_OPTIONAL,
        MetatheoryCampaignStageRole.DECISION_ASSURANCE_OPTIONAL,
    ),
    output_schema_ids=_metatheory_terminal_schemas(
        MetatheoryCampaignStageRole.SCIENTIFIC_SOURCE_QUALIFICATION,
        MetatheoryCampaignStageRole.COORDINATE_EVALUATION,
        MetatheoryCampaignStageRole.ATLAS_QUALIFICATION_OPTIONAL,
        MetatheoryCampaignStageRole.DECISION_ASSURANCE_OPTIONAL,
        MetatheoryCampaignStageRole.PROPERTY_SURVIVAL_OPTIONAL,
    ),
    maximum_outcome_access=OutcomeAccess.OUTCOME_BLIND,
    permissions=(
        CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
        CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
    ),
)
EXECUTABLE_METATHEORY_EVALUATOR_MANIFEST = _metatheory_manifest(
    capability_key=EXECUTABLE_METATHEORY_EVALUATOR_KEY,
    kind=CapabilityKind.EVALUATOR,
    config_type=MetatheoryConformanceEvaluatorConfig,
    input_schema_ids=_metatheory_terminal_schemas(
        MetatheoryCampaignStageRole.TARGET_ACQUISITION,
    ),
    output_schema_ids=_metatheory_terminal_schemas(
        MetatheoryCampaignStageRole.REVEAL_AND_ADJUDICATION,
    ),
    maximum_outcome_access=OutcomeAccess.EVALUATION_REVEALED,
    permissions=(
        CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
        CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
    ),
)
EXECUTABLE_METATHEORY_PREDICTION_MANIFEST = _metatheory_manifest(
    capability_key=EXECUTABLE_METATHEORY_PREDICTION_KEY,
    kind=CapabilityKind.EVALUATOR,
    config_type=MetatheoryConformancePredictionConfig,
    input_schema_ids=_metatheory_terminal_schemas(
        MetatheoryCampaignStageRole.PROPERTY_SURVIVAL_OPTIONAL,
    ),
    output_schema_ids=_metatheory_terminal_schemas(
        MetatheoryCampaignStageRole.PREDICTION_ISSUE,
    ),
    maximum_outcome_access=OutcomeAccess.OUTCOME_BLIND,
    permissions=(
        CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
        CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
    ),
)
EXECUTABLE_METATHEORY_REPORTER_MANIFEST = _metatheory_manifest(
    capability_key=EXECUTABLE_METATHEORY_REPORTER_KEY,
    kind=CapabilityKind.REPORTER,
    config_type=MetatheoryConformanceReporterConfig,
    input_schema_ids=_metatheory_terminal_schemas(
        MetatheoryCampaignStageRole.OBSTRUCTION_CLOSEOUT,
    ),
    output_schema_ids=(ScientificAdjudicationRecord.SCHEMA,),
    maximum_outcome_access=OutcomeAccess.EVALUATION_REVEALED,
    permissions=(
        CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
        CapabilityPermission.READ_OUTCOME_VISIBLE,
        CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
    ),
)
EXECUTABLE_METATHEORY_CONFORMANCE_MANIFESTS = tuple(
    sorted(
        (
            EXECUTABLE_METATHEORY_ACQUISITION_MANIFEST,
            EXECUTABLE_METATHEORY_DEVELOPMENT_MANIFEST,
            EXECUTABLE_METATHEORY_EVALUATOR_MANIFEST,
            EXECUTABLE_METATHEORY_PREDICTION_MANIFEST,
            EXECUTABLE_METATHEORY_QUALIFICATION_MANIFEST,
            EXECUTABLE_METATHEORY_REPORTER_MANIFEST,
        ),
        key=lambda value: value.registry_id,
    )
)


INTERVAL_PROPERTY_COMPARISON_CANDIDATE_REGISTRATION = CandidateCapabilityRegistration(
    manifest=INTERVAL_PROPERTY_COMPARISON_MANIFEST,
    provider_key='structural-transport.interval-property-comparison-provider',
    provider_version="1.0.0",
    config_media_type=STRUCTURAL_TRANSPORT_MEDIA_TYPE,
    maximum_config_bytes=_MAXIMUM_CONFIG_BYTES,
)
TRANSFORMED_PROPERTY_COMPARISON_CANDIDATE_REGISTRATION = CandidateCapabilityRegistration(
    manifest=TRANSFORMED_PROPERTY_COMPARISON_MANIFEST,
    provider_key='structural-transport.transformed-property-comparison-provider',
    provider_version="1.0.0",
    config_media_type=STRUCTURAL_TRANSPORT_MEDIA_TYPE,
    maximum_config_bytes=_MAXIMUM_CONFIG_BYTES,
)
PROSPECTIVE_STRUCTURAL_RECURRENCE_CANDIDATE_REGISTRATION = CandidateCapabilityRegistration(
    manifest=PROSPECTIVE_STRUCTURAL_RECURRENCE_MANIFEST,
    provider_key='structural-transport.structural-recurrence-prospective-provider',
    provider_version="1.0.0",
    config_media_type=STRUCTURAL_TRANSPORT_MEDIA_TYPE,
    maximum_config_bytes=_MAXIMUM_CONFIG_BYTES,
)
STRUCTURAL_DEVELOPMENT_BRIDGE_CANDIDATE_REGISTRATION = CandidateCapabilityRegistration(
    manifest=STRUCTURAL_DEVELOPMENT_BRIDGE_MANIFEST,
    provider_key="structural-transport.development-bridge-provider",
    provider_version="1.0.0",
    config_media_type=STRUCTURAL_TRANSPORT_MEDIA_TYPE,
    maximum_config_bytes=_MAXIMUM_CONFIG_BYTES,
)
STRUCTURAL_RECURRENCE_TRANSPORT_CANDIDATE_REGISTRATION = CandidateCapabilityRegistration(
    manifest=STRUCTURAL_RECURRENCE_TRANSPORT_MANIFEST,
    provider_key='structural-transport.structural-recurrence-transport-provider',
    provider_version="1.0.0",
    config_media_type=STRUCTURAL_TRANSPORT_MEDIA_TYPE,
    maximum_config_bytes=_MAXIMUM_CONFIG_BYTES,
)
STRUCTURAL_TARGET_BRIDGE_CANDIDATE_REGISTRATION = CandidateCapabilityRegistration(
    manifest=STRUCTURAL_TARGET_BRIDGE_MANIFEST,
    provider_key="structural-transport.target-bridge-provider",
    provider_version="1.0.0",
    config_media_type=STRUCTURAL_TRANSPORT_MEDIA_TYPE,
    maximum_config_bytes=_MAXIMUM_CONFIG_BYTES,
)
STRUCTURAL_REPORTER_CANDIDATE_REGISTRATION = CandidateCapabilityRegistration(
    manifest=STRUCTURAL_REPORTER_MANIFEST,
    provider_key="structural-transport.reporter-provider",
    provider_version="1.0.0",
    config_media_type=STRUCTURAL_TRANSPORT_MEDIA_TYPE,
    maximum_config_bytes=_MAXIMUM_CONFIG_BYTES,
)
EXECUTABLE_METATHEORY_CONFORMANCE_CANDIDATE_REGISTRATIONS = tuple(
    CandidateCapabilityRegistration(
        manifest=value,
        provider_key=f"structural-transport.{value.capability_key}-provider",
        provider_version="1.0.0",
        config_media_type=STRUCTURAL_TRANSPORT_MEDIA_TYPE,
        maximum_config_bytes=_MAXIMUM_CONFIG_BYTES,
    )
    for value in EXECUTABLE_METATHEORY_CONFORMANCE_MANIFESTS
)

STRUCTURAL_FACE_PROFILE_REGISTRATION = ExtensionProfileRegistration(
    registration_id="profile-registration.structural-face-handoff",
    profile_key="structural-face-handoff",
    profile_version="1.0.0",
    kind=ExtensionProfileKind.OBJECTIVE_TERMINAL,
    profile_schema=StructuralFaceHandoffProfile.SCHEMA,
    profile_schema_sha256=sha256(StructuralFaceHandoffProfile.SCHEMA.encode()).hexdigest(),
    implementation_sha256=STRUCTURAL_TRANSPORT_PROVIDER_IMPLEMENTATION_SHA256,
)
EXECUTABLE_METATHEORY_PROFILE_REGISTRATION = ExtensionProfileRegistration(
    registration_id="profile-registration.executable-metatheory-campaign",
    profile_key="executable-metatheory-campaign",
    profile_version="1.0.0",
    kind=ExtensionProfileKind.LINKED_CAMPAIGN,
    profile_schema=MetatheoryCampaignProfile.SCHEMA,
    profile_schema_sha256=sha256(MetatheoryCampaignProfile.SCHEMA.encode()).hexdigest(),
    implementation_sha256=STRUCTURAL_TRANSPORT_PROVIDER_IMPLEMENTATION_SHA256,
)


def _component(
    *,
    registration_id: str,
    component_key: str,
    kind: ExtensionComponentKind,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    component_version: str = "1.0.0",
    implementation_sha256: str = STRUCTURAL_TRANSPORT_PROVIDER_IMPLEMENTATION_SHA256,
) -> ExtensionComponentRegistration:
    return ExtensionComponentRegistration(
        registration_id=registration_id,
        component_key=component_key,
        component_version=component_version,
        kind=kind,
        input_schema_ids=tuple(sorted(inputs)),
        output_schema_ids=tuple(sorted(outputs)),
        implementation_sha256=implementation_sha256,
    )


INTERVAL_PROPERTY_COMPARISON_METHOD = ExtensionComponentRegistration(
    registration_id="component.interval-property-comparison",
    component_key=INTERVAL_PROPERTY_COMPARISON_METHOD_KEY,
    component_version=INTERVAL_PROPERTY_COMPARISON_METHOD_VERSION,
    kind=ExtensionComponentKind.METHOD,
    input_schema_ids=INTERVAL_PROPERTY_COMPARISON_MANIFEST.input_schema_ids,
    output_schema_ids=INTERVAL_PROPERTY_COMPARISON_MANIFEST.output_schema_ids,
    implementation_sha256=INTERVAL_PROPERTY_COMPARISON_IMPLEMENTATION_SHA256,
)
INTERVAL_PROPERTY_COMPARISON_CONFIG_DECODER = _component(
    registration_id='component.interval-property-comparison-method-spec-decoder',
    component_key='interval-property-comparison-method-spec-decoder',
    kind=ExtensionComponentKind.CONFIG_DECODER,
    inputs=(IntervalPropertyComparisonMethodSpec.SCHEMA,),
    outputs=(IntervalPropertyComparisonMethodSpec.SCHEMA,),
)
INTERVAL_PROPERTY_COMPARISON_ARTIFACT_VALIDATOR = _component(
    registration_id='component.interval-property-comparison-result-validator',
    component_key='interval-property-comparison-result-validator',
    kind=ExtensionComponentKind.ARTIFACT_VALIDATOR,
    inputs=(IntervalPropertyComparisonResult.SCHEMA,),
    outputs=('empirical-lawhood/runtime/artifact-generic-validation',),
)
INTERVAL_PROPERTY_COMPARISON_RUNTIME_PROVIDER = _component(
    registration_id='component.interval-property-comparison-runtime-provider',
    component_key=INTERVAL_PROPERTY_COMPARISON_CANDIDATE_REGISTRATION.provider_key,
    kind=ExtensionComponentKind.RUNTIME_PROVIDER,
    inputs=INTERVAL_PROPERTY_COMPARISON_MANIFEST.input_schema_ids,
    outputs=INTERVAL_PROPERTY_COMPARISON_MANIFEST.output_schema_ids,
)
TRANSFORMED_PROPERTY_COMPARISON_METHOD = ExtensionComponentRegistration(
    registration_id="component.transformed-property-comparison",
    component_key=TRANSFORMED_PROPERTY_COMPARISON_METHOD_KEY,
    component_version=TRANSFORMED_PROPERTY_COMPARISON_METHOD_VERSION,
    kind=ExtensionComponentKind.METHOD,
    input_schema_ids=TRANSFORMED_PROPERTY_COMPARISON_MANIFEST.input_schema_ids,
    output_schema_ids=TRANSFORMED_PROPERTY_COMPARISON_MANIFEST.output_schema_ids,
    implementation_sha256=TRANSFORMED_PROPERTY_COMPARISON_IMPLEMENTATION_SHA256,
)
TRANSFORMED_PROPERTY_COMPARISON_CONFIG_DECODER = _component(
    registration_id='component.transformed-property-comparison-method-spec-decoder',
    component_key='transformed-property-comparison-method-spec-decoder',
    component_version="1.0.0",
    kind=ExtensionComponentKind.CONFIG_DECODER,
    inputs=(TransformedPropertyComparisonMethodSpec.SCHEMA,),
    outputs=(TransformedPropertyComparisonMethodSpec.SCHEMA,),
)
TRANSFORMED_PROPERTY_COMPARISON_ARTIFACT_VALIDATOR = _component(
    registration_id='component.transformed-property-comparison-result-validator',
    component_key='transformed-property-comparison-result-validator',
    component_version="1.0.0",
    kind=ExtensionComponentKind.ARTIFACT_VALIDATOR,
    inputs=(TransformedPropertyComparisonResult.SCHEMA,),
    outputs=('empirical-lawhood/runtime/artifact-generic-validation',),
)
TRANSFORMED_PROPERTY_COMPARISON_RUNTIME_PROVIDER = _component(
    registration_id='component.transformed-property-comparison-runtime-provider',
    component_key=TRANSFORMED_PROPERTY_COMPARISON_CANDIDATE_REGISTRATION.provider_key,
    component_version="1.0.0",
    kind=ExtensionComponentKind.RUNTIME_PROVIDER,
    inputs=TRANSFORMED_PROPERTY_COMPARISON_MANIFEST.input_schema_ids,
    outputs=TRANSFORMED_PROPERTY_COMPARISON_MANIFEST.output_schema_ids,
)

PROSPECTIVE_STRUCTURAL_RECURRENCE_METHOD = ExtensionComponentRegistration(
    registration_id='component.structural-recurrence-prospective-method',
    component_key=PROSPECTIVE_STRUCTURAL_RECURRENCE_METHOD_KEY,
    component_version=PROSPECTIVE_STRUCTURAL_RECURRENCE_METHOD_VERSION,
    kind=ExtensionComponentKind.METHOD,
    input_schema_ids=PROSPECTIVE_STRUCTURAL_RECURRENCE_MANIFEST.input_schema_ids,
    output_schema_ids=PROSPECTIVE_STRUCTURAL_RECURRENCE_MANIFEST.output_schema_ids,
    implementation_sha256=PROSPECTIVE_STRUCTURAL_RECURRENCE_IMPLEMENTATION_SHA256,
)
PROSPECTIVE_STRUCTURAL_RECURRENCE_CONFIG_DECODER = _component(
    registration_id='component.structural-recurrence-prospective-method-spec-decoder',
    component_key='structural-recurrence-prospective-method-spec-decoder',
    kind=ExtensionComponentKind.CONFIG_DECODER,
    inputs=(ProspectiveStructuralRecurrenceMethodSpec.SCHEMA,),
    outputs=(ProspectiveStructuralRecurrenceMethodSpec.SCHEMA,),
)
PROSPECTIVE_STRUCTURAL_RECURRENCE_ARTIFACT_VALIDATOR = _component(
    registration_id='component.structural-recurrence-prospective-artifact-validator',
    component_key='structural-recurrence-prospective-artifact-validator',
    kind=ExtensionComponentKind.ARTIFACT_VALIDATOR,
    inputs=_PROSPECTIVE_STRUCTURAL_RECURRENCE_OUTPUT_SCHEMAS,
    outputs=('empirical-lawhood/runtime/artifact-generic-validation',),
)
PROSPECTIVE_STRUCTURAL_RECURRENCE_RUNTIME_PROVIDER = _component(
    registration_id='component.structural-recurrence-prospective-runtime-provider',
    component_key=PROSPECTIVE_STRUCTURAL_RECURRENCE_CANDIDATE_REGISTRATION.provider_key,
    kind=ExtensionComponentKind.RUNTIME_PROVIDER,
    inputs=_PROSPECTIVE_STRUCTURAL_RECURRENCE_INPUT_SCHEMAS,
    outputs=_PROSPECTIVE_STRUCTURAL_RECURRENCE_OUTPUT_SCHEMAS,
)


def _campaign_components(
    *,
    token: str,
    manifest: CapabilityManifest,
    candidate: CandidateCapabilityRegistration,
) -> tuple[
    ExtensionComponentRegistration,
    ExtensionComponentRegistration,
    ExtensionComponentRegistration,
    ExtensionComponentRegistration,
]:
    method = _component(
        registration_id=f"component.{token}-method",
        component_key=manifest.capability_key,
        kind=ExtensionComponentKind.METHOD,
        inputs=manifest.input_schema_ids,
        outputs=manifest.output_schema_ids,
        implementation_sha256=manifest.implementation_sha256,
    )
    decoder = _component(
        registration_id=f"component.{token}-config-decoder",
        component_key=f"{token}-config-decoder",
        kind=ExtensionComponentKind.CONFIG_DECODER,
        inputs=(manifest.config_schema,),
        outputs=(manifest.config_schema,),
    )
    validator = _component(
        registration_id=f"component.{token}-artifact-validator",
        component_key=f"{token}-artifact-validator",
        kind=ExtensionComponentKind.ARTIFACT_VALIDATOR,
        inputs=manifest.output_schema_ids,
        outputs=('empirical-lawhood/runtime/artifact-generic-validation',),
    )
    provider = _component(
        registration_id=f"component.{token}-runtime-provider",
        component_key=candidate.provider_key,
        kind=ExtensionComponentKind.RUNTIME_PROVIDER,
        inputs=manifest.input_schema_ids,
        outputs=manifest.output_schema_ids,
    )
    return method, decoder, validator, provider


(
    STRUCTURAL_DEVELOPMENT_BRIDGE_METHOD,
    STRUCTURAL_DEVELOPMENT_BRIDGE_CONFIG_DECODER,
    STRUCTURAL_DEVELOPMENT_BRIDGE_ARTIFACT_VALIDATOR,
    STRUCTURAL_DEVELOPMENT_BRIDGE_RUNTIME_PROVIDER,
) = _campaign_components(
    token="structural-development-bridge",
    manifest=STRUCTURAL_DEVELOPMENT_BRIDGE_MANIFEST,
    candidate=STRUCTURAL_DEVELOPMENT_BRIDGE_CANDIDATE_REGISTRATION,
)
(
    STRUCTURAL_RECURRENCE_TRANSPORT_METHOD,
    STRUCTURAL_RECURRENCE_TRANSPORT_CONFIG_DECODER,
    STRUCTURAL_RECURRENCE_TRANSPORT_ARTIFACT_VALIDATOR,
    STRUCTURAL_RECURRENCE_TRANSPORT_RUNTIME_PROVIDER,
) = _campaign_components(
    token='structural-recurrence-transport',
    manifest=STRUCTURAL_RECURRENCE_TRANSPORT_MANIFEST,
    candidate=STRUCTURAL_RECURRENCE_TRANSPORT_CANDIDATE_REGISTRATION,
)
(
    STRUCTURAL_TARGET_BRIDGE_METHOD,
    STRUCTURAL_TARGET_BRIDGE_CONFIG_DECODER,
    STRUCTURAL_TARGET_BRIDGE_ARTIFACT_VALIDATOR,
    STRUCTURAL_TARGET_BRIDGE_RUNTIME_PROVIDER,
) = _campaign_components(
    token="structural-target-bridge",
    manifest=STRUCTURAL_TARGET_BRIDGE_MANIFEST,
    candidate=STRUCTURAL_TARGET_BRIDGE_CANDIDATE_REGISTRATION,
)
(
    STRUCTURAL_REPORTER_METHOD,
    STRUCTURAL_REPORTER_CONFIG_DECODER,
    STRUCTURAL_REPORTER_ARTIFACT_VALIDATOR,
    STRUCTURAL_REPORTER_RUNTIME_PROVIDER,
) = _campaign_components(
    token="structural-reporter",
    manifest=STRUCTURAL_REPORTER_MANIFEST,
    candidate=STRUCTURAL_REPORTER_CANDIDATE_REGISTRATION,
)


def _metatheory_components(
    token: str,
    manifest: CapabilityManifest,
) -> tuple[
    ExtensionComponentRegistration,
    ExtensionComponentRegistration,
    ExtensionComponentRegistration,
    ExtensionComponentRegistration,
]:
    candidate = next(
        value
        for value in EXECUTABLE_METATHEORY_CONFORMANCE_CANDIDATE_REGISTRATIONS
        if value.manifest == manifest
    )
    return _campaign_components(token=token, manifest=manifest, candidate=candidate)


(
    EXECUTABLE_METATHEORY_ACQUISITION_METHOD,
    EXECUTABLE_METATHEORY_ACQUISITION_CONFIG_DECODER,
    EXECUTABLE_METATHEORY_ACQUISITION_ARTIFACT_VALIDATOR,
    EXECUTABLE_METATHEORY_ACQUISITION_RUNTIME_PROVIDER,
) = _metatheory_components(
    "executable-metatheory-acquisition",
    EXECUTABLE_METATHEORY_ACQUISITION_MANIFEST,
)
(
    EXECUTABLE_METATHEORY_DEVELOPMENT_METHOD,
    EXECUTABLE_METATHEORY_DEVELOPMENT_CONFIG_DECODER,
    EXECUTABLE_METATHEORY_DEVELOPMENT_ARTIFACT_VALIDATOR,
    EXECUTABLE_METATHEORY_DEVELOPMENT_RUNTIME_PROVIDER,
) = _metatheory_components(
    "executable-metatheory-development",
    EXECUTABLE_METATHEORY_DEVELOPMENT_MANIFEST,
)
(
    EXECUTABLE_METATHEORY_EVALUATOR_METHOD,
    EXECUTABLE_METATHEORY_EVALUATOR_CONFIG_DECODER,
    EXECUTABLE_METATHEORY_EVALUATOR_ARTIFACT_VALIDATOR,
    EXECUTABLE_METATHEORY_EVALUATOR_RUNTIME_PROVIDER,
) = _metatheory_components(
    "executable-metatheory-evaluator",
    EXECUTABLE_METATHEORY_EVALUATOR_MANIFEST,
)
(
    EXECUTABLE_METATHEORY_PREDICTION_METHOD,
    EXECUTABLE_METATHEORY_PREDICTION_CONFIG_DECODER,
    EXECUTABLE_METATHEORY_PREDICTION_ARTIFACT_VALIDATOR,
    EXECUTABLE_METATHEORY_PREDICTION_RUNTIME_PROVIDER,
) = _metatheory_components(
    "executable-metatheory-prediction",
    EXECUTABLE_METATHEORY_PREDICTION_MANIFEST,
)
(
    EXECUTABLE_METATHEORY_QUALIFICATION_METHOD,
    EXECUTABLE_METATHEORY_QUALIFICATION_CONFIG_DECODER,
    EXECUTABLE_METATHEORY_QUALIFICATION_ARTIFACT_VALIDATOR,
    EXECUTABLE_METATHEORY_QUALIFICATION_RUNTIME_PROVIDER,
) = _metatheory_components(
    "executable-metatheory-qualification",
    EXECUTABLE_METATHEORY_QUALIFICATION_MANIFEST,
)
(
    EXECUTABLE_METATHEORY_REPORTER_METHOD,
    EXECUTABLE_METATHEORY_REPORTER_CONFIG_DECODER,
    EXECUTABLE_METATHEORY_REPORTER_ARTIFACT_VALIDATOR,
    EXECUTABLE_METATHEORY_REPORTER_RUNTIME_PROVIDER,
) = _metatheory_components(
    "executable-metatheory-reporter",
    EXECUTABLE_METATHEORY_REPORTER_MANIFEST,
)
EXECUTABLE_METATHEORY_CONFORMANCE_COMPONENTS = tuple(
    value
    for group in (
        (
            EXECUTABLE_METATHEORY_ACQUISITION_METHOD,
            EXECUTABLE_METATHEORY_ACQUISITION_CONFIG_DECODER,
            EXECUTABLE_METATHEORY_ACQUISITION_ARTIFACT_VALIDATOR,
            EXECUTABLE_METATHEORY_ACQUISITION_RUNTIME_PROVIDER,
        ),
        (
            EXECUTABLE_METATHEORY_DEVELOPMENT_METHOD,
            EXECUTABLE_METATHEORY_DEVELOPMENT_CONFIG_DECODER,
            EXECUTABLE_METATHEORY_DEVELOPMENT_ARTIFACT_VALIDATOR,
            EXECUTABLE_METATHEORY_DEVELOPMENT_RUNTIME_PROVIDER,
        ),
        (
            EXECUTABLE_METATHEORY_EVALUATOR_METHOD,
            EXECUTABLE_METATHEORY_EVALUATOR_CONFIG_DECODER,
            EXECUTABLE_METATHEORY_EVALUATOR_ARTIFACT_VALIDATOR,
            EXECUTABLE_METATHEORY_EVALUATOR_RUNTIME_PROVIDER,
        ),
        (
            EXECUTABLE_METATHEORY_PREDICTION_METHOD,
            EXECUTABLE_METATHEORY_PREDICTION_CONFIG_DECODER,
            EXECUTABLE_METATHEORY_PREDICTION_ARTIFACT_VALIDATOR,
            EXECUTABLE_METATHEORY_PREDICTION_RUNTIME_PROVIDER,
        ),
        (
            EXECUTABLE_METATHEORY_QUALIFICATION_METHOD,
            EXECUTABLE_METATHEORY_QUALIFICATION_CONFIG_DECODER,
            EXECUTABLE_METATHEORY_QUALIFICATION_ARTIFACT_VALIDATOR,
            EXECUTABLE_METATHEORY_QUALIFICATION_RUNTIME_PROVIDER,
        ),
        (
            EXECUTABLE_METATHEORY_REPORTER_METHOD,
            EXECUTABLE_METATHEORY_REPORTER_CONFIG_DECODER,
            EXECUTABLE_METATHEORY_REPORTER_ARTIFACT_VALIDATOR,
            EXECUTABLE_METATHEORY_REPORTER_RUNTIME_PROVIDER,
        ),
    )
    for value in group
)

SOURCE_FREE_PROPERTY_TRANSPORT_MANIFESTS = source_free_property_transport_capability_registry().capabilities
SOURCE_FREE_PROPERTY_TRANSPORT_CANDIDATE_REGISTRATIONS = tuple(
    CandidateCapabilityRegistration(
        manifest=manifest,
        provider_key=f"structural-transport.{manifest.capability_key}-provider",
        provider_version="1.0.0",
        config_media_type=STRUCTURAL_TRANSPORT_MEDIA_TYPE,
        maximum_config_bytes=_MAXIMUM_CONFIG_BYTES,
    )
    for manifest in SOURCE_FREE_PROPERTY_TRANSPORT_MANIFESTS
)
_SOURCE_FREE_PROPERTY_TRANSPORT_CANDIDATE_BY_ID = {
    value.manifest.registry_id: value for value in SOURCE_FREE_PROPERTY_TRANSPORT_CANDIDATE_REGISTRATIONS
}
SOURCE_FREE_PROPERTY_TRANSPORT_EXECUTABLE_COMPONENTS = tuple(
    (
        role,
        manifest,
        SOURCE_FREE_PROPERTY_TRANSPORT_CONFIG_TYPE_BY_ROLE[role],
        *_campaign_components(
            token=f"source-free-property-transport-{role.value.lower().replace('_', '-')}",
            manifest=manifest,
            candidate=_SOURCE_FREE_PROPERTY_TRANSPORT_CANDIDATE_BY_ID[manifest.registry_id],
        ),
    )
    for role in MetatheoryCampaignStageRole
    if role in SOURCE_FREE_PROPERTY_TRANSPORT_CONFIG_TYPE_BY_ROLE
    for manifest in (
        next(
            value
            for value in SOURCE_FREE_PROPERTY_TRANSPORT_MANIFESTS
            if value.config_schema == SOURCE_FREE_PROPERTY_TRANSPORT_CONFIG_TYPE_BY_ROLE[role].SCHEMA
        ),
    )
)
SOURCE_FREE_PROPERTY_TRANSPORT_COMPONENTS = tuple(
    component
    for _, _, _, method, decoder, validator, provider in SOURCE_FREE_PROPERTY_TRANSPORT_EXECUTABLE_COMPONENTS
    for component in (method, decoder, validator, provider)
)

EXTENSION_BUNDLE_CONTRIBUTION = ExtensionBundleContribution(
    contribution_id="extension-contribution.structural-transport",
    contribution_version="1.0.0",
    kind=ExtensionContributionKind.METHOD,
    evidence_profile_registries=(),
    capability_manifests=tuple(
        sorted(
            (
                INTERVAL_PROPERTY_COMPARISON_MANIFEST,
                TRANSFORMED_PROPERTY_COMPARISON_MANIFEST,
                *EXECUTABLE_METATHEORY_CONFORMANCE_MANIFESTS,
                *SOURCE_FREE_PROPERTY_TRANSPORT_MANIFESTS,
                PROSPECTIVE_STRUCTURAL_RECURRENCE_MANIFEST,
                STRUCTURAL_DEVELOPMENT_BRIDGE_MANIFEST,
                STRUCTURAL_RECURRENCE_TRANSPORT_MANIFEST,
                STRUCTURAL_REPORTER_MANIFEST,
                STRUCTURAL_TARGET_BRIDGE_MANIFEST,
            ),
            key=lambda value: value.registry_id,
        )
    ),
    candidate_capability_registrations=tuple(
        sorted(
            (
                INTERVAL_PROPERTY_COMPARISON_CANDIDATE_REGISTRATION,
                TRANSFORMED_PROPERTY_COMPARISON_CANDIDATE_REGISTRATION,
                *EXECUTABLE_METATHEORY_CONFORMANCE_CANDIDATE_REGISTRATIONS,
                *SOURCE_FREE_PROPERTY_TRANSPORT_CANDIDATE_REGISTRATIONS,
                PROSPECTIVE_STRUCTURAL_RECURRENCE_CANDIDATE_REGISTRATION,
                STRUCTURAL_DEVELOPMENT_BRIDGE_CANDIDATE_REGISTRATION,
                STRUCTURAL_RECURRENCE_TRANSPORT_CANDIDATE_REGISTRATION,
                STRUCTURAL_REPORTER_CANDIDATE_REGISTRATION,
                STRUCTURAL_TARGET_BRIDGE_CANDIDATE_REGISTRATION,
            ),
            key=lambda value: value.registration_id,
        )
    ),
    dataset_capability_registries=(),
    profile_registrations=tuple(
        sorted(
            (
                EXECUTABLE_METATHEORY_PROFILE_REGISTRATION,
                STRUCTURAL_FACE_PROFILE_REGISTRATION,
            ),
            key=lambda value: value.registration_id,
        )
    ),
    method_registrations=tuple(
        sorted(
            (
                ObjectIdentity.from_record(
                    INTERVAL_PROPERTY_COMPARISON_METHOD.registration_id,
                    INTERVAL_PROPERTY_COMPARISON_METHOD,
                ),
                ObjectIdentity.from_record(
                    TRANSFORMED_PROPERTY_COMPARISON_METHOD.registration_id,
                    TRANSFORMED_PROPERTY_COMPARISON_METHOD,
                ),
                *(
                    ObjectIdentity.from_record(value.registration_id, value)
                    for value in EXECUTABLE_METATHEORY_CONFORMANCE_COMPONENTS
                    if value.kind is ExtensionComponentKind.METHOD
                ),
                *(
                    ObjectIdentity.from_record(value.registration_id, value)
                    for value in SOURCE_FREE_PROPERTY_TRANSPORT_COMPONENTS
                    if value.kind is ExtensionComponentKind.METHOD
                ),
                ObjectIdentity.from_record(
                    PROSPECTIVE_STRUCTURAL_RECURRENCE_METHOD.registration_id,
                    PROSPECTIVE_STRUCTURAL_RECURRENCE_METHOD,
                ),
                *(
                    ObjectIdentity.from_record(value.registration_id, value)
                    for value in (
                        STRUCTURAL_DEVELOPMENT_BRIDGE_METHOD,
                        STRUCTURAL_RECURRENCE_TRANSPORT_METHOD,
                        STRUCTURAL_REPORTER_METHOD,
                        STRUCTURAL_TARGET_BRIDGE_METHOD,
                    )
                ),
            ),
            key=lambda value: value.object_id,
        )
    ),
    config_decoders=tuple(
        ObjectIdentity.from_record(value.registration_id, value)
        for value in sorted(
            (
                INTERVAL_PROPERTY_COMPARISON_CONFIG_DECODER,
                TRANSFORMED_PROPERTY_COMPARISON_CONFIG_DECODER,
                *(
                    value
                    for value in EXECUTABLE_METATHEORY_CONFORMANCE_COMPONENTS
                    if value.kind is ExtensionComponentKind.CONFIG_DECODER
                ),
                *(
                    value
                    for value in SOURCE_FREE_PROPERTY_TRANSPORT_COMPONENTS
                    if value.kind is ExtensionComponentKind.CONFIG_DECODER
                ),
                PROSPECTIVE_STRUCTURAL_RECURRENCE_CONFIG_DECODER,
                STRUCTURAL_DEVELOPMENT_BRIDGE_CONFIG_DECODER,
                STRUCTURAL_RECURRENCE_TRANSPORT_CONFIG_DECODER,
                STRUCTURAL_REPORTER_CONFIG_DECODER,
                STRUCTURAL_TARGET_BRIDGE_CONFIG_DECODER,
            ),
            key=lambda value: value.registration_id,
        )
    ),
    artifact_validators=tuple(
        ObjectIdentity.from_record(value.registration_id, value)
        for value in sorted(
            (
                INTERVAL_PROPERTY_COMPARISON_ARTIFACT_VALIDATOR,
                TRANSFORMED_PROPERTY_COMPARISON_ARTIFACT_VALIDATOR,
                *(
                    value
                    for value in EXECUTABLE_METATHEORY_CONFORMANCE_COMPONENTS
                    if value.kind is ExtensionComponentKind.ARTIFACT_VALIDATOR
                ),
                *(
                    value
                    for value in SOURCE_FREE_PROPERTY_TRANSPORT_COMPONENTS
                    if value.kind is ExtensionComponentKind.ARTIFACT_VALIDATOR
                ),
                PROSPECTIVE_STRUCTURAL_RECURRENCE_ARTIFACT_VALIDATOR,
                STRUCTURAL_DEVELOPMENT_BRIDGE_ARTIFACT_VALIDATOR,
                STRUCTURAL_RECURRENCE_TRANSPORT_ARTIFACT_VALIDATOR,
                STRUCTURAL_REPORTER_ARTIFACT_VALIDATOR,
                STRUCTURAL_TARGET_BRIDGE_ARTIFACT_VALIDATOR,
            ),
            key=lambda value: value.registration_id,
        )
    ),
    runtime_providers=tuple(
        ObjectIdentity.from_record(value.registration_id, value)
        for value in sorted(
            (
                INTERVAL_PROPERTY_COMPARISON_RUNTIME_PROVIDER,
                TRANSFORMED_PROPERTY_COMPARISON_RUNTIME_PROVIDER,
                *(
                    value
                    for value in EXECUTABLE_METATHEORY_CONFORMANCE_COMPONENTS
                    if value.kind is ExtensionComponentKind.RUNTIME_PROVIDER
                ),
                *(
                    value
                    for value in SOURCE_FREE_PROPERTY_TRANSPORT_COMPONENTS
                    if value.kind is ExtensionComponentKind.RUNTIME_PROVIDER
                ),
                PROSPECTIVE_STRUCTURAL_RECURRENCE_RUNTIME_PROVIDER,
                STRUCTURAL_DEVELOPMENT_BRIDGE_RUNTIME_PROVIDER,
                STRUCTURAL_RECURRENCE_TRANSPORT_RUNTIME_PROVIDER,
                STRUCTURAL_REPORTER_RUNTIME_PROVIDER,
                STRUCTURAL_TARGET_BRIDGE_RUNTIME_PROVIDER,
            ),
            key=lambda value: value.registration_id,
        )
    ),
    study_authors=(),
    grants_authority=False,
    embeds_scientific_payload=False,
)


__all__ = [
    "EXTENSION_BUNDLE_CONTRIBUTION",
    "SOURCE_FREE_PROPERTY_TRANSPORT_CANDIDATE_REGISTRATIONS",
    "SOURCE_FREE_PROPERTY_TRANSPORT_COMPONENTS",
    "SOURCE_FREE_PROPERTY_TRANSPORT_EXECUTABLE_COMPONENTS",
    "SOURCE_FREE_PROPERTY_TRANSPORT_MANIFESTS",
]
