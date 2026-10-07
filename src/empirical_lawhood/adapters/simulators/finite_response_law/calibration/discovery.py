"Fresh calibration native measurement metadata, without qualification or source effects."

from dataclasses import replace

from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.source_qualification import PredecessorBoundSourceQualificationExperiment
from empirical_lawhood.runtime.source_qualification import PredecessorBoundSourceQualificationSubstrateBinding
from empirical_lawhood.runtime.extension_bundles import ExtensionContributionKind
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.capabilities import CapabilityPermission
from empirical_lawhood.adapters.composition.discovery import manifest, bundle
from empirical_lawhood.adapters.methods.finite_response_law.extension_bundle import PROJECTION_CAPABILITY as OLD_PROJECTION, EVALUATION_CAPABILITY as OLD_EVALUATION
from empirical_lawhood.adapters.methods.finite_response_law.assigned_native_records import FiniteResponseLawAssignedCalibrationProjectionConfig, FiniteResponseLawAssignedCalibrationEvaluationConfig, FiniteResponseLawAssignedCalibrationViewObservation, FiniteResponseLawAssignedCalibrationNativeEvaluation
from ..discovery import SOURCE_CAPABILITY as OLD_SOURCE
from ..assigned_contracts import FiniteResponseLawAssignedCalibrationConfig
from ..source_outputs import FiniteResponseLawAssignedCalibrationTaskResult
from ..native_artifact import NATIVE_PAIR_SCHEMA

NAMESPACE = "finite-response-law.calibration"
SOURCE_RECORDS: tuple[type[CanonicalRecord], ...] = (
    FiniteResponseLawAssignedCalibrationConfig,
    PredecessorBoundSourceQualificationExperiment,
    PredecessorBoundSourceQualificationSubstrateBinding,
)
SOURCE_CAPABILITY = manifest(
    "source",
    FiniteResponseLawAssignedCalibrationConfig,
    OLD_SOURCE.kind,
    (
        FiniteResponseLawAssignedCalibrationConfig.SCHEMA,
        FiniteResponseLawAssignedCalibrationTaskResult.SCHEMA,
        NATIVE_PAIR_SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
    ),
    (
        FiniteResponseLawAssignedCalibrationTaskResult.SCHEMA,
        NATIVE_PAIR_SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
    ),
    EvidenceCeiling.MEASUREMENT,
    OutcomeAccess.EVALUATION_SEALED,
    1800,
    namespace=NAMESPACE,
    resources=OLD_SOURCE.resource_ceiling,
    implementation_id="fresh-32-root-single-assigned-parent-native-census",
    conformance_ids=(
        "coupled-views-two-futures",
        "exact-fresh-calibration-census",
        "unchanged-native-marcher",
    ),
    read_permission=(CapabilityPermission.READ_OUTCOME_VISIBLE if 'source' == "evaluation" else CapabilityPermission.READ_SEALED_OUTCOMES),
)
PROJECTION_CAPABILITY = manifest(
    "projection",
    FiniteResponseLawAssignedCalibrationProjectionConfig,
    OLD_PROJECTION.kind,
    (
        FiniteResponseLawAssignedCalibrationProjectionConfig.SCHEMA,
        FiniteResponseLawAssignedCalibrationTaskResult.SCHEMA,
        NATIVE_PAIR_SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
    ),
    (
        FiniteResponseLawAssignedCalibrationViewObservation.SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
    ),
    EvidenceCeiling.MEASUREMENT,
    OutcomeAccess.EVALUATION_SEALED,
    120,
    namespace=NAMESPACE,
    resources=OLD_PROJECTION.resource_ceiling,
    implementation_id="fresh-native-192-paired-hold-same-reducer",
    conformance_ids=(
        "actual-signed-mate-and-hold",
        "complete-future-view-census",
        "unchanged-receiver",
    ),
    read_permission=(CapabilityPermission.READ_OUTCOME_VISIBLE if 'projection' == "evaluation" else CapabilityPermission.READ_SEALED_OUTCOMES),
)
EVALUATION_CAPABILITY = manifest(
    "evaluation",
    FiniteResponseLawAssignedCalibrationEvaluationConfig,
    OLD_EVALUATION.kind,
    (
        FiniteResponseLawAssignedCalibrationEvaluationConfig.SCHEMA,
        FiniteResponseLawAssignedCalibrationViewObservation.SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
    ),
    (
        FiniteResponseLawAssignedCalibrationNativeEvaluation.SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
        ScientificAdjudicationRecord.SCHEMA,
    ),
    EvidenceCeiling.MEASUREMENT,
    OutcomeAccess.EVALUATOR_REVEAL,
    120,
    namespace=NAMESPACE,
    # 32 roots x 2 views, plus the canonical evaluation configuration.
    resources=replace(
        OLD_EVALUATION.resource_ceiling,
        source_scan_bytes=64 * PROJECTION_CAPABILITY.resource_ceiling.output_bytes
        + 1024**2,
    ),
    implementation_id="fresh-full-native-census-completion-not-law-qualification",
    conformance_ids=(
        "full-32-root-denominator",
        "no-qualification-promotion",
        "separate-native-adjudication",
    ),
    read_permission=(CapabilityPermission.READ_OUTCOME_VISIBLE if 'evaluation' == "evaluation" else CapabilityPermission.READ_SEALED_OUTCOMES),
)
# This fresh evaluator owns the protected read. Historical evaluators consume
# already exposed summaries and retain their existing permission contracts.
EVALUATION_CAPABILITY = replace(
    EVALUATION_CAPABILITY,
    permissions=tuple(
        sorted(
            (
                CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
                CapabilityPermission.READ_SEALED_OUTCOMES,
                CapabilityPermission.REVEAL_OUTCOMES,
            )
        )
    ),
)
_source, SOURCE_COMPONENTS = bundle(
    "source",
    ExtensionContributionKind.SOURCE,
    SOURCE_CAPABILITY,
    (FiniteResponseLawAssignedCalibrationConfig,),
    NATIVE_PAIR_SCHEMA,
    namespace=NAMESPACE,
)
_projection, PROJECTION_COMPONENTS = bundle(
    "projection",
    ExtensionContributionKind.METHOD,
    PROJECTION_CAPABILITY,
    (FiniteResponseLawAssignedCalibrationProjectionConfig,),
    None,
    namespace=NAMESPACE,
)
_evaluation, EVALUATION_COMPONENTS = bundle(
    "evaluation",
    ExtensionContributionKind.METHOD,
    EVALUATION_CAPABILITY,
    (FiniteResponseLawAssignedCalibrationEvaluationConfig,),
    None,
    namespace=NAMESPACE,
)
_parts = (_source, _projection, _evaluation)
EXTENSION_BUNDLE_CONTRIBUTION = replace(
    _source,
    contribution_id=f"extension-contribution.{NAMESPACE}",
    capability_manifests=tuple(
        sorted(
            (v for p in _parts for v in p.capability_manifests),
            key=lambda v: v.registry_id,
        )
    ),
    candidate_capability_registrations=tuple(
        sorted(
            (v for p in _parts for v in p.candidate_capability_registrations),
            key=lambda v: v.registration_id,
        )
    ),
    profile_registrations=tuple(
        sorted(
            (v for p in _parts for v in p.profile_registrations),
            key=lambda v: v.registration_id,
        )
    ),
    config_decoders=tuple(
        sorted(
            (v for p in _parts for v in p.config_decoders), key=lambda v: v.object_id
        )
    ),
    runtime_providers=tuple(
        sorted(
            (v for p in _parts for v in p.runtime_providers), key=lambda v: v.object_id
        )
    ),
)
