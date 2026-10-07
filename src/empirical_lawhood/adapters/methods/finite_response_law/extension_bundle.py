"Paired-native projection and completion discovery; no law/control overlay."

from dataclasses import replace

from empirical_lawhood.adapters.simulators.finite_response_law.native_artifact import NATIVE_PAIR_SCHEMA
from empirical_lawhood.adapters.simulators.finite_response_law.source_outputs import FiniteResponseLawNativeTaskResult
from empirical_lawhood.adapters.composition.discovery import bundle, manifest
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.artifacts import CanonicalTaskReceipt
from empirical_lawhood.runtime.capabilities import CapabilityKind, CapabilityPermission
from empirical_lawhood.runtime.extension_bundles import ExtensionContributionKind
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope

from .assigned_native_records import FiniteResponseLawAssignedCalibrationNativeEvaluation
from .calibration_operands import PREDICTION_SCHEMA
from .method_records import COEFFICIENT_SCHEMA, MANIFEST_SCHEMA, READOUT_SCHEMA, REPORT_SCHEMA, FiniteResponseLawAssignedCalibrationMethodConfig, FiniteResponseLawAssignedCalibrationReport, FiniteResponseLawAssignedQualificationReport
from .native_records import FiniteResponseLawNativeEvaluationConfig, FiniteResponseLawNativeEvaluation, FiniteResponseLawNativeViewObservation, FiniteResponseLawProjectionConfig
from .science import PROGRAMME

PROJECTION_CAPABILITY = manifest(
    "projection",
    FiniteResponseLawProjectionConfig,
    CapabilityKind.TRANSFORM,
    (
        FiniteResponseLawProjectionConfig.SCHEMA,
        FiniteResponseLawNativeTaskResult.SCHEMA,
        NATIVE_PAIR_SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
    ),
    (FiniteResponseLawNativeViewObservation.SCHEMA, LinkedCampaignStageEnvelope.SCHEMA),
    EvidenceCeiling.MEASUREMENT,
    OutcomeAccess.EVALUATION_SEALED,
    120,
    namespace=PROGRAMME,
    resources=ResourceBudget(1, 1024**3, 0, 120, 1024**3, 1024**2),
    implementation_id="exact-native-192-paired-hold-projection",
    conformance_ids=(
        "exact-flh-projection-census",
        "paired-hold-lineage",
        "preserved-native-response-reducer",
    ),
    read_permission=(CapabilityPermission.READ_OUTCOME_VISIBLE if 'projection' == "evaluation" else CapabilityPermission.READ_SEALED_OUTCOMES),
)
EVALUATION_CAPABILITY = manifest(
    "evaluation",
    FiniteResponseLawNativeEvaluationConfig,
    CapabilityKind.EVALUATOR,
    (
        FiniteResponseLawNativeEvaluationConfig.SCHEMA,
        FiniteResponseLawNativeViewObservation.SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
    ),
    (
        FiniteResponseLawNativeEvaluation.SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
        ScientificAdjudicationRecord.SCHEMA,
    ),
    EvidenceCeiling.MEASUREMENT,
    OutcomeAccess.EVALUATION_REVEALED,
    120,
    namespace=PROGRAMME,
    resources=ResourceBudget(1, 1024**3, 0, 120, 64 * 1024**2, 8 * 1024**2),
    implementation_id="native-census-measurement-completion-only",
    conformance_ids=(
        "complete-native-census",
        "no-oracle-or-lawhood-promotion",
        "separate-source-adjudication",
    ),
    read_permission=(CapabilityPermission.READ_OUTCOME_VISIBLE if 'evaluation' == "evaluation" else CapabilityPermission.READ_SEALED_OUTCOMES),
)
_projection, PROJECTION_COMPONENTS = bundle(
    "projection",
    ExtensionContributionKind.METHOD,
    PROJECTION_CAPABILITY,
    (FiniteResponseLawProjectionConfig,),
    None,
    namespace=PROGRAMME,
)
_evaluation, EVALUATION_COMPONENTS = bundle(
    "evaluation",
    ExtensionContributionKind.METHOD,
    EVALUATION_CAPABILITY,
    (FiniteResponseLawNativeEvaluationConfig,),
    None,
    namespace=PROGRAMME,
)
METHOD_CAPABILITY = manifest(
    "calibration-qualification",
    FiniteResponseLawAssignedCalibrationMethodConfig,
    CapabilityKind.EVALUATOR,
    (
        FiniteResponseLawAssignedCalibrationMethodConfig.SCHEMA,
        FiniteResponseLawAssignedQualificationReport.SCHEMA,
        ObjectIdentity.SCHEMA,
        FiniteResponseLawAssignedCalibrationNativeEvaluation.SCHEMA,
        CanonicalTaskReceipt.SCHEMA,
        REPORT_SCHEMA,
        COEFFICIENT_SCHEMA,
        MANIFEST_SCHEMA,
        FiniteResponseLawAssignedCalibrationReport.SCHEMA,
        PREDICTION_SCHEMA,
        READOUT_SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
    ),
    (
        FiniteResponseLawAssignedCalibrationReport.SCHEMA,
        FiniteResponseLawAssignedQualificationReport.SCHEMA,
        ObjectIdentity.SCHEMA,
        PREDICTION_SCHEMA,
        READOUT_SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
        ScientificAdjudicationRecord.SCHEMA,
    ),
    EvidenceCeiling.LOCAL_LAW,
    OutcomeAccess.EVALUATOR_REVEAL,
    120,
    namespace=PROGRAMME,
    resources=ResourceBudget(1, 1024**3, 0, 120, 32 * 1024**2, 8 * 1024**2),
    read_permission=CapabilityPermission.READ_SEALED_OUTCOMES,
    implementation_id="frozen-32-root-rank-30-four-singleton-joint-qualification-separate-adjudication",
    conformance_ids=(
        "committed-calibration-custody",
        "finite-nonfinite-unevaluable-truth-table",
        "sole-law-qualification-owner",
    ),
)
METHOD_CAPABILITY = replace(
    METHOD_CAPABILITY,
    permissions=tuple(
        sorted((*METHOD_CAPABILITY.permissions, CapabilityPermission.REVEAL_OUTCOMES))
    ),
)
_method, METHOD_COMPONENTS = bundle(
    "calibration-qualification",
    ExtensionContributionKind.METHOD,
    METHOD_CAPABILITY,
    (FiniteResponseLawAssignedCalibrationMethodConfig,),
    None,
    namespace=PROGRAMME,
)
EXTENSION_BUNDLE_CONTRIBUTION = replace(
    _projection,
    contribution_id=f"extension-contribution.{PROGRAMME}.methods",
    capability_manifests=tuple(
        sorted(
            (
                *_projection.capability_manifests,
                *_evaluation.capability_manifests,
                *_method.capability_manifests,
            ),
            key=lambda v: v.registry_id,
        )
    ),
    candidate_capability_registrations=tuple(
        sorted(
            (
                *_projection.candidate_capability_registrations,
                *_evaluation.candidate_capability_registrations,
                *_method.candidate_capability_registrations,
            ),
            key=lambda v: v.manifest.registry_id,
        )
    ),
    profile_registrations=tuple(
        sorted(
            (
                *_projection.profile_registrations,
                *_evaluation.profile_registrations,
                *_method.profile_registrations,
            ),
            key=lambda v: v.registration_id,
        )
    ),
    config_decoders=tuple(
        sorted(
            (
                *_projection.config_decoders,
                *_evaluation.config_decoders,
                *_method.config_decoders,
            ),
            key=lambda v: v.object_id,
        )
    ),
    runtime_providers=tuple(
        sorted(
            (
                *_projection.runtime_providers,
                *_evaluation.runtime_providers,
                *_method.runtime_providers,
            ),
            key=lambda v: v.object_id,
        )
    ),
)
