"Static source qualification scalar projection and finite charter evaluation discovery."

from dataclasses import replace

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.capabilities import CapabilityKind, CapabilityPermission
from empirical_lawhood.runtime.extension_bundles import ExtensionContributionKind
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from empirical_lawhood.adapters.composition.discovery import bundle, manifest
from empirical_lawhood.adapters.simulators.prepared_response.native_pair import NATIVE_PAIR_SCHEMA
from empirical_lawhood.adapters.simulators.prepared_response.source_outputs import PreparedNativeTaskResult
from .qualification_records import PreparedResponseSourceQualificationProjectionConfig, PreparedResponseSourceQualificationViewObservation
from .qualification import PreparedResponseSourceQualificationEvaluationConfig, PreparedResponseSourceQualificationEvaluation
from .development_projection import PreparedResponseDevelopmentProjectionConfig, PreparedResponseDevelopmentViewProjection
from .development_fit import PreparedResponseDevelopmentFitConfig, PreparedResponseDevelopmentModelFit
from .development_selection import PreparedResponseDevelopmentNominalLibrary, PreparedResponseDevelopmentSelectionConfig
from .development_benchmarks import PreparedResponseDevelopmentConstantGainReport
from .policy_decision import PreparedParentDecision, PreparedPolicyDecisionConfig
from .calibration_records import PreparedResponseCalibrationCalibratedLibrary, PreparedResponseCalibrationCalibrationConfig, PreparedResponseCalibrationProjectionConfig, PreparedResponseCalibrationViewProjection
from empirical_lawhood.adapters.simulators.prepared_response.policy_native import PreparedResponseCalibrationNativeTaskResult


SOURCE_QUALIFICATION_NAMESPACE = "prepared-response.source-qualification"
DEVELOPMENT_NAMESPACE = "prepared-response.dependent-refinement"
SOURCE_QUALIFICATION_PROJECTION_CAPABILITY = manifest(
    "projection",
    PreparedResponseSourceQualificationProjectionConfig,
    CapabilityKind.TRANSFORM,
    (
        PreparedResponseSourceQualificationProjectionConfig.SCHEMA,
        PreparedNativeTaskResult.SCHEMA,
        NATIVE_PAIR_SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
    ),
    (PreparedResponseSourceQualificationViewObservation.SCHEMA, LinkedCampaignStageEnvelope.SCHEMA),
    EvidenceCeiling.RESPONSE,
    OutcomeAccess.EVALUATION_SEALED,
    180,
    namespace=SOURCE_QUALIFICATION_NAMESPACE,
    resources=ResourceBudget(1, 1024**3, 0, 180, 2 * 1024**3, 1024**2),
    implementation_id="source-qualification-absolute-two-port-native-projection",
    conformance_ids=(
        "complete-assigned-root-census",
        "matched-innovation-preservation",
        "no-source-contact-in-projection",
    ),
    read_permission=(CapabilityPermission.READ_OUTCOME_VISIBLE if 'projection' == "evaluation" else CapabilityPermission.READ_SEALED_OUTCOMES),
)
SOURCE_QUALIFICATION_EVALUATION_CAPABILITY = manifest(
    "evaluation",
    PreparedResponseSourceQualificationEvaluationConfig,
    CapabilityKind.EVALUATOR,
    (
        PreparedResponseSourceQualificationEvaluationConfig.SCHEMA,
        PreparedResponseSourceQualificationViewObservation.SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
    ),
    (
        PreparedResponseSourceQualificationEvaluation.SCHEMA,
        ScientificAdjudicationRecord.SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
    ),
    EvidenceCeiling.RESPONSE,
    OutcomeAccess.EVALUATION_REVEALED,
    180,
    namespace=SOURCE_QUALIFICATION_NAMESPACE,
    resources=ResourceBudget(1, 1024**3, 0, 180, 128 * 1024**2, 2 * 1024**2),
    implementation_id="source-qualification-forty-eight-charter-both-context-nomination",
    conformance_ids=(
        "all-assigned-source-qualification-roots-retained",
        "both-context-charter-intersection",
        "fixed-parent-command-comparators",
        "valid-negative-and-unevaluable-q-stops",
    ),
    read_permission=(CapabilityPermission.READ_OUTCOME_VISIBLE if 'evaluation' == "evaluation" else CapabilityPermission.READ_SEALED_OUTCOMES),
)
DEVELOPMENT_PROJECTION_CAPABILITY = manifest(
    "projection",
    PreparedResponseDevelopmentProjectionConfig,
    CapabilityKind.TRANSFORM,
    (
        PreparedResponseDevelopmentProjectionConfig.SCHEMA,
        PreparedNativeTaskResult.SCHEMA,
        NATIVE_PAIR_SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
    ),
    (PreparedResponseDevelopmentViewProjection.SCHEMA, LinkedCampaignStageEnvelope.SCHEMA),
    EvidenceCeiling.RESPONSE,
    OutcomeAccess.DEVELOPMENT_VISIBLE,
    300,
    namespace=DEVELOPMENT_NAMESPACE,
    resources=ResourceBudget(1, 3 * 1024**3, 0, 300, 2 * 1024**3, 2 * 1024**2),
    read_permission=CapabilityPermission.READ_OUTCOME_VISIBLE,
    implementation_id="dependent-refinement-absolute-two-port-per-view-projection",
    conformance_ids=(
        "complete-assigned-development-root-census",
        "no-source-contact-in-projection",
        "paired-acquisition-separate-view-products",
    ),
)
DEVELOPMENT_FIT_CAPABILITY = manifest(
    "fit",
    PreparedResponseDevelopmentFitConfig,
    CapabilityKind.ANALYSIS,
    (
        PreparedResponseDevelopmentFitConfig.SCHEMA,
        PreparedResponseDevelopmentViewProjection.SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
    ),
    (PreparedResponseDevelopmentModelFit.SCHEMA, LinkedCampaignStageEnvelope.SCHEMA),
    EvidenceCeiling.LOCAL_LAW,
    OutcomeAccess.DEVELOPMENT_VISIBLE,
    1800,
    namespace=DEVELOPMENT_NAMESPACE,
    resources=ResourceBudget(1, 4 * 1024**3, 0, 1800, 512 * 1024**2, 18 * 1024**2),
    read_permission=CapabilityPermission.READ_OUTCOME_VISIBLE,
    implementation_id="dependent-refinement-four-family-whole-root-nested-bilinear-fit",
    conformance_ids=(
        "all-assigned-development-roots-retained",
        "authored-whole-root-nested-crossfit",
        "nominal-products-do-not-qualify-a-law",
    ),
)
DEVELOPMENT_SELECTION_CAPABILITY = manifest(
    "selection",
    PreparedResponseDevelopmentSelectionConfig,
    CapabilityKind.HYPOTHESIS_ADJUDICATOR,
    (
        PreparedResponseDevelopmentSelectionConfig.SCHEMA,
        PreparedResponseDevelopmentViewProjection.SCHEMA,
        PreparedResponseDevelopmentModelFit.SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
    ),
    (
        PreparedResponseDevelopmentNominalLibrary.SCHEMA,
        PreparedResponseDevelopmentConstantGainReport.SCHEMA,
        ScientificAdjudicationRecord.SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
    ),
    EvidenceCeiling.RESPONSE,
    OutcomeAccess.EVALUATION_REVEALED,
    1800,
    namespace=DEVELOPMENT_NAMESPACE,
    resources=ResourceBudget(1, 4 * 1024**3, 0, 1800, 1024**3, 32 * 1024**2),
    read_permission=CapabilityPermission.READ_OUTCOME_VISIBLE,
    implementation_id="dependent-refinement-two-context-selection-joint-handoff-scenarios-constant-gain",
    conformance_ids=(
        "all-failed-candidates-retained",
        "both-context-eligibility-intersection",
        "constant-gain-reuses-held-baselines-without-calibration-promotion",
        "nominal-library-grants-no-law-authority",
        "training-only-baseline-and-intervals",
    ),
)
CALIBRATION_POLICY_CAPABILITY = manifest(
    "policy",
    PreparedPolicyDecisionConfig,
    CapabilityKind.PROSPECTIVE_NOMINATOR,
    (
        PreparedPolicyDecisionConfig.SCHEMA,
        PreparedResponseCalibrationNativeTaskResult.SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
    ),
    (PreparedParentDecision.SCHEMA, LinkedCampaignStageEnvelope.SCHEMA),
    EvidenceCeiling.RESPONSE,
    OutcomeAccess.EVALUATION_SEALED,
    120,
    namespace="prepared-response.fresh-response-calibration",
    resources=ResourceBudget(1, 1024**3, 0, 120, 64 * 1024**2, 1024**2),
    implementation_id="development-frozen-target-blind-adequacy-constrained-parent",
    conformance_ids=(
        "four-development-frozen-policy-roles",
        "no-target-or-future-input",
        "preparent-primary-view-cutoff",
    ),
    read_permission=(CapabilityPermission.READ_OUTCOME_VISIBLE if 'policy' == "evaluation" else CapabilityPermission.READ_SEALED_OUTCOMES),
)
CALIBRATION_PROJECTION_CAPABILITY = manifest(
    "projection",
    PreparedResponseCalibrationProjectionConfig,
    CapabilityKind.TRANSFORM,
    (
        PreparedResponseCalibrationProjectionConfig.SCHEMA,
        PreparedResponseCalibrationNativeTaskResult.SCHEMA,
        PreparedParentDecision.SCHEMA,
        NATIVE_PAIR_SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
    ),
    (PreparedResponseCalibrationViewProjection.SCHEMA, LinkedCampaignStageEnvelope.SCHEMA),
    EvidenceCeiling.RESPONSE,
    OutcomeAccess.EVALUATION_SEALED,
    300,
    namespace="prepared-response.fresh-response-calibration",
    resources=ResourceBudget(1, 3 * 1024**3, 0, 300, 256 * 1024**2, 2 * 1024**2),
    implementation_id="fresh-calibration-authenticated-policy-chart-parent-work-projection",
    conformance_ids=(
        "complete-policy-word-view-census",
        "exact-invocation-predecessor-common-start-parent-joins",
        "no-native-effect-in-projection",
        "same-innovation-hold-preservation",
    ),
    read_permission=(CapabilityPermission.READ_OUTCOME_VISIBLE if 'projection' == "evaluation" else CapabilityPermission.READ_SEALED_OUTCOMES),
)
CALIBRATION_CALIBRATION_CAPABILITY = manifest(
    "calibration",
    PreparedResponseCalibrationCalibrationConfig,
    CapabilityKind.EVALUATOR,
    (
        PreparedResponseCalibrationCalibrationConfig.SCHEMA,
        PreparedResponseCalibrationViewProjection.SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
    ),
    (
        PreparedResponseCalibrationCalibratedLibrary.SCHEMA,
        ScientificAdjudicationRecord.SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
    ),
    EvidenceCeiling.RESPONSE,
    OutcomeAccess.EVALUATION_REVEALED,
    1800,
    namespace="prepared-response.fresh-response-calibration",
    resources=ResourceBudget(1, 4 * 1024**3, 0, 1800, 8 * 1024**3, 32 * 1024**2),
    read_permission=CapabilityPermission.READ_OUTCOME_VISIBLE,
    implementation_id="fresh-calibration-config-bound-root-maximum-split-conformal",
    conformance_ids=(
        "fixed-order-statistic-93",
        "invalid-coordinate-infinite-score",
        "ninety-six-root-context-census",
    ),
)
_projection, SOURCE_QUALIFICATION_PROJECTION_COMPONENTS = bundle(
    "projection",
    ExtensionContributionKind.METHOD,
    SOURCE_QUALIFICATION_PROJECTION_CAPABILITY,
    (PreparedResponseSourceQualificationProjectionConfig,),
    None,
    namespace=SOURCE_QUALIFICATION_NAMESPACE,
)
_evaluation, SOURCE_QUALIFICATION_EVALUATION_COMPONENTS = bundle(
    "evaluation",
    ExtensionContributionKind.METHOD,
    SOURCE_QUALIFICATION_EVALUATION_CAPABILITY,
    (PreparedResponseSourceQualificationEvaluationConfig,),
    None,
    namespace=SOURCE_QUALIFICATION_NAMESPACE,
)
_development_projection, DEVELOPMENT_PROJECTION_COMPONENTS = bundle(
    "projection",
    ExtensionContributionKind.METHOD,
    DEVELOPMENT_PROJECTION_CAPABILITY,
    (PreparedResponseDevelopmentProjectionConfig,),
    None,
    namespace=DEVELOPMENT_NAMESPACE,
)
_development_fit, DEVELOPMENT_FIT_COMPONENTS = bundle(
    "fit",
    ExtensionContributionKind.METHOD,
    DEVELOPMENT_FIT_CAPABILITY,
    (PreparedResponseDevelopmentFitConfig,),
    None,
    namespace=DEVELOPMENT_NAMESPACE,
)
_development_selection, DEVELOPMENT_SELECTION_COMPONENTS = bundle(
    "selection",
    ExtensionContributionKind.METHOD,
    DEVELOPMENT_SELECTION_CAPABILITY,
    (PreparedResponseDevelopmentSelectionConfig,),
    None,
    namespace=DEVELOPMENT_NAMESPACE,
)
_calibration_policy, CALIBRATION_POLICY_COMPONENTS = bundle(
    "policy",
    ExtensionContributionKind.METHOD,
    CALIBRATION_POLICY_CAPABILITY,
    (PreparedPolicyDecisionConfig,),
    None,
    namespace="prepared-response.fresh-response-calibration",
    maximum_config_bytes=16 * 1024**2,
)
_calibration_projection, CALIBRATION_PROJECTION_COMPONENTS = bundle(
    "projection",
    ExtensionContributionKind.METHOD,
    CALIBRATION_PROJECTION_CAPABILITY,
    (PreparedResponseCalibrationProjectionConfig,),
    None,
    namespace="prepared-response.fresh-response-calibration",
    maximum_config_bytes=16 * 1024**2,
)
_calibration_evaluation, CALIBRATION_CALIBRATION_COMPONENTS = bundle(
    "calibration",
    ExtensionContributionKind.METHOD,
    CALIBRATION_CALIBRATION_CAPABILITY,
    (PreparedResponseCalibrationCalibrationConfig,),
    None,
    namespace="prepared-response.fresh-response-calibration",
    maximum_config_bytes=16 * 1024**2,
)
EXTENSION_BUNDLE_CONTRIBUTION = replace(
    _projection,
    contribution_id=f"extension-contribution.{SOURCE_QUALIFICATION_NAMESPACE}.methods",
    capability_manifests=tuple(
        sorted(
            (
                *_projection.capability_manifests,
                *_evaluation.capability_manifests,
                *_development_projection.capability_manifests,
                *_development_fit.capability_manifests,
                *_development_selection.capability_manifests,
                *_calibration_policy.capability_manifests,
                *_calibration_projection.capability_manifests,
                *_calibration_evaluation.capability_manifests,
            ),
            key=lambda value: value.registry_id,
        )
    ),
    candidate_capability_registrations=tuple(
        sorted(
            (
                *_projection.candidate_capability_registrations,
                *_evaluation.candidate_capability_registrations,
                *_development_projection.candidate_capability_registrations,
                *_development_fit.candidate_capability_registrations,
                *_development_selection.candidate_capability_registrations,
                *_calibration_policy.candidate_capability_registrations,
                *_calibration_projection.candidate_capability_registrations,
                *_calibration_evaluation.candidate_capability_registrations,
            ),
            key=lambda value: value.manifest.registry_id,
        )
    ),
    profile_registrations=tuple(
        sorted(
            (
                *_projection.profile_registrations,
                *_evaluation.profile_registrations,
                *_development_projection.profile_registrations,
                *_development_fit.profile_registrations,
                *_development_selection.profile_registrations,
                *_calibration_policy.profile_registrations,
                *_calibration_projection.profile_registrations,
                *_calibration_evaluation.profile_registrations,
            ),
            key=lambda value: value.registration_id,
        )
    ),
    config_decoders=tuple(
        sorted(
            (
                *_projection.config_decoders,
                *_evaluation.config_decoders,
                *_development_projection.config_decoders,
                *_development_fit.config_decoders,
                *_development_selection.config_decoders,
                *_calibration_policy.config_decoders,
                *_calibration_projection.config_decoders,
                *_calibration_evaluation.config_decoders,
            ),
            key=lambda value: value.object_id,
        )
    ),
    runtime_providers=tuple(
        sorted(
            (
                *_projection.runtime_providers,
                *_evaluation.runtime_providers,
                *_development_projection.runtime_providers,
                *_development_fit.runtime_providers,
                *_development_selection.runtime_providers,
                *_calibration_policy.runtime_providers,
                *_calibration_projection.runtime_providers,
                *_calibration_evaluation.runtime_providers,
            ),
            key=lambda value: value.object_id,
        )
    ),
)
