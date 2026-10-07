"Finite response-law evaluation fresh native, causal-control, projection and sealed-completion metadata."

from dataclasses import replace

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.runtime.capabilities import CapabilityKind, CapabilityPermission
from empirical_lawhood.runtime.extension_bundles import ExtensionContributionKind
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.artifacts import CanonicalTaskReceipt
from empirical_lawhood.planning.source_qualification import QualifiedSourceUseExperiment
from empirical_lawhood.runtime.source_qualification import QualifiedSourceUseSubstrateBinding
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.adapters.methods.finite_response_law.evaluation_results import FiniteResponseLawEvaluationRevealConfig, FiniteResponseLawEvaluationCohort
from empirical_lawhood.adapters.methods.finite_response_law.control_reveal import FiniteResponseLawProspectiveRootEvaluation
from empirical_lawhood.adapters.methods.finite_response_law.evaluation_readout import FiniteResponseLawRootInferenceOperands
from empirical_lawhood.adapters.composition.discovery import manifest, bundle
from empirical_lawhood.adapters.methods.finite_response_law.control_records import FiniteResponseLawControlConfig, FiniteResponseLawRootForecast, FiniteResponseLawRootRequestReveal, FiniteResponseLawRootControlLock, FiniteResponseLawRootParentJoin
from empirical_lawhood.adapters.methods.finite_response_law.control_closeout import FiniteResponseLawRootSealedControl
from empirical_lawhood.adapters.methods.finite_response_law.method_records import FiniteResponseLawAssignedQualificationReport
from empirical_lawhood.adapters.methods.finite_response_law.assigned_native_records import FiniteResponseLawAssignedCalibrationNativeEvaluation, FiniteResponseLawAssignedEvaluationProjectionConfig, FiniteResponseLawAssignedEvaluationCompletionConfig, FiniteResponseLawAssignedEvaluationViewObservation, FiniteResponseLawAssignedEvaluationNativeCompletion
from ..assigned_contracts import FiniteResponseLawAssignedEvaluationConfig
from ..source_outputs import FiniteResponseLawAssignedEvaluationTaskResult
from ..native_artifact import NATIVE_PAIR_SCHEMA
from ..calibration.discovery import SOURCE_CAPABILITY as CAL_SOURCE, PROJECTION_CAPABILITY as CAL_PROJECTION

NAMESPACE = "finite-response-law.evaluation"
SOURCE_CAPABILITY = manifest(
    "source",
    FiniteResponseLawAssignedEvaluationConfig,
    CAL_SOURCE.kind,
    (
        FiniteResponseLawAssignedEvaluationConfig.SCHEMA,
        FiniteResponseLawAssignedEvaluationTaskResult.SCHEMA,
        NATIVE_PAIR_SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
        FiniteResponseLawRootControlLock.SCHEMA,
        FiniteResponseLawRootParentJoin.SCHEMA,
    ),
    (FiniteResponseLawAssignedEvaluationTaskResult.SCHEMA, NATIVE_PAIR_SCHEMA, LinkedCampaignStageEnvelope.SCHEMA),
    EvidenceCeiling.MEASUREMENT,
    OutcomeAccess.EVALUATION_SEALED,
    1800,
    namespace=NAMESPACE,
    resources=replace(CAL_SOURCE.resource_ceiling, source_scan_bytes=192 * 1024**2),
    implementation_id="fresh-64-root-source-with-persisted-pre-parent-control-guards",
    conformance_ids=(
        "exact-fresh-evaluation-census",
        "pre-parent-lock-before-native-effects",
        "unchanged-native-marcher",
    ),
    read_permission=(CapabilityPermission.READ_OUTCOME_VISIBLE if 'source' == "evaluation" else CapabilityPermission.READ_SEALED_OUTCOMES),
)
PROJECTION_CAPABILITY = manifest(
    "projection",
    FiniteResponseLawAssignedEvaluationProjectionConfig,
    CapabilityKind.TRANSFORM,
    (
        FiniteResponseLawAssignedEvaluationProjectionConfig.SCHEMA,
        FiniteResponseLawAssignedEvaluationTaskResult.SCHEMA,
        NATIVE_PAIR_SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
    ),
    (FiniteResponseLawAssignedEvaluationViewObservation.SCHEMA, LinkedCampaignStageEnvelope.SCHEMA),
    EvidenceCeiling.MEASUREMENT,
    OutcomeAccess.EVALUATION_SEALED,
    120,
    namespace=NAMESPACE,
    resources=CAL_PROJECTION.resource_ceiling,
    implementation_id="fresh-evaluation-same-paired-native-reducer",
    conformance_ids=(
        "actual-signed-mate-and-hold",
        "complete-future-view-census",
        "unchanged-receiver",
    ),
    read_permission=(CapabilityPermission.READ_OUTCOME_VISIBLE if 'projection' == "evaluation" else CapabilityPermission.READ_SEALED_OUTCOMES),
)
EVALUATION_CAPABILITY = manifest(
    "sealed-completion",
    FiniteResponseLawAssignedEvaluationCompletionConfig,
    CapabilityKind.EVALUATOR,
    (
        FiniteResponseLawAssignedEvaluationCompletionConfig.SCHEMA,
        FiniteResponseLawAssignedEvaluationViewObservation.SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
        FiniteResponseLawRootParentJoin.SCHEMA,
        FiniteResponseLawRootSealedControl.SCHEMA,
    ),
    (FiniteResponseLawAssignedEvaluationNativeCompletion.SCHEMA, FiniteResponseLawRootSealedControl.SCHEMA),
    EvidenceCeiling.MEASUREMENT,
    OutcomeAccess.EVALUATION_SEALED,
    120,
    namespace=NAMESPACE,
    resources=ResourceBudget(1, 1024**3, 0, 120, 2 * 1024**3, 16 * 1024**2),
    implementation_id="complete-64-root-sealed-native-census-no-adjudication",
    conformance_ids=("complete-64-root-native-census", "recovery-before-separate-reveal"),
    read_permission=(CapabilityPermission.READ_OUTCOME_VISIBLE if 'sealed-completion' == "evaluation" else CapabilityPermission.READ_SEALED_OUTCOMES),
)
CONTROL_CAPABILITY = manifest(
    "control",
    FiniteResponseLawControlConfig,
    CapabilityKind.CONTROLLER_SYNTHESIZER,
    (
        FiniteResponseLawControlConfig.SCHEMA,
        FiniteResponseLawAssignedQualificationReport.SCHEMA,
        FiniteResponseLawAssignedCalibrationNativeEvaluation.SCHEMA,
        CanonicalTaskReceipt.SCHEMA,
        FiniteResponseLawAssignedEvaluationTaskResult.SCHEMA,
        NATIVE_PAIR_SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
        FiniteResponseLawRootForecast.SCHEMA,
        FiniteResponseLawRootRequestReveal.SCHEMA,
        FiniteResponseLawRootControlLock.SCHEMA,
        FiniteResponseLawRootParentJoin.SCHEMA,
    ),
    (
        FiniteResponseLawRootForecast.SCHEMA,
        FiniteResponseLawRootRequestReveal.SCHEMA,
        FiniteResponseLawRootControlLock.SCHEMA,
        FiniteResponseLawRootParentJoin.SCHEMA,
        FiniteResponseLawAssignedQualificationReport.SCHEMA,
    ),
    EvidenceCeiling.ADMISSION,
    OutcomeAccess.OUTCOME_BLIND,
    180,
    namespace=NAMESPACE,
    resources=ResourceBudget(1, 1024**3, 0, 180, 256 * 1024**2, 128 * 1024**2),
    implementation_id='finite-response-law-admission-native-magnitude-prepared-lock-owners',
    conformance_ids=(
        "cancel-without-reselection",
        "corrected-current-controller-contract",
        "six-consumers-before-parent",
    ),
    read_permission=(CapabilityPermission.READ_OUTCOME_VISIBLE if 'control' == "evaluation" else CapabilityPermission.READ_SEALED_OUTCOMES),
)
CONTROL_CAPABILITY = replace(
    CONTROL_CAPABILITY,
    permissions=tuple(
        p
        for p in CONTROL_CAPABILITY.permissions
        if p is not CapabilityPermission.READ_SEALED_OUTCOMES
    ),
)
REVEAL_CAPABILITY = manifest(
    "reveal",
    FiniteResponseLawEvaluationRevealConfig,
    CapabilityKind.EVALUATOR,
    (
        FiniteResponseLawEvaluationRevealConfig.SCHEMA,
        FiniteResponseLawAssignedEvaluationNativeCompletion.SCHEMA,
        FiniteResponseLawRootSealedControl.SCHEMA,
        FiniteResponseLawAssignedEvaluationViewObservation.SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
        FiniteResponseLawProspectiveRootEvaluation.SCHEMA,
        FiniteResponseLawRootInferenceOperands.SCHEMA,
        FiniteResponseLawEvaluationCohort.SCHEMA,
        FiniteResponseLawAssignedQualificationReport.SCHEMA,
        FiniteResponseLawAssignedCalibrationNativeEvaluation.SCHEMA,
        CanonicalTaskReceipt.SCHEMA,
    ),
    (
        FiniteResponseLawProspectiveRootEvaluation.SCHEMA,
        FiniteResponseLawRootInferenceOperands.SCHEMA,
        FiniteResponseLawEvaluationCohort.SCHEMA,
        ScientificAdjudicationRecord.SCHEMA,
    ),
    EvidenceCeiling.CONTROLLER_USE,
    OutcomeAccess.EVALUATION_REVEALED,
    120,
    namespace=NAMESPACE,
    resources=ResourceBudget(1, 2 * 1024**3, 0, 120, 2 * 1024**3, 16 * 1024**2),
    implementation_id='finite-response-law-prospective-evaluation-all-root-scalar-inference-and-adjudication',
    conformance_ids=(
        "all-assigned-root-inference-no-complete-case-filter",
        'independent-scalar-prospective-evaluation-agreement',
        "recovery-before-outcome-reveal",
    ),
    read_permission=(CapabilityPermission.READ_OUTCOME_VISIBLE if 'reveal' == "evaluation" else CapabilityPermission.READ_SEALED_OUTCOMES),
)
REVEAL_CAPABILITY = replace(
    REVEAL_CAPABILITY,
    permissions=tuple(
        sorted(
            set(REVEAL_CAPABILITY.permissions)
            | {
                CapabilityPermission.READ_SEALED_OUTCOMES,
                CapabilityPermission.REVEAL_OUTCOMES,
                CapabilityPermission.READ_OUTCOME_VISIBLE,
            },
            key=lambda p: p.value,
        )
    ),
)
_source, SOURCE_COMPONENTS = bundle(
    "source",
    ExtensionContributionKind.SOURCE,
    SOURCE_CAPABILITY,
    (FiniteResponseLawAssignedEvaluationConfig, QualifiedSourceUseExperiment, QualifiedSourceUseSubstrateBinding),
    NATIVE_PAIR_SCHEMA,
    namespace=NAMESPACE,
)
_projection, PROJECTION_COMPONENTS = bundle(
    "projection",
    ExtensionContributionKind.METHOD,
    PROJECTION_CAPABILITY,
    (FiniteResponseLawAssignedEvaluationProjectionConfig,),
    None,
    namespace=NAMESPACE,
)
_evaluation, EVALUATION_COMPONENTS = bundle(
    "sealed-completion",
    ExtensionContributionKind.METHOD,
    EVALUATION_CAPABILITY,
    (FiniteResponseLawAssignedEvaluationCompletionConfig,),
    None,
    namespace=NAMESPACE,
)
_control, CONTROL_COMPONENTS = bundle(
    "control",
    ExtensionContributionKind.METHOD,
    CONTROL_CAPABILITY,
    (FiniteResponseLawControlConfig,),
    None,
    namespace=NAMESPACE,
)
_reveal, REVEAL_COMPONENTS = bundle(
    "reveal",
    ExtensionContributionKind.METHOD,
    REVEAL_CAPABILITY,
    (FiniteResponseLawEvaluationRevealConfig,),
    None,
    namespace=NAMESPACE,
)
_parts = (_source, _projection, _evaluation, _control, _reveal)
EXTENSION_BUNDLE_CONTRIBUTION = replace(
    _source,
    contribution_id=f"extension-contribution.{NAMESPACE}",
    capability_manifests=tuple(
        sorted((v for p in _parts for v in p.capability_manifests), key=lambda v: v.registry_id)
    ),
    candidate_capability_registrations=tuple(
        sorted(
            (v for p in _parts for v in p.candidate_capability_registrations),
            key=lambda v: v.registration_id,
        )
    ),
    profile_registrations=tuple(
        sorted(
            (v for p in _parts for v in p.profile_registrations), key=lambda v: v.registration_id
        )
    ),
    config_decoders=tuple(
        sorted((v for p in _parts for v in p.config_decoders), key=lambda v: v.object_id)
    ),
    runtime_providers=tuple(
        sorted((v for p in _parts for v in p.runtime_providers), key=lambda v: v.object_id)
    ),
)
