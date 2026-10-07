"""Nonacquiring retained-projection completion; no new numerical science."""

from dataclasses import replace
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.artifacts import CanonicalTaskReceipt
from empirical_lawhood.runtime.capabilities import CapabilityKind, CapabilityPermission
from empirical_lawhood.runtime.extension_bundles import ExtensionContributionKind
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from empirical_lawhood.adapters.composition.discovery import manifest, bundle
from ..native_records import FiniteResponseLawCalibrationViewObservation, FiniteResponseLawCalibrationNativeEvaluation
from ..assigned_native_records import FiniteResponseLawAssignedEvaluationViewObservation, FiniteResponseLawAssignedEvaluationNativeCompletion
from .contracts import FiniteResponseLawAssignedRetainedCompletionConfig, FiniteResponseLawRetainedCompletionConfig, PROSPECTIVE_EVALUATION_COMPLETION_PREFIX, PREFIX
from empirical_lawhood.adapters.simulators.finite_response_law.fresh_contracts import FiniteResponseLawCalibrationConfig
from empirical_lawhood.adapters.simulators.finite_response_law.assigned_contracts import FiniteResponseLawAssignedEvaluationConfig

CAPABILITY = manifest(
    "completion",
    FiniteResponseLawRetainedCompletionConfig,
    CapabilityKind.EVALUATOR,
    (
        FiniteResponseLawRetainedCompletionConfig.SCHEMA,
        FiniteResponseLawCalibrationConfig.SCHEMA,
        ObjectIdentity.SCHEMA,
        FiniteResponseLawCalibrationViewObservation.SCHEMA,
        CanonicalTaskReceipt.SCHEMA,
        FiniteResponseLawCalibrationNativeEvaluation.SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
    ),
    (
        ObjectIdentity.SCHEMA,
        FiniteResponseLawCalibrationNativeEvaluation.SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
        ScientificAdjudicationRecord.SCHEMA,
    ),
    EvidenceCeiling.MEASUREMENT,
    OutcomeAccess.EVALUATOR_REVEAL,
    180,
    namespace=PREFIX,
    resources=ResourceBudget(1, 1024**3, 0, 180, 96 * 1024**2, 16 * 1024**2),
    read_permission=CapabilityPermission.READ_SEALED_OUTCOMES,
    implementation_id="retained-64-projection-custody-separate-adjudication",
    conformance_ids=(
        "exact-retained-projection-receipts",
        "no-native-recomputation",
        "separate-publication-evidence-classes",
    ),
)
CAPABILITY = replace(
    CAPABILITY,
    permissions=tuple(sorted((*CAPABILITY.permissions, CapabilityPermission.REVEAL_OUTCOMES))),
)
EXTENSION_BUNDLE_CONTRIBUTION, COMPONENTS = bundle(
    "completion",
    ExtensionContributionKind.METHOD,
    CAPABILITY,
    (FiniteResponseLawRetainedCompletionConfig,),
    None,
    namespace=PREFIX,
)
ASSIGNED_CAPABILITY = manifest(
    "completion",
    FiniteResponseLawAssignedRetainedCompletionConfig,
    CapabilityKind.EVALUATOR,
    (
        FiniteResponseLawAssignedRetainedCompletionConfig.SCHEMA,
        FiniteResponseLawAssignedEvaluationConfig.SCHEMA,
        ObjectIdentity.SCHEMA,
        FiniteResponseLawAssignedEvaluationViewObservation.SCHEMA,
        CanonicalTaskReceipt.SCHEMA,
        FiniteResponseLawAssignedEvaluationNativeCompletion.SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
    ),
    (
        ObjectIdentity.SCHEMA,
        FiniteResponseLawAssignedEvaluationNativeCompletion.SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
        ScientificAdjudicationRecord.SCHEMA,
    ),
    EvidenceCeiling.MEASUREMENT,
    OutcomeAccess.EVALUATOR_REVEAL,
    180,
    namespace=PROSPECTIVE_EVALUATION_COMPLETION_PREFIX,
    resources=ResourceBudget(1, 1024**3, 0, 180, 96 * 1024**2, 16 * 1024**2),
    read_permission=CapabilityPermission.READ_SEALED_OUTCOMES,
    implementation_id="assigned-retained-128-projection-custody-separate-adjudication",
    conformance_ids=(
        "exact-retained-projection-receipts",
        "no-native-recomputation",
        "separate-publication-evidence-classes",
    ),
)
ASSIGNED_CAPABILITY = replace(
    ASSIGNED_CAPABILITY,
    permissions=tuple(sorted((*ASSIGNED_CAPABILITY.permissions, CapabilityPermission.REVEAL_OUTCOMES))),
)
_assigned, ASSIGNED_COMPONENTS = bundle(
    "completion",
    ExtensionContributionKind.METHOD,
    ASSIGNED_CAPABILITY,
    (FiniteResponseLawAssignedRetainedCompletionConfig,),
    None,
    namespace=PROSPECTIVE_EVALUATION_COMPLETION_PREFIX,
)
_parts = (EXTENSION_BUNDLE_CONTRIBUTION, _assigned)
EXTENSION_BUNDLE_CONTRIBUTION = replace(
    EXTENSION_BUNDLE_CONTRIBUTION,
    capability_manifests=tuple(
        sorted((v for part in _parts for v in part.capability_manifests), key=lambda v: v.registry_id)
    ),
    candidate_capability_registrations=tuple(
        sorted(
            (v for part in _parts for v in part.candidate_capability_registrations),
            key=lambda v: v.registration_id,
        )
    ),
    profile_registrations=tuple(
        sorted((v for part in _parts for v in part.profile_registrations), key=lambda v: v.registration_id)
    ),
    config_decoders=tuple(
        sorted((v for part in _parts for v in part.config_decoders), key=lambda v: v.object_id)
    ),
    runtime_providers=tuple(
        sorted((v for part in _parts for v in part.runtime_providers), key=lambda v: v.object_id)
    ),
)
