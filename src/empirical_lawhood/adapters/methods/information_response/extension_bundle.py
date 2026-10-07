"Import-safe information response prediction projection and evaluation discovery."
from empirical_lawhood.runtime.capabilities import CapabilityPermission


from dataclasses import replace

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.capabilities import CapabilityKind
from empirical_lawhood.runtime.extension_bundles import ExtensionContributionKind
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from empirical_lawhood.adapters.composition.discovery import manifest, bundle
from empirical_lawhood.adapters.simulators.prepared_response.native_pair import NATIVE_PAIR_SCHEMA
from empirical_lawhood.adapters.simulators.prepared_response.source_outputs import PreparedNativeTaskResult
from .models import PROGRAMME
from .records import InformationResponseCommittedPrediction, InformationResponseProjectionConfig, InformationResponseEvaluationConfig, InformationResponseViewObservation, InformationResponseEvaluation

PROJECTION_CAPABILITY = manifest(
    "projection",
    InformationResponseProjectionConfig,
    CapabilityKind.TRANSFORM,
    (
        InformationResponseProjectionConfig.SCHEMA,
        PreparedNativeTaskResult.SCHEMA,
        InformationResponseCommittedPrediction.SCHEMA,
        NATIVE_PAIR_SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
    ),
    (InformationResponseViewObservation.SCHEMA, LinkedCampaignStageEnvelope.SCHEMA),
    EvidenceCeiling.RESPONSE,
    OutcomeAccess.EVALUATION_SEALED,
    60,
    namespace=PROGRAMME,
    resources=ResourceBudget(1, 3 * 1024**3, 0, 60, 1024**3, 1024**2),
    implementation_id="information-prediction-paired-native-contrast-projection",
    read_permission=(CapabilityPermission.READ_OUTCOME_VISIBLE if 'projection' == "evaluation" else CapabilityPermission.READ_SEALED_OUTCOMES),
    conformance_ids=("exact-frozen-source-qualification-roster", 'no-scientific-retry', 'separate-root-view-action-clocks'),
)
EVALUATION_CAPABILITY = manifest(
    "evaluation",
    InformationResponseEvaluationConfig,
    CapabilityKind.EVALUATOR,
    (
        InformationResponseEvaluationConfig.SCHEMA,
        InformationResponseViewObservation.SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
    ),
    (
        InformationResponseEvaluation.SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
        ScientificAdjudicationRecord.SCHEMA,
    ),
    EvidenceCeiling.RESPONSE,
    OutcomeAccess.EVALUATION_REVEALED,
    60,
    namespace=PROGRAMME,
    resources=ResourceBudget(1, 3 * 1024**3, 0, 60, 256 * 1024**2, 1024**2),
    implementation_id="independent-context-exact-root-prediction-comparison",
    read_permission=(CapabilityPermission.READ_OUTCOME_VISIBLE if 'evaluation' == "evaluation" else CapabilityPermission.READ_SEALED_OUTCOMES),
    conformance_ids=("exact-frozen-source-qualification-roster", 'no-scientific-retry', 'separate-root-view-action-clocks'),
)
_projection, PROJECTION_COMPONENTS = bundle(
    "projection",
    ExtensionContributionKind.METHOD,
    PROJECTION_CAPABILITY,
    (InformationResponseProjectionConfig,),
    None,
    namespace=PROGRAMME,
)
_evaluation, EVALUATION_COMPONENTS = bundle(
    "evaluation",
    ExtensionContributionKind.METHOD,
    EVALUATION_CAPABILITY,
    (InformationResponseEvaluationConfig,),
    None,
    namespace=PROGRAMME,
)
EXTENSION_BUNDLE_CONTRIBUTION = replace(
    _projection,
    contribution_id=f"extension-contribution.{PROGRAMME}.methods",
    capability_manifests=tuple(
        sorted(
            (*_projection.capability_manifests, *_evaluation.capability_manifests),
            key=lambda v: v.registry_id,
        )
    ),
    candidate_capability_registrations=tuple(
        sorted(
            (
                *_projection.candidate_capability_registrations,
                *_evaluation.candidate_capability_registrations,
            ),
            key=lambda v: v.manifest.registry_id,
        )
    ),
    profile_registrations=tuple(
        sorted(
            (*_projection.profile_registrations, *_evaluation.profile_registrations),
            key=lambda v: v.registration_id,
        )
    ),
    config_decoders=tuple(
        sorted(
            (*_projection.config_decoders, *_evaluation.config_decoders), key=lambda v: v.object_id
        )
    ),
    runtime_providers=tuple(
        sorted(
            (*_projection.runtime_providers, *_evaluation.runtime_providers),
            key=lambda v: v.object_id,
        )
    ),
)
