"""preparation-policy development source/projection/screen discovery metadata."""

from dataclasses import replace

from empirical_lawhood.adapters.methods.finite_response_law.law_payloads import FiniteResponseLawLowerPayload
from empirical_lawhood.adapters.methods.finite_response_law.extension_bundle import EVALUATION_CAPABILITY as OLD_EVALUATION_CAPABILITY, PROJECTION_CAPABILITY as OLD_PROJECTION_CAPABILITY
from empirical_lawhood.adapters.methods.finite_response_law.preparation_policy_projection import FiniteResponseLawPreparationPolicyProjectionConfig, FiniteResponseLawPreparationPolicyRootPanel
from empirical_lawhood.adapters.methods.finite_response_law.preparation_screen_results import FiniteResponseLawRetainedProspectiveCloseoutReference, FiniteResponseLawPreparationScreenResult, FiniteResponseLawPreparationPolicyScreenConfig, FiniteResponseLawPreparationPolicyUpperPayloadFreeze
from empirical_lawhood.adapters.methods.finite_response_law.science import PROGRAMME
from empirical_lawhood.adapters.simulators.finite_response_law.preparation_policy_contracts import FiniteResponseLawPreparationPolicyNativeConfig
from empirical_lawhood.adapters.simulators.finite_response_law.preparation_policy_native_artifact import PREPARATION_POLICY_NATIVE_PAIR_SCHEMA
from empirical_lawhood.adapters.simulators.finite_response_law.preparation_policy_source_outputs import FiniteResponseLawPreparationPolicyNativeTaskResult
from empirical_lawhood.adapters.simulators.prepared_response.native_pair import NATIVE_PAIR_SCHEMA as RETAINED_PAIR_SCHEMA
from empirical_lawhood.adapters.simulators.prepared_response.source_outputs import PreparedNativeTaskResult
from empirical_lawhood.adapters.composition.discovery import bundle, manifest
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.source_qualification import PredecessorBoundSourceQualificationExperiment
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.artifacts import CanonicalTaskReceipt
from empirical_lawhood.runtime.capabilities import CapabilityKind, CapabilityPermission
from empirical_lawhood.runtime.extension_bundles import ExtensionContributionKind
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.source_qualification import PredecessorBoundSourceQualificationSubstrateBinding

NAMESPACE = f"{PROGRAMME}.preparation-policy"
SOURCE_RECORDS: tuple[type[CanonicalRecord], ...] = (
    FiniteResponseLawPreparationPolicyNativeConfig,
    PredecessorBoundSourceQualificationExperiment,
    PredecessorBoundSourceQualificationSubstrateBinding,
)
SOURCE_CAPABILITY = manifest(
    "source",
    FiniteResponseLawPreparationPolicyNativeConfig,
    CapabilityKind.SIMULATOR,
    (
        FiniteResponseLawPreparationPolicyNativeConfig.SCHEMA,
        FiniteResponseLawPreparationPolicyNativeTaskResult.SCHEMA,
        PREPARATION_POLICY_NATIVE_PAIR_SCHEMA,
        PreparedNativeTaskResult.SCHEMA,
        RETAINED_PAIR_SCHEMA,
        CanonicalTaskReceipt.SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
    ),
    (
        FiniteResponseLawPreparationPolicyNativeTaskResult.SCHEMA,
        PREPARATION_POLICY_NATIVE_PAIR_SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
    ),
    EvidenceCeiling.MEASUREMENT,
    OutcomeAccess.EVALUATION_SEALED,
    86_400,
    namespace=NAMESPACE,
    resources=ResourceBudget(1, 1024**3, 0, 86_400, 64 * 1024**2, 16 * 1024**2),
    implementation_id="retained-prefix-nine-schedule-plus400-native",
    conformance_ids=(
        "exact-24-retained-prefixes",
        "nine-plus400-schedules",
        "shared-innovations-paired-views-futures",
    ),
    read_permission=(CapabilityPermission.READ_OUTCOME_VISIBLE if 'source' == "evaluation" else CapabilityPermission.READ_SEALED_OUTCOMES),
)
SOURCE_CAPABILITY = replace(
    SOURCE_CAPABILITY,
    permissions=tuple(
        sorted((*SOURCE_CAPABILITY.permissions, CapabilityPermission.READ_OUTCOME_VISIBLE))
    ),
)
PROJECTION_CAPABILITY = manifest(
    "projection",
    FiniteResponseLawPreparationPolicyProjectionConfig,
    OLD_PROJECTION_CAPABILITY.kind,
    (
        FiniteResponseLawPreparationPolicyProjectionConfig.SCHEMA,
        FiniteResponseLawPreparationPolicyNativeTaskResult.SCHEMA,
        PREPARATION_POLICY_NATIVE_PAIR_SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
    ),
    (FiniteResponseLawPreparationPolicyRootPanel.SCHEMA, LinkedCampaignStageEnvelope.SCHEMA),
    EvidenceCeiling.MEASUREMENT,
    OutcomeAccess.EVALUATION_SEALED,
    3_600,
    namespace=NAMESPACE,
    resources=ResourceBudget(1, 1024**3, 0, 3_600, 3 * 1024**3, 8 * 1024**2),
    implementation_id="complete-nine-schedule-root-projection",
    conformance_ids=("all-171-source-results", "explicit-O-D-masks", "native-feature-cutoffs"),
    read_permission=(CapabilityPermission.READ_OUTCOME_VISIBLE if 'projection' == "evaluation" else CapabilityPermission.READ_SEALED_OUTCOMES),
)
EVALUATION_CAPABILITY = manifest(
    "evaluation",
    FiniteResponseLawPreparationPolicyScreenConfig,
    OLD_EVALUATION_CAPABILITY.kind,
    (
        FiniteResponseLawPreparationPolicyScreenConfig.SCHEMA,
        FiniteResponseLawPreparationPolicyRootPanel.SCHEMA,
        FiniteResponseLawLowerPayload.SCHEMA,
        'empirical-lawhood/methods/finite-response-law/finite-response-law-qualification-report',
        ScientificAdjudicationRecord.SCHEMA,
        FiniteResponseLawRetainedProspectiveCloseoutReference.SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
    ),
    (
        FiniteResponseLawPreparationScreenResult.SCHEMA,
        FiniteResponseLawPreparationPolicyUpperPayloadFreeze.SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
        ScientificAdjudicationRecord.SCHEMA,
    ),
    EvidenceCeiling.RESPONSE,
    OutcomeAccess.EVALUATOR_REVEAL,
    3_600,
    namespace=NAMESPACE,
    resources=ResourceBudget(1, 1024**3, 0, 3_600, 256 * 1024**2, 16 * 1024**2),
    implementation_id="preparation-policy-cross-fitted-provider-adequacy-use-screen",
    conformance_ids=(
        "four-outer-three-coefficient-folds",
        "separate-full-menu-A-actual-J-joined-C",
        "unchanged-qualified-lower",
    ),
    read_permission=(CapabilityPermission.READ_OUTCOME_VISIBLE if 'evaluation' == "evaluation" else CapabilityPermission.READ_SEALED_OUTCOMES),
)
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
    (FiniteResponseLawPreparationPolicyNativeConfig,),
    PREPARATION_POLICY_NATIVE_PAIR_SCHEMA,
    namespace=NAMESPACE,
    maximum_config_bytes=2 * 1024**2,
)
_projection, PROJECTION_COMPONENTS = bundle(
    "projection",
    ExtensionContributionKind.METHOD,
    PROJECTION_CAPABILITY,
    (FiniteResponseLawPreparationPolicyProjectionConfig,),
    None,
    namespace=NAMESPACE,
)
_evaluation, EVALUATION_COMPONENTS = bundle(
    "evaluation",
    ExtensionContributionKind.METHOD,
    EVALUATION_CAPABILITY,
    (FiniteResponseLawPreparationPolicyScreenConfig,),
    None,
    namespace=NAMESPACE,
)
_parts = (_source, _projection, _evaluation)
EXTENSION_BUNDLE_CONTRIBUTION = replace(
    _source,
    contribution_id=f"extension-contribution.{NAMESPACE}",
    capability_manifests=tuple(
        sorted(
            (value for part in _parts for value in part.capability_manifests),
            key=lambda value: value.registry_id,
        )
    ),
    candidate_capability_registrations=tuple(
        sorted(
            (value for part in _parts for value in part.candidate_capability_registrations),
            key=lambda value: value.registration_id,
        )
    ),
    profile_registrations=tuple(
        sorted(
            (value for part in _parts for value in part.profile_registrations),
            key=lambda value: value.registration_id,
        )
    ),
    config_decoders=tuple(
        sorted(
            (value for part in _parts for value in part.config_decoders),
            key=lambda value: value.object_id,
        )
    ),
    runtime_providers=tuple(
        sorted(
            (value for part in _parts for value in part.runtime_providers),
            key=lambda value: value.object_id,
        )
    ),
)
