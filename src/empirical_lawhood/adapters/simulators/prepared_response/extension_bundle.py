"Import-safe native source qualification/dependent refinement discovery, distinct from source readiness and issue."

from dataclasses import replace

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.source_qualification import FreshSourceQualificationExperiment
from empirical_lawhood.runtime.capabilities import CapabilityKind, CapabilityPermission
from empirical_lawhood.runtime.extension_bundles import ExtensionContributionKind
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.source_qualification import FreshSourceQualificationSubstrateBinding
from empirical_lawhood.adapters.composition.discovery import bundle, manifest
from .contracts import PreparedNativeSpec
from .native_pair import NATIVE_PAIR_SCHEMA
from .source_outputs import PreparedNativeTaskResult
from .policy_native import PreparedResponseCalibrationNativeTaskResult
from empirical_lawhood.adapters.methods.prepared_response.policy_decision import PreparedParentDecision


SOURCE_QUALIFICATION_NAMESPACE = "prepared-response.source-qualification"
DEVELOPMENT_NAMESPACE = "prepared-response.dependent-refinement"
SOURCE_QUALIFICATION_SOURCE_RECORDS: tuple[type[CanonicalRecord], ...] = (
    PreparedNativeSpec,
    FreshSourceQualificationExperiment,
    FreshSourceQualificationSubstrateBinding,
)
SOURCE_QUALIFICATION_SOURCE_CAPABILITY = manifest(
    "source",
    PreparedNativeSpec,
    CapabilityKind.SIMULATOR,
    (
        PreparedNativeSpec.SCHEMA,
        PreparedNativeTaskResult.SCHEMA,
        NATIVE_PAIR_SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
    ),
    (PreparedNativeTaskResult.SCHEMA, NATIVE_PAIR_SCHEMA, LinkedCampaignStageEnvelope.SCHEMA),
    EvidenceCeiling.MEASUREMENT,
    OutcomeAccess.EVALUATION_SEALED,
    1800,
    namespace=SOURCE_QUALIFICATION_NAMESPACE,
    resources=ResourceBudget(1, 768 * 1024**2, 0, 1800, 32 * 1024**2, 20 * 1024**2),
    implementation_id="source-qualification-two-port-common-start-native-pair",
    conformance_ids=(
        "authenticated-common-start-continuation",
        "exact-source-qualification-native-phase-census",
        "no-scientific-retry",
        "paired-numerical-view-delivery",
    ),
    read_permission=(CapabilityPermission.READ_OUTCOME_VISIBLE if 'source' == "evaluation" else CapabilityPermission.READ_SEALED_OUTCOMES),
)
EXTENSION_BUNDLE_CONTRIBUTION, SOURCE_QUALIFICATION_SOURCE_COMPONENTS = bundle(
    "source",
    ExtensionContributionKind.SOURCE,
    SOURCE_QUALIFICATION_SOURCE_CAPABILITY,
    (PreparedNativeSpec,),
    NATIVE_PAIR_SCHEMA,
    namespace=SOURCE_QUALIFICATION_NAMESPACE,
)
DEVELOPMENT_SOURCE_CAPABILITY = manifest(
    "source",
    PreparedNativeSpec,
    CapabilityKind.SIMULATOR,
    (
        PreparedNativeSpec.SCHEMA,
        PreparedNativeTaskResult.SCHEMA,
        NATIVE_PAIR_SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
    ),
    (PreparedNativeTaskResult.SCHEMA, NATIVE_PAIR_SCHEMA, LinkedCampaignStageEnvelope.SCHEMA),
    EvidenceCeiling.MEASUREMENT,
    OutcomeAccess.DEVELOPMENT_VISIBLE,
    1800,
    namespace=DEVELOPMENT_NAMESPACE,
    resources=ResourceBudget(1, 768 * 1024**2, 0, 1800, 32 * 1024**2, 20 * 1024**2),
    read_permission=CapabilityPermission.READ_OUTCOME_VISIBLE,
    implementation_id="dependent-refinement-two-port-common-start-native-pair",
    conformance_ids=(
        "authenticated-common-start-continuation",
        "exact-dependent-refinement-native-phase-census",
        "no-scientific-retry",
        "paired-numerical-view-delivery",
    ),
)
_development_source, DEVELOPMENT_SOURCE_COMPONENTS = bundle(
    "source",
    ExtensionContributionKind.SOURCE,
    DEVELOPMENT_SOURCE_CAPABILITY,
    (PreparedNativeSpec,),
    NATIVE_PAIR_SCHEMA,
    namespace=DEVELOPMENT_NAMESPACE,
)
CALIBRATION_SOURCE_CAPABILITY = manifest(
    "source",
    PreparedNativeSpec,
    CapabilityKind.SIMULATOR,
    (
        PreparedNativeSpec.SCHEMA,
        PreparedResponseCalibrationNativeTaskResult.SCHEMA,
        PreparedParentDecision.SCHEMA,
        NATIVE_PAIR_SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
    ),
    (
        PreparedResponseCalibrationNativeTaskResult.SCHEMA,
        NATIVE_PAIR_SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
    ),
    EvidenceCeiling.MEASUREMENT,
    OutcomeAccess.EVALUATION_SEALED,
    1800,
    namespace="prepared-response.fresh-response-calibration",
    resources=ResourceBudget(1, 768 * 1024**2, 0, 1800, 64 * 1024**2, 20 * 1024**2),
    implementation_id="fresh-calibration-policy-keyed-common-start-native-pair",
    conformance_ids=(
        "authenticated-common-start-continuation",
        "development-frozen-target-blind-policy-decision",
        "exact-fresh-response-calibration-native-phase-census",
        "no-scientific-retry",
        "paired-numerical-view-delivery",
    ),
    read_permission=(CapabilityPermission.READ_OUTCOME_VISIBLE if 'source' == "evaluation" else CapabilityPermission.READ_SEALED_OUTCOMES),
)
_calibration_source, CALIBRATION_SOURCE_COMPONENTS = bundle(
    "source",
    ExtensionContributionKind.SOURCE,
    CALIBRATION_SOURCE_CAPABILITY,
    (PreparedNativeSpec,),
    NATIVE_PAIR_SCHEMA,
    namespace="prepared-response.fresh-response-calibration",
)
EXTENSION_BUNDLE_CONTRIBUTION = replace(
    EXTENSION_BUNDLE_CONTRIBUTION,
    contribution_id="extension-contribution.prepared-response.source-qualification-dependent-refinement-fresh-calibration.source",
    capability_manifests=tuple(
        sorted(
            (
                *EXTENSION_BUNDLE_CONTRIBUTION.capability_manifests,
                *_development_source.capability_manifests,
                *_calibration_source.capability_manifests,
            ),
            key=lambda value: value.registry_id,
        )
    ),
    candidate_capability_registrations=tuple(
        sorted(
            (
                *EXTENSION_BUNDLE_CONTRIBUTION.candidate_capability_registrations,
                *_development_source.candidate_capability_registrations,
                *_calibration_source.candidate_capability_registrations,
            ),
            key=lambda value: value.manifest.registry_id,
        )
    ),
    profile_registrations=tuple(
        sorted(
            (
                *EXTENSION_BUNDLE_CONTRIBUTION.profile_registrations,
                *_development_source.profile_registrations,
                *_calibration_source.profile_registrations,
            ),
            key=lambda value: value.registration_id,
        )
    ),
    config_decoders=tuple(
        sorted(
            (
                *EXTENSION_BUNDLE_CONTRIBUTION.config_decoders,
                *_development_source.config_decoders,
                *_calibration_source.config_decoders,
            ),
            key=lambda value: value.object_id,
        )
    ),
    runtime_providers=tuple(
        sorted(
            (
                *EXTENSION_BUNDLE_CONTRIBUTION.runtime_providers,
                *_development_source.runtime_providers,
                *_calibration_source.runtime_providers,
            ),
            key=lambda value: value.object_id,
        )
    ),
    artifact_validators=tuple(
        sorted(
            (
                *EXTENSION_BUNDLE_CONTRIBUTION.artifact_validators,
                *_development_source.artifact_validators,
                *_calibration_source.artifact_validators,
            ),
            key=lambda value: value.object_id,
        )
    ),
)
