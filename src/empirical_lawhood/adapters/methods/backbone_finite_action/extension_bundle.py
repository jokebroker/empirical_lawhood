"""Static finite-action capability, profile and validation declaration."""

from __future__ import annotations

import hashlib

from empirical_lawhood.adapters.methods.finite_action_identification import FiniteActionIdentificationConfig, FiniteActionIdentificationResult
from empirical_lawhood.adapters.methods.finite_action_registration import (
    FINITE_ACTION_CONFIG_MEDIA_TYPE,
    FINITE_ACTION_IDENTIFICATION_KEY,
    FINITE_ACTION_IDENTIFICATION_VERSION,
    FINITE_ACTION_MAXIMUM_CONFIG_BYTES,
    FINITE_ACTION_PROVIDER_KEY,
    finite_action_identification_manifest,
)
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.planning.evidence_profiles import ExperimentObjectiveProfile, build_observation_evidence_world_registry
from empirical_lawhood.runtime.candidate_composition import (
    CandidateCapabilityRegistration,
)
from empirical_lawhood.runtime.extension_bundles import ExtensionBundleContribution, ExtensionComponentKind, ExtensionComponentRegistration, ExtensionContributionKind, ExtensionProfileKind, ExtensionProfileRegistration


_IDENTIFICATION_IMPLEMENTATION_SHA256 = (
    "ec0fecc4fec007aa0e8909d677235f07bcd3107195960a9d38caccd6127197a8"
)
_REGISTRATION_IMPLEMENTATION_SHA256 = (
    "068565658890e6c6960417d4485d415cfd313e7450a4d3bd02b271e35584ef52"
)
_PROFILE_IMPLEMENTATION_SHA256 = (
    "1838a6f708feee9554c0465610ccb7c1771bb37e77cc44ab1b5415a963966baa"
)


FINITE_ACTION_MANIFEST = finite_action_identification_manifest(
    implementation_sha256=_IDENTIFICATION_IMPLEMENTATION_SHA256
)

FINITE_ACTION_CANDIDATE_REGISTRATION = CandidateCapabilityRegistration(
    manifest=FINITE_ACTION_MANIFEST,
    provider_key=FINITE_ACTION_PROVIDER_KEY,
    provider_version=FINITE_ACTION_IDENTIFICATION_VERSION,
    config_media_type=FINITE_ACTION_CONFIG_MEDIA_TYPE,
    maximum_config_bytes=FINITE_ACTION_MAXIMUM_CONFIG_BYTES,
)

LAW_CONTROL_OBJECTIVE_PROFILE_REGISTRATION = ExtensionProfileRegistration(
    registration_id='profile-registration.law-controller-use',
    profile_key='law-controller-use',
    profile_version="1.0.0",
    kind=ExtensionProfileKind.OBJECTIVE_TERMINAL,
    profile_schema=ExperimentObjectiveProfile.SCHEMA,
    profile_schema_sha256=hashlib.sha256(
        ExperimentObjectiveProfile.SCHEMA.encode()
    ).hexdigest(),
    implementation_sha256=_PROFILE_IMPLEMENTATION_SHA256,
)

FINITE_ACTION_METHOD = ExtensionComponentRegistration(
    registration_id="component.finite-action-identification",
    component_key=FINITE_ACTION_IDENTIFICATION_KEY,
    component_version=FINITE_ACTION_IDENTIFICATION_VERSION,
    kind=ExtensionComponentKind.METHOD,
    input_schema_ids=FINITE_ACTION_MANIFEST.input_schema_ids,
    output_schema_ids=FINITE_ACTION_MANIFEST.output_schema_ids,
    implementation_sha256=_IDENTIFICATION_IMPLEMENTATION_SHA256,
)

FINITE_ACTION_CONFIG_DECODER = ExtensionComponentRegistration(
    registration_id="component.finite-action-config-decoder",
    component_key="finite-action-config-decoder",
    component_version="1.0.0",
    kind=ExtensionComponentKind.CONFIG_DECODER,
    input_schema_ids=(FiniteActionIdentificationConfig.SCHEMA,),
    output_schema_ids=(FiniteActionIdentificationConfig.SCHEMA,),
    implementation_sha256=_REGISTRATION_IMPLEMENTATION_SHA256,
)

FINITE_ACTION_ARTIFACT_VALIDATOR = ExtensionComponentRegistration(
    registration_id="component.finite-action-artifact-validator",
    component_key="finite-action-artifact-validator",
    component_version="1.0.0",
    kind=ExtensionComponentKind.ARTIFACT_VALIDATOR,
    input_schema_ids=(FiniteActionIdentificationResult.SCHEMA,),
    output_schema_ids=('empirical-lawhood/runtime/artifact-generic-validation',),
    implementation_sha256=_REGISTRATION_IMPLEMENTATION_SHA256,
)

EXTENSION_BUNDLE_CONTRIBUTION = ExtensionBundleContribution(
    contribution_id="extension-contribution.finite-action",
    contribution_version="1.0.0",
    kind=ExtensionContributionKind.METHOD,
    evidence_profile_registries=(build_observation_evidence_world_registry(),),
    capability_manifests=(FINITE_ACTION_MANIFEST,),
    candidate_capability_registrations=(FINITE_ACTION_CANDIDATE_REGISTRATION,),
    dataset_capability_registries=(),
    profile_registrations=(LAW_CONTROL_OBJECTIVE_PROFILE_REGISTRATION,),
    method_registrations=(
        ObjectIdentity.from_record(
            FINITE_ACTION_METHOD.registration_id,
            FINITE_ACTION_METHOD,
        ),
    ),
    config_decoders=(
        ObjectIdentity.from_record(
            FINITE_ACTION_CONFIG_DECODER.registration_id,
            FINITE_ACTION_CONFIG_DECODER,
        ),
    ),
    artifact_validators=(
        ObjectIdentity.from_record(
            FINITE_ACTION_ARTIFACT_VALIDATOR.registration_id,
            FINITE_ACTION_ARTIFACT_VALIDATOR,
        ),
    ),
    runtime_providers=(),
    study_authors=(),
    grants_authority=False,
    embeds_scientific_payload=False,
)


__all__ = ["EXTENSION_BUNDLE_CONTRIBUTION"]
