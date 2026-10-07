"""Static partial-morphism terminal declaration."""

from __future__ import annotations

import hashlib

from empirical_lawhood.adapters.methods.partial_morphism import PartialMorphismAssessment, PartialMorphismCellEvidence, partial_morphism_evaluator_registration
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.planning.contrasting_objectives import PartialMorphismSpec
from empirical_lawhood.planning.evidence_profiles import ExperimentObjectiveProfile
from empirical_lawhood.runtime.extension_bundles import ExtensionBundleContribution, ExtensionContributionKind, ExtensionProfileKind, ExtensionProfileRegistration


_IMPLEMENTATION_SHA256 = (
    "698017c84184b319a53c900bafddb68bc5e9af3944ba153d2a159fa505097311"
)
_PROFILE_IMPLEMENTATION_SHA256 = (
    "1838a6f708feee9554c0465610ccb7c1771bb37e77cc44ab1b5415a963966baa"
)


PARTIAL_MORPHISM_PROFILE_REGISTRATION = ExtensionProfileRegistration(
    registration_id="profile-registration.partial-morphism",
    profile_key="partial-morphism",
    profile_version="1.0.0",
    kind=ExtensionProfileKind.OBJECTIVE_TERMINAL,
    profile_schema=ExperimentObjectiveProfile.SCHEMA,
    profile_schema_sha256=hashlib.sha256(
        ExperimentObjectiveProfile.SCHEMA.encode()
    ).hexdigest(),
    implementation_sha256=_PROFILE_IMPLEMENTATION_SHA256,
)

PARTIAL_MORPHISM_EVALUATOR = partial_morphism_evaluator_registration(
    implementation_sha256=_IMPLEMENTATION_SHA256
)

EXTENSION_BUNDLE_CONTRIBUTION = ExtensionBundleContribution(
    contribution_id="extension-contribution.partial-morphism",
    contribution_version="1.0.0",
    kind=ExtensionContributionKind.METHOD,
    evidence_profile_registries=(),
    capability_manifests=(),
    candidate_capability_registrations=(),
    dataset_capability_registries=(),
    profile_registrations=(PARTIAL_MORPHISM_PROFILE_REGISTRATION,),
    method_registrations=(
        ObjectIdentity.from_record(
            PARTIAL_MORPHISM_EVALUATOR.registration_id,
            PARTIAL_MORPHISM_EVALUATOR,
        ),
    ),
    config_decoders=(),
    artifact_validators=(),
    runtime_providers=(),
    study_authors=(),
    grants_authority=False,
    embeds_scientific_payload=False,
)


assert PartialMorphismSpec.SCHEMA in PARTIAL_MORPHISM_EVALUATOR.input_schema_ids
assert (
    PartialMorphismCellEvidence.SCHEMA in PARTIAL_MORPHISM_EVALUATOR.input_schema_ids
)
assert PARTIAL_MORPHISM_EVALUATOR.output_schema_id == PartialMorphismAssessment.SCHEMA

__all__ = ["EXTENSION_BUNDLE_CONTRIBUTION"]
