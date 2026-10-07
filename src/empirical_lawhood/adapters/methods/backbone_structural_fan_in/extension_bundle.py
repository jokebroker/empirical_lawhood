"""Static compact-handoff and structural fan-in declaration."""

from __future__ import annotations

import hashlib

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.planning.contrasting_objectives import CompactScientificHandoff, MetatheoryFanInAssessment, StructuralHandoffProfile
from empirical_lawhood.runtime.extension_bundles import ExtensionBundleContribution, ExtensionComponentKind, ExtensionComponentRegistration, ExtensionContributionKind, ExtensionProfileKind, ExtensionProfileRegistration


_IMPLEMENTATION_SHA256 = (
    "0e36ad1268eaece43a5125f56c8252fa5eeab66487c2d9c2034487f0e6a5fc33"
)


STRUCTURAL_PROFILE_REGISTRATION = ExtensionProfileRegistration(
    registration_id="profile-registration.structural-fan-in",
    profile_key="structural-fan-in",
    profile_version="1.0.0",
    kind=ExtensionProfileKind.OBJECTIVE_TERMINAL,
    profile_schema=StructuralHandoffProfile.SCHEMA,
    profile_schema_sha256=hashlib.sha256(
        StructuralHandoffProfile.SCHEMA.encode()
    ).hexdigest(),
    implementation_sha256=_IMPLEMENTATION_SHA256,
)

STRUCTURAL_FAN_IN_METHOD = ExtensionComponentRegistration(
    registration_id="component.structural-fan-in",
    component_key="structural-fan-in",
    component_version="1.0.0",
    kind=ExtensionComponentKind.METHOD,
    input_schema_ids=(CompactScientificHandoff.SCHEMA,),
    output_schema_ids=(MetatheoryFanInAssessment.SCHEMA,),
    implementation_sha256=_IMPLEMENTATION_SHA256,
)

EXTENSION_BUNDLE_CONTRIBUTION = ExtensionBundleContribution(
    contribution_id="extension-contribution.structural-fan-in",
    contribution_version="1.0.0",
    kind=ExtensionContributionKind.STRUCTURAL,
    evidence_profile_registries=(),
    capability_manifests=(),
    candidate_capability_registrations=(),
    dataset_capability_registries=(),
    profile_registrations=(STRUCTURAL_PROFILE_REGISTRATION,),
    method_registrations=(
        ObjectIdentity.from_record(
            STRUCTURAL_FAN_IN_METHOD.registration_id,
            STRUCTURAL_FAN_IN_METHOD,
        ),
    ),
    config_decoders=(),
    artifact_validators=(),
    runtime_providers=(),
    study_authors=(),
    grants_authority=False,
    embeds_scientific_payload=False,
)


__all__ = ["EXTENSION_BUNDLE_CONTRIBUTION"]
