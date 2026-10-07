"""Import-safe generated discovery for parameterised FreeGSNKE execution."""

from __future__ import annotations

import hashlib

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.runtime.capabilities import (
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
)
from empirical_lawhood.runtime.extension_bundles import ExtensionBundleContribution, ExtensionComponentKind, ExtensionComponentRegistration, ExtensionContributionKind
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope

from .parameterised_provider import FREEGSNKE_PARAMETERISED_PROJECTION_KEY, FREEGSNKE_PARAMETERISED_PROVIDER_KEY, FREEGSNKE_PARAMETERISED_VERSION, FreeGsnkeParameterisedEpisodeResult, FreeGsnkeParameterisedProjectionConfig, FreeGsnkeParameterisedProjection, FreeGsnkeParameterisedProviderConfig


FREEGSNKE_PARAMETERISED_IMPLEMENTATION_SHA256 = hashlib.sha256(
    b"freegsnke-neutral-parameterised-provider"
).hexdigest()


def _capability(
    *,
    key: str,
    kind: CapabilityKind,
    config_schema: str,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    deterministic: bool,
) -> CapabilityManifest:
    return CapabilityManifest(
        capability_key=key,
        capability_version=FREEGSNKE_PARAMETERISED_VERSION,
        kind=kind,
        config_schema=config_schema,
        config_schema_sha256=hashlib.sha256(config_schema.encode()).hexdigest(),
        input_schema_ids=tuple(sorted(inputs)),
        output_schema_ids=tuple(sorted(outputs)),
        permissions=(
            CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
            CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
        ),
        maximum_evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
        maximum_outcome_access=OutcomeAccess.EVALUATION_SEALED,
        resource_ceiling=ResourceBudget(
            8,
            64 * 1024**3,
            0,
            14_400,
            16 * 1024**3,
            2 * 1024**3,
        ),
        deterministic=deterministic,
        seed_required=False,
        language_id="python",
        runtime_id="freegsnke-parameterised",
        requires_clean_commit=True,
        requires_active_mount=True,
        requires_network=False,
        conformance_check_ids=tuple(
            sorted(
                (
                    "many-views-one-episode",
                    "numerical-failure-preserved",
                    "one-execution-per-acquisition-group",
                    "plural-native-clocks-and-receivers",
                    "requested-accepted-applied-realized-distinct",
                )
            )
        ),
        implementation_sha256=FREEGSNKE_PARAMETERISED_IMPLEMENTATION_SHA256,
    )


FREEGSNKE_PARAMETERISED_ACQUISITION_CAPABILITY = _capability(
    key=FREEGSNKE_PARAMETERISED_PROVIDER_KEY,
    kind=CapabilityKind.SIMULATOR,
    config_schema=FreeGsnkeParameterisedProviderConfig.SCHEMA,
    inputs=(FreeGsnkeParameterisedProviderConfig.SCHEMA,),
    outputs=(FreeGsnkeParameterisedEpisodeResult.SCHEMA, LinkedCampaignStageEnvelope.SCHEMA),
    deterministic=False,
)
FREEGSNKE_PARAMETERISED_PROJECTION_CAPABILITY = _capability(
    key=FREEGSNKE_PARAMETERISED_PROJECTION_KEY,
    kind=CapabilityKind.TRANSFORM,
    config_schema=FreeGsnkeParameterisedProjectionConfig.SCHEMA,
    inputs=(FreeGsnkeParameterisedEpisodeResult.SCHEMA, LinkedCampaignStageEnvelope.SCHEMA),
    outputs=(FreeGsnkeParameterisedProjection.SCHEMA, LinkedCampaignStageEnvelope.SCHEMA),
    deterministic=True,
)


def _decoder(label: str, schema: str) -> ExtensionComponentRegistration:
    return ExtensionComponentRegistration(
        registration_id=f"component.freegsnke-{label}-decoder",
        component_key=f"freegsnke-{label}-decoder",
        component_version="1.0.0",
        kind=ExtensionComponentKind.CONFIG_DECODER,
        input_schema_ids=(schema,),
        output_schema_ids=(schema,),
        implementation_sha256=FREEGSNKE_PARAMETERISED_IMPLEMENTATION_SHA256,
    )


FREEGSNKE_PARAMETERISED_CONFIG_DECODER = _decoder(
    "parameterised-provider-config",
    FreeGsnkeParameterisedProviderConfig.SCHEMA,
)
FREEGSNKE_PARAMETERISED_PROJECTION_CONFIG_DECODER = _decoder(
    "parameterised-projection-config",
    FreeGsnkeParameterisedProjectionConfig.SCHEMA,
)


def _provider(label: str, manifest: CapabilityManifest) -> ExtensionComponentRegistration:
    return ExtensionComponentRegistration(
        registration_id=f"component.freegsnke-{label}-provider",
        component_key=f"freegsnke-{label}-provider",
        component_version="1.0.0",
        kind=ExtensionComponentKind.RUNTIME_PROVIDER,
        input_schema_ids=manifest.input_schema_ids,
        output_schema_ids=manifest.output_schema_ids,
        implementation_sha256=FREEGSNKE_PARAMETERISED_IMPLEMENTATION_SHA256,
    )


FREEGSNKE_PARAMETERISED_ACQUISITION_PROVIDER = _provider(
    "parameterised-acquisition",
    FREEGSNKE_PARAMETERISED_ACQUISITION_CAPABILITY,
)
FREEGSNKE_PARAMETERISED_PROJECTION_PROVIDER = _provider(
    "parameterised-projection",
    FREEGSNKE_PARAMETERISED_PROJECTION_CAPABILITY,
)


EXTENSION_BUNDLE_CONTRIBUTION = ExtensionBundleContribution(
    contribution_id="extension-contribution.freegsnke-parameterised",
    contribution_version="1.0.0",
    kind=ExtensionContributionKind.SOURCE,
    evidence_profile_registries=(),
    capability_manifests=tuple(
        sorted(
            (
                FREEGSNKE_PARAMETERISED_ACQUISITION_CAPABILITY,
                FREEGSNKE_PARAMETERISED_PROJECTION_CAPABILITY,
            ),
            key=lambda value: value.registry_id,
        )
    ),
    candidate_capability_registrations=(),
    dataset_capability_registries=(),
    profile_registrations=(),
    method_registrations=(),
    config_decoders=tuple(
        sorted(
            (
                ObjectIdentity.from_record(value.registration_id, value)
                for value in (
                    FREEGSNKE_PARAMETERISED_CONFIG_DECODER,
                    FREEGSNKE_PARAMETERISED_PROJECTION_CONFIG_DECODER,
                )
            ),
            key=lambda value: value.object_id,
        )
    ),
    artifact_validators=(),
    runtime_providers=tuple(
        sorted(
            (
                ObjectIdentity.from_record(value.registration_id, value)
                for value in (
                    FREEGSNKE_PARAMETERISED_ACQUISITION_PROVIDER,
                    FREEGSNKE_PARAMETERISED_PROJECTION_PROVIDER,
                )
            ),
            key=lambda value: value.object_id,
        )
    ),
    study_authors=(),
    grants_authority=False,
    embeds_scientific_payload=False,
)


__all__ = [
    "EXTENSION_BUNDLE_CONTRIBUTION",
    "FREEGSNKE_PARAMETERISED_ACQUISITION_CAPABILITY",
    "FREEGSNKE_PARAMETERISED_ACQUISITION_PROVIDER",
    "FREEGSNKE_PARAMETERISED_CONFIG_DECODER",
    "FREEGSNKE_PARAMETERISED_IMPLEMENTATION_SHA256",
    "FREEGSNKE_PARAMETERISED_PROJECTION_CAPABILITY",
    "FREEGSNKE_PARAMETERISED_PROJECTION_CONFIG_DECODER",
    "FREEGSNKE_PARAMETERISED_PROJECTION_PROVIDER",
]
