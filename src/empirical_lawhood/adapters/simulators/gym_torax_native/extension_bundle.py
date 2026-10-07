"Import-light discovery for parameterised Gym--TORAX acquisition."

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

from .field_metadata_contracts import GymToraxFieldMetadataNativeEpisode
from .extraction_manifest import gym_torax_current_implementation_sha256
from .projection import GYM_TORAX_PARAMETERISED_PROJECTION_KEY, GYM_TORAX_PARAMETERISED_PROJECTION_VERSION, GymToraxNativeProjection, GymToraxParameterisedProjectionConfig
from .source import GYM_TORAX_PARAMETERISED_PROVIDER_KEY, GYM_TORAX_PARAMETERISED_PROVIDER_VERSION, GymToraxParameterisedAcquisitionManifest, GymToraxParameterisedSourceConfig


GYM_TORAX_PARAMETERISED_IMPLEMENTATION_SHA256 = gym_torax_current_implementation_sha256()
GYM_TORAX_PARAMETERISED_PROJECTION_IMPLEMENTATION_SHA256 = GYM_TORAX_PARAMETERISED_IMPLEMENTATION_SHA256

GYM_TORAX_PARAMETERISED_CAPABILITY = CapabilityManifest(
    capability_key=GYM_TORAX_PARAMETERISED_PROVIDER_KEY,
    capability_version=GYM_TORAX_PARAMETERISED_PROVIDER_VERSION,
    kind=CapabilityKind.SIMULATOR,
    config_schema=GymToraxParameterisedSourceConfig.SCHEMA,
    config_schema_sha256=hashlib.sha256(
        GymToraxParameterisedSourceConfig.SCHEMA.encode()
    ).hexdigest(),
    input_schema_ids=(GymToraxParameterisedAcquisitionManifest.SCHEMA,),
    output_schema_ids=tuple(
        sorted((GymToraxFieldMetadataNativeEpisode.SCHEMA, LinkedCampaignStageEnvelope.SCHEMA))
    ),
    permissions=(
        CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
        CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
    ),
    maximum_evidence_ceiling=EvidenceCeiling.MEASUREMENT,
    maximum_outcome_access=OutcomeAccess.EVALUATION_SEALED,
    resource_ceiling=ResourceBudget(
        cpu_cores=4,
        memory_bytes=16 * 1024**3,
        gpu_devices=0,
        wall_time_seconds=14_400,
        source_scan_bytes=2 * 1024**3,
        output_bytes=512 * 1024**2,
    ),
    deterministic=False,
    seed_required=True,
    language_id="python",
    runtime_id='gym-torax-native.field-metadata-acquisition',
    requires_clean_commit=True,
    requires_active_mount=False,
    requires_network=False,
    conformance_check_ids=(
        "exact-action-clock-and-field-metadata",
        "one-acquisition-per-group",
        "one-native-record-resident",
        "typed-partial-invalid-and-exception-causes",
    ),
    implementation_sha256=GYM_TORAX_PARAMETERISED_IMPLEMENTATION_SHA256,
)

GYM_TORAX_PARAMETERISED_PROJECTION_CAPABILITY = CapabilityManifest(
    capability_key=GYM_TORAX_PARAMETERISED_PROJECTION_KEY,
    capability_version=GYM_TORAX_PARAMETERISED_PROJECTION_VERSION,
    kind=CapabilityKind.TRANSFORM,
    config_schema=GymToraxParameterisedProjectionConfig.SCHEMA,
    config_schema_sha256=hashlib.sha256(
        GymToraxParameterisedProjectionConfig.SCHEMA.encode()
    ).hexdigest(),
    input_schema_ids=tuple(
        sorted((GymToraxFieldMetadataNativeEpisode.SCHEMA, LinkedCampaignStageEnvelope.SCHEMA))
    ),
    output_schema_ids=tuple(
        sorted((GymToraxNativeProjection.SCHEMA, LinkedCampaignStageEnvelope.SCHEMA))
    ),
    permissions=(),
    maximum_evidence_ceiling=EvidenceCeiling.MEASUREMENT,
    maximum_outcome_access=OutcomeAccess.EVALUATION_SEALED,
    resource_ceiling=ResourceBudget(
        cpu_cores=2,
        memory_bytes=2 * 1024**3,
        gpu_devices=0,
        wall_time_seconds=600,
        source_scan_bytes=0,
        output_bytes=512 * 1024**2,
    ),
    deterministic=True,
    seed_required=False,
    language_id="python",
    runtime_id='gym-torax-native.field-metadata-projection',
    requires_clean_commit=True,
    requires_active_mount=False,
    requires_network=False,
    conformance_check_ids=(
        "many-views-one-episode",
        "native-status-axes-preserved",
        "no-adapter-scientific-verdict",
        "one-view-no-reexecution",
    ),
    implementation_sha256=GYM_TORAX_PARAMETERISED_PROJECTION_IMPLEMENTATION_SHA256,
)

GYM_TORAX_PARAMETERISED_CONFIG_DECODER = ExtensionComponentRegistration(
    registration_id='component.gym-torax-parameterised-source-config-decoder',
    component_key="gym-torax-parameterised-source-config-decoder",
    component_version="1.0.0",
    kind=ExtensionComponentKind.CONFIG_DECODER,
    input_schema_ids=(GymToraxParameterisedSourceConfig.SCHEMA,),
    output_schema_ids=(GymToraxParameterisedSourceConfig.SCHEMA,),
    implementation_sha256=GYM_TORAX_PARAMETERISED_IMPLEMENTATION_SHA256,
)

GYM_TORAX_PARAMETERISED_ACQUISITION_MANIFEST_DECODER = ExtensionComponentRegistration(
    registration_id='component.gym-torax-parameterised-acquisition-manifest-decoder',
    component_key="gym-torax-parameterised-acquisition-manifest-decoder",
    component_version="1.0.0",
    kind=ExtensionComponentKind.CONFIG_DECODER,
    input_schema_ids=(GymToraxParameterisedAcquisitionManifest.SCHEMA,),
    output_schema_ids=(GymToraxParameterisedAcquisitionManifest.SCHEMA,),
    implementation_sha256=GYM_TORAX_PARAMETERISED_IMPLEMENTATION_SHA256,
)

GYM_TORAX_PARAMETERISED_PROJECTION_CONFIG_DECODER = ExtensionComponentRegistration(
    registration_id='component.gym-torax-parameterised-projection-config-decoder',
    component_key="gym-torax-parameterised-projection-config-decoder",
    component_version="1.0.0",
    kind=ExtensionComponentKind.CONFIG_DECODER,
    input_schema_ids=(GymToraxParameterisedProjectionConfig.SCHEMA,),
    output_schema_ids=(GymToraxParameterisedProjectionConfig.SCHEMA,),
    implementation_sha256=GYM_TORAX_PARAMETERISED_PROJECTION_IMPLEMENTATION_SHA256,
)

GYM_TORAX_PARAMETERISED_EPISODE_VALIDATOR = ExtensionComponentRegistration(
    registration_id='component.gym-torax-field-metadata-episode-validator',
    component_key='gym-torax-field-metadata-episode-validator',
    component_version="1.0.0",
    kind=ExtensionComponentKind.ARTIFACT_VALIDATOR,
    input_schema_ids=(GymToraxFieldMetadataNativeEpisode.SCHEMA,),
    output_schema_ids=('empirical-lawhood/runtime/artifact-generic-validation',),
    implementation_sha256=GYM_TORAX_PARAMETERISED_IMPLEMENTATION_SHA256,
)

GYM_TORAX_PARAMETERISED_RUNTIME_PROVIDER = ExtensionComponentRegistration(
    registration_id='component.gym-torax-parameterised-runtime-provider',
    component_key="gym-torax-parameterised-runtime-provider",
    component_version="1.0.0",
    kind=ExtensionComponentKind.RUNTIME_PROVIDER,
    input_schema_ids=GYM_TORAX_PARAMETERISED_CAPABILITY.input_schema_ids,
    output_schema_ids=GYM_TORAX_PARAMETERISED_CAPABILITY.output_schema_ids,
    implementation_sha256=GYM_TORAX_PARAMETERISED_IMPLEMENTATION_SHA256,
)

GYM_TORAX_PARAMETERISED_PROJECTION_RUNTIME_PROVIDER = ExtensionComponentRegistration(
    registration_id='component.gym-torax-parameterised-projection-provider',
    component_key="gym-torax-parameterised-projection-provider",
    component_version="1.0.0",
    kind=ExtensionComponentKind.RUNTIME_PROVIDER,
    input_schema_ids=GYM_TORAX_PARAMETERISED_PROJECTION_CAPABILITY.input_schema_ids,
    output_schema_ids=GYM_TORAX_PARAMETERISED_PROJECTION_CAPABILITY.output_schema_ids,
    implementation_sha256=GYM_TORAX_PARAMETERISED_PROJECTION_IMPLEMENTATION_SHA256,
)

EXTENSION_BUNDLE_CONTRIBUTION = ExtensionBundleContribution(
    contribution_id='extension-contribution.gym-torax-parameterised',
    contribution_version="1.0.0",
    kind=ExtensionContributionKind.SOURCE,
    evidence_profile_registries=(),
    capability_manifests=tuple(
        sorted(
            (
                GYM_TORAX_PARAMETERISED_CAPABILITY,
                GYM_TORAX_PARAMETERISED_PROJECTION_CAPABILITY,
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
                    GYM_TORAX_PARAMETERISED_ACQUISITION_MANIFEST_DECODER,
                    GYM_TORAX_PARAMETERISED_CONFIG_DECODER,
                    GYM_TORAX_PARAMETERISED_PROJECTION_CONFIG_DECODER,
                )
            ),
            key=lambda value: value.object_id,
        )
    ),
    artifact_validators=(
        ObjectIdentity.from_record(
            GYM_TORAX_PARAMETERISED_EPISODE_VALIDATOR.registration_id,
            GYM_TORAX_PARAMETERISED_EPISODE_VALIDATOR,
        ),
    ),
    runtime_providers=tuple(
        sorted(
            (
                ObjectIdentity.from_record(value.registration_id, value)
                for value in (
                    GYM_TORAX_PARAMETERISED_RUNTIME_PROVIDER,
                    GYM_TORAX_PARAMETERISED_PROJECTION_RUNTIME_PROVIDER,
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
    "GYM_TORAX_PARAMETERISED_CAPABILITY",
    "GYM_TORAX_PARAMETERISED_ACQUISITION_MANIFEST_DECODER",
    "GYM_TORAX_PARAMETERISED_CONFIG_DECODER",
    "GYM_TORAX_PARAMETERISED_EPISODE_VALIDATOR",
    "GYM_TORAX_PARAMETERISED_IMPLEMENTATION_SHA256",
    "GYM_TORAX_PARAMETERISED_PROJECTION_CAPABILITY",
    "GYM_TORAX_PARAMETERISED_PROJECTION_CONFIG_DECODER",
    "GYM_TORAX_PARAMETERISED_PROJECTION_IMPLEMENTATION_SHA256",
    "GYM_TORAX_PARAMETERISED_PROJECTION_RUNTIME_PROVIDER",
    "GYM_TORAX_PARAMETERISED_RUNTIME_PROVIDER",
]
