"""Import-safe discovery for the MAST-U query-manifest source context."""

from __future__ import annotations

import hashlib

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.planning.source_pipelines import SourcePipelineProfile
from empirical_lawhood.runtime.capabilities import (
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
)
from empirical_lawhood.runtime.datasets import (
    DatasetCapabilityLimits,
    DatasetCapabilityRegistry,
    DatasetTransformRegistration,
)
from empirical_lawhood.runtime.extension_bundles import ExtensionBundleContribution, ExtensionComponentKind, ExtensionComponentRegistration, ExtensionContributionKind
from empirical_lawhood.runtime.profile_compilation import SourceProfileCompilerBinding
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope

from .parameterised_provider import MASTU_ARCHIVE_ACQUISITION_PROVIDER_KEY, MASTU_ARCHIVE_COMPACT_PROVIDER_KEY, MASTU_ARCHIVE_PROJECTION_PROVIDER_KEY, MASTU_ARCHIVE_PROVIDER_VERSION, MASTUArchiveCompactProviderConfig, MASTUArchiveParameterisedProviderConfig
from .archive_receiver_projection import MastArchiveReceiverProjectionConfig, MastArchiveReceiverScientificProjection
from .query_source import MASTUArchiveAcquisitionResult, MASTUArchiveCompactManifestResult, MASTUArchiveQueryManifest, MASTUArchiveQuerySourceConfig


_IMPLEMENTATION_SHA256 = hashlib.sha256(b"mastu-query-source-context").hexdigest()
_CANONICAL_OUTPUT_SCHEMA = 'empirical-lawhood/physical/mast-archive-response-qualification/mastu-archive-canonical-observations'
_TRANSFORM_CONFIG_SCHEMA = 'empirical-lawhood/physical/mast-archive-response-qualification/mastu-query-object-projection'
MASTU_PARAMETERISED_IMPLEMENTATION_SHA256 = hashlib.sha256(
    b"mastu-archive-issued-acquisition-projection-compact"
).hexdigest()


def _parameterised_capability(
    *,
    key: str,
    kind: CapabilityKind,
    config_schema: str,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    permissions: tuple[CapabilityPermission, ...],
    deterministic: bool,
) -> CapabilityManifest:
    return CapabilityManifest(
        capability_key=key,
        capability_version=MASTU_ARCHIVE_PROVIDER_VERSION,
        kind=kind,
        config_schema=config_schema,
        config_schema_sha256=hashlib.sha256(config_schema.encode()).hexdigest(),
        input_schema_ids=tuple(sorted(inputs)),
        output_schema_ids=tuple(sorted(outputs)),
        permissions=tuple(sorted(permissions)),
        maximum_evidence_ceiling=EvidenceCeiling.MEASUREMENT,
        maximum_outcome_access=OutcomeAccess.EVALUATION_SEALED,
        resource_ceiling=ResourceBudget(2, 4 * 1024**3, 0, 3600, 64 * 1024**3, 64 * 1024**3),
        deterministic=deterministic,
        seed_required=False,
        language_id="python",
        runtime_id="mastu-archive-parameterised",
        requires_clean_commit=True,
        requires_active_mount=True,
        requires_network=key == MASTU_ARCHIVE_ACQUISITION_PROVIDER_KEY,
        conformance_check_ids=tuple(
            sorted(
                (
                    "authority-before-contact",
                    "one-acquisition-per-group",
                    "custody-before-next-retrieval",
                    "many-views-one-object",
                    "no-reacquisition-recovery",
                )
            )
        ),
        implementation_sha256=MASTU_PARAMETERISED_IMPLEMENTATION_SHA256,
    )


MASTU_PARAMETERISED_ACQUISITION_CAPABILITY = _parameterised_capability(
    key=MASTU_ARCHIVE_ACQUISITION_PROVIDER_KEY,
    kind=CapabilityKind.SOURCE,
    config_schema=MASTUArchiveParameterisedProviderConfig.SCHEMA,
    inputs=(MASTUArchiveQueryManifest.SCHEMA,),
    outputs=(MASTUArchiveAcquisitionResult.SCHEMA, LinkedCampaignStageEnvelope.SCHEMA),
    permissions=(
        CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
        CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
    ),
    deterministic=False,
)
MASTU_PARAMETERISED_PROJECTION_CAPABILITY = _parameterised_capability(
    key=MASTU_ARCHIVE_PROJECTION_PROVIDER_KEY,
    kind=CapabilityKind.TRANSFORM,
    config_schema=MastArchiveReceiverProjectionConfig.SCHEMA,
    inputs=(MASTUArchiveAcquisitionResult.SCHEMA, LinkedCampaignStageEnvelope.SCHEMA),
    outputs=(MastArchiveReceiverScientificProjection.SCHEMA, LinkedCampaignStageEnvelope.SCHEMA),
    permissions=(
        CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
        CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
    ),
    deterministic=True,
)
MASTU_PARAMETERISED_COMPACT_CAPABILITY = _parameterised_capability(
    key=MASTU_ARCHIVE_COMPACT_PROVIDER_KEY,
    kind=CapabilityKind.TRANSFORM,
    config_schema=MASTUArchiveCompactProviderConfig.SCHEMA,
    inputs=(MASTUArchiveAcquisitionResult.SCHEMA, LinkedCampaignStageEnvelope.SCHEMA),
    outputs=(MASTUArchiveCompactManifestResult.SCHEMA, LinkedCampaignStageEnvelope.SCHEMA),
    permissions=(CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,),
    deterministic=True,
)

MASTU_QUERY_SOURCE_MANIFEST = CapabilityManifest(
    capability_key="mastu-archive.query-source",
    capability_version="1.0.0",
    kind=CapabilityKind.SOURCE,
    config_schema=MASTUArchiveQuerySourceConfig.SCHEMA,
    config_schema_sha256=hashlib.sha256(
        MASTUArchiveQuerySourceConfig.SCHEMA.encode()
    ).hexdigest(),
    input_schema_ids=(),
    output_schema_ids=(MASTUArchiveQueryManifest.SCHEMA,),
    permissions=(CapabilityPermission.READ_EXTERNAL_ARTIFACTS,),
    maximum_evidence_ceiling=EvidenceCeiling.MEASUREMENT,
    maximum_outcome_access=OutcomeAccess.OUTCOME_BLIND,
    resource_ceiling=ResourceBudget(2, 2 * 1024**3, 0, 3600, 64 * 1024**3, 64 * 1024**3),
    deterministic=False,
    seed_required=False,
    language_id="python",
    runtime_id="mastu-uda-query",
    requires_clean_commit=True,
    requires_active_mount=False,
    requires_network=True,
    conformance_check_ids=(
        "manifest-frozen-before-contact",
        "receipt-before-next-retrieval",
        "response-properties-learned-after-receipt",
    ),
    implementation_sha256=_IMPLEMENTATION_SHA256,
)

MASTU_QUERY_PROJECTION_MANIFEST = CapabilityManifest(
    capability_key="mastu-archive.query-object-projection",
    capability_version="1.0.0",
    kind=CapabilityKind.TRANSFORM,
    config_schema=_TRANSFORM_CONFIG_SCHEMA,
    config_schema_sha256=hashlib.sha256(_TRANSFORM_CONFIG_SCHEMA.encode()).hexdigest(),
    input_schema_ids=(MASTUArchiveQueryManifest.SCHEMA,),
    output_schema_ids=(_CANONICAL_OUTPUT_SCHEMA,),
    permissions=(
        CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
        CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
    ),
    maximum_evidence_ceiling=EvidenceCeiling.MEASUREMENT,
    maximum_outcome_access=OutcomeAccess.OUTCOME_BLIND,
    resource_ceiling=ResourceBudget(2, 2 * 1024**3, 0, 3600, 64 * 1024**3, 64 * 1024**3),
    deterministic=True,
    seed_required=False,
    language_id="python",
    runtime_id="mastu-query-projection",
    requires_clean_commit=True,
    requires_active_mount=True,
    requires_network=False,
    conformance_check_ids=(
        "native-unit-frame-clock-preservation",
        "physical-unit-and-view-lineage",
    ),
    implementation_sha256=_IMPLEMENTATION_SHA256,
)

MASTU_QUERY_DATASET_REGISTRY = DatasetCapabilityRegistry(
    registry_id="dataset-registry.mastu-query-source",
    providers=(),
    inspectors=(),
    transforms=(
        DatasetTransformRegistration(
            capability=MASTU_QUERY_PROJECTION_MANIFEST,
            accepted_media_types=("application/json",),
            produced_media_types=("application/vnd.apache.parquet",),
            source_format_profile_ids=("mastu-query-manifest",),
            destination_format_profile_ids=("mastu-canonical-observations",),
            limits=DatasetCapabilityLimits(
                max_input_bytes=64 * 1024**3,
                max_output_bytes=64 * 1024**3,
                max_records=10_000_000,
                max_files=100_000,
                max_archive_members=100_000,
            ),
        ),
    ),
    binding_compilers=(),
    storage_verifiers=(),
    evidence_verifiers=(),
)

MASTU_QUERY_CONFIG_DECODERS = tuple(
    ExtensionComponentRegistration(
        registration_id=f"component.mastu-query-{label}-decoder",
        component_key=f"mastu-query-{label}-decoder",
        component_version="1.0.0",
        kind=ExtensionComponentKind.CONFIG_DECODER,
        input_schema_ids=(record_type.SCHEMA,),
        output_schema_ids=(record_type.SCHEMA,),
        implementation_sha256=_IMPLEMENTATION_SHA256,
    )
    for label, record_type in (
        ("manifest", MASTUArchiveQueryManifest),
        ("source-config", MASTUArchiveQuerySourceConfig),
        ("parameterised-provider-config", MASTUArchiveParameterisedProviderConfig),
        ("compact-provider-config", MASTUArchiveCompactProviderConfig),
        ("pf-projection-config", MastArchiveReceiverProjectionConfig),
    )
)
MASTU_QUERY_SOURCE_PROVIDER = ExtensionComponentRegistration(
    registration_id="component.mastu-query-source-provider",
    component_key="mastu-query-source-provider",
    component_version="1.0.0",
    kind=ExtensionComponentKind.RUNTIME_PROVIDER,
    input_schema_ids=(MASTUArchiveQueryManifest.SCHEMA,),
    output_schema_ids=(MASTUArchiveQueryManifest.SCHEMA,),
    implementation_sha256=_IMPLEMENTATION_SHA256,
)
MASTU_QUERY_PROJECTION_SELECTION = ExtensionComponentRegistration(
    registration_id="component.mastu-query-projection-selection",
    component_key="mastu-query-projection-selection",
    component_version="1.0.0",
    kind=ExtensionComponentKind.RUNTIME_PROVIDER,
    input_schema_ids=(SourcePipelineProfile.SCHEMA,),
    output_schema_ids=MASTU_QUERY_PROJECTION_MANIFEST.output_schema_ids,
    implementation_sha256=_IMPLEMENTATION_SHA256,
)
MASTU_QUERY_PROFILE_COMPILER = ExtensionComponentRegistration(
    registration_id="component.mastu-query-profile-compiler",
    component_key="mastu-query-profile-compiler",
    component_version="1.0.0",
    kind=ExtensionComponentKind.RUNTIME_PROVIDER,
    input_schema_ids=(),
    output_schema_ids=(SourceProfileCompilerBinding.SCHEMA,),
    implementation_sha256=_IMPLEMENTATION_SHA256,
)


def _runtime_provider_component(
    label: str,
    manifest: CapabilityManifest,
) -> ExtensionComponentRegistration:
    return ExtensionComponentRegistration(
        registration_id=f"component.mastu-{label}-runtime-provider",
        component_key=f"mastu-{label}-runtime-provider",
        component_version="1.0.0",
        kind=ExtensionComponentKind.RUNTIME_PROVIDER,
        input_schema_ids=manifest.input_schema_ids,
        output_schema_ids=manifest.output_schema_ids,
        implementation_sha256=MASTU_PARAMETERISED_IMPLEMENTATION_SHA256,
    )


MASTU_PARAMETERISED_ACQUISITION_PROVIDER = _runtime_provider_component(
    "parameterised-acquisition",
    MASTU_PARAMETERISED_ACQUISITION_CAPABILITY,
)
MASTU_PARAMETERISED_PROJECTION_PROVIDER = _runtime_provider_component(
    "parameterised-projection",
    MASTU_PARAMETERISED_PROJECTION_CAPABILITY,
)
MASTU_PARAMETERISED_COMPACT_PROVIDER = _runtime_provider_component(
    "parameterised-compact",
    MASTU_PARAMETERISED_COMPACT_CAPABILITY,
)

EXTENSION_BUNDLE_CONTRIBUTION = ExtensionBundleContribution(
    contribution_id="extension-contribution.mastu-query-source",
    contribution_version="1.0.0",
    kind=ExtensionContributionKind.SOURCE,
    evidence_profile_registries=(),
    capability_manifests=tuple(
        sorted(
            (
                MASTU_PARAMETERISED_ACQUISITION_CAPABILITY,
                MASTU_PARAMETERISED_COMPACT_CAPABILITY,
                MASTU_PARAMETERISED_PROJECTION_CAPABILITY,
                MASTU_QUERY_PROJECTION_MANIFEST,
                MASTU_QUERY_SOURCE_MANIFEST,
            ),
            key=lambda value: value.registry_id,
        )
    ),
    candidate_capability_registrations=(),
    dataset_capability_registries=(MASTU_QUERY_DATASET_REGISTRY,),
    profile_registrations=(),
    method_registrations=(),
    config_decoders=tuple(
        sorted(
            (
                ObjectIdentity.from_record(value.registration_id, value)
                for value in MASTU_QUERY_CONFIG_DECODERS
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
                    MASTU_QUERY_PROFILE_COMPILER,
                    MASTU_QUERY_PROJECTION_SELECTION,
                    MASTU_QUERY_SOURCE_PROVIDER,
                    MASTU_PARAMETERISED_ACQUISITION_PROVIDER,
                    MASTU_PARAMETERISED_COMPACT_PROVIDER,
                    MASTU_PARAMETERISED_PROJECTION_PROVIDER,
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
    "MASTU_QUERY_CONFIG_DECODERS",
    "MASTU_QUERY_DATASET_REGISTRY",
    "MASTU_QUERY_PROFILE_COMPILER",
    "MASTU_QUERY_PROJECTION_SELECTION",
    "MASTU_QUERY_PROJECTION_MANIFEST",
    "MASTU_QUERY_SOURCE_MANIFEST",
    "MASTU_QUERY_SOURCE_PROVIDER",
    "MASTU_PARAMETERISED_ACQUISITION_CAPABILITY",
    "MASTU_PARAMETERISED_ACQUISITION_PROVIDER",
    "MASTU_PARAMETERISED_COMPACT_CAPABILITY",
    "MASTU_PARAMETERISED_COMPACT_PROVIDER",
    "MASTU_PARAMETERISED_IMPLEMENTATION_SHA256",
    "MASTU_PARAMETERISED_PROJECTION_CAPABILITY",
    "MASTU_PARAMETERISED_PROJECTION_PROVIDER",
]
