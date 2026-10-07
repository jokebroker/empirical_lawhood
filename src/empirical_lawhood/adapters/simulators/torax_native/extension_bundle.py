"""Static source-pipeline declaration for the native TORAX pilot seam."""

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
from empirical_lawhood.runtime.extension_bundles import ExtensionBundleContribution, ExtensionComponentKind, ExtensionComponentRegistration, ExtensionContributionKind, ExtensionProfileKind, ExtensionProfileRegistration
from empirical_lawhood.runtime.profile_compilation import SourceProfileCompilerBinding


_PLANNING_IMPLEMENTATION_SHA256 = (
    "7d91f9feacc1e490242e2a5f6d59d28fdcc3fd4b559ae2f8c5a1b5c2a463f565"
)
_RUNTIME_IMPLEMENTATION_SHA256 = (
    "21787c7629fd4d6a5ae8e42a268e7d38c17dfec36793cae3c7b2da538a36831e"
)
_CONFIG_SCHEMA = 'empirical-lawhood/simulators/torax-native/source-normalization'


SOURCE_PIPELINE_PROFILE_REGISTRATION = ExtensionProfileRegistration(
    registration_id="profile-registration.source-pipeline",
    profile_key="source-pipeline",
    profile_version="1.0.0",
    kind=ExtensionProfileKind.SOURCE_PIPELINE,
    profile_schema=SourcePipelineProfile.SCHEMA,
    profile_schema_sha256=hashlib.sha256(
        SourcePipelineProfile.SCHEMA.encode()
    ).hexdigest(),
    implementation_sha256=_PLANNING_IMPLEMENTATION_SHA256,
)

SOURCE_TRANSFORM_MANIFEST = CapabilityManifest(
    capability_key="source-pipeline.registered-transform",
    capability_version="1.0.0",
    kind=CapabilityKind.TRANSFORM,
    config_schema=_CONFIG_SCHEMA,
    config_schema_sha256=hashlib.sha256(_CONFIG_SCHEMA.encode()).hexdigest(),
    input_schema_ids=('empirical-lawhood/evidence/native-source',),
    output_schema_ids=('empirical-lawhood/evidence/canonical-source',),
    permissions=(
        CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
        CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
    ),
    maximum_evidence_ceiling=EvidenceCeiling.MEASUREMENT,
    maximum_outcome_access=OutcomeAccess.OUTCOME_BLIND,
    resource_ceiling=ResourceBudget(
        cpu_cores=4,
        memory_bytes=8 * 1024**3,
        gpu_devices=0,
        wall_time_seconds=3_600,
        source_scan_bytes=8 * 1024**3,
        output_bytes=8 * 1024**3,
    ),
    deterministic=True,
    seed_required=False,
    language_id="python",
    runtime_id="cpython-source-pipeline",
    requires_clean_commit=True,
    requires_active_mount=True,
    requires_network=False,
    conformance_check_ids=(
        "bounded-registered-edge-chain",
        "native-coordinate-preservation",
        "physical-unit-and-invalid-row-accounting",
        "receipt-first-recovery",
    ),
    implementation_sha256=_RUNTIME_IMPLEMENTATION_SHA256,
)

TORAX_SIMULATED_SOURCE_MANIFEST = CapabilityManifest(
    capability_key="torax-native.simulated-source",
    capability_version="1.0.0",
    kind=CapabilityKind.SOURCE,
    config_schema=_CONFIG_SCHEMA,
    config_schema_sha256=hashlib.sha256(_CONFIG_SCHEMA.encode()).hexdigest(),
    input_schema_ids=(),
    output_schema_ids=('empirical-lawhood/evidence/native-source',),
    permissions=(),
    maximum_evidence_ceiling=EvidenceCeiling.MEASUREMENT,
    maximum_outcome_access=OutcomeAccess.OUTCOME_BLIND,
    resource_ceiling=SOURCE_TRANSFORM_MANIFEST.resource_ceiling,
    deterministic=False,
    seed_required=True,
    language_id="python",
    runtime_id="torax-native-source",
    requires_clean_commit=True,
    requires_active_mount=False,
    requires_network=False,
    conformance_check_ids=(
        "adapter-owned-realization",
        "native-clock-unit-frame-preservation",
    ),
    implementation_sha256=_RUNTIME_IMPLEMENTATION_SHA256,
)

SOURCE_DATASET_CAPABILITY_REGISTRY = DatasetCapabilityRegistry(
    registry_id="dataset-registry.backbone-source-pipeline",
    providers=(),
    inspectors=(),
    transforms=(
        DatasetTransformRegistration(
            capability=SOURCE_TRANSFORM_MANIFEST,
            accepted_media_types=("application/octet-stream",),
            produced_media_types=("application/vnd.apache.parquet",),
            source_format_profile_ids=("native-source",),
            destination_format_profile_ids=('canonical-source',),
            limits=DatasetCapabilityLimits(
                max_input_bytes=8 * 1024**3,
                max_output_bytes=8 * 1024**3,
                max_records=10_000_000,
                max_files=10_000,
                max_archive_members=100_000,
            ),
        ),
    ),
    binding_compilers=(),
    storage_verifiers=(),
    evidence_verifiers=(),
)

SOURCE_CONFIG_DECODER = ExtensionComponentRegistration(
    registration_id="component.source-pipeline-config-decoder",
    component_key="source-pipeline-config-decoder",
    component_version="1.0.0",
    kind=ExtensionComponentKind.CONFIG_DECODER,
    input_schema_ids=(SourcePipelineProfile.SCHEMA,),
    output_schema_ids=(SourcePipelineProfile.SCHEMA,),
    implementation_sha256=_PLANNING_IMPLEMENTATION_SHA256,
)

SOURCE_ARTIFACT_VALIDATOR = ExtensionComponentRegistration(
    registration_id="component.source-pipeline-artifact-validator",
    component_key="source-pipeline-artifact-validator",
    component_version="1.0.0",
    kind=ExtensionComponentKind.ARTIFACT_VALIDATOR,
    input_schema_ids=('empirical-lawhood/evidence/canonical-source',),
    output_schema_ids=('empirical-lawhood/runtime/artifact-generic-validation',),
    implementation_sha256=_RUNTIME_IMPLEMENTATION_SHA256,
)

TORAX_SIMULATED_SOURCE_PROVIDER = ExtensionComponentRegistration(
    registration_id="component.torax-simulated-source-provider",
    component_key="torax-simulated-source-provider",
    component_version="1.0.0",
    kind=ExtensionComponentKind.RUNTIME_PROVIDER,
    input_schema_ids=(SourcePipelineProfile.SCHEMA,),
    output_schema_ids=TORAX_SIMULATED_SOURCE_MANIFEST.output_schema_ids,
    implementation_sha256=_RUNTIME_IMPLEMENTATION_SHA256,
)

TORAX_SOURCE_PROFILE_COMPILER = ExtensionComponentRegistration(
    registration_id="component.torax-source-profile-compiler",
    component_key="torax-source-profile-compiler",
    component_version="1.0.0",
    kind=ExtensionComponentKind.RUNTIME_PROVIDER,
    input_schema_ids=(),
    output_schema_ids=(SourceProfileCompilerBinding.SCHEMA,),
    implementation_sha256=_RUNTIME_IMPLEMENTATION_SHA256,
)

EXTENSION_BUNDLE_CONTRIBUTION = ExtensionBundleContribution(
    contribution_id="extension-contribution.torax-native-source",
    contribution_version="1.0.0",
    kind=ExtensionContributionKind.SOURCE,
    evidence_profile_registries=(),
    capability_manifests=(SOURCE_TRANSFORM_MANIFEST, TORAX_SIMULATED_SOURCE_MANIFEST),
    candidate_capability_registrations=(),
    dataset_capability_registries=(SOURCE_DATASET_CAPABILITY_REGISTRY,),
    profile_registrations=(SOURCE_PIPELINE_PROFILE_REGISTRATION,),
    method_registrations=(),
    config_decoders=(
        ObjectIdentity.from_record(
            SOURCE_CONFIG_DECODER.registration_id,
            SOURCE_CONFIG_DECODER,
        ),
    ),
    artifact_validators=(
        ObjectIdentity.from_record(
            SOURCE_ARTIFACT_VALIDATOR.registration_id,
            SOURCE_ARTIFACT_VALIDATOR,
        ),
    ),
    runtime_providers=tuple(
        ObjectIdentity.from_record(value.registration_id, value)
        for value in (TORAX_SIMULATED_SOURCE_PROVIDER, TORAX_SOURCE_PROFILE_COMPILER)
    ),
    study_authors=(),
    grants_authority=False,
    embeds_scientific_payload=False,
)


__all__ = ["EXTENSION_BUNDLE_CONTRIBUTION"]
