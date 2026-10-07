"""Closed static capability profile for the pinned Glenn dataset adapter.

This module composes canonical registry metadata only.  It never opens held
source bytes, discovers the live Python environment, hashes repository files,
or manufactures process-local dataset implementations.  Runtime and
environment artifact identities, the exact config reference, and every
implementation digest are explicit inputs supplied by a trusted composition
root.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Final

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256
from empirical_lawhood.planning.dataset_manifests import (
    DatasetCapabilityBinding,
    DatasetCapabilityKind,
    DatasetRegistrationManifest,
    DatasetTransformationManifest,
)
from empirical_lawhood.planning.datasets import (
    CurrentMaterializationVerification,
    DatasetMaterialization,
    DatasetMaterializationVerificationReceipt,
)
from empirical_lawhood.runtime.capabilities import (
    CapabilityConfigRef,
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
    CapabilityRegistry,
)
from empirical_lawhood.runtime.datasets import (
    DatasetCapabilityLimits,
    DatasetCapabilityRegistry,
    DatasetEvidenceVerification,
    DatasetEvidenceVerifierRegistration,
    DatasetFormatInspectorRegistration,
    DatasetImplementationRegistry,
    DatasetSupportingIdentityRegistry,
    DatasetTransformRegistration,
    validate_dataset_capability_bindings,
)

from .contracts import GLENN_2026_PROFILE
from .manifests import GlennRetainedComparisonInputs, GLENN_ARCHIVE_FULL_SCAN_READ_LIMIT_BYTES, GLENN_ARCHIVE_SIZE_BYTES, GLENN_HISTORICAL_OBSERVATIONS_SIZE_BYTES, GLENN_READ_SET_SIZE_BYTES, GlennRegistrationManifestDependencies, GlennTransformationManifestDependencies
from .transform import GLENN_OUTPUT_SCHEMA_ID, GLENN_PARQUET_PROFILE_ID


GLENN_CONFIG_REGISTRY_KEY: Final = "dataset.config.glenn"
GLENN_ENVIRONMENT_REGISTRY_KEY: Final = "dataset.environment.glenn"
GLENN_EVIDENCE_VERIFIER_REGISTRY_KEY: Final = "dataset.evidence-verifier.glenn"
GLENN_INSPECTOR_REGISTRY_KEY: Final = "dataset.inspector.glenn"
GLENN_RUNTIME_REGISTRY_KEY: Final = "dataset.runtime.glenn"
GLENN_TRANSFORM_REGISTRY_KEY: Final = "dataset.transform.glenn"

GLENN_CAPABILITY_REGISTRY_ID: Final = "capabilities.dataset.glenn"
GLENN_DATASET_CAPABILITY_REGISTRY_ID: Final = "dataset-capabilities.glenn"
GLENN_SUPPORTING_IDENTITY_REGISTRY_ID: Final = "dataset-support.glenn"
GLENN_CAPABILITY_VERSION: Final = "1.0.0"

GLENN_ARCHIVE_FORMAT_PROFILE_ID: Final = "zip.safe-member-inventory"
GLENN_ARCHIVE_INSPECTION_PROFILE_ID: Final = GLENN_2026_PROFILE.profile_id
GLENN_MAXIMUM_OUTPUT_BYTES: Final = 64 * 1024**2
GLENN_MAXIMUM_METADATA_BYTES: Final = 1024**2
GLENN_MAXIMUM_EVIDENCE_RECORD_BYTES: Final = 1024**2
GLENN_MAXIMUM_EVIDENCE_BYTES: Final = 2 * GLENN_MAXIMUM_EVIDENCE_RECORD_BYTES
GLENN_TRANSFORM_INPUT_BYTES: Final = (
    GLENN_ARCHIVE_SIZE_BYTES
    + GLENN_READ_SET_SIZE_BYTES
    + GLENN_HISTORICAL_OBSERVATIONS_SIZE_BYTES
)
GLENN_TRANSFORM_SOURCE_SCAN_BYTES: Final = (
    GLENN_ARCHIVE_FULL_SCAN_READ_LIMIT_BYTES
    + GLENN_READ_SET_SIZE_BYTES
    + GLENN_HISTORICAL_OBSERVATIONS_SIZE_BYTES
)
GLENN_EXTERNAL_EVIDENCE_SCHEMA_IDS: Final = tuple(
    sorted(
        (
            DatasetMaterializationVerificationReceipt.SCHEMA,
            DatasetRegistrationManifest.SCHEMA,
            DatasetTransformationManifest.SCHEMA,
        )
    )
)

_EVIDENCE_VERIFICATION_OUTPUTS: Final = (DatasetEvidenceVerification.SCHEMA,)
_INSPECTOR_INPUT_SCHEMAS: Final = tuple(
    sorted((DatasetMaterialization.SCHEMA, DatasetRegistrationManifest.SCHEMA))
)
_INSPECTOR_OUTPUT_SCHEMAS: Final = tuple(
    sorted((CurrentMaterializationVerification.SCHEMA, DatasetMaterialization.SCHEMA))
)
_TRANSFORM_INPUT_SCHEMAS: Final = (DatasetTransformationManifest.SCHEMA,)
_TRANSFORM_OUTPUT_SCHEMAS: Final = tuple(
    sorted((DatasetMaterialization.SCHEMA, GLENN_OUTPUT_SCHEMA_ID))
)


def _validate_profile_inputs(
    *,
    config: CapabilityConfigRef,
    runtime: ObjectIdentity,
    environment: ObjectIdentity,
    inspector_implementation_sha256: str,
    transform_implementation_sha256: str,
    evidence_verifier_implementation_sha256: str,
) -> None:
    if not isinstance(config, CapabilityConfigRef):
        raise ValueError("config must be an exact CapabilityConfigRef")
    if config.config_id != GLENN_CONFIG_REGISTRY_KEY:
        raise ValueError("config has the wrong Glenn registry key")
    for identity, field_name, expected_id in (
        (runtime, "runtime", GLENN_RUNTIME_REGISTRY_KEY),
        (environment, "environment", GLENN_ENVIRONMENT_REGISTRY_KEY),
    ):
        if not isinstance(identity, ObjectIdentity):
            raise ValueError(f"{field_name} must be an exact ObjectIdentity")
        if identity.object_id != expected_id:
            raise ValueError(f"{field_name} has the wrong Glenn registry key")
        if identity.object_schema != ArtifactIdentity.SCHEMA:
            raise ValueError(f"{field_name} must identify an ArtifactIdentity")
        if identity.object_version != ArtifactIdentity.VERSION:
            raise ValueError(f"{field_name} has the wrong ArtifactIdentity version")
    for field_name, value in (
        ("inspector_implementation_sha256", inspector_implementation_sha256),
        ("transform_implementation_sha256", transform_implementation_sha256),
        (
            "evidence_verifier_implementation_sha256",
            evidence_verifier_implementation_sha256,
        ),
    ):
        validate_sha256(value, field_name=field_name)


def _archive_resource_ceiling(
    *,
    source_scan_bytes: int,
    output_bytes: int,
) -> ResourceBudget:
    return ResourceBudget(
        cpu_cores=4,
        memory_bytes=4 * 1024**3,
        gpu_devices=0,
        wall_time_seconds=3600,
        source_scan_bytes=source_scan_bytes,
        output_bytes=output_bytes,
    )


def _inspector_limits() -> DatasetCapabilityLimits:
    return DatasetCapabilityLimits(
        max_input_bytes=GLENN_ARCHIVE_SIZE_BYTES,
        max_output_bytes=GLENN_MAXIMUM_METADATA_BYTES,
        max_records=GLENN_2026_PROFILE.limits.maximum_members,
        max_files=1,
        max_archive_members=GLENN_2026_PROFILE.limits.maximum_members,
    )


def _transform_limits() -> DatasetCapabilityLimits:
    return DatasetCapabilityLimits(
        max_input_bytes=GLENN_TRANSFORM_INPUT_BYTES,
        max_output_bytes=GLENN_MAXIMUM_OUTPUT_BYTES,
        max_records=GLENN_2026_PROFILE.expected_rows,
        max_files=3,
        max_archive_members=GLENN_2026_PROFILE.limits.maximum_members,
    )


def _evidence_limits() -> DatasetCapabilityLimits:
    return DatasetCapabilityLimits(
        max_input_bytes=GLENN_MAXIMUM_EVIDENCE_BYTES,
        max_output_bytes=2 * GLENN_MAXIMUM_METADATA_BYTES,
        max_records=2,
        max_files=2,
        max_archive_members=0,
    )


def _manifest(
    *,
    capability_key: str,
    kind: CapabilityKind,
    config: CapabilityConfigRef,
    runtime: ObjectIdentity,
    implementation_sha256: str,
    input_schema_ids: tuple[str, ...],
    output_schema_ids: tuple[str, ...],
    permissions: tuple[CapabilityPermission, ...],
    maximum_outcome_access: OutcomeAccess,
    resources: ResourceBudget,
    conformance_check_ids: tuple[str, ...],
) -> CapabilityManifest:
    return CapabilityManifest(
        capability_key=capability_key,
        capability_version=GLENN_CAPABILITY_VERSION,
        kind=kind,
        config_schema=config.config_schema,
        config_schema_sha256=config.config_schema_sha256,
        input_schema_ids=input_schema_ids,
        output_schema_ids=output_schema_ids,
        permissions=permissions,
        maximum_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        maximum_outcome_access=maximum_outcome_access,
        resource_ceiling=resources,
        deterministic=True,
        seed_required=False,
        language_id="python",
        runtime_id=runtime.object_id,
        requires_clean_commit=True,
        requires_active_mount=True,
        requires_network=False,
        conformance_check_ids=conformance_check_ids,
        implementation_sha256=implementation_sha256,
    )


def _build_registries(
    *,
    config: CapabilityConfigRef,
    runtime: ObjectIdentity,
    environment: ObjectIdentity,
    inspector_implementation_sha256: str,
    transform_implementation_sha256: str,
    evidence_verifier_implementation_sha256: str,
) -> tuple[
    CapabilityRegistry, DatasetCapabilityRegistry, DatasetSupportingIdentityRegistry
]:
    inspector = _manifest(
        capability_key=GLENN_INSPECTOR_REGISTRY_KEY,
        kind=CapabilityKind.SOURCE,
        config=config,
        runtime=runtime,
        implementation_sha256=inspector_implementation_sha256,
        input_schema_ids=_INSPECTOR_INPUT_SCHEMAS,
        output_schema_ids=_INSPECTOR_OUTPUT_SCHEMAS,
        permissions=(CapabilityPermission.READ_EXTERNAL_ARTIFACTS,),
        maximum_outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        resources=_archive_resource_ceiling(
            source_scan_bytes=GLENN_ARCHIVE_FULL_SCAN_READ_LIMIT_BYTES,
            output_bytes=GLENN_MAXIMUM_METADATA_BYTES,
        ),
        conformance_check_ids=(
            "glenn-archive-contract",
            "glenn-safe-member-selection",
        ),
    )
    transform = _manifest(
        capability_key=GLENN_TRANSFORM_REGISTRY_KEY,
        kind=CapabilityKind.TRANSFORM,
        config=config,
        runtime=runtime,
        implementation_sha256=transform_implementation_sha256,
        input_schema_ids=_TRANSFORM_INPUT_SCHEMAS,
        output_schema_ids=_TRANSFORM_OUTPUT_SCHEMAS,
        permissions=(
            CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
            CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
        ),
        maximum_outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        resources=_archive_resource_ceiling(
            source_scan_bytes=GLENN_TRANSFORM_SOURCE_SCAN_BYTES,
            output_bytes=GLENN_MAXIMUM_OUTPUT_BYTES,
        ),
        conformance_check_ids=(
            "glenn-causal-field-contract",
            "glenn-deterministic-parquet",
            "glenn-safe-member-selection",
        ),
    )
    evidence_verifier = _manifest(
        capability_key=GLENN_EVIDENCE_VERIFIER_REGISTRY_KEY,
        kind=CapabilityKind.SOURCE,
        config=config,
        runtime=runtime,
        implementation_sha256=evidence_verifier_implementation_sha256,
        input_schema_ids=GLENN_EXTERNAL_EVIDENCE_SCHEMA_IDS,
        output_schema_ids=_EVIDENCE_VERIFICATION_OUTPUTS,
        permissions=(CapabilityPermission.READ_EXTERNAL_ARTIFACTS,),
        maximum_outcome_access=OutcomeAccess.OUTCOME_BLIND,
        resources=ResourceBudget(
            cpu_cores=1,
            memory_bytes=16 * 1024**2,
            gpu_devices=0,
            wall_time_seconds=60,
            source_scan_bytes=2 * GLENN_MAXIMUM_EVIDENCE_BYTES,
            output_bytes=2 * GLENN_MAXIMUM_METADATA_BYTES,
        ),
        conformance_check_ids=(
            "dataset-evidence-guard",
            "dataset-evidence-paired-record-byte-cap",
        ),
    )
    capability_registry = CapabilityRegistry(
        registry_id=GLENN_CAPABILITY_REGISTRY_ID,
        capabilities=tuple(
            sorted(
                (inspector, transform, evidence_verifier),
                key=lambda value: value.registry_id,
            )
        ),
    )
    dataset_registry = DatasetCapabilityRegistry(
        registry_id=GLENN_DATASET_CAPABILITY_REGISTRY_ID,
        providers=(),
        inspectors=(
            DatasetFormatInspectorRegistration(
                capability=inspector,
                accepted_media_types=("application/zip",),
                supported_format_profile_ids=(GLENN_ARCHIVE_FORMAT_PROFILE_ID,),
                limits=_inspector_limits(),
            ),
        ),
        transforms=(
            DatasetTransformRegistration(
                capability=transform,
                accepted_media_types=("application/zip",),
                produced_media_types=("application/vnd.apache.parquet",),
                source_format_profile_ids=(GLENN_ARCHIVE_FORMAT_PROFILE_ID,),
                destination_format_profile_ids=(GLENN_PARQUET_PROFILE_ID,),
                limits=_transform_limits(),
            ),
        ),
        binding_compilers=(),
        storage_verifiers=(),
        evidence_verifiers=(
            DatasetEvidenceVerifierRegistration(
                registry_id=GLENN_EVIDENCE_VERIFIER_REGISTRY_KEY,
                capability=evidence_verifier,
                supported_evidence_schema_ids=GLENN_EXTERNAL_EVIDENCE_SCHEMA_IDS,
                limits=_evidence_limits(),
            ),
        ),
    )
    supporting_registry = DatasetSupportingIdentityRegistry(
        registry_id=GLENN_SUPPORTING_IDENTITY_REGISTRY_ID,
        configs=(ObjectIdentity.from_record(GLENN_CONFIG_REGISTRY_KEY, config),),
        runtimes=(runtime,),
        environments=(environment,),
    )
    return capability_registry, dataset_registry, supporting_registry


@dataclass(frozen=True, slots=True)
class GlennDatasetRegistryProfile(CanonicalRecord):
    """One exact, canonical static Glenn registry closure."""

    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/physical/glenn/glenn-dataset-registry-profile'
    )

    config: CapabilityConfigRef
    runtime: ObjectIdentity
    environment: ObjectIdentity
    inspector_implementation_sha256: str
    transform_implementation_sha256: str
    evidence_verifier_implementation_sha256: str
    capability_registry: CapabilityRegistry
    dataset_registry: DatasetCapabilityRegistry
    supporting_registry: DatasetSupportingIdentityRegistry

    def __post_init__(self) -> None:
        _validate_profile_inputs(
            config=self.config,
            runtime=self.runtime,
            environment=self.environment,
            inspector_implementation_sha256=self.inspector_implementation_sha256,
            transform_implementation_sha256=self.transform_implementation_sha256,
            evidence_verifier_implementation_sha256=(
                self.evidence_verifier_implementation_sha256
            ),
        )
        expected_capability, expected_dataset, expected_supporting = _build_registries(
            config=self.config,
            runtime=self.runtime,
            environment=self.environment,
            inspector_implementation_sha256=self.inspector_implementation_sha256,
            transform_implementation_sha256=self.transform_implementation_sha256,
            evidence_verifier_implementation_sha256=(
                self.evidence_verifier_implementation_sha256
            ),
        )
        if self.capability_registry != expected_capability:
            raise ValueError("capability_registry differs from the exact Glenn closure")
        if self.dataset_registry != expected_dataset:
            raise ValueError("dataset_registry differs from the exact Glenn closure")
        if self.supporting_registry != expected_supporting:
            raise ValueError("supporting_registry differs from the exact Glenn closure")


def build_glenn_dataset_registry_profile(
    *,
    config: CapabilityConfigRef,
    runtime: ObjectIdentity,
    environment: ObjectIdentity,
    inspector_implementation_sha256: str,
    transform_implementation_sha256: str,
    evidence_verifier_implementation_sha256: str,
) -> GlennDatasetRegistryProfile:
    """Build the static profile only from explicit immutable identities."""

    _validate_profile_inputs(
        config=config,
        runtime=runtime,
        environment=environment,
        inspector_implementation_sha256=inspector_implementation_sha256,
        transform_implementation_sha256=transform_implementation_sha256,
        evidence_verifier_implementation_sha256=evidence_verifier_implementation_sha256,
    )
    capability_registry, dataset_registry, supporting_registry = _build_registries(
        config=config,
        runtime=runtime,
        environment=environment,
        inspector_implementation_sha256=inspector_implementation_sha256,
        transform_implementation_sha256=transform_implementation_sha256,
        evidence_verifier_implementation_sha256=evidence_verifier_implementation_sha256,
    )
    return GlennDatasetRegistryProfile(
        config=config,
        runtime=runtime,
        environment=environment,
        inspector_implementation_sha256=inspector_implementation_sha256,
        transform_implementation_sha256=transform_implementation_sha256,
        evidence_verifier_implementation_sha256=evidence_verifier_implementation_sha256,
        capability_registry=capability_registry,
        dataset_registry=dataset_registry,
        supporting_registry=supporting_registry,
    )


def glenn_dataset_capability_bindings(
    profile: GlennDatasetRegistryProfile,
    *,
    include_transform: bool,
    implementation_registry: DatasetImplementationRegistry | None = None,
) -> tuple[DatasetCapabilityBinding, ...]:
    """Derive exact sorted authoring bindings and optionally replay implementations."""

    if not isinstance(profile, GlennDatasetRegistryProfile):
        raise TypeError("profile must be a GlennDatasetRegistryProfile")
    if not isinstance(include_transform, bool):
        raise ValueError("include_transform must be boolean")
    inspector = profile.capability_registry.resolve(
        GLENN_INSPECTOR_REGISTRY_KEY,
        GLENN_CAPABILITY_VERSION,
    )
    evidence_verifier = profile.dataset_registry.evidence_verifier(
        GLENN_EVIDENCE_VERIFIER_REGISTRY_KEY
    )
    values = [
        DatasetCapabilityBinding(
            binding_id="capability.config",
            kind=DatasetCapabilityKind.CONFIG,
            registry_key=GLENN_CONFIG_REGISTRY_KEY,
            implementation=profile.supporting_registry.resolve(
                DatasetCapabilityKind.CONFIG,
                GLENN_CONFIG_REGISTRY_KEY,
            ),
        ),
        DatasetCapabilityBinding(
            binding_id="capability.environment",
            kind=DatasetCapabilityKind.ENVIRONMENT,
            registry_key=GLENN_ENVIRONMENT_REGISTRY_KEY,
            implementation=profile.supporting_registry.resolve(
                DatasetCapabilityKind.ENVIRONMENT,
                GLENN_ENVIRONMENT_REGISTRY_KEY,
            ),
        ),
        DatasetCapabilityBinding(
            binding_id="capability.evidence-verifier",
            kind=DatasetCapabilityKind.EVIDENCE_VERIFIER,
            registry_key=GLENN_EVIDENCE_VERIFIER_REGISTRY_KEY,
            implementation=ObjectIdentity.from_record(
                GLENN_EVIDENCE_VERIFIER_REGISTRY_KEY,
                evidence_verifier,
            ),
        ),
        DatasetCapabilityBinding(
            binding_id="capability.inspector",
            kind=DatasetCapabilityKind.INSPECTOR,
            registry_key=GLENN_INSPECTOR_REGISTRY_KEY,
            implementation=ObjectIdentity.from_record(
                GLENN_INSPECTOR_REGISTRY_KEY,
                inspector,
            ),
        ),
        DatasetCapabilityBinding(
            binding_id="capability.runtime",
            kind=DatasetCapabilityKind.RUNTIME,
            registry_key=GLENN_RUNTIME_REGISTRY_KEY,
            implementation=profile.supporting_registry.resolve(
                DatasetCapabilityKind.RUNTIME,
                GLENN_RUNTIME_REGISTRY_KEY,
            ),
        ),
    ]
    if include_transform:
        transform = profile.capability_registry.resolve(
            GLENN_TRANSFORM_REGISTRY_KEY,
            GLENN_CAPABILITY_VERSION,
        )
        values.append(
            DatasetCapabilityBinding(
                binding_id="capability.transform",
                kind=DatasetCapabilityKind.TRANSFORM,
                registry_key=GLENN_TRANSFORM_REGISTRY_KEY,
                implementation=ObjectIdentity.from_record(
                    GLENN_TRANSFORM_REGISTRY_KEY,
                    transform,
                ),
            )
        )
    bindings = tuple(sorted(values, key=lambda value: value.binding_id))
    if implementation_registry is not None:
        validate_dataset_capability_bindings(
            bindings,
            capability_registry=profile.capability_registry,
            dataset_registry=profile.dataset_registry,
            implementation_registry=implementation_registry,
            supporting_registry=profile.supporting_registry,
        )
    return bindings


def compose_glenn_registration_manifest_dependencies(
    profile: GlennDatasetRegistryProfile,
    *,
    source_storage_root: ObjectIdentity,
    control_storage_root: ObjectIdentity,
    complete_release_selector_sha256: str,
    implementation_registry: DatasetImplementationRegistry | None = None,
) -> GlennRegistrationManifestDependencies:
    """Compose the no-download registration manifest's exact dependencies."""

    return GlennRegistrationManifestDependencies(
        source_storage_root=source_storage_root,
        control_storage_root=control_storage_root,
        complete_release_selector_sha256=complete_release_selector_sha256,
        capabilities=glenn_dataset_capability_bindings(
            profile,
            include_transform=False,
            implementation_registry=implementation_registry,
        ),
    )


def compose_glenn_transformation_manifest_dependencies(
    profile: GlennDatasetRegistryProfile,
    *,
    historical_inputs: GlennRetainedComparisonInputs,
    source_storage_root: ObjectIdentity,
    selector_storage_root: ObjectIdentity,
    destination_storage_root: ObjectIdentity,
    historical_storage_root: ObjectIdentity,
    control_storage_root: ObjectIdentity,
    archive_materialization: ObjectIdentity,
    historical_observations_artifact: ObjectIdentity,
    historical_complete_selector_sha256: str,
    expected_output_physical_sha256: str | None = None,
    expected_output_byte_size: int | None = None,
    implementation_registry: DatasetImplementationRegistry | None = None,
) -> GlennTransformationManifestDependencies:
    """Compose the exact held-archive transformation manifest dependencies."""

    return GlennTransformationManifestDependencies(
        historical_inputs=historical_inputs,
        source_storage_root=source_storage_root,
        selector_storage_root=selector_storage_root,
        destination_storage_root=destination_storage_root,
        historical_storage_root=historical_storage_root,
        control_storage_root=control_storage_root,
        archive_materialization=archive_materialization,
        historical_observations_artifact=historical_observations_artifact,
        historical_complete_selector_sha256=historical_complete_selector_sha256,
        capabilities=glenn_dataset_capability_bindings(
            profile,
            include_transform=True,
            implementation_registry=implementation_registry,
        ),
        expected_output_physical_sha256=expected_output_physical_sha256,
        expected_output_byte_size=expected_output_byte_size,
    )


__all__ = [
    "GLENN_ARCHIVE_FORMAT_PROFILE_ID",
    "GLENN_ARCHIVE_INSPECTION_PROFILE_ID",
    "GLENN_CAPABILITY_REGISTRY_ID",
    "GLENN_CAPABILITY_VERSION",
    "GLENN_CONFIG_REGISTRY_KEY",
    "GLENN_DATASET_CAPABILITY_REGISTRY_ID",
    "GLENN_ENVIRONMENT_REGISTRY_KEY",
    "GLENN_EVIDENCE_VERIFIER_REGISTRY_KEY",
    "GLENN_EXTERNAL_EVIDENCE_SCHEMA_IDS",
    "GLENN_INSPECTOR_REGISTRY_KEY",
    "GLENN_MAXIMUM_EVIDENCE_BYTES",
    "GLENN_MAXIMUM_EVIDENCE_RECORD_BYTES",
    "GLENN_MAXIMUM_METADATA_BYTES",
    "GLENN_MAXIMUM_OUTPUT_BYTES",
    "GLENN_RUNTIME_REGISTRY_KEY",
    "GLENN_SUPPORTING_IDENTITY_REGISTRY_ID",
    "GLENN_TRANSFORM_REGISTRY_KEY",
    "GLENN_TRANSFORM_INPUT_BYTES",
    "GLENN_TRANSFORM_SOURCE_SCAN_BYTES",
    "GlennDatasetRegistryProfile",
    "build_glenn_dataset_registry_profile",
    "compose_glenn_registration_manifest_dependencies",
    "compose_glenn_transformation_manifest_dependencies",
    "glenn_dataset_capability_bindings",
]
