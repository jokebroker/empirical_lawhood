"""Static dataset capability closure shared by held simulator registrations."""

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
    DatasetDirectorySelectorManifest,
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
    validate_dataset_capability_bindings,
)


SIMULATOR_DATASET_CONFIG_KEY: Final = "dataset.config.simulator-directory"
SIMULATOR_DATASET_ENVIRONMENT_KEY: Final = "dataset.environment.simulator-directory"
SIMULATOR_DATASET_EVIDENCE_VERIFIER_KEY: Final = "dataset.evidence-verifier.simulator-directory"
SIMULATOR_DATASET_INSPECTOR_KEY: Final = "dataset.inspector.simulator-directory"
SIMULATOR_DATASET_RUNTIME_KEY: Final = "dataset.runtime.simulator-directory"
SIMULATOR_DATASET_CAPABILITY_VERSION: Final = "1.0.0"
SIMULATOR_DIRECTORY_MEDIA_TYPE: Final = "application/vnd.empirical-lawhood.allowlisted-directory+json"
SIMULATOR_DIRECTORY_FORMAT_PROFILE_ID: Final = "directory.allowlisted-content"

SIMULATOR_MAX_SOURCE_BYTES: Final = 8 * 1024**3
SIMULATOR_MAX_SOURCE_FILES: Final = 10_000
SIMULATOR_MAX_METADATA_BYTES: Final = 10_000_000
SIMULATOR_MAX_SINGLE_FILE_BYTES: Final = 2 * 1024**3

_EXTERNAL_EVIDENCE_SCHEMAS: Final = tuple(
    sorted(
        (
            DatasetMaterializationVerificationReceipt.SCHEMA,
            DatasetRegistrationManifest.SCHEMA,
            DatasetTransformationManifest.SCHEMA,
        )
    )
)


def _manifest(
    *,
    capability_key: str,
    config: CapabilityConfigRef,
    runtime: ObjectIdentity,
    implementation_sha256: str,
    input_schema_ids: tuple[str, ...],
    output_schema_ids: tuple[str, ...],
    maximum_outcome_access: OutcomeAccess,
    resources: ResourceBudget,
    conformance_check_ids: tuple[str, ...],
) -> CapabilityManifest:
    return CapabilityManifest(
        capability_key=capability_key,
        capability_version=SIMULATOR_DATASET_CAPABILITY_VERSION,
        kind=CapabilityKind.SOURCE,
        config_schema=config.config_schema,
        config_schema_sha256=config.config_schema_sha256,
        input_schema_ids=input_schema_ids,
        output_schema_ids=output_schema_ids,
        permissions=(CapabilityPermission.READ_EXTERNAL_ARTIFACTS,),
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
    evidence_verifier_implementation_sha256: str,
) -> tuple[CapabilityRegistry, DatasetCapabilityRegistry, DatasetSupportingIdentityRegistry]:
    inspector = _manifest(
        capability_key=SIMULATOR_DATASET_INSPECTOR_KEY,
        config=config,
        runtime=runtime,
        implementation_sha256=inspector_implementation_sha256,
        input_schema_ids=tuple(
            sorted(
                (
                    DatasetDirectorySelectorManifest.SCHEMA,
                    DatasetMaterialization.SCHEMA,
                    DatasetRegistrationManifest.SCHEMA,
                )
            )
        ),
        output_schema_ids=tuple(
            sorted(
                (
                    CurrentMaterializationVerification.SCHEMA,
                    DatasetMaterialization.SCHEMA,
                )
            )
        ),
        maximum_outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        resources=ResourceBudget(
            cpu_cores=4,
            memory_bytes=2 * 1024**3,
            gpu_devices=0,
            wall_time_seconds=7200,
            source_scan_bytes=SIMULATOR_MAX_SOURCE_BYTES,
            output_bytes=SIMULATOR_MAX_METADATA_BYTES,
        ),
        conformance_check_ids=(
            "directory-literal-allowlist",
            "directory-no-follow-stable-read",
            "directory-selector-content-digest",
            "directory-zero-network-no-write",
        ),
    )
    evidence_verifier = _manifest(
        capability_key=SIMULATOR_DATASET_EVIDENCE_VERIFIER_KEY,
        config=config,
        runtime=runtime,
        implementation_sha256=evidence_verifier_implementation_sha256,
        input_schema_ids=_EXTERNAL_EVIDENCE_SCHEMAS,
        output_schema_ids=(DatasetEvidenceVerification.SCHEMA,),
        maximum_outcome_access=OutcomeAccess.OUTCOME_BLIND,
        resources=ResourceBudget(
            cpu_cores=1,
            memory_bytes=64 * 1024**2,
            gpu_devices=0,
            wall_time_seconds=120,
            source_scan_bytes=2 * SIMULATOR_MAX_METADATA_BYTES,
            output_bytes=2 * SIMULATOR_MAX_METADATA_BYTES,
        ),
        conformance_check_ids=(
            "dataset-evidence-guard",
            "dataset-evidence-paired-record-byte-cap",
        ),
    )
    capability_registry = CapabilityRegistry(
        registry_id="capabilities.dataset.simulator-directory",
        capabilities=tuple(
            sorted((inspector, evidence_verifier), key=lambda value: value.registry_id)
        ),
    )
    dataset_registry = DatasetCapabilityRegistry(
        registry_id="dataset-capabilities.simulator-directory",
        providers=(),
        inspectors=(
            DatasetFormatInspectorRegistration(
                capability=inspector,
                accepted_media_types=(SIMULATOR_DIRECTORY_MEDIA_TYPE,),
                supported_format_profile_ids=(SIMULATOR_DIRECTORY_FORMAT_PROFILE_ID,),
                limits=DatasetCapabilityLimits(
                    max_input_bytes=SIMULATOR_MAX_SOURCE_BYTES,
                    max_output_bytes=SIMULATOR_MAX_METADATA_BYTES,
                    max_records=SIMULATOR_MAX_SOURCE_FILES,
                    max_files=SIMULATOR_MAX_SOURCE_FILES,
                    max_archive_members=0,
                ),
            ),
        ),
        transforms=(),
        binding_compilers=(),
        storage_verifiers=(),
        evidence_verifiers=(
            DatasetEvidenceVerifierRegistration(
                registry_id=SIMULATOR_DATASET_EVIDENCE_VERIFIER_KEY,
                capability=evidence_verifier,
                supported_evidence_schema_ids=_EXTERNAL_EVIDENCE_SCHEMAS,
                limits=DatasetCapabilityLimits(
                    max_input_bytes=SIMULATOR_MAX_METADATA_BYTES,
                    max_output_bytes=SIMULATOR_MAX_METADATA_BYTES,
                    max_records=2,
                    max_files=2,
                    max_archive_members=0,
                ),
            ),
        ),
    )
    supporting_registry = DatasetSupportingIdentityRegistry(
        registry_id="dataset-support.simulator-directory",
        configs=(ObjectIdentity.from_record(SIMULATOR_DATASET_CONFIG_KEY, config),),
        runtimes=(runtime,),
        environments=(environment,),
    )
    return capability_registry, dataset_registry, supporting_registry


@dataclass(frozen=True, slots=True)
class SimulatorDatasetRegistryProfile(CanonicalRecord):
    """Exact static registry closure for held simulator directory selections."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/simulator-dataset-registry-profile'

    config: CapabilityConfigRef
    runtime: ObjectIdentity
    environment: ObjectIdentity
    inspector_implementation_sha256: str
    evidence_verifier_implementation_sha256: str
    capability_registry: CapabilityRegistry
    dataset_registry: DatasetCapabilityRegistry
    supporting_registry: DatasetSupportingIdentityRegistry

    def __post_init__(self) -> None:
        if not isinstance(self.config, CapabilityConfigRef):
            raise ValueError("config must be a CapabilityConfigRef")
        if self.config.config_id != SIMULATOR_DATASET_CONFIG_KEY:
            raise ValueError("config has the wrong simulator dataset key")
        for value, expected_id, field_name in (
            (self.runtime, SIMULATOR_DATASET_RUNTIME_KEY, "runtime"),
            (self.environment, SIMULATOR_DATASET_ENVIRONMENT_KEY, "environment"),
        ):
            if (
                not isinstance(value, ObjectIdentity)
                or value.object_id != expected_id
                or value.object_schema != ArtifactIdentity.SCHEMA
            ):
                raise ValueError(f"{field_name} has another exact artifact identity")
        validate_sha256(
            self.inspector_implementation_sha256,
            field_name="inspector_implementation_sha256",
        )
        validate_sha256(
            self.evidence_verifier_implementation_sha256,
            field_name="evidence_verifier_implementation_sha256",
        )
        expected = _build_registries(
            config=self.config,
            runtime=self.runtime,
            environment=self.environment,
            inspector_implementation_sha256=self.inspector_implementation_sha256,
            evidence_verifier_implementation_sha256=(self.evidence_verifier_implementation_sha256),
        )
        if (self.capability_registry, self.dataset_registry, self.supporting_registry) != expected:
            raise ValueError("simulator dataset registry profile is not reproducible")


def build_simulator_dataset_registry_profile(
    *,
    config: CapabilityConfigRef,
    runtime: ObjectIdentity,
    environment: ObjectIdentity,
    inspector_implementation_sha256: str,
    evidence_verifier_implementation_sha256: str,
) -> SimulatorDatasetRegistryProfile:
    capability, dataset, supporting = _build_registries(
        config=config,
        runtime=runtime,
        environment=environment,
        inspector_implementation_sha256=inspector_implementation_sha256,
        evidence_verifier_implementation_sha256=evidence_verifier_implementation_sha256,
    )
    return SimulatorDatasetRegistryProfile(
        config=config,
        runtime=runtime,
        environment=environment,
        inspector_implementation_sha256=inspector_implementation_sha256,
        evidence_verifier_implementation_sha256=evidence_verifier_implementation_sha256,
        capability_registry=capability,
        dataset_registry=dataset,
        supporting_registry=supporting,
    )


def simulator_dataset_capability_bindings(
    profile: SimulatorDatasetRegistryProfile,
    *,
    implementation_registry: DatasetImplementationRegistry | None = None,
) -> tuple[DatasetCapabilityBinding, ...]:
    """Derive the exact five bindings required by a registration manifest."""

    if not isinstance(profile, SimulatorDatasetRegistryProfile):
        raise TypeError("profile must be a SimulatorDatasetRegistryProfile")
    inspector = profile.capability_registry.resolve(
        SIMULATOR_DATASET_INSPECTOR_KEY,
        SIMULATOR_DATASET_CAPABILITY_VERSION,
    )
    verifier = profile.dataset_registry.evidence_verifier(SIMULATOR_DATASET_EVIDENCE_VERIFIER_KEY)
    values = (
        DatasetCapabilityBinding(
            binding_id="capability.config",
            kind=DatasetCapabilityKind.CONFIG,
            registry_key=SIMULATOR_DATASET_CONFIG_KEY,
            implementation=profile.supporting_registry.resolve(
                DatasetCapabilityKind.CONFIG,
                SIMULATOR_DATASET_CONFIG_KEY,
            ),
        ),
        DatasetCapabilityBinding(
            binding_id="capability.environment",
            kind=DatasetCapabilityKind.ENVIRONMENT,
            registry_key=SIMULATOR_DATASET_ENVIRONMENT_KEY,
            implementation=profile.supporting_registry.resolve(
                DatasetCapabilityKind.ENVIRONMENT,
                SIMULATOR_DATASET_ENVIRONMENT_KEY,
            ),
        ),
        DatasetCapabilityBinding(
            binding_id="capability.evidence-verifier",
            kind=DatasetCapabilityKind.EVIDENCE_VERIFIER,
            registry_key=SIMULATOR_DATASET_EVIDENCE_VERIFIER_KEY,
            implementation=ObjectIdentity.from_record(
                SIMULATOR_DATASET_EVIDENCE_VERIFIER_KEY,
                verifier,
            ),
        ),
        DatasetCapabilityBinding(
            binding_id="capability.inspector",
            kind=DatasetCapabilityKind.INSPECTOR,
            registry_key=SIMULATOR_DATASET_INSPECTOR_KEY,
            implementation=ObjectIdentity.from_record(
                SIMULATOR_DATASET_INSPECTOR_KEY,
                inspector,
            ),
        ),
        DatasetCapabilityBinding(
            binding_id="capability.runtime",
            kind=DatasetCapabilityKind.RUNTIME,
            registry_key=SIMULATOR_DATASET_RUNTIME_KEY,
            implementation=profile.supporting_registry.resolve(
                DatasetCapabilityKind.RUNTIME,
                SIMULATOR_DATASET_RUNTIME_KEY,
            ),
        ),
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


__all__ = [
    "SIMULATOR_DATASET_CAPABILITY_VERSION",
    "SIMULATOR_DATASET_CONFIG_KEY",
    "SIMULATOR_DATASET_ENVIRONMENT_KEY",
    "SIMULATOR_DATASET_EVIDENCE_VERIFIER_KEY",
    "SIMULATOR_DATASET_INSPECTOR_KEY",
    "SIMULATOR_DATASET_RUNTIME_KEY",
    "SIMULATOR_DIRECTORY_FORMAT_PROFILE_ID",
    "SIMULATOR_DIRECTORY_MEDIA_TYPE",
    "SIMULATOR_MAX_METADATA_BYTES",
    "SIMULATOR_MAX_SINGLE_FILE_BYTES",
    "SIMULATOR_MAX_SOURCE_BYTES",
    "SIMULATOR_MAX_SOURCE_FILES",
    "SimulatorDatasetRegistryProfile",
    "build_simulator_dataset_registry_profile",
    "simulator_dataset_capability_bindings",
]
