"""Exact non-writing production composition for Glenn reproduction authoring."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import ClassVar, Final

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    validate_relative_locator,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.planning.dataset_authority import (
    DATASET_EXTERNAL_ROOT_CONTRACT_SCHEMA,
)
from empirical_lawhood.planning.dataset_manifests import (
    DatasetRegistrationManifest,
    DatasetTransformationManifest,
)
from empirical_lawhood.runtime.capabilities import (
    CapabilityConfigRef,
    ImplementationSourceClosure,
)
from empirical_lawhood.runtime.datasets import DatasetImplementationRegistry

from .contracts import GLENN_2026_PROFILE
from .manifests import GlennRetainedComparisonInputs, GLENN_ARCHIVE_RELATIVE_LOCATOR, GLENN_ARCHIVE_SHA256, GLENN_ARCHIVE_SIZE_BYTES, GLENN_HISTORICAL_OBSERVATIONS_RELATIVE_LOCATOR, GLENN_HISTORICAL_OBSERVATIONS_SIZE_BYTES, build_glenn_registration_manifest, build_glenn_transformation_manifest
from .registry import (
    GLENN_CONFIG_REGISTRY_KEY,
    GLENN_ENVIRONMENT_REGISTRY_KEY,
    GLENN_RUNTIME_REGISTRY_KEY,
    GlennDatasetRegistryProfile,
    build_glenn_dataset_registry_profile,
    compose_glenn_registration_manifest_dependencies,
    compose_glenn_transformation_manifest_dependencies,
)
from .transform import GLENN_PARQUET_PROFILE_ID


GLENN_PRODUCTION_PROFILE_ID: Final = "glenn-reproduction-ready"
GLENN_INSPECTOR_CLOSURE_ID: Final = "implementation.glenn-inspector"
GLENN_TRANSFORM_CLOSURE_ID: Final = "implementation.glenn-transform"
GLENN_EVIDENCE_VERIFIER_CLOSURE_ID: Final = "implementation.glenn-evidence-verifier"
GLENN_CONFIG_ARTIFACT_ID: Final = "artifact.glenn-dataset-capability-config"
GLENN_HISTORICAL_ARTIFACT_ID: Final = "artifact.glenn-g1-historical-observations"


@dataclass(frozen=True, slots=True)
class GlennDatasetCapabilityConfig(CanonicalRecord):
    """Closed static settings shared by the exact Glenn dataset capabilities."""

    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/physical/glenn/glenn-dataset-capability-config'
    )

    profile_id: str
    profile_version: str
    archive_profile_id: str
    output_profile_id: str
    expected_row_count: int
    selected_member_ids: tuple[str, ...]
    network_required: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.profile_id, field_name="profile_id")
        validate_semantic_version(self.profile_version)
        for field_name, value in (
            ("archive_profile_id", self.archive_profile_id),
            ("output_profile_id", self.output_profile_id),
        ):
            validate_stable_id(value, field_name=field_name)
        if (
            self.profile_id != GLENN_PRODUCTION_PROFILE_ID
            or self.profile_version != "1.0.0"
        ):
            raise ValueError(
                "Glenn capability config has the wrong production identity"
            )
        if self.archive_profile_id != GLENN_2026_PROFILE.profile_id:
            raise ValueError("Glenn capability config has the wrong archive profile")
        if self.output_profile_id != GLENN_PARQUET_PROFILE_ID:
            raise ValueError("Glenn capability config has the wrong output profile")
        if self.expected_row_count != GLENN_2026_PROFILE.expected_rows:
            raise ValueError("Glenn capability config has the wrong expected row count")
        expected_members = tuple(
            value.member_id for value in GLENN_2026_PROFILE.members
        )
        if self.selected_member_ids != expected_members:
            raise ValueError(
                "Glenn capability config changes the exact safe-member selector"
            )
        if not isinstance(self.network_required, bool) or self.network_required:
            raise ValueError("held-source Glenn reproduction must require zero network")


@dataclass(frozen=True, slots=True)
class GlennHeldObjectExpectation(CanonicalRecord):
    """One exact held object named by a complete-release selector."""

    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/physical/glenn/glenn-held-object-expectation'
    )

    object_id: str
    relative_locator: str
    expected_sha256: str
    expected_size_bytes: int
    media_type: str

    def __post_init__(self) -> None:
        validate_stable_id(self.object_id, field_name="object_id")
        validate_relative_locator(self.relative_locator)
        validate_sha256(self.expected_sha256, field_name="expected_sha256")
        if (
            not isinstance(self.expected_size_bytes, int)
            or isinstance(self.expected_size_bytes, bool)
            or self.expected_size_bytes <= 0
        ):
            raise ValueError("expected_size_bytes must be a positive integer")
        if (
            not isinstance(self.media_type, str)
            or not self.media_type
            or self.media_type != self.media_type.strip()
        ):
            raise ValueError("media_type must be a nonblank exact string")


@dataclass(frozen=True, slots=True)
class GlennHeldObjectSelectorManifest(CanonicalRecord):
    """Canonical selector bytes for exact held release/comparison objects."""

    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/physical/glenn/glenn-held-object-selector-manifest'
    )

    selector_id: str
    subject_id: str
    complete_subject: bool
    objects: tuple[GlennHeldObjectExpectation, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.selector_id, field_name="selector_id")
        validate_stable_id(self.subject_id, field_name="subject_id")
        if not isinstance(self.complete_subject, bool) or not self.complete_subject:
            raise ValueError(
                "Glenn held-object selector must be complete for its subject"
            )
        if not isinstance(self.objects, tuple) or not self.objects:
            raise ValueError("Glenn held-object selector must name exact objects")
        require_sorted_unique_ids(
            self.objects,
            attribute="object_id",
            field_name="objects",
        )


def _capability_config() -> GlennDatasetCapabilityConfig:
    return GlennDatasetCapabilityConfig(
        profile_id=GLENN_PRODUCTION_PROFILE_ID,
        profile_version="1.0.0",
        archive_profile_id=GLENN_2026_PROFILE.profile_id,
        output_profile_id=GLENN_PARQUET_PROFILE_ID,
        expected_row_count=GLENN_2026_PROFILE.expected_rows,
        selected_member_ids=tuple(
            value.member_id for value in GLENN_2026_PROFILE.members
        ),
        network_required=False,
    )


def _config_ref(config: GlennDatasetCapabilityConfig) -> CapabilityConfigRef:
    return CapabilityConfigRef(
        config_id=GLENN_CONFIG_REGISTRY_KEY,
        config_schema=config.SCHEMA,
        config_schema_sha256=hashlib.sha256(config.SCHEMA.encode("utf-8")).hexdigest(),
        content_sha256=config.fingerprint(),
        artifact_id=GLENN_CONFIG_ARTIFACT_ID,
    )


def _registration_selector() -> GlennHeldObjectSelectorManifest:
    return GlennHeldObjectSelectorManifest(
        selector_id="selector.glenn-complete-release",
        subject_id="release.glenn-2026",
        complete_subject=True,
        objects=(
            GlennHeldObjectExpectation(
                object_id="object.glenn-2026-archive",
                relative_locator=GLENN_ARCHIVE_RELATIVE_LOCATOR,
                expected_sha256=GLENN_ARCHIVE_SHA256,
                expected_size_bytes=GLENN_ARCHIVE_SIZE_BYTES,
                media_type="application/zip",
            ),
        ),
    )


def _historical_selector(
    inputs: GlennRetainedComparisonInputs,
) -> GlennHeldObjectSelectorManifest:
    return GlennHeldObjectSelectorManifest(
        selector_id="selector.glenn-historical-observations",
        subject_id=GLENN_HISTORICAL_ARTIFACT_ID,
        complete_subject=True,
        objects=(
            GlennHeldObjectExpectation(
                object_id="object.glenn-g1-historical-observations",
                relative_locator=GLENN_HISTORICAL_OBSERVATIONS_RELATIVE_LOCATOR,
                expected_sha256=inputs.observations_sha256,
                expected_size_bytes=GLENN_HISTORICAL_OBSERVATIONS_SIZE_BYTES,
                media_type="text/csv",
            ),
        ),
    )


def _historical_artifact(inputs: GlennRetainedComparisonInputs) -> ArtifactIdentity:
    return ArtifactIdentity(
        artifact_id=GLENN_HISTORICAL_ARTIFACT_ID,
        role="comparison-only",
        payload_schema="empirical-lawhood/physical/glenn/historical-observations-csv",
        sha256=inputs.observations_sha256,
        media_type="text/csv",
        size_bytes=GLENN_HISTORICAL_OBSERVATIONS_SIZE_BYTES,
    )


def _require_artifact(
    value: ArtifactIdentity,
    *,
    artifact_id: str,
    field_name: str,
) -> None:
    if not isinstance(value, ArtifactIdentity) or value.artifact_id != artifact_id:
        raise ValueError(f"{field_name} has the wrong exact artifact identity")


def _require_closure(
    value: ImplementationSourceClosure,
    *,
    closure_id: str,
    implementation_commit: str,
    field_name: str,
) -> None:
    if not isinstance(value, ImplementationSourceClosure):
        raise ValueError(f"{field_name} must be an ImplementationSourceClosure")
    if (
        value.closure_id != closure_id
        or value.implementation_commit != implementation_commit
    ):
        raise ValueError(f"{field_name} has the wrong exact implementation closure")


@dataclass(frozen=True, slots=True)
class GlennProductionStaticComposition(CanonicalRecord):
    """Canonical production inputs that can exist before source registration."""

    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/physical/glenn/glenn-production-static-composition'
    )

    historical_inputs: GlennRetainedComparisonInputs
    VERSION: ClassVar[str] = '1.0.0'
    implementation_commit: str
    storage_root: ObjectIdentity
    config: GlennDatasetCapabilityConfig
    runtime_artifact: ArtifactIdentity
    environment_artifact: ArtifactIdentity
    inspector_closure: ImplementationSourceClosure
    transform_closure: ImplementationSourceClosure
    evidence_verifier_closure: ImplementationSourceClosure
    registration_selector: GlennHeldObjectSelectorManifest
    historical_selector: GlennHeldObjectSelectorManifest
    historical_artifact: ArtifactIdentity
    registry_profile: GlennDatasetRegistryProfile
    registration_manifest: DatasetRegistrationManifest

    def __post_init__(self) -> None:
        if (
            not isinstance(self.storage_root, ObjectIdentity)
            or self.storage_root.object_schema != DATASET_EXTERNAL_ROOT_CONTRACT_SCHEMA
        ):
            raise ValueError("storage_root must identify one current external root")
        if self.config != _capability_config():
            raise ValueError("config differs from the exact Glenn production config")
        _require_artifact(
            self.runtime_artifact,
            artifact_id=GLENN_RUNTIME_REGISTRY_KEY,
            field_name="runtime_artifact",
        )
        _require_artifact(
            self.environment_artifact,
            artifact_id=GLENN_ENVIRONMENT_REGISTRY_KEY,
            field_name="environment_artifact",
        )
        for value, closure_id, field_name in (
            (self.inspector_closure, GLENN_INSPECTOR_CLOSURE_ID, "inspector_closure"),
            (self.transform_closure, GLENN_TRANSFORM_CLOSURE_ID, "transform_closure"),
            (
                self.evidence_verifier_closure,
                GLENN_EVIDENCE_VERIFIER_CLOSURE_ID,
                "evidence_verifier_closure",
            ),
        ):
            _require_closure(
                value,
                closure_id=closure_id,
                implementation_commit=self.implementation_commit,
                field_name=field_name,
            )
        if self.registration_selector != _registration_selector():
            raise ValueError(
                "registration_selector differs from the exact held release"
            )
        if self.historical_selector != _historical_selector(self.historical_inputs):
            raise ValueError(
                "historical_selector differs from the exact comparison artifact"
            )
        if self.historical_artifact != _historical_artifact(self.historical_inputs):
            raise ValueError(
                "historical_artifact differs from its fixed physical identity"
            )
        expected_profile = build_glenn_dataset_registry_profile(
            config=_config_ref(self.config),
            runtime=ObjectIdentity.from_record(
                self.runtime_artifact.artifact_id,
                self.runtime_artifact,
            ),
            environment=ObjectIdentity.from_record(
                self.environment_artifact.artifact_id,
                self.environment_artifact,
            ),
            inspector_implementation_sha256=self.inspector_closure.fingerprint(),
            transform_implementation_sha256=self.transform_closure.fingerprint(),
            evidence_verifier_implementation_sha256=(
                self.evidence_verifier_closure.fingerprint()
            ),
        )
        if self.registry_profile != expected_profile:
            raise ValueError(
                "registry_profile differs from the exact production closure"
            )
        expected_dependencies = compose_glenn_registration_manifest_dependencies(
            expected_profile,
            source_storage_root=self.storage_root,
            control_storage_root=self.storage_root,
            complete_release_selector_sha256=self.registration_selector.fingerprint(),
        )
        expected_manifest = build_glenn_registration_manifest(expected_dependencies)
        if self.registration_manifest != expected_manifest:
            raise ValueError(
                "registration_manifest differs from the exact production inputs"
            )

    def build_transformation_manifest(
        self,
        *,
        archive_materialization: ObjectIdentity,
        expected_output_physical_sha256: str | None = None,
        expected_output_byte_size: int | None = None,
        implementation_registry: DatasetImplementationRegistry | None = None,
    ) -> DatasetTransformationManifest:
        """Render only after registration supplies the exact parent identity."""

        dependencies = compose_glenn_transformation_manifest_dependencies(
            self.registry_profile,
            historical_inputs=self.historical_inputs,
            source_storage_root=self.storage_root,
            selector_storage_root=self.storage_root,
            destination_storage_root=self.storage_root,
            historical_storage_root=self.storage_root,
            control_storage_root=self.storage_root,
            archive_materialization=archive_materialization,
            historical_observations_artifact=ObjectIdentity.from_record(
                self.historical_artifact.artifact_id,
                self.historical_artifact,
            ),
            historical_complete_selector_sha256=self.historical_selector.fingerprint(),
            expected_output_physical_sha256=expected_output_physical_sha256,
            expected_output_byte_size=expected_output_byte_size,
            implementation_registry=implementation_registry,
        )
        return build_glenn_transformation_manifest(dependencies)


def build_glenn_production_static_composition(
    *,
    historical_inputs: GlennRetainedComparisonInputs,
    implementation_commit: str,
    storage_root: ObjectIdentity,
    runtime_artifact: ArtifactIdentity,
    environment_artifact: ArtifactIdentity,
    inspector_closure: ImplementationSourceClosure,
    transform_closure: ImplementationSourceClosure,
    evidence_verifier_closure: ImplementationSourceClosure,
) -> GlennProductionStaticComposition:
    """Compose exact authoring records without opening any held dataset byte."""

    config = _capability_config()
    profile = build_glenn_dataset_registry_profile(
        config=_config_ref(config),
        runtime=ObjectIdentity.from_record(
            runtime_artifact.artifact_id, runtime_artifact
        ),
        environment=ObjectIdentity.from_record(
            environment_artifact.artifact_id,
            environment_artifact,
        ),
        inspector_implementation_sha256=inspector_closure.fingerprint(),
        transform_implementation_sha256=transform_closure.fingerprint(),
        evidence_verifier_implementation_sha256=evidence_verifier_closure.fingerprint(),
    )
    registration_selector = _registration_selector()
    dependencies = compose_glenn_registration_manifest_dependencies(
        profile,
        source_storage_root=storage_root,
        control_storage_root=storage_root,
        complete_release_selector_sha256=registration_selector.fingerprint(),
    )
    return GlennProductionStaticComposition(
        historical_inputs=historical_inputs,
        implementation_commit=implementation_commit,
        storage_root=storage_root,
        config=config,
        runtime_artifact=runtime_artifact,
        environment_artifact=environment_artifact,
        inspector_closure=inspector_closure,
        transform_closure=transform_closure,
        evidence_verifier_closure=evidence_verifier_closure,
        registration_selector=registration_selector,
        historical_selector=_historical_selector(historical_inputs),
        historical_artifact=_historical_artifact(historical_inputs),
        registry_profile=profile,
        registration_manifest=build_glenn_registration_manifest(dependencies),
    )


__all__ = [
    "GLENN_CONFIG_ARTIFACT_ID",
    "GLENN_EVIDENCE_VERIFIER_CLOSURE_ID",
    "GLENN_HISTORICAL_ARTIFACT_ID",
    "GLENN_INSPECTOR_CLOSURE_ID",
    "GLENN_PRODUCTION_PROFILE_ID",
    "GLENN_TRANSFORM_CLOSURE_ID",
    "GlennDatasetCapabilityConfig",
    "GlennHeldObjectExpectation",
    "GlennHeldObjectSelectorManifest",
    "GlennProductionStaticComposition",
    "build_glenn_production_static_composition",
]
