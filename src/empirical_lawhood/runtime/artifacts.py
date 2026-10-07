"""Safe logical/physical artifact and common receipt contracts."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import PurePosixPath
from typing import ClassVar, Final, Protocol

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling, inherited_visibility
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    MAX_RELATIVE_LOCATOR_BYTES,
    CanonicalRecord,
    canonical_json_bytes,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_relative_locator as validate_relative_locator,
    validate_schema,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.status import OperationalStatus


ARTIFACT_PLANE_MINIMUM_FREE_BYTES: Final[int] = 100 * 1024**3
MAX_ARTIFACT_PUBLICATION_MEMBERS: Final[int] = 256
MAX_ARTIFACT_PUBLICATION_PATH_BYTES: Final[int] = MAX_RELATIVE_LOCATOR_BYTES
MAX_ARTIFACT_PUBLICATION_AGGREGATE_BYTES: Final[int] = 1024**4
MAX_ARTIFACT_SEMANTIC_VALIDATION_REGISTRATIONS: Final[int] = 100_000





class ArtifactProfile(StrEnum):
    CANONICAL_JSON = "CANONICAL_JSON"
    ARROW_IPC = "ARROW_IPC"
    PARQUET = "PARQUET"
    AUDITED_HDF5 = "AUDITED_HDF5"
    NUMPY_NO_PICKLE = "NUMPY_NO_PICKLE"
    JSONL_CHUNKS = "JSONL_CHUNKS"
    ONNX = "ONNX"
    SAFETENSORS = "SAFETENSORS"
    TEXT_PARAMETERS = "TEXT_PARAMETERS"
    RAW_SOURCE_BYTES = "RAW_SOURCE_BYTES"


@dataclass(frozen=True, slots=True)
class TabularFieldContract:
    """Infrastructure-independent semantic contract for one Arrow field."""

    name: str
    physical_type: str
    logical_type: str
    nullable: bool
    native_unit: str
    coordinate_frame: str
    clock_id: str | None

    def __post_init__(self) -> None:
        for field_name in (
            "name",
            "physical_type",
            "logical_type",
            "native_unit",
            "coordinate_frame",
        ):
            validate_nonempty(getattr(self, field_name), field_name=field_name)
        if self.clock_id is not None:
            validate_nonempty(self.clock_id, field_name="clock_id")


@dataclass(frozen=True, slots=True)
class TabularPayloadContract:
    """Runtime-owned table semantics composed into a physical Arrow validator."""

    payload_schema: str
    fields: tuple[TabularFieldContract, ...]
    primary_key_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_schema(self.payload_schema)
        if not self.fields:
            raise ValueError("tabular payload contract requires fields")
        names = tuple(value.name for value in self.fields)
        if len(set(names)) != len(names):
            raise ValueError("tabular payload field names repeat")
        if (
            not self.primary_key_fields
            or len(set(self.primary_key_fields)) != len(self.primary_key_fields)
            or not set(self.primary_key_fields).issubset(names)
        ):
            raise ValueError("tabular payload primary key is invalid")


_EXECUTABLE_SERIALIZATION_MEDIA = ("joblib", "pickle", "python-object")
_OUTCOME_ACCESS_RANK = {
    OutcomeAccess.OUTCOME_BLIND: 0,
    OutcomeAccess.EVALUATION_SEALED: 1,
    OutcomeAccess.DEVELOPMENT_VISIBLE: 2,
    OutcomeAccess.EVALUATOR_REVEAL: 3,
    OutcomeAccess.EVALUATION_REVEALED: 3,
    OutcomeAccess.PRIVILEGED_TRUTH: 4,
}


def most_restrictive_outcome_access(
    *values: OutcomeAccess,
) -> OutcomeAccess:
    if not values:
        raise ValueError("at least one outcome-access value is required")
    return max(values, key=lambda value: _OUTCOME_ACCESS_RANK[value])


def derived_outcome_access(
    declared: OutcomeAccess,
    *parent_values: OutcomeAccess,
) -> OutcomeAccess:
    """Resolve an output lane without exposing newly sealed evaluation bytes.

    A sealed evaluation acquisition may be scheduled from an immutable
    development-visible freeze.  The freeze remains visible in lineage and in
    the inherited visibility ceiling, but it must not relabel the newly
    generated evaluation receiver as development-visible.  Already revealed
    or privileged parents cannot be resealed.
    """

    if declared is OutcomeAccess.EVALUATION_SEALED and all(
        value
        in {
            OutcomeAccess.OUTCOME_BLIND,
            OutcomeAccess.DEVELOPMENT_VISIBLE,
            OutcomeAccess.EVALUATION_SEALED,
        }
        for value in parent_values
    ):
        return OutcomeAccess.EVALUATION_SEALED
    return most_restrictive_outcome_access(declared, *parent_values)


@dataclass(frozen=True, slots=True)
class ArtifactLineageParent(CanonicalRecord):
    """Exact evidence parent and the ceiling/access inherited from it."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/artifact-lineage-parent'

    identity: ObjectIdentity
    visibility_ceiling: VisibilityCeiling
    outcome_access: OutcomeAccess


def lineage_parent_sort_key(
    parent: ArtifactLineageParent,
) -> tuple[str, str, str, str]:
    """Order exact parents without conflating IDs reused by different schemas."""

    identity = parent.identity
    return (
        identity.object_id,
        identity.object_schema,
        identity.object_version,
        identity.object_fingerprint,
    )


@dataclass(frozen=True, slots=True)
class ExternalRootContract(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/external-root-contract'

    storage_root_id: str
    logical_name: str
    canonical_path: str
    required_mount_path: str
    mount_contract_schema: str
    minimum_free_bytes: int
    expected_mount_source: str | None = None
    expected_volume_identity: str | None = None
    allowed_filesystem_types: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_stable_id(self.storage_root_id, field_name="storage_root_id")
        validate_nonempty(self.logical_name, field_name="logical_name")
        for name, value in (
            ("canonical_path", self.canonical_path),
            ("required_mount_path", self.required_mount_path),
        ):
            validate_nonempty(value, field_name=name)
            if not value.startswith("/"):
                raise ValueError(f"{name} must be absolute")
        validate_schema(self.mount_contract_schema)
        if self.minimum_free_bytes <= 0:
            raise ValueError("external root free-space floor must be positive")
        if self.expected_mount_source is not None:
            validate_nonempty(self.expected_mount_source, field_name="expected_mount_source")
            if not self.expected_mount_source.startswith("/"):
                raise ValueError("expected_mount_source must be an absolute device path")
        if self.expected_volume_identity is not None:
            validate_nonempty(
                self.expected_volume_identity,
                field_name="expected_volume_identity",
            )
        require_sorted_unique_strings(
            self.allowed_filesystem_types,
            field_name="allowed_filesystem_types",
        )


@dataclass(frozen=True, slots=True)
class ArtifactGenericValidation(CanonicalRecord):
    """Exact infrastructure-owned profile validator bound to one payload contract."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/artifact-generic-validation'

    validator_key: str
    validator_version: str
    validator_implementation_sha256: str
    payload_schema: str
    profile: ArtifactProfile

    def __post_init__(self) -> None:
        validate_stable_id(self.validator_key, field_name="validator_key")
        validate_semantic_version(self.validator_version)
        validate_sha256(
            self.validator_implementation_sha256,
            field_name="validator_implementation_sha256",
        )
        validate_schema(self.payload_schema)


@dataclass(frozen=True, slots=True)
class ArtifactSemanticValidation(CanonicalRecord):
    """Exact replayable semantic checks bound to one artifact identity."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/artifact-semantic-validation'

    validator_key: str
    validator_version: str
    validator_implementation_sha256: str
    payload_schema: str
    profile: ArtifactProfile
    top_level_keys: tuple[str, ...] = ()
    value_keys: tuple[str, ...] = ()
    field_bindings: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        validate_stable_id(self.validator_key, field_name="validator_key")
        validate_semantic_version(self.validator_version)
        validate_sha256(
            self.validator_implementation_sha256,
            field_name="validator_implementation_sha256",
        )
        validate_schema(self.payload_schema)
        require_sorted_unique_strings(
            self.top_level_keys,
            field_name="artifact semantic top-level keys",
        )
        require_sorted_unique_strings(
            self.value_keys,
            field_name="artifact semantic value keys",
        )
        binding_keys = tuple(key for key, _value in self.field_bindings)
        if tuple(sorted(set(binding_keys))) != binding_keys:
            raise ValueError("artifact semantic field bindings must have sorted unique keys")
        for key, value in self.field_bindings:
            validate_nonempty(key, field_name="artifact semantic binding key")
            validate_nonempty(value, field_name="artifact semantic binding value")
            if key not in self.top_level_keys:
                raise ValueError("artifact semantic binding key is absent from document shape")
        if self.profile is ArtifactProfile.CANONICAL_JSON:
            if not self.top_level_keys:
                raise ValueError("canonical JSON semantic validation requires a document shape")
        elif self.top_level_keys or self.value_keys or self.field_bindings:
            raise ValueError(
                "non-JSON semantic validation is delegated to its registered profile/schema"
            )


@dataclass(frozen=True, slots=True)
class ArtifactSemanticValidationRegistration(CanonicalRecord):
    """One exact logical-artifact binding to an immutable semantic validator."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/artifact-semantic-validation-registration'

    logical_artifact_id: str
    validation: ArtifactSemanticValidation

    def __post_init__(self) -> None:
        validate_stable_id(self.logical_artifact_id, field_name="logical_artifact_id")


@dataclass(frozen=True, slots=True)
class ArtifactSemanticValidationRegistry(CanonicalRecord):
    """Immutable expected semantic contracts for a bounded artifact set."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/artifact-semantic-validation-registry'

    registry_id: str
    registrations: tuple[ArtifactSemanticValidationRegistration, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.registry_id, field_name="registry_id")
        if len(self.registrations) > MAX_ARTIFACT_SEMANTIC_VALIDATION_REGISTRATIONS:
            raise ValueError("artifact semantic validation registry exceeds its registration limit")
        logical_ids = tuple(item.logical_artifact_id for item in self.registrations)
        if tuple(sorted(set(logical_ids))) != logical_ids:
            raise ValueError(
                "artifact semantic validation registrations must have sorted unique "
                "logical artifact IDs"
            )
        implementation_by_identity: dict[tuple[str, str], str] = {}
        for registration in self.registrations:
            validation = registration.validation
            identity = (validation.validator_key, validation.validator_version)
            existing = implementation_by_identity.setdefault(
                identity,
                validation.validator_implementation_sha256,
            )
            if existing != validation.validator_implementation_sha256:
                raise ValueError(
                    "one semantic validator key/version cannot name multiple implementations"
                )

    @classmethod
    def from_registrations(
        cls,
        *,
        registry_id: str,
        registrations: tuple[ArtifactSemanticValidationRegistration, ...],
    ) -> ArtifactSemanticValidationRegistry:
        """Canonicalize caller-provided registrations without mutating the registry later."""

        if len(registrations) > MAX_ARTIFACT_SEMANTIC_VALIDATION_REGISTRATIONS:
            raise ValueError("artifact semantic validation registry exceeds its registration limit")
        by_logical_id: dict[str, ArtifactSemanticValidationRegistration] = {}
        for registration in registrations:
            existing = by_logical_id.get(registration.logical_artifact_id)
            if existing is not None and existing != registration:
                raise ValueError(
                    "one logical artifact cannot register multiple semantic validations"
                )
            by_logical_id[registration.logical_artifact_id] = registration
        return cls(
            registry_id=registry_id,
            registrations=tuple(by_logical_id[logical_id] for logical_id in sorted(by_logical_id)),
        )

    def resolve(
        self,
        logical_artifact_id: str,
        observed: ArtifactSemanticValidation | None,
    ) -> ArtifactSemanticValidation | None:
        """Resolve one manifest declaration against the exact immutable registration."""

        validate_stable_id(logical_artifact_id, field_name="logical_artifact_id")
        lower = 0
        upper = len(self.registrations)
        expected: ArtifactSemanticValidation | None = None
        while lower < upper:
            midpoint = (lower + upper) // 2
            registration = self.registrations[midpoint]
            if registration.logical_artifact_id < logical_artifact_id:
                lower = midpoint + 1
            elif registration.logical_artifact_id > logical_artifact_id:
                upper = midpoint
            else:
                expected = registration.validation
                break
        if expected is None and observed is None:
            return None
        if expected is None:
            raise KeyError(
                f"artifact {logical_artifact_id!r} lacks a registered semantic validator"
            )
        if observed is None:
            raise ValueError("artifact semantic validation omits its registered contract")
        if (
            observed.validator_key != expected.validator_key
            or observed.validator_version != expected.validator_version
        ):
            raise ValueError("artifact semantic validator identity differs from its registration")
        if observed.validator_implementation_sha256 != expected.validator_implementation_sha256:
            raise ValueError(
                "artifact semantic validator implementation differs from its registration"
            )
        if observed != expected:
            raise ValueError("artifact semantic validation differs from its registration")
        return expected


@dataclass(frozen=True, slots=True)
class LogicalArtifactIdentity(CanonicalRecord):
    """Storage-layout-independent identity of one scientific object."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/logical-artifact-identity'

    logical_artifact_id: str
    content_sha256: str
    payload_schema: str
    profile: ArtifactProfile
    media_type: str
    visibility_ceiling: VisibilityCeiling
    parent_visibility_ceilings: tuple[VisibilityCeiling, ...]
    outcome_access: OutcomeAccess
    generic_validation: ArtifactGenericValidation | None = None
    lineage_parents: tuple[ArtifactLineageParent, ...] = ()
    semantic_validation: ArtifactSemanticValidation | None = None

    def __post_init__(self) -> None:
        self._validate(allow_missing_generic_validation=False)

    def _validate(self, *, allow_missing_generic_validation: bool) -> None:
        validate_stable_id(self.logical_artifact_id, field_name="logical_artifact_id")
        validate_sha256(self.content_sha256, field_name="content_sha256")
        validate_schema(self.payload_schema)
        validate_nonempty(self.media_type, field_name="media_type")
        if any(fragment in self.media_type.lower() for fragment in _EXECUTABLE_SERIALIZATION_MEDIA):
            raise ValueError("unsafe executable serialization media is forbidden")
        if self.generic_validation is None:
            if not allow_missing_generic_validation:
                raise ValueError("current artifact requires generic validator identity")
        elif (
            self.generic_validation.payload_schema != self.payload_schema
            or self.generic_validation.profile is not self.profile
        ):
            raise ValueError("artifact generic validation differs from identity profile/schema")
        if self.semantic_validation is not None and (
            self.semantic_validation.payload_schema != self.payload_schema
            or self.semantic_validation.profile is not self.profile
        ):
            raise ValueError("artifact semantic validation differs from identity profile/schema")
        required = inherited_visibility(self.parent_visibility_ceilings, self.outcome_access)
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(required):
            raise ValueError("artifact visibility cannot be lowered from its lineage")
        parent_keys = tuple(lineage_parent_sort_key(parent) for parent in self.lineage_parents)
        if tuple(sorted(set(parent_keys))) != parent_keys:
            raise ValueError("artifact lineage parents must have sorted unique identities")
        lineage_ceilings = tuple(
            sorted(
                (parent.visibility_ceiling for parent in self.lineage_parents),
                key=lambda value: value.value,
            )
        )
        declared_ceilings = tuple(
            sorted(self.parent_visibility_ceilings, key=lambda value: value.value)
        )
        if self.lineage_parents and lineage_ceilings != declared_ceilings:
            raise ValueError("artifact parent identities and ceilings differ")
        if self.lineage_parents:
            inherited_access = max(
                (parent.outcome_access for parent in self.lineage_parents),
                key=lambda value: _OUTCOME_ACCESS_RANK[value],
            )
            sealed_from_unrevealed_lineage = (
                self.outcome_access is OutcomeAccess.EVALUATION_SEALED
                and all(
                    parent.outcome_access
                    in {
                        OutcomeAccess.OUTCOME_BLIND,
                        OutcomeAccess.DEVELOPMENT_VISIBLE,
                        OutcomeAccess.EVALUATION_SEALED,
                    }
                    for parent in self.lineage_parents
                )
            )
            if (
                not sealed_from_unrevealed_lineage
                and _OUTCOME_ACCESS_RANK[self.outcome_access]
                < _OUTCOME_ACCESS_RANK[inherited_access]
            ):
                raise ValueError("artifact outcome access is lower than a lineage parent")
            if (
                any(
                    parent.outcome_access is OutcomeAccess.EVALUATION_SEALED
                    for parent in self.lineage_parents
                )
                and self.outcome_access is OutcomeAccess.OUTCOME_BLIND
            ):
                raise ValueError("sealed lineage cannot be relabeled outcome-blind")





@dataclass(frozen=True, slots=True)
class ArtifactMaterialization(CanonicalRecord):
    """One physical shard/file realization of a logical artifact."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/artifact-materialization'

    materialization_id: str
    logical_artifact_id: str
    storage_root_id: str
    relative_path: str
    physical_sha256: str
    size_bytes: int
    compression: str
    partition_selector: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("materialization_id", self.materialization_id),
            ("logical_artifact_id", self.logical_artifact_id),
            ("storage_root_id", self.storage_root_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_relative_locator(self.relative_path)
        validate_sha256(self.physical_sha256, field_name="physical_sha256")
        if self.size_bytes < 0:
            raise ValueError("artifact size must be nonnegative")
        validate_nonempty(self.compression, field_name="compression")
        if self.partition_selector is not None:
            validate_nonempty(self.partition_selector, field_name="partition_selector")
            if len(self.partition_selector.encode("utf-8")) > 512:
                raise ValueError("partition selector exceeds the bounded locator budget")


@dataclass(frozen=True, slots=True)
class ArtifactPublicationMember(CanonicalRecord):
    """Exact physical and logical identity of one publication member."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/artifact-publication-member'

    materialization_id: str
    logical_artifact_id: str
    logical_identity_sha256: str
    storage_root_id: str
    relative_path: str
    physical_sha256: str
    size_bytes: int
    visibility_ceiling: VisibilityCeiling
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name, value in (
            ("materialization_id", self.materialization_id),
            ("logical_artifact_id", self.logical_artifact_id),
            ("storage_root_id", self.storage_root_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_sha256(
            self.logical_identity_sha256,
            field_name="logical_identity_sha256",
        )
        validate_relative_locator(self.relative_path)
        validate_sha256(self.physical_sha256, field_name="physical_sha256")
        if self.size_bytes < 0:
            raise ValueError("publication member size must be nonnegative")
        if len(self.relative_path.encode("utf-8")) > MAX_ARTIFACT_PUBLICATION_PATH_BYTES:
            raise ValueError("publication member path exceeds its byte bound")
        path_parts = PurePosixPath(self.relative_path).parts
        if ".publication-batches" in path_parts or any(
            part.endswith(".manifest.json") for part in path_parts
        ):
            raise ValueError("publication member path uses a reserved control form")


def validate_publication_scope_root(value: str) -> str:
    """Validate one explicit directory authority, including the external-root sentinel."""

    if value == ".":
        return value
    validate_relative_locator(value)
    if len(value.encode("utf-8")) > MAX_ARTIFACT_PUBLICATION_PATH_BYTES:
        raise ValueError("publication scope root exceeds its byte bound")
    if any(part == ".publication-batches" for part in PurePosixPath(value).parts):
        raise ValueError("publication scope root uses a reserved control directory")
    return value


@dataclass(frozen=True, slots=True)
class ArtifactPublicationScope(CanonicalRecord):
    """Immutable storage, path, visibility and outcome authority for one batch."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/artifact-publication-scope'

    publication_scope_id: str
    storage_root_id: str
    relative_root: str
    visibility_ceiling: VisibilityCeiling
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(
            self.publication_scope_id,
            field_name="publication_scope_id",
        )
        validate_stable_id(self.storage_root_id, field_name="storage_root_id")
        validate_publication_scope_root(self.relative_root)

    def contains(self, relative_path: str) -> bool:
        validate_relative_locator(relative_path)
        if self.relative_root == ".":
            return True
        root_parts = PurePosixPath(self.relative_root).parts
        path_parts = PurePosixPath(relative_path).parts
        return len(path_parts) > len(root_parts) and path_parts[: len(root_parts)] == root_parts


def artifact_publication_member(
    logical: LogicalArtifactIdentity,
    materialization: ArtifactMaterialization,
) -> ArtifactPublicationMember:
    if logical.logical_artifact_id != materialization.logical_artifact_id:
        raise ValueError("publication member logical and physical identities differ")
    return ArtifactPublicationMember(
        materialization_id=materialization.materialization_id,
        logical_artifact_id=materialization.logical_artifact_id,
        logical_identity_sha256=(
            logical.fingerprint()
        ),
        storage_root_id=materialization.storage_root_id,
        relative_path=materialization.relative_path,
        physical_sha256=materialization.physical_sha256,
        size_bytes=materialization.size_bytes,
        visibility_ceiling=logical.visibility_ceiling,
        outcome_access=logical.outcome_access,
    )


def artifact_publication_batch_id(
    publication_scope: ArtifactPublicationScope,
    members: tuple[ArtifactPublicationMember, ...],
) -> str:
    if not members:
        raise ValueError("artifact publication requires at least one member")
    digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "publication_scope": publication_scope,
                "members": members,
            }
        )
    ).hexdigest()
    return f"artifact-publication.{digest}"


def artifact_publication_commit_relative_path(
    publication_scope: ArtifactPublicationScope,
    publication_batch_id: str,
) -> str:
    return _artifact_publication_control_relative_path(
        publication_scope,
        publication_batch_id,
        suffix="commit.json",
    )


def artifact_publication_intent_relative_path(
    publication_scope: ArtifactPublicationScope,
    publication_batch_id: str,
) -> str:
    return _artifact_publication_control_relative_path(
        publication_scope,
        publication_batch_id,
        suffix="intent.json",
    )


def _artifact_publication_control_relative_path(
    publication_scope: ArtifactPublicationScope,
    publication_batch_id: str,
    *,
    suffix: str,
) -> str:
    validate_stable_id(publication_batch_id, field_name="publication_batch_id")
    scope_token = publication_scope.fingerprint()[:24]
    root_parts = (
        ()
        if publication_scope.relative_root == "."
        else PurePosixPath(publication_scope.relative_root).parts
    )
    relative = PurePosixPath(
        *root_parts,
        ".publication-batches",
        scope_token,
        f"{publication_batch_id}.{suffix}",
    ).as_posix()
    if len(relative.encode("utf-8")) > MAX_ARTIFACT_PUBLICATION_PATH_BYTES:
        raise ValueError("artifact publication control path exceeds its byte bound")
    return validate_relative_locator(relative)


def _validate_publication_members(
    publication_scope: ArtifactPublicationScope,
    members: tuple[ArtifactPublicationMember, ...],
) -> None:
    require_sorted_unique_ids(
        members,
        attribute="materialization_id",
        field_name="publication members",
    )
    if not members:
        raise ValueError("artifact publication requires at least one member")
    if len(members) > MAX_ARTIFACT_PUBLICATION_MEMBERS:
        raise ValueError("artifact publication exceeds its member-count bound")
    paths = tuple(member.relative_path for member in members)
    if len(set(paths)) != len(paths):
        raise ValueError("artifact publication member paths must be unique")
    if len({member.storage_root_id for member in members}) != 1:
        raise ValueError("artifact publication members must share one storage root")
    if any(member.storage_root_id != publication_scope.storage_root_id for member in members):
        raise ValueError("artifact publication member lies on another storage root")
    if any(not publication_scope.contains(member.relative_path) for member in members):
        raise ValueError("artifact publication member lies outside its explicit scope")
    if any(
        member.visibility_ceiling is not publication_scope.visibility_ceiling
        or member.outcome_access is not publication_scope.outcome_access
        for member in members
    ):
        raise ValueError("artifact publication members have mixed evidence classes")
    if sum(member.size_bytes for member in members) > MAX_ARTIFACT_PUBLICATION_AGGREGATE_BYTES:
        raise ValueError("artifact publication exceeds its aggregate byte bound")


@dataclass(frozen=True, slots=True)
class ArtifactPublicationCommit(CanonicalRecord):
    """The single durable visibility point for one immutable artifact batch."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/artifact-publication-commit'

    publication_batch_id: str
    publication_scope: ArtifactPublicationScope
    members: tuple[ArtifactPublicationMember, ...]

    def __post_init__(self) -> None:
        validate_stable_id(
            self.publication_batch_id,
            field_name="publication_batch_id",
        )
        _validate_publication_members(self.publication_scope, self.members)
        if self.publication_batch_id != artifact_publication_batch_id(
            self.publication_scope,
            self.members,
        ):
            raise ValueError("publication batch ID differs from its exact members")


@dataclass(frozen=True, slots=True)
class ArtifactPublicationIntent(CanonicalRecord):
    """Durable ownership of paths absent before an uncommitted batch copy."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/artifact-publication-intent'

    publication_batch_id: str
    publication_scope: ArtifactPublicationScope
    members: tuple[ArtifactPublicationMember, ...]
    absent_relative_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(
            self.publication_batch_id,
            field_name="publication_batch_id",
        )
        _validate_publication_members(self.publication_scope, self.members)
        if self.publication_batch_id != artifact_publication_batch_id(
            self.publication_scope,
            self.members,
        ):
            raise ValueError("publication batch ID differs from its exact members")
        require_sorted_unique_strings(
            self.absent_relative_paths,
            field_name="publication intent absent paths",
        )
        permitted_paths = {
            relative_path
            for member in self.members
            for relative_path in (
                member.relative_path,
                f"{member.relative_path}.manifest.json",
            )
        }
        for relative_path in self.absent_relative_paths:
            validate_relative_locator(relative_path)
            if relative_path not in permitted_paths:
                raise ValueError("publication intent contains a nonmember path")
        absent = set(self.absent_relative_paths)
        for member in self.members:
            if (
                member.relative_path in absent
                and f"{member.relative_path}.manifest.json" not in absent
            ):
                raise ValueError("publication intent records a manifest-first pair")


@dataclass(frozen=True, slots=True)
class ArtifactPublicationBinding(CanonicalRecord):
    """Manifest binding to a hash-checked atomic batch commit marker."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/artifact-publication-binding'

    publication_batch_id: str
    publication_scope: ArtifactPublicationScope
    commit_marker_relative_path: str
    commit_marker_sha256: str
    commit_marker_size_bytes: int
    members: tuple[ArtifactPublicationMember, ...]

    def __post_init__(self) -> None:
        validate_stable_id(
            self.publication_batch_id,
            field_name="publication_batch_id",
        )
        _validate_publication_members(self.publication_scope, self.members)
        if self.publication_batch_id != artifact_publication_batch_id(
            self.publication_scope,
            self.members,
        ):
            raise ValueError("publication batch ID differs from its exact members")
        expected_marker_path = artifact_publication_commit_relative_path(
            self.publication_scope,
            self.publication_batch_id,
        )
        if self.commit_marker_relative_path != expected_marker_path:
            raise ValueError("publication commit marker path is not batch-derived")
        validate_sha256(
            self.commit_marker_sha256,
            field_name="commit_marker_sha256",
        )
        if self.commit_marker_size_bytes <= 0:
            raise ValueError("publication commit marker size must be positive")


@dataclass(frozen=True, slots=True)
class ArtifactWriteRequest:
    logical_artifact_id: str
    relative_path: str
    payload_schema: str
    profile: ArtifactProfile
    media_type: str
    publication_scope_id: str
    publication_scope_relative_root: str
    payload: bytes
    visibility_ceiling: VisibilityCeiling
    parent_visibility_ceilings: tuple[VisibilityCeiling, ...]
    outcome_access: OutcomeAccess
    logical_content_sha256: str | None = None
    compression: str = "none"
    partition_selector: str | None = None
    lineage_parents: tuple[ArtifactLineageParent, ...] = ()
    read_only_source_input: bool = False
    minimum_free_bytes: int = 0
    semantic_validation: ArtifactSemanticValidation | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        validate_stable_id(self.logical_artifact_id, field_name="logical_artifact_id")
        validate_relative_locator(self.relative_path)
        validate_schema(self.payload_schema)
        validate_nonempty(self.media_type, field_name="media_type")
        validate_stable_id(self.publication_scope_id, field_name="publication_scope_id")
        validate_publication_scope_root(self.publication_scope_relative_root)
        if any(fragment in self.media_type.lower() for fragment in _EXECUTABLE_SERIALIZATION_MEDIA):
            raise ValueError("unsafe executable serialization media is forbidden")
        if not isinstance(self.payload, bytes):
            raise ValueError("artifact write payload must be immutable bytes")
        if self.logical_content_sha256 is not None:
            validate_sha256(
                self.logical_content_sha256,
                field_name="logical_content_sha256",
            )
        validate_nonempty(self.compression, field_name="compression")
        if self.minimum_free_bytes < 0:
            raise ValueError("operation free-space floor must be nonnegative")
        if self.semantic_validation is not None and (
            self.semantic_validation.payload_schema != self.payload_schema
            or self.semantic_validation.profile is not self.profile
        ):
            raise ValueError("artifact semantic validation differs from profile/schema")
        parent_keys = tuple(lineage_parent_sort_key(parent) for parent in self.lineage_parents)
        if tuple(sorted(set(parent_keys))) != parent_keys:
            raise ValueError("artifact lineage parents must have sorted unique identities")
        lineage_ceilings = tuple(
            sorted(
                (parent.visibility_ceiling for parent in self.lineage_parents),
                key=lambda value: value.value,
            )
        )
        declared_ceilings = tuple(
            sorted(self.parent_visibility_ceilings, key=lambda value: value.value)
        )
        if (
            self.parent_visibility_ceilings
            and not self.lineage_parents
            and not self.read_only_source_input
        ):
            raise ValueError("artifact request parent ceilings require exact identities")
        if self.lineage_parents and lineage_ceilings != declared_ceilings:
            raise ValueError("artifact request parent identities and ceilings differ")

    def as_stream(self) -> ArtifactStreamWriteRequest:
        """Publish already serialized control bytes through the bounded stream port."""

        size = len(self.payload)
        chunk_size = min(1024 * 1024, max(1, size))
        return ArtifactStreamWriteRequest(
            logical_artifact_id=self.logical_artifact_id,
            relative_path=self.relative_path,
            payload_schema=self.payload_schema,
            profile=self.profile,
            media_type=self.media_type,
            publication_scope_id=self.publication_scope_id,
            publication_scope_relative_root=self.publication_scope_relative_root,
            chunks=(
                self.payload[start : start + chunk_size] for start in range(0, size, chunk_size)
            ),
            maximum_bytes=max(1, size),
            maximum_chunk_bytes=chunk_size,
            expected_size_bytes=size,
            expected_physical_sha256=hashlib.sha256(self.payload).hexdigest(),
            visibility_ceiling=self.visibility_ceiling,
            parent_visibility_ceilings=self.parent_visibility_ceilings,
            outcome_access=self.outcome_access,
            logical_content_sha256=self.logical_content_sha256,
            compression=self.compression,
            partition_selector=self.partition_selector,
            lineage_parents=self.lineage_parents,
            read_only_source_input=self.read_only_source_input,
            minimum_free_bytes=self.minimum_free_bytes,
            semantic_validation=self.semantic_validation,
        )


@dataclass(frozen=True, slots=True)
class ArtifactStreamWriteRequest:
    """One bounded chunk source for an immutable artifact publication."""

    logical_artifact_id: str
    relative_path: str
    payload_schema: str
    profile: ArtifactProfile
    media_type: str
    publication_scope_id: str
    publication_scope_relative_root: str
    chunks: Iterable[bytes] = field(repr=False, compare=False)
    maximum_bytes: int = 0
    maximum_chunk_bytes: int = 1024 * 1024
    expected_size_bytes: int | None = None
    expected_physical_sha256: str | None = None
    visibility_ceiling: VisibilityCeiling = VisibilityCeiling.PROSPECTIVE
    parent_visibility_ceilings: tuple[VisibilityCeiling, ...] = ()
    outcome_access: OutcomeAccess = OutcomeAccess.OUTCOME_BLIND
    logical_content_sha256: str | None = None
    compression: str = "none"
    partition_selector: str | None = None
    lineage_parents: tuple[ArtifactLineageParent, ...] = ()
    read_only_source_input: bool = False
    minimum_free_bytes: int = 0
    semantic_validation: ArtifactSemanticValidation | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        validate_stable_id(self.logical_artifact_id, field_name="logical_artifact_id")
        validate_relative_locator(self.relative_path)
        validate_schema(self.payload_schema)
        validate_nonempty(self.media_type, field_name="media_type")
        validate_stable_id(self.publication_scope_id, field_name="publication_scope_id")
        validate_publication_scope_root(self.publication_scope_relative_root)
        if any(fragment in self.media_type.lower() for fragment in _EXECUTABLE_SERIALIZATION_MEDIA):
            raise ValueError("unsafe executable serialization media is forbidden")
        if self.maximum_bytes <= 0 or self.maximum_chunk_bytes <= 0:
            raise ValueError("stream byte and chunk limits must be positive")
        if self.maximum_chunk_bytes > self.maximum_bytes:
            raise ValueError("stream chunk limit cannot exceed its total byte limit")
        if (self.expected_size_bytes is None) != (self.expected_physical_sha256 is None):
            raise ValueError("stream expected size and digest must be declared together")
        if self.expected_size_bytes is not None:
            if self.expected_size_bytes < 0 or self.expected_size_bytes > self.maximum_bytes:
                raise ValueError("stream expected size exceeds its total byte limit")
            assert self.expected_physical_sha256 is not None
            validate_sha256(
                self.expected_physical_sha256,
                field_name="expected_physical_sha256",
            )
        if self.logical_content_sha256 is not None:
            validate_sha256(
                self.logical_content_sha256,
                field_name="logical_content_sha256",
            )
        validate_nonempty(self.compression, field_name="compression")
        if self.minimum_free_bytes < 0:
            raise ValueError("operation free-space floor must be nonnegative")
        if self.semantic_validation is not None and (
            self.semantic_validation.payload_schema != self.payload_schema
            or self.semantic_validation.profile is not self.profile
        ):
            raise ValueError("artifact stream semantic validation differs from profile/schema")
        parent_keys = tuple(lineage_parent_sort_key(parent) for parent in self.lineage_parents)
        if tuple(sorted(set(parent_keys))) != parent_keys:
            raise ValueError("artifact lineage parents must have sorted unique identities")
        lineage_ceilings = tuple(
            sorted(
                (parent.visibility_ceiling for parent in self.lineage_parents),
                key=lambda value: value.value,
            )
        )
        declared_ceilings = tuple(
            sorted(self.parent_visibility_ceilings, key=lambda value: value.value)
        )
        if (
            self.parent_visibility_ceilings
            and not self.lineage_parents
            and not self.read_only_source_input
        ):
            raise ValueError("artifact stream parent ceilings require exact identities")
        if self.lineage_parents and lineage_ceilings != declared_ceilings:
            raise ValueError("artifact stream parent identities and ceilings differ")


@dataclass(frozen=True, slots=True)
class ArtifactManifest(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/artifact-manifest'

    logical: LogicalArtifactIdentity
    materialization: ArtifactMaterialization
    publication: ArtifactPublicationBinding | None = None

    def __post_init__(self) -> None:
        if self.logical.logical_artifact_id != self.materialization.logical_artifact_id:
            raise ValueError("artifact manifest binds different logical/physical IDs")
        if self.publication is not None and (
            artifact_publication_member(self.logical, self.materialization)
            not in self.publication.members
        ):
            raise ValueError("artifact publication does not include its materialization")





@dataclass(frozen=True, slots=True)
class ArtifactWriteResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/artifact-write-result'

    logical: LogicalArtifactIdentity
    materialization: ArtifactMaterialization
    manifest_materialization: ArtifactMaterialization
    created: bool

    def __post_init__(self) -> None:
        if self.materialization.logical_artifact_id != self.logical.logical_artifact_id:
            raise ValueError("materialization and logical artifact differ")
        if self.manifest_materialization.storage_root_id != (self.materialization.storage_root_id):
            raise ValueError("artifact and manifest roots differ")


@dataclass(frozen=True, slots=True)
class ReceiptCheck(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/receipt-check'

    check_id: str
    passed: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.check_id, field_name="check_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.passed and self.reason_codes:
            raise ValueError("passing receipt checks cannot retain failure reasons")
        if not self.passed and not self.reason_codes:
            raise ValueError("failing receipt checks require reason codes")


@dataclass(frozen=True, slots=True)
class CanonicalTaskReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/canonical-task-receipt'

    receipt_id: str
    run_id: str
    task_id: str
    attempt_id: str
    implementation_commit: str
    input_materialization_ids: tuple[str, ...]
    output_materializations: tuple[ArtifactMaterialization, ...]
    output_logical_artifacts: tuple[LogicalArtifactIdentity, ...]
    checks: tuple[ReceiptCheck, ...]
    operational_status: OperationalStatus
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("receipt_id", self.receipt_id),
            ("run_id", self.run_id),
            ("task_id", self.task_id),
            ("attempt_id", self.attempt_id),
        ):
            validate_stable_id(value, field_name=name)
        if len(self.implementation_commit) != 40 or any(
            char not in "0123456789abcdef" for char in self.implementation_commit
        ):
            raise ValueError("implementation commit must be a lowercase Git SHA-1")
        require_sorted_unique_strings(
            self.input_materialization_ids,
            field_name="input_materialization_ids",
        )
        require_sorted_unique_ids(
            self.output_materializations,
            attribute="materialization_id",
            field_name="output_materializations",
        )
        require_sorted_unique_ids(
            self.output_logical_artifacts,
            attribute="logical_artifact_id",
            field_name="output_logical_artifacts",
        )
        if self.output_logical_artifacts and tuple(
            value.logical_artifact_id for value in self.output_logical_artifacts
        ) != tuple(value.logical_artifact_id for value in self.output_materializations):
            raise ValueError("receipt logical and physical outputs differ")
        require_sorted_unique_ids(self.checks, attribute="check_id", field_name="checks")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        all_pass = all(check.passed for check in self.checks)
        if self.operational_status is OperationalStatus.SUCCEEDED and not all_pass:
            raise ValueError("successful receipt cannot contain a failed check")
        if self.operational_status is not OperationalStatus.SUCCEEDED and not self.reason_codes:
            raise ValueError("non-success receipt requires terminal reasons")





class ArtifactWriter(Protocol):
    def write(self, request: ArtifactWriteRequest) -> ArtifactWriteResult: ...

    def write_stream(self, request: ArtifactStreamWriteRequest) -> ArtifactWriteResult: ...

    def write_batch(
        self,
        requests: tuple[ArtifactWriteRequest | ArtifactStreamWriteRequest, ...],
    ) -> tuple[ArtifactWriteResult, ...]: ...

    def verify(self, materialization: ArtifactMaterialization) -> None: ...

    def verify_manifest(self, manifest: ArtifactManifest) -> None: ...

    def verify_manifests(self, manifests: tuple[ArtifactManifest, ...]) -> None: ...


def verify_artifact_manifests(
    writer: ArtifactWriter, manifests: tuple[ArtifactManifest, ...]
) -> None:
    # Deduplicate complete immutable records before chunking. A conflicting
    # logical/materialization/publication binding must still reach the writer;
    # equality of a path or logical ID alone is insufficient.
    manifests = tuple(dict.fromkeys(manifests))
    for start in range(0, len(manifests), MAX_ARTIFACT_PUBLICATION_MEMBERS):
        writer.verify_manifests(manifests[start : start + MAX_ARTIFACT_PUBLICATION_MEMBERS])
