"""Strict deployment-local storage profile, separate from scientific identity."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import ClassVar

from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_semantic_version,
    validate_stable_id,
)

from .artifacts import ExternalRootContract


class OperatorStorageAccessMode(StrEnum):
    READ_ONLY = "READ_ONLY"
    READ_WRITE = "READ_WRITE"


@dataclass(frozen=True, slots=True)
class OperatorStorageProfile(CanonicalRecord):
    """Portable operator choices; this record grants no authority."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/operator-storage-profile'

    profile_id: str
    backend_key: str
    backend_version: str
    external_root_locator: str
    required_mount_path: str
    artifact_namespace: str
    scientific_scratch_namespace: str
    access_mode: OperatorStorageAccessMode
    expected_mount_source: str | None
    expected_volume_identity: str | None
    allowed_filesystem_types: tuple[str, ...]
    containment_policy_key: str
    minimum_free_bytes: int
    read_only_mirror_ids: tuple[str, ...]
    maximum_parallel_tasks: int
    authority_granted: bool

    def __post_init__(self) -> None:
        for name in (
            "profile_id",
            "backend_key",
            "artifact_namespace",
            "scientific_scratch_namespace",
            "containment_policy_key",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_semantic_version(self.backend_version)
        if self.backend_key != "external-filesystem" or self.backend_version != "1.0.0":
            raise ValueError("operator profile selects an unregistered storage backend")
        if self.containment_policy_key != "strict-mount-contained-no-symlink":
            raise ValueError("operator profile cannot weaken storage containment")
        for name in ("external_root_locator", "required_mount_path"):
            value = getattr(self, name)
            if not value.startswith("/"):
                raise ValueError(f"{name} must be absolute")
        if self.external_root_locator == "/" or self.required_mount_path == "/":
            raise ValueError("operator profile cannot select the filesystem root")
        if self.expected_mount_source is not None and not self.expected_mount_source.startswith(
            "/"
        ):
            raise ValueError("expected_mount_source must be an absolute device path")
        require_sorted_unique_strings(
            self.allowed_filesystem_types,
            field_name="allowed_filesystem_types",
        )
        require_sorted_unique_strings(
            self.read_only_mirror_ids,
            field_name="read_only_mirror_ids",
        )
        if self.minimum_free_bytes <= 0:
            raise ValueError("operator profile free-space floor must be positive")
        if not 1 <= self.maximum_parallel_tasks <= 64:
            raise ValueError("operator profile concurrency lies outside the registered envelope")
        if self.authority_granted:
            raise ValueError("operator profile cannot grant operation authority")


def resolve_external_root_contract(
    profile: OperatorStorageProfile,
    *,
    repo_root: Path,
    home_root: Path,
    temporary_root: Path = Path("/tmp"),
    for_write: bool = True,
) -> ExternalRootContract:
    """Resolve a validated profile; read-only diagnosis never grants write access."""

    if for_write and profile.access_mode is not OperatorStorageAccessMode.READ_WRITE:
        raise PermissionError("scientific execution requires a read/write operator profile")
    external = Path(profile.external_root_locator)
    mount = Path(profile.required_mount_path)
    artifact = external / profile.artifact_namespace
    scratch = external / profile.scientific_scratch_namespace
    forbidden = (
        repo_root.resolve(strict=True),
        home_root.resolve(strict=True),
        temporary_root.resolve(strict=True),
    )
    for candidate in (external, artifact, scratch):
        if not candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("operator storage locator is not canonical and absolute")
        resolved = candidate.resolve(strict=False)
        if any(resolved == value or resolved.is_relative_to(value) for value in forbidden):
            raise ValueError("operator storage locator enters a forbidden local root")
        if not resolved.is_relative_to(mount.resolve(strict=for_write)):
            raise ValueError("operator storage locator lies outside its required mount")
    if artifact == scratch or artifact.is_relative_to(scratch) or scratch.is_relative_to(artifact):
        raise ValueError("artifact and scientific scratch namespaces must be disjoint")
    fingerprint = profile.fingerprint()
    return ExternalRootContract(
        storage_root_id=f"operator-storage.{fingerprint}",
        logical_name=f"operator-selected artifact root for {profile.profile_id}",
        canonical_path=str(artifact),
        required_mount_path=str(mount),
        mount_contract_schema=OperatorStorageProfile.SCHEMA,
        minimum_free_bytes=profile.minimum_free_bytes,
        expected_mount_source=profile.expected_mount_source,
        expected_volume_identity=profile.expected_volume_identity,
        allowed_filesystem_types=profile.allowed_filesystem_types,
    )


def resolve_scientific_scratch_contract(
    profile: OperatorStorageProfile,
    *,
    repo_root: Path,
    home_root: Path,
    temporary_root: Path = Path("/tmp"),
) -> ExternalRootContract:
    """Resolve the disjoint scientific-scratch namespace under the same profile."""

    artifact_contract = resolve_external_root_contract(
        profile,
        repo_root=repo_root,
        home_root=home_root,
        temporary_root=temporary_root,
    )
    fingerprint = profile.fingerprint()
    return ExternalRootContract(
        storage_root_id=f"operator-scratch.{fingerprint}",
        logical_name=f"operator-selected scientific scratch for {profile.profile_id}",
        canonical_path=str(
            Path(profile.external_root_locator) / profile.scientific_scratch_namespace
        ),
        required_mount_path=artifact_contract.required_mount_path,
        mount_contract_schema=artifact_contract.mount_contract_schema,
        minimum_free_bytes=artifact_contract.minimum_free_bytes,
        expected_mount_source=artifact_contract.expected_mount_source,
        expected_volume_identity=artifact_contract.expected_volume_identity,
        allowed_filesystem_types=artifact_contract.allowed_filesystem_types,
    )


__all__ = [
    "OperatorStorageAccessMode",
    "OperatorStorageProfile",
    "resolve_external_root_contract",
    "resolve_scientific_scratch_contract",
]
