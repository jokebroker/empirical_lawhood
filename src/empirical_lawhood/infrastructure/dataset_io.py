"""Guarded read-only access to exact, already-held dataset source files."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import hashlib
import os
import stat
from pathlib import PurePosixPath
from typing import BinaryIO, Final

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import canonical_json_bytes, validate_sha256
from empirical_lawhood.kernel.time import parse_utc_timestamp
from empirical_lawhood.planning.dataset_authority import DatasetStorageScope
from empirical_lawhood.planning.dataset_manifests import DatasetDirectorySelectorManifest
from empirical_lawhood.runtime.dataset_io import (
    DatasetDirectoryGuardReceipt,
    DatasetSourceGuardReceipt,
    GuardedDatasetSource,
)

from .artifacts import (
    ArtifactIdentityConflict,
    GuardedExternalRoot,
)


_STREAM_CHUNK_BYTES: Final[int] = 1024 * 1024


def _secure_open_directory(path: str) -> int:
    """Open an absolute directory without following any path-component symlink."""

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    if not nofollow or not directory:
        raise ArtifactIdentityConflict("secure descriptor-relative opening is unavailable")
    if not os.path.isabs(path):
        raise ArtifactIdentityConflict("guarded storage root is not absolute")
    flags = os.O_RDONLY | directory | nofollow | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(os.path.sep, flags)
    try:
        for component in tuple(value for value in path.split(os.path.sep) if value):
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _secure_open_relative(root_path: str, relative_locator: str) -> int:
    """Open one regular-file candidate beneath a retained root descriptor."""

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:
        raise ArtifactIdentityConflict("dataset source no-follow opening is unavailable")
    components = tuple(relative_locator.split("/"))
    if not components or any(not value or value in {".", ".."} for value in components):
        raise ArtifactIdentityConflict("dataset source locator is not component-safe")
    directory_flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | nofollow | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        root_descriptor = _secure_open_directory(root_path)
        parent_descriptor = root_descriptor
        try:
            for component in components[:-1]:
                next_descriptor = os.open(component, directory_flags, dir_fd=parent_descriptor)
                if parent_descriptor != root_descriptor:
                    os.close(parent_descriptor)
                parent_descriptor = next_descriptor
            return os.open(
                components[-1],
                os.O_RDONLY | nofollow | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent_descriptor,
            )
        finally:
            if parent_descriptor != root_descriptor:
                os.close(parent_descriptor)
            os.close(root_descriptor)
    except OSError as error:
        raise ArtifactIdentityConflict(
            "dataset source descriptor-relative opening failed"
        ) from error


def _descriptor_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _hash_descriptor(
    descriptor: int,
    *,
    expected_size_bytes: int,
    maximum_bytes: int,
) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    remaining = expected_size_bytes
    while remaining:
        chunk = os.read(descriptor, min(_STREAM_CHUNK_BYTES, remaining))
        if not chunk:
            raise ArtifactIdentityConflict("held dataset source was truncated")
        if len(chunk) > remaining:
            raise ArtifactIdentityConflict("held dataset source exceeded its expected size")
        digest.update(chunk)
        remaining -= len(chunk)
    if os.read(descriptor, 1):
        raise ArtifactIdentityConflict("held dataset source exceeded its exact byte envelope")
    if expected_size_bytes > maximum_bytes:
        raise ArtifactIdentityConflict("held dataset source exceeds its read-work envelope")
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def _guard_receipt_id(
    *,
    storage_root: ObjectIdentity,
    source_scope: DatasetStorageScope,
    observed_size_bytes: int,
    observed_sha256: str,
    verified_at_utc: str,
) -> str:
    subject = canonical_json_bytes(
        {
            "observed_sha256": observed_sha256,
            "observed_size_bytes": observed_size_bytes,
            "source_scope_fingerprint": source_scope.fingerprint(),
            "storage_root_fingerprint": storage_root.object_fingerprint,
            "verified_at_utc": verified_at_utc,
        }
    )
    return f"dataset-source-guard.{hashlib.sha256(subject).hexdigest()[:24]}"


class ExternalDatasetSourceOpener:
    """Open exact source scopes as stable, bounded, path-free streams."""

    def __init__(self, root: GuardedExternalRoot) -> None:
        self.root = root

    @contextmanager
    def open_exact(
        self,
        *,
        source_scope: DatasetStorageScope,
        expected_size_bytes: int,
        expected_sha256: str,
        maximum_bytes: int,
        trusted_at_utc: str,
    ) -> Iterator[GuardedDatasetSource]:
        if not isinstance(source_scope, DatasetStorageScope):
            raise ValueError("source_scope must be a DatasetStorageScope")
        if (
            not isinstance(expected_size_bytes, int)
            or isinstance(expected_size_bytes, bool)
            or expected_size_bytes < 0
        ):
            raise ValueError("expected_size_bytes must be a nonnegative integer")
        if (
            not isinstance(maximum_bytes, int)
            or isinstance(maximum_bytes, bool)
            or maximum_bytes <= 0
            or expected_size_bytes > maximum_bytes
        ):
            raise ValueError("source byte expectation exceeds its positive work envelope")
        validate_sha256(expected_sha256, field_name="expected_sha256")
        parse_utc_timestamp(trusted_at_utc, field_name="trusted_at_utc")

        root_identity = ObjectIdentity.from_record(
            self.root.contract.storage_root_id,
            self.root.contract,
        )
        if source_scope.storage_root != root_identity:
            raise ArtifactIdentityConflict("dataset source scope binds another storage root")
        # Keep the ordinary root diagnostic/containment checks, then open every
        # component relative to retained descriptors so a parent swap cannot
        # escape between that check and ``open``.
        self.root.resolve(source_scope.relative_prefix, for_write=False)
        descriptor = _secure_open_relative(
            self.root.contract.canonical_path,
            source_scope.relative_prefix,
        )
        handle: BinaryIO | None = None
        completion: list[DatasetSourceGuardReceipt] = []
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ArtifactIdentityConflict("held dataset source is not a regular file")
            if before.st_size != expected_size_bytes:
                raise ArtifactIdentityConflict("held dataset source size differs from expectation")
            observed_sha256 = _hash_descriptor(
                descriptor,
                expected_size_bytes=expected_size_bytes,
                maximum_bytes=maximum_bytes,
            )
            after_hash = os.fstat(descriptor)
            if _descriptor_identity(before) != _descriptor_identity(after_hash):
                raise ArtifactIdentityConflict("held dataset source changed during preflight")
            if observed_sha256 != expected_sha256:
                raise ArtifactIdentityConflict(
                    "held dataset source digest differs from expectation"
                )
            receipt = DatasetSourceGuardReceipt(
                guard_receipt_id=_guard_receipt_id(
                    storage_root=root_identity,
                    source_scope=source_scope,
                    observed_size_bytes=expected_size_bytes,
                    observed_sha256=observed_sha256,
                    verified_at_utc=trusted_at_utc,
                ),
                storage_root=root_identity,
                source_scope=source_scope,
                relative_locator=source_scope.relative_prefix,
                observed_size_bytes=expected_size_bytes,
                observed_sha256=observed_sha256,
                verified_at_utc=trusted_at_utc,
                regular_file=True,
                opened_read_only=True,
                nofollow_enforced=True,
                complete_eof=True,
                file_identity_stable=True,
                network_bytes=0,
            )
            handle = os.fdopen(descriptor, "rb", closefd=True)
            descriptor = -1
            guarded = GuardedDatasetSource(
                stream=handle,
                guard_receipt_id=receipt.guard_receipt_id,
                _completed_receipt_reader=(lambda: completion[0] if completion else None),
            )
            yield guarded

            handle.seek(0)
            final_sha256 = _hash_descriptor(
                handle.fileno(),
                expected_size_bytes=expected_size_bytes,
                maximum_bytes=maximum_bytes,
            )
            final = os.fstat(handle.fileno())
            if final_sha256 != observed_sha256 or _descriptor_identity(
                after_hash
            ) != _descriptor_identity(final):
                raise ArtifactIdentityConflict("held dataset source changed during adapter use")
            rebound_descriptor = _secure_open_relative(
                self.root.contract.canonical_path,
                source_scope.relative_prefix,
            )
            try:
                rebound = os.fstat(rebound_descriptor)
                if _descriptor_identity(rebound) != _descriptor_identity(final):
                    raise ArtifactIdentityConflict(
                        "held dataset source locator changed during adapter use"
                    )
            finally:
                os.close(rebound_descriptor)
            completion.append(receipt)
        finally:
            if handle is not None:
                handle.close()
            elif descriptor >= 0:
                os.close(descriptor)


class ExternalDatasetDirectorySourceInspector:
    """Hash an exact literal member allowlist without scanning a directory."""

    def __init__(self, root: GuardedExternalRoot) -> None:
        self.root = root

    def inspect_exact(
        self,
        *,
        source_scope: DatasetStorageScope,
        selector: DatasetDirectorySelectorManifest,
        maximum_bytes: int,
        maximum_files: int,
        maximum_single_file_bytes: int,
        trusted_at_utc: str,
    ) -> DatasetDirectoryGuardReceipt:
        if not isinstance(source_scope, DatasetStorageScope):
            raise ValueError("source_scope must be a DatasetStorageScope")
        if not isinstance(selector, DatasetDirectorySelectorManifest):
            raise ValueError("selector must be a DatasetDirectorySelectorManifest")
        for field_name, value in (
            ("maximum_bytes", maximum_bytes),
            ("maximum_files", maximum_files),
            ("maximum_single_file_bytes", maximum_single_file_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if len(selector.members) > maximum_files:
            raise ValueError("directory selector exceeds its file-work envelope")
        if selector.expected_total_size_bytes > maximum_bytes:
            raise ValueError("directory selector exceeds its byte-work envelope")
        if any(value.expected_size_bytes > maximum_single_file_bytes for value in selector.members):
            raise ValueError("directory selector exceeds its single-file envelope")
        parse_utc_timestamp(trusted_at_utc, field_name="trusted_at_utc")

        root_identity = ObjectIdentity.from_record(
            self.root.contract.storage_root_id,
            self.root.contract,
        )
        if source_scope.storage_root != root_identity:
            raise ArtifactIdentityConflict("directory source scope binds another root")
        self.root.resolve(source_scope.relative_prefix, for_write=False)

        for member in selector.members:
            relative_locator = PurePosixPath(
                source_scope.relative_prefix,
                member.relative_locator,
            ).as_posix()
            self.root.resolve(relative_locator, for_write=False)
            descriptor = _secure_open_relative(
                self.root.contract.canonical_path,
                relative_locator,
            )
            rebound_descriptor = -1
            try:
                before = os.fstat(descriptor)
                if not stat.S_ISREG(before.st_mode):
                    raise ArtifactIdentityConflict(
                        "directory selector member is not a regular file"
                    )
                if before.st_size != member.expected_size_bytes:
                    raise ArtifactIdentityConflict(
                        "directory selector member size differs from expectation"
                    )
                observed_sha256 = _hash_descriptor(
                    descriptor,
                    expected_size_bytes=member.expected_size_bytes,
                    maximum_bytes=maximum_single_file_bytes,
                )
                after_hash = os.fstat(descriptor)
                if _descriptor_identity(before) != _descriptor_identity(after_hash):
                    raise ArtifactIdentityConflict(
                        "directory selector member changed during hashing"
                    )
                if observed_sha256 != member.expected_physical_sha256:
                    raise ArtifactIdentityConflict(
                        "directory selector member digest differs from expectation"
                    )
                rebound_descriptor = _secure_open_relative(
                    self.root.contract.canonical_path,
                    relative_locator,
                )
                rebound = os.fstat(rebound_descriptor)
                if _descriptor_identity(after_hash) != _descriptor_identity(rebound):
                    raise ArtifactIdentityConflict(
                        "directory selector member locator changed during verification"
                    )
            finally:
                if rebound_descriptor >= 0:
                    os.close(rebound_descriptor)
                os.close(descriptor)

        selector_identity = ObjectIdentity.from_record(selector.selector_id, selector)
        receipt_subject = canonical_json_bytes(
            {
                "selector_fingerprint": selector_identity.object_fingerprint,
                "source_scope_fingerprint": source_scope.fingerprint(),
                "storage_root_fingerprint": root_identity.object_fingerprint,
                "verified_at_utc": trusted_at_utc,
            }
        )
        return DatasetDirectoryGuardReceipt(
            guard_receipt_id=(
                f"dataset-directory-guard.{hashlib.sha256(receipt_subject).hexdigest()[:24]}"
            ),
            storage_root=root_identity,
            source_scope=source_scope,
            selector=selector_identity,
            observed_content_sha256=selector.expected_content_sha256,
            observed_size_bytes=selector.expected_total_size_bytes,
            observed_file_count=len(selector.members),
            verified_at_utc=trusted_at_utc,
            regular_files=True,
            opened_read_only=True,
            nofollow_enforced=True,
            complete_eof=True,
            file_identities_stable=True,
            literal_allowlist_only=True,
            network_bytes=0,
        )


__all__ = [
    "ExternalDatasetDirectorySourceInspector",
    "ExternalDatasetSourceOpener",
]
