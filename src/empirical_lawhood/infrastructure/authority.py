"""Guarded external persistence for complete approval records."""

from __future__ import annotations

from empirical_lawhood.planning.approval import FrozenRetrospectiveApproval

from collections.abc import Callable
import os
from pathlib import Path
import stat
from typing import Generic, TypeVar

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.approval import APPROVAL_SIGNATURE_VERSION, ApprovalCheckerRegistry, ApprovalGateAttestation, ApprovalSignatureAlgorithm, DurableAuthorizationRecord, FrozenApprovalProposal, FrozenIssuedStudyApprovalProposal
from empirical_lawhood.planning.dataset_authority import (
    DatasetAuthorizationSignatureAlgorithm,
)
from empirical_lawhood.runtime.artifacts import (
    ArtifactLineageParent,
    ArtifactManifest,
    ArtifactProfile,
    ArtifactWriteRequest,
)

from .artifacts import ExternalArtifactPlane
from .bounded_io import BoundedFileIOError, read_bounded_bytes
from .task_receipts import decode_artifact_manifest


RecordT = TypeVar("RecordT", bound=CanonicalRecord)
MAX_AUTHORITY_RECORD_BYTES = 10_000_000
_ED25519_PRIVATE_KEY_BYTES = 32


class Ed25519DatasetAuthorizationSigner:
    """Owner-only local signer for exact dataset operation authorizations."""

    signature_algorithm = DatasetAuthorizationSignatureAlgorithm.ED25519
    signature_version = "1.0.0"

    __slots__ = ("_private_key", "verification_key_hex")

    def __init__(self, private_key: Ed25519PrivateKey) -> None:
        self._private_key = private_key
        public_bytes = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        self.verification_key_hex = public_bytes.hex()

    @classmethod
    def create_private_key_file(cls, path: Path) -> Ed25519DatasetAuthorizationSigner:
        """Create one new 0600 raw key below an existing owner-only directory."""

        parent_descriptor = _open_owner_only_key_directory(path)
        descriptor: int | None = None
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path.name, flags, 0o600, dir_fd=parent_descriptor)
            private_key = Ed25519PrivateKey.generate()
            private_bytes = private_key.private_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PrivateFormat.Raw,
                encryption_algorithm=serialization.NoEncryption(),
            )
            _write_all(descriptor, private_bytes)
            os.fsync(descriptor)
        finally:
            if descriptor is not None:
                os.close(descriptor)
            os.close(parent_descriptor)
        return cls.from_private_key_file(path)

    @classmethod
    def from_private_key_file(cls, path: Path) -> Ed25519DatasetAuthorizationSigner:
        """Load exactly 32 secret bytes without following a key-file symlink."""

        parent_descriptor = _open_owner_only_key_directory(path)
        descriptor: int | None = None
        try:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path.name, flags, dir_fd=parent_descriptor)
            before = os.fstat(descriptor)
            _validate_private_key_file(before)
            private_bytes = _read_exact(descriptor, _ED25519_PRIVATE_KEY_BYTES)
            after = os.fstat(descriptor)
            if _file_identity(before) != _file_identity(after):
                raise PermissionError("dataset operator private key changed while loading")
        finally:
            if descriptor is not None:
                os.close(descriptor)
            os.close(parent_descriptor)
        return cls(Ed25519PrivateKey.from_private_bytes(private_bytes))

    def sign(self, payload: bytes) -> bytes:
        return self._private_key.sign(payload)


def _open_owner_only_key_directory(path: Path) -> int:
    if not isinstance(path, Path) or path.name in {"", ".", ".."}:
        raise ValueError("dataset operator key path must name one file")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path.parent, flags)
    try:
        observed = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(observed.st_mode)
            or observed.st_uid != os.geteuid()
            or stat.S_IMODE(observed.st_mode) & 0o077
        ):
            raise PermissionError("dataset operator key directory must be owner-only")
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _validate_private_key_file(observed: os.stat_result) -> None:
    if (
        not stat.S_ISREG(observed.st_mode)
        or observed.st_uid != os.geteuid()
        or observed.st_nlink != 1
        or stat.S_IMODE(observed.st_mode) != 0o600
        or observed.st_size != _ED25519_PRIVATE_KEY_BYTES
    ):
        raise PermissionError("dataset operator private key must be one owner-only 0600 file")


def _read_exact(descriptor: int, size_bytes: int) -> bytes:
    chunks: list[bytes] = []
    observed = 0
    while observed < size_bytes:
        chunk = os.read(descriptor, size_bytes - observed)
        if not chunk:
            break
        chunks.append(chunk)
        observed += len(chunk)
    if observed != size_bytes or os.read(descriptor, 1):
        raise PermissionError("dataset operator private key has an invalid byte length")
    return b"".join(chunks)


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        offset += os.write(descriptor, payload[offset:])


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


class Ed25519ApprovalAttestationSigner:
    """Private, non-canonical signer for a trusted approval-checker issuer."""

    signature_algorithm = ApprovalSignatureAlgorithm.ED25519
    signature_version = APPROVAL_SIGNATURE_VERSION

    __slots__ = ("_private_key", "verification_key_hex")

    def __init__(self, private_key: Ed25519PrivateKey) -> None:
        self._private_key = private_key
        public_bytes = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        self.verification_key_hex = public_bytes.hex()

    @classmethod
    def from_private_bytes(cls, private_key: bytes) -> Ed25519ApprovalAttestationSigner:
        """Construct an issuer from exactly 32 secret bytes supplied out of band."""

        if len(private_key) != 32:
            raise ValueError("Ed25519 approval private keys must contain exactly 32 bytes")
        return cls(Ed25519PrivateKey.from_private_bytes(private_key))

    @classmethod
    def create_private_key_file(cls, path: Path) -> Ed25519ApprovalAttestationSigner:
        """Create one new 0600 raw checker key below an owner-only directory."""

        parent_descriptor = _open_owner_only_key_directory(path)
        descriptor: int | None = None
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path.name, flags, 0o600, dir_fd=parent_descriptor)
            private_key = Ed25519PrivateKey.generate()
            private_bytes = private_key.private_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PrivateFormat.Raw,
                encryption_algorithm=serialization.NoEncryption(),
            )
            _write_all(descriptor, private_bytes)
            os.fsync(descriptor)
        finally:
            if descriptor is not None:
                os.close(descriptor)
            os.close(parent_descriptor)
        return cls.from_private_key_file(path)

    @classmethod
    def from_private_key_file(cls, path: Path) -> Ed25519ApprovalAttestationSigner:
        """Load exactly 32 checker secret bytes without following a symlink."""

        parent_descriptor = _open_owner_only_key_directory(path)
        descriptor: int | None = None
        try:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path.name, flags, dir_fd=parent_descriptor)
            before = os.fstat(descriptor)
            _validate_private_key_file(before)
            private_bytes = _read_exact(descriptor, _ED25519_PRIVATE_KEY_BYTES)
            after = os.fstat(descriptor)
            if _file_identity(before) != _file_identity(after):
                raise PermissionError("approval checker private key changed while loading")
        finally:
            if descriptor is not None:
                os.close(descriptor)
            os.close(parent_descriptor)
        return cls(Ed25519PrivateKey.from_private_bytes(private_bytes))

    def sign(self, payload: bytes) -> bytes:
        return self._private_key.sign(payload)


def load_owner_installed_approval_checker_registry(
    path: Path,
) -> ApprovalCheckerRegistry:
    """Load one exact owner-installed public checker trust root fail-closed."""

    parent_descriptor = _open_owner_only_key_directory(path)
    descriptor: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path.name, flags, dir_fd=parent_descriptor)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_size <= 0
            or before.st_size > MAX_AUTHORITY_RECORD_BYTES
        ):
            raise PermissionError(
                "approval checker trust root must be one bounded owner-only 0600 file"
            )
        payload = _read_exact(descriptor, before.st_size)
        after = os.fstat(descriptor)
        if _file_identity(before) != _file_identity(after):
            raise PermissionError("approval checker trust root changed while loading")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_descriptor)
    registry = decode_canonical_bytes(
        payload,
        ApprovalCheckerRegistry,
        maximum_bytes=MAX_AUTHORITY_RECORD_BYTES,
    )
    if registry.canonical_bytes() != payload:
        raise ValueError("approval checker trust root is not canonical")
    return registry


class _ExternalCanonicalRecordStore(Generic[RecordT]):
    """One immutable canonical-record namespace below the guarded artifact root."""

    def __init__(
        self,
        artifact_plane: ExternalArtifactPlane,
        *,
        namespace: str,
        record_type: type[RecordT],
        decoder: Callable[[bytes], RecordT],
    ) -> None:
        self._artifact_plane = artifact_plane
        self._namespace = namespace
        self._record_type = record_type
        self._accepted_schemas = (
            (record_type.SCHEMA, FrozenRetrospectiveApproval.SCHEMA)
            if record_type is FrozenIssuedStudyApprovalProposal
            else (record_type.SCHEMA,)
        )
        self._decoder = decoder

    def _relative_path(self, object_id: str) -> str:
        return f"authority/{self._namespace}/{object_id}.json"

    def persist(
        self,
        record: RecordT,
        *,
        lineage_parents: tuple[ArtifactLineageParent, ...] = (),
    ) -> ObjectIdentity:
        if not isinstance(record, self._record_type):
            raise TypeError("approval store received another canonical record type")
        object_id = record_id(record)
        result = self._artifact_plane.write(
            ArtifactWriteRequest(
                logical_artifact_id=object_id,
                relative_path=self._relative_path(object_id),
                payload_schema=record.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type="application/json",
                publication_scope_id="authority-records",
                publication_scope_relative_root="authority",
                payload=record.canonical_bytes(),
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                parent_visibility_ceilings=tuple(
                    parent.visibility_ceiling for parent in lineage_parents
                ),
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
                logical_content_sha256=record.fingerprint(),
                lineage_parents=lineage_parents,
            )
        )
        expected = ObjectIdentity.from_record(object_id, record)
        if (
            result.logical.logical_artifact_id != expected.object_id
            or result.logical.content_sha256 != expected.object_fingerprint
        ):
            raise RuntimeError("approval store persisted a different canonical identity")
        return expected

    def load(self, object_id: str) -> RecordT:
        relative_path = self._relative_path(object_id)
        payload_path = self._artifact_plane.root.resolve(relative_path, for_write=False)
        manifest_path = self._artifact_plane.root.resolve(
            f"{relative_path}.manifest.json",
            for_write=False,
        )
        if not payload_path.is_file() or not manifest_path.is_file():
            raise KeyError(object_id)
        manifest = decode_artifact_manifest(_read_bounded(manifest_path))
        if (
            manifest.logical.logical_artifact_id != object_id
            or manifest.logical.payload_schema not in self._accepted_schemas
            or manifest.logical.profile is not ArtifactProfile.CANONICAL_JSON
            or manifest.logical.media_type != "application/json"
            or manifest.logical.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or manifest.materialization.relative_path != relative_path
            or manifest.materialization.size_bytes > MAX_AUTHORITY_RECORD_BYTES
        ):
            raise ValueError("approval store manifest differs from its closed contract")
        self._artifact_plane.verify_manifest(
            ArtifactManifest(
                logical=manifest.logical,
                materialization=manifest.materialization,
            )
        )
        payload = _read_bounded(payload_path)
        try:
            value = self._decoder(payload)
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError("approval store payload is not a canonical record") from error
        if not isinstance(value, self._record_type):
            raise ValueError("approval store payload has another canonical type")
        if (
            value.canonical_bytes() != payload
            or record_id(value) != object_id
            or value.fingerprint() != manifest.logical.content_sha256
        ):
            raise ValueError("approval store payload identity differs from its manifest")
        return value


class ExternalFrozenApprovalProposalStore:
    """Durable immutable proposal/obligations store on the artifact plane."""

    def __init__(
        self,
        artifact_plane: ExternalArtifactPlane,
        *,
        decoder: Callable[[bytes], FrozenApprovalProposal],
    ) -> None:
        self._store = _ExternalCanonicalRecordStore(
            artifact_plane,
            namespace="frozen-proposals",
            record_type=FrozenApprovalProposal,
            decoder=decoder,
        )

    def persist(self, frozen: FrozenApprovalProposal) -> ObjectIdentity:
        return self._store.persist(frozen)

    def load(self, frozen_proposal_id: str) -> FrozenApprovalProposal:
        return self._store.load(frozen_proposal_id)


class ExternalFrozenStudyApprovalProposalStore:
    """Immutable additive ordinary-origin proposal store on the artifact plane."""

    def __init__(
        self,
        artifact_plane: ExternalArtifactPlane,
        *,
        decoder: Callable[[bytes], FrozenIssuedStudyApprovalProposal],
    ) -> None:
        self._store = _ExternalCanonicalRecordStore(
            artifact_plane,
            namespace="frozen-programme-proposals",
            record_type=FrozenIssuedStudyApprovalProposal,
            decoder=decoder,
        )

    def persist(self, frozen: FrozenIssuedStudyApprovalProposal) -> ObjectIdentity:
        return self._store.persist(frozen)

    def load(self, frozen_proposal_id: str) -> FrozenIssuedStudyApprovalProposal:
        return self._store.load(frozen_proposal_id)


class ExternalAuthorizationRecordStore:
    """Durable immutable complete-envelope authorization store."""

    def __init__(
        self,
        artifact_plane: ExternalArtifactPlane,
        *,
        decoder: Callable[[bytes], DurableAuthorizationRecord],
    ) -> None:
        self._store = _ExternalCanonicalRecordStore(
            artifact_plane,
            namespace="authorization-records",
            record_type=DurableAuthorizationRecord,
            decoder=decoder,
        )

    def persist(self, record: DurableAuthorizationRecord) -> ObjectIdentity:
        parent = ArtifactLineageParent(
            identity=record.envelope.frozen_proposal,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
        )
        return self._store.persist(record, lineage_parents=(parent,))

    def load(self, authorization_id: str) -> DurableAuthorizationRecord:
        return self._store.load(authorization_id)


class ExternalApprovalGateAttestationStore:
    """Durable immutable attestations issued by registered approval checkers."""

    def __init__(
        self,
        artifact_plane: ExternalArtifactPlane,
        *,
        decoder: Callable[[bytes], ApprovalGateAttestation],
        checker_registry: ApprovalCheckerRegistry,
    ) -> None:
        self._checker_registry = checker_registry
        self._store = _ExternalCanonicalRecordStore(
            artifact_plane,
            namespace="gate-attestations",
            record_type=ApprovalGateAttestation,
            decoder=decoder,
        )

    def persist(self, attestation: ApprovalGateAttestation) -> ObjectIdentity:
        self._checker_registry.validate(attestation)
        parent = ArtifactLineageParent(
            identity=attestation.subject,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
        )
        return self._store.persist(attestation, lineage_parents=(parent,))

    def load(self, attestation_id: str) -> ApprovalGateAttestation:
        attestation = self._store.load(attestation_id)
        self._checker_registry.validate(attestation)
        return attestation


def record_id(record: CanonicalRecord) -> str:
    for attribute in ("attestation_id", "frozen_proposal_id", "authorization_id"):
        value = getattr(record, attribute, None)
        if isinstance(value, str):
            return value
    raise TypeError("approval store record lacks a stable identity")


def _read_bounded(path: Path) -> bytes:
    try:
        return read_bounded_bytes(
            path,
            maximum_bytes=MAX_AUTHORITY_RECORD_BYTES,
        )
    except (BoundedFileIOError, OSError) as error:
        raise ValueError("approval store payload violates its bounded file contract") from error


__all__ = [
    "Ed25519ApprovalAttestationSigner",
    "ExternalApprovalGateAttestationStore",
    "ExternalAuthorizationRecordStore",
    "ExternalFrozenApprovalProposalStore",
    'ExternalFrozenStudyApprovalProposalStore',
    "load_owner_installed_approval_checker_registry",
]
