"""Guarded, immutable external artifact plane and safe serializers."""

from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import json
import math
import os
import shutil
import stat
import threading
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable, Iterator, Protocol
from multiprocessing.reduction import DupFd

import numpy as np
import numpy.typing as npt
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from empirical_lawhood.kernel.serialization import (
    canonical_json_bytes,
)
from empirical_lawhood.runtime.artifacts import (
    ArtifactManifest,
    ArtifactMaterialization,
    ArtifactProfile,
    ArtifactPublicationBinding,
    ArtifactPublicationCommit,
    ArtifactPublicationIntent,
    ArtifactPublicationScope,
    ArtifactSemanticValidation,
    ArtifactSemanticValidationRegistry,
    artifact_publication_batch_id,
    artifact_publication_commit_relative_path,
    artifact_publication_intent_relative_path,
    artifact_publication_member,
    ArtifactStreamWriteRequest,
    ArtifactWriteRequest,
    ArtifactWriteResult,
    ExternalRootContract,
    LogicalArtifactIdentity,
    MAX_ARTIFACT_PUBLICATION_AGGREGATE_BYTES,
    MAX_ARTIFACT_PUBLICATION_MEMBERS,
    MAX_ARTIFACT_PUBLICATION_PATH_BYTES,
    validate_relative_locator,
)
from empirical_lawhood.runtime.execution import (
    VerifiedArtifactInput,
    WorkerInputBinding,
    WorkerInputKind,
    WorkerInputPort,
)

from .bounded_io import (
    BoundedFileIOError,
    MAX_ARTIFACT_MANIFEST_BYTES,
    MAX_ARTIFACT_PUBLICATION_INTENT_BYTES,
    MAX_ARTIFACT_PUBLICATION_MARKER_BYTES,
    bounded_file_sha256,
    file_matches_bytes,
    read_bounded_bytes,
)
from .bounded_process import BoundedProcessError, run_bounded_command
from .file_locks import exclusive_file_lock
from .artifact_validation import (
    ArrowFieldContract as ArrowFieldContract,
    ArrowTableContract as ArrowTableContract,
    ArtifactIdentityConflict as ArtifactIdentityConflict,
    ArtifactInputLimitExceeded as ArtifactInputLimitExceeded,
    ArtifactPlaneError as ArtifactPlaneError,
    ArtifactProfileValidatorRegistry as ArtifactProfileValidatorRegistry,
    ExternalRootUnavailable as ExternalRootUnavailable,
    GenericValidationCompatibility as GenericValidationCompatibility,
    GenericValidatorImplementationCompatibility as GenericValidatorImplementationCompatibility,
    HDF5AttributeContract as HDF5AttributeContract,
    HDF5AttributeKind as HDF5AttributeKind,
    HDF5DatasetStorageContract as HDF5DatasetStorageContract,
    HDF5InventoryContract as HDF5InventoryContract,
    HDF5ObjectContract as HDF5ObjectContract,
    HDF5ObjectKind as HDF5ObjectKind,
    MAX_CANONICAL_JSON_DEPTH as MAX_CANONICAL_JSON_DEPTH,
    MAX_CANONICAL_JSON_TOKEN_BYTES as MAX_CANONICAL_JSON_TOKEN_BYTES,
    MAX_GENERIC_VALIDATION_COMPATIBILITIES as MAX_GENERIC_VALIDATION_COMPATIBILITIES,
    MAX_IN_MEMORY_ARTIFACT_BYTES as MAX_IN_MEMORY_ARTIFACT_BYTES,
    MAX_JSONL_RECORD_BYTES as MAX_JSONL_RECORD_BYTES,
    MAX_MODEL_PAYLOAD_BYTES as MAX_MODEL_PAYLOAD_BYTES,
    MAX_SAFETENSORS_HEADER_BYTES as MAX_SAFETENSORS_HEADER_BYTES,
    MAX_TEXT_PARAMETER_BYTES as MAX_TEXT_PARAMETER_BYTES,
    MAX_VALIDATOR_IMPLEMENTATION_SOURCE_BYTES as MAX_VALIDATOR_IMPLEMENTATION_SOURCE_BYTES,
    NumpyArrayContract as NumpyArrayContract,
    ParquetColumnContract as ParquetColumnContract,
    ParquetTableContract as ParquetTableContract,
    STREAM_CHUNK_BYTES as STREAM_CHUNK_BYTES,
    _CanonicalJsonFileInspector as _CanonicalJsonFileInspector,
    _decoded_json_bytes as _decoded_json_bytes,
    _metadata_json as _metadata_json,
    _strict_json as _strict_json,
    _validate_metadata_pairs as _validate_metadata_pairs,
    numpy_no_pickle_bytes as numpy_no_pickle_bytes,
    parquet_bytes as parquet_bytes,
    validate_numpy_no_pickle as validate_numpy_no_pickle,
)


@dataclass(slots=True)
class BoundedSerializedArtifact:
    """One verified external-scratch serialization, consumed as bounded bytes."""

    size_bytes: int
    physical_sha256: str
    _path: Path
    _consumed: bool = False

    def chunks(self, maximum_chunk_bytes: int = STREAM_CHUNK_BYTES) -> Iterator[bytes]:
        if maximum_chunk_bytes <= 0 or maximum_chunk_bytes > STREAM_CHUNK_BYTES:
            raise ValueError("serialized artifact chunk bound is invalid")
        if self._consumed:
            raise ArtifactIdentityConflict("serialized artifact was already consumed")
        observed_size, observed_sha256 = ExternalArtifactPlane._stream_identity(
            self._path,
            maximum_bytes=self.size_bytes,
        )
        if observed_size != self.size_bytes or observed_sha256 != self.physical_sha256:
            raise ArtifactIdentityConflict("serialized scratch identity drifted")
        self._consumed = True
        descriptor = os.open(
            self._path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        digest = hashlib.sha256()
        remaining = self.size_bytes
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_size != self.size_bytes:
                raise ArtifactIdentityConflict("serialized scratch size or type drifted")
            while remaining:
                chunk = os.read(descriptor, min(maximum_chunk_bytes, remaining))
                if not chunk:
                    raise ArtifactIdentityConflict("serialized scratch was truncated")
                remaining -= len(chunk)
                digest.update(chunk)
                yield chunk
            if os.read(descriptor, 1):
                raise ArtifactIdentityConflict("serialized scratch exceeded its byte bound")
            after = os.fstat(descriptor)
            before_identity = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            after_identity = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            if before_identity != after_identity or digest.hexdigest() != self.physical_sha256:
                raise ArtifactIdentityConflict(
                    "serialized scratch identity drifted during streaming"
                )
        finally:
            os.close(descriptor)

    def close(self) -> None:
        if self._path.exists():
            self._path.unlink()


def _anonymous_input_descriptor() -> int:
    """Return one sealable anonymous descriptor without a filesystem locator."""

    # The bundled CPython omits os.memfd_create even on Linux.  Calling the
    # libc wrapper preserves the same kernel primitive without falling back to
    # a named or directory-revealing temporary file.
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        memfd_create = libc.memfd_create
    except AttributeError as error:
        raise ArtifactInputLimitExceeded(
            "anonymous path-free worker inputs are unavailable"
        ) from error
    memfd_create.argtypes = (ctypes.c_char_p, ctypes.c_uint)
    memfd_create.restype = ctypes.c_int
    # linux/memfd.h: MFD_CLOEXEC | MFD_ALLOW_SEALING
    descriptor = int(memfd_create(b"empirical-lawhood-worker-input", 0x0001 | 0x0002))
    if descriptor < 0:
        error_number = ctypes.get_errno()
        raise ArtifactInputLimitExceeded(
            "anonymous path-free worker inputs are unavailable"
        ) from OSError(error_number, os.strerror(error_number))
    return descriptor


def _seal_anonymous_input(descriptor: int) -> None:
    """Make an anonymous staged input immutable before exposing it."""

    # linux/fcntl.h: F_ADD_SEALS with F_SEAL_SEAL, SHRINK, GROW and WRITE.
    try:
        fcntl.fcntl(descriptor, 1033, 0x0001 | 0x0002 | 0x0004 | 0x0008)
    except OSError as error:
        raise ArtifactInputLimitExceeded(
            "anonymous worker input could not be made immutable"
        ) from error


class _PreopenedBoundedReader:
    """Pickle-safe sequential reader backed only by a duplicated descriptor."""

    def __init__(self, descriptor: int, *, maximum_bytes: int) -> None:
        self._descriptor: int | Any | None = descriptor
        self._maximum_bytes = maximum_bytes
        self._bytes_read = 0
        self._handle: BinaryIO | None = None

    def __getstate__(self) -> dict[str, object]:
        """Duplicate the descriptor only into the spawned-process payload."""

        state = self.__dict__.copy()
        descriptor = self._descriptor
        if isinstance(descriptor, int):
            state["_descriptor"] = DupFd(descriptor)
        return state

    @property
    def bytes_read(self) -> int:
        return self._bytes_read

    def _binary_handle(self) -> BinaryIO:
        if self._handle is None:
            descriptor_reference = self._descriptor
            if descriptor_reference is None:
                raise ValueError("worker input is closed")
            descriptor = (
                descriptor_reference
                if isinstance(descriptor_reference, int)
                else descriptor_reference.detach()
            )
            self._descriptor = None
            self._handle = os.fdopen(descriptor, "rb", closefd=True)
        return self._handle

    def read(self, size: int = -1) -> bytes:
        if size < -1:
            raise ValueError("input read size must be -1 or nonnegative")
        remaining = self._maximum_bytes - self._bytes_read
        requested = remaining + 1 if size == -1 else min(size, remaining + 1)
        payload = self._binary_handle().read(requested)
        if len(payload) > remaining:
            raise ArtifactInputLimitExceeded("worker exceeded its input-scan budget")
        self._bytes_read += len(payload)
        return payload

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
            return
        descriptor_reference = self._descriptor
        self._descriptor = None
        if descriptor_reference is None:
            return
        if isinstance(descriptor_reference, int):
            try:
                os.close(descriptor_reference)
            except OSError as error:
                # A long-lived spawned worker can outlive the resource-sharer
                # copy of this descriptor.  EBADF therefore means the port is
                # already closed, which is the exact requested postcondition.
                if error.errno != errno.EBADF:
                    raise
            return
        try:
            descriptor = descriptor_reference.detach()
        except (EOFError, ValueError):
            return
        try:
            os.close(descriptor)
        except OSError as error:
            if error.errno != errno.EBADF:
                raise








@dataclass(frozen=True, slots=True)
class FilesystemState:
    canonical_root: str
    active_mount: bool
    writable: bool
    free_bytes: int
    path_is_symlink: bool
    mount_source: str | None = None
    volume_identity: str | None = None
    filesystem_type: str | None = None


@dataclass(frozen=True, slots=True)
class ExternalStorageDiagnostic:
    mount_active: bool
    canonical_contained: bool
    symlink_safe: bool
    mount_source: str | None
    mount_source_matches: bool
    volume_identity: str | None
    volume_identity_matches: bool
    filesystem_type: str | None
    filesystem_allowed: bool
    writable: bool
    observed_free_bytes: int
    effective_write_floor_bytes: int
    read_ready: bool
    write_ready: bool
    reason_codes: tuple[str, ...]


class FilesystemInspector(Protocol):
    def inspect(self, contract: ExternalRootContract) -> FilesystemState: ...


class LocalFilesystemInspector:
    def __init__(self) -> None:
        self._bound: tuple[FilesystemState, tuple[int, int], tuple[int, int]] | None = None

    def inspect_bound(self, contract: ExternalRootContract) -> FilesystemState:
        """Check the bound volume cheaply between full operation preflights."""
        if self._bound is None or self._bound[0].canonical_root != contract.canonical_path:
            return self.inspect(contract)
        state, root_identity, mount_identity = self._bound
        root, mount = Path(contract.canonical_path), Path(contract.required_mount_path)
        try:
            current_root, current_mount = root.stat(), mount.stat()
            if (
                (current_root.st_dev, current_root.st_ino) != root_identity
                or (current_mount.st_dev, current_mount.st_ino) != mount_identity
                or not os.path.ismount(mount)
                or root.resolve(strict=True) != root
                or mount.resolve(strict=True) != mount
            ):
                raise ExternalRootUnavailable("bound external volume was removed or replaced")
            return replace(
                state, writable=os.access(root, os.W_OK), free_bytes=shutil.disk_usage(root).free
            )
        except OSError as error:
            raise ExternalRootUnavailable("bound external volume is unavailable") from error

    def inspect(self, contract: ExternalRootContract) -> FilesystemState:
        root = Path(contract.canonical_path)
        mount = Path(contract.required_mount_path)
        resolved_root = root.resolve(strict=True)
        resolved_mount = mount.resolve(strict=True)
        try:
            resolved_root.relative_to(resolved_mount)
        except ValueError as error:
            raise ExternalRootUnavailable("external root escapes its required mount") from error
        usage = shutil.disk_usage(resolved_root)
        try:
            completed = run_bounded_command(
                [
                    "findmnt",
                    "--json",
                    "--target",
                    str(resolved_mount),
                    "--output",
                    "SOURCE,FSTYPE,UUID,TARGET",
                ],
                timeout_seconds=5,
                maximum_stdout_bytes=64 * 1024,
                maximum_stderr_bytes=16 * 1024,
            )
            if completed.returncode != 0:
                raise BoundedProcessError("findmnt returned a non-zero status")
            filesystems = json.loads(completed.stdout.decode("utf-8")).get("filesystems", [])
            mount_fact = filesystems[0] if len(filesystems) == 1 else {}
        except (
            BoundedProcessError,
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as error:
            raise ExternalRootUnavailable("cannot inspect the required mount identity") from error
        target = mount_fact.get("target")
        exact_mount = target == str(resolved_mount)
        state = FilesystemState(
            canonical_root=str(resolved_root),
            active_mount=os.path.ismount(resolved_mount) and exact_mount,
            writable=os.access(resolved_root, os.W_OK),
            free_bytes=usage.free,
            path_is_symlink=root.is_symlink() or mount.is_symlink(),
            mount_source=mount_fact.get("source"),
            volume_identity=mount_fact.get("uuid"),
            filesystem_type=mount_fact.get("fstype"),
        )
        root_stat, mount_stat = resolved_root.stat(), resolved_mount.stat()
        if root_stat.st_dev != mount_stat.st_dev:
            raise ExternalRootUnavailable("external root belongs to a different mounted volume")
        self._bound = (
            state,
            (root_stat.st_dev, root_stat.st_ino),
            (mount_stat.st_dev, mount_stat.st_ino),
        )
        return state


class GuardedExternalRoot:
    def __init__(
        self,
        contract: ExternalRootContract,
        inspector: FilesystemInspector | None = None,
    ) -> None:
        self.contract = contract
        self.inspector = inspector or LocalFilesystemInspector()

    def diagnostic(
        self,
        *,
        operation_minimum_free_bytes: int = 0,
    ) -> ExternalStorageDiagnostic:
        if operation_minimum_free_bytes < 0:
            raise ValueError("operation free-space floor must be nonnegative")
        state = self.inspector.inspect(self.contract)
        canonical_contained = state.canonical_root == self.contract.canonical_path
        symlink_safe = not state.path_is_symlink
        source_matches = (
            self.contract.expected_mount_source is None
            or state.mount_source == self.contract.expected_mount_source
        )
        volume_matches = (
            self.contract.expected_volume_identity is None
            or state.volume_identity == self.contract.expected_volume_identity
        )
        filesystem_allowed = (
            not self.contract.allowed_filesystem_types
            or state.filesystem_type in self.contract.allowed_filesystem_types
        )
        effective_floor = max(
            self.contract.minimum_free_bytes,
            operation_minimum_free_bytes,
        )
        reasons = []
        if not canonical_contained:
            reasons.append("EXTERNAL_ROOT_IDENTITY_DRIFT")
        if not state.active_mount:
            reasons.append("EXTERNAL_MOUNT_INACTIVE")
        if not symlink_safe:
            reasons.append("EXTERNAL_ROOT_SYMLINKED")
        if not source_matches:
            reasons.append("EXTERNAL_MOUNT_SOURCE_MISMATCH")
        if not volume_matches:
            reasons.append("EXTERNAL_VOLUME_IDENTITY_MISMATCH")
        if not filesystem_allowed:
            reasons.append("EXTERNAL_FILESYSTEM_NOT_ALLOWED")
        read_ready = not reasons
        if not state.writable:
            reasons.append("EXTERNAL_STORAGE_NOT_WRITABLE")
        if state.free_bytes < effective_floor:
            reasons.append("EXTERNAL_FREE_SPACE_BELOW_FLOOR")
        return ExternalStorageDiagnostic(
            mount_active=state.active_mount,
            canonical_contained=canonical_contained,
            symlink_safe=symlink_safe,
            mount_source=state.mount_source,
            mount_source_matches=source_matches,
            volume_identity=state.volume_identity,
            volume_identity_matches=volume_matches,
            filesystem_type=state.filesystem_type,
            filesystem_allowed=filesystem_allowed,
            writable=state.writable,
            observed_free_bytes=state.free_bytes,
            effective_write_floor_bytes=effective_floor,
            read_ready=read_ready,
            write_ready=read_ready and state.writable and state.free_bytes >= effective_floor,
            reason_codes=tuple(sorted(reasons)),
        )

    def verify(
        self,
        *,
        for_write: bool,
        operation_minimum_free_bytes: int = 0,
    ) -> Path:
        return self._verify(
            for_write=for_write,
            operation_minimum_free_bytes=operation_minimum_free_bytes,
            refresh_mount=True,
        )

    def _verify(
        self, *, for_write: bool, operation_minimum_free_bytes: int, refresh_mount: bool
    ) -> Path:
        if operation_minimum_free_bytes < 0:
            raise ValueError("operation free-space floor must be nonnegative")
        state = (
            self.inspector.inspect_bound(self.contract)
            if not refresh_mount and isinstance(self.inspector, LocalFilesystemInspector)
            else self.inspector.inspect(self.contract)
        )
        if state.canonical_root != self.contract.canonical_path:
            raise ExternalRootUnavailable("external root canonical identity drifted")
        if not state.active_mount:
            raise ExternalRootUnavailable("required external mount is not active")
        if state.path_is_symlink:
            raise ExternalRootUnavailable("external root or mount is symlinked")
        if (
            self.contract.expected_mount_source is not None
            and state.mount_source != self.contract.expected_mount_source
        ):
            raise ExternalRootUnavailable("external mount source identity differs")
        if (
            self.contract.expected_volume_identity is not None
            and state.volume_identity != self.contract.expected_volume_identity
        ):
            raise ExternalRootUnavailable("external volume identity differs")
        if self.contract.allowed_filesystem_types and (
            state.filesystem_type not in self.contract.allowed_filesystem_types
        ):
            raise ExternalRootUnavailable("external filesystem type is not allowed")
        if for_write:
            effective_floor = max(
                self.contract.minimum_free_bytes,
                operation_minimum_free_bytes,
            )
            if state.free_bytes < effective_floor:
                raise ExternalRootUnavailable("external root violates its free-space floor")
            if not state.writable:
                raise ExternalRootUnavailable("external root is not writable")
        return Path(state.canonical_root)

    def resolve(
        self,
        relative_path: str,
        *,
        for_write: bool,
        operation_minimum_free_bytes: int = 0,
    ) -> Path:
        validate_relative_locator(relative_path)
        root = self._verify(
            for_write=for_write,
            operation_minimum_free_bytes=operation_minimum_free_bytes,
            refresh_mount=False,
        )
        candidate = root.joinpath(*Path(relative_path).parts)
        resolved_parent = candidate.parent.resolve(strict=False)
        try:
            resolved_parent.relative_to(root)
        except ValueError as error:
            raise ExternalRootUnavailable("artifact path escapes external root") from error
        current = root
        for part in Path(relative_path).parts[:-1]:
            current = current / part
            if current.exists() and current.is_symlink():
                raise ExternalRootUnavailable("artifact parent contains a symlink")
        if candidate.exists() and candidate.is_symlink():
            raise ExternalRootUnavailable("artifact destination is a symlink")
        return candidate


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _materialization_id(logical_id: str, relative_path: str, sha256: str) -> str:
    digest = hashlib.sha256(f"{logical_id}\0{relative_path}\0{sha256}".encode()).hexdigest()
    return f"{logical_id}.{digest[:24]}"


@dataclass(frozen=True, slots=True)
class _StagedArtifactPublication:
    logical: LogicalArtifactIdentity
    materialization: ArtifactMaterialization
    manifest_materialization: ArtifactMaterialization
    manifest_payload: bytes
    artifact_path: Path
    sidecar_path: Path
    staged_artifact_path: Path
    staged_sidecar_path: Path


@dataclass(frozen=True, slots=True)
class _StagedArtifactBatch:
    members: tuple[_StagedArtifactPublication, ...]
    publication: ArtifactPublicationBinding
    commit_payload: bytes
    commit_path: Path
    intent_path: Path
    staged_commit_path: Path


class ExternalArtifactPlane:
    def __init__(
        self,
        root: GuardedExternalRoot,
        validators: ArtifactProfileValidatorRegistry | None = None,
        semantic_validations: ArtifactSemanticValidationRegistry | None = None,
    ) -> None:
        self.root = root
        self.validators = validators or ArtifactProfileValidatorRegistry()
        self.semantic_validations = semantic_validations or ArtifactSemanticValidationRegistry(
            registry_id="unregistered-artifact-semantics",
            registrations=(),
        )

    def with_semantic_validation_registry(
        self,
        registry: ArtifactSemanticValidationRegistry,
    ) -> ExternalArtifactPlane:
        """Return a plane view bound to one immutable semantic registry."""

        return ExternalArtifactPlane(
            self.root,
            self.validators,
            registry,
        )

    def _require_registered_semantics(
        self,
        logical_artifact_id: str,
        validation: ArtifactSemanticValidation | None,
    ) -> None:
        try:
            self.semantic_validations.resolve(logical_artifact_id, validation)
        except KeyError as error:
            raise ArtifactIdentityConflict(
                "artifact lacks an immutable registered semantic validator"
            ) from error
        except ValueError as error:
            raise ArtifactIdentityConflict(str(error)) from error

    def _serializer_path(
        self,
        suffix: str,
        *,
        maximum_bytes: int,
        minimum_free_bytes: int,
    ) -> Path:
        if maximum_bytes <= 0 or minimum_free_bytes < 0:
            raise ValueError("serializer bounds are invalid")
        token = hashlib.sha256(
            f"{os.getpid()}\0{threading.get_ident()}\0{time.time_ns()}".encode()
        ).hexdigest()[:24]
        path = self.root.resolve(
            f"scratch/artifact-serialization/{token}{suffix}",
            for_write=True,
            operation_minimum_free_bytes=max(maximum_bytes, minimum_free_bytes),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _finish_serialization(
        path: Path,
        *,
        maximum_bytes: int,
    ) -> BoundedSerializedArtifact:
        size_bytes, physical_sha256 = ExternalArtifactPlane._stream_identity(
            path,
            maximum_bytes=maximum_bytes,
        )
        if size_bytes > maximum_bytes:
            raise ArtifactIdentityConflict("serialized artifact exceeds its total byte limit")
        return BoundedSerializedArtifact(
            size_bytes=size_bytes,
            physical_sha256=physical_sha256,
            _path=path,
        )

    def serialize_numpy_chunks(
        self,
        *,
        payload_schema: str,
        chunks: Iterable[npt.NDArray[Any]],
        maximum_bytes: int,
        maximum_input_chunk_bytes: int = STREAM_CHUNK_BYTES,
        minimum_free_bytes: int = 0,
    ) -> BoundedSerializedArtifact:
        """Incrementally build one no-pickle NPY file in guarded scratch."""

        if maximum_input_chunk_bytes <= 0 or maximum_input_chunk_bytes > maximum_bytes:
            raise ValueError("NumPy input chunk bound is invalid")
        contract = self.validators.numpy_contract(payload_schema)
        dtype = np.dtype(contract.dtype)
        if dtype.hasobject or not contract.shape:
            raise ArtifactIdentityConflict(
                "incremental NumPy serialization requires a non-object array with rank"
            )
        if any(dimension is None for dimension in contract.shape):
            raise ArtifactIdentityConflict(
                "incremental NumPy serialization requires an exact registered shape"
            )
        exact_shape = tuple(dimension for dimension in contract.shape if dimension is not None)
        if math.prod(exact_shape) * dtype.itemsize > maximum_bytes:
            raise ArtifactIdentityConflict(
                "registered NumPy array exceeds the serializer byte limit"
            )
        path = self._serializer_path(
            ".npy",
            maximum_bytes=maximum_bytes,
            minimum_free_bytes=minimum_free_bytes,
        )
        try:
            target = np.lib.format.open_memmap(
                path,
                mode="w+",
                dtype=dtype,
                shape=exact_shape,
            )
            offset = 0
            try:
                for chunk in chunks:
                    if (
                        not isinstance(chunk, np.ndarray)
                        or chunk.dtype != dtype
                        or chunk.dtype.hasobject
                        or chunk.ndim != len(exact_shape)
                        or chunk.shape[1:] != exact_shape[1:]
                    ):
                        raise ArtifactIdentityConflict(
                            "NumPy input chunk differs from its registered contract"
                        )
                    if chunk.nbytes > maximum_input_chunk_bytes:
                        raise ArtifactIdentityConflict("NumPy input chunk exceeds its byte limit")
                    next_offset = offset + chunk.shape[0]
                    if next_offset > exact_shape[0]:
                        raise ArtifactIdentityConflict(
                            "NumPy input chunks exceed the registered shape"
                        )
                    target[offset:next_offset] = chunk
                    offset = next_offset
                if offset != exact_shape[0]:
                    raise ArtifactIdentityConflict(
                        "NumPy input chunks do not fill the registered shape"
                    )
                target.flush()
            finally:
                del target
            return self._finish_serialization(path, maximum_bytes=maximum_bytes)
        except Exception:
            if path.exists():
                path.unlink()
            raise

    def serialize_parquet_batches(
        self,
        *,
        payload_schema: str,
        schema: pa.Schema,
        batches: Iterable[pa.RecordBatch],
        maximum_bytes: int,
        maximum_rows: int,
        maximum_batch_rows: int = 65_536,
        maximum_input_batch_bytes: int = STREAM_CHUNK_BYTES,
        minimum_free_bytes: int = 0,
    ) -> BoundedSerializedArtifact:
        """Write bounded Arrow batches as uncompressed Parquet in external scratch."""

        self.validators.validate_table_schema(
            ArtifactProfile.PARQUET,
            payload_schema,
            schema,
        )
        if (
            min(maximum_rows, maximum_batch_rows, maximum_input_batch_bytes) <= 0
            or maximum_input_batch_bytes > maximum_bytes
        ):
            raise ValueError("Parquet batch bounds must be positive")
        path = self._serializer_path(
            ".parquet",
            maximum_bytes=maximum_bytes,
            minimum_free_bytes=minimum_free_bytes,
        )
        rows = 0
        try:
            with pa.OSFile(str(path), "wb") as sink:
                with pq.ParquetWriter(
                    sink,
                    schema,
                    compression="NONE",
                    use_dictionary=False,
                ) as writer:
                    for batch in batches:
                        self._validate_record_batch(
                            batch,
                            schema,
                            maximum_batch_rows=maximum_batch_rows,
                            maximum_input_batch_bytes=maximum_input_batch_bytes,
                        )
                        rows += batch.num_rows
                        if rows > maximum_rows:
                            raise ArtifactIdentityConflict(
                                "Parquet rows exceed the serializer limit"
                            )
                        writer.write_batch(batch)
                        if sink.tell() > maximum_bytes:
                            raise ArtifactIdentityConflict(
                                "Parquet serialization exceeds its byte limit"
                            )
            return self._finish_serialization(path, maximum_bytes=maximum_bytes)
        except Exception:
            if path.exists():
                path.unlink()
            raise

    def serialize_arrow_batches(
        self,
        *,
        payload_schema: str,
        schema: pa.Schema,
        batches: Iterable[pa.RecordBatch],
        maximum_bytes: int,
        maximum_rows: int,
        maximum_batch_rows: int = 65_536,
        maximum_input_batch_bytes: int = STREAM_CHUNK_BYTES,
        minimum_free_bytes: int = 0,
    ) -> BoundedSerializedArtifact:
        """Write bounded record batches as an Arrow IPC file in external scratch."""

        self.validators.validate_table_schema(
            ArtifactProfile.ARROW_IPC,
            payload_schema,
            schema,
        )
        if (
            min(maximum_rows, maximum_batch_rows, maximum_input_batch_bytes) <= 0
            or maximum_input_batch_bytes > maximum_bytes
        ):
            raise ValueError("Arrow batch bounds must be positive")
        path = self._serializer_path(
            ".arrow",
            maximum_bytes=maximum_bytes,
            minimum_free_bytes=minimum_free_bytes,
        )
        rows = 0
        try:
            with pa.OSFile(str(path), "wb") as sink:
                with pa.ipc.new_file(sink, schema) as writer:
                    for batch in batches:
                        self._validate_record_batch(
                            batch,
                            schema,
                            maximum_batch_rows=maximum_batch_rows,
                            maximum_input_batch_bytes=maximum_input_batch_bytes,
                        )
                        rows += batch.num_rows
                        if rows > maximum_rows:
                            raise ArtifactIdentityConflict("Arrow rows exceed the serializer limit")
                        writer.write_batch(batch)
                        if sink.tell() > maximum_bytes:
                            raise ArtifactIdentityConflict(
                                "Arrow serialization exceeds its byte limit"
                            )
            return self._finish_serialization(path, maximum_bytes=maximum_bytes)
        except Exception:
            if path.exists():
                path.unlink()
            raise

    @staticmethod
    def _validate_record_batch(
        batch: pa.RecordBatch,
        schema: pa.Schema,
        *,
        maximum_batch_rows: int,
        maximum_input_batch_bytes: int,
    ) -> None:
        if not isinstance(batch, pa.RecordBatch) or not batch.schema.equals(
            schema,
            check_metadata=True,
        ):
            raise ArtifactIdentityConflict("Arrow input batch differs from its declared schema")
        if batch.num_rows > maximum_batch_rows or batch.nbytes > maximum_input_batch_bytes:
            raise ArtifactIdentityConflict("Arrow input batch exceeds its bound")

    def write(self, request: ArtifactWriteRequest) -> ArtifactWriteResult:
        return self.write_batch((request,))[0]

    def write_stream(
        self,
        request: ArtifactStreamWriteRequest,
    ) -> ArtifactWriteResult:
        return self.write_batch((request,))[0]

    def write_batch(
        self,
        requests: tuple[ArtifactWriteRequest | ArtifactStreamWriteRequest, ...],
    ) -> tuple[ArtifactWriteResult, ...]:
        """Validate every member before publishing an immutable output batch.

        Staging is bounded by each request and occurs only in hidden files on
        the guarded external filesystem.  Publication then holds every member
        lock in deterministic order.  A caught publication failure removes
        only destinations created by this batch; matching pre-existing
        immutable pairs are never removed.
        """

        if not requests:
            raise ValueError("artifact publication batch cannot be empty")
        publication_scope = self._preflight_publication(requests)
        relative_paths = tuple(request.relative_path for request in requests)
        all_destinations = (*relative_paths, *(f"{path}.manifest.json" for path in relative_paths))
        if len(set(all_destinations)) != len(all_destinations):
            raise ArtifactIdentityConflict("artifact publication batch paths overlap")
        token = hashlib.sha256(
            (
                f"{os.getpid()}\0{threading.get_ident()}\0{time.time_ns()}\0"
                + "\0".join(relative_paths)
            ).encode("utf-8")
        ).hexdigest()[:24]
        staged: list[_StagedArtifactPublication] = []
        batch: _StagedArtifactBatch | None = None
        try:
            for index, request in enumerate(requests):
                staged.append(self._stage_publication(request, token=token, index=index))
            batch = self._bind_staged_batch(
                tuple(staged),
                publication_scope=publication_scope,
                token=token,
            )
            results = self._publish_staged_batch(batch)
        finally:
            for item in staged:
                self._unlink_if_present(item.staged_artifact_path)
                self._unlink_if_present(item.staged_sidecar_path)
            if batch is not None:
                self._unlink_if_present(batch.staged_commit_path)
        return results

    def _preflight_publication(
        self,
        requests: tuple[ArtifactWriteRequest | ArtifactStreamWriteRequest, ...],
    ) -> ArtifactPublicationScope:
        """Reject unbounded or cross-authority batches before staging any bytes."""

        if len(requests) > MAX_ARTIFACT_PUBLICATION_MEMBERS:
            raise ArtifactIdentityConflict("artifact publication exceeds its member-count bound")
        first = requests[0]
        publication_scope = ArtifactPublicationScope(
            publication_scope_id=first.publication_scope_id,
            storage_root_id=self.root.contract.storage_root_id,
            relative_root=first.publication_scope_relative_root,
            visibility_ceiling=first.visibility_ceiling,
            outcome_access=first.outcome_access,
        )
        declared_bytes = 0
        prospective: list[tuple[LogicalArtifactIdentity, ArtifactMaterialization]] = []
        for request in requests:
            if (
                request.publication_scope_id != publication_scope.publication_scope_id
                or request.publication_scope_relative_root != publication_scope.relative_root
            ):
                raise ArtifactIdentityConflict("artifact publication crosses explicit scopes")
            if (
                request.visibility_ceiling is not publication_scope.visibility_ceiling
                or request.outcome_access is not publication_scope.outcome_access
            ):
                raise ArtifactIdentityConflict("artifact publication mixes evidence classes")
            if not publication_scope.contains(request.relative_path):
                raise ArtifactIdentityConflict("artifact path lies outside its publication scope")
            if (
                any(
                    part.endswith(".manifest.json")
                    for part in PurePosixPath(request.relative_path).parts
                )
                or ".publication-batches" in PurePosixPath(request.relative_path).parts
                or len(f"{request.relative_path}.manifest.json".encode("utf-8"))
                > MAX_ARTIFACT_PUBLICATION_PATH_BYTES
            ):
                raise ArtifactIdentityConflict("artifact path violates publication path policy")
            declared_size = (
                len(request.payload)
                if isinstance(request, ArtifactWriteRequest)
                else request.maximum_bytes
            )
            declared_bytes += declared_size
            if declared_bytes > MAX_ARTIFACT_PUBLICATION_AGGREGATE_BYTES:
                raise ArtifactIdentityConflict(
                    "artifact publication exceeds its aggregate byte bound"
                )
            physical_sha256 = (
                _sha256(request.payload)
                if isinstance(request, ArtifactWriteRequest)
                else (
                    request.expected_physical_sha256 or request.logical_content_sha256 or ("0" * 64)
                )
            )
            if isinstance(request, ArtifactStreamWriteRequest):
                prospective_size = request.expected_size_bytes or request.maximum_bytes
            else:
                prospective_size = len(request.payload)
            logical, materialization, _manifest, _payload = self._prepare_identity(
                request,
                physical_sha256=physical_sha256,
                size_bytes=prospective_size,
            )
            prospective.append((logical, materialization))
        members = tuple(
            sorted(
                (
                    artifact_publication_member(logical, materialization)
                    for logical, materialization in prospective
                ),
                key=lambda value: value.materialization_id,
            )
        )
        batch_id = artifact_publication_batch_id(publication_scope, members)
        commit = ArtifactPublicationCommit(
            publication_batch_id=batch_id,
            publication_scope=publication_scope,
            members=members,
        )
        commit_payload = commit.canonical_bytes()
        if len(commit_payload) > MAX_ARTIFACT_PUBLICATION_MARKER_BYTES:
            raise ArtifactIdentityConflict("artifact publication marker exceeds its byte bound")
        binding = ArtifactPublicationBinding(
            publication_batch_id=batch_id,
            publication_scope=publication_scope,
            commit_marker_relative_path=artifact_publication_commit_relative_path(
                publication_scope,
                batch_id,
            ),
            commit_marker_sha256=_sha256(commit_payload),
            commit_marker_size_bytes=len(commit_payload),
            members=members,
        )
        for logical, materialization in prospective:
            manifest_payload = ArtifactManifest(
                logical=logical,
                materialization=materialization,
                publication=binding,
            ).canonical_bytes()
            if len(manifest_payload) > MAX_ARTIFACT_MANIFEST_BYTES:
                raise ArtifactIdentityConflict("artifact manifest exceeds its byte bound")
        intent = ArtifactPublicationIntent(
            publication_batch_id=batch_id,
            publication_scope=publication_scope,
            members=members,
            absent_relative_paths=tuple(
                sorted(
                    path
                    for member in members
                    for path in (
                        member.relative_path,
                        f"{member.relative_path}.manifest.json",
                    )
                )
            ),
        )
        if len(intent.canonical_bytes()) > MAX_ARTIFACT_PUBLICATION_INTENT_BYTES:
            raise ArtifactIdentityConflict("artifact publication intent exceeds its byte bound")
        return publication_scope

    def _bind_staged_batch(
        self,
        staged: tuple[_StagedArtifactPublication, ...],
        *,
        publication_scope: ArtifactPublicationScope,
        token: str,
    ) -> _StagedArtifactBatch:
        members = tuple(
            sorted(
                (
                    artifact_publication_member(item.logical, item.materialization)
                    for item in staged
                ),
                key=lambda value: value.materialization_id,
            )
        )
        publication_batch_id = artifact_publication_batch_id(publication_scope, members)
        commit = ArtifactPublicationCommit(
            publication_batch_id=publication_batch_id,
            publication_scope=publication_scope,
            members=members,
        )
        commit_payload = commit.canonical_bytes()
        if len(commit_payload) > MAX_ARTIFACT_PUBLICATION_MARKER_BYTES:
            raise ArtifactIdentityConflict("artifact publication marker exceeds its byte bound")
        commit_relative_path = artifact_publication_commit_relative_path(
            publication_scope,
            publication_batch_id,
        )
        publication = ArtifactPublicationBinding(
            publication_batch_id=publication_batch_id,
            publication_scope=publication_scope,
            commit_marker_relative_path=commit_relative_path,
            commit_marker_sha256=_sha256(commit_payload),
            commit_marker_size_bytes=len(commit_payload),
            members=members,
        )
        rebound: list[_StagedArtifactPublication] = []
        for item in staged:
            manifest_payload = ArtifactManifest(
                logical=item.logical,
                materialization=item.materialization,
                publication=publication,
            ).canonical_bytes()
            if len(manifest_payload) > MAX_ARTIFACT_MANIFEST_BYTES:
                raise ArtifactIdentityConflict("artifact manifest exceeds its byte bound")
            manifest_materialization = self._manifest_materialization(
                logical_artifact_id=item.logical.logical_artifact_id,
                relative_path=f"{item.materialization.relative_path}.manifest.json",
                payload=manifest_payload,
            )
            self._write_staged_bytes(item.staged_sidecar_path, manifest_payload)
            rebound.append(
                _StagedArtifactPublication(
                    logical=item.logical,
                    materialization=item.materialization,
                    manifest_materialization=manifest_materialization,
                    manifest_payload=manifest_payload,
                    artifact_path=item.artifact_path,
                    sidecar_path=item.sidecar_path,
                    staged_artifact_path=item.staged_artifact_path,
                    staged_sidecar_path=item.staged_sidecar_path,
                )
            )
        commit_path = self.root.resolve(
            commit_relative_path,
            for_write=True,
        )
        intent_path = self.root.resolve(
            artifact_publication_intent_relative_path(
                publication_scope,
                publication_batch_id,
            ),
            for_write=True,
        )
        if {commit_path, intent_path} & {
            path for item in rebound for path in (item.artifact_path, item.sidecar_path)
        }:
            raise ArtifactIdentityConflict(
                "artifact publication control path overlaps a batch member"
            )
        commit_path.parent.mkdir(parents=True, exist_ok=True)
        staged_commit_path = commit_path.with_name(f".{commit_path.name}.{token}.partial")
        self._write_staged_bytes(staged_commit_path, commit_payload)
        return _StagedArtifactBatch(
            members=tuple(rebound),
            publication=publication,
            commit_payload=commit_payload,
            commit_path=commit_path,
            intent_path=intent_path,
            staged_commit_path=staged_commit_path,
        )

    def _stage_publication(
        self,
        request: ArtifactWriteRequest | ArtifactStreamWriteRequest,
        *,
        token: str,
        index: int,
    ) -> _StagedArtifactPublication:
        if request.read_only_source_input:
            raise ArtifactIdentityConflict("historical artifact contracts are read-only")
        self._require_registered_semantics(
            request.logical_artifact_id,
            request.semantic_validation,
        )
        if request.compression != "none" or request.partition_selector is not None:
            raise ArtifactIdentityConflict(
                "compressed or partitioned artifacts require an equivalence proof contract"
            )
        if isinstance(request, ArtifactWriteRequest) and (
            len(request.payload) > MAX_IN_MEMORY_ARTIFACT_BYTES
        ):
            raise ArtifactIdentityConflict(
                "in-memory artifact payload exceeds its convenience limit; use write_stream"
            )
        artifact_path = self.root.resolve(
            request.relative_path,
            for_write=True,
            operation_minimum_free_bytes=request.minimum_free_bytes,
        )
        sidecar_path = self.root.resolve(
            f"{request.relative_path}.manifest.json",
            for_write=True,
            operation_minimum_free_bytes=request.minimum_free_bytes,
        )
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        staged_artifact_path = artifact_path.with_name(
            f".{artifact_path.name}.{token}.{index:04d}.batch.partial"
        )
        staged_sidecar_path = sidecar_path.with_name(
            f".{sidecar_path.name}.{token}.{index:04d}.batch.partial"
        )
        try:
            if isinstance(request, ArtifactWriteRequest):
                (
                    logical,
                    materialization,
                    manifest_materialization,
                    manifest_payload,
                ) = self._prepare(request)
                self._write_staged_bytes(staged_artifact_path, request.payload)
                observed_size, observed_sha256 = self._stream_identity(
                    staged_artifact_path,
                    maximum_bytes=len(request.payload),
                )
                if (
                    observed_size != materialization.size_bytes
                    or observed_sha256 != materialization.physical_sha256
                ):
                    raise ArtifactIdentityConflict("staged artifact identity changed")
                self.validators.validate_file(staged_artifact_path, request)
            else:
                size_bytes, physical_sha256 = self._write_staged_stream(
                    staged_artifact_path,
                    request,
                )
                if request.expected_size_bytes is not None and (
                    size_bytes != request.expected_size_bytes
                    or physical_sha256 != request.expected_physical_sha256
                ):
                    raise ArtifactIdentityConflict(
                        "artifact stream differs from its expected physical identity"
                    )
                self.validators.validate_file(staged_artifact_path, request)
                (
                    logical,
                    materialization,
                    manifest_materialization,
                    manifest_payload,
                ) = self._prepare_identity(
                    request,
                    physical_sha256=physical_sha256,
                    size_bytes=size_bytes,
                )
            # The sidecar gains its immutable sibling-set binding only after
            # every payload is staged. Write that final sidecar once in
            # _bind_staged_batch, before any destination is published.
            return _StagedArtifactPublication(
                logical=logical,
                materialization=materialization,
                manifest_materialization=manifest_materialization,
                manifest_payload=manifest_payload,
                artifact_path=artifact_path,
                sidecar_path=sidecar_path,
                staged_artifact_path=staged_artifact_path,
                staged_sidecar_path=staged_sidecar_path,
            )
        except BaseException:
            self._unlink_if_present(staged_artifact_path)
            self._unlink_if_present(staged_sidecar_path)
            raise

    @staticmethod
    def _write_staged_bytes(path: Path, payload: bytes) -> None:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            stat.S_IRUSR | stat.S_IWUSR,
        )
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            ExternalArtifactPlane._unlink_if_present(path)
            raise

    @staticmethod
    def _write_staged_stream(
        path: Path,
        request: ArtifactStreamWriteRequest,
    ) -> tuple[int, str]:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            stat.S_IRUSR | stat.S_IWUSR,
        )
        digest = hashlib.sha256()
        size_bytes = 0
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                for chunk in request.chunks:
                    if not isinstance(chunk, bytes):
                        raise ArtifactIdentityConflict(
                            "artifact stream chunks must be immutable bytes"
                        )
                    if len(chunk) > request.maximum_chunk_bytes:
                        raise ArtifactIdentityConflict(
                            "artifact stream chunk exceeds its byte limit"
                        )
                    size_bytes += len(chunk)
                    if size_bytes > request.maximum_bytes:
                        raise ArtifactIdentityConflict(
                            "artifact stream exceeds its total byte limit"
                        )
                    handle.write(chunk)
                    digest.update(chunk)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            ExternalArtifactPlane._unlink_if_present(path)
            raise
        return size_bytes, digest.hexdigest()

    def _publish_staged_batch(
        self,
        batch: _StagedArtifactBatch,
    ) -> tuple[ArtifactWriteResult, ...]:
        staged = batch.members
        created_flags: list[bool] = []
        lock_order = tuple(sorted(staged, key=lambda item: str(item.artifact_path)))
        commit_published = False
        intent: ArtifactPublicationIntent | None = None
        owned_paths_validated = False
        already_committed = False
        absolute_paths = {
            relative_path: path
            for item in staged
            for relative_path, path in (
                (item.materialization.relative_path, item.artifact_path),
                (item.manifest_materialization.relative_path, item.sidecar_path),
            )
        }
        component_sources = {
            relative_path: (destination, source, maximum_bytes)
            for item in staged
            for relative_path, destination, source, maximum_bytes in (
                (
                    item.materialization.relative_path,
                    item.artifact_path,
                    item.staged_artifact_path,
                    item.materialization.size_bytes,
                ),
                (
                    item.manifest_materialization.relative_path,
                    item.sidecar_path,
                    item.staged_sidecar_path,
                    item.manifest_materialization.size_bytes,
                ),
            )
        }
        component_identities = {
            value.relative_path: (value.size_bytes, value.physical_sha256)
            for item in staged
            for value in (item.materialization, item.manifest_materialization)
        }
        try:
            with ExitStack() as locks:
                locks.enter_context(self._publication_batch_lock(batch.commit_path))
                for item in lock_order:
                    locks.enter_context(self._pair_lock(item.artifact_path, item.sidecar_path))
                commit_exists = batch.commit_path.exists()
                if commit_exists and not self._file_equals(
                    batch.commit_path,
                    batch.commit_payload,
                ):
                    raise ArtifactIdentityConflict(
                        "immutable publication commit identity already exists"
                    )
                if batch.intent_path.exists():
                    intent = self._read_publication_intent(batch.intent_path)
                    if (
                        intent.publication_batch_id != batch.publication.publication_batch_id
                        or intent.members != batch.publication.members
                    ):
                        raise ArtifactIdentityConflict(
                            "publication intent belongs to another exact batch"
                        )

                if commit_exists:
                    for item in staged:
                        if not item.artifact_path.is_file() or not item.sidecar_path.is_file():
                            raise ArtifactIdentityConflict(
                                "committed artifact publication is incomplete"
                            )
                        try:
                            artifact_exact = self._component_is_exact_or_prefix(
                                item.artifact_path,
                                item.staged_artifact_path,
                                maximum_bytes=item.materialization.size_bytes,
                            )
                            sidecar_exact = self._component_is_exact_or_prefix(
                                item.sidecar_path,
                                item.staged_sidecar_path,
                                maximum_bytes=item.manifest_materialization.size_bytes,
                            )
                        except ArtifactIdentityConflict as error:
                            raise ArtifactIdentityConflict(
                                "committed artifact publication is incomplete"
                            ) from error
                        if not artifact_exact or not sidecar_exact:
                            raise ArtifactIdentityConflict(
                                "committed artifact publication is incomplete"
                            )
                        created_flags.append(False)
                    already_committed = True
                    if intent is not None:
                        self._remove_publication_intent(batch.intent_path)
                else:
                    components = tuple(
                        (
                            relative_path,
                            destination,
                            source,
                            maximum_bytes,
                        )
                        for item in staged
                        for relative_path, destination, source, maximum_bytes in (
                            (
                                item.materialization.relative_path,
                                item.artifact_path,
                                item.staged_artifact_path,
                                item.materialization.size_bytes,
                            ),
                            (
                                item.manifest_materialization.relative_path,
                                item.sidecar_path,
                                item.staged_sidecar_path,
                                item.manifest_materialization.size_bytes,
                            ),
                        )
                    )
                    if intent is None:
                        absent_relative_paths: list[str] = []
                        exact_paths: set[str] = set()
                        for relative_path, destination, source, maximum_bytes in components:
                            if not destination.exists():
                                absent_relative_paths.append(relative_path)
                                continue
                            try:
                                exact = self._component_is_exact_or_prefix(
                                    destination,
                                    source,
                                    maximum_bytes=maximum_bytes,
                                )
                            except ArtifactIdentityConflict as error:
                                raise ArtifactIdentityConflict(
                                    "immutable artifact identity already exists"
                                ) from error
                            if not exact:
                                raise ArtifactIdentityConflict(
                                    "immutable artifact identity already exists"
                                )
                            exact_paths.add(relative_path)
                        for item in staged:
                            if item.sidecar_path.exists() and not item.artifact_path.exists():
                                raise ArtifactIdentityConflict(
                                    "artifact publication has a manifest-first partial"
                                )
                            created_flags.append(
                                item.materialization.relative_path not in exact_paths
                                or item.manifest_materialization.relative_path not in exact_paths
                            )
                        intent = ArtifactPublicationIntent(
                            publication_batch_id=(batch.publication.publication_batch_id),
                            publication_scope=batch.publication.publication_scope,
                            members=batch.publication.members,
                            absent_relative_paths=tuple(sorted(absent_relative_paths)),
                        )
                        self._write_publication_intent(batch.intent_path, intent)
                        owned_paths_validated = True
                    else:
                        owned = set(intent.absent_relative_paths)
                        exact_paths = set()
                        prefix_paths: list[Path] = []
                        for relative_path, destination, source, maximum_bytes in components:
                            if not destination.exists():
                                if relative_path not in owned:
                                    raise ArtifactIdentityConflict(
                                        "pre-existing publication member disappeared"
                                    )
                                continue
                            exact = self._component_is_exact_or_prefix(
                                destination,
                                source,
                                maximum_bytes=maximum_bytes,
                            )
                            if relative_path not in owned:
                                if not exact:
                                    raise ArtifactIdentityConflict(
                                        "pre-existing publication member changed"
                                    )
                                exact_paths.add(relative_path)
                            elif exact:
                                exact_paths.add(relative_path)
                            else:
                                prefix_paths.append(destination)
                        for item in staged:
                            created_flags.append(
                                item.materialization.relative_path not in exact_paths
                                or item.manifest_materialization.relative_path not in exact_paths
                            )
                        owned_paths_validated = True
                        prefix_parents: set[Path] = set()
                        for path in prefix_paths:
                            self._unlink_if_present(path)
                            prefix_parents.add(path.parent)
                        for parent in sorted(prefix_parents, key=str):
                            self._fsync_directory(parent)

                    for relative_path, destination, source, maximum_bytes in components:
                        if destination.exists():
                            continue
                        if relative_path not in set(intent.absent_relative_paths):
                            raise ArtifactIdentityConflict(
                                "pre-existing publication member disappeared"
                            )
                        identity = component_identities[relative_path]
                        self._require_component_identity(source, identity)
                        # Both staged files have already been fsynced. Exclusive
                        # rename keeps the final path all-or-absent without a
                        # second write, fsync and write-mode close of each file.
                        self._move_file_once(source, destination)
                        self._require_component_identity(destination, identity)
                    parent_paths = {
                        path.parent
                        for item in staged
                        for path in (item.artifact_path, item.sidecar_path)
                    }
                    for parent in sorted(parent_paths, key=str):
                        self._fsync_directory(parent)
                    try:
                        self._rename_noreplace(
                            batch.staged_commit_path,
                            batch.commit_path,
                        )
                    except FileExistsError as error:
                        if not self._file_equals(batch.commit_path, batch.commit_payload):
                            raise ArtifactIdentityConflict(
                                "immutable publication commit identity raced"
                            ) from error
                    commit_published = True
                    self._fsync_directory(batch.commit_path.parent)
                    self._remove_publication_intent(batch.intent_path)
        except BaseException:
            if not commit_published and owned_paths_validated and intent is not None:
                rollback_parents: set[Path] = set()
                for relative_path in reversed(intent.absent_relative_paths):
                    path = absolute_paths[relative_path]
                    if path.exists():
                        _destination, source, maximum_bytes = component_sources[relative_path]
                        try:
                            if source.exists():
                                self._component_is_exact_or_prefix(
                                    path,
                                    source,
                                    maximum_bytes=maximum_bytes,
                                )
                            else:
                                # A completed rename consumed its staging path,
                                # including when a later operation raised. Delete
                                # only the exact content frozen before publication.
                                self._require_component_identity(
                                    path, component_identities[relative_path]
                                )
                        except ArtifactIdentityConflict:
                            # A noncooperating writer won the path with other
                            # bytes; preserve that evidence instead of deleting it.
                            continue
                        rollback_parents.add(path.parent)
                        self._unlink_if_present(path)
                for parent in sorted(rollback_parents, key=str):
                    self._fsync_directory(parent)
                self._remove_publication_intent(batch.intent_path)
            raise
        if already_committed:
            commit_published = True
        return tuple(
            ArtifactWriteResult(
                logical=item.logical,
                materialization=item.materialization,
                manifest_materialization=item.manifest_materialization,
                created=created,
            )
            for item, created in zip(staged, created_flags, strict=True)
        )

    @staticmethod
    def _component_is_exact_or_prefix(
        candidate: Path,
        expected: Path,
        *,
        maximum_bytes: int,
    ) -> bool:
        """Return exactness after proving candidate is a stable expected prefix."""

        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        candidate_descriptor: int | None = None
        expected_descriptor: int | None = None
        try:
            candidate_descriptor = os.open(candidate, flags)
            expected_descriptor = os.open(expected, flags)
            candidate_before = os.fstat(candidate_descriptor)
            expected_before = os.fstat(expected_descriptor)
            if (
                not stat.S_ISREG(candidate_before.st_mode)
                or not stat.S_ISREG(expected_before.st_mode)
                or expected_before.st_size != maximum_bytes
                or candidate_before.st_size > maximum_bytes
            ):
                raise ArtifactIdentityConflict(
                    "batch-owned partial differs from its expected prefix"
                )
            remaining = candidate_before.st_size
            while remaining:
                candidate_chunk = os.read(
                    candidate_descriptor,
                    min(STREAM_CHUNK_BYTES, remaining),
                )
                expected_chunk = os.read(expected_descriptor, len(candidate_chunk))
                if not candidate_chunk or candidate_chunk != expected_chunk:
                    raise ArtifactIdentityConflict(
                        "batch-owned partial differs from its expected prefix"
                    )
                remaining -= len(candidate_chunk)
            candidate_after = os.fstat(candidate_descriptor)
            expected_after = os.fstat(expected_descriptor)
            candidate_identity = (
                candidate_before.st_dev,
                candidate_before.st_ino,
                candidate_before.st_size,
                candidate_before.st_mtime_ns,
                candidate_before.st_ctime_ns,
            )
            expected_identity = (
                expected_before.st_dev,
                expected_before.st_ino,
                expected_before.st_size,
                expected_before.st_mtime_ns,
                expected_before.st_ctime_ns,
            )
            if candidate_identity != (
                candidate_after.st_dev,
                candidate_after.st_ino,
                candidate_after.st_size,
                candidate_after.st_mtime_ns,
                candidate_after.st_ctime_ns,
            ) or expected_identity != (
                expected_after.st_dev,
                expected_after.st_ino,
                expected_after.st_size,
                expected_after.st_mtime_ns,
                expected_after.st_ctime_ns,
            ):
                raise ArtifactIdentityConflict(
                    "publication component changed during prefix verification"
                )
            return candidate_before.st_size == maximum_bytes
        except OSError as error:
            raise ArtifactIdentityConflict(
                "publication component cannot be verified as a stable prefix"
            ) from error
        finally:
            if expected_descriptor is not None:
                os.close(expected_descriptor)
            if candidate_descriptor is not None:
                os.close(candidate_descriptor)

    @staticmethod
    def _read_publication_intent(path: Path) -> ArtifactPublicationIntent:
        try:
            payload = read_bounded_bytes(
                path,
                maximum_bytes=MAX_ARTIFACT_PUBLICATION_INTENT_BYTES,
            )
            from .task_receipts import decode_artifact_publication_intent

            return decode_artifact_publication_intent(payload)
        except (BoundedFileIOError, OSError, ValueError) as error:
            raise ArtifactIdentityConflict("artifact publication intent is invalid") from error

    @staticmethod
    def _write_publication_intent(
        path: Path,
        intent: ArtifactPublicationIntent,
    ) -> None:
        payload = intent.canonical_bytes()
        if len(payload) > MAX_ARTIFACT_PUBLICATION_INTENT_BYTES:
            raise ArtifactIdentityConflict("artifact publication intent exceeds its byte bound")
        token = hashlib.sha256(
            f"{os.getpid()}\0{threading.get_ident()}\0{time.time_ns()}".encode()
        ).hexdigest()[:24]
        staged_path = path.with_name(f".{path.name}.{token}.partial")
        try:
            ExternalArtifactPlane._write_staged_bytes(staged_path, payload)
            try:
                ExternalArtifactPlane._rename_noreplace(staged_path, path)
            except FileExistsError as error:
                raise ArtifactIdentityConflict(
                    "artifact publication intent already exists"
                ) from error
            ExternalArtifactPlane._fsync_directory(path.parent)
        finally:
            ExternalArtifactPlane._unlink_if_present(staged_path)

    @staticmethod
    def _remove_publication_intent(path: Path) -> None:
        if path.exists():
            path.unlink()
            ExternalArtifactPlane._fsync_directory(path.parent)

    @staticmethod
    def _rename_noreplace(source: Path, destination: Path) -> None:
        """Atomically publish a staged control file without replacement.

        Linux ``renameat2(RENAME_NOREPLACE)`` is supported by the production
        vfat mount through the VFS.  Failing closed is safer than falling back
        to a check-then-replace sequence on an unsupported platform.
        """

        libc = ctypes.CDLL(None, use_errno=True)
        try:
            renameat2 = libc.renameat2
        except AttributeError as error:
            raise ArtifactIdentityConflict(
                "exclusive atomic control publication is unavailable"
            ) from error
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        result = int(
            renameat2(
                -100,
                os.fsencode(source),
                -100,
                os.fsencode(destination),
                1,
            )
        )
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(error_number, os.strerror(error_number), destination)
        raise ArtifactIdentityConflict("exclusive atomic control publication failed") from OSError(
            error_number, os.strerror(error_number), destination
        )

    @staticmethod
    def _require_component_identity(path: Path, identity: tuple[int, str]) -> None:
        try:
            observed = ExternalArtifactPlane._stream_identity(path, maximum_bytes=identity[0])
        except OSError as error:
            raise ArtifactIdentityConflict(
                "publication component identity is unavailable"
            ) from error
        if observed != identity:
            raise ArtifactIdentityConflict("artifact identity changed during batch publication")

    @staticmethod
    def _move_file_once(source: Path, destination: Path) -> None:
        ExternalArtifactPlane._rename_noreplace(source, destination)

    @staticmethod
    def _file_equals(path: Path, expected: bytes) -> bool:
        try:
            return file_matches_bytes(path, expected)
        except (BoundedFileIOError, OSError):
            return False

    @staticmethod
    def _unlink_if_present(path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def open(
        self,
        value: VerifiedArtifactInput,
        *,
        binding: WorkerInputBinding | None = None,
        maximum_bytes: int,
    ) -> WorkerInputPort:
        return self.open_many(
            (value,),
            bindings=(
                binding or WorkerInputBinding.from_verified(value, kind=WorkerInputKind.EXTERNAL),
            ),
            maximum_bytes=maximum_bytes,
        )[0]

    def open_many(
        self,
        values: tuple[VerifiedArtifactInput, ...],
        *,
        bindings: tuple[WorkerInputBinding, ...],
        maximum_bytes: int,
    ) -> tuple[WorkerInputPort, ...]:
        """Authenticate each batch once and seal the bytes actually given to workers."""
        if maximum_bytes < 0 or len(values) != len(bindings):
            raise ValueError("worker input count or budget is invalid")
        if sum(value.materialization.size_bytes for value in values) > maximum_bytes:
            raise ArtifactInputLimitExceeded(
                "frozen input size exceeds the worker input-scan budget"
            )
        requested: dict[str, tuple[WorkerInputBinding, int]] = {}
        for value, binding in zip(values, bindings, strict=True):
            if binding != WorkerInputBinding.from_verified(value, kind=binding.kind):
                raise ArtifactIdentityConflict(
                    "worker input binding differs from the verified materialization"
                )
            path = value.materialization.relative_path
            if path in requested:
                raise ArtifactIdentityConflict("worker input materialization is duplicated")
            requested[path] = (
                binding,
                maximum_bytes if len(values) == 1 else value.materialization.size_bytes,
            )
        ports = self._verify_manifests(
            tuple(
                ArtifactManifest(logical=value.logical, materialization=value.materialization)
                for value in values
            ),
            requested,
        )
        return tuple(ports[value.materialization.relative_path] for value in values)

    def recover_partial(self, request: ArtifactWriteRequest) -> ArtifactWriteResult:
        """Complete an exact uncommitted partial through the batch protocol."""

        return self.write(request)

    def _prepare(
        self,
        request: ArtifactWriteRequest,
    ) -> tuple[
        LogicalArtifactIdentity,
        ArtifactMaterialization,
        ArtifactMaterialization,
        bytes,
    ]:
        if request.read_only_source_input:
            raise ArtifactIdentityConflict("historical artifact contracts are read-only")
        if request.compression != "none" or request.partition_selector is not None:
            raise ArtifactIdentityConflict(
                "compressed or partitioned artifacts require an equivalence proof contract"
            )
        physical_sha256 = _sha256(request.payload)
        if (
            request.logical_content_sha256 is not None
            and request.logical_content_sha256 != physical_sha256
        ):
            raise ArtifactIdentityConflict(
                "logical content identity lacks a verified equivalence proof"
            )
        self.validators.validate(request)
        return self._prepare_identity(
            request,
            physical_sha256=physical_sha256,
            size_bytes=len(request.payload),
        )

    def _prepare_identity(
        self,
        request: ArtifactWriteRequest | ArtifactStreamWriteRequest,
        *,
        physical_sha256: str,
        size_bytes: int,
    ) -> tuple[
        LogicalArtifactIdentity,
        ArtifactMaterialization,
        ArtifactMaterialization,
        bytes,
    ]:
        if (
            request.logical_content_sha256 is not None
            and request.logical_content_sha256 != physical_sha256
        ):
            raise ArtifactIdentityConflict(
                "logical content identity lacks a verified equivalence proof"
            )
        logical_sha256 = physical_sha256
        logical = LogicalArtifactIdentity(
            logical_artifact_id=request.logical_artifact_id,
            content_sha256=logical_sha256,
            payload_schema=request.payload_schema,
            profile=request.profile,
            media_type=request.media_type,
            visibility_ceiling=request.visibility_ceiling,
            parent_visibility_ceilings=request.parent_visibility_ceilings,
            outcome_access=request.outcome_access,
            generic_validation=self.validators.generic_validation(
                profile=request.profile,
                payload_schema=request.payload_schema,
            ),
            lineage_parents=request.lineage_parents,
            semantic_validation=request.semantic_validation,
        )
        materialization = ArtifactMaterialization(
            materialization_id=_materialization_id(
                request.logical_artifact_id,
                request.relative_path,
                physical_sha256,
            ),
            logical_artifact_id=request.logical_artifact_id,
            storage_root_id=self.root.contract.storage_root_id,
            relative_path=request.relative_path,
            physical_sha256=physical_sha256,
            size_bytes=size_bytes,
            compression=request.compression,
            partition_selector=request.partition_selector,
        )
        manifest_payload = canonical_json_bytes(
            ArtifactManifest(logical=logical, materialization=materialization)
        )
        manifest_path = f"{request.relative_path}.manifest.json"
        manifest_materialization = self._manifest_materialization(
            logical_artifact_id=request.logical_artifact_id,
            relative_path=manifest_path,
            payload=manifest_payload,
        )
        return (
            logical,
            materialization,
            manifest_materialization,
            manifest_payload,
        )

    def _manifest_materialization(
        self,
        *,
        logical_artifact_id: str,
        relative_path: str,
        payload: bytes,
    ) -> ArtifactMaterialization:
        manifest_logical_id = f"{logical_artifact_id}-manifest"
        physical_sha256 = _sha256(payload)
        return ArtifactMaterialization(
            materialization_id=_materialization_id(
                manifest_logical_id,
                relative_path,
                physical_sha256,
            ),
            logical_artifact_id=manifest_logical_id,
            storage_root_id=self.root.contract.storage_root_id,
            relative_path=relative_path,
            physical_sha256=physical_sha256,
            size_bytes=len(payload),
            compression="none",
        )

    def verify(self, materialization: ArtifactMaterialization) -> None:
        if materialization.storage_root_id != self.root.contract.storage_root_id:
            raise ArtifactIdentityConflict("artifact belongs to another storage root")
        path = self.root.resolve(materialization.relative_path, for_write=False)
        if not path.is_file():
            raise ArtifactIdentityConflict("artifact path is unavailable")
        size_bytes, physical_sha256 = self._stream_identity(
            path,
            maximum_bytes=materialization.size_bytes,
        )
        if (
            size_bytes != materialization.size_bytes
            or physical_sha256 != materialization.physical_sha256
        ):
            raise ArtifactIdentityConflict("artifact bytes differ from their identity")

    def verify_manifest(self, manifest: ArtifactManifest) -> None:
        self.verify_manifests((manifest,))

    def verify_manifests(self, manifests: tuple[ArtifactManifest, ...]) -> None:
        if len(manifests) > MAX_ARTIFACT_PUBLICATION_MEMBERS:
            raise ArtifactIdentityConflict("artifact verification batch exceeds its bound")
        self._verify_manifests(manifests, {})

    def _verify_manifests(
        self,
        manifests: tuple[ArtifactManifest, ...],
        requested: dict[str, tuple[WorkerInputBinding, int]],
    ) -> dict[str, WorkerInputPort]:
        """Verify each committed sibling set once within a stable read operation.

        Physical bytes and semantics are checked afresh on every invocation.
        Snapshot guards span all members, sidecars and commit markers; no
        verification or authority decision survives the operation.
        """

        verified_batches: set[str] = set()
        snapshots: dict[Path, tuple[int, int, int, int, int]] = {}
        ports: dict[str, WorkerInputPort] = {}
        complete = False
        try:
            for manifest in manifests:
                self._verify_manifest_in_batch(
                    manifest, verified_batches, snapshots, requested, ports
                )
            for path, identity in snapshots.items():
                if self._descriptor_identity(os.stat(path, follow_symlinks=False)) != identity:
                    raise ArtifactIdentityConflict("artifact batch changed during verification")
            if ports.keys() != requested.keys():
                raise ArtifactIdentityConflict("worker input was absent from its committed batch")
            complete = True
            return ports
        except OSError as error:
            raise ArtifactIdentityConflict("artifact batch snapshot became unavailable") from error
        finally:
            if not complete:
                for port in ports.values():
                    port.close()

    def _track_verification_snapshot(
        self, path: Path, snapshots: dict[Path, tuple[int, int, int, int, int]]
    ) -> None:
        observed = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(observed.st_mode):
            raise ArtifactIdentityConflict("artifact verification snapshot is not a regular file")
        identity = self._descriptor_identity(observed)
        if snapshots.setdefault(path, identity) != identity:
            raise ArtifactIdentityConflict("artifact batch changed during verification")

    def _verify_manifest_in_batch(
        self,
        manifest: ArtifactManifest,
        verified_batches: set[str],
        snapshots: dict[Path, tuple[int, int, int, int, int]],
        requested: dict[str, tuple[WorkerInputBinding, int]],
        ports: dict[str, WorkerInputPort],
    ) -> None:
        sidecar = self.root.resolve(
            f"{manifest.materialization.relative_path}.manifest.json",
            for_write=False,
        )
        if not sidecar.is_file():
            raise ArtifactIdentityConflict("artifact manifest sidecar is unavailable")
        self._track_verification_snapshot(sidecar, snapshots)
        persisted = self._read_current_manifest(sidecar)
        if (
            persisted.logical != manifest.logical
            or persisted.materialization != manifest.materialization
            or (manifest.publication is not None and persisted.publication != manifest.publication)
        ):
            raise ArtifactIdentityConflict("artifact manifest differs from its sidecar")
        publication = persisted.publication
        if publication is None:
            raise ArtifactIdentityConflict(
                "compatibility manifest requires a qualified current publication"
            )
        batch_key = publication.fingerprint()
        if batch_key in verified_batches:
            return
        expected_commit = ArtifactPublicationCommit(
            publication_batch_id=publication.publication_batch_id,
            publication_scope=publication.publication_scope,
            members=publication.members,
        ).canonical_bytes()
        if (
            len(expected_commit) > MAX_ARTIFACT_PUBLICATION_MARKER_BYTES
            or len(expected_commit) != publication.commit_marker_size_bytes
            or hashlib.sha256(expected_commit).hexdigest() != publication.commit_marker_sha256
        ):
            raise ArtifactIdentityConflict("artifact publication binding differs from its commit")
        commit_path = self.root.resolve(
            publication.commit_marker_relative_path,
            for_write=False,
        )
        try:
            self._track_verification_snapshot(commit_path, snapshots)
            committed = file_matches_bytes(commit_path, expected_commit)
        except (BoundedFileIOError, OSError) as error:
            raise ArtifactIdentityConflict(
                "artifact publication is not durably committed"
            ) from error
        if not committed:
            raise ArtifactIdentityConflict("artifact publication is not durably committed")

        # A member is valid only if the exact committed batch remains complete.
        for member in publication.members:
            sibling_sidecar = self.root.resolve(
                f"{member.relative_path}.manifest.json",
                for_write=False,
            )
            self._track_verification_snapshot(sibling_sidecar, snapshots)
            sibling = self._read_current_manifest(sibling_sidecar)
            if (
                sibling.publication != publication
                or artifact_publication_member(sibling.logical, sibling.materialization) != member
            ):
                raise ArtifactIdentityConflict(
                    "artifact publication sibling differs from its committed member set"
                )
            self._track_verification_snapshot(
                self.root.resolve(member.relative_path, for_write=False), snapshots
            )
            if member.relative_path in requested:
                binding, limit = requested[member.relative_path]
                port = self._verify_manifest_snapshot(sibling, binding=binding, maximum_bytes=limit)
                assert port is not None
                ports[member.relative_path] = port
            else:
                self._verify_manifest_snapshot(sibling)
        verified_batches.add(batch_key)

    @staticmethod
    def _descriptor_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    @staticmethod
    def _read_current_manifest(path: Path) -> ArtifactManifest:
        try:
            sidecar_payload = read_bounded_bytes(
                path,
                maximum_bytes=MAX_ARTIFACT_MANIFEST_BYTES,
            )
            # Imported lazily because the receipt store composes this plane.
            from .task_receipts import decode_artifact_manifest

            persisted = decode_artifact_manifest(sidecar_payload)
        except (BoundedFileIOError, OSError, ValueError) as error:
            raise ArtifactIdentityConflict("artifact manifest sidecar is invalid") from error
        if persisted.publication is None:
            raise ArtifactIdentityConflict(
                "compatibility manifest requires a qualified current publication"
            )
        return persisted

    def _verify_manifest_snapshot(
        self,
        manifest: ArtifactManifest,
        *,
        binding: WorkerInputBinding | None = None,
        maximum_bytes: int = 0,
    ) -> WorkerInputPort | None:
        materialization = manifest.materialization
        if materialization.storage_root_id != self.root.contract.storage_root_id:
            raise ArtifactIdentityConflict("artifact belongs to another storage root")
        if manifest.publication is None:
            raise ArtifactIdentityConflict("qualified artifact lacks a publication binding")
        if manifest.logical.content_sha256 != materialization.physical_sha256:
            raise ArtifactIdentityConflict(
                "logical content identity lacks a verified equivalence proof"
            )
        if materialization.compression != "none" or materialization.partition_selector is not None:
            raise ArtifactIdentityConflict(
                "compressed or partitioned artifacts lack an equivalence proof contract"
            )
        generic_validation = manifest.logical.generic_validation
        if generic_validation is None:
            raise ArtifactIdentityConflict(
                "compatibility manifest lacks a qualified generic validator registration"
            )
        if (
            generic_validation.profile is not manifest.logical.profile
            or generic_validation.payload_schema != manifest.logical.payload_schema
        ):
            raise ArtifactIdentityConflict(
                "artifact generic validation differs from its logical identity"
            )
        self.validators.require_generic_validation(generic_validation)
        self._require_registered_semantics(
            manifest.logical.logical_artifact_id,
            manifest.logical.semantic_validation,
        )
        path = self.root.resolve(materialization.relative_path, for_write=False)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as error:
            raise ArtifactIdentityConflict("artifact path is unavailable") from error
        anonymous: int | None = None
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ArtifactIdentityConflict("artifact type differs from identity")
            if before.st_size != materialization.size_bytes:
                raise ArtifactIdentityConflict("artifact bytes differ from their identity")
            digest = hashlib.sha256()
            observed_bytes = 0
            if binding is not None:
                anonymous = _anonymous_input_descriptor()
            while True:
                chunk = os.read(descriptor, STREAM_CHUNK_BYTES)
                if not chunk:
                    break
                observed_bytes += len(chunk)
                if observed_bytes > materialization.size_bytes:
                    raise ArtifactIdentityConflict("artifact identity scan exceeded its bound")
                digest.update(chunk)
                if anonymous is not None:
                    view = memoryview(chunk)
                    while view:
                        written = os.write(anonymous, view)
                        if written <= 0:
                            raise OSError("worker input copy made no progress")
                        view = view[written:]
            if (
                observed_bytes != materialization.size_bytes
                or digest.hexdigest() != materialization.physical_sha256
            ):
                raise ArtifactIdentityConflict("artifact bytes differ from their identity")
            if anonymous is not None:
                os.lseek(anonymous, 0, os.SEEK_SET)
                _seal_anonymous_input(anonymous)
            descriptor_path = Path(
                f"/proc/self/fd/{descriptor if anonymous is None else anonymous}"
            )
            logical = manifest.logical
            self.validators.validate_file(
                descriptor_path,
                ArtifactStreamWriteRequest(
                    logical_artifact_id=logical.logical_artifact_id,
                    relative_path=materialization.relative_path,
                    payload_schema=logical.payload_schema,
                    profile=logical.profile,
                    media_type=logical.media_type,
                    publication_scope_id=(
                        manifest.publication.publication_scope.publication_scope_id
                    ),
                    publication_scope_relative_root=(
                        manifest.publication.publication_scope.relative_root
                    ),
                    chunks=(),
                    maximum_bytes=max(1, materialization.size_bytes),
                    maximum_chunk_bytes=min(
                        STREAM_CHUNK_BYTES,
                        max(1, materialization.size_bytes),
                    ),
                    visibility_ceiling=logical.visibility_ceiling,
                    parent_visibility_ceilings=logical.parent_visibility_ceilings,
                    outcome_access=logical.outcome_access,
                    compression=materialization.compression,
                    partition_selector=materialization.partition_selector,
                    lineage_parents=logical.lineage_parents,
                    minimum_free_bytes=0,
                    semantic_validation=logical.semantic_validation,
                ),
            )
            after = os.fstat(descriptor)
            named = os.stat(path, follow_symlinks=False)
            if self._descriptor_identity(before) != self._descriptor_identity(after) or (
                named.st_dev,
                named.st_ino,
            ) != (before.st_dev, before.st_ino):
                raise ArtifactIdentityConflict(
                    "artifact changed or was replaced during semantic verification"
                )
            if anonymous is not None:
                assert binding is not None
                port = WorkerInputPort(
                    binding=binding,
                    _reader=_PreopenedBoundedReader(anonymous, maximum_bytes=maximum_bytes),
                )
                anonymous = None
                return port
            return None
        except OSError as error:
            raise ArtifactIdentityConflict("artifact snapshot verification failed") from error
        finally:
            os.close(descriptor)
            if anonymous is not None:
                os.close(anonymous)

    @staticmethod
    def _stream_identity(path: Path, *, maximum_bytes: int) -> tuple[int, str]:
        try:
            return bounded_file_sha256(path, maximum_bytes=maximum_bytes)
        except BoundedFileIOError as error:
            raise ArtifactIdentityConflict("artifact identity scan exceeded its bound") from error

    def _write_pair_once(
        self,
        artifact_path: Path,
        artifact_payload: bytes,
        manifest_path: Path,
        manifest_payload: bytes,
    ) -> bool:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with self._pair_lock(artifact_path, manifest_path):
            artifact_exists = artifact_path.exists()
            manifest_exists = manifest_path.exists()
            if artifact_exists or manifest_exists:
                if artifact_exists != manifest_exists:
                    raise ArtifactIdentityConflict("artifact/manifest pair is partial")
                try:
                    matches = file_matches_bytes(
                        artifact_path,
                        artifact_payload,
                    ) and file_matches_bytes(manifest_path, manifest_payload)
                except BoundedFileIOError as error:
                    raise ArtifactIdentityConflict(
                        "immutable artifact identity comparison failed"
                    ) from error
                if not matches:
                    raise ArtifactIdentityConflict("immutable artifact identity already exists")
                return False
            self._write_atomic(artifact_path, artifact_payload)
            try:
                self._write_atomic(manifest_path, manifest_payload)
            except Exception:
                # Preserve the partial artifact for identity-checked recovery.
                raise
            self._fsync_directory(artifact_path.parent)
            if manifest_path.parent != artifact_path.parent:
                self._fsync_directory(manifest_path.parent)
            return True

    @staticmethod
    @contextmanager
    def _pair_lock(artifact_path: Path, manifest_path: Path):  # type: ignore[no-untyped-def]
        token = hashlib.sha256(
            f"{artifact_path.name}\0{manifest_path.name}".encode("utf-8")
        ).hexdigest()[:24]
        lock_path = artifact_path.parent / f".artifact-pair.{token}.lock"
        with exclusive_file_lock(lock_path):
            yield

    @staticmethod
    @contextmanager
    def _publication_batch_lock(commit_path: Path):  # type: ignore[no-untyped-def]
        token = hashlib.sha256(str(commit_path).encode("utf-8")).hexdigest()[:24]
        lock_path = commit_path.parent / f".publication-batch.{token}.lock"
        with exclusive_file_lock(lock_path):
            yield

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _write_atomic(path: Path, payload: bytes) -> None:
        token = f"{os.getpid()}.{threading.get_ident()}"
        temporary = path.with_name(f".{path.name}.{token}.partial")
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            stat.S_IRUSR | stat.S_IWUSR,
        )
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except Exception:
            if temporary.exists():
                temporary.unlink()
            raise
