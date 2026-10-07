"""Guarded filesystem implementations for held VCC bytes and task scratch.

The scientific runners depend only on the protocols in :mod:`ports`.  These
implementations are injected at the API composition boundary and accept one
exact source manifest/root; callers never supply an arbitrary path.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, closing
import hashlib
import os
from pathlib import Path
import stat
from typing import BinaryIO, cast

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.serialization import validate_stable_id

from .contracts import (
    VirtualCellContractError,
    VirtualCellSourceManifest,
    VirtualCellSourceObject,
)
from .dataset import SegmentedBinaryReader
from .ports import VirtualCellScratchWorkspace


_HASH_CHUNK_BYTES = 8 * 1024**2
_MAXIMUM_SCRATCH_ENTRIES = 10_000
_ALLOWED_READ_MODES = frozenset({"rb", "r+b"})


def _require_plain_name(value: str, *, field_name: str) -> None:
    if (
        not value
        or len(value.encode("utf-8")) > 255
        or value in {".", ".."}
        or Path(value).name != value
        or "/" in value
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError(f"{field_name} must be one bounded plain filename")


def _existing_ancestor(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        if candidate.parent == candidate:
            raise VirtualCellContractError("external path has no existing ancestor")
        candidate = candidate.parent
    return candidate


class ExternalStoragePreflight:
    """Validate one active external mount and its authorized subtree."""

    def __init__(
        self,
        *,
        mount_root: Path,
        authorized_root: Path,
        require_mount: bool = True,
    ) -> None:
        self.mount_root = mount_root.absolute()
        self.authorized_root = authorized_root.absolute()
        self.require_mount = require_mount

    @staticmethod
    def _require_directory(path: Path) -> None:
        observed = path.lstat()
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
            raise VirtualCellContractError("external storage component is not a real directory")

    def _require_chain(self, path: Path) -> None:
        current = self.mount_root
        self._require_directory(current)
        try:
            relative = path.relative_to(self.mount_root)
        except ValueError as error:
            raise VirtualCellContractError("external path escapes the active mount") from error
        for component in relative.parts:
            current = current / component
            if current.exists() or current.is_symlink():
                self._require_directory(current)

    def validate(
        self,
        path: Path,
        *,
        writable: bool,
        minimum_free_bytes: int,
    ) -> None:
        if isinstance(minimum_free_bytes, bool) or minimum_free_bytes < 0:
            raise ValueError("minimum_free_bytes must be nonnegative")
        if not self.mount_root.is_absolute() or not self.authorized_root.is_absolute():
            raise VirtualCellContractError("external storage roots must be absolute")
        if self.require_mount and not os.path.ismount(self.mount_root):
            raise VirtualCellContractError("required external mount is not active")
        self._require_chain(self.authorized_root)
        try:
            path.relative_to(self.authorized_root)
        except ValueError as error:
            raise VirtualCellContractError("external path escapes its authorized root") from error
        ancestor = _existing_ancestor(path)
        self._require_chain(ancestor)
        if ancestor.resolve(strict=True) != ancestor:
            raise VirtualCellContractError("external path resolves through a symlink")
        if writable and not os.access(ancestor, os.W_OK | os.X_OK):
            raise VirtualCellContractError("external destination is not writable")
        stats = os.statvfs(ancestor)
        available = stats.f_bavail * stats.f_frsize
        if available < minimum_free_bytes:
            raise VirtualCellContractError("external destination lacks the required free space")


class HeldVirtualCellSourcePort:
    """Read exactly the ordered parts named by one custody-qualified manifest."""

    def __init__(
        self,
        *,
        source_root: Path,
        manifest: VirtualCellSourceManifest,
        preflight: ExternalStoragePreflight,
    ) -> None:
        self._source_root = source_root.absolute()
        self._manifest = manifest
        self._preflight = preflight
        self._objects = {value.object_id: value for value in manifest.objects}
        self._verified: set[str] = set()
        preflight.validate(self._source_root, writable=False, minimum_free_bytes=0)
        observed = self._source_root.lstat()
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
            raise VirtualCellContractError("held VCC source root is not a real directory")

    def _require_registered(self, source: VirtualCellSourceObject) -> None:
        if self._objects.get(source.object_id) != source:
            raise VirtualCellContractError("source object is not bound by the held manifest")

    def _part_path(self, relative_locator: str) -> Path:
        path = self._source_root.joinpath(*relative_locator.split("/")).absolute()
        try:
            path.relative_to(self._source_root)
        except ValueError as error:
            raise VirtualCellContractError("source part escapes its exact custody root") from error
        current = self._source_root
        for component in Path(relative_locator).parts:
            current = current / component
            observed = current.lstat()
            if stat.S_ISLNK(observed.st_mode):
                raise VirtualCellContractError("source part resolves through a symlink")
        observed = path.lstat()
        if not stat.S_ISREG(observed.st_mode):
            raise VirtualCellContractError("source part is not a regular file")
        return path

    def verify_available(self, source: VirtualCellSourceObject) -> None:
        self._require_registered(source)
        identity = source.fingerprint()
        if identity in self._verified:
            return
        object_digest = hashlib.sha256()
        object_size = 0
        for part in source.parts:
            path = self._part_path(part.relative_locator)
            observed_size = path.stat().st_size
            if observed_size != part.size_bytes:
                raise VirtualCellContractError("source part byte count differs from custody")
            part_digest = hashlib.sha256()
            with path.open("rb") as stream:
                while payload := stream.read(_HASH_CHUNK_BYTES):
                    part_digest.update(payload)
                    object_digest.update(payload)
                    object_size += len(payload)
            if part_digest.hexdigest() != part.sha256:
                raise VirtualCellContractError("source part SHA-256 differs from custody")
        if object_size != source.size_bytes or object_digest.hexdigest() != source.sha256:
            raise VirtualCellContractError("source object identity differs from custody")
        self._verified.add(identity)

    @staticmethod
    def _authorize_open(source: VirtualCellSourceObject, access: OutcomeAccess) -> None:
        if source.sealed:
            if access is not OutcomeAccess.EVALUATOR_REVEAL:
                raise PermissionError("sealed VCC test responses are evaluator-only")
            return
        if "response/development" in source.role:
            if access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
                raise PermissionError("development responses require development visibility")
            return
        if access is not OutcomeAccess.OUTCOME_BLIND:
            raise PermissionError("VCC prefix/registry objects require outcome-blind access")

    def open_object(
        self,
        source: VirtualCellSourceObject,
        *,
        outcome_access: OutcomeAccess,
    ) -> AbstractContextManager[BinaryIO]:
        self._require_registered(source)
        self._authorize_open(source, outcome_access)
        self.verify_available(source)
        streams: list[BinaryIO] = []
        try:
            for part in source.parts:
                streams.append(self._part_path(part.relative_locator).open("rb"))
            reader = SegmentedBinaryReader(
                tuple(streams),
                tuple(value.size_bytes for value in source.parts),
            )
        except Exception:
            for stream in streams:
                stream.close()
            raise
        return closing(cast(BinaryIO, reader))


class ExternalVirtualCellScratchWorkspace:
    """One exclusive, fail-closed task workspace below external scratch."""

    def __init__(self, *, workspace: Path, scratch_root: Path) -> None:
        self._workspace = workspace
        self._scratch_root = scratch_root
        self._reserved: set[Path] = set()
        self._closed = False

    @property
    def workspace_id(self) -> str:
        return self._workspace.name

    def reserve_path(self, filename: str) -> Path:
        if self._closed:
            raise ValueError("scratch workspace is closed")
        _require_plain_name(filename, field_name="filename")
        path = self._workspace / filename
        if path in self._reserved or path.exists() or path.is_symlink():
            raise VirtualCellContractError("scratch destination already exists or is reserved")
        self._reserved.add(path)
        return path

    def open_existing(self, filename: str, mode: str) -> BinaryIO:
        if self._closed:
            raise ValueError("scratch workspace is closed")
        _require_plain_name(filename, field_name="filename")
        if mode not in _ALLOWED_READ_MODES:
            raise ValueError("scratch existing-file mode is not permitted")
        path = self._workspace / filename
        if path not in self._reserved:
            raise VirtualCellContractError("scratch path was not reserved by this workspace")
        observed = path.lstat()
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
            raise VirtualCellContractError("scratch path is not a regular file")
        return cast(BinaryIO, path.open(mode))

    def _remove_tree(self, path: Path, counter: list[int]) -> None:
        try:
            observed = path.lstat()
        except FileNotFoundError:
            return
        counter[0] += 1
        if counter[0] > _MAXIMUM_SCRATCH_ENTRIES:
            raise VirtualCellContractError("scratch cleanup exceeds its bounded entry count")
        if stat.S_ISLNK(observed.st_mode):
            raise VirtualCellContractError("scratch cleanup refuses a symlink")
        if stat.S_ISDIR(observed.st_mode):
            for child in sorted(path.iterdir(), key=lambda value: value.name):
                self._remove_tree(child, counter)
            path.rmdir()
        elif stat.S_ISREG(observed.st_mode):
            path.unlink()
        else:
            raise VirtualCellContractError("scratch cleanup refuses a special file")

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._workspace.relative_to(self._scratch_root)
        except ValueError as error:
            raise VirtualCellContractError("scratch workspace escaped its root") from error
        counter = [0]
        for path in sorted(self._reserved, key=lambda value: value.name):
            self._remove_tree(path, counter)
        for child in sorted(self._workspace.iterdir(), key=lambda value: value.name):
            self._remove_tree(child, counter)
        self._workspace.rmdir()
        parent = self._workspace.parent
        while parent != self._scratch_root:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent
        self._closed = True

    def __enter__(self) -> ExternalVirtualCellScratchWorkspace:
        return self

    def __exit__(self, *_error: object) -> None:
        self.close()


class ExternalVirtualCellScratchPort:
    """Allocate exact external workspaces with no repository/home/tmp fallback."""

    def __init__(
        self,
        *,
        scratch_root: Path,
        preflight: ExternalStoragePreflight,
        minimum_free_bytes: int,
    ) -> None:
        if isinstance(minimum_free_bytes, bool) or minimum_free_bytes <= 0:
            raise ValueError("minimum scratch free-space envelope must be positive")
        self._scratch_root = scratch_root.absolute()
        self._preflight = preflight
        self._minimum_free_bytes = minimum_free_bytes

    @staticmethod
    def _component(value: str, *, field_name: str) -> str:
        validate_stable_id(value, field_name=field_name)
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]

    def allocate(
        self,
        *,
        run_id: str,
        task_id: str,
        attempt_id: str,
    ) -> VirtualCellScratchWorkspace:
        self._preflight.validate(
            self._scratch_root,
            writable=True,
            minimum_free_bytes=self._minimum_free_bytes,
        )
        parent = self._scratch_root.parent
        self._scratch_root.mkdir(exist_ok=True)
        if self._scratch_root.parent != parent:
            raise VirtualCellContractError("scratch root parent changed during allocation")
        observed = self._scratch_root.lstat()
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
            raise VirtualCellContractError("scratch root is not a real directory")
        components = (
            self._component(run_id, field_name="run_id"),
            self._component(task_id, field_name="task_id"),
            self._component(attempt_id, field_name="attempt_id"),
        )
        current = self._scratch_root
        for component in components[:-1]:
            current = current / component
            current.mkdir(exist_ok=True)
            if stat.S_ISLNK(current.lstat().st_mode):
                raise VirtualCellContractError("scratch path resolves through a symlink")
        workspace = current / components[-1]
        workspace.mkdir()
        return ExternalVirtualCellScratchWorkspace(
            workspace=workspace,
            scratch_root=self._scratch_root,
        )


__all__ = [
    "ExternalStoragePreflight",
    "ExternalVirtualCellScratchPort",
    "ExternalVirtualCellScratchWorkspace",
    "HeldVirtualCellSourcePort",
]
