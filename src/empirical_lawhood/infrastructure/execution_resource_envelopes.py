"""Guarded incremental state for deadline-free execution resource envelopes."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Final

from empirical_lawhood.infrastructure.artifacts import GuardedExternalRoot
from empirical_lawhood.infrastructure.bounded_io import read_bounded_bytes
from empirical_lawhood.infrastructure.file_locks import exclusive_file_lock
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    canonical_json_bytes,
    validate_relative_locator,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.runtime.execution_envelope import ExecutionResourceEnvelopeEvent, JitGraphSignatureManifest, reconstruct_execution_resource_envelope, replay_execution_resource_event, RunExecutionResourceEnvelope

from empirical_lawhood.runtime.recovery import RESOURCE_ENVELOPE_JOURNAL_ROOT

MAX_EXECUTION_RESOURCE_ENVELOPE_STATE_BYTES: Final[int] = 256 * 1024 * 1024
DEFAULT_EXECUTION_RESOURCE_ENVELOPE_STATE_ROOT: Final[str] = RESOURCE_ENVELOPE_JOURNAL_ROOT


@dataclass(frozen=True, slots=True)
class _JournalOrigin(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/infrastructure/resource-envelope-journal-origin'
    envelope: RunExecutionResourceEnvelope
    jit_manifest: JitGraphSignatureManifest | None

    def __post_init__(self) -> None:
        initial = reconstruct_execution_resource_envelope(
            envelope_id=self.envelope.envelope_id,
            spec=self.envelope.spec,
            execution_plan=self.envelope.execution_plan,
            events=(),
            jit_graph_signature_manifest=self.jit_manifest,
        )
        if self.envelope != initial:
            raise ValueError("resource-envelope journal must start at the issued initial state")


@dataclass(frozen=True, slots=True)
class _JournalCheckpoint(_JournalOrigin):
    """An additive origin whose entire retained prefix is replay-verified."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/infrastructure/resource-envelope-journal-checkpoint'

    def __post_init__(self) -> None:
        replayed = reconstruct_execution_resource_envelope(
            envelope_id=self.envelope.envelope_id,
            spec=self.envelope.spec,
            execution_plan=self.envelope.execution_plan,
            events=self.envelope.events,
            jit_graph_signature_manifest=self.jit_manifest,
        )
        if replayed != self.envelope:
            raise ValueError("resource-envelope checkpoint differs from its complete event replay")


@dataclass(frozen=True, slots=True)
class _JournalHead(CanonicalRecord):
    """Durable commit pointer for the one authoritative operational event log."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/infrastructure/resource-envelope-journal-head'
    base_sha256: str
    journal_sha256: str
    journal_bytes: int
    event_count: int
    materialized_bytes: int

    def __post_init__(self) -> None:
        for name in ("base_sha256", "journal_sha256"):
            validate_sha256(getattr(self, name), field_name=name)
        if min(self.journal_bytes, self.event_count, self.materialized_bytes) < 0:
            raise ValueError("resource-envelope journal head has invalid bounds")


@dataclass(frozen=True, slots=True)
class _JournalPaths:
    origin: Path
    journal: Path
    head: Path


class ExternalExecutionResourceEnvelopeStore:
    """One fsynced event append followed by an atomic, content-bound head.

    Initial load and explicit inspection authenticate and replay all committed
    bytes. An active operation retains that verified prefix and extends its hash
    with bytes it has itself fsynced. A head or file-identity change invalidates
    the prefix and forces full authentication. Filesystem metadata alone never
    establishes an authenticated prefix. Uncommitted tails are discarded on retry.
    """

    def __init__(
        self,
        root: GuardedExternalRoot,
        *,
        state_root_relative_path: str = DEFAULT_EXECUTION_RESOURCE_ENVELOPE_STATE_ROOT,
        maximum_state_bytes: int = MAX_EXECUTION_RESOURCE_ENVELOPE_STATE_BYTES,
    ) -> None:
        validate_relative_locator(state_root_relative_path)
        if not 0 < maximum_state_bytes <= MAX_EXECUTION_RESOURCE_ENVELOPE_STATE_BYTES:
            raise ValueError("resource-envelope state byte limit is invalid")
        self.root = root
        self.state_root_relative_path = state_root_relative_path
        self.maximum_state_bytes = maximum_state_bytes
        self.maximum_journal_bytes = maximum_state_bytes
        self._cached: (
            tuple[
                _JournalHead,
                RunExecutionResourceEnvelope,
                _JournalOrigin,
                tuple[tuple[int, ...], ...],
            ]
            | None
        ) = None
        self._journal_hasher = hashlib.sha256()
        self._cache_lock = threading.RLock()

    @staticmethod
    def _token(envelope_id: str) -> str:
        validate_stable_id(envelope_id, field_name="envelope_id")
        return hashlib.sha256(envelope_id.encode("utf-8")).hexdigest()

    def _relative_path(self, envelope_id: str, suffix: str) -> str:
        return f"{self.state_root_relative_path}/{self._token(envelope_id)}.{suffix}"

    def _path(self, envelope_id: str, suffix: str) -> Path:
        return self.root.resolve(self._relative_path(envelope_id, suffix), for_write=True)

    @contextmanager
    def _locked(self, envelope_id: str) -> Iterator[_JournalPaths]:
        state_path = self._path(envelope_id, "json")
        if not state_path.parent.exists():
            state_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            state_path = self._path(envelope_id, "json")
        lock_path = self._path(envelope_id, "lock")
        # The cached prefix/hash belongs to one envelope at a time. Serialize
        # this instance as well as each external log so another run cannot
        # replace its cached origin between validation and append.
        with self._cache_lock, exclusive_file_lock(lock_path):
            paths = _JournalPaths(
                state_path, self._path(envelope_id, "journal"), self._path(envelope_id, "head")
            )
            try:
                # Resolve once per locked operation, never across operations.
                # Reads retain no-follow/type/identity checks; each writer below
                # also revalidates its path immediately before mutation.
                yield paths
            finally:
                # Detect mount/space loss and ancestor substitution even when
                # the final reads only used the transaction's resolved paths.
                self.root.resolve(self.state_root_relative_path, for_write=True)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _write_all(descriptor: int, payload: bytes) -> None:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("resource-envelope write made no progress")
            view = view[written:]

    def _write_atomic(self, path: Path, record: CanonicalRecord) -> None:
        payload = record.canonical_bytes()
        if len(payload) > self.maximum_state_bytes:
            raise ValueError("resource-envelope state exceeds its byte limit")
        token = hashlib.sha256(
            f"{os.getpid()}\0{threading.get_ident()}\0{record.fingerprint()}".encode()
        ).hexdigest()[:24]
        temporary = path.with_name(f".{path.name}.{token}.partial")
        descriptor: int | None = None
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                stat.S_IRUSR | stat.S_IWUSR,
            )
            self._write_all(descriptor, payload)
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            os.replace(temporary, path)
            self._fsync_directory(path.parent)
        finally:
            if descriptor is not None:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _stamp(path: Path) -> tuple[int, ...]:
        try:
            value = path.stat(follow_symlinks=False)
        except FileNotFoundError:
            return ()
        if not stat.S_ISREG(value.st_mode):
            raise ValueError("resource-envelope file is not regular")
        return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)

    def _stamps(self, paths: _JournalPaths) -> tuple[tuple[int, ...], ...]:
        return self._stamp(paths.origin), self._stamp(paths.journal)

    def _head(self, paths: _JournalPaths) -> _JournalHead:
        try:
            head = decode_canonical_bytes(
                read_bounded_bytes(paths.head, maximum_bytes=4096),
                _JournalHead,
                maximum_bytes=4096,
            )
        except FileNotFoundError as error:
            raise ValueError("resource-envelope journal commit pointer is missing") from error
        if (
            head.journal_bytes > self.maximum_journal_bytes
            or head.materialized_bytes > self.maximum_state_bytes
        ):
            raise ValueError("resource-envelope journal bound differs")
        return head

    def _read(
        self, paths: _JournalPaths, envelope_id: str
    ) -> tuple[RunExecutionResourceEnvelope, str, bytes]:
        before = self._stamps(paths)
        base = read_bounded_bytes(paths.origin, maximum_bytes=self.maximum_state_bytes)
        base_sha = hashlib.sha256(base).hexdigest()
        head = self._head(paths)
        if head.base_sha256 != base_sha:
            raise ValueError("resource-envelope journal base differs")
        journal = b""
        if head.journal_bytes:
            journal = read_bounded_bytes(paths.journal, maximum_bytes=self.maximum_journal_bytes)[
                : head.journal_bytes
            ]
        if len(journal) != head.journal_bytes:
            raise ValueError("resource-envelope committed journal is truncated")
        hasher = hashlib.sha256(journal)
        if hasher.hexdigest() != head.journal_sha256:
            raise ValueError("resource-envelope journal digest differs")
        origin = decode_canonical_bytes(
            base,
            _JournalCheckpoint
            if json.loads(base).get("schema") == _JournalCheckpoint.SCHEMA
            else _JournalOrigin,
            maximum_bytes=self.maximum_state_bytes,
        )
        envelope = origin.envelope
        journal_events = []
        for line in journal.splitlines(keepends=True):
            if not line.endswith(b"\n"):
                raise ValueError("resource-envelope journal event is incomplete")
            event = decode_canonical_bytes(
                line, ExecutionResourceEnvelopeEvent, maximum_bytes=self.maximum_state_bytes
            )
            if event.canonical_bytes() != line:
                raise ValueError("resource-envelope journal event is noncanonical")
            journal_events.append(event)
        if len(journal_events) > 64:
            envelope = reconstruct_execution_resource_envelope(
                envelope_id=envelope.envelope_id, spec=envelope.spec,
                execution_plan=envelope.execution_plan,
                events=(*envelope.events, *journal_events),
                jit_graph_signature_manifest=origin.jit_manifest,
            )
        else:
            for event in journal_events:
                envelope = replay_execution_resource_event(
                    envelope, event, jit_graph_signature_manifest=origin.jit_manifest
                )
        if envelope.envelope_id != envelope_id or len(envelope.events) != head.event_count:
            raise ValueError("resource-envelope committed state identity differs")
        if len(envelope.canonical_bytes()) != head.materialized_bytes:
            raise ValueError("resource-envelope materialized byte bound differs")
        if before != self._stamps(paths) or self._head(paths) != head:
            raise ValueError("resource-envelope journal changed during authentication")
        self._journal_hasher = hasher
        self._cached = (head, envelope, origin, before)
        return envelope, base_sha, journal

    def _current(self, paths: _JournalPaths, envelope_id: str) -> RunExecutionResourceEnvelope:
        cached = self._cached
        if cached is not None and cached[1].envelope_id == envelope_id:
            if self._head(paths) == cached[0] and self._stamps(paths) == cached[3]:
                return cached[1]
        return self._read(paths, envelope_id)[0]

    def create(
        self,
        envelope: RunExecutionResourceEnvelope,
        *,
        jit_graph_signature_manifest: JitGraphSignatureManifest | None = None,
    ) -> None:
        origin = (_JournalCheckpoint if envelope.events else _JournalOrigin)(
            envelope=envelope, jit_manifest=jit_graph_signature_manifest
        )
        if len(origin.canonical_bytes()) > self.maximum_state_bytes:
            raise ValueError("resource-envelope origin exceeds its byte limit")
        head = _JournalHead(
            base_sha256=origin.fingerprint(),
            journal_sha256=hashlib.sha256(b"").hexdigest(),
            journal_bytes=0,
            event_count=len(envelope.events),
            materialized_bytes=len(envelope.canonical_bytes()),
        )
        with self._locked(envelope.envelope_id) as paths:
            if paths.origin.exists():
                current, _, _ = self._read(paths, envelope.envelope_id)
                if current == envelope and self._cached is not None and self._cached[2] == origin:
                    return
                raise ValueError("resource envelope already exists with different state")
            if paths.journal.exists():
                raise ValueError("resource-envelope checkpoint is missing for a committed journal")
            head_path = paths.head
            if head_path.exists():
                if self._head(paths) != head:
                    raise ValueError("resource-envelope initialization commit differs")
            else:
                # Commit the exact empty origin identity first. A crash before
                # origin publication can then finish that same initialization;
                # a missing head never masquerades as an unspent existing run.
                self._write_atomic(self._path(envelope.envelope_id, "head"), head)
            self._write_atomic(self._path(envelope.envelope_id, "json"), origin)
            self._journal_hasher = hashlib.sha256()
            self._cached = (head, envelope, origin, self._stamps(paths))

    def load(self, envelope_id: str) -> RunExecutionResourceEnvelope:
        """Authenticate the complete external prefix for explicit inspection."""
        with self._locked(envelope_id) as paths:
            if not paths.origin.is_file():
                raise KeyError(envelope_id)
            return self._read(paths, envelope_id)[0]

    def current(self, envelope_id: str) -> RunExecutionResourceEnvelope:
        """Read within the active operation, extending only a verified prefix."""
        with self._locked(envelope_id) as paths:
            if not paths.origin.is_file():
                raise KeyError(envelope_id)
            return self._current(paths, envelope_id)

    @staticmethod
    def _state_byte_delta(
        current: RunExecutionResourceEnvelope, updated: RunExecutionResourceEnvelope
    ) -> int:
        # Cell/token rosters are fixed. Only changed records contribute bytes;
        # historical events and unchanged siblings are never serialized.
        delta = sum(
            len(new.canonical_bytes()) - len(old.canonical_bytes())
            for old, new in (
                *zip(current.cells, updated.cells, strict=True),
                *zip(current.child_token_states, updated.child_token_states, strict=True),
            )
            if old != new
        )
        if current.roster_decisions != updated.roster_decisions:
            delta += len(canonical_json_bytes(updated.roster_decisions)) - len(
                canonical_json_bytes(current.roster_decisions)
            )
        return delta

    def compare_and_append(
        self, *, expected: RunExecutionResourceEnvelope, updated: RunExecutionResourceEnvelope
    ) -> None:
        with self._locked(updated.envelope_id) as paths:
            current = self._current(paths, updated.envelope_id)
            if len(current.events) != len(expected.events) or current != expected:
                raise ValueError("resource-envelope compare-and-append conflict")
            if (
                len(updated.events) != len(current.events) + 1
                or updated.events[:-1] != current.events
            ):
                raise ValueError("resource-envelope update is not one event append")
            assert self._cached is not None
            head, _, origin, stamps = self._cached
            event = updated.events[-1]
            replayed = replay_execution_resource_event(
                current, event, jit_graph_signature_manifest=origin.jit_manifest
            )
            if replayed != updated:
                raise ValueError("resource-envelope update differs from its replayed event")
            payload = event.canonical_bytes()
            materialized_bytes = (
                head.materialized_bytes
                + len(payload)
                - 1
                + bool(current.events)
                + self._state_byte_delta(current, updated)
            )
            delta = payload
            if (
                materialized_bytes > self.maximum_state_bytes
                or head.journal_bytes + len(delta) > self.maximum_journal_bytes
            ):
                raise ValueError("resource-envelope journal or state exceeds its byte limit")
            hasher = self._journal_hasher.copy()
            hasher.update(delta)
            journal_path = self._path(updated.envelope_id, "journal")
            descriptor = os.open(
                journal_path,
                os.O_RDWR
                | os.O_CREAT
                | (os.O_EXCL if not stamps[1] else 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                stat.S_IRUSR | stat.S_IWUSR,
            )
            try:
                value = os.fstat(descriptor)
                if not stat.S_ISREG(value.st_mode) or (
                    stamps[1]
                    and (
                        value.st_dev,
                        value.st_ino,
                        value.st_size,
                        value.st_mtime_ns,
                        value.st_ctime_ns,
                    )
                    != stamps[1]
                ):
                    raise ValueError("resource-envelope journal changed before append")
                os.ftruncate(descriptor, head.journal_bytes)
                os.lseek(descriptor, head.journal_bytes, os.SEEK_SET)
                self._write_all(descriptor, delta)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            self._fsync_directory(journal_path.parent)
            committed = _JournalHead(
                base_sha256=head.base_sha256,
                journal_sha256=hasher.hexdigest(),
                journal_bytes=head.journal_bytes + len(delta),
                event_count=len(updated.events),
                materialized_bytes=materialized_bytes,
            )
            self._write_atomic(self._path(updated.envelope_id, "head"), committed)
            self._journal_hasher = hasher
            self._cached = (committed, updated, origin, self._stamps(paths))


__all__ = [
    "DEFAULT_EXECUTION_RESOURCE_ENVELOPE_STATE_ROOT",
    'ExternalExecutionResourceEnvelopeStore',
    "MAX_EXECUTION_RESOURCE_ENVELOPE_STATE_BYTES",
]
