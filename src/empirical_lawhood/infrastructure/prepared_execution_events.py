"""Bounded immutable prepared-execution prefixes on the guarded artifact plane."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from threading import RLock

from empirical_lawhood.infrastructure.artifacts import ArtifactIdentityConflict, ExternalArtifactPlane
from empirical_lawhood.infrastructure.bounded_io import MAX_ARTIFACT_MANIFEST_BYTES, read_bounded_bytes
from empirical_lawhood.infrastructure.file_locks import exclusive_file_lock
from empirical_lawhood.infrastructure.task_receipts import decode_artifact_manifest
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_relative_locator,
    validate_schema,
    validate_stable_id,
)
from empirical_lawhood.runtime.artifacts import ArtifactProfile, ArtifactWriteRequest
from empirical_lawhood.runtime.controller_evaluation_nested import PreparedExecutionEventPrefix, ProspectiveBindingConflict


MAX_PREPARED_RECORD_BYTES = 128 * 1024 * 1024
MAX_PREPARED_PREFIX_BYTES = 1024 * 1024
MAX_PREPARED_EVENTS = 256


class ExternalPreparedExecutionEventStore:
    """No mutable scientific head: each committed prefix is an immutable publication.

    At most 256 numbered snapshots are inspected. A per-prefix advisory lock
    serializes compare-and-append across processes. Publication success precedes
    acknowledgement; after a lost acknowledgement the recovered reservation is
    still consumed. The existing artifact plane owns partial-publication repair.
    """

    def __init__(
        self,
        plane: ExternalArtifactPlane,
        *,
        state_root_relative_path: str,
        record_schemas: tuple[str, ...],
        minimum_free_bytes: int,
        maximum_record_bytes: int = MAX_PREPARED_RECORD_BYTES,
    ) -> None:
        validate_relative_locator(state_root_relative_path)
        require_sorted_unique_strings(
            record_schemas, field_name="record_schemas", allow_empty=False
        )
        for schema in record_schemas:
            validate_schema(schema)
        if minimum_free_bytes < 0 or not 0 < maximum_record_bytes <= MAX_PREPARED_RECORD_BYTES:
            raise ValueError("prepared event store has invalid storage bounds")
        self.plane = plane
        self.state_root_relative_path = state_root_relative_path
        self.record_schemas = frozenset(record_schemas)
        self.minimum_free_bytes = minimum_free_bytes
        self.maximum_record_bytes = maximum_record_bytes
        self._lock = RLock()

    @staticmethod
    def _token(value: str) -> str:
        validate_stable_id(value, field_name="prepared storage identity")
        return sha256(value.encode("utf-8")).hexdigest()

    def _scope(self, prefix_id: str) -> str:
        return f"{self.state_root_relative_path}/{self._token(prefix_id)}"

    def _record_path(self, prefix_id: str, object_id: str) -> str:
        return f"{self._scope(prefix_id)}/records/{self._token(object_id)}.json"

    def _prefix_path(self, prefix_id: str, sequence: int) -> str:
        if not 1 <= sequence <= MAX_PREPARED_EVENTS:
            raise ValueError("prepared sequence exceeds the bounded snapshot roster")
        return f"{self._scope(prefix_id)}/prefixes/{sequence:03}.json"

    def _path(self, relative: str, *, for_write: bool) -> Path:
        return self.plane.root.resolve(
            relative, for_write=for_write, operation_minimum_free_bytes=self.minimum_free_bytes
        )

    @contextmanager
    def _locked(self, prefix_id: str) -> Iterator[None]:
        self.plane.root.verify(for_write=True, operation_minimum_free_bytes=self.minimum_free_bytes)
        lock_relative = f"{self._scope(prefix_id)}/append.lock"
        path = self._path(lock_relative, for_write=True)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = self._path(lock_relative, for_write=True)
        with self._lock, exclusive_file_lock(path):
            try:
                yield
            finally:
                self.plane.root.verify(
                    for_write=True, operation_minimum_free_bytes=self.minimum_free_bytes
                )

    def _write(
        self,
        *,
        prefix_id: str,
        relative: str,
        object_id: str,
        record: CanonicalRecord,
    ) -> ArtifactIdentity:
        payload = record.canonical_bytes()
        maximum = (
            MAX_PREPARED_PREFIX_BYTES
            if isinstance(record, PreparedExecutionEventPrefix)
            else self.maximum_record_bytes
        )
        if not payload or len(payload) > maximum:
            raise ValueError("prepared control record exceeds its declared byte bound")
        result = self.plane.write(
            ArtifactWriteRequest(
                logical_artifact_id=object_id,
                relative_path=relative,
                payload_schema=record.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type="application/json",
                publication_scope_id=f"prepared-events.{self._token(prefix_id)}",
                publication_scope_relative_root=self._scope(prefix_id),
                payload=payload,
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                parent_visibility_ceilings=(),
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
                minimum_free_bytes=self.minimum_free_bytes,
            )
        )
        if result.logical.content_sha256 != record.fingerprint():
            raise ArtifactIdentityConflict("prepared publication changed its canonical subject")
        return ArtifactIdentity(
            artifact_id=object_id,
            role="prepared-control-record",
            payload_schema=record.SCHEMA,
            sha256=record.fingerprint(),
            media_type="application/json",
            size_bytes=len(payload),
        )

    def publish_prepared_record(
        self,
        *,
        prefix_id: str,
        object_id: str,
        record: CanonicalRecord,
    ) -> ArtifactIdentity:
        if record.SCHEMA not in self.record_schemas:
            raise ValueError("prepared control-record schema is outside the injected roster")
        return self._write(
            prefix_id=prefix_id,
            relative=self._record_path(prefix_id, object_id),
            object_id=object_id,
            record=record,
        )

    def _read(self, relative: str, *, maximum_bytes: int) -> tuple[bytes, str, str]:
        self.plane.root.verify(for_write=False)
        path = self._path(relative, for_write=False)
        manifest_path = self._path(f"{relative}.manifest.json", for_write=False)
        manifest = decode_artifact_manifest(
            read_bounded_bytes(manifest_path, maximum_bytes=MAX_ARTIFACT_MANIFEST_BYTES)
        )
        if (
            manifest.materialization.relative_path != relative
            or manifest.materialization.size_bytes > maximum_bytes
            or manifest.logical.profile is not ArtifactProfile.CANONICAL_JSON
            or manifest.logical.media_type != "application/json"
            or manifest.logical.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
            or manifest.logical.outcome_access is not OutcomeAccess.OUTCOME_BLIND
        ):
            raise ArtifactIdentityConflict(
                "prepared manifest changes its bounded control-record contract"
            )
        self.plane.verify_manifest(manifest)
        payload = read_bounded_bytes(path, maximum_bytes=maximum_bytes)
        if sha256(payload).hexdigest() != manifest.logical.content_sha256:
            raise ArtifactIdentityConflict("prepared record changed during authenticated read")
        return payload, manifest.logical.logical_artifact_id, manifest.logical.payload_schema

    def read_prepared_record(
        self,
        *,
        prefix_id: str,
        subject: ObjectIdentity,
        artifact: ArtifactIdentity,
    ) -> bytes:
        if subject.object_schema not in self.record_schemas:
            raise ValueError("prepared subject schema is outside the injected roster")
        payload, object_id, schema = self._read(
            self._record_path(prefix_id, subject.object_id), maximum_bytes=self.maximum_record_bytes
        )
        expected = ArtifactIdentity(
            subject.object_id,
            "prepared-control-record",
            schema,
            sha256(payload).hexdigest(),
            "application/json",
            len(payload),
        )
        if (
            object_id != subject.object_id
            or schema != subject.object_schema
            or expected != artifact
            or artifact.sha256 != subject.object_fingerprint
            or json.loads(payload).get("version") != subject.object_version
            or json.loads(payload).get("schema") != subject.object_schema
        ):
            raise ArtifactIdentityConflict("prepared subject/artifact identity changed")
        return payload

    def verify_prepared_subject(
        self,
        *,
        prefix_id: str,
        subject: ObjectIdentity,
        artifact: ArtifactIdentity,
    ) -> None:
        self.read_prepared_record(prefix_id=prefix_id, subject=subject, artifact=artifact)

    def _snapshot_sequences(self, prefix_id: str) -> tuple[int, ...]:
        directory = self._path(f"{self._scope(prefix_id)}/prefixes", for_write=False)
        if not directory.exists():
            return ()
        sequences: set[int] = set()
        with os.scandir(directory) as entries:
            for count, entry in enumerate(entries, 1):
                if count > 8 * MAX_PREPARED_EVENTS:
                    raise ValueError("prepared snapshot directory exceeds its bounded inventory")
                if entry.is_symlink():
                    raise ArtifactIdentityConflict("prepared snapshot directory contains a symlink")
                matched = re.fullmatch(r"([0-9]{3})\.json(?:\.manifest\.json)?", entry.name)
                if matched is not None:
                    sequence = int(matched.group(1))
                    if not 1 <= sequence <= MAX_PREPARED_EVENTS:
                        raise ValueError("prepared snapshot sequence exceeds its declared bound")
                    sequences.add(sequence)
        ordered = tuple(sorted(sequences))
        if ordered != tuple(range(1, len(ordered) + 1)):
            raise ProspectiveBindingConflict("PREPARED_PERSISTED_PREFIX_GAP")
        return ordered

    def _load(
        self, prefix_id: str, *, stop_before: int | None = None
    ) -> PreparedExecutionEventPrefix:
        current: PreparedExecutionEventPrefix | None = None
        for sequence in self._snapshot_sequences(prefix_id):
            if stop_before is not None and sequence >= stop_before:
                break
            relative = self._prefix_path(prefix_id, sequence)
            payload, object_id, schema = self._read(
                relative, maximum_bytes=MAX_PREPARED_PREFIX_BYTES
            )
            prefix = decode_canonical_bytes(
                payload, PreparedExecutionEventPrefix, maximum_bytes=MAX_PREPARED_PREFIX_BYTES
            )
            if (
                prefix.prefix_id != prefix_id
                or len(prefix.events) != sequence
                or schema != prefix.SCHEMA
                or object_id != f"{prefix_id}.prefix.{sequence}"
                or (
                    current is not None
                    and (
                        prefix.events[:-1] != current.events
                        or prefix.root_id != current.root_id
                        or prefix.policy_id != current.policy_id
                    )
                )
            ):
                raise ProspectiveBindingConflict("PREPARED_PREFIX_CONTINUITY_CONFLICT")
            event = prefix.events[-1]
            self.verify_prepared_subject(
                prefix_id=prefix_id, subject=event.subject, artifact=event.subject_artifact
            )
            current = prefix
        if current is None:
            raise KeyError(prefix_id)
        return current

    def load_prepared(self, prefix_id: str) -> PreparedExecutionEventPrefix:
        # Immutable snapshots permit concurrent reads; an incomplete publication
        # raises rather than treating a possibly consumed reservation as absent.
        return self._load(prefix_id)

    def _write_prefix(self, prefix: PreparedExecutionEventPrefix) -> None:
        count = len(prefix.events)
        event = prefix.events[-1]
        self.verify_prepared_subject(
            prefix_id=prefix.prefix_id, subject=event.subject, artifact=event.subject_artifact
        )
        self._write(
            prefix_id=prefix.prefix_id,
            relative=self._prefix_path(prefix.prefix_id, count),
            object_id=f"{prefix.prefix_id}.prefix.{count}",
            record=prefix,
        )

    def create_prepared(self, prefix: PreparedExecutionEventPrefix) -> None:
        if len(prefix.events) != 1:
            raise ProspectiveBindingConflict("PREPARED_INITIAL_PREFIX_REQUIRED")
        with self._locked(prefix.prefix_id):
            relative = self._prefix_path(prefix.prefix_id, 1)
            if self._path(relative, for_write=False).exists():
                existing = self._load(prefix.prefix_id, stop_before=2)
                if existing != prefix:
                    raise ProspectiveBindingConflict("PREPARED_DESIGN_CONFLICT")
            else:
                self._write_prefix(prefix)

    def compare_and_append_prepared(
        self,
        *,
        expected_prefix_sha256: str,
        updated: PreparedExecutionEventPrefix,
    ) -> None:
        with self._locked(updated.prefix_id):
            current = self._load(updated.prefix_id)
            if (
                current.fingerprint() != expected_prefix_sha256
                or updated.events[:-1] != current.events
                or updated.root_id != current.root_id
                or updated.policy_id != current.policy_id
            ):
                raise ProspectiveBindingConflict("PREPARED_BINDING_CONFLICT")
            self._write_prefix(updated)

    def recover_prepared_publication(self, expected: PreparedExecutionEventPrefix) -> None:
        """Repair only an exact pending publication; never invoke a source or task.

        The caller supplies the original pending prefix from its persisted
        publication intent/control record. Divergent bytes are refused by the
        artifact plane, including when its payload/sidecar/commit is partial.
        """
        with self._locked(expected.prefix_id):
            relative = self._prefix_path(expected.prefix_id, len(expected.events))
            if (
                not self._path(relative, for_write=False).exists()
                and not self._path(f"{relative}.manifest.json", for_write=False).exists()
            ):
                raise ProspectiveBindingConflict("PREPARED_RECOVERY_HAS_NO_PENDING_PUBLICATION")
            count = len(expected.events)
            if count > 1:
                current = self._load(expected.prefix_id, stop_before=count)
                if (
                    expected.events[:-1] != current.events
                    or expected.root_id != current.root_id
                    or expected.policy_id != current.policy_id
                ):
                    raise ProspectiveBindingConflict("PREPARED_RECOVERY_PREFIX_CONFLICT")
            self._write_prefix(expected)
