"""Guarded immutable custody for cumulative campaign elapsed-budget ledgers."""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from threading import RLock
from typing import Final

from empirical_lawhood.infrastructure.artifacts import ArtifactIdentityConflict, ExternalArtifactPlane
from empirical_lawhood.infrastructure.bounded_io import MAX_ARTIFACT_MANIFEST_BYTES, read_bounded_bytes
from empirical_lawhood.infrastructure.file_locks import exclusive_file_lock
from empirical_lawhood.infrastructure.task_receipts import decode_artifact_manifest
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.serialization import validate_relative_locator, validate_sha256, validate_stable_id
from empirical_lawhood.runtime.artifacts import ArtifactProfile, ArtifactWriteRequest
from empirical_lawhood.runtime.campaign_elapsed_budget import CampaignElapsedLedger


MAX_CAMPAIGN_ELAPSED_SNAPSHOTS: Final[int] = 128
MAX_CAMPAIGN_ELAPSED_LEDGER_BYTES: Final[int] = 4 * 1024 * 1024


# Original custody root must remain visible to prevent resetting accumulated cost.
_HISTORICAL_ELAPSED_JOURNAL_ROOT = "control/campaign-elapsed-budgets/v1"


class CampaignElapsedBudgetConflict(RuntimeError):
    """The durable elapsed ledger differs from the caller's authenticated prefix."""


class ExternalCampaignElapsedBudgetStore:
    """Write-once ledger snapshots serialized by one guarded advisory lock.

    The store has no mutable head.  Loading authenticates every numbered
    snapshot and every adjacent state transition.  A lost acknowledgement can
    therefore be recovered by reading the already-published latest snapshot;
    it cannot reset accumulated time or replace an envelope binding.
    """

    def __init__(
        self,
        plane: ExternalArtifactPlane,
        *,
        state_root_relative_path: str = "control/campaign-elapsed-budgets",
        minimum_free_bytes: int = 0,
        maximum_ledger_bytes: int = MAX_CAMPAIGN_ELAPSED_LEDGER_BYTES,
    ) -> None:
        validate_relative_locator(state_root_relative_path)
        if minimum_free_bytes < 0 or not 0 < maximum_ledger_bytes <= MAX_CAMPAIGN_ELAPSED_LEDGER_BYTES:
            raise ValueError("campaign elapsed-budget store has invalid storage bounds")
        # A new default must not hide cumulative costs held under the original root.
        if state_root_relative_path == "control/campaign-elapsed-budgets":
            original_state_root = _HISTORICAL_ELAPSED_JOURNAL_ROOT
            if plane.root.resolve(original_state_root, for_write=False).exists():
                raise CampaignElapsedBudgetConflict(
                    "CAMPAIGN_ELAPSED_ORIGINAL_JOURNAL_REQUIRES_VERIFIED_MIGRATION"
                )
        self.plane = plane
        self.state_root_relative_path = state_root_relative_path
        self.minimum_free_bytes = minimum_free_bytes
        self.maximum_ledger_bytes = maximum_ledger_bytes
        self._lock = RLock()

    @staticmethod
    def _token(ledger_id: str) -> str:
        validate_stable_id(ledger_id, field_name="ledger_id")
        return sha256(ledger_id.encode("utf-8")).hexdigest()

    def _scope(self, ledger_id: str) -> str:
        return f"{self.state_root_relative_path}/{self._token(ledger_id)}"

    def _snapshot_path(self, ledger_id: str, sequence: int) -> str:
        if not 1 <= sequence <= MAX_CAMPAIGN_ELAPSED_SNAPSHOTS:
            raise ValueError("campaign elapsed-budget snapshot ceiling is exceeded")
        return f"{self._scope(ledger_id)}/snapshots/{sequence:03d}.json"

    def _path(self, relative: str, *, for_write: bool) -> Path:
        return self.plane.root.resolve(
            relative,
            for_write=for_write,
            operation_minimum_free_bytes=self.minimum_free_bytes,
        )

    @contextmanager
    def _locked(self, ledger_id: str) -> Iterator[None]:
        self.plane.root.verify(
            for_write=True, operation_minimum_free_bytes=self.minimum_free_bytes
        )
        relative = f"{self._scope(ledger_id)}/append.lock"
        path = self._path(relative, for_write=True)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = self._path(relative, for_write=True)
        with self._lock, exclusive_file_lock(path):
            try:
                yield
            finally:
                self.plane.root.verify(
                    for_write=True,
                    operation_minimum_free_bytes=self.minimum_free_bytes,
                )

    @staticmethod
    def _is_exact_close(
        current: CampaignElapsedLedger,
        updated: CampaignElapsedLedger,
        *,
        exhausted: bool,
    ) -> bool:
        if (
            current.open_interval_id is None
            or updated.open_interval_id is not None
            or updated.open_interval_kind is not None
            or updated.open_interval_started_utc is not None
            or updated.open_envelope_binding is not None
            or updated.envelope_bindings != current.envelope_bindings
            or updated.exhausted is not exhausted
            or len(updated.closed_intervals) != len(current.closed_intervals) + 1
            or updated.closed_intervals[:-1] != current.closed_intervals
        ):
            return False
        assert current.open_interval_kind is not None
        assert current.open_interval_started_utc is not None
        assert current.open_envelope_binding is not None
        interval = updated.closed_intervals[-1]
        return (
            interval.interval_id == current.open_interval_id
            and interval.envelope_binding == current.open_envelope_binding
            and interval.kind == current.open_interval_kind
            and interval.opened_at_utc == current.open_interval_started_utc
        )

    @classmethod
    def _valid_transition(
        cls, current: CampaignElapsedLedger, updated: CampaignElapsedLedger
    ) -> bool:
        if current.ledger_id != updated.ledger_id or current.budget != updated.budget:
            return False
        # Bind exactly one new conditional-stage envelope.
        if (
            not current.exhausted
            and current.open_interval_id is None
            and updated.closed_intervals == current.closed_intervals
            and updated.open_interval_id is None
            and updated.exhausted is current.exhausted
            and len(updated.envelope_bindings) == len(current.envelope_bindings) + 1
            and updated.envelope_bindings[:-1] == current.envelope_bindings
        ):
            return True
        # Open exactly one active launch/recovery interval.
        if (
            not current.exhausted
            and current.open_interval_id is None
            and updated.envelope_bindings == current.envelope_bindings
            and updated.closed_intervals == current.closed_intervals
            and not updated.exhausted
            and updated.open_interval_id is not None
        ):
            return True
        # Close/reconcile one active interval without exhausting the campaign.
        if cls._is_exact_close(current, updated, exhausted=current.exhausted):
            return True
        # Mark an idle ledger exhausted, or close one active interval and exhaust.
        return (
            not current.exhausted
            and updated.exhausted
            and (
                (
                    current.open_interval_id is None
                    and updated.envelope_bindings == current.envelope_bindings
                    and updated.closed_intervals == current.closed_intervals
                    and updated.open_interval_id is None
                )
                or cls._is_exact_close(current, updated, exhausted=True)
            )
        )

    def _write(self, ledger: CampaignElapsedLedger, sequence: int) -> None:
        payload = ledger.canonical_bytes()
        if not payload or len(payload) > self.maximum_ledger_bytes:
            raise ValueError("campaign elapsed ledger exceeds its declared byte bound")
        relative = self._snapshot_path(ledger.ledger_id, sequence)
        result = self.plane.write(
            ArtifactWriteRequest(
                logical_artifact_id=f"{ledger.ledger_id}.snapshot.{sequence:03d}",
                relative_path=relative,
                payload_schema=ledger.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type="application/json",
                publication_scope_id=f"campaign-elapsed.{self._token(ledger.ledger_id)}",
                publication_scope_relative_root=self._scope(ledger.ledger_id),
                payload=payload,
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                parent_visibility_ceilings=(),
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
                minimum_free_bytes=self.minimum_free_bytes,
            )
        )
        if result.logical.content_sha256 != ledger.fingerprint():
            raise ArtifactIdentityConflict("campaign elapsed ledger changed during publication")

    def _snapshot_sequences(self, ledger_id: str) -> tuple[int, ...]:
        directory = self._path(f"{self._scope(ledger_id)}/snapshots", for_write=False)
        if not directory.exists():
            return ()
        sequences: set[int] = set()
        with os.scandir(directory) as entries:
            for count, entry in enumerate(entries, 1):
                if count > 4 * MAX_CAMPAIGN_ELAPSED_SNAPSHOTS:
                    raise CampaignElapsedBudgetConflict("CAMPAIGN_ELAPSED_SNAPSHOT_INVENTORY_EXCEEDED")
                if entry.is_symlink():
                    raise ArtifactIdentityConflict("campaign elapsed snapshot directory contains a symlink")
                match = re.fullmatch(r"([0-9]{3})\.json(?:\.manifest\.json)?", entry.name)
                if match is not None:
                    sequences.add(int(match.group(1)))
        ordered = tuple(sorted(sequences))
        if ordered != tuple(range(1, len(ordered) + 1)):
            raise CampaignElapsedBudgetConflict("CAMPAIGN_ELAPSED_SNAPSHOT_GAP")
        if ordered and ordered[-1] > MAX_CAMPAIGN_ELAPSED_SNAPSHOTS:
            raise CampaignElapsedBudgetConflict("CAMPAIGN_ELAPSED_SNAPSHOT_CEILING_EXCEEDED")
        return ordered

    def _read_snapshot(self, ledger_id: str, sequence: int) -> CampaignElapsedLedger:
        relative = self._snapshot_path(ledger_id, sequence)
        path = self._path(relative, for_write=False)
        manifest_path = self._path(f"{relative}.manifest.json", for_write=False)
        manifest = decode_artifact_manifest(
            read_bounded_bytes(manifest_path, maximum_bytes=MAX_ARTIFACT_MANIFEST_BYTES)
        )
        if (
            manifest.materialization.relative_path != relative
            or manifest.materialization.size_bytes > self.maximum_ledger_bytes
            or manifest.logical.logical_artifact_id
            != f"{ledger_id}.snapshot.{sequence:03d}"
            or manifest.logical.payload_schema != CampaignElapsedLedger.SCHEMA
            or manifest.logical.profile is not ArtifactProfile.CANONICAL_JSON
            or manifest.logical.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
            or manifest.logical.outcome_access is not OutcomeAccess.OUTCOME_BLIND
        ):
            raise ArtifactIdentityConflict("campaign elapsed snapshot manifest differs")
        self.plane.verify_manifest(manifest)
        payload = read_bounded_bytes(path, maximum_bytes=self.maximum_ledger_bytes)
        if sha256(payload).hexdigest() != manifest.logical.content_sha256:
            raise ArtifactIdentityConflict("campaign elapsed snapshot changed during read")
        ledger = decode_canonical_bytes(
            payload,
            CampaignElapsedLedger,
            maximum_bytes=self.maximum_ledger_bytes,
        )
        if ledger.ledger_id != ledger_id:
            raise ArtifactIdentityConflict("campaign elapsed snapshot substitutes its ledger")
        return ledger

    def _load(self, ledger_id: str) -> tuple[int, CampaignElapsedLedger]:
        current: CampaignElapsedLedger | None = None
        sequence = 0
        for sequence in self._snapshot_sequences(ledger_id):
            observed = self._read_snapshot(ledger_id, sequence)
            if current is not None and not self._valid_transition(current, observed):
                raise CampaignElapsedBudgetConflict("CAMPAIGN_ELAPSED_TRANSITION_CONFLICT")
            current = observed
        if current is None:
            raise KeyError(ledger_id)
        return sequence, current

    def create(self, ledger: CampaignElapsedLedger) -> None:
        if (
            ledger.envelope_bindings
            or ledger.closed_intervals
            or ledger.open_interval_id is not None
            or ledger.exhausted
        ):
            raise CampaignElapsedBudgetConflict("CAMPAIGN_ELAPSED_INITIAL_LEDGER_REQUIRED")
        with self._locked(ledger.ledger_id):
            if self._path(self._snapshot_path(ledger.ledger_id, 1), for_write=False).exists():
                _, existing = self._load(ledger.ledger_id)
                if existing != ledger:
                    raise CampaignElapsedBudgetConflict("CAMPAIGN_ELAPSED_INITIAL_LEDGER_CONFLICT")
                return
            self._write(ledger, 1)

    def load(self, ledger_id: str) -> CampaignElapsedLedger:
        with self._locked(ledger_id):
            return self._load(ledger_id)[1]

    def compare_and_append(
        self,
        *,
        expected_ledger_sha256: str,
        updated: CampaignElapsedLedger,
    ) -> None:
        validate_sha256(expected_ledger_sha256, field_name="expected_ledger_sha256")
        with self._locked(updated.ledger_id):
            sequence, current = self._load(updated.ledger_id)
            if current.fingerprint() != expected_ledger_sha256:
                raise CampaignElapsedBudgetConflict("CAMPAIGN_ELAPSED_COMPARE_AND_APPEND_CONFLICT")
            if not self._valid_transition(current, updated):
                raise CampaignElapsedBudgetConflict("CAMPAIGN_ELAPSED_ILLEGAL_TRANSITION")
            self._write(updated, sequence + 1)


__all__ = [
    "CampaignElapsedBudgetConflict",
    "ExternalCampaignElapsedBudgetStore",
    "MAX_CAMPAIGN_ELAPSED_LEDGER_BYTES",
    "MAX_CAMPAIGN_ELAPSED_SNAPSHOTS",
]
