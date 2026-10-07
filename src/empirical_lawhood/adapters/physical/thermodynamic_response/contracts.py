"""Synthetic-testable contracts for external-owner physical episode imports.

These records do not open paths, issue commands, grant authority, reveal sealed
outcomes or publish bytes. They bind the evidence that an external acquisition
plane already produced and permit only exact receipt reconciliation after it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.response_algebra import ActionStage
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.thermodynamic_response import ActionInterval, ActionStageObservation
from empirical_lawhood.kernel.time import parse_utc_timestamp, validate_utc_timestamp
from empirical_lawhood.kernel.worlds import WorldKind


class ActionJournalEntryDisposition(StrEnum):
    RECORDED = "RECORDED"
    REJECTED = "REJECTED"
    PARTIAL = "PARTIAL"
    UNCERTAIN = "UNCERTAIN"


class PhysicalEpisodeRole(StrEnum):
    COMMISSIONING = "COMMISSIONING"
    CALIBRATION = "CALIBRATION"
    DEVELOPMENT = "DEVELOPMENT"
    EVALUATION = "EVALUATION"


class PhysicalEpisodeTerminalStatus(StrEnum):
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    STOPPED = "STOPPED"
    UNCERTAIN_ACTION = "UNCERTAIN_ACTION"
    INVALID = "INVALID"


class ReceiptReconciliationDisposition(StrEnum):
    EXACT_RECEIPT_REPLAY = "EXACT_RECEIPT_REPLAY"
    RECEIPT_MISSING = "RECEIPT_MISSING"


@dataclass(frozen=True, slots=True)
class ExternalAuthorityEvidence(CanonicalRecord):
    """Outcome-blind proof that an external owner authorized one exact episode."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/thermodynamic-response/external-authority-evidence'

    evidence_id: str
    episode_id: str
    external_owner_id: str
    issuer_id: str
    operator_id: str
    external_command_system_id: str
    apparatus_id: str
    hardware_ids: tuple[str, ...]
    protocol_id: str
    preparation_id: str
    calibration_ids: tuple[str, ...]
    authorized_action_quantity_ids: tuple[str, ...]
    authorized_measurement_quantity_ids: tuple[str, ...]
    authority_scope_id: str
    valid_from_utc: str
    valid_until_utc: str
    owner_interlock_id: str
    owner_stop_authority_id: str
    evidence_receipt_id: str
    evidence_receipt_sha256: str
    outcome_access: OutcomeAccess
    external_owner_retains_command: bool
    external_owner_retains_interlock: bool
    external_owner_retains_emergency_stop: bool
    platform_command_authorized: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("evidence_id", self.evidence_id),
            ("episode_id", self.episode_id),
            ("external_owner_id", self.external_owner_id),
            ("issuer_id", self.issuer_id),
            ("operator_id", self.operator_id),
            ("external_command_system_id", self.external_command_system_id),
            ("apparatus_id", self.apparatus_id),
            ("protocol_id", self.protocol_id),
            ("preparation_id", self.preparation_id),
            ("authority_scope_id", self.authority_scope_id),
            ("owner_interlock_id", self.owner_interlock_id),
            ("owner_stop_authority_id", self.owner_stop_authority_id),
            ("evidence_receipt_id", self.evidence_receipt_id),
        ):
            validate_stable_id(value, field_name=name)
        for name, string_values in (
            ("hardware_ids", self.hardware_ids),
            ("calibration_ids", self.calibration_ids),
            ("authorized_action_quantity_ids", self.authorized_action_quantity_ids),
            (
                "authorized_measurement_quantity_ids",
                self.authorized_measurement_quantity_ids,
            ),
        ):
            require_sorted_unique_strings(string_values, field_name=name, allow_empty=False)
            for value in string_values:
                validate_stable_id(value, field_name=name)
        start = parse_utc_timestamp(self.valid_from_utc, field_name="valid_from_utc")
        end = parse_utc_timestamp(self.valid_until_utc, field_name="valid_until_utc")
        if end <= start:
            raise ValueError("external authority validity end must follow its start")
        validate_sha256(self.evidence_receipt_sha256, field_name="evidence_receipt_sha256")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("external authority evidence must remain outcome-blind")
        if not (
            self.external_owner_retains_command
            and self.external_owner_retains_interlock
            and self.external_owner_retains_emergency_stop
        ):
            raise ValueError("external owner must retain command, interlock and stop authority")
        if self.platform_command_authorized:
            raise ValueError("external-import authority cannot authorize platform command")


@dataclass(frozen=True, slots=True)
class ActionJournalEntry(CanonicalRecord):
    """One append-only requested/accepted/applied/realized stage entry."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/thermodynamic-response/action-journal-entry'

    entry_id: str
    episode_id: str
    sequence_index: int
    previous_entry_sha256: str | None
    recorded_at_utc: str
    external_actor_id: str
    authority_evidence_id: str
    action_event_id: str
    stage: ActionStage
    stage_observations: tuple[ActionStageObservation, ...]
    interval: ActionInterval
    source_record_sha256: str
    disposition: ActionJournalEntryDisposition
    physical_action_may_have_occurred: bool
    physical_retry_permitted: bool
    receipt_reconciliation_permitted: bool
    interlock_active: bool
    clipped: bool
    saturated: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("entry_id", self.entry_id),
            ("episode_id", self.episode_id),
            ("external_actor_id", self.external_actor_id),
            ("authority_evidence_id", self.authority_evidence_id),
            ("action_event_id", self.action_event_id),
        ):
            validate_stable_id(value, field_name=name)
        if not 0 <= self.sequence_index <= 1_000_000:
            raise ValueError("action-journal sequence index is outside its bound")
        if self.previous_entry_sha256 is not None:
            validate_sha256(self.previous_entry_sha256, field_name="previous_entry_sha256")
        validate_utc_timestamp(self.recorded_at_utc, field_name="recorded_at_utc")
        validate_sha256(self.source_record_sha256, field_name="source_record_sha256")
        require_sorted_unique_ids(
            self.stage_observations,
            attribute="observation_id",
            field_name="stage_observations",
        )
        if not self.stage_observations:
            raise ValueError("action-journal entry requires stage observations")
        if any(value.stage is not self.stage for value in self.stage_observations):
            raise ValueError("action-journal observations differ from their declared stage")
        if len({value.quantity_id for value in self.stage_observations}) != len(
            self.stage_observations
        ):
            raise ValueError("one action-journal stage repeats an action quantity")
        if any(
            value.clock_id != self.interval.clock_id
            or not self.interval.start <= value.clock_coordinate <= self.interval.end
            for value in self.stage_observations
        ):
            raise ValueError("action-journal observations lie outside their mapped interval")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.physical_retry_permitted:
            raise ValueError("physical action is never retryable through the importer")
        if not self.receipt_reconciliation_permitted:
            raise ValueError("episode recovery must permit receipt reconciliation")
        if self.stage in {ActionStage.REQUESTED, ActionStage.ACCEPTED} and (
            self.physical_action_may_have_occurred
        ):
            raise ValueError("pre-application stages cannot assert physical action")
        flags = self.interlock_active or self.clipped or self.saturated
        if self.disposition is ActionJournalEntryDisposition.RECORDED:
            if flags or self.reason_codes:
                raise ValueError("recorded action-journal entry cannot carry failure flags")
            if self.stage in {ActionStage.APPLIED, ActionStage.REALIZED} and not (
                self.physical_action_may_have_occurred
            ):
                raise ValueError("recorded applied/realized stage requires physical occurrence")
        elif not self.reason_codes:
            raise ValueError("non-recorded action-journal entry requires reasons")
        if self.disposition is ActionJournalEntryDisposition.REJECTED and (
            self.physical_action_may_have_occurred
        ):
            raise ValueError("rejected action cannot assert physical occurrence")
        if self.disposition is ActionJournalEntryDisposition.UNCERTAIN and not (
            self.physical_action_may_have_occurred
        ):
            raise ValueError("uncertain action must preserve possible physical occurrence")


def action_journal_sha256(entries: tuple[ActionJournalEntry, ...]) -> str:
    """Hash ordered fixed-length entry fingerprints without reordering chronology."""

    if not entries:
        raise ValueError("action journal must not be empty")
    payload = "".join(value.fingerprint() for value in entries).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _validate_journal(
    entries: tuple[ActionJournalEntry, ...],
    *,
    episode_id: str,
    authority_evidence_id: str,
) -> None:
    if not entries or len(entries) > 1_000_000:
        raise ValueError("physical episode requires a bounded nonempty action journal")
    if len({value.entry_id for value in entries}) != len(entries):
        raise ValueError("action journal repeats an entry identity")
    expected_previous: str | None = None
    previous_recorded_at = None
    stage_by_event: dict[str, list[ActionStage]] = {}
    for index, entry in enumerate(entries):
        if entry.sequence_index != index:
            raise ValueError("action-journal sequence is not contiguous")
        if entry.previous_entry_sha256 != expected_previous:
            raise ValueError("action-journal hash chain is broken")
        if entry.episode_id != episode_id:
            raise ValueError("action-journal entry belongs to another episode")
        if entry.authority_evidence_id != authority_evidence_id:
            raise ValueError("action-journal entry uses another authority evidence record")
        recorded_at = parse_utc_timestamp(entry.recorded_at_utc)
        if previous_recorded_at is not None and recorded_at < previous_recorded_at:
            raise ValueError("action-journal recording chronology regresses")
        stage_by_event.setdefault(entry.action_event_id, []).append(entry.stage)
        expected_previous = entry.fingerprint()
        previous_recorded_at = recorded_at
    stage_order = tuple(ActionStage)
    for stages in stage_by_event.values():
        if tuple(stages) != stage_order[: len(stages)]:
            raise ValueError("action-event stages are duplicated, skipped or out of order")


@dataclass(frozen=True, slots=True)
class PhysicalEpisodeBundle(CanonicalRecord):
    """One externally commanded, immutable, physical independent-unit episode."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/thermodynamic-response/physical-episode-bundle'

    bundle_id: str
    episode_id: str
    system_id: str
    independent_unit_id: str
    preparation_id: str
    protocol_id: str
    apparatus_id: str
    hardware_ids: tuple[str, ...]
    calibration_ids: tuple[str, ...]
    synchronization_id: str
    source_clock_ids: tuple[str, ...]
    canonical_clock_id: str
    clock_mapping_artifacts: tuple[ArtifactIdentity, ...]
    started_at_utc: str
    ended_at_utc: str
    role: PhysicalEpisodeRole
    terminal_status: PhysicalEpisodeTerminalStatus
    world_kind: WorldKind
    evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    authority_evidence: ExternalAuthorityEvidence
    attempt_id: str
    attempt_ordinal: int
    externally_owned_command_plane: bool
    physical_retry_permitted: bool
    receipt_reconciliation_only: bool
    action_journal: tuple[ActionJournalEntry, ...]
    action_journal_sha256: str
    action_quantity_ids: tuple[str, ...]
    receiver_quantity_ids: tuple[str, ...]
    bath_quantity_ids: tuple[str, ...]
    exchange_quantity_ids: tuple[str, ...]
    validity_quantity_ids: tuple[str, ...]
    action_artifacts: tuple[ArtifactIdentity, ...]
    receiver_artifacts: tuple[ArtifactIdentity, ...]
    bath_artifacts: tuple[ArtifactIdentity, ...]
    exchange_artifacts: tuple[ArtifactIdentity, ...]
    validity_artifacts: tuple[ArtifactIdentity, ...]
    calibration_artifacts: tuple[ArtifactIdentity, ...]
    instrument_artifacts: tuple[ArtifactIdentity, ...]
    interlock_triggered: bool
    stop_triggered: bool
    hold_applied: bool
    physical_sink_reason_codes: tuple[str, ...]
    observation_failure_reason_codes: tuple[str, ...]
    epistemic_limitation_reason_codes: tuple[str, ...]
    terminal_source_sha256: str
    acquisition_receipt_id: str
    acquisition_receipt_sha256: str
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("bundle_id", self.bundle_id),
            ("episode_id", self.episode_id),
            ("system_id", self.system_id),
            ("independent_unit_id", self.independent_unit_id),
            ("preparation_id", self.preparation_id),
            ("protocol_id", self.protocol_id),
            ("apparatus_id", self.apparatus_id),
            ("synchronization_id", self.synchronization_id),
            ("canonical_clock_id", self.canonical_clock_id),
            ("attempt_id", self.attempt_id),
            ("acquisition_receipt_id", self.acquisition_receipt_id),
        ):
            validate_stable_id(value, field_name=name)
        for name, values in (
            ("hardware_ids", self.hardware_ids),
            ("calibration_ids", self.calibration_ids),
            ("source_clock_ids", self.source_clock_ids),
            ("action_quantity_ids", self.action_quantity_ids),
            ("receiver_quantity_ids", self.receiver_quantity_ids),
            ("bath_quantity_ids", self.bath_quantity_ids),
            ("exchange_quantity_ids", self.exchange_quantity_ids),
            ("validity_quantity_ids", self.validity_quantity_ids),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)
            for value in values:
                validate_stable_id(value, field_name=name)
        if self.canonical_clock_id not in self.source_clock_ids:
            raise ValueError("canonical episode clock is outside its source clock inventory")
        require_sorted_unique_ids(
            self.clock_mapping_artifacts,
            attribute="artifact_id",
            field_name="clock_mapping_artifacts",
        )
        if len(self.source_clock_ids) > 1 and not self.clock_mapping_artifacts:
            raise ValueError("multi-clock episode requires mapping artifacts")
        start = parse_utc_timestamp(self.started_at_utc, field_name="started_at_utc")
        end = parse_utc_timestamp(self.ended_at_utc, field_name="ended_at_utc")
        if end < start:
            raise ValueError("physical episode end precedes its start")
        if self.world_kind is not WorldKind.PHYSICAL_EXPERIMENT:
            raise ValueError("physical episode bundle requires a physical evidence world")
        if self.evidence_ceiling is not EvidenceCeiling.MEASUREMENT:
            raise ValueError("raw physical episode is capped at measurement")
        expected_access = (
            OutcomeAccess.EVALUATION_SEALED
            if self.role is PhysicalEpisodeRole.EVALUATION
            else OutcomeAccess.DEVELOPMENT_VISIBLE
        )
        if self.outcome_access is not expected_access:
            raise ValueError("physical episode role and outcome access differ")
        authority = self.authority_evidence
        if (
            authority.episode_id != self.episode_id
            or authority.preparation_id != self.preparation_id
            or authority.protocol_id != self.protocol_id
            or authority.apparatus_id != self.apparatus_id
            or any(
                entry.external_actor_id != authority.operator_id for entry in self.action_journal
            )
        ):
            raise ValueError("physical episode differs from its external authority evidence")
        if not set(self.hardware_ids) <= set(authority.hardware_ids):
            raise ValueError("physical episode hardware lies outside external authority")
        if not set(self.calibration_ids) <= set(authority.calibration_ids):
            raise ValueError("physical episode calibration lies outside external authority")
        if start < parse_utc_timestamp(authority.valid_from_utc) or end > parse_utc_timestamp(
            authority.valid_until_utc
        ):
            raise ValueError("physical episode lies outside external authority validity")
        if not self.externally_owned_command_plane:
            raise ValueError("physical episode importer cannot own the external command plane")
        if self.physical_retry_permitted or self.attempt_ordinal != 1:
            raise ValueError("physical episode permits exactly one physical attempt")
        if not self.receipt_reconciliation_only:
            raise ValueError("episode recovery is limited to receipt reconciliation")
        _validate_journal(
            self.action_journal,
            episode_id=self.episode_id,
            authority_evidence_id=authority.evidence_id,
        )
        if self.action_journal_sha256 != action_journal_sha256(self.action_journal):
            raise ValueError("physical episode action-journal digest differs")
        journal_action_ids = {
            observation.quantity_id
            for entry in self.action_journal
            for observation in entry.stage_observations
        }
        if journal_action_ids != set(self.action_quantity_ids):
            raise ValueError("physical episode action quantities differ from its journal")
        if not journal_action_ids <= set(authority.authorized_action_quantity_ids):
            raise ValueError("physical episode action lies outside external authority")
        measurement_ids = {
            *self.receiver_quantity_ids,
            *self.bath_quantity_ids,
            *self.exchange_quantity_ids,
            *self.validity_quantity_ids,
        }
        if not measurement_ids <= set(authority.authorized_measurement_quantity_ids):
            raise ValueError("physical episode measurement lies outside external authority")
        artifact_families = (
            self.clock_mapping_artifacts,
            self.action_artifacts,
            self.receiver_artifacts,
            self.bath_artifacts,
            self.exchange_artifacts,
            self.validity_artifacts,
            self.calibration_artifacts,
            self.instrument_artifacts,
        )
        for artifacts, name, allow_empty in (
            (self.action_artifacts, "action_artifacts", False),
            (self.receiver_artifacts, "receiver_artifacts", False),
            (self.bath_artifacts, "bath_artifacts", False),
            (self.exchange_artifacts, "exchange_artifacts", False),
            (self.validity_artifacts, "validity_artifacts", False),
            (self.calibration_artifacts, "calibration_artifacts", False),
            (self.instrument_artifacts, "instrument_artifacts", False),
        ):
            require_sorted_unique_ids(
                artifacts,
                attribute="artifact_id",
                field_name=name,
            )
            if not allow_empty and not artifacts:
                raise ValueError(f"physical episode requires {name}")
        all_artifact_ids = [value.artifact_id for values in artifact_families for value in values]
        if len(set(all_artifact_ids)) != len(all_artifact_ids):
            raise ValueError("physical episode artifact roles overlap")
        stage_order = tuple(ActionStage)
        stages_by_event: dict[str, list[ActionStage]] = {}
        for entry in self.action_journal:
            stages_by_event.setdefault(entry.action_event_id, []).append(entry.stage)
        journal_complete = all(tuple(value) == stage_order for value in stages_by_event.values())
        all_entries_recorded = all(
            value.disposition is ActionJournalEntryDisposition.RECORDED
            for value in self.action_journal
        )
        any_uncertain = any(
            value.disposition is ActionJournalEntryDisposition.UNCERTAIN
            for value in self.action_journal
        )
        require_sorted_unique_strings(
            self.physical_sink_reason_codes,
            field_name="physical_sink_reason_codes",
        )
        require_sorted_unique_strings(
            self.observation_failure_reason_codes,
            field_name="observation_failure_reason_codes",
        )
        require_sorted_unique_strings(
            self.epistemic_limitation_reason_codes,
            field_name="epistemic_limitation_reason_codes",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        failure_flags = self.interlock_triggered or self.stop_triggered or any_uncertain
        if self.terminal_status is PhysicalEpisodeTerminalStatus.COMPLETED:
            if (
                failure_flags
                or not journal_complete
                or not all_entries_recorded
                or self.reason_codes
            ):
                raise ValueError("completed physical episode has incomplete or invalid action")
        elif not self.reason_codes:
            raise ValueError("noncompleted physical episode requires terminal reasons")
        if self.terminal_status is PhysicalEpisodeTerminalStatus.UNCERTAIN_ACTION and not (
            any_uncertain
        ):
            raise ValueError("uncertain-action episode requires an uncertain journal entry")
        if any_uncertain and self.terminal_status is not (
            PhysicalEpisodeTerminalStatus.UNCERTAIN_ACTION
        ):
            raise ValueError("uncertain journal entry requires uncertain-action terminal status")
        if (self.interlock_triggered or self.stop_triggered) and not self.hold_applied:
            raise ValueError("interlock or stop requires an applied hold")
        validate_sha256(self.terminal_source_sha256, field_name="terminal_source_sha256")
        validate_sha256(
            self.acquisition_receipt_sha256,
            field_name="acquisition_receipt_sha256",
        )


@dataclass(frozen=True, slots=True)
class PhysicalEpisodePublication(CanonicalRecord):
    """Receipt-linked outcome-blind and sealed products for one episode."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/thermodynamic-response/physical-episode-publication'

    publication_id: str
    episode_id: str
    bundle_id: str
    bundle_sha256: str
    action_journal_sha256: str
    split_supported: bool
    entire_episode_sealed: bool
    delivery_publication_id: str
    delivery_artifact_ids: tuple[str, ...]
    delivery_outcome_access: OutcomeAccess
    delivery_receipt_id: str
    delivery_receipt_sha256: str
    sealed_publication_id: str
    sealed_artifact_ids: tuple[str, ...]
    sealed_outcome_access: OutcomeAccess
    sealed_receipt_id: str
    sealed_receipt_sha256: str
    redacted_delivery_attestation_artifact_id: str | None
    physical_retry_permitted: bool
    receipt_reconciliation_only: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("publication_id", self.publication_id),
            ("episode_id", self.episode_id),
            ("bundle_id", self.bundle_id),
            ("delivery_publication_id", self.delivery_publication_id),
            ("delivery_receipt_id", self.delivery_receipt_id),
            ("sealed_publication_id", self.sealed_publication_id),
            ("sealed_receipt_id", self.sealed_receipt_id),
        ):
            validate_stable_id(value, field_name=name)
        for name, digest in (
            ("bundle_sha256", self.bundle_sha256),
            ("action_journal_sha256", self.action_journal_sha256),
            ("delivery_receipt_sha256", self.delivery_receipt_sha256),
            ("sealed_receipt_sha256", self.sealed_receipt_sha256),
        ):
            validate_sha256(digest, field_name=name)
        for name, values in (
            ("delivery_artifact_ids", self.delivery_artifact_ids),
            ("sealed_artifact_ids", self.sealed_artifact_ids),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)
            for value in values:
                validate_stable_id(value, field_name=name)
        if set(self.delivery_artifact_ids) & set(self.sealed_artifact_ids):
            raise ValueError("delivery and sealed publication artifacts overlap")
        if self.delivery_outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("delivery/validity publication must remain outcome-blind")
        if self.sealed_outcome_access is not OutcomeAccess.EVALUATION_SEALED:
            raise ValueError("episode outcome publication must remain evaluation-sealed")
        if self.split_supported == self.entire_episode_sealed:
            raise ValueError("split support and entire-episode sealing are inconsistent")
        if self.split_supported:
            if self.redacted_delivery_attestation_artifact_id is not None:
                raise ValueError("split publication does not use a redacted attestation")
        else:
            if self.redacted_delivery_attestation_artifact_id is None:
                raise ValueError("whole-episode sealing requires a redacted attestation")
            validate_stable_id(
                self.redacted_delivery_attestation_artifact_id,
                field_name="redacted_delivery_attestation_artifact_id",
            )
            if self.delivery_artifact_ids != (self.redacted_delivery_attestation_artifact_id,):
                raise ValueError("whole-episode delivery product must be only its attestation")
        if self.physical_retry_permitted:
            raise ValueError("publication recovery cannot retry physical action")
        if not self.receipt_reconciliation_only:
            raise ValueError("publication recovery is limited to exact receipt reconciliation")


def validate_episode_publication(
    bundle: PhysicalEpisodeBundle,
    publication: PhysicalEpisodePublication,
) -> None:
    """Bind a publication to the exact episode and enforce outcome separation."""

    if (
        publication.episode_id != bundle.episode_id
        or publication.bundle_id != bundle.bundle_id
        or publication.bundle_sha256 != bundle.fingerprint()
        or publication.action_journal_sha256 != bundle.action_journal_sha256
    ):
        raise ValueError("physical episode publication differs from its exact bundle")
    if bundle.role is not PhysicalEpisodeRole.EVALUATION:
        raise ValueError("split/sealed episode publication is reserved for evaluation evidence")
    outcome_blind_ids = {
        value.artifact_id
        for values in (
            bundle.action_artifacts,
            bundle.validity_artifacts,
            bundle.calibration_artifacts,
            bundle.clock_mapping_artifacts,
        )
        for value in values
    }
    protected_ids = {
        value.artifact_id
        for values in (
            bundle.receiver_artifacts,
            bundle.bath_artifacts,
            bundle.exchange_artifacts,
            bundle.instrument_artifacts,
        )
        for value in values
    }
    if publication.split_supported:
        if set(publication.delivery_artifact_ids) != outcome_blind_ids:
            raise ValueError("delivery publication differs from outcome-blind episode roles")
        if set(publication.sealed_artifact_ids) != protected_ids:
            raise ValueError("sealed publication differs from protected episode roles")
    else:
        if set(publication.sealed_artifact_ids) != outcome_blind_ids | protected_ids:
            raise ValueError("whole-episode seal differs from the episode artifact inventory")


def reconcile_episode_publication_receipts(
    publication: PhysicalEpisodePublication,
    *,
    observed_delivery_receipt_id: str | None,
    observed_delivery_receipt_sha256: str | None,
    observed_sealed_receipt_id: str | None,
    observed_sealed_receipt_sha256: str | None,
) -> ReceiptReconciliationDisposition:
    """Reconcile already committed receipts; never repeat acquisition or publication."""

    observed = (
        observed_delivery_receipt_id,
        observed_delivery_receipt_sha256,
        observed_sealed_receipt_id,
        observed_sealed_receipt_sha256,
    )
    if any(value is None for value in observed):
        return ReceiptReconciliationDisposition.RECEIPT_MISSING
    if observed != (
        publication.delivery_receipt_id,
        publication.delivery_receipt_sha256,
        publication.sealed_receipt_id,
        publication.sealed_receipt_sha256,
    ):
        raise ValueError("observed episode publication receipt differs from frozen identity")
    return ReceiptReconciliationDisposition.EXACT_RECEIPT_REPLAY


__all__ = [
    "ActionJournalEntry",
    "ActionJournalEntryDisposition",
    "ExternalAuthorityEvidence",
    "PhysicalEpisodeBundle",
    "PhysicalEpisodePublication",
    "PhysicalEpisodeRole",
    "PhysicalEpisodeTerminalStatus",
    "ReceiptReconciliationDisposition",
    "action_journal_sha256",
    "reconcile_episode_publication_receipts",
    "validate_episode_publication",
]
