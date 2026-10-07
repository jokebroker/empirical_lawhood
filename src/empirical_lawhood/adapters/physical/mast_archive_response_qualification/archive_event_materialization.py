"Pure FAIR-MAST event, command and receiver-clock materialization."

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_stable_id,
)

from .contracts import ArchiveActionClass, ArchiveLawSpec, MastArchiveSourceFeasibilityAmendment, ArchiveResponseClass, classify_archive_response


class ArchiveEventDisposition(StrEnum):
    UNIQUE_ACTIVE_EVENT = "UNIQUE_ACTIVE_EVENT"
    NO_ACTIVE_EVENT = "NO_ACTIVE_EVENT"
    MULTIPLE_DISJOINT_MAXIMUM_PLATEAUS = "MULTIPLE_DISJOINT_MAXIMUM_PLATEAUS"
    INSUFFICIENT_ACTION_WINDOW = "INSUFFICIENT_ACTION_WINDOW"


class ArchiveCommandDisposition(StrEnum):
    COMMAND_STABLE = "COMMAND_STABLE"
    SIMULTANEOUS_COMMAND_CHANGE = "SIMULTANEOUS_COMMAND_CHANGE"
    COMMAND_STATUS_UNQUALIFIED = "COMMAND_STATUS_UNQUALIFIED"


class ArchiveVerticalSliceDisposition(StrEnum):
    COMPLETE_ACTIVE_ROW = "COMPLETE_ACTIVE_ROW"
    NO_ACTIVE_EVENT = "NO_ACTIVE_EVENT"
    EVENT_UNEVALUABLE = "EVENT_UNEVALUABLE"
    CONCURRENT_COMMAND_INVALID = "CONCURRENT_COMMAND_INVALID"
    STATE_HISTORY_UNEVALUABLE = "STATE_HISTORY_UNEVALUABLE"
    ENDPOINT_UNEVALUABLE = "ENDPOINT_UNEVALUABLE"


class ArchiveActionScreenDisposition(StrEnum):
    ELIGIBLE_NBI_DOWN = "ELIGIBLE_NBI_DOWN"
    ELIGIBLE_NBI_UP = "ELIGIBLE_NBI_UP"
    NO_ACTIVE_EVENT = "NO_ACTIVE_EVENT"
    EVENT_UNEVALUABLE = "EVENT_UNEVALUABLE"
    CONCURRENT_COMMAND_INVALID = "CONCURRENT_COMMAND_INVALID"


@dataclass(frozen=True, slots=True)
class ArchiveEventNomination(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive-response-qualification/archive-event-nomination'

    nomination_id: str
    shot_id: int
    disposition: ArchiveEventDisposition
    event_time_s: Decimal | None
    action_class: ArchiveActionClass | None
    pre_median_w: Decimal | None
    post_median_w: Decimal | None
    signed_change_w: Decimal | None
    maximum_plateau_count: int
    maximum_pivot_count: int
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.nomination_id, field_name="nomination_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.shot_id <= 0 or self.maximum_plateau_count < 0 or self.maximum_pivot_count < 0:
            raise ValueError("archive event nomination counts or shot identity are invalid")
        values = (
            self.event_time_s,
            self.pre_median_w,
            self.post_median_w,
            self.signed_change_w,
        )
        for name, value in zip(
            ("event_time_s", "pre_median_w", "post_median_w", "signed_change_w"),
            values,
            strict=True,
        ):
            if value is not None:
                validate_decimal(value, field_name=name)
        if self.disposition is ArchiveEventDisposition.UNIQUE_ACTIVE_EVENT:
            if any(value is None for value in values) or self.action_class not in {
                ArchiveActionClass.NBI_DOWN,
                ArchiveActionClass.NBI_UP,
            }:
                raise ValueError("unique event lacks a complete realized active action")
            if self.maximum_plateau_count != 1 or self.maximum_pivot_count <= 0:
                raise ValueError("unique event does not bind one maximum plateau")
        elif any(value is not None for value in values) or self.action_class is not None:
            raise ValueError("non-event disposition carries an active event")


@dataclass(frozen=True, slots=True)
class ArchiveCommandStability(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive-response-qualification/archive-command-stability'

    audit_id: str
    array_path: str
    disposition: ArchiveCommandDisposition
    observed_channel_count: int
    observed_sample_count: int
    changed_channel_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.audit_id, field_name="audit_id")
        if (
            self.observed_channel_count <= 0
            or self.observed_sample_count < 0
            or tuple(sorted(set(self.changed_channel_indices))) != self.changed_channel_indices
            or any(
                value < 0 or value >= self.observed_channel_count
                for value in self.changed_channel_indices
            )
        ):
            raise ValueError("archive command-stability dimensions are invalid")
        if self.disposition is ArchiveCommandDisposition.COMMAND_STABLE and (
            self.observed_sample_count == 0 or self.changed_channel_indices
        ):
            raise ValueError("stable command audit lacks complete constant samples")
        if (
            self.disposition is ArchiveCommandDisposition.SIMULTANEOUS_COMMAND_CHANGE
            and not self.changed_channel_indices
        ):
            raise ValueError("changed command audit names no changed channel")


@dataclass(frozen=True, slots=True)
class ArchiveHistorySignalContact(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive-response-qualification/archive-history-signal-contact'

    contact_id: str
    signal_id: str
    block_sample_counts: tuple[int, ...]
    block_contact_complete: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.contact_id, field_name="contact_id")
        validate_stable_id(self.signal_id, field_name="signal_id")
        if len(self.block_sample_counts) != 4 or any(
            value < 0 for value in self.block_sample_counts
        ):
            raise ValueError("archive history contact does not bind four valid blocks")
        if self.block_contact_complete is not all(value >= 2 for value in self.block_sample_counts):
            raise ValueError("archive history contact completeness differs from sample counts")


@dataclass(frozen=True, slots=True)
class ArchiveReceiverContact(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive-response-qualification/archive-receiver-contact'

    contact_id: str
    reference_sample_count: int
    reference_median_ev: Decimal
    early_placebo_source_index: int
    early_placebo_time_s: Decimal
    early_placebo_ev: Decimal
    late_placebo_source_index: int
    late_placebo_time_s: Decimal
    late_placebo_ev: Decimal
    endpoint_source_index: int
    endpoint_time_s: Decimal
    endpoint_ev: Decimal
    response_ev: Decimal
    response_class: ArchiveResponseClass

    def __post_init__(self) -> None:
        validate_stable_id(self.contact_id, field_name="contact_id")
        if self.reference_sample_count < 2 or any(
            value < 0
            for value in (
                self.early_placebo_source_index,
                self.late_placebo_source_index,
                self.endpoint_source_index,
            )
        ):
            raise ValueError("archive receiver contact lacks required samples")
        for name in (
            "reference_median_ev",
            "early_placebo_time_s",
            "early_placebo_ev",
            "late_placebo_time_s",
            "late_placebo_ev",
            "endpoint_time_s",
            "endpoint_ev",
            "response_ev",
        ):
            validate_decimal(getattr(self, name), field_name=name)


@dataclass(frozen=True, slots=True)
class ArchiveVerticalSliceRow(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive-response-qualification/archive-vertical-slice-row'

    row_id: str
    shot_id: int
    campaign_id: str
    event: ArchiveEventNomination
    command_audits: tuple[ArchiveCommandStability, ...]
    denominator_field_ids: tuple[str, ...]
    history_contacts: tuple[ArchiveHistorySignalContact, ...]
    action_projection_sample_count: int
    receiver_contact: ArchiveReceiverContact | None
    disposition: ArchiveVerticalSliceDisposition
    reason_codes: tuple[str, ...]
    excluded_from_protected_evidence: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.row_id, field_name="row_id")
        if self.shot_id <= 0 or self.campaign_id not in {"M8", "M9"}:
            raise ValueError("archive vertical-slice row has an invalid shot/campaign")
        if self.event.shot_id != self.shot_id:
            raise ValueError("archive vertical-slice event belongs to another shot")
        require_sorted_unique_ids(
            self.command_audits,
            attribute="audit_id",
            field_name="command_audits",
        )
        require_sorted_unique_strings(
            self.denominator_field_ids,
            field_name="denominator_field_ids",
        )
        require_sorted_unique_ids(
            self.history_contacts,
            attribute="contact_id",
            field_name="history_contacts",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.action_projection_sample_count < 0 or not self.excluded_from_protected_evidence:
            raise ValueError("archive vertical-slice row could promote contaminated evidence")
        no_downstream_contact = (
            not self.command_audits
            and not self.denominator_field_ids
            and not self.history_contacts
            and self.action_projection_sample_count == 0
            and self.receiver_contact is None
        )
        stable_active_event = (
            self.event.disposition is ArchiveEventDisposition.UNIQUE_ACTIVE_EVENT
            and len(self.command_audits) == 4
            and all(
                value.disposition is ArchiveCommandDisposition.COMMAND_STABLE
                for value in self.command_audits
            )
        )
        state_history_complete = (
            len(self.denominator_field_ids) == 8
            and len(self.history_contacts) == 5
            and all(value.block_contact_complete for value in self.history_contacts)
            and self.action_projection_sample_count == 7
        )
        if self.disposition is ArchiveVerticalSliceDisposition.COMPLETE_ACTIVE_ROW:
            if (
                not stable_active_event
                or not state_history_complete
                or self.receiver_contact is None
                or self.reason_codes
            ):
                raise ValueError("complete archive vertical-slice row lacks a required operand")
        elif self.disposition is ArchiveVerticalSliceDisposition.NO_ACTIVE_EVENT:
            if (
                self.event.disposition is not ArchiveEventDisposition.NO_ACTIVE_EVENT
                or not no_downstream_contact
                or not self.reason_codes
            ):
                raise ValueError("no-event archive row accessed downstream operands")
        elif self.disposition is ArchiveVerticalSliceDisposition.EVENT_UNEVALUABLE:
            if (
                self.event.disposition
                in {
                    ArchiveEventDisposition.UNIQUE_ACTIVE_EVENT,
                    ArchiveEventDisposition.NO_ACTIVE_EVENT,
                }
                or not no_downstream_contact
                or not self.reason_codes
            ):
                raise ValueError("event-unevaluable archive row is inconsistent")
        elif self.disposition is ArchiveVerticalSliceDisposition.CONCURRENT_COMMAND_INVALID:
            if (
                self.event.disposition is not ArchiveEventDisposition.UNIQUE_ACTIVE_EVENT
                or len(self.command_audits) != 4
                or all(
                    value.disposition is ArchiveCommandDisposition.COMMAND_STABLE
                    for value in self.command_audits
                )
                or self.denominator_field_ids
                or self.history_contacts
                or self.action_projection_sample_count
                or self.receiver_contact is not None
                or not self.reason_codes
            ):
                raise ValueError("command-invalid archive row is inconsistent")
        elif self.disposition is ArchiveVerticalSliceDisposition.STATE_HISTORY_UNEVALUABLE:
            if (
                not stable_active_event
                or state_history_complete
                or self.receiver_contact is not None
                or not self.reason_codes
            ):
                raise ValueError("state/history-unevaluable archive row is inconsistent")
        elif (
            not stable_active_event
            or not state_history_complete
            or self.receiver_contact is not None
            or not self.reason_codes
        ):
            raise ValueError("endpoint-unevaluable archive row is inconsistent")


@dataclass(frozen=True, slots=True)
class ArchiveVerticalSliceBatch(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive-response-qualification/archive-vertical-slice-batch'

    batch_id: str
    amendment_id: str
    rows: tuple[ArchiveVerticalSliceRow, ...]
    source_chunk_sha256: tuple[str, ...]
    terminal: bool
    excluded_from_protected_evidence: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.batch_id, field_name="batch_id")
        validate_stable_id(self.amendment_id, field_name="amendment_id")
        require_sorted_unique_ids(self.rows, attribute="row_id", field_name="rows")
        require_sorted_unique_strings(
            self.source_chunk_sha256,
            field_name="source_chunk_sha256",
            allow_empty=False,
        )
        if (
            tuple(value.shot_id for value in self.rows) != (27582, 29643)
            or not self.terminal
            or not self.excluded_from_protected_evidence
        ):
            raise ValueError("archive development slice is not the bounded contaminated batch")


@dataclass(frozen=True, slots=True)
class ArchiveActionScreenRow(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive-response-qualification/archive-action-screen-row'

    row_id: str
    shot_id: int
    campaign_id: str
    event: ArchiveEventNomination
    command_audits: tuple[ArchiveCommandStability, ...]
    disposition: ArchiveActionScreenDisposition
    reason_codes: tuple[str, ...]
    receiver_value_accessed: bool
    excluded_from_protected_evidence: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.row_id, field_name="row_id")
        require_sorted_unique_ids(
            self.command_audits,
            attribute="audit_id",
            field_name="command_audits",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if (
            self.shot_id <= 0
            or self.campaign_id not in {"M8", "M9"}
            or self.event.shot_id != self.shot_id
            or self.receiver_value_accessed
            or not self.excluded_from_protected_evidence
        ):
            raise ValueError("archive action-screen row crosses its evidence boundary")
        if self.disposition in {
            ArchiveActionScreenDisposition.ELIGIBLE_NBI_DOWN,
            ArchiveActionScreenDisposition.ELIGIBLE_NBI_UP,
        }:
            expected_action = (
                ArchiveActionClass.NBI_DOWN
                if self.disposition is ArchiveActionScreenDisposition.ELIGIBLE_NBI_DOWN
                else ArchiveActionClass.NBI_UP
            )
            if (
                self.event.disposition is not ArchiveEventDisposition.UNIQUE_ACTIVE_EVENT
                or self.event.action_class is not expected_action
                or len(self.command_audits) != 4
                or any(
                    value.disposition is not ArchiveCommandDisposition.COMMAND_STABLE
                    for value in self.command_audits
                )
                or self.reason_codes
            ):
                raise ValueError("eligible archive action-screen row is incomplete")
        elif self.disposition is ArchiveActionScreenDisposition.NO_ACTIVE_EVENT:
            if (
                self.event.disposition is not ArchiveEventDisposition.NO_ACTIVE_EVENT
                or self.command_audits
            ):
                raise ValueError("no-event action-screen row accessed command operands")
        elif self.disposition is ArchiveActionScreenDisposition.EVENT_UNEVALUABLE:
            if (
                self.event.disposition
                in {
                    ArchiveEventDisposition.UNIQUE_ACTIVE_EVENT,
                    ArchiveEventDisposition.NO_ACTIVE_EVENT,
                }
                or self.command_audits
            ):
                raise ValueError("event-unevaluable action-screen row is inconsistent")
        elif (
            self.event.disposition is not ArchiveEventDisposition.UNIQUE_ACTIVE_EVENT
            or len(self.command_audits) != 4
            or all(
                value.disposition is ArchiveCommandDisposition.COMMAND_STABLE
                for value in self.command_audits
            )
        ):
            raise ValueError("command-invalid action-screen row has no command failure")


@dataclass(frozen=True, slots=True)
class ArchiveActionScreenCampaignSummary(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive-response-qualification/archive-action-screen-campaign-summary'

    summary_id: str
    campaign_id: str
    attempted_count: int
    eligible_down_count: int
    eligible_up_count: int
    no_active_event_count: int
    event_unevaluable_count: int
    command_invalid_count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.summary_id, field_name="summary_id")
        counts = (
            self.eligible_down_count,
            self.eligible_up_count,
            self.no_active_event_count,
            self.event_unevaluable_count,
            self.command_invalid_count,
        )
        if (
            self.campaign_id not in {"M8", "M9"}
            or self.attempted_count != 10
            or any(value < 0 for value in counts)
            or sum(counts) != self.attempted_count
        ):
            raise ValueError("archive action-screen campaign accounting differs")


@dataclass(frozen=True, slots=True)
class ArchiveActionScreenBatch(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive-response-qualification/archive-action-screen-batch'

    batch_id: str
    screen_id: str
    rows: tuple[ArchiveActionScreenRow, ...]
    campaign_summaries: tuple[ArchiveActionScreenCampaignSummary, ...]
    source_chunk_sha256: tuple[str, ...]
    terminal: bool
    receiver_value_access_count: int
    excluded_from_protected_evidence: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.batch_id, field_name="batch_id")
        validate_stable_id(self.screen_id, field_name="screen_id")
        require_sorted_unique_ids(self.rows, attribute="row_id", field_name="rows")
        require_sorted_unique_ids(
            self.campaign_summaries,
            attribute="summary_id",
            field_name="campaign_summaries",
        )
        require_sorted_unique_strings(
            self.source_chunk_sha256,
            field_name="source_chunk_sha256",
            allow_empty=False,
        )
        if (
            len(self.rows) != 20
            or tuple(value.campaign_id for value in self.campaign_summaries) != ("M8", "M9")
            or not self.terminal
            or self.receiver_value_access_count != 0
            or not self.excluded_from_protected_evidence
        ):
            raise ValueError("archive action-screen batch is not the bounded terminal sample")
        for summary in self.campaign_summaries:
            members = [value for value in self.rows if value.campaign_id == summary.campaign_id]
            expected = {
                "eligible_down_count": sum(
                    value.disposition is ArchiveActionScreenDisposition.ELIGIBLE_NBI_DOWN
                    for value in members
                ),
                "eligible_up_count": sum(
                    value.disposition is ArchiveActionScreenDisposition.ELIGIBLE_NBI_UP
                    for value in members
                ),
                "no_active_event_count": sum(
                    value.disposition is ArchiveActionScreenDisposition.NO_ACTIVE_EVENT
                    for value in members
                ),
                "event_unevaluable_count": sum(
                    value.disposition is ArchiveActionScreenDisposition.EVENT_UNEVALUABLE
                    for value in members
                ),
                "command_invalid_count": sum(
                    value.disposition is ArchiveActionScreenDisposition.CONCURRENT_COMMAND_INVALID
                    for value in members
                ),
            }
            if len(members) != summary.attempted_count or any(
                getattr(summary, name) != count for name, count in expected.items()
            ):
                raise ValueError("archive action-screen summary does not derive from its rows")


def _as_source_vector(name: str, value: npt.ArrayLike) -> npt.NDArray[np.float64]:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be one nonempty source vector")
    return result


def _decimal(value: float) -> Decimal:
    return Decimal(format(value, ".17g"))


def _median(value: npt.NDArray[np.float64]) -> float:
    return float(np.median(value))


def nominate_archive_event(
    *,
    shot_id: int,
    source_time_s: npt.ArrayLike,
    realized_nbi_power_w: npt.ArrayLike,
    nbi_start_s: Decimal,
    nbi_end_s: Decimal,
    amendment: MastArchiveSourceFeasibilityAmendment,
) -> ArchiveEventNomination:
    "Nominate the event without receiver or non-action fields."

    if amendment.nbi_hold_available:
        raise ValueError("archive event implementation cannot enable HOLD")
    time_s = _as_source_vector("source_time_s", source_time_s)
    power_w = _as_source_vector("realized_nbi_power_w", realized_nbi_power_w)
    if time_s.shape != power_w.shape or np.any(np.diff(time_s) <= 0):
        raise ValueError("NBI source time/power shapes or clock order differ")
    start_s = float(nbi_start_s)
    end_s = float(nbi_end_s)
    candidates: list[tuple[int, float, float, float]] = []
    had_complete_window = False
    for index, pivot_s in enumerate(time_s):
        if not np.isfinite(pivot_s) or not (
            start_s + 0.020 - 1e-12 <= pivot_s <= end_s - 0.080 + 1e-12
        ):
            continue
        pre = power_w[
            (time_s >= pivot_s - 0.010 - 1e-12)
            & (time_s <= pivot_s - 0.002 + 1e-12)
            & np.isfinite(power_w)
        ]
        post = power_w[
            (time_s >= pivot_s + 0.002 - 1e-12)
            & (time_s <= pivot_s + 0.010 + 1e-12)
            & np.isfinite(power_w)
        ]
        if pre.size < 2 or post.size < 2:
            continue
        had_complete_window = True
        pre_median = _median(pre)
        post_median = _median(post)
        signed_change = post_median - pre_median
        if abs(signed_change) >= 100_000.0:
            candidates.append((index, pre_median, post_median, signed_change))
    if not candidates:
        disposition = (
            ArchiveEventDisposition.NO_ACTIVE_EVENT
            if had_complete_window
            else ArchiveEventDisposition.INSUFFICIENT_ACTION_WINDOW
        )
        reason = "NO_ACTIVE_NBI_CHANGE" if had_complete_window else "INSUFFICIENT_ACTION_WINDOW"
        return ArchiveEventNomination(
            nomination_id=f"archive-action-event-nomination.{shot_id}",
            shot_id=shot_id,
            disposition=disposition,
            event_time_s=None,
            action_class=None,
            pre_median_w=None,
            post_median_w=None,
            signed_change_w=None,
            maximum_plateau_count=0,
            maximum_pivot_count=0,
            reason_codes=(reason,),
        )
    greatest = max(abs(value[3]) for value in candidates)
    maxima = [value for value in candidates if abs(value[3]) == greatest]
    plateau_count = 1 + sum(
        right[0] != left[0] + 1 for left, right in zip(maxima, maxima[1:], strict=False)
    )
    if plateau_count != 1:
        return ArchiveEventNomination(
            nomination_id=f"archive-action-event-nomination.{shot_id}",
            shot_id=shot_id,
            disposition=ArchiveEventDisposition.MULTIPLE_DISJOINT_MAXIMUM_PLATEAUS,
            event_time_s=None,
            action_class=None,
            pre_median_w=None,
            post_median_w=None,
            signed_change_w=None,
            maximum_plateau_count=plateau_count,
            maximum_pivot_count=len(maxima),
            reason_codes=("MULTIPLE_DISJOINT_MAXIMUM_PLATEAUS",),
        )
    index, pre_median, post_median, signed_change = maxima[0]
    action_class = ArchiveActionClass.NBI_UP if signed_change > 0 else ArchiveActionClass.NBI_DOWN
    return ArchiveEventNomination(
        nomination_id=f"archive-action-event-nomination.{shot_id}",
        shot_id=shot_id,
        disposition=ArchiveEventDisposition.UNIQUE_ACTIVE_EVENT,
        event_time_s=_decimal(float(time_s[index])),
        action_class=action_class,
        pre_median_w=_decimal(pre_median),
        post_median_w=_decimal(post_median),
        signed_change_w=_decimal(signed_change),
        maximum_plateau_count=1,
        maximum_pivot_count=len(maxima),
        reason_codes=(),
    )


def audit_archive_command_stability(
    *,
    audit_id: str,
    array_path: str,
    source_time_s: npt.ArrayLike,
    command_values: npt.ArrayLike,
    event_time_s: Decimal,
    amendment: MastArchiveSourceFeasibilityAmendment,
) -> ArchiveCommandStability:
    """Require exact command constancy on the declared Level-2 window."""

    if array_path not in amendment.concurrent_command_array_paths:
        raise ValueError("command array is outside the development exclusion inventory")
    time_s = _as_source_vector("source_time_s", source_time_s)
    values = np.asarray(command_values, dtype=np.float64)
    if values.ndim == 1:
        values = values[np.newaxis, :]
    if values.ndim != 2 or values.shape[-1] != time_s.size:
        raise ValueError("command array must have time on its final axis")
    event_s = float(event_time_s)
    mask = (time_s >= event_s - 0.010 - 1e-12) & (time_s <= event_s + 0.010 + 1e-12)
    window = values[:, mask]
    if window.shape[1] == 0 or not np.all(np.isfinite(window)):
        return ArchiveCommandStability(
            audit_id=audit_id,
            array_path=array_path,
            disposition=ArchiveCommandDisposition.COMMAND_STATUS_UNQUALIFIED,
            observed_channel_count=values.shape[0],
            observed_sample_count=window.shape[1],
            changed_channel_indices=(),
        )
    changed = tuple(int(index) for index in np.flatnonzero(np.any(window != window[:, :1], axis=1)))
    disposition = (
        ArchiveCommandDisposition.SIMULTANEOUS_COMMAND_CHANGE
        if changed
        else ArchiveCommandDisposition.COMMAND_STABLE
    )
    return ArchiveCommandStability(
        audit_id=audit_id,
        array_path=array_path,
        disposition=disposition,
        observed_channel_count=values.shape[0],
        observed_sample_count=window.shape[1],
        changed_channel_indices=changed,
    )


def nearest_receiver_sample_index(
    *,
    receiver_time_s: npt.ArrayLike,
    target_time_s: Decimal,
    amendment: MastArchiveSourceFeasibilityAmendment,
) -> int | None:
    """Return the nearest in-tolerance sample, choosing the earlier exact tie."""

    time_s = _as_source_vector("receiver_time_s", receiver_time_s)
    finite_indices = np.flatnonzero(np.isfinite(time_s))
    if finite_indices.size == 0:
        return None
    target = float(target_time_s)
    errors = np.abs(time_s[finite_indices] - target)
    minimum = float(np.min(errors))
    tolerance_s = float(amendment.receiver_sample_tolerance_ms) / 1000.0
    if minimum > tolerance_s + 1e-12:
        return None
    tied = finite_indices[np.isclose(errors, minimum, rtol=0.0, atol=1e-12)]
    return int(tied[np.argmin(time_s[tied])])


def audit_archive_history_contact(
    *,
    shot_id: int,
    signal_id: str,
    source_time_s: npt.ArrayLike,
    signal_values: npt.ArrayLike,
    event_time_s: Decimal,
    archive_law: ArchiveLawSpec,
) -> ArchiveHistorySignalContact:
    """Contact the four frozen causal-history blocks without imputation."""

    if signal_id not in archive_law.state_history.history_signal_ids:
        raise ValueError("history signal lies outside the archive-law inventory")
    time_s = _as_source_vector("source_time_s", source_time_s)
    values = _as_source_vector("signal_values", signal_values)
    if time_s.shape != values.shape or np.any(np.diff(time_s) <= 0):
        raise ValueError("history source time/value shapes or clock order differ")
    event_s = float(event_time_s)
    relative_ms = (time_s - event_s) * 1000.0
    block_counts = []
    for block in archive_law.state_history.history_blocks_ms:
        start_ms, end_ms = (float(value) for value in block.split(":"))
        at_start = np.isclose(relative_ms, start_ms, rtol=0.0, atol=1e-9)
        at_end = np.isclose(relative_ms, end_ms, rtol=0.0, atol=1e-9)
        mask = (
            ((relative_ms > start_ms) | at_start)
            & (relative_ms < end_ms)
            & ~at_end
            & np.isfinite(values)
        )
        block_counts.append(int(np.count_nonzero(mask)))
    counts = tuple(block_counts)
    return ArchiveHistorySignalContact(
        contact_id=f"archive-history-signal-contact.{shot_id}.{signal_id}",
        signal_id=signal_id,
        block_sample_counts=counts,
        block_contact_complete=all(value >= 2 for value in counts),
    )


def materialize_archive_receiver_contact(
    *,
    shot_id: int,
    receiver_time_s: npt.ArrayLike,
    receiver_ev: npt.ArrayLike,
    event_time_s: Decimal,
    archive_law: ArchiveLawSpec,
    amendment: MastArchiveSourceFeasibilityAmendment,
) -> ArchiveReceiverContact | None:
    """Materialize the unchanged endpoint and amended nearest-clock rule."""

    time_s = _as_source_vector("receiver_time_s", receiver_time_s)
    values_ev = _as_source_vector("receiver_ev", receiver_ev)
    if time_s.shape != values_ev.shape or np.any(np.diff(time_s) <= 0):
        raise ValueError("receiver source time/value shapes or clock order differ")
    event_s = float(event_time_s)
    relative_ms = (time_s - event_s) * 1000.0
    reference_start_ms = float(archive_law.endpoint.reference_window_start_ms)
    reference_end_ms = float(archive_law.endpoint.reference_window_end_ms)
    reference_mask = (
        (
            (relative_ms > reference_start_ms)
            | np.isclose(relative_ms, reference_start_ms, rtol=0.0, atol=1e-9)
        )
        & (relative_ms < reference_end_ms)
        & ~np.isclose(relative_ms, reference_end_ms, rtol=0.0, atol=1e-9)
        & np.isfinite(values_ev)
    )
    reference_values = values_ev[reference_mask]
    if reference_values.size < 2:
        return None
    offsets_s = (
        Decimal("-0.042"),
        Decimal("-0.002"),
        archive_law.endpoint.endpoint_offset_ms / Decimal(1000),
    )
    indices = tuple(
        nearest_receiver_sample_index(
            receiver_time_s=time_s,
            target_time_s=event_time_s + offset,
            amendment=amendment,
        )
        for offset in offsets_s
    )
    if any(index is None for index in indices):
        return None
    early_index, late_index, endpoint_index = indices
    assert early_index is not None
    assert late_index is not None
    assert endpoint_index is not None
    selected_values = values_ev[[early_index, late_index, endpoint_index]]
    if not np.all(np.isfinite(selected_values)):
        return None
    reference_median = _median(reference_values)
    response = float(values_ev[endpoint_index]) - reference_median
    if not (
        float(archive_law.endpoint.receiver_valid_min_ev)
        <= float(values_ev[endpoint_index])
        <= float(archive_law.endpoint.receiver_valid_max_ev)
    ):
        return None
    response_decimal = _decimal(response)
    return ArchiveReceiverContact(
        contact_id=f"archive-receiver-contact.{shot_id}",
        reference_sample_count=int(reference_values.size),
        reference_median_ev=_decimal(reference_median),
        early_placebo_source_index=early_index,
        early_placebo_time_s=_decimal(float(time_s[early_index])),
        early_placebo_ev=_decimal(float(values_ev[early_index])),
        late_placebo_source_index=late_index,
        late_placebo_time_s=_decimal(float(time_s[late_index])),
        late_placebo_ev=_decimal(float(values_ev[late_index])),
        endpoint_source_index=endpoint_index,
        endpoint_time_s=_decimal(float(time_s[endpoint_index])),
        endpoint_ev=_decimal(float(values_ev[endpoint_index])),
        response_ev=response_decimal,
        response_class=classify_archive_response(
            response_decimal,
            endpoint=archive_law.endpoint,
        ),
    )


__all__ = [
    'ArchiveActionScreenBatch',
    'ArchiveActionScreenCampaignSummary',
    'ArchiveActionScreenDisposition',
    'ArchiveActionScreenRow',
    'ArchiveCommandDisposition',
    'ArchiveCommandStability',
    'ArchiveEventDisposition',
    'ArchiveEventNomination',
    'ArchiveHistorySignalContact',
    'ArchiveReceiverContact',
    'ArchiveVerticalSliceBatch',
    'ArchiveVerticalSliceDisposition',
    'ArchiveVerticalSliceRow',
    'audit_archive_command_stability',
    'audit_archive_history_contact',
    'materialize_archive_receiver_contact',
    'nearest_receiver_sample_index',
    'nominate_archive_event',
]
