"""Durable event-sourced execution envelope with nonrefundable reservations."""

from __future__ import annotations

import hashlib
from dataclasses import InitVar, dataclass, replace
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    canonical_json_bytes,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_schema,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.time import parse_utc_timestamp


class ExecutionEnvelopeState(StrEnum):
    ISSUED = "ISSUED"
    RESERVED = "RESERVED"
    LAUNCHED = "LAUNCHED"
    RECEIPT_OBSERVED = "RECEIPT_OBSERVED"
    PUBLISHED = "PUBLISHED"
    RECOVERY_PUBLICATION = "RECOVERY_PUBLICATION"
    TERMINAL = "TERMINAL"
    RESOURCE_STOP = "RESOURCE_STOP"
    UNKNOWN_COMPLETION = "UNKNOWN_COMPLETION"


class ExecutionEnvelopeEventKind(StrEnum):
    RESERVATION = "RESERVATION"
    LAUNCH = "LAUNCH"
    RECEIPT = "RECEIPT"
    OPERATIONAL_FAILURE = "OPERATIONAL_FAILURE"
    DUPLICATE_RECEIPT = "DUPLICATE_RECEIPT"
    PUBLICATION = "PUBLICATION"
    RECOVERY_PUBLICATION_STARTED = "RECOVERY_PUBLICATION_STARTED"
    RECOVERY_PUBLICATION_FINISHED = "RECOVERY_PUBLICATION_FINISHED"
    TERMINAL = "TERMINAL"
    RESOURCE_STOP = "RESOURCE_STOP"
    UNKNOWN_COMPLETION = "UNKNOWN_COMPLETION"


@dataclass(frozen=True, slots=True)
class ExecutionEnvelopeCellSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/execution-envelope-cell-spec'

    cell_id: str
    task_id: str
    physical_independent_unit_id: str
    maximum_physical_tokens: int
    maximum_retry_tokens: int

    def __post_init__(self) -> None:
        for name, value in (
            ("cell_id", self.cell_id),
            ("task_id", self.task_id),
            ("physical_independent_unit_id", self.physical_independent_unit_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.maximum_physical_tokens < 1 or self.maximum_retry_tokens < 0:
            raise ValueError("execution-envelope cell token limits are invalid")


@dataclass(frozen=True, slots=True)
class ExecutionEnvelopeSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/execution-envelope-spec'

    envelope_spec_id: str
    issued_study_extensions: ObjectIdentity
    cells: tuple[ExecutionEnvelopeCellSpec, ...]
    global_physical_token_limit: int
    global_retry_token_limit: int
    allowlisted_retry_reason_codes: tuple[str, ...]
    deadline_utc: str
    resource_contract_sha256: str
    nonrefundable_reservations: bool
    first_valid_success_wins: bool
    unknown_completion_requires_new_disposition: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.envelope_spec_id, field_name="envelope_spec_id")
        require_sorted_unique_ids(self.cells, attribute="cell_id", field_name="cells")
        require_sorted_unique_strings(
            self.allowlisted_retry_reason_codes,
            field_name="allowlisted_retry_reason_codes",
        )
        parse_utc_timestamp(self.deadline_utc, field_name="deadline_utc")
        validate_sha256(
            self.resource_contract_sha256,
            field_name="resource_contract_sha256",
        )
        if not self.cells:
            raise ValueError("execution envelope requires cells")
        if self.global_physical_token_limit < len(self.cells) or self.global_retry_token_limit < 0:
            raise ValueError("execution-envelope global token limits are invalid")
        cell_coordinates = tuple(
            (value.task_id, value.physical_independent_unit_id) for value in self.cells
        )
        if len(set(cell_coordinates)) != len(cell_coordinates):
            raise ValueError("execution-envelope task/unit coordinates must be unique")
        if len({value.task_id for value in self.cells}) != len(self.cells):
            raise ValueError("standard execution-envelope cells must map one-to-one to tasks")
        if any(
            value.maximum_physical_tokens != value.maximum_retry_tokens + 1 for value in self.cells
        ):
            raise ValueError(
                "execution-envelope physical limits must equal initial plus retry tokens"
            )
        if not (
            self.nonrefundable_reservations
            and self.first_valid_success_wins
            and self.unknown_completion_requires_new_disposition
        ):
            raise ValueError("execution envelope weakens a frozen durability rule")


@dataclass(frozen=True, slots=True)
class ExecutionEnvelopeEvent(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/execution-envelope-event'

    event_id: str
    sequence_number: int
    occurred_at_utc: str
    kind: ExecutionEnvelopeEventKind
    cell_id: str
    attempt_id: str | None
    from_state: ExecutionEnvelopeState
    to_state: ExecutionEnvelopeState
    reason_code: str | None
    receipt: ObjectIdentity | None
    artifact: ObjectIdentity | None
    valid_receipt: bool | None
    scientific_terminal: bool | None

    def __post_init__(self) -> None:
        validate_stable_id(self.event_id, field_name="event_id")
        validate_stable_id(self.cell_id, field_name="cell_id")
        if self.attempt_id is not None:
            validate_stable_id(self.attempt_id, field_name="attempt_id")
        if self.reason_code is not None:
            validate_nonempty(self.reason_code, field_name="reason_code")
        if self.sequence_number < 1:
            raise ValueError("execution-envelope event sequence must be positive")
        parse_utc_timestamp(self.occurred_at_utc, field_name="occurred_at_utc")


@dataclass(frozen=True, slots=True)
class ExecutionEnvelopeCellState(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/execution-envelope-cell-state'

    cell_id: str
    state: ExecutionEnvelopeState
    physical_tokens_reserved: int
    retry_tokens_reserved: int
    current_attempt_id: str | None
    first_valid_receipt: ObjectIdentity | None
    published_artifact: ObjectIdentity | None
    duplicate_receipts: tuple[ObjectIdentity, ...]
    last_operational_reason_code: str | None

    def __post_init__(self) -> None:
        validate_stable_id(self.cell_id, field_name="cell_id")
        require_sorted_unique_ids(
            self.duplicate_receipts,
            attribute="object_id",
            field_name="duplicate_receipts",
        )
        if self.physical_tokens_reserved < 0 or self.retry_tokens_reserved < 0:
            raise ValueError("execution-envelope token counts cannot be negative")
        if self.retry_tokens_reserved > max(0, self.physical_tokens_reserved - 1):
            raise ValueError("execution-envelope retry tokens exceed reservations")
        if self.state is ExecutionEnvelopeState.ISSUED and (
            self.physical_tokens_reserved
            or self.current_attempt_id is not None
            or self.first_valid_receipt is not None
            or self.published_artifact is not None
        ):
            raise ValueError("issued envelope cell already contains execution state")
        if self.published_artifact is not None and self.first_valid_receipt is None:
            raise ValueError("published envelope artifact lacks a first valid receipt")


@dataclass(frozen=True, slots=True)
class RunExecutionEnvelope(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/run-execution-envelope'

    envelope_id: str
    spec: ExecutionEnvelopeSpec
    execution_plan: ObjectIdentity
    cells: tuple[ExecutionEnvelopeCellState, ...]
    events: tuple[ExecutionEnvelopeEvent, ...]
    global_physical_tokens_reserved: int
    global_retry_tokens_reserved: int

    def __post_init__(self) -> None:
        validate_stable_id(self.envelope_id, field_name="envelope_id")
        require_sorted_unique_ids(self.cells, attribute="cell_id", field_name="cells")
        if tuple(value.cell_id for value in self.cells) != tuple(
            value.cell_id for value in self.spec.cells
        ):
            raise ValueError("execution-envelope state roster differs from its spec")
        event_ids = tuple(value.event_id for value in self.events)
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("execution-envelope event identity is duplicated")
        if tuple(value.sequence_number for value in self.events) != tuple(
            range(1, len(self.events) + 1)
        ):
            raise ValueError("execution-envelope event sequence is not contiguous")
        if any(
            parse_utc_timestamp(later.occurred_at_utc, field_name="occurred_at_utc")
            < parse_utc_timestamp(earlier.occurred_at_utc, field_name="occurred_at_utc")
            for earlier, later in zip(self.events, self.events[1:], strict=False)
        ):
            raise ValueError("execution-envelope events are not time ordered")
        expected_physical = sum(value.physical_tokens_reserved for value in self.cells)
        expected_retry = sum(value.retry_tokens_reserved for value in self.cells)
        if (
            self.global_physical_tokens_reserved != expected_physical
            or self.global_retry_tokens_reserved != expected_retry
        ):
            raise ValueError("execution-envelope global tokens differ from cell reservations")
        if (
            expected_physical > self.spec.global_physical_token_limit
            or expected_retry > self.spec.global_retry_token_limit
        ):
            raise ValueError("execution-envelope global token limit is exceeded")

    def cell(self, cell_id: str) -> ExecutionEnvelopeCellState:
        for value in self.cells:
            if value.cell_id == cell_id:
                return value
        raise KeyError(cell_id)


class RunExecutionEnvelopeMachine:
    """Pure transitions; durability is supplied by an injected compare-and-append store."""

    @staticmethod
    def initial(
        *,
        envelope_id: str,
        spec: ExecutionEnvelopeSpec,
        execution_plan: ObjectIdentity,
    ) -> RunExecutionEnvelope:
        return RunExecutionEnvelope(
            envelope_id=envelope_id,
            spec=spec,
            execution_plan=execution_plan,
            cells=tuple(
                ExecutionEnvelopeCellState(
                    cell_id=value.cell_id,
                    state=ExecutionEnvelopeState.ISSUED,
                    physical_tokens_reserved=0,
                    retry_tokens_reserved=0,
                    current_attempt_id=None,
                    first_valid_receipt=None,
                    published_artifact=None,
                    duplicate_receipts=(),
                    last_operational_reason_code=None,
                )
                for value in spec.cells
            ),
            events=(),
            global_physical_tokens_reserved=0,
            global_retry_tokens_reserved=0,
        )

    @staticmethod
    def _append(
        envelope: RunExecutionEnvelope,
        *,
        event: ExecutionEnvelopeEvent,
        cell: ExecutionEnvelopeCellState,
    ) -> RunExecutionEnvelope:
        current = envelope.cell(cell.cell_id)
        if event.sequence_number != len(envelope.events) + 1:
            raise ValueError("execution-envelope append sequence differs")
        if event.from_state is not current.state or event.to_state is not cell.state:
            raise ValueError("execution-envelope event/state transition differs")
        cells = tuple(cell if value.cell_id == cell.cell_id else value for value in envelope.cells)
        return replace(
            envelope,
            cells=cells,
            events=(*envelope.events, event),
            global_physical_tokens_reserved=sum(value.physical_tokens_reserved for value in cells),
            global_retry_tokens_reserved=sum(value.retry_tokens_reserved for value in cells),
        )

    @staticmethod
    def _event(
        envelope: RunExecutionEnvelope,
        *,
        event_id: str,
        occurred_at_utc: str,
        kind: ExecutionEnvelopeEventKind,
        cell_id: str,
        attempt_id: str | None,
        from_state: ExecutionEnvelopeState,
        to_state: ExecutionEnvelopeState,
        reason_code: str | None = None,
        receipt: ObjectIdentity | None = None,
        artifact: ObjectIdentity | None = None,
        valid_receipt: bool | None = None,
        scientific_terminal: bool | None = None,
    ) -> ExecutionEnvelopeEvent:
        return ExecutionEnvelopeEvent(
            event_id=event_id,
            sequence_number=len(envelope.events) + 1,
            occurred_at_utc=occurred_at_utc,
            kind=kind,
            cell_id=cell_id,
            attempt_id=attempt_id,
            from_state=from_state,
            to_state=to_state,
            reason_code=reason_code,
            receipt=receipt,
            artifact=artifact,
            valid_receipt=valid_receipt,
            scientific_terminal=scientific_terminal,
        )

    @classmethod
    def reserve(
        cls,
        envelope: RunExecutionEnvelope,
        *,
        event_id: str,
        occurred_at_utc: str,
        cell_id: str,
        attempt_id: str,
        retry_reason_code: str | None = None,
    ) -> RunExecutionEnvelope:
        cell = envelope.cell(cell_id)
        spec = next(value for value in envelope.spec.cells if value.cell_id == cell_id)
        retry = cell.physical_tokens_reserved > 0
        if parse_utc_timestamp(
            occurred_at_utc,
            field_name="occurred_at_utc",
        ) > parse_utc_timestamp(envelope.spec.deadline_utc, field_name="deadline_utc"):
            raise ValueError("execution-envelope reservation is after its deadline")
        if retry:
            if (
                cell.state is not ExecutionEnvelopeState.RECEIPT_OBSERVED
                or cell.first_valid_receipt is not None
                or retry_reason_code is None
                or retry_reason_code != cell.last_operational_reason_code
                or retry_reason_code not in envelope.spec.allowlisted_retry_reason_codes
            ):
                raise ValueError("execution-envelope retry is not allowlisted")
        elif cell.state is not ExecutionEnvelopeState.ISSUED or retry_reason_code is not None:
            raise ValueError("first execution-envelope reservation is malformed")
        physical = cell.physical_tokens_reserved + 1
        retries = cell.retry_tokens_reserved + int(retry)
        if (
            physical > spec.maximum_physical_tokens
            or retries > spec.maximum_retry_tokens
            or envelope.global_physical_tokens_reserved + 1
            > envelope.spec.global_physical_token_limit
            or envelope.global_retry_tokens_reserved + int(retry)
            > envelope.spec.global_retry_token_limit
        ):
            raise ValueError("execution-envelope reservation exceeds a token limit")
        updated = replace(
            cell,
            state=ExecutionEnvelopeState.RESERVED,
            physical_tokens_reserved=physical,
            retry_tokens_reserved=retries,
            current_attempt_id=attempt_id,
            last_operational_reason_code=None,
        )
        event = cls._event(
            envelope,
            event_id=event_id,
            occurred_at_utc=occurred_at_utc,
            kind=ExecutionEnvelopeEventKind.RESERVATION,
            cell_id=cell_id,
            attempt_id=attempt_id,
            from_state=cell.state,
            to_state=updated.state,
            reason_code=retry_reason_code,
        )
        return cls._append(envelope, event=event, cell=updated)

    @classmethod
    def launch(
        cls,
        envelope: RunExecutionEnvelope,
        *,
        event_id: str,
        occurred_at_utc: str,
        cell_id: str,
        attempt_id: str,
    ) -> RunExecutionEnvelope:
        cell = envelope.cell(cell_id)
        if (
            cell.state is not ExecutionEnvelopeState.RESERVED
            or cell.current_attempt_id != attempt_id
        ):
            raise ValueError("execution-envelope launch lacks its durable reservation")
        updated = replace(cell, state=ExecutionEnvelopeState.LAUNCHED)
        return cls._append(
            envelope,
            event=cls._event(
                envelope,
                event_id=event_id,
                occurred_at_utc=occurred_at_utc,
                kind=ExecutionEnvelopeEventKind.LAUNCH,
                cell_id=cell_id,
                attempt_id=attempt_id,
                from_state=cell.state,
                to_state=updated.state,
            ),
            cell=updated,
        )

    @classmethod
    def observe_receipt(
        cls,
        envelope: RunExecutionEnvelope,
        *,
        event_id: str,
        occurred_at_utc: str,
        cell_id: str,
        attempt_id: str,
        receipt: ObjectIdentity,
        valid_receipt: bool,
        scientific_terminal: bool,
        operational_reason_code: str | None = None,
    ) -> RunExecutionEnvelope:
        cell = envelope.cell(cell_id)
        if cell.state is not ExecutionEnvelopeState.LAUNCHED or (
            cell.current_attempt_id != attempt_id
        ):
            raise ValueError("execution-envelope receipt lacks an exact launched attempt")
        if valid_receipt:
            if operational_reason_code is not None or cell.first_valid_receipt is not None:
                raise ValueError("first-valid execution receipt is malformed")
            first = receipt
        else:
            if scientific_terminal or operational_reason_code is None:
                raise ValueError("operational failure receipt is malformed")
            first = None
        updated = replace(
            cell,
            state=ExecutionEnvelopeState.RECEIPT_OBSERVED,
            first_valid_receipt=first,
            last_operational_reason_code=operational_reason_code,
        )
        return cls._append(
            envelope,
            event=cls._event(
                envelope,
                event_id=event_id,
                occurred_at_utc=occurred_at_utc,
                kind=ExecutionEnvelopeEventKind.RECEIPT,
                cell_id=cell_id,
                attempt_id=attempt_id,
                from_state=cell.state,
                to_state=updated.state,
                reason_code=operational_reason_code,
                receipt=receipt,
                valid_receipt=valid_receipt,
                scientific_terminal=scientific_terminal,
            ),
            cell=updated,
        )

    @classmethod
    def observe_operational_failure(
        cls,
        envelope: RunExecutionEnvelope,
        *,
        event_id: str,
        occurred_at_utc: str,
        cell_id: str,
        attempt_id: str,
        reason_code: str,
    ) -> RunExecutionEnvelope:
        cell = envelope.cell(cell_id)
        if (
            cell.state is not ExecutionEnvelopeState.LAUNCHED
            or cell.current_attempt_id != attempt_id
        ):
            raise ValueError("execution-envelope failure lacks an exact launched attempt")
        updated = replace(
            cell,
            state=ExecutionEnvelopeState.RECEIPT_OBSERVED,
            last_operational_reason_code=reason_code,
        )
        return cls._append(
            envelope,
            event=cls._event(
                envelope,
                event_id=event_id,
                occurred_at_utc=occurred_at_utc,
                kind=ExecutionEnvelopeEventKind.OPERATIONAL_FAILURE,
                cell_id=cell_id,
                attempt_id=attempt_id,
                from_state=cell.state,
                to_state=updated.state,
                reason_code=reason_code,
                valid_receipt=False,
                scientific_terminal=False,
            ),
            cell=updated,
        )

    @classmethod
    def retain_duplicate_receipt(
        cls,
        envelope: RunExecutionEnvelope,
        *,
        event_id: str,
        occurred_at_utc: str,
        cell_id: str,
        receipt: ObjectIdentity,
    ) -> RunExecutionEnvelope:
        cell = envelope.cell(cell_id)
        if (
            cell.state
            not in {
                ExecutionEnvelopeState.RECEIPT_OBSERVED,
                ExecutionEnvelopeState.PUBLISHED,
                ExecutionEnvelopeState.TERMINAL,
            }
            or cell.first_valid_receipt is None
        ):
            raise ValueError("duplicate execution receipt precedes first valid success")
        duplicates = tuple(
            sorted(
                set((*cell.duplicate_receipts, receipt)),
                key=lambda value: value.object_id,
            )
        )
        updated = replace(cell, duplicate_receipts=duplicates)
        return cls._append(
            envelope,
            event=cls._event(
                envelope,
                event_id=event_id,
                occurred_at_utc=occurred_at_utc,
                kind=ExecutionEnvelopeEventKind.DUPLICATE_RECEIPT,
                cell_id=cell_id,
                attempt_id=cell.current_attempt_id,
                from_state=cell.state,
                to_state=cell.state,
                receipt=receipt,
                valid_receipt=True,
            ),
            cell=updated,
        )

    @classmethod
    def publish(
        cls,
        envelope: RunExecutionEnvelope,
        *,
        event_id: str,
        occurred_at_utc: str,
        cell_id: str,
        artifact: ObjectIdentity,
        recovery: bool = False,
    ) -> RunExecutionEnvelope:
        cell = envelope.cell(cell_id)
        expected_state = (
            ExecutionEnvelopeState.RECOVERY_PUBLICATION
            if recovery
            else ExecutionEnvelopeState.RECEIPT_OBSERVED
        )
        if (
            cell.state is not expected_state
            or cell.first_valid_receipt is None
            or cell.published_artifact is not None
        ):
            raise ValueError("execution-envelope publication lacks first valid bytes")
        updated = replace(
            cell,
            state=ExecutionEnvelopeState.PUBLISHED,
            published_artifact=artifact,
        )
        return cls._append(
            envelope,
            event=cls._event(
                envelope,
                event_id=event_id,
                occurred_at_utc=occurred_at_utc,
                kind=(
                    ExecutionEnvelopeEventKind.RECOVERY_PUBLICATION_FINISHED
                    if recovery
                    else ExecutionEnvelopeEventKind.PUBLICATION
                ),
                cell_id=cell_id,
                attempt_id=cell.current_attempt_id,
                from_state=cell.state,
                to_state=updated.state,
                receipt=cell.first_valid_receipt,
                artifact=artifact,
                valid_receipt=True,
            ),
            cell=updated,
        )

    @classmethod
    def begin_recovery_publication(
        cls,
        envelope: RunExecutionEnvelope,
        *,
        event_id: str,
        occurred_at_utc: str,
        cell_id: str,
    ) -> RunExecutionEnvelope:
        cell = envelope.cell(cell_id)
        if (
            cell.state is not ExecutionEnvelopeState.RECEIPT_OBSERVED
            or cell.first_valid_receipt is None
            or cell.published_artifact is not None
        ):
            raise ValueError("recovery can only finish an interrupted publication")
        updated = replace(cell, state=ExecutionEnvelopeState.RECOVERY_PUBLICATION)
        return cls._append(
            envelope,
            event=cls._event(
                envelope,
                event_id=event_id,
                occurred_at_utc=occurred_at_utc,
                kind=ExecutionEnvelopeEventKind.RECOVERY_PUBLICATION_STARTED,
                cell_id=cell_id,
                attempt_id=cell.current_attempt_id,
                from_state=cell.state,
                to_state=updated.state,
                receipt=cell.first_valid_receipt,
                valid_receipt=True,
            ),
            cell=updated,
        )

    @classmethod
    def terminal(
        cls,
        envelope: RunExecutionEnvelope,
        *,
        event_id: str,
        occurred_at_utc: str,
        cell_id: str,
    ) -> RunExecutionEnvelope:
        cell = envelope.cell(cell_id)
        if cell.state is not ExecutionEnvelopeState.PUBLISHED:
            raise ValueError("execution-envelope terminal requires publication")
        updated = replace(cell, state=ExecutionEnvelopeState.TERMINAL)
        return cls._append(
            envelope,
            event=cls._event(
                envelope,
                event_id=event_id,
                occurred_at_utc=occurred_at_utc,
                kind=ExecutionEnvelopeEventKind.TERMINAL,
                cell_id=cell_id,
                attempt_id=cell.current_attempt_id,
                from_state=cell.state,
                to_state=updated.state,
                receipt=cell.first_valid_receipt,
                artifact=cell.published_artifact,
                valid_receipt=True,
            ),
            cell=updated,
        )

    @classmethod
    def stop(
        cls,
        envelope: RunExecutionEnvelope,
        *,
        event_id: str,
        occurred_at_utc: str,
        cell_id: str,
        reason_code: str,
    ) -> RunExecutionEnvelope:
        cell = envelope.cell(cell_id)
        if cell.state not in {
            ExecutionEnvelopeState.ISSUED,
            ExecutionEnvelopeState.RESERVED,
            ExecutionEnvelopeState.LAUNCHED,
            ExecutionEnvelopeState.RECEIPT_OBSERVED,
        }:
            raise ValueError("resource stop occurs outside the prelaunch boundary")
        if (
            cell.state is ExecutionEnvelopeState.RECEIPT_OBSERVED
            and cell.first_valid_receipt is not None
        ):
            raise ValueError("resource stop cannot replace valid scientific receipt truth")
        updated = replace(
            cell,
            state=ExecutionEnvelopeState.RESOURCE_STOP,
            last_operational_reason_code=reason_code,
        )
        return cls._append(
            envelope,
            event=cls._event(
                envelope,
                event_id=event_id,
                occurred_at_utc=occurred_at_utc,
                kind=ExecutionEnvelopeEventKind.RESOURCE_STOP,
                cell_id=cell_id,
                attempt_id=cell.current_attempt_id,
                from_state=cell.state,
                to_state=updated.state,
                reason_code=reason_code,
            ),
            cell=updated,
        )

    @classmethod
    def unknown_completion(
        cls,
        envelope: RunExecutionEnvelope,
        *,
        event_id: str,
        occurred_at_utc: str,
        cell_id: str,
        reason_code: str,
    ) -> RunExecutionEnvelope:
        cell = envelope.cell(cell_id)
        if cell.state is not ExecutionEnvelopeState.LAUNCHED:
            raise ValueError("unknown completion requires a launched reservation")
        updated = replace(
            cell,
            state=ExecutionEnvelopeState.UNKNOWN_COMPLETION,
            last_operational_reason_code=reason_code,
        )
        return cls._append(
            envelope,
            event=cls._event(
                envelope,
                event_id=event_id,
                occurred_at_utc=occurred_at_utc,
                kind=ExecutionEnvelopeEventKind.UNKNOWN_COMPLETION,
                cell_id=cell_id,
                attempt_id=cell.current_attempt_id,
                from_state=cell.state,
                to_state=updated.state,
                reason_code=reason_code,
            ),
            cell=updated,
        )


def reconstruct_execution_envelope(
    *,
    envelope_id: str,
    spec: ExecutionEnvelopeSpec,
    execution_plan: ObjectIdentity,
    events: tuple[ExecutionEnvelopeEvent, ...],
) -> RunExecutionEnvelope:
    """Rebuild only from authoritative external events; never infer/refund/retry."""

    machine = RunExecutionEnvelopeMachine
    envelope = machine.initial(
        envelope_id=envelope_id,
        spec=spec,
        execution_plan=execution_plan,
    )
    for event in events:
        kwargs = {
            "event_id": event.event_id,
            "occurred_at_utc": event.occurred_at_utc,
            "cell_id": event.cell_id,
        }
        if event.kind is ExecutionEnvelopeEventKind.RESERVATION:
            assert event.attempt_id is not None
            envelope = machine.reserve(
                envelope,
                attempt_id=event.attempt_id,
                retry_reason_code=event.reason_code,
                **kwargs,
            )
        elif event.kind is ExecutionEnvelopeEventKind.LAUNCH:
            assert event.attempt_id is not None
            envelope = machine.launch(
                envelope,
                attempt_id=event.attempt_id,
                **kwargs,
            )
        elif event.kind is ExecutionEnvelopeEventKind.RECEIPT:
            assert event.attempt_id is not None and event.receipt is not None
            assert event.valid_receipt is not None and event.scientific_terminal is not None
            envelope = machine.observe_receipt(
                envelope,
                attempt_id=event.attempt_id,
                receipt=event.receipt,
                valid_receipt=event.valid_receipt,
                scientific_terminal=event.scientific_terminal,
                operational_reason_code=event.reason_code,
                **kwargs,
            )
        elif event.kind is ExecutionEnvelopeEventKind.OPERATIONAL_FAILURE:
            assert event.attempt_id is not None and event.reason_code is not None
            envelope = machine.observe_operational_failure(
                envelope,
                attempt_id=event.attempt_id,
                reason_code=event.reason_code,
                **kwargs,
            )
        elif event.kind is ExecutionEnvelopeEventKind.DUPLICATE_RECEIPT:
            assert event.receipt is not None
            envelope = machine.retain_duplicate_receipt(
                envelope,
                receipt=event.receipt,
                **kwargs,
            )
        elif event.kind is ExecutionEnvelopeEventKind.PUBLICATION:
            assert event.artifact is not None
            envelope = machine.publish(
                envelope,
                event_id=event.event_id,
                occurred_at_utc=event.occurred_at_utc,
                cell_id=event.cell_id,
                artifact=event.artifact,
            )
        elif event.kind is ExecutionEnvelopeEventKind.RECOVERY_PUBLICATION_STARTED:
            envelope = machine.begin_recovery_publication(envelope, **kwargs)
        elif event.kind is ExecutionEnvelopeEventKind.RECOVERY_PUBLICATION_FINISHED:
            assert event.artifact is not None
            envelope = machine.publish(
                envelope,
                event_id=event.event_id,
                occurred_at_utc=event.occurred_at_utc,
                cell_id=event.cell_id,
                artifact=event.artifact,
                recovery=True,
            )
        elif event.kind is ExecutionEnvelopeEventKind.TERMINAL:
            envelope = machine.terminal(envelope, **kwargs)
        elif event.kind is ExecutionEnvelopeEventKind.RESOURCE_STOP:
            assert event.reason_code is not None
            envelope = machine.stop(
                envelope,
                reason_code=event.reason_code,
                **kwargs,
            )
        else:
            assert event.reason_code is not None
            envelope = machine.unknown_completion(
                envelope,
                reason_code=event.reason_code,
                **kwargs,
            )
        if envelope.events[-1] != event:
            raise ValueError("reconstructed execution-envelope event bytes differ")
    return envelope


# These deadline-free records are separate from the deadline-bearing envelope.
# Each family keeps its own validators and immutable replay provenance.
# The deadline-free envelope counts attempts on every graph task and
# physical/retry cost only on native simulator launches.


@dataclass(frozen=True, slots=True)
class JitExpressionFieldValue(CanonicalRecord):
    """One canonical cell field and whether it changes the traced expression."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/jit-expression-field-value'

    field_id: str
    canonical_value_sha256: str
    expression_changing: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.field_id, field_name="field_id")
        validate_sha256(
            self.canonical_value_sha256,
            field_name="canonical_value_sha256",
        )


def jit_graph_signature_sha256(
    fields: tuple[JitExpressionFieldValue, ...],
) -> str:
    """Hash only expression-changing coordinates; dynamic values reuse a graph."""

    return hashlib.sha256(
        canonical_json_bytes(
            tuple(
                {
                    "field_id": value.field_id,
                    "canonical_value_sha256": value.canonical_value_sha256,
                }
                for value in fields
                if value.expression_changing
            )
        )
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class JitCellSignatureProjection(CanonicalRecord):
    """Canonical projection of one simulator cell onto static and dynamic fields."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/jit-cell-signature-projection'

    cell_id: str
    fields: tuple[JitExpressionFieldValue, ...]
    signature_sha256: str

    def __post_init__(self) -> None:
        validate_stable_id(self.cell_id, field_name="cell_id")
        require_sorted_unique_ids(self.fields, attribute="field_id", field_name="fields")
        if not self.fields or not any(value.expression_changing for value in self.fields):
            raise ValueError("JIT projection requires an expression-changing field")
        validate_sha256(self.signature_sha256, field_name="signature_sha256")
        if self.signature_sha256 != jit_graph_signature_sha256(self.fields):
            raise ValueError("JIT graph signature differs from its canonical projection")


@dataclass(frozen=True, slots=True)
class PredevelopmentJitSignatureCensus(CanonicalRecord):
    """Outcome-free census proving the finite development expression graph."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/predevelopment-jit-signature-census'

    census_id: str
    expression_field_probe: ObjectIdentity
    persistent_cache: ObjectIdentity
    expression_changing_field_ids: tuple[str, ...]
    runtime_dynamic_field_ids: tuple[str, ...]
    cell_projections: tuple[JitCellSignatureProjection, ...]
    maximum_distinct_signatures: int

    def __post_init__(self) -> None:
        validate_stable_id(self.census_id, field_name="census_id")
        require_sorted_unique_strings(
            self.expression_changing_field_ids,
            field_name="expression_changing_field_ids",
        )
        require_sorted_unique_strings(
            self.runtime_dynamic_field_ids,
            field_name="runtime_dynamic_field_ids",
        )
        if set(self.expression_changing_field_ids).intersection(self.runtime_dynamic_field_ids):
            raise ValueError("JIT fields cannot be both static and runtime-dynamic")
        require_sorted_unique_ids(
            self.cell_projections,
            attribute="cell_id",
            field_name="cell_projections",
        )
        if not self.cell_projections:
            if (
                self.expression_changing_field_ids
                or self.runtime_dynamic_field_ids
                or self.maximum_distinct_signatures != 0
            ):
                raise ValueError("empty JIT census must be an explicit zero-cell contract")
            return
        if self.maximum_distinct_signatures <= 0:
            raise ValueError("nonempty JIT census requires a positive signature ceiling")
        expected_fields = set(self.expression_changing_field_ids).union(
            self.runtime_dynamic_field_ids
        )
        for projection in self.cell_projections:
            if {value.field_id for value in projection.fields} != expected_fields:
                raise ValueError("JIT census cell has an incomplete field projection")
            for field in projection.fields:
                if field.expression_changing != (
                    field.field_id in self.expression_changing_field_ids
                ):
                    raise ValueError("JIT census field classification drifted")
        signatures = {value.signature_sha256 for value in self.cell_projections}
        if len(signatures) > self.maximum_distinct_signatures:
            raise ValueError("predevelopment JIT signature ceiling is exceeded")


@dataclass(frozen=True, slots=True)
class JitGraphSignatureObservation(CanonicalRecord):
    """Cold/compile/cache/warm evidence for one manifested expression graph."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/jit-graph-signature-observation'

    signature_sha256: str
    cold_trace_receipt: ObjectIdentity
    cold_compile_receipt: ObjectIdentity
    persistent_cache_receipt: ObjectIdentity
    warm_execution_receipts: tuple[ObjectIdentity, ...]

    def __post_init__(self) -> None:
        validate_sha256(self.signature_sha256, field_name="signature_sha256")
        require_sorted_unique_ids(
            self.warm_execution_receipts,
            attribute="object_id",
            field_name="warm_execution_receipts",
        )
        if not self.warm_execution_receipts:
            raise ValueError("JIT signature observation requires warm execution evidence")


@dataclass(frozen=True, slots=True)
class JitGraphSignatureManifest(CanonicalRecord):
    """Complete maximum-graph cell-to-signature manifest used at launch."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/jit-graph-signature-manifest'

    manifest_id: str
    predevelopment_census: ObjectIdentity
    persistent_cache: ObjectIdentity
    cell_projections: tuple[JitCellSignatureProjection, ...]
    observations: tuple[JitGraphSignatureObservation, ...]
    maximum_distinct_signatures: int

    def __post_init__(self) -> None:
        validate_stable_id(self.manifest_id, field_name="manifest_id")
        if self.predevelopment_census.object_schema != (PredevelopmentJitSignatureCensus.SCHEMA):
            raise ValueError("JIT manifest binds another census schema")
        require_sorted_unique_ids(
            self.cell_projections,
            attribute="cell_id",
            field_name="cell_projections",
        )
        require_sorted_unique_ids(
            self.observations,
            attribute="signature_sha256",
            field_name="observations",
        )
        if not self.cell_projections:
            if self.observations or self.maximum_distinct_signatures != 0:
                raise ValueError("empty JIT manifest must be an explicit zero-cell contract")
            return
        if self.maximum_distinct_signatures <= 0:
            raise ValueError("nonempty JIT manifest requires a positive ceiling")
        signatures = {value.signature_sha256 for value in self.cell_projections}
        if len(signatures) > self.maximum_distinct_signatures:
            raise ValueError("issued JIT signature ceiling is exceeded")
        if signatures != {value.signature_sha256 for value in self.observations}:
            raise ValueError("JIT observations do not cover the exact manifested signatures")

    def require_projection(
        self,
        projection: JitCellSignatureProjection,
    ) -> JitCellSignatureProjection:
        """Refuse an unmanifested cell or any static/dynamic projection drift."""

        matches = tuple(
            value for value in self.cell_projections if value.cell_id == projection.cell_id
        )
        if len(matches) != 1 or matches[0] != projection:
            raise ValueError("simulator cell has an unmanifested JIT projection")
        return matches[0]


@dataclass(frozen=True, slots=True)
class ProgressHeartbeat(CanonicalRecord):
    """One native-work heartbeat; elapsed duration is deliberately absent."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/progress-heartbeat'

    heartbeat_id: str
    task_id: str
    attempt_id: str
    work_unit_id: str
    work_counter: Decimal

    def __post_init__(self) -> None:
        for name, value in (
            ("heartbeat_id", self.heartbeat_id),
            ("task_id", self.task_id),
            ("attempt_id", self.attempt_id),
            ("work_unit_id", self.work_unit_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_decimal(
            self.work_counter,
            field_name="work_counter",
            minimum=Decimal(0),
        )


@dataclass(frozen=True, slots=True)
class ProgressLivenessContract(CanonicalRecord):
    """Source-qualified stalled-progress rule, never a total task deadline."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/progress-liveness-contract'

    contract_id: str
    heartbeat_schema: str
    work_unit_id: str
    maximum_no_progress_gap_seconds: Decimal
    monotone_counter_required: bool
    progressing_worker_has_no_elapsed_limit: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.contract_id, field_name="contract_id")
        validate_schema(self.heartbeat_schema)
        validate_stable_id(self.work_unit_id, field_name="work_unit_id")
        validate_decimal(
            self.maximum_no_progress_gap_seconds,
            field_name="maximum_no_progress_gap_seconds",
            minimum=Decimal("0.000001"),
        )
        if self.heartbeat_schema != ProgressHeartbeat.SCHEMA:
            raise ValueError("progress liveness requires the canonical heartbeat schema")
        if not (self.monotone_counter_required and self.progressing_worker_has_no_elapsed_limit):
            raise ValueError("progress liveness weakens a deadline-free invariant")


@dataclass(frozen=True, slots=True)
class NonTimeResourceBudget(CanonicalRecord):
    """Per-task computability bounds with no wall-time coordinate."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/non-time-resource-budget'

    budget_id: str
    cpu_cores: int
    memory_bytes: int
    output_bytes: int
    source_request_limit: int
    source_byte_limit: int

    def __post_init__(self) -> None:
        validate_stable_id(self.budget_id, field_name="budget_id")
        if self.cpu_cores <= 0 or self.memory_bytes <= 0 or self.output_bytes < 0:
            raise ValueError("non-time compute budget is invalid")
        if self.source_request_limit < 0 or self.source_byte_limit < 0:
            raise ValueError("non-time source budget is invalid")


@dataclass(frozen=True, slots=True)
class ExecutionResourceTaskCellSpec(CanonicalRecord):
    """Task costs and progress, with JIT evidence only for compiled runners."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/execution-resource-task-cell-spec'

    cell_id: str
    task_id: str
    child_id: str
    physical_independent_unit_id: str | None
    preparation_unit_id: str | None
    maximum_attempts: int
    physical_execution_cost: int
    retry_token_cost: int
    native_simulator_launch: bool
    resource_budget: NonTimeResourceBudget
    progress_liveness: ProgressLivenessContract | None
    jit_cell_id: str | None
    jit_signature_sha256: str | None

    def __post_init__(self) -> None:
        for name, value in (
            ("cell_id", self.cell_id),
            ("task_id", self.task_id),
            ("child_id", self.child_id),
        ):
            validate_stable_id(value, field_name=name)
        units = (self.physical_independent_unit_id, self.preparation_unit_id)
        if (units[0] is None) != (units[1] is None) or (
            self.native_simulator_launch and units[0] is None
        ):
            raise ValueError("native tasks require both physical-unit and preparation identities")
        for unit_name, unit_value in zip(
            ("physical_independent_unit_id", "preparation_unit_id"), units
        ):
            if unit_value is not None:
                validate_stable_id(unit_value, field_name=unit_name)
        if type(self.native_simulator_launch) is not bool or any(
            type(value) is not int
            for value in (
                self.maximum_attempts,
                self.physical_execution_cost,
                self.retry_token_cost,
            )
        ):
            raise ValueError(
                "resource task costs/attempts and native applicability require exact types"
            )
        if self.maximum_attempts not in {1, 2}:
            raise ValueError("resource-envelope task cells require one or two attempts")
        if self.physical_execution_cost not in {0, 1} or self.retry_token_cost not in {0, 1}:
            raise ValueError("resource-envelope costs must be zero or one")
        if self.native_simulator_launch != (self.physical_execution_cost == 1):
            raise ValueError("physical execution cost must identify native simulator launch")
        if self.retry_token_cost != self.physical_execution_cost:
            raise ValueError("only a native simulator relaunch may consume a retry token")
        if self.native_simulator_launch and self.progress_liveness is None:
            raise ValueError("native simulator cell lacks progress binding")
        if (self.jit_cell_id is None) != (self.jit_signature_sha256 is None):
            raise ValueError("JIT cell identity and signature must be supplied together")
        if self.jit_cell_id is not None:
            assert self.jit_signature_sha256 is not None
            validate_stable_id(self.jit_cell_id, field_name="jit_cell_id")
            validate_sha256(self.jit_signature_sha256, field_name="jit_signature_sha256")


def validate_optional_jit_identities(
    census: ObjectIdentity | None, manifest: ObjectIdentity | None
) -> None:
    """An absent compilation contract has neither census nor manifest."""
    if census is None and manifest is None:
        return
    if (
        census is None
        or manifest is None
        or census.object_schema != PredevelopmentJitSignatureCensus.SCHEMA
        or manifest.object_schema != JitGraphSignatureManifest.SCHEMA
    ):
        raise ValueError("JIT census/manifest identities are partial or incompatible")


@dataclass(frozen=True, slots=True)
class ChildResourceTokenLimit(CanonicalRecord):
    """Nontransferable execution and retry ceilings for one child."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/child-resource-token-limit'

    child_id: str
    physical_execution_token_limit: int
    retry_token_limit: int

    def __post_init__(self) -> None:
        validate_stable_id(self.child_id, field_name="child_id")
        if self.physical_execution_token_limit < 0 or self.retry_token_limit < 0:
            raise ValueError("child resource token limits cannot be negative")


@dataclass(frozen=True, slots=True)
class RosterCapacityBranchSpec(CanonicalRecord):
    """One preissued branch; groups are selected exactly once and outcome-blind."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/roster-capacity-branch-spec'

    branch_id: str
    group_id: str
    child_id: str
    cell_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("branch_id", self.branch_id),
            ("group_id", self.group_id),
            ("child_id", self.child_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(
            self.cell_ids,
            field_name="cell_ids",
            allow_empty=False,
        )


@dataclass(frozen=True, slots=True)
class RosterCapacityDecision(CanonicalRecord):
    """Outcome-blind selection of one already issued capacity branch."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/roster-capacity-decision'

    decision_id: str
    group_id: str
    selected_branch_id: str
    decision_receipt: ObjectIdentity
    outcome_blind: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("decision_id", self.decision_id),
            ("group_id", self.group_id),
            ("selected_branch_id", self.selected_branch_id),
        ):
            validate_stable_id(value, field_name=name)
        if not self.outcome_blind:
            raise ValueError("roster capacity decision must be outcome-blind")


@dataclass(frozen=True, slots=True)
class ExecutionResourceEnvelopeSpec(CanonicalRecord):
    """Deadline-free finite resource contract for one issued execution graph."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/execution-resource-envelope-spec'

    envelope_spec_id: str
    issued_study_extensions: ObjectIdentity
    task_cells: tuple[ExecutionResourceTaskCellSpec, ...]
    child_token_limits: tuple[ChildResourceTokenLimit, ...]
    roster_capacity_branches: tuple[RosterCapacityBranchSpec, ...]
    allowlisted_retry_reason_codes: tuple[str, ...]
    allowlisted_resource_stop_reason_codes: tuple[str, ...]
    predevelopment_jit_signature_census: ObjectIdentity | None
    jit_graph_signature_manifest: ObjectIdentity | None
    maximum_parallel_tasks: int
    aggregate_memory_ceiling_bytes: int
    resource_lock_ids: tuple[str, ...]
    non_gating_forecast_records: tuple[ObjectIdentity, ...]
    resource_contract_sha256: str
    nonrefundable_reservations: bool
    first_valid_success_wins: bool
    unknown_completion_requires_new_disposition: bool
    elapsed_time_has_no_control_effect: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.envelope_spec_id, field_name="envelope_spec_id")
        if self.issued_study_extensions.object_schema != (
            'empirical-lawhood/runtime/issued-extension-set'
        ):
            raise ValueError("resource envelope binds another extension-set schema")
        require_sorted_unique_ids(
            self.task_cells,
            attribute="cell_id",
            field_name="task_cells",
        )
        if not self.task_cells:
            raise ValueError("resource envelope requires one cell for every graph task")
        if len({value.task_id for value in self.task_cells}) != len(self.task_cells):
            raise ValueError("resource-envelope cells must map one-to-one to tasks")
        require_sorted_unique_ids(
            self.child_token_limits,
            attribute="child_id",
            field_name="child_token_limits",
        )
        require_sorted_unique_ids(
            self.roster_capacity_branches,
            attribute="branch_id",
            field_name="roster_capacity_branches",
        )
        require_sorted_unique_strings(
            self.allowlisted_retry_reason_codes,
            field_name="allowlisted_retry_reason_codes",
        )
        require_sorted_unique_strings(
            self.allowlisted_resource_stop_reason_codes,
            field_name="allowlisted_resource_stop_reason_codes",
        )
        require_sorted_unique_strings(self.resource_lock_ids, field_name="resource_lock_ids")
        require_sorted_unique_ids(
            self.non_gating_forecast_records,
            attribute="object_id",
            field_name="non_gating_forecast_records",
        )
        validate_sha256(self.resource_contract_sha256, field_name="resource_contract_sha256")
        validate_optional_jit_identities(
            self.predevelopment_jit_signature_census, self.jit_graph_signature_manifest
        )
        if any(cell.jit_cell_id is not None for cell in self.task_cells) != (
            self.jit_graph_signature_manifest is not None
        ):
            raise ValueError(
                "resource envelope JIT evidence must match compiled task applicability"
            )
        if self.maximum_parallel_tasks <= 0 or self.aggregate_memory_ceiling_bytes <= 0:
            raise ValueError("resource envelope parallelism/memory ceiling is invalid")
        if any(
            value.resource_budget.memory_bytes > self.aggregate_memory_ceiling_bytes
            for value in self.task_cells
        ):
            raise ValueError("task memory exceeds the aggregate envelope ceiling")
        if not (
            self.nonrefundable_reservations
            and self.first_valid_success_wins
            and self.unknown_completion_requires_new_disposition
            and self.elapsed_time_has_no_control_effect
        ):
            raise ValueError("resource envelope weakens a frozen durability rule")
        self._validate_branches_and_tokens()

    def _validate_branches_and_tokens(self) -> None:
        cells = {value.cell_id: value for value in self.task_cells}
        limits = {value.child_id: value for value in self.child_token_limits}
        if set(limits) != {value.child_id for value in self.task_cells}:
            raise ValueError("child token limits do not cover the exact child roster")
        branches_by_group: dict[str, list[RosterCapacityBranchSpec]] = {}
        groups_by_cell: dict[str, set[str]] = {}
        for branch in self.roster_capacity_branches:
            branches_by_group.setdefault(branch.group_id, []).append(branch)
            for cell_id in branch.cell_ids:
                cell = cells.get(cell_id)
                if cell is None or cell.child_id != branch.child_id:
                    raise ValueError("roster branch contains an unknown/cross-child cell")
                groups_by_cell.setdefault(cell_id, set()).add(branch.group_id)
        if any(len(values) < 2 for values in branches_by_group.values()):
            raise ValueError("capacity branch groups require at least two choices")
        if any(len(values) != 1 for values in groups_by_cell.values()):
            raise ValueError("an optional cell cannot belong to multiple branch groups")
        for group in branches_by_group.values():
            if len({value.child_id for value in group}) != 1:
                raise ValueError("a capacity branch group cannot transfer child resources")

        optional_cells = set(groups_by_cell)
        for child_id, limit in limits.items():
            mandatory = tuple(
                value
                for value in self.task_cells
                if value.child_id == child_id and value.cell_id not in optional_cells
            )
            physical = sum(
                value.physical_execution_cost * value.maximum_attempts for value in mandatory
            )
            retries = sum(
                value.retry_token_cost * (value.maximum_attempts - 1) for value in mandatory
            )
            for group in branches_by_group.values():
                if group[0].child_id != child_id:
                    continue
                physical += max(
                    sum(
                        cells[cell_id].physical_execution_cost * cells[cell_id].maximum_attempts
                        for cell_id in branch.cell_ids
                    )
                    for branch in group
                )
                retries += max(
                    sum(
                        cells[cell_id].retry_token_cost * (cells[cell_id].maximum_attempts - 1)
                        for cell_id in branch.cell_ids
                    )
                    for branch in group
                )
            if (
                limit.physical_execution_token_limit != physical
                or limit.retry_token_limit != retries
            ):
                raise ValueError("child token ceiling differs from exact maximum demand")

    def cell(self, cell_id: str) -> ExecutionResourceTaskCellSpec:
        for value in self.task_cells:
            if value.cell_id == cell_id:
                return value
        raise KeyError(cell_id)

    def cell_for_task(self, task_id: str) -> ExecutionResourceTaskCellSpec:
        for value in self.task_cells:
            if value.task_id == task_id:
                return value
        raise KeyError(task_id)


def validate_execution_jit_evidence(
    spec: ExecutionResourceEnvelopeSpec,
    census: PredevelopmentJitSignatureCensus | None,
    manifest: JitGraphSignatureManifest | None,
) -> None:
    """One shared join for declared cells, actual evidence and its identities."""
    if census is None and manifest is None:
        if spec.jit_graph_signature_manifest is not None:
            raise ValueError("compiled tasks lack JIT evidence")
        return
    if census is None or manifest is None:
        raise ValueError("JIT census and manifest must be supplied together")
    census_identity = ObjectIdentity.from_record(census.census_id, census)
    if (
        spec.predevelopment_jit_signature_census != census_identity
        or spec.jit_graph_signature_manifest
        != ObjectIdentity.from_record(manifest.manifest_id, manifest)
        or manifest.predevelopment_census != census_identity
        or manifest.persistent_cache != census.persistent_cache
    ):
        raise ValueError("resource/JIT evidence identities are discontinuous")
    declared = tuple(
        sorted(
            (cell.jit_cell_id, cell.jit_signature_sha256)
            for cell in spec.task_cells
            if cell.jit_cell_id is not None
        )
    )
    if declared != tuple(
        (cell.cell_id, cell.signature_sha256) for cell in manifest.cell_projections
    ):
        raise ValueError("compiled task/JIT manifest cell roster is not exact")


class ExecutionResourceEnvelopeState(StrEnum):
    PENDING_BRANCH = "PENDING_BRANCH"
    ISSUED = "ISSUED"
    RESERVED = "RESERVED"
    LAUNCHED = "LAUNCHED"
    RECEIPT_OBSERVED = "RECEIPT_OBSERVED"
    RECOVERY_PUBLICATION = "RECOVERY_PUBLICATION"
    PUBLISHED = "PUBLISHED"
    TERMINAL = "TERMINAL"
    ZERO_CALL_SKIPPED = "ZERO_CALL_SKIPPED"
    RESOURCE_STOP = "RESOURCE_STOP"
    UNKNOWN_COMPLETION = "UNKNOWN_COMPLETION"


class ExecutionResourceEnvelopeEventKind(StrEnum):
    ROSTER_BRANCH_SELECTED = "ROSTER_BRANCH_SELECTED"
    RESERVATION = "RESERVATION"
    LAUNCH = "LAUNCH"
    PROGRESS_HEARTBEAT = "PROGRESS_HEARTBEAT"
    PROGRESS_STALLED = "PROGRESS_STALLED"
    RECEIPT = "RECEIPT"
    OPERATIONAL_FAILURE = "OPERATIONAL_FAILURE"
    DUPLICATE_RECEIPT = "DUPLICATE_RECEIPT"
    PUBLICATION = "PUBLICATION"
    RECOVERY_PUBLICATION_STARTED = "RECOVERY_PUBLICATION_STARTED"
    ARTIFACT_BEFORE_RECEIPT_RECOVERY = "ARTIFACT_BEFORE_RECEIPT_RECOVERY"
    RECOVERY_PUBLICATION_FINISHED = "RECOVERY_PUBLICATION_FINISHED"
    TERMINAL = "TERMINAL"
    RESOURCE_STOP = "RESOURCE_STOP"
    UNKNOWN_COMPLETION = "UNKNOWN_COMPLETION"


@dataclass(frozen=True, slots=True)
class ExecutionResourceEnvelopeEvent(CanonicalRecord):
    "One canonical transition; branch selection may update several cells."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/execution-resource-envelope-event'

    event_id: str
    sequence_number: int
    occurred_at_utc: str
    kind: ExecutionResourceEnvelopeEventKind
    cell_id: str | None
    affected_cell_ids: tuple[str, ...]
    attempt_id: str | None
    from_state: ExecutionResourceEnvelopeState | None
    to_state: ExecutionResourceEnvelopeState | None
    reason_code: str | None
    receipt: ObjectIdentity | None
    artifact: ObjectIdentity | None
    recovery_proof: ObjectIdentity | None
    heartbeat: ProgressHeartbeat | None
    jit_projection: JitCellSignatureProjection | None
    roster_decision: RosterCapacityDecision | None
    valid_receipt: bool | None
    scientific_terminal: bool | None

    def __post_init__(self) -> None:
        validate_stable_id(self.event_id, field_name="event_id")
        if self.sequence_number < 1:
            raise ValueError("resource-envelope event sequence must be positive")
        parse_utc_timestamp(self.occurred_at_utc, field_name="occurred_at_utc")
        require_sorted_unique_strings(
            self.affected_cell_ids,
            field_name="affected_cell_ids",
        )
        if self.reason_code is not None:
            validate_nonempty(self.reason_code, field_name="reason_code")
        if self.kind is ExecutionResourceEnvelopeEventKind.ROSTER_BRANCH_SELECTED:
            if (
                self.roster_decision is None
                or not self.affected_cell_ids
                or self.cell_id is not None
                or self.attempt_id is not None
                or self.from_state is not None
                or self.to_state is not None
            ):
                raise ValueError("resource-envelope branch event is malformed")
        else:
            if (
                self.cell_id is None
                or self.from_state is None
                or self.to_state is None
                or self.affected_cell_ids
                or self.roster_decision is not None
            ):
                raise ValueError("resource-envelope cell event is malformed")
            validate_stable_id(self.cell_id, field_name="cell_id")
        if self.attempt_id is not None:
            validate_stable_id(self.attempt_id, field_name="attempt_id")
        if self.kind is ExecutionResourceEnvelopeEventKind.PROGRESS_HEARTBEAT:
            if self.heartbeat is None:
                raise ValueError("progress event lacks its heartbeat")
        elif self.heartbeat is not None:
            raise ValueError("non-progress event contains a heartbeat")
        if self.kind is not ExecutionResourceEnvelopeEventKind.LAUNCH and (
            self.jit_projection is not None
        ):
            raise ValueError("only a launch event may bind a JIT projection")
        if self.kind in {
            ExecutionResourceEnvelopeEventKind.RECOVERY_PUBLICATION_STARTED,
            ExecutionResourceEnvelopeEventKind.ARTIFACT_BEFORE_RECEIPT_RECOVERY,
        }:
            if self.receipt is None or self.artifact is None or self.recovery_proof is None:
                raise ValueError("recovery publication event lacks proven receipt/artifact")
        elif self.recovery_proof is not None:
            raise ValueError("non-recovery event contains a recovery proof")


@dataclass(frozen=True, slots=True)
class ExecutionResourceCellState(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/execution-resource-cell-state'

    cell_id: str
    state: ExecutionResourceEnvelopeState
    attempt_ids: tuple[str, ...]
    physical_tokens_reserved: int
    retry_tokens_reserved: int
    current_attempt_id: str | None
    first_valid_receipt: ObjectIdentity | None
    published_artifact: ObjectIdentity | None
    recovery_candidate_artifact: ObjectIdentity | None
    recovery_proof: ObjectIdentity | None
    duplicate_receipts: tuple[ObjectIdentity, ...]
    last_operational_reason_code: str | None
    last_progress_counter: Decimal | None
    last_progress_at_utc: str | None

    def __post_init__(self) -> None:
        validate_stable_id(self.cell_id, field_name="cell_id")
        require_sorted_unique_strings(
            tuple(sorted(self.attempt_ids)),
            field_name="attempt_ids",
        )
        for value in self.attempt_ids:
            validate_stable_id(value, field_name="attempt_ids")
        if len(set(self.attempt_ids)) != len(self.attempt_ids):
            raise ValueError("resource-envelope attempt identity is duplicated")
        require_sorted_unique_ids(
            self.duplicate_receipts,
            attribute="object_id",
            field_name="duplicate_receipts",
        )
        if self.physical_tokens_reserved < 0 or self.retry_tokens_reserved < 0:
            raise ValueError("resource-envelope token counts cannot be negative")
        if self.current_attempt_id is not None:
            validate_stable_id(self.current_attempt_id, field_name="current_attempt_id")
            if self.current_attempt_id not in self.attempt_ids:
                raise ValueError("current attempt is absent from reserved attempt history")
        if (self.recovery_candidate_artifact is None) != (self.recovery_proof is None):
            raise ValueError("recovery artifact/proof must be present together")
        if self.state is ExecutionResourceEnvelopeState.RECOVERY_PUBLICATION and (
            self.first_valid_receipt is None or self.recovery_candidate_artifact is None
        ):
            raise ValueError("recovery publication lacks proven first-valid work")
        if self.state is not ExecutionResourceEnvelopeState.RECOVERY_PUBLICATION and (
            self.recovery_candidate_artifact is not None
        ):
            raise ValueError("recovery candidate escaped its publication state")
        if self.published_artifact is not None and self.first_valid_receipt is None:
            raise ValueError("published resource-envelope artifact lacks valid receipt")
        if (self.last_progress_counter is None) != (self.last_progress_at_utc is None):
            raise ValueError("progress counter/time must be present together")
        if self.last_progress_counter is not None:
            validate_decimal(
                self.last_progress_counter,
                field_name="last_progress_counter",
                minimum=Decimal(0),
            )
            assert self.last_progress_at_utc is not None
            parse_utc_timestamp(self.last_progress_at_utc, field_name="last_progress_at_utc")


@dataclass(frozen=True, slots=True)
class ChildResourceTokenState(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/child-resource-token-state'

    child_id: str
    physical_tokens_reserved: int
    retry_tokens_reserved: int

    def __post_init__(self) -> None:
        validate_stable_id(self.child_id, field_name="child_id")
        if self.physical_tokens_reserved < 0 or self.retry_tokens_reserved < 0:
            raise ValueError("child resource token state cannot be negative")


class _ResourceEnvelopeEventIndex(CanonicalRecord):
    # Derived immutable lookup; not a serialized record field or authority.
    __slots__ = ("_event_ids",)
    _event_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class RunExecutionResourceEnvelope(_ResourceEnvelopeEventIndex):
    """Durable deadline-free execution prefix with child-local reservations."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/run-execution-resource-envelope'

    envelope_id: str
    spec: ExecutionResourceEnvelopeSpec
    execution_plan: ObjectIdentity
    cells: tuple[ExecutionResourceCellState, ...]
    roster_decisions: tuple[RosterCapacityDecision, ...]
    child_token_states: tuple[ChildResourceTokenState, ...]
    events: tuple[ExecutionResourceEnvelopeEvent, ...]

    _previous: InitVar[RunExecutionResourceEnvelope | None] = None

    def __post_init__(self, _previous: RunExecutionResourceEnvelope | None) -> None:
        validate_stable_id(self.envelope_id, field_name="envelope_id")
        require_sorted_unique_ids(self.cells, attribute="cell_id", field_name="cells")
        if tuple(value.cell_id for value in self.cells) != tuple(
            value.cell_id for value in self.spec.task_cells
        ):
            raise ValueError("resource-envelope state roster differs from its spec")
        require_sorted_unique_ids(
            self.roster_decisions,
            attribute="group_id",
            field_name="roster_decisions",
        )
        require_sorted_unique_ids(
            self.child_token_states,
            attribute="child_id",
            field_name="child_token_states",
        )
        if tuple(value.child_id for value in self.child_token_states) != tuple(
            value.child_id for value in self.spec.child_token_limits
        ):
            raise ValueError("resource-envelope child token roster differs")
        if _previous is None:
            event_ids = frozenset(value.event_id for value in self.events)
            object.__setattr__(self, "_event_ids", event_ids)
            if len(event_ids) != len(self.events):
                raise ValueError("resource-envelope event identity is duplicated")
            if tuple(value.sequence_number for value in self.events) != tuple(
                range(1, len(self.events) + 1)
            ):
                raise ValueError("resource-envelope event sequence is not contiguous")
            if any(
                parse_utc_timestamp(later.occurred_at_utc, field_name="occurred_at_utc")
                < parse_utc_timestamp(earlier.occurred_at_utc, field_name="occurred_at_utc")
                for earlier, later in zip(self.events, self.events[1:], strict=False)
            ):
                raise ValueError("resource-envelope events are not time ordered")
        else:
            if (
                len(self.events) != len(_previous.events) + 1
                or self.events[:-1] != _previous.events
            ):
                raise ValueError("resource-envelope update is not one event append")
            event = self.events[-1]
            if event.sequence_number != len(self.events):
                raise ValueError("resource-envelope append sequence differs")
            if event.event_id in _previous._event_ids:
                raise ValueError("resource-envelope event identity is duplicated")
            object.__setattr__(self, "_event_ids", _previous._event_ids | {event.event_id})
            if _previous.events and parse_utc_timestamp(
                event.occurred_at_utc, field_name="occurred_at_utc"
            ) < parse_utc_timestamp(
                _previous.events[-1].occurred_at_utc, field_name="occurred_at_utc"
            ):
                raise ValueError("resource-envelope events are not time ordered")
        specs = {value.cell_id: value for value in self.spec.task_cells}
        limits = {value.child_id: value for value in self.spec.child_token_limits}
        expected_by_child: dict[str, tuple[int, int]] = {}
        for child_id in limits:
            child_cells = tuple(
                value for value in self.cells if specs[value.cell_id].child_id == child_id
            )
            expected_by_child[child_id] = (
                sum(value.physical_tokens_reserved for value in child_cells),
                sum(value.retry_tokens_reserved for value in child_cells),
            )
        for state in self.child_token_states:
            expected = expected_by_child[state.child_id]
            limit = limits[state.child_id]
            if (state.physical_tokens_reserved, state.retry_tokens_reserved) != expected:
                raise ValueError("child token state differs from cell reservations")
            if (
                state.physical_tokens_reserved > limit.physical_execution_token_limit
                or state.retry_tokens_reserved > limit.retry_token_limit
            ):
                raise ValueError("child-local resource token ceiling is exceeded")

    def cell(self, cell_id: str) -> ExecutionResourceCellState:
        for value in self.cells:
            if value.cell_id == cell_id:
                return value
        raise KeyError(cell_id)


class RunExecutionResourceEnvelopeMachine:
    "Pure resource-envelope transitions; no transition reads a total elapsed-time target."

    @staticmethod
    def initial(
        *,
        envelope_id: str,
        spec: ExecutionResourceEnvelopeSpec,
        execution_plan: ObjectIdentity,
    ) -> RunExecutionResourceEnvelope:
        optional = {
            cell_id for branch in spec.roster_capacity_branches for cell_id in branch.cell_ids
        }
        cells = tuple(
            ExecutionResourceCellState(
                cell_id=value.cell_id,
                state=(
                    ExecutionResourceEnvelopeState.PENDING_BRANCH
                    if value.cell_id in optional
                    else ExecutionResourceEnvelopeState.ISSUED
                ),
                attempt_ids=(),
                physical_tokens_reserved=0,
                retry_tokens_reserved=0,
                current_attempt_id=None,
                first_valid_receipt=None,
                published_artifact=None,
                recovery_candidate_artifact=None,
                recovery_proof=None,
                duplicate_receipts=(),
                last_operational_reason_code=None,
                last_progress_counter=None,
                last_progress_at_utc=None,
            )
            for value in spec.task_cells
        )
        return RunExecutionResourceEnvelope(
            envelope_id=envelope_id,
            spec=spec,
            execution_plan=execution_plan,
            cells=cells,
            roster_decisions=(),
            child_token_states=tuple(
                ChildResourceTokenState(
                    child_id=value.child_id,
                    physical_tokens_reserved=0,
                    retry_tokens_reserved=0,
                )
                for value in spec.child_token_limits
            ),
            events=(),
        )

    @staticmethod
    def _token_states(
        envelope: RunExecutionResourceEnvelope,
        cells: tuple[ExecutionResourceCellState, ...],
    ) -> tuple[ChildResourceTokenState, ...]:
        specs = {value.cell_id: value for value in envelope.spec.task_cells}
        return tuple(
            ChildResourceTokenState(
                child_id=limit.child_id,
                physical_tokens_reserved=sum(
                    value.physical_tokens_reserved
                    for value in cells
                    if specs[value.cell_id].child_id == limit.child_id
                ),
                retry_tokens_reserved=sum(
                    value.retry_tokens_reserved
                    for value in cells
                    if specs[value.cell_id].child_id == limit.child_id
                ),
            )
            for limit in envelope.spec.child_token_limits
        )

    @classmethod
    def _append(
        cls,
        envelope: RunExecutionResourceEnvelope,
        *,
        event: ExecutionResourceEnvelopeEvent,
        cells: tuple[ExecutionResourceCellState, ...],
        roster_decisions: tuple[RosterCapacityDecision, ...] | None = None,
    ) -> RunExecutionResourceEnvelope:
        if event.sequence_number != len(envelope.events) + 1:
            raise ValueError("resource-envelope append sequence differs")
        return replace(
            envelope,
            cells=cells,
            roster_decisions=(
                envelope.roster_decisions if roster_decisions is None else roster_decisions
            ),
            child_token_states=cls._token_states(envelope, cells),
            events=(*envelope.events, event),
            _previous=envelope,
        )

    @staticmethod
    def _replace_cell(
        envelope: RunExecutionResourceEnvelope,
        updated: ExecutionResourceCellState,
    ) -> tuple[ExecutionResourceCellState, ...]:
        return tuple(
            updated if value.cell_id == updated.cell_id else value for value in envelope.cells
        )

    @staticmethod
    def _event(
        envelope: RunExecutionResourceEnvelope,
        *,
        event_id: str,
        occurred_at_utc: str,
        kind: ExecutionResourceEnvelopeEventKind,
        cell_id: str | None = None,
        affected_cell_ids: tuple[str, ...] = (),
        attempt_id: str | None = None,
        from_state: ExecutionResourceEnvelopeState | None = None,
        to_state: ExecutionResourceEnvelopeState | None = None,
        reason_code: str | None = None,
        receipt: ObjectIdentity | None = None,
        artifact: ObjectIdentity | None = None,
        recovery_proof: ObjectIdentity | None = None,
        heartbeat: ProgressHeartbeat | None = None,
        jit_projection: JitCellSignatureProjection | None = None,
        roster_decision: RosterCapacityDecision | None = None,
        valid_receipt: bool | None = None,
        scientific_terminal: bool | None = None,
    ) -> ExecutionResourceEnvelopeEvent:
        return ExecutionResourceEnvelopeEvent(
            event_id=event_id,
            sequence_number=len(envelope.events) + 1,
            occurred_at_utc=occurred_at_utc,
            kind=kind,
            cell_id=cell_id,
            affected_cell_ids=affected_cell_ids,
            attempt_id=attempt_id,
            from_state=from_state,
            to_state=to_state,
            reason_code=reason_code,
            receipt=receipt,
            artifact=artifact,
            recovery_proof=recovery_proof,
            heartbeat=heartbeat,
            jit_projection=jit_projection,
            roster_decision=roster_decision,
            valid_receipt=valid_receipt,
            scientific_terminal=scientific_terminal,
        )

    @classmethod
    def select_roster_branch(
        cls,
        envelope: RunExecutionResourceEnvelope,
        *,
        event_id: str,
        occurred_at_utc: str,
        decision: RosterCapacityDecision,
    ) -> RunExecutionResourceEnvelope:
        if any(value.group_id == decision.group_id for value in envelope.roster_decisions):
            raise ValueError("roster capacity group was already selected")
        branches = tuple(
            value
            for value in envelope.spec.roster_capacity_branches
            if value.group_id == decision.group_id
        )
        selected = tuple(
            value for value in branches if value.branch_id == decision.selected_branch_id
        )
        if len(selected) != 1:
            raise ValueError("roster capacity decision selects an unknown branch")
        affected = tuple(sorted({cell for branch in branches for cell in branch.cell_ids}))
        selected_cells = set(selected[0].cell_ids)
        updated_cells = []
        for cell in envelope.cells:
            if cell.cell_id not in affected:
                updated_cells.append(cell)
                continue
            if cell.state is not ExecutionResourceEnvelopeState.PENDING_BRANCH:
                raise ValueError("roster branch cell left its pending boundary")
            updated_cells.append(
                replace(
                    cell,
                    state=(
                        ExecutionResourceEnvelopeState.ISSUED
                        if cell.cell_id in selected_cells
                        else ExecutionResourceEnvelopeState.ZERO_CALL_SKIPPED
                    ),
                    last_operational_reason_code=(
                        None
                        if cell.cell_id in selected_cells
                        else "UNSELECTED_ROSTER_CAPACITY_BRANCH"
                    ),
                )
            )
        decisions = tuple(
            sorted((*envelope.roster_decisions, decision), key=lambda value: value.group_id)
        )
        return cls._append(
            envelope,
            event=cls._event(
                envelope,
                event_id=event_id,
                occurred_at_utc=occurred_at_utc,
                kind=ExecutionResourceEnvelopeEventKind.ROSTER_BRANCH_SELECTED,
                affected_cell_ids=affected,
                roster_decision=decision,
            ),
            cells=tuple(updated_cells),
            roster_decisions=decisions,
        )

    @classmethod
    def reserve(
        cls,
        envelope: RunExecutionResourceEnvelope,
        *,
        event_id: str,
        occurred_at_utc: str,
        cell_id: str,
        attempt_id: str,
        retry_reason_code: str | None = None,
    ) -> RunExecutionResourceEnvelope:
        # occurred_at_utc is validated and ordered, but never compared with a
        # deadline because this schema deliberately has none.
        cell = envelope.cell(cell_id)
        spec = envelope.spec.cell(cell_id)
        retry = bool(cell.attempt_ids)
        if any(attempt_id in value.attempt_ids for value in envelope.cells):
            raise ValueError("resource-envelope attempt identity is already reserved")
        if retry:
            if (
                cell.state is not ExecutionResourceEnvelopeState.RECEIPT_OBSERVED
                or cell.first_valid_receipt is not None
                or retry_reason_code is None
                or retry_reason_code != cell.last_operational_reason_code
                or retry_reason_code not in envelope.spec.allowlisted_retry_reason_codes
            ):
                raise ValueError("resource-envelope retry is not allowlisted")
        elif (
            cell.state is not ExecutionResourceEnvelopeState.ISSUED
            or retry_reason_code is not None
        ):
            raise ValueError("first resource-envelope reservation is malformed")
        if len(cell.attempt_ids) >= spec.maximum_attempts:
            raise ValueError("resource-envelope maximum attempts are exhausted")
        updated = replace(
            cell,
            state=ExecutionResourceEnvelopeState.RESERVED,
            attempt_ids=(*cell.attempt_ids, attempt_id),
            physical_tokens_reserved=(cell.physical_tokens_reserved + spec.physical_execution_cost),
            retry_tokens_reserved=(
                cell.retry_tokens_reserved + (spec.retry_token_cost if retry else 0)
            ),
            current_attempt_id=attempt_id,
            last_operational_reason_code=None,
            last_progress_counter=None,
            last_progress_at_utc=None,
        )
        return cls._append(
            envelope,
            event=cls._event(
                envelope,
                event_id=event_id,
                occurred_at_utc=occurred_at_utc,
                kind=ExecutionResourceEnvelopeEventKind.RESERVATION,
                cell_id=cell_id,
                attempt_id=attempt_id,
                from_state=cell.state,
                to_state=updated.state,
                reason_code=retry_reason_code,
            ),
            cells=cls._replace_cell(envelope, updated),
        )

    @classmethod
    def launch(
        cls,
        envelope: RunExecutionResourceEnvelope,
        *,
        event_id: str,
        occurred_at_utc: str,
        cell_id: str,
        attempt_id: str,
        jit_projection: JitCellSignatureProjection | None = None,
        jit_manifest: JitGraphSignatureManifest | None = None,
    ) -> RunExecutionResourceEnvelope:
        cell = envelope.cell(cell_id)
        spec = envelope.spec.cell(cell_id)
        if (
            cell.state is not ExecutionResourceEnvelopeState.RESERVED
            or cell.current_attempt_id != attempt_id
        ):
            raise ValueError("resource-envelope launch lacks durable reservation")
        if spec.jit_cell_id is not None:
            if jit_projection is None or jit_manifest is None:
                raise ValueError("compiled task launch lacks manifested JIT projection")
            jit_manifest.require_projection(jit_projection)
            if (
                jit_projection.cell_id != spec.jit_cell_id
                or jit_projection.signature_sha256 != spec.jit_signature_sha256
                or ObjectIdentity.from_record(jit_manifest.manifest_id, jit_manifest)
                != envelope.spec.jit_graph_signature_manifest
            ):
                raise ValueError("compiled task launch JIT identity drifted")
        elif jit_projection is not None or jit_manifest is not None:
            raise ValueError("uncompiled task launch cannot bind JIT evidence")
        updated = replace(
            cell,
            state=ExecutionResourceEnvelopeState.LAUNCHED,
            last_progress_counter=(Decimal(0) if spec.progress_liveness is not None else None),
            last_progress_at_utc=(occurred_at_utc if spec.progress_liveness is not None else None),
        )
        return cls._append(
            envelope,
            event=cls._event(
                envelope,
                event_id=event_id,
                occurred_at_utc=occurred_at_utc,
                kind=ExecutionResourceEnvelopeEventKind.LAUNCH,
                cell_id=cell_id,
                attempt_id=attempt_id,
                from_state=cell.state,
                to_state=updated.state,
                jit_projection=jit_projection,
            ),
            cells=cls._replace_cell(envelope, updated),
        )

    @classmethod
    def heartbeat(
        cls,
        envelope: RunExecutionResourceEnvelope,
        *,
        event_id: str,
        occurred_at_utc: str,
        cell_id: str,
        heartbeat: ProgressHeartbeat,
    ) -> RunExecutionResourceEnvelope:
        cell = envelope.cell(cell_id)
        spec = envelope.spec.cell(cell_id)
        contract = spec.progress_liveness
        if (
            cell.state is not ExecutionResourceEnvelopeState.LAUNCHED
            or contract is None
            or cell.current_attempt_id != heartbeat.attempt_id
            or spec.task_id != heartbeat.task_id
            or contract.work_unit_id != heartbeat.work_unit_id
            or cell.last_progress_counter is None
            or heartbeat.work_counter <= cell.last_progress_counter
        ):
            raise ValueError("progress heartbeat is stale, substituted or non-monotone")
        updated = replace(
            cell,
            last_progress_counter=heartbeat.work_counter,
            last_progress_at_utc=occurred_at_utc,
        )
        return cls._append(
            envelope,
            event=cls._event(
                envelope,
                event_id=event_id,
                occurred_at_utc=occurred_at_utc,
                kind=ExecutionResourceEnvelopeEventKind.PROGRESS_HEARTBEAT,
                cell_id=cell_id,
                attempt_id=heartbeat.attempt_id,
                from_state=cell.state,
                to_state=updated.state,
                heartbeat=heartbeat,
            ),
            cells=cls._replace_cell(envelope, updated),
        )

    @classmethod
    def progress_stalled(
        cls,
        envelope: RunExecutionResourceEnvelope,
        *,
        event_id: str,
        occurred_at_utc: str,
        cell_id: str,
        attempt_id: str,
    ) -> RunExecutionResourceEnvelope:
        cell = envelope.cell(cell_id)
        contract = envelope.spec.cell(cell_id).progress_liveness
        if (
            cell.state is not ExecutionResourceEnvelopeState.LAUNCHED
            or cell.current_attempt_id != attempt_id
            or contract is None
            or cell.last_progress_at_utc is None
        ):
            raise ValueError("stalled-progress transition lacks a live progress contract")
        elapsed = parse_utc_timestamp(
            occurred_at_utc,
            field_name="occurred_at_utc",
        ) - parse_utc_timestamp(cell.last_progress_at_utc, field_name="last_progress_at_utc")
        if Decimal(str(elapsed.total_seconds())) < contract.maximum_no_progress_gap_seconds:
            raise ValueError("worker is still inside its no-progress qualification gap")
        reason = "WORKER_PROGRESS_STALLED"
        if reason not in envelope.spec.allowlisted_retry_reason_codes:
            raise ValueError("stalled-progress retry reason is not allowlisted")
        updated = replace(
            cell,
            state=ExecutionResourceEnvelopeState.RECEIPT_OBSERVED,
            last_operational_reason_code=reason,
        )
        return cls._append(
            envelope,
            event=cls._event(
                envelope,
                event_id=event_id,
                occurred_at_utc=occurred_at_utc,
                kind=ExecutionResourceEnvelopeEventKind.PROGRESS_STALLED,
                cell_id=cell_id,
                attempt_id=attempt_id,
                from_state=cell.state,
                to_state=updated.state,
                reason_code=reason,
                valid_receipt=False,
                scientific_terminal=False,
            ),
            cells=cls._replace_cell(envelope, updated),
        )

    @classmethod
    def observe_receipt(
        cls,
        envelope: RunExecutionResourceEnvelope,
        *,
        event_id: str,
        occurred_at_utc: str,
        cell_id: str,
        attempt_id: str,
        receipt: ObjectIdentity,
        valid_receipt: bool,
        scientific_terminal: bool,
        operational_reason_code: str | None = None,
    ) -> RunExecutionResourceEnvelope:
        cell = envelope.cell(cell_id)
        if (
            cell.state is not ExecutionResourceEnvelopeState.LAUNCHED
            or cell.current_attempt_id != attempt_id
        ):
            raise ValueError("resource-envelope receipt lacks exact launched attempt")
        if valid_receipt:
            if operational_reason_code is not None or cell.first_valid_receipt is not None:
                raise ValueError("first-valid resource-envelope receipt is malformed")
            first = receipt
        else:
            if scientific_terminal or operational_reason_code is None:
                raise ValueError("operational failure receipt is malformed")
            first = None
        updated = replace(
            cell,
            state=ExecutionResourceEnvelopeState.RECEIPT_OBSERVED,
            first_valid_receipt=first,
            last_operational_reason_code=operational_reason_code,
        )
        return cls._append(
            envelope,
            event=cls._event(
                envelope,
                event_id=event_id,
                occurred_at_utc=occurred_at_utc,
                kind=ExecutionResourceEnvelopeEventKind.RECEIPT,
                cell_id=cell_id,
                attempt_id=attempt_id,
                from_state=cell.state,
                to_state=updated.state,
                reason_code=operational_reason_code,
                receipt=receipt,
                valid_receipt=valid_receipt,
                scientific_terminal=scientific_terminal,
            ),
            cells=cls._replace_cell(envelope, updated),
        )

    @classmethod
    def observe_operational_failure(
        cls,
        envelope: RunExecutionResourceEnvelope,
        *,
        event_id: str,
        occurred_at_utc: str,
        cell_id: str,
        attempt_id: str,
        reason_code: str,
    ) -> RunExecutionResourceEnvelope:
        cell = envelope.cell(cell_id)
        if (
            cell.state is not ExecutionResourceEnvelopeState.LAUNCHED
            or cell.current_attempt_id != attempt_id
        ):
            raise ValueError("resource-envelope failure lacks exact launched attempt")
        updated = replace(
            cell,
            state=ExecutionResourceEnvelopeState.RECEIPT_OBSERVED,
            last_operational_reason_code=reason_code,
        )
        return cls._append(
            envelope,
            event=cls._event(
                envelope,
                event_id=event_id,
                occurred_at_utc=occurred_at_utc,
                kind=ExecutionResourceEnvelopeEventKind.OPERATIONAL_FAILURE,
                cell_id=cell_id,
                attempt_id=attempt_id,
                from_state=cell.state,
                to_state=updated.state,
                reason_code=reason_code,
                valid_receipt=False,
                scientific_terminal=False,
            ),
            cells=cls._replace_cell(envelope, updated),
        )

    @classmethod
    def retain_duplicate_receipt(
        cls,
        envelope: RunExecutionResourceEnvelope,
        *,
        event_id: str,
        occurred_at_utc: str,
        cell_id: str,
        receipt: ObjectIdentity,
    ) -> RunExecutionResourceEnvelope:
        cell = envelope.cell(cell_id)
        if (
            cell.state
            not in {
                ExecutionResourceEnvelopeState.RECEIPT_OBSERVED,
                ExecutionResourceEnvelopeState.RECOVERY_PUBLICATION,
                ExecutionResourceEnvelopeState.PUBLISHED,
                ExecutionResourceEnvelopeState.TERMINAL,
            }
            or cell.first_valid_receipt is None
        ):
            raise ValueError("duplicate receipt precedes first valid success")
        updated = replace(
            cell,
            duplicate_receipts=tuple(
                sorted(
                    set((*cell.duplicate_receipts, receipt)),
                    key=lambda value: value.object_id,
                )
            ),
        )
        return cls._append(
            envelope,
            event=cls._event(
                envelope,
                event_id=event_id,
                occurred_at_utc=occurred_at_utc,
                kind=ExecutionResourceEnvelopeEventKind.DUPLICATE_RECEIPT,
                cell_id=cell_id,
                attempt_id=cell.current_attempt_id,
                from_state=cell.state,
                to_state=updated.state,
                receipt=receipt,
                valid_receipt=True,
            ),
            cells=cls._replace_cell(envelope, updated),
        )

    @classmethod
    def publish(
        cls,
        envelope: RunExecutionResourceEnvelope,
        *,
        event_id: str,
        occurred_at_utc: str,
        cell_id: str,
        artifact: ObjectIdentity,
    ) -> RunExecutionResourceEnvelope:
        cell = envelope.cell(cell_id)
        if (
            cell.state is not ExecutionResourceEnvelopeState.RECEIPT_OBSERVED
            or cell.first_valid_receipt is None
            or cell.published_artifact is not None
        ):
            raise ValueError("resource-envelope publication lacks first valid bytes")
        updated = replace(
            cell,
            state=ExecutionResourceEnvelopeState.PUBLISHED,
            published_artifact=artifact,
        )
        return cls._append(
            envelope,
            event=cls._event(
                envelope,
                event_id=event_id,
                occurred_at_utc=occurred_at_utc,
                kind=ExecutionResourceEnvelopeEventKind.PUBLICATION,
                cell_id=cell_id,
                attempt_id=cell.current_attempt_id,
                from_state=cell.state,
                to_state=updated.state,
                receipt=cell.first_valid_receipt,
                artifact=artifact,
                valid_receipt=True,
            ),
            cells=cls._replace_cell(envelope, updated),
        )

    @classmethod
    def begin_recovery_publication(
        cls,
        envelope: RunExecutionResourceEnvelope,
        *,
        event_id: str,
        occurred_at_utc: str,
        cell_id: str,
        artifact: ObjectIdentity,
        recovery_proof: ObjectIdentity,
    ) -> RunExecutionResourceEnvelope:
        cell = envelope.cell(cell_id)
        if (
            cell.state is not ExecutionResourceEnvelopeState.RECEIPT_OBSERVED
            or cell.first_valid_receipt is None
            or cell.published_artifact is not None
        ):
            raise ValueError("recovery publication lacks a durable valid receipt")
        updated = replace(
            cell,
            state=ExecutionResourceEnvelopeState.RECOVERY_PUBLICATION,
            recovery_candidate_artifact=artifact,
            recovery_proof=recovery_proof,
        )
        return cls._append(
            envelope,
            event=cls._event(
                envelope,
                event_id=event_id,
                occurred_at_utc=occurred_at_utc,
                kind=ExecutionResourceEnvelopeEventKind.RECOVERY_PUBLICATION_STARTED,
                cell_id=cell_id,
                attempt_id=cell.current_attempt_id,
                from_state=cell.state,
                to_state=updated.state,
                receipt=cell.first_valid_receipt,
                artifact=artifact,
                recovery_proof=recovery_proof,
                valid_receipt=True,
            ),
            cells=cls._replace_cell(envelope, updated),
        )

    @classmethod
    def recover_artifact_before_receipt(
        cls,
        envelope: RunExecutionResourceEnvelope,
        *,
        event_id: str,
        occurred_at_utc: str,
        cell_id: str,
        attempt_id: str,
        receipt: ObjectIdentity,
        artifact: ObjectIdentity,
        recovery_proof: ObjectIdentity,
    ) -> RunExecutionResourceEnvelope:
        """Bind proven already-written work without reserving or launching again."""

        cell = envelope.cell(cell_id)
        if (
            cell.state is not ExecutionResourceEnvelopeState.LAUNCHED
            or cell.current_attempt_id != attempt_id
            or cell.first_valid_receipt is not None
        ):
            raise ValueError("artifact-before-receipt recovery lacks exact launched work")
        updated = replace(
            cell,
            state=ExecutionResourceEnvelopeState.RECOVERY_PUBLICATION,
            first_valid_receipt=receipt,
            recovery_candidate_artifact=artifact,
            recovery_proof=recovery_proof,
        )
        return cls._append(
            envelope,
            event=cls._event(
                envelope,
                event_id=event_id,
                occurred_at_utc=occurred_at_utc,
                kind=(ExecutionResourceEnvelopeEventKind.ARTIFACT_BEFORE_RECEIPT_RECOVERY),
                cell_id=cell_id,
                attempt_id=attempt_id,
                from_state=cell.state,
                to_state=updated.state,
                receipt=receipt,
                artifact=artifact,
                recovery_proof=recovery_proof,
                valid_receipt=True,
                scientific_terminal=True,
            ),
            cells=cls._replace_cell(envelope, updated),
        )

    @classmethod
    def finish_recovery_publication(
        cls,
        envelope: RunExecutionResourceEnvelope,
        *,
        event_id: str,
        occurred_at_utc: str,
        cell_id: str,
    ) -> RunExecutionResourceEnvelope:
        cell = envelope.cell(cell_id)
        if (
            cell.state is not ExecutionResourceEnvelopeState.RECOVERY_PUBLICATION
            or cell.first_valid_receipt is None
            or cell.recovery_candidate_artifact is None
            or cell.recovery_proof is None
        ):
            raise ValueError("recovery publication is incomplete")
        artifact = cell.recovery_candidate_artifact
        updated = replace(
            cell,
            state=ExecutionResourceEnvelopeState.PUBLISHED,
            published_artifact=artifact,
            recovery_candidate_artifact=None,
            recovery_proof=None,
        )
        return cls._append(
            envelope,
            event=cls._event(
                envelope,
                event_id=event_id,
                occurred_at_utc=occurred_at_utc,
                kind=ExecutionResourceEnvelopeEventKind.RECOVERY_PUBLICATION_FINISHED,
                cell_id=cell_id,
                attempt_id=cell.current_attempt_id,
                from_state=cell.state,
                to_state=updated.state,
                receipt=cell.first_valid_receipt,
                artifact=artifact,
                valid_receipt=True,
            ),
            cells=cls._replace_cell(envelope, updated),
        )

    @classmethod
    def terminal(
        cls,
        envelope: RunExecutionResourceEnvelope,
        *,
        event_id: str,
        occurred_at_utc: str,
        cell_id: str,
    ) -> RunExecutionResourceEnvelope:
        cell = envelope.cell(cell_id)
        if cell.state is not ExecutionResourceEnvelopeState.PUBLISHED:
            raise ValueError("resource-envelope terminal requires publication")
        updated = replace(cell, state=ExecutionResourceEnvelopeState.TERMINAL)
        return cls._append(
            envelope,
            event=cls._event(
                envelope,
                event_id=event_id,
                occurred_at_utc=occurred_at_utc,
                kind=ExecutionResourceEnvelopeEventKind.TERMINAL,
                cell_id=cell_id,
                attempt_id=cell.current_attempt_id,
                from_state=cell.state,
                to_state=updated.state,
                receipt=cell.first_valid_receipt,
                artifact=cell.published_artifact,
                valid_receipt=True,
            ),
            cells=cls._replace_cell(envelope, updated),
        )

    @classmethod
    def stop(
        cls,
        envelope: RunExecutionResourceEnvelope,
        *,
        event_id: str,
        occurred_at_utc: str,
        cell_id: str,
        reason_code: str,
    ) -> RunExecutionResourceEnvelope:
        cell = envelope.cell(cell_id)
        if reason_code not in envelope.spec.allowlisted_resource_stop_reason_codes:
            raise ValueError("resource stop reason is not predeclared")
        if cell.state not in {
            ExecutionResourceEnvelopeState.ISSUED,
            ExecutionResourceEnvelopeState.RESERVED,
            ExecutionResourceEnvelopeState.LAUNCHED,
            ExecutionResourceEnvelopeState.RECEIPT_OBSERVED,
        }:
            raise ValueError("resource stop occurs outside a stoppable boundary")
        if (
            cell.state is ExecutionResourceEnvelopeState.RECEIPT_OBSERVED
            and cell.first_valid_receipt is not None
        ):
            raise ValueError("resource stop cannot replace valid scientific receipt truth")
        updated = replace(
            cell,
            state=ExecutionResourceEnvelopeState.RESOURCE_STOP,
            last_operational_reason_code=reason_code,
        )
        return cls._append(
            envelope,
            event=cls._event(
                envelope,
                event_id=event_id,
                occurred_at_utc=occurred_at_utc,
                kind=ExecutionResourceEnvelopeEventKind.RESOURCE_STOP,
                cell_id=cell_id,
                attempt_id=cell.current_attempt_id,
                from_state=cell.state,
                to_state=updated.state,
                reason_code=reason_code,
            ),
            cells=cls._replace_cell(envelope, updated),
        )

    @classmethod
    def unknown_completion(
        cls,
        envelope: RunExecutionResourceEnvelope,
        *,
        event_id: str,
        occurred_at_utc: str,
        cell_id: str,
        reason_code: str,
    ) -> RunExecutionResourceEnvelope:
        cell = envelope.cell(cell_id)
        if cell.state is not ExecutionResourceEnvelopeState.LAUNCHED:
            raise ValueError("unknown completion requires launched reservation")
        updated = replace(
            cell,
            state=ExecutionResourceEnvelopeState.UNKNOWN_COMPLETION,
            last_operational_reason_code=reason_code,
        )
        return cls._append(
            envelope,
            event=cls._event(
                envelope,
                event_id=event_id,
                occurred_at_utc=occurred_at_utc,
                kind=ExecutionResourceEnvelopeEventKind.UNKNOWN_COMPLETION,
                cell_id=cell_id,
                attempt_id=cell.current_attempt_id,
                from_state=cell.state,
                to_state=updated.state,
                reason_code=reason_code,
            ),
            cells=cls._replace_cell(envelope, updated),
        )


def reconstruct_execution_resource_envelope(
    *,
    envelope_id: str,
    spec: ExecutionResourceEnvelopeSpec,
    execution_plan: ObjectIdentity,
    events: tuple[ExecutionResourceEnvelopeEvent, ...],
    jit_graph_signature_manifest: JitGraphSignatureManifest | None,
) -> RunExecutionResourceEnvelope:
    """Rebuild the current envelope from its authoritative event prefix."""

    if (
        None
        if jit_graph_signature_manifest is None
        else ObjectIdentity.from_record(
            jit_graph_signature_manifest.manifest_id,
            jit_graph_signature_manifest,
        )
    ) != spec.jit_graph_signature_manifest:
        raise ValueError("resource-envelope reconstruction JIT manifest differs")
    # Without conditional rosters or JIT, transitions are cell-local and the
    # child ceilings are exactly the sum of cell maxima. Authenticate each
    # cell with the same transition owner, then validate the global sequence
    # and summed charges once instead of scanning every cell at every event.
    if len(spec.task_cells) > 1 and not spec.roster_capacity_branches and jit_graph_signature_manifest is None:
        by_cell: dict[str, list[ExecutionResourceEnvelopeEvent]] = {
            cell.cell_id: [] for cell in spec.task_cells
        }
        for event in events:
            if event.cell_id not in by_cell or event.kind is ExecutionResourceEnvelopeEventKind.ROSTER_BRANCH_SELECTED:
                raise ValueError("resource event does not belong to an independent cell")
            assert event.cell_id is not None
            by_cell[event.cell_id].append(event)
        cells = []
        totals = {limit.child_id: [0, 0] for limit in spec.child_token_limits}
        limits = {limit.child_id: limit for limit in spec.child_token_limits}
        for cell in spec.task_cells:
            local_spec = replace(
                spec, task_cells=(cell,), child_token_limits=(replace(
                    limits[cell.child_id],
                    physical_execution_token_limit=cell.physical_execution_cost * cell.maximum_attempts,
                    retry_token_limit=cell.retry_token_cost * (cell.maximum_attempts - 1),
                ),),
            )
            local = reconstruct_execution_resource_envelope(
                envelope_id=envelope_id, spec=local_spec, execution_plan=execution_plan,
                events=tuple(replace(event, sequence_number=i) for i, event in enumerate(by_cell[cell.cell_id], 1)),
                jit_graph_signature_manifest=None,
            )
            state, = local.cells
            cells.append(state)
            totals[cell.child_id][0] += state.physical_tokens_reserved
            totals[cell.child_id][1] += state.retry_tokens_reserved
        return RunExecutionResourceEnvelope(
            envelope_id=envelope_id, spec=spec, execution_plan=execution_plan,
            cells=tuple(cells), roster_decisions=(),
            child_token_states=tuple(ChildResourceTokenState(limit.child_id, *totals[limit.child_id]) for limit in spec.child_token_limits),
            events=events,
        )
    machine = RunExecutionResourceEnvelopeMachine
    envelope = machine.initial(
        envelope_id=envelope_id,
        spec=spec,
        execution_plan=execution_plan,
    )
    for event in events:
        envelope = replay_execution_resource_event(
            envelope, event, jit_graph_signature_manifest=jit_graph_signature_manifest
        )
    return envelope


def replay_execution_resource_event(
    envelope: RunExecutionResourceEnvelope,
    event: ExecutionResourceEnvelopeEvent,
    *,
    jit_graph_signature_manifest: JitGraphSignatureManifest | None,
) -> RunExecutionResourceEnvelope:
    """Apply and authenticate one event using the sole transition owner."""

    machine = RunExecutionResourceEnvelopeMachine
    base = {
        "event_id": event.event_id,
        "occurred_at_utc": event.occurred_at_utc,
    }
    if event.kind is ExecutionResourceEnvelopeEventKind.ROSTER_BRANCH_SELECTED:
        assert event.roster_decision is not None
        envelope = machine.select_roster_branch(
            envelope,
            decision=event.roster_decision,
            **base,
        )
    else:
        assert event.cell_id is not None
        cell = {"cell_id": event.cell_id}
        if event.kind is ExecutionResourceEnvelopeEventKind.RESERVATION:
            assert event.attempt_id is not None
            envelope = machine.reserve(
                envelope,
                attempt_id=event.attempt_id,
                retry_reason_code=event.reason_code,
                **base,
                **cell,
            )
        elif event.kind is ExecutionResourceEnvelopeEventKind.LAUNCH:
            assert event.attempt_id is not None
            envelope = machine.launch(
                envelope,
                attempt_id=event.attempt_id,
                jit_projection=event.jit_projection,
                jit_manifest=(
                    jit_graph_signature_manifest if event.jit_projection is not None else None
                ),
                **base,
                **cell,
            )
        elif event.kind is ExecutionResourceEnvelopeEventKind.PROGRESS_HEARTBEAT:
            assert event.heartbeat is not None
            envelope = machine.heartbeat(
                envelope,
                heartbeat=event.heartbeat,
                **base,
                **cell,
            )
        elif event.kind is ExecutionResourceEnvelopeEventKind.PROGRESS_STALLED:
            assert event.attempt_id is not None
            envelope = machine.progress_stalled(
                envelope,
                attempt_id=event.attempt_id,
                **base,
                **cell,
            )
        elif event.kind is ExecutionResourceEnvelopeEventKind.RECEIPT:
            assert event.attempt_id is not None and event.receipt is not None
            assert event.valid_receipt is not None and event.scientific_terminal is not None
            envelope = machine.observe_receipt(
                envelope,
                attempt_id=event.attempt_id,
                receipt=event.receipt,
                valid_receipt=event.valid_receipt,
                scientific_terminal=event.scientific_terminal,
                operational_reason_code=event.reason_code,
                **base,
                **cell,
            )
        elif event.kind is ExecutionResourceEnvelopeEventKind.OPERATIONAL_FAILURE:
            assert event.attempt_id is not None and event.reason_code is not None
            envelope = machine.observe_operational_failure(
                envelope,
                attempt_id=event.attempt_id,
                reason_code=event.reason_code,
                **base,
                **cell,
            )
        elif event.kind is ExecutionResourceEnvelopeEventKind.DUPLICATE_RECEIPT:
            assert event.receipt is not None
            envelope = machine.retain_duplicate_receipt(
                envelope,
                receipt=event.receipt,
                **base,
                **cell,
            )
        elif event.kind is ExecutionResourceEnvelopeEventKind.PUBLICATION:
            assert event.artifact is not None
            envelope = machine.publish(
                envelope,
                artifact=event.artifact,
                **base,
                **cell,
            )
        elif event.kind is (ExecutionResourceEnvelopeEventKind.RECOVERY_PUBLICATION_STARTED):
            assert event.artifact is not None and event.recovery_proof is not None
            envelope = machine.begin_recovery_publication(
                envelope,
                artifact=event.artifact,
                recovery_proof=event.recovery_proof,
                **base,
                **cell,
            )
        elif event.kind is (ExecutionResourceEnvelopeEventKind.ARTIFACT_BEFORE_RECEIPT_RECOVERY):
            assert event.attempt_id is not None and event.receipt is not None
            assert event.artifact is not None and event.recovery_proof is not None
            envelope = machine.recover_artifact_before_receipt(
                envelope,
                attempt_id=event.attempt_id,
                receipt=event.receipt,
                artifact=event.artifact,
                recovery_proof=event.recovery_proof,
                **base,
                **cell,
            )
        elif event.kind is (ExecutionResourceEnvelopeEventKind.RECOVERY_PUBLICATION_FINISHED):
            envelope = machine.finish_recovery_publication(
                envelope,
                **base,
                **cell,
            )
        elif event.kind is ExecutionResourceEnvelopeEventKind.TERMINAL:
            envelope = machine.terminal(envelope, **base, **cell)
        elif event.kind is ExecutionResourceEnvelopeEventKind.RESOURCE_STOP:
            assert event.reason_code is not None
            envelope = machine.stop(
                envelope,
                reason_code=event.reason_code,
                **base,
                **cell,
            )
        else:
            assert event.reason_code is not None
            envelope = machine.unknown_completion(
                envelope,
                reason_code=event.reason_code,
                **base,
                **cell,
            )
    if envelope.events[-1] != event:
        raise ValueError("reconstructed resource-envelope event bytes differ")
    return envelope


__all__ = [
    'ChildResourceTokenLimit',
    'ChildResourceTokenState',
    "ExecutionEnvelopeCellSpec",
    "ExecutionEnvelopeCellState",
    "ExecutionEnvelopeEvent",
    "ExecutionEnvelopeEventKind",
    "ExecutionEnvelopeSpec",
    "ExecutionEnvelopeState",
    'ExecutionResourceCellState',
    'ExecutionResourceEnvelopeEventKind',
    'ExecutionResourceEnvelopeEvent',
    'ExecutionResourceEnvelopeSpec',
    'ExecutionResourceEnvelopeState',
    'ExecutionResourceTaskCellSpec',
    'JitCellSignatureProjection',
    'JitExpressionFieldValue',
    'JitGraphSignatureManifest',
    'JitGraphSignatureObservation',
    'NonTimeResourceBudget',
    'PredevelopmentJitSignatureCensus',
    'ProgressHeartbeat',
    'ProgressLivenessContract',
    'RosterCapacityBranchSpec',
    'RosterCapacityDecision',
    "RunExecutionEnvelope",
    "RunExecutionEnvelopeMachine",
    'RunExecutionResourceEnvelopeMachine',
    'RunExecutionResourceEnvelope',
    "jit_graph_signature_sha256",
    "reconstruct_execution_envelope",
    'reconstruct_execution_resource_envelope',
    'replay_execution_resource_event',
]
