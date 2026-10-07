"""Outcome-blind maintenance stops at a fully receipted execution boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256, validate_stable_id
from empirical_lawhood.runtime.execution_envelope import ExecutionResourceEnvelopeState, RunExecutionResourceEnvelope


@dataclass(frozen=True, slots=True)
class MaintenanceStopAuthority(CanonicalRecord):
    """Exact owner-requested local service stop; never a retry authorization."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/maintenance-stop-authority'

    stop_id: str
    run_id: str
    owner_id: str
    owner_instruction_sha256: str
    service_name: str
    main_pid: int
    main_start_ticks: int
    service_exec_start_sha256: str
    execution_plan: ObjectIdentity

    def __post_init__(self) -> None:
        for name in ("stop_id", "run_id", "owner_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in ("owner_instruction_sha256", "service_exec_start_sha256"):
            validate_sha256(getattr(self, name), field_name=name)
        if (
            not self.service_name.endswith(".service")
            or not self.service_name.startswith("empirical-lawhood-")
            or any(
                c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.@"
                for c in self.service_name
            )
        ):
            raise ValueError("maintenance service name is outside the local campaign namespace")
        if self.main_pid <= 1 or self.main_start_ticks <= 0:
            raise ValueError("maintenance service process identity is invalid")


@dataclass(frozen=True, slots=True)
class MaintenanceReceiptBoundary(CanonicalRecord):
    """An operational checkpoint; it grants neither retries nor scientific support."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/maintenance-receipt-boundary'

    run_id: str
    envelope: ObjectIdentity
    execution_plan: ObjectIdentity
    completed_receipts: tuple[ObjectIdentity, ...]
    unstarted_task_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.run_id, field_name="run_id")
        receipt_ids = tuple(value.object_id for value in self.completed_receipts)
        if receipt_ids != tuple(sorted(set(receipt_ids))):
            raise ValueError("maintenance receipts must be sorted and unique")
        if self.unstarted_task_ids != tuple(sorted(set(self.unstarted_task_ids))):
            raise ValueError("maintenance unstarted tasks must be sorted and unique")
        for value in self.unstarted_task_ids:
            validate_stable_id(value, field_name="unstarted_task_ids")


@dataclass(frozen=True, slots=True)
class ContinuationBoundInterruptionAuthority(CanonicalRecord):
    """Owner-authorized interruption, preserving the old issue without retries.

    A separately issued continuation consumes the retained interruption record.
    This authority grants no native execution and cannot amend the old plan.
    """

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/continuation-bound-interruption-authority'

    service_stop: MaintenanceStopAuthority
    amendment_document_sha256: str
    continuation_run_id: str
    maximum_open_attempts: int

    def __post_init__(self) -> None:
        validate_sha256(self.amendment_document_sha256, field_name="amendment_document_sha256")
        validate_stable_id(self.continuation_run_id, field_name="continuation_run_id")
        if self.continuation_run_id == self.service_stop.run_id:
            raise ValueError("an interrupted zero-retry issue requires a distinct continuation")
        if type(self.maximum_open_attempts) is not int or not 1 <= self.maximum_open_attempts <= 64:
            raise ValueError("maintenance interruption needs an explicit bounded attempt census")


@dataclass(frozen=True, slots=True)
class MaintenanceProcessStopAuthority(CanonicalRecord):
    """Owner-bound stop of one direct process and its bounded live descendants."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/maintenance-process-stop-authority'

    stop_id: str
    run_id: str
    owner_id: str
    owner_instruction_sha256: str
    main_pid: int
    main_start_ticks: int
    command_line_sha256: str
    maximum_processes: int
    execution_plan: ObjectIdentity

    def __post_init__(self) -> None:
        for name in ("stop_id", "run_id", "owner_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in ("owner_instruction_sha256", "command_line_sha256"):
            validate_sha256(getattr(self, name), field_name=name)
        if self.main_pid <= 1 or self.main_start_ticks <= 0:
            raise ValueError("maintenance process identity is invalid")
        if type(self.maximum_processes) is not int or not 1 <= self.maximum_processes <= 128:
            raise ValueError("maintenance process stop needs a bounded descendant census")


@dataclass(frozen=True, slots=True)
class InterruptionAuthority(CanonicalRecord):
    """Stop authority independent of any future continuation or amendment."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/interruption-authority'

    stop: MaintenanceStopAuthority | MaintenanceProcessStopAuthority
    maximum_open_attempts: int

    def __post_init__(self) -> None:
        if type(self.maximum_open_attempts) is not int or not 0 <= self.maximum_open_attempts <= 64:
            raise ValueError("maintenance interruption needs an explicit bounded attempt census")


def maintenance_receipt_boundary(
    run_id: str, envelope: RunExecutionResourceEnvelope
) -> MaintenanceReceiptBoundary | None:
    """Refuse a pause while any reservation, source work or publication is open.

    This check must be followed by receipt authentication and process stop while
    retaining the same exclusive reservation-store lock. A prior observation
    outside that lock cannot authorize a stop.
    """

    validate_stable_id(run_id, field_name="run_id")
    if envelope.envelope_id != f"run-resource-envelope.{run_id}":
        raise ValueError("maintenance envelope belongs to another run")
    completed: list[ObjectIdentity] = []
    unstarted: list[str] = []
    for spec, cell in zip(envelope.spec.task_cells, envelope.cells, strict=True):
        if cell.state is ExecutionResourceEnvelopeState.ISSUED:
            if (
                cell.attempt_ids
                or cell.current_attempt_id is not None
                or cell.first_valid_receipt is not None
                or cell.published_artifact is not None
                or cell.physical_tokens_reserved
                or cell.retry_tokens_reserved
            ):
                raise ValueError("unstarted maintenance cell contains spent work")
            unstarted.append(spec.task_id)
        elif cell.state is ExecutionResourceEnvelopeState.TERMINAL:
            receipt = cell.first_valid_receipt
            if (
                receipt is None
                or cell.published_artifact != receipt
                or cell.current_attempt_id is None
                or not cell.attempt_ids
                or receipt.object_schema != 'empirical-lawhood/runtime/canonical-task-receipt'
            ):
                raise ValueError("terminal maintenance cell lacks a published receipt")
            completed.append(receipt)
        else:
            return None
    return MaintenanceReceiptBoundary(
        run_id=run_id,
        envelope=ObjectIdentity.from_record(envelope.envelope_id, envelope),
        execution_plan=envelope.execution_plan,
        completed_receipts=tuple(sorted(completed, key=lambda value: value.object_id)),
        unstarted_task_ids=tuple(sorted(unstarted)),
    )
