"""Immutable per-wave recovery authority and append-only task event contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
from typing import TYPE_CHECKING, ClassVar, Final, Self

if TYPE_CHECKING:
    from empirical_lawhood.runtime.retry_amendment import LeaseExpiryRetryAmendment

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    canonical_json_bytes,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_relative_locator,
    validate_schema,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.runtime.artifacts import CanonicalTaskReceipt
from empirical_lawhood.runtime.execution import OperationalFailureClass, TaskBlockReason
from empirical_lawhood.runtime.execution_envelope import validate_optional_jit_identities
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan, CandidateExecutionPlan, EnvelopeExecutionPlan, ExecutionPlan, ProtocolExecutionTask


RESOURCE_ENVELOPE_JOURNAL_ROOT: Final[str] = 'operational/execution-resource-envelopes'

MAX_RECOVERY_TASKS: Final[int] = 10_000
MAX_RECOVERY_ATTEMPTS_PER_TASK: Final[int] = 1_000
MAX_RECOVERY_CANDIDATES: Final[int] = 100_000
MAX_SANITIZED_EXCEPTION_TYPE_BYTES: Final[int] = 128
MAX_SANITIZED_MESSAGE_BYTES: Final[int] = 256


class TaskRecoveryDisposition(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    NOT_ATTEMPTED = "NOT_ATTEMPTED"


class RecoveryTerminalDisposition(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True, slots=True)
class OperationalFailureDiagnostic(CanonicalRecord):
    """Bounded causal diagnostic that can never serve as scientific evidence."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/operational-failure-diagnostic'

    failure_class: OperationalFailureClass
    retryable: bool
    sanitized_exception_type: str | None
    sanitized_message: str | None
    diagnostic_sha256: str
    scientific_evidence: bool = False

    def __post_init__(self) -> None:
        if self.retryable is not self.failure_class.retryable:
            raise ValueError("operational failure retryability differs from its class")
        for name, value, maximum_bytes in (
            (
                "sanitized_exception_type",
                self.sanitized_exception_type,
                MAX_SANITIZED_EXCEPTION_TYPE_BYTES,
            ),
            ("sanitized_message", self.sanitized_message, MAX_SANITIZED_MESSAGE_BYTES),
        ):
            if value is None:
                continue
            if not value or len(value.encode("utf-8")) > maximum_bytes:
                raise ValueError(f"{name} is empty or exceeds its byte bound")
            if any(ord(character) < 32 or ord(character) == 127 for character in value):
                raise ValueError(f"{name} contains a control character")
        if self.scientific_evidence:
            raise ValueError("operational diagnostics cannot become scientific evidence")
        validate_sha256(self.diagnostic_sha256, field_name="diagnostic_sha256")
        if self.diagnostic_sha256 != self.expected_diagnostic_sha256(
            failure_class=self.failure_class,
            retryable=self.retryable,
            sanitized_exception_type=self.sanitized_exception_type,
            sanitized_message=self.sanitized_message,
        ):
            raise ValueError("operational diagnostic hash differs from its bounded fields")

    @staticmethod
    def expected_diagnostic_sha256(
        *,
        failure_class: OperationalFailureClass,
        retryable: bool,
        sanitized_exception_type: str | None,
        sanitized_message: str | None,
    ) -> str:
        return hashlib.sha256(
            canonical_json_bytes(
                {
                    "failure_class": failure_class.value,
                    "retryable": retryable,
                    "sanitized_exception_type": sanitized_exception_type,
                    "sanitized_message": sanitized_message,
                    "scientific_evidence": False,
                }
            )
        ).hexdigest()

    @classmethod
    def create(
        cls,
        failure_class: OperationalFailureClass,
        *,
        sanitized_exception_type: str | None,
        sanitized_message: str | None,
    ) -> Self:
        retryable = failure_class.retryable
        return cls(
            failure_class=failure_class,
            retryable=retryable,
            sanitized_exception_type=sanitized_exception_type,
            sanitized_message=sanitized_message,
            diagnostic_sha256=cls.expected_diagnostic_sha256(
                failure_class=failure_class,
                retryable=retryable,
                sanitized_exception_type=sanitized_exception_type,
                sanitized_message=sanitized_message,
            ),
        )


@dataclass(frozen=True, slots=True)
class RecoveryAttemptCandidate(CanonicalRecord):
    """One exact receipt/event location for one frozen task attempt."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/recovery-attempt-candidate'

    attempt_id: str
    ordinal: int
    receipt_relative_path: str
    event_relative_path: str
    receipt_schema: str
    event_schema: str

    def __post_init__(self) -> None:
        validate_stable_id(self.attempt_id, field_name="attempt_id")
        if self.ordinal <= 0:
            raise ValueError("recovery attempt ordinal must be positive")
        validate_relative_locator(self.receipt_relative_path)
        validate_relative_locator(self.event_relative_path)
        validate_schema(self.receipt_schema)
        validate_schema(self.event_schema)


@dataclass(frozen=True, slots=True)
class RecoveryTaskBinding(CanonicalRecord):
    """Frozen plan-to-recovery compatibility map for one execution task."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/recovery-task-binding'

    task_id: str
    topological_ordinal: int
    dependency_task_ids: tuple[str, ...]
    maximum_attempts: int
    capability_key: str
    capability_version: str
    capability_implementation_sha256: str
    output_schema_ids: tuple[str, ...]
    attempts: tuple[RecoveryAttemptCandidate, ...]
    nonattempt_event_relative_path: str

    def __post_init__(self) -> None:
        validate_stable_id(self.task_id, field_name="task_id")
        if self.topological_ordinal < 0:
            raise ValueError("task topological ordinal must be nonnegative")
        require_sorted_unique_strings(
            self.dependency_task_ids,
            field_name="dependency_task_ids",
        )
        if not 0 < self.maximum_attempts <= MAX_RECOVERY_ATTEMPTS_PER_TASK:
            raise ValueError("task maximum attempts exceed the recovery bound")
        validate_stable_id(self.capability_key, field_name="capability_key")
        validate_semantic_version(self.capability_version)
        validate_sha256(
            self.capability_implementation_sha256,
            field_name="capability_implementation_sha256",
        )
        require_sorted_unique_strings(
            self.output_schema_ids,
            field_name="output_schema_ids",
            allow_empty=False,
        )
        if len(self.attempts) != self.maximum_attempts:
            raise ValueError("recovery candidates do not cover every frozen attempt")
        if tuple(candidate.ordinal for candidate in self.attempts) != tuple(
            range(1, self.maximum_attempts + 1)
        ):
            raise ValueError("recovery attempt candidates are not ordinal-complete")
        validate_relative_locator(self.nonattempt_event_relative_path)


@dataclass(frozen=True, slots=True)
class ProtocolRunRecoveryIndex(CanonicalRecord):
    """Immutable external authority for bounded recovery of one execution wave."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/protocol-run-recovery-index'
    EXECUTION_PLAN_SCHEMA: ClassVar[str] = ProtocolExecutionPlan.SCHEMA
    TASK_EVENT_SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/task-recovery-event'
    TERMINAL_EVENT_SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/protocol-run-recovery-terminal-event'

    recovery_index_id: str
    wave_id: str
    run_id: str
    execution_plan: ObjectIdentity
    authority_identities: tuple[ObjectIdentity, ...]
    registry_sha256: str
    implementation_commit: str
    index_relative_path: str
    terminal_event_relative_path: str
    terminal_event_schema: str
    tasks: tuple[RecoveryTaskBinding, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("recovery_index_id", self.recovery_index_id),
            ("wave_id", self.wave_id),
            ("run_id", self.run_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.execution_plan.object_schema != self.EXECUTION_PLAN_SCHEMA:
            raise ValueError("recovery index must bind an execution plan")
        require_sorted_unique_ids(
            self.authority_identities,
            attribute="object_id",
            field_name="authority_identities",
        )
        if not self.authority_identities:
            raise ValueError("recovery index requires exact execution authority")
        validate_sha256(self.registry_sha256, field_name="registry_sha256")
        if len(self.implementation_commit) != 40 or any(
            value not in "0123456789abcdef" for value in self.implementation_commit
        ):
            raise ValueError("implementation_commit must be a lowercase Git SHA-1")
        validate_relative_locator(self.index_relative_path)
        validate_relative_locator(self.terminal_event_relative_path)
        validate_schema(self.terminal_event_schema)
        if self.terminal_event_schema != self.TERMINAL_EVENT_SCHEMA:
            raise ValueError("recovery index terminal-event schema is unsupported")
        require_sorted_unique_ids(self.tasks, attribute="task_id", field_name="tasks")
        if not self.tasks or len(self.tasks) > MAX_RECOVERY_TASKS:
            raise ValueError("recovery index task count is outside its bound")
        if sum(len(task.attempts) for task in self.tasks) > MAX_RECOVERY_CANDIDATES:
            raise ValueError("recovery index exceeds its aggregate candidate bound")
        by_id = {task.task_id: task for task in self.tasks}
        if tuple(sorted(task.topological_ordinal for task in self.tasks)) != tuple(
            range(len(self.tasks))
        ):
            raise ValueError("recovery task topology is not complete")
        for task in self.tasks:
            if any(
                dependency not in by_id
                or by_id[dependency].topological_ordinal >= task.topological_ordinal
                for dependency in task.dependency_task_ids
            ):
                raise ValueError("recovery task dependency order is invalid")

    @classmethod
    def from_execution_plan(
        cls,
        plan: ProtocolExecutionPlan,
        *,
        run_id: str,
        wave_id: str,
        authority_identities: tuple[ObjectIdentity, ...],
    ) -> Self:
        validate_stable_id(run_id, field_name="run_id")
        validate_stable_id(wave_id, field_name="wave_id")
        if run_id != plan.source_plan.object_id:
            raise ValueError("recovery run differs from the execution source plan")
        order = plan.topological_task_ids()
        ordinal_by_id = {task_id: index for index, task_id in enumerate(order)}
        tasks = tuple(
            sorted(
                (
                    _task_binding(
                        plan_task,
                        run_id=run_id,
                        wave_id=wave_id,
                        topological_ordinal=ordinal_by_id[plan_task.task_id],
                        event_schema=cls.TASK_EVENT_SCHEMA,
                    )
                    for plan_task in plan.tasks
                ),
                key=lambda value: value.task_id,
            )
        )
        return cls(
            recovery_index_id=f"recovery-index.{run_id}.{wave_id}",
            wave_id=wave_id,
            run_id=run_id,
            execution_plan=ObjectIdentity.from_record(plan.execution_plan_id, plan),
            authority_identities=tuple(
                sorted(authority_identities, key=lambda value: value.object_id)
            ),
            registry_sha256=plan.registry_sha256,
            implementation_commit=plan.implementation_commit,
            index_relative_path=(f"runs/{run_id}/recovery/{wave_id}/run-recovery-index.json"),
            terminal_event_relative_path=(f"runs/{run_id}/recovery/{wave_id}/terminal.json"),
            terminal_event_schema=cls.TERMINAL_EVENT_SCHEMA,
            tasks=tasks,
        )

    def validate_plan(
        self, plan: ProtocolExecutionPlan, *, retry_amendment: LeaseExpiryRetryAmendment | None = None
    ) -> None:
        if retry_amendment is not None:
            if retry_amendment.identity not in self.authority_identities:
                raise ValueError("recovery index lacks the exact retry amendment")
            base = RunRecoveryIndex.from_execution_plan(
                plan,
                run_id=self.run_id,
                wave_id=self.wave_id,
                authority_identities=tuple(
                    value
                    for value in self.authority_identities
                    if value != retry_amendment.identity
                ),
            )
            if self != retry_amendment.recovery_index(plan, base):
                raise ValueError("recovery index differs from its bounded retry amendment")
            return
        expected = type(self).from_execution_plan(
            plan,
            run_id=self.run_id,
            wave_id=self.wave_id,
            authority_identities=self.authority_identities,
        )
        if expected != self:
            raise ValueError("run recovery index differs from the frozen execution plan")

    def task(self, task_id: str) -> RecoveryTaskBinding:
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        raise KeyError(task_id)


@dataclass(frozen=True, slots=True)
class CandidateRunRecoveryIndex(ProtocolRunRecoveryIndex):
    """Recovery authority for an exact-edge execution plan."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/candidate-run-recovery-index'
    EXECUTION_PLAN_SCHEMA: ClassVar[str] = CandidateExecutionPlan.SCHEMA
    TASK_EVENT_SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/task-recovery-event'
    TERMINAL_EVENT_SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/candidate-run-recovery-terminal-event'


@dataclass(frozen=True, slots=True)
class EnvelopeRunRecoveryIndex(CandidateRunRecoveryIndex):
    """Exact-edge recovery plus issued extension/envelope event roots."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/envelope-run-recovery-index'
    EXECUTION_PLAN_SCHEMA: ClassVar[str] = EnvelopeExecutionPlan.SCHEMA
    TASK_EVENT_SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/task-recovery-event'
    TERMINAL_EVENT_SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/envelope-run-recovery-terminal-event'

    issued_extension_set: ObjectIdentity
    execution_envelope_spec: ObjectIdentity
    envelope_event_root_relative_path: str

    def __post_init__(self) -> None:
        super(EnvelopeRunRecoveryIndex, self).__post_init__()
        if (
            self.issued_extension_set.object_schema
            != 'empirical-lawhood/runtime/issued-extension-set'
            or self.execution_envelope_spec.object_schema
            != 'empirical-lawhood/runtime/execution-envelope-spec'
        ):
            raise ValueError("execution envelope recovery index has an incompatible extension or envelope identity")
        validate_relative_locator(self.envelope_event_root_relative_path)

    @classmethod
    def from_execution_plan(
        cls,
        plan: ProtocolExecutionPlan,
        *,
        run_id: str,
        wave_id: str,
        authority_identities: tuple[ObjectIdentity, ...],
    ) -> Self:
        if not isinstance(plan, EnvelopeExecutionPlan):
            raise ValueError("execution envelope recovery index requires an execution envelope plan")
        validate_stable_id(run_id, field_name="run_id")
        validate_stable_id(wave_id, field_name="wave_id")
        if run_id != plan.source_plan.object_id:
            raise ValueError("recovery run differs from the execution source plan")
        order = plan.topological_task_ids()
        ordinal_by_id = {task_id: index for index, task_id in enumerate(order)}
        tasks = tuple(
            sorted(
                (
                    _task_binding(
                        plan_task,
                        run_id=run_id,
                        wave_id=wave_id,
                        topological_ordinal=ordinal_by_id[plan_task.task_id],
                        event_schema=cls.TASK_EVENT_SCHEMA,
                    )
                    for plan_task in plan.tasks
                ),
                key=lambda value: value.task_id,
            )
        )
        recovery_root = f"runs/{run_id}/recovery/{wave_id}"
        return cls(
            recovery_index_id=f"envelope-run-recovery-index.{run_id}.{wave_id}",
            wave_id=wave_id,
            run_id=run_id,
            execution_plan=ObjectIdentity.from_record(plan.execution_plan_id, plan),
            authority_identities=tuple(
                sorted(authority_identities, key=lambda value: value.object_id)
            ),
            registry_sha256=plan.registry_sha256,
            implementation_commit=plan.implementation_commit,
            index_relative_path=f"{recovery_root}/envelope-run-recovery-index.json",
            terminal_event_relative_path=f"{recovery_root}/envelope-terminal.json",
            terminal_event_schema=cls.TERMINAL_EVENT_SCHEMA,
            tasks=tasks,
            issued_extension_set=plan.issued_extension_set,
            execution_envelope_spec=plan.execution_envelope_spec,
            envelope_event_root_relative_path=f"{recovery_root}/envelope-events",
        )


@dataclass(frozen=True, slots=True)
class RunRecoveryIndex(CandidateRunRecoveryIndex):
    """Current recovery identity, with JIT evidence only when applicable."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/run-recovery-index'
    EXECUTION_PLAN_SCHEMA: ClassVar[str] = ExecutionPlan.SCHEMA
    TASK_EVENT_SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/task-recovery-event'
    TERMINAL_EVENT_SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/run-recovery-terminal-event'

    issued_extension_set: ObjectIdentity
    execution_resource_envelope_spec: ObjectIdentity
    predevelopment_jit_signature_census: ObjectIdentity | None
    jit_graph_signature_manifest: ObjectIdentity | None
    resource_envelope_event_root_relative_path: str

    def __post_init__(self) -> None:
        super(RunRecoveryIndex, self).__post_init__()
        expected = (
            'empirical-lawhood/runtime/issued-extension-set',
            'empirical-lawhood/runtime/execution-resource-envelope-spec',
        )
        observed = (
            self.issued_extension_set.object_schema,
            self.execution_resource_envelope_spec.object_schema,
        )
        if observed != expected:
            raise ValueError("resource envelope recovery index has an incompatible resource identity")
        validate_optional_jit_identities(
            self.predevelopment_jit_signature_census, self.jit_graph_signature_manifest
        )
        validate_relative_locator(self.resource_envelope_event_root_relative_path)

    @classmethod
    def from_execution_plan(
        cls,
        plan: ProtocolExecutionPlan,
        *,
        run_id: str,
        wave_id: str,
        authority_identities: tuple[ObjectIdentity, ...],
    ) -> Self:
        if not isinstance(plan, ExecutionPlan):
            raise ValueError("resource envelope recovery index requires a resource execution plan")
        validate_stable_id(run_id, field_name="run_id")
        validate_stable_id(wave_id, field_name="wave_id")
        if run_id != plan.source_plan.object_id:
            raise ValueError("recovery run differs from the execution source plan")
        order = plan.topological_task_ids()
        ordinal_by_id = {task_id: index for index, task_id in enumerate(order)}
        tasks = tuple(
            sorted(
                (
                    _task_binding(
                        plan_task,
                        run_id=run_id,
                        wave_id=wave_id,
                        topological_ordinal=ordinal_by_id[plan_task.task_id],
                        event_schema=cls.TASK_EVENT_SCHEMA,
                    )
                    for plan_task in plan.tasks
                ),
                key=lambda value: value.task_id,
            )
        )
        recovery_root = f"runs/{run_id}/recovery/{wave_id}"
        return cls(
            recovery_index_id=f"resource-run-recovery-index.{run_id}.{wave_id}",
            wave_id=wave_id,
            run_id=run_id,
            execution_plan=ObjectIdentity.from_record(plan.execution_plan_id, plan),
            authority_identities=tuple(
                sorted(authority_identities, key=lambda value: value.object_id)
            ),
            registry_sha256=plan.registry_sha256,
            implementation_commit=plan.implementation_commit,
            index_relative_path=f"{recovery_root}/resource-run-recovery-index.json",
            terminal_event_relative_path=f"{recovery_root}/resource-terminal.json",
            terminal_event_schema=cls.TERMINAL_EVENT_SCHEMA,
            tasks=tasks,
            issued_extension_set=plan.issued_extension_set,
            execution_resource_envelope_spec=plan.execution_resource_envelope_spec,
            predevelopment_jit_signature_census=(plan.predevelopment_jit_signature_census),
            jit_graph_signature_manifest=plan.jit_graph_signature_manifest,
            resource_envelope_event_root_relative_path=RESOURCE_ENVELOPE_JOURNAL_ROOT,
        )


def build_run_recovery_index(
    plan: ProtocolExecutionPlan,
    *,
    run_id: str,
    wave_id: str,
    authority_identities: tuple[ObjectIdentity, ...],
) -> ProtocolRunRecoveryIndex:
    """Select the recovery schema matching the immutable execution-plan version."""

    index_type = (
        RunRecoveryIndex
        if isinstance(plan, ExecutionPlan)
        else EnvelopeRunRecoveryIndex
        if isinstance(plan, EnvelopeExecutionPlan)
        else CandidateRunRecoveryIndex
        if isinstance(plan, CandidateExecutionPlan)
        else ProtocolRunRecoveryIndex
    )
    return index_type.from_execution_plan(
        plan,
        run_id=run_id,
        wave_id=wave_id,
        authority_identities=authority_identities,
    )


def _task_binding(
    task: ProtocolExecutionTask,
    *,
    run_id: str,
    wave_id: str,
    topological_ordinal: int,
    event_schema: str,
) -> RecoveryTaskBinding:
    if task.maximum_attempts > MAX_RECOVERY_ATTEMPTS_PER_TASK:
        raise ValueError("execution plan maximum attempts exceed the recovery bound")
    recovery_root = f"runs/{run_id}/recovery/{wave_id}"
    attempts = tuple(
        RecoveryAttemptCandidate(
            attempt_id=f"{run_id}.{task.task_id}.attempt-{ordinal:03d}",
            ordinal=ordinal,
            receipt_relative_path=(
                f"runs/{run_id}/receipts/{task.task_id}/"
                f"{run_id}.{task.task_id}.attempt-{ordinal:03d}.json"
            ),
            event_relative_path=(
                f"{recovery_root}/events/{task.task_id}/attempt-{ordinal:03d}.json"
            ),
            receipt_schema=CanonicalTaskReceipt.SCHEMA,
            event_schema=event_schema,
        )
        for ordinal in range(1, task.maximum_attempts + 1)
    )
    return RecoveryTaskBinding(
        task_id=task.task_id,
        topological_ordinal=topological_ordinal,
        dependency_task_ids=task.dependency_task_ids,
        maximum_attempts=task.maximum_attempts,
        capability_key=task.capability.capability_key,
        capability_version=task.capability.capability_version,
        capability_implementation_sha256=task.capability_implementation_sha256,
        output_schema_ids=tuple(sorted({output.payload_schema for output in task.outputs})),
        attempts=attempts,
        nonattempt_event_relative_path=(
            f"{recovery_root}/events/{task.task_id}/not-attempted.json"
        ),
    )


@dataclass(frozen=True, slots=True)
class ProtocolTaskRecoveryEvent(CanonicalRecord):
    """Append-only terminal projection separate from its immutable index."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/protocol-task-recovery-event'
    RECOVERY_INDEX_SCHEMA: ClassVar[str] = ProtocolRunRecoveryIndex.SCHEMA
    RECOVERY_INDEX_SCHEMAS: ClassVar[tuple[str, ...]] = (ProtocolRunRecoveryIndex.SCHEMA,)

    event_id: str
    recovery_index: ObjectIdentity
    run_id: str
    task_id: str
    attempt_id: str | None
    attempt_ordinal: int | None
    disposition: TaskRecoveryDisposition
    receipt_id: str | None
    receipt_relative_path: str | None
    receipt_sha256: str | None
    receipt_schema: str | None
    output_materialization_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("event_id", self.event_id),
            ("run_id", self.run_id),
            ("task_id", self.task_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.recovery_index.object_schema not in self.RECOVERY_INDEX_SCHEMAS:
            raise ValueError("task recovery event binds another index schema")
        require_sorted_unique_strings(
            self.output_materialization_ids,
            field_name="output_materialization_ids",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        receipt_values = (
            self.receipt_id,
            self.receipt_relative_path,
            self.receipt_sha256,
            self.receipt_schema,
        )
        if self.disposition is TaskRecoveryDisposition.SUCCEEDED:
            if self.attempt_id is None or self.attempt_ordinal is None:
                raise ValueError("successful recovery event requires an attempt")
            if any(value is None for value in receipt_values):
                raise ValueError("successful recovery event requires exact receipt identity")
            if self.reason_codes or not self.output_materialization_ids:
                raise ValueError("successful recovery event has invalid terminal fields")
        elif self.disposition in {
            TaskRecoveryDisposition.FAILED,
            TaskRecoveryDisposition.BLOCKED,
        }:
            if self.attempt_id is None or self.attempt_ordinal is None:
                raise ValueError("failed recovery event requires an attempt")
            if any(value is not None for value in receipt_values):
                raise ValueError("failed recovery event cannot claim a receipt")
            if len(self.reason_codes) != 1 or self.output_materialization_ids:
                raise ValueError("unsuccessful recovery event requires only terminal reasons")
        else:
            if self.attempt_id is not None or self.attempt_ordinal is not None:
                raise ValueError("nonattempt recovery event cannot claim an attempt")
            if any(value is not None for value in receipt_values):
                raise ValueError("nonattempt recovery event cannot claim a receipt")
            if len(self.reason_codes) != 1 or self.output_materialization_ids:
                raise ValueError("nonattempt recovery event requires only terminal reasons")
        if self.attempt_id is not None:
            validate_stable_id(self.attempt_id, field_name="attempt_id")
        if self.attempt_ordinal is not None and self.attempt_ordinal <= 0:
            raise ValueError("task recovery attempt ordinal must be positive")
        if self.receipt_id is not None:
            validate_stable_id(self.receipt_id, field_name="receipt_id")
        if self.receipt_relative_path is not None:
            validate_relative_locator(self.receipt_relative_path)
        if self.receipt_sha256 is not None:
            validate_sha256(self.receipt_sha256, field_name="receipt_sha256")
        if self.receipt_schema is not None:
            validate_schema(self.receipt_schema)

    @classmethod
    def from_receipt(
        cls,
        index: ProtocolRunRecoveryIndex,
        receipt: CanonicalTaskReceipt,
    ) -> Self:
        task = index.task(receipt.task_id)
        candidate = next(
            (value for value in task.attempts if value.attempt_id == receipt.attempt_id),
            None,
        )
        if candidate is None:
            raise ValueError("task receipt lies outside the bounded recovery candidates")
        return cls(
            event_id=f"recovery-event.{receipt.attempt_id}.succeeded",
            recovery_index=ObjectIdentity.from_record(index.recovery_index_id, index),
            run_id=index.run_id,
            task_id=receipt.task_id,
            attempt_id=receipt.attempt_id,
            attempt_ordinal=candidate.ordinal,
            disposition=TaskRecoveryDisposition.SUCCEEDED,
            receipt_id=receipt.receipt_id,
            receipt_relative_path=candidate.receipt_relative_path,
            receipt_sha256=receipt.fingerprint(),
            receipt_schema=receipt.SCHEMA,
            output_materialization_ids=tuple(
                sorted(value.materialization_id for value in receipt.output_materializations)
            ),
            reason_codes=(),
        )


@dataclass(frozen=True, slots=True)
class CandidateTaskRecoveryEvent(ProtocolTaskRecoveryEvent):
    "Append-only event shared by exact-edge and deadline-free execution machines."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/candidate-task-recovery-event'
    RECOVERY_INDEX_SCHEMA: ClassVar[str] = CandidateRunRecoveryIndex.SCHEMA
    RECOVERY_INDEX_SCHEMAS: ClassVar[tuple[str, ...]] = (
        CandidateRunRecoveryIndex.SCHEMA,
        EnvelopeRunRecoveryIndex.SCHEMA,
        RunRecoveryIndex.SCHEMA,
    )


@dataclass(frozen=True, slots=True)
class TaskRecoveryEvent(ProtocolTaskRecoveryEvent):
    """Current bounded event retaining non-scientific operational causality."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/task-recovery-event'
    RECOVERY_INDEX_SCHEMA: ClassVar[str] = ProtocolRunRecoveryIndex.SCHEMA
    RECOVERY_INDEX_SCHEMAS: ClassVar[tuple[str, ...]] = (
        ProtocolRunRecoveryIndex.SCHEMA,
        CandidateRunRecoveryIndex.SCHEMA,
        EnvelopeRunRecoveryIndex.SCHEMA,
        RunRecoveryIndex.SCHEMA,
    )

    failure: OperationalFailureDiagnostic | None = None

    def __post_init__(self) -> None:
        super(TaskRecoveryEvent, self).__post_init__()
        if self.disposition in {
            TaskRecoveryDisposition.FAILED,
            TaskRecoveryDisposition.BLOCKED,
        }:
            if self.failure is None:
                raise ValueError("unsuccessful current recovery event requires a typed diagnostic")
            if self.disposition is TaskRecoveryDisposition.BLOCKED:
                TaskBlockReason(self.reason_codes[0])
        elif self.failure is not None:
            raise ValueError("nonfailure recovery event cannot carry a failure diagnostic")


@dataclass(frozen=True, slots=True)
class RecoveryTerminalAttempt(CanonicalRecord):
    """Exact compact terminal state for one attempted or blocked task row."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/recovery-terminal-attempt'

    attempt_id: str
    task_id: str
    ordinal: int
    disposition: RecoveryTerminalDisposition
    reason_code: str | None
    receipt_id: str | None
    receipt_sha256: str | None

    def __post_init__(self) -> None:
        validate_stable_id(self.attempt_id, field_name="attempt_id")
        validate_stable_id(self.task_id, field_name="task_id")
        if self.ordinal <= 0:
            raise ValueError("terminal attempt ordinal must be positive")
        if self.disposition is RecoveryTerminalDisposition.SUCCEEDED:
            if self.reason_code is not None or self.receipt_id is None:
                raise ValueError("successful terminal attempt requires only a receipt")
        else:
            if self.reason_code is None or self.receipt_id is not None:
                raise ValueError("non-success terminal attempt requires only a reason")
        if self.reason_code is not None:
            validate_stable_id(self.reason_code, field_name="reason_code")
        if self.receipt_id is not None:
            validate_stable_id(self.receipt_id, field_name="receipt_id")
        if (self.receipt_id is None) is not (self.receipt_sha256 is None):
            raise ValueError("terminal receipt identity must be complete")
        if self.receipt_sha256 is not None:
            validate_sha256(self.receipt_sha256, field_name="receipt_sha256")


@dataclass(frozen=True, slots=True)
class ProtocolRunRecoveryTerminalEvent(CanonicalRecord):
    """Immutable closure proving the exact externally recoverable wave state."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/protocol-run-recovery-terminal-event'
    RECOVERY_INDEX_SCHEMA: ClassVar[str] = ProtocolRunRecoveryIndex.SCHEMA

    terminal_event_id: str
    recovery_index: ObjectIdentity
    run_id: str
    operational_status: str
    attempts: tuple[RecoveryTerminalAttempt, ...]
    completed_task_ids: tuple[str, ...]
    failed_task_ids: tuple[str, ...]
    blocked_task_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.terminal_event_id, field_name="terminal_event_id")
        validate_stable_id(self.run_id, field_name="run_id")
        if self.recovery_index.object_schema != self.RECOVERY_INDEX_SCHEMA:
            raise ValueError("terminal event binds another recovery index schema")
        if self.operational_status not in {"SUCCEEDED", "FAILED", "BLOCKED"}:
            raise ValueError("terminal event has a nonterminal operational status")
        attempt_keys = tuple((value.task_id, value.ordinal) for value in self.attempts)
        if tuple(sorted(set(attempt_keys))) != attempt_keys:
            raise ValueError("terminal attempts must be task/ordinal sorted and unique")
        task_ids = tuple(sorted({value.task_id for value in self.attempts}))
        for task_id in task_ids:
            ordinals = tuple(value.ordinal for value in self.attempts if value.task_id == task_id)
            if ordinals != tuple(range(1, len(ordinals) + 1)):
                raise ValueError("terminal attempt ordinals must be contiguous per task")
        for name, values in (
            ("completed_task_ids", self.completed_task_ids),
            ("failed_task_ids", self.failed_task_ids),
            ("blocked_task_ids", self.blocked_task_ids),
        ):
            require_sorted_unique_strings(values, field_name=name)
        if set(self.completed_task_ids).intersection(
            {*self.failed_task_ids, *self.blocked_task_ids}
        ) or set(self.failed_task_ids).intersection(self.blocked_task_ids):
            raise ValueError("terminal task sets must be disjoint")
        terminal_by_task: dict[str, RecoveryTerminalDisposition] = {}
        for attempt in self.attempts:
            terminal_by_task[attempt.task_id] = attempt.disposition
        expected_completed = tuple(
            sorted(
                task_id
                for task_id, disposition in terminal_by_task.items()
                if disposition is RecoveryTerminalDisposition.SUCCEEDED
            )
        )
        expected_failed = tuple(
            sorted(
                task_id
                for task_id, disposition in terminal_by_task.items()
                if disposition is RecoveryTerminalDisposition.FAILED
            )
        )
        expected_blocked = tuple(
            sorted(
                task_id
                for task_id, disposition in terminal_by_task.items()
                if disposition is RecoveryTerminalDisposition.BLOCKED
            )
        )
        if (
            expected_completed != self.completed_task_ids
            or expected_failed != self.failed_task_ids
            or expected_blocked != self.blocked_task_ids
        ):
            raise ValueError("terminal task sets differ from their latest attempts")
        if self.operational_status == "SUCCEEDED" and (
            self.failed_task_ids or self.blocked_task_ids
        ):
            raise ValueError("successful terminal event contains non-success tasks")
        if self.operational_status == "FAILED" and not self.failed_task_ids:
            raise ValueError("failed terminal event lacks failed tasks")
        if self.operational_status == "BLOCKED" and not self.blocked_task_ids:
            raise ValueError("blocked terminal event lacks blocked tasks")


@dataclass(frozen=True, slots=True)
class CandidateRunRecoveryTerminalEvent(ProtocolRunRecoveryTerminalEvent):
    """Terminal recovery closure for an exact-edge recovery index."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/candidate-run-recovery-terminal-event'
    RECOVERY_INDEX_SCHEMA: ClassVar[str] = CandidateRunRecoveryIndex.SCHEMA


@dataclass(frozen=True, slots=True)
class EnvelopeRunRecoveryTerminalEvent(CandidateRunRecoveryTerminalEvent):
    """Terminal closure for extension/envelope recovery authority."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/envelope-run-recovery-terminal-event'
    RECOVERY_INDEX_SCHEMA: ClassVar[str] = EnvelopeRunRecoveryIndex.SCHEMA


@dataclass(frozen=True, slots=True)
class RunRecoveryTerminalEvent(CandidateRunRecoveryTerminalEvent):
    """Terminal closure for deadline-free resource/JIT recovery authority."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/run-recovery-terminal-event'
    RECOVERY_INDEX_SCHEMA: ClassVar[str] = RunRecoveryIndex.SCHEMA


@dataclass(frozen=True, slots=True)
class RunRecoveryReport:
    """Bounded reconciliation outcome; never itself scientific evidence."""

    candidate_count: int
    receipt_probe_count: int
    event_probe_count: int
    recovered_attempt_count: int
    recovered_task_ids: tuple[str, ...]
    recovered_receipt_ids: tuple[str, ...]
    recovered_receipts: tuple[CanonicalTaskReceipt, ...]
    terminal_status: str | None
    completed_task_ids: tuple[str, ...]
    failed_task_ids: tuple[str, ...]
    blocked_task_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            min(
                self.candidate_count,
                self.receipt_probe_count,
                self.event_probe_count,
                self.recovered_attempt_count,
            )
            < 0
        ):
            raise ValueError("recovery report counts must be nonnegative")
        if self.receipt_probe_count > self.candidate_count:
            raise ValueError("recovery receipt probes exceed bounded candidates")
        if self.event_probe_count > 2 * self.candidate_count:
            raise ValueError("recovery event probes exceed bounded candidates")
        require_sorted_unique_strings(
            self.recovered_task_ids,
            field_name="recovered_task_ids",
        )
        require_sorted_unique_strings(
            self.recovered_receipt_ids,
            field_name="recovered_receipt_ids",
        )
        if tuple(receipt.receipt_id for receipt in self.recovered_receipts) != (
            self.recovered_receipt_ids
        ):
            raise ValueError("recovery report receipt records differ from their IDs")
        for name, values in (
            ("completed_task_ids", self.completed_task_ids),
            ("failed_task_ids", self.failed_task_ids),
            ("blocked_task_ids", self.blocked_task_ids),
        ):
            require_sorted_unique_strings(values, field_name=name)
        if self.terminal_status is None:
            if self.completed_task_ids or self.failed_task_ids or self.blocked_task_ids:
                raise ValueError("nonterminal recovery report has terminal task sets")
        elif self.terminal_status not in {"SUCCEEDED", "FAILED", "BLOCKED"}:
            raise ValueError("recovery report terminal status is invalid")
