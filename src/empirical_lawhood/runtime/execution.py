"""Capability-isolated task, receipt and operational-repository ports."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import Protocol

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.serialization import (
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_schema,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.status import OperationalStatus

from .artifacts import (
    ArtifactMaterialization,
    ArtifactProfile,
    ArtifactWriteResult,
    CanonicalTaskReceipt,
    LogicalArtifactIdentity,
    ReceiptCheck,
)
from .adjudication import ScientificAdjudicationContext
from .capabilities import (
    CapabilityConfigRef,
    CapabilityManifest,
    CapabilityPermission,
    CapabilityRegistry,
)
from .plans import ProtocolExecutionPlan, ProtocolExecutionTask
from .execution_envelope import ExecutionResourceTaskCellSpec


class TaskAttemptDisposition(StrEnum):
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


class OperationalFailureClass(StrEnum):
    """Closed non-scientific failure algebra controlling automatic retry."""

    CONTRACT_REFUSAL = "CONTRACT_REFUSAL"
    RESOURCE_REFUSAL = "RESOURCE_REFUSAL"
    PROVIDER_RUNNER_DEFECT = "PROVIDER_RUNNER_DEFECT"
    CHILD_PROCESS_FAILURE = "CHILD_PROCESS_FAILURE"
    TRANSIENT_INFRASTRUCTURE_FAILURE = "TRANSIENT_INFRASTRUCTURE_FAILURE"
    RETRY_EXHAUSTION = "RETRY_EXHAUSTION"
    RECOVERY_OBSTRUCTION = "RECOVERY_OBSTRUCTION"

    @property
    def retryable(self) -> bool:
        """Return the automatic in-run retry policy owned by this class."""

        return self in {
            OperationalFailureClass.CHILD_PROCESS_FAILURE,
            OperationalFailureClass.TRANSIENT_INFRASTRUCTURE_FAILURE,
        }

    @property
    def reason_code(self) -> str:
        return self.value.casefold().replace("_", "-")


class TaskBlockKind(StrEnum):
    """Whether a blocked task may be reconsidered on an explicit resume."""

    RETRYABLE = "RETRYABLE"
    DURABLE = "DURABLE"


class TaskBlockReason(StrEnum):
    """Closed scheduler block taxonomy with an explicit recovery contract."""

    DEPENDENCY_BLOCKED = "dependency-blocked"
    DEPENDENCY_FAILED = "dependency-failed"
    RESOURCE_LOCK_UNAVAILABLE = "resource-lock-unavailable"
    RESOURCE_COMPUTABILITY_UNAVAILABLE = "resource-computability-unavailable"
    AUTHORITY_REQUIRED = "authority-required"
    UNKNOWN_COMPLETION = "unknown-completion"
    CAMPAIGN_ELAPSED_BUDGET_EXHAUSTED = "campaign-elapsed-budget-exhausted"
    # Read compatibility for local projections written before block kinds were
    # separated. The old reason represented an unavailable prerequisite and is
    # therefore conservatively re-evaluated on resume.
    HISTORICAL_DEPENDENCY_NOT_SUCCEEDED = "dependency-not-succeeded"

    @property
    def kind(self) -> TaskBlockKind:
        if self in {
            TaskBlockReason.DEPENDENCY_FAILED,
            TaskBlockReason.UNKNOWN_COMPLETION,
            TaskBlockReason.CAMPAIGN_ELAPSED_BUDGET_EXHAUSTED,
        }:
            return TaskBlockKind.DURABLE
        return TaskBlockKind.RETRYABLE


class WorkerIsolationProfile(StrEnum):
    """Truthful local-worker boundary requested by the frozen execution lane."""

    TRUSTED_LOCAL = "TRUSTED_LOCAL"
    EXPLORATION_NO_NETWORK = "EXPLORATION_NO_NETWORK"


class ExecutionAssuranceProfile(StrEnum):
    """Operational assurance selected independently of scientific identity."""

    TRUSTED_LOCAL = "TRUSTED_LOCAL"
    STRICT_ISOLATED = "STRICT_ISOLATED"


class WorkerInputKind(StrEnum):
    """Frozen origin of one path-free worker input."""

    DEPENDENCY = "DEPENDENCY"
    EXTERNAL = "EXTERNAL"


class BoundedInputReader(Protocol):
    """Sequential, path-free reader over one already verified materialization."""

    @property
    def bytes_read(self) -> int: ...

    def read(self, size: int = -1) -> bytes: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class WorkerInputBinding:
    """Immutable input identity safe to expose across the worker boundary.

    The binding deliberately excludes storage roots and relative locators.  It
    is the exact lineage identity paired with one already authorized input
    port, not a means of locating the underlying artifact.
    """

    artifact_id: str
    materialization_id: str
    payload_schema: str
    media_type: str
    size_bytes: int
    visibility_ceiling: VisibilityCeiling
    outcome_access: OutcomeAccess
    kind: WorkerInputKind

    def __post_init__(self) -> None:
        validate_stable_id(self.artifact_id, field_name="artifact_id")
        validate_stable_id(self.materialization_id, field_name="materialization_id")
        validate_schema(self.payload_schema)
        validate_nonempty(self.media_type, field_name="media_type")
        if self.size_bytes < 0:
            raise ValueError("worker input binding size must be nonnegative")

    @classmethod
    def from_verified(
        cls,
        value: VerifiedArtifactInput,
        *,
        kind: WorkerInputKind,
    ) -> WorkerInputBinding:
        return cls(
            artifact_id=value.logical.logical_artifact_id,
            materialization_id=value.materialization.materialization_id,
            payload_schema=value.logical.payload_schema,
            media_type=value.logical.media_type,
            size_bytes=value.materialization.size_bytes,
            visibility_ceiling=value.logical.visibility_ceiling,
            outcome_access=value.logical.outcome_access,
            kind=kind,
        )


@dataclass(frozen=True, slots=True)
class WorkerInputPort:
    """Capability-scoped input handle over one anonymous preopened input."""

    binding: WorkerInputBinding
    _reader: BoundedInputReader = field(repr=False, compare=False)

    @property
    def artifact_id(self) -> str:
        return self.binding.artifact_id

    @property
    def materialization_id(self) -> str:
        return self.binding.materialization_id

    @property
    def payload_schema(self) -> str:
        return self.binding.payload_schema

    @property
    def media_type(self) -> str:
        return self.binding.media_type

    @property
    def size_bytes(self) -> int:
        return self.binding.size_bytes

    @property
    def visibility_ceiling(self) -> VisibilityCeiling:
        return self.binding.visibility_ceiling

    @property
    def outcome_access(self) -> OutcomeAccess:
        return self.binding.outcome_access

    @property
    def kind(self) -> WorkerInputKind:
        return self.binding.kind

    @property
    def bytes_read(self) -> int:
        return self._reader.bytes_read

    def read(self, size: int = -1) -> bytes:
        return self._reader.read(size)

    def close(self) -> None:
        self._reader.close()


@dataclass(frozen=True, slots=True)
class WorkerOutputPort:
    """Path-free output contract exposed to an isolated task runner."""

    output_id: str
    payload_schema: str
    profile: ArtifactProfile
    media_type: str
    logical_artifact_id: str | None = None

    def __post_init__(self) -> None:
        validate_stable_id(self.output_id, field_name="output_id")
        validate_schema(self.payload_schema)
        validate_nonempty(self.media_type, field_name="media_type")
        if self.logical_artifact_id is not None:
            validate_stable_id(
                self.logical_artifact_id,
                field_name="logical_artifact_id",
            )


@dataclass(frozen=True, slots=True)
class VerifiedArtifactInput:
    """One externally resolved, content-verified read-only task input."""

    logical: LogicalArtifactIdentity
    materialization: ArtifactMaterialization

    def __post_init__(self) -> None:
        if self.logical.logical_artifact_id != self.materialization.logical_artifact_id:
            raise ValueError("verified external input logical/materialized identities differ")

    @property
    def artifact_id(self) -> str:
        return self.logical.logical_artifact_id


@dataclass(frozen=True, slots=True)
class DependencyReceiptBinding:
    """Path-free proof that dependency materializations came from one receipt."""

    receipt_id: str
    task_id: str
    output_materialization_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        validate_stable_id(self.task_id, field_name="task_id")
        require_sorted_unique_strings(
            self.output_materialization_ids,
            field_name="output_materialization_ids",
            allow_empty=False,
        )


@dataclass(frozen=True, slots=True)
class TaskContext:
    """Complete worker authority, without stores, database sessions or paths.

    Local Python runners are trusted, statically registered code. This narrow
    context is an authority and I/O contract; it is not an OS sandbox.
    """

    run_id: str
    task_id: str
    attempt_id: str
    config: CapabilityConfigRef
    input_bindings: tuple[WorkerInputBinding, ...]
    input_ports: tuple[WorkerInputPort, ...]
    output_ports: tuple[WorkerOutputPort, ...]
    permissions: tuple[CapabilityPermission, ...]
    outcome_access: OutcomeAccess
    resource_budget: ResourceBudget
    isolation_profile: WorkerIsolationProfile
    dependency_receipts: tuple[DependencyReceiptBinding, ...] = ()
    scientific_adjudication_context: ScientificAdjudicationContext | None = None
    cpu_affinity: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("run_id", self.run_id),
            ("task_id", self.task_id),
            ("attempt_id", self.attempt_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_ids(
            self.input_bindings,
            attribute="artifact_id",
            field_name="input_bindings",
        )
        materialization_ids = tuple(value.materialization_id for value in self.input_bindings)
        if len(set(materialization_ids)) != len(materialization_ids):
            raise ValueError("input_bindings must have unique materialization_id values")
        require_sorted_unique_ids(
            self.input_ports,
            attribute="artifact_id",
            field_name="input_ports",
        )
        require_sorted_unique_ids(
            self.output_ports,
            attribute="output_id",
            field_name="output_ports",
        )
        require_sorted_unique_strings(self.permissions, field_name="permissions")
        require_sorted_unique_ids(
            self.dependency_receipts,
            attribute="receipt_id",
            field_name="dependency_receipts",
        )
        if (
            tuple(sorted(set(self.cpu_affinity))) != self.cpu_affinity
            or any(value < 0 for value in self.cpu_affinity)
            or len(self.cpu_affinity) > self.resource_budget.cpu_cores
        ):
            raise ValueError("worker CPU affinity is invalid")
        receipt_materialization_ids = tuple(
            sorted(
                materialization_id
                for receipt in self.dependency_receipts
                for materialization_id in receipt.output_materialization_ids
            )
        )
        dependency_materialization_ids = tuple(
            sorted(
                value.materialization_id
                for value in self.input_bindings
                if value.kind is WorkerInputKind.DEPENDENCY
            )
        )
        if receipt_materialization_ids != dependency_materialization_ids:
            raise ValueError("worker dependency inputs differ from receipt-bound materializations")
        if tuple(port.binding for port in self.input_ports) != self.input_bindings:
            raise ValueError("worker input ports differ from the exact frozen input bindings")

    @property
    def external_input_artifact_ids(self) -> tuple[str, ...]:
        return tuple(
            value.artifact_id
            for value in self.input_bindings
            if value.kind is WorkerInputKind.EXTERNAL
        )

    @property
    def input_materialization_ids(self) -> tuple[str, ...]:
        return tuple(sorted(value.materialization_id for value in self.input_bindings))

    @property
    def dependency_input_materialization_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                value.materialization_id
                for value in self.input_bindings
                if value.kind is WorkerInputKind.DEPENDENCY
            )
        )

    @property
    def dependency_receipt_ids(self) -> tuple[str, ...]:
        return tuple(value.receipt_id for value in self.dependency_receipts)


@dataclass(frozen=True, slots=True)
class TaskOutputPayload:
    output_id: str
    payload: bytes
    logical_content_sha256: str | None = None

    def __post_init__(self) -> None:
        validate_stable_id(self.output_id, field_name="output_id")
        if not isinstance(self.payload, bytes):
            raise ValueError("task output payload must be immutable bytes")
        if self.logical_content_sha256 is not None:
            validate_sha256(
                self.logical_content_sha256,
                field_name="logical_content_sha256",
            )


class TaskOutputSource(Protocol):
    def chunks(self, maximum_chunk_bytes: int) -> Iterator[bytes]: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class StreamedTaskOutput:
    """Parent-side bounded stream produced without crossing the worker pipe whole."""

    output_id: str
    size_bytes: int
    physical_sha256: str
    source: TaskOutputSource = field(repr=False, compare=False)
    logical_content_sha256: str | None = None

    def __post_init__(self) -> None:
        validate_stable_id(self.output_id, field_name="output_id")
        if self.size_bytes < 0:
            raise ValueError("streamed task output size must be nonnegative")
        validate_sha256(self.physical_sha256, field_name="physical_sha256")
        if self.logical_content_sha256 is not None:
            validate_sha256(
                self.logical_content_sha256,
                field_name="logical_content_sha256",
            )


class StreamingOutputEmitter(Protocol):
    def write(self, output_id: str, chunk: bytes) -> None: ...


@dataclass(frozen=True, slots=True)
class RunnerResult:
    outputs: tuple[TaskOutputPayload | StreamedTaskOutput, ...]
    checks: tuple[ReceiptCheck, ...]

    def __post_init__(self) -> None:
        require_sorted_unique_ids(self.outputs, attribute="output_id", field_name="outputs")
        require_sorted_unique_ids(self.checks, attribute="check_id", field_name="checks")


class TaskProgressEmitter(Protocol):
    """Report strictly increasing completed work in the declared native unit."""

    def advance(self, work_counter: Decimal) -> None: ...


class TaskRunner(Protocol):
    manifest: CapabilityManifest

    def execute(self, context: TaskContext) -> RunnerResult: ...


def validate_runner_resource_interface(
    runner: TaskRunner,
    cell: ExecutionResourceTaskCellSpec,
) -> None:
    """Check declared execution interfaces without invoking any runner code."""
    if callable(getattr(runner, "jit_signature_projection", None)) != (
        cell.jit_cell_id is not None
    ):
        raise ValueError("runner JIT applicability differs from its declared task cell")
    if cell.progress_liveness is not None and not callable(
        getattr(runner, "execute_with_progress", None)
    ):
        raise ValueError("runner lacks its declared progress interface")


class StreamingTaskRunner(Protocol):
    manifest: CapabilityManifest

    def execute_streaming(
        self,
        context: TaskContext,
        emitter: StreamingOutputEmitter,
    ) -> tuple[ReceiptCheck, ...]: ...


@dataclass(frozen=True, slots=True)
class ExecutorEnforcementCapability:
    """Typed evidence for the limits an executor actually enforces.

    This is deliberately independent of source-control identity.  A caller may
    relax Git replay checks for a synthetic fixture without thereby asserting
    that its executor enforces compute or I/O limits.
    """

    capability_id: str
    cpu_time_limit: bool
    address_space_limit: bool
    wall_time_limit: bool
    source_scan_limit: bool
    scratch_limit: bool
    output_limit: bool
    network_isolation: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.capability_id, field_name="capability_id")

    @property
    def enforces_resources(self) -> bool:
        return all(
            (
                self.cpu_time_limit,
                self.address_space_limit,
                self.wall_time_limit,
                self.source_scan_limit,
                self.scratch_limit,
                self.output_limit,
            )
        )


class TaskExecutor(Protocol):
    """Infrastructure boundary for isolated task execution."""

    enforcement_capability: ExecutorEnforcementCapability

    def execute(
        self,
        runner: TaskRunner,
        context: TaskContext,
        *,
        timeout_seconds: int,
    ) -> RunnerResult: ...


class RunnerRegistry:
    """Explicit in-process binding; configuration can name only registered manifests."""

    def __init__(
        self,
        runners: tuple[TaskRunner, ...],
        *,
        registry_sha256: str,
    ) -> None:
        validate_sha256(registry_sha256, field_name="registry_sha256")
        identifiers = tuple(
            f"{runner.manifest.capability_key}@{runner.manifest.capability_version}"
            for runner in runners
        )
        if tuple(sorted(set(identifiers))) != identifiers:
            raise ValueError("runner registry must be sorted and unique")
        self._runners = runners
        self.registry_sha256 = registry_sha256

    def resolve(self, task: ProtocolExecutionTask) -> TaskRunner:
        for runner in self._runners:
            manifest = runner.manifest
            if (
                manifest.capability_key == task.capability.capability_key
                and manifest.capability_version == task.capability.capability_version
            ):
                if manifest.implementation_sha256 != task.capability_implementation_sha256:
                    raise ValueError("runner implementation drifted after plan freeze")
                CapabilityRegistry(
                    registry_id="runner-conformance",
                    capabilities=(manifest,),
                ).require(task.capability)
                return runner
        raise KeyError(
            "runner is not registered: "
            f"{task.capability.capability_key}@{task.capability.capability_version}"
        )


@dataclass(frozen=True, slots=True)
class OperationalAttempt:
    attempt_id: str
    run_id: str
    task_id: str
    ordinal: int
    disposition: TaskAttemptDisposition
    reason_code: str | None
    lease_id: str | None
    lease_expires_epoch_seconds: int | None

    def __post_init__(self) -> None:
        for name, value in (
            ("attempt_id", self.attempt_id),
            ("run_id", self.run_id),
            ("task_id", self.task_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.ordinal <= 0:
            raise ValueError("attempt ordinal must be positive")
        if self.reason_code is not None:
            validate_stable_id(self.reason_code, field_name="reason_code")
        if self.lease_id is not None:
            validate_stable_id(self.lease_id, field_name="lease_id")
        has_lease = self.lease_id is not None or self.lease_expires_epoch_seconds is not None
        if self.disposition is TaskAttemptDisposition.RUNNING:
            if self.lease_id is None or self.lease_expires_epoch_seconds is None:
                raise ValueError("running attempts require a complete lease")
            if self.reason_code is not None:
                raise ValueError("running attempts cannot have terminal reasons")
        elif has_lease:
            raise ValueError("terminal attempts cannot retain a lease")
        if (
            self.disposition
            in {
                TaskAttemptDisposition.FAILED,
                TaskAttemptDisposition.BLOCKED,
            }
            and self.reason_code is None
        ):
            raise ValueError("failed or blocked attempts require a reason")
        if self.disposition is TaskAttemptDisposition.BLOCKED:
            assert self.reason_code is not None
            TaskBlockReason(self.reason_code)

    @property
    def block_kind(self) -> TaskBlockKind | None:
        if self.disposition is not TaskAttemptDisposition.BLOCKED:
            return None
        assert self.reason_code is not None
        return TaskBlockReason(self.reason_code).kind


class OperationalRepository(Protocol):
    def register_run(self, run_id: str, plan_sha256: str) -> None: ...

    def run_status(self, run_id: str) -> OperationalStatus: ...

    def set_run_status(self, run_id: str, status: OperationalStatus) -> None: ...

    def attempts(self, run_id: str) -> tuple[OperationalAttempt, ...]: ...

    def start_attempt(
        self,
        *,
        run_id: str,
        task_id: str,
        attempt_id: str,
        lease_id: str,
        owner_id: str,
        expires_epoch_seconds: int,
    ) -> None: ...

    def complete_attempt(self, attempt_id: str) -> None: ...

    def fail_attempt(self, attempt_id: str, reason_code: str) -> None: ...

    def block_attempt(
        self,
        attempt_id: str,
        reason: TaskBlockReason,
    ) -> None: ...

    def block_task(
        self,
        run_id: str,
        task_id: str,
        reason: TaskBlockReason,
    ) -> None: ...


class ExternalInputResolver(Protocol):
    def resolve(
        self,
        logical_artifact_ids: tuple[str, ...],
    ) -> tuple[VerifiedArtifactInput, ...]: ...


class WorkerInputPortFactory(Protocol):
    def open_many(
        self,
        values: tuple[VerifiedArtifactInput, ...],
        *,
        bindings: tuple[WorkerInputBinding, ...],
        maximum_bytes: int,
    ) -> tuple[WorkerInputPort, ...]: ...

    def open(
        self,
        value: VerifiedArtifactInput,
        *,
        binding: WorkerInputBinding | None = None,
        maximum_bytes: int,
    ) -> WorkerInputPort: ...


class TaskReceiptStore(Protocol):
    def commit(
        self,
        receipt: CanonicalTaskReceipt,
        *,
        visibility_ceiling: VisibilityCeiling,
        outcome_access: OutcomeAccess,
    ) -> ArtifactWriteResult: ...

    def read(self, run_id: str, task_id: str, attempt_id: str) -> CanonicalTaskReceipt | None: ...


class EpochClock(Protocol):
    def now(self) -> int: ...


@dataclass(frozen=True, slots=True)
class SchedulerResult:
    run_id: str
    status: OperationalStatus
    completed_task_ids: tuple[str, ...]
    failed_task_ids: tuple[str, ...]
    blocked_task_ids: tuple[str, ...]
    receipt_ids: tuple[str, ...]
    receipts: tuple[CanonicalTaskReceipt, ...] = ()

    def __post_init__(self) -> None:
        validate_stable_id(self.run_id, field_name="run_id")
        for field_name, values in (
            ("completed_task_ids", self.completed_task_ids),
            ("failed_task_ids", self.failed_task_ids),
            ("blocked_task_ids", self.blocked_task_ids),
            ("receipt_ids", self.receipt_ids),
        ):
            require_sorted_unique_strings(values, field_name=field_name)
        if set(self.completed_task_ids).intersection(
            {*self.failed_task_ids, *self.blocked_task_ids}
        ):
            raise ValueError("scheduler terminal task sets must be disjoint")
        if self.receipts:
            observed = tuple(receipt.receipt_id for receipt in self.receipts)
            if observed != self.receipt_ids:
                raise ValueError("scheduler receipt records differ from receipt identities")
        if self.status is OperationalStatus.SUCCEEDED and (
            self.failed_task_ids or self.blocked_task_ids
        ):
            raise ValueError("successful scheduler result cannot retain failures")
        if self.status is OperationalStatus.FAILED and not self.failed_task_ids:
            raise ValueError("failed scheduler result requires failed tasks")
        if self.status is OperationalStatus.BLOCKED and not self.blocked_task_ids:
            raise ValueError("blocked scheduler result requires blocked tasks")


class Scheduler(Protocol):
    def execute(self, run_id: str, plan: ProtocolExecutionPlan) -> SchedulerResult: ...
