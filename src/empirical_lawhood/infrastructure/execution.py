"""Local capability-isolated scheduler with receipt-first crash reconciliation."""

from __future__ import annotations

import json
import hashlib
import multiprocessing
import os
import resource
import signal
import stat
import threading
import time
from datetime import datetime, timezone
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, BinaryIO, Callable, Protocol

from empirical_lawhood.infrastructure.worker_bootstrap import (
    join_worker_process,
    numerical_spawn_environment,
)

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import validate_stable_id
from empirical_lawhood.kernel.status import OperationalStatus
from empirical_lawhood.runtime.artifacts import (
    ArtifactManifest,
    ArtifactLineageParent,
    ArtifactProfile,
    ArtifactStreamWriteRequest,
    ArtifactWriter,
    ArtifactWriteRequest,
    CanonicalTaskReceipt,
    ReceiptCheck,
    lineage_parent_sort_key,
    derived_outcome_access,
    most_restrictive_outcome_access,
    verify_artifact_manifests,
)
from empirical_lawhood.runtime.execution_envelope import ExecutionEnvelopeEvent, ExecutionEnvelopeSpec, ExecutionEnvelopeState, ExecutionResourceEnvelopeState, ExecutionResourceEnvelopeSpec, ExecutionResourceTaskCellSpec, JitCellSignatureProjection, JitGraphSignatureManifest, ProgressHeartbeat, ProgressLivenessContract, RosterCapacityDecision, RunExecutionEnvelope, RunExecutionEnvelopeMachine, RunExecutionResourceEnvelopeMachine, RunExecutionResourceEnvelope
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationContext
from empirical_lawhood.runtime.input_access import planned_input_access_allowed
from empirical_lawhood.runtime.campaign_elapsed_budget import CampaignElapsedReservationPlan, DurableCampaignElapsedBudgetCoordinator
from empirical_lawhood.runtime.execution import validate_runner_resource_interface, ExecutorEnforcementCapability, ExecutionAssuranceProfile, DependencyReceiptBinding, EpochClock, ExternalInputResolver, OperationalAttempt, OperationalFailureClass, OperationalRepository, RunnerRegistry, RunnerResult, SchedulerResult, StreamedTaskOutput, StreamingOutputEmitter, TaskAttemptDisposition, TaskBlockKind, TaskBlockReason, TaskContext, TaskExecutor, TaskOutputPayload, TaskReceiptStore, TaskRunner, VerifiedArtifactInput, WorkerInputBinding, WorkerInputKind, WorkerInputPort, WorkerInputPortFactory, WorkerIsolationProfile, WorkerOutputPort
from empirical_lawhood.runtime.plans import BarrierKind, ProtocolExecutionPlan, ProtocolExecutionTask, ExecutionTask, ExecutionPlan, PlanLane
from empirical_lawhood.runtime.providers import CapabilityOutputSemanticContract
from empirical_lawhood.runtime.retry_amendment import LeaseExpiryRetryAmendment, RetainedCustodyCompletionAmendment
from empirical_lawhood.runtime.recovery import OperationalFailureDiagnostic, ProtocolRunRecoveryIndex, EnvelopeRunRecoveryIndex, RunRecoveryIndex, RunRecoveryReport

from .artifacts import (
    ArtifactInputLimitExceeded,
    GuardedExternalRoot,
    STREAM_CHUNK_BYTES,
)


class ExecutionErrorKind(StrEnum):
    """Stable service-boundary classification for expected execution failures."""

    VALIDATION = "VALIDATION"
    IDENTITY = "IDENTITY"
    IMMUTABLE_CONFLICT = "IMMUTABLE_CONFLICT"
    LIVE_CONCURRENCY = "LIVE_CONCURRENCY"
    CAPABILITY = "CAPABILITY"
    CUSTODY_STORAGE = "CUSTODY_STORAGE"
    TASK_FAILURE = "TASK_FAILURE"
    PREREQUISITE = "PREREQUISITE"
    INTERNAL = "INTERNAL"


class SchedulerConsistencyError(RuntimeError):
    pass


class LiveLeaseError(RuntimeError):
    pass


class InjectedSchedulerCrash(RuntimeError):
    pass


class ResourceLockUnavailable(RuntimeError):
    pass


class TaskProcessError(RuntimeError):
    """Parent-side worker failure retaining only bounded sanitized child detail."""

    def __init__(
        self,
        message: str,
        *,
        child_exception_type: str | None = None,
        child_message: str | None = None,
    ) -> None:
        super().__init__(message)
        self.sanitized_exception_type = _sanitize_exception_type(
            child_exception_type or type(self).__name__
        )
        self.sanitized_message = _sanitize_child_message(
            self.sanitized_exception_type,
            child_message,
        )


class ResourceAdmissionError(RuntimeError):
    """The frozen task cannot be enforced within the local compute boundary."""


class ProgressStalledError(RuntimeError):
    """Native work made no monotone progress for its source-qualified gap."""


class CampaignElapsedBoundaryReached(RuntimeError):
    """The absolute cumulative campaign-time boundary stopped in-flight work."""


def _sanitize_exception_type(value: str) -> str:
    sanitized = "".join(
        character
        for character in value
        if character.isascii() and (character.isalnum() or character in {"_", "."})
    )
    return (sanitized or "Exception")[:128]


def _sanitize_child_message(exception_type: str, raw_message: str | None) -> str:
    """Map untrusted child text to an allowlisted causal phrase; never persist it."""

    del raw_message
    lowered = exception_type.casefold()
    if "assert" in lowered:
        return "child process assertion failed"
    if "value" in lowered or "validation" in lowered:
        return "child process rejected a value"
    if "memory" in lowered:
        return "child process memory failure"
    if "timeout" in lowered:
        return "child process exceeded its frozen time budget"
    return "child process reported a bounded failure"


def operational_failure_diagnostic(
    error: BaseException,
    *,
    failure_class: OperationalFailureClass | None = None,
) -> OperationalFailureDiagnostic:
    """Classify one scheduler failure and retain only non-scientific safe fields."""

    resolved = failure_class
    if resolved is None:
        if isinstance(error, SchedulerConsistencyError):
            resolved = OperationalFailureClass.CONTRACT_REFUSAL
        elif isinstance(error, (ResourceAdmissionError, ArtifactInputLimitExceeded)):
            resolved = OperationalFailureClass.RESOURCE_REFUSAL
        elif isinstance(error, (TaskProcessError, TimeoutError, ProgressStalledError)):
            resolved = OperationalFailureClass.CHILD_PROCESS_FAILURE
        elif isinstance(error, OSError):
            resolved = OperationalFailureClass.TRANSIENT_INFRASTRUCTURE_FAILURE
        else:
            resolved = OperationalFailureClass.PROVIDER_RUNNER_DEFECT
    exception_type = (
        error.sanitized_exception_type
        if isinstance(error, TaskProcessError)
        else _sanitize_exception_type(type(error).__name__)
    )
    if isinstance(error, TaskProcessError):
        message = error.sanitized_message
    elif resolved is OperationalFailureClass.CONTRACT_REFUSAL:
        message = "execution contract refused the task result"
    elif resolved is OperationalFailureClass.RESOURCE_REFUSAL:
        message = "local resources refused the frozen task"
    elif resolved is OperationalFailureClass.PROVIDER_RUNNER_DEFECT:
        message = "provider or runner violated its deterministic contract"
    elif resolved is OperationalFailureClass.CHILD_PROCESS_FAILURE:
        message = "child process failed within its bounded execution contract"
    elif resolved is OperationalFailureClass.TRANSIENT_INFRASTRUCTURE_FAILURE:
        message = "transient infrastructure operation failed"
    elif resolved is OperationalFailureClass.RETRY_EXHAUSTION:
        message = "automatic retry budget was exhausted"
    else:
        message = "bounded recovery was obstructed"
    return OperationalFailureDiagnostic.create(
        resolved,
        sanitized_exception_type=exception_type,
        sanitized_message=message,
    )


# Bounded convenience surface for non-streaming runners. The measured
# maximum child allocation is 176,160,768 bytes; 256 MiB is the smallest
# ordinary power-of-two guard above it.  External artifact publication remains
# streamed and independently bounded.
MAX_IN_MEMORY_TASK_OUTPUT_BYTES = 256 * 1024**2
MAX_SCRATCH_SCAN_ENTRIES = 10_000


@dataclass(frozen=True, slots=True)
class ExecutionResourceCapacity:
    cpu_cores: int
    memory_bytes: int
    gpu_devices: int
    enforced_cpu_limit: bool
    enforced_address_space_limit: bool
    enforced_no_network: bool

    def __post_init__(self) -> None:
        if self.cpu_cores <= 0 or self.memory_bytes <= 0 or self.gpu_devices < 0:
            raise ValueError("local execution capacity is invalid")


@dataclass(frozen=True, slots=True)
class ExecutionResourceAdmission:
    """Read-only computability decision for one frozen execution plan."""

    admitted: bool
    required_cpu_cores: int
    required_memory_bytes: int
    required_gpu_devices: int
    required_scratch_bytes: int
    required_source_scan_bytes: int
    required_output_bytes: int
    available_cpu_cores: int
    available_memory_bytes: int
    available_gpu_devices: int
    available_scratch_bytes: int
    reason_codes: tuple[str, ...]
    assurance_codes: tuple[str, ...]


class LocalExecutionResourceAdmitter:
    """Preflight frozen resources against observed local enforcement capacity."""

    def __init__(self, capacity: ExecutionResourceCapacity | None = None) -> None:
        self.capacity = capacity or self.detect_capacity()

    @staticmethod
    def detect_capacity() -> ExecutionResourceCapacity:
        cpu_cores = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else (os.cpu_count() or 1)
        )
        memory_bytes = 0
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            physical_pages = int(os.sysconf("SC_PHYS_PAGES"))
            memory_bytes = page_size * physical_pages
        except (OSError, ValueError):
            pass
        if memory_bytes <= 0:
            memory_bytes = 1
        return ExecutionResourceCapacity(
            cpu_cores=cpu_cores,
            memory_bytes=memory_bytes,
            gpu_devices=0,
            enforced_cpu_limit=hasattr(resource, "RLIMIT_CPU"),
            enforced_address_space_limit=hasattr(resource, "RLIMIT_AS"),
            # No network namespace/cgroup implementation is composed yet.
            # Strict isolation stops; trusted-local records the limitation.
            enforced_no_network=False,
        )

    def assess(
        self,
        plan: ProtocolExecutionPlan,
        *,
        available_scratch_bytes: int,
        executor_enforces_resources: bool,
        executor_enforces_no_network: bool,
        assurance_profile: ExecutionAssuranceProfile = ExecutionAssuranceProfile.STRICT_ISOLATED,
    ) -> ExecutionResourceAdmission:
        if available_scratch_bytes < 0:
            raise ValueError("available scratch bytes must be nonnegative")
        budgets = tuple(task.capability.requested_resources for task in plan.tasks)
        required_cpu = max(value.cpu_cores for value in budgets)
        required_memory = max(value.memory_bytes for value in budgets)
        required_gpu = max(value.gpu_devices for value in budgets)
        # Resource admission derives scratch from each task's frozen output
        # allocation until a separately authored scratch field is versioned.
        required_scratch = max(value.output_bytes for value in budgets)
        required_scan = max(value.source_scan_bytes for value in budgets)
        required_output = max(value.output_bytes for value in budgets)
        reasons: list[str] = []
        assurances: list[str] = []
        if required_cpu > self.capacity.cpu_cores:
            reasons.append("CPU_CAPACITY_UNAVAILABLE")
        if required_memory > self.capacity.memory_bytes:
            reasons.append("MEMORY_CAPACITY_UNAVAILABLE")
        if required_gpu > self.capacity.gpu_devices:
            reasons.append("GPU_CAPACITY_UNAVAILABLE")
        if required_scratch > available_scratch_bytes:
            reasons.append("SCRATCH_CAPACITY_UNAVAILABLE")
        if assurance_profile is ExecutionAssuranceProfile.STRICT_ISOLATED:
            if not executor_enforces_resources:
                reasons.append("EXECUTOR_RESOURCE_ENFORCEMENT_UNAVAILABLE")
            if not self.capacity.enforced_cpu_limit:
                reasons.append("CPU_LIMIT_ENFORCEMENT_UNAVAILABLE")
            if not self.capacity.enforced_address_space_limit:
                reasons.append("MEMORY_LIMIT_ENFORCEMENT_UNAVAILABLE")
            if plan.lane is PlanLane.EXPLORATORY and (
                not self.capacity.enforced_no_network or not executor_enforces_no_network
            ):
                reasons.append("NO_NETWORK_ISOLATION_UNAVAILABLE")
        else:
            assurances.append("AGGREGATE_DESCENDANT_LIMITS_NOT_ENFORCED")
            if plan.lane is PlanLane.EXPLORATORY:
                assurances.extend(
                    (
                        "NETWORK_PERMISSION_NOT_REQUESTED",
                        "OS_NETWORK_ISOLATION_NOT_ENFORCED",
                    )
                )
        reason_codes = tuple(sorted(set(reasons)))
        return ExecutionResourceAdmission(
            admitted=not reason_codes,
            required_cpu_cores=required_cpu,
            required_memory_bytes=required_memory,
            required_gpu_devices=required_gpu,
            required_scratch_bytes=required_scratch,
            required_source_scan_bytes=required_scan,
            required_output_bytes=required_output,
            available_cpu_cores=self.capacity.cpu_cores,
            available_memory_bytes=self.capacity.memory_bytes,
            available_gpu_devices=self.capacity.gpu_devices,
            available_scratch_bytes=available_scratch_bytes,
            reason_codes=reason_codes,
            assurance_codes=tuple(sorted(set(assurances))),
        )


class FailureInjector(Protocol):
    def after_receipt_commit(
        self,
        run_id: str,
        task_id: str,
        attempt_id: str,
    ) -> None: ...


class RunRecoveryStore(Protocol):
    def freeze(self, index: ProtocolRunRecoveryIndex) -> None: ...

    def reconcile(
        self,
        index: ProtocolRunRecoveryIndex,
        plan: ProtocolExecutionPlan,
        repository: OperationalRepository,
        receipt_store: TaskReceiptStore,
        *,
        retry_amendment: LeaseExpiryRetryAmendment | None = None,
    ) -> RunRecoveryReport: ...

    def record_receipt(
        self,
        index: ProtocolRunRecoveryIndex,
        receipt: CanonicalTaskReceipt,
    ) -> None: ...

    def record_failure(
        self,
        index: ProtocolRunRecoveryIndex,
        *,
        task_id: str,
        attempt_id: str,
        reason_code: str,
        failure: OperationalFailureDiagnostic,
        blocked: bool = False,
    ) -> None: ...

    def record_not_attempted(
        self,
        index: ProtocolRunRecoveryIndex,
        *,
        task_id: str,
        reason: TaskBlockReason,
    ) -> None: ...

    def record_terminal(
        self,
        index: ProtocolRunRecoveryIndex,
        *,
        status: OperationalStatus,
        attempts: tuple[OperationalAttempt, ...],
        receipts: tuple[CanonicalTaskReceipt, ...],
    ) -> None: ...


class ExecutionEnvelopeEventStore(Protocol):
    """External immutable sink for each already-durable envelope transition."""

    def append_envelope_event(
        self,
        index: EnvelopeRunRecoveryIndex,
        event: ExecutionEnvelopeEvent,
        *,
        visibility_ceiling: VisibilityCeiling,
        outcome_access: OutcomeAccess,
    ) -> str: ...


class ExecutionEnvelopeEventClock(Protocol):
    def now_utc(self) -> str: ...


class NoFailureInjector:
    def after_receipt_commit(
        self,
        run_id: str,
        task_id: str,
        attempt_id: str,
    ) -> None:
        return None


class SystemEpochClock:
    def now(self) -> int:
        return int(time.time())


class SystemExecutionEnvelopeEventClock:
    def now_utc(self) -> str:
        return (
            datetime.now(timezone.utc)
            .isoformat(timespec="microseconds")
            .replace(
                "+00:00",
                "Z",
            )
        )


class DirectTaskExecutor:
    """Deterministic synthetic-test executor; it is not production enforcement."""

    enforcement_capability = ExecutorEnforcementCapability(
        capability_id="direct-task-executor",
        cpu_time_limit=False,
        address_space_limit=False,
        wall_time_limit=False,
        source_scan_limit=False,
        scratch_limit=False,
        output_limit=False,
        network_isolation=False,
    )

    def execute(
        self,
        runner: TaskRunner,
        context: TaskContext,
        *,
        timeout_seconds: int,
    ) -> RunnerResult:
        if timeout_seconds <= 0:
            raise ValueError("task timeout must be positive")
        return runner.execute(context)

    def execute_deadline_free(
        self,
        runner: TaskRunner,
        context: TaskContext,
        *,
        progress_contract: ProgressLivenessContract | None,
        progress_callback: Callable[[Decimal], None],
    ) -> RunnerResult:
        "Synthetic-only path; production stall enforcement is process based."

        if progress_contract is None:
            return runner.execute(context)
        execute_with_progress = getattr(runner, "execute_with_progress", None)
        if not callable(execute_with_progress):
            raise ResourceAdmissionError(
                "native deadline-free runner lacks a monotone progress interface"
            )
        result = execute_with_progress(
            context,
            _DirectProgressEmitter(progress_callback),
        )
        if not isinstance(result, RunnerResult):
            raise ResourceAdmissionError("progress runner returned an invalid result")
        return result

    def execute_deadline_free_until(
        self,
        runner: TaskRunner,
        context: TaskContext,
        *,
        progress_contract: ProgressLivenessContract | None,
        progress_callback: Callable[[Decimal], None],
        hard_remaining_seconds: Decimal,
    ) -> RunnerResult:
        """Synthetic boundary probe; production enforcement is process based."""

        if hard_remaining_seconds <= 0:
            raise CampaignElapsedBoundaryReached(context.task_id)
        started = time.monotonic()
        result = self.execute_deadline_free(
            runner,
            context,
            progress_contract=progress_contract,
            progress_callback=progress_callback,
        )
        if Decimal(str(time.monotonic() - started)) >= hard_remaining_seconds:
            raise CampaignElapsedBoundaryReached(context.task_id)
        return result


class _DirectProgressEmitter:
    def __init__(self, callback: Callable[[Decimal], None]) -> None:
        self._callback = callback
        self._last = Decimal(0)

    def advance(self, work_counter: Decimal) -> None:
        if not isinstance(work_counter, Decimal) or work_counter <= self._last:
            raise ResourceAdmissionError("runner progress must be a monotone Decimal counter")
        self._last = work_counter
        self._callback(work_counter)


@dataclass(frozen=True, slots=True)
class _ProcessMessage:
    result: RunnerResult | None
    error_type: str | None
    error_message: str | None
    error_kind: str | None


@dataclass(frozen=True, slots=True)
class _ProcessStarted:
    process_group_id: int


@dataclass(frozen=True, slots=True)
class _ProcessTaskFinished:
    """A chunk member has returned and released every input capability."""

    attempt_id: str


@dataclass(slots=True)
class _GroupedWorker:
    process: Any
    connection: Connection
    key: tuple[object, ...]
    tasks_used: int = 0


@dataclass(frozen=True, slots=True)
class _ProcessChunk:
    output_id: str
    payload: bytes


@dataclass(frozen=True, slots=True)
class _ProcessStreamComplete:
    checks: tuple[ReceiptCheck, ...]
    logical_content_sha256_by_output: tuple[tuple[str, str | None], ...] = ()


@dataclass(frozen=True, slots=True)
class _ProcessProgress:
    work_counter: Decimal


class _PipeOutputEmitter(StreamingOutputEmitter):
    def __init__(self, connection: Connection, context: TaskContext) -> None:
        self._connection = connection
        self._output_ids = {value.output_id for value in context.output_ports}
        self._maximum_bytes = context.resource_budget.output_bytes
        self._written = 0

    def write(self, output_id: str, chunk: bytes) -> None:
        if output_id not in self._output_ids:
            raise ResourceAdmissionError("streaming runner used an unknown output port")
        if not isinstance(chunk, bytes):
            raise ResourceAdmissionError("streaming runner chunks must be immutable bytes")
        if len(chunk) > STREAM_CHUNK_BYTES:
            raise ResourceAdmissionError("streaming runner chunk exceeds its byte limit")
        self._written += len(chunk)
        if self._written > self._maximum_bytes:
            raise ResourceAdmissionError("streaming runner exceeded its output-byte budget")
        self._connection.send(_ProcessChunk(output_id, chunk))


class _PipeProgressEmitter:
    def __init__(self, connection: Connection) -> None:
        self._connection = connection
        self._last = Decimal(0)

    def advance(self, work_counter: Decimal) -> None:
        if not isinstance(work_counter, Decimal) or work_counter <= self._last:
            raise ResourceAdmissionError("runner progress must be a monotone Decimal counter")
        self._last = work_counter
        self._connection.send(_ProcessProgress(work_counter))


class _PathOutputSource:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._consumed = False

    def chunks(self, maximum_chunk_bytes: int):  # type: ignore[no-untyped-def]
        if maximum_chunk_bytes <= 0:
            raise ValueError("output stream chunk limit must be positive")
        if self._consumed:
            raise ResourceAdmissionError("task output stream was already consumed")
        self._consumed = True
        with self._path.open("rb") as handle:
            while True:
                chunk = handle.read(maximum_chunk_bytes)
                if not chunk:
                    break
                yield chunk

    def close(self) -> None:
        if self._path.exists():
            self._path.unlink()


class _ScratchOutputCollector:
    def __init__(self, context: TaskContext, scratch_directory: Path) -> None:
        self._maximum_bytes = context.resource_budget.output_bytes
        self._written = 0
        self._paths: dict[str, Path] = {}
        self._handles: dict[str, BinaryIO] = {}
        self._digests: dict[str, Any] = {}
        self._sizes: dict[str, int] = {}
        for port in context.output_ports:
            path = scratch_directory / f".worker-output.{port.output_id}.partial"
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                stat.S_IRUSR | stat.S_IWUSR,
            )
            self._paths[port.output_id] = path
            self._handles[port.output_id] = os.fdopen(descriptor, "wb", closefd=True)
            self._digests[port.output_id] = hashlib.sha256()
            self._sizes[port.output_id] = 0

    def write(self, message: _ProcessChunk) -> None:
        handle = self._handles.get(message.output_id)
        if handle is None:
            raise ResourceAdmissionError("worker emitted an unknown output stream")
        if len(message.payload) > STREAM_CHUNK_BYTES:
            raise ResourceAdmissionError("worker output stream chunk exceeds its limit")
        self._written += len(message.payload)
        if self._written > self._maximum_bytes:
            raise ResourceAdmissionError("worker output stream exceeds its budget")
        handle.write(message.payload)
        self._digests[message.output_id].update(message.payload)
        self._sizes[message.output_id] += len(message.payload)

    def finish(
        self,
        logical_content_sha256_by_output: tuple[tuple[str, str | None], ...] = (),
    ) -> tuple[StreamedTaskOutput, ...]:
        logical_digests = dict(logical_content_sha256_by_output)
        if len(logical_digests) != len(logical_content_sha256_by_output) or set(
            logical_digests
        ).difference(self._handles):
            raise ResourceAdmissionError("streamed output identities are invalid")
        outputs = []
        for output_id in sorted(self._handles):
            handle = self._handles[output_id]
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            outputs.append(
                StreamedTaskOutput(
                    output_id=output_id,
                    size_bytes=self._sizes[output_id],
                    physical_sha256=self._digests[output_id].hexdigest(),
                    source=_PathOutputSource(self._paths[output_id]),
                    logical_content_sha256=logical_digests.get(output_id),
                )
            )
        self._handles.clear()
        return tuple(outputs)

    def abort(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()
        for path in self._paths.values():
            if path.exists():
                path.unlink()


def _apply_resource_limits(
    budget: ResourceBudget,
    *,
    cpu_affinity: tuple[int, ...] = (),
    deadline_free: bool = False,
) -> None:
    if not deadline_free:
        cpu_seconds = max(1, budget.cpu_cores * budget.wall_time_seconds)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    resource.setrlimit(
        resource.RLIMIT_AS,
        (budget.memory_bytes, budget.memory_bytes),
    )
    file_limit = max(1, budget.output_bytes)
    resource.setrlimit(resource.RLIMIT_FSIZE, (file_limit, file_limit))
    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        available = tuple(sorted(os.sched_getaffinity(0)))
        if cpu_affinity:
            if len(cpu_affinity) > budget.cpu_cores or not set(cpu_affinity).issubset(available):
                raise ResourceAdmissionError("worker CPU placement is unavailable")
            os.sched_setaffinity(0, set(cpu_affinity))
        elif len(available) > budget.cpu_cores:
            os.sched_setaffinity(0, set(available[: budget.cpu_cores]))


def _sanitize_worker_process(
    context: TaskContext,
    scratch_directory: Path,
    *,
    deadline_free: bool = False,
) -> None:
    from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

    if context.isolation_profile is WorkerIsolationProfile.EXPLORATION_NO_NETWORK:
        raise ResourceAdmissionError("enforceable exploration no-network isolation is unavailable")
    os.chdir(scratch_directory)
    os.environ.clear()
    os.environ.update(
        {
            "HOME": str(scratch_directory),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
            "TMPDIR": str(scratch_directory),
        }
    )
    # Fork and an existing forkserver can already have loaded native runtimes.
    # New spawn interpreters also receive these limits before their first import.
    threadpool_limits(limits=1)
    _apply_resource_limits(
        context.resource_budget,
        cpu_affinity=context.cpu_affinity,
        deadline_free=deadline_free,
    )


def _bounded_scratch_bytes(
    scratch_directory: Path,
    *,
    maximum_bytes: int,
    maximum_entries: int = MAX_SCRATCH_SCAN_ENTRIES,
) -> int:
    """Inspect scratch with explicit byte and entry stops.

    This is bounded defense in depth.  It is intentionally not represented as
    an aggregate scratch-quota capability.
    """

    if maximum_bytes < 0 or maximum_entries <= 0:
        raise ValueError("scratch inspection limits are invalid")
    pending = [scratch_directory]
    entry_count = 0
    total_bytes = 0
    try:
        while pending:
            directory = pending.pop()
            try:
                entries = os.scandir(directory)
            except FileNotFoundError:
                # A trusted runner may prune an intermediate directory between
                # bounded scans.  A vanished entry contributes no retained
                # scratch and is observed again if it reappears.
                continue
            with entries:
                for entry in entries:
                    entry_count += 1
                    if entry_count > maximum_entries:
                        raise ResourceAdmissionError(
                            "task scratch entry count exceeds the bounded inspection limit"
                        )
                    try:
                        entry_state = entry.stat(follow_symlinks=False)
                    except FileNotFoundError:
                        # QE and similar fixed-profile tools atomically replace
                        # and prune intermediates while the parent observes the
                        # task tree.  Treat only an exact vanished entry as a
                        # transient race; every retained entry remains bounded.
                        continue
                    if stat.S_ISLNK(entry_state.st_mode):
                        raise ResourceAdmissionError("task scratch contains a symbolic link")
                    if stat.S_ISDIR(entry_state.st_mode):
                        pending.append(Path(entry.path))
                        continue
                    if not stat.S_ISREG(entry_state.st_mode):
                        raise ResourceAdmissionError("task scratch contains a non-file entry")
                    total_bytes += entry_state.st_size
                    if total_bytes > maximum_bytes:
                        return total_bytes
    except ResourceAdmissionError:
        raise
    except OSError as error:
        raise ResourceAdmissionError("task scratch cannot be inspected safely") from error
    return total_bytes


def _terminate_process_group(
    process: multiprocessing.Process,
    *,
    process_group_id: int | None,
) -> None:
    """Terminate the worker and any descendants in its acknowledged group."""

    if process_group_id == process.pid and process_group_id is not None:
        try:
            os.killpg(process_group_id, signal.SIGTERM)
        except ProcessLookupError:
            pass
        join_worker_process(process, timeout=0.25)
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
        join_worker_process(process, timeout=0.25)
    if process.is_alive():
        process.terminate()
        join_worker_process(process, timeout=0.25)
    if process.is_alive() and hasattr(process, "kill"):
        process.kill()
        join_worker_process(process)


def _execute_in_child(
    runner: TaskRunner,
    context: TaskContext,
    connection: Connection,
    scratch_directory: Path,
    progress_contract: ProgressLivenessContract | None = None,
    deadline_free: bool = False,
    expected_source_root: str | None = None,
    grouped: bool = False,
) -> None:
    try:
        if not grouped:
            os.setsid()
        connection.send(_ProcessStarted(os.getpgrp()))
        if (
            expected_source_root is not None
            and str(Path(__file__).resolve().parents[2]) != expected_source_root
        ):
            raise ResourceAdmissionError("worker imported a different installed source tree")
        _sanitize_worker_process(
            context,
            scratch_directory,
            deadline_free=deadline_free,
        )
        execute_streaming = getattr(runner, "execute_streaming", None)
        execute_streaming_with_progress = getattr(
            runner,
            "execute_streaming_with_progress",
            None,
        )
        execute_with_progress = getattr(runner, "execute_with_progress", None)
        if progress_contract is not None and callable(execute_streaming_with_progress):
            checks = execute_streaming_with_progress(
                context,
                _PipeOutputEmitter(connection, context),
                _PipeProgressEmitter(connection),
            )
            if not isinstance(checks, tuple) or not all(
                isinstance(check, ReceiptCheck) for check in checks
            ):
                raise ResourceAdmissionError("streaming runner returned invalid receipt checks")
            connection.send(_ProcessStreamComplete(checks))
        elif progress_contract is not None and callable(execute_with_progress):
            result = execute_with_progress(
                context,
                _PipeProgressEmitter(connection),
            )
            _return_nonstreaming_result(connection, context, result)
        elif progress_contract is not None:
            raise ResourceAdmissionError(
                "native deadline-free runner lacks a monotone progress interface"
            )
        elif callable(execute_streaming):
            checks = execute_streaming(
                context,
                _PipeOutputEmitter(connection, context),
            )
            if not isinstance(checks, tuple) or not all(
                isinstance(check, ReceiptCheck) for check in checks
            ):
                raise ResourceAdmissionError("streaming runner returned invalid receipt checks")
            connection.send(_ProcessStreamComplete(checks))
        else:
            result = runner.execute(context)
            _return_nonstreaming_result(connection, context, result)
    except (ResourceAdmissionError, ArtifactInputLimitExceeded) as error:
        connection.send(
            _ProcessMessage(
                None,
                type(error).__name__,
                str(error),
                "RESOURCE",
            )
        )
    except BaseException as error:  # noqa: BLE001 - child errors cross a typed boundary
        connection.send(
            _ProcessMessage(
                None,
                type(error).__name__,
                str(error),
                "TASK",
            )
        )
    finally:
        if not grouped:
            connection.close()


def _execute_task_chunk(connection: Connection, expected_source_root: str) -> None:
    """At most two pure tasks; each keeps its own ports, scratch and result."""

    os.setsid()
    original_affinity = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None
    try:
        for _ in range(2):
            runner, context, scratch_directory = connection.recv()
            if original_affinity is not None:
                os.sched_setaffinity(0, original_affinity)
            _execute_in_child(
                runner,
                context,
                connection,
                scratch_directory,
                deadline_free=True,
                expected_source_root=expected_source_root,
                grouped=True,
            )
            for port in context.input_ports:
                port.close()
            connection.send(_ProcessTaskFinished(context.attempt_id))
            del runner, context, scratch_directory
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()


def _return_nonstreaming_result(
    connection: Connection,
    context: TaskContext,
    result: RunnerResult,
) -> None:
    "Return bounded in-memory outputs without sending a large pipe message.\n\n    An in-memory runner may construct an in-memory result within its declared task\n    budget, but external publication accepts large artifacts only through its\n    streamed path.  Convert any output larger than one IPC chunk into the same\n    parent-side guarded scratch stream used by native streaming runners.\n    "

    buffered_outputs = tuple(
        output for output in result.outputs if isinstance(output, TaskOutputPayload)
    )
    if len(buffered_outputs) != len(result.outputs):
        raise ResourceAdmissionError("non-streaming runner returned a streamed output handle")
    output_bytes = sum(len(output.payload) for output in buffered_outputs)
    if output_bytes > min(
        context.resource_budget.output_bytes,
        MAX_IN_MEMORY_TASK_OUTPUT_BYTES,
    ):
        raise ResourceAdmissionError("in-memory task output exceeds its bounded convenience limit")
    if any(len(output.payload) > STREAM_CHUNK_BYTES for output in buffered_outputs):
        emitter = _PipeOutputEmitter(connection, context)
        for output in buffered_outputs:
            for start in range(0, len(output.payload), STREAM_CHUNK_BYTES):
                emitter.write(output.output_id, output.payload[start : start + STREAM_CHUNK_BYTES])
        connection.send(
            _ProcessStreamComplete(
                checks=result.checks,
                logical_content_sha256_by_output=tuple(
                    (output.output_id, output.logical_content_sha256) for output in buffered_outputs
                ),
            )
        )
        return
    connection.send(_ProcessMessage(result, None, None, None))


class _ProgressPersistence:
    """Persist monotone progress without blocking native output on telemetry.

    Keep one callback in flight and the latest unpersisted counter. Every pipe
    sample is checked for monotonicity and liveness before submission; samples
    superseded here have not created canonical events. The final counter must
    cross the durable callback before a task result can be acknowledged.
    """

    def __init__(self, callback: Callable[[Decimal], None]) -> None:
        self._callback = callback
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="native-progress")
        self._lock = threading.Lock()
        self._pending: Decimal | None = None
        self._future: Future[None] | None = None
        self._running = False
        self._failure: BaseException | None = None

    def _persist(self, counter: Decimal) -> None:
        try:
            while True:
                self._callback(counter)
                with self._lock:
                    if self._pending is None:
                        self._running = False
                        return
                    counter, self._pending = self._pending, None
        except BaseException as error:
            with self._lock:
                self._failure = error
                self._pending = None
                self._running = False
            raise

    def check(self) -> None:
        with self._lock:
            if self._failure is not None:
                raise self._failure

    def submit(self, counter: Decimal) -> None:
        with self._lock:
            if self._failure is not None:
                raise self._failure
            if self._running:
                self._pending = counter
            else:
                self._running = True
                self._future = self._pool.submit(self._persist, counter)

    def drain(self) -> None:
        # The pipe owner calls drain after its final sample, so no producer can
        # submit while this future consumes the final pending counter.
        if self._future is not None:
            self._future.result()
        self.check()

    def close(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)


class LocalProcessExecutor:
    """Enforced trusted-local process boundary; explicitly not an OS sandbox."""

    enforcement_capability = ExecutorEnforcementCapability(
        capability_id="trusted-local-process",
        # RLIMIT_CPU and RLIMIT_AS are per-process and a trusted runner may
        # create descendants.  The timeout/process-group mechanics below are
        # defense in depth, not aggregate quota evidence.
        cpu_time_limit=False,
        address_space_limit=False,
        wall_time_limit=False,
        source_scan_limit=True,
        # Scratch is observed by a bounded scan, not an aggregate filesystem
        # quota, so it must not be advertised as enforced.
        scratch_limit=False,
        output_limit=True,
        network_isolation=False,
    )

    def __init__(
        self,
        *,
        scratch_root: GuardedExternalRoot | None = None,
        start_method: str = "spawn",
    ) -> None:
        if start_method not in multiprocessing.get_all_start_methods():
            raise ValueError("unsupported multiprocessing start method")
        self._context: Any = multiprocessing.get_context(start_method)
        self._scratch_root = scratch_root
        self._group_lock = threading.Lock()
        self._idle_workers: dict[tuple[object, ...], _GroupedWorker] = {}

    @staticmethod
    def _stop_group_worker(worker: _GroupedWorker) -> None:
        worker.connection.close()
        _terminate_process_group(worker.process, process_group_id=worker.process.pid)

    def close_run_workers(self, run_id: str | None = None) -> None:
        with self._group_lock:
            for key in tuple(self._idle_workers):
                if run_id is None or key[0] == run_id:
                    self._stop_group_worker(self._idle_workers.pop(key))

    def execute_deadline_free_in_group(
        self,
        runner: TaskRunner,
        context: TaskContext,
        *,
        execution_group_id: str,
        progress_contract: ProgressLivenessContract | None,
        progress_callback: Callable[[Decimal], None],
    ) -> RunnerResult:
        if (
            getattr(runner, "worker_chunk_limit", None) != 2
            or progress_contract is not None
            or context.scientific_adjudication_context is not None
            or context.isolation_profile is not WorkerIsolationProfile.TRUSTED_LOCAL
        ):
            raise ResourceAdmissionError("runner is not eligible for a bounded pure work chunk")
        key = (
            context.run_id,
            execution_group_id,
            runner.manifest.fingerprint(),
            context.config.fingerprint(),
            context.permissions,
            context.outcome_access,
            context.resource_budget.fingerprint(),
            context.isolation_profile,
        )
        with self._group_lock:
            worker = self._idle_workers.pop(key, None)
            # Idle memory never accumulates beside newly admitted workers.
            # A matching worker replaces exactly the same admitted budget.
            if worker is None:
                for idle in self._idle_workers.values():
                    self._stop_group_worker(idle)
                self._idle_workers.clear()
                parent, child = self._context.Pipe(duplex=True)
                process = self._context.Process(
                    target=_execute_task_chunk,
                    args=(child, str(Path(__file__).resolve().parents[2])),
                    name=f"empirical-lawhood-chunk-{context.task_id}",
                )
                try:
                    with numerical_spawn_environment():
                        process.start()
                except BaseException:
                    parent.close()
                    child.close()
                    raise
                child.close()
                worker = _GroupedWorker(process, parent, key)
        worker.tasks_used += 1
        try:
            result = self._execute_deadline_free(
                runner,
                context,
                progress_contract=None,
                progress_callback=progress_callback,
                worker=worker,
            )
        except BaseException:
            self._stop_group_worker(worker)
            raise
        if worker.tasks_used < 2:
            with self._group_lock:
                self._idle_workers[key] = worker
        return result

    def execute(
        self,
        runner: TaskRunner,
        context: TaskContext,
        *,
        timeout_seconds: int,
    ) -> RunnerResult:
        self.close_run_workers()
        if timeout_seconds <= 0:
            raise ValueError("task timeout must be positive")
        if timeout_seconds != context.resource_budget.wall_time_seconds:
            raise ResourceAdmissionError("worker timeout differs from frozen budget")
        if self._scratch_root is None:
            raise ResourceAdmissionError("scoped external scratch is not composed")
        scratch_directory = self._scratch_root.resolve(
            f"scratch/task-execution/{context.run_id}/{context.attempt_id}",
            for_write=True,
            operation_minimum_free_bytes=context.resource_budget.output_bytes,
        )
        if scratch_directory.exists():
            raise ResourceAdmissionError("task scratch identity already exists")
        scratch_directory.mkdir(parents=True)
        collector: _ScratchOutputCollector | None = None
        parent, child = self._context.Pipe(duplex=False)
        process = self._context.Process(
            target=_execute_in_child,
            args=(
                runner,
                context,
                child,
                scratch_directory,
                None,
                False,
                str(Path(__file__).resolve().parents[2]),
            ),
            name=f"empirical-lawhood-{context.task_id}",
        )
        message: object | None = None
        process_group_id: int | None = None
        try:
            with numerical_spawn_environment():
                process.start()
            child.close()
            deadline = time.monotonic() + timeout_seconds
            while message is None:
                if parent.poll(0.05):
                    try:
                        observed = parent.recv()
                    except EOFError:
                        break
                    if isinstance(observed, _ProcessStarted):
                        if process_group_id is not None or observed.process_group_id != process.pid:
                            raise TaskProcessError(
                                "worker process-group acknowledgement is invalid"
                            )
                        process_group_id = observed.process_group_id
                        continue
                    if isinstance(observed, _ProcessChunk):
                        if process_group_id is None:
                            raise TaskProcessError(
                                "worker emitted output before isolation acknowledgement"
                            )
                        if collector is None:
                            collector = _ScratchOutputCollector(context, scratch_directory)
                        collector.write(observed)
                        continue
                    if isinstance(observed, _ProcessStreamComplete) and collector is None:
                        collector = _ScratchOutputCollector(context, scratch_directory)
                    message = observed
                    break
                if time.monotonic() >= deadline:
                    _terminate_process_group(
                        process,
                        process_group_id=process_group_id,
                    )
                    raise TimeoutError(f"task process timed out: {context.task_id}")
                scratch_bytes = _bounded_scratch_bytes(
                    scratch_directory,
                    maximum_bytes=context.resource_budget.output_bytes,
                )
                if scratch_bytes > context.resource_budget.output_bytes:
                    _terminate_process_group(
                        process,
                        process_group_id=process_group_id,
                    )
                    raise ResourceAdmissionError("task exceeded its derived scratch budget")
                if not process.is_alive():
                    break
            if message is None and parent.poll():
                try:
                    message = parent.recv()
                except EOFError:
                    message = None
            if message is None:
                join_worker_process(process)
                raise TaskProcessError(
                    f"task process exited without a typed result (exitcode={process.exitcode})"
                )
            join_worker_process(process)
        finally:
            parent.close()
            if process.is_alive():
                _terminate_process_group(
                    process,
                    process_group_id=process_group_id,
                )
            elif process_group_id is not None:
                # A runner may leave descendants alive after returning.  The
                # acknowledged worker group is always dedicated to this task.
                try:
                    os.killpg(process_group_id, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    os.killpg(process_group_id, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if collector is not None and not isinstance(
                message,
                _ProcessStreamComplete,
            ):
                collector.abort()
        if isinstance(message, _ProcessStreamComplete):
            if collector is None:
                raise TaskProcessError("stream completion lacks an output collector")
            try:
                outputs = collector.finish(message.logical_content_sha256_by_output)
            except Exception:
                collector.abort()
                raise
            try:
                result = RunnerResult(outputs=outputs, checks=message.checks)
            except Exception:
                for output in outputs:
                    output.source.close()
                raise
        elif isinstance(message, _ProcessMessage):
            if message.result is None:
                if message.error_kind == "RESOURCE":
                    raise ResourceAdmissionError("task resource boundary is unavailable")
                raise TaskProcessError(
                    "task child process reported a typed failure",
                    child_exception_type=message.error_type,
                    child_message=message.error_message,
                )
            result = message.result
        else:
            raise TaskProcessError("task process returned an invalid message")
        if process.exitcode != 0:
            self._close_streamed_outputs(result)
            raise TaskProcessError(f"task process exited with code {process.exitcode}")
        scratch_bytes = _bounded_scratch_bytes(
            scratch_directory,
            maximum_bytes=context.resource_budget.output_bytes,
        )
        if scratch_bytes > context.resource_budget.output_bytes:
            self._close_streamed_outputs(result)
            raise ResourceAdmissionError("task exceeded its derived scratch budget")
        return result

    def execute_deadline_free(
        self,
        runner: TaskRunner,
        context: TaskContext,
        *,
        progress_contract: ProgressLivenessContract | None,
        progress_callback: Callable[[Decimal], None],
    ) -> RunnerResult:
        self.close_run_workers()
        return self._execute_deadline_free(
            runner,
            context,
            progress_contract=progress_contract,
            progress_callback=progress_callback,
        )

    def execute_deadline_free_until(
        self,
        runner: TaskRunner,
        context: TaskContext,
        *,
        progress_contract: ProgressLivenessContract | None,
        progress_callback: Callable[[Decimal], None],
        hard_remaining_seconds: Decimal,
    ) -> RunnerResult:
        "Enforce the companion campaign boundary with its exact execution-envelope prerequisites."

        if hard_remaining_seconds <= 0:
            raise CampaignElapsedBoundaryReached(context.task_id)
        self.close_run_workers()
        return self._execute_deadline_free(
            runner,
            context,
            progress_contract=progress_contract,
            progress_callback=progress_callback,
            campaign_hard_limit_seconds=hard_remaining_seconds,
        )

    def _execute_deadline_free(
        self,
        runner: TaskRunner,
        context: TaskContext,
        *,
        progress_contract: ProgressLivenessContract | None,
        progress_callback: Callable[[Decimal], None],
        worker: _GroupedWorker | None = None,
        campaign_hard_limit_seconds: Decimal | None = None,
    ) -> RunnerResult:
        "Execute without a total-duration stop; only qualified stalls terminate.\n\n        The ``ResourceBudget.wall_time_seconds`` field is deliberately\n        neither read nor translated into an RLIMIT on this deadline-free path.  Memory,\n        output, scratch and CPU-placement limits remain enforced.\n        "

        if self._scratch_root is None:
            raise ResourceAdmissionError("scoped external scratch is not composed")
        scratch_directory = self._scratch_root.resolve(
            f"scratch/task-execution/{context.run_id}/{context.attempt_id}",
            for_write=True,
            operation_minimum_free_bytes=context.resource_budget.output_bytes,
        )
        if scratch_directory.exists():
            raise ResourceAdmissionError("task scratch identity already exists")
        scratch_directory.mkdir(parents=True)
        collector: _ScratchOutputCollector | None = None
        if worker is None:
            parent, child = self._context.Pipe(duplex=False)
            process = self._context.Process(
                target=_execute_in_child,
                args=(
                    runner,
                    context,
                    child,
                    scratch_directory,
                    progress_contract,
                    True,
                    str(Path(__file__).resolve().parents[2]),
                ),
                name=f"empirical-lawhood-{context.task_id}",
            )
        else:
            parent, process = worker.connection, worker.process
        task_finished = False
        keep_alive = False
        message: object | None = None
        process_group_id: int | None = None
        last_progress_counter = Decimal(0)
        last_progress_monotonic = time.monotonic()
        no_progress_gap = (
            None
            if progress_contract is None
            else float(progress_contract.maximum_no_progress_gap_seconds)
        )
        campaign_deadline_monotonic = (
            None
            if campaign_hard_limit_seconds is None
            else time.monotonic() + float(campaign_hard_limit_seconds)
        )
        progress = None if progress_contract is None else _ProgressPersistence(progress_callback)
        progress_drained = progress is None
        try:
            if worker is None:
                with numerical_spawn_environment():
                    process.start()
                child.close()
            else:
                parent.send((runner, context, scratch_directory))
            while message is None or (worker is not None and not task_finished):
                if progress is not None:
                    progress.check()
                if parent.poll(0.05):
                    try:
                        observed = parent.recv()
                    except EOFError:
                        break
                    if isinstance(observed, _ProcessTaskFinished):
                        if (
                            worker is None
                            or message is None
                            or observed.attempt_id != context.attempt_id
                        ):
                            raise TaskProcessError("worker chunk completion is invalid")
                        task_finished = True
                        continue
                    if message is not None:
                        raise TaskProcessError("worker emitted data after its task result")
                    if isinstance(observed, _ProcessStarted):
                        if process_group_id is not None or observed.process_group_id != process.pid:
                            raise TaskProcessError(
                                "worker process-group acknowledgement is invalid"
                            )
                        process_group_id = observed.process_group_id
                        continue
                    if isinstance(observed, _ProcessProgress):
                        if process_group_id is None or progress_contract is None:
                            raise TaskProcessError(
                                "worker emitted progress outside its live progress contract"
                            )
                        if observed.work_counter <= last_progress_counter:
                            raise TaskProcessError("worker progress counter is non-monotone")
                        last_progress_counter = observed.work_counter
                        last_progress_monotonic = time.monotonic()
                        assert progress is not None
                        progress.submit(observed.work_counter)
                        continue
                    if isinstance(observed, _ProcessChunk):
                        if process_group_id is None:
                            raise TaskProcessError(
                                "worker emitted output before isolation acknowledgement"
                            )
                        if collector is None:
                            collector = _ScratchOutputCollector(context, scratch_directory)
                        collector.write(observed)
                        continue
                    if isinstance(observed, _ProcessStreamComplete) and collector is None:
                        collector = _ScratchOutputCollector(context, scratch_directory)
                    message = observed
                    if worker is None:
                        break
                    continue
                if (
                    no_progress_gap is not None
                    and time.monotonic() - last_progress_monotonic >= no_progress_gap
                ):
                    _terminate_process_group(
                        process,
                        process_group_id=process_group_id,
                    )
                    raise ProgressStalledError(
                        f"task process made no qualified progress: {context.task_id}"
                    )
                if (
                    campaign_deadline_monotonic is not None
                    and time.monotonic() >= campaign_deadline_monotonic
                ):
                    _terminate_process_group(
                        process,
                        process_group_id=process_group_id,
                    )
                    raise CampaignElapsedBoundaryReached(context.task_id)
                scratch_bytes = _bounded_scratch_bytes(
                    scratch_directory,
                    maximum_bytes=context.resource_budget.output_bytes,
                )
                if scratch_bytes > context.resource_budget.output_bytes:
                    _terminate_process_group(
                        process,
                        process_group_id=process_group_id,
                    )
                    raise ResourceAdmissionError("task exceeded its derived scratch budget")
                if not process.is_alive():
                    break
            if message is None and parent.poll():
                try:
                    message = parent.recv()
                except EOFError:
                    message = None
            if (
                campaign_deadline_monotonic is not None
                and time.monotonic() >= campaign_deadline_monotonic
            ):
                _terminate_process_group(
                    process,
                    process_group_id=process_group_id,
                )
                raise CampaignElapsedBoundaryReached(context.task_id)
            if message is None:
                join_worker_process(process)
                raise TaskProcessError(
                    f"task process exited without a typed result (exitcode={process.exitcode})"
                )
            if worker is not None and not task_finished:
                raise TaskProcessError("worker chunk exited before releasing task inputs")
            if progress is not None:
                # The final observed counter must be durable before a receipt.
                progress.drain()
                progress_drained = True
            keep_alive = worker is not None and worker.tasks_used < 2
            if not keep_alive:
                join_worker_process(process)
        finally:
            if progress is not None:
                progress.close()
            if not keep_alive:
                parent.close()
            if not keep_alive and process.is_alive():
                _terminate_process_group(
                    process,
                    process_group_id=process_group_id,
                )
            elif not keep_alive and process_group_id is not None:
                try:
                    os.killpg(process_group_id, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    os.killpg(process_group_id, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if collector is not None and (
                not isinstance(message, _ProcessStreamComplete)
                or (worker is not None and not task_finished)
                or not progress_drained
            ):
                collector.abort()
        if isinstance(message, _ProcessStreamComplete):
            if collector is None:
                raise TaskProcessError("stream completion lacks an output collector")
            try:
                outputs = collector.finish(message.logical_content_sha256_by_output)
            except Exception:
                collector.abort()
                raise
            try:
                result = RunnerResult(outputs=outputs, checks=message.checks)
            except Exception:
                for output in outputs:
                    output.source.close()
                raise
        elif isinstance(message, _ProcessMessage):
            if message.result is None:
                if message.error_kind == "RESOURCE":
                    raise ResourceAdmissionError("task resource boundary is unavailable")
                raise TaskProcessError(
                    "task child process reported a typed failure",
                    child_exception_type=message.error_type,
                    child_message=message.error_message,
                )
            result = message.result
        else:
            raise TaskProcessError("task process returned an invalid message")
        if not keep_alive and process.exitcode != 0:
            self._close_streamed_outputs(result)
            raise TaskProcessError(f"task process exited with code {process.exitcode}")
        scratch_bytes = _bounded_scratch_bytes(
            scratch_directory,
            maximum_bytes=context.resource_budget.output_bytes,
        )
        if scratch_bytes > context.resource_budget.output_bytes:
            self._close_streamed_outputs(result)
            raise ResourceAdmissionError("task exceeded its derived scratch budget")
        return result

    @staticmethod
    def _close_streamed_outputs(result: RunnerResult) -> None:
        for output in result.outputs:
            if isinstance(output, StreamedTaskOutput):
                output.source.close()


class ResourceLockManager:
    def __init__(self) -> None:
        self._held: set[str] = set()

    def acquire(self, lock_ids: tuple[str, ...]) -> bool:
        if self._held.intersection(lock_ids):
            return False
        self._held.update(lock_ids)
        return True

    def release(self, lock_ids: tuple[str, ...]) -> None:
        self._held.difference_update(lock_ids)

    def reserve_for_test(self, lock_id: str) -> None:
        validate_stable_id(lock_id, field_name="lock_id")
        self._held.add(lock_id)


@dataclass(frozen=True, slots=True)
class _TaskRecovery:
    completed: bool
    durable_blocked: bool
    spent_attempt_count: int
    next_attempt_number: int
    receipt: CanonicalTaskReceipt | None


@dataclass(frozen=True, slots=True)
class _TaskOutcome:
    disposition: TaskAttemptDisposition
    receipt: CanonicalTaskReceipt | None = None

    def __post_init__(self) -> None:
        if (self.disposition is TaskAttemptDisposition.SUCCEEDED) != (self.receipt is not None):
            raise ValueError("only a successful task outcome may carry a receipt")


@dataclass(frozen=True, slots=True)
class _PreparedTask:
    task: ProtocolExecutionTask
    dependency_outputs: tuple[VerifiedArtifactInput, ...]
    external_inputs: tuple[VerifiedArtifactInput, ...]
    dependency_receipts: tuple[DependencyReceiptBinding, ...]
    spent_attempt_count: int
    next_attempt_number: int


@dataclass(slots=True)
class _ParallelTaskState:
    prepared: _PreparedTask
    remaining_attempts: int
    next_attempt_number: int


@dataclass(frozen=True, slots=True)
class _ParallelAttempt:
    state: _ParallelTaskState
    attempt_id: str
    cpu_affinity: tuple[int, ...]


class _RetryableBlockRecorded(RuntimeError):
    """A running attempt was durably closed as a retryable block."""


class LocalScheduler:
    """One-writer local executor; workers receive only an immutable TaskContext."""

    def __init__(
        self,
        *,
        operational_repository: OperationalRepository,
        artifact_writer: ArtifactWriter,
        receipt_store: TaskReceiptStore,
        recovery_index: ProtocolRunRecoveryIndex,
        recovery_store: RunRecoveryStore,
        runner_registry: RunnerRegistry,
        implementation_commit: str,
        external_input_resolver: ExternalInputResolver | None = None,
        owner_id: str = "local-scheduler",
        lease_seconds: int = 60,
        clock: EpochClock | None = None,
        lock_manager: ResourceLockManager | None = None,
        failure_injector: FailureInjector | None = None,
        executor: TaskExecutor | None = None,
        input_port_factory: WorkerInputPortFactory | None = None,
        lane: PlanLane = PlanLane.PROSPECTIVE,
        minimum_free_bytes: int = 0,
        output_semantic_contracts: tuple[CapabilityOutputSemanticContract, ...] = (),
        adjudication_task_id: str | None = None,
        scientific_adjudication_context: ScientificAdjudicationContext | None = None,
        assurance_profile: ExecutionAssuranceProfile = ExecutionAssuranceProfile.STRICT_ISOLATED,
        assurance_codes: tuple[str, ...] = (),
        maximum_parallel_tasks: int = 1,
        parallel_resource_capacity: ExecutionResourceCapacity | None = None,
        execution_envelope_coordinator: DurableExecutionEnvelopeCoordinator | None = None,
        execution_envelope_event_store: ExecutionEnvelopeEventStore | None = None,
        execution_envelope_id: str | None = None,
        execution_resource_envelope_coordinator: (
            DurableExecutionResourceEnvelopeCoordinator | None
        ) = None,
        execution_resource_envelope_id: str | None = None,
        campaign_elapsed_budget_coordinator: (
            DurableCampaignElapsedBudgetCoordinator | None
        ) = None,
        campaign_elapsed_reservation_plan: CampaignElapsedReservationPlan | None = None,
        jit_graph_signature_manifest: JitGraphSignatureManifest | None = None,
        roster_capacity_decisions: tuple[RosterCapacityDecision, ...] = (),
        execution_envelope_event_clock: ExecutionEnvelopeEventClock | None = None,
        authorized_barriers: tuple[BarrierKind, ...] = (BarrierKind.REVEAL,),
        retry_amendment: LeaseExpiryRetryAmendment | None = None,
    ) -> None:
        validate_stable_id(owner_id, field_name="owner_id")
        if lease_seconds <= 0:
            raise ValueError("scheduler lease duration must be positive")
        if minimum_free_bytes < 0:
            raise ValueError("scheduler free-space floor must be nonnegative")
        if (adjudication_task_id is None) != (scientific_adjudication_context is None):
            raise ValueError("adjudication task and context must be composed together")
        if adjudication_task_id is not None:
            validate_stable_id(adjudication_task_id, field_name="adjudication_task_id")
        semantic_keys = tuple(contract.key for contract in output_semantic_contracts)
        if tuple(sorted(set(semantic_keys), key=str)) != semantic_keys:
            raise ValueError("output semantic contracts must be sorted and unique")
        if tuple(sorted(set(assurance_codes))) != assurance_codes:
            raise ValueError("execution assurance codes must be sorted and unique")
        if maximum_parallel_tasks <= 0:
            raise ValueError("maximum parallel tasks must be positive")
        if maximum_parallel_tasks > 1 and parallel_resource_capacity is None:
            raise ValueError("parallel execution requires an explicit resource capacity")
        envelope_values = (
            execution_envelope_coordinator,
            execution_envelope_event_store,
            execution_envelope_id,
        )
        if any(value is None for value in envelope_values) != all(
            value is None for value in envelope_values
        ):
            raise ValueError("execution-envelope scheduler composition is incomplete")
        if execution_envelope_id is not None:
            validate_stable_id(execution_envelope_id, field_name="execution_envelope_id")
        resource_envelope_values = (
            execution_resource_envelope_coordinator,
            execution_resource_envelope_id,
        )
        if any(value is None for value in resource_envelope_values) != all(
            value is None for value in resource_envelope_values
        ):
            raise ValueError("resource envelope scheduler composition is incomplete")
        if (
            jit_graph_signature_manifest is not None
            and execution_resource_envelope_coordinator is None
        ):
            raise ValueError("JIT manifest requires resource-envelope composition")
        if execution_resource_envelope_id is not None:
            validate_stable_id(
                execution_resource_envelope_id,
                field_name="execution_resource_envelope_id",
            )
        elapsed_values = (
            campaign_elapsed_budget_coordinator,
            campaign_elapsed_reservation_plan,
        )
        if any(value is None for value in elapsed_values) != all(
            value is None for value in elapsed_values
        ):
            raise ValueError("campaign elapsed-budget scheduler composition is incomplete")
        if campaign_elapsed_budget_coordinator is not None and (
            execution_resource_envelope_coordinator is None
            or campaign_elapsed_reservation_plan is None
            or campaign_elapsed_reservation_plan.budget
            != campaign_elapsed_budget_coordinator.budget_identity
        ):
            raise ValueError(
                "campaign elapsed budget requires its exact deadline-free resource envelope"
            )
        if execution_envelope_coordinator is not None and (
            execution_resource_envelope_coordinator is not None
        ):
            raise ValueError("deadline-bearing and deadline-free envelopes are exclusive")
        decision_groups = tuple(value.group_id for value in roster_capacity_decisions)
        if tuple(sorted(set(decision_groups))) != decision_groups:
            raise ValueError("roster capacity decisions must be sorted and group-unique")
        if roster_capacity_decisions and execution_resource_envelope_coordinator is None:
            raise ValueError("roster capacity decisions require resource envelope")
        if tuple(sorted(set(authorized_barriers), key=lambda value: value.value)) != (
            authorized_barriers
        ):
            raise ValueError("authorized barriers must be sorted and unique")
        available_cpu_ids = (
            tuple(sorted(os.sched_getaffinity(0)))
            if hasattr(os, "sched_getaffinity")
            else tuple(range(os.cpu_count() or 1))
        )
        if (
            maximum_parallel_tasks > 1
            and parallel_resource_capacity is not None
            and parallel_resource_capacity.cpu_cores > len(available_cpu_ids)
        ):
            raise ValueError("parallel CPU capacity exceeds the available affinity")
        self.operational_repository = operational_repository
        self.artifact_writer = artifact_writer
        self.receipt_store = receipt_store
        self.recovery_index = recovery_index
        self.recovery_store = recovery_store
        self.runner_registry = runner_registry
        self.implementation_commit = implementation_commit
        self.external_input_resolver = external_input_resolver
        self.owner_id = owner_id
        self.lease_seconds = lease_seconds
        self.clock = clock or SystemEpochClock()
        self.lock_manager = lock_manager or ResourceLockManager()
        self.failure_injector = failure_injector or NoFailureInjector()
        self.executor = executor or LocalProcessExecutor()
        self.input_port_factory = input_port_factory
        self.lane = lane
        self.minimum_free_bytes = minimum_free_bytes
        self.output_semantic_contracts = {
            contract.key: contract for contract in output_semantic_contracts
        }
        self.adjudication_task_id = adjudication_task_id
        self.scientific_adjudication_context = scientific_adjudication_context
        self.assurance_profile = assurance_profile
        self.assurance_codes = assurance_codes
        self.maximum_parallel_tasks = maximum_parallel_tasks
        self.parallel_resource_capacity = parallel_resource_capacity
        self.parallel_cpu_ids = (
            available_cpu_ids[: parallel_resource_capacity.cpu_cores]
            if maximum_parallel_tasks > 1 and parallel_resource_capacity is not None
            else ()
        )
        self.execution_envelope_coordinator = execution_envelope_coordinator
        self.execution_envelope_event_store = execution_envelope_event_store
        self.execution_envelope_id = execution_envelope_id
        self.execution_resource_envelope_coordinator = execution_resource_envelope_coordinator
        self.execution_resource_envelope_id = execution_resource_envelope_id
        self.campaign_elapsed_budget_coordinator = campaign_elapsed_budget_coordinator
        self.campaign_elapsed_reservation_plan = campaign_elapsed_reservation_plan
        self._campaign_elapsed_interval_id: str | None = None
        self.jit_graph_signature_manifest = jit_graph_signature_manifest
        self.roster_capacity_decisions = roster_capacity_decisions
        self._resource_envelope_transition_lock = threading.RLock()
        self.execution_envelope_event_clock = (
            execution_envelope_event_clock or SystemExecutionEnvelopeEventClock()
        )
        self.authorized_barriers = frozenset(authorized_barriers)
        self.retry_amendment = retry_amendment

    def _envelope_cell_id(self, task_id: str) -> str:
        coordinator = self.execution_envelope_coordinator
        envelope_id = self.execution_envelope_id
        if coordinator is None or envelope_id is None:
            raise SchedulerConsistencyError("execution envelope is not composed")
        matches = tuple(
            value.cell_id
            for value in coordinator.load(envelope_id).spec.cells
            if value.task_id == task_id
        )
        if len(matches) != 1:
            raise SchedulerConsistencyError(
                "execution-envelope task does not resolve to exactly one cell"
            )
        return matches[0]

    def _resource_envelope_cell_id(
        self, task_id: str, *, envelope: RunExecutionResourceEnvelope | None = None
    ) -> str:
        coordinator = self.execution_resource_envelope_coordinator
        envelope_id = self.execution_resource_envelope_id
        if coordinator is None or envelope_id is None:
            raise SchedulerConsistencyError("resource envelope is not composed")
        if envelope is None:
            envelope = coordinator.current(envelope_id)
        if envelope.envelope_id != envelope_id:
            raise SchedulerConsistencyError("resource-envelope lookup received a substitution")
        matches = tuple(
            value.cell_id for value in envelope.spec.task_cells if value.task_id == task_id
        )
        if len(matches) != 1:
            raise SchedulerConsistencyError(
                "resource-envelope task does not resolve to exactly one cell"
            )
        return matches[0]

    def _resource_progress_contract(
        self,
        task_id: str,
    ) -> ProgressLivenessContract | None:
        coordinator = self.execution_resource_envelope_coordinator
        envelope_id = self.execution_resource_envelope_id
        if coordinator is None or envelope_id is None:
            return None
        envelope = coordinator.current(envelope_id)
        cell_id = self._resource_envelope_cell_id(task_id, envelope=envelope)
        return envelope.spec.cell(cell_id).progress_liveness

    def _record_resource_progress(
        self,
        task_id: str,
        attempt_id: str,
        work_counter: Decimal,
    ) -> None:
        coordinator = self.execution_resource_envelope_coordinator
        envelope_id = self.execution_resource_envelope_id
        if coordinator is None or envelope_id is None:
            raise SchedulerConsistencyError("progress callback lacks resource envelope")
        with self._resource_envelope_transition_lock:
            envelope = coordinator.current(envelope_id)
            cell_id = self._resource_envelope_cell_id(task_id, envelope=envelope)
            contract = envelope.spec.cell(cell_id).progress_liveness
            if contract is None:
                raise SchedulerConsistencyError("non-native task emitted progress")
            sequence = len(envelope.events) + 1
            heartbeat = ProgressHeartbeat(
                heartbeat_id=f"heartbeat.{attempt_id}.{sequence:08d}",
                task_id=task_id,
                attempt_id=attempt_id,
                work_unit_id=contract.work_unit_id,
                work_counter=work_counter,
            )
            coordinator.heartbeat(
                envelope_id=envelope_id,
                event_id=f"resource-envelope-event.{heartbeat.heartbeat_id}",
                occurred_at_utc=self.execution_envelope_event_clock.now_utc(),
                cell_id=cell_id,
                heartbeat=heartbeat,
                current=envelope,
            )

    def _record_resource_progress_stalled(
        self,
        task_id: str,
        attempt_id: str,
    ) -> None:
        coordinator = self.execution_resource_envelope_coordinator
        envelope_id = self.execution_resource_envelope_id
        if coordinator is None or envelope_id is None:
            raise SchedulerConsistencyError("stalled progress lacks resource envelope")
        with self._resource_envelope_transition_lock:
            cell_id = self._resource_envelope_cell_id(task_id)
            coordinator.progress_stalled(
                envelope_id=envelope_id,
                event_id=f"resource-envelope-event.{attempt_id}.progress-stalled",
                occurred_at_utc=self.execution_envelope_event_clock.now_utc(),
                cell_id=cell_id,
                attempt_id=attempt_id,
            )

    def _retain_envelope_transition(
        self,
        envelope: RunExecutionEnvelope,
        *,
        visibility_ceiling: VisibilityCeiling = VisibilityCeiling.PROSPECTIVE,
        outcome_access: OutcomeAccess = OutcomeAccess.OUTCOME_BLIND,
    ) -> RunExecutionEnvelope:
        sink = self.execution_envelope_event_store
        if sink is None:
            return envelope
        if not isinstance(self.recovery_index, EnvelopeRunRecoveryIndex) or not envelope.events:
            raise SchedulerConsistencyError(
                "execution envelope requires execution envelope external recovery custody"
            )
        sink.append_envelope_event(
            self.recovery_index,
            envelope.events[-1],
            visibility_ceiling=visibility_ceiling,
            outcome_access=outcome_access,
        )
        return envelope

    def _sync_envelope_events(self) -> None:
        coordinator = self.execution_envelope_coordinator
        envelope_id = self.execution_envelope_id
        sink = self.execution_envelope_event_store
        if coordinator is None or envelope_id is None or sink is None:
            return
        if not isinstance(self.recovery_index, EnvelopeRunRecoveryIndex):
            raise SchedulerConsistencyError("execution envelope lacks execution envelope recovery authority")
        for envelope_event in coordinator.load(envelope_id).events:
            sink.append_envelope_event(
                self.recovery_index,
                envelope_event,
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
            )

    def _reserve_envelope_attempt(self, task_id: str, attempt_id: str) -> None:
        resource_coordinator = self.execution_resource_envelope_coordinator
        resource_envelope_id = self.execution_resource_envelope_id
        if resource_coordinator is not None and resource_envelope_id is not None:
            with self._resource_envelope_transition_lock:
                cell_id = self._resource_envelope_cell_id(task_id)
                cell = resource_coordinator.current(resource_envelope_id).cell(cell_id)
                if (
                    cell.state is ExecutionResourceEnvelopeState.RESERVED
                    and cell.current_attempt_id == attempt_id
                ):
                    return
                retry_reason = (
                    cell.last_operational_reason_code
                    if cell.state is ExecutionResourceEnvelopeState.RECEIPT_OBSERVED
                    else None
                )
                resource_coordinator.reserve(
                    envelope_id=resource_envelope_id,
                    event_id=f"resource-envelope-event.{attempt_id}.reservation",
                    occurred_at_utc=self.execution_envelope_event_clock.now_utc(),
                    cell_id=cell_id,
                    attempt_id=attempt_id,
                    retry_reason_code=retry_reason,
                )
            return
        coordinator = self.execution_envelope_coordinator
        envelope_id = self.execution_envelope_id
        if coordinator is None or envelope_id is None:
            return
        envelope_cell_id = self._envelope_cell_id(task_id)
        envelope_cell = coordinator.load(envelope_id).cell(envelope_cell_id)
        if (
            envelope_cell.state is ExecutionEnvelopeState.RESERVED
            and envelope_cell.current_attempt_id == attempt_id
        ):
            return
        envelope_retry_reason = (
            envelope_cell.last_operational_reason_code
            if envelope_cell.state is ExecutionEnvelopeState.RECEIPT_OBSERVED
            else None
        )
        updated_envelope = coordinator.reserve(
            envelope_id=envelope_id,
            event_id=f"envelope-event.{attempt_id}.reservation",
            occurred_at_utc=self.execution_envelope_event_clock.now_utc(),
            cell_id=envelope_cell_id,
            attempt_id=attempt_id,
            retry_reason_code=envelope_retry_reason,
        )
        self._retain_envelope_transition(updated_envelope)

    def _admit_campaign_elapsed_reservation(self, task_id: str, attempt_id: str) -> bool:
        coordinator = self.campaign_elapsed_budget_coordinator
        plan = self.campaign_elapsed_reservation_plan
        if coordinator is None or plan is None:
            return True
        if coordinator.current().exhausted:
            return False
        spec = self._worker_group_specs.get(task_id)
        if spec is None:
            raise SchedulerConsistencyError("elapsed reservation lacks its resource task cell")
        if not spec.native_simulator_launch:
            return True
        at_utc = self.execution_envelope_event_clock.now_utc()
        admission = coordinator.admit_reservation(
            admission_id=f"elapsed-admission.{attempt_id}",
            at_utc=at_utc,
            projected_active_seconds=plan.projected_seconds(task_id),
        )
        return admission.admitted

    def _launch_envelope_attempt(self, task_id: str, attempt_id: str) -> None:
        resource_coordinator = self.execution_resource_envelope_coordinator
        resource_envelope_id = self.execution_resource_envelope_id
        if resource_coordinator is not None and resource_envelope_id is not None:
            with self._resource_envelope_transition_lock:
                cell_id = self._resource_envelope_cell_id(task_id)
                envelope = resource_coordinator.current(resource_envelope_id)
                cell = envelope.cell(cell_id)
                if (
                    cell.state is ExecutionResourceEnvelopeState.LAUNCHED
                    and cell.current_attempt_id == attempt_id
                ):
                    return
                spec = envelope.spec.cell(cell_id)
                manifest = self.jit_graph_signature_manifest
                projection = None
                if spec.jit_cell_id is not None:
                    if manifest is None:
                        raise SchedulerConsistencyError(
                            "compiled launch lacks its issued JIT manifest"
                        )
                    matches = tuple(
                        value
                        for value in manifest.cell_projections
                        if value.cell_id == spec.jit_cell_id
                    )
                    if len(matches) != 1:
                        raise SchedulerConsistencyError(
                            "native launch cell lacks one manifested JIT projection"
                        )
                    projection = matches[0]
                resource_coordinator.launch(
                    envelope_id=resource_envelope_id,
                    event_id=f"resource-envelope-event.{attempt_id}.launch",
                    occurred_at_utc=self.execution_envelope_event_clock.now_utc(),
                    cell_id=cell_id,
                    attempt_id=attempt_id,
                    jit_projection=projection,
                    jit_graph_signature_manifest=(manifest if projection is not None else None),
                )
            return
        coordinator = self.execution_envelope_coordinator
        envelope_id = self.execution_envelope_id
        if coordinator is None or envelope_id is None:
            return
        envelope_cell_id = self._envelope_cell_id(task_id)
        envelope_cell = coordinator.load(envelope_id).cell(envelope_cell_id)
        if (
            envelope_cell.state is ExecutionEnvelopeState.LAUNCHED
            and envelope_cell.current_attempt_id == attempt_id
        ):
            return
        updated_envelope = coordinator.launch(
            envelope_id=envelope_id,
            event_id=f"envelope-event.{attempt_id}.launch",
            occurred_at_utc=self.execution_envelope_event_clock.now_utc(),
            cell_id=envelope_cell_id,
            attempt_id=attempt_id,
        )
        self._retain_envelope_transition(updated_envelope)

    def _record_envelope_operational_failure(
        self,
        task_id: str,
        attempt_id: str,
        *,
        reason_code: str,
    ) -> None:
        resource_coordinator = self.execution_resource_envelope_coordinator
        resource_envelope_id = self.execution_resource_envelope_id
        if resource_coordinator is not None and resource_envelope_id is not None:
            with self._resource_envelope_transition_lock:
                cell_id = self._resource_envelope_cell_id(task_id)
                cell = resource_coordinator.current(resource_envelope_id).cell(cell_id)
                if cell.state is ExecutionResourceEnvelopeState.RECEIPT_OBSERVED:
                    return
                resource_coordinator.observe_operational_failure(
                    envelope_id=resource_envelope_id,
                    event_id=f"resource-envelope-event.{attempt_id}.operational-failure",
                    occurred_at_utc=self.execution_envelope_event_clock.now_utc(),
                    cell_id=cell_id,
                    attempt_id=attempt_id,
                    reason_code=reason_code,
                )
            return
        coordinator = self.execution_envelope_coordinator
        envelope_id = self.execution_envelope_id
        if coordinator is None or envelope_id is None:
            return
        envelope_cell_id = self._envelope_cell_id(task_id)
        envelope_cell = coordinator.load(envelope_id).cell(envelope_cell_id)
        if envelope_cell.state is ExecutionEnvelopeState.RECEIPT_OBSERVED:
            return
        updated_envelope = coordinator.observe_operational_failure(
            envelope_id=envelope_id,
            event_id=f"envelope-event.{attempt_id}.operational-failure",
            occurred_at_utc=self.execution_envelope_event_clock.now_utc(),
            cell_id=envelope_cell_id,
            attempt_id=attempt_id,
            reason_code=reason_code,
        )
        self._retain_envelope_transition(updated_envelope)

    def _stop_envelope_task(self, task_id: str, *, reason_code: str) -> None:
        resource_coordinator = self.execution_resource_envelope_coordinator
        resource_envelope_id = self.execution_resource_envelope_id
        if resource_coordinator is not None and resource_envelope_id is not None:
            with self._resource_envelope_transition_lock:
                cell_id = self._resource_envelope_cell_id(task_id)
                cell = resource_coordinator.current(resource_envelope_id).cell(cell_id)
                if cell.state is ExecutionResourceEnvelopeState.RESOURCE_STOP:
                    return
                resource_coordinator.stop(
                    envelope_id=resource_envelope_id,
                    event_id=(
                        f"resource-envelope-event.{resource_envelope_id}.{cell_id}.resource-stop"
                    ),
                    occurred_at_utc=self.execution_envelope_event_clock.now_utc(),
                    cell_id=cell_id,
                    reason_code=reason_code,
                )
            return
        coordinator = self.execution_envelope_coordinator
        envelope_id = self.execution_envelope_id
        if coordinator is None or envelope_id is None:
            return
        envelope_cell_id = self._envelope_cell_id(task_id)
        envelope_cell = coordinator.load(envelope_id).cell(envelope_cell_id)
        if envelope_cell.state is ExecutionEnvelopeState.RESOURCE_STOP:
            return
        updated_envelope = coordinator.stop(
            envelope_id=envelope_id,
            event_id=f"envelope-event.{envelope_id}.{envelope_cell_id}.resource-stop",
            occurred_at_utc=self.execution_envelope_event_clock.now_utc(),
            cell_id=envelope_cell_id,
            reason_code=reason_code,
        )
        self._retain_envelope_transition(updated_envelope)

    def _unknown_envelope_completion(
        self,
        task_id: str,
        attempt_id: str,
        *,
        reason_code: str = "UNKNOWN_COMPLETION_AFTER_EXPIRED_LEASE",
    ) -> None:
        resource_coordinator = self.execution_resource_envelope_coordinator
        resource_envelope_id = self.execution_resource_envelope_id
        if resource_coordinator is not None and resource_envelope_id is not None:
            with self._resource_envelope_transition_lock:
                cell_id = self._resource_envelope_cell_id(task_id)
                cell = resource_coordinator.current(resource_envelope_id).cell(cell_id)
                if cell.state is ExecutionResourceEnvelopeState.UNKNOWN_COMPLETION:
                    return
                resource_coordinator.unknown_completion(
                    envelope_id=resource_envelope_id,
                    event_id=f"resource-envelope-event.{attempt_id}.unknown-completion",
                    occurred_at_utc=self.execution_envelope_event_clock.now_utc(),
                    cell_id=cell_id,
                    reason_code=reason_code,
                )
            return
        coordinator = self.execution_envelope_coordinator
        envelope_id = self.execution_envelope_id
        if coordinator is None or envelope_id is None:
            return
        envelope_cell_id = self._envelope_cell_id(task_id)
        envelope_cell = coordinator.load(envelope_id).cell(envelope_cell_id)
        if envelope_cell.state is ExecutionEnvelopeState.UNKNOWN_COMPLETION:
            return
        updated_envelope = coordinator.unknown_completion(
            envelope_id=envelope_id,
            event_id=f"envelope-event.{attempt_id}.unknown-completion",
            occurred_at_utc=self.execution_envelope_event_clock.now_utc(),
            cell_id=envelope_cell_id,
            reason_code=reason_code,
        )
        self._retain_envelope_transition(updated_envelope)

    def _record_campaign_elapsed_boundary(self, task_id: str, attempt_id: str) -> None:
        """Durably retain an interrupted launch as unknown before exhausting time."""

        coordinator = self.campaign_elapsed_budget_coordinator
        if coordinator is None:
            raise SchedulerConsistencyError("campaign boundary lacks its durable coordinator")
        self._unknown_envelope_completion(
            task_id,
            attempt_id,
            reason_code="UNKNOWN_COMPLETION_AT_CAMPAIGN_ELAPSED_BOUNDARY",
        )
        failure = operational_failure_diagnostic(
            CampaignElapsedBoundaryReached(task_id),
            failure_class=OperationalFailureClass.RECOVERY_OBSTRUCTION,
        )
        self.recovery_store.record_failure(
            self.recovery_index,
            task_id=task_id,
            attempt_id=attempt_id,
            reason_code=TaskBlockReason.UNKNOWN_COMPLETION.value,
            failure=failure,
            blocked=True,
        )
        self.operational_repository.block_attempt(
            attempt_id,
            TaskBlockReason.UNKNOWN_COMPLETION,
        )
        coordinator.stop_at_boundary(at_utc=self.execution_envelope_event_clock.now_utc())
        self._campaign_elapsed_interval_id = None

    def _finalize_envelope_receipt(
        self,
        task_id: str,
        attempt_id: str,
        receipt: CanonicalTaskReceipt,
        *,
        recovery: bool,
    ) -> None:
        resource_coordinator = self.execution_resource_envelope_coordinator
        resource_envelope_id = self.execution_resource_envelope_id
        if resource_coordinator is not None and resource_envelope_id is not None:
            with self._resource_envelope_transition_lock:
                cell_id = self._resource_envelope_cell_id(task_id)
                receipt_identity = ObjectIdentity.from_record(receipt.receipt_id, receipt)
                current = resource_coordinator.current(resource_envelope_id)
                cell = current.cell(cell_id)
                if recovery and cell.state is ExecutionResourceEnvelopeState.LAUNCHED:
                    # The canonical receipt binds the already-written output
                    # materializations and therefore serves as the immutable
                    # artifact/recovery proof without another launch or token.
                    current = resource_coordinator.recover_artifact_before_receipt(
                        envelope_id=resource_envelope_id,
                        event_id=(f"resource-envelope-event.{attempt_id}.artifact-before-receipt"),
                        occurred_at_utc=self.execution_envelope_event_clock.now_utc(),
                        cell_id=cell_id,
                        attempt_id=attempt_id,
                        receipt=receipt_identity,
                        artifact=receipt_identity,
                        recovery_proof=receipt_identity,
                    )
                    cell = current.cell(cell_id)
                elif cell.state is ExecutionResourceEnvelopeState.LAUNCHED:
                    current = resource_coordinator.observe_receipt(
                        envelope_id=resource_envelope_id,
                        event_id=f"resource-envelope-event.{attempt_id}.receipt",
                        occurred_at_utc=self.execution_envelope_event_clock.now_utc(),
                        cell_id=cell_id,
                        attempt_id=attempt_id,
                        receipt=receipt_identity,
                        valid_receipt=True,
                        scientific_terminal=True,
                    )
                    cell = current.cell(cell_id)
                if (
                    recovery
                    and cell.state is ExecutionResourceEnvelopeState.RECEIPT_OBSERVED
                    and cell.first_valid_receipt == receipt_identity
                ):
                    current = resource_coordinator.begin_recovery_publication(
                        envelope_id=resource_envelope_id,
                        event_id=(
                            f"resource-envelope-event.{attempt_id}.recovery-publication-start"
                        ),
                        occurred_at_utc=self.execution_envelope_event_clock.now_utc(),
                        cell_id=cell_id,
                        artifact=receipt_identity,
                        recovery_proof=receipt_identity,
                    )
                    cell = current.cell(cell_id)
                if (
                    not recovery
                    and cell.state is ExecutionResourceEnvelopeState.RECEIPT_OBSERVED
                    and cell.first_valid_receipt == receipt_identity
                ):
                    current = resource_coordinator.publish(
                        envelope_id=resource_envelope_id,
                        event_id=f"resource-envelope-event.{attempt_id}.publication",
                        occurred_at_utc=self.execution_envelope_event_clock.now_utc(),
                        cell_id=cell_id,
                        artifact=receipt_identity,
                    )
                    cell = current.cell(cell_id)
                elif (
                    recovery
                    and cell.state is ExecutionResourceEnvelopeState.RECOVERY_PUBLICATION
                    and cell.first_valid_receipt == receipt_identity
                ):
                    current = resource_coordinator.publish(
                        envelope_id=resource_envelope_id,
                        event_id=f"resource-envelope-event.{attempt_id}.publication",
                        occurred_at_utc=self.execution_envelope_event_clock.now_utc(),
                        cell_id=cell_id,
                        recovery=True,
                    )
                    cell = current.cell(cell_id)
                if (
                    cell.state is ExecutionResourceEnvelopeState.PUBLISHED
                    and cell.published_artifact == receipt_identity
                ):
                    current = resource_coordinator.terminal(
                        envelope_id=resource_envelope_id,
                        event_id=f"resource-envelope-event.{attempt_id}.terminal",
                        occurred_at_utc=self.execution_envelope_event_clock.now_utc(),
                        cell_id=cell_id,
                    )
                    cell = current.cell(cell_id)
                if (
                    cell.state is not ExecutionResourceEnvelopeState.TERMINAL
                    or cell.first_valid_receipt != receipt_identity
                    or cell.published_artifact != receipt_identity
                ):
                    raise SchedulerConsistencyError(
                        "resource envelope differs from its first valid task receipt"
                    )
                return
        coordinator = self.execution_envelope_coordinator
        envelope_id = self.execution_envelope_id
        if coordinator is None or envelope_id is None:
            return
        envelope_cell_id = self._envelope_cell_id(task_id)
        envelope_receipt_identity = ObjectIdentity.from_record(receipt.receipt_id, receipt)
        envelope_visibility = VisibilityCeiling.most_restrictive(
            VisibilityCeiling.PROSPECTIVE,
            *(value.visibility_ceiling for value in receipt.output_logical_artifacts),
        )
        envelope_outcome_access = most_restrictive_outcome_access(
            OutcomeAccess.OUTCOME_BLIND,
            *(value.outcome_access for value in receipt.output_logical_artifacts),
        )
        current_envelope = coordinator.load(envelope_id)
        envelope_cell = current_envelope.cell(envelope_cell_id)
        if envelope_cell.state is ExecutionEnvelopeState.LAUNCHED:
            current_envelope = coordinator.observe_receipt(
                envelope_id=envelope_id,
                event_id=f"envelope-event.{attempt_id}.receipt",
                occurred_at_utc=self.execution_envelope_event_clock.now_utc(),
                cell_id=envelope_cell_id,
                attempt_id=attempt_id,
                receipt=envelope_receipt_identity,
                valid_receipt=True,
                scientific_terminal=True,
            )
            self._retain_envelope_transition(
                current_envelope,
                visibility_ceiling=envelope_visibility,
                outcome_access=envelope_outcome_access,
            )
            envelope_cell = current_envelope.cell(envelope_cell_id)
        if (
            envelope_cell.state is ExecutionEnvelopeState.RECEIPT_OBSERVED
            and envelope_cell.first_valid_receipt == envelope_receipt_identity
        ):
            if recovery:
                current_envelope = coordinator.begin_recovery_publication(
                    envelope_id=envelope_id,
                    event_id=f"envelope-event.{attempt_id}.recovery-publication-start",
                    occurred_at_utc=self.execution_envelope_event_clock.now_utc(),
                    cell_id=envelope_cell_id,
                )
                self._retain_envelope_transition(
                    current_envelope,
                    visibility_ceiling=envelope_visibility,
                    outcome_access=envelope_outcome_access,
                )
            current_envelope = coordinator.publish(
                envelope_id=envelope_id,
                event_id=f"envelope-event.{attempt_id}.publication",
                occurred_at_utc=self.execution_envelope_event_clock.now_utc(),
                cell_id=envelope_cell_id,
                artifact=envelope_receipt_identity,
                recovery=recovery,
            )
            self._retain_envelope_transition(
                current_envelope,
                visibility_ceiling=envelope_visibility,
                outcome_access=envelope_outcome_access,
            )
            envelope_cell = current_envelope.cell(envelope_cell_id)
        if (
            envelope_cell.state is ExecutionEnvelopeState.PUBLISHED
            and envelope_cell.published_artifact == envelope_receipt_identity
        ):
            current_envelope = coordinator.terminal(
                envelope_id=envelope_id,
                event_id=f"envelope-event.{attempt_id}.terminal",
                occurred_at_utc=self.execution_envelope_event_clock.now_utc(),
                cell_id=envelope_cell_id,
            )
            self._retain_envelope_transition(
                current_envelope,
                visibility_ceiling=envelope_visibility,
                outcome_access=envelope_outcome_access,
            )
            envelope_cell = current_envelope.cell(envelope_cell_id)
        if (
            envelope_cell.state is not ExecutionEnvelopeState.TERMINAL
            or envelope_cell.first_valid_receipt != envelope_receipt_identity
            or envelope_cell.published_artifact != envelope_receipt_identity
        ):
            raise SchedulerConsistencyError(
                "execution envelope differs from its first valid task receipt"
            )

    def execute(self, run_id: str, plan: ProtocolExecutionPlan) -> SchedulerResult:
        self._worker_group_ids: dict[str, str | None] = {}
        self._worker_group_specs: dict[str, ExecutionResourceTaskCellSpec] = {}
        self._begin_campaign_elapsed_interval(run_id)
        try:
            return self._execute(run_id, plan)
        finally:
            self._close_campaign_elapsed_interval()
            close_workers = getattr(self.executor, "close_run_workers", None)
            if callable(close_workers):
                close_workers(run_id)

    def _begin_campaign_elapsed_interval(self, run_id: str) -> None:
        coordinator = self.campaign_elapsed_budget_coordinator
        plan = self.campaign_elapsed_reservation_plan
        if coordinator is None or plan is None:
            return
        now = self.execution_envelope_event_clock.now_utc()
        current = coordinator.ensure_initialized()
        if current.open_interval_id is not None:
            current = coordinator.reconcile_open_interval(observed_close_utc=now)
        if (
            current.exhausted
            or current.active_seconds_at(now) >= coordinator.budget.hard_ceiling_seconds
        ):
            if not current.exhausted:
                coordinator.stop_at_boundary(at_utc=now)
            return
        current = coordinator.bind_envelope(
            binding_id=plan.envelope_binding_id,
            stage_id=plan.stage_id,
            governed_envelope=plan.governed_envelope,
            at_utc=now,
        )
        kind = "RECOVERY" if self.operational_repository.attempts(run_id) else "LAUNCH"
        interval_id = (
            f"{coordinator.ledger_id}.{plan.stage_id}.{kind.lower()}."
            f"{len(current.closed_intervals) + 1:03d}"
        )
        coordinator.open_interval(interval_id=interval_id, kind=kind, at_utc=now)
        self._campaign_elapsed_interval_id = interval_id

    def _close_campaign_elapsed_interval(self) -> None:
        coordinator = self.campaign_elapsed_budget_coordinator
        interval_id = self._campaign_elapsed_interval_id
        self._campaign_elapsed_interval_id = None
        if coordinator is None or interval_id is None:
            return
        current = coordinator.current()
        if current.open_interval_id == interval_id:
            coordinator.close_interval(at_utc=self.execution_envelope_event_clock.now_utc())
        elif current.open_interval_id is not None:
            raise SchedulerConsistencyError("another campaign elapsed interval replaced this run")

    def _execute(self, run_id: str, plan: ProtocolExecutionPlan) -> SchedulerResult:
        validate_stable_id(run_id, field_name="run_id")
        if run_id != plan.source_plan.object_id:
            raise SchedulerConsistencyError("run ID differs from the frozen source plan")
        if plan.implementation_commit != self.implementation_commit:
            raise SchedulerConsistencyError("execution commit differs from frozen plan")
        if plan.registry_sha256 != self.runner_registry.registry_sha256:
            raise SchedulerConsistencyError("runner registry differs from frozen plan")
        if plan.lane is not self.lane:
            raise SchedulerConsistencyError("scheduler lane differs from frozen plan")
        if self.scientific_adjudication_context is not None and (
            self.scientific_adjudication_context.execution_plan
            != ObjectIdentity.from_record(plan.execution_plan_id, plan)
        ):
            raise SchedulerConsistencyError(
                "scientific adjudication context binds another execution plan"
            )
        try:
            self.recovery_index.validate_plan(plan, retry_amendment=self.retry_amendment)
        except ValueError as error:
            raise SchedulerConsistencyError(
                "run recovery index differs from the frozen plan"
            ) from error
        if self.recovery_index.run_id != run_id:
            raise SchedulerConsistencyError("run recovery index binds another run")
        if self.execution_resource_envelope_coordinator is not None:
            if not isinstance(self.recovery_index, RunRecoveryIndex) or not isinstance(
                plan,
                ExecutionPlan,
            ):
                raise SchedulerConsistencyError(
                    "resource envelope requires a resource execution plan and recovery index"
                )
            assert self.execution_resource_envelope_id is not None
            resource_envelope = self.execution_resource_envelope_coordinator.current(
                self.execution_resource_envelope_id
            )
            if resource_envelope.execution_plan != ObjectIdentity.from_record(
                plan.execution_plan_id,
                plan,
            ):
                raise SchedulerConsistencyError("resource envelope binds another execution plan")
            self._worker_group_specs = {
                value.task_id: value for value in resource_envelope.spec.task_cells
            }
            if self.campaign_elapsed_reservation_plan is not None:
                self.campaign_elapsed_reservation_plan.validate_task_census(resource_envelope.spec)
                elapsed = self.campaign_elapsed_budget_coordinator
                assert elapsed is not None
                ledger = elapsed.current()
                if (
                    not ledger.envelope_bindings
                    or ledger.envelope_bindings[-1].binding_id
                    != self.campaign_elapsed_reservation_plan.envelope_binding_id
                    or ledger.envelope_bindings[-1].governed_envelope
                    != self.campaign_elapsed_reservation_plan.governed_envelope
                ):
                    raise SchedulerConsistencyError(
                        "campaign elapsed ledger lacks the current execution envelope"
                    )
            if (
                None
                if self.jit_graph_signature_manifest is None
                else ObjectIdentity.from_record(
                    self.jit_graph_signature_manifest.manifest_id,
                    self.jit_graph_signature_manifest,
                )
            ) != plan.jit_graph_signature_manifest:
                raise SchedulerConsistencyError(
                    "scheduler JIT manifest differs from the resource execution plan"
                )
            existing_groups = {value.group_id for value in resource_envelope.roster_decisions}
            for decision in self.roster_capacity_decisions:
                if decision.group_id in existing_groups:
                    continue
                resource_envelope = (
                    self.execution_resource_envelope_coordinator.select_roster_branch(
                        envelope_id=self.execution_resource_envelope_id,
                        event_id=f"resource-envelope-event.{decision.decision_id}",
                        occurred_at_utc=self.execution_envelope_event_clock.now_utc(),
                        decision=decision,
                    )
                )
                existing_groups.add(decision.group_id)
            required_groups = {
                value.group_id for value in resource_envelope.spec.roster_capacity_branches
            }
            if existing_groups != required_groups:
                raise SchedulerConsistencyError(
                    "resource envelope lacks an exact outcome-blind roster decision"
                )
        elif isinstance(plan, ExecutionPlan):
            raise SchedulerConsistencyError(
                "resource execution plan requires deadline-free resource-envelope dispatch"
            )
        if self.execution_envelope_coordinator is not None:
            if not isinstance(self.recovery_index, EnvelopeRunRecoveryIndex):
                raise SchedulerConsistencyError("execution envelope requires an execution envelope recovery index")
            assert self.execution_envelope_id is not None
            envelope = self.execution_envelope_coordinator.load(self.execution_envelope_id)
            if envelope.execution_plan != ObjectIdentity.from_record(
                plan.execution_plan_id,
                plan,
            ):
                raise SchedulerConsistencyError("execution envelope binds another execution plan")
        self.recovery_store.freeze(self.recovery_index)
        self._sync_envelope_events()
        recovery = self.recovery_store.reconcile(
            self.recovery_index,
            plan,
            self.operational_repository,
            self.receipt_store,
            retry_amendment=self.retry_amendment,
        )
        if recovery.terminal_status is not None:
            self._validate_terminal_replay(plan, recovery)
            return SchedulerResult(
                run_id=run_id,
                status=OperationalStatus(recovery.terminal_status),
                completed_task_ids=recovery.completed_task_ids,
                failed_task_ids=recovery.failed_task_ids,
                blocked_task_ids=recovery.blocked_task_ids,
                receipt_ids=recovery.recovered_receipt_ids,
                receipts=recovery.recovered_receipts,
            )
        self.operational_repository.register_run(run_id, plan.fingerprint())
        if self.operational_repository.run_status(run_id) is not OperationalStatus.SUCCEEDED:
            self.operational_repository.set_run_status(run_id, OperationalStatus.RUNNING)
        tasks = {task.task_id: task for task in plan.tasks}
        outputs: dict[str, tuple[VerifiedArtifactInput, ...]] = {}
        receipts: dict[str, CanonicalTaskReceipt] = {}
        if isinstance(self.retry_amendment, RetainedCustodyCompletionAmendment):
            # Reconcile has authenticated these receipts against the retained
            # terminal. Open only inputs consumed by the remaining pure task.
            for receipt in recovery.recovered_receipts:
                outputs[receipt.task_id] = self._receipt_outputs(receipt)
                receipts[receipt.task_id] = receipt
            if set(tasks) - set(receipts) != set(self.retry_amendment.task_ids):
                raise SchedulerConsistencyError("minimal retry has work outside its pure-task scope")
        failed: set[str] = set()
        blocked: set[str] = set()
        attempt_lists: dict[str, list[OperationalAttempt]] = {}
        for attempt in self.operational_repository.attempts(run_id):
            attempt_lists.setdefault(attempt.task_id, []).append(attempt)
        attempts_by_task = {task_id: tuple(values) for task_id, values in attempt_lists.items()}

        if self.maximum_parallel_tasks == 1:
            for task_id in plan.topological_task_ids():
                if task_id in receipts:
                    continue
                task = tasks[task_id]
                outcome = self._process_task(
                    run_id,
                    task,
                    outputs,
                    receipts,
                    failed,
                    blocked,
                    attempts_by_task.get(task_id, ()),
                )
                self._record_task_outcome(
                    task_id,
                    outcome,
                    outputs=outputs,
                    receipts=receipts,
                    failed=failed,
                    blocked=blocked,
                )
        else:
            self._process_parallel_tasks(
                run_id,
                tuple(tasks[task_id] for task_id in plan.topological_task_ids() if task_id not in receipts),
                outputs=outputs,
                receipts=receipts,
                failed=failed,
                blocked=blocked,
                attempts_by_task=attempts_by_task,
            )

        status = self._terminal_status(failed, blocked)
        terminal_receipts = tuple(sorted(receipts.values(), key=lambda value: value.receipt_id))
        self.recovery_store.record_terminal(
            self.recovery_index,
            status=status,
            attempts=self.operational_repository.attempts(run_id),
            receipts=terminal_receipts,
        )
        self.operational_repository.set_run_status(run_id, status)
        return SchedulerResult(
            run_id=run_id,
            status=status,
            completed_task_ids=tuple(sorted(outputs)),
            failed_task_ids=tuple(sorted(failed)),
            blocked_task_ids=tuple(sorted(blocked)),
            receipt_ids=tuple(sorted(receipt.receipt_id for receipt in receipts.values())),
            receipts=terminal_receipts,
        )

    def _validate_terminal_replay(
        self,
        plan: ProtocolExecutionPlan,
        recovery: RunRecoveryReport,
    ) -> None:
        """Recheck current inputs/implementations without reopening terminal work."""

        tasks = {task.task_id: task for task in plan.tasks}
        receipts = {receipt.task_id: receipt for receipt in recovery.recovered_receipts}
        attempts = {
            attempt.attempt_id: attempt
            for attempt in self.operational_repository.attempts(self.recovery_index.run_id)
        }
        outputs: dict[str, tuple[VerifiedArtifactInput, ...]] = {}
        for task_id in plan.topological_task_ids():
            task = tasks[task_id]
            self.runner_registry.resolve(task)
            receipt = receipts.get(task_id)
            if receipt is None:
                continue
            dependency_outputs = self._dependency_outputs(task, outputs)
            external_inputs = self._resolve_external_inputs(task)
            try:
                attempt = attempts[receipt.attempt_id]
            except KeyError as error:
                raise SchedulerConsistencyError(
                    "terminal receipt lacks reconstructed operational state"
                ) from error
            self._validate_receipt(
                receipt,
                receipt.run_id,
                task,
                attempt,
                dependency_outputs,
                external_inputs,
            )
            outputs[task_id] = self._receipt_outputs(receipt)

    @staticmethod
    def _terminal_status(failed: set[str], blocked: set[str]) -> OperationalStatus:
        if failed:
            return OperationalStatus.FAILED
        if blocked:
            return OperationalStatus.BLOCKED
        return OperationalStatus.SUCCEEDED

    def _process_task(
        self,
        run_id: str,
        task: ProtocolExecutionTask,
        outputs: dict[str, tuple[VerifiedArtifactInput, ...]],
        receipts: dict[str, CanonicalTaskReceipt],
        failed_dependencies: set[str],
        blocked_dependencies: set[str],
        prior_attempts: tuple[OperationalAttempt, ...],
    ) -> _TaskOutcome:
        prepared = self._prepare_task(
            run_id,
            task,
            outputs,
            receipts,
            failed_dependencies,
            blocked_dependencies,
            prior_attempts,
        )
        if isinstance(prepared, _TaskOutcome):
            return prepared
        try:
            receipt = self._execute_with_retries(
                run_id,
                task,
                prepared.dependency_outputs,
                prepared.external_inputs,
                prepared.dependency_receipts,
                spent_attempt_count=prepared.spent_attempt_count,
                next_attempt_number=prepared.next_attempt_number,
            )
        except ResourceLockUnavailable:
            self.operational_repository.block_task(
                run_id,
                task.task_id,
                TaskBlockReason.RESOURCE_LOCK_UNAVAILABLE,
            )
            return _TaskOutcome(TaskAttemptDisposition.BLOCKED)
        except _RetryableBlockRecorded:
            return _TaskOutcome(TaskAttemptDisposition.BLOCKED)
        if receipt is None:
            return _TaskOutcome(TaskAttemptDisposition.FAILED)
        return _TaskOutcome(TaskAttemptDisposition.SUCCEEDED, receipt)

    def _prepare_task(
        self,
        run_id: str,
        task: ProtocolExecutionTask,
        outputs: dict[str, tuple[VerifiedArtifactInput, ...]],
        receipts: dict[str, CanonicalTaskReceipt],
        failed_dependencies: set[str],
        blocked_dependencies: set[str],
        prior_attempts: tuple[OperationalAttempt, ...],
    ) -> _PreparedTask | _TaskOutcome:
        elapsed_exhausted = (
            self.campaign_elapsed_budget_coordinator is not None
            and self.campaign_elapsed_budget_coordinator.current().exhausted
        )
        if elapsed_exhausted:
            if not any(
                attempt.disposition is TaskAttemptDisposition.BLOCKED
                and attempt.block_kind is TaskBlockKind.DURABLE
                for attempt in prior_attempts
            ):
                self.recovery_store.record_not_attempted(
                    self.recovery_index,
                    task_id=task.task_id,
                    reason=TaskBlockReason.CAMPAIGN_ELAPSED_BUDGET_EXHAUSTED,
                )
                self.operational_repository.block_task(
                    run_id,
                    task.task_id,
                    TaskBlockReason.CAMPAIGN_ELAPSED_BUDGET_EXHAUSTED,
                )
            return _TaskOutcome(TaskAttemptDisposition.BLOCKED)
        dependency_ids = set(task.dependency_task_ids)
        if dependency_ids.intersection(failed_dependencies):
            self.recovery_store.record_not_attempted(
                self.recovery_index,
                task_id=task.task_id,
                reason=TaskBlockReason.DEPENDENCY_FAILED,
            )
            self.operational_repository.block_task(
                run_id,
                task.task_id,
                TaskBlockReason.DEPENDENCY_FAILED,
            )
            return _TaskOutcome(TaskAttemptDisposition.BLOCKED)
        if dependency_ids.intersection(blocked_dependencies):
            self.operational_repository.block_task(
                run_id,
                task.task_id,
                TaskBlockReason.DEPENDENCY_BLOCKED,
            )
            return _TaskOutcome(TaskAttemptDisposition.BLOCKED)
        dependency_outputs = self._dependency_outputs(task, outputs)
        dependency_receipts = self._dependency_receipt_bindings(
            task,
            dependency_outputs=dependency_outputs,
            receipts=receipts,
        )
        if task.barrier is BarrierKind.REVEAL and task.barrier not in self.authorized_barriers:
            self.operational_repository.block_task(
                run_id,
                task.task_id,
                TaskBlockReason.AUTHORITY_REQUIRED,
            )
            return _TaskOutcome(TaskAttemptDisposition.BLOCKED)
        try:
            external_inputs = self._resolve_external_inputs(task)
        except (ResourceAdmissionError, ArtifactInputLimitExceeded):
            self.operational_repository.block_task(
                run_id,
                task.task_id,
                TaskBlockReason.RESOURCE_COMPUTABILITY_UNAVAILABLE,
            )
            return _TaskOutcome(TaskAttemptDisposition.BLOCKED)
        recovery = self._recover_task(
            run_id,
            task,
            dependency_outputs,
            external_inputs,
            prior_attempts,
        )
        if recovery.completed:
            return _TaskOutcome(TaskAttemptDisposition.SUCCEEDED, recovery.receipt)
        if recovery.durable_blocked:
            return _TaskOutcome(TaskAttemptDisposition.BLOCKED)
        if recovery.spent_attempt_count >= self.recovery_index.task(task.task_id).maximum_attempts:
            latest_attempt = max(
                prior_attempts,
                key=lambda value: value.ordinal,
                default=None,
            )
            if (
                latest_attempt is not None
                and latest_attempt.disposition is TaskAttemptDisposition.BLOCKED
                and latest_attempt.block_kind is TaskBlockKind.RETRYABLE
            ):
                return _TaskOutcome(TaskAttemptDisposition.BLOCKED)
            return _TaskOutcome(TaskAttemptDisposition.FAILED)
        return _PreparedTask(
            task=task,
            dependency_outputs=dependency_outputs,
            external_inputs=external_inputs,
            dependency_receipts=dependency_receipts,
            spent_attempt_count=recovery.spent_attempt_count,
            next_attempt_number=recovery.next_attempt_number,
        )

    @staticmethod
    def _record_task_outcome(
        task_id: str,
        outcome: _TaskOutcome,
        *,
        outputs: dict[str, tuple[VerifiedArtifactInput, ...]],
        receipts: dict[str, CanonicalTaskReceipt],
        failed: set[str],
        blocked: set[str],
    ) -> None:
        if outcome.disposition is TaskAttemptDisposition.SUCCEEDED:
            assert outcome.receipt is not None
            outputs[task_id] = LocalScheduler._receipt_outputs(outcome.receipt)
            receipts[task_id] = outcome.receipt
        elif outcome.disposition is TaskAttemptDisposition.BLOCKED:
            blocked.add(task_id)
        else:
            failed.add(task_id)

    def _process_parallel_tasks(
        self,
        run_id: str,
        tasks: tuple[ProtocolExecutionTask, ...],
        *,
        outputs: dict[str, tuple[VerifiedArtifactInput, ...]],
        receipts: dict[str, CanonicalTaskReceipt],
        failed: set[str],
        blocked: set[str],
        attempts_by_task: dict[str, tuple[OperationalAttempt, ...]],
    ) -> None:
        """Refill slots after durable completion using the frozen dependency graph."""

        unprepared = {task.task_id: task for task in tasks}
        pending: list[_ParallelTaskState] = []
        running: dict[Future[CanonicalTaskReceipt], _ParallelAttempt] = {}
        with ThreadPoolExecutor(
            max_workers=self.maximum_parallel_tasks, thread_name_prefix="empirical-lawhood-task"
        ) as pool:

            def admit() -> None:
                nonlocal pending
                admitted, deferred = self._start_parallel_batch(
                    run_id,
                    tuple(sorted(pending, key=lambda state: state.prepared.task.task_id)),
                    active=tuple(running.values()),
                    outputs=outputs,
                    receipts=receipts,
                    failed=failed,
                    blocked=blocked,
                )
                pending = list(deferred)
                for attempt in admitted:
                    future = pool.submit(
                        self._run_and_commit_parallel_attempt,
                        run_id,
                        attempt,
                    )
                    running[future] = attempt

            try:
                while unprepared or pending or running:
                    admit()
                    settled = set(outputs) | failed | blocked
                    ready = tuple(
                        sorted(
                            (
                                task
                                for task in unprepared.values()
                                if set(task.dependency_task_ids).issubset(settled)
                            ),
                            key=lambda task: (
                                self._task_worker_group(task) or task.task_id,
                                task.task_id,
                            ),
                        )
                    )
                    prepared_any = False
                    preparation_slots = self.maximum_parallel_tasks - len(running)
                    prepared_count = 0
                    for task in ready:
                        if prepared_count >= preparation_slots:
                            break
                        prepared = self._prepare_task(
                            run_id,
                            task,
                            outputs,
                            receipts,
                            failed,
                            blocked,
                            attempts_by_task.get(task.task_id, ()),
                        )
                        del unprepared[task.task_id]
                        prepared_any = True
                        prepared_count += 1
                        if isinstance(prepared, _TaskOutcome):
                            self._record_task_outcome(
                                task.task_id,
                                prepared,
                                outputs=outputs,
                                receipts=receipts,
                                failed=failed,
                                blocked=blocked,
                            )
                        else:
                            pending.append(
                                _ParallelTaskState(
                                    prepared=prepared,
                                    remaining_attempts=self.recovery_index.task(
                                        task.task_id
                                    ).maximum_attempts
                                    - prepared.spent_attempt_count,
                                    next_attempt_number=prepared.next_attempt_number,
                                )
                            )
                    admit()
                    if len(running) < self.maximum_parallel_tasks and any(
                        set(task.dependency_task_ids).issubset(set(outputs) | failed | blocked)
                        for task in unprepared.values()
                    ):
                        continue
                    if not running:
                        if pending or (unprepared and not prepared_any):
                            raise SchedulerConsistencyError(
                                "parallel scheduler has no admissible ready task"
                            )
                        continue
                    completed, _ = wait(running, return_when=FIRST_COMPLETED)
                    for future in sorted(
                        completed, key=lambda item: running[item].state.prepared.task.task_id
                    ):
                        attempt = running.pop(future)
                        pending.extend(
                            self._finish_parallel_attempts(
                                run_id,
                                (attempt,),
                                futures={attempt.attempt_id: future},
                                outputs=outputs,
                                receipts=receipts,
                                failed=failed,
                                blocked=blocked,
                            )
                        )
            finally:
                # A coordinator failure cannot leave locks or streamed results
                # owned by abandoned futures. Their issued attempts remain for
                # exact recovery/unknown-completion handling, never silent rerun.
                for future, attempt in running.items():
                    if not future.cancel():
                        try:
                            future.result()
                        except BaseException:
                            pass
                    self.lock_manager.release(attempt.state.prepared.task.resource_lock_ids)

    def _run_and_commit_parallel_attempt(
        self, run_id: str, attempt: _ParallelAttempt
    ) -> CanonicalTaskReceipt:
        """Keep bounded output publication inside its already-admitted task slot.

        SQLite, recovery acknowledgement and dependency release remain on the
        scheduler's owning thread. Another admitted task can compute or publish
        while this task's immutable output and receipt files enter custody.
        """

        prepared = attempt.state.prepared
        result = self._execute_runner(
            run_id,
            prepared.task,
            attempt.attempt_id,
            prepared.dependency_outputs,
            prepared.external_inputs,
            prepared.dependency_receipts,
            attempt.cpu_affinity,
        )
        try:
            return self._commit_runner_result(
                run_id,
                prepared.task,
                attempt.attempt_id,
                prepared.dependency_outputs,
                prepared.external_inputs,
                result,
            )
        finally:
            self._close_runner_outputs(result)

    def _start_parallel_batch(
        self,
        run_id: str,
        states: tuple[_ParallelTaskState, ...],
        *,
        active: tuple[_ParallelAttempt, ...] = (),
        outputs: dict[str, tuple[VerifiedArtifactInput, ...]],
        receipts: dict[str, CanonicalTaskReceipt],
        failed: set[str],
        blocked: set[str],
    ) -> tuple[tuple[_ParallelAttempt, ...], tuple[_ParallelTaskState, ...]]:
        capacity = self.parallel_resource_capacity
        if capacity is None:
            raise SchedulerConsistencyError("parallel resource capacity is unavailable")
        selected: list[_ParallelAttempt] = []
        deferred: list[_ParallelTaskState] = []
        active_tasks = tuple(attempt.state.prepared.task for attempt in active)
        active_worker_groups = {self._task_worker_group(task) for task in active_tasks} - {None}
        selected_lock_ids = {lock for task in active_tasks for lock in task.resource_lock_ids}
        used_cpu_cores = sum(task.capability.requested_resources.cpu_cores for task in active_tasks)
        used_memory_bytes = sum(
            task.capability.requested_resources.memory_bytes for task in active_tasks
        )
        used_gpu_devices = sum(
            task.capability.requested_resources.gpu_devices for task in active_tasks
        )
        occupied_cpu_ids = {cpu for attempt in active for cpu in attempt.cpu_affinity}
        free_cpu_ids = tuple(cpu for cpu in self.parallel_cpu_ids if cpu not in occupied_cpu_ids)
        for state in states:
            task = state.prepared.task
            budget = task.capability.requested_resources
            if (
                budget.cpu_cores > capacity.cpu_cores
                or budget.memory_bytes > capacity.memory_bytes
                or budget.gpu_devices > capacity.gpu_devices
            ):
                self.operational_repository.block_task(
                    run_id,
                    task.task_id,
                    TaskBlockReason.RESOURCE_COMPUTABILITY_UNAVAILABLE,
                )
                self._record_task_outcome(
                    task.task_id,
                    _TaskOutcome(TaskAttemptDisposition.BLOCKED),
                    outputs=outputs,
                    receipts=receipts,
                    failed=failed,
                    blocked=blocked,
                )
                continue
            capacity_available = (
                len(selected) + len(active) < self.maximum_parallel_tasks
                and used_cpu_cores + budget.cpu_cores <= capacity.cpu_cores
                and used_memory_bytes + budget.memory_bytes <= capacity.memory_bytes
                and used_gpu_devices + budget.gpu_devices <= capacity.gpu_devices
            )
            internally_locked = bool(selected_lock_ids.intersection(task.resource_lock_ids))
            worker_group = self._task_worker_group(task)
            if not capacity_available or internally_locked or worker_group in active_worker_groups:
                deferred.append(state)
                continue
            if not self.lock_manager.acquire(task.resource_lock_ids):
                self.operational_repository.block_task(
                    run_id,
                    task.task_id,
                    TaskBlockReason.RESOURCE_LOCK_UNAVAILABLE,
                )
                self._record_task_outcome(
                    task.task_id,
                    _TaskOutcome(TaskAttemptDisposition.BLOCKED),
                    outputs=outputs,
                    receipts=receipts,
                    failed=failed,
                    blocked=blocked,
                )
                continue
            attempt_number = state.next_attempt_number
            attempt_id = self.recovery_index.task(task.task_id).attempts[attempt_number - 1].attempt_id
            if not self._admit_campaign_elapsed_reservation(task.task_id, attempt_id):
                self.lock_manager.release(task.resource_lock_ids)
                self._stop_envelope_task(
                    task.task_id,
                    reason_code="CAMPAIGN_ELAPSED_BUDGET_INSUFFICIENT_FOR_PROJECTED_TASK",
                )
                self.recovery_store.record_not_attempted(
                    self.recovery_index,
                    task_id=task.task_id,
                    reason=TaskBlockReason.CAMPAIGN_ELAPSED_BUDGET_EXHAUSTED,
                )
                self.operational_repository.block_task(
                    run_id,
                    task.task_id,
                    TaskBlockReason.CAMPAIGN_ELAPSED_BUDGET_EXHAUSTED,
                )
                self._record_task_outcome(
                    task.task_id,
                    _TaskOutcome(TaskAttemptDisposition.BLOCKED),
                    outputs=outputs,
                    receipts=receipts,
                    failed=failed,
                    blocked=blocked,
                )
                continue
            try:
                self._reserve_envelope_attempt(task.task_id, attempt_id)
                self.operational_repository.start_attempt(
                    run_id=run_id,
                    task_id=task.task_id,
                    attempt_id=attempt_id,
                    lease_id=f"lease.{attempt_id}",
                    owner_id=self.owner_id,
                    expires_epoch_seconds=self.clock.now() + self.lease_seconds,
                )
                self._launch_envelope_attempt(task.task_id, attempt_id)
            except BaseException:
                self.lock_manager.release(task.resource_lock_ids)
                raise
            state.remaining_attempts -= 1
            state.next_attempt_number += 1
            if worker_group is not None:
                active_worker_groups.add(worker_group)
            selected.append(
                _ParallelAttempt(
                    state=state,
                    attempt_id=attempt_id,
                    cpu_affinity=free_cpu_ids[: budget.cpu_cores],
                )
            )
            free_cpu_ids = free_cpu_ids[budget.cpu_cores :]
            selected_lock_ids.update(task.resource_lock_ids)
            used_cpu_cores += budget.cpu_cores
            used_memory_bytes += budget.memory_bytes
            used_gpu_devices += budget.gpu_devices
        return tuple(selected), tuple(deferred)

    def _finish_parallel_attempts(
        self,
        run_id: str,
        attempts: tuple[_ParallelAttempt, ...],
        *,
        futures: dict[str, Future[CanonicalTaskReceipt]],
        outputs: dict[str, tuple[VerifiedArtifactInput, ...]],
        receipts: dict[str, CanonicalTaskReceipt],
        failed: set[str],
        blocked: set[str],
    ) -> tuple[_ParallelTaskState, ...]:
        committed: list[tuple[_ParallelAttempt, CanonicalTaskReceipt]] = []
        retries: list[_ParallelTaskState] = []
        try:
            for attempt in attempts:
                task = attempt.state.prepared.task
                try:
                    receipt = futures[attempt.attempt_id].result()
                except CampaignElapsedBoundaryReached:
                    self._record_campaign_elapsed_boundary(
                        task.task_id,
                        attempt.attempt_id,
                    )
                    self._record_task_outcome(
                        task.task_id,
                        _TaskOutcome(TaskAttemptDisposition.BLOCKED),
                        outputs=outputs,
                        receipts=receipts,
                        failed=failed,
                        blocked=blocked,
                    )
                except ProgressStalledError as error:
                    failure = operational_failure_diagnostic(
                        error,
                        failure_class=(
                            OperationalFailureClass.CHILD_PROCESS_FAILURE
                            if attempt.state.remaining_attempts
                            else OperationalFailureClass.RETRY_EXHAUSTION
                        ),
                    )
                    self._record_resource_progress_stalled(
                        task.task_id,
                        attempt.attempt_id,
                    )
                    self.recovery_store.record_failure(
                        self.recovery_index,
                        task_id=task.task_id,
                        attempt_id=attempt.attempt_id,
                        reason_code=failure.failure_class.reason_code,
                        failure=failure,
                    )
                    self.operational_repository.fail_attempt(
                        attempt.attempt_id,
                        failure.failure_class.reason_code,
                    )
                    if attempt.state.remaining_attempts:
                        retries.append(attempt.state)
                    else:
                        self._stop_envelope_task(
                            task.task_id,
                            reason_code=failure.failure_class.value,
                        )
                        self._record_task_outcome(
                            task.task_id,
                            _TaskOutcome(TaskAttemptDisposition.FAILED),
                            outputs=outputs,
                            receipts=receipts,
                            failed=failed,
                            blocked=blocked,
                        )
                except (ResourceAdmissionError, ArtifactInputLimitExceeded) as error:
                    failure = operational_failure_diagnostic(error)
                    self._record_envelope_operational_failure(
                        task.task_id,
                        attempt.attempt_id,
                        reason_code=failure.failure_class.value,
                    )
                    self.recovery_store.record_failure(
                        self.recovery_index,
                        task_id=task.task_id,
                        attempt_id=attempt.attempt_id,
                        reason_code=TaskBlockReason.RESOURCE_COMPUTABILITY_UNAVAILABLE.value,
                        failure=failure,
                        blocked=True,
                    )
                    self._stop_envelope_task(
                        task.task_id,
                        reason_code="RESOURCE_COMPUTABILITY_UNAVAILABLE",
                    )
                    self.operational_repository.block_attempt(
                        attempt.attempt_id,
                        TaskBlockReason.RESOURCE_COMPUTABILITY_UNAVAILABLE,
                    )
                    self._record_task_outcome(
                        task.task_id,
                        _TaskOutcome(TaskAttemptDisposition.BLOCKED),
                        outputs=outputs,
                        receipts=receipts,
                        failed=failed,
                        blocked=blocked,
                    )
                except Exception as error:
                    failure = operational_failure_diagnostic(error)
                    if failure.retryable and not attempt.state.remaining_attempts:
                        failure = operational_failure_diagnostic(
                            error,
                            failure_class=OperationalFailureClass.RETRY_EXHAUSTION,
                        )
                    self._record_envelope_operational_failure(
                        task.task_id,
                        attempt.attempt_id,
                        reason_code=failure.failure_class.value,
                    )
                    self.recovery_store.record_failure(
                        self.recovery_index,
                        task_id=task.task_id,
                        attempt_id=attempt.attempt_id,
                        reason_code=failure.failure_class.reason_code,
                        failure=failure,
                    )
                    self.operational_repository.fail_attempt(
                        attempt.attempt_id,
                        failure.failure_class.reason_code,
                    )
                    if failure.retryable and attempt.state.remaining_attempts:
                        retries.append(attempt.state)
                    else:
                        self._stop_envelope_task(
                            task.task_id,
                            reason_code=failure.failure_class.value,
                        )
                        self._record_task_outcome(
                            task.task_id,
                            _TaskOutcome(TaskAttemptDisposition.FAILED),
                            outputs=outputs,
                            receipts=receipts,
                            failed=failed,
                            blocked=blocked,
                        )
                else:
                    committed.append((attempt, receipt))
                finally:
                    self.lock_manager.release(task.resource_lock_ids)
        finally:
            for attempt in attempts:
                self.lock_manager.release(attempt.state.prepared.task.resource_lock_ids)

        for attempt, receipt in committed:
            task_id = attempt.state.prepared.task.task_id
            self.recovery_store.record_receipt(self.recovery_index, receipt)
            self._finalize_envelope_receipt(
                task_id,
                attempt.attempt_id,
                receipt,
                recovery=False,
            )
            self.failure_injector.after_receipt_commit(
                run_id,
                task_id,
                attempt.attempt_id,
            )
            self.operational_repository.complete_attempt(attempt.attempt_id)
            self._record_task_outcome(
                task_id,
                _TaskOutcome(TaskAttemptDisposition.SUCCEEDED, receipt),
                outputs=outputs,
                receipts=receipts,
                failed=failed,
                blocked=blocked,
            )
        return tuple(sorted(retries, key=lambda value: value.prepared.task.task_id))

    @staticmethod
    def _close_runner_outputs(result: RunnerResult) -> None:
        for output in result.outputs:
            if isinstance(output, StreamedTaskOutput):
                output.source.close()

    @staticmethod
    def _dependency_outputs(
        task: ProtocolExecutionTask,
        outputs: dict[str, tuple[VerifiedArtifactInput, ...]],
    ) -> tuple[VerifiedArtifactInput, ...]:
        if any(dependency not in outputs for dependency in task.dependency_task_ids):
            raise SchedulerConsistencyError("ready task lacks dependency materializations")
        if isinstance(task, ExecutionTask):
            selected: dict[str, VerifiedArtifactInput] = {}
            for spec in task.scientific_inputs:
                if spec.producer_task_id is None:
                    continue
                available = outputs[spec.producer_task_id]
                matches = tuple(
                    value
                    for value in available
                    if value.logical.logical_artifact_id == spec.operational_logical_artifact_id
                )
                if len(matches) != 1:
                    raise SchedulerConsistencyError(
                        "exact scientific dependency output is unavailable"
                    )
                value = matches[0]
                if (
                    value.logical.payload_schema != spec.payload_schema
                    or value.logical.media_type != spec.media_type
                    or value.materialization.size_bytes > spec.maximum_size_bytes
                ):
                    raise SchedulerConsistencyError(
                        "scientific dependency materialization differs from its edge"
                    )
                selected[value.materialization.materialization_id] = value
            return tuple(
                sorted(
                    selected.values(),
                    key=lambda item: item.materialization.materialization_id,
                )
            )
        return tuple(
            sorted(
                (
                    output
                    for dependency in task.dependency_task_ids
                    for output in outputs[dependency]
                ),
                key=lambda item: item.materialization.materialization_id,
            )
        )

    @staticmethod
    def _dependency_receipt_bindings(
        task: ProtocolExecutionTask,
        *,
        dependency_outputs: tuple[VerifiedArtifactInput, ...],
        receipts: dict[str, CanonicalTaskReceipt],
    ) -> tuple[DependencyReceiptBinding, ...]:
        selected_materialization_ids = {
            value.logical.logical_artifact_id: value.materialization.materialization_id
            for value in dependency_outputs
        }
        bindings: list[DependencyReceiptBinding] = []
        for dependency in task.dependency_task_ids:
            receipt = receipts[dependency]
            if isinstance(task, ExecutionTask):
                expected_logical_ids = {
                    value.operational_logical_artifact_id
                    for value in task.scientific_inputs
                    if value.producer_task_id == dependency
                }
                materialization_ids = tuple(
                    sorted(selected_materialization_ids[value] for value in expected_logical_ids)
                )
            else:
                materialization_ids = tuple(
                    value.materialization_id for value in receipt.output_materializations
                )
            if not set(materialization_ids).issubset(
                {value.materialization_id for value in receipt.output_materializations}
            ):
                raise SchedulerConsistencyError(
                    "selected dependency output is absent from its receipt"
                )
            bindings.append(
                DependencyReceiptBinding(
                    receipt_id=receipt.receipt_id,
                    task_id=dependency,
                    output_materialization_ids=materialization_ids,
                )
            )
        return tuple(sorted(bindings, key=lambda value: value.receipt_id))

    @staticmethod
    def _receipt_outputs(
        receipt: CanonicalTaskReceipt,
    ) -> tuple[VerifiedArtifactInput, ...]:
        if len(receipt.output_logical_artifacts) != len(receipt.output_materializations):
            raise SchedulerConsistencyError("task receipt lacks exact logical output identities")
        return tuple(
            VerifiedArtifactInput(logical, materialization)
            for logical, materialization in zip(
                receipt.output_logical_artifacts,
                receipt.output_materializations,
                strict=True,
            )
        )

    def _recover_task(
        self,
        run_id: str,
        task: ProtocolExecutionTask,
        dependency_outputs: tuple[VerifiedArtifactInput, ...],
        external_inputs: tuple[VerifiedArtifactInput, ...],
        attempts: tuple[OperationalAttempt, ...],
    ) -> _TaskRecovery:
        if any(attempt.run_id != run_id or attempt.task_id != task.task_id for attempt in attempts):
            raise SchedulerConsistencyError("cached operational attempts bind another task")
        candidate_ordinals = {
            candidate.attempt_id: candidate.ordinal
            for candidate in self.recovery_index.task(task.task_id).attempts
        }
        spent_attempt_count = sum(
            attempt.attempt_id in candidate_ordinals
            and attempt.disposition
            in {
                TaskAttemptDisposition.FAILED,
                TaskAttemptDisposition.BLOCKED,
            }
            for attempt in attempts
        )
        durable_blocked = any(
            attempt.disposition is TaskAttemptDisposition.BLOCKED
            and attempt.block_kind is TaskBlockKind.DURABLE
            and not (
                self.retry_amendment is not None
                and attempt.reason_code == TaskBlockReason.DEPENDENCY_FAILED.value
                and attempt.attempt_id not in candidate_ordinals
            )
            for attempt in attempts
        )
        next_attempt_number = (
            max(
                (
                    candidate_ordinals[attempt.attempt_id]
                    for attempt in attempts
                    if attempt.attempt_id in candidate_ordinals
                ),
                default=0,
            )
            + 1
        )
        for attempt in attempts:
            if attempt.disposition not in {
                TaskAttemptDisposition.RUNNING,
                TaskAttemptDisposition.SUCCEEDED,
            }:
                continue
            receipt = self.receipt_store.read(
                run_id,
                task.task_id,
                attempt.attempt_id,
            )
            if receipt is not None:
                self._validate_receipt(
                    receipt,
                    run_id,
                    task,
                    attempt,
                    dependency_outputs,
                    external_inputs,
                )
                if attempt.disposition is TaskAttemptDisposition.RUNNING:
                    self.operational_repository.complete_attempt(attempt.attempt_id)
                self._finalize_envelope_receipt(
                    task.task_id,
                    attempt.attempt_id,
                    receipt,
                    recovery=True,
                )
                return _TaskRecovery(
                    True,
                    False,
                    spent_attempt_count,
                    next_attempt_number,
                    receipt,
                )
            if attempt.disposition is TaskAttemptDisposition.SUCCEEDED:
                raise SchedulerConsistencyError("successful task has no durable receipt")
            elapsed_exhausted = (
                self.campaign_elapsed_budget_coordinator is not None
                and self.campaign_elapsed_budget_coordinator.current().exhausted
            )
            if self.execution_resource_envelope_coordinator is not None and elapsed_exhausted:
                self._record_campaign_elapsed_boundary(task.task_id, attempt.attempt_id)
                return _TaskRecovery(
                    False,
                    True,
                    spent_attempt_count + 1,
                    next_attempt_number,
                    None,
                )
            if (
                attempt.lease_expires_epoch_seconds is None
                or attempt.lease_expires_epoch_seconds > self.clock.now()
            ):
                raise LiveLeaseError("task has an active lease and no receipt")
            if self.execution_envelope_coordinator is not None:
                self._unknown_envelope_completion(task.task_id, attempt.attempt_id)
                self.operational_repository.block_attempt(
                    attempt.attempt_id,
                    TaskBlockReason.UNKNOWN_COMPLETION,
                )
                return _TaskRecovery(
                    False,
                    True,
                    spent_attempt_count + 1,
                    next_attempt_number,
                    None,
                )
            self.operational_repository.fail_attempt(
                attempt.attempt_id,
                "lease-expired",
            )
            spent_attempt_count += 1
        return _TaskRecovery(
            False,
            durable_blocked,
            spent_attempt_count,
            next_attempt_number,
            None,
        )

    def _execute_with_retries(
        self,
        run_id: str,
        task: ProtocolExecutionTask,
        dependency_outputs: tuple[VerifiedArtifactInput, ...],
        external_inputs: tuple[VerifiedArtifactInput, ...],
        dependency_receipts: tuple[DependencyReceiptBinding, ...],
        *,
        spent_attempt_count: int,
        next_attempt_number: int,
    ) -> CanonicalTaskReceipt | None:
        remaining_attempts = (
            self.recovery_index.task(task.task_id).maximum_attempts - spent_attempt_count
        )
        for offset in range(remaining_attempts):
            if not self.lock_manager.acquire(task.resource_lock_ids):
                raise ResourceLockUnavailable(task.task_id)
            attempt_number = next_attempt_number + offset
            attempt_id = self.recovery_index.task(task.task_id).attempts[attempt_number - 1].attempt_id
            lease_id = f"lease.{attempt_id}"
            if not self._admit_campaign_elapsed_reservation(task.task_id, attempt_id):
                self._stop_envelope_task(
                    task.task_id,
                    reason_code="CAMPAIGN_ELAPSED_BUDGET_INSUFFICIENT_FOR_PROJECTED_TASK",
                )
                self.recovery_store.record_not_attempted(
                    self.recovery_index,
                    task_id=task.task_id,
                    reason=TaskBlockReason.CAMPAIGN_ELAPSED_BUDGET_EXHAUSTED,
                )
                self.operational_repository.block_task(
                    run_id,
                    task.task_id,
                    TaskBlockReason.CAMPAIGN_ELAPSED_BUDGET_EXHAUSTED,
                )
                self.lock_manager.release(task.resource_lock_ids)
                raise _RetryableBlockRecorded
            self._reserve_envelope_attempt(task.task_id, attempt_id)
            self.operational_repository.start_attempt(
                run_id=run_id,
                task_id=task.task_id,
                attempt_id=attempt_id,
                lease_id=lease_id,
                owner_id=self.owner_id,
                expires_epoch_seconds=self.clock.now() + self.lease_seconds,
            )
            self._launch_envelope_attempt(task.task_id, attempt_id)
            try:
                receipt = self._run_attempt(
                    run_id,
                    task,
                    attempt_id,
                    dependency_outputs,
                    external_inputs,
                    dependency_receipts,
                )
            except InjectedSchedulerCrash:
                raise
            except CampaignElapsedBoundaryReached as error:
                self._record_campaign_elapsed_boundary(task.task_id, attempt_id)
                raise _RetryableBlockRecorded from error
            except ProgressStalledError as error:
                failure = operational_failure_diagnostic(
                    error,
                    failure_class=(
                        OperationalFailureClass.CHILD_PROCESS_FAILURE
                        if offset + 1 < remaining_attempts
                        else OperationalFailureClass.RETRY_EXHAUSTION
                    ),
                )
                self._record_resource_progress_stalled(task.task_id, attempt_id)
                self.recovery_store.record_failure(
                    self.recovery_index,
                    task_id=task.task_id,
                    attempt_id=attempt_id,
                    reason_code=failure.failure_class.reason_code,
                    failure=failure,
                )
                self.operational_repository.fail_attempt(
                    attempt_id,
                    failure.failure_class.reason_code,
                )
                if offset + 1 == remaining_attempts:
                    self._stop_envelope_task(
                        task.task_id,
                        reason_code="RETRY_BUDGET_EXHAUSTED",
                    )
                continue
            except (ResourceAdmissionError, ArtifactInputLimitExceeded) as error:
                failure = operational_failure_diagnostic(error)
                self._record_envelope_operational_failure(
                    task.task_id,
                    attempt_id,
                    reason_code=failure.failure_class.value,
                )
                self.recovery_store.record_failure(
                    self.recovery_index,
                    task_id=task.task_id,
                    attempt_id=attempt_id,
                    reason_code=TaskBlockReason.RESOURCE_COMPUTABILITY_UNAVAILABLE.value,
                    failure=failure,
                    blocked=True,
                )
                self._stop_envelope_task(
                    task.task_id,
                    reason_code="RESOURCE_COMPUTABILITY_UNAVAILABLE",
                )
                self.operational_repository.block_attempt(
                    attempt_id,
                    TaskBlockReason.RESOURCE_COMPUTABILITY_UNAVAILABLE,
                )
                raise _RetryableBlockRecorded from error
            except Exception as error:
                failure = operational_failure_diagnostic(error)
                if failure.retryable and offset + 1 == remaining_attempts:
                    failure = operational_failure_diagnostic(
                        error,
                        failure_class=OperationalFailureClass.RETRY_EXHAUSTION,
                    )
                self._record_envelope_operational_failure(
                    task.task_id,
                    attempt_id,
                    reason_code=failure.failure_class.value,
                )
                self.recovery_store.record_failure(
                    self.recovery_index,
                    task_id=task.task_id,
                    attempt_id=attempt_id,
                    reason_code=failure.failure_class.reason_code,
                    failure=failure,
                )
                self.operational_repository.fail_attempt(
                    attempt_id,
                    failure.failure_class.reason_code,
                )
                if not failure.retryable:
                    self._stop_envelope_task(
                        task.task_id,
                        reason_code=failure.failure_class.value,
                    )
                    return None
                continue
            finally:
                self.lock_manager.release(task.resource_lock_ids)
            self.recovery_store.record_receipt(self.recovery_index, receipt)
            self._finalize_envelope_receipt(
                task.task_id,
                attempt_id,
                receipt,
                recovery=False,
            )
            self.failure_injector.after_receipt_commit(
                run_id,
                task.task_id,
                attempt_id,
            )
            self.operational_repository.complete_attempt(attempt_id)
            return receipt
        return None

    def _run_attempt(
        self,
        run_id: str,
        task: ProtocolExecutionTask,
        attempt_id: str,
        dependency_outputs: tuple[VerifiedArtifactInput, ...],
        external_inputs: tuple[VerifiedArtifactInput, ...],
        dependency_receipts: tuple[DependencyReceiptBinding, ...],
    ) -> CanonicalTaskReceipt:
        result = self._execute_runner(
            run_id,
            task,
            attempt_id,
            dependency_outputs,
            external_inputs,
            dependency_receipts,
        )
        try:
            return self._commit_runner_result(
                run_id,
                task,
                attempt_id,
                dependency_outputs,
                external_inputs,
                result,
            )
        finally:
            for output in result.outputs:
                if isinstance(output, StreamedTaskOutput):
                    output.source.close()

    def _task_worker_group(self, task: ProtocolExecutionTask) -> str | None:
        if self.execution_resource_envelope_coordinator is None:
            return None
        if task.task_id not in self._worker_group_ids:
            group = None
            runner = self.runner_registry.resolve(task)
            if (
                getattr(runner, "worker_chunk_limit", None) == 2
                and callable(getattr(self.executor, "execute_deadline_free_in_group", None))
                and task.task_id != self.adjudication_task_id
                and self.assurance_profile is not ExecutionAssuranceProfile.STRICT_ISOLATED
            ):
                spec = self._worker_group_specs[task.task_id]
                if (
                    not spec.native_simulator_launch
                    and spec.progress_liveness is None
                    and spec.preparation_unit_id is not None
                    and spec.physical_independent_unit_id is not None
                ):
                    group = f"{spec.preparation_unit_id}:{spec.physical_independent_unit_id}"
            self._worker_group_ids[task.task_id] = group
        return self._worker_group_ids[task.task_id]

    def _execute_runner(
        self,
        run_id: str,
        task: ProtocolExecutionTask,
        attempt_id: str,
        dependency_outputs: tuple[VerifiedArtifactInput, ...],
        external_inputs: tuple[VerifiedArtifactInput, ...],
        dependency_receipts: tuple[DependencyReceiptBinding, ...],
        cpu_affinity: tuple[int, ...] = (),
    ) -> RunnerResult:
        """Run one isolated worker without committing its scientific outputs."""

        runner = self.runner_registry.resolve(task)
        input_bindings, input_ports = self._worker_inputs(
            task,
            dependency_outputs,
            external_inputs,
        )
        context = TaskContext(
            run_id=run_id,
            task_id=task.task_id,
            attempt_id=attempt_id,
            config=task.capability.config,
            input_bindings=input_bindings,
            input_ports=input_ports,
            output_ports=tuple(
                WorkerOutputPort(
                    output_id=output.output_id,
                    payload_schema=output.payload_schema,
                    profile=output.profile,
                    media_type=output.media_type,
                    logical_artifact_id=output.logical_artifact_id,
                )
                for output in task.outputs
            ),
            permissions=task.capability.required_permissions,
            outcome_access=task.capability.requested_outcome_access,
            resource_budget=task.capability.requested_resources,
            isolation_profile=(
                WorkerIsolationProfile.EXPLORATION_NO_NETWORK
                if self.lane is PlanLane.EXPLORATORY
                and self.assurance_profile is ExecutionAssuranceProfile.STRICT_ISOLATED
                else WorkerIsolationProfile.TRUSTED_LOCAL
            ),
            dependency_receipts=dependency_receipts,
            scientific_adjudication_context=(
                self.scientific_adjudication_context
                if task.task_id == self.adjudication_task_id
                else None
            ),
            cpu_affinity=cpu_affinity,
        )
        result: RunnerResult | None = None
        input_close_error: Exception | None = None
        try:
            if self.execution_resource_envelope_coordinator is not None:
                cell_id = self._resource_envelope_cell_id(task.task_id)
                assert self.execution_resource_envelope_id is not None
                resource_spec = self.execution_resource_envelope_coordinator.current(
                    self.execution_resource_envelope_id
                ).spec.cell(cell_id)
                try:
                    validate_runner_resource_interface(runner, resource_spec)
                except ValueError as error:
                    raise ResourceAdmissionError(str(error)) from error
                projection_builder = getattr(runner, "jit_signature_projection", None)
                if callable(projection_builder):
                    observed_projection = projection_builder(context)
                    manifest = self.jit_graph_signature_manifest
                    if (
                        not isinstance(
                            observed_projection,
                            JitCellSignatureProjection,
                        )
                        or manifest is None
                    ):
                        raise ResourceAdmissionError(
                            "native runner returned invalid JIT projection evidence"
                        )
                    try:
                        manifest.require_projection(observed_projection)
                    except ValueError as error:
                        raise ResourceAdmissionError(
                            "native runner attempted an unmanifested JIT graph"
                        ) from error
                    if (
                        observed_projection.cell_id != resource_spec.jit_cell_id
                        or observed_projection.signature_sha256
                        != resource_spec.jit_signature_sha256
                    ):
                        raise ResourceAdmissionError(
                            "native runner JIT projection differs from its task cell"
                        )
                deadline_free_execute = getattr(
                    self.executor,
                    "execute_deadline_free",
                    None,
                )
                if not callable(deadline_free_execute):
                    raise ResourceAdmissionError(
                        "resource execution plan requires a deadline-free executor"
                    )
                worker_group = self._task_worker_group(task)
                execution_kwargs: dict[str, object] = {}
                if worker_group is not None:
                    deadline_free_execute = getattr(self.executor, "execute_deadline_free_in_group")
                    execution_kwargs["execution_group_id"] = worker_group
                if (
                    resource_spec.native_simulator_launch
                    and self.campaign_elapsed_budget_coordinator is not None
                ):
                    if worker_group is not None:
                        raise SchedulerConsistencyError(
                            "native elapsed-governed work cannot use a reusable worker group"
                        )
                    deadline_free_execute = getattr(
                        self.executor,
                        "execute_deadline_free_until",
                        None,
                    )
                    if not callable(deadline_free_execute):
                        raise ResourceAdmissionError(
                            "campaign elapsed budget requires a hard-boundary executor"
                        )
                    execution_kwargs["hard_remaining_seconds"] = (
                        self.campaign_elapsed_budget_coordinator.hard_remaining_seconds(
                            at_utc=self.execution_envelope_event_clock.now_utc()
                        )
                    )
                result = deadline_free_execute(
                    runner,
                    context,
                    progress_contract=resource_spec.progress_liveness,
                    progress_callback=lambda counter: self._record_resource_progress(
                        task.task_id,
                        attempt_id,
                        counter,
                    ),
                    **execution_kwargs,
                )
            else:
                result = self.executor.execute(
                    runner,
                    context,
                    timeout_seconds=task.capability.requested_resources.wall_time_seconds,
                )
        finally:
            for port in input_ports:
                try:
                    port.close()
                except Exception as error:  # noqa: PERF203 - close every authorized port
                    if input_close_error is None:
                        input_close_error = error
            if input_close_error is not None:
                if result is not None:
                    for output in result.outputs:
                        if isinstance(output, StreamedTaskOutput):
                            output.source.close()
                raise SchedulerConsistencyError(
                    "worker input port could not be closed"
                ) from input_close_error
        assert result is not None
        return result

    def _commit_runner_result(
        self,
        run_id: str,
        task: ProtocolExecutionTask,
        attempt_id: str,
        dependency_outputs: tuple[VerifiedArtifactInput, ...],
        external_inputs: tuple[VerifiedArtifactInput, ...],
        result: RunnerResult,
    ) -> CanonicalTaskReceipt:
        try:
            return self._publish_runner_result(
                run_id, task, attempt_id, dependency_outputs, external_inputs, result
            )
        except BaseException:
            close_workers = getattr(self.executor, "close_run_workers", None)
            if callable(close_workers):
                close_workers(run_id)
            raise

    def _publish_runner_result(
        self,
        run_id: str,
        task: ProtocolExecutionTask,
        attempt_id: str,
        dependency_outputs: tuple[VerifiedArtifactInput, ...],
        external_inputs: tuple[VerifiedArtifactInput, ...],
        result: RunnerResult,
    ) -> CanonicalTaskReceipt:
        output_bytes = sum(
            len(output.payload) if isinstance(output, TaskOutputPayload) else output.size_bytes
            for output in result.outputs
        )
        if output_bytes > task.capability.requested_resources.output_bytes:
            raise SchedulerConsistencyError("runner exceeded its frozen output-byte budget")
        expected_output_ids = tuple(output.output_id for output in task.outputs)
        observed_output_ids = tuple(output.output_id for output in result.outputs)
        if observed_output_ids != expected_output_ids:
            raise SchedulerConsistencyError("runner output ports differ from frozen task")
        if not result.checks or not all(check.passed for check in result.checks):
            raise SchedulerConsistencyError("runner output checks did not all pass")
        self._validate_output_profiles(task, result)
        self._validate_output_semantics(task, result)
        payloads = {output.output_id: output for output in result.outputs}
        write_requests: list[ArtifactWriteRequest | ArtifactStreamWriteRequest] = []
        for spec in task.outputs:
            payload = payloads[spec.output_id]
            fallback_ceiling = (
                VisibilityCeiling.most_restrictive(*spec.parent_visibility_ceilings)
                if spec.parent_visibility_ceilings
                else spec.visibility_ceiling
            )
            parents = [
                ArtifactLineageParent(
                    identity=ObjectIdentity.from_record(
                        value.logical.logical_artifact_id,
                        value.logical,
                    ),
                    visibility_ceiling=value.logical.visibility_ceiling,
                    outcome_access=value.logical.outcome_access,
                )
                for value in external_inputs
            ]
            parents.extend(
                ArtifactLineageParent(
                    identity=ObjectIdentity.from_record(
                        value.logical.logical_artifact_id,
                        value.logical,
                    ),
                    visibility_ceiling=value.logical.visibility_ceiling,
                    outcome_access=value.logical.outcome_access,
                )
                for value in dependency_outputs
            )
            if not parents and spec.parent_visibility_ceilings:
                parents.append(
                    ArtifactLineageParent(
                        identity=ObjectIdentity.from_record(task.task_id, task),
                        visibility_ceiling=fallback_ceiling,
                        outcome_access=spec.outcome_access,
                    )
                )
            lineage_parents = tuple(sorted(parents, key=lineage_parent_sort_key))
            effective_outcome_access = derived_outcome_access(
                spec.outcome_access,
                *(parent.outcome_access for parent in lineage_parents),
            )
            parent_ceilings = tuple(parent.visibility_ceiling for parent in lineage_parents)
            semantic_contract = self.output_semantic_contracts.get(
                (
                    task.capability.capability_key,
                    task.capability.capability_version,
                    spec.payload_schema,
                    spec.profile,
                )
            )
            semantic_validation = (
                None
                if semantic_contract is None
                else semantic_contract.artifact_validation(
                    task_id=task.task_id,
                    output_id=spec.output_id,
                )
            )
            if isinstance(payload, TaskOutputPayload):
                write_requests.append(
                    ArtifactWriteRequest(
                        logical_artifact_id=spec.logical_artifact_id,
                        relative_path=spec.relative_path,
                        payload_schema=spec.payload_schema,
                        profile=spec.profile,
                        media_type=spec.media_type,
                        publication_scope_id=(f"task-output-scope.{run_id}.{task.task_id}"),
                        publication_scope_relative_root=f"runs/{run_id}",
                        payload=payload.payload,
                        visibility_ceiling=spec.visibility_ceiling,
                        parent_visibility_ceilings=parent_ceilings,
                        outcome_access=effective_outcome_access,
                        logical_content_sha256=payload.logical_content_sha256,
                        lineage_parents=lineage_parents,
                        minimum_free_bytes=self.minimum_free_bytes,
                        semantic_validation=semantic_validation,
                    )
                )
            else:
                maximum_chunk_bytes = min(
                    STREAM_CHUNK_BYTES,
                    task.capability.requested_resources.output_bytes,
                )
                write_requests.append(
                    ArtifactStreamWriteRequest(
                        logical_artifact_id=spec.logical_artifact_id,
                        relative_path=spec.relative_path,
                        payload_schema=spec.payload_schema,
                        profile=spec.profile,
                        media_type=spec.media_type,
                        publication_scope_id=(f"task-output-scope.{run_id}.{task.task_id}"),
                        publication_scope_relative_root=f"runs/{run_id}",
                        chunks=payload.source.chunks(maximum_chunk_bytes),
                        maximum_bytes=task.capability.requested_resources.output_bytes,
                        maximum_chunk_bytes=maximum_chunk_bytes,
                        expected_size_bytes=payload.size_bytes,
                        expected_physical_sha256=payload.physical_sha256,
                        visibility_ceiling=spec.visibility_ceiling,
                        parent_visibility_ceilings=parent_ceilings,
                        outcome_access=effective_outcome_access,
                        logical_content_sha256=payload.logical_content_sha256,
                        lineage_parents=lineage_parents,
                        minimum_free_bytes=self.minimum_free_bytes,
                        semantic_validation=semantic_validation,
                    )
                )
        write_results = list(self.artifact_writer.write_batch(tuple(write_requests)))
        if len(write_results) != len(task.outputs):
            raise SchedulerConsistencyError("artifact writer returned another output batch shape")
        for spec, payload, write_result in zip(
            task.outputs,
            result.outputs,
            write_results,
            strict=True,
        ):
            if isinstance(payload, StreamedTaskOutput):
                if (
                    write_result.materialization.size_bytes != payload.size_bytes
                    or write_result.materialization.physical_sha256 != payload.physical_sha256
                ):
                    raise SchedulerConsistencyError(
                        "streamed output identity changed before publication"
                    )
        ordered_results = tuple(
            sorted(
                write_results,
                key=lambda item: item.materialization.materialization_id,
            )
        )
        assurance_checks = (
            ReceiptCheck(
                f"execution-assurance-{self.assurance_profile.value.lower().replace('_', '-')}",
                True,
                (),
            ),
            *(
                ReceiptCheck(code.lower().replace("_", "-"), True, ())
                for code in self.assurance_codes
            ),
        )
        receipt = CanonicalTaskReceipt(
            receipt_id=f"receipt.{attempt_id}",
            run_id=run_id,
            task_id=task.task_id,
            attempt_id=attempt_id,
            implementation_commit=self.implementation_commit,
            input_materialization_ids=self._input_materialization_ids(
                dependency_outputs,
                external_inputs,
            ),
            output_materializations=tuple(item.materialization for item in ordered_results),
            output_logical_artifacts=tuple(item.logical for item in ordered_results),
            checks=tuple(
                sorted(
                    (*result.checks, *assurance_checks),
                    key=lambda value: value.check_id,
                )
            ),
            operational_status=OperationalStatus.SUCCEEDED,
            reason_codes=(),
        )
        output_visibility = VisibilityCeiling.most_restrictive(
            *(output.visibility_ceiling for output in task.outputs)
        )
        self.receipt_store.commit(
            receipt,
            visibility_ceiling=output_visibility,
            outcome_access=most_restrictive_outcome_access(
                task.capability.requested_outcome_access,
                *(item.logical.outcome_access for item in ordered_results),
            ),
        )
        return receipt

    def _worker_inputs(
        self,
        task: ProtocolExecutionTask,
        dependency_outputs: tuple[VerifiedArtifactInput, ...],
        external_inputs: tuple[VerifiedArtifactInput, ...],
    ) -> tuple[tuple[WorkerInputBinding, ...], tuple[WorkerInputPort, ...]]:
        inputs = tuple(
            sorted(
                (
                    *(
                        (
                            value,
                            WorkerInputBinding.from_verified(
                                value,
                                kind=WorkerInputKind.DEPENDENCY,
                            ),
                        )
                        for value in dependency_outputs
                    ),
                    *(
                        (
                            value,
                            WorkerInputBinding.from_verified(
                                value,
                                kind=WorkerInputKind.EXTERNAL,
                            ),
                        )
                        for value in external_inputs
                    ),
                ),
                key=lambda item: item[1].artifact_id,
            )
        )
        bindings = tuple(binding for _value, binding in inputs)
        artifact_ids = tuple(binding.artifact_id for binding in bindings)
        materialization_ids = tuple(binding.materialization_id for binding in bindings)
        if len(set(artifact_ids)) != len(artifact_ids) or len(set(materialization_ids)) != len(
            materialization_ids
        ):
            raise SchedulerConsistencyError("frozen task inputs are not uniquely identified")
        budget = task.capability.requested_resources.source_scan_bytes
        if not inputs:
            return (), ()
        unauthorized = tuple(
            binding.artifact_id
            for value, binding in inputs
            if not self._input_is_authorized_for_execution(task, value)
        )
        if unauthorized:
            raise ResourceAdmissionError(
                "frozen input lacks read or outcome authority: " + ",".join(unauthorized)
            )
        if budget <= 0:
            raise ResourceAdmissionError("frozen inputs lack a positive source-scan budget")
        if self.input_port_factory is None:
            raise ResourceAdmissionError("bounded worker input ports are not composed")
        if sum(value.materialization.size_bytes for value, _binding in inputs) > budget:
            raise ResourceAdmissionError("frozen inputs exceed the source-scan budget")
        result = self.input_port_factory.open_many(
            tuple(value for value, _binding in inputs), bindings=bindings, maximum_bytes=budget
        )
        if tuple(port.binding for port in result) != bindings:
            for port in result:
                port.close()
            raise SchedulerConsistencyError("worker input factory returned a false lineage binding")
        return bindings, result

    def _input_is_authorized_for_execution(
        self,
        task: ProtocolExecutionTask,
        value: VerifiedArtifactInput,
    ) -> bool:
        return planned_input_access_allowed(
            task,
            value.logical.logical_artifact_id,
            value.logical.outcome_access,
            value.logical.visibility_ceiling,
            reveal_barrier_authorized=(
                task.barrier is BarrierKind.REVEAL
                and BarrierKind.REVEAL in self.authorized_barriers
            ),
        )

    @staticmethod
    def _input_is_authorized(task: ProtocolExecutionTask, value: VerifiedArtifactInput) -> bool:
        return planned_input_access_allowed(
            task,
            value.logical.logical_artifact_id,
            value.logical.outcome_access,
            value.logical.visibility_ceiling,
        )

    @staticmethod
    def _validate_output_profiles(task: ProtocolExecutionTask, result: RunnerResult) -> None:
        payloads = {
            output.output_id: output.payload
            for output in result.outputs
            if isinstance(output, TaskOutputPayload)
        }
        for spec in task.outputs:
            if spec.output_id not in payloads:
                continue
            if spec.profile is not ArtifactProfile.CANONICAL_JSON:
                continue
            try:
                decoded = json.loads(payloads[spec.output_id].decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise SchedulerConsistencyError(
                    "canonical JSON output is not valid UTF-8 JSON"
                ) from error
            try:
                canonical = (
                    json.dumps(
                        decoded,
                        allow_nan=False,
                        ensure_ascii=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                ).encode("utf-8")
            except ValueError as error:
                raise SchedulerConsistencyError(
                    "canonical JSON output contains a non-finite value"
                ) from error
            if canonical != payloads[spec.output_id]:
                raise SchedulerConsistencyError("canonical JSON output bytes are not canonical")

    def _validate_output_semantics(
        self,
        task: ProtocolExecutionTask,
        result: RunnerResult,
    ) -> None:
        if not self.output_semantic_contracts:
            return
        payloads = {
            output.output_id: output.payload
            for output in result.outputs
            if isinstance(output, TaskOutputPayload)
        }
        for spec in task.outputs:
            key = (
                task.capability.capability_key,
                task.capability.capability_version,
                spec.payload_schema,
                spec.profile,
            )
            contract = self.output_semantic_contracts.get(key)
            if contract is None:
                raise SchedulerConsistencyError(
                    "capability output lacks a registered semantic contract"
                )
            if contract.capability_implementation_sha256 != task.capability_implementation_sha256:
                raise SchedulerConsistencyError(
                    "capability semantic validator implementation differs from the frozen task"
                )
            if spec.output_id not in payloads:
                continue
            if contract.profile is not ArtifactProfile.CANONICAL_JSON:
                continue
            try:
                decoded = json.loads(payloads[spec.output_id].decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise SchedulerConsistencyError(
                    "capability semantic validator rejected the output"
                ) from error
            if not isinstance(decoded, dict) or tuple(sorted(decoded)) != (contract.top_level_keys):
                raise SchedulerConsistencyError(
                    "capability output document shape differs from its contract"
                )
            if decoded.get("schema") != contract.payload_schema:
                raise SchedulerConsistencyError(
                    "capability output schema differs from its semantic contract"
                )
            if contract.value_keys:
                value = decoded.get("value")
                if not isinstance(value, dict) or tuple(sorted(value)) != contract.value_keys:
                    raise SchedulerConsistencyError(
                        "capability output value shape differs from its contract"
                    )
                if decoded.get("version") != contract.record_version:
                    raise SchedulerConsistencyError(
                        "capability output record version differs from its contract"
                    )
            for field_name, expected in (
                (contract.capability_key_field, task.capability.capability_key),
                (contract.task_id_field, task.task_id),
                (contract.output_id_field, spec.output_id),
            ):
                if field_name is not None and decoded.get(field_name) != expected:
                    raise SchedulerConsistencyError(
                        "capability output context binding differs from its contract"
                    )

    def _validate_receipt(
        self,
        receipt: CanonicalTaskReceipt,
        run_id: str,
        task: ProtocolExecutionTask,
        attempt: OperationalAttempt,
        dependency_outputs: tuple[VerifiedArtifactInput, ...],
        external_inputs: tuple[VerifiedArtifactInput, ...],
    ) -> None:
        if (
            receipt.run_id != run_id
            or receipt.task_id != task.task_id
            or receipt.attempt_id != attempt.attempt_id
            or receipt.implementation_commit != self.implementation_commit
            or receipt.operational_status is not OperationalStatus.SUCCEEDED
        ):
            raise SchedulerConsistencyError("task receipt identity differs from execution")
        expected_inputs = self._input_materialization_ids(
            dependency_outputs,
            external_inputs,
        )
        if receipt.input_materialization_ids != expected_inputs:
            raise SchedulerConsistencyError("task receipt input lineage differs")
        expected_outputs = tuple(sorted(output.logical_artifact_id for output in task.outputs))
        observed_outputs = tuple(
            sorted(output.logical_artifact_id for output in receipt.output_materializations)
        )
        if expected_outputs != observed_outputs:
            raise SchedulerConsistencyError("task receipt output identity differs")
        if self.output_semantic_contracts:
            logical_by_id = {
                output.logical_artifact_id: output for output in receipt.output_logical_artifacts
            }
            for spec in task.outputs:
                contract = self.output_semantic_contracts.get(
                    (
                        task.capability.capability_key,
                        task.capability.capability_version,
                        spec.payload_schema,
                        spec.profile,
                    )
                )
                if contract is None:
                    raise SchedulerConsistencyError(
                        "task receipt lacks a registered semantic contract"
                    )
                if (
                    contract.capability_implementation_sha256
                    != task.capability_implementation_sha256
                ):
                    raise SchedulerConsistencyError(
                        "task receipt semantic validator differs from the frozen implementation"
                    )
                logical = logical_by_id.get(spec.logical_artifact_id)
                expected_validation = contract.artifact_validation(
                    task_id=task.task_id,
                    output_id=spec.output_id,
                )
                if logical is None or logical.semantic_validation != expected_validation:
                    raise SchedulerConsistencyError(
                        "task receipt semantic validation differs from the frozen contract"
                    )
        verify_artifact_manifests(
            self.artifact_writer,
            tuple(
                ArtifactManifest(
                    logical=output.logical,
                    materialization=output.materialization,
                )
                for output in (
                    *dependency_outputs,
                    *external_inputs,
                    *self._receipt_outputs(receipt),
                )
            ),
        )

    @staticmethod
    def _input_materialization_ids(
        dependency_outputs: tuple[VerifiedArtifactInput, ...],
        external_inputs: tuple[VerifiedArtifactInput, ...],
    ) -> tuple[str, ...]:
        return tuple(
            sorted(
                (
                    *(value.materialization.materialization_id for value in dependency_outputs),
                    *(value.materialization.materialization_id for value in external_inputs),
                )
            )
        )

    def _resolve_external_inputs(
        self,
        task: ProtocolExecutionTask,
    ) -> tuple[VerifiedArtifactInput, ...]:
        expected_ids = task.external_input_artifact_ids
        if not expected_ids:
            return ()
        if self.external_input_resolver is None:
            raise SchedulerConsistencyError(
                "task has external inputs but no verified input resolver"
            )
        resolved = self.external_input_resolver.resolve(expected_ids)
        observed_ids = tuple(input_.artifact_id for input_ in resolved)
        if observed_ids != expected_ids:
            raise SchedulerConsistencyError("resolved external inputs differ from the frozen task")
        specs = {value.logical_artifact_id: value for value in task.external_inputs}
        for input_ in resolved:
            spec = specs[input_.artifact_id]
            if (
                spec.expected_content_sha256 is not None
                and input_.logical.content_sha256 != spec.expected_content_sha256
            ):
                raise SchedulerConsistencyError(
                    "resolved external input content identity differs from the frozen task"
                )
            if (
                spec.expected_payload_schema is not None
                and input_.logical.payload_schema != spec.expected_payload_schema
            ):
                raise SchedulerConsistencyError(
                    "resolved external input schema differs from the frozen task"
                )
            if (
                spec.expected_media_type is not None
                and input_.logical.media_type != spec.expected_media_type
            ):
                raise SchedulerConsistencyError(
                    "resolved external input media type differs from the frozen task"
                )
            if (
                spec.expected_size_bytes is not None
                and input_.materialization.size_bytes != spec.expected_size_bytes
            ):
                raise SchedulerConsistencyError(
                    "resolved external input size differs from the frozen task"
                )
            if spec.identity_scope_sha256 is not None and not any(
                parent.identity.object_fingerprint == spec.identity_scope_sha256
                for parent in input_.logical.lineage_parents
            ):
                raise SchedulerConsistencyError(
                    "resolved external input enclosing-scope identity differs"
                )
            if (
                spec.expected_visibility_ceiling is not None
                and input_.logical.visibility_ceiling is not spec.expected_visibility_ceiling
            ):
                raise SchedulerConsistencyError(
                    "resolved external input visibility differs from the frozen task"
                )
            if (
                spec.expected_outcome_access is not None
                and input_.logical.outcome_access is not spec.expected_outcome_access
            ):
                raise SchedulerConsistencyError(
                    "resolved external input outcome access differs from the frozen task"
                )
        config = next(
            (
                input_
                for input_ in resolved
                if input_.artifact_id == task.capability.config.artifact_id
            ),
            None,
        )
        if config is None:
            raise SchedulerConsistencyError("task config artifact was not resolved")
        if (
            config.logical.content_sha256 != task.capability.config.content_sha256
            or config.logical.payload_schema != task.capability.config.config_schema
        ):
            raise SchedulerConsistencyError(
                "resolved config identity differs from the frozen capability config"
            )
        return resolved


class DurableExecutionEnvelopeStore(Protocol):
    """Externally durable compare-and-append authority for envelope events."""

    def create(self, envelope: RunExecutionEnvelope) -> None: ...

    def load(self, envelope_id: str) -> RunExecutionEnvelope: ...

    def compare_and_append(
        self,
        *,
        expected_envelope_sha256: str,
        updated: RunExecutionEnvelope,
    ) -> None: ...


class DurableExecutionEnvelopeCoordinator:
    """Persist each transition; reservation is a separate prelaunch commit."""

    def __init__(self, store: DurableExecutionEnvelopeStore) -> None:
        self._store = store

    def initialize(
        self,
        *,
        envelope_id: str,
        spec: ExecutionEnvelopeSpec,
        execution_plan: ObjectIdentity,
    ) -> RunExecutionEnvelope:
        envelope = RunExecutionEnvelopeMachine.initial(
            envelope_id=envelope_id,
            spec=spec,
            execution_plan=execution_plan,
        )
        self._store.create(envelope)
        return envelope

    def _current(self, envelope_id: str) -> RunExecutionEnvelope:
        current = self._store.load(envelope_id)
        if current.envelope_id != envelope_id:
            raise SchedulerConsistencyError(
                "durable execution-envelope store returned a substitution"
            )
        return current

    def load(self, envelope_id: str) -> RunExecutionEnvelope:
        """Load exact persisted state without applying an implicit transition."""

        return self._current(envelope_id)

    def ensure_initialized(
        self,
        *,
        envelope_id: str,
        spec: ExecutionEnvelopeSpec,
        execution_plan: ObjectIdentity,
    ) -> RunExecutionEnvelope:
        """Create once or validate the exact durable envelope on resume."""

        try:
            current = self._current(envelope_id)
        except (KeyError, FileNotFoundError):
            return self.initialize(
                envelope_id=envelope_id,
                spec=spec,
                execution_plan=execution_plan,
            )
        if current.spec != spec or current.execution_plan != execution_plan:
            raise SchedulerConsistencyError(
                "durable execution envelope differs from the issued plan"
            )
        return current

    def _persist(
        self,
        current: RunExecutionEnvelope,
        updated: RunExecutionEnvelope,
    ) -> RunExecutionEnvelope:
        if (
            updated.events[:-1] != current.events
            or updated.events[-1].sequence_number != len(current.events) + 1
        ):
            raise SchedulerConsistencyError("execution-envelope transition is not one append")
        self._store.compare_and_append(
            expected_envelope_sha256=current.fingerprint(),
            updated=updated,
        )
        return updated

    def reserve(
        self,
        *,
        envelope_id: str,
        event_id: str,
        occurred_at_utc: str,
        cell_id: str,
        attempt_id: str,
        retry_reason_code: str | None = None,
    ) -> RunExecutionEnvelope:
        current = self._current(envelope_id)
        return self._persist(
            current,
            RunExecutionEnvelopeMachine.reserve(
                current,
                event_id=event_id,
                occurred_at_utc=occurred_at_utc,
                cell_id=cell_id,
                attempt_id=attempt_id,
                retry_reason_code=retry_reason_code,
            ),
        )

    def launch(
        self,
        *,
        envelope_id: str,
        event_id: str,
        occurred_at_utc: str,
        cell_id: str,
        attempt_id: str,
    ) -> RunExecutionEnvelope:
        current = self._current(envelope_id)
        return self._persist(
            current,
            RunExecutionEnvelopeMachine.launch(
                current,
                event_id=event_id,
                occurred_at_utc=occurred_at_utc,
                cell_id=cell_id,
                attempt_id=attempt_id,
            ),
        )

    def observe_receipt(
        self,
        *,
        envelope_id: str,
        event_id: str,
        occurred_at_utc: str,
        cell_id: str,
        attempt_id: str,
        receipt: ObjectIdentity,
        valid_receipt: bool,
        scientific_terminal: bool,
        operational_reason_code: str | None = None,
    ) -> RunExecutionEnvelope:
        current = self._current(envelope_id)
        return self._persist(
            current,
            RunExecutionEnvelopeMachine.observe_receipt(
                current,
                event_id=event_id,
                occurred_at_utc=occurred_at_utc,
                cell_id=cell_id,
                attempt_id=attempt_id,
                receipt=receipt,
                valid_receipt=valid_receipt,
                scientific_terminal=scientific_terminal,
                operational_reason_code=operational_reason_code,
            ),
        )

    def observe_operational_failure(
        self,
        *,
        envelope_id: str,
        event_id: str,
        occurred_at_utc: str,
        cell_id: str,
        attempt_id: str,
        reason_code: str,
    ) -> RunExecutionEnvelope:
        current = self._current(envelope_id)
        return self._persist(
            current,
            RunExecutionEnvelopeMachine.observe_operational_failure(
                current,
                event_id=event_id,
                occurred_at_utc=occurred_at_utc,
                cell_id=cell_id,
                attempt_id=attempt_id,
                reason_code=reason_code,
            ),
        )

    def begin_recovery_publication(
        self,
        *,
        envelope_id: str,
        event_id: str,
        occurred_at_utc: str,
        cell_id: str,
    ) -> RunExecutionEnvelope:
        current = self._current(envelope_id)
        return self._persist(
            current,
            RunExecutionEnvelopeMachine.begin_recovery_publication(
                current,
                event_id=event_id,
                occurred_at_utc=occurred_at_utc,
                cell_id=cell_id,
            ),
        )

    def publish(
        self,
        *,
        envelope_id: str,
        event_id: str,
        occurred_at_utc: str,
        cell_id: str,
        artifact: ObjectIdentity,
        recovery: bool = False,
    ) -> RunExecutionEnvelope:
        current = self._current(envelope_id)
        return self._persist(
            current,
            RunExecutionEnvelopeMachine.publish(
                current,
                event_id=event_id,
                occurred_at_utc=occurred_at_utc,
                cell_id=cell_id,
                artifact=artifact,
                recovery=recovery,
            ),
        )

    def terminal(
        self,
        *,
        envelope_id: str,
        event_id: str,
        occurred_at_utc: str,
        cell_id: str,
    ) -> RunExecutionEnvelope:
        current = self._current(envelope_id)
        return self._persist(
            current,
            RunExecutionEnvelopeMachine.terminal(
                current,
                event_id=event_id,
                occurred_at_utc=occurred_at_utc,
                cell_id=cell_id,
            ),
        )

    def retain_duplicate_receipt(
        self,
        *,
        envelope_id: str,
        event_id: str,
        occurred_at_utc: str,
        cell_id: str,
        receipt: ObjectIdentity,
    ) -> RunExecutionEnvelope:
        current = self._current(envelope_id)
        return self._persist(
            current,
            RunExecutionEnvelopeMachine.retain_duplicate_receipt(
                current,
                event_id=event_id,
                occurred_at_utc=occurred_at_utc,
                cell_id=cell_id,
                receipt=receipt,
            ),
        )

    def stop(
        self,
        *,
        envelope_id: str,
        event_id: str,
        occurred_at_utc: str,
        cell_id: str,
        reason_code: str,
    ) -> RunExecutionEnvelope:
        current = self._current(envelope_id)
        return self._persist(
            current,
            RunExecutionEnvelopeMachine.stop(
                current,
                event_id=event_id,
                occurred_at_utc=occurred_at_utc,
                cell_id=cell_id,
                reason_code=reason_code,
            ),
        )

    def unknown_completion(
        self,
        *,
        envelope_id: str,
        event_id: str,
        occurred_at_utc: str,
        cell_id: str,
        reason_code: str,
    ) -> RunExecutionEnvelope:
        current = self._current(envelope_id)
        return self._persist(
            current,
            RunExecutionEnvelopeMachine.unknown_completion(
                current,
                event_id=event_id,
                occurred_at_utc=occurred_at_utc,
                cell_id=cell_id,
                reason_code=reason_code,
            ),
        )


class DurableExecutionResourceEnvelopeStore(Protocol):
    "Durable compare-and-append authority for the deadline-free prefix."

    def create(
        self,
        envelope: RunExecutionResourceEnvelope,
        *,
        jit_graph_signature_manifest: JitGraphSignatureManifest | None = None,
    ) -> None: ...

    def load(self, envelope_id: str) -> RunExecutionResourceEnvelope: ...

    def current(self, envelope_id: str) -> RunExecutionResourceEnvelope: ...

    def compare_and_append(
        self,
        *,
        expected: RunExecutionResourceEnvelope,
        updated: RunExecutionResourceEnvelope,
    ) -> None: ...


class DurableExecutionResourceEnvelopeCoordinator:
    """Persist the current resource transitions through the existing journal."""

    def __init__(self, store: DurableExecutionResourceEnvelopeStore) -> None:
        self._store = store

    def ensure_checkpoint(
        self,
        checkpoint: RunExecutionResourceEnvelope,
        *,
        jit_graph_signature_manifest: JitGraphSignatureManifest | None,
    ) -> RunExecutionResourceEnvelope:
        """Seed a distinct journal with an authenticated, replayable prefix."""
        try:
            current = self.current(checkpoint.envelope_id)
        except (KeyError, FileNotFoundError):
            self._store.create(
                checkpoint, jit_graph_signature_manifest=jit_graph_signature_manifest
            )
            return checkpoint
        if (
            current.spec != checkpoint.spec
            or current.execution_plan != checkpoint.execution_plan
            or current.events[: len(checkpoint.events)] != checkpoint.events
        ):
            raise SchedulerConsistencyError(
                "resource checkpoint conflicts with its retained prefix"
            )
        return current

    def initialize(
        self,
        *,
        envelope_id: str,
        spec: ExecutionResourceEnvelopeSpec,
        execution_plan: ObjectIdentity,
        jit_graph_signature_manifest: JitGraphSignatureManifest | None,
    ) -> RunExecutionResourceEnvelope:
        if (
            None
            if jit_graph_signature_manifest is None
            else ObjectIdentity.from_record(
                jit_graph_signature_manifest.manifest_id,
                jit_graph_signature_manifest,
            )
        ) != spec.jit_graph_signature_manifest:
            raise SchedulerConsistencyError("resource-envelope initialization JIT manifest differs")
        envelope = RunExecutionResourceEnvelopeMachine.initial(
            envelope_id=envelope_id,
            spec=spec,
            execution_plan=execution_plan,
        )
        self._store.create(envelope, jit_graph_signature_manifest=jit_graph_signature_manifest)
        return envelope

    def _current(self, envelope_id: str) -> RunExecutionResourceEnvelope:
        current = self._store.current(envelope_id)
        if current.envelope_id != envelope_id:
            raise SchedulerConsistencyError(
                "durable resource-envelope store returned a substitution"
            )
        return current

    def current(self, envelope_id: str) -> RunExecutionResourceEnvelope:
        return self._current(envelope_id)

    def load(self, envelope_id: str) -> RunExecutionResourceEnvelope:
        return self._store.load(envelope_id)

    def ensure_initialized(
        self,
        *,
        envelope_id: str,
        spec: ExecutionResourceEnvelopeSpec,
        execution_plan: ObjectIdentity,
        jit_graph_signature_manifest: JitGraphSignatureManifest | None,
    ) -> RunExecutionResourceEnvelope:
        try:
            current = self.load(envelope_id)
        except (KeyError, FileNotFoundError):
            return self.initialize(
                envelope_id=envelope_id,
                spec=spec,
                execution_plan=execution_plan,
                jit_graph_signature_manifest=jit_graph_signature_manifest,
            )
        if current.spec != spec or current.execution_plan != execution_plan:
            raise SchedulerConsistencyError(
                "durable resource envelope differs from the issued plan"
            )
        if (
            None
            if jit_graph_signature_manifest is None
            else ObjectIdentity.from_record(
                jit_graph_signature_manifest.manifest_id,
                jit_graph_signature_manifest,
            )
        ) != spec.jit_graph_signature_manifest:
            raise SchedulerConsistencyError("durable resource envelope JIT manifest differs")
        return current

    def _persist(
        self,
        current: RunExecutionResourceEnvelope,
        updated: RunExecutionResourceEnvelope,
    ) -> RunExecutionResourceEnvelope:
        if (
            updated.events[:-1] != current.events
            or updated.events[-1].sequence_number != len(current.events) + 1
        ):
            raise SchedulerConsistencyError("resource-envelope transition is not one append")
        self._store.compare_and_append(
            expected=current,
            updated=updated,
        )
        return updated

    def select_roster_branch(
        self,
        *,
        envelope_id: str,
        event_id: str,
        occurred_at_utc: str,
        decision: RosterCapacityDecision,
    ) -> RunExecutionResourceEnvelope:
        current = self._current(envelope_id)
        return self._persist(
            current,
            RunExecutionResourceEnvelopeMachine.select_roster_branch(
                current,
                event_id=event_id,
                occurred_at_utc=occurred_at_utc,
                decision=decision,
            ),
        )

    def reserve(
        self,
        *,
        envelope_id: str,
        event_id: str,
        occurred_at_utc: str,
        cell_id: str,
        attempt_id: str,
        retry_reason_code: str | None = None,
    ) -> RunExecutionResourceEnvelope:
        current = self._current(envelope_id)
        return self._persist(
            current,
            RunExecutionResourceEnvelopeMachine.reserve(
                current,
                event_id=event_id,
                occurred_at_utc=occurred_at_utc,
                cell_id=cell_id,
                attempt_id=attempt_id,
                retry_reason_code=retry_reason_code,
            ),
        )

    def launch(
        self,
        *,
        envelope_id: str,
        event_id: str,
        occurred_at_utc: str,
        cell_id: str,
        attempt_id: str,
        jit_projection: JitCellSignatureProjection | None,
        jit_graph_signature_manifest: JitGraphSignatureManifest | None,
    ) -> RunExecutionResourceEnvelope:
        current = self._current(envelope_id)
        return self._persist(
            current,
            RunExecutionResourceEnvelopeMachine.launch(
                current,
                event_id=event_id,
                occurred_at_utc=occurred_at_utc,
                cell_id=cell_id,
                attempt_id=attempt_id,
                jit_projection=jit_projection,
                jit_manifest=jit_graph_signature_manifest,
            ),
        )

    def heartbeat(
        self,
        *,
        envelope_id: str,
        event_id: str,
        occurred_at_utc: str,
        cell_id: str,
        heartbeat: ProgressHeartbeat,
        current: RunExecutionResourceEnvelope | None = None,
    ) -> RunExecutionResourceEnvelope:
        # The store compares this snapshot against its authenticated committed
        # prefix under the lock, so stale snapshots cannot overwrite another event.
        if current is None:
            current = self._current(envelope_id)
        if current.envelope_id != envelope_id:
            raise SchedulerConsistencyError("heartbeat snapshot envelope identity differs")
        return self._persist(
            current,
            RunExecutionResourceEnvelopeMachine.heartbeat(
                current,
                event_id=event_id,
                occurred_at_utc=occurred_at_utc,
                cell_id=cell_id,
                heartbeat=heartbeat,
            ),
        )

    def progress_stalled(
        self,
        *,
        envelope_id: str,
        event_id: str,
        occurred_at_utc: str,
        cell_id: str,
        attempt_id: str,
    ) -> RunExecutionResourceEnvelope:
        current = self._current(envelope_id)
        return self._persist(
            current,
            RunExecutionResourceEnvelopeMachine.progress_stalled(
                current,
                event_id=event_id,
                occurred_at_utc=occurred_at_utc,
                cell_id=cell_id,
                attempt_id=attempt_id,
            ),
        )

    def observe_receipt(
        self,
        *,
        envelope_id: str,
        event_id: str,
        occurred_at_utc: str,
        cell_id: str,
        attempt_id: str,
        receipt: ObjectIdentity,
        valid_receipt: bool,
        scientific_terminal: bool,
        operational_reason_code: str | None = None,
    ) -> RunExecutionResourceEnvelope:
        current = self._current(envelope_id)
        return self._persist(
            current,
            RunExecutionResourceEnvelopeMachine.observe_receipt(
                current,
                event_id=event_id,
                occurred_at_utc=occurred_at_utc,
                cell_id=cell_id,
                attempt_id=attempt_id,
                receipt=receipt,
                valid_receipt=valid_receipt,
                scientific_terminal=scientific_terminal,
                operational_reason_code=operational_reason_code,
            ),
        )

    def observe_operational_failure(
        self,
        *,
        envelope_id: str,
        event_id: str,
        occurred_at_utc: str,
        cell_id: str,
        attempt_id: str,
        reason_code: str,
    ) -> RunExecutionResourceEnvelope:
        current = self._current(envelope_id)
        return self._persist(
            current,
            RunExecutionResourceEnvelopeMachine.observe_operational_failure(
                current,
                event_id=event_id,
                occurred_at_utc=occurred_at_utc,
                cell_id=cell_id,
                attempt_id=attempt_id,
                reason_code=reason_code,
            ),
        )

    def begin_recovery_publication(
        self,
        *,
        envelope_id: str,
        event_id: str,
        occurred_at_utc: str,
        cell_id: str,
        artifact: ObjectIdentity,
        recovery_proof: ObjectIdentity,
    ) -> RunExecutionResourceEnvelope:
        current = self._current(envelope_id)
        return self._persist(
            current,
            RunExecutionResourceEnvelopeMachine.begin_recovery_publication(
                current,
                event_id=event_id,
                occurred_at_utc=occurred_at_utc,
                cell_id=cell_id,
                artifact=artifact,
                recovery_proof=recovery_proof,
            ),
        )

    def recover_artifact_before_receipt(
        self,
        *,
        envelope_id: str,
        event_id: str,
        occurred_at_utc: str,
        cell_id: str,
        attempt_id: str,
        receipt: ObjectIdentity,
        artifact: ObjectIdentity,
        recovery_proof: ObjectIdentity,
    ) -> RunExecutionResourceEnvelope:
        current = self._current(envelope_id)
        return self._persist(
            current,
            RunExecutionResourceEnvelopeMachine.recover_artifact_before_receipt(
                current,
                event_id=event_id,
                occurred_at_utc=occurred_at_utc,
                cell_id=cell_id,
                attempt_id=attempt_id,
                receipt=receipt,
                artifact=artifact,
                recovery_proof=recovery_proof,
            ),
        )

    def publish(
        self,
        *,
        envelope_id: str,
        event_id: str,
        occurred_at_utc: str,
        cell_id: str,
        artifact: ObjectIdentity | None = None,
        recovery: bool = False,
    ) -> RunExecutionResourceEnvelope:
        current = self._current(envelope_id)
        if not recovery and artifact is None:
            raise ValueError("ordinary resource-envelope publication requires an artifact")
        if recovery:
            updated = RunExecutionResourceEnvelopeMachine.finish_recovery_publication(
                current,
                event_id=event_id,
                occurred_at_utc=occurred_at_utc,
                cell_id=cell_id,
            )
        else:
            assert artifact is not None
            updated = RunExecutionResourceEnvelopeMachine.publish(
                current,
                event_id=event_id,
                occurred_at_utc=occurred_at_utc,
                cell_id=cell_id,
                artifact=artifact,
            )
        return self._persist(current, updated)

    def terminal(
        self,
        *,
        envelope_id: str,
        event_id: str,
        occurred_at_utc: str,
        cell_id: str,
    ) -> RunExecutionResourceEnvelope:
        current = self._current(envelope_id)
        return self._persist(
            current,
            RunExecutionResourceEnvelopeMachine.terminal(
                current,
                event_id=event_id,
                occurred_at_utc=occurred_at_utc,
                cell_id=cell_id,
            ),
        )

    def retain_duplicate_receipt(
        self,
        *,
        envelope_id: str,
        event_id: str,
        occurred_at_utc: str,
        cell_id: str,
        receipt: ObjectIdentity,
    ) -> RunExecutionResourceEnvelope:
        current = self._current(envelope_id)
        return self._persist(
            current,
            RunExecutionResourceEnvelopeMachine.retain_duplicate_receipt(
                current,
                event_id=event_id,
                occurred_at_utc=occurred_at_utc,
                cell_id=cell_id,
                receipt=receipt,
            ),
        )

    def stop(
        self,
        *,
        envelope_id: str,
        event_id: str,
        occurred_at_utc: str,
        cell_id: str,
        reason_code: str,
    ) -> RunExecutionResourceEnvelope:
        current = self._current(envelope_id)
        return self._persist(
            current,
            RunExecutionResourceEnvelopeMachine.stop(
                current,
                event_id=event_id,
                occurred_at_utc=occurred_at_utc,
                cell_id=cell_id,
                reason_code=reason_code,
            ),
        )

    def unknown_completion(
        self,
        *,
        envelope_id: str,
        event_id: str,
        occurred_at_utc: str,
        cell_id: str,
        reason_code: str,
    ) -> RunExecutionResourceEnvelope:
        current = self._current(envelope_id)
        return self._persist(
            current,
            RunExecutionResourceEnvelopeMachine.unknown_completion(
                current,
                event_id=event_id,
                occurred_at_utc=occurred_at_utc,
                cell_id=cell_id,
                reason_code=reason_code,
            ),
        )
