# SPDX-License-Identifier: MPL-2.0
# Adapted from the source project; synthetic software conformance only.
from __future__ import annotations

import hashlib
import json
import os
import resource
import threading
import time
from dataclasses import fields, replace
from pathlib import Path

import pytest
from sqlalchemy import func, select
from tests.runtime_platform.support import IMPLEMENTATION_COMMIT, ExplorationFixture, ProtocolFixture, budget, digest

from empirical_lawhood.adapters.exploration.conformance import _measure, _prepare
from empirical_lawhood.adapters.exploration.registry import SYNTHESIS_KEY, exploration_capability_keys, exploration_capability_registry
from empirical_lawhood.adapters.reference_worlds import ReferenceWorldKind, get_reference_world
from empirical_lawhood.infrastructure.artifacts import (
    ExternalArtifactPlane,
)
from empirical_lawhood.infrastructure.artifacts import (
    ArtifactIdentityConflict,
    FilesystemState,
    GuardedExternalRoot,
    STREAM_CHUNK_BYTES,
)
from empirical_lawhood.infrastructure.execution import (
    DirectTaskExecutor,
    ExecutionResourceCapacity,
    InjectedSchedulerCrash,
    LiveLeaseError,
    LocalProcessExecutor,
    LocalExecutionResourceAdmitter,
    LocalScheduler,
    ResourceAdmissionError,
    ResourceLockManager,
    SchedulerConsistencyError,
    TaskProcessError,
    _bounded_scratch_bytes,
)
from empirical_lawhood.infrastructure.recovery import ExternalRunRecoveryStore
from empirical_lawhood.infrastructure.sql import (
    SQLiteOperationalRepository,
    create_catalog_engine,
    upgrade_catalog,
)
from empirical_lawhood.infrastructure.sql.schema import run_event
from empirical_lawhood.infrastructure.task_receipts import (
    ExternalTaskReceiptStore,
    decode_artifact_manifest,
    decode_task_receipt,
)
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalizationError,
    CanonicalRecord,
    canonical_json_bytes,
)
from empirical_lawhood.kernel.status import OperationalStatus
from empirical_lawhood.planning.discovery import DiagnosticMeasureKind, ThresholdDirection
from empirical_lawhood.runtime.artifacts import (
    ArtifactLineageParent,
    ArtifactMaterialization,
    ArtifactProfile,
    ArtifactSemanticValidationRegistry,
    ArtifactWriteRequest,
    ExternalRootContract,
    ReceiptCheck,
    lineage_parent_sort_key,
)
from empirical_lawhood.runtime.capabilities import CapabilityManifest, CapabilityPermission
from empirical_lawhood.runtime.compiler import compile_exploration_execution_plan
from empirical_lawhood.runtime.execution import (
    ExecutionAssuranceProfile,
    RunnerRegistry,
    RunnerResult,
    StreamedTaskOutput,
    StreamingOutputEmitter,
    TaskAttemptDisposition,
    TaskBlockKind,
    TaskBlockReason,
    TaskContext,
    TaskExecutor,
    TaskOutputPayload,
    TaskRunner,
    VerifiedArtifactInput,
    WorkerInputBinding,
    WorkerInputKind,
    WorkerInputPort,
    WorkerIsolationProfile,
    WorkerOutputPort,
)
from empirical_lawhood.runtime.plans import BarrierKind, ProtocolExecutionPlan, PlanLane
from empirical_lawhood.runtime.providers import (
    CapabilityOutputSemanticContract,
    capability_semantic_validation_registry,
)
from empirical_lawhood.runtime.recovery import ProtocolRunRecoveryIndex


class StaticInspector:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()

    def inspect(self, _contract: ExternalRootContract) -> FilesystemState:
        return FilesystemState(
            canonical_root=str(self.path),
            active_mount=True,
            writable=True,
            free_bytes=100_000_000,
            path_is_symlink=False,
        )


class _BindingReader:
    bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        del size
        return b""

    def close(self) -> None:
        return None


def test_task_context_orders_inputs_by_artifact_not_materialization_identity(
    protocol_fixture: ProtocolFixture,
) -> None:
    task = next(
        value for value in protocol_fixture.execution_plan.tasks if value.task_id == "prepare"
    )
    bindings = (
        WorkerInputBinding(
            artifact_id="artifact.input-a",
            materialization_id="materialization.z-content-address",
            payload_schema='empirical-lawhood/testing/fixtures/input',
            media_type="application/json",
            size_bytes=0,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            kind=WorkerInputKind.EXTERNAL,
        ),
        WorkerInputBinding(
            artifact_id="artifact.input-b",
            materialization_id="materialization.a-content-address",
            payload_schema='empirical-lawhood/testing/fixtures/input',
            media_type="application/json",
            size_bytes=0,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            kind=WorkerInputKind.EXTERNAL,
        ),
    )

    context = TaskContext(
        run_id=protocol_fixture.run_plan.run_plan_id,
        task_id=task.task_id,
        attempt_id=f"{protocol_fixture.run_plan.run_plan_id}.{task.task_id}.attempt-001",
        config=task.capability.config,
        input_bindings=bindings,
        input_ports=tuple(
            WorkerInputPort(binding=value, _reader=_BindingReader()) for value in bindings
        ),
        output_ports=(),
        permissions=task.capability.required_permissions,
        outcome_access=task.capability.requested_outcome_access,
        resource_budget=task.capability.requested_resources,
        isolation_profile=WorkerIsolationProfile.TRUSTED_LOCAL,
    )

    assert context.input_materialization_ids == (
        "materialization.a-content-address",
        "materialization.z-content-address",
    )
    with pytest.raises(ValueError, match="unique materialization_id"):
        replace(
            context,
            input_bindings=(
                bindings[0],
                replace(bindings[1], materialization_id=bindings[0].materialization_id),
            ),
            input_ports=(
                WorkerInputPort(binding=bindings[0], _reader=_BindingReader()),
                WorkerInputPort(
                    binding=replace(
                        bindings[1],
                        materialization_id=bindings[0].materialization_id,
                    ),
                    _reader=_BindingReader(),
                ),
            ),
        )


def _plane(
    path: Path,
    semantic_validations: ArtifactSemanticValidationRegistry | None = None,
) -> ExternalArtifactPlane:
    contract = ExternalRootContract(
        storage_root_id='test-reference-external',
        logical_name="R5 synthetic external artifact root",
        canonical_path=str(path.resolve()),
        required_mount_path=str(path.resolve()),
        mount_contract_schema='empirical-lawhood/testing/fixtures/reference-world-pipeline/mount-contract',
        minimum_free_bytes=1,
    )
    return ExternalArtifactPlane(
        GuardedExternalRoot(contract, StaticInspector(path)),
        semantic_validations=semantic_validations,
    )


def _recovery(
    plane: ExternalArtifactPlane,
    plan: ProtocolExecutionPlan,
    *,
    authority: ObjectIdentity | None = None,
    wave_id: str = "wave.test",
) -> tuple[ProtocolRunRecoveryIndex, ExternalRunRecoveryStore]:
    authority_identity = authority or ObjectIdentity(
        object_id="authorization.synthetic-recovery",
        object_schema='empirical-lawhood/testing/fixtures/recovery-authority',
        object_version="1.0.0",
        object_fingerprint=digest("synthetic recovery authority"),
    )
    return (
        ProtocolRunRecoveryIndex.from_execution_plan(
            plan,
            run_id=plan.source_plan.object_id,
            wave_id=wave_id,
            authority_identities=(authority_identity,),
        ),
        ExternalRunRecoveryStore(plane),
    )


class DeterministicRunner:
    def __init__(self, manifest: CapabilityManifest, *, failures: int = 0) -> None:
        self.manifest = manifest
        self.failures = failures
        self.contexts: list[TaskContext] = []

    def execute(self, context: TaskContext) -> RunnerResult:
        self.contexts.append(context)
        if self.failures:
            self.failures -= 1
            raise OSError("injected transient runner failure")
        outputs = tuple(
            TaskOutputPayload(
                output_id=port.output_id,
                payload=canonical_json_bytes(
                    {
                        "capability": self.manifest.registry_id,
                        "external_input_artifact_ids": context.external_input_artifact_ids,
                        "input_materialization_ids": (context.dependency_input_materialization_ids),
                        "output_id": port.output_id,
                        "schema": port.payload_schema,
                        "task_id": context.task_id,
                    }
                ),
            )
            for port in context.output_ports
        )
        return RunnerResult(
            outputs=outputs,
            checks=(
                ReceiptCheck(
                    check_id="worker-contract-passed",
                    passed=True,
                    reason_codes=(),
                ),
            ),
        )


class LossyDecimalCustodyRunner(DeterministicRunner):
    """Exercise the actual new-custody guard through the scheduler boundary."""

    def execute(self, context: TaskContext) -> RunnerResult:
        from decimal import Decimal, localcontext
        from empirical_lawhood.adapters._decimal_custody import require_decimal_operands_preserved
        from tests.test_decimal_output_custody import DecimalOperandRecord

        self.contexts.append(context)
        original = DecimalOperandRecord(Decimal("1.23456789012345678901234567890123456789"))
        with localcontext() as arithmetic:
            arithmetic.prec = 28
            require_decimal_operands_preserved(
                original, original.canonical_bytes(), maximum_bytes=context.resource_budget.output_bytes,
            )
        raise AssertionError("lossy custody must refuse before a runner result")


class MaliciousPathProbeRunner(DeterministicRunner):
    """Probe every reachable worker-context value and open descriptor link."""

    def __init__(self, manifest: CapabilityManifest, *, forbidden_locator: str) -> None:
        super().__init__(manifest)
        self.forbidden_locator = forbidden_locator
        self.probed = False

    def execute(self, context: TaskContext) -> RunnerResult:
        pending: list[object] = [context]
        visited: set[int] = set()
        while pending:
            value = pending.pop()
            identity = id(value)
            if identity in visited:
                continue
            visited.add(identity)
            assert not isinstance(
                value,
                (ArtifactMaterialization, Path, VerifiedArtifactInput),
            )
            if isinstance(value, str):
                assert self.forbidden_locator not in value
            elif isinstance(value, dict):
                pending.extend(value.keys())
                pending.extend(value.values())
            elif isinstance(value, (tuple, list, set, frozenset)):
                pending.extend(value)
            elif hasattr(value, "__dataclass_fields__"):
                pending.extend(getattr(value, field.name) for field in fields(value))
            elif value.__class__.__module__.startswith(
                ("empirical_lawhood.", "multiprocessing.")
            ) and hasattr(value, "__dict__"):
                pending.extend(vars(value).values())
        for descriptor_name in os.listdir("/proc/self/fd"):
            try:
                target = os.readlink(f"/proc/self/fd/{descriptor_name}")
            except FileNotFoundError:
                continue
            assert self.forbidden_locator not in target
        self.probed = True
        return super().execute(context)


class RecoveringCapacityExecutor:
    """Test executor whose local capacity can be restored between resumes."""

    def __init__(self) -> None:
        self.capacity_available = False
        self._delegate = DirectTaskExecutor()

    def execute(
        self,
        runner: TaskRunner,
        context: TaskContext,
        *,
        timeout_seconds: int,
    ) -> RunnerResult:
        if not self.capacity_available:
            raise ResourceAdmissionError("injected local capacity unavailable")
        return self._delegate.execute(
            runner,
            context,
            timeout_seconds=timeout_seconds,
        )


class ConcurrencyObservingExecutor:
    """Test executor wrapper that measures overlap without changing results."""

    def __init__(self, delegate: TaskExecutor | None = None) -> None:
        self._delegate = delegate or DirectTaskExecutor()
        self._lock = threading.Lock()
        self._active_development = 0
        self.peak_development = 0
        self.development_affinities: dict[str, tuple[int, ...]] = {}
        self.errors: list[str] = []

    def execute(
        self,
        runner: TaskRunner,
        context: TaskContext,
        *,
        timeout_seconds: int,
    ) -> RunnerResult:
        development = context.task_id in {"develop-a", "develop-b"}
        if development:
            with self._lock:
                self._active_development += 1
                self.development_affinities[context.task_id] = context.cpu_affinity
                self.peak_development = max(
                    self.peak_development,
                    self._active_development,
                )
        try:
            if development:
                time.sleep(0.05)
            try:
                return self._delegate.execute(
                    runner,
                    context,
                    timeout_seconds=timeout_seconds,
                )
            except Exception as error:
                with self._lock:
                    self.errors.append(f"{context.task_id}:{type(error).__name__}:{error}")
                raise
        finally:
            if development:
                with self._lock:
                    self._active_development -= 1


class ProcessIdentityRunner:
    def __init__(self, manifest: CapabilityManifest, *, fail: bool = False) -> None:
        self.manifest = manifest
        self.fail = fail

    def execute(self, context: TaskContext) -> RunnerResult:
        if self.fail:
            raise RuntimeError("isolated child failure")
        return RunnerResult(
            outputs=tuple(
                TaskOutputPayload(
                    output_id=port.output_id,
                    payload=str(os.getpid()).encode("ascii"),
                )
                for port in context.output_ports
            ),
            checks=(ReceiptCheck("child-process-passed", True, ()),),
        )


class LargeLegacyOutputRunner:
    def __init__(self, manifest: CapabilityManifest) -> None:
        self.manifest = manifest

    def execute(self, context: TaskContext) -> RunnerResult:
        payload = b"x" * (STREAM_CHUNK_BYTES + 1)
        return RunnerResult(
            outputs=(
                TaskOutputPayload(
                    output_id=context.output_ports[0].output_id,
                    payload=payload,
                    logical_content_sha256=hashlib.sha256(payload).hexdigest(),
                ),
            ),
            checks=(ReceiptCheck("large-streamed-output-complete", True, ()),),
        )


class HardExitRunner:
    def __init__(self, manifest: CapabilityManifest) -> None:
        self.manifest = manifest

    def execute(self, context: TaskContext) -> RunnerResult:
        os._exit(7)


class ProcessBoundaryProbeRunner:
    def __init__(self, manifest: CapabilityManifest) -> None:
        self.manifest = manifest

    def execute(self, context: TaskContext) -> RunnerResult:
        payload = canonical_json_bytes(
            {
                "schema": 'empirical-lawhood/testing/fixtures/process-boundary-probe',
                "value": {
                    "affinity_count": len(os.sched_getaffinity(0)),
                    "cpu_limit": resource.getrlimit(resource.RLIMIT_CPU)[0],
                    "cwd": os.getcwd(),
                    "environment_keys": tuple(sorted(os.environ)),
                    "file_limit": resource.getrlimit(resource.RLIMIT_FSIZE)[0],
                    "memory_limit": resource.getrlimit(resource.RLIMIT_AS)[0],
                },
                "version": "1.0.0",
            }
        )
        return RunnerResult(
            outputs=(
                TaskOutputPayload(
                    output_id=context.output_ports[0].output_id,
                    payload=payload,
                ),
            ),
            checks=(ReceiptCheck("process-boundary-observed", True, ()),),
        )


class ScratchBudgetProbeRunner:
    def __init__(self, manifest: CapabilityManifest) -> None:
        self.manifest = manifest

    def execute(self, context: TaskContext) -> RunnerResult:
        Path("scratch-a.bin").write_bytes(b"a" * 600)
        Path("scratch-b.bin").write_bytes(b"b" * 600)
        return RunnerResult(
            outputs=(TaskOutputPayload(context.output_ports[0].output_id, b"x"),),
            checks=(ReceiptCheck("scratch-probe-complete", True, ()),),
        )


class ScratchEntryFloodRunner:
    def __init__(self, manifest: CapabilityManifest) -> None:
        self.manifest = manifest

    def execute(self, context: TaskContext) -> RunnerResult:
        for index in range(10_001):
            Path(f"empty-{index:05d}").touch()
        return RunnerResult(
            outputs=(TaskOutputPayload(context.output_ports[0].output_id, b"x"),),
            checks=(ReceiptCheck("scratch-entry-probe-complete", True, ()),),
        )


class DescendantProcessProbeRunner:
    def __init__(self, manifest: CapabilityManifest, *, detached: bool) -> None:
        self.manifest = manifest
        self.detached = detached

    def execute(self, context: TaskContext) -> RunnerResult:
        ready_read, ready_write = os.pipe()
        process_id = os.fork()
        if process_id == 0:
            try:
                os.close(ready_read)
                if self.detached:
                    os.setsid()
                os.write(ready_write, b"1")
                os.close(ready_write)
                time.sleep(0.75 if self.detached else 30)
            finally:
                os._exit(0)
        os.close(ready_write)
        try:
            if os.read(ready_read, 1) != b"1":
                raise RuntimeError("descendant process did not acknowledge its boundary")
        finally:
            os.close(ready_read)
        return RunnerResult(
            outputs=(
                TaskOutputPayload(
                    context.output_ports[0].output_id,
                    str(process_id).encode("ascii"),
                ),
            ),
            checks=(ReceiptCheck("descendant-process-started", True, ()),),
        )


def _process_is_running(process_id: int) -> bool:
    stat_path = Path(f"/proc/{process_id}/stat")
    try:
        fields = stat_path.read_text(encoding="ascii").split()
    except OSError:
        return False
    return len(fields) > 2 and fields[2] != "Z"


class GeneratedJsonlStreamingRunner:
    def __init__(
        self,
        manifest: CapabilityManifest,
        *,
        record_count: int,
        fail_after_records: int | None = None,
        emitted_schema: str | None = None,
        unsorted_checks: bool = False,
    ) -> None:
        self.manifest = manifest
        self.record_count = record_count
        self.fail_after_records = fail_after_records
        self.emitted_schema = emitted_schema
        self.unsorted_checks = unsorted_checks

    def execute(self, _context: TaskContext) -> RunnerResult:
        raise AssertionError("streaming runner used the in-memory execution surface")

    def execute_streaming(
        self,
        context: TaskContext,
        emitter: StreamingOutputEmitter,
    ) -> tuple[ReceiptCheck, ...]:
        output = context.output_ports[0]
        pending = bytearray()
        for sequence in range(self.record_count):
            pending.extend(
                canonical_json_bytes(
                    {
                        "schema": self.emitted_schema or output.payload_schema,
                        "sequence": sequence,
                        "value": "x" * 997,
                    }
                )
            )
            if len(pending) >= 64 * 1024:
                emitter.write(output.output_id, bytes(pending))
                pending.clear()
            if self.fail_after_records == sequence + 1:
                raise RuntimeError("injected mid-stream failure")
        if pending:
            emitter.write(output.output_id, bytes(pending))
        if self.unsorted_checks:
            return (
                ReceiptCheck("streaming-output-z-complete", True, ()),
                ReceiptCheck("streaming-output-a-bounded", True, ()),
            )
        return (ReceiptCheck("streaming-output-complete", True, ()),)


class _ControlledOutputSource:
    def __init__(self, payload: bytes, *, fail_midstream: bool) -> None:
        self.payload = payload
        self.fail_midstream = fail_midstream
        self.consumed = False
        self.closed = False

    def chunks(self, maximum_chunk_bytes: int):  # type: ignore[no-untyped-def]
        if self.consumed:
            raise RuntimeError("test output source was consumed twice")
        self.consumed = True
        split = min(maximum_chunk_bytes, max(1, len(self.payload) // 2))
        yield self.payload[:split]
        if self.fail_midstream:
            raise RuntimeError("injected second-output stream failure")
        for offset in range(split, len(self.payload), maximum_chunk_bytes):
            yield self.payload[offset : offset + maximum_chunk_bytes]

    def close(self) -> None:
        self.closed = True


class _TwoOutputRunner:
    def __init__(
        self,
        manifest: CapabilityManifest,
        *,
        failure_mode: str | None,
    ) -> None:
        self.manifest = manifest
        self.failure_mode = failure_mode
        self.sources: list[_ControlledOutputSource] = []

    def execute(self, context: TaskContext) -> RunnerResult:
        first, second = context.output_ports
        first_payload = canonical_json_bytes({"schema": first.payload_schema, "value": "first"})
        second_payload = canonical_json_bytes(
            (
                {"schema": second.payload_schema, "unexpected": "second"}
                if self.failure_mode == "semantic"
                else {"schema": second.payload_schema, "value": "second"}
            )
        )
        source = _ControlledOutputSource(
            second_payload,
            fail_midstream=self.failure_mode == "midstream",
        )
        self.sources.append(source)
        return RunnerResult(
            outputs=(
                TaskOutputPayload(first.output_id, first_payload),
                StreamedTaskOutput(
                    output_id=second.output_id,
                    size_bytes=len(second_payload),
                    physical_sha256=(
                        "0" * 64
                        if self.failure_mode == "sha256"
                        else hashlib.sha256(second_payload).hexdigest()
                    ),
                    source=source,
                ),
            ),
            checks=(ReceiptCheck("two-output-runner-complete", True, ()),),
        )


class _CrashAfterFirstFailedAttemptRepository(SQLiteOperationalRepository):
    def __init__(self, engine) -> None:  # type: ignore[no-untyped-def]
        super().__init__(engine)
        self.triggered = False

    def fail_attempt(self, attempt_id: str, reason_code: str) -> None:
        super().fail_attempt(attempt_id, reason_code)
        if not self.triggered:
            self.triggered = True
            raise InjectedSchedulerCrash("after-failed-attempt")


class CrashAfterReceipt:
    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.triggered = False

    def after_receipt_commit(
        self,
        _run_id: str,
        task_id: str,
        _attempt_id: str,
    ) -> None:
        if task_id == self.task_id and not self.triggered:
            self.triggered = True
            raise InjectedSchedulerCrash(task_id)


class FixedClock:
    def __init__(self, now: int) -> None:
        self.value = now

    def now(self) -> int:
        return self.value


class StaticExternalInputResolver:
    def __init__(self, inputs: tuple[VerifiedArtifactInput, ...]) -> None:
        self._inputs = {input_.artifact_id: input_ for input_ in inputs}

    def resolve(
        self,
        logical_artifact_ids: tuple[str, ...],
    ) -> tuple[VerifiedArtifactInput, ...]:
        return tuple(self._inputs[artifact_id] for artifact_id in logical_artifact_ids)


class AccessCountingResolver:
    def __init__(self, delegate: StaticExternalInputResolver) -> None:
        self.delegate = delegate
        self.calls: list[tuple[str, ...]] = []

    def resolve(
        self,
        logical_artifact_ids: tuple[str, ...],
    ) -> tuple[VerifiedArtifactInput, ...]:
        self.calls.append(logical_artifact_ids)
        return self.delegate.resolve(logical_artifact_ids)


class DriftedConfigResolver:
    def __init__(self, delegate: StaticExternalInputResolver) -> None:
        self.delegate = delegate

    def resolve(
        self,
        logical_artifact_ids: tuple[str, ...],
    ) -> tuple[VerifiedArtifactInput, ...]:
        resolved = self.delegate.resolve(logical_artifact_ids)
        config = resolved[0]
        return (
            VerifiedArtifactInput(
                replace(config.logical, content_sha256="0" * 64),
                config.materialization,
            ),
            *resolved[1:],
        )


class FalseLineageInputPortFactory:
    def __init__(self, delegate: ExternalArtifactPlane) -> None:
        self.delegate = delegate

    def open_many(self, values, *, bindings, maximum_bytes):
        return tuple(
            replace(
                port,
                binding=replace(port.binding, artifact_id=f"forged.{port.binding.artifact_id}"),
            )
            for port in self.delegate.open_many(
                values, bindings=bindings, maximum_bytes=maximum_bytes
            )
        )


def _external_inputs(
    plane: ExternalArtifactPlane,
    protocol: ProtocolFixture,
) -> StaticExternalInputResolver:
    configs = {
        task.capability.config.artifact_id: task.capability.config
        for task in protocol.execution_plan.tasks
    }
    artifact_ids = tuple(
        sorted(
            {
                artifact_id
                for task in protocol.execution_plan.tasks
                for artifact_id in task.external_input_artifact_ids
            }
        )
    )
    resolved = []
    for artifact_id in artifact_ids:
        config = configs.get(artifact_id)
        if config is None:
            payload = b"sealed outcomes"
            payload_schema = 'empirical-lawhood/reference-worlds/reference-adjudication-fixture'
            logical_content_sha256 = None
            outcome_access = OutcomeAccess.EVALUATION_SEALED
        else:
            payload = f"config payload for {task_key(config.artifact_id)}".encode()
            payload_schema = config.config_schema
            logical_content_sha256 = config.content_sha256
            outcome_access = OutcomeAccess.OUTCOME_BLIND
        if config is None:
            spec = next(
                value
                for task in protocol.execution_plan.tasks
                for value in task.external_inputs
                if value.logical_artifact_id == artifact_id
            )
            assert spec.identity_scope_sha256 is not None
            parent_identity = ObjectIdentity(
                object_id=spec.input_id,
                object_schema='empirical-lawhood/runtime/external-input-scope',
                object_version="1.0.0",
                object_fingerprint=spec.identity_scope_sha256,
            )
        else:
            parent_identity = ObjectIdentity.from_record(config.config_id, config)
        parent = ArtifactLineageParent(
            identity=parent_identity,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            outcome_access=outcome_access,
        )
        result = plane.write(
            ArtifactWriteRequest(
                logical_artifact_id=artifact_id,
                relative_path=f"inputs/{artifact_id}.txt",
                payload_schema=payload_schema,
                profile=ArtifactProfile.TEXT_PARAMETERS,
                media_type="text/plain",
                publication_scope_id="scheduler-external-inputs",
                publication_scope_relative_root="inputs",
                payload=payload,
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                parent_visibility_ceilings=(parent.visibility_ceiling,),
                outcome_access=outcome_access,
                logical_content_sha256=logical_content_sha256,
                lineage_parents=(parent,),
            )
        )
        resolved.append(VerifiedArtifactInput(result.logical, result.materialization))
    return StaticExternalInputResolver(tuple(resolved))


def task_key(config_artifact_id: str) -> str:
    return config_artifact_id.removeprefix("config-artifact.")


def _runtime(
    tmp_path: Path,
    protocol: ProtocolFixture,
    *,
    runner_failures: dict[str, int] | None = None,
    failure_injector: CrashAfterReceipt | None = None,
    clock: FixedClock | None = None,
    lock_manager: ResourceLockManager | None = None,
    executor: TaskExecutor | None = None,
    output_semantic_contracts: tuple[CapabilityOutputSemanticContract, ...] = (),
    assurance_profile: ExecutionAssuranceProfile = ExecutionAssuranceProfile.STRICT_ISOLATED,
    assurance_codes: tuple[str, ...] = (),
    maximum_parallel_tasks: int = 1,
    parallel_resource_capacity: ExecutionResourceCapacity | None = None,
    authorized_barriers: tuple[BarrierKind, ...] = (BarrierKind.REVEAL,),
) -> tuple[
    LocalScheduler,
    SQLiteOperationalRepository,
    ExternalArtifactPlane,
    ExternalTaskReceiptStore,
    tuple[DeterministicRunner, ...],
]:
    semantic_validations = (
        None
        if not output_semantic_contracts
        else capability_semantic_validation_registry(
            registry_id=(f"test-execution-semantics.{protocol.execution_plan.execution_plan_id}"),
            plan=protocol.execution_plan,
            contracts=output_semantic_contracts,
        )
    )
    plane = _plane(tmp_path, semantic_validations)
    engine = create_catalog_engine(f"sqlite+pysqlite:///{tmp_path / 'catalog.sqlite3'}")
    upgrade_catalog(engine)
    repository = SQLiteOperationalRepository(engine)
    failures = runner_failures or {}
    runners = tuple(
        DeterministicRunner(
            manifest,
            failures=failures.get(manifest.capability_key, 0),
        )
        for manifest in protocol.registry.capabilities
    )
    registry = RunnerRegistry(
        runners,
        registry_sha256=protocol.registry.fingerprint(),
    )
    receipts = ExternalTaskReceiptStore(plane)
    recovery_index, recovery_store = _recovery(
        plane,
        protocol.execution_plan,
        authority=protocol.run_plan.authorization,
    )
    input_resolver = _external_inputs(plane, protocol)
    scheduler = LocalScheduler(
        operational_repository=repository,
        artifact_writer=plane,
        receipt_store=receipts,
        recovery_index=recovery_index,
        recovery_store=recovery_store,
        runner_registry=registry,
        implementation_commit=IMPLEMENTATION_COMMIT,
        external_input_resolver=input_resolver,
        clock=clock,
        lock_manager=lock_manager,
        failure_injector=failure_injector,
        executor=executor or DirectTaskExecutor(),
        input_port_factory=plane,
        output_semantic_contracts=output_semantic_contracts,
        assurance_profile=assurance_profile,
        assurance_codes=assurance_codes,
        maximum_parallel_tasks=maximum_parallel_tasks,
        parallel_resource_capacity=parallel_resource_capacity,
        authorized_barriers=authorized_barriers,
    )
    return scheduler, repository, plane, receipts, runners


def _runner(runners: tuple[DeterministicRunner, ...], key: str) -> DeterministicRunner:
    return next(runner for runner in runners if runner.manifest.capability_key == key)


def _deterministic_semantic_contracts(
    protocol: ProtocolFixture,
) -> tuple[CapabilityOutputSemanticContract, ...]:
    top_level_keys = tuple(
        sorted(
            (
                "capability",
                "external_input_artifact_ids",
                "input_materialization_ids",
                "output_id",
                "schema",
                "task_id",
            )
        )
    )
    return tuple(
        sorted(
            (
                CapabilityOutputSemanticContract.from_manifest(
                    protocol.registry.resolve(
                        task.capability.capability_key,
                        task.capability.capability_version,
                    ),
                    payload_schema=output.payload_schema,
                    profile=output.profile,
                    top_level_keys=top_level_keys,
                    task_id_field="task_id",
                    output_id_field="output_id",
                )
                for task in protocol.execution_plan.tasks
                for output in task.outputs
            ),
            key=lambda contract: str(contract.key),
        )
    )


def _streaming_runtime(
    root: Path,
    protocol: ProtocolFixture,
    *,
    fail_after_records: int | None = None,
    emitted_schema: str | None = None,
    semantic_contract_schema: str | None = None,
) -> tuple[
    LocalScheduler,
    SQLiteOperationalRepository,
    ExternalArtifactPlane,
    ExternalTaskReceiptStore,
    ProtocolExecutionPlan,
]:
    root.mkdir(parents=True, exist_ok=True)
    engine = create_catalog_engine(f"sqlite+pysqlite:///{root / 'catalog.sqlite3'}")
    upgrade_catalog(engine)
    repository = SQLiteOperationalRepository(engine)
    original = next(task for task in protocol.execution_plan.tasks if task.task_id == "prepare")
    schema = 'empirical-lawhood/testing/fixtures/streaming-task-output'
    requested_resources = replace(
        original.capability.requested_resources,
        memory_bytes=512 * 1024**2,
        wall_time_seconds=10,
        output_bytes=1024**2,
    )
    task = replace(
        original,
        dependency_task_ids=(),
        external_inputs=(),
        maximum_attempts=1,
        capability=replace(
            original.capability,
            required_output_schema_ids=(schema,),
            requested_resources=requested_resources,
        ),
        outputs=(
            replace(
                original.outputs[0],
                logical_artifact_id="artifact.streaming-task-output",
                relative_path=(
                    f"runs/{protocol.execution_plan.source_plan.object_id}/streams/"
                    "streaming-task-output.jsonl"
                ),
                payload_schema=schema,
                profile=ArtifactProfile.JSONL_CHUNKS,
                media_type="application/x-ndjson",
            ),
        ),
    )
    original_manifest = protocol.registry.resolve(
        task.capability.capability_key,
        task.capability.capability_version,
    )
    manifest = replace(
        original_manifest,
        output_schema_ids=(schema,),
        resource_ceiling=requested_resources,
    )
    runner = GeneratedJsonlStreamingRunner(
        manifest,
        record_count=400,
        fail_after_records=fail_after_records,
        emitted_schema=emitted_schema,
    )
    plan = replace(
        protocol.execution_plan,
        execution_plan_id=(
            "execution-plan.streaming-task-failure"
            if fail_after_records is not None
            else "execution-plan.streaming-task-success"
        ),
        tasks=(task,),
    )
    contracts = (
        CapabilityOutputSemanticContract(
            capability_key=task.capability.capability_key,
            capability_version=task.capability.capability_version,
            capability_implementation_sha256=manifest.implementation_sha256,
            payload_schema=semantic_contract_schema or schema,
            profile=ArtifactProfile.JSONL_CHUNKS,
        ),
    )
    try:
        semantic_validations = capability_semantic_validation_registry(
            registry_id=f"test-execution-semantics.{plan.execution_plan_id}",
            plan=plan,
            contracts=contracts,
        )
    except ValueError:
        # The negative fixture deliberately gives the scheduler an unregistered
        # schema; it must reach the ordinary failed-attempt path without a write.
        semantic_validations = None
    plane = _plane(root, semantic_validations)
    receipts = ExternalTaskReceiptStore(plane)
    recovery_index, recovery_store = _recovery(plane, plan)
    scheduler = LocalScheduler(
        operational_repository=repository,
        artifact_writer=plane,
        receipt_store=receipts,
        recovery_index=recovery_index,
        recovery_store=recovery_store,
        runner_registry=RunnerRegistry(
            (runner,),
            registry_sha256=plan.registry_sha256,
        ),
        implementation_commit=IMPLEMENTATION_COMMIT,
        executor=LocalProcessExecutor(scratch_root=plane.root),
        input_port_factory=plane,
        output_semantic_contracts=contracts,
    )
    return scheduler, repository, plane, receipts, plan


def test_branched_scheduler_commits_receipts_then_replays_without_duplicate_work(
    tmp_path: Path,
    protocol_fixture: ProtocolFixture,
) -> None:
    scheduler, repository, _plane_value, _receipts, runners = _runtime(
        tmp_path,
        protocol_fixture,
    )
    run_id = protocol_fixture.run_plan.run_plan_id
    result = scheduler.execute(run_id, protocol_fixture.execution_plan)
    assert result.status is OperationalStatus.SUCCEEDED
    assert result.completed_task_ids == (
        "develop-a",
        "develop-b",
        "evaluate",
        "freeze",
        "prepare",
        "report",
    )
    assert len(repository.attempts(result.run_id)) == 6
    assert all(
        attempt.disposition is TaskAttemptDisposition.SUCCEEDED
        for attempt in repository.attempts(result.run_id)
    )
    assert all(len(runner.contexts) == 1 for runner in runners)
    assert _runner(runners, "reference.evaluate").contexts[0].external_input_artifact_ids == (
        "config-artifact.reference.evaluate",
        "reference-sealed-outcomes",
    )
    evaluation_attempt = next(
        attempt for attempt in repository.attempts(result.run_id) if attempt.task_id == "evaluate"
    )
    evaluation_receipt = _receipts.read(
        evaluation_attempt.run_id,
        evaluation_attempt.task_id,
        evaluation_attempt.attempt_id,
    )
    assert evaluation_receipt is not None
    evaluation_external_inputs = scheduler.external_input_resolver
    assert evaluation_external_inputs is not None
    expected_external_materializations = {
        input_.materialization.materialization_id
        for input_ in evaluation_external_inputs.resolve(
            ("config-artifact.reference.evaluate", "reference-sealed-outcomes")
        )
    }
    assert expected_external_materializations.issubset(evaluation_receipt.input_materialization_ids)

    context_fields = {field.name for field in fields(TaskContext)}
    assert context_fields.isdisjoint(
        {
            "artifact_writer",
            "database",
            "external_inputs",
            "input_materializations",
            "output_paths",
            "output_specs",
            "repository",
            "store",
        }
    )
    assert all(
        not hasattr(port, "relative_path")
        for runner in runners
        for context in runner.contexts
        for port in context.output_ports
    )
    with repository.engine.connect() as connection:
        event_count = connection.execute(select(func.count()).select_from(run_event)).scalar_one()
    attempt_ids = tuple(attempt.attempt_id for attempt in repository.attempts(result.run_id))
    receipt_files = tuple(sorted(tmp_path.glob(f"runs/{run_id}/receipts/**/*.json")))

    replayed = scheduler.execute(run_id, protocol_fixture.execution_plan)
    assert replayed == result
    assert all(len(runner.contexts) == 1 for runner in runners)
    assert (
        tuple(attempt.attempt_id for attempt in repository.attempts(result.run_id)) == attempt_ids
    )
    assert tuple(sorted(tmp_path.glob(f"runs/{run_id}/receipts/**/*.json"))) == (receipt_files)
    with repository.engine.connect() as connection:
        replay_event_count = connection.execute(
            select(func.count()).select_from(run_event)
        ).scalar_one()
    assert replay_event_count == event_count
    repository.engine.dispose()


def test_reveal_barrier_pauses_after_sealed_predecessors_and_resumes_same_plan(
    tmp_path: Path,
    protocol_fixture: ProtocolFixture,
) -> None:
    paused_scheduler, repository, _plane_value, _receipts, paused_runners = _runtime(
        tmp_path,
        protocol_fixture,
        authorized_barriers=(),
    )
    resolver = paused_scheduler.external_input_resolver
    assert isinstance(resolver, StaticExternalInputResolver)
    access_counter = AccessCountingResolver(resolver)
    paused_scheduler.external_input_resolver = access_counter
    run_id = protocol_fixture.run_plan.run_plan_id
    paused = paused_scheduler.execute(run_id, protocol_fixture.execution_plan)

    assert paused.status is OperationalStatus.BLOCKED
    assert paused.completed_task_ids == ("develop-a", "develop-b", "freeze", "prepare")
    assert paused.blocked_task_ids == ("evaluate", "report")
    evaluate_attempt = next(
        value for value in repository.attempts(run_id) if value.task_id == "evaluate"
    )
    assert evaluate_attempt.disposition is TaskAttemptDisposition.BLOCKED
    assert evaluate_attempt.reason_code == TaskBlockReason.AUTHORITY_REQUIRED.value
    assert len(_runner(paused_runners, "reference.evaluate").contexts) == 0
    assert all(
        "reference-sealed-outcomes" not in artifact_ids for artifact_ids in access_counter.calls
    )
    assert not tuple(tmp_path.glob(f"runs/{run_id}/recovery/**/terminal*.json"))
    repository.engine.dispose()

    resumed_scheduler, resumed_repository, _plane_value, _receipts, resumed_runners = _runtime(
        tmp_path,
        protocol_fixture,
        authorized_barriers=(BarrierKind.REVEAL,),
    )
    resumed = resumed_scheduler.execute(run_id, protocol_fixture.execution_plan)

    assert resumed.status is OperationalStatus.SUCCEEDED
    assert resumed.completed_task_ids == tuple(
        value.task_id for value in protocol_fixture.execution_plan.tasks
    )
    assert len(_runner(resumed_runners, "reference.evaluate").contexts) == 1
    assert sum(len(value.contexts) for value in resumed_runners) == 2
    assert tuple(
        value.attempt_id
        for value in resumed_repository.attempts(run_id)
        if value.disposition is TaskAttemptDisposition.SUCCEEDED
    )
    resumed_repository.engine.dispose()


def test_malicious_runner_cannot_reconstruct_a_sealed_input_locator(
    tmp_path: Path,
    protocol_fixture: ProtocolFixture,
) -> None:
    scheduler, repository, _plane_value, _receipts, runners = _runtime(
        tmp_path,
        protocol_fixture,
    )
    resolver = scheduler.external_input_resolver
    assert resolver is not None
    sealed = resolver.resolve(("reference-sealed-outcomes",))[0]
    evaluator = protocol_fixture.registry.resolve("reference.evaluate", "1.0.0")
    probe = MaliciousPathProbeRunner(
        evaluator,
        forbidden_locator=sealed.materialization.relative_path,
    )
    composed_runners = tuple(
        probe if runner.manifest.capability_key == "reference.evaluate" else runner
        for runner in runners
    )
    scheduler.runner_registry = RunnerRegistry(
        composed_runners,
        registry_sha256=protocol_fixture.registry.fingerprint(),
    )

    result = scheduler.execute(
        protocol_fixture.run_plan.run_plan_id,
        protocol_fixture.execution_plan,
    )

    assert result.status is OperationalStatus.SUCCEEDED
    assert probe.probed
    repository.engine.dispose()


def test_scheduler_rejects_run_and_registry_identity_drift(
    tmp_path: Path,
    protocol_fixture: ProtocolFixture,
) -> None:
    scheduler, repository, _plane_value, _receipts, _runners = _runtime(
        tmp_path,
        protocol_fixture,
    )
    with pytest.raises(SchedulerConsistencyError, match="run ID differs"):
        scheduler.execute("different-run", protocol_fixture.execution_plan)
    with pytest.raises(SchedulerConsistencyError, match="runner registry differs"):
        scheduler.execute(
            protocol_fixture.run_plan.run_plan_id,
            replace(protocol_fixture.execution_plan, registry_sha256="0" * 64),
        )
    repository.engine.dispose()


def test_scheduler_requires_verified_external_inputs_and_exact_config_identity(
    tmp_path: Path,
    protocol_fixture: ProtocolFixture,
) -> None:
    scheduler, repository, _plane_value, _receipts, _runners = _runtime(
        tmp_path,
        protocol_fixture,
    )
    resolver = scheduler.external_input_resolver
    assert isinstance(resolver, StaticExternalInputResolver)
    scheduler.external_input_resolver = None
    with pytest.raises(SchedulerConsistencyError, match="no verified input resolver"):
        scheduler.execute(
            protocol_fixture.run_plan.run_plan_id,
            protocol_fixture.execution_plan,
        )
    assert repository.attempts(protocol_fixture.run_plan.run_plan_id) == ()

    scheduler.external_input_resolver = DriftedConfigResolver(resolver)
    with pytest.raises(SchedulerConsistencyError, match="content identity"):
        scheduler.execute(
            protocol_fixture.run_plan.run_plan_id,
            protocol_fixture.execution_plan,
        )
    assert repository.attempts(protocol_fixture.run_plan.run_plan_id) == ()
    repository.engine.dispose()


def test_scheduler_replay_rejects_external_input_byte_drift(
    tmp_path: Path,
    protocol_fixture: ProtocolFixture,
) -> None:
    scheduler, repository, _plane_value, _receipts, _runners = _runtime(
        tmp_path,
        protocol_fixture,
    )
    run_id = protocol_fixture.run_plan.run_plan_id
    assert scheduler.execute(run_id, protocol_fixture.execution_plan).status is (
        OperationalStatus.SUCCEEDED
    )
    resolver = scheduler.external_input_resolver
    assert isinstance(resolver, StaticExternalInputResolver)
    prepare_input = resolver.resolve(("config-artifact.reference.prepare",))[0]
    (tmp_path / prepare_input.materialization.relative_path).write_bytes(b"drift")
    with pytest.raises(ArtifactIdentityConflict, match="artifact (?:bytes|size).+identity"):
        scheduler.execute(run_id, protocol_fixture.execution_plan)
    repository.engine.dispose()


def test_exploration_plan_crash_resume_is_read_only_and_nonpromotable(
    tmp_path: Path,
    exploration_fixture: ExplorationFixture,
) -> None:
    execution = compile_exploration_execution_plan(
        exploration_plan=exploration_fixture.plan,
        snapshot=exploration_fixture.snapshot,
        snapshot_verification=exploration_fixture.verification,
        registry=exploration_fixture.registry,
        implementation_commit=IMPLEMENTATION_COMMIT,
        synthesis_capability_key="exploration.synthesis",
        synthesis_capability_version="1.0.0",
        synthesis_budget=budget(source_scan_bytes=10_000),
    )
    plane = _plane(tmp_path)
    records: dict[str, CanonicalRecord] = {
        f"analysis-spec.{proposal.analysis.analysis_id}": proposal.analysis
        for proposal in exploration_fixture.plan.proposals
    }
    records[f"exploration-plan.{exploration_fixture.plan.plan_id}"] = exploration_fixture.plan
    resolved_inputs = []
    scope_parent = ArtifactLineageParent(
        identity=ObjectIdentity.from_record(
            exploration_fixture.snapshot.snapshot_id,
            exploration_fixture.snapshot,
        ),
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
    )
    for artifact_id, record in sorted(records.items()):
        parent = ArtifactLineageParent(
            identity=ObjectIdentity.from_record(artifact_id, record),
            visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
            outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        )
        parents = tuple(sorted((parent, scope_parent), key=lineage_parent_sort_key))
        result = plane.write(
            ArtifactWriteRequest(
                logical_artifact_id=artifact_id,
                relative_path=f"inputs/{artifact_id}.json",
                payload_schema=record.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type="application/json",
                publication_scope_id="scheduler-exploration-inputs",
                publication_scope_relative_root="inputs",
                payload=record.canonical_bytes(),
                visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
                parent_visibility_ceilings=tuple(value.visibility_ceiling for value in parents),
                outcome_access=OutcomeAccess.EVALUATION_REVEALED,
                logical_content_sha256=record.fingerprint(),
                lineage_parents=parents,
            )
        )
        resolved_inputs.append(VerifiedArtifactInput(result.logical, result.materialization))
    source = exploration_fixture.snapshot.artifacts[0]
    source_parent = ArtifactLineageParent(
        identity=ObjectIdentity.from_record(source.artifact_id, source),
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
    )
    source_parents = tuple(sorted((source_parent, scope_parent), key=lineage_parent_sort_key))
    source_result = plane.write(
        ArtifactWriteRequest(
            logical_artifact_id=source.artifact_id,
            relative_path=f"inputs/{source.artifact_id}.txt",
            payload_schema=source.payload_schema,
            profile=ArtifactProfile.TEXT_PARAMETERS,
            media_type=source.media_type,
            publication_scope_id="scheduler-exploration-inputs",
            publication_scope_relative_root="inputs",
            payload=b'synthetic source data',
            visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
            parent_visibility_ceilings=tuple(value.visibility_ceiling for value in source_parents),
            outcome_access=OutcomeAccess.EVALUATION_REVEALED,
            logical_content_sha256=source.sha256,
            lineage_parents=source_parents,
        )
    )
    assert source_result.materialization.size_bytes == source.size_bytes
    resolved_inputs.append(
        VerifiedArtifactInput(source_result.logical, source_result.materialization)
    )
    resolver = StaticExternalInputResolver(
        tuple(sorted(resolved_inputs, key=lambda value: value.artifact_id))
    )
    engine = create_catalog_engine(f"sqlite+pysqlite:///{tmp_path / 'catalog.sqlite3'}")
    upgrade_catalog(engine)
    repository = SQLiteOperationalRepository(engine)
    runners = tuple(
        DeterministicRunner(manifest) for manifest in exploration_fixture.registry.capabilities
    )
    registry = RunnerRegistry(
        runners,
        registry_sha256=exploration_fixture.registry.fingerprint(),
    )
    receipts = ExternalTaskReceiptStore(plane)
    recovery_index, recovery_store = _recovery(plane, execution)
    crash = CrashAfterReceipt('explore.synthetic-exploration-analysis-1')
    scheduler = LocalScheduler(
        operational_repository=repository,
        artifact_writer=plane,
        receipt_store=receipts,
        recovery_index=recovery_index,
        recovery_store=recovery_store,
        runner_registry=registry,
        implementation_commit=IMPLEMENTATION_COMMIT,
        external_input_resolver=resolver,
        failure_injector=crash,
        executor=DirectTaskExecutor(),
        input_port_factory=plane,
        lane=execution.lane,
    )
    with pytest.raises(InjectedSchedulerCrash, match='explore.synthetic-exploration-analysis-1'):
        scheduler.execute(exploration_fixture.plan.plan_id, execution)
    scheduler_result = LocalScheduler(
        operational_repository=repository,
        artifact_writer=plane,
        receipt_store=receipts,
        recovery_index=recovery_index,
        recovery_store=recovery_store,
        runner_registry=registry,
        implementation_commit=IMPLEMENTATION_COMMIT,
        external_input_resolver=resolver,
        executor=DirectTaskExecutor(),
        input_port_factory=plane,
        lane=execution.lane,
    ).execute(exploration_fixture.plan.plan_id, execution)
    assert scheduler_result.status is OperationalStatus.SUCCEEDED
    assert scheduler_result.completed_task_ids == (
        "exploration-synthesis",
        'explore.synthetic-exploration-analysis-1',
    )
    assert tuple(len(runner.contexts) for runner in runners) == (1, 1)
    assert all(not task.outputs[0].visibility_ceiling.is_promotable for task in execution.tasks)
    assert all(
        input_.visibility_ceiling is VisibilityCeiling.OUTCOME_VISIBLE
        for runner in runners
        for context in runner.contexts
        for input_ in context.input_bindings
        if input_.kind is WorkerInputKind.EXTERNAL
    )
    engine.dispose()


@pytest.mark.parametrize("declared_version", ("1.0.0", "2.0.0"))
@pytest.mark.parametrize("emitted_version", ("1.0.0", "2.0.0"))
def test_inline_canonical_output_uses_declared_record_version(
    tmp_path: Path,
    protocol_fixture: ProtocolFixture,
    monkeypatch: pytest.MonkeyPatch,
    declared_version: str,
    emitted_version: str,
) -> None:
    def execute(self: DeterministicRunner, context: TaskContext) -> RunnerResult:
        return RunnerResult(
            tuple(
                TaskOutputPayload(
                    port.output_id,
                    canonical_json_bytes(
                        {
                            "schema": port.payload_schema,
                            "version": emitted_version,
                            "value": {"marker": "synthetic-version-contract"},
                        }
                    ),
                )
                for port in context.output_ports
            ),
            (ReceiptCheck("version-fixture", True, ()),),
        )

    monkeypatch.setattr(DeterministicRunner, "execute", execute)
    contracts = tuple(
        sorted(
            (
                CapabilityOutputSemanticContract.from_manifest(
                    protocol_fixture.registry.resolve(
                        task.capability.capability_key, task.capability.capability_version
                    ),
                    payload_schema=output.payload_schema,
                    profile=output.profile,
                    top_level_keys=("schema", "value", "version"),
                    value_keys=("marker",),
                    record_version=declared_version,
                )
                for task in protocol_fixture.execution_plan.tasks
                for output in task.outputs
            ),
            key=lambda contract: str(contract.key),
        )
    )
    scheduler, repository, _, _, _ = _runtime(
        tmp_path, protocol_fixture, output_semantic_contracts=contracts
    )
    try:
        result = scheduler.execute(
            protocol_fixture.run_plan.run_plan_id, protocol_fixture.execution_plan
        )
        assert result.status is (
            OperationalStatus.SUCCEEDED
            if declared_version == emitted_version
            else OperationalStatus.FAILED
        )
        if declared_version != emitted_version:
            first = next(
                t
                for t in protocol_fixture.execution_plan.tasks
                if t.task_id in result.failed_task_ids
            )
            assert all(not tmp_path.joinpath(o.relative_path).exists() for o in first.outputs)
    finally:
        repository.engine.dispose()


def test_capability_specific_semantic_shape_fails_before_output_commit(
    tmp_path: Path,
    protocol_fixture: ProtocolFixture,
) -> None:
    contracts = tuple(
        sorted(
            (
                CapabilityOutputSemanticContract(
                    capability_key=task.capability.capability_key,
                    capability_version=task.capability.capability_version,
                    capability_implementation_sha256=protocol_fixture.registry.resolve(
                        task.capability.capability_key,
                        task.capability.capability_version,
                    ).implementation_sha256,
                    payload_schema=output.payload_schema,
                    profile=output.profile,
                    top_level_keys=tuple(
                        sorted(
                            (
                                "capability",
                                "external_input_artifact_ids",
                                "input_materialization_ids",
                                "output_id",
                                "required_scientific_field",
                                "schema",
                                "task_id",
                            )
                        )
                    ),
                    task_id_field="task_id",
                    output_id_field="output_id",
                )
                for task in protocol_fixture.execution_plan.tasks
                for output in task.outputs
            ),
            key=lambda contract: str(contract.key),
        )
    )
    scheduler, repository, _plane_value, _receipts, _runners = _runtime(
        tmp_path,
        protocol_fixture,
        output_semantic_contracts=contracts,
    )

    result = scheduler.execute(
        protocol_fixture.run_plan.run_plan_id,
        protocol_fixture.execution_plan,
    )

    assert result.status is OperationalStatus.FAILED
    assert result.failed_task_ids
    first_failed = next(
        task
        for task in protocol_fixture.execution_plan.tasks
        if task.task_id in result.failed_task_ids
    )
    assert all(
        not tmp_path.joinpath(output.relative_path).exists() for output in first_failed.outputs
    )
    repository.engine.dispose()


@pytest.mark.parametrize(
    ("field_name", "replacement", "reason"),
    (
        ("capability_key", "forged.semantic-validator", "lacks a registered"),
        ("capability_version", "9.9.9", "lacks a registered"),
        (
            "capability_implementation_sha256",
            "0" * 64,
            "semantic validator differs",
        ),
    ),
)
def test_semantic_validator_identity_substitution_fails_receipt_replay(
    tmp_path: Path,
    protocol_fixture: ProtocolFixture,
    field_name: str,
    replacement: str,
    reason: str,
) -> None:
    contracts = _deterministic_semantic_contracts(protocol_fixture)
    scheduler, repository, plane, receipts, runners = _runtime(
        tmp_path,
        protocol_fixture,
        output_semantic_contracts=contracts,
    )
    run_id = protocol_fixture.run_plan.run_plan_id
    assert scheduler.execute(run_id, protocol_fixture.execution_plan).status is (
        OperationalStatus.SUCCEEDED
    )

    substituted = tuple(
        sorted(
            (
                replace(contracts[0], **{field_name: replacement}),
                *contracts[1:],
            ),
            key=lambda contract: str(contract.key),
        )
    )
    resumed = LocalScheduler(
        operational_repository=repository,
        artifact_writer=plane,
        receipt_store=receipts,
        recovery_index=scheduler.recovery_index,
        recovery_store=scheduler.recovery_store,
        runner_registry=RunnerRegistry(
            runners,
            registry_sha256=protocol_fixture.registry.fingerprint(),
        ),
        implementation_commit=IMPLEMENTATION_COMMIT,
        external_input_resolver=_external_inputs(plane, protocol_fixture),
        executor=DirectTaskExecutor(),
        input_port_factory=plane,
        output_semantic_contracts=substituted,
    )

    with pytest.raises(SchedulerConsistencyError, match=reason):
        resumed.execute(run_id, protocol_fixture.execution_plan)

    repository.engine.dispose()


def test_exploration_compiled_wave_crash_reconciles_without_duplicate_analysis(
    tmp_path: Path,
) -> None:
    world = get_reference_world(ReferenceWorldKind.PLANTED_RELATIONAL_ANOMALY)
    prepared = _prepare(
        case_id="runtime-crash-reconcile",
        kind=ReferenceWorldKind.PLANTED_RELATIONAL_ANOMALY,
        measures=(
            _measure(
                world,
                "runtime-coordinate-instability",
                DiagnosticMeasureKind.COORDINATE_INSTABILITY,
                "0.8",
                "0.2",
                ThresholdDirection.ABOVE_MAXIMUM,
                12,
            ),
        ),
        axis_overrides={"representation": ("native",), "transform": ("native",)},
    )
    plan = prepared.planned.plan
    assert plan is not None
    implementation_hashes = {key: digest(key) for key in exploration_capability_keys()}
    capability_registry = exploration_capability_registry(implementation_hashes)
    execution = compile_exploration_execution_plan(
        exploration_plan=plan,
        snapshot=prepared.snapshot,
        snapshot_verification=prepared.verification,
        registry=capability_registry,
        implementation_commit=IMPLEMENTATION_COMMIT,
        synthesis_capability_key=SYNTHESIS_KEY,
        synthesis_capability_version="1.0.0",
        synthesis_budget=budget(source_scan_bytes=1_000_000),
    )
    analysis_task = next(task for task in execution.tasks if task.stage.value == "EXPLORE")
    synthesis_task = next(task for task in execution.tasks if task.stage.value == "SYNTHESIZE")
    assert synthesis_task.capability.kind.value == "HYPOTHESIS_SYNTHESIZER"
    assert synthesis_task.dependency_task_ids == (analysis_task.task_id,)

    plane = _plane(tmp_path)
    records: dict[str, CanonicalRecord] = {
        f"analysis-spec.{proposal.analysis.analysis_id}": proposal.analysis
        for proposal in plan.proposals
    }
    records[f"exploration-plan.{plan.plan_id}"] = plan
    resolved_inputs = []
    scope_parent = ArtifactLineageParent(
        identity=ObjectIdentity.from_record(
            prepared.snapshot.snapshot_id,
            prepared.snapshot,
        ),
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
    )
    for index, (artifact_id, record) in enumerate(sorted(records.items()), start=1):
        parent = ArtifactLineageParent(
            identity=ObjectIdentity.from_record(artifact_id, record),
            visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
            outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        )
        parents = tuple(sorted((parent, scope_parent), key=lineage_parent_sort_key))
        result = plane.write(
            ArtifactWriteRequest(
                logical_artifact_id=artifact_id,
                relative_path=f"inputs/exploration-config-{index}.json",
                payload_schema=record.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type="application/json",
                publication_scope_id="scheduler-exploration-inputs",
                publication_scope_relative_root="inputs",
                payload=record.canonical_bytes(),
                visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
                parent_visibility_ceilings=tuple(value.visibility_ceiling for value in parents),
                outcome_access=OutcomeAccess.EVALUATION_REVEALED,
                logical_content_sha256=record.fingerprint(),
                lineage_parents=parents,
            )
        )
        resolved_inputs.append(VerifiedArtifactInput(result.logical, result.materialization))
    source = prepared.snapshot.artifacts[0]
    source_parent = ArtifactLineageParent(
        identity=ObjectIdentity.from_record(source.artifact_id, source),
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
    )
    source_parents = tuple(sorted((source_parent, scope_parent), key=lineage_parent_sort_key))
    source_result = plane.write(
        ArtifactWriteRequest(
            logical_artifact_id=source.artifact_id,
            relative_path=f"inputs/{source.artifact_id}.json",
            payload_schema=source.payload_schema,
            profile=ArtifactProfile.CANONICAL_JSON,
            media_type=source.media_type,
            publication_scope_id="scheduler-exploration-inputs",
            publication_scope_relative_root="inputs",
            payload=prepared.projection.canonical_bytes(),
            visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
            parent_visibility_ceilings=tuple(value.visibility_ceiling for value in source_parents),
            outcome_access=OutcomeAccess.EVALUATION_REVEALED,
            logical_content_sha256=source.sha256,
            lineage_parents=source_parents,
        )
    )
    resolved_inputs.append(
        VerifiedArtifactInput(source_result.logical, source_result.materialization)
    )
    resolver = StaticExternalInputResolver(
        tuple(sorted(resolved_inputs, key=lambda value: value.artifact_id))
    )
    engine = create_catalog_engine(f"sqlite+pysqlite:///{tmp_path / 'catalog.sqlite3'}")
    upgrade_catalog(engine)
    repository = SQLiteOperationalRepository(engine)
    runners = tuple(DeterministicRunner(manifest) for manifest in capability_registry.capabilities)
    runner_registry = RunnerRegistry(runners, registry_sha256=capability_registry.fingerprint())
    receipts = ExternalTaskReceiptStore(plane)
    recovery_index, recovery_store = _recovery(plane, execution)
    crash = CrashAfterReceipt(analysis_task.task_id)
    crashing_scheduler = LocalScheduler(
        operational_repository=repository,
        artifact_writer=plane,
        receipt_store=receipts,
        recovery_index=recovery_index,
        recovery_store=recovery_store,
        runner_registry=runner_registry,
        implementation_commit=IMPLEMENTATION_COMMIT,
        external_input_resolver=resolver,
        failure_injector=crash,
        executor=DirectTaskExecutor(),
        input_port_factory=plane,
        lane=execution.lane,
    )
    with pytest.raises(InjectedSchedulerCrash, match=analysis_task.task_id):
        crashing_scheduler.execute(plan.plan_id, execution)
    resumed = LocalScheduler(
        operational_repository=repository,
        artifact_writer=plane,
        receipt_store=receipts,
        recovery_index=recovery_index,
        recovery_store=recovery_store,
        runner_registry=runner_registry,
        implementation_commit=IMPLEMENTATION_COMMIT,
        external_input_resolver=resolver,
        executor=DirectTaskExecutor(),
        input_port_factory=plane,
        lane=execution.lane,
    ).execute(plan.plan_id, execution)
    assert resumed.status is OperationalStatus.SUCCEEDED
    assert resumed.completed_task_ids == ("exploration-synthesis", analysis_task.task_id)
    assert len(_runner(runners, analysis_task.capability.capability_key).contexts) == 1
    assert len(_runner(runners, SYNTHESIS_KEY).contexts) == 1
    assert all(not task.outputs[0].visibility_ceiling.is_promotable for task in execution.tasks)
    engine.dispose()


@pytest.mark.parametrize("child", [False, True])
def test_lossy_custody_failure_is_bounded_and_creates_no_success_receipt(
    tmp_path: Path, protocol_fixture: ProtocolFixture, monkeypatch, child: bool,
) -> None:
    monkeypatch.setitem(globals(), "DeterministicRunner", LossyDecimalCustodyRunner)
    executor = (LocalProcessExecutor(scratch_root=_plane(tmp_path).root)
                if child else DirectTaskExecutor())
    scheduler, repository, _plane_value, receipts, _runners = _runtime(
        tmp_path, protocol_fixture, executor=executor,
    )
    try:
        run_id = protocol_fixture.run_plan.run_plan_id
        result = scheduler.execute(run_id, protocol_fixture.execution_plan)
        assert result.status is OperationalStatus.FAILED
        attempts = repository.attempts(run_id)
        assert attempts
        assert any(value.disposition is TaskAttemptDisposition.FAILED for value in attempts)
        for task in protocol_fixture.execution_plan.tasks:
            attempted = tuple(value for value in attempts if value.task_id == task.task_id)
            if attempted:
                failed = tuple(value for value in attempted if value.disposition is TaskAttemptDisposition.FAILED)
                if failed:
                    assert len(failed) == (task.maximum_attempts if child else 1)
                    expected_reasons = ({"child-process-failure", "retry-exhaustion"} if child
                                        else {"provider-runner-defect"})
                    assert {value.reason_code for value in failed} <= expected_reasons
                else:
                    assert all(value.disposition is TaskAttemptDisposition.BLOCKED for value in attempted)
                assert all(receipts.read(run_id, value.task_id, value.attempt_id) is None for value in attempted)
        assert not tuple(tmp_path.rglob(".worker-output.*.partial"))
    finally:
        repository.engine.dispose()


def test_retry_preserves_failed_attempt_and_commits_only_successful_receipt(
    tmp_path: Path,
    protocol_fixture: ProtocolFixture,
) -> None:
    scheduler, repository, _plane_value, _receipts, runners = _runtime(
        tmp_path,
        protocol_fixture,
        runner_failures={"reference.develop-a": 1},
    )
    run_id = protocol_fixture.run_plan.run_plan_id
    result = scheduler.execute(run_id, protocol_fixture.execution_plan)
    assert result.status is OperationalStatus.SUCCEEDED
    attempts = repository.attempts(result.run_id)
    develop_attempts = tuple(attempt for attempt in attempts if attempt.task_id == "develop-a")
    assert tuple(attempt.disposition for attempt in develop_attempts) == (
        TaskAttemptDisposition.FAILED,
        TaskAttemptDisposition.SUCCEEDED,
    )
    assert _runner(runners, "reference.develop-a").contexts.__len__() == 2
    assert not (
        tmp_path / f"runs/{run_id}/receipts/develop-a/{run_id}.develop-a.attempt-001.json"
    ).exists()
    assert (
        tmp_path / f"runs/{run_id}/receipts/develop-a/{run_id}.develop-a.attempt-002.json"
    ).is_file()
    repository.engine.dispose()


def test_receipt_first_crash_recovers_running_attempt_without_rerunning_worker(
    tmp_path: Path,
    protocol_fixture: ProtocolFixture,
) -> None:
    crash = CrashAfterReceipt("freeze")
    scheduler, repository, plane, receipts, runners = _runtime(
        tmp_path,
        protocol_fixture,
        failure_injector=crash,
    )
    with pytest.raises(InjectedSchedulerCrash, match="freeze"):
        scheduler.execute(protocol_fixture.run_plan.run_plan_id, protocol_fixture.execution_plan)
    freeze_attempt = next(
        attempt
        for attempt in repository.attempts(protocol_fixture.run_plan.run_plan_id)
        if attempt.task_id == "freeze"
    )
    assert freeze_attempt.disposition is TaskAttemptDisposition.RUNNING
    receipt = receipts.read(
        freeze_attempt.run_id,
        freeze_attempt.task_id,
        freeze_attempt.attempt_id,
    )
    assert receipt is not None
    for materialization in receipt.output_materializations:
        plane.verify(materialization)

    resumed = LocalScheduler(
        operational_repository=repository,
        artifact_writer=plane,
        receipt_store=receipts,
        recovery_index=scheduler.recovery_index,
        recovery_store=scheduler.recovery_store,
        runner_registry=RunnerRegistry(
            runners,
            registry_sha256=protocol_fixture.registry.fingerprint(),
        ),
        implementation_commit=IMPLEMENTATION_COMMIT,
        external_input_resolver=scheduler.external_input_resolver,
        executor=DirectTaskExecutor(),
        input_port_factory=plane,
    ).execute(protocol_fixture.run_plan.run_plan_id, protocol_fixture.execution_plan)
    assert resumed.status is OperationalStatus.SUCCEEDED
    assert _runner(runners, "reference.freeze").contexts.__len__() == 1
    assert _runner(runners, "reference.evaluate").contexts.__len__() == 1
    assert (
        next(
            attempt
            for attempt in repository.attempts(protocol_fixture.run_plan.run_plan_id)
            if attempt.task_id == "freeze"
        ).disposition
        is TaskAttemptDisposition.SUCCEEDED
    )
    repository.engine.dispose()


def test_local_process_executor_isolates_worker_and_reports_typed_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    protocol_fixture: ProtocolFixture,
) -> None:
    task = next(task for task in protocol_fixture.execution_plan.tasks if task.task_id == "prepare")
    manifest = protocol_fixture.registry.resolve(
        task.capability.capability_key,
        task.capability.capability_version,
    )
    context = TaskContext(
        run_id=protocol_fixture.run_plan.run_plan_id,
        task_id=task.task_id,
        attempt_id=f"{protocol_fixture.run_plan.run_plan_id}.prepare.attempt-001",
        config=task.capability.config,
        input_bindings=(),
        input_ports=(),
        output_ports=tuple(
            WorkerOutputPort(
                output_id=output.output_id,
                payload_schema=output.payload_schema,
                profile=output.profile,
                media_type=output.media_type,
            )
            for output in task.outputs
        ),
        permissions=task.capability.required_permissions,
        outcome_access=task.capability.requested_outcome_access,
        resource_budget=replace(
            task.capability.requested_resources,
            memory_bytes=512 * 1024**2,
            wall_time_seconds=5,
        ),
        isolation_profile=WorkerIsolationProfile.TRUSTED_LOCAL,
    )
    executor = LocalProcessExecutor(scratch_root=_plane(tmp_path).root)
    enforcement = executor.enforcement_capability
    assert enforcement.source_scan_limit
    assert enforcement.output_limit
    assert not enforcement.cpu_time_limit
    assert not enforcement.address_space_limit
    assert not enforcement.wall_time_limit
    assert not enforcement.scratch_limit
    assert not enforcement.enforces_resources
    monkeypatch.setenv("ICF_YOLO_TEST_SECRET", "must-not-cross-worker-boundary")
    result = executor.execute(
        ProcessIdentityRunner(manifest),
        context,
        timeout_seconds=5,
    )
    identity_output = result.outputs[0]
    assert isinstance(identity_output, TaskOutputPayload)
    assert int(identity_output.payload.decode("ascii")) != os.getpid()
    with pytest.raises(TaskProcessError, match="typed failure") as captured_failure:
        executor.execute(
            ProcessIdentityRunner(manifest, fail=True),
            replace(
                context,
                attempt_id=f"{protocol_fixture.run_plan.run_plan_id}.prepare.attempt-002",
            ),
            timeout_seconds=5,
        )
    assert captured_failure.value.sanitized_exception_type == "RuntimeError"
    assert captured_failure.value.sanitized_message == "child process reported a bounded failure"
    assert "isolated child failure" not in str(captured_failure.value)
    with pytest.raises(TaskProcessError, match=r"typed result \(exitcode=7\)"):
        executor.execute(
            HardExitRunner(manifest),
            replace(
                context,
                attempt_id=f"{protocol_fixture.run_plan.run_plan_id}.prepare.attempt-010",
            ),
            timeout_seconds=5,
        )
    probed = executor.execute(
        ProcessBoundaryProbeRunner(manifest),
        replace(
            context,
            attempt_id=f"{protocol_fixture.run_plan.run_plan_id}.prepare.attempt-003",
        ),
        timeout_seconds=5,
    )
    probe_output = probed.outputs[0]
    assert isinstance(probe_output, TaskOutputPayload)
    observed = json.loads(probe_output.payload)
    facts = observed["value"]
    assert facts["cpu_limit"] == 5
    assert facts["memory_limit"] == 512 * 1024**2
    assert facts["file_limit"] == context.resource_budget.output_bytes
    assert facts["affinity_count"] <= context.resource_budget.cpu_cores
    assert facts["environment_keys"] == [
        "HOME",
        "LANG",
        "LC_ALL",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "PYTHONHASHSEED",
        "TMPDIR",
        "VECLIB_MAXIMUM_THREADS",
    ]
    assert facts["cwd"].endswith(
        f"scratch/task-execution/{context.run_id}/{context.run_id}.prepare.attempt-003"
    )

    with pytest.raises(ResourceAdmissionError, match="resource boundary"):
        executor.execute(
            ProcessIdentityRunner(manifest),
            replace(
                context,
                attempt_id=f"{protocol_fixture.run_plan.run_plan_id}.prepare.attempt-004",
                isolation_profile=WorkerIsolationProfile.EXPLORATION_NO_NETWORK,
            ),
            timeout_seconds=5,
        )

    with pytest.raises(ResourceAdmissionError, match="resource boundary"):
        executor.execute(
            GeneratedJsonlStreamingRunner(manifest, record_count=400),
            replace(
                context,
                attempt_id=f"{protocol_fixture.run_plan.run_plan_id}.prepare.attempt-005",
                resource_budget=replace(context.resource_budget, output_bytes=1_000),
            ),
            timeout_seconds=5,
        )
    assert not tuple(tmp_path.rglob(".worker-output.*.partial"))

    with pytest.raises(ValueError, match="checks must have sorted, unique check_id"):
        executor.execute(
            GeneratedJsonlStreamingRunner(
                manifest,
                record_count=0,
                unsorted_checks=True,
            ),
            replace(
                context,
                attempt_id=f"{protocol_fixture.run_plan.run_plan_id}.prepare.attempt-011",
            ),
            timeout_seconds=5,
        )
    assert not tuple(tmp_path.rglob(".worker-output.*.partial"))

    with pytest.raises(ResourceAdmissionError, match="scratch budget"):
        executor.execute(
            ScratchBudgetProbeRunner(manifest),
            replace(
                context,
                attempt_id=f"{protocol_fixture.run_plan.run_plan_id}.prepare.attempt-006",
                resource_budget=replace(context.resource_budget, output_bytes=1_000),
            ),
            timeout_seconds=5,
        )

    with pytest.raises(ResourceAdmissionError, match="scratch entry count"):
        executor.execute(
            ScratchEntryFloodRunner(manifest),
            replace(
                context,
                attempt_id=f"{protocol_fixture.run_plan.run_plan_id}.prepare.attempt-007",
            ),
            timeout_seconds=5,
        )

    grouped = executor.execute(
        DescendantProcessProbeRunner(manifest, detached=False),
        replace(
            context,
            attempt_id=f"{protocol_fixture.run_plan.run_plan_id}.prepare.attempt-008",
        ),
        timeout_seconds=5,
    )
    grouped_output = grouped.outputs[0]
    assert isinstance(grouped_output, TaskOutputPayload)
    grouped_process_id = int(grouped_output.payload.decode("ascii"))
    deadline = time.monotonic() + 2
    while _process_is_running(grouped_process_id) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not _process_is_running(grouped_process_id)

    detached = executor.execute(
        DescendantProcessProbeRunner(manifest, detached=True),
        replace(
            context,
            attempt_id=f"{protocol_fixture.run_plan.run_plan_id}.prepare.attempt-009",
        ),
        timeout_seconds=5,
    )
    detached_output = detached.outputs[0]
    assert isinstance(detached_output, TaskOutputPayload)
    detached_process_id = int(detached_output.payload.decode("ascii"))
    assert _process_is_running(detached_process_id)
    deadline = time.monotonic() + 2
    while _process_is_running(detached_process_id) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not _process_is_running(detached_process_id)
    assert not executor.enforcement_capability.enforces_resources


def test_local_process_executor_streams_large_streamed_output_to_guarded_scratch(
    tmp_path: Path,
    protocol_fixture: ProtocolFixture,
) -> None:
    task = next(task for task in protocol_fixture.execution_plan.tasks if task.task_id == "prepare")
    manifest = protocol_fixture.registry.resolve(
        task.capability.capability_key,
        task.capability.capability_version,
    )
    source_output = task.outputs[0]
    context = TaskContext(
        run_id=protocol_fixture.run_plan.run_plan_id,
        task_id=task.task_id,
        attempt_id=f"{protocol_fixture.run_plan.run_plan_id}.prepare.large-output-attempt",
        config=task.capability.config,
        input_bindings=(),
        input_ports=(),
        output_ports=(
            WorkerOutputPort(
                output_id=source_output.output_id,
                payload_schema=source_output.payload_schema,
                profile=source_output.profile,
                media_type=source_output.media_type,
            ),
        ),
        permissions=task.capability.required_permissions,
        outcome_access=task.capability.requested_outcome_access,
        resource_budget=replace(
            task.capability.requested_resources,
            memory_bytes=512 * 1024**2,
            output_bytes=2 * STREAM_CHUNK_BYTES,
            wall_time_seconds=5,
        ),
        isolation_profile=WorkerIsolationProfile.TRUSTED_LOCAL,
    )
    result = LocalProcessExecutor(scratch_root=_plane(tmp_path).root).execute(
        LargeLegacyOutputRunner(manifest),
        context,
        timeout_seconds=5,
    )

    assert len(result.outputs) == 1
    output = result.outputs[0]
    assert isinstance(output, StreamedTaskOutput)
    assert output.size_bytes == STREAM_CHUNK_BYTES + 1
    assert (
        output.logical_content_sha256 == hashlib.sha256(b"x" * (STREAM_CHUNK_BYTES + 1)).hexdigest()
    )
    digest = hashlib.sha256()
    observed_size = 0
    for chunk in output.source.chunks(STREAM_CHUNK_BYTES):
        assert len(chunk) <= STREAM_CHUNK_BYTES
        observed_size += len(chunk)
        digest.update(chunk)
    assert observed_size == output.size_bytes
    assert digest.hexdigest() == output.physical_sha256
    output.source.close()
    assert not tuple(tmp_path.rglob(".worker-output.*.partial"))


def test_bounded_scratch_scan_tolerates_an_exact_vanished_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transient = tmp_path / "transient.bin"
    transient.write_bytes(b"intermediate")
    original_scandir = os.scandir

    class VanishingEntryScan:
        def __init__(self, path: os.PathLike[str] | str) -> None:
            self.path = Path(path)
            self.scan = original_scandir(path)

        def __enter__(self) -> VanishingEntryScan:
            self.scan.__enter__()
            return self

        def __iter__(self) -> object:
            entries = tuple(self.scan)
            if self.path == tmp_path:
                transient.unlink()
            return iter(entries)

        def __exit__(self, *args: object) -> object:
            return self.scan.__exit__(*args)

    monkeypatch.setattr(
        "empirical_lawhood.infrastructure.execution.os.scandir",
        VanishingEntryScan,
    )

    assert _bounded_scratch_bytes(tmp_path, maximum_bytes=1024) == 0


def test_streaming_worker_commits_validated_artifact_then_receipt(
    tmp_path: Path,
    protocol_fixture: ProtocolFixture,
) -> None:
    scheduler, repository, plane, receipts, plan = _streaming_runtime(
        tmp_path,
        protocol_fixture,
    )
    run_id = plan.source_plan.object_id
    result = scheduler.execute(run_id, plan)

    assert result.status is OperationalStatus.SUCCEEDED
    assert result.completed_task_ids == ("prepare",)
    attempt = repository.attempts(run_id)[0]
    receipt = receipts.read(run_id, "prepare", attempt.attempt_id)
    assert receipt is not None
    assert receipt.operational_status is OperationalStatus.SUCCEEDED
    assert len(receipt.output_materializations) == 1
    output = receipt.output_materializations[0]
    assert output.size_bytes > 400_000
    assert output.relative_path == f"runs/{run_id}/streams/streaming-task-output.jsonl"
    plane.verify(output)
    assert not tuple(tmp_path.rglob(".worker-output.*.partial"))
    repository.engine.dispose()


@pytest.mark.parametrize("failure_mode", ("midstream", "semantic", "sha256"))
def test_two_output_failure_publishes_neither_output_and_restart_retries_cleanly(
    tmp_path: Path,
    protocol_fixture: ProtocolFixture,
    failure_mode: str,
) -> None:
    engine = create_catalog_engine(f"sqlite+pysqlite:///{tmp_path / 'catalog.sqlite3'}")
    upgrade_catalog(engine)
    repository = _CrashAfterFirstFailedAttemptRepository(engine)
    original = next(
        task for task in protocol_fixture.execution_plan.tasks if task.task_id == "prepare"
    )
    first_schema = 'empirical-lawhood/testing/fixtures/batch-first'
    second_schema = 'empirical-lawhood/testing/fixtures/batch-second'
    resources = replace(
        original.capability.requested_resources,
        output_bytes=16_384,
    )
    outputs = (
        replace(
            original.outputs[0],
            output_id="prepare.batch-first",
            logical_artifact_id="artifact.prepare.batch-first",
            relative_path=(
                f"runs/{protocol_fixture.execution_plan.source_plan.object_id}/"
                "two-output/first.json"
            ),
            payload_schema=first_schema,
        ),
        replace(
            original.outputs[0],
            output_id="prepare.batch-second",
            logical_artifact_id="artifact.prepare.batch-second",
            relative_path=(
                f"runs/{protocol_fixture.execution_plan.source_plan.object_id}/"
                "two-output/second.json"
            ),
            payload_schema=second_schema,
        ),
    )
    task = replace(
        original,
        dependency_task_ids=(),
        external_inputs=(),
        outputs=outputs,
        maximum_attempts=2,
        capability=replace(
            original.capability,
            required_input_schema_ids=(),
            required_output_schema_ids=(first_schema, second_schema),
            requested_resources=resources,
        ),
    )
    original_manifest = protocol_fixture.registry.resolve(
        task.capability.capability_key,
        task.capability.capability_version,
    )
    manifest = replace(
        original_manifest,
        input_schema_ids=(),
        output_schema_ids=(first_schema, second_schema),
        resource_ceiling=resources,
    )
    plan = replace(
        protocol_fixture.execution_plan,
        execution_plan_id=f"execution-plan.two-output-{failure_mode}",
        tasks=(task,),
    )
    contracts = tuple(
        CapabilityOutputSemanticContract(
            capability_key=task.capability.capability_key,
            capability_version=task.capability.capability_version,
            capability_implementation_sha256=manifest.implementation_sha256,
            payload_schema=schema,
            profile=ArtifactProfile.CANONICAL_JSON,
            top_level_keys=("schema", "value"),
        )
        for schema in (first_schema, second_schema)
    )
    semantic_validations = capability_semantic_validation_registry(
        registry_id=f"test-execution-semantics.{plan.execution_plan_id}",
        plan=plan,
        contracts=contracts,
    )
    plane = _plane(tmp_path, semantic_validations)
    receipts = ExternalTaskReceiptStore(plane)
    recovery_index, recovery_store = _recovery(plane, plan)
    failing_runner = _TwoOutputRunner(manifest, failure_mode=failure_mode)
    scheduler = LocalScheduler(
        operational_repository=repository,
        artifact_writer=plane,
        receipt_store=receipts,
        recovery_index=recovery_index,
        recovery_store=recovery_store,
        runner_registry=RunnerRegistry(
            (failing_runner,),
            registry_sha256=plan.registry_sha256,
        ),
        implementation_commit=IMPLEMENTATION_COMMIT,
        executor=DirectTaskExecutor(),
        input_port_factory=plane,
        output_semantic_contracts=contracts,
    )
    run_id = plan.source_plan.object_id

    with pytest.raises(InjectedSchedulerCrash, match="after-failed-attempt"):
        scheduler.execute(run_id, plan)

    failed_attempt = repository.attempts(run_id)[0]
    assert failed_attempt.disposition is TaskAttemptDisposition.FAILED
    assert receipts.read(run_id, task.task_id, failed_attempt.attempt_id) is None
    assert failing_runner.sources[0].closed
    for output in outputs:
        assert not (tmp_path / output.relative_path).exists()
        assert not (tmp_path / f"{output.relative_path}.manifest.json").exists()
    assert not tuple(tmp_path.rglob("*.batch.partial"))

    corrected_runner = _TwoOutputRunner(manifest, failure_mode=None)
    resumed = LocalScheduler(
        operational_repository=repository,
        artifact_writer=plane,
        receipt_store=receipts,
        recovery_index=recovery_index,
        recovery_store=recovery_store,
        runner_registry=RunnerRegistry(
            (corrected_runner,),
            registry_sha256=plan.registry_sha256,
        ),
        implementation_commit=IMPLEMENTATION_COMMIT,
        executor=DirectTaskExecutor(),
        input_port_factory=plane,
        output_semantic_contracts=contracts,
    ).execute(run_id, plan)

    assert resumed.status is OperationalStatus.SUCCEEDED
    assert tuple(attempt.disposition for attempt in repository.attempts(run_id)) == (
        TaskAttemptDisposition.FAILED,
        TaskAttemptDisposition.SUCCEEDED,
    )
    receipt = receipts.read(
        run_id,
        task.task_id,
        repository.attempts(run_id)[-1].attempt_id,
    )
    assert receipt is not None
    assert len(receipt.output_materializations) == 2
    for output in outputs:
        assert (tmp_path / output.relative_path).is_file()
        assert (tmp_path / f"{output.relative_path}.manifest.json").is_file()
    assert corrected_runner.sources[0].closed
    assert not tuple(tmp_path.rglob("*.batch.partial"))
    engine.dispose()


def test_midstream_worker_failure_commits_no_output_manifest_receipt_or_success(
    tmp_path: Path,
    protocol_fixture: ProtocolFixture,
) -> None:
    scheduler, repository, _plane_value, receipts, plan = _streaming_runtime(
        tmp_path,
        protocol_fixture,
        fail_after_records=100,
    )
    run_id = plan.source_plan.object_id
    result = scheduler.execute(run_id, plan)

    assert result.status is OperationalStatus.FAILED
    assert result.failed_task_ids == ("prepare",)
    attempt = repository.attempts(run_id)[0]
    assert attempt.disposition is TaskAttemptDisposition.FAILED
    assert receipts.read(run_id, "prepare", attempt.attempt_id) is None
    assert not (tmp_path / "streams/streaming-task-output.jsonl").exists()
    assert not (tmp_path / "streams/streaming-task-output.jsonl.manifest.json").exists()
    assert not tuple(tmp_path.rglob(".worker-output.*.partial"))
    repository.engine.dispose()


def test_input_close_failure_closes_already_collected_streams(
    tmp_path: Path,
    protocol_fixture: ProtocolFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler, repository, _plane_value, _receipts, _runners = _runtime(
        tmp_path,
        protocol_fixture,
    )
    task = protocol_fixture.execution_plan.tasks[0]
    binding = WorkerInputBinding(
        artifact_id="artifact.synthetic-close-failure",
        materialization_id="materialization.synthetic-close-failure",
        payload_schema='empirical-lawhood/testing/fixtures/synthetic-close-failure',
        media_type="application/json",
        size_bytes=0,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        kind=WorkerInputKind.EXTERNAL,
    )

    class FailingCloseReader:
        bytes_read = 0

        def read(self, _size: int = -1) -> bytes:
            return b""

        def close(self) -> None:
            raise OSError("synthetic input close failure")

    class TrackingSource:
        closed = False

        def chunks(self, _maximum_chunk_bytes: int):  # type: ignore[no-untyped-def]
            yield b"x"

        def close(self) -> None:
            self.closed = True

    source = TrackingSource()

    class CollectedStreamExecutor:
        enforcement_capability = DirectTaskExecutor.enforcement_capability

        def execute(
            self,
            _runner: TaskRunner,
            _context: TaskContext,
            *,
            timeout_seconds: int,
        ) -> RunnerResult:
            assert timeout_seconds == task.capability.requested_resources.wall_time_seconds
            return RunnerResult(
                outputs=(
                    StreamedTaskOutput(
                        output_id=task.outputs[0].output_id,
                        size_bytes=1,
                        physical_sha256=hashlib.sha256(b"x").hexdigest(),
                        source=source,
                    ),
                ),
                checks=(ReceiptCheck("synthetic-stream-collected", True, ()),),
            )

    scheduler.executor = CollectedStreamExecutor()
    monkeypatch.setattr(
        scheduler,
        "_worker_inputs",
        lambda *_args: (
            (binding,),
            (WorkerInputPort(binding=binding, _reader=FailingCloseReader()),),
        ),
    )

    with pytest.raises(SchedulerConsistencyError, match="input port could not be closed"):
        scheduler._run_attempt(
            protocol_fixture.execution_plan.source_plan.object_id,
            task,
            "attempt.synthetic-input-close-failure",
            (),
            (),
            (),
        )

    assert source.closed
    repository.engine.dispose()


@pytest.mark.parametrize(
    ("emitted_schema", "semantic_contract_schema"),
    (
        ('empirical-lawhood/testing/fixtures/wrong-stream-schema', None),
        (None, 'empirical-lawhood/testing/fixtures/unregistered-stream-contract'),
    ),
)
def test_streamed_output_requires_matching_registered_profile_schema_before_publication(
    tmp_path: Path,
    protocol_fixture: ProtocolFixture,
    emitted_schema: str | None,
    semantic_contract_schema: str | None,
) -> None:
    scheduler, repository, _plane_value, receipts, plan = _streaming_runtime(
        tmp_path,
        protocol_fixture,
        emitted_schema=emitted_schema,
        semantic_contract_schema=semantic_contract_schema,
    )
    run_id = plan.source_plan.object_id

    result = scheduler.execute(run_id, plan)

    assert result.status is OperationalStatus.FAILED
    assert result.failed_task_ids == ("prepare",)
    attempt = repository.attempts(run_id)[0]
    assert attempt.disposition is TaskAttemptDisposition.FAILED
    assert receipts.read(run_id, "prepare", attempt.attempt_id) is None
    assert not (tmp_path / "streams/streaming-task-output.jsonl").exists()
    assert not (tmp_path / "streams/streaming-task-output.jsonl.manifest.json").exists()
    assert not tuple(tmp_path.rglob(".worker-output.*.partial"))
    repository.engine.dispose()


def test_scheduler_requires_exact_authorized_bounded_input_ports(
    tmp_path: Path,
    protocol_fixture: ProtocolFixture,
) -> None:
    scheduler, repository, _plane_value, _receipts, _runners = _runtime(
        tmp_path,
        protocol_fixture,
    )
    task = next(
        value for value in protocol_fixture.execution_plan.tasks if value.task_id == "evaluate"
    )
    assert scheduler.external_input_resolver is not None
    inputs = scheduler.external_input_resolver.resolve(task.external_input_artifact_ids)
    total = sum(value.materialization.size_bytes for value in inputs)
    bounded_task = replace(
        task,
        capability=replace(
            task.capability,
            requested_resources=replace(
                task.capability.requested_resources,
                source_scan_bytes=total,
            ),
        ),
    )

    bindings, ports = scheduler._worker_inputs(bounded_task, (), inputs)
    assert tuple(port.binding for port in ports) == bindings
    assert tuple(port.artifact_id for port in ports) == tuple(value.artifact_id for value in inputs)
    assert sum(len(port.read()) for port in ports) == total
    assert sum(port.bytes_read for port in ports) == total
    for port in ports:
        port.close()

    without_sealed_authority = replace(
        bounded_task,
        capability=replace(
            bounded_task.capability,
            required_permissions=tuple(
                permission
                for permission in bounded_task.capability.required_permissions
                if permission is not CapabilityPermission.READ_SEALED_OUTCOMES
            ),
        ),
    )
    with pytest.raises(ResourceAdmissionError, match="lacks read or outcome authority"):
        scheduler._worker_inputs(without_sealed_authority, (), inputs)

    with pytest.raises(ResourceAdmissionError, match="source-scan budget"):
        scheduler._worker_inputs(
            replace(
                bounded_task,
                capability=replace(
                    bounded_task.capability,
                    requested_resources=replace(
                        bounded_task.capability.requested_resources,
                        source_scan_bytes=total - 1,
                    ),
                ),
            ),
            (),
            inputs,
        )
    with pytest.raises(ResourceAdmissionError, match="positive source-scan budget"):
        scheduler._worker_inputs(
            replace(
                bounded_task,
                capability=replace(
                    bounded_task.capability,
                    requested_resources=replace(
                        bounded_task.capability.requested_resources,
                        source_scan_bytes=0,
                    ),
                ),
            ),
            (),
            inputs,
        )

    scheduler.input_port_factory = FalseLineageInputPortFactory(_plane_value)
    with pytest.raises(SchedulerConsistencyError, match="false lineage"):
        scheduler._worker_inputs(bounded_task, (), inputs)
    repository.engine.dispose()


def test_authorized_reveal_barrier_opens_sealed_inputs_without_static_authority(
    tmp_path: Path,
    protocol_fixture: ProtocolFixture,
) -> None:
    scheduler, repository, _plane_value, _receipts, _runners = _runtime(
        tmp_path,
        protocol_fixture,
    )
    task = next(
        value for value in protocol_fixture.execution_plan.tasks if value.task_id == "evaluate"
    )
    assert scheduler.external_input_resolver is not None
    inputs = scheduler.external_input_resolver.resolve(task.external_input_artifact_ids)
    total = sum(value.materialization.size_bytes for value in inputs)
    evaluator = replace(
        task,
        barrier=BarrierKind.REVEAL,
        capability=replace(
            task.capability,
            requested_outcome_access=OutcomeAccess.EVALUATION_REVEALED,
            required_permissions=tuple(
                sorted(
                    {
                        CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                        CapabilityPermission.READ_OUTCOME_VISIBLE,
                    }
                )
            ),
            requested_resources=replace(
                task.capability.requested_resources,
                source_scan_bytes=total,
            ),
        ),
    )

    bindings, ports = scheduler._worker_inputs(evaluator, (), inputs)
    assert tuple(port.binding for port in ports) == bindings
    for port in ports:
        port.close()
    scheduler.authorized_barriers = frozenset()
    with pytest.raises(ResourceAdmissionError, match="lacks read or outcome authority"):
        scheduler._worker_inputs(evaluator, (), inputs)
    repository.engine.dispose()


def test_sealed_generation_can_read_development_freeze_only_with_permission(
    tmp_path: Path,
    protocol_fixture: ProtocolFixture,
) -> None:
    plane = _plane(tmp_path)
    resolver = _external_inputs(plane, protocol_fixture)
    task = next(
        value for value in protocol_fixture.execution_plan.tasks if value.task_id == "evaluate"
    )
    source = resolver.resolve(task.external_input_artifact_ids)[0]
    development_freeze = VerifiedArtifactInput(
        replace(
            source.logical,
            visibility_ceiling=VisibilityCeiling.DEVELOPMENT_ONLY,
            outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        ),
        source.materialization,
    )
    sealed_generation = replace(
        task,
        capability=replace(
            task.capability,
            requested_outcome_access=OutcomeAccess.EVALUATION_SEALED,
            required_permissions=tuple(
                sorted(
                    {
                        CapabilityPermission.READ_DEVELOPMENT,
                        CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                    }
                )
            ),
        ),
    )

    assert LocalScheduler._input_is_authorized(sealed_generation, development_freeze)
    assert not LocalScheduler._input_is_authorized(
        replace(
            sealed_generation,
            capability=replace(
                sealed_generation.capability,
                required_permissions=(CapabilityPermission.READ_EXTERNAL_ARTIFACTS,),
            ),
        ),
        development_freeze,
    )


def test_sealed_dependent_task_can_read_revealed_parent_only_with_permission(
    tmp_path: Path,
    protocol_fixture: ProtocolFixture,
) -> None:
    plane = _plane(tmp_path)
    resolver = _external_inputs(plane, protocol_fixture)
    task = next(
        value for value in protocol_fixture.execution_plan.tasks if value.task_id == "evaluate"
    )
    source = resolver.resolve(task.external_input_artifact_ids)[0]
    revealed_parent = VerifiedArtifactInput(
        replace(
            source.logical,
            visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
            outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        ),
        source.materialization,
    )
    sealed_dependent_task = replace(
        task,
        capability=replace(
            task.capability,
            requested_outcome_access=OutcomeAccess.EVALUATION_SEALED,
            required_permissions=tuple(
                sorted(
                    {
                        CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                        CapabilityPermission.READ_OUTCOME_VISIBLE,
                    }
                )
            ),
        ),
    )

    assert LocalScheduler._input_is_authorized(sealed_dependent_task, revealed_parent)
    assert not LocalScheduler._input_is_authorized(
        replace(
            sealed_dependent_task,
            capability=replace(
                sealed_dependent_task.capability,
                required_permissions=(CapabilityPermission.READ_EXTERNAL_ARTIFACTS,),
            ),
        ),
        revealed_parent,
    )


def test_privileged_truth_requires_reveal_capability_and_evaluator_access(
    tmp_path: Path,
    protocol_fixture: ProtocolFixture,
) -> None:
    plane = _plane(tmp_path)
    resolver = _external_inputs(plane, protocol_fixture)
    task = next(
        value for value in protocol_fixture.execution_plan.tasks if value.task_id == "evaluate"
    )
    source = resolver.resolve(task.external_input_artifact_ids)[0]
    privileged_truth = VerifiedArtifactInput(
        replace(
            source.logical,
            visibility_ceiling=VisibilityCeiling.PRIVILEGED_TRUTH,
            outcome_access=OutcomeAccess.PRIVILEGED_TRUTH,
        ),
        source.materialization,
    )
    revealing_evaluator = replace(
        task,
        capability=replace(
            task.capability,
            requested_outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
            required_permissions=tuple(
                sorted(
                    {
                        CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                        CapabilityPermission.REVEAL_OUTCOMES,
                    }
                )
            ),
        ),
    )

    assert LocalScheduler._input_is_authorized(revealing_evaluator, privileged_truth)
    assert not LocalScheduler._input_is_authorized(
        replace(
            revealing_evaluator,
            capability=replace(
                revealing_evaluator.capability,
                requested_outcome_access=OutcomeAccess.EVALUATION_REVEALED,
            ),
        ),
        privileged_truth,
    )
    assert not LocalScheduler._input_is_authorized(
        replace(
            revealing_evaluator,
            capability=replace(
                revealing_evaluator.capability,
                required_permissions=(CapabilityPermission.READ_EXTERNAL_ARTIFACTS,),
            ),
        ),
        privileged_truth,
    )


def test_resource_admission_covers_compute_scratch_and_exploration_isolation(
    protocol_fixture: ProtocolFixture,
) -> None:
    capacity = ExecutionResourceCapacity(
        cpu_cores=1,
        memory_bytes=2_000_000,
        gpu_devices=0,
        enforced_cpu_limit=True,
        enforced_address_space_limit=True,
        enforced_no_network=False,
    )
    admitter = LocalExecutionResourceAdmitter(capacity)
    plan = protocol_fixture.execution_plan
    admitted = admitter.assess(
        plan,
        available_scratch_bytes=10_000,
        executor_enforces_resources=True,
        executor_enforces_no_network=False,
    )
    assert admitted.admitted

    first = plan.tasks[0]
    excessive = replace(
        first,
        capability=replace(
            first.capability,
            requested_resources=replace(
                first.capability.requested_resources,
                cpu_cores=2,
                memory_bytes=2_000_001,
                gpu_devices=1,
            ),
        ),
    )
    refused = admitter.assess(
        replace(
            plan, tasks=tuple(sorted((excessive, *plan.tasks[1:]), key=lambda value: value.task_id))
        ),
        available_scratch_bytes=0,
        executor_enforces_resources=False,
        executor_enforces_no_network=False,
    )
    assert not refused.admitted
    assert set(refused.reason_codes) == {
        "CPU_CAPACITY_UNAVAILABLE",
        "EXECUTOR_RESOURCE_ENFORCEMENT_UNAVAILABLE",
        "GPU_CAPACITY_UNAVAILABLE",
        "MEMORY_CAPACITY_UNAVAILABLE",
        "SCRATCH_CAPACITY_UNAVAILABLE",
    }

    exploration = admitter.assess(
        replace(plan, lane=PlanLane.EXPLORATORY),
        available_scratch_bytes=10_000,
        executor_enforces_resources=True,
        executor_enforces_no_network=True,
    )
    assert exploration.reason_codes == ("NO_NETWORK_ISOLATION_UNAVAILABLE",)

    trusted = admitter.assess(
        replace(plan, lane=PlanLane.EXPLORATORY),
        available_scratch_bytes=10_000,
        executor_enforces_resources=False,
        executor_enforces_no_network=False,
        assurance_profile=ExecutionAssuranceProfile.TRUSTED_LOCAL,
    )
    assert trusted.admitted
    assert trusted.reason_codes == ()
    assert trusted.assurance_codes == (
        "AGGREGATE_DESCENDANT_LIMITS_NOT_ENFORCED",
        "NETWORK_PERMISSION_NOT_REQUESTED",
        "OS_NETWORK_ISOLATION_NOT_ENFORCED",
    )


def test_trusted_local_assurance_is_persisted_in_task_receipts(
    tmp_path: Path,
    protocol_fixture: ProtocolFixture,
) -> None:
    scheduler, repository, _plane_value, receipts, _runners = _runtime(
        tmp_path,
        protocol_fixture,
        assurance_profile=ExecutionAssuranceProfile.TRUSTED_LOCAL,
        assurance_codes=("AGGREGATE_DESCENDANT_LIMITS_NOT_ENFORCED",),
    )

    result = scheduler.execute(
        protocol_fixture.run_plan.run_plan_id,
        protocol_fixture.execution_plan,
    )

    assert result.status is OperationalStatus.SUCCEEDED
    for task_id, receipt_id in zip(
        result.completed_task_ids,
        result.receipt_ids,
        strict=True,
    ):
        receipt = receipts.read(
            protocol_fixture.run_plan.run_plan_id,
            task_id,
            receipt_id.removeprefix("receipt."),
        )
        assert receipt is not None
        assert {value.check_id for value in receipt.checks}.issuperset(
            {
                "aggregate-descendant-limits-not-enforced",
                "execution-assurance-trusted-local",
            }
        )
    repository.engine.dispose()


def test_live_lease_blocks_duplicate_execution_and_expired_lease_retries(
    tmp_path: Path,
    protocol_fixture: ProtocolFixture,
) -> None:
    clock = FixedClock(100)
    scheduler, repository, _plane_value, _receipts, runners = _runtime(
        tmp_path,
        protocol_fixture,
        clock=clock,
    )
    run_id = protocol_fixture.run_plan.run_plan_id
    repository.register_run(run_id, protocol_fixture.execution_plan.fingerprint())
    repository.start_attempt(
        run_id=run_id,
        task_id="prepare",
        attempt_id=f"{run_id}.prepare.attempt-001",
        lease_id=f"lease.{run_id}.prepare.attempt-001",
        owner_id="other-scheduler",
        expires_epoch_seconds=101,
    )
    with pytest.raises(LiveLeaseError, match="active lease"):
        scheduler.execute(run_id, protocol_fixture.execution_plan)
    assert not _runner(runners, "reference.prepare").contexts

    clock.value = 102
    result = scheduler.execute(run_id, protocol_fixture.execution_plan)
    assert result.status is OperationalStatus.SUCCEEDED
    prepare_attempts = tuple(
        attempt for attempt in repository.attempts(run_id) if attempt.task_id == "prepare"
    )
    assert tuple(attempt.disposition for attempt in prepare_attempts) == (
        TaskAttemptDisposition.FAILED,
        TaskAttemptDisposition.SUCCEEDED,
    )
    assert prepare_attempts[0].reason_code == "lease-expired"
    repository.engine.dispose()


def test_failed_branch_does_not_suppress_sibling_and_blocks_join(
    tmp_path: Path,
    protocol_fixture: ProtocolFixture,
) -> None:
    scheduler, repository, _plane_value, _receipts, runners = _runtime(
        tmp_path,
        protocol_fixture,
        runner_failures={"reference.develop-a": 2},
    )
    result = scheduler.execute(
        protocol_fixture.run_plan.run_plan_id,
        protocol_fixture.execution_plan,
    )
    assert result.status is OperationalStatus.FAILED
    assert result.completed_task_ids == ("develop-b", "prepare")
    assert result.failed_task_ids == ("develop-a",)
    assert result.blocked_task_ids == ("evaluate", "freeze", "report")
    assert len(_runner(runners, "reference.develop-b").contexts) == 1
    assert not _runner(runners, "reference.freeze").contexts
    repository.engine.dispose()


def test_parallel_scheduler_runs_independent_ready_tasks_concurrently(
    tmp_path: Path,
    protocol_fixture: ProtocolFixture,
) -> None:
    executor = ConcurrencyObservingExecutor()
    scheduler, repository, _plane_value, _receipts, _runners = _runtime(
        tmp_path,
        protocol_fixture,
        executor=executor,
        maximum_parallel_tasks=2,
        parallel_resource_capacity=ExecutionResourceCapacity(
            cpu_cores=2,
            memory_bytes=2_000_000,
            gpu_devices=0,
            enforced_cpu_limit=True,
            enforced_address_space_limit=True,
            enforced_no_network=False,
        ),
    )

    result = scheduler.execute(
        protocol_fixture.run_plan.run_plan_id,
        protocol_fixture.execution_plan,
    )

    assert result.status is OperationalStatus.SUCCEEDED
    assert executor.peak_development == 2
    assert all(executor.development_affinities.values())
    assert not set(executor.development_affinities["develop-a"]).intersection(
        executor.development_affinities["develop-b"]
    )
    assert len(result.receipts) == len(protocol_fixture.execution_plan.tasks)
    repository.engine.dispose()


def test_parallel_scheduler_unlocks_dependent_task_before_unrelated_sibling_finishes(
    tmp_path, protocol_fixture
):
    plan = replace(
        protocol_fixture.execution_plan,
        tasks=tuple(
            replace(task, dependency_task_ids=("develop-a",)) if task.task_id == "freeze" else task
            for task in protocol_fixture.execution_plan.tasks
        ),
    )
    dependent_task_started = threading.Event()
    sibling_started = threading.Event()
    observed = []

    class OrderedExecutor:
        def execute(self, runner, context, *, timeout_seconds):
            if context.task_id == "develop-b":
                sibling_started.set()
                assert dependent_task_started.wait(10), "follow-up waited for an unrelated sibling"
            if context.task_id == "freeze":
                assert sibling_started.wait(10)
                dependent_task_started.set()
            observed.append(context.task_id)
            return runner.execute(context)

    scheduler, repository, _, _, _ = _runtime(
        tmp_path,
        replace(protocol_fixture, execution_plan=plan),
        executor=OrderedExecutor(),
        maximum_parallel_tasks=2,
        parallel_resource_capacity=ExecutionResourceCapacity(
            cpu_cores=2,
            memory_bytes=2_000_000,
            gpu_devices=0,
            enforced_cpu_limit=True,
            enforced_address_space_limit=True,
            enforced_no_network=False,
        ),
    )
    result = scheduler.execute(protocol_fixture.run_plan.run_plan_id, plan)
    assert result.status is OperationalStatus.SUCCEEDED
    assert observed.index("freeze") < observed.index("develop-b")
    repository.engine.dispose()


def test_slow_sibling_publication_does_not_block_ready_dependent_task(tmp_path, protocol_fixture):
    plan = replace(
        protocol_fixture.execution_plan,
        tasks=tuple(
            replace(task, dependency_task_ids=("develop-a",)) if task.task_id == "freeze" else task
            for task in protocol_fixture.execution_plan.tasks
        ),
    )
    dependent_task_started = threading.Event()
    publication_started = threading.Event()

    class Executor:
        def execute(self, runner, context, *, timeout_seconds):
            if context.task_id == "freeze":
                assert publication_started.wait(10)
                dependent_task_started.set()
            return runner.execute(context)

    scheduler, repository, _, receipts, _ = _runtime(
        tmp_path,
        replace(protocol_fixture, execution_plan=plan),
        executor=Executor(),
        maximum_parallel_tasks=2,
        parallel_resource_capacity=ExecutionResourceCapacity(
            cpu_cores=2,
            memory_bytes=2_000_000,
            gpu_devices=0,
            enforced_cpu_limit=True,
            enforced_address_space_limit=True,
            enforced_no_network=False,
        ),
    )
    commit = receipts.commit

    def slow_commit(receipt, **kwargs):
        if receipt.task_id == "develop-b":
            publication_started.set()
            assert dependent_task_started.wait(10), "publication blocked an independent follow-up"
        return commit(receipt, **kwargs)

    receipts.commit = slow_commit
    try:
        result = scheduler.execute(protocol_fixture.run_plan.run_plan_id, plan)
        assert result.status is OperationalStatus.SUCCEEDED
        assert len(result.receipts) == len(plan.tasks)
        assert dependent_task_started.is_set()
    finally:
        repository.engine.dispose()


def test_parallel_scheduler_serializes_shared_resource_locks_without_blocking(
    tmp_path: Path,
    protocol_fixture: ProtocolFixture,
) -> None:
    shared_lock_tasks = tuple(
        replace(task, resource_lock_ids=("development-shared",))
        if task.task_id in {"develop-a", "develop-b"}
        else task
        for task in protocol_fixture.execution_plan.tasks
    )
    execution_plan = replace(protocol_fixture.execution_plan, tasks=shared_lock_tasks)
    fixture = replace(protocol_fixture, execution_plan=execution_plan)
    executor = ConcurrencyObservingExecutor()
    scheduler, repository, _plane_value, _receipts, _runners = _runtime(
        tmp_path,
        fixture,
        executor=executor,
        maximum_parallel_tasks=2,
        parallel_resource_capacity=ExecutionResourceCapacity(
            cpu_cores=2,
            memory_bytes=2_000_000,
            gpu_devices=0,
            enforced_cpu_limit=True,
            enforced_address_space_limit=True,
            enforced_no_network=False,
        ),
    )

    result = scheduler.execute(fixture.run_plan.run_plan_id, execution_plan)

    assert result.status is OperationalStatus.SUCCEEDED
    assert result.blocked_task_ids == ()
    assert executor.peak_development == 1
    repository.engine.dispose()


def test_parallel_scheduler_preserves_retry_and_terminal_replay(
    tmp_path: Path,
    protocol_fixture: ProtocolFixture,
) -> None:
    capacity = ExecutionResourceCapacity(
        cpu_cores=2,
        memory_bytes=2_000_000,
        gpu_devices=0,
        enforced_cpu_limit=True,
        enforced_address_space_limit=True,
        enforced_no_network=False,
    )
    scheduler, repository, _plane_value, _receipts, runners = _runtime(
        tmp_path,
        protocol_fixture,
        runner_failures={"reference.develop-a": 1},
        maximum_parallel_tasks=2,
        parallel_resource_capacity=capacity,
    )

    first = scheduler.execute(
        protocol_fixture.run_plan.run_plan_id,
        protocol_fixture.execution_plan,
    )
    replay = scheduler.execute(
        protocol_fixture.run_plan.run_plan_id,
        protocol_fixture.execution_plan,
    )

    assert first.status is replay.status is OperationalStatus.SUCCEEDED
    assert first.receipts == replay.receipts
    assert len(_runner(runners, "reference.develop-a").contexts) == 2
    assert len(_runner(runners, "reference.develop-b").contexts) == 1
    repository.engine.dispose()


def test_parallel_scheduler_composes_with_spawned_process_executor(
    tmp_path: Path,
    protocol_fixture: ProtocolFixture,
) -> None:
    execution_plan = replace(
        protocol_fixture.execution_plan,
        tasks=tuple(
            replace(
                task,
                capability=replace(
                    task.capability,
                    requested_resources=replace(
                        task.capability.requested_resources,
                        wall_time_seconds=5,
                    ),
                ),
            )
            for task in protocol_fixture.execution_plan.tasks
        ),
    )
    fixture = replace(protocol_fixture, execution_plan=execution_plan)
    capacity = ExecutionResourceCapacity(
        cpu_cores=2,
        memory_bytes=2_000_000,
        gpu_devices=0,
        enforced_cpu_limit=True,
        enforced_address_space_limit=True,
        enforced_no_network=False,
    )
    scheduler, repository, plane, _receipts, _runners = _runtime(
        tmp_path,
        fixture,
        maximum_parallel_tasks=2,
        parallel_resource_capacity=capacity,
    )
    executor = ConcurrencyObservingExecutor(LocalProcessExecutor(scratch_root=plane.root))
    scheduler.executor = executor

    result = scheduler.execute(
        fixture.run_plan.run_plan_id,
        execution_plan,
    )

    assert result.status is OperationalStatus.SUCCEEDED, executor.errors
    assert executor.peak_development == 2
    assert not set(executor.development_affinities["develop-a"]).intersection(
        executor.development_affinities["develop-b"]
    )
    repository.engine.dispose()


def test_resource_lock_fails_closed_as_blocked_not_failed(
    tmp_path: Path,
    protocol_fixture: ProtocolFixture,
) -> None:
    locks = ResourceLockManager()
    locks.reserve_for_test("reference-simulator")
    scheduler, repository, _plane_value, _receipts, runners = _runtime(
        tmp_path,
        protocol_fixture,
        lock_manager=locks,
    )
    result = scheduler.execute(
        protocol_fixture.run_plan.run_plan_id,
        protocol_fixture.execution_plan,
    )
    assert result.status is OperationalStatus.BLOCKED
    assert result.failed_task_ids == ()
    assert result.blocked_task_ids == (
        "develop-a",
        "develop-b",
        "evaluate",
        "freeze",
        "prepare",
        "report",
    )
    assert all(not runner.contexts for runner in runners)
    repository.engine.dispose()


def test_resource_lock_blocks_are_distinct_and_resume_after_release(
    tmp_path: Path,
    protocol_fixture: ProtocolFixture,
) -> None:
    locks = ResourceLockManager()
    locks.reserve_for_test("reference-simulator")
    scheduler, repository, _plane_value, _receipts, _runners = _runtime(
        tmp_path,
        protocol_fixture,
        lock_manager=locks,
    )
    run_id = protocol_fixture.run_plan.run_plan_id

    first = scheduler.execute(run_id, protocol_fixture.execution_plan)
    second = scheduler.execute(run_id, protocol_fixture.execution_plan)
    assert first.status is OperationalStatus.BLOCKED
    assert second.status is OperationalStatus.BLOCKED

    prepare_blocks = tuple(
        attempt for attempt in repository.attempts(run_id) if attempt.task_id == "prepare"
    )
    assert tuple(attempt.ordinal for attempt in prepare_blocks) == (1, 2)
    assert len({attempt.attempt_id for attempt in prepare_blocks}) == 2
    assert all(
        attempt.reason_code == TaskBlockReason.RESOURCE_LOCK_UNAVAILABLE.value
        and attempt.block_kind is TaskBlockKind.RETRYABLE
        for attempt in prepare_blocks
    )

    locks.release(("reference-simulator",))
    resumed = scheduler.execute(run_id, protocol_fixture.execution_plan)
    assert resumed.status is OperationalStatus.SUCCEEDED
    assert resumed.completed_task_ids == (
        "develop-a",
        "develop-b",
        "evaluate",
        "freeze",
        "prepare",
        "report",
    )
    assert resumed.failed_task_ids == ()
    assert resumed.blocked_task_ids == ()
    prepare_history = tuple(
        attempt for attempt in repository.attempts(run_id) if attempt.task_id == "prepare"
    )
    assert tuple(attempt.disposition for attempt in prepare_history) == (
        TaskAttemptDisposition.BLOCKED,
        TaskAttemptDisposition.BLOCKED,
        TaskAttemptDisposition.SUCCEEDED,
    )
    assert prepare_history[-1].ordinal == 3
    repository.engine.dispose()


def test_resource_capacity_block_closes_attempt_and_recovers_without_retry_loss(
    tmp_path: Path,
    protocol_fixture: ProtocolFixture,
) -> None:
    executor = RecoveringCapacityExecutor()
    scheduler, repository, _plane_value, _receipts, _runners = _runtime(
        tmp_path,
        protocol_fixture,
        executor=executor,
    )
    run_id = protocol_fixture.run_plan.run_plan_id

    blocked = scheduler.execute(run_id, protocol_fixture.execution_plan)
    assert blocked.status is OperationalStatus.BLOCKED
    prepare_block = next(
        attempt for attempt in repository.attempts(run_id) if attempt.task_id == "prepare"
    )
    assert prepare_block.disposition is TaskAttemptDisposition.BLOCKED
    assert prepare_block.reason_code == (TaskBlockReason.RESOURCE_COMPUTABILITY_UNAVAILABLE.value)
    assert prepare_block.block_kind is TaskBlockKind.RETRYABLE
    assert prepare_block.lease_id is None

    executor.capacity_available = True
    resumed = scheduler.execute(run_id, protocol_fixture.execution_plan)
    assert resumed.status is OperationalStatus.SUCCEEDED
    assert resumed.failed_task_ids == ()
    assert resumed.blocked_task_ids == ()
    prepare_history = tuple(
        attempt for attempt in repository.attempts(run_id) if attempt.task_id == "prepare"
    )
    assert tuple(attempt.disposition for attempt in prepare_history) == (
        TaskAttemptDisposition.BLOCKED,
        TaskAttemptDisposition.SUCCEEDED,
    )
    assert tuple(attempt.ordinal for attempt in prepare_history) == (1, 2)
    repository.engine.dispose()


def test_exhausted_resource_capacity_blocks_close_as_blocked_and_replay(
    tmp_path: Path,
    protocol_fixture: ProtocolFixture,
) -> None:
    executor = RecoveringCapacityExecutor()
    scheduler, repository, _plane_value, _receipts, runners = _runtime(
        tmp_path,
        protocol_fixture,
        executor=executor,
    )
    run_id = protocol_fixture.run_plan.run_plan_id
    maximum_attempts = next(
        task.maximum_attempts
        for task in protocol_fixture.execution_plan.tasks
        if task.task_id == "prepare"
    )

    results = tuple(
        scheduler.execute(run_id, protocol_fixture.execution_plan) for _ in range(maximum_attempts)
    )

    assert all(result.status is OperationalStatus.BLOCKED for result in results)
    prepare_history = tuple(
        attempt for attempt in repository.attempts(run_id) if attempt.task_id == "prepare"
    )
    assert len(prepare_history) == maximum_attempts
    assert all(
        attempt.disposition is TaskAttemptDisposition.BLOCKED
        and attempt.block_kind is TaskBlockKind.RETRYABLE
        for attempt in prepare_history
    )
    calls_before_replay = sum(len(runner.contexts) for runner in runners)

    replayed = scheduler.execute(run_id, protocol_fixture.execution_plan)

    assert replayed.status is OperationalStatus.BLOCKED
    assert replayed.failed_task_ids == ()
    assert "prepare" in replayed.blocked_task_ids
    assert sum(len(runner.contexts) for runner in runners) == calls_before_replay
    repository.engine.dispose()


def test_durable_block_remains_stopped_across_resume(
    tmp_path: Path,
    protocol_fixture: ProtocolFixture,
) -> None:
    scheduler, repository, _plane_value, _receipts, runners = _runtime(
        tmp_path,
        protocol_fixture,
    )
    run_id = protocol_fixture.run_plan.run_plan_id
    repository.register_run(run_id, protocol_fixture.execution_plan.fingerprint())
    repository.block_task(
        run_id,
        "prepare",
        TaskBlockReason.DEPENDENCY_FAILED,
    )

    first = scheduler.execute(run_id, protocol_fixture.execution_plan)
    second = scheduler.execute(run_id, protocol_fixture.execution_plan)
    assert first.status is OperationalStatus.BLOCKED
    assert second.status is OperationalStatus.BLOCKED
    assert not _runner(runners, "reference.prepare").contexts
    prepare_history = tuple(
        attempt for attempt in repository.attempts(run_id) if attempt.task_id == "prepare"
    )
    assert len(prepare_history) == 1
    assert prepare_history[0].block_kind is TaskBlockKind.DURABLE
    repository.engine.dispose()


def test_receipt_decoder_rejects_noncanonical_and_partial_pairs(
    tmp_path: Path,
    protocol_fixture: ProtocolFixture,
) -> None:
    scheduler, repository, _plane_value, receipts, _runners = _runtime(
        tmp_path,
        protocol_fixture,
    )
    run_id = protocol_fixture.run_plan.run_plan_id
    scheduler.execute(run_id, protocol_fixture.execution_plan)
    attempt = repository.attempts(run_id)[0]
    receipt = receipts.read(attempt.run_id, attempt.task_id, attempt.attempt_id)
    assert receipt is not None
    relative = ExternalTaskReceiptStore._relative_path(
        attempt.run_id,
        attempt.task_id,
        attempt.attempt_id,
    )
    payload = (tmp_path / relative).read_bytes()
    manifest_payload = (tmp_path / f"{relative}.manifest.json").read_bytes()
    assert decode_task_receipt(payload) == receipt
    assert decode_artifact_manifest(manifest_payload).logical.content_sha256 == (
        receipt.fingerprint()
    )
    with pytest.raises(CanonicalizationError, match="not canonical"):
        decode_task_receipt(payload + b" ")

    (tmp_path / f"{relative}.manifest.json").unlink()
    with pytest.raises(ArtifactIdentityConflict, match="partial"):
        receipts.read(attempt.run_id, attempt.task_id, attempt.attempt_id)
    repository.engine.dispose()
