"Finite response-law evaluation native source binding: persisted choices precede every parent effect."

from collections.abc import Callable
from dataclasses import replace
from typing import cast

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.capabilities import CapabilityManifest, CapabilityRegistry
from empirical_lawhood.runtime.controller_evaluation_nested import (
    DurablePreparedExecutionEventStore,
    PreparedExecutionEventKind,
)
from empirical_lawhood.runtime.execution import RunnerResult, TaskContext, TaskRunner
from empirical_lawhood.adapters.methods.finite_response_law.control_locking import execution_prefix_id
from empirical_lawhood.adapters.methods.finite_response_law.control_parent import authenticate_parent_reservations
from empirical_lawhood.adapters.methods.finite_response_law.control_records import FiniteResponseLawRootControlLock, FiniteResponseLawRootParentJoin
from empirical_lawhood.adapters.simulators.response_geometry_prospective.provider import decode_port
from .contracts import FiniteResponseLawNativeInvocation
from .evaluation_contracts import FiniteResponseLawEvaluationConfig, FiniteResponseLawEvaluationInvocation
from .assigned_contracts import FiniteResponseLawAssignedEvaluationConfig, FiniteResponseLawAssignedEvaluationInvocation
from .provider import FiniteResponseLawSourceTask, FiniteResponseLawSourceProvider
from .source_outputs import FiniteResponseLawNativeTaskResult


def control_dependency(
    invocation: FiniteResponseLawEvaluationInvocation | FiniteResponseLawAssignedEvaluationInvocation,
) -> str | None:
    if invocation.phase == "prefix":
        return None
    return f"{invocation.root.stage_unit}.{'lock-choices' if invocation.phase == 'parent' else 'join-parent'}"


class _GuardedNativeTask(FiniteResponseLawSourceTask):
    def __init__(
        self,
        manifest: CapabilityManifest,
        config: FiniteResponseLawEvaluationConfig | FiniteResponseLawAssignedEvaluationConfig,
        guard: FiniteResponseLawRootControlLock | FiniteResponseLawRootParentJoin,
    ) -> None:
        super().__init__(manifest, config)
        self.guard = guard

    def validate_predecessor(
        self, task: FiniteResponseLawNativeInvocation, previous: FiniteResponseLawNativeTaskResult
    ) -> None:
        expected = (
            self.guard.forecast.prefix
            if isinstance(self.guard, FiniteResponseLawRootControlLock)
            else self.guard.parent
        )
        if previous != expected:
            raise ValueError("Finite response-law evaluation physical predecessor differs from the immutable control guard")


class FiniteResponseLawEvaluationSourceTask(FiniteResponseLawSourceTask):
    def __init__(
        self,
        manifest: CapabilityManifest,
        config: FiniteResponseLawEvaluationConfig | FiniteResponseLawAssignedEvaluationConfig,
        store: DurablePreparedExecutionEventStore,
    ) -> None:
        super().__init__(manifest, config)
        self.evaluation_config, self.store = config, store

    def _execute(
        self, context: TaskContext, progress: Callable[[int], None] | None
    ) -> RunnerResult:
        try:
            invocation = self.invocations.get(context.task_id)
            if type(invocation) is not FiniteResponseLawAssignedEvaluationInvocation:
                raise ValueError("Finite response-law evaluation source is outside its exact fresh census")
            dependency = control_dependency(invocation)
            if dependency is None:
                return super()._execute(context, progress)
            receipts = tuple(r for r in context.dependency_receipts if r.task_id == dependency)
            if len(receipts) != 1:
                raise ValueError("Finite response-law evaluation native effect has no prior committed control task receipt")
            controls = tuple(
                p
                for p in context.input_ports
                if p.materialization_id in receipts[0].output_materialization_ids
            )
            expected_schema = (
                FiniteResponseLawRootControlLock.SCHEMA
                if invocation.phase == "parent"
                else FiniteResponseLawRootParentJoin.SCHEMA
            )
            if len(controls) != 1 or controls[0].payload_schema != expected_schema:
                raise ValueError("Finite response-law evaluation source control guard omits or substitutes its output")
            guard: FiniteResponseLawRootControlLock | FiniteResponseLawRootParentJoin
            if invocation.phase == "parent":
                guard = decode_port(controls[0], FiniteResponseLawRootControlLock, maximum=16 * 1024**2)
                lock = guard
                authenticate_parent_reservations(lock, self.store, before_parent=True)
            else:
                guard = decode_port(controls[0], FiniteResponseLawRootParentJoin, maximum=16 * 1024**2)
                lock = guard.lock
                authenticate_parent_reservations(lock, self.store, before_parent=False)
                for design, returned in zip(lock.designs, guard.returns, strict=True):
                    prefix_id = execution_prefix_id(design.root_id, design.policy_id)
                    prefix = self.store.load_prepared(prefix_id)
                    events = tuple(
                        e
                        for e in prefix.events
                        if e.kind is PreparedExecutionEventKind.PARENT_RETURNED
                    )
                    if len(events) != 1 or events[0].subject != ObjectIdentity.from_record(
                        returned.return_id, returned
                    ):
                        raise ValueError("Finite response-law evaluation future lacks its actual persisted parent return")
                    if (
                        self.store.read_prepared_record(
                            prefix_id=prefix_id,
                            subject=events[0].subject,
                            artifact=events[0].subject_artifact,
                        )
                        != returned.canonical_bytes()
                    ):
                        raise ValueError("Finite response-law evaluation future substitutes its actual parent return")
            if (
                lock.forecast.root != invocation.root
                or lock.forecast.prefix.invocation.source != invocation.source
            ):
                raise ValueError("Finite response-law evaluation source guard substitutes the fresh root or frozen source")
            # The production DAG and outer task receipt retain BOTH dependencies.
            # Delegate only native inputs to the unchanged physical marcher;
            # consuming a guard never consumes/reopens the predecessor stream.
            remaining = tuple(p for p in context.input_ports if p not in controls)
            physical_context = replace(
                context,
                input_ports=remaining,
                input_bindings=tuple(p.binding for p in remaining),
                dependency_receipts=tuple(
                    r for r in context.dependency_receipts if r is not receipts[0]
                ),
            )
            task = _GuardedNativeTask(self.manifest, self.evaluation_config, guard)
            return task._execute(physical_context, progress)
        finally:
            for port in context.input_ports:
                port.close()


class FiniteResponseLawEvaluationSourceProvider(FiniteResponseLawSourceProvider):
    """Same native provider/output contracts, with exact causal dependencies."""

    def bind_control_store(self, store: DurablePreparedExecutionEventStore) -> None:
        if type(self.config) is not FiniteResponseLawAssignedEvaluationConfig or hasattr(self, "control_store"):
            raise ValueError("Finite response-law evaluation source store must bind once to the exact evaluation provider")
        self.control_store = store

    def expected_dependencies(self, invocation: FiniteResponseLawNativeInvocation) -> tuple[str, ...]:
        if type(invocation) is not FiniteResponseLawAssignedEvaluationInvocation:
            raise ValueError("Finite response-law evaluation source changes its versioned invocation")
        dependency = control_dependency(invocation)
        return tuple(
            sorted((*invocation.dependency_task_ids, *((dependency,) if dependency else ())))
        )

    def runners(
        self, registry: CapabilityRegistry, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records or not hasattr(self, "control_store"):
            raise ValueError("Finite response-law evaluation source lacks its installed registry or durable control store")
        return (
            cast(
                TaskRunner,
                FiniteResponseLawEvaluationSourceTask(
                    self.manifest, cast(FiniteResponseLawAssignedEvaluationConfig, self.config), self.control_store
                ),
            ),
        )
