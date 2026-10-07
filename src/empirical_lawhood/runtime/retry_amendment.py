"""Bounded operational retries of an unchanged, interrupted simulation plan.

This is an explicit owner amendment, not an automatic scientific retry policy.
The failed terminal, original plan and every completed receipt remain immutable.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.time import parse_utc_timestamp
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_sha256,
    validate_relative_locator,
    validate_stable_id,
)
from empirical_lawhood.runtime.execution_envelope import ExecutionResourceEnvelopeSpec, ExecutionResourceEnvelopeState, ExecutionResourceEnvelopeEventKind, reconstruct_execution_resource_envelope, RunExecutionResourceEnvelope
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan, ExecutionPlan
from empirical_lawhood.runtime.campaign_elapsed_budget import CampaignElapsedBudgetSpec, CampaignElapsedLedger
from empirical_lawhood.runtime.recovery import RecoveryTerminalDisposition, ProtocolRunRecoveryIndex, RunRecoveryIndex, ProtocolRunRecoveryTerminalEvent, _task_binding


@dataclass(frozen=True, slots=True)
class OperationalSourceAmendment(CanonicalRecord):
    """Authorize a bounded operational-only repair release without retrying tasks."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/operational-source-amendment'

    amendment_id: str
    run_id: str
    execution_plan: ObjectIdentity
    obstruction_result_sha256: str
    owner_instruction_sha256: str
    original_implementation_commit: str
    repair_implementation_commit: str
    operational_query_work_limit: int
    created_at_utc: str

    def __post_init__(self) -> None:
        validate_stable_id(self.amendment_id, field_name="amendment_id")
        validate_stable_id(self.run_id, field_name="run_id")
        if self.execution_plan.object_schema != ExecutionPlan.SCHEMA:
            raise ValueError("operational source amendment requires an ExecutionPlan")
        validate_sha256(self.obstruction_result_sha256, field_name="obstruction_result_sha256")
        validate_sha256(self.owner_instruction_sha256, field_name="owner_instruction_sha256")
        for commit in (self.original_implementation_commit, self.repair_implementation_commit):
            if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
                raise ValueError("operational source amendment requires exact Git commits")
        if not 100_000 < self.operational_query_work_limit <= 10_000_000:
            raise ValueError("operational query work limit lies outside its bounded repair range")
        parse_utc_timestamp(self.created_at_utc, field_name="created_at_utc")

    @property
    def identity(self) -> ObjectIdentity:
        return ObjectIdentity.from_record(self.amendment_id, self)


@dataclass(frozen=True, slots=True)
class LeaseExpiryRetryAmendment(CanonicalRecord):
    """One additional attempt for each exact lease-expired, receipt-free task."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/lease-expiry-retry-amendment'

    amendment_id: str
    execution_plan: ObjectIdentity
    prior_recovery_index: ObjectIdentity
    prior_terminal: ObjectIdentity
    prior_resource_envelope: ObjectIdentity
    task_ids: tuple[str, ...]
    owner_instruction_sha256: str
    original_implementation_commit: str
    repair_implementation_commit: str
    created_at_utc: str

    def __post_init__(self) -> None:
        validate_stable_id(self.amendment_id, field_name="amendment_id")
        validate_sha256(self.owner_instruction_sha256, field_name="owner_instruction_sha256")
        require_sorted_unique_strings(self.task_ids, field_name="task_ids", allow_empty=False)
        if len(self.task_ids) > 64:
            raise ValueError("operational amendment exceeds 64 interrupted tasks")
        for commit in (self.original_implementation_commit, self.repair_implementation_commit):
            if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
                raise ValueError("retry implementation identity is not an exact Git commit")
        parse_utc_timestamp(self.created_at_utc, field_name="created_at_utc")
        if self.execution_plan.object_schema != ExecutionPlan.SCHEMA:
            raise ValueError("operational retry requires the current execution plan")
        if self.prior_recovery_index.object_schema != RunRecoveryIndex.SCHEMA:
            raise ValueError("operational retry requires the current recovery index")
        if self.prior_terminal.object_schema != RunRecoveryIndex.TERMINAL_EVENT_SCHEMA:
            raise ValueError("operational retry requires the exact prior terminal")
        if self.prior_resource_envelope.object_schema != RunExecutionResourceEnvelope.SCHEMA:
            raise ValueError("operational retry requires the prior resource prefix")

    @property
    def identity(self) -> ObjectIdentity:
        return ObjectIdentity.from_record(self.amendment_id, self)

    @property
    def wave_id(self) -> str:
        return f"wave.{self.amendment_id}"

    @property
    def envelope_id(self) -> str:
        return f"run-resource-envelope.{self.amendment_id}"

    @property
    def failure_reason(self) -> str:
        return "lease-expired"

    def _validate_identity(
        self,
        plan: ProtocolExecutionPlan,
        index: ProtocolRunRecoveryIndex,
        terminal: ProtocolRunRecoveryTerminalEvent,
        envelope: RunExecutionResourceEnvelope,
    ) -> None:
        index.validate_plan(plan)
        if (
            self.execution_plan != ObjectIdentity.from_record(plan.execution_plan_id, plan)
            or self.original_implementation_commit != plan.implementation_commit
            or self.prior_recovery_index
            != ObjectIdentity.from_record(index.recovery_index_id, index)
            or self.prior_terminal
            != ObjectIdentity.from_record(terminal.terminal_event_id, terminal)
            or self.prior_resource_envelope
            != ObjectIdentity.from_record(envelope.envelope_id, envelope)
            or terminal.recovery_index != self.prior_recovery_index
            or envelope.execution_plan != self.execution_plan
        ):
            raise ValueError("retry amendment substitutes prior execution evidence")
        if terminal.operational_status != "FAILED" or terminal.failed_task_ids != self.task_ids:
            raise ValueError("retry amendment must cover exactly the interrupted failures")

    def validate_prior(
        self,
        plan: ProtocolExecutionPlan,
        index: ProtocolRunRecoveryIndex,
        terminal: ProtocolRunRecoveryTerminalEvent,
        envelope: RunExecutionResourceEnvelope,
    ) -> None:
        self._validate_identity(plan, index, terminal, envelope)
        for task_id in self.task_ids:
            attempts = tuple(
                a
                for a in terminal.attempts
                if a.task_id == task_id
                and a.disposition is not RecoveryTerminalDisposition.BLOCKED
            )
            cell_spec = envelope.spec.cell_for_task(task_id)
            cell = envelope.cell(cell_spec.cell_id)
            if (
                len(attempts) != 1
                or attempts[0].ordinal != 1
                or attempts[0].disposition is not RecoveryTerminalDisposition.FAILED
                or attempts[0].reason_code != "lease-expired"
                or attempts[0].receipt_id is not None
                or index.task(task_id).maximum_attempts != 1
                or cell_spec.maximum_attempts != 1
                or not cell_spec.native_simulator_launch
                or cell.state is not ExecutionResourceEnvelopeState.LAUNCHED
                or cell.attempt_ids != (attempts[0].attempt_id,)
                or cell.current_attempt_id != attempts[0].attempt_id
                or cell.first_valid_receipt is not None
                or cell.published_artifact is not None
                or cell.recovery_candidate_artifact is not None
            ):
                raise ValueError("retry target is not a single interrupted simulator attempt")

    def recovery_index(self, plan: ProtocolExecutionPlan, original: ProtocolRunRecoveryIndex) -> ProtocolRunRecoveryIndex:
        """Derive new locators and one extra candidate without editing the plan."""
        if original.execution_plan != self.execution_plan:
            raise ValueError("retry index binds another execution plan")
        base = RunRecoveryIndex.from_execution_plan(
            plan,
            run_id=original.run_id,
            wave_id=self.wave_id,
            authority_identities=tuple(
                sorted(
                    (*original.authority_identities, self.identity),
                    key=lambda value: value.object_id,
                )
            ),
        )
        tasks = {task.task_id: task for task in plan.tasks}
        return replace(
            base,
            tasks=tuple(
                _task_binding(
                    replace(tasks[binding.task_id], maximum_attempts=2),
                    run_id=base.run_id,
                    wave_id=base.wave_id,
                    topological_ordinal=binding.topological_ordinal,
                    event_schema=base.TASK_EVENT_SCHEMA,
                )
                if binding.task_id in self.task_ids
                else binding
                for binding in base.tasks
            ),
        )

    @property
    def resource_failure_reason(self) -> str:
        return "lease-expired"

    def resource_spec(
        self, original: ExecutionResourceEnvelopeSpec
    ) -> ExecutionResourceEnvelopeSpec:
        """Keep every prior charge; increase only the named cells' retry demand."""
        if original.roster_capacity_branches:
            raise ValueError("retry amendment does not change conditional resource branches")
        cells = tuple(
            replace(cell, maximum_attempts=2) if cell.task_id in self.task_ids else cell
            for cell in original.task_cells
        )
        return replace(
            original,
            envelope_spec_id=f"resource-spec.{self.amendment_id}",
            task_cells=cells,
            allowlisted_retry_reason_codes=tuple(
                sorted({*original.allowlisted_retry_reason_codes, self.resource_failure_reason})
            ),
            child_token_limits=tuple(
                replace(
                    limit,
                    physical_execution_token_limit=sum(
                        c.physical_execution_cost * c.maximum_attempts
                        for c in cells
                        if c.child_id == limit.child_id
                    ),
                    retry_token_limit=sum(
                        c.retry_token_cost * (c.maximum_attempts - 1)
                        for c in cells
                        if c.child_id == limit.child_id
                    ),
                )
                for limit in original.child_token_limits
            ),
        )


@dataclass(frozen=True, slots=True)
class DiagnosedPureTaskRetryAmendment(LeaseExpiryRetryAmendment):
    """Owner-authorized repair of exactly identified, receipt-free pure tasks.

    The prior terminal and resource history remain authoritative. The derived
    resource branch lifts only the named retry-exhaustion stops; every launch,
    reservation, failure, charge and completed publication remains in its replay.
    It does not authorize retrying native acquisition or scientific negatives.
    """

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/diagnosed-pure-task-retry-amendment'
    failure_diagnostic_sha256: str

    def __post_init__(self) -> None:
        LeaseExpiryRetryAmendment.__post_init__(self)
        validate_sha256(self.failure_diagnostic_sha256, field_name="failure_diagnostic_sha256")

    @property
    def failure_reason(self) -> str:
        return "retry-exhaustion"

    @property
    def resource_failure_reason(self) -> str:
        return "RETRY_EXHAUSTION"

    def validate_prior(
        self,
        plan: ProtocolExecutionPlan,
        index: ProtocolRunRecoveryIndex,
        terminal: ProtocolRunRecoveryTerminalEvent,
        envelope: RunExecutionResourceEnvelope,
    ) -> None:
        self._validate_identity(plan, index, terminal, envelope)
        if envelope.spec.jit_graph_signature_manifest is not None:
            raise ValueError("pure-task repair cannot alter JIT execution")
        for task_id in self.task_ids:
            attempts = tuple(
                a
                for a in terminal.attempts
                if a.task_id == task_id
                and a.disposition is not RecoveryTerminalDisposition.BLOCKED
            )
            spec = envelope.spec.cell_for_task(task_id)
            cell = envelope.cell(spec.cell_id)
            events = tuple(e for e in envelope.events if e.cell_id == cell.cell_id)
            if (
                len(attempts) != 1
                or attempts[0].attempt_id
                != index.task(task_id).attempts[0].attempt_id
                or (
                    attempts[0].ordinal != 1
                    and not isinstance(self, (ResourcePureTaskRetryAmendment, ChainedResourceRetryAmendment))
                )
                or attempts[0].disposition is not RecoveryTerminalDisposition.FAILED
                or attempts[0].reason_code != self.failure_reason
                or attempts[0].receipt_id is not None
                or index.task(task_id).maximum_attempts != 1
                or spec.maximum_attempts != 1
                or spec.native_simulator_launch
                or spec.physical_execution_cost != 0
                or spec.resource_budget.source_request_limit != 0
                or cell.state is not ExecutionResourceEnvelopeState.RESOURCE_STOP
                or cell.attempt_ids != (attempts[0].attempt_id,)
                or cell.current_attempt_id != attempts[0].attempt_id
                or cell.first_valid_receipt is not None
                or cell.published_artifact is not None
                or cell.recovery_candidate_artifact is not None
                or cell.last_operational_reason_code != self.resource_failure_reason
                or len(events) != 4
                or tuple(e.kind for e in events)
                != (
                    ExecutionResourceEnvelopeEventKind.RESERVATION,
                    ExecutionResourceEnvelopeEventKind.LAUNCH,
                    ExecutionResourceEnvelopeEventKind.OPERATIONAL_FAILURE,
                    ExecutionResourceEnvelopeEventKind.RESOURCE_STOP,
                )
                or events[-2].reason_code != self.resource_failure_reason
                or events[-2].scientific_terminal is not False
            ):
                raise ValueError("repair target is not a single receipt-free pure-task failure")

    def resource_checkpoint(
        self, original: RunExecutionResourceEnvelope
    ) -> RunExecutionResourceEnvelope:
        cells = {original.spec.cell_for_task(t).cell_id for t in self.task_ids}
        retained = tuple(
            e
            for e in original.events
            if not (
                e.cell_id in cells and e.kind is ExecutionResourceEnvelopeEventKind.RESOURCE_STOP
            )
        )
        return reconstruct_execution_resource_envelope(
            envelope_id=self.envelope_id,
            spec=self.resource_spec(original.spec),
            execution_plan=original.execution_plan,
            events=tuple(replace(e, sequence_number=i) for i, e in enumerate(retained, 1)),
            jit_graph_signature_manifest=None,
        )


@dataclass(frozen=True, slots=True)
class ChainedPureTaskRetryAmendment(DiagnosedPureTaskRetryAmendment):
    "A separately authorized pure-task branch after a failed diagnosed pure-task repair.\n\n    Both failed branches remain immutable. Native charges and cumulative elapsed\n    accounting are retained; only zero-cost, nonacquiring tasks can branch again.\n    The new branch has its own attempt identity, never a reused attempt-002.\n    "

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/chained-pure-task-retry-amendment'
    prior_repair_path: str
    prior_repair: ObjectIdentity
    prior_repair_terminal: ObjectIdentity
    prior_repair_resource: ObjectIdentity

    def __post_init__(self) -> None:
        DiagnosedPureTaskRetryAmendment.__post_init__(self)
        validate_relative_locator(self.prior_repair_path)
        if (
            self.prior_repair.object_schema != DiagnosedPureTaskRetryAmendment.SCHEMA
            or self.prior_repair_terminal.object_schema != RunRecoveryIndex.TERMINAL_EVENT_SCHEMA
            or self.prior_repair_resource.object_schema != RunExecutionResourceEnvelope.SCHEMA
        ):
            raise ValueError("repair branch requires its exact failed diagnosed pure-task predecessor")

    def recovery_index(self, plan: ProtocolExecutionPlan, original: ProtocolRunRecoveryIndex) -> ProtocolRunRecoveryIndex:
        index = DiagnosedPureTaskRetryAmendment.recovery_index(self, plan, original)
        tasks = []
        for binding in index.tasks:
            if binding.task_id in self.task_ids:
                attempt = binding.attempts[1]
                attempt_id = (
                    f"{index.run_id}.{binding.task_id}.repair-{self.fingerprint()[:12]}.attempt-002"
                )
                candidate = replace(
                    attempt,
                    attempt_id=attempt_id,
                    receipt_relative_path=f"runs/{index.run_id}/receipts/{binding.task_id}/{attempt_id}.json",
                )
                binding = replace(binding, attempts=(binding.attempts[0], candidate))
            tasks.append(binding)
        return replace(index, tasks=tuple(tasks))


@dataclass(frozen=True, slots=True)
class CompletedRepairRetryAmendment(DiagnosedPureTaskRetryAmendment):
    "Finish one pure task after a chained pure-task repair has produced completed evidence.\n\n    Retain the successful repair's entire attempt roster and resource journal.\n    Only a new, receipt-free operational failure receives one additional attempt.\n    "

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/completed-repair-retry-amendment'
    completed_repair: ChainedPureTaskRetryAmendment

    def __post_init__(self) -> None:
        DiagnosedPureTaskRetryAmendment.__post_init__(self)
        if (
            len(self.task_ids) != 1
            or set(self.task_ids) & set(self.completed_repair.task_ids)
            or self.execution_plan != self.completed_repair.execution_plan
            or self.original_implementation_commit
            != self.completed_repair.original_implementation_commit
        ):
            raise ValueError("closure repair requires one new pure failure after the same chained pure-task plan")

    def predecessor_index(self, plan: ProtocolExecutionPlan, base: ProtocolRunRecoveryIndex) -> ProtocolRunRecoveryIndex:
        original = replace(
            base,
            authority_identities=tuple(
                a
                for a in base.authority_identities
                if a not in (self.identity, self.completed_repair.identity)
            ),
        )
        return self.completed_repair.recovery_index(plan, original)

    def recovery_index(self, plan: ProtocolExecutionPlan, original: ProtocolRunRecoveryIndex) -> ProtocolRunRecoveryIndex:
        prior = self.predecessor_index(plan, original)
        index = DiagnosedPureTaskRetryAmendment.recovery_index(self, plan, prior)
        tasks = []
        for binding in index.tasks:
            if binding.task_id in self.task_ids:
                attempt = binding.attempts[1]
                attempt_id = (
                    f"{index.run_id}.{binding.task_id}.repair-{self.fingerprint()[:12]}.attempt-002"
                )
                binding = replace(
                    binding,
                    attempts=(
                        binding.attempts[0],
                        replace(
                            attempt,
                            attempt_id=attempt_id,
                            receipt_relative_path=f"runs/{index.run_id}/receipts/{binding.task_id}/{attempt_id}.json",
                        ),
                    ),
                )
            else:
                old = prior.task(binding.task_id)
                binding = replace(
                    old,
                    nonattempt_event_relative_path=binding.nonattempt_event_relative_path,
                    attempts=tuple(
                        replace(
                            a,
                            event_relative_path=a.event_relative_path.replace(
                                f"/recovery/{prior.wave_id}/", f"/recovery/{index.wave_id}/"
                            ),
                        )
                        for a in old.attempts
                    ),
                )
            tasks.append(binding)
        return replace(index, tasks=tuple(tasks))

    def validate_prior(
        self,
        plan: ProtocolExecutionPlan,
        index: ProtocolRunRecoveryIndex,
        terminal: ProtocolRunRecoveryTerminalEvent,
        envelope: RunExecutionResourceEnvelope,
    ) -> None:
        index.validate_plan(plan, retry_amendment=self.completed_repair)
        if (
            self.execution_plan != ObjectIdentity.from_record(plan.execution_plan_id, plan)
            or self.original_implementation_commit != plan.implementation_commit
            or self.prior_recovery_index
            != ObjectIdentity.from_record(index.recovery_index_id, index)
            or self.prior_terminal
            != ObjectIdentity.from_record(terminal.terminal_event_id, terminal)
            or self.prior_resource_envelope
            != ObjectIdentity.from_record(envelope.envelope_id, envelope)
            or terminal.recovery_index != self.prior_recovery_index
            or envelope.execution_plan != self.execution_plan
            or envelope.envelope_id != self.completed_repair.envelope_id
            or terminal.operational_status != "FAILED"
            or terminal.failed_task_ids != self.task_ids
            or not set(self.completed_repair.task_ids) <= set(terminal.completed_task_ids)
            or envelope.spec.jit_graph_signature_manifest is not None
        ):
            raise ValueError("closure repair substitutes the completed predecessor or its charges")
        task_id = self.task_ids[0]
        binding = index.task(task_id)
        attempts = tuple(
            a
            for a in terminal.attempts
            if a.task_id == task_id and a.disposition is not RecoveryTerminalDisposition.BLOCKED
        )
        spec = envelope.spec.cell_for_task(task_id)
        cell = envelope.cell(spec.cell_id)
        events = tuple(e for e in envelope.events if e.cell_id == cell.cell_id)
        if (
            len(attempts) != 1
            or attempts[0].attempt_id != binding.attempts[0].attempt_id
            or attempts[0].disposition is not RecoveryTerminalDisposition.FAILED
            or attempts[0].reason_code != self.failure_reason
            or attempts[0].receipt_id is not None
            or binding.maximum_attempts != 1
            or spec.maximum_attempts != 1
            or spec.native_simulator_launch
            or spec.physical_execution_cost != 0
            or spec.resource_budget.source_request_limit != 0
            or cell.state is not ExecutionResourceEnvelopeState.RESOURCE_STOP
            or cell.attempt_ids != (attempts[0].attempt_id,)
            or cell.current_attempt_id != attempts[0].attempt_id
            or cell.first_valid_receipt is not None
            or cell.published_artifact is not None
            or cell.recovery_candidate_artifact is not None
            or cell.last_operational_reason_code != self.resource_failure_reason
            or tuple(e.kind for e in events)
            != (
                ExecutionResourceEnvelopeEventKind.RESERVATION,
                ExecutionResourceEnvelopeEventKind.LAUNCH,
                ExecutionResourceEnvelopeEventKind.OPERATIONAL_FAILURE,
                ExecutionResourceEnvelopeEventKind.RESOURCE_STOP,
            )
            or events[-2].reason_code != self.resource_failure_reason
            or events[-2].scientific_terminal is not False
        ):
            raise ValueError("closure target is not one receipt-free pure-task failure")

    def resource_spec(
        self, original: ExecutionResourceEnvelopeSpec
    ) -> ExecutionResourceEnvelopeSpec:
        return DiagnosedPureTaskRetryAmendment.resource_spec(
            self, self.completed_repair.resource_spec(original)
        )


@dataclass(frozen=True, slots=True)
class ElapsedClosureRetryAmendment(CompletedRepairRetryAmendment):
    """A finite additional closure allowance linked to the unchanged prior ledger."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/elapsed-closure-retry-amendment'
    prior_elapsed_ledger: ObjectIdentity
    prior_elapsed_budget: ObjectIdentity
    additional_closure_seconds: int
    budget_instruction_sha256: str

    def __post_init__(self) -> None:
        CompletedRepairRetryAmendment.__post_init__(self)
        if (
            self.prior_elapsed_ledger.object_schema != CampaignElapsedLedger.SCHEMA
            or self.prior_elapsed_budget.object_schema != CampaignElapsedBudgetSpec.SCHEMA
            or type(self.additional_closure_seconds) is not int
            or not 0 < self.additional_closure_seconds <= 7200
        ):
            raise ValueError(
                "closure allowance requires exact prior accounting and at most two hours"
            )
        validate_sha256(self.budget_instruction_sha256, field_name="budget_instruction_sha256")

    def closure_budget(self, original: CampaignElapsedBudgetSpec) -> CampaignElapsedBudgetSpec:
        if ObjectIdentity.from_record(original.budget_id, original) != self.prior_elapsed_budget:
            raise ValueError("closure allowance substitutes the original elapsed budget")
        return CampaignElapsedBudgetSpec(
            budget_id=f"closure-budget.{self.amendment_id}",
            campaign_anchor=original.campaign_anchor,
            cumulative_elapsed_ceiling_seconds=self.additional_closure_seconds,
            reserve_fraction=Decimal(0),
        )

    def validate_elapsed(
        self, original: CampaignElapsedBudgetSpec, ledger: CampaignElapsedLedger
    ) -> None:
        self.closure_budget(original)
        if (
            ObjectIdentity.from_record(ledger.ledger_id, ledger) != self.prior_elapsed_ledger
            or ledger.budget != self.prior_elapsed_budget
            or ledger.open_interval_id is not None
        ):
            raise ValueError("closure allowance cannot discard or replace prior elapsed charges")

    def validate_prior(
        self,
        plan: ProtocolExecutionPlan,
        index: ProtocolRunRecoveryIndex,
        terminal: ProtocolRunRecoveryTerminalEvent,
        envelope: RunExecutionResourceEnvelope,
    ) -> None:
        CompletedRepairRetryAmendment.validate_prior(self, plan, index, terminal, envelope)
        remaining = set(t.task_id for t in plan.tasks) - set(terminal.completed_task_ids)
        for task_id in remaining:
            cell = envelope.spec.cell_for_task(task_id)
            if (
                cell.native_simulator_launch
                or cell.physical_execution_cost
                or cell.resource_budget.source_request_limit
            ):
                raise ValueError("additional closure time cannot admit new source acquisition")


@dataclass(frozen=True, slots=True)
class ResourcePureTaskRetryAmendment(DiagnosedPureTaskRetryAmendment):
    "Retry receipt-free pure work in a resource-envelope campaign."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/resource-pure-task-retry-amendment'


@dataclass(frozen=True, slots=True)
class ChainedResourceRetryAmendment(ChainedPureTaskRetryAmendment):
    "One separately authorized pure-task branch after a failed resource-envelope pure-task repair.\n\n    Reuse the chained pure-task repair's custody checks and distinct attempt identity, retaining the resource-envelope pure-task\n    predecessor and its failed terminal without authorizing native acquisition.\n    "

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/chained-resource-retry-amendment'

    def __post_init__(self) -> None:
        DiagnosedPureTaskRetryAmendment.__post_init__(self)
        validate_relative_locator(self.prior_repair_path)
        if (
            self.prior_repair.object_schema != ResourcePureTaskRetryAmendment.SCHEMA
            or self.prior_repair_terminal.object_schema != RunRecoveryIndex.TERMINAL_EVENT_SCHEMA
            or self.prior_repair_resource.object_schema != RunExecutionResourceEnvelope.SCHEMA
        ):
            raise ValueError("repair branch requires its exact failed resource-envelope pure-task predecessor")


@dataclass(frozen=True, slots=True)
class RetainedCustodyCompletionAmendment(ChainedResourceRetryAmendment):
    "Pure-task completion reusing the successful custody of a failed chained resource-repair wave.\n\n    The retained terminal authenticates completed receipts. Only their consumers'\n    actual inputs are opened; completed tasks are never redispatched.\n    "

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/retained-custody-completion-amendment'
    retained_terminal_path: str

    def __post_init__(self) -> None:
        DiagnosedPureTaskRetryAmendment.__post_init__(self)
        validate_relative_locator(self.prior_repair_path)
        validate_relative_locator(self.retained_terminal_path)
        if (
            self.prior_repair.object_schema != ChainedResourceRetryAmendment.SCHEMA
            or self.prior_repair_terminal.object_schema != RunRecoveryIndex.TERMINAL_EVENT_SCHEMA
            or self.prior_repair_resource.object_schema != RunExecutionResourceEnvelope.SCHEMA
        ):
            raise ValueError("minimal pure-task completion requires its failed chained resource-repair predecessor")
