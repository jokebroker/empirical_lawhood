"""Authenticated preparation of the existing scheduler's bounded retry route."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from empirical_lawhood.infrastructure.artifacts import ExternalArtifactPlane
from empirical_lawhood.infrastructure.bounded_process import run_bounded_command
from empirical_lawhood.infrastructure.recovery import ExternalRunRecoveryStore
from empirical_lawhood.infrastructure.task_receipts import ExternalTaskReceiptStore
from empirical_lawhood.infrastructure.bounded_io import read_bounded_bytes
from empirical_lawhood.infrastructure.execution_resource_envelopes import ExternalExecutionResourceEnvelopeStore
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.runtime.execution_envelope import RunExecutionResourceEnvelopeMachine, RunExecutionResourceEnvelope
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.recovery import ProtocolRunRecoveryIndex, ProtocolRunRecoveryTerminalEvent, TaskRecoveryEvent
from empirical_lawhood.runtime.retry_amendment import OperationalSourceAmendment, LeaseExpiryRetryAmendment, DiagnosedPureTaskRetryAmendment, ChainedPureTaskRetryAmendment, CompletedRepairRetryAmendment, ElapsedClosureRetryAmendment, ResourcePureTaskRetryAmendment, ChainedResourceRetryAmendment, RetainedCustodyCompletionAmendment
from empirical_lawhood.runtime.campaign_elapsed_budget import CampaignElapsedBudgetSpec, CampaignElapsedBudgetMachine, CampaignElapsedLedger, DurableCampaignElapsedBudgetCoordinator, DurableCampaignElapsedBudgetStore


def closure_elapsed_coordinator(
    amendment: ElapsedClosureRetryAmendment,
    original: CampaignElapsedBudgetSpec,
    store: DurableCampaignElapsedBudgetStore,
) -> tuple[DurableCampaignElapsedBudgetCoordinator, CampaignElapsedLedger]:
    """Authenticate retained charges and inspect the supplemental ledger without writes."""
    prior = store.load(amendment.prior_elapsed_ledger.object_id)
    amendment.validate_elapsed(original, prior)
    budget = amendment.closure_budget(original)
    ledger_id = f"campaign-elapsed-ledger.{budget.budget_id}"
    coordinator = DurableCampaignElapsedBudgetCoordinator(
        store, budget=budget, ledger_id=ledger_id
    )
    try:
        ledger = coordinator.current()
    except (KeyError, FileNotFoundError):
        ledger = CampaignElapsedBudgetMachine.initial(ledger_id=ledger_id, budget=budget)
    return coordinator, ledger


# A repair release may change these operational seams, never a scientific
# adapter, capability registry, plan compiler, configuration or dependency lock.
OPERATIONAL_REPAIR_PATHS = frozenset(
    {
        "src/empirical_lawhood/api/execution.py",
        "src/empirical_lawhood/api/facade.py",
        "src/empirical_lawhood/api/results.py",
        "src/empirical_lawhood/infrastructure/execution.py",
        "src/empirical_lawhood/infrastructure/execution_resource_envelopes.py",
        "src/empirical_lawhood/infrastructure/recovery.py",
        "src/empirical_lawhood/infrastructure/retry_amendment.py",
        "src/empirical_lawhood/runtime/recovery.py",
        "src/empirical_lawhood/runtime/retry_amendment.py",
        "src/empirical_lawhood/infrastructure/sql/operations.py",
    }
)


def verify_operational_repair_source(
    repository: Path, amendment: LeaseExpiryRetryAmendment | OperationalSourceAmendment
) -> None:
    from empirical_lawhood.infrastructure.source_origin import require_executing_target_source

    require_executing_target_source(repository)
    def git(*args: str) -> str:
        result = run_bounded_command(
            ["git", *args],
            cwd=repository,
            timeout_seconds=10,
            maximum_stdout_bytes=1024 * 1024,
            maximum_stderr_bytes=16 * 1024,
        )
        if result.returncode:
            raise ValueError("operational repair Git inspection failed")
        return result.stdout.decode("utf-8").strip()

    if (
        git("status", "--porcelain")
        or git("rev-parse", "HEAD") != amendment.repair_implementation_commit
    ):
        raise ValueError("operational repair requires its exact clean release")
    changed = set(
        git(
            "diff",
            "--name-only",
            amendment.original_implementation_commit,
            amendment.repair_implementation_commit,
            "--",
        ).splitlines()
    )
    executable_changes = {p for p in changed if not p.startswith(("tests/", "docs/"))}
    allowed = OPERATIONAL_REPAIR_PATHS
    if isinstance(amendment, DiagnosedPureTaskRetryAmendment):
        allowed = allowed | {
            'src/empirical_lawhood/adapters/methods/finite_response_law/control_provider.py',
            "src/empirical_lawhood/infrastructure/sql/database.py",
        }
    if isinstance(amendment, CompletedRepairRetryAmendment):
        allowed = allowed | {
            'src/empirical_lawhood/adapters/methods/finite_response_law/evaluation_provider.py',
            'src/empirical_lawhood/adapters/methods/finite_response_law/evaluation_results.py',
        }
    if isinstance(amendment, (ResourcePureTaskRetryAmendment, ChainedResourceRetryAmendment)):
        allowed = allowed | {
            'src/empirical_lawhood/adapters/methods/finite_response_law/preparation_policy_screen.py',
        }
    if isinstance(amendment, ChainedResourceRetryAmendment):
        allowed = allowed | {
            'src/empirical_lawhood/adapters/methods/finite_response_law/preparation_policy_provider.py',
        }
    if isinstance(amendment, RetainedCustodyCompletionAmendment):
        allowed = allowed | {"src/empirical_lawhood/runtime/execution_envelope.py"}
    if not executable_changes or not executable_changes <= allowed:
        raise ValueError("operational repair changes science or another implementation seam")
    if (
        Path(__file__).resolve()
        != repository.resolve() / "src/empirical_lawhood/infrastructure/retry_amendment.py"
    ):
        raise ValueError("operational repair is imported outside its authenticated worktree")


@dataclass(frozen=True, slots=True)
class PreparedOperationalRetry:
    index: ProtocolRunRecoveryIndex
    prior_terminal: ProtocolRunRecoveryTerminalEvent
    resource_checkpoint: RunExecutionResourceEnvelope
    prior_index: ProtocolRunRecoveryIndex
    retained_failures: tuple[TaskRecoveryEvent, ...] = ()


def prepare_operational_retry(
    amendment: LeaseExpiryRetryAmendment,
    *,
    plan: ProtocolExecutionPlan,
    original_index: ProtocolRunRecoveryIndex,
    original_envelope: RunExecutionResourceEnvelope,
    recovery_store: ExternalRunRecoveryStore,
    receipt_store: ExternalTaskReceiptStore,
    artifact_plane: ExternalArtifactPlane,
) -> PreparedOperationalRetry:
    """No writes, workers or source contact; authenticate exact prior custody."""
    if isinstance(amendment, CompletedRepairRetryAmendment):
        original_index = amendment.predecessor_index(plan, original_index)
        original_envelope = ExternalExecutionResourceEnvelopeStore(artifact_plane.root).load(
            amendment.completed_repair.envelope_id
        )
    saved_index = recovery_store.read_index(
        original_index.index_relative_path, expected_schema=original_index.SCHEMA
    )
    terminal = recovery_store.read_terminal(original_index)
    if saved_index != original_index or terminal is None:
        raise ValueError("operational retry lacks the exact persisted index and failed terminal")
    amendment.validate_prior(plan, original_index, terminal, original_envelope)
    if isinstance(amendment, ChainedPureTaskRetryAmendment):
        prior = decode_canonical_bytes(
            read_bounded_bytes(
                artifact_plane.root.resolve(amendment.prior_repair_path, for_write=False),
                maximum_bytes=65536,
            ),
            ChainedResourceRetryAmendment
            if isinstance(amendment, RetainedCustodyCompletionAmendment)
            else ResourcePureTaskRetryAmendment
            if isinstance(amendment, ChainedResourceRetryAmendment)
            else DiagnosedPureTaskRetryAmendment,
            maximum_bytes=65536,
        )
        prior.validate_prior(plan, original_index, terminal, original_envelope)
        prior_index = prior.recovery_index(plan, original_index)
        if isinstance(amendment, RetainedCustodyCompletionAmendment) and amendment.retained_terminal_path != prior_index.terminal_event_relative_path:
            raise ValueError("minimal completion substitutes its retained terminal locator")
        saved_prior = recovery_store.read_index(
            prior_index.index_relative_path, expected_schema=prior_index.SCHEMA
        )
        prior_terminal = recovery_store.read_terminal(prior_index)
        prior_resource = ExternalExecutionResourceEnvelopeStore(artifact_plane.root).load(
            prior.envelope_id
        )
        if (
            prior.identity != amendment.prior_repair
            or saved_prior != prior_index
            or prior_terminal is None
            or ObjectIdentity.from_record(prior_terminal.terminal_event_id, prior_terminal)
            != amendment.prior_repair_terminal
            or ObjectIdentity.from_record(prior_resource.envelope_id, prior_resource)
            != amendment.prior_repair_resource
            or prior.task_ids != amendment.task_ids
            or prior_terminal.operational_status != "FAILED"
            or prior_terminal.failed_task_ids != amendment.task_ids
            or prior_terminal.completed_task_ids != terminal.completed_task_ids
            or prior_resource.spec != prior.resource_spec(original_envelope.spec)
            or prior_resource.child_token_states != original_envelope.child_token_states
        ):
            raise ValueError("repair branch substitutes its failed predecessor or drops charges")
        target_cells = {original_envelope.spec.cell_for_task(t).cell_id for t in amendment.task_ids}
        if any(
            c != original_envelope.cell(c.cell_id)
            for c in prior_resource.cells
            if c.cell_id not in target_cells
        ):
            raise ValueError("repair branch cannot discard new work in its predecessor")
        for task_id in amendment.task_ids:
            binding = prior_index.task(task_id)
            event = recovery_store.read_event(prior_index, binding.attempts[1].event_relative_path)
            cell = prior_resource.cell(prior_resource.spec.cell_for_task(task_id).cell_id)
            if (
                any(
                    receipt_store.read(prior_index.run_id, task_id, a.attempt_id) is not None
                    for a in binding.attempts
                )
                or cell.attempt_ids != tuple(a.attempt_id for a in binding.attempts)
                or cell.state.value != "RESOURCE_STOP"
                or cell.first_valid_receipt is not None
                or event is None
                or event.failure is None
                or event.receipt_id is not None
                or event.output_materialization_ids
                or event.failure.scientific_evidence
                or event.failure.diagnostic_sha256 != amendment.failure_diagnostic_sha256
                or event.reason_codes != (amendment.failure_reason,)
            ):
                raise ValueError("repair branch predecessor is not receipt-free pure failure")
    index = amendment.recovery_index(plan, original_index)
    entered = artifact_plane.root.resolve(index.index_relative_path, for_write=False).exists()
    if (
        entered
        and recovery_store.read_index(index.index_relative_path, expected_schema=index.SCHEMA)
        != index
    ):
        raise ValueError("persisted continuation differs from its approved amendment")
    for task_id in amendment.task_ids:
        binding = original_index.task(task_id)
        if (
            receipt_store.read(original_index.run_id, task_id, binding.attempts[0].attempt_id)
            is not None
        ):
            raise ValueError("operational retry would repeat a completed receipt")
        if (
            not entered
            and artifact_plane.root.resolve(
                f"runs/{original_index.run_id}/outputs/{task_id}", for_write=False
            ).exists()
        ):
            raise ValueError("interrupted task has output custody requiring publication recovery")
        event = recovery_store.read_event(original_index, binding.attempts[0].event_relative_path)
        if isinstance(amendment, DiagnosedPureTaskRetryAmendment):
            if (
                event is None
                or event.task_id != task_id
                or event.attempt_id != binding.attempts[0].attempt_id
                or event.reason_codes != (amendment.failure_reason,)
                or event.receipt_id is not None
                or event.output_materialization_ids
                or event.failure is None
                or event.failure.scientific_evidence
                or event.failure.diagnostic_sha256 != amendment.failure_diagnostic_sha256
            ):
                raise ValueError("pure-task retry lacks its exact prior failure diagnostic")
        elif event is not None:
            raise ValueError("operational retry target already has a durable task event")
    if isinstance(amendment, DiagnosedPureTaskRetryAmendment):
        checkpoint = amendment.resource_checkpoint(original_envelope)
        index.validate_plan(plan, retry_amendment=amendment)
        retained_failures = []
        if isinstance(amendment, CompletedRepairRetryAmendment):
            for binding in original_index.tasks:
                for candidate in binding.attempts:
                    event = recovery_store.read_event(original_index, candidate.event_relative_path)
                    if event is not None and event.disposition.value == "FAILED":
                        if (
                            event.failure is None
                            or event.receipt_id is not None
                            or event.failure.scientific_evidence
                        ):
                            raise ValueError("closure predecessor failure is not operational")
                        retained_failures.append(event)
        return PreparedOperationalRetry(
            index, terminal, checkpoint, original_index, tuple(retained_failures)
        )
    checkpoint = replace(
        original_envelope,
        envelope_id=amendment.envelope_id,
        spec=amendment.resource_spec(original_envelope.spec),
    )
    for task_id in amendment.task_ids:
        cell = checkpoint.cell(checkpoint.spec.cell_for_task(task_id).cell_id)
        assert cell.current_attempt_id is not None
        checkpoint = RunExecutionResourceEnvelopeMachine.observe_operational_failure(
            checkpoint,
            event_id=f"{amendment.amendment_id}.{task_id}.lease-expired",
            occurred_at_utc=amendment.created_at_utc,
            cell_id=cell.cell_id,
            attempt_id=cell.current_attempt_id,
            reason_code="lease-expired",
        )
    index.validate_plan(plan, retry_amendment=amendment)
    return PreparedOperationalRetry(index, terminal, checkpoint, original_index)
