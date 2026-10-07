"Join actual finite response-law evaluation parent receipts to frozen choices; cancel without reselecting."

from collections.abc import Callable
from typing import cast

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.status import OperationalStatus
from empirical_lawhood.planning.nested_controller_evaluation import PreparedFutureRole
from empirical_lawhood.runtime.artifacts import CanonicalTaskReceipt
from empirical_lawhood.runtime.controller_compiler import CompiledLeastMagnitudeControllerStudy
from empirical_lawhood.runtime.controller_runtime import CommitmentDisposition
from empirical_lawhood.runtime.controller_evaluation_nested import DurablePreparedExecutionEventStore, PreparedExecutionEventKind, PreparedParentReturn, PreparedParentDisposition, ProspectiveEvaluationBindingCoordinator
from empirical_lawhood.adapters.simulators.finite_response_law.source_outputs import FiniteResponseLawEvaluationTaskResult, decode_task_native
from .control_locking import execution_prefix_id
from .control_records import FiniteResponseLawRootControlLock, FiniteResponseLawRootParentJoin
from .evaluation_native_records import FiniteResponseLawEvaluationInterface
from .native_projection import _calibration_interface


def authenticate_parent_reservations(
    lock: FiniteResponseLawRootControlLock,
    store: DurablePreparedExecutionEventStore,
    *,
    before_parent: bool,
) -> None:
    """Require actual persisted generic locks, not a caller's lookalike records."""
    for design, parent in zip(lock.designs, lock.parents, strict=True):
        prefix_id = execution_prefix_id(design.root_id, design.policy_id)
        prefix = store.load_prepared(prefix_id)
        expected = (
            (
                PreparedExecutionEventKind.DESIGN_FROZEN,
                ObjectIdentity.from_record(design.binding_id, design),
                design,
            ),
            (
                PreparedExecutionEventKind.PARENT_RESERVED,
                ObjectIdentity.from_record(parent.commitment_id, parent),
                parent,
            ),
        )
        if (prefix.root_id, prefix.policy_id) != (design.root_id, design.policy_id) or (
            len(prefix.events) != 2 if before_parent else len(prefix.events) < 2
        ):
            raise ValueError("Finite response-law evaluation native execution lacks the exact pre-parent event boundary")
        for event, (kind, identity, record) in zip(prefix.events[:2], expected, strict=True):
            if (
                event.kind is not kind
                or event.subject != identity
                or store.read_prepared_record(
                    prefix_id=prefix_id, subject=identity, artifact=event.subject_artifact
                )
                != record.canonical_bytes()
            ):
                raise ValueError("Finite response-law evaluation native execution substitutes an immutable pre-parent lock")


def join_root_parent(
    *,
    lock: FiniteResponseLawRootControlLock,
    parent: FiniteResponseLawEvaluationTaskResult,
    parent_bytes: bytes,
    receipt: CanonicalTaskReceipt,
    artifact: ArtifactIdentity,
    store: DurablePreparedExecutionEventStore,
    now: Callable[[], str],
) -> FiniteResponseLawRootParentJoin:
    """The existing source already ran; authenticate its exact completed receipt."""
    if (
        receipt.task_id != parent.invocation.task_id
        or parent.invocation.phase != "parent"
        or parent.invocation.root != lock.forecast.root
        or parent.invocation.source != lock.forecast.prefix.invocation.source
        or parent.predecessors
        != (ObjectIdentity.from_record(lock.forecast.prefix.result_id, lock.forecast.prefix),)
        or receipt.operational_status is not OperationalStatus.SUCCEEDED
        or artifact.sha256 != parent.fingerprint()
        or artifact.payload_schema != parent.SCHEMA
    ):
        raise ValueError("Finite response-law evaluation parent join lacks its exact successful source task receipt")
    logical = tuple(
        a for a in receipt.output_logical_artifacts if a.logical_artifact_id == artifact.artifact_id
    )
    physical = tuple(
        a for a in receipt.output_materializations if a.logical_artifact_id == artifact.artifact_id
    )
    if (
        len(logical) != 1
        or len(physical) != 1
        or logical[0].content_sha256 != artifact.sha256
        or physical[0].physical_sha256 != artifact.sha256
        or physical[0].size_bytes != artifact.size_bytes
        or physical[0].compression != "none"
        or physical[0].partition_selector is not None
    ):
        raise ValueError("Finite response-law evaluation parent result is detached from source custody")
    authenticate_parent_reservations(lock, store, before_parent=True)
    pair = decode_task_native(parent, parent_bytes)
    interface = cast(
        FiniteResponseLawEvaluationInterface | None,
        _calibration_interface(
            None if pair is None or not parent.native_complete else pair[0], parent
        ),
    )
    work = (
        (None, None)
        if pair is None
        else tuple(v.delivery.parent_absolute_density_work for v in pair)
    )
    use_allowed = interface is not None and all(w is not None and w <= 32 for w in work)
    coordinator = ProspectiveEvaluationBindingCoordinator(prepared_store=store)
    returns, instances = [], []
    receipt_id = ObjectIdentity.from_record(receipt.receipt_id, receipt)
    handoff_id = (
        None
        if interface is None
        else ObjectIdentity.from_record(
            f"{lock.forecast.root.stage_unit}.actual-handoff", interface
        )
    )
    for design, early, compiled_artifact in zip(
        lock.designs, lock.parents, lock.compiled_artifacts, strict=True
    ):
        prefix_id = execution_prefix_id(design.root_id, design.policy_id)
        if interface is not None:
            assert handoff_id is not None
            store.publish_prepared_record(
                prefix_id=prefix_id, object_id=handoff_id.object_id, record=interface
            )
        returned = PreparedParentReturn(
            f"{prefix_id}.actual-parent",
            ObjectIdentity.from_record(design.binding_id, design),
            design.common_checkpoint,
            design.parent_commitment,
            receipt_id,
            artifact,
            handoff_id,
            PreparedParentDisposition.HANDOFF_AVAILABLE
            if interface is not None
            else PreparedParentDisposition.NO_HANDOFF,
            () if interface is not None else ("FLH_PARENT_HANDOFF_UNAVAILABLE",),
        )
        coordinator.record_prepared_parent_return(
            prefix_id=prefix_id, design=design, returned=returned, occurred_at_utc=now()
        )
        instance = None
        if interface is not None:
            raw = store.read_prepared_record(
                prefix_id=prefix_id,
                subject=early.decision.compiled_study,
                artifact=compiled_artifact,
            )
            compiled = decode_canonical_bytes(
                raw, CompiledLeastMagnitudeControllerStudy, maximum_bytes=16 * 1024**2
            )
            instance = coordinator.bind_prepared_forecast_instance(
                prefix_id=prefix_id,
                binding_id=f"{prefix_id}.actual-instance",
                design=design,
                compiled=compiled,
                returned=returned,
                forecast_parent=early,
                occurred_at_utc=now(),
            )
            coordinator.store_prepared_commitment(
                prefix_id=prefix_id,
                instance=instance,
                commitment=early.decision,
                occurred_at_utc=now(),
            )
            if use_allowed and early.decision.disposition is not CommitmentDisposition.NONATTEMPT:
                coordinator.reserve_prepared_task(
                    prefix_id=prefix_id, commitment=early.decision, occurred_at_utc=now()
                )
            slot = next(
                s
                for s in design.evaluation_plan.futures
                if s.role is PreparedFutureRole.MATCHED_HOLD
            )
            coordinator.reserve_prepared_matched_hold(
                prefix_id=prefix_id,
                design=design,
                returned=returned,
                slot=slot,
                occurred_at_utc=now(),
            )
        returns.append(returned)
        instances.append(instance)
    return FiniteResponseLawRootParentJoin(
        lock,
        parent,
        receipt_id,
        artifact,
        interface,
        (work[0], work[1]),
        tuple(returns),
        tuple(instances),
    )
