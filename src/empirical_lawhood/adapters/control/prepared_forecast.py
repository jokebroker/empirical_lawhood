"Recoverable forecast-parent assembly through the prepared controller-use event owner.\n\nNo native acquisition, law qualification or outcome judgement occurs here.\n"

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.controller_study import ImplementationRole
from empirical_lawhood.planning.nested_controller_evaluation import PreparedFutureRole, CommonStartControllerEvaluationPlan, PreparedInterfaceEvaluationSlice, prepared_interface_evaluation_slice
from empirical_lawhood.runtime.controller_evaluation_nested import DurablePreparedExecutionEventStore, PreparedDesignBindingReceipt, PreparedExecutionEvent, PreparedExecutionEventPrefix, PreparedExecutionEventKind, PreparedForecastParentCommitment, PreparedInstanceBindingReceipt, PreparedParentDisposition, PreparedParentReturn, PreparedProbeCommitment, ProspectiveEvaluationBindingCoordinator
from empirical_lawhood.runtime.controller_runtime import CommitmentDisposition
from empirical_lawhood.runtime.controller_evaluation_nested import SealedPreparedFutureLocator, SealedPreparedForecastPolicyBundle, PreparedFutureCompletion, PreparedExecutionCensus
from empirical_lawhood.runtime.artifacts import CanonicalTaskReceipt

from empirical_lawhood.runtime.controller_compiler import CompiledDeliveryControllerStudy
from empirical_lawhood.runtime.controller_runtime import DeliveryControllerDecisionCommitment, DeliveryControllerTickReceipt, TickDisposition
from empirical_lawhood.runtime.controller_evaluation_trajectory import TrajectoryCallbackLink
from empirical_lawhood.planning.finite_response_geometry import FiniteTaskFunctionalSpec, FiniteResponseSet


def prepared_event(
    prefix: PreparedExecutionEventPrefix,
    kind: PreparedExecutionEventKind,
    *,
    slot_id: str | None = None,
) -> PreparedExecutionEvent | None:
    """Inspect an authenticated snapshot already returned by the durable owner."""
    rows = tuple(
        event
        for event in prefix.events
        if event.kind is kind and (slot_id is None or event.slot_id == slot_id)
    )
    if len(rows) > 1:
        raise ValueError("prepared controller-use recovery found duplicate prepared-owner events")
    return rows[0] if rows else None


def same_prepared_event(event: PreparedExecutionEvent | None, subject: ObjectIdentity) -> bool:
    if event is None:
        return False
    if event.subject != subject:
        raise ValueError("prepared controller-use recovery changed a sealed prepared-owner subject")
    return True


@dataclass(frozen=True, slots=True)
class PreparedForecastLock(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/control/prepared-forecast-lock'
    prefix_id: str
    design: PreparedDesignBindingReceipt
    parent: PreparedForecastParentCommitment
    returned: PreparedParentReturn
    instance: PreparedInstanceBindingReceipt
    probes: tuple[PreparedProbeCommitment, ...]

    def __post_init__(self) -> None:
        if (
            self.design.parent_commitment
            != ObjectIdentity.from_record(self.parent.commitment_id, self.parent)
            or self.returned.parent_commitment != self.design.parent_commitment
            or self.instance.design_binding
            != ObjectIdentity.from_record(self.design.binding_id, self.design)
            or any(p.slot.role is not PreparedFutureRole.AUDIT_PROBE for p in self.probes)
        ):
            raise ValueError("forecast lock changed its prepared parent, instance or probe roles")

    def evaluation_for(
        self, compiled: CompiledDeliveryControllerStudy
    ) -> PreparedInterfaceEvaluationSlice:
        """Recover the frozen root/policy view without reopening the whole census.

        Callers must obtain this lock from authenticated custody. Initial freeze
        still compares the slice with the full plan through the coordinator.
        """
        self.parent.validate_binding(self.design, compiled)
        if self.instance.compiled_study != ObjectIdentity.from_record(
            compiled.compiled_study_id, compiled
        ) or self.instance.parent_return != ObjectIdentity.from_record(
            self.returned.return_id, self.returned
        ):
            raise ValueError("prepared lock substituted its compiled instance or parent return")
        return self.design.evaluation_plan


def freeze_forecast_request(
    *,
    prefix_id: str,
    policy_id: str,
    root: str,
    compiled: CompiledDeliveryControllerStudy,
    commitment: DeliveryControllerDecisionCommitment,
    task: FiniteTaskFunctionalSpec,
    audit_forecasts: tuple[FiniteResponseSet, ...],
    checkpoint: ObjectIdentity,
    causal_receipt: CanonicalTaskReceipt | DeliveryControllerTickReceipt,
    causal_artifact: ArtifactIdentity,
    issued_study: ObjectIdentity,
    evaluation: CommonStartControllerEvaluationPlan,
    store: DurablePreparedExecutionEventStore,
    occurred_at_utc: str,
    causal_link: TrajectoryCallbackLink | None = None,
) -> PreparedForecastLock:
    """Persist the exact forecast, receipted parent and assigned future reservations."""
    if (
        causal_artifact.sha256 != checkpoint.object_fingerprint
        or causal_artifact.payload_schema != checkpoint.object_schema
    ):
        raise ValueError("forecast lock lacks its authenticated parent artifact")
    if isinstance(causal_receipt, CanonicalTaskReceipt):
        if causal_link is not None or not any(
            a.logical_artifact_id == causal_artifact.artifact_id
            and a.content_sha256 == causal_artifact.sha256
            and a.payload_schema == causal_artifact.payload_schema
            for a in causal_receipt.output_logical_artifacts
        ):
            raise ValueError("forecast lock lacks its authenticated parent receipt")
        source_receipt = ObjectIdentity.from_record(causal_receipt.receipt_id, causal_receipt)
    else:
        source_receipt = ObjectIdentity.from_record(causal_receipt.tick_id, causal_receipt)
        word = causal_receipt.delivery_trace.expected_action_word
        actual = commitment.observation
        if (
            causal_link is None
            or causal_link.previous_tick != source_receipt
            or causal_link.root != root
            or causal_link.causal_input != causal_artifact
            or causal_receipt.disposition is not TickDisposition.ACTION_DELIVERED
            or not causal_receipt.delivery_trace.exact
            or word is None
            or causal_receipt.commitment.observation.independent_unit_id != root
            or actual.independent_unit_id != root
            or actual.coordinate.clock_id != word.ordering_clock_id
            or actual.coordinate.time_unit != word.ordering_time_unit
            or actual.coordinate.coordinate_frame != word.ordering_coordinate_frame
            or actual.coordinate.origin is not word.ordering_origin
            or actual.coordinate.coordinate
            != max(o.realized.coordinate.coordinate + o.duration for o in word.occurrences)
            or causal_artifact not in actual.input_artifacts
            or not any(
                a.artifact_id == causal_link.link_id
                and a.sha256 == causal_link.fingerprint()
                and a.payload_schema == causal_link.SCHEMA
                for a in actual.input_artifacts
            )
        ):
            raise ValueError(
                "forecast callback parent is not the actual completed predecessor and next native cutoff"
            )
    plan_identity = ObjectIdentity.from_record(evaluation.evaluation_plan_id, evaluation)
    parent = PreparedForecastParentCommitment(
        f"{prefix_id}.parent-lock",
        plan_identity,
        root,
        policy_id,
        checkpoint,
        checkpoint,
        commitment,
        task,
        occurred_at_utc,
    )
    evaluator = compiled.implementation(ImplementationRole.OUTCOME_EVALUATOR)
    design = PreparedDesignBindingReceipt(
        f"{prefix_id}.design",
        issued_study,
        prepared_interface_evaluation_slice(evaluation, root_id=root, policy_id=policy_id),
        root,
        policy_id,
        checkpoint,
        ObjectIdentity.from_record(parent.commitment_id, parent),
        evaluator,
        occurred_at_utc,
    )
    coordinator = ProspectiveEvaluationBindingCoordinator(prepared_store=store)
    prefix = coordinator.freeze_prepared_forecast_design(
        prefix_id=prefix_id,
        binding=design,
        plan=evaluation,
        forecast_parent=parent,
        compiled=compiled,
    )
    parent_artifact = store.publish_prepared_record(
        prefix_id=prefix_id, object_id=parent.commitment_id, record=parent
    )
    if not same_prepared_event(
        prepared_event(prefix, PreparedExecutionEventKind.PARENT_RESERVED),
        design.parent_commitment,
    ):
        prefix = coordinator.record_prepared_boundary(
            prefix_id=prefix_id,
            design=design,
            kind=PreparedExecutionEventKind.PARENT_RESERVED,
            subject=design.parent_commitment,
            artifact=parent_artifact,
            occurred_at_utc=occurred_at_utc,
        )
    returned = PreparedParentReturn(
        f"{prefix_id}.actual-parent",
        ObjectIdentity.from_record(design.binding_id, design),
        checkpoint,
        design.parent_commitment,
        source_receipt,
        causal_artifact,
        compiled.study.instance_binding.public_handoff,
        PreparedParentDisposition.HANDOFF_AVAILABLE,
        (),
    )
    returned_id = ObjectIdentity.from_record(returned.return_id, returned)
    if not same_prepared_event(
        prepared_event(prefix, PreparedExecutionEventKind.PARENT_RETURNED),
        returned_id,
    ):
        prefix = coordinator.record_prepared_parent_return(
            prefix_id=prefix_id,
            design=design,
            returned=returned,
            occurred_at_utc=occurred_at_utc,
        )
    bound = prepared_event(prefix, PreparedExecutionEventKind.INSTANCE_BOUND)
    if bound is None:
        instance = coordinator.bind_prepared_forecast_instance(
            prefix_id=prefix_id,
            binding_id=f"{prefix_id}.actual-instance",
            design=design,
            compiled=compiled,
            returned=returned,
            forecast_parent=parent,
            occurred_at_utc=occurred_at_utc,
        )
    else:
        raw = store.read_prepared_record(
            prefix_id=prefix_id, subject=bound.subject, artifact=bound.subject_artifact
        )
        instance = decode_canonical_bytes(
            raw, PreparedInstanceBindingReceipt, maximum_bytes=16 * 1024**2
        )
        if (
            instance.binding_id != f"{prefix_id}.actual-instance"
            or instance.design_binding != ObjectIdentity.from_record(design.binding_id, design)
            or instance.parent_return != returned_id
            or instance.compiled_study
            != ObjectIdentity.from_record(compiled.compiled_study_id, compiled)
        ):
            raise ValueError("prepared controller-use recovery changed the frozen controller instance")
    prefix = coordinator.store_prepared_commitment(
        prefix_id=prefix_id,
        instance=instance,
        commitment=commitment,
        occurred_at_utc=occurred_at_utc,
    )
    commitment_id = ObjectIdentity.from_record(commitment.commitment_id, commitment)
    if commitment.disposition is not CommitmentDisposition.NONATTEMPT and not same_prepared_event(
        prepared_event(prefix, PreparedExecutionEventKind.TASK_RESERVED),
        commitment_id,
    ):
        prefix = coordinator.reserve_prepared_task(
            prefix_id=prefix_id,
            commitment=commitment,
            occurred_at_utc=occurred_at_utc,
        )
    hold = next(
        slot
        for slot in evaluation.futures
        if slot.root_id == root
        and slot.policy_id == policy_id
        and slot.role is PreparedFutureRole.MATCHED_HOLD
    )
    if not same_prepared_event(
        prepared_event(
            prefix,
            PreparedExecutionEventKind.MATCHED_HOLD_RESERVED,
            slot_id=hold.slot_id,
        ),
        ObjectIdentity.from_record(hold.slot_id, hold),
    ):
        prefix = coordinator.reserve_prepared_matched_hold(
            prefix_id=prefix_id,
            design=design,
            returned=returned,
            slot=hold,
            occurred_at_utc=occurred_at_utc,
        )
    audit_slots = tuple(
        sorted(
            (
                slot
                for slot in evaluation.futures
                if slot.root_id == root
                and slot.policy_id == policy_id
                and slot.role is PreparedFutureRole.AUDIT_PROBE
            ),
            key=lambda slot: slot.ordinal,
        )
    )
    probes = tuple(
        PreparedProbeCommitment(
            f"{slot.slot_id}.probe",
            slot,
            ObjectIdentity.from_record(design.binding_id, design),
            ObjectIdentity.from_record(returned.return_id, returned),
            forecast,
        )
        for slot, forecast in zip(audit_slots, audit_forecasts, strict=True)
    )
    for probe in probes:
        if not same_prepared_event(
            prepared_event(
                prefix,
                PreparedExecutionEventKind.PROBE_RESERVED,
                slot_id=probe.slot.slot_id,
            ),
            ObjectIdentity.from_record(probe.probe_id, probe),
        ):
            prefix = coordinator.reserve_prepared_probe(
                prefix_id=prefix_id,
                design=design,
                returned=returned,
                probe=probe,
                occurred_at_utc=occurred_at_utc,
            )
    return PreparedForecastLock(
        prefix_id,
        design,
        parent,
        returned,
        instance,
        probes,
    )


def seal_forecast_request(
    *,
    lock: PreparedForecastLock,
    locators: tuple[SealedPreparedFutureLocator, ...],
    store: DurablePreparedExecutionEventStore,
    occurred_at_utc: str,
) -> SealedPreparedForecastPolicyBundle:
    """Persist all assigned future completions and let the sole owner check the census."""
    prefix_id = lock.prefix_id
    coordinator = ProspectiveEvaluationBindingCoordinator(prepared_store=store)
    prefix = store.load_prepared(prefix_id)
    completions = []
    for slot in lock.design.evaluation_plan.futures:
        reservation, completion_kind = {
            PreparedFutureRole.COMMITTED_TASK: (
                PreparedExecutionEventKind.TASK_RESERVED,
                PreparedExecutionEventKind.TASK_COMPLETED,
            ),
            PreparedFutureRole.MATCHED_HOLD: (
                PreparedExecutionEventKind.MATCHED_HOLD_RESERVED,
                PreparedExecutionEventKind.MATCHED_HOLD_COMPLETED,
            ),
            PreparedFutureRole.AUDIT_PROBE: (
                PreparedExecutionEventKind.PROBE_RESERVED,
                PreparedExecutionEventKind.PROBE_COMPLETED,
            ),
        }[slot.role]
        if prepared_event(prefix, reservation, slot_id=slot.slot_id) is None:
            continue
        completion = PreparedFutureCompletion(
            f"{slot.slot_id}.completion",
            ObjectIdentity.from_record(slot.slot_id, slot),
            tuple(
                sorted(
                    (v for v in locators if v.slot_id == slot.slot_id), key=lambda v: v.locator_id
                )
            ),
        )
        if not same_prepared_event(
            prepared_event(prefix, completion_kind, slot_id=slot.slot_id),
            ObjectIdentity.from_record(completion.completion_id, completion),
        ):
            prefix = coordinator.record_prepared_completion(
                prefix_id=prefix_id,
                design=lock.design,
                completion=completion,
                occurred_at_utc=occurred_at_utc,
            )
        completions.append(completion)
    census = PreparedExecutionCensus(
        f"{prefix_id}.census",
        ObjectIdentity.from_record(lock.design.binding_id, lock.design),
        tuple(sorted(locators, key=lambda v: v.locator_id)),
    )
    if not same_prepared_event(
        prepared_event(prefix, PreparedExecutionEventKind.TERMINAL),
        ObjectIdentity.from_record(census.census_id, census),
    ):
        prefix = coordinator.close_prepared_execution(
            prefix_id=prefix_id, design=lock.design, census=census, occurred_at_utc=occurred_at_utc
        )
    return SealedPreparedForecastPolicyBundle(
        f"{prefix_id}.sealed",
        lock.design,
        lock.returned,
        lock.instance,
        lock.parent.task,
        lock.parent.decision,
        lock.probes,
        prefix,
        census,
        tuple(sorted(completions, key=lambda v: v.completion_id)),
        lock.parent,
    )
