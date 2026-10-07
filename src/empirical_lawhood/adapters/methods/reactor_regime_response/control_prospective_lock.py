"Freeze four installed prepared-policy controller use consumers before D native actions.\n\nThe source adapter supplies the already receipted parent preparation and the\nactual compiled admission commitments. This owner persists the generic prepared-event\nchain; a local selector or a later chart cannot stand in for it.\n"

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.controller_study import ImplementationRole
from empirical_lawhood.planning.nested_controller_evaluation import PreparedFutureRole, CoupledRealizationControllerEvaluationPlan, prepared_interface_evaluation_slice
from empirical_lawhood.runtime.controller_evaluation_nested import DurablePreparedExecutionEventStore, PreparedDesignBindingReceipt, PreparedExecutionEvent, PreparedExecutionEventKind, PreparedForecastParentCommitment, PreparedInstanceBindingReceipt, PreparedParentDisposition, PreparedParentReturn, PreparedProbeCommitment, ProspectiveEvaluationBindingCoordinator
from empirical_lawhood.runtime.controller_runtime import CommitmentDisposition
from empirical_lawhood.runtime.artifacts import CanonicalTaskReceipt

from .control_owner import PreparedReactorRequest
from .records import RegimeCausalPreparation


def _existing_event(
    store: DurablePreparedExecutionEventStore,
    prefix_id: str,
    kind: PreparedExecutionEventKind,
    *,
    slot_id: str | None = None,
) -> PreparedExecutionEvent | None:
    try:
        prefix = store.load_prepared(prefix_id)
    except KeyError:
        return None
    rows = tuple(
        event for event in prefix.events
        if event.kind is kind and (slot_id is None or event.slot_id == slot_id)
    )
    if len(rows) > 1:
        raise ValueError("D controller-use recovery found duplicate prepared-owner events")
    return rows[0] if rows else None


def _same_event(
    event: PreparedExecutionEvent | None, subject: ObjectIdentity
) -> bool:
    if event is None:
        return False
    if event.subject != subject:
        raise ValueError("D controller-use recovery changed a sealed prepared-owner subject")
    return True


@dataclass(frozen=True, slots=True)
class RegimeDPreparedRequestLock(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/regime-d-prepared-request-lock'

    request_index: int
    prefix_id: str
    design: PreparedDesignBindingReceipt
    parent: PreparedForecastParentCommitment
    returned: PreparedParentReturn
    instance: PreparedInstanceBindingReceipt
    probes: tuple[PreparedProbeCommitment, PreparedProbeCommitment]

    def __post_init__(self) -> None:
        if (
            self.request_index not in range(4)
            or self.prefix_id != (
                f"{self.design.root_id}.request-{self.request_index}.prepared"
            )
            or self.design.policy_id != f"reactor-regime-request-{self.request_index}"
            or self.design.parent_commitment
            != ObjectIdentity.from_record(self.parent.commitment_id, self.parent)
            or self.returned.parent_commitment != self.design.parent_commitment
            or self.instance.design_binding
            != ObjectIdentity.from_record(self.design.binding_id, self.design)
            or tuple(probe.slot.ordinal for probe in self.probes) != (0, 1)
            or any(probe.slot.role is not PreparedFutureRole.AUDIT_PROBE for probe in self.probes)
        ):
            raise ValueError("D controller use lock changed its frozen request and two audit words")


def freeze_prepared_request(
    *,
    root: str,
    request_index: int,
    prepared: PreparedReactorRequest,
    causal: RegimeCausalPreparation,
    causal_receipt: CanonicalTaskReceipt,
    causal_artifact: ArtifactIdentity,
    issued_study: ObjectIdentity,
    evaluation: CoupledRealizationControllerEvaluationPlan,
    store: DurablePreparedExecutionEventStore,
    occurred_at_utc: str,
) -> RegimeDPreparedRequestLock:
    """Durably bind the early commitment, parent handoff and future slots."""
    if (
        request_index not in range(4)
        or causal.root != root
        or causal_artifact.sha256 != causal.fingerprint()
        or causal_artifact.payload_schema != causal.SCHEMA
        or causal_receipt.task_id != f"regime.prepare.{root}"
        or not any(
            logical.content_sha256 == causal.fingerprint()
            and logical.payload_schema == causal.SCHEMA
            and logical.logical_artifact_id == causal_artifact.artifact_id
            for logical in causal_receipt.output_logical_artifacts
        )
        or root not in {row.root_id for row in evaluation.roots}
    ):
        raise ValueError("D controller use lock lacks its exact receipted prepared parent")
    policy_id = f"reactor-regime-request-{request_index}"
    prefix_id = f"{root}.request-{request_index}.prepared"
    checkpoint = ObjectIdentity.from_record(f"{root}.causal-preparation", causal)
    plan_identity = ObjectIdentity.from_record(evaluation.evaluation_plan_id, evaluation)
    parent = PreparedForecastParentCommitment(
        f"{prefix_id}.parent-lock", plan_identity, root, policy_id,
        checkpoint, checkpoint, prepared.commitment, prepared.task,
        occurred_at_utc,
    )
    evaluator = prepared.compiled.implementation(ImplementationRole.OUTCOME_EVALUATOR)
    design = PreparedDesignBindingReceipt(
        f"{prefix_id}.design",
        issued_study,
        prepared_interface_evaluation_slice(
            evaluation, root_id=root, policy_id=policy_id
        ),
        root, policy_id, checkpoint,
        ObjectIdentity.from_record(parent.commitment_id, parent),
        evaluator, occurred_at_utc,
    )
    coordinator = ProspectiveEvaluationBindingCoordinator(prepared_store=store)
    coordinator.freeze_prepared_forecast_design(
        prefix_id=prefix_id, binding=design, plan=evaluation,
        forecast_parent=parent, compiled=prepared.compiled,
    )
    parent_artifact = store.publish_prepared_record(
        prefix_id=prefix_id, object_id=parent.commitment_id, record=parent
    )
    if not _same_event(
        _existing_event(store, prefix_id, PreparedExecutionEventKind.PARENT_RESERVED),
        design.parent_commitment,
    ):
        coordinator.record_prepared_boundary(
            prefix_id=prefix_id, design=design,
            kind=PreparedExecutionEventKind.PARENT_RESERVED,
            subject=design.parent_commitment, artifact=parent_artifact,
            occurred_at_utc=occurred_at_utc,
        )
    returned = PreparedParentReturn(
        f"{prefix_id}.actual-parent",
        ObjectIdentity.from_record(design.binding_id, design),
        checkpoint, design.parent_commitment,
        ObjectIdentity.from_record(causal_receipt.receipt_id, causal_receipt),
        causal_artifact,
        prepared.compiled.study.instance_binding.public_handoff,
        PreparedParentDisposition.HANDOFF_AVAILABLE,
        (),
    )
    returned_id = ObjectIdentity.from_record(returned.return_id, returned)
    if not _same_event(
        _existing_event(store, prefix_id, PreparedExecutionEventKind.PARENT_RETURNED),
        returned_id,
    ):
        coordinator.record_prepared_parent_return(
            prefix_id=prefix_id, design=design, returned=returned,
            occurred_at_utc=occurred_at_utc,
        )
    bound = _existing_event(store, prefix_id, PreparedExecutionEventKind.INSTANCE_BOUND)
    if bound is None:
        instance = coordinator.bind_prepared_forecast_instance(
            prefix_id=prefix_id, binding_id=f"{prefix_id}.actual-instance",
            design=design, compiled=prepared.compiled, returned=returned,
            forecast_parent=parent, occurred_at_utc=occurred_at_utc,
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
            != ObjectIdentity.from_record(
                prepared.compiled.compiled_study_id, prepared.compiled
            )
        ):
            raise ValueError("D controller-use recovery changed the frozen controller instance")
    coordinator.store_prepared_commitment(
        prefix_id=prefix_id, instance=instance,
        commitment=prepared.commitment, occurred_at_utc=occurred_at_utc,
    )
    commitment_id = ObjectIdentity.from_record(
        prepared.commitment.commitment_id, prepared.commitment
    )
    if (
        prepared.commitment.disposition is not CommitmentDisposition.NONATTEMPT
        and not _same_event(
            _existing_event(store, prefix_id, PreparedExecutionEventKind.TASK_RESERVED),
            commitment_id,
        )
    ):
        coordinator.reserve_prepared_task(
            prefix_id=prefix_id, commitment=prepared.commitment,
            occurred_at_utc=occurred_at_utc,
        )
    hold = next(
        slot for slot in evaluation.futures
        if slot.root_id == root and slot.policy_id == policy_id
        and slot.role is PreparedFutureRole.MATCHED_HOLD
    )
    if not _same_event(
        _existing_event(
            store, prefix_id, PreparedExecutionEventKind.MATCHED_HOLD_RESERVED,
            slot_id=hold.slot_id,
        ),
        ObjectIdentity.from_record(hold.slot_id, hold),
    ):
        coordinator.reserve_prepared_matched_hold(
            prefix_id=prefix_id, design=design, returned=returned,
            slot=hold, occurred_at_utc=occurred_at_utc,
        )
    audit_slots = tuple(sorted((
        slot for slot in evaluation.futures
        if slot.root_id == root and slot.policy_id == policy_id
        and slot.role is PreparedFutureRole.AUDIT_PROBE
    ), key=lambda slot: slot.ordinal))
    if len(audit_slots) != 2:
        raise ValueError("D controller use lock lacks both positive audit words")
    probes = tuple(
        PreparedProbeCommitment(
            f"{slot.slot_id}.probe",
            slot,
            ObjectIdentity.from_record(design.binding_id, design),
            ObjectIdentity.from_record(returned.return_id, returned),
            forecast,
        )
        for slot, forecast in zip(audit_slots, prepared.audit_forecasts, strict=True)
    )
    for probe in probes:
        if not _same_event(
            _existing_event(
                store, prefix_id, PreparedExecutionEventKind.PROBE_RESERVED,
                slot_id=probe.slot.slot_id,
            ),
            ObjectIdentity.from_record(probe.probe_id, probe),
        ):
            coordinator.reserve_prepared_probe(
                prefix_id=prefix_id, design=design, returned=returned,
                probe=probe, occurred_at_utc=occurred_at_utc,
            )
    return RegimeDPreparedRequestLock(
        request_index, prefix_id, design, parent, returned, instance,
        probes,  # type: ignore[arg-type]
    )
