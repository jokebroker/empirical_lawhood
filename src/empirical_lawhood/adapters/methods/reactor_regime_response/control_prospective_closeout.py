"Sealed D native custody mapped to the installed prepared controller-use event owner."

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.adapters.simulators.reactor_regime_response.prospective_records import RegimeDNativeRoot
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.nested_controller_evaluation import (
    PreparedFutureRole,
)
from empirical_lawhood.runtime.artifacts import CanonicalTaskReceipt
from empirical_lawhood.runtime.controller_evaluation_nested import DurablePreparedExecutionEventStore, PreparedExecutionCensus, PreparedExecutionEventKind, PreparedFutureCompletion, PreparedFutureDisposition, ProspectiveEvaluationBindingCoordinator, SealedPreparedForecastPolicyBundle, SealedPreparedFutureLocator
from empirical_lawhood.runtime.controller_runtime import TickDisposition

from .control_prospective_lock import _existing_event, _same_event
from .control_prospective_plan import ReactorRegimeResponsePreparedProspectivePlanBundle
from .control_measure import measure_native_root
from .prospective_decision import CausalValidityRegimeAssignment
from .records import RegimeCausalPreparation, RegimePrivatePreparation


@dataclass(frozen=True, slots=True)
class ReactorRegimeResponseRootSealedProspective(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/reactor-regime-response-root-sealed-prospective'

    root: str
    native: ObjectIdentity
    plan: ObjectIdentity
    source_receipt: ObjectIdentity
    bundles: tuple[SealedPreparedForecastPolicyBundle, ...]

    def __post_init__(self) -> None:
        if (
            self.native.object_schema != RegimeDNativeRoot.SCHEMA
            or self.plan.object_schema != ReactorRegimeResponsePreparedProspectivePlanBundle.SCHEMA
            or self.source_receipt.object_schema != CanonicalTaskReceipt.SCHEMA
            or len(self.bundles) != 4
            or tuple(bundle.design.policy_id for bundle in self.bundles)
            != tuple(f"reactor-regime-request-{index}" for index in range(4))
            or any(bundle.design.root_id != self.root for bundle in self.bundles)
        ):
            raise ValueError("D sealed controller-use root changed its four installed consumers")


@dataclass(frozen=True, slots=True)
class RegimeDRootSealResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/regime-d-root-seal-result'

    root: str
    native: ObjectIdentity
    plan: ObjectIdentity
    sealed: ReactorRegimeResponseRootSealedProspective | None
    nonentry_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.native.object_schema != RegimeDNativeRoot.SCHEMA
            or self.plan.object_schema != ReactorRegimeResponsePreparedProspectivePlanBundle.SCHEMA
            or (self.sealed is None) != bool(self.nonentry_reasons)
            or (self.sealed is not None and (
                self.sealed.root != self.root
                or self.sealed.native != self.native
                or self.sealed.plan != self.plan
            ))
        ):
            raise ValueError("D root controller use seal lost its entry or nonentry reason")


def seal_native_root(
    *,
    native: RegimeDNativeRoot,
    plan_bundle: ReactorRegimeResponsePreparedProspectivePlanBundle,
    assignment: CausalValidityRegimeAssignment,
    causal: RegimeCausalPreparation,
    private: RegimePrivatePreparation,
    receipt: CanonicalTaskReceipt,
    artifact: ArtifactIdentity,
    store: DurablePreparedExecutionEventStore,
    occurred_at_utc: str,
) -> ReactorRegimeResponseRootSealedProspective | None:
    """Close an eligible root; ineligible roots retain their native nonentry."""
    root = native.root
    if (
        receipt.task_id != f"regime.action.{root}"
        or artifact.sha256 != native.fingerprint()
        or artifact.payload_schema != native.SCHEMA
        or not any(
            logical.logical_artifact_id == artifact.artifact_id
            and logical.content_sha256 == artifact.sha256
            for logical in receipt.output_logical_artifacts
        )
        or tuple(row.root for row in plan_bundle.assigned_roots)
        != tuple(sorted(row.root for row in plan_bundle.assigned_roots))
    ):
        raise ValueError("D controller use closure lacks its authenticated all-assigned native root")
    assigned = next(row for row in plan_bundle.assigned_roots if row.root == root)
    if not assigned.eligible:
        if native.locks or native.native_calls or native.controller_evaluation_plan is not None:
            raise ValueError("D nonentered root borrowed a prepared controller-use owner")
        return None
    plan = plan_bundle.plan
    if (
        plan is None
        or native.controller_evaluation_plan
        != ObjectIdentity.from_record(plan.evaluation_plan_id, plan)
    ):
        raise ValueError("D controller use closure changed its precontact all-root plan")
    measured = measure_native_root(
        native=native, assignment=assignment, causal=causal, private=private,
    )
    branch = {
        (row.role, row.index, row.view): row for row in native.branches
    }
    if len(branch) != 14:
        raise ValueError("D controller use closure lost a native request or chart slot")
    source_receipt = ObjectIdentity.from_record(receipt.receipt_id, receipt)
    coordinator = ProspectiveEvaluationBindingCoordinator(prepared_store=store)
    bundles = []
    for index, lock in enumerate(native.locks):
        prefix_id = lock.prefix_id
        slots = tuple(
            slot for slot in plan.futures
            if slot.root_id == root and slot.policy_id == lock.design.policy_id
        )
        if len(slots) != 4:
            raise ValueError("D controller use closure changed a request's task/reference/audit census")
        locators = []
        completions = []
        events = store.load_prepared(prefix_id).events
        for slot in slots:
            reserved_kind = {
                PreparedFutureRole.COMMITTED_TASK: PreparedExecutionEventKind.TASK_RESERVED,
                PreparedFutureRole.MATCHED_HOLD: PreparedExecutionEventKind.MATCHED_HOLD_RESERVED,
                PreparedFutureRole.AUDIT_PROBE: PreparedExecutionEventKind.PROBE_RESERVED,
            }[slot.role]
            reserved = any(
                event.kind is reserved_kind and event.slot_id == slot.slot_id
                for event in events
            )
            if slot.role is PreparedFutureRole.COMMITTED_TASK:
                key = ("PRIMARY", index)
            elif slot.role is PreparedFutureRole.MATCHED_HOLD:
                key = ("EVALUATOR", 0)
            else:
                key = ("EVALUATOR", slot.ordinal + 1)
            local = []
            for view, view_id in enumerate(plan.numerical_view_ids):
                observed = branch[key[0], key[1], view]
                tick = native.ticks[index]
                delivered = (
                    tick is not None
                    and tick.disposition is TickDisposition.ACTION_DELIVERED
                ) if slot.role is PreparedFutureRole.COMMITTED_TASK else True
                measured_available = (
                    measured.owner[index] is not None
                    if slot.role is PreparedFutureRole.COMMITTED_TASK else
                    measured.chart[view] is not None
                    if slot.role is PreparedFutureRole.MATCHED_HOLD else
                    measured.chart[2 * (slot.ordinal + 1) + view] is not None
                )
                complete = (
                    reserved and observed.status != "NONATTEMPT"
                    and delivered and measured_available
                )
                disposition = (
                    PreparedFutureDisposition.COMPLETED if complete else
                    PreparedFutureDisposition.NONATTEMPT
                    if not reserved or observed.status == "NONATTEMPT" or not delivered
                    else PreparedFutureDisposition.INVALID_OBSERVATION
                )
                local.append(SealedPreparedFutureLocator(
                    f"{slot.slot_id}.{view_id}.locator",
                    slot.slot_id, view_id, disposition,
                    source_receipt if reserved else None,
                    artifact if reserved else None,
                    () if complete else (f"REACTOR_D_{disposition.value}",),
                    OutcomeAccess.EVALUATION_SEALED,
                ))
            local = sorted(local, key=lambda row: row.locator_id)
            locators.extend(local)
            if reserved:
                completion = PreparedFutureCompletion(
                    f"{slot.slot_id}.completion",
                    ObjectIdentity.from_record(slot.slot_id, slot),
                    tuple(local),
                )
                kind = {
                    PreparedFutureRole.COMMITTED_TASK: PreparedExecutionEventKind.TASK_COMPLETED,
                    PreparedFutureRole.MATCHED_HOLD: PreparedExecutionEventKind.MATCHED_HOLD_COMPLETED,
                    PreparedFutureRole.AUDIT_PROBE: PreparedExecutionEventKind.PROBE_COMPLETED,
                }[slot.role]
                if not _same_event(
                    _existing_event(store, prefix_id, kind, slot_id=slot.slot_id),
                    ObjectIdentity.from_record(completion.completion_id, completion),
                ):
                    coordinator.record_prepared_completion(
                        prefix_id=prefix_id, design=lock.design,
                        completion=completion, occurred_at_utc=occurred_at_utc,
                    )
                completions.append(completion)
        census = PreparedExecutionCensus(
            f"{prefix_id}.census",
            ObjectIdentity.from_record(lock.design.binding_id, lock.design),
            tuple(sorted(locators, key=lambda row: row.locator_id)),
        )
        if not _same_event(
            _existing_event(store, prefix_id, PreparedExecutionEventKind.TERMINAL),
            ObjectIdentity.from_record(census.census_id, census),
        ):
            terminal = coordinator.close_prepared_execution(
                prefix_id=prefix_id, design=lock.design,
                census=census, occurred_at_utc=occurred_at_utc,
            )
        else:
            terminal = store.load_prepared(prefix_id)
        bundles.append(SealedPreparedForecastPolicyBundle(
            f"{prefix_id}.sealed", lock.design, lock.returned, lock.instance,
            lock.parent.task, lock.parent.decision, lock.probes,
            terminal, census,
            tuple(sorted(completions, key=lambda value: value.completion_id)),
            lock.parent,
        ))
    return ReactorRegimeResponseRootSealedProspective(
        root,
        ObjectIdentity.from_record(f"{root}.native-action", native),
        ObjectIdentity.from_record("regime.d-plan", plan_bundle),
        source_receipt,
        tuple(bundles),
    )
