"Close immutable finite response-law evaluation consumer censuses over already acquired native views.\n\nThis is a sealed custody operation, not reveal, selection or adjudication.\nUnreserved/cancelled consumers cannot borrow successful shadow outcomes.\n"

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.controller_compiler import CompiledLeastMagnitudeControllerStudy
from empirical_lawhood.runtime.controller_evaluation_nested import DurablePreparedExecutionEventStore, ProspectiveEvaluationBindingCoordinator, PreparedExecutionEventKind, PreparedFutureDisposition, PreparedFutureCompletion, PreparedExecutionCensus, SealedPreparedFutureLocator, SealedPreparedForecastPolicyBundle
from empirical_lawhood.planning.nested_controller_evaluation import PreparedFutureRole
from .control_records import FiniteResponseLawRootParentJoin
from .control_locking import execution_prefix_id
from .control_delivery import _force
from .evaluation_native_records import FiniteResponseLawEvaluationViewObservation
from .assigned_native_records import FiniteResponseLawAssignedEvaluationViewObservation
from empirical_lawhood.adapters.simulators.finite_response_law.assigned_contracts import FiniteResponseLawAssignedEvaluationRoot
from .paired_assay import paired_native_assay


@dataclass(frozen=True, slots=True)
class FiniteResponseLawRootSealedControl(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-response-law/finite-response-law-root-sealed-control'
    join: FiniteResponseLawRootParentJoin
    views: tuple[ObjectIdentity, ...]
    bundles: tuple[SealedPreparedForecastPolicyBundle, ...]

    def __post_init__(self) -> None:
        lock = self.join.lock
        view_type = (
            FiniteResponseLawAssignedEvaluationViewObservation
            if type(lock.forecast.root) is FiniteResponseLawAssignedEvaluationRoot
            else FiniteResponseLawEvaluationViewObservation
        )
        if (
            len(self.views) != 2
            or len(set(self.views)) != 2
            or any(v.object_schema != view_type.SCHEMA for v in self.views)
            or tuple(b.design for b in self.bundles) != lock.designs
            or tuple(b.forecast_parent for b in self.bundles) != lock.parents
            or tuple(b.parent_return for b in self.bundles) != self.join.returns
            or tuple(b.instance for b in self.bundles) != self.join.instances
        ):
            raise ValueError("Finite response-law evaluation sealed census substitutes its actual parent or frozen consumers")

    @property
    def result_id(self) -> str:
        return f"{self.join.lock.forecast.root.stage_unit}.sealed-control"


def read_compiled(
    join: FiniteResponseLawRootParentJoin, index: int, store: DurablePreparedExecutionEventStore
) -> CompiledLeastMagnitudeControllerStudy:
    design, early = join.lock.designs[index], join.lock.parents[index]
    return decode_canonical_bytes(
        store.read_prepared_record(
            prefix_id=execution_prefix_id(design.root_id, design.policy_id),
            subject=early.decision.compiled_study,
            artifact=join.lock.compiled_artifacts[index],
        ),
        CompiledLeastMagnitudeControllerStudy,
        maximum_bytes=16 * 1024**2,
    )


def slot_available(
    join: FiniteResponseLawRootParentJoin,
    index: int,
    view: FiniteResponseLawEvaluationViewObservation,
    views: tuple[FiniteResponseLawEvaluationViewObservation, ...],
    role: PreparedFutureRole,
) -> bool:
    "Only genuine complete observations can enter a completed controller-use locator."
    if role is PreparedFutureRole.MATCHED_HOLD:
        words = tuple(
            w for w in view.words if w.invocation.word is not None and w.invocation.word.sign == 0
        )
        return len(words) == 2 and all(
            w.complete and w.maximum_force_error == 0 and all(v is not None for v in w.outputs)
            for w in words
        )
    action = join.lock.parents[index].decision.action_binding
    if action is None:
        return False
    assay = paired_native_assay(
        views, parent=join.lock.forecast.root.assigned_parent, word=_force(action.action_word)
    )
    return all(r.source_valid for r in assay.readouts if r.refinement == view.refinement)


def seal_root(
    *,
    join: FiniteResponseLawRootParentJoin,
    views: tuple[FiniteResponseLawEvaluationViewObservation, ...],
    artifacts: tuple[ArtifactIdentity, ...],
    receipts: tuple[ObjectIdentity, ...],
    store: DurablePreparedExecutionEventStore,
    now: Callable[[], str],
) -> FiniteResponseLawRootSealedControl:
    root = join.lock.forecast.root
    if (
        tuple(v.refinement for v in views) != (1, 2)
        or any(v.root != root for v in views)
        or len(artifacts) != 2
        or len(receipts) != 2
        or any(
            a.sha256 != v.fingerprint() or a.payload_schema != v.SCHEMA
            for a, v in zip(artifacts, views, strict=True)
        )
    ):
        raise ValueError("Finite response-law evaluation sealed closure requires both authenticated views of its exact root")
    coordinator = ProspectiveEvaluationBindingCoordinator(prepared_store=store)
    bundles = []
    for i, (design, early, returned, instance) in enumerate(
        zip(join.lock.designs, join.lock.parents, join.returns, join.instances, strict=True)
    ):
        prefix_id = execution_prefix_id(design.root_id, design.policy_id)
        prefix = store.load_prepared(prefix_id)
        locators, completions = [], []
        for slot in design.evaluation_plan.futures:
            if slot.root_id != design.root_id or slot.policy_id != design.policy_id:
                continue
            reservation = (
                PreparedExecutionEventKind.MATCHED_HOLD_RESERVED
                if slot.role is PreparedFutureRole.MATCHED_HOLD
                else PreparedExecutionEventKind.TASK_RESERVED
            )
            reserved = any(
                e.kind is reservation and e.slot_id == slot.slot_id for e in prefix.events
            )
            local = []
            for view, artifact, receipt, view_id in zip(
                views, artifacts, receipts, design.evaluation_plan.numerical_view_ids, strict=True
            ):
                available = reserved and slot_available(join, i, view, views, slot.role)
                disposition = (
                    PreparedFutureDisposition.COMPLETED
                    if available
                    else PreparedFutureDisposition.INVALID_OBSERVATION
                    if reserved
                    else PreparedFutureDisposition.NO_HANDOFF
                    if join.handoff is None
                    else PreparedFutureDisposition.OPERATIONAL_STOP
                    if not join.use_allowed
                    else PreparedFutureDisposition.NONATTEMPT
                )
                local.append(
                    SealedPreparedFutureLocator(
                        f"{slot.slot_id}.{view_id}.locator",
                        slot.slot_id,
                        view_id,
                        disposition,
                        receipt if reserved else None,
                        artifact if reserved else None,
                        () if available else (f"FLH_{disposition.value}",),
                        OutcomeAccess.EVALUATION_SEALED,
                    )
                )
            local = sorted(local, key=lambda locator: locator.locator_id)
            locators.extend(local)
            if reserved:
                completion = PreparedFutureCompletion(
                    f"{slot.slot_id}.completion",
                    ObjectIdentity.from_record(slot.slot_id, slot),
                    tuple(local),
                )
                coordinator.record_prepared_completion(
                    prefix_id=prefix_id, design=design, completion=completion, occurred_at_utc=now()
                )
                completions.append(completion)
        census = PreparedExecutionCensus(
            f"{prefix_id}.census",
            ObjectIdentity.from_record(design.binding_id, design),
            tuple(sorted(locators, key=lambda locator: locator.locator_id)),
        )
        terminal = coordinator.close_prepared_execution(
            prefix_id=prefix_id, design=design, census=census, occurred_at_utc=now()
        )
        bundles.append(
            SealedPreparedForecastPolicyBundle(
                f"{prefix_id}.sealed",
                design,
                returned,
                instance,
                early.task if instance is not None else None,
                early.decision if instance is not None else None,
                (),
                terminal,
                census,
                tuple(sorted(completions, key=lambda c: c.completion_id)),
                early,
            )
        )
    return FiniteResponseLawRootSealedControl(
        join, tuple(ObjectIdentity.from_record(v.report_id, v) for v in views), tuple(bundles)
    )
