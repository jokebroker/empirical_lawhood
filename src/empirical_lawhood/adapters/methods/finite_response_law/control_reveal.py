"Map actual paired native measurements into the existing prepared controller-use owner."

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar

from empirical_lawhood.kernel.control import OperationalDeliveryState, ScientificCommitmentKind
from empirical_lawhood.planning.controller_study import DeliveryEquivalenceSpec
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.controller_study import ImplementationRole
from empirical_lawhood.planning.nested_controller_evaluation import PreparedFutureRole
from empirical_lawhood.runtime.controller_runtime import ExactActionDeliveryTrace
from empirical_lawhood.runtime.controller_evaluation_nested import DurablePreparedExecutionEventStore, PreparedFutureDisposition, PreparedNativeReadout, RevealedPreparedFuture, RevealedPreparedForecastPolicyBundle, PreparedPolicyUnitEvaluation, NestedControllerUseEvaluator
from .consumer import FiniteResponseLawNativeUnitReadoutMap, response_coordinates, SPEC
from .control_closeout import FiniteResponseLawRootSealedControl, read_compiled
from .control_delivery import _force
from .control_evaluation import hold_delivery_coordinate
from .evaluation_native_records import FiniteResponseLawEvaluationViewObservation
from .paired_assay import paired_native_assay, prepared_paired_readouts


@dataclass(frozen=True, slots=True)
class FiniteResponseLawProspectiveRootEvaluation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-response-law/finite-response-law-prospective-root-evaluation'
    sealed: ObjectIdentity
    revealed: tuple[RevealedPreparedForecastPolicyBundle, ...]
    units: tuple[PreparedPolicyUnitEvaluation, ...]

    def __post_init__(self) -> None:
        if (
            self.sealed.object_schema != FiniteResponseLawRootSealedControl.SCHEMA
            or not self.units
            or len(self.units) != len(self.revealed)
            or len({u.root_id for u in self.units}) != 1
            or tuple(u.policy_id for u in self.units)
            != tuple(sorted({u.policy_id for u in self.units}))
            or any(
                u.revealed_bundle != ObjectIdentity.from_record(r.reveal_id, r)
                for u, r in zip(self.units, self.revealed, strict=True)
            )
        ):
            raise ValueError("Finite response-law evaluation controller-use report changes its all-consumer generic evaluation census")


def reveal_root(
    *,
    sealed: FiniteResponseLawRootSealedControl,
    views: tuple[FiniteResponseLawEvaluationViewObservation, ...],
    reveal_authority: ObjectIdentity,
    store: DurablePreparedExecutionEventStore,
) -> FiniteResponseLawProspectiveRootEvaluation:
    """Caller owns typed reveal authority and authenticated view-task custody."""
    if tuple(ObjectIdentity.from_record(v.report_id, v) for v in views) != sealed.views:
        raise ValueError("Finite response-law evaluation reveal substitutes the two projections fixed in its sealed census")
    mapping = FiniteResponseLawNativeUnitReadoutMap(
        response_coordinates(ObjectIdentity.from_record("flh-science", SPEC))
    )
    revealed, units = [], []
    for i, bundle in enumerate(sealed.bundles):
        plan = bundle.design.evaluation_plan
        compiled = None if bundle.instance is None else read_compiled(sealed.join, i, store)
        outcomes = []
        slots = {s.slot_id: s for s in plan.futures}
        by_view = dict(zip(plan.numerical_view_ids, views, strict=True))
        for locator in bundle.locators:
            slot, view = slots[locator.slot_id], by_view[locator.view_id]
            readouts: tuple[PreparedNativeReadout, ...] = ()
            preservation: tuple[NamedDecimal, ...] = ()
            trace = None
            if locator.disposition is PreparedFutureDisposition.COMPLETED:
                assert compiled is not None
                hold = slot.role is PreparedFutureRole.MATCHED_HOLD
                if hold:
                    word = plan.matched_hold_word
                    commitment = ObjectIdentity.from_record(slot.slot_id, slot)
                else:
                    decision = bundle.forecast_parent.decision
                    assert decision.action_binding is not None
                    word = decision.action_binding.action_word
                    commitment = ObjectIdentity.from_record(decision.commitment_id, decision)
                observed = next(
                    o
                    for o in view.delivery_observations
                    if o.expected_occurrence_id == word.occurrences[0].occurrence_id
                )
                # Exact full traces for BOTH independent purposes were reduced
                # before reveal. No requested value is promoted to observation.
                if not observed.complete or observed.reason_codes:
                    raise ValueError("Completed finite response-law evaluation locator lacks actual exact native delivery")
                equivalence = compiled.study.delivery_equivalence
                if (
                    not isinstance(equivalence, DeliveryEquivalenceSpec)
                    or equivalence.stage_value_tolerance.value != 0
                ):
                    raise ValueError(
                        "Finite response-law evaluation reveal lost its exact zero-tolerance native delivery binding"
                    )
                trace = ExactActionDeliveryTrace(
                    f"{locator.locator_id}.delivery",
                    commitment,
                    compiled.implementation(ImplementationRole.DELIVERY),
                    ScientificCommitmentKind.MEASURED_HOLD
                    if hold
                    else ScientificCommitmentKind.ACTION,
                    word,
                    (observed,),
                    OperationalDeliveryState.DELIVERED,
                    (),
                )
                if hold:
                    assert observed.realized is not None
                    readouts = (
                        PreparedNativeReadout(
                            hold_delivery_coordinate(word), observed.realized.value, D(0)
                        ),
                    )
                else:
                    assay = paired_native_assay(
                        views, parent=view.root.assigned_parent, word=_force(word)
                    )
                    readouts = mapping.readouts(
                        prepared_paired_readouts(
                            assay, mapping.native_coordinates, refinement=view.refinement
                        )
                    )
                    predicate = next(
                        p
                        for p in plan.evaluator_boundary.outcome_predicates
                        if p.predicate_id in plan.future_preservation_predicate_ids
                    )
                    preservation = (NamedDecimal(predicate.quantity_id, D(1), "1"),)
            outcomes.append(
                RevealedPreparedFuture(
                    f"{locator.locator_id}.revealed",
                    ObjectIdentity.from_record(locator.locator_id, locator),
                    locator.slot_id,
                    locator.view_id,
                    readouts,
                    preservation,
                    trace,
                    OutcomeAccess.EVALUATOR_REVEAL,
                )
            )
        predicate = next(
            p
            for p in plan.evaluator_boundary.outcome_predicates
            if p.predicate_id in plan.parent_preservation_predicate_ids
        )
        work = sealed.join.parent_work
        parent_values = (
            ()
            if any(w is None for w in work)
            else (
                NamedDecimal(
                    predicate.quantity_id, D(32) - max(w for w in work if w is not None), "1"
                ),
            )
        )
        result = RevealedPreparedForecastPolicyBundle(
            f"{bundle.bundle_id}.reveal",
            bundle,
            reveal_authority,
            tuple(sorted(outcomes, key=lambda o: o.outcome_id)),
            parent_values,
        )
        evaluator = NestedControllerUseEvaluator(bundle.design.evaluator_binding, plan.reducer)
        units.append(evaluator.evaluate_prepared_unit(revealed=result, compiled=compiled))
        revealed.append(result)
    return FiniteResponseLawProspectiveRootEvaluation(
        ObjectIdentity.from_record(sealed.result_id, sealed), tuple(revealed), tuple(units)
    )
