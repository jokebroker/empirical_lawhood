"Separate D reveal through the installed prepared-policy controller-use evaluator."

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar


from empirical_lawhood.adapters.simulators.reactor_selected_action_response.prospective_records import ClassicalNativeRoot
from empirical_lawhood.kernel.admission import GateStatus
from empirical_lawhood.kernel.control import ScientificCommitmentKind
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.controller_study import ImplementationRole
from empirical_lawhood.planning.nested_controller_evaluation import PreparedFutureRole
from empirical_lawhood.runtime.controller_evaluation_nested import NestedControllerUseEvaluator, SealedPreparedFutureLocator, PreparedProbeCommitment, PreparedFutureDisposition, PreparedNativeReadout, PreparedPolicyUnitEvaluation, RevealedPreparedForecastPolicyBundle, RevealedPreparedFuture
from empirical_lawhood.runtime.controller_runtime import TickDisposition, ExactActionDeliveryTrace

from .control_measure import ClassicalMeasuredOperands, ClassicalMeasuredWord, measure_native_root
from .control_prospective_closeout import ClassicalSealResult
from .control_prospective_plan import ReactorSelectedActionResponseProspectivePlan
from .records import ClassicalDecision
from .records import ClassicalCausal, ClassicalPrivate
from .science import RECEIVERS


@dataclass(frozen=True, slots=True)
class ClassicalRevealResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-selected-action-response/classical-reveal-result'
    root: str
    sealed: ObjectIdentity
    native: ObjectIdentity
    measured: ClassicalMeasuredOperands
    revealed: tuple[RevealedPreparedForecastPolicyBundle, ...]
    units: tuple[PreparedPolicyUnitEvaluation, ...]
    evaluable: bool
    safe: bool
    success: bool
    robust_cooling_K: D | None
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            len(self.units) not in (0, 1)
            or len(self.revealed) != len(self.units)
            or (
                self.success
                and (
                    not self.evaluable
                    or not self.safe
                    or self.reasons
                    or self.robust_cooling_K is None
                )
            )
        ):
            raise ValueError("classical controller-use reveal lost its owner result or negative face")


def _readouts(
    row: ClassicalMeasuredWord | None,
    plan: ReactorSelectedActionResponseProspectivePlan,
) -> tuple[PreparedNativeReadout, ...]:
    if row is None or plan.plan is None:
        return ()
    result = []
    for tolerance in plan.plan.readout_tolerances:
        coordinate = tolerance.coordinate
        if coordinate.quantity_id == RECEIVERS[0]:
            value = row.measured_peak_K
        elif coordinate.quantity_id == RECEIVERS[1]:
            value = row.measured_cooling_K
        else:
            raise ValueError("D controller use readout changed its temperature/cooling receiver")
        result.append(PreparedNativeReadout(coordinate, value, tolerance.numerical_tolerance))
    return tuple(sorted(result, key=lambda item: item.coordinate.coordinate_id))


def _reference_trace(
    locator: SealedPreparedFutureLocator,
    native: ClassicalNativeRoot,
    plan: ReactorSelectedActionResponseProspectivePlan,
    callback: int,
    view: int,
    compiled_index: int,
    probe: PreparedProbeCommitment | None = None,
) -> ExactActionDeliveryTrace | None:
    """Map evaluator roles to the shared native feed-trace authenticator."""
    from empirical_lawhood.adapters.simulators.reactor_regime_response.feed_trace import measured_feed_trace

    compiled = native.compiled[compiled_index]
    if plan.plan is None or compiled is None:
        return None
    slot = next(s for s in plan.plan.futures if s.slot_id == locator.slot_id)
    word = (
        next(w.word for w in plan.plan.root_hold_words if w.root_id == native.root)
        if probe is None
        else next(
            w
            for w in plan.plan.audit_action_words
            if ObjectIdentity.from_record(w.word_id, w) == slot.assigned_probe_word
        )
    )
    episode = next(
        b.episode()
        for b in native.branches
        if (b.role, b.index, b.view)
        == ("EVALUATOR", 0 if probe is None else slot.ordinal + 1, view)
    )
    return measured_feed_trace(
        episode=episode,
        callback=callback,
        word=word,
        implementation=compiled.implementation(ImplementationRole.DELIVERY),
        commitment=ObjectIdentity.from_record(slot.slot_id, slot)
        if probe is None
        else ObjectIdentity.from_record(probe.probe_id, probe),
        kind=ScientificCommitmentKind.MEASURED_HOLD
        if probe is None
        else ScientificCommitmentKind.ACTION,
        trace_id=f"{locator.locator_id}.reference-delivery"
        if probe is None
        else f"{locator.locator_id}.probe-delivery",
    )


def reveal_native_root(
    *,
    sealed: ClassicalSealResult,
    native: ClassicalNativeRoot,
    plan_bundle: ReactorSelectedActionResponseProspectivePlan,
    assignment: ClassicalDecision,
    causal: ClassicalCausal,
    private: ClassicalPrivate,
    reveal_authority: ObjectIdentity,
) -> ClassicalRevealResult:
    """Use frozen intervals, actual owner ticks and both saved native views."""
    root = native.root
    if (
        sealed.root != root
        or sealed.native != ObjectIdentity.from_record(f"{root}.native-action", native)
        or sealed.plan != ObjectIdentity.from_record("classical.prospective-evaluation-plan", plan_bundle)
        or assignment.root != root
        or native.assignment != ObjectIdentity.from_record(assignment.decision_id, assignment)
    ):
        raise ValueError("D reveal substituted its sealed root, assignment or plan")
    measured = measure_native_root(
        native=native, assignment=assignment, causal=causal, private=private
    )
    decision = assignment
    if sealed.sealed is None or decision is None or not decision.causal_preparation_valid:
        if sealed.sealed is not None or native.locks:
            raise ValueError("D nonentry acquired a prepared controller-use owner")
        return ClassicalRevealResult(
            root,
            ObjectIdentity.from_record(f"{root}.prospective-evaluation-seal", sealed),
            ObjectIdentity.from_record(f"{root}.native-action", native),
            measured,
            (),
            (),
            False,
            False,
            False,
            None,
            sealed.nonentry_reasons,
        )
    assert decision.callback is not None
    plan = plan_bundle.plan
    if plan is None:
        raise ValueError("eligible D reveal lacks its frozen prepared controller-use plan")
    chart = measured.chart
    owner = measured.owner
    future_predicate = next(
        predicate
        for predicate in plan.evaluator_boundary.outcome_predicates
        if predicate.predicate_id in plan.future_preservation_predicate_ids
    )
    parent_predicate = next(
        predicate
        for predicate in plan.evaluator_boundary.outcome_predicates
        if predicate.predicate_id in plan.parent_preservation_predicate_ids
    )
    from empirical_lawhood.adapters.simulators.reactor_selected_action_response.acquisition import donor_episode

    maxima = tuple(
        float(
            donor_episode(causal, private, view)
            .grid[: decision.callback * (20 if view else 10) + 1, 1]
            .max()
        )
        for view in (0, 1)
    )
    parent_values = (NamedDecimal(parent_predicate.quantity_id, D(repr(max(maxima))), "K"),)
    revealed = []
    units = []
    slots = {slot.slot_id: slot for slot in plan.futures}
    by_view = {view_id: view for view, view_id in enumerate(plan.numerical_view_ids)}
    for index, bundle in enumerate(sealed.sealed.bundles):
        compiled = native.compiled[index]
        if compiled is None:
            raise ValueError("D prepared controller-use owner lacks its compiled programme")
        outcomes = []
        for locator in bundle.locators:
            slot = slots[locator.slot_id]
            view = by_view[locator.view_id]
            if slot.role is PreparedFutureRole.COMMITTED_TASK:
                pair = owner[index]
                row = None if pair is None else pair[view]
            elif slot.role is PreparedFutureRole.MATCHED_HOLD:
                row = chart[view]
            else:
                row = chart[2 * (slot.ordinal + 1) + view]
            available = (
                locator.disposition is PreparedFutureDisposition.COMPLETED and row is not None
            )
            trace = None
            if available and slot.role is PreparedFutureRole.COMMITTED_TASK:
                tick = native.ticks[index]
                if tick is not None and tick.disposition is TickDisposition.ACTION_DELIVERED:
                    trace = tick.delivery_trace
            elif available and slot.role is PreparedFutureRole.MATCHED_HOLD:
                trace = _reference_trace(
                    locator=locator,
                    native=native,
                    plan=plan_bundle,
                    callback=decision.callback,
                    view=view,
                    compiled_index=index,
                )
            elif available and slot.role is PreparedFutureRole.AUDIT_PROBE:
                probe = next(probe for probe in bundle.probes if probe.slot.slot_id == slot.slot_id)
                trace = _reference_trace(
                    locator=locator,
                    native=native,
                    plan=plan_bundle,
                    probe=probe,
                    callback=decision.callback,
                    view=view,
                    compiled_index=index,
                )
            outcomes.append(
                RevealedPreparedFuture(
                    f"{locator.locator_id}.revealed",
                    ObjectIdentity.from_record(locator.locator_id, locator),
                    locator.slot_id,
                    locator.view_id,
                    _readouts(row, plan_bundle) if available else (),
                    (NamedDecimal(future_predicate.quantity_id, row.measured_peak_K, "K"),)
                    if available
                    and row is not None
                    and slot.role
                    in (PreparedFutureRole.COMMITTED_TASK, PreparedFutureRole.AUDIT_PROBE)
                    else (),
                    trace,
                    OutcomeAccess.EVALUATOR_REVEAL,
                )
            )
        incomplete = tuple(
            (
                outcome.slot_id,
                outcome.view_id,
                len(outcome.readouts),
                outcome.delivery_trace is not None,
            )
            for outcome in outcomes
            if (not outcome.readouts or outcome.delivery_trace is None)
            and next(
                locator
                for locator in bundle.locators
                if (locator.slot_id, locator.view_id) == (outcome.slot_id, outcome.view_id)
            ).disposition
            is PreparedFutureDisposition.COMPLETED
        )
        if incomplete:
            raise ValueError(
                f'completed reactor controller use future lacks its verified readout/trace: {incomplete}'
            )
        result = RevealedPreparedForecastPolicyBundle(
            f"{bundle.bundle_id}.reveal",
            bundle,
            reveal_authority,
            tuple(sorted(outcomes, key=lambda value: value.outcome_id)),
            parent_values,
        )
        unit = NestedControllerUseEvaluator(
            bundle.design.evaluator_binding, plan.reducer
        ).evaluate_prepared_unit(revealed=result, compiled=compiled)
        revealed.append(result)
        units.append(unit)
    from .config import ClassicalDesign

    design = ClassicalDesign()
    pair = measured.owner[0]
    reasons = set(measured.reasons)
    evaluable = pair is not None and all(row is not None for row in measured.chart)
    safe = max(maxima) <= float(design.temperature_limit_K)
    robust = None
    if pair is not None:
        robust = min(row.measured_cooling_K for row in pair)
        safe = safe and all(row.measured_peak_K <= design.temperature_limit_K for row in pair)
        if not all(
            design.response_lower_K <= row.measured_cooling_K <= design.response_upper_K
            for row in pair
        ):
            reasons.add("PROSPECTIVE_RESPONSE_BOUND_FAILED")
        if decision.observed_temperature_K is None or not all(
            abs(row.measured_peak_K - decision.observed_temperature_K) <= D(1) for row in pair
        ):
            reasons.add("PROSPECTIVE_CAUSAL_SAFETY_ENVELOPE_FAILED")
        if (
            abs(pair[0].measured_peak_K - pair[1].measured_peak_K)
            > design.numerical_peak_tolerance_K
            or abs(pair[0].measured_cooling_K - pair[1].measured_cooling_K)
            > design.numerical_cooling_tolerance_K
        ):
            reasons.add("PROSPECTIVE_NUMERICAL_AGREEMENT_FAILED")
        if not all(
            row.receipt_valid and abs(row.delivered_mass_kg - D(".16")) <= D("1e-12")
            for row in pair
        ):
            reasons.add("PROSPECTIVE_NATIVE_DELIVERY_FAILED")
    if not safe:
        reasons.add("PROSPECTIVE_NATIVE_SAFETY_FAILED")
    if not evaluable:
        reasons.add("PROSPECTIVE_MEASUREMENT_UNEVALUABLE")
    if not all(
        unit.admitted
        and unit.task_success
        and unit.probe_adequacy is GateStatus.PASS
        and unit.probe_coverage is GateStatus.PASS
        for unit in units
    ):
        reasons.add("PREPARED_CONTROLLER_USE_OWNER_DID_NOT_VALIDATE_USE")
    return ClassicalRevealResult(
        root,
        ObjectIdentity.from_record(f"{root}.prospective-evaluation-seal", sealed),
        ObjectIdentity.from_record(f"{root}.native-action", native),
        measured,
        tuple(revealed),
        tuple(units),
        evaluable,
        safe,
        not reasons,
        robust,
        tuple(sorted(reasons)),
    )
