"Separate D reveal through the installed prepared-policy controller-use evaluator."

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar

import numpy as np

from empirical_lawhood.adapters.simulators.reactor_regime_response.prospective_records import RegimeDNativeRoot
from empirical_lawhood.kernel.action_contracts import ObservedActionOccurrence
from empirical_lawhood.kernel.admission import GateStatus
from empirical_lawhood.kernel.control import OperationalDeliveryState, ScientificCommitmentKind
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.controller_study import ImplementationRole
from empirical_lawhood.planning.nested_controller_evaluation import PreparedFutureRole
from empirical_lawhood.runtime.controller_evaluation_nested import NestedControllerUseEvaluator, PreparedFutureDisposition, PreparedNativeReadout, PreparedPolicyUnitEvaluation, RevealedPreparedForecastPolicyBundle, RevealedPreparedFuture, SealedPreparedFutureLocator, PreparedProbeCommitment
from empirical_lawhood.runtime.controller_runtime import ExactActionDeliveryTrace, TickDisposition

from .control_math import Admission, REQUESTS_K, evaluate_root
from .control_measure import RegimeDMeasuredOperands, RegimeDMeasuredWord, measure_native_root
from .control_prospective_closeout import RegimeDRootSealResult
from .control_prospective_plan import ReactorRegimeResponsePreparedProspectivePlanBundle
from .comparison_math import POLICIES, PolicyRootOutcome
from .prospective_decision import CausalValidityRegimeAssignment
from .records import RegimeCausalPreparation, RegimePrivatePreparation
from .science import RECEIVERS


@dataclass(frozen=True, slots=True)
class RegimeDPolicyRoot(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/regime-d-policy-root'

    root: str
    policy: str
    source: str
    evaluable: bool
    choices: tuple[int | None, int | None, int | None, int | None]
    successes: tuple[bool, bool, bool, bool]
    delivered_mass_kg: tuple[D | None, D | None, D | None, D | None]
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.policy not in POLICIES
            or self.source != ("OWNER" if self.policy == "EL" else "SHADOW_CHART")
            or len(self.choices) != 4
            or len(self.successes) != 4
            or len(self.delivered_mass_kg) != 4
            or (not self.evaluable and not self.reasons)
            or any(
                choice not in (None, 1, 2)
                or (choice is None and success)
                or (mass is not None and not D(0) <= mass <= D(".32"))
                for choice, success, mass in zip(
                    self.choices, self.successes, self.delivered_mass_kg, strict=True
                )
            )
        ):
            raise ValueError("D policy root changed a frozen choice or native source")

    def to_comparison(self) -> PolicyRootOutcome:
        if not self.evaluable:
            raise ValueError("unevaluable D shadow policy cannot enter a favorable comparison")
        return PolicyRootOutcome(
            self.root, self.policy, self.source, self.choices,
            self.successes,
            tuple(
                None if value is None else float(value)
                for value in self.delivered_mass_kg
            ),  # type: ignore[arg-type]
            tuple(choice is None for choice in self.choices),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class RegimeDRootRevealResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/regime-d-root-reveal-result'

    root: str
    sealed: ObjectIdentity
    assignment: ObjectIdentity
    native: ObjectIdentity
    measured: RegimeDMeasuredOperands
    revealed: tuple[RevealedPreparedForecastPolicyBundle, ...]
    units: tuple[PreparedPolicyUnitEvaluation, ...]
    policies: tuple[RegimeDPolicyRoot, ...]
    physical_A: bool
    physical_request_J: tuple[bool, bool, bool, bool]
    physical_F: bool
    A: bool
    request_J: tuple[bool, bool, bool, bool]
    J: bool
    C: bool
    F: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.sealed.object_schema != RegimeDRootSealResult.SCHEMA
            or self.assignment.object_schema != CausalValidityRegimeAssignment.SCHEMA
            or self.native.object_schema != RegimeDNativeRoot.SCHEMA
            or len(self.revealed) not in (0, 4)
            or len(self.revealed) != len(self.units)
            or tuple(row.policy for row in self.policies) != POLICIES
            or any(row.root != self.root for row in self.policies)
            or self.J != all(self.request_J)
            or self.C != (self.A and self.J)
            or (self.F and not (self.physical_F or any(
                unit.admitted and not unit.task_success for unit in self.units
            )))
        ):
            raise ValueError("D controller-use reveal changed its same-root A/J/C/F or owner census")


def _readouts(
    row: RegimeDMeasuredWord | None,
    plan: ReactorRegimeResponsePreparedProspectivePlanBundle,
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
        result.append(PreparedNativeReadout(
            coordinate, value, tolerance.numerical_tolerance
        ))
    return tuple(sorted(result, key=lambda item: item.coordinate.coordinate_id))


def _hold_trace(
    *,
    locator: SealedPreparedFutureLocator,
    native: RegimeDNativeRoot,
    plan: ReactorRegimeResponsePreparedProspectivePlanBundle,
    callback: int,
    view: int,
    compiled_index: int,
) -> ExactActionDeliveryTrace | None:
    if plan.plan is None:
        return None
    episode = next(
        row.episode() for row in native.branches
        if (row.role, row.index, row.view) == ("EVALUATOR", 0, view)
    )
    compiled = native.compiled[compiled_index]
    if (episode is None or compiled is None
            or min(len(episode.exposure), len(episode.requests), len(episode.stages)) <= callback):
        return None
    exposure = episode.exposure[callback]
    if (
        episode.requests[callback, 0] != 0
        or episode.stages[callback, 0] != 0
        or episode.stages[callback, 2] != 0
        or not np.all(exposure[:, 2] == 0)
        or not np.array_equal(
            exposure[:, 0],
            callback * 10 + np.arange(len(exposure)) * episode.dt,
        )
    ):
        return None
    word = plan.plan.matched_hold_word
    if any(
        any(value.value != 0 for value in (
            occurrence.requested, occurrence.accepted,
            occurrence.applied, occurrence.realized,
        ))
        for occurrence in word.occurrences
    ):
        return None
    expected_runs = tuple(sorted(
        (
            float(occurrence.realized.coordinate.coordinate),
            float(occurrence.duration),
        )
        for occurrence in word.occurrences
    ))
    clock = float(callback * 10)
    for start, duration in expected_runs:
        if start != clock or duration <= 0:
            return None
        clock += duration
    if clock != float(callback * 10 + 10) or any(
        float(occurrence.requested.coordinate.coordinate) != callback * 10
        or float(occurrence.accepted.coordinate.coordinate) != callback * 10
        for occurrence in word.occurrences
    ):
        return None
    observed = tuple(
        ObservedActionOccurrence(
            f"observed.{occurrence.occurrence_id}",
            occurrence.occurrence_id,
            occurrence.requested, occurrence.accepted,
            occurrence.applied, occurrence.realized, (),
        )
        for occurrence in word.occurrences
    )
    slot = next(
        slot for slot in plan.plan.futures if slot.slot_id == locator.slot_id
    )
    return ExactActionDeliveryTrace(
        f"{locator.locator_id}.reference-delivery",
        ObjectIdentity.from_record(slot.slot_id, slot),
        compiled.implementation(ImplementationRole.DELIVERY),
        ScientificCommitmentKind.MEASURED_HOLD,
        word, observed, OperationalDeliveryState.DELIVERED, (),
    )


def _probe_trace(
    *, locator: SealedPreparedFutureLocator,
    native: RegimeDNativeRoot,
    plan: ReactorRegimeResponsePreparedProspectivePlanBundle,
    probe: PreparedProbeCommitment,
    callback: int, view: int, compiled_index: int,
) -> ExactActionDeliveryTrace | None:
    """Authenticate an evaluator probe's actual scalar native stages."""
    if plan.plan is None:
        return None
    slot = probe.slot
    if slot.slot_id != locator.slot_id or slot.assigned_probe_word is None:
        return None
    words = tuple(
        word for word in plan.plan.audit_action_words
        if ObjectIdentity.from_record(word.word_id, word) == slot.assigned_probe_word
    )
    if len(words) != 1:
        return None
    word = words[0]
    episode = next(
        row.episode() for row in native.branches
        if (row.role, row.index, row.view)
        == ("EVALUATOR", slot.ordinal + 1, view)
    )
    compiled = native.compiled[compiled_index]
    if (episode is None or compiled is None
            or min(len(episode.exposure), len(episode.requests), len(episode.stages)) <= callback):
        return None
    exposure = episode.exposure[callback]
    requested = D(repr(float(episode.requests[callback, 0])))
    accepted = D(repr(float(episode.stages[callback, 0])))
    applied = D(repr(float(episode.stages[callback, 2])))
    dt = D(1) if view == 0 else D(".5")
    clock = D(0)
    for occurrence in word.occurrences:
        start = occurrence.realized.coordinate.coordinate - D(callback * 10)
        duration = occurrence.duration
        if (
            start != clock or duration <= 0
            or start / dt != int(start / dt)
            or duration / dt != int(duration / dt)
            or occurrence.requested.value != requested
            or occurrence.accepted.value != accepted
            or occurrence.applied.value != applied
        ):
            return None
        first, last = int(start / dt), int((start + duration) / dt)
        if last > len(exposure) or any(
            D(repr(float(row[0]))) != D(callback * 10) + D(step) * dt
            or D(repr(float(row[1]))) != dt
            or D(repr(float(row[2]))) != occurrence.realized.value
            for step, row in enumerate(exposure[first:last], start=first)
        ):
            return None
        clock += duration
    if clock != D(10):
        return None
    observed = tuple(
        ObservedActionOccurrence(
            f"observed.{occurrence.occurrence_id}", occurrence.occurrence_id,
            occurrence.requested, occurrence.accepted,
            occurrence.applied, occurrence.realized, (),
        )
        for occurrence in word.occurrences
    )
    return ExactActionDeliveryTrace(
        f"{locator.locator_id}.probe-delivery",
        ObjectIdentity.from_record(probe.probe_id, probe),
        compiled.implementation(ImplementationRole.DELIVERY),
        ScientificCommitmentKind.ACTION,
        word, observed, OperationalDeliveryState.DELIVERED, (),
    )


def reveal_native_root(
    *,
    sealed: RegimeDRootSealResult,
    native: RegimeDNativeRoot,
    plan_bundle: ReactorRegimeResponsePreparedProspectivePlanBundle,
    assignment: CausalValidityRegimeAssignment,
    causal: RegimeCausalPreparation,
    private: RegimePrivatePreparation,
    reveal_authority: ObjectIdentity,
) -> RegimeDRootRevealResult:
    """Use frozen intervals, actual owner ticks and both saved native views."""
    root = native.root
    if (
        sealed.root != root
        or sealed.native != ObjectIdentity.from_record(f"{root}.native-action", native)
        or sealed.plan != ObjectIdentity.from_record("regime.d-plan", plan_bundle)
        or assignment.root != root
        or native.assignment != ObjectIdentity.from_record(assignment.assignment_id, assignment)
    ):
        raise ValueError("D reveal substituted its sealed root, assignment or plan")
    measured = measure_native_root(
        native=native, assignment=assignment, causal=causal, private=private
    )
    decision = assignment.decision
    if sealed.sealed is None or decision is None or not decision.causal_preparation_valid:
        if sealed.sealed is not None or native.locks:
            raise ValueError("D nonentry acquired a prepared controller-use owner")
        return RegimeDRootRevealResult(
            root,
            ObjectIdentity.from_record(f"{root}.d-seal", sealed),
            ObjectIdentity.from_record(assignment.assignment_id, assignment),
            ObjectIdentity.from_record(f"{root}.native-action", native),
            measured, (), (),
            tuple(
                RegimeDPolicyRoot(
                    root, policy, "OWNER" if policy == "EL" else "SHADOW_CHART",
                    True, (None,) * 4, (False,) * 4, (None,) * 4,
                    ("D_SELECTED_PREPARATION_NONENTRY",),
                )
                for policy in POLICIES
            ),
            False, (False,) * 4, False,
            False, (False,) * 4, False, False, False,
            sealed.nonentry_reasons,
        )
    plan = plan_bundle.plan
    if plan is None:
        raise ValueError("eligible D reveal lacks its frozen prepared controller-use plan")
    chart = measured.chart
    owner = measured.owner
    future_predicate = next(
        predicate for predicate in plan.evaluator_boundary.outcome_predicates
        if predicate.predicate_id in plan.future_preservation_predicate_ids
    )
    parent_predicate = next(
        predicate for predicate in plan.evaluator_boundary.outcome_predicates
        if predicate.predicate_id in plan.parent_preservation_predicate_ids
    )
    from .preparation_evidence import preparation_evidence

    preparation = preparation_evidence(causal, private, decision.route)
    maxima = (preparation.nominal_max_K, preparation.refined_max_K)
    parent_values = () if any(value is None for value in maxima) else (
        NamedDecimal(
            parent_predicate.quantity_id, D(repr(max(value for value in maxima if value is not None))), "K"
        ),
    )
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
                trace = _hold_trace(
                    locator=locator, native=native, plan=plan_bundle,
                    callback=decision.callback, view=view, compiled_index=index,
                )
            elif available and slot.role is PreparedFutureRole.AUDIT_PROBE:
                probe = next(
                    probe for probe in bundle.probes
                    if probe.slot.slot_id == slot.slot_id
                )
                trace = _probe_trace(
                    locator=locator, native=native, plan=plan_bundle,
                    probe=probe, callback=decision.callback, view=view,
                    compiled_index=index,
                )
            outcomes.append(RevealedPreparedFuture(
                f"{locator.locator_id}.revealed",
                ObjectIdentity.from_record(locator.locator_id, locator),
                locator.slot_id, locator.view_id,
                _readouts(row, plan_bundle) if available else (),
                (NamedDecimal(future_predicate.quantity_id, row.measured_peak_K, "K"),)
                if available and row is not None and slot.role is PreparedFutureRole.COMMITTED_TASK
                else (),
                trace, OutcomeAccess.EVALUATOR_REVEAL,
            ))
        incomplete = tuple(
            (outcome.slot_id, outcome.view_id, len(outcome.readouts),
             outcome.delivery_trace is not None)
            for outcome in outcomes
            if (
                not outcome.readouts or outcome.delivery_trace is None
            )
            and next(locator for locator in bundle.locators
                     if (locator.slot_id, locator.view_id)
                     == (outcome.slot_id, outcome.view_id)).disposition
            is PreparedFutureDisposition.COMPLETED
        )
        if incomplete:
            raise ValueError(f'completed reactor controller use future lacks its verified readout/trace: {incomplete}')
        result = RevealedPreparedForecastPolicyBundle(
            f"{bundle.bundle_id}.reveal",
            bundle, reveal_authority,
            tuple(sorted(outcomes, key=lambda value: value.outcome_id)),
            parent_values,
        )
        unit = NestedControllerUseEvaluator(
            bundle.design.evaluator_binding, plan.reducer
        ).evaluate_prepared_unit(revealed=result, compiled=compiled)
        revealed.append(result)
        units.append(unit)
    forecasts = tuple(value.to_word() for value in decision.primary_words)
    admissions = tuple(
        Admission(
            REQUESTS_K[index], choice,
            () if choice is None else (choice,),
            decision.refusal_reasons[index],
        )
        for index, choice in enumerate(decision.policies[0].choices)
    )
    delivered = tuple(
        choice if tick is not None
        and tick.disposition is TickDisposition.ACTION_DELIVERED
        and tick.delivery_trace.exact
        else None
        for choice, tick in zip(
            decision.policies[0].choices, native.ticks, strict=True
        )
    )
    measured_preparation_safe = preparation.safe
    physical = evaluate_root(
        forecasts,
        tuple(value.to_measured() for value in chart if value is not None),
        admissions,
        preparation_safe=measured_preparation_safe,
        numerically_valid=True,
        delivered_words=delivered,
        primary_outcomes=tuple(
            None if pair is None else (pair[0].to_measured(), pair[1].to_measured())
            for pair in owner
        ),
    )
    request_J = physical.successes
    adequate = bool(
        physical.adequate
        and all(
            not physical.successes[index] or unit.task_success
            for index, unit in enumerate(units)
        )
        and all(
            unit.probe_adequacy is GateStatus.PASS
            and unit.probe_coverage is GateStatus.PASS
            for unit in units
        )
    )
    false_admission = bool(
        physical.false_admission
        or any(unit.admitted and not unit.task_success for unit in units)
    )
    policies = []
    for policy in decision.policies:
        if not policy.evaluable:
            policies.append(RegimeDPolicyRoot(
                root, policy.policy,
                "OWNER" if policy.policy == "EL" else "SHADOW_CHART",
                False, policy.choices, (False,) * 4, (None,) * 4,
                policy.reasons,
            ))
            continue
        successes: list[bool] = []
        masses: list[D | None] = []
        for request, choice in enumerate(policy.choices):
            if choice is None:
                successes.append(False)
                masses.append(None)
                continue
            if policy.policy == "EL":
                pair = owner[request]
                successes.append(request_J[request])
                masses.append(None if pair is None else pair[0].delivered_mass_kg)
            else:
                nominal = chart[2 * choice]
                refined = chart[2 * choice + 1]
                successes.append(bool(
                    nominal is not None and refined is not None
                    and measured_preparation_safe
                    and all(
                        value.measured_cooling_K >= D(repr(REQUESTS_K[request]))
                        and value.measured_peak_K <= D("356.2")
                        and value.dose_valid and value.window_valid
                        and value.receipt_valid and value.observation_valid
                        for value in (nominal, refined) if value is not None
                    )
                ))
                masses.append(None if nominal is None else nominal.delivered_mass_kg)
        policies.append(RegimeDPolicyRoot(
            root, policy.policy,
            "OWNER" if policy.policy == "EL" else "SHADOW_CHART",
            policy.evaluable,
            policy.choices,
            tuple(successes),  # type: ignore[arg-type]
            tuple(masses),  # type: ignore[arg-type]
            policy.reasons,
        ))
    return RegimeDRootRevealResult(
        root,
        ObjectIdentity.from_record(f"{root}.d-seal", sealed),
        ObjectIdentity.from_record(assignment.assignment_id, assignment),
        ObjectIdentity.from_record(f"{root}.native-action", native),
        measured,
        tuple(revealed), tuple(units), tuple(policies),
        physical.adequate, physical.successes, physical.false_admission,
        adequate, request_J, all(request_J),
        adequate and all(request_J), false_admission,
        tuple(sorted(set(physical.evaluability_reasons) | set(measured.reasons))),
    )
