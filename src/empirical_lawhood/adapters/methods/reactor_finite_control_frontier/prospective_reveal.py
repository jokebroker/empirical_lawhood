"Separate reveal through the shared prepared-unit controller-use evaluator."

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar

from empirical_lawhood.adapters.simulators.reactor_finite_control_frontier.prospective import FrontierNativeRoot
from empirical_lawhood.adapters.simulators.reactor_finite_control_frontier.trace import measured_trace
from empirical_lawhood.kernel.admission import GateStatus
from empirical_lawhood.kernel.control import ScientificCommitmentKind
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.controller_study import ImplementationRole
from empirical_lawhood.planning.nested_controller_evaluation import PreparedFutureRole
from empirical_lawhood.runtime.controller_evaluation_nested import NestedControllerUseEvaluator, PreparedFutureDisposition, PreparedNativeReadout, PreparedPolicyUnitEvaluation, RevealedPreparedForecastPolicyBundle, RevealedPreparedFuture
from empirical_lawhood.runtime.canonical_record_archive import CanonicalRecordArchive
from .config import ZERO
from .law_terminal import FrontierLaws
from .prospective_lock import FrontierFrozenRoot
from .prospective_plan import ReactorFiniteControlFrontierProspectivePlan
from .prospective_seal import FrontierSealedRoot
from .qualification import bound_event
from .science import RECEIVERS
from .selection import FrontierUseRequest


@dataclass(frozen=True, slots=True)
class FrontierUseResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-finite-control-frontier/frontier-use-result'
    request: FrontierUseRequest
    admitted: bool
    evaluable: bool
    success: bool
    unsafe: bool
    false_admission: bool
    reasons: tuple[str, ...]
    revealed_archive: CanonicalRecordArchive | None
    unit: PreparedPolicyUnitEvaluation | None

    @property
    def revealed(self) -> RevealedPreparedForecastPolicyBundle | None:
        if self.revealed_archive is None:
            return None
        archive = self.revealed_archive
        value = decode_canonical_bytes(
            archive.unpack(),
            RevealedPreparedForecastPolicyBundle,
            maximum_bytes=archive.decoded_bytes,
        )
        if (
            archive.subject != ObjectIdentity.from_record(value.reveal_id, value)
            or value.sealed.design.policy_id != f"el.{self.request.request_id}"
        ):
            raise ValueError("revealed owner archive substituted its exact request")
        return value

    def __post_init__(self) -> None:
        if (self.unit is None) != (self.revealed_archive is None) or (
            self.revealed_archive is not None
            and (
                self.unit is None
                or self.revealed_archive.subject != self.unit.revealed_bundle
                or self.revealed_archive.subject.object_schema
                != RevealedPreparedForecastPolicyBundle.SCHEMA
            )
        ):
            raise ValueError("Controller use unit lost its exact archived owner reveal")
        if self.success and (
            not self.admitted
            or not self.evaluable
            or self.unsafe
            or self.false_admission
            or self.reasons
            or self.unit is None
        ):
            raise ValueError("Controller use success contradicts its whole assigned event")
        if self.false_admission != (self.admitted and self.evaluable and not self.success):
            raise ValueError("known admitted failure changed its denominator")


@dataclass(frozen=True, slots=True)
class FrontierRevealedRoot(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-finite-control-frontier/frontier-revealed-root'
    root: str
    sealed: ObjectIdentity
    uses: tuple[FrontierUseResult, ...]

    @property
    def record_id(self) -> str:
        return f"{self.root}.frontier-reveal"


def reveal_root(
    *,
    sealed: FrontierSealedRoot,
    native: FrontierNativeRoot,
    frozen: FrontierFrozenRoot,
    laws: FrontierLaws,
    plan: ReactorFiniteControlFrontierProspectivePlan,
    reveal_authority: ObjectIdentity,
) -> FrontierRevealedRoot:
    if (
        sealed.native != ObjectIdentity.from_record(native.record_id, native)
        or native.frozen != ObjectIdentity.from_record(frozen.record_id, frozen)
        or sealed.plan != ObjectIdentity.from_record(plan.record_id, plan)
        or frozen.plan != sealed.plan
        or plan.laws != ObjectIdentity.from_record(laws.record_id, laws)
    ):
        raise ValueError("Controller-use reveal changed its sealed source/plan/law identity")
    results = []
    data = native.assay.arrays.unpack()
    for use, measured, actual in zip(frozen.uses, sealed.measured.uses, native.uses, strict=True):
        request = use.request
        if use.projection is None:
            results.append(
                FrontierUseResult(
                    request, False, True, False, False, False, use.reasons, None, None
                )
            )
            continue
        bundle = sealed.bundle_for(request)
        assert use.compiled is not None and use.lock is not None
        evaluation = use.lock.evaluation_for(use.compiled)
        if bundle.design != use.lock.design:
            raise ValueError("sealed reveal substituted its precontact design")
        assert (
            evaluation is not None and use.compiled is not None and measured.observation is not None
        )
        row = measured.observation
        audit = next(
            o for o in sealed.measured.chart.observations if o.coordinate == row.coordinate
        )
        parent_predicate = next(
            p
            for p in evaluation.evaluator_boundary.outcome_predicates
            if p.predicate_id in evaluation.parent_preservation_predicate_ids
        )
        future_predicate = next(
            p
            for p in evaluation.evaluator_boundary.outcome_predicates
            if p.predicate_id in evaluation.future_preservation_predicate_ids
        )
        prefix_values = [
            D(repr(float(data[f"{request.context}_v{v}_prefix_peak"][0])))
            for v in (0, 1)
            if f"{request.context}_v{v}_prefix_peak" in data
        ]
        parent_values = (
            ()
            if len(prefix_values) != 2
            else (NamedDecimal(parent_predicate.quantity_id, max(prefix_values), "K"),)
        )
        outcomes = []
        for locator in bundle.locators:
            slot = next(s for s in evaluation.futures if s.slot_id == locator.slot_id)
            view = evaluation.numerical_view_ids.index(locator.view_id)
            current = row if slot.role is PreparedFutureRole.COMMITTED_TASK else audit
            available = (
                locator.disposition is PreparedFutureDisposition.COMPLETED and current.evaluable
            )
            readouts: tuple[PreparedNativeReadout, ...] = ()
            preservation: tuple[NamedDecimal, ...] = ()
            trace = None
            if available:
                reference = slot.role is PreparedFutureRole.MATCHED_HOLD
                peak = (
                    current.reference_peaks_K[view] if reference else current.action_peaks_K[view]
                )
                cooling = D(0) if reference else current.effects_K[view]
                readouts = tuple(
                    sorted(
                        (
                            PreparedNativeReadout(
                                t.coordinate,
                                peak if t.coordinate.quantity_id == RECEIVERS[0] else cooling,
                                t.numerical_tolerance,
                            )
                            for t in evaluation.readout_tolerances
                        ),
                        key=lambda r: r.coordinate.coordinate_id,
                    )
                )
                if not reference:
                    preservation = (NamedDecimal(future_predicate.quantity_id, peak, "K"),)
                if slot.role is PreparedFutureRole.COMMITTED_TASK:
                    trace = None if actual.tick is None else actual.tick.delivery_trace
                else:
                    probe = (
                        None
                        if reference
                        else next(p for p in bundle.probes if p.slot.slot_id == slot.slot_id)
                    )
                    word = evaluation.matched_hold_word if reference else use.projection.word
                    if not reference and (
                        ObjectIdentity.from_record(word.word_id, word) != slot.assigned_probe_word
                    ):
                        raise ValueError("audit branch substituted its frozen probe word")
                    trace = measured_trace(
                        data=data,
                        key=f"{request.context}_{(ZERO if reference else use.projection.pulse).word_id}_v0",
                        word=word,
                        implementation=use.compiled.implementation(ImplementationRole.DELIVERY),
                        commitment=ObjectIdentity.from_record(slot.slot_id, slot)
                        if probe is None
                        else ObjectIdentity.from_record(probe.probe_id, probe),
                        kind=ScientificCommitmentKind.MEASURED_HOLD
                        if reference
                        else ScientificCommitmentKind.ACTION,
                        trace_id=f"{locator.locator_id}.nominal-delivery",
                    )
                if trace is None:
                    raise ValueError("completed future lacks authenticated nominal delivery trace")
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
        revealed = RevealedPreparedForecastPolicyBundle(
            f"{bundle.bundle_id}.reveal",
            bundle,
            reveal_authority,
            tuple(sorted(outcomes, key=lambda o: o.outcome_id)),
            parent_values,
        )
        unit = NestedControllerUseEvaluator(
            bundle.design.evaluator_binding, evaluation.reducer
        ).evaluate_prepared_unit(revealed=revealed, compiled=use.compiled)
        bound = next(
            r.payload.bound for r in laws.rows if r.payload.bound.coordinate == row.coordinate
        )
        _, event_reasons = bound_event(bound, row)
        reasons = set(event_reasons)
        evaluable = row.evaluable and audit.evaluable and len(prefix_values) == 2
        unsafe = (
            (measured.guard is not None and measured.guard.unsafe)
            or (row.evaluable and (not row.prefix_safe or not row.action_safe))
            or any(v > D("356.2") for v in prefix_values)
        )
        if not measured.owner_delivered:
            reasons.add("OWNER_DELIVERY_FAILED")
        if row.evaluable and (
            min(row.effects_K) < request.request_K
            or max(row.applied_masses_kg) > request.budget_kg + D("1e-10")
        ):
            reasons.add("REQUEST_OR_FEED_BUDGET_FAILED")
        if (
            not unit.admitted
            or not unit.task_success
            or unit.probe_adequacy is not GateStatus.PASS
            or unit.probe_coverage is not GateStatus.PASS
        ):
            reasons.add("PREPARED_CONTROLLER_USE_OWNER_DID_NOT_VALIDATE_USE")
        if not evaluable:
            reasons.add("MISSING_MANDATORY_EVIDENCE")
        success = not reasons and not unsafe
        results.append(
            FrontierUseResult(
                request,
                True,
                evaluable,
                success,
                unsafe,
                evaluable and not success,
                tuple(sorted(reasons)),
                CanonicalRecordArchive.pack(revealed.reveal_id, revealed),
                unit,
            )
        )
    return FrontierRevealedRoot(
        native.root, ObjectIdentity.from_record(sealed.record_id, sealed), tuple(results)
    )
