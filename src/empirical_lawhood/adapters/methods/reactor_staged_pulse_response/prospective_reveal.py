"Separate reveal to prepared controller-use owners, followed by the bounded sequence join."

from dataclasses import dataclass
from decimal import Decimal as D
from typing import Any, ClassVar

from empirical_lawhood.adapters.simulators.reactor_staged_pulse_response.prospective import ClassicalNativeRoot, ClassicalNativeStage, ClassicalNativeUse
from empirical_lawhood.adapters.simulators.reactor_prefix_response.finite_trace import measured_trace
from empirical_lawhood.kernel.admission import GateStatus
from empirical_lawhood.kernel.control import ScientificCommitmentKind
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.controller_study import ImplementationRole
from empirical_lawhood.planning.nested_controller_evaluation import PreparedFutureRole
from empirical_lawhood.runtime.canonical_record_archive import CanonicalRecordArchive
from empirical_lawhood.runtime.controller_evaluation_nested import NestedControllerUseEvaluator, PreparedFutureDisposition, PreparedNativeReadout, PreparedPolicyUnitEvaluation, RevealedPreparedForecastPolicyBundle, RevealedPreparedFuture
from empirical_lawhood.runtime.controller_evaluation_sequence import BoundedSequenceUnitEvaluation, RevealedBoundedSequence, SequenceChildEvidence, SequenceEpisodeMeasurement
from .discovery import physical_service
from .measurement import _window
from .prospective_measure import actual_window, reference_window
from .prospective_plan import ReactorStagedPulseResponseProspectivePlan
from .prospective_seal import ClassicalSealedRoot
from .prediction import predicted_event


@dataclass(frozen=True, slots=True)
class ClassicalRevealedStage(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-staged-pulse-response/classical-revealed-stage'
    frozen: ObjectIdentity
    revealed: CanonicalRecordArchive | None
    unit: PreparedPolicyUnitEvaluation | None
    evaluable: bool
    success: bool
    false_admission: bool | None
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            (self.revealed is None) != (self.unit is None)
            or (
                self.unit is not None
                and self.revealed is not None
                and self.revealed.subject != self.unit.revealed_bundle
            )
            or (
                self.success
                and (
                    not self.evaluable
                    or self.false_admission is not False
                    or self.unit is None
                    or not self.unit.task_success
                )
            )
        ):
            raise ValueError("stage result substitutes its exact prepared owner or unknown outcome")


@dataclass(frozen=True, slots=True)
class ClassicalUseResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-staged-pulse-response/classical-use-result'
    policy: str
    request_id: str
    admitted: bool
    evaluable: bool
    strict_success: bool
    common_service: bool
    false_admission: bool | None
    false_task: bool | None
    unsafe: bool
    attempted_mass_kg: D | None
    stages: tuple[ClassicalRevealedStage, ...]
    sequence: BoundedSequenceUnitEvaluation | None
    sequence_bundle: CanonicalRecordArchive | None
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not 1 <= len(self.stages) <= 2
            or (self.evaluable and (self.false_admission is None or self.false_task is None))
            or (
                self.strict_success
                and (
                    not self.admitted
                    or not self.evaluable
                    or self.false_admission is not False
                    or self.unsafe
                )
            )
            or (self.sequence is None) != (self.sequence_bundle is None)
            or (
                self.sequence is not None
                and self.sequence_bundle is not None
                and self.sequence.bundle != self.sequence_bundle.subject
            )
        ):
            raise ValueError(
                "use result changes its strict/common event or known/unknown failure denominator"
            )


@dataclass(frozen=True, slots=True)
class ClassicalRevealedRoot(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-staged-pulse-response/classical-revealed-root'
    root: str
    sealed: ObjectIdentity
    uses: tuple[ClassicalUseResult, ...]

    @property
    def record_id(self) -> str:
        return f"{self.root}.revealed-outcomes"


def reveal_stage(
    sealed: ClassicalSealedRoot,
    native: ClassicalNativeRoot,
    actual: ClassicalNativeStage,
    authority: ObjectIdentity,
) -> ClassicalRevealedStage:
    frozen = actual.unpack()
    if frozen.lock is None:
        known = "LOCAL_LAW_OR_CAUSAL_PREREQUISITE_NONENTRY" in frozen.reasons
        return ClassicalRevealedStage(
            actual.frozen.subject,
            None,
            None,
            known,
            False,
            False if known else None,
            frozen.reasons,
        )
    assert frozen.compiled is not None and frozen.predicted is not None
    bundle = sealed.bundle_for(actual.frozen.subject)
    evaluation = frozen.lock.evaluation_for(frozen.compiled)
    measured = next(m for m in sealed.measured.stages if m.frozen == actual.frozen.subject)
    row = measured.observation
    outcomes = []
    for locator in bundle.locators:
        slot = next(s for s in evaluation.futures if s.slot_id == locator.slot_id)
        index = evaluation.numerical_view_ids.index(locator.view_id)
        reference = slot.role is PreparedFutureRole.MATCHED_HOLD
        readouts: tuple[PreparedNativeReadout, ...] = ()
        preservation: tuple[NamedDecimal, ...] = ()
        trace = None
        if locator.disposition is PreparedFutureDisposition.COMPLETED:
            if reference:
                reference_raw = reference_window(native, actual, bool(index))
                decoded = None if reference_raw is None else _window(reference_raw)
                if decoded is None:
                    raise ValueError("completed reference lacks its paired native window")
                peak = D(repr(float(decoded[0]["grid"][:, 1].max())))
                values = {
                    t.coordinate.quantity_id: peak
                    if t.coordinate.quantity_id.startswith("s")
                    else D(0)
                    for t in evaluation.readout_tolerances
                }
            else:
                assert row is not None and row.evaluable
                values = {v.value_id: v.value for v in row.values[index]}
            readouts = tuple(
                sorted(
                    (
                        PreparedNativeReadout(
                            t.coordinate,
                            min(
                                values[t.coordinate.quantity_id],
                                frozen.predicted.interval(t.coordinate.quantity_id)[0],
                            )
                            if not reference
                            and t.coordinate.quantity_id in frozen.predicted.saturated_receivers
                            else values[t.coordinate.quantity_id],
                            D(0),
                        )
                        for t in evaluation.readout_tolerances
                    ),
                    key=lambda r: r.coordinate.coordinate_id,
                )
            )
            if not reference:
                assert row is not None
                raw_values = {
                    "local-action-guard-maximum-temperature": values["s2"]
                    if frozen.stage_kind == "joint"
                    else values["s"],
                    "raw-numerics-valid": D(int(row.numerical_valid)),
                    "native-mass-valid": D(
                        int(
                            row.delivery_valid
                            and measured.owner_delivered
                            and measured.reference_valid
                            and frozen.predicted.projected_mass_kg is not None
                            and all(
                                abs(m - frozen.predicted.projected_mass_kg) <= D("1e-12")
                                for m in row.applied_masses_kg
                            )
                        )
                    ),
                    "assigned-budget-valid": D(
                        int(max(row.applied_masses_kg) <= frozen.request.budget_kg + D("1e-10"))
                    ),
                }
                preservation = tuple(
                    sorted(
                        (
                            NamedDecimal(
                                p.quantity_id,
                                raw_values[p.quantity_id],
                                p.upper.unit
                                if p.upper is not None
                                else p.lower.unit
                                if p.lower is not None
                                else "1",
                            )
                            for p in evaluation.evaluator_boundary.outcome_predicates
                            if p.predicate_id in evaluation.future_preservation_predicate_ids
                        ),
                        key=lambda v: v.value_id,
                    )
                )
                trace = None if actual.tick is None else actual.tick.delivery_trace
            else:
                nominal = reference_window(native, actual, False)
                assert nominal is not None
                trace = measured_trace(
                    data={f"reference_{k}": v for k, v in nominal.arrays.unpack().items()},
                    key="reference",
                    word=evaluation.matched_hold_word,
                    implementation=frozen.compiled.implementation(ImplementationRole.DELIVERY),
                    commitment=ObjectIdentity.from_record(slot.slot_id, slot),
                    kind=ScientificCommitmentKind.MEASURED_HOLD,
                    trace_id=f"{locator.locator_id}.actual-reference",
                    guard_s=nominal.duration_s,
                )
                if trace is None:
                    # A known wrong delivery is retained by the raw native gate;
                    # missing delivery evidence remains unknown to the owner.
                    readouts = ()
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
    parent = next(
        p
        for p in evaluation.evaluator_boundary.outcome_predicates
        if p.predicate_id in evaluation.parent_preservation_predicate_ids
    )
    parent_values = (
        ()
        if len(measured.parent_peaks_K) != 2
        else (NamedDecimal(parent.quantity_id, max(measured.parent_peaks_K), "K"),)
    )
    revealed = RevealedPreparedForecastPolicyBundle(
        f"{bundle.bundle_id}.reveal",
        bundle,
        authority,
        tuple(sorted(outcomes, key=lambda o: o.outcome_id)),
        parent_values,
    )
    unit = NestedControllerUseEvaluator(
        bundle.design.evaluator_binding, evaluation.reducer
    ).evaluate_prepared_unit(
        revealed=revealed, compiled=frozen.compiled if bundle.instance is not None else None
    )
    statuses = (unit.native_target, unit.delivery, unit.preservation, unit.numerics)
    evaluable = (
        all(s is not GateStatus.UNEVALUABLE for s in statuses)
        if frozen.admitted
        else actual.tick is not None
    )
    failed = any(s is GateStatus.FAIL for s in statuses) if frozen.admitted else False
    success = unit.task_success and evaluable
    reasons = set(() if success else ("OWNED_CONTROLLER_USE_OBLIGATION_NOT_PASSED",))
    if frozen.admitted and row is not None:
        event, raw_reasons = predicted_event(frozen.predicted, row)
        # The explicit raw gates and selected-law task must agree with the
        # adapter's independently recomputed event; never overwrite the owner.
        if success and not event:
            raise ValueError(f'prepared controller use omitted a declared raw-law conjunct: {raw_reasons}')
        reasons.update(raw_reasons)
    if not evaluable:
        reasons.add("MISSING_MANDATORY_EVIDENCE")
    return ClassicalRevealedStage(
        actual.frozen.subject,
        CanonicalRecordArchive.pack(revealed.reveal_id, revealed),
        unit,
        evaluable,
        success,
        True if failed else False if evaluable else None,
        tuple(sorted(reasons)),
    )


def _sequence(
    *,
    plan: ReactorStagedPulseResponseProspectivePlan,
    native: ClassicalNativeRoot,
    sealed: ClassicalSealedRoot,
    actual: ClassicalNativeUse,
    results: tuple[ClassicalRevealedStage, ...],
    publisher: Any,
) -> tuple[BoundedSequenceUnitEvaluation | None, CanonicalRecordArchive | None]:
    policy_id = f"{actual.policy.lower()}.{actual.request.request_id.lower()}"
    sequence = next(
        (s for s in plan.sequences if s.policy_id == policy_id and native.root in s.roots), None
    )
    if sequence is None or results[0].revealed is None:
        return None, None
    children = []
    for index, (stage, result) in enumerate(zip(actual.stages, results, strict=True)):
        if result.revealed is None:
            break
        frozen = stage.unpack()
        assert frozen.compiled is not None
        if index and stage.tick is None:
            break
        children.append(
            SequenceChildEvidence(
                index,
                result.revealed,
                None
                if stage.tick is None
                else CanonicalRecordArchive.pack(
                    frozen.compiled.compiled_study_id, frozen.compiled
                ),
                None
                if stage.tick is None
                else CanonicalRecordArchive.pack(stage.tick.tick_id, stage.tick),
                None if stage.tick is None else frozen.link,
            )
        )
    episode = None
    if len(children) == 2 and all(s.unpack().admitted for s in actual.stages):
        measured = next(
            m for m in sealed.measured.stages if m.frozen == actual.stages[1].frozen.subject
        )
        row = measured.observation
        if row is not None and row.evaluable:
            windows = tuple(actual_window(actual, actual.stages[1], v) for v in (False, True))
            assert all(w is not None for w in windows)
            artifacts = tuple(
                publisher.publish_record(f"{native.root}.{policy_id}.actual-v{i}", w)
                for i, w in enumerate(windows)
                if w is not None
            )
            refs = tuple(
                publisher.publish_record(
                    f"{native.root}.{policy_id}.{w.branch_id}.v{int(w.refined)}.reference", w
                )
                for w in native.references
            )
            raw = tuple(
                tuple(
                    NamedDecimal(
                        r.receiver_id,
                        row.applied_masses_kg[i]
                        if r.receiver_id == "mass"
                        else row.pair(r.receiver_id)[i],
                        r.unit,
                    )
                    for r in sequence.receivers
                )
                for i in (0, 1)
            )
            first = actual.stages[0].unpack().causal.callback
            assert first is not None
            episode = SequenceEpisodeMeasurement(
                native.root,
                plan.source,
                native.scenario,
                D(first * 10),
                D(240),
                sequence.numerical_views,
                sealed.native_artifact,
                sealed.source_receipt,
                (artifacts[0], artifacts[1]),
                refs,
                (raw[0], raw[1]),
                tuple((r.receiver_id, r.observation_operator) for r in sequence.receivers),
            )
    bundle = RevealedBoundedSequence(
        f"{native.root}.{policy_id}.sequence-reveal",
        ObjectIdentity.from_record(sequence.evaluation_plan_id, sequence),
        native.root,
        native.scenario,
        tuple(children),
        episode,
    )
    joined = NestedControllerUseEvaluator(
        sequence.evaluator, sequence.reducer
    ).evaluate_sequence_unit(plan=sequence, bundle=bundle)
    return joined, CanonicalRecordArchive.pack(bundle.bundle_id, bundle)


def reveal_root(
    *,
    sealed: ClassicalSealedRoot,
    native: ClassicalNativeRoot,
    plan: ReactorStagedPulseResponseProspectivePlan,
    reveal_authority: ObjectIdentity,
    publisher: Any,
) -> ClassicalRevealedRoot:
    if sealed.native != ObjectIdentity.from_record(
        native.record_id, native
    ) or sealed.plan != ObjectIdentity.from_record(plan.record_id, plan):
        raise ValueError("prospective reveal substitutes its exact sealed native result")
    uses = []
    for actual in native.uses:
        results = tuple(
            reveal_stage(sealed, native, stage, reveal_authority) for stage in actual.stages
        )
        sequence, archive = (
            _sequence(
                plan=plan,
                native=native,
                sealed=sealed,
                actual=actual,
                results=results,
                publisher=publisher,
            )
            if plan.block == "staged-sequence-comparison" and actual.policy != "ONE_PULSE"
            else (None, None)
        )
        admitted = actual.stages[0].unpack().admitted
        measured = tuple(
            next(m for m in sealed.measured.stages if m.frozen == s.frozen.subject)
            for s in actual.stages
        )
        evaluable = (
            sequence.evaluable
            if sequence is not None
            else all(r.evaluable for r in results)
            and not (admitted and plan.block == "staged-sequence-comparison" and actual.policy != "ONE_PULSE")
        )
        success = (
            sequence.success
            if sequence is not None
            else results[0].success
            if plan.block != "staged-sequence-comparison" or actual.policy == "ONE_PULSE"
            else False
        )
        failed = (
            sequence.false_admission
            if sequence is not None
            else True
            if any(r.false_admission is True for r in results)
            else False
            if evaluable
            else None
        )
        target = measured[-1].observation
        complete_joint = (
            len(measured) == 2 and target is not None and target.coordinate.kind == "joint"
        )
        physical = (
            target is not None
            and measured[-1].owner_delivered
            and physical_service(
                target,
                actual.request.required_K,
                actual.request.budget_kg,
                second=actual.request.second_K
                if plan.block == "staged-sequence-comparison" and actual.policy != "ONE_PULSE"
                else None,
            )
        )
        common = physical and (plan.block != "staged-sequence-comparison" or complete_joint)
        unsafe = (
            actual.known_unsafe
            or any(m.observation is not None and m.observation.unsafe for m in measured)
            or any(v > D("356.2") for m in measured for v in m.parent_peaks_K)
        )
        if admitted and unsafe:
            failed = True
            success = False
        mass = (
            sum((D(10) * d.applied_feed_kg_s for s in actual.stages for d in s.deliveries), D(0))
            if not admitted or all(s.tick is not None for s in actual.stages)
            else None
        )
        reasons = set(r for result in results for r in result.reasons)
        if sequence is not None:
            reasons.update(sequence.reason_codes)
        false_task = bool(admitted and not physical) if evaluable else True if unsafe else None
        uses.append(
            ClassicalUseResult(
                actual.policy,
                actual.request.request_id,
                admitted,
                evaluable,
                success,
                bool(common),
                failed,
                false_task,
                unsafe,
                mass,
                results,
                sequence,
                archive,
                tuple(sorted(reasons)),
            )
        )
    return ClassicalRevealedRoot(
        native.root, ObjectIdentity.from_record(sealed.record_id, sealed), tuple(uses)
    )
