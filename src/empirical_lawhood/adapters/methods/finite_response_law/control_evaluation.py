"Bind finite response-law evaluation consumer use to the existing prepared controller-use owner.\n\nBoth independent future purposes remain labelled coordinates. Native acquisition\nidentities live in the explicit bundle map; six logical consumers do not create\nsix physical panels. Full-menu prediction information is a separate endpoint,\nnot the prepared owner's optional random-probe adequacy statistic.\n"

from dataclasses import dataclass, replace
from decimal import Decimal as D
from typing import ClassVar

from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord, ActionWordSupportStatus
from empirical_lawhood.kernel.admission import AdmissionGateKind
from empirical_lawhood.kernel.causal_contracts import PredicateDirection, ReceiverInterval
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.controller_study import EvaluatorBoundarySpec, ImplementationBinding, ImplementationRole
from empirical_lawhood.planning.evidence_geometry import GatePredicateKind, GatePredicateSpec
from empirical_lawhood.planning.finite_response_geometry import FiniteReadoutKind, FiniteResponseCoordinate
from empirical_lawhood.planning.nested_controller_evaluation import PreparedFutureRole, PreparedFutureSlot, PreparedRootAssignment, PreparedPolicyAssignment, PreparedReadoutTolerance, PreparedInterfaceReducerRegistration, CommonStartControllerEvaluationPlan
from empirical_lawhood.runtime.controller_evaluation_nested import SealedPreparedFutureLocator, RevealedPreparedFuture
from empirical_lawhood.adapters.simulators.finite_response_law.evaluation_contracts import FiniteResponseLawEvaluationConfig, FiniteResponseLawEvaluationRoot, FiniteResponseLawEvaluationInvocation
from empirical_lawhood.adapters.simulators.finite_response_law.assigned_contracts import FiniteResponseLawAssignedEvaluationConfig, FiniteResponseLawAssignedEvaluationInvocation, FiniteResponseLawAssignedEvaluationRoot
from .consumer import SPEC, NAMES, FiniteResponseLawNativeUnitReadoutMap
from .control_plan import clock
from .law_binding import NATIVE_WORDS, native_action_word, output_quantities

POLICIES = tuple(
    sorted(f"{b}.consumer-{c}" for b in ("composed", "cached", "direct") for c in (0, 1))
)


def hold_delivery_coordinate(word: OccurrenceActionWord) -> FiniteResponseCoordinate:
    """Actual zero-force delivery, never a fabricated HOLD differential response.

    Both purpose traces must be exact before this observation is available.
    The generic HOLD reference's numerical check is therefore redundant with
    exact delivery; it imposes no additional response-convergence requirement.
    """
    event = word.occurrences[0].realized
    return FiniteResponseCoordinate(
        "matched-hold.realized-native-force",
        event.quantity_id,
        "finite-response-law.native-force-trace",
        ObjectIdentity.from_record("flh-science", SPEC),
        FiniteReadoutKind.EARLY,
        event.native_unit,
        event.native_action_frame,
        event.coordinate.clock_id,
        event.coordinate.coordinate,
        event.coordinate.coordinate,
    )


def measured_hold_word(denominator_id: str, history_id: str) -> OccurrenceActionWord:
    """Declare measured zero forcing; never a qualified controller fallback."""
    pulse = native_action_word(
        NATIVE_WORDS[0], denominator_id=denominator_id, history_id=history_id
    )
    stem = "finite-response-law.prospective-control.measured-hold"
    occurrence = pulse.occurrences[0]
    zero = replace(
        occurrence,
        occurrence_id=f"{stem}.interval.0",
        requested=replace(occurrence.requested, value=D(0)),
        accepted=replace(occurrence.accepted, value=D(0)),
        applied=replace(occurrence.applied, value=D(0)),
        realized=replace(occurrence.realized, value=D(0)),
    )
    return replace(
        pulse,
        word_id=stem,
        occurrences=(zero,),
        groups=tuple(
            replace(
                g,
                group_id=f"{stem}.group.{i}",
                members=tuple(replace(m, occurrence_id=zero.occurrence_id) for m in g.members),
            )
            for i, g in enumerate(pulse.groups)
        ),
        support_status=ActionWordSupportStatus.UNEVALUABLE,
        reason_codes=("SUPPORT_MEASURED_REFERENCE_ONLY_NO_QUALIFIED_HOLD_FIBRE",),
    )


@dataclass(frozen=True, slots=True)
class FiniteResponseLawEvaluationAcquisitionBundle(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-response-law/finite-response-law-evaluation-acquisition-bundle'
    source: ObjectIdentity
    root: FiniteResponseLawEvaluationRoot | FiniteResponseLawAssignedEvaluationRoot
    role: PreparedFutureRole
    invocations: tuple[FiniteResponseLawEvaluationInvocation | FiniteResponseLawAssignedEvaluationInvocation, ...]

    def __post_init__(self) -> None:
        from empirical_lawhood.adapters.simulators.prepared_response.contracts import PreparedForceWord

        words = (*NATIVE_WORDS, PreparedForceWord(D(0), 0, 0))
        if self.role is PreparedFutureRole.MATCHED_HOLD:
            words = (words[-1],)
        elif self.role is not PreparedFutureRole.COMMITTED_TASK:
            raise ValueError("Finite response-law evaluation use bundles are the full paired panel or actual HOLD reference")
        invocation_type = (
            FiniteResponseLawAssignedEvaluationInvocation
            if type(self.root) is FiniteResponseLawAssignedEvaluationRoot
            else FiniteResponseLawEvaluationInvocation
        )
        expected = tuple(
            sorted(
                (
                    invocation_type(
                        self.source, self.root, "future", self.root.assigned_parent, w, p
                    )
                    for w in words
                    for p in ("future-1", "future-2")
                ),
                key=lambda t: t.task_id,
            )
        )
        if self.invocations != expected:
            raise ValueError(
                "Finite response-law evaluation bundle substitutes its exact native acquisitions or future purposes"
            )

    @property
    def bundle_id(self) -> str:
        return f"{self.root.stage_unit}.{self.role.value.lower()}.native-bundle"


def evaluation_design(
    *,
    source: FiniteResponseLawEvaluationConfig | FiniteResponseLawAssignedEvaluationConfig,
    evaluator: ImplementationBinding,
    readout_map: FiniteResponseLawNativeUnitReadoutMap,
    view_ids: tuple[str, ...],
    hold_word: OccurrenceActionWord,
) -> tuple[
    CommonStartControllerEvaluationPlan, tuple[FiniteResponseLawEvaluationAcquisitionBundle, ...]
]:
    "A use-only controller-use census: 64 roots, six policies, shared finite native panels.\n\n    Optional generic random-probe slots are inapplicable here. The independently\n    declared full eight-word information-loss census remains mandatory outside\n    this use reducer, with its own masks and all-root inference.\n    "
    if evaluator.role is not ImplementationRole.OUTCOME_EVALUATOR or len(view_ids) != 2:
        raise ValueError("Finite response-law evaluation requires its actual controller-use owner and both numerical views")
    stem = "finite-response-law.prospective-control"
    source_id = ObjectIdentity.from_record(source.spec_id, source)
    recipe = ObjectIdentity.from_record("flh-science", SPEC)
    roots = tuple(
        PreparedRootAssignment(
            r.stage_unit,
            r.assigned_parent,
            "finite-native-response",
            f"{r.stage_unit}.prefix-stream",
        )
        for r in source.roots
    )
    policies = tuple(
        PreparedPolicyAssignment(p, source_id, recipe, ("assigned-parent", "primary-prefix"))
        for p in POLICIES
    )
    bundles = []
    futures = []
    for root in source.roots:
        for role in (PreparedFutureRole.COMMITTED_TASK, PreparedFutureRole.MATCHED_HOLD):
            words = (
                source.words
                if role is PreparedFutureRole.COMMITTED_TASK
                else tuple(w for w in source.words if w.sign == 0)
            )
            bundle = FiniteResponseLawEvaluationAcquisitionBundle(
                source_id,
                root,
                role,
                tuple(
                    sorted(
                        (
                            (
                                FiniteResponseLawAssignedEvaluationInvocation
                                if type(source) is FiniteResponseLawAssignedEvaluationConfig
                                else FiniteResponseLawEvaluationInvocation
                            )(
                                source_id, root, "future", root.assigned_parent, w, purpose
                            )
                            for w in words
                            for purpose in ("future-1", "future-2")
                        ),
                        key=lambda t: t.task_id,
                    )
                ),
            )
            bundles.append(bundle)
            for policy in policies:
                futures.append(
                    PreparedFutureSlot(
                        f"{root.stage_unit}.{policy.policy_id}.{role.value.lower()}",
                        root.stage_unit,
                        policy.policy_id,
                        role,
                        0,
                        bundle.bundle_id,
                        f"{root.stage_unit}.two-future-stream-bundle",
                        ObjectIdentity.from_record(hold_word.word_id, hold_word)
                        if role is PreparedFutureRole.MATCHED_HOLD
                        else None,
                        ObjectIdentity.from_record(bundle.bundle_id, bundle),
                        ObjectIdentity.from_record(root.root_id, root),
                    )
                )
    predicates = tuple(
        GatePredicateSpec(
            f"{stem}.{name}",
            AdmissionGateKind.EFFORT
            if name == "parent-work"
            else AdmissionGateKind.OBSERVATION_VALIDITY,
            output_quantities()[0].quantity_id,
            PredicateDirection.AT_LEAST,
            f"{stem}.{name}-margin",
            ReceiverInterval(clock(start), clock(end)),
            GatePredicateKind.SCALAR_AT_LEAST,
            NamedDecimal("zero", D(0), "1"),
            None,
            None,
            None,
            None,
            (f"{stem}.{name}-constraint",),
            evaluator.reference,
        )
        for name, start, end in (("future-validity", 4368, 4560), ("parent-work", 4096, 4368))
    )
    boundary = EvaluatorBoundarySpec(
        f"{stem}.evaluator-boundary",
        evaluator.binding_id,
        f"{stem}.outcome-custodian",
        f"{stem}.reveal-authority",
        f"{stem}.evaluator",
        SealedPreparedFutureLocator.SCHEMA,
        RevealedPreparedFuture.SCHEMA,
        tuple(p.quantity_id for p in predicates),
        predicates,
    )
    reducer = PreparedInterfaceReducerRegistration(
        f"{stem}.reducer",
        evaluator.reference.capability_key,
        evaluator.reference.capability_version,
        evaluator.config_sha256,
        evaluator.implementation_sha256,
        ("NUMERICAL_VIEW", "FUTURE_ROLE", "ROOT_POLICY", "ROOT"),
        recipe,
    )
    plan = CommonStartControllerEvaluationPlan(
        f"{stem}.evaluation-plan",
        roots,
        policies,
        tuple(sorted(futures, key=lambda f: f.slot_id)),
        view_ids,
        tuple(
            PreparedReadoutTolerance(
                f"{c.coordinate_id}.tolerance",
                c,
                SPEC.delta[NAMES.index(c.coordinate_id.split(".", 1)[1])] / 8,
                SPEC.delta[NAMES.index(c.coordinate_id.split(".", 1)[1])],
            )
            for c in readout_map.coordinates
        )
        + (
            PreparedReadoutTolerance(
                "matched-hold.realized-native-force.tolerance",
                hold_delivery_coordinate(hold_word),
                # One native force unit is only a positive codec bound. Exact
                # zero-forcing trace equality is required independently; no
                # nonzero command can pass through this numerical tolerance.
                D(1),
                D(1),
            ),
        ),
        (),
        0,
        None,
        hold_word,
        recipe,
        ("assigned-parent", "primary-prefix"),
        ("consumer-a-request", "consumer-b-request"),
        boundary,
        (f"{stem}.parent-work",),
        (f"{stem}.future-validity",),
        reducer,
        f"{stem}.pre-parent",
        OutcomeAccess.EVALUATION_SEALED,
        VisibilityCeiling.PROSPECTIVE,
        EvidenceCeiling.CONTROLLER_USE,
    )
    return plan, tuple(sorted(bundles, key=lambda b: b.bundle_id))
