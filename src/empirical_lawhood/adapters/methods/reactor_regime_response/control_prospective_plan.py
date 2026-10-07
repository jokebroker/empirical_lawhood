"Freeze the 64-root prepared controller-use census after D preparation, before actions."

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar

from empirical_lawhood.adapters.simulators.reactor_causal_response.design import draw_scenario
from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_design import ReactorAssignedScenario
from empirical_lawhood.kernel.admission import AdmissionGateKind
from empirical_lawhood.kernel.causal_contracts import (
    PredicateDirection,
    ReceiverInterval,
    TemporalPredicateSemantics,
)
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.time import ClockCoordinate, CoordinateOrigin
from empirical_lawhood.planning.controller_study import EvaluatorBoundarySpec, ImplementationRole
from empirical_lawhood.planning.evidence_geometry import GatePredicateKind, GatePredicateSpec
from empirical_lawhood.planning.finite_response_geometry import FiniteReadoutKind, FiniteResponseCoordinate
from empirical_lawhood.planning.nested_controller_evaluation import PreparedFutureRole, PreparedFutureSlot, CoupledRealizationControllerEvaluationPlan, PreparedInterfaceReducerRegistration, PreparedNativeRealizationCoupling, PreparedPolicyAssignment, PreparedReadoutTolerance, PreparedRootAssignment, PreparedRootHoldWord
from empirical_lawhood.runtime.controller_evaluation_nested import RevealedPreparedFuture, SealedPreparedFutureLocator

from .config import ROOTS, ReactorRegimeResponseDesign
from .control_clock import LOCAL_CLOCK, LOCAL_FRAME, ReactorLocalClockMap
from .control_math import REQUESTS_K
from .control_owner import frozen_consumer
from .control_plan import regime_control_law_context
from .control_services import implementation_payloads
from .law_terminal import RegimeJointLawResult
from .prospective_decision import CausalValidityRegimeAssignment, CausalValidityRegimeDecision
from .records import RegimeCausalPreparation
from .science import RECEIVERS


@dataclass(frozen=True, slots=True)
class ReactorRegimeResponseProspectiveStatisticsSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/reactor-regime-response-prospective-statistics-spec'

    spec_id: str = 'reactor-regime-response-four-request-prospective-evaluation-statistics'
    assigned_root_count: int = 64
    requests_K: tuple[D, ...] = tuple(D(repr(value)) for value in REQUESTS_K)
    joined_lower_floor: D = D(".70")
    false_admission_upper_ceiling: D = D(".10")
    one_sided_confidence: D = D(".95")
    direct_minimum_improvement: D = D(".05")
    direct_alpha: D = D(".05")

    def __post_init__(self) -> None:
        if (
            self.spec_id != 'reactor-regime-response-four-request-prospective-evaluation-statistics'
            or self.assigned_root_count != 64
            or self.requests_K != tuple(D(repr(value)) for value in REQUESTS_K)
            or self.joined_lower_floor != D(".70")
            or self.false_admission_upper_ceiling != D(".10")
            or self.one_sided_confidence != D(".95")
            or self.direct_minimum_improvement != D(".05")
            or self.direct_alpha != D(".05")
        ):
            raise ValueError("D statistics changed the frozen all-assigned use criterion")


@dataclass(frozen=True, slots=True)
class ReactorRegimeResponseProspectiveAcquisitionPrecommitment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/reactor-regime-response-prospective-acquisition-precommitment'

    commitment_id: str
    root: str
    role: PreparedFutureRole
    request_index: int | None
    word_index: int | None
    scenario: ObjectIdentity
    views: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.root not in {root for root, role, _, _ in ROOTS if role == "prospective"}
            or self.views != ("reactor-native", "reactor-refined")
            or (
                self.role is PreparedFutureRole.COMMITTED_TASK
                and (self.request_index not in range(4) or self.word_index is not None)
            )
            or (
                self.role is not PreparedFutureRole.COMMITTED_TASK
                and (self.request_index is not None or self.word_index not in (0, 1, 2))
            )
        ):
            raise ValueError("D acquisition precommitment changes its fixed branch role")


@dataclass(frozen=True, slots=True)
class ReactorRegimeResponseProspectiveAssignedRoot(CanonicalRecord):
    """All-assigned denominator, including roots with no safe D contact."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/reactor-regime-response-prospective-assigned-root'

    root: str
    seed: int
    scenario: ObjectIdentity
    eligible: bool
    decision: ObjectIdentity | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            (self.root, self.seed)
            not in {(root, seed) for root, role, _, seed in ROOTS if role == "prospective"}
            or self.scenario.object_schema != ReactorAssignedScenario.SCHEMA
            or (self.decision is not None and self.decision.object_schema != CausalValidityRegimeDecision.SCHEMA)
            or (self.eligible and (self.decision is None or self.reason_codes))
            or (not self.eligible and not self.reason_codes)
        ):
            raise ValueError("D assigned-root record changes its honest contact census")


@dataclass(frozen=True, slots=True)
class ReactorRegimeResponsePreparedProspectivePlanBundle(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/reactor-regime-response-prepared-prospective-plan-bundle'

    plan: CoupledRealizationControllerEvaluationPlan | None
    statistics: ReactorRegimeResponseProspectiveStatisticsSpec
    assigned_roots: tuple[ReactorRegimeResponseProspectiveAssignedRoot, ...]
    acquisitions: tuple[ReactorRegimeResponseProspectiveAcquisitionPrecommitment, ...]
    clock_maps: tuple[ReactorLocalClockMap, ...]
    scenarios: tuple[ReactorAssignedScenario, ...]

    def __post_init__(self) -> None:
        expected = tuple(root for root, role, _, _ in ROOTS if role == "prospective")
        eligible = tuple(row.root for row in self.assigned_roots if row.eligible)
        if (
            self.statistics != ReactorRegimeResponseProspectiveStatisticsSpec()
            or tuple(row.root for row in self.assigned_roots) != expected
            or tuple(row.unit_id for row in self.scenarios) != expected
            or len(self.acquisitions) != 7 * len(eligible)
            or len(self.clock_maps) != len(eligible)
            or (self.plan is None) != (not eligible)
            or self.plan is not None
            and tuple(root.root_id for root in self.plan.roots) != eligible
        ):
            raise ValueError("D prepared controller-use bundle changes its 64-root and native call census")


def _local_coordinate(value: D) -> ClockCoordinate:
    return ClockCoordinate(
        LOCAL_CLOCK, value, "s", LOCAL_FRAME, CoordinateOrigin.EPISODE_RELATIVE
    )


def _prospective_predicate(stem: str, kind: AdmissionGateKind, quantity: str, start: D, end: D, evaluator) -> GatePredicateSpec:  # type: ignore[no-untyped-def]
    return GatePredicateSpec(
        f"{stem}.{kind.value.lower()}",
        kind,
        RECEIVERS[0],
        PredicateDirection.AT_MOST,
        quantity,
        ReceiverInterval(_local_coordinate(start), _local_coordinate(end)),
        GatePredicateKind.SCALAR_AT_MOST,
        None,
        NamedDecimal("absolute-reactor-temperature-limit", D("356.2"), "K"),
        None,
        None,
        TemporalPredicateSemantics.ALWAYS_PRESERVED_PATH
        if kind is AdmissionGateKind.BASELINE_PRESERVATION else None,
        (f"{stem}.native-temperature-and-validity",),
        evaluator,
    )


def build_prepared_prospective_plan(
    *,
    design: ReactorRegimeResponseDesign,
    report: RegimeJointLawResult,
    prepared: tuple[
        tuple[
            RegimeCausalPreparation,
            CausalValidityRegimeDecision | CausalValidityRegimeAssignment | None,
        ], ...
    ],
    producer: ObjectIdentity,
    resource: ObjectIdentity,
    authority: ObjectIdentity,
) -> ReactorRegimeResponsePreparedProspectivePlanBundle:
    """All 64 roots and four nested requests, with one scenario realization/root."""
    assigned = tuple((root, seed) for root, role, _, seed in ROOTS if role == "prospective")
    if (
        design != ReactorRegimeResponseDesign()
        or tuple(causal.root for causal, _ in prepared) != tuple(root for root, _ in assigned)
        or any(
            causal.seed != seed or causal.role != "prospective"
            or (decision is not None and decision.root != causal.root)
            or (
                isinstance(decision, CausalValidityRegimeAssignment)
                and decision.causal_preparation
                != ObjectIdentity.from_record(f"{causal.root}.causal-preparation", causal)
            )
            for (causal, decision), (_, seed) in zip(prepared, assigned, strict=True)
        )
    ):
        raise ValueError("Controller-use plan lacks the exact 64-root D preparation census")
    consumer = frozen_consumer(report)
    consumer_id = ObjectIdentity.from_record(consumer.consumer_id, consumer)
    prospective_evaluation_binding = next(
        binding for binding, _ in implementation_payloads()
        if binding.role is ImplementationRole.OUTCOME_EVALUATOR
    )
    views = report.family.axis_map.qualification_view_ids
    # The local receiver chart and readout identities are invariant across roots.
    coordinates = tuple(sorted((
        FiniteResponseCoordinate(
            receiver, receiver, receiver, consumer.command_expansion,
            FiniteReadoutKind.WINDOW_MAXIMUM if index == 0 else FiniteReadoutKind.ENDPOINT,
            "K", LOCAL_FRAME, LOCAL_CLOCK,
            D(0) if index == 0 else D(10), D(10),
        )
        for index, receiver in enumerate(RECEIVERS)
    ), key=lambda value: value.coordinate_id))
    tolerances = tuple(sorted((
        PreparedReadoutTolerance(
            f"reactor-regime-tolerance.{coordinate.coordinate_id}",
            coordinate,
            D(".01") if coordinate.quantity_id == RECEIVERS[0] else D(".000001"),
            D(".26") if coordinate.quantity_id == RECEIVERS[0] else D(".005051"),
        )
        for coordinate in coordinates
    ), key=lambda value: value.tolerance_id))
    parent = _prospective_predicate(
        'reactor-regime-response-prospective-evaluation.parent', AdmissionGateKind.PHYSICAL_SINK,
        "selected-prefix-maximum-temperature", D(0), D(0), prospective_evaluation_binding.reference,
    )
    future = _prospective_predicate(
        'reactor-regime-response-prospective-evaluation.future', AdmissionGateKind.BASELINE_PRESERVATION,
        "local-branch-maximum-temperature", D(0), D(10), prospective_evaluation_binding.reference,
    )
    boundary = EvaluatorBoundarySpec(
        'reactor-regime-response-prospective-evaluation-boundary',
        prospective_evaluation_binding.binding_id,
        "reactor-native-outcome-custodian",
        "reactor-separate-reveal-authority",
        "reactor-prepared-controller-evaluator",
        SealedPreparedFutureLocator.SCHEMA,
        RevealedPreparedFuture.SCHEMA,
        tuple(sorted((parent.quantity_id, future.quantity_id))),
        tuple(sorted((parent, future), key=lambda value: value.predicate_id)),
    )
    statistics = ReactorRegimeResponseProspectiveStatisticsSpec()
    reducer = PreparedInterfaceReducerRegistration(
        'reactor-regime-response-prospective-evaluation-reducer',
        prospective_evaluation_binding.reference.capability_key,
        prospective_evaluation_binding.reference.capability_version,
        prospective_evaluation_binding.config_sha256,
        prospective_evaluation_binding.implementation_sha256,
        ("NUMERICAL_VIEW", "FUTURE_ROLE", "ROOT_POLICY", "ROOT"),
        ObjectIdentity.from_record(statistics.spec_id, statistics),
    )
    roots = []
    policies = tuple(
        PreparedPolicyAssignment(
            f"reactor-regime-request-{index}",
            report.payload.nomination,
            consumer_id,
            ("assigned-scenario",),
        )
        for index in range(4)
    )
    futures = []
    commitments = []
    couplings = []
    holds = []
    clocks = []
    scenarios = []
    assigned_roots = []
    audit_words = {}
    for (causal, assigned_decision), (root, seed) in zip(prepared, assigned, strict=True):
        decision = (
            assigned_decision.decision
            if isinstance(assigned_decision, CausalValidityRegimeAssignment)
            else assigned_decision
        )
        scenario = draw_scenario(root, "heldout", seed)
        scenarios.append(scenario)
        scenario_id = ObjectIdentity.from_record(scenario.unit_id, scenario)
        eligible = decision is not None and decision.causal_preparation_valid
        reasons: tuple[str, ...]
        if eligible:
            reasons = ()
        elif decision is not None:
            reasons = decision.causal_reasons
        elif isinstance(assigned_decision, CausalValidityRegimeAssignment):
            reasons = assigned_decision.nonentry_reasons
        else:
            reasons = ("SELECTED_PREPARATION_NO_CONTACT",)
        assigned_roots.append(ReactorRegimeResponseProspectiveAssignedRoot(
            root,
            seed,
            scenario_id,
            eligible,
            None if decision is None else ObjectIdentity.from_record(decision.decision_id, decision),
            reasons,
        ))
        if not eligible:
            continue
        assert decision is not None
        stream = f"reactor-scenario.{root}.{seed}"
        roots.append(PreparedRootAssignment(root, decision.route, "paired-cooling-four-request", stream))
        coupling = PreparedNativeRealizationCoupling(
            f"{root}.native-realization", root, stream, scenario_id
        )
        coupling_id = ObjectIdentity.from_record(coupling.coupling_id, coupling)
        couplings.append(coupling)
        context = regime_control_law_context(
            report=report, causal=causal, decision=decision,
            request_K=D(repr(REQUESTS_K[0])),
            producer=producer, resource=resource, authority=authority,
        )
        words = context.word_maps
        holds.append(PreparedRootHoldWord(root, words[0].scalar_word))
        clocks.append(ReactorLocalClockMap(
            f"{root}.callback-{decision.callback:04d}.local-clock",
            root, decision.callback, D(decision.callback * 10),
        ))
        for word in words[1:]:
            audit_words[word.scalar_word.word_id] = word.scalar_word
        for index, policy in enumerate(policies):
            for role, ordinal, word_index in (
                (PreparedFutureRole.COMMITTED_TASK, 0, None),
                (PreparedFutureRole.MATCHED_HOLD, 0, 0),
                (PreparedFutureRole.AUDIT_PROBE, 0, 1),
                (PreparedFutureRole.AUDIT_PROBE, 1, 2),
            ):
                acquisition_id = (
                    f"{root}.request-{index}.task"
                    if word_index is None else f"{root}.evaluator-word-{word_index}"
                )
                commitment = ReactorRegimeResponseProspectiveAcquisitionPrecommitment(
                    f"precommitment.{acquisition_id}", root, role,
                    index if word_index is None else None,
                    word_index, scenario_id, views,
                )
                if commitment not in commitments:
                    commitments.append(commitment)
                slot = PreparedFutureSlot(
                    f"{root}.{policy.policy_id}.{role.value.lower()}.{ordinal}",
                    root, policy.policy_id, role, ordinal,
                    acquisition_id, stream,
                    None if word_index is None else ObjectIdentity.from_record(
                        words[word_index].scalar_word.word_id, words[word_index].scalar_word
                    ),
                    ObjectIdentity.from_record(commitment.commitment_id, commitment),
                    coupling_id,
                )
                futures.append(slot)
    plan = None if not roots else CoupledRealizationControllerEvaluationPlan(
        'reactor-regime-response-four-request-prospective-evaluation',
        tuple(sorted(roots, key=lambda value: value.root_id)),
        policies,
        tuple(sorted(futures, key=lambda value: value.slot_id)),
        views,
        tolerances,
        tuple(sorted(audit_words.values(), key=lambda value: value.word_id)),
        2,
        report.payload.nomination,
        holds[0].word,
        consumer.recipe,
        ("assigned-scenario",),
        ("measured-local-chart",),
        boundary,
        (parent.predicate_id,),
        (future.predicate_id,),
        reducer,
        "reactor-regime-selected-preaction-callback",
        OutcomeAccess.EVALUATION_SEALED,
        VisibilityCeiling.PROSPECTIVE,
        EvidenceCeiling.CONTROLLER_USE,
        tuple(sorted(couplings, key=lambda value: value.root_id)),
        tuple(sorted(holds, key=lambda value: value.root_id)),
    )
    return ReactorRegimeResponsePreparedProspectivePlanBundle(
        plan,
        statistics,
        tuple(sorted(assigned_roots, key=lambda value: value.root)),
        tuple(sorted(commitments, key=lambda value: value.commitment_id)),
        tuple(sorted(clocks, key=lambda value: value.map_id)),
        tuple(sorted(scenarios, key=lambda value: value.unit_id)),
    )
