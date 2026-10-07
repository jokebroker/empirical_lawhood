"""Pre-t0 complete prepared-owner plans, with no additional native audit probes."""

from dataclasses import dataclass, replace
from decimal import Decimal as D
from functools import lru_cache
from typing import ClassVar

from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.causal import CausalOperands, causal_operands
from empirical_lawhood.adapters.simulators.reactor_staged_pulse_response.words import project_pulse
from empirical_lawhood.adapters.simulators.reactor_causal_response.design import draw_scenario
from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_design import ReactorBatchSource
from empirical_lawhood.adapters.simulators.reactor_causal_response.delivery import coordinate as native_coordinate
from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord
from empirical_lawhood.kernel.admission import AdmissionGateKind
from empirical_lawhood.kernel.causal_contracts import (
    PredicateDirection,
    ReceiverInterval,
    TemporalPredicateSemantics,
)
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.time import ClockCoordinate, CoordinateOrigin
from empirical_lawhood.planning.controller_study import EvaluatorBoundarySpec, ImplementationRole
from empirical_lawhood.planning.bounded_sequence_evaluation import BoundedSequenceEvaluationPlan, SequenceReceiverLimit, SequenceStageContract
from empirical_lawhood.planning.evidence_geometry import GatePredicateKind, GatePredicateSpec
from empirical_lawhood.planning.nested_controller_evaluation import PreparedFutureRole, PreparedFutureSlot, CoupledRealizationControllerEvaluationPlan, PreparedInterfaceReducerRegistration, PreparedNativeRealizationCoupling, PreparedPolicyAssignment, PreparedReadoutTolerance, PreparedRootAssignment, PreparedRootHoldWord
from empirical_lawhood.runtime.canonical_record_archive import CanonicalRecordArchive
from empirical_lawhood.runtime.controller_evaluation_nested import RevealedPreparedFuture, SealedPreparedFutureLocator
from .config import ARMS, BASE, FIRST, LOCAL_REQUESTS, MENUS, PAIR_REQUESTS, SECONDS, SEQUENCES, ZERO, ClassicalDesign, Request, assignment, roots
from .control_records import ClassicalReadout, consumer, response_coordinates
from .control_services import implementation_payloads
from .law_terminal import ClassicalLaws
from .nomination import initial_pair_opportunities
from .prediction import HORIZON, ClassicalPredictionSeal
from .records import ClassicalPreparation
from .science import LOCAL_CLOCK, LOCAL_FRAME


@dataclass(frozen=True, slots=True)
class ClassicalAcquisition(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-staged-pulse-response/classical-acquisition'
    root: str
    block: str
    branch: str
    scenario: ObjectIdentity

    def __post_init__(self) -> None:
        if (
            assignment(self.root)[:2] != (self.block, "prospective")
            or self.scenario.object_id != self.root
        ):
            raise ValueError("future acquisition changes its preassigned native realization")

    @property
    def record_id(self) -> str:
        return f"{self.root}.{self.branch}.acquisition"


@dataclass(frozen=True, slots=True)
class ReactorStagedPulseResponseProspectiveUsePlan(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-staged-pulse-response/reactor-staged-pulse-response-prospective-use-plan'
    block: str
    policy: str
    request: Request
    stage_kind: str
    prerequisite_entered: bool
    assigned: tuple[tuple[str, bool, tuple[str, ...]], ...]
    plan: CoupledRealizationControllerEvaluationPlan | None
    acquisitions: tuple[ClassicalAcquisition, ...]

    def __post_init__(self) -> None:
        entered = tuple(root for root, yes, _ in self.assigned if yes)
        if (
            tuple(r for r, _, _ in self.assigned) != roots(self.block, "prospective")
            or (self.plan is None) != (not entered)
            or (
                self.plan is not None
                and (
                    tuple(r.root_id for r in self.plan.roots) != entered
                    or self.plan.audit_probes_per_policy != 0
                )
            )
            or len(self.acquisitions) != 2 * len(entered)
        ):
            raise ValueError(
                "Controller use use plan changes its complete root census or adds undeclared acquisitions"
            )

    @property
    def record_id(self) -> str:
        return f"reactor-staged-pulse-response.{self.block}.{self.policy.lower()}.{self.request.request_id.lower()}.{self.stage_kind}.prospective-evaluation"


@dataclass(frozen=True, slots=True)
class ReactorStagedPulseResponseProspectivePlan(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-staged-pulse-response/reactor-staged-pulse-response-prospective-plan'
    block: str
    laws: ObjectIdentity
    seals: tuple[ObjectIdentity, ...]
    uses: tuple[CanonicalRecordArchive, ...]
    source: ObjectIdentity
    sequences: tuple[BoundedSequenceEvaluationPlan, ...]

    def __post_init__(self) -> None:
        if (
            self.laws.object_schema != ClassicalLaws.SCHEMA
            or len(self.seals) != 64
            or len(self.uses) != {"base-menu-comparison": 16, "expanded-menu-comparison": 8, "staged-sequence-comparison": 10}[self.block]
            or any(a.subject.object_schema != ReactorStagedPulseResponseProspectiveUsePlan.SCHEMA for a in self.uses)
        ):
            raise ValueError("Controller-use plan changes its complete frozen treatment/request/stage census")
        if (
            self.source.object_schema != ReactorBatchSource.SCHEMA
            or (self.block != "staged-sequence-comparison" and self.sequences)
            or len({(s.policy_id, s.request_id) for s in self.sequences}) != len(self.sequences)
            or any(s.source != self.source for s in self.sequences)
        ):
            raise ValueError("sequence plan changes its assigned source or policy census")

    @property
    def record_id(self) -> str:
        return f"reactor-staged-pulse-response.{self.block}.prospective-evaluation-plan"

    def use(self, policy: str, request: Request, stage_kind: str) -> ReactorStagedPulseResponseProspectiveUsePlan:
        key = f"reactor-staged-pulse-response.{self.block}.{policy.lower()}.{request.request_id.lower()}.{stage_kind}.prospective-evaluation"
        archive = next(a for a in self.uses if a.subject.object_id == key)
        return _decode_use(archive)


@lru_cache(maxsize=32)
def _decode_use(archive: CanonicalRecordArchive) -> ReactorStagedPulseResponseProspectiveUsePlan:
    """Reuse the same validated immutable plan across its assigned root reductions."""
    result = decode_canonical_bytes(
        archive.unpack(), ReactorStagedPulseResponseProspectiveUsePlan, maximum_bytes=archive.decoded_bytes
    )
    if ObjectIdentity.from_record(result.record_id, result) != archive.subject:
        raise ValueError("Controller use archive substitutes the exact owner plan")
    return result


def _build_use(
    laws: ClassicalLaws,
    preparations: tuple[ClassicalPreparation, ...],
    seals: tuple[ClassicalPredictionSeal, ...],
    policy: str,
    request: Request,
    kind: str,
    causal_inputs: dict[tuple[str, str], CausalOperands | None],
) -> ReactorStagedPulseResponseProspectiveUsePlan:
    block = laws.operands.nomination.block
    stem = f"reactor-staged-pulse-response.{block}.{policy.lower()}.{request.request_id.lower()}.{kind}.prospective-evaluation"
    readout = ClassicalReadout(
        policy, request, kind, ObjectIdentity.from_record(laws.record_id, laws)
    )
    arm = (
        policy
        if policy in ARMS
        else "EL_SERVICE_MENU"
        if policy in MENUS
        else "EL_SEQUENCE_RELATION"
    )
    reports = tuple(
        r
        for r in laws.rows
        if r.payload.recipe.recipe_id in laws.qualified
        and r.payload.recipe.arm == arm
        and (policy != "BASE" or r.payload.recipe.bound.coordinate.pulse in BASE)
    )
    prerequisite = (
        bool(reports)
        if block != "staged-sequence-comparison"
        else any(
            r.payload.recipe.bound.coordinate.kind
            == ("baseline" if policy == "ONE_PULSE" else "first")
            for r in reports
        )
        and (
            policy == "ONE_PULSE"
            or any(r.payload.recipe.bound.coordinate.kind == "joint" for r in reports)
        )
    )
    relevant = tuple(
        r
        for r in reports
        if r.payload.recipe.bound.coordinate.kind == kind
        and (
            kind != "local"
            or (
                r.payload.recipe.bound.coordinate.context,
                r.payload.recipe.bound.coordinate.horizon_s,
            )
            == (request.context, request.horizon_s)
        )
    )
    if not prerequisite or not relevant:
        # A missing qualified fibre has no predictive consumer or prepared
        # acquisition. Retain every assigned root as a known nonattempt.
        return ReactorStagedPulseResponseProspectiveUsePlan(
            block,
            policy,
            request,
            kind,
            prerequisite,
            tuple((p.root, False, ("LOCAL_LAW_OR_CAUSAL_PREREQUISITE_NONENTRY",)) for p in preparations),
            None,
            (),
        )
    frozen = consumer(laws, readout)
    binding = next(
        b for b, _ in implementation_payloads() if b.role is ImplementationRole.OUTCOME_EVALUATOR
    )
    tolerances = tuple(
        sorted(
            (
                PreparedReadoutTolerance(
                    f"{stem}.{c.coordinate_id}.tolerance",
                    c,
                    D(".01")
                    if c.quantity_id.startswith("s")
                    else D(".000024")
                    if c.quantity_id == "g"
                    else D(".000001") * (request.horizon_s if kind == "local" else 10) / 10,
                    D(".01") if c.quantity_id.startswith("s") else D(".000001"),
                )
                for c in response_coordinates(readout)
            ),
            key=lambda t: t.tolerance_id,
        )
    )

    def point(t: D) -> ClockCoordinate:
        return ClockCoordinate(LOCAL_CLOCK, t, "s", LOCAL_FRAME, CoordinateOrigin.EPISODE_RELATIVE)

    def predicate(gate: AdmissionGateKind, end: D) -> GatePredicateSpec:
        return GatePredicateSpec(
            f"{stem}.{gate.value.lower()}",
            gate,
            "s",
            PredicateDirection.AT_MOST,
            "prepared-prefix-maximum-temperature"
            if end == 0
            else "local-action-guard-maximum-temperature",
            ReceiverInterval(point(D(0)), point(end)),
            GatePredicateKind.SCALAR_AT_MOST,
            None,
            NamedDecimal("native-reactor-temperature-limit", D("356.2"), "K"),
            None,
            None,
            TemporalPredicateSemantics.ALWAYS_PRESERVED_PATH if end else None,
            (f"{stem}.native-safety",),
            binding.reference,
        )

    parent = predicate(AdmissionGateKind.PHYSICAL_SINK, D(0))
    future = predicate(AdmissionGateKind.BASELINE_PRESERVATION, D(readout.guard_s))
    # Saturation cannot hide a raw discrepancy, changed native word or budget
    # violation. These conjuncts use the existing preservation predicate owner.
    raw_gates = tuple(
        GatePredicateSpec(
            f"{stem}.{name}",
            AdmissionGateKind.BASELINE_PRESERVATION,
            name,
            PredicateDirection.AT_LEAST,
            name,
            ReceiverInterval(point(D(0)), point(D(readout.guard_s))),
            GatePredicateKind.SCALAR_AT_LEAST,
            NamedDecimal(f"{name}.required", D(1), "1"),
            None,
            None,
            None,
            TemporalPredicateSemantics.ALWAYS_PRESERVED_PATH,
            (f"{stem}.native-safety",),
            binding.reference,
        )
        for name in ("raw-numerics-valid", "native-mass-valid", "assigned-budget-valid")
    )
    predicates = (parent, future, *raw_gates)
    boundary = EvaluatorBoundarySpec(
        f"{stem}.boundary",
        binding.binding_id,
        "reactor-native-outcome-custodian",
        "reactor-separate-reveal-authority",
        "reactor-prepared-controller-evaluator",
        SealedPreparedFutureLocator.SCHEMA,
        RevealedPreparedFuture.SCHEMA,
        tuple(sorted(p.quantity_id for p in predicates)),
        tuple(sorted(predicates, key=lambda p: p.predicate_id)),
    )
    design = ClassicalDesign()
    reducer = PreparedInterfaceReducerRegistration(
        "reactor-staged-pulse-response.prepared-reducer",
        binding.reference.capability_key,
        binding.reference.capability_version,
        binding.config_sha256,
        binding.implementation_sha256,
        ("NUMERICAL_VIEW", "FUTURE_ROLE", "ROOT_POLICY", "ROOT"),
        ObjectIdentity.from_record(design.config_id, design),
    )
    policy_id = f"{policy.lower()}.{request.request_id.lower()}"
    allocation = PreparedPolicyAssignment(
        policy_id,
        ObjectIdentity.from_record(f"request.{request.request_id.lower()}", request),
        ObjectIdentity.from_record(frozen.consumer_id, frozen),
        ("assigned-scenario",),
    )
    assignments, entered_roots, holds, futures, couplings, acquisitions = [], [], [], [], [], []
    for preparation, seal in zip(preparations, seals, strict=True):
        root = preparation.root
        ca = next(
            c
            for c in preparation.contexts
            if c.context == (request.context if kind == "local" else "early")
        )
        inputs = causal_inputs[root, ca.context]
        entered = prerequisite and bool(relevant) and inputs is not None
        history_id = ca.record_id
        if kind in ("first", "joint") and inputs is not None:
            first_mapping = project_pulse(
                inputs.observation,
                inputs.previous,
                FIRST,
                decision_id=f"{root}.first-projection",
                history_id=ca.record_id,
                horizon_id=HORIZON,
                receiver_id="c1",
            )
            # This supplies only known actuator time/dose to the all-zero
            # reference word. No predicted temperature becomes an observation.
            at_second = replace(
                inputs.observation,
                time=inputs.observation.time + 120,
                dose=inputs.observation.dose + float(first_mapping.realized_mass_kg),
            )
            previous = (0.0, inputs.previous[1])
            masses = {
                w.word_id: project_pulse(
                    at_second,
                    previous,
                    w,
                    decision_id=f"{root}.{w.word_id}.second-projection",
                    history_id="prospective-induced-history",
                    horizon_id=HORIZON,
                    receiver_id="c2",
                ).applied_mass_kg
                for w in SECONDS
            }
            initial = next(p for p in seal.points if p.context == "early")
            opportunities = initial_pair_opportunities(
                laws.operands.nomination,
                initial,
                laws.qualified,
                request.request_id,
                masses,
                fixed=policy == "FIXED_SEQUENCE",
            )
            entered &= bool(opportunities)
            if kind == "joint":
                inputs = replace(inputs, observation=at_second, previous=previous)
                history_id = f"{root}.{policy_id}.induced.causal-capture"
        assignments.append(
            (root, entered, () if entered else ("LOCAL_LAW_OR_CAUSAL_PREREQUISITE_NONENTRY",))
        )
        if not entered:
            continue
        assert inputs is not None
        scenario = draw_scenario(root, "heldout", assignment(root)[2])
        scenario_id = ObjectIdentity.from_record(root, scenario)
        stream = f"reactor-scenario.{root}.{scenario.seed}"
        entered_roots.append(
            PreparedRootAssignment(
                root,
                request.context if kind == "local" else kind,
                "classical-finite-response-service",
                stream,
            )
        )
        coupling = PreparedNativeRealizationCoupling(
            f"{root}.native-realization", root, stream, scenario_id
        )
        couplings.append(coupling)
        word: OccurrenceActionWord = project_pulse(
            inputs.observation,
            inputs.previous,
            ZERO,
            decision_id=f"{root}.{policy_id}.{kind}.hold",
            history_id=history_id,
            horizon_id=HORIZON,
            guard_s=readout.guard_s,
            receiver_id="c2" if kind == "joint" else "c1" if kind in ("first", "baseline") else "c",
        ).word
        holds.append(PreparedRootHoldWord(root, word))
        for role in (PreparedFutureRole.COMMITTED_TASK, PreparedFutureRole.MATCHED_HOLD):
            branch = (
                f"{policy_id}.actual"
                if role is PreparedFutureRole.COMMITTED_TASK
                else f"{request.context}.zero"
                if kind == "local"
                else "a0"
                if kind == "joint"
                else "00"
            )
            acquisition = ClassicalAcquisition(root, block, branch, scenario_id)
            acquisitions.append(acquisition)
            futures.append(
                PreparedFutureSlot(
                    f"{root}.{policy_id}.{kind}.{role.value.lower()}",
                    root,
                    policy_id,
                    role,
                    0,
                    acquisition.record_id,
                    stream,
                    None
                    if role is PreparedFutureRole.COMMITTED_TASK
                    else ObjectIdentity.from_record(word.word_id, word),
                    ObjectIdentity.from_record(acquisition.record_id, acquisition),
                    ObjectIdentity.from_record(coupling.coupling_id, coupling),
                )
            )
    evaluation = (
        None
        if not entered_roots
        else CoupledRealizationControllerEvaluationPlan(
            stem,
            tuple(entered_roots),
            (allocation,),
            tuple(sorted(futures, key=lambda f: f.slot_id)),
            ("reactor-native", "reactor-refined"),
            tolerances,
            (),
            0,
            None,
            holds[0].word,
            frozen.recipe,
            ("assigned-scenario",),
            ("measured-local-or-induced-chart",),
            boundary,
            (parent.predicate_id,),
            tuple(sorted(p.predicate_id for p in (future, *raw_gates))),
            reducer,
            "reactor-staged-pulse-response-preaction-cutoff",
            OutcomeAccess.EVALUATION_SEALED,
            VisibilityCeiling.PROSPECTIVE,
            EvidenceCeiling.CONTROLLER_USE,
            tuple(couplings),
            tuple(holds),
        )
    )
    return ReactorStagedPulseResponseProspectiveUsePlan(
        block,
        policy,
        request,
        kind,
        prerequisite,
        tuple(assignments),
        evaluation,
        tuple(acquisitions),
    )


def build_prospective_plan(
    laws: ClassicalLaws,
    preparations: tuple[ClassicalPreparation, ...],
    seals: tuple[ClassicalPredictionSeal, ...],
    source: ReactorBatchSource,
) -> ReactorStagedPulseResponseProspectivePlan:
    block = laws.operands.nomination.block
    assigned = roots(block, "prospective")
    if (
        tuple(p.root for p in preparations) != assigned
        or tuple(s.root for s in seals) != assigned
        or any(
            s.preparation != ObjectIdentity.from_record(p.record_id, p)
            or s.nomination
            != ObjectIdentity.from_record(
                laws.operands.nomination.record_id, laws.operands.nomination
            )
            for p, s in zip(preparations, seals, strict=True)
        )
    ):
        raise ValueError("Controller-use plan substitutes a frozen recipe or an assigned causal preparation")
    census = tuple(
        (policy, request, kind)
        for policy in (ARMS if block == "base-menu-comparison" else MENUS if block == "expanded-menu-comparison" else SEQUENCES)
        for request in (LOCAL_REQUESTS if block != "staged-sequence-comparison" else PAIR_REQUESTS)
        for kind in (
            ("local",)
            if block != "staged-sequence-comparison"
            else ("baseline",)
            if policy == "ONE_PULSE"
            else ("first", "joint")
        )
    )
    causal_inputs = {
        (p.root, c.context): causal_operands(c, laws.rows[0].payload.prepared_domain)
        for p in preparations
        for c in p.contexts
    }
    uses = tuple(
        _build_use(laws, preparations, seals, policy, request, kind, causal_inputs)
        for policy, request, kind in census
    )
    source_id = ObjectIdentity.from_record("reactor-staged-pulse-response.native-source", source)
    if any(p.source_sha256 != source.fingerprint() for p in preparations):
        raise ValueError("prospective plan substitutes its pinned native source")
    sequences = []
    if block == "staged-sequence-comparison":
        evaluator = next(
            b
            for b, _ in implementation_payloads()
            if b.role is ImplementationRole.OUTCOME_EVALUATOR
        )
        for policy in SEQUENCES[:2]:
            for request in PAIR_REQUESTS:
                children = tuple(
                    next(
                        u
                        for u in uses
                        if (u.policy, u.request, u.stage_kind) == (policy, request, kind)
                    )
                    for kind in ("first", "joint")
                )
                first, second = (c.plan for c in children)
                if first is None or second is None:
                    continue
                if (
                    first.roots != tuple(replace(r, context_id="first") for r in second.roots)
                    or first.realization_couplings != second.realization_couplings
                ):
                    raise ValueError(
                        "sequence children changed their common preassigned root census"
                    )
                readout = ClassicalSequenceReadout(
                    request, ObjectIdentity.from_record(laws.record_id, laws)
                )
                operator = ObjectIdentity.from_record(readout.record_id, readout)
                limits = tuple(
                    SequenceReceiverLimit(key, unit, operator, lo, hi, tol)
                    for key, unit, lo, hi, tol in (
                        ("c1", "K", request.required_K, None, D(".000001")),
                        ("c2", "K", request.second_K, None, D(".000001")),
                        ("g", "K", D(0), None, D(".000024")),
                        ("mass", "kg", D(0), request.budget_kg + D("1e-10"), D("1e-12")),
                        ("s", "K", None, D("356.2"), D(".01")),
                    )
                )
                stages = tuple(
                    SequenceStageContract(
                        i,
                        D(120 * i),
                        ObjectIdentity.from_record(child.evaluation_plan_id, child),
                        child.policies[0].child_recipe,
                    )
                    for i, child in enumerate((first, second))
                )
                sequences.append(
                    BoundedSequenceEvaluationPlan(
                        f"reactor-staged-pulse-response.{policy.lower()}.{request.request_id.lower()}.sequence",
                        tuple(r.root_id for r in first.roots),
                        f"{policy.lower()}.{request.request_id.lower()}",
                        request.request_id.lower(),
                        (stages[0], stages[1]),
                        native_coordinate(D(0)),
                        D(240),
                        source_id,
                        ObjectIdentity.from_record(laws.record_id, laws),
                        ("reactor-native", "reactor-refined"),
                        limits,
                        evaluator,
                        first.reducer,
                        first.realization_couplings,
                    )
                )
    return ReactorStagedPulseResponseProspectivePlan(
        block,
        ObjectIdentity.from_record(laws.record_id, laws),
        tuple(ObjectIdentity.from_record(s.record_id, s) for s in seals),
        tuple(CanonicalRecordArchive.pack(u.record_id, u) for u in uses),
        source_id,
        tuple(sequences),
    )


@dataclass(frozen=True, slots=True)
class ClassicalSequenceReadout(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-staged-pulse-response/classical-sequence-readout'
    request: Request
    laws: ObjectIdentity
    definition: str = "actual-00-a0-aw-paired-native-c1-c2-g-maximum-temperature-applied-mass-240s"

    def __post_init__(self) -> None:
        if (
            self.request not in PAIR_REQUESTS
            or self.laws.object_schema != ClassicalLaws.SCHEMA
            or self.definition
            != "actual-00-a0-aw-paired-native-c1-c2-g-maximum-temperature-applied-mass-240s"
        ):
            raise ValueError("sequence readout changes its declared raw native receiver")

    @property
    def record_id(self) -> str:
        return f"reactor-staged-pulse-response.{self.request.request_id.lower()}.sequence-readout"
