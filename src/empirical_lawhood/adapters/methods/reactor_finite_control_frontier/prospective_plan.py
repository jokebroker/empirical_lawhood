"Frozen request-specific prepared controller-use censuses; physical roots stay explicit."

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar

from empirical_lawhood.adapters.simulators.reactor_causal_response.design import draw_scenario
from empirical_lawhood.adapters.simulators.reactor_finite_control_frontier.words import project_pulse
from empirical_lawhood.kernel.admission import AdmissionGateKind
from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord
from empirical_lawhood.kernel.causal_contracts import (
    PredicateDirection,
    ReceiverInterval,
    TemporalPredicateSemantics,
)
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.runtime.canonical_record_archive import CanonicalRecordArchive
from empirical_lawhood.kernel.time import ClockCoordinate, CoordinateOrigin
from empirical_lawhood.planning.controller_study import EvaluatorBoundarySpec, ImplementationRole
from empirical_lawhood.planning.evidence_geometry import GatePredicateKind, GatePredicateSpec
from empirical_lawhood.planning.nested_controller_evaluation import PreparedFutureRole, PreparedFutureSlot, CoupledRealizationControllerEvaluationPlan, PreparedInterfaceReducerRegistration, PreparedNativeRealizationCoupling, PreparedPolicyAssignment, PreparedReadoutTolerance, PreparedRootAssignment, PreparedRootHoldWord
from empirical_lawhood.runtime.controller_evaluation_nested import RevealedPreparedFuture, SealedPreparedFutureLocator
from .causal import causal_operands
from .config import ROOTS, ZERO, FrontierDesign, Pulse, assignment
from .control_records import FrontierReadout, consumer, response_coordinates
from .control_services import implementation_payloads
from .law_terminal import FrontierLaws
from .records import FrontierPreparation
from .selection import FrontierPredictionSeal, FrontierUseRequest
from .science import HORIZON, LOCAL_CLOCK, LOCAL_FRAME, RECEIVERS

PROSPECTIVE_ROOTS = tuple(r for r, role, _, _ in ROOTS if role == "prospective")


@dataclass(frozen=True, slots=True)
class FrontierAcquisition(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-finite-control-frontier/frontier-acquisition'
    root: str
    context: str
    request_id: str | None
    role: PreparedFutureRole
    pulse: Pulse | None
    scenario: ObjectIdentity

    @property
    def record_id(self) -> str:
        return (
            f"{self.root}.primary.{self.request_id}"
            if self.role is PreparedFutureRole.COMMITTED_TASK
            else f"{self.root}.evaluator.{self.context}.{self.pulse.word_id if self.pulse else 'missing'}"
        )


@dataclass(frozen=True, slots=True)
class ReactorFiniteControlFrontierProspectiveRequestPlan(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-finite-control-frontier/reactor-finite-control-frontier-prospective-request-plan'
    request: FrontierUseRequest
    assigned: tuple[tuple[str, bool, tuple[str, ...]], ...]
    plan: CoupledRealizationControllerEvaluationPlan | None
    acquisitions: tuple[FrontierAcquisition, ...]

    @property
    def record_id(self) -> str:
        return f"reactor-finite-control-frontier.{self.request.request_id}.prospective-evaluation-request"

    def __post_init__(self) -> None:
        entered = tuple(r for r, yes, _ in self.assigned if yes)
        if (
            tuple(r for r, _, _ in self.assigned) != PROSPECTIVE_ROOTS
            or (self.plan is None) != (not entered)
            or self.plan is not None
            and tuple(r.root_id for r in self.plan.roots) != entered
            or len(self.acquisitions) != len(entered) * 3
        ):
            raise ValueError("Controller use request plan changed its all-assigned versus entered denominator")


@dataclass(frozen=True, slots=True)
class ReactorFiniteControlFrontierProspectiveRequestArchive(CanonicalRecord):
    """Lossless existing archive transport, decoded one request at a time."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-finite-control-frontier/reactor-finite-control-frontier-prospective-request-archive'
    request: FrontierUseRequest
    archive: CanonicalRecordArchive

    def __post_init__(self) -> None:
        if (
            self.archive.subject.object_schema != ReactorFiniteControlFrontierProspectiveRequestPlan.SCHEMA
            or self.archive.subject.object_id
            != f"reactor-finite-control-frontier.{self.request.request_id}.prospective-evaluation-request"
        ):
            raise ValueError("archived controller use request changed its scientific subject")

    @classmethod
    def pack(cls, request: ReactorFiniteControlFrontierProspectiveRequestPlan) -> 'ReactorFiniteControlFrontierProspectiveRequestArchive':
        return cls(request.request, CanonicalRecordArchive.pack(request.record_id, request))

    def unpack(self) -> ReactorFiniteControlFrontierProspectiveRequestPlan:
        result = decode_canonical_bytes(
            self.archive.unpack(),
            ReactorFiniteControlFrontierProspectiveRequestPlan,
            maximum_bytes=self.archive.decoded_bytes,
        )
        if result.request != self.request:
            raise ValueError("archived controller use request substituted its causal request")
        return result


@dataclass(frozen=True, slots=True)
class ReactorFiniteControlFrontierProspectivePlan(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-finite-control-frontier/reactor-finite-control-frontier-prospective-plan'
    laws: ObjectIdentity
    seals: tuple[ObjectIdentity, ...]
    requests: tuple[ReactorFiniteControlFrontierProspectiveRequestArchive, ...]

    def __post_init__(self) -> None:
        if (
            self.laws.object_schema != FrontierLaws.SCHEMA
            or len(self.seals) != 64
            or len(self.requests) > 24
        ):
            raise ValueError("Controller-use plan lost its qualified parent or fixed root/request allocation")

    @property
    def record_id(self) -> str:
        return "reactor-finite-control-frontier.prospective-evaluation-plan"


def build_prospective_plan(
    laws: FrontierLaws,
    preparations: tuple[FrontierPreparation, ...],
    seals: tuple[FrontierPredictionSeal, ...],
    requests: tuple[FrontierUseRequest, ...],
) -> ReactorFiniteControlFrontierProspectivePlan:
    if (
        tuple(p.root for p in preparations) != PROSPECTIVE_ROOTS
        or tuple(s.root for s in seals) != PROSPECTIVE_ROOTS
    ):
        raise ValueError("Controller-use plan cannot omit or replace an assigned physical root")
    binding = next(
        b for b, _ in implementation_payloads() if b.role is ImplementationRole.OUTCOME_EVALUATOR
    )
    views = ("reactor-native", "reactor-refined")
    bundles = []
    for request in requests:
        stem = f"reactor-finite-control-frontier.{request.request_id}.prospective-evaluation"
        readout = FrontierReadout(request.context, request.horizon_s)
        frozen = consumer(laws, readout)
        reports = tuple(
            r
            for r in laws.rows
            if r.payload.bound.coordinate.context == request.context
            and r.payload.bound.coordinate.horizon_s == request.horizon_s
            and r.payload.bound.coordinate.coordinate_id in laws.qualified
        )
        widths = tuple(
            (r.payload.bound.upper_K - r.payload.bound.lower_K) / 2
            for r in reports
            if r.payload.bound.upper_K is not None and r.payload.bound.lower_K is not None
        )
        tolerances = tuple(
            sorted(
                (
                    PreparedReadoutTolerance(
                        f"{stem}.{c.coordinate_id}.tolerance",
                        c,
                        D(".01")
                        if c.quantity_id == RECEIVERS[0]
                        else D(".000001") * D(request.horizon_s) / 10,
                        D("356.2") if c.quantity_id == RECEIVERS[0] else max(widths),
                    )
                    for c in response_coordinates(readout)
                ),
                key=lambda t: t.tolerance_id,
            )
        )

        def point(t: D) -> ClockCoordinate:
            return ClockCoordinate(
                LOCAL_CLOCK, t, "s", LOCAL_FRAME, CoordinateOrigin.EPISODE_RELATIVE
            )

        def predicate(kind: AdmissionGateKind, start: D, end: D) -> GatePredicateSpec:
            return GatePredicateSpec(
                f"{stem}.{kind.value.lower()}",
                kind,
                RECEIVERS[0],
                PredicateDirection.AT_MOST,
                "prepared-prefix-maximum-temperature"
                if end == 0
                else "local-action-guard-maximum-temperature",
                ReceiverInterval(point(start), point(end)),
                GatePredicateKind.SCALAR_AT_MOST,
                None,
                NamedDecimal("native-reactor-temperature-limit", D("356.2"), "K"),
                None,
                None,
                TemporalPredicateSemantics.ALWAYS_PRESERVED_PATH if end else None,
                (f"{stem}.native-safety",),
                binding.reference,
            )

        parent, future = (
            predicate(AdmissionGateKind.PHYSICAL_SINK, D(0), D(0)),
            predicate(AdmissionGateKind.BASELINE_PRESERVATION, D(0), D(120)),
        )
        boundary = EvaluatorBoundarySpec(
            f"{stem}.boundary",
            binding.binding_id,
            "reactor-native-outcome-custodian",
            "reactor-separate-reveal-authority",
            "reactor-prepared-controller-evaluator",
            SealedPreparedFutureLocator.SCHEMA,
            RevealedPreparedFuture.SCHEMA,
            tuple(sorted((parent.quantity_id, future.quantity_id))),
            tuple(sorted((parent, future), key=lambda p: p.predicate_id)),
        )
        design = FrontierDesign()
        reducer = PreparedInterfaceReducerRegistration(
            f"{stem}.reducer",
            binding.reference.capability_key,
            binding.reference.capability_version,
            binding.config_sha256,
            binding.implementation_sha256,
            ("NUMERICAL_VIEW", "FUTURE_ROLE", "ROOT_POLICY", "ROOT"),
            ObjectIdentity.from_record(design.config_id, design),
        )
        policy = PreparedPolicyAssignment(
            f"el.{request.request_id}",
            ObjectIdentity.from_record(request.request_id, request),
            ObjectIdentity.from_record(frozen.consumer_id, frozen),
            ("assigned-scenario",),
        )
        roots, assigned, futures, acquisitions, couplings, holds, audits = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for preparation, seal in zip(preparations, seals, strict=True):
            root = preparation.root
            if seal.preparation != ObjectIdentity.from_record(preparation.record_id, preparation):
                raise ValueError("Controller-use plan substituted a sealed causal preparation")
            pulse = next(w for r, w in seal.primary if r == request)
            ca = next(c for c in preparation.contexts if c.context == request.context)
            operands = causal_operands(ca, reports[0].payload.prepared_domain)
            entered = pulse is not None and operands is not None
            assigned.append((root, entered, () if entered else ("CAUSAL_NONATTEMPT",)))
            if not entered:
                continue
            assert pulse is not None and operands is not None
            scenario = draw_scenario(root, "heldout", assignment(root)[1])
            scenario_id = ObjectIdentity.from_record(scenario.unit_id, scenario)
            stream = f"reactor-scenario.{root}.{assignment(root)[1]}"
            roots.append(
                PreparedRootAssignment(
                    root, request.context, "paired-finite-pulse-peak-response", stream
                )
            )
            coupling = PreparedNativeRealizationCoupling(
                f"{root}.native-realization", root, stream, scenario_id
            )
            couplings.append(coupling)

            def word(w: Pulse) -> OccurrenceActionWord:
                return project_pulse(
                    operands.observation,
                    operands.previous,
                    w,
                    decision_id=f"{root}.{ca.context}.{w.word_id}",
                    history_id=f"{root}.{ca.context}.causal",
                    horizon_id=HORIZON,
                ).word

            hold, action = word(ZERO), word(pulse)
            holds.append(PreparedRootHoldWord(root, hold))
            audits.append(action)
            for role, w in (
                (PreparedFutureRole.COMMITTED_TASK, None),
                (PreparedFutureRole.MATCHED_HOLD, ZERO),
                (PreparedFutureRole.AUDIT_PROBE, pulse),
            ):
                acquisition = FrontierAcquisition(
                    root,
                    ca.context,
                    request.request_id if w is None else None,
                    role,
                    w,
                    scenario_id,
                )
                acquisitions.append(acquisition)
                futures.append(
                    PreparedFutureSlot(
                        f"{root}.{request.request_id}.{role.value.lower()}",
                        root,
                        policy.policy_id,
                        role,
                        0,
                        acquisition.record_id,
                        stream,
                        None
                        if w is None
                        else ObjectIdentity.from_record(
                            (hold if w == ZERO else action).word_id, hold if w == ZERO else action
                        ),
                        ObjectIdentity.from_record(acquisition.record_id, acquisition),
                        ObjectIdentity.from_record(coupling.coupling_id, coupling),
                    )
                )
        plan = (
            None
            if not roots
            else CoupledRealizationControllerEvaluationPlan(
                stem,
                tuple(roots),
                (policy,),
                tuple(sorted(futures, key=lambda s: s.slot_id)),
                views,
                tolerances,
                tuple(sorted(audits, key=lambda w: w.word_id)),
                1,
                ObjectIdentity.from_record(request.request_id, request),
                holds[0].word,
                frozen.recipe,
                ("assigned-scenario",),
                ("measured-local-chart",),
                boundary,
                (parent.predicate_id,),
                (future.predicate_id,),
                reducer,
                "reactor-frontier-preaction-callback",
                OutcomeAccess.EVALUATION_SEALED,
                VisibilityCeiling.PROSPECTIVE,
                EvidenceCeiling.CONTROLLER_USE,
                tuple(couplings),
                tuple(holds),
            )
        )
        bundles.append(
            ReactorFiniteControlFrontierProspectiveRequestArchive.pack(
                ReactorFiniteControlFrontierProspectiveRequestPlan(request, tuple(assigned), plan, tuple(acquisitions))
            )
        )
    return ReactorFiniteControlFrontierProspectivePlan(
        ObjectIdentity.from_record(laws.record_id, laws),
        tuple(ObjectIdentity.from_record(s.record_id, s) for s in seals),
        tuple(bundles),
    )
