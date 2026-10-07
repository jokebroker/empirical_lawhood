"Independent admission owners, the installed finite ranker, then one commitment."

from dataclasses import dataclass
from decimal import Decimal as D

from empirical_lawhood.adapters.control.composition import ControllerStudyComposition
from empirical_lawhood.adapters.control.publication import ControlPublisher
from empirical_lawhood.adapters.control.finite_campaign import finite_control_study, finite_control_services
from empirical_lawhood.adapters.methods.law_assessment import CandidatePayloadReader
from empirical_lawhood.adapters.methods.receiver_conditioned_io.finite_action_mpc import (
    FiniteMPCConfig,
    FiniteMPCAggregationRule,
    FiniteMPCTieBreakRule,
    FiniteMPCProposal,
)
from empirical_lawhood.adapters.simulators.reactor_causal_response.delivery import BatchDeliverySession, coordinate
from empirical_lawhood.adapters.simulators.reactor_finite_control_frontier.delivery import FrontierPulseDeliveryPort
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import EvidenceLink, EvidenceRelation, ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.planning.controller_study import ImplementationRole, DeliveryControllerStudy
from empirical_lawhood.planning.finite_response_geometry import FiniteResponseSet, FiniteTaskFunctionalSpec
from empirical_lawhood.planning.nested_controller_evaluation import CoupledRealizationControllerEvaluationPlan
from empirical_lawhood.runtime.controller_compiler import CompiledDeliveryControllerStudy, AdmissionCandidateAudit
from empirical_lawhood.runtime.controller_runtime import DeliveryControllerDecisionCommitment, RuntimeObservation
from empirical_lawhood.adapters.control.finite_ranking import AssessedFiniteAction, rank_finite_actions
from .control_causal import causal_binding
from .control_corpus import control_corpus
from .control_plan import FrontierControlContext, control_context
from .control_prediction import predict
from .control_records import FrontierClockMap, FrontierReadout, consumer
from .control_services import implementation_payloads, implementation_configurations
from .law_terminal import FrontierLaw, FrontierLaws
from .records import FrontierContext
from .selection import FrontierUseRequest, pulse_tie_key
from .science import CHART, frontier_system


def ranker_config() -> FiniteMPCConfig:
    return FiniteMPCConfig(
        "reactor-finite-control-frontier.feed-effort",
        D(0),
        D(0),
        D(1),
        D(0),
        "1",
        "1",
        FiniteMPCAggregationRule.WORST_MEMBER_COST,
        FiniteMPCTieBreakRule.UTILITY_DESCENDING_CANDIDATE_ID_ASCENDING,
    )


@dataclass(frozen=True)
class FrontierAssessedCandidate:
    context: FrontierControlContext
    study: DeliveryControllerStudy
    audits: tuple[AdmissionCandidateAudit, ...]
    services: ControllerStudyComposition
    observation: RuntimeObservation
    task: FiniteTaskFunctionalSpec
    forecasts: tuple[FiniteResponseSet, ...]


@dataclass(frozen=True)
class FrontierPreparedRequest:
    candidate: FrontierAssessedCandidate
    compiled: CompiledDeliveryControllerStudy
    commitment: DeliveryControllerDecisionCommitment


def assess_candidate(
    *,
    laws: FrontierLaws,
    report: FrontierLaw,
    causal: FrontierContext,
    request: FrontierUseRequest,
    publisher: ControlPublisher,
    reader: CandidatePayloadReader,
    session: BatchDeliverySession,
    authority: ObjectIdentity,
    resources: ObjectIdentity,
    evaluation: CoupledRealizationControllerEvaluationPlan,
) -> FrontierAssessedCandidate:
    implementations = tuple(b for b, _ in implementation_payloads())
    for config in implementation_configurations():
        if publisher.publish_record(config.config_id, config) != next(
            b.reference.payload for b in implementations if b.role is config.role
        ):
            raise ValueError("controller configuration publication changed its exact identity")
    producer = ObjectIdentity.from_record("reactor-finite-control-frontier.admission-owner", implementations[0])
    context = control_context(
        report=report,
        causal=causal,
        request=request,
        producer=producer,
        resource=resources,
        authority=authority,
    )
    causal_artifact = publisher.publish_record(f"{causal.root}.{causal.context}.causal", causal)
    table = predict(context, reader, causal_artifact)
    assert causal.callback is not None
    readout = FrontierReadout(causal.context, request.horizon_s)
    clock = FrontierClockMap(causal.root, causal.callback, readout)
    frozen = consumer(laws, readout)
    row = report.operands
    calibration = publisher.publish_record(f"{row.record_id}.calibration", row)
    artifacts = tuple(
        sorted(
            (
                causal_artifact,
                publisher.publish_record(readout.record_id, readout),
                publisher.publish_record(clock.map_id, clock),
                publisher.publish_record(table.table_id, table),
                calibration,
                report.candidate.payload_publication.artifact,
            ),
            key=lambda a: a.artifact_id,
        )
    )
    evidence = (
        EvidenceLink(
            f"{context.plan.plan_id}.causal-evidence",
            EvidenceRelation.DERIVED_FROM,
            ObjectIdentity.from_record(table.table_id, table),
            ObjectIdentity.from_record(f"{causal.root}.{causal.context}.causal", causal),
            tuple(a.artifact_id for a in artifacts),
            frontier_system().world.world_id,
            context.plan.information_cutoff.cutoff_id,
            OutcomeAccess.OUTCOME_BLIND,
            VisibilityCeiling.PROSPECTIVE,
            (VisibilityCeiling.PROSPECTIVE,),
            "One independently qualified pulse; exact causal prefix and frozen inner-window/guard compatibility.",
        ),
    )
    corpus = control_corpus(
        context=context,
        table=table,
        consumer=frozen,
        calibration_artifact=calibration,
        source_artifacts=artifacts,
        evidence=evidence,
    )
    publisher.publish_record(corpus.corpus_id, corpus)
    raw = causal.arrays.unpack()["observations"][-1]
    observation = RuntimeObservation(
        f"{context.plan.plan_id}.observation",
        causal.root,
        tuple(
            sorted(
                (
                    NamedDecimal(k, D(repr(float(v))), u)
                    for k, v, u in (
                        ("callback-time", raw[0], "s"),
                        ("observed-temperature", raw[1], "K"),
                        ("observed-jacket", raw[2], "K"),
                        ("observed-dose", raw[3], "kg"),
                    )
                ),
                key=lambda v: v.value_id,
            )
        ),
        coordinate(D(causal.callback * 10)),
        artifacts,
        OutcomeAccess.OUTCOME_BLIND,
    )
    publisher.publish_record(observation.observation_id, observation)
    programme = finite_control_study(
        system=frontier_system(),
        corpus=corpus,
        action_bindings=(causal_binding(context, evidence),),
        implementations=implementations,
        observation=observation,
        frozen_recipe=ObjectIdentity.from_record(frozen.consumer_id, frozen),
        compiler_release_id="reactor-finite-control-frontier.finite-admission-controller-compiler",
        decision_budget_seconds=D(1),
        prospective_evaluation=ObjectIdentity.from_record(
            evaluation.evaluation_plan_id, evaluation
        ),
    )
    publisher.publish_record(programme.study_id, programme)
    delivery = FrontierPulseDeliveryPort(
        next(b for b in implementations if b.role is ImplementationRole.DELIVERY),
        session,
        context.word,
    )
    services = finite_control_services(programme, evaluation, delivery)
    audits = services.assess_finite_candidates(programme)
    for audit in audits:
        publisher.publish_record(audit.audit_id, audit)
    return FrontierAssessedCandidate(
        context,
        programme,
        audits,
        services,
        observation,
        corpus.reachability_receipts[0].request.task,
        tuple(r.request.response_set for r in corpus.reachability_receipts),
    )


def rank_and_commit(
    candidates: tuple[FrontierAssessedCandidate, ...], publisher: ControlPublisher
) -> tuple[FiniteMPCProposal, FrontierPreparedRequest | None]:
    if not candidates or len({(c.context.causal.root, c.context.request) for c in candidates}) != 1:
        raise ValueError(
            "ranking requires one exact root/request and its independent admission candidates"
        )
    first = candidates[0]
    stem = f"{first.context.causal.root}.{first.context.request.request_id}.finite-ranking"
    actions = tuple(
        AssessedFiniteAction(
            f"{stem}.{pulse_tie_key(c.context.word.pulse)}",
            c.context.word.word,
            c.context.word.applied_mass_kg / D("2.10"),
            c.study,
            c.audits,
            c.services,
            c.observation,
        )
        for c in candidates
    )
    proposal, committed = rank_finite_actions(
        actions=actions, stem=stem, chart=CHART, config=ranker_config(), publisher=publisher
    )
    if committed is None:
        return proposal, None
    chosen = next(c for c, a in zip(candidates, actions, strict=True) if a == committed.assessed)
    return proposal, FrontierPreparedRequest(chosen, committed.compiled, committed.commitment)
