"Staged-pulse causal bindings to the shared assessment, finite ranking and commitment route."

from dataclasses import dataclass, replace
from decimal import Decimal as D

from empirical_lawhood.adapters.control.composition import ControllerStudyComposition
from empirical_lawhood.adapters.control.finite_campaign import finite_control_study, finite_control_services
from empirical_lawhood.adapters.control.finite_ranking import AssessedFiniteAction, rank_finite_actions
from empirical_lawhood.adapters.control.publication import ControlPublisher
from empirical_lawhood.adapters.methods.law_assessment import CandidatePayloadReader
from empirical_lawhood.adapters.methods.receiver_conditioned_io.finite_action_mpc import FiniteMPCProposal
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.control_owner import ranker_config
from empirical_lawhood.adapters.simulators.reactor_staged_pulse_response.words import CHART, ClassicalPulseDeliveryPort
from empirical_lawhood.adapters.simulators.reactor_causal_response.delivery import BatchDeliverySession, coordinate
from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_design import ReactorBatchSource
from empirical_lawhood.adapters.simulators.reactor_prefix_response.finite_support import finite_feed_causal_binding
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import EvidenceLink, EvidenceRelation, ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.planning.controller_study import DeliveryControllerStudy, ImplementationRole
from empirical_lawhood.planning.finite_response_geometry import FiniteResponseSet, FiniteTaskFunctionalSpec
from empirical_lawhood.planning.nested_controller_evaluation import CoupledRealizationControllerEvaluationPlan
from empirical_lawhood.runtime.controller_compiler import AdmissionCandidateAudit, CompiledDeliveryControllerStudy
from empirical_lawhood.runtime.controller_evaluation_trajectory import TrajectoryCallbackLink
from empirical_lawhood.runtime.controller_runtime import DeliveryControllerDecisionCommitment, RuntimeObservation
from .config import Request
from .control_corpus import control_corpus
from .control_plan import ClassicalControlContext, control_context
from .control_prediction import predict_control
from .control_records import ClassicalClockMap, consumer
from .control_services import implementation_configurations, implementation_payloads
from .law_terminal import ClassicalLaws, ClassicalLaw
from .prediction import ClassicalPredictedBound
from .records import ClassicalContext
from .science import UNIT, staged_pulse_reactor_system


@dataclass(frozen=True)
class ClassicalAssessedCandidate:
    context: ClassicalControlContext
    study: DeliveryControllerStudy
    audits: tuple[AdmissionCandidateAudit, ...]
    services: ControllerStudyComposition
    observation: RuntimeObservation
    task: FiniteTaskFunctionalSpec
    forecasts: tuple[FiniteResponseSet, ...]
    link: TrajectoryCallbackLink


@dataclass(frozen=True)
class ClassicalPreparedRequest:
    candidate: ClassicalAssessedCandidate
    compiled: CompiledDeliveryControllerStudy
    commitment: DeliveryControllerDecisionCommitment


def assess_candidate(
    *,
    laws: ClassicalLaws,
    report: ClassicalLaw,
    causal: ClassicalContext,
    request: Request,
    policy: str,
    predicted: ClassicalPredictedBound,
    source: ReactorBatchSource,
    publisher: ControlPublisher,
    reader: CandidatePayloadReader,
    session: BatchDeliverySession,
    authority: ObjectIdentity,
    resources: ObjectIdentity,
    evaluation: CoupledRealizationControllerEvaluationPlan,
) -> ClassicalAssessedCandidate:
    implementations = tuple(b for b, _ in implementation_payloads())
    for config in implementation_configurations():
        if publisher.publish_record(config.config_id, config) != next(
            b.reference.payload for b in implementations if b.role is config.role
        ):
            raise ValueError("controller publication changed the installed finite owner")
    producer = ObjectIdentity.from_record("reactor-staged-pulse-response.admission-owner", implementations[0])
    context = control_context(
        laws=laws,
        report=report,
        causal=causal,
        request=request,
        policy=policy,
        predicted=predicted,
        producer=producer,
        resource=resources,
        authority=authority,
    )
    causal_artifact = publisher.publish_record(causal.record_id, causal)
    table = predict_control(context, source, reader, causal_artifact)
    assert causal.callback is not None
    clock = ClassicalClockMap(
        causal.root,
        causal.callback,
        causal.first_callback if causal.first_callback is not None else causal.callback,
        context.readout,
    )
    frozen = consumer(laws, context.readout)
    link = TrajectoryCallbackLink(
        f"{causal.root}.{policy.lower()}.{request.request_id.lower()}.{context.readout.stage_kind}.link",
        causal.root,
        int(causal.context == "induced"),
        causal.predecessor,
        causal_artifact,
    )
    row = report.operands
    calibration = publisher.publish_record(f"{row.record_id}.calibration", row)
    artifacts = tuple(
        sorted(
            (
                causal_artifact,
                publisher.publish_record(link.link_id, link),
                publisher.publish_record(context.readout.record_id, context.readout),
                publisher.publish_record(clock.map_id, clock),
                publisher.publish_record(table.table_id, table),
                calibration,
                report.candidate.payload_publication.artifact,
            ),
            key=lambda a: a.artifact_id,
        )
    )
    system = staged_pulse_reactor_system(report.payload.recipe.bound.coordinate)
    evidence = (
        EvidenceLink(
            f"{context.plan.plan_id}.causal-evidence",
            EvidenceRelation.DERIVED_FROM,
            ObjectIdentity.from_record(table.table_id, table),
            ObjectIdentity.from_record(causal.record_id, causal),
            tuple(a.artifact_id for a in artifacts),
            system.world.world_id,
            context.plan.information_cutoff.cutoff_id,
            OutcomeAccess.OUTCOME_BLIND,
            VisibilityCeiling.PROSPECTIVE,
            (VisibilityCeiling.PROSPECTIVE,),
            "One qualified finite word under its exact actual prefix and predecessor; explicit raw/saturated receiver map and native clock.",
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
    values = tuple(
        sorted(
            (
                NamedDecimal(k, D(repr(float(v))), unit)
                for k, v, unit in (
                    ("callback-time", raw[0], "s"),
                    ("observed-temperature", raw[1], "K"),
                    ("observed-jacket", raw[2], "K"),
                    ("observed-dose", raw[3], "kg"),
                )
            ),
            key=lambda v: v.value_id,
        )
    )
    observation = RuntimeObservation(
        f"{context.plan.plan_id}.observation",
        causal.root,
        values,
        coordinate(D(causal.callback * 10)),
        artifacts,
        OutcomeAccess.OUTCOME_BLIND,
    )
    publisher.publish_record(observation.observation_id, observation)
    law = report.qualification.response_law
    assert law is not None
    action = finite_feed_causal_binding(
        plan=context.plan,
        law=law,
        word=context.word.word,
        native_mapping=ObjectIdentity.from_record(context.word.word.word_id, context.word),
        views=report.family.axis_map.qualification_view_ids,
        unit=UNIT,
        guard_s=context.readout.guard_s,
        evidence=evidence,
    )
    programme = finite_control_study(
        system=system,
        corpus=corpus,
        action_bindings=(action,),
        implementations=implementations,
        observation=observation,
        frozen_recipe=ObjectIdentity.from_record(frozen.consumer_id, frozen),
        compiler_release_id="reactor-staged-pulse-response.finite-admission-controller-compiler",
        decision_budget_seconds=D(1),
        prospective_evaluation=ObjectIdentity.from_record(
            evaluation.evaluation_plan_id, evaluation
        ),
    )
    publisher.publish_record(programme.study_id, programme)
    delivery = ClassicalPulseDeliveryPort(
        next(b for b in implementations if b.role is ImplementationRole.DELIVERY),
        session,
        context.word,
    )
    services = finite_control_services(programme, evaluation, delivery)
    audits = services.assess_finite_candidates(programme)
    for audit in audits:
        publisher.publish_record(audit.audit_id, audit)
    return ClassicalAssessedCandidate(
        context,
        programme,
        audits,
        services,
        observation,
        corpus.reachability_receipts[0].request.task,
        tuple(r.request.response_set for r in corpus.reachability_receipts),
        link,
    )


def rank_and_commit(
    candidates: tuple[ClassicalAssessedCandidate, ...],
    publisher: ControlPublisher,
    *,
    commit_nonattempt: bool = False,
) -> tuple[FiniteMPCProposal, ClassicalPreparedRequest | None]:
    if (
        not candidates
        or len(
            {
                (
                    c.context.causal.root,
                    c.context.request,
                    c.context.policy,
                    c.context.readout.stage_kind,
                )
                for c in candidates
            }
        )
        != 1
    ):
        raise ValueError("ranking mixes separate owner requests or native cutoffs")
    first = candidates[0].context
    stem = f"{first.causal.root}.{first.policy.lower()}.{first.request.request_id.lower()}.{first.readout.stage_kind}.ranking"
    actions = []
    for c in candidates:
        word = c.context.word
        mass = c.context.predicted.projected_mass_kg
        assert mass is not None
        key = f"{stem}.d{word.pulse.duration_s:03d}.f{int(word.pulse.rate_kg_s * 1000):03d}.{word.pulse.word_id}"
        actions.append(
            AssessedFiniteAction(
                key, word.word, mass / D("2.10"), c.study, c.audits, c.services, c.observation
            )
        )
    proposal, committed = rank_finite_actions(
        actions=tuple(actions),
        stem=stem,
        chart=CHART,
        config=replace(ranker_config(), config_id="reactor-staged-pulse-response.feed-effort"),
        publisher=publisher,
        commit_nonattempt=commit_nonattempt,
    )
    if committed is None:
        return proposal, None
    chosen = next(c for c, a in zip(candidates, actions, strict=True) if a == committed.assessed)
    return proposal, ClassicalPreparedRequest(chosen, committed.compiled, committed.commitment)
