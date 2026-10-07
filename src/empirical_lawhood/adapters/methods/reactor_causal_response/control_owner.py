"""One causal callback through the ordinary author/compiler/runtime owners."""

from dataclasses import dataclass
from decimal import Decimal as D
from typing import Protocol
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity, EvidenceLink, EvidenceRelation
from empirical_lawhood.kernel.references import ArtifactIdentity, NamedDecimal
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.controller_study import FrozenFeedbackConsumer, ImplementationRole
from empirical_lawhood.planning.trajectory_controller_evaluation import TrajectoryControllerEvaluationPlan
from empirical_lawhood.runtime.controller_compiler import CompiledDeliveryControllerStudy
from empirical_lawhood.runtime.controller_evaluation_trajectory import TrajectoryCallbackLink
from empirical_lawhood.runtime.controller_runtime import RuntimeObservation, DeliveryControllerDecisionCommitment, DeliveryControllerTickReceipt
from empirical_lawhood.adapters.control.composition import ControllerStudyComposition
from empirical_lawhood.adapters.methods.law_assessment import CandidatePayloadReader
from empirical_lawhood.adapters.simulators.reactor_causal_response.delivery import ReactorProfileDeliveryPort, BatchDeliverySession, coordinate
from .config import EmpiricalRecipe, CONFIRMATION_ROOTS
from .science import empirical_system
from .terminal import EmpiricalQualificationResult
from .numerical import Decision, Observation
from .control_plan import control_law_context
from .control_prediction import EmpiricalCallbackInput, predict_table
from .control_corpus import control_corpus
from .control_causal import causal_bindings
from .control_study import control_study
from .control_services import implementation_payloads, control_services


class CallbackRecordPublisher(Protocol):
    def publish_record(self, object_id: str, record: CanonicalRecord) -> ArtifactIdentity: ...


def frozen_consumer(report: EmpiricalQualificationResult) -> FrozenFeedbackConsumer:
    recipe = EmpiricalRecipe()
    law = report.qualification.response_law
    if (
        law is None
        or report.payload.q is None
        or report.design != ObjectIdentity.from_record(recipe.config_id, recipe)
    ):
        raise ValueError("control requires its supported frozen recipe/law/calibration")
    return FrozenFeedbackConsumer(
        "reactor-empirical-frozen-consumer",
        report.design,
        (law.evaluator.payload,),
        ObjectIdentity.from_record("reactor-empirical-calibration-operands", report.operands),
        ObjectIdentity.from_record("reactor-empirical-fixed-selector", recipe),
        ObjectIdentity.from_record("reactor-empirical-native-profile-expansion", recipe),
    )


def trajectory_plan(report: EmpiricalQualificationResult) -> TrajectoryControllerEvaluationPlan:
    consumer = frozen_consumer(report)
    binding = next(
        b for b, _ in implementation_payloads() if b.role is ImplementationRole.OUTCOME_EVALUATOR
    )
    return TrajectoryControllerEvaluationPlan(
        'reactor-causal-response-confirmation-prospective-evaluation',
        ObjectIdentity.from_record(consumer.consumer_id, consumer),
        CONFIRMATION_ROOTS,
        2880,
        D(10),
        report.family.axis_map.qualification_view_ids,
        (
            NamedDecimal("reactor-peak-temperature", D(".01"), "K"),
            NamedDecimal("reactor-end-conversion", D(".0002"), "1"),
            NamedDecimal("reactor-dose", D(".000001"), "kg"),
        ),
        (
            NamedDecimal("reactor-peak-temperature", D(".26"), "K"),
            NamedDecimal("reactor-end-conversion", D(".0102"), "1"),
        ),
        NamedDecimal("reactor-peak-temperature", D("356.2"), "K"),
        (
            NamedDecimal("reactor-dose", D(repr(0.999 * 287.3)), "kg"),
            NamedDecimal("reactor-end-conversion", D(".98"), "1"),
        ),
        D("1.06"),
        D("1.04"),
        binding,
    )


@dataclass(frozen=True)
class CausalReactorPreparedUseCallback:
    compiled: CompiledDeliveryControllerStudy
    commitment: DeliveryControllerDecisionCommitment
    services: ControllerStudyComposition
    delivery: ReactorProfileDeliveryPort
    causal: EmpiricalCallbackInput
    link: TrajectoryCallbackLink

    def deliver(self, publisher: CallbackRecordPublisher) -> DeliveryControllerTickReceipt:
        tick = self.services.deliver_prepared_commitment(self.compiled, self.commitment)
        publisher.publish_record(tick.tick_id, tick)
        return tick


def prepare_callback(
    *,
    report: EmpiricalQualificationResult,
    root: str,
    observation: Observation,
    previous: tuple[float, float],
    decision: Decision,
    previous_tick: ObjectIdentity | None,
    publisher: CallbackRecordPublisher,
    reader: CandidatePayloadReader,
    session: BatchDeliverySession,
    calibration_artifact: ArtifactIdentity,
    authority: ObjectIdentity,
    resources: ObjectIdentity,
    evaluation: TrajectoryControllerEvaluationPlan | None = None,
) -> CausalReactorPreparedUseCallback:
    consumer = frozen_consumer(report)
    system = empirical_system(EmpiricalRecipe())
    stem = f"{root}.{int(observation.time) // 10:04d}"

    def decimal(x: float) -> D:
        return D(repr(float(x)))

    causal = EmpiricalCallbackInput(
        f"{stem}.causal",
        root,
        int(observation.time) // 10,
        (
            decimal(observation.time),
            decimal(observation.temperature),
            decimal(observation.jacket),
            decimal(observation.dose),
        ),
        (decimal(previous[0]), decimal(previous[1])),
        decision.history_digest,
        previous_tick,
        tuple(tuple(decimal(x) for x in c.features) for c in decision.candidates),
    )
    causal_artifact = publisher.publish_record(causal.input_id, causal)
    link = TrajectoryCallbackLink(
        f"{stem}.link", root, causal.callback, previous_tick, causal_artifact
    )
    link_artifact = publisher.publish_record(link.link_id, link)
    implementations = tuple(
        b
        for b, _ in implementation_payloads()
        if evaluation is not None or b.role is not ImplementationRole.OUTCOME_EVALUATOR
    )
    producer = ObjectIdentity.from_record("reactor-causal-response-admission-owner-binding", implementations[0])
    method = implementations[0].reference
    context = control_law_context(
        system=system,
        report=report,
        decision=decision,
        root=root,
        time=decimal(observation.time),
        method=method,
        producer=producer,
        resource=resources,
        authority=authority,
    )
    table = predict_table(context, report, causal, reader)
    table_artifact = publisher.publish_record(table.table_id, table)
    artifacts = tuple(
        sorted(
            (
                causal_artifact,
                link_artifact,
                table_artifact,
                calibration_artifact,
                report.candidate.payload_publication.artifact,
                method.payload,
            ),
            key=lambda a: a.artifact_id,
        )
    )
    evidence = (
        EvidenceLink(
            f"{stem}.causal-evidence",
            EvidenceRelation.DERIVED_FROM,
            ObjectIdentity.from_record(table.table_id, table),
            ObjectIdentity.from_record(causal.input_id, causal),
            tuple(a.artifact_id for a in artifacts),
            system.world.world_id,
            context.plan.information_cutoff.cutoff_id,
            OutcomeAccess.OUTCOME_BLIND,
            VisibilityCeiling.PROSPECTIVE,
            (VisibilityCeiling.PROSPECTIVE,),
            "Frozen prior law and complete causal callback inputs; no future receiver labels.",
        ),
    )
    corpus = control_corpus(
        context=context,
        causal=causal,
        table=table,
        consumer=consumer,
        calibration_artifact=calibration_artifact,
        source_artifacts=artifacts,
        evidence_links=evidence,
    )
    runtime_observation = RuntimeObservation(
        f"{stem}.observation",
        root,
        tuple(
            NamedDecimal(key, value, unit)
            for key, value, unit in sorted(
                (
                    ("observed-dose", causal.observation[3], "kg"),
                    ("observed-jacket", causal.observation[2], "K"),
                    ("observed-temperature", causal.observation[1], "K"),
                    ("callback-time", causal.observation[0], "s"),
                )
            )
        ),
        coordinate(decimal(observation.time)),
        artifacts,
        OutcomeAccess.OUTCOME_BLIND,
    )
    programme = control_study(
        system=system,
        corpus=corpus,
        action_bindings=causal_bindings(context, report, evidence),
        implementations=implementations,
        observation=runtime_observation,
        frozen_recipe=ObjectIdentity.from_record(consumer.consumer_id, consumer),
        compiler_release_id='reactor-causal-response-finite-law-compiler',
        decision_budget_seconds=D(1),
        prospective_evaluation=None
        if evaluation is None
        else ObjectIdentity.from_record(evaluation.evaluation_plan_id, evaluation),
    )
    # Selection belongs to the compiler. The delivery port resolves only the
    # eventual exact compiled word; it cannot optimize or replace that choice.
    delivery = ReactorProfileDeliveryPort(
        next(b for b in implementations if b.role is ImplementationRole.DELIVERY),
        session,
        context.word_maps[decision.selected if decision.selected is not None else 0],
    )
    services = control_services(programme, evaluation, delivery)
    compiled = services.compile(programme)
    publisher.publish_record(compiled.compiled_study_id, compiled)
    commitment = services.prepare_commitment(
        compiled, runtime_observation, commitment_coordinate=runtime_observation.coordinate
    )
    expected = None if decision.selected is None else context.word_maps[decision.selected].word
    if (
        None if commitment.action_binding is None else commitment.action_binding.action_word
    ) != expected:
        raise ValueError("ordinary owner selection differs from frozen numerical/export selector")
    publisher.publish_record(commitment.commitment_id, commitment)
    return CausalReactorPreparedUseCallback(compiled, commitment, services, delivery, causal, link)
