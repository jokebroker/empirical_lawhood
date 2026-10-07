"Prepare one reactor cooling request through the installed finite admission owner."

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal as D
from typing import Protocol

from empirical_lawhood.adapters.control.composition import ControllerStudyComposition
from empirical_lawhood.adapters.methods.law_assessment import CandidatePayloadReader
from empirical_lawhood.adapters.simulators.reactor_causal_response.delivery import BatchDeliverySession, coordinate
from empirical_lawhood.adapters.simulators.reactor_regime_response.feed_word import ReactorScalarFeedDeliveryPort
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import EvidenceLink, EvidenceRelation, ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity, NamedDecimal
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.controller_study import FrozenFeedbackConsumer, ImplementationRole
from empirical_lawhood.planning.nested_controller_evaluation import CoupledRealizationControllerEvaluationPlan
from empirical_lawhood.planning.finite_response_geometry import FiniteResponseSet, FiniteTaskFunctionalSpec
from empirical_lawhood.runtime.controller_compiler import CompiledDeliveryControllerStudy
from empirical_lawhood.runtime.controller_runtime import DeliveryControllerDecisionCommitment, DeliveryControllerTickReceipt, RuntimeObservation

from .control_causal import causal_bindings
from empirical_lawhood.adapters.methods.reactor_regime_response.control_clock import ReactorLocalClockMap
from .control_corpus import control_corpus
from .control_plan import control_context
from .control_prediction import predict_control_table
from empirical_lawhood.adapters.control.finite_campaign import finite_control_study
from .control_services import control_services, implementation_payloads, implementation_configurations
from .law_terminal import ClassicalLaw
from .records import ClassicalDecision
from .records import ClassicalCausal
from .science import classical_system


class ControlRecordPublisher(Protocol):
    def publish_record(self, object_id: str, record: CanonicalRecord) -> ArtifactIdentity: ...


def frozen_consumer(report: ClassicalLaw) -> FrozenFeedbackConsumer:
    law = report.qualification.response_law
    if law is None or report.payload.route is None:
        raise ValueError("reactor consumer requires its supported selected joint law")
    design = report.family.members[0].config
    return FrozenFeedbackConsumer(
        "reactor-classical-selected-action-consumer",
        design,
        (law.evaluator.payload,),
        report.payload.calibration,
        report.payload.nomination,
        design,
    )


@dataclass(frozen=True)
class PreparedClassicalRequest:
    compiled: CompiledDeliveryControllerStudy
    commitment: DeliveryControllerDecisionCommitment
    services: ControllerStudyComposition
    delivery: ReactorScalarFeedDeliveryPort
    request_K: D
    task: FiniteTaskFunctionalSpec
    audit_forecasts: tuple[FiniteResponseSet, ...]

    def __post_init__(self) -> None:
        if len(
            self.audit_forecasts
        ) != 1 or self.commitment.instance_binding.task_functional != ObjectIdentity.from_record(
            self.task.functional_id, self.task
        ):
            raise ValueError("prepared reactor request lost its exact controller use task and audit chart")

    def deliver(self, publisher: ControlRecordPublisher) -> DeliveryControllerTickReceipt:
        tick = self.services.deliver_prepared_commitment(self.compiled, self.commitment)
        publisher.publish_record(tick.tick_id, tick)
        return tick


def prepare_request(
    *,
    report: ClassicalLaw,
    causal: ClassicalCausal,
    decision: ClassicalDecision,
    request_K: D,
    publisher: ControlRecordPublisher,
    reader: CandidatePayloadReader,
    session: BatchDeliverySession,
    calibration_artifact: ArtifactIdentity,
    authority: ObjectIdentity,
    resources: ObjectIdentity,
    evaluation: CoupledRealizationControllerEvaluationPlan,
) -> PreparedClassicalRequest:
    """Seal corpus, compile and commit before this request's native branch."""
    if (
        request_K != report.payload.design.request_K
        or not decision.causal_preparation_valid
        or calibration_artifact.sha256 != report.payload.calibration.object_fingerprint
    ):
        raise ValueError("reactor request lacks its safe frozen C calibration")
    implementations = tuple(binding for binding, _ in implementation_payloads())
    for configuration in implementation_configurations():
        artifact = publisher.publish_record(configuration.config_id, configuration)
        expected = next(
            binding.reference.payload
            for binding in implementations
            if binding.role is configuration.role
        )
        if artifact != expected:
            raise ValueError("classical controller implementation configuration custody differs")
    producer = ObjectIdentity.from_record("reactor-selected-action-response-admission-owner-binding", implementations[0])
    consumer = frozen_consumer(report)
    context = control_context(
        report=report,
        causal=causal,
        decision=decision,
        producer=producer,
        resource=resources,
        authority=authority,
    )
    table = predict_control_table(context, reader)
    assert decision.callback is not None
    clock_map = ReactorLocalClockMap(
        f"{causal.root}.callback-{decision.callback:04d}.local-clock",
        causal.root,
        decision.callback,
        D(decision.callback * 10),
    )
    causal_artifact = publisher.publish_record(f"{causal.root}.causal-preparation", causal)
    decision_artifact = publisher.publish_record(decision.decision_id, decision)
    table_artifact = publisher.publish_record(table.table_id, table)
    clock_artifact = publisher.publish_record(clock_map.map_id, clock_map)
    method = context.plan.gate_predicates[0].evaluator
    artifacts = tuple(
        sorted(
            {
                value.artifact_id: value
                for value in (
                    causal_artifact,
                    decision_artifact,
                    table_artifact,
                    clock_artifact,
                    calibration_artifact,
                    report.candidate.payload_publication.artifact,
                    method.payload,
                )
            }.values(),
            key=lambda value: value.artifact_id,
        )
    )
    evidence = (
        EvidenceLink(
            f"{context.plan.plan_id}.causal-evidence",
            EvidenceRelation.DERIVED_FROM,
            ObjectIdentity.from_record(table.table_id, table),
            ObjectIdentity.from_record(f"{causal.root}.causal-preparation", causal),
            tuple(value.artifact_id for value in artifacts),
            classical_system().world.world_id,
            context.plan.information_cutoff.cutoff_id,
            OutcomeAccess.OUTCOME_BLIND,
            VisibilityCeiling.PROSPECTIVE,
            (VisibilityCeiling.PROSPECTIVE,),
            "Frozen law qualification law, exact selected causal prefix and the selected finite feed-word interval.",
        ),
    )
    corpus = control_corpus(
        context=context,
        table=table,
        consumer=consumer,
        calibration_artifact=calibration_artifact,
        source_artifacts=artifacts,
        evidence_links=evidence,
    )
    publisher.publish_record(corpus.corpus_id, corpus)
    prefix = "exploration_unshifted" if decision.route == "prepared_t0" else decision.route[0]
    observation = causal.arrays.unpack()[f"{prefix}_v0_observations"][decision.callback]
    moment = D(repr(float(observation[0])))
    runtime_observation = RuntimeObservation(
        f"{context.plan.plan_id}.observation",
        causal.root,
        tuple(
            sorted(
                (
                    NamedDecimal("callback-time", moment, "s"),
                    NamedDecimal("observed-dose", D(repr(float(observation[3]))), "kg"),
                    NamedDecimal("observed-jacket", D(repr(float(observation[2]))), "K"),
                    NamedDecimal("observed-temperature", D(repr(float(observation[1]))), "K"),
                ),
                key=lambda value: value.value_id,
            )
        ),
        coordinate(moment),
        artifacts,
        OutcomeAccess.OUTCOME_BLIND,
    )
    publisher.publish_record(runtime_observation.observation_id, runtime_observation)
    programme = finite_control_study(
        system=classical_system(),
        corpus=corpus,
        action_bindings=causal_bindings(context, report, evidence),
        implementations=implementations,
        observation=runtime_observation,
        frozen_recipe=ObjectIdentity.from_record(consumer.consumer_id, consumer),
        compiler_release_id='reactor-selected-action-response-finite-law-compiler',
        decision_budget_seconds=D(1),
        prospective_evaluation=ObjectIdentity.from_record(
            evaluation.evaluation_plan_id, evaluation
        ),
    )
    publisher.publish_record(programme.study_id, programme)
    delivery = ReactorScalarFeedDeliveryPort(
        next(binding for binding in implementations if binding.role is ImplementationRole.DELIVERY),
        session,
        context.word_maps[1],
    )
    services = control_services(programme, evaluation, delivery)
    compiled = services.compile(programme)
    publisher.publish_record(compiled.compiled_study_id, compiled)
    commitment = services.prepare_commitment(
        compiled, runtime_observation, commitment_coordinate=runtime_observation.coordinate
    )
    actual_word = (
        None if commitment.action_binding is None else commitment.action_binding.action_word
    )
    if actual_word is not None and actual_word != context.word_maps[1].scalar_word:
        raise ValueError("classical admission owner selected another native word")
    publisher.publish_record(commitment.commitment_id, commitment)
    audit_forecasts = tuple(
        receipt.request.response_set for receipt in corpus.reachability_receipts
    )
    if len(audit_forecasts) != 1 or any(
        not value.evaluation_bindings
        or any(
            binding.action_word
            != ObjectIdentity.from_record(word.scalar_word.word_id, word.scalar_word)
            for binding in value.evaluation_bindings
        )
        for value, word in zip(audit_forecasts, context.word_maps[1:], strict=True)
    ):
        raise ValueError("D controller use audit forecasts changed the frozen selected positive native word")
    return PreparedClassicalRequest(
        compiled,
        commitment,
        services,
        delivery,
        request_K,
        corpus.reachability_receipts[0].request.task,
        audit_forecasts,
    )
