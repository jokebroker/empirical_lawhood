"One native owner delivery after durable admission/controller-use commitment, then separate audits."

from dataclasses import dataclass
from typing import Callable

from empirical_lawhood.adapters.methods.law_assessment import CandidatePayloadReader
from empirical_lawhood.adapters.methods.reactor_selected_action_response.control_owner import ControlRecordPublisher, PreparedClassicalRequest, prepare_request
from empirical_lawhood.adapters.methods.reactor_selected_action_response.control_prospective_lock import PreparedForecastLock, freeze_prepared_request
from empirical_lawhood.adapters.methods.reactor_selected_action_response.control_prospective_plan import ReactorSelectedActionResponseProspectivePlan
from empirical_lawhood.adapters.methods.reactor_selected_action_response.law_terminal import ClassicalLaw
from empirical_lawhood.adapters.methods.reactor_selected_action_response.records import ClassicalCausal, ClassicalDecision, ClassicalPrivate
from empirical_lawhood.adapters.simulators.reactor_causal_response.acquisition import NativeEpisode, TapeController, acquire_episode
from empirical_lawhood.adapters.simulators.reactor_causal_response.design import draw_scenario
from empirical_lawhood.adapters.simulators.reactor_regime_response.committed_branch import acquire_committed_nominal_branch, acquire_committed_refined_replay
from empirical_lawhood.adapters.simulators.reactor_regime_response.panel import assay_tape
from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_design import ReactorBatchSource
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.runtime.artifacts import CanonicalTaskReceipt
from empirical_lawhood.runtime.controller_evaluation_nested import (
    DurablePreparedExecutionEventStore,
    ProspectiveEvaluationBindingCoordinator,
)
from empirical_lawhood.runtime.controller_runtime import DeliveryControllerTickReceipt, CommitmentDisposition
from .acquisition import donor_episode


class UnboundSession:
    def advance(self, command: object) -> None:
        raise RuntimeError("classical precontact commitment attempted native delivery")


@dataclass(frozen=True)
class ClassicalRootAcquisition:
    root: str
    prepared: tuple[PreparedClassicalRequest | None, ...]
    nominal: tuple[NativeEpisode | None, ...]
    refined: tuple[NativeEpisode | None, ...]
    evaluator_chart: tuple[NativeEpisode, ...]
    ticks: tuple[DeliveryControllerTickReceipt | None, ...]
    native_calls: int
    reasons: tuple[str, ...]
    locks: tuple[PreparedForecastLock, ...] = ()

    def __post_init__(self) -> None:
        if (
            any(len(x) != 1 for x in (self.prepared, self.nominal, self.refined, self.ticks))
            or len(self.evaluator_chart) not in (0, 4)
            or len(self.locks) not in (0, 1)
            or self.native_calls
            != sum(x is not None for x in (*self.nominal, *self.refined))
            + len(self.evaluator_chart)
        ):
            raise ValueError(
                "classical acquisition changed its one request and reference/audit census"
            )


def acquire_prospective_root(
    *,
    source: ReactorBatchSource,
    causal: ClassicalCausal,
    private: ClassicalPrivate,
    assignment: ClassicalDecision,
    report: ClassicalLaw,
    prospective_plan: ReactorSelectedActionResponseProspectivePlan,
    calibration_artifact: ArtifactIdentity,
    publisher: ControlRecordPublisher,
    reader: CandidatePayloadReader,
    authority: ObjectIdentity,
    resources: ObjectIdentity,
    causal_receipt: CanonicalTaskReceipt,
    causal_artifact: ArtifactIdentity,
    issued_study: ObjectIdentity,
    prepared_store: DurablePreparedExecutionEventStore,
    occurred_at_utc: str,
    progress: Callable[[int], None] | None = None,
    acquire: Callable[..., NativeEpisode] = acquire_episode,
) -> ClassicalRootAcquisition:
    root = causal.root
    if (
        causal.role != "prospective"
        or assignment.root != root
        or assignment.causal_preparation
        != ObjectIdentity.from_record(f"{root}.causal-preparation", causal)
        or private.causal_preparation != assignment.causal_preparation
        or source.fingerprint() != causal.source_sha256
        or authority.object_schema != 'empirical-lawhood/planning/study-operation-authority'
    ):
        raise ValueError("classical prospective source or execution authority differs")
    assigned = next(row for row in prospective_plan.assigned_roots if row.root == root)
    if assigned.eligible != assignment.causal_preparation_valid:
        raise ValueError("classical controller use changed its sealed root eligibility")
    if not assigned.eligible:
        return ClassicalRootAcquisition(
            root, (None,), (None,), (None,), (), (None,), 0, assigned.reason_codes
        )
    plan = prospective_plan.plan
    callback = assignment.callback
    assert plan is not None and callback is not None
    scenario = draw_scenario(root, "heldout", causal.seed)
    if assigned.scenario != ObjectIdentity.from_record(scenario.unit_id, scenario):
        raise ValueError("classical controller use native realization differs")
    nominal_base, refined_base = (donor_episode(causal, private, view) for view in (0, 1))
    prepared = prepare_request(
        report=report,
        causal=causal,
        decision=assignment,
        request_K=report.payload.design.request_K,
        publisher=publisher,
        reader=reader,
        session=UnboundSession(),  # type: ignore[arg-type]
        calibration_artifact=calibration_artifact,
        authority=assignment.issue_authority,
        resources=resources,
        evaluation=plan,
    )
    lock = freeze_prepared_request(
        root=root,
        request_index=0,
        prepared=prepared,
        causal=causal,
        causal_receipt=causal_receipt,
        causal_artifact=causal_artifact,
        issued_study=issued_study,
        evaluation=plan,
        store=prepared_store,
        occurred_at_utc=occurred_at_utc,
    )
    coordinator = ProspectiveEvaluationBindingCoordinator(prepared_store=prepared_store)
    calls = 0

    def episode_progress(offset: int) -> Callable[[int], None]:
        def emit(value: int) -> None:
            if progress is not None:
                progress(offset * 2880 + value)

        return emit

    nominal = refined = None
    if prepared.commitment.disposition is CommitmentDisposition.NONATTEMPT:
        tick = prepared.deliver(publisher)
    else:
        nominal, owner = acquire_committed_nominal_branch(
            source=source,
            scenario=scenario,
            base=nominal_base,
            callback=callback,
            request_index=0,
            prepared=prepared,
            publisher=publisher,
            acquire=acquire,
            progress=episode_progress(calls),
        )
        calls += 1
        tick = owner.tick
        refined = acquire_committed_refined_replay(
            source=source,
            scenario=scenario,
            base=nominal_base,
            refined_preparation=refined_base,
            callback=callback,
            feed_kg_s=0.016,
            request_id=f"{root}.request-0",
            acquire=acquire,
            progress=episode_progress(calls),
        )
        calls += 1
    if tick is not None:
        coordinator.store_prepared_delivery(prefix_id=lock.prefix_id, tick=tick)
    chart = []
    for word, feed in enumerate((0.0, 0.016)):
        for view, base in enumerate((nominal_base, refined_base)):
            chart.append(
                acquire(
                    source,
                    scenario,
                    f"{root}.evaluator-word-{word}.view-{view}",
                    TapeController(assay_tape(base, callback, feed)),
                    dt=base.dt,
                    progress=episode_progress(calls),
                )
            )
            calls += 1
    return ClassicalRootAcquisition(
        root, (prepared,), (nominal,), (refined,), tuple(chart), (tick,), calls, (), (lock,)
    )
