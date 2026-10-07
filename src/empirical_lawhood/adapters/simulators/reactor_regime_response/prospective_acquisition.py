"""One D root's precommitted, separate native cooling branches.

This is an adapter component. The public D task and its immutable output
contract must bind it before any fresh D source contact is authorized.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal as D
from typing import Callable

import numpy as np

from empirical_lawhood.adapters.methods.law_assessment import CandidatePayloadReader
from empirical_lawhood.adapters.methods.reactor_regime_response.control_math import REQUESTS_K
from empirical_lawhood.adapters.methods.reactor_regime_response.control_owner import ControlRecordPublisher, PreparedReactorRequest, prepare_request
from empirical_lawhood.adapters.methods.reactor_regime_response.control_prospective_plan import ReactorRegimeResponsePreparedProspectivePlanBundle
from empirical_lawhood.adapters.methods.reactor_regime_response.control_prospective_lock import RegimeDPreparedRequestLock, freeze_prepared_request
from empirical_lawhood.adapters.methods.reactor_regime_response.law_terminal import RegimeJointLawResult
from empirical_lawhood.adapters.methods.reactor_regime_response.prospective_decision import CausalValidityRegimeAssignment
from empirical_lawhood.adapters.methods.reactor_regime_response.records import RegimeCausalPreparation, RegimePrivatePreparation
from empirical_lawhood.adapters.simulators.reactor_causal_response.acquisition import NativeEpisode
from empirical_lawhood.adapters.simulators.reactor_causal_response.design import draw_scenario
from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_design import ReactorBatchSource
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.runtime.controller_runtime import DeliveryControllerTickReceipt, TickDisposition
from empirical_lawhood.runtime.artifacts import CanonicalTaskReceipt
from empirical_lawhood.runtime.controller_evaluation_nested import (
    DurablePreparedExecutionEventStore,
    ProspectiveEvaluationBindingCoordinator,
)

from .acquisition import _episode
from .committed_branch import acquire_committed_nominal_branch, acquire_committed_refined_replay, acquire_prepared_evaluator_chart


class _UnboundSession:
    def advance(self, command: object) -> None:
        raise RuntimeError("precontact admission commitment attempted a native delivery")


@dataclass(frozen=True)
class ProspectiveRootAcquisition:
    root: str
    prepared: tuple[PreparedReactorRequest | None, ...]
    nominal: tuple[NativeEpisode | None, ...]
    refined: tuple[NativeEpisode | None, ...]
    evaluator_chart: tuple[NativeEpisode, ...]
    ticks: tuple[DeliveryControllerTickReceipt | None, ...]
    native_calls: int
    reasons: tuple[str, ...]
    locks: tuple[RegimeDPreparedRequestLock, ...] = ()

    def __post_init__(self) -> None:
        if (
            len(self.prepared) != 4
            or len(self.nominal) != 4
            or len(self.refined) != 4
            or len(self.ticks) != 4
            or len(self.evaluator_chart) not in (0, 6)
            or self.native_calls != sum(value is not None for value in (*self.nominal, *self.refined))
            + len(self.evaluator_chart)
            or self.native_calls > 14
            or len(self.locks) not in (0, 4)
        ):
            raise ValueError("D action changes its four branches or six-word-view chart")


def acquire_prospective_root(
    *,
    source: ReactorBatchSource,
    causal: RegimeCausalPreparation,
    private: RegimePrivatePreparation,
    assignment: CausalValidityRegimeAssignment,
    report: RegimeJointLawResult,
    prospective_plan: ReactorRegimeResponsePreparedProspectivePlanBundle,
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
) -> ProspectiveRootAcquisition:
    "Commit all four requests first, then perform each independent branch.\n\n    Only the installed finite admission delivery port may supply a nominal primary\n    action. The complete chart remains postdecision evaluator evidence.\n    "
    root = causal.root
    if (
        causal.role != "prospective"
        or (assignment.root, assignment.seed) != (root, causal.seed)
        or assignment.causal_preparation
        != ObjectIdentity.from_record(f"{root}.causal-preparation", causal)
        or private.causal_preparation != assignment.causal_preparation
        or source.fingerprint() != causal.source_sha256
    ):
        raise ValueError("D native root changed its released assignment or source")
    assigned = next(item for item in prospective_plan.assigned_roots if item.root == root)
    decision = assignment.decision
    if assigned.eligible != (decision is not None and decision.causal_preparation_valid):
        raise ValueError("D prepared census changed this root's admission eligibility")
    if not assigned.eligible:
        return ProspectiveRootAcquisition(
            root, (None,) * 4, (None,) * 4, (None,) * 4, (),
            (None,) * 4, 0, assigned.reason_codes,
        )
    if prospective_plan.plan is None or decision is None:
        raise ValueError("eligible D root lacks its frozen controller-use plan or decision")
    if authority.object_schema != 'empirical-lawhood/planning/study-operation-authority':
        raise ValueError("D native acquisition lacks its separate execution authority")
    scenario = draw_scenario(root, "heldout", causal.seed)
    if assigned.scenario != ObjectIdentity.from_record(scenario.unit_id, scenario):
        raise ValueError("D source changed its shared native scenario realization")
    prefix = "exploration_unshifted" if decision.route == "prepared_t0" else decision.route[0]
    causal_arrays, private_arrays = causal.arrays.unpack(), private.arrays.unpack()
    nominal_base = _episode(root, f"{prefix}_v0", causal_arrays, private_arrays)
    refined_base = _episode(root, f"{prefix}_v1", causal_arrays, private_arrays)
    if nominal_base is None:
        raise ValueError("eligible D decision lost its nominal causal prefix")

    # The inert session fails if a commitment path touches the source before
    # every request has a durable control record. The nominal owner replaces it
    # only at that branch's exact callback.
    prepared = tuple(
        prepare_request(
            report=report,
            causal=causal,
            decision=decision,
            request_K=D(repr(request)),
            publisher=publisher,
            reader=reader,
            session=_UnboundSession(),  # type: ignore[arg-type]
            calibration_artifact=calibration_artifact,
            authority=decision.issue_authority,
            resources=resources,
            evaluation=prospective_plan.plan,
        )
        for request in REQUESTS_K
    )
    locks = tuple(
        freeze_prepared_request(
            root=root, request_index=index, prepared=request,
            causal=causal, causal_receipt=causal_receipt,
            causal_artifact=causal_artifact,
            issued_study=issued_study,
            evaluation=prospective_plan.plan, store=prepared_store,
            occurred_at_utc=occurred_at_utc,
        )
        for index, request in enumerate(prepared)
    )
    if (nominal_base.requests.shape != (2880, 2)
            or not np.isfinite(nominal_base.requests).all()):
        # The decisions and their locks survive. An unavailable exploration-unshifted tail cannot
        # retrospectively remove a contact or authorize an invented policy.
        return ProspectiveRootAcquisition(
            root, prepared, (None,) * 4, (None,) * 4, (), (None,) * 4, 0,
            ("DECLARED_DONOR_CONTINUATION_UNAVAILABLE",), locks,
        )
    choices = decision.policies[0].choices
    coordinator = ProspectiveEvaluationBindingCoordinator(
        prepared_store=prepared_store
    )
    nominal: list[NativeEpisode | None] = []
    refined: list[NativeEpisode | None] = []
    ticks: list[DeliveryControllerTickReceipt | None] = []
    calls = 0

    def report_progress(value: int, *, offset: int) -> None:
        if progress is not None:
            progress(offset * 2880 + value)

    def episode_progress(offset: int) -> Callable[[int], None]:
        def emit(value: int) -> None:
            report_progress(value, offset=offset)
        return emit

    for index, (choice, request) in enumerate(zip(choices, prepared, strict=True)):
        if choice is None:
            nominal.append(None)
            refined.append(None)
            refused = request.deliver(publisher)
            if refused.disposition is not TickDisposition.NONATTEMPT:
                raise ValueError("refused D request generated an owner native action")
            coordinator.store_prepared_delivery(
                prefix_id=locks[index].prefix_id, tick=refused
            )
            ticks.append(refused)
            continue
        result, owner = acquire_committed_nominal_branch(
            source=source,
            scenario=scenario,
            base=nominal_base,
            callback=decision.callback,
            request_index=index,
            prepared=request,
            publisher=publisher,
            progress=episode_progress(calls),
        )
        calls += 1
        nominal.append(result)
        if owner.tick is not None:
            coordinator.store_prepared_delivery(
                prefix_id=locks[index].prefix_id, tick=owner.tick
            )
        ticks.append(owner.tick)
        replay = acquire_committed_refined_replay(
            source=source,
            scenario=scenario,
            base=nominal_base,
            refined_preparation=refined_base,
            callback=decision.callback,
            feed_kg_s=float(request.delivery.mapping.full_native.command.feed_kg_s),
            request_id=f"{root}.request-{index}",
            progress=episode_progress(calls),
        )
        calls += 1
        refined.append(replay)
    chart = acquire_prepared_evaluator_chart(
        source=source,
        scenario=scenario,
        nominal_preparation=nominal_base,
        refined_preparation=refined_base,
        callback=decision.callback,
        progress=episode_progress(calls),
    )
    calls += len(chart)
    return ProspectiveRootAcquisition(
        root, prepared, tuple(nominal), tuple(refined), chart,
        tuple(ticks), calls, (), locks,
    )
