"One request-specific admission plan from an unchanged prepared root and local law."

from __future__ import annotations

from empirical_lawhood.adapters.control.finite_gates import bounded_response_gate

from dataclasses import dataclass, replace
from decimal import Decimal as D

import numpy as np

from empirical_lawhood.adapters.geometry.services import AdmissionReceiptPlanValidator
from empirical_lawhood.adapters.methods.reactor_causal_response.numerical import Observation
from empirical_lawhood.adapters.simulators.reactor_causal_response.interface import Actuator
from empirical_lawhood.adapters.simulators.reactor_regime_response.feed_word import ReactorFeedOnlyWordMap
from empirical_lawhood.kernel.action_contracts import ActionDeliveryStage
from empirical_lawhood.kernel.admission import AdmissionGateKind
from empirical_lawhood.kernel.causal_contracts import (
    ReceiverInterval,
)
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.time import CausalPhase, InformationCutoff
from empirical_lawhood.planning.evidence_geometry import GatePredicateSpec, ReceiptAdmissionActionFibre, ReceiptAdmissionPlannedCoordinate, ReceiptAdmissionReceiptProductionPlan, ReceiptAdmissionSupportCell
from empirical_lawhood.planning.finite_action_support import support_map_for_plan
from empirical_lawhood.adapters.simulators.reactor_causal_response.delivery import coordinate

from .causal_contexts import causal_contexts
from .control_basis import regime_control_basis
from .control_math import REQUESTS_K
from .law_terminal import RESULT_ID, RegimeJointLawResult
from .prospective_decision import CausalValidityRegimeDecision
from .records import RegimeCausalPreparation
from .science import RECEIVERS, regime_system


@dataclass(frozen=True, slots=True)
class RegimeControlLawContext:
    report: RegimeJointLawResult
    causal: RegimeCausalPreparation
    decision: CausalValidityRegimeDecision
    request_K: D
    word_maps: tuple[ReactorFeedOnlyWordMap, ...]
    plan: ReceiptAdmissionReceiptProductionPlan


def _gate(
    stem: str,
    kind: AdmissionGateKind,
    request_K: D,
    time: D,
    evaluator: object,
) -> GatePredicateSpec:
    from empirical_lawhood.kernel.references import ExecutableReference

    if not isinstance(evaluator, ExecutableReference):
        raise TypeError("Admission gate lacks the exact qualified evaluator")
    return bounded_response_gate(
        stem=stem,
        kind=kind,
        response_receiver=RECEIVERS[1],
        safety_receiver=RECEIVERS[0],
        response_minimum=NamedDecimal("response-minimum", request_K, "K"),
        safety_maximum=NamedDecimal("safety-maximum", D("356.2"), "K"),
        protected_interval=ReceiverInterval(coordinate(time), coordinate(time + D(10))),
        evaluator=evaluator,
    )


def regime_control_law_context(
    *,
    report: RegimeJointLawResult,
    causal: RegimeCausalPreparation,
    decision: CausalValidityRegimeDecision,
    request_K: D,
    producer: ObjectIdentity,
    resource: ObjectIdentity,
    authority: ObjectIdentity,
) -> RegimeControlLawContext:
    "Build distinct admission request programmes from one exact causal prefix."
    if request_K not in tuple(D(repr(value)) for value in REQUESTS_K):
        raise ValueError("Admission request is outside the four frozen cooling targets")
    if not decision.causal_preparation_valid:
        raise ValueError("Admission cannot admit from an invalid or unsafe selected preparation")
    if (
        decision.root != causal.root
        or decision.causal_preparation
        != ObjectIdentity.from_record(f"{causal.root}.causal-preparation", causal)
        or decision.qualified_law != ObjectIdentity.from_record(RESULT_ID, report)
        or decision.route != report.payload.route
        or decision.issue_authority != authority
    ):
        raise ValueError("Admission changed the saved pre-outcome root, law or authority")
    system = regime_system()
    batch, assembly, models = regime_control_basis(report)
    law = report.qualification.response_law
    assert law is not None
    axis = report.family.axis_map.bindings[0]
    member = models.members[0]
    context = next(value for value in causal_contexts(causal) if value.name == decision.route)
    callback = context.callback
    if (
        callback is None
        or callback != decision.callback
        or context.projected_masses_kg is None
        or context.input_sha256 is None
    ):
        raise ValueError("Admission source lacks the exact selected causal callback")
    arrays = causal.arrays.unpack()
    prefix = "exploration_unshifted" if decision.route == "prepared_t0" else decision.route[0]
    observations = arrays[f"{prefix}_v0_observations"]
    stages = arrays[f"{prefix}_v0_stages"]
    if (
        not np.isfinite(observations[: callback + 1]).all()
        or not np.isfinite(stages[:callback]).all()
    ):
        raise ValueError("Admission selected native prefix has invalid causal operands")
    observation = Observation(*map(float, observations[callback]))
    previous = (float(stages[callback - 1, 2]), float(stages[callback - 1, 3]))
    stem = f"{causal.root}.{callback:04d}.request-{REQUESTS_K.index(float(request_K))}"
    maps = tuple(
        ReactorFeedOnlyWordMap.from_projection(
            Actuator().project_request(observation, previous, (feed, previous[1]), 1 + 3 * index),
            # All four tasks use one prepared three-word native chart.  The
            # request target changes the controller admission task, never the action identity.
            decision_id=f"{causal.root}.{callback:04d}.shared-word-{index}",
            history_id=f"{causal.root}.selected-causal-prefix",
        )
        for index, feed in enumerate((0.0, 0.016, 0.032))
    )
    masses = tuple(
        float(sum(dt * feed for _, dt, feed, _ in value.full_native.exposure)) for value in maps
    )
    if any(abs(a - b) > 1e-12 for a, b in zip(masses, context.projected_masses_kg, strict=True)):
        raise ValueError("Admission scalar word changed the sealed projected delivered masses")
    fibres = tuple(
        sorted(
            (
                ReceiptAdmissionActionFibre(
                    f"{stem}.word-{index}",
                    maps[index].scalar_word,
                    tuple(value.occurrence_id for value in maps[index].scalar_word.occurrences),
                    tuple(ActionDeliveryStage),
                )
                for index in (1, 2)
            ),
            key=lambda value: value.action_binding_id,
        )
    )
    cell = ReceiptAdmissionSupportCell(
        axis.denominator_member_id,
        law.chart_id,
        axis.denominator_member_id,
        tuple(sorted(maps[index].scalar_word.word_id for index in (1, 2))),
    )
    coordinates = tuple(
        ReceiptAdmissionPlannedCoordinate(
            f"{stem}.word-{index}.coordinate",
            ObjectIdentity.from_record(member.binding_id, member),
            member.response_law,
            member.qualification_result,
            member.denominator_member_id,
            member.candidate_version_ids[0],
            member.qualification_view_ids,
            ObjectIdentity.from_record(fibres[index - 1].action_binding_id, fibres[index - 1]),
            ObjectIdentity.from_record(cell.cell_id, cell),
        )
        for index in (1, 2)
    )
    cutoff = InformationCutoff(
        f"{stem}.preaction", "reactor-clock", CausalPhase.PRE_ACTION, D(callback * 10)
    )
    evaluator = law.evaluator
    plan = ReceiptAdmissionReceiptProductionPlan(
        f"{stem}.admission",
        assembly.atlas,
        ObjectIdentity.from_record(batch.batch_id, batch),
        models,
        member.denominator_member_id,
        fibres,
        (cell,),
        coordinates,
        tuple(
            sorted(
                (
                    _gate(stem, kind, request_K, D(callback * 10), evaluator)
                    for kind in AdmissionGateKind
                ),
                key=lambda value: value.predicate_id,
            )
        ),
        f"{stem}.initial",
        law.relation.horizon,
        (f"{stem}.constraint.reachability",),
        producer,
        producer,
        producer,
        resource,
        resource,
        resource,
        ObjectIdentity.from_record(system.world.world_id, system.world),
        cutoff,
        OutcomeAccess.OUTCOME_BLIND,
        authority,
        EvidenceCeiling.ADMISSION,
        VisibilityCeiling.PROSPECTIVE,
    )
    mapping = support_map_for_plan(plan)
    models = replace(models, extensions=(mapping.extension,))
    plan = replace(plan, model_set=models)
    AdmissionReceiptPlanValidator.validate(batch=batch, plan=plan)
    return RegimeControlLawContext(report, causal, decision, request_K, maps, plan)
