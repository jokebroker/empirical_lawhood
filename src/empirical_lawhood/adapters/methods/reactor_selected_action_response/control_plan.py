"One selected native word enters the existing noncompensating admission owners."

from dataclasses import dataclass, replace
from decimal import Decimal as D

from empirical_lawhood.adapters.geometry.services import AdmissionReceiptPlanValidator
from empirical_lawhood.adapters.methods.reactor_regime_response.control_plan import _gate
from empirical_lawhood.adapters.simulators.reactor_causal_response.interface import Actuator
from empirical_lawhood.adapters.simulators.reactor_regime_response.feed_word import ReactorFeedOnlyWordMap
from empirical_lawhood.kernel.action_contracts import ActionDeliveryStage
from empirical_lawhood.kernel.admission import AdmissionGateKind
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.time import CausalPhase, InformationCutoff
from empirical_lawhood.planning.evidence_geometry import ReceiptAdmissionActionFibre, ReceiptAdmissionPlannedCoordinate, ReceiptAdmissionReceiptProductionPlan, ReceiptAdmissionSupportCell
from empirical_lawhood.planning.finite_action_support import support_map_for_plan
from .causal import inputs
from .control_basis import classical_control_basis
from .law_terminal import ClassicalLaw, RESULT_ID
from .records import ClassicalCausal, ClassicalDecision
from .science import classical_system


@dataclass(frozen=True, slots=True)
class ClassicalControlContext:
    report: ClassicalLaw
    causal: ClassicalCausal
    decision: ClassicalDecision
    request_K: D
    word_maps: tuple[ReactorFeedOnlyWordMap, ...]
    plan: ReceiptAdmissionReceiptProductionPlan


def control_context(
    *,
    report: ClassicalLaw,
    causal: ClassicalCausal,
    decision: ClassicalDecision,
    producer: ObjectIdentity,
    resource: ObjectIdentity,
    authority: ObjectIdentity,
) -> ClassicalControlContext:
    if (
        not decision.causal_preparation_valid
        or decision.root != causal.root
        or decision.causal_preparation
        != ObjectIdentity.from_record(f"{causal.root}.causal-preparation", causal)
        or decision.qualified_law != ObjectIdentity.from_record(RESULT_ID, report)
        or decision.issue_authority != authority
    ):
        raise ValueError("classical admission lacks the exact causal handoff, qualified law or authority")
    _, _, observation, previous = inputs(causal)
    callback = causal.callback
    assert callback is not None
    if callback != decision.callback:
        raise ValueError("classical admission changed callback")
    batch, assembly, models = classical_control_basis(report)
    law = report.qualification.response_law
    assert law is not None
    axis = report.family.axis_map.bindings[0]
    member = models.members[0]
    stem = f"{causal.root}.{callback:04d}.selected-action"
    maps = tuple(
        ReactorFeedOnlyWordMap.from_projection(
            Actuator().project_request(observation, previous, (feed, previous[1]), 1 + 3 * i),
            decision_id=f"{causal.root}.{callback:04d}.word-{i}",
            history_id=f"{causal.root}.selected-causal-prefix",
        )
        for i, feed in enumerate((0.0, 0.016))
    )
    if any(
        abs(sum(dt * feed for _, dt, feed, _ in m.full_native.exposure) - target) > 1e-12
        for m, target in zip(maps, (D(0), D(".16")), strict=True)
    ):
        raise ValueError("classical admission native word has another delivered mass")
    fibre = ReceiptAdmissionActionFibre(
        f"{stem}.word-1",
        maps[1].scalar_word,
        tuple(o.occurrence_id for o in maps[1].scalar_word.occurrences),
        tuple(ActionDeliveryStage),
    )
    cell = ReceiptAdmissionSupportCell(
        axis.denominator_member_id,
        law.chart_id,
        axis.denominator_member_id,
        (maps[1].scalar_word.word_id,),
    )
    coordinate = ReceiptAdmissionPlannedCoordinate(
        f"{stem}.word-1.coordinate",
        ObjectIdentity.from_record(member.binding_id, member),
        member.response_law,
        member.qualification_result,
        member.denominator_member_id,
        member.candidate_version_ids[0],
        member.qualification_view_ids,
        ObjectIdentity.from_record(fibre.action_binding_id, fibre),
        ObjectIdentity.from_record(cell.cell_id, cell),
    )
    cutoff = InformationCutoff(
        f"{stem}.preaction", "reactor-clock", CausalPhase.PRE_ACTION, D(callback * 10)
    )
    system = classical_system()
    request = report.payload.design.request_K
    plan = ReceiptAdmissionReceiptProductionPlan(
        f"{stem}.admission",
        assembly.atlas,
        ObjectIdentity.from_record(batch.batch_id, batch),
        models,
        member.denominator_member_id,
        (fibre,),
        (cell,),
        (coordinate,),
        tuple(
            sorted(
                (
                    _gate(stem, kind, request, D(callback * 10), law.evaluator)
                    for kind in AdmissionGateKind
                ),
                key=lambda v: v.predicate_id,
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
    support = support_map_for_plan(plan)
    plan = replace(plan, model_set=replace(models, extensions=(support.extension,)))
    AdmissionReceiptPlanValidator.validate(batch=batch, plan=plan)
    return ClassicalControlContext(report, causal, decision, request, maps, plan)
