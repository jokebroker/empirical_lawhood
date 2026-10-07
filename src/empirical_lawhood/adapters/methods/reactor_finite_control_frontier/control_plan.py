"One independently qualified pulse enters the unchanged finite admission owners."

from dataclasses import dataclass, replace
from decimal import Decimal as D
from functools import lru_cache
from empirical_lawhood.adapters.control.local_law_basis import qualified_local_basis

from empirical_lawhood.adapters.geometry.services import AdmissionReceiptPlanValidator
from empirical_lawhood.adapters.methods.contracts import (
    LawQualificationBatch,
)
from empirical_lawhood.adapters.control.finite_gates import bounded_response_gate
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.adapters.simulators.reactor_causal_response.delivery import coordinate
from empirical_lawhood.adapters.simulators.reactor_finite_control_frontier.words import PulseProjection, project_pulse
from empirical_lawhood.kernel.action_contracts import ActionDeliveryStage
from empirical_lawhood.kernel.admission import AdmissionGateKind
from empirical_lawhood.kernel.causal_contracts import ReceiverInterval
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.models import ModelSetSpec
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.status import ScientificStatus
from empirical_lawhood.kernel.time import CausalPhase, InformationCutoff
from empirical_lawhood.planning.evidence_geometry import ReceiptAdmissionActionFibre, ReceiptAdmissionPlannedCoordinate, ReceiptAdmissionReceiptProductionPlan, ReceiptAdmissionSupportCell
from empirical_lawhood.planning.finite_action_support import support_map_for_plan
from empirical_lawhood.planning.geometry import (
    BatchAtlasAssemblyResult,
)
from .causal import causal_operands
from .config import Pulse
from .law_terminal import FrontierLaw
from .records import FrontierContext
from .selection import FrontierUseRequest
from .science import CHART, HORIZON, RECEIVERS, frontier_system


@lru_cache(maxsize=72)
def control_basis(
    report: FrontierLaw,
) -> tuple[LawQualificationBatch, BatchAtlasAssemblyResult, ModelSetSpec]:
    system, q, stem = frontier_system(), report.qualification, report.payload.stem
    law = q.response_law
    if q.scientific_status is not ScientificStatus.SUPPORTED or law is None:
        raise ValueError("Admission requires this exact independently supported pulse")
    axis = report.family.axis_map.bindings[0]
    return qualified_local_basis(
        system=system,
        qualification=q,
        axis=axis,
        stem=stem,
        scope_description="One independently qualified finite pulse/window under its exact causal context.",
        nontransport_description="Other actions are separate local relations; they are not alternative models of this pulse.",
    )


@dataclass(frozen=True, slots=True)
class FrontierControlContext:
    report: FrontierLaw
    causal: FrontierContext
    request: FrontierUseRequest
    word: PulseProjection
    plan: ReceiptAdmissionReceiptProductionPlan


def control_context(
    *,
    report: FrontierLaw,
    causal: FrontierContext,
    request: FrontierUseRequest,
    producer: ObjectIdentity,
    resource: ObjectIdentity,
    authority: ObjectIdentity,
) -> FrontierControlContext:
    c = report.payload.bound.coordinate
    inputs = causal_operands(causal, report.payload.prepared_domain)
    if (
        inputs is None
        or (request.context, request.horizon_s) != (c.context, c.horizon_s)
        or causal.context != c.context
    ):
        raise ValueError("Admission changed its supported causal context/window")
    batch, atlas, models = control_basis(report)
    law = report.qualification.response_law
    assert law is not None and causal.callback is not None
    stem = f"{causal.root}.{request.request_id}.{c.pulse.word_id}"

    def mapping(pulse: Pulse) -> PulseProjection:
        return project_pulse(
            inputs.observation,
            inputs.previous,
            pulse,
            decision_id=f"{causal.root}.{causal.context}.{pulse.word_id}",
            history_id=f"{causal.root}.{causal.context}.causal",
            horizon_id=HORIZON,
        )

    word = mapping(c.pulse)
    fibre = ReceiptAdmissionActionFibre(
        f"{stem}.action",
        word.word,
        tuple(o.occurrence_id for o in word.word.occurrences),
        tuple(ActionDeliveryStage),
    )
    member = models.members[0]
    cell = ReceiptAdmissionSupportCell(CHART, CHART, CHART, (word.word.word_id,))
    planned = ReceiptAdmissionPlannedCoordinate(
        f"{stem}.coordinate",
        ObjectIdentity.from_record(member.binding_id, member),
        member.response_law,
        member.qualification_result,
        member.denominator_member_id,
        member.candidate_version_ids[0],
        member.qualification_view_ids,
        ObjectIdentity.from_record(fibre.action_binding_id, fibre),
        ObjectIdentity.from_record(cell.cell_id, cell),
    )
    time = D(causal.callback * 10)
    gates = tuple(
        sorted(
            (
                bounded_response_gate(
                    stem=stem,
                    kind=kind,
                    response_receiver=RECEIVERS[1],
                    safety_receiver=RECEIVERS[0],
                    response_minimum=NamedDecimal("response-minimum", request.request_K, "K"),
                    safety_maximum=NamedDecimal("safety-maximum", D("356.2"), "K"),
                    protected_interval=ReceiverInterval(coordinate(time), coordinate(time + 120)),
                    evaluator=law.evaluator,
                )
                for kind in AdmissionGateKind
            ),
            key=lambda g: g.predicate_id,
        )
    )
    system = frontier_system()
    plan = ReceiptAdmissionReceiptProductionPlan(
        f"{stem}.admission",
        atlas.atlas,
        ObjectIdentity.from_record(batch.batch_id, batch),
        models,
        member.denominator_member_id,
        (fibre,),
        (cell,),
        (planned,),
        gates,
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
        InformationCutoff(f"{stem}.preaction", "reactor-clock", CausalPhase.PRE_ACTION, time),
        OutcomeAccess.OUTCOME_BLIND,
        authority,
        EvidenceCeiling.ADMISSION,
        VisibilityCeiling.PROSPECTIVE,
    )
    plan = replace(
        plan, model_set=replace(models, extensions=(support_map_for_plan(plan).extension,))
    )
    AdmissionReceiptPlanValidator.validate(batch=batch, plan=plan)
    return FrontierControlContext(report, causal, request, word, plan)
