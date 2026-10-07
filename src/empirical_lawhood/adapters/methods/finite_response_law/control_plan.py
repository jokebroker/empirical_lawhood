"Bind retained finite response-law qualification to the existing atlas and finite admission owners."

from dataclasses import dataclass, replace
from decimal import Decimal as D

from empirical_lawhood.kernel.action_contracts import ActionDeliveryStage, ActionWordSupportStatus
from empirical_lawhood.kernel.admission import AdmissionGateKind
from empirical_lawhood.kernel.causal_contracts import (
    PredicateDirection,
    ReceiverInterval,
    TemporalPredicateSemantics,
)
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.identification import LawQualificationResult
from empirical_lawhood.kernel.models import ModelIntersectionSemantics, ModelMemberLawBinding, ModelSetSpec
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ExecutableReference, NamedDecimal
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.kernel.time import CausalPhase, ClockCoordinate, CoordinateOrigin, InformationCutoff
from empirical_lawhood.planning.evidence_geometry import GatePredicateKind, GatePredicateSpec, ReceiptAdmissionActionFibre, ReceiptAdmissionSupportCell, ReceiptAdmissionPlannedCoordinate, ReceiptAdmissionReceiptProductionPlan
from empirical_lawhood.planning.geometry import (
    AtlasDomainCell,
    QualificationBatchAtlasAssemblySpec,
    BatchAtlasAssemblyResult,
)
from empirical_lawhood.runtime.candidate_payloads import CandidatePayloadPublicationReceipt
from empirical_lawhood.adapters.methods.contracts import (
    LawCandidateAxisMap,
    LawQualificationBatch,
    LawQualificationBatchCoordinate,
    LawQualificationBatchCoordinateDeclaration,
    LawQualificationCoordinateDisposition,
)
from empirical_lawhood.adapters.geometry.services import AtlasAssembler, AdmissionReceiptPlanValidator
from .control_word import FiniteResponseLawControllerWordMap, controller_word_map
from .law_binding import HORIZON, NATIVE_WORDS, native_action_word, output_quantities
from .method_records import FiniteResponseLawQualificationReport
from .science import PROGRAMME
from empirical_lawhood.adapters.simulators.finite_response_law.roster import Q_CLOCK, Q_FRAME


def clock(value: int) -> ClockCoordinate:
    return ClockCoordinate(Q_CLOCK, D(value), "reference-tick", Q_FRAME, CoordinateOrigin.ABSOLUTE)


@dataclass(frozen=True, slots=True)
class FiniteResponseLawControlLawContext:
    system: SystemSpec
    qualification: LawQualificationResult
    axes: LawCandidateAxisMap
    publication: CandidatePayloadPublicationReceipt
    batch: LawQualificationBatch
    assembly: BatchAtlasAssemblyResult
    model_set: ModelSetSpec
    word_maps: tuple[FiniteResponseLawControllerWordMap, ...]
    plan: ReceiptAdmissionReceiptProductionPlan


def qualified_model_set(
    *, report: FiniteResponseLawQualificationReport, boundary: str, world_id: str
) -> ModelSetSpec:
    "Freeze the same exact finite response-law qualification member at both outer and inner controller boundaries."
    if boundary not in ("composed", "cached", "direct"):
        raise ValueError("Finite response-law evaluation model set requires a frozen pre-parent boundary")
    index = tuple(b.boundary for b in report.calibration.boundaries).index(boundary)
    qualification = report.qualifications[index]
    law = qualification.response_law
    axes = report.families[index].ledger.axis_map
    if law is None or len(axes.bindings) != 1:
        raise ValueError("Finite response-law evaluation cannot freeze an unavailable or substituted qualified law")
    axis = axes.bindings[0]
    stem = f"{PROGRAMME}.{boundary}.prospective-control"
    member = ModelMemberLawBinding(
        f"{stem}.member",
        axis.denominator_member_id,
        law.relation.relation_id,
        ObjectIdentity.from_record(law.law_id, law),
        ObjectIdentity.from_record(qualification.result_id, qualification),
        (axis.candidate_version_member_id,),
        axis.qualification_view_ids,
        axis.claimed_property_ids,
        axis.nontransported_property_ids,
    )
    return ModelSetSpec(
        f"{stem}.models",
        world_id,
        (member,),
        (law.relation.relation_id,),
        'Exact law-qualified frozen boundary; no model selection.',
        "The declared eight-word chart with frozen causal feature support.",
        (law.obligations.uncertainty.uncertainty_id,),
        law.obligations.validity,
        ModelIntersectionSemantics.QUALIFIED_INTERSECTION,
        EvidenceCeiling.ADMISSION,
        OutcomeAccess.OUTCOME_BLIND,
        (VisibilityCeiling.PROSPECTIVE,),
        VisibilityCeiling.PROSPECTIVE,
    )


def control_law_context(
    *,
    system: SystemSpec,
    report: FiniteResponseLawQualificationReport,
    boundary: str,
    method: ExecutableReference,
    producer: ObjectIdentity,
    resource: ObjectIdentity,
    authority: ObjectIdentity,
) -> FiniteResponseLawControlLawContext:
    "No qualification/refitting: preserve the exact finite response-law qualification law and its whole chart."
    if boundary not in ("composed", "cached", "direct"):
        raise ValueError("Only frozen pre-parent laws are prospective consumers")
    index = tuple(b.boundary for b in report.calibration.boundaries).index(boundary)
    qualification, family, publication = (
        report.qualifications[index],
        report.families[index],
        report.publications[index],
    )
    law = qualification.response_law
    if law is None or not dict(report.calibration.joint_opportunities)["composed"]:
        raise ValueError("Finite response-law evaluation requires eligible primary and an available boundary")
    axes = family.ledger.axis_map
    if len(axes.bindings) != 1 or len(axes.qualification_view_ids) != 2:
        raise ValueError("Finite response-law control changes its frozen member/version/view roster")
    axis = axes.bindings[0]
    stem = f"{PROGRAMME}.{boundary}.prospective-control"
    declaration = LawQualificationBatchCoordinateDeclaration(
        f"{stem}.coordinate",
        f"{stem}.domain",
        law.chart_id,
        law.obligations.support.denominator_cell_ids[0],
        axis.denominator_member_id,
    )
    batch = LawQualificationBatch(
        f"{stem}.batch",
        system.system_id,
        system.world.world_id,
        (declaration,),
        (
            LawQualificationBatchCoordinate(
                declaration.coordinate_id,
                declaration.domain_cell_id,
                declaration.chart_id,
                declaration.denominator_cell_id,
                declaration.denominator_member_id,
                qualification,
                LawQualificationCoordinateDisposition.SUPPORTED_LAW,
                ObjectIdentity.from_record(law.law_id, law),
                None,
                (),
                tuple(e.link_id for e in qualification.evidence_links),
            ),
        ),
        (),
        qualification.evidence_links,
        EvidenceCeiling.LOCAL_LAW,
        VisibilityCeiling.PROSPECTIVE,
    )
    assembly_spec = QualificationBatchAtlasAssemblySpec(
        f"{stem}.assembly",
        ObjectIdentity.from_record(system.system_id, system),
        ObjectIdentity.from_record(batch.batch_id, batch),
        (
            AtlasDomainCell(
                declaration.domain_cell_id, law.chart_id, declaration.denominator_cell_id
            ),
        ),
        (),
        EvidenceCeiling.LOCAL_LAW,
        VisibilityCeiling.PROSPECTIVE,
    )
    assembly = AtlasAssembler().assemble_batch(
        system=system, batch=batch, spec=assembly_spec, evidence_links=batch.evidence_links
    )
    if not isinstance(assembly, BatchAtlasAssemblyResult):
        raise ValueError("Qualified finite response-law boundary did not yield its exact atlas")
    models = qualified_model_set(report=report, boundary=boundary, world_id=system.world.world_id)
    member = models.members[0]
    maps = tuple(
        controller_word_map(
            replace(
                native_action_word(
                    force,
                    denominator_id=axis.denominator_member_id,
                    history_id=law.relation.history_quantity_ids[0],
                ),
                support_status=ActionWordSupportStatus.SUPPORTED,
                reason_codes=(),
            )
        )
        for force in NATIVE_WORDS
    )
    fibres = tuple(
        sorted(
            (
                ReceiptAdmissionActionFibre(
                    f"{stem}.word-{i}",
                    m.controller_word,
                    tuple(o.occurrence_id for o in m.controller_word.occurrences),
                    tuple(ActionDeliveryStage),
                )
                for i, m in enumerate(maps)
            ),
            key=lambda f: f.action_binding_id,
        )
    )
    cell = ReceiptAdmissionSupportCell(
        axis.denominator_member_id,
        law.chart_id,
        axis.denominator_member_id,
        tuple(sorted(m.controller_word.word_id for m in maps)),
    )
    cutoff = InformationCutoff(f"{stem}.preparent", Q_CLOCK, CausalPhase.PRE_ACTION, D(4096))
    coordinates = tuple(
        ReceiptAdmissionPlannedCoordinate(
            f"{stem}.word-{i}.coordinate",
            ObjectIdentity.from_record(member.binding_id, member),
            member.response_law,
            member.qualification_result,
            member.denominator_member_id,
            member.candidate_version_ids[0],
            member.qualification_view_ids,
            ObjectIdentity.from_record(f.action_binding_id, f),
            ObjectIdentity.from_record(cell.cell_id, cell),
        )
        for i, f in enumerate(fibres)
    )
    predicates = tuple(
        sorted(
            (
                GatePredicateSpec(
                    f"{stem}.gate.{k.value.lower()}",
                    k,
                    output_quantities()[0].quantity_id,
                    PredicateDirection.AT_LEAST,
                    f"{stem}.margin.{k.value.lower()}",
                    ReceiverInterval(clock(4096), clock(4096))
                    if k
                    in (AdmissionGateKind.PHYSICAL_SINK, AdmissionGateKind.OBSERVATION_VALIDITY)
                    else ReceiverInterval(clock(4368), clock(4560)),
                    GatePredicateKind.SCALAR_AT_LEAST,
                    NamedDecimal(f"{stem}.zero.{k.value.lower()}", D(0), "1"),
                    None,
                    None,
                    None,
                    TemporalPredicateSemantics.ALWAYS_PRESERVED_PATH
                    if k is AdmissionGateKind.BASELINE_PRESERVATION
                    else None,
                    (f"{stem}.constraint.{k.value.lower()}",),
                    method,
                )
                for k in AdmissionGateKind
            ),
            key=lambda p: p.predicate_id,
        )
    )
    plan = ReceiptAdmissionReceiptProductionPlan(
        f"{stem}.admission",
        assembly.atlas,
        ObjectIdentity.from_record(batch.batch_id, batch),
        models,
        member.denominator_member_id,
        fibres,
        (cell,),
        coordinates,
        predicates,
        f"{stem}.initial",
        HORIZON,
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
    from empirical_lawhood.planning.finite_action_support import support_map_for_plan

    mapping = support_map_for_plan(plan)
    models = replace(models, extensions=(mapping.extension,))
    plan = replace(plan, model_set=models)
    AdmissionReceiptPlanValidator.validate(batch=batch, plan=plan)
    return FiniteResponseLawControlLawContext(
        system, qualification, axes, publication, batch, assembly, models, maps, plan
    )
