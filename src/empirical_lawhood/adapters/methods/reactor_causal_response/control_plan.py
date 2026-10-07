"Bind the exact empirical qualification to ordinary atlas and finite admission owners."

from dataclasses import dataclass, replace
from decimal import Decimal as D
from functools import lru_cache

from empirical_lawhood.kernel.action_contracts import ActionDeliveryStage
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
from empirical_lawhood.kernel.time import CausalPhase, InformationCutoff
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
from empirical_lawhood.adapters.simulators.reactor_causal_response.delivery import ReactorExposureWordMap, coordinate as clock
from .terminal import EmpiricalQualificationResult
from .science import PREFIX as PROGRAMME, CLOCK as Q_CLOCK
from .numerical import Decision


@dataclass(frozen=True, slots=True)
class EmpiricalControlLawContext:
    system: SystemSpec
    qualification: LawQualificationResult
    axes: LawCandidateAxisMap
    publication: CandidatePayloadPublicationReceipt
    batch: LawQualificationBatch
    assembly: BatchAtlasAssemblyResult
    model_set: ModelSetSpec
    word_maps: tuple[ReactorExposureWordMap, ...]
    plan: ReceiptAdmissionReceiptProductionPlan


def qualified_model_set(*, report: EmpiricalQualificationResult, world_id: str) -> ModelSetSpec:
    """Preserve the exact qualified member and chosen-policy limitation."""
    qualification = report.qualification
    law = qualification.response_law
    axes = report.family.axis_map
    if law is None or len(axes.bindings) != 1:
        raise ValueError("control requires the supported frozen empirical law")
    axis = axes.bindings[0]
    stem = f"{PROGRAMME}.control"
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
        "Exact qualified frozen empirical consumer; no online model selection.",
        "Nine-word query chart; coverage only for the frozen chosen policy path.",
        (law.obligations.uncertainty.uncertainty_id,),
        law.obligations.validity,
        ModelIntersectionSemantics.QUALIFIED_INTERSECTION,
        EvidenceCeiling.ADMISSION,
        OutcomeAccess.OUTCOME_BLIND,
        (VisibilityCeiling.PROSPECTIVE,),
        VisibilityCeiling.PROSPECTIVE,
    )


@lru_cache(maxsize=4)
def _qualified_basis(
    system: SystemSpec, report: EmpiricalQualificationResult
) -> tuple[LawQualificationBatch, BatchAtlasAssemblyResult, ModelSetSpec]:
    qualification, family = report.qualification, report.family
    law = qualification.response_law
    if law is None:
        raise ValueError("control requires supported useful empirical qualification")
    axes = family.axis_map
    if len(axes.bindings) != 1 or len(axes.qualification_view_ids) != 2:
        raise ValueError("empirical control changed its frozen member/version/view roster")
    axis = axes.bindings[0]
    stem = f"{PROGRAMME}.qualified-parent"
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
        raise ValueError("Qualified empirical parent did not yield its exact atlas")
    models = qualified_model_set(report=report, world_id=system.world.world_id)
    return batch, assembly, models


def control_law_context(
    *,
    system: SystemSpec,
    report: EmpiricalQualificationResult,
    decision: Decision,
    root: str,
    time: D,
    method: ExecutableReference,
    producer: ObjectIdentity,
    resource: ObjectIdentity,
    authority: ObjectIdentity,
) -> EmpiricalControlLawContext:
    """One causal callback instance; no new qualification or fitted parameters."""
    batch, assembly, models = _qualified_basis(system, report)
    qualification, axes, publication = (
        report.qualification,
        report.family.axis_map,
        report.candidate.payload_publication,
    )
    law = qualification.response_law
    assert law is not None
    axis = axes.bindings[0]
    member = models.members[0]
    stem = f"{root}.{int(time) // 10:04d}"
    maps = tuple(
        ReactorExposureWordMap.from_projection(
            c.projection,
            decision_id=f"{stem}.word-{i}",
            history_id=law.relation.history_quantity_ids[0],
            horizon_id=law.relation.horizon.horizon_id,
        )
        for i, c in enumerate(decision.candidates)
    )
    fibres = tuple(
        sorted(
            (
                ReceiptAdmissionActionFibre(
                    f"{stem}.word-{i}",
                    m.word,
                    tuple(o.occurrence_id for o in m.word.occurrences),
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
        tuple(sorted(m.word.word_id for m in maps)),
    )
    cutoff = InformationCutoff(f"{stem}.preaction", Q_CLOCK, CausalPhase.PRE_ACTION, time)
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
                    "reactor-peak-temperature",
                    PredicateDirection.AT_LEAST,
                    f"{stem}.margin.{k.value.lower()}",
                    ReceiverInterval(clock(time), clock(time))
                    if k
                    in (AdmissionGateKind.PHYSICAL_SINK, AdmissionGateKind.OBSERVATION_VALIDITY)
                    else ReceiverInterval(clock(time), clock(time + D(10))),
                    GatePredicateKind.SCALAR_AT_LEAST,
                    NamedDecimal(
                        f"{stem}.zero.{k.value.lower()}",
                        D(0),
                        "K"
                        if k
                        in (
                            AdmissionGateKind.TARGET,
                            AdmissionGateKind.PHYSICAL_SINK,
                            AdmissionGateKind.BASELINE_PRESERVATION,
                            AdmissionGateKind.REACHABILITY,
                        )
                        else "1",
                    ),
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
    from empirical_lawhood.planning.finite_action_support import support_map_for_plan

    mapping = support_map_for_plan(plan)
    models = replace(models, extensions=(mapping.extension,))
    plan = replace(plan, model_set=models)
    AdmissionReceiptPlanValidator.validate(batch=batch, plan=plan)
    return EmpiricalControlLawContext(
        system, qualification, axes, publication, batch, assembly, models, maps, plan
    )
