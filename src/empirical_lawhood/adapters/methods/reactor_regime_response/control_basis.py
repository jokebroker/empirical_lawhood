"""Carry the exact sole-owner joint law into the installed atlas/model-set owners."""

from functools import lru_cache

from empirical_lawhood.adapters.geometry.services import AtlasAssembler
from empirical_lawhood.adapters.methods.contracts import (
    LawQualificationBatch,
    LawQualificationBatchCoordinate,
    LawQualificationBatchCoordinateDeclaration,
    LawQualificationCoordinateDisposition,
)
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.models import ModelIntersectionSemantics, ModelMemberLawBinding, ModelSetSpec
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.status import ScientificStatus
from empirical_lawhood.planning.geometry import (
    AtlasDomainCell,
    BatchAtlasAssemblyResult,
    QualificationBatchAtlasAssemblySpec,
)

from .config import PREFIX
from .law_terminal import RegimeJointLawResult
from .science import regime_system


@lru_cache(maxsize=4)
def regime_control_basis(
    report: RegimeJointLawResult,
) -> tuple[LawQualificationBatch, BatchAtlasAssemblyResult, ModelSetSpec]:
    system = regime_system()
    qualification = report.qualification
    law = qualification.response_law
    if (
        qualification.scientific_status is not ScientificStatus.SUPPORTED
        or law is None
        or report.family.system != ObjectIdentity.from_record(system.system_id, system)
        or report.payload.route is None
    ):
        raise ValueError("Admission basis requires the exact supported selected joint reactor law")
    axes = report.family.axis_map
    if len(axes.bindings) != 1 or len(axes.qualification_view_ids) != 2:
        raise ValueError("Admission basis changed the frozen member and two-view roster")
    axis = axes.bindings[0]
    stem = f"{PREFIX}.qualified-parent"
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
        (LawQualificationBatchCoordinate(
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
            tuple(link.link_id for link in qualification.evidence_links),
        ),),
        (),
        qualification.evidence_links,
        EvidenceCeiling.LOCAL_LAW,
        VisibilityCeiling.PROSPECTIVE,
    )
    spec = QualificationBatchAtlasAssemblySpec(
        f"{stem}.assembly",
        ObjectIdentity.from_record(system.system_id, system),
        ObjectIdentity.from_record(batch.batch_id, batch),
        (AtlasDomainCell(
            declaration.domain_cell_id,
            law.chart_id,
            declaration.denominator_cell_id,
        ),),
        (),
        EvidenceCeiling.LOCAL_LAW,
        VisibilityCeiling.PROSPECTIVE,
    )
    assembly = AtlasAssembler().assemble_batch(
        system=system, batch=batch, spec=spec, evidence_links=batch.evidence_links
    )
    if not isinstance(assembly, BatchAtlasAssemblyResult):
        raise ValueError("Admission atlas owner did not preserve the qualified reactor member")
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
    model_set = ModelSetSpec(
        f"{stem}.models",
        system.world.world_id,
        (member,),
        (law.relation.relation_id,),
        "One unchanged selected-route joint law with measured zero-feed reference.",
        "Three scalar feed words under an unchanged absolute jacket denominator.",
        (law.obligations.uncertainty.uncertainty_id,),
        law.obligations.validity,
        ModelIntersectionSemantics.QUALIFIED_INTERSECTION,
        EvidenceCeiling.ADMISSION,
        OutcomeAccess.OUTCOME_BLIND,
        (VisibilityCeiling.PROSPECTIVE,),
        VisibilityCeiling.PROSPECTIVE,
    )
    return batch, assembly, model_set
