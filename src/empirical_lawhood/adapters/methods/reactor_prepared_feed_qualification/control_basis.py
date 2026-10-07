"""Preserve the joint qualified parent through existing atlas and model-set owners."""

from functools import lru_cache
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.models import ModelIntersectionSemantics, ModelMemberLawBinding, ModelSetSpec
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.planning.geometry import (
    AtlasDomainCell,
    QualificationBatchAtlasAssemblySpec,
    BatchAtlasAssemblyResult,
)
from empirical_lawhood.adapters.methods.contracts import (
    LawQualificationBatch,
    LawQualificationBatchCoordinate,
    LawQualificationBatchCoordinateDeclaration,
    LawQualificationCoordinateDisposition,
)
from empirical_lawhood.adapters.geometry.services import AtlasAssembler
from .terminal import FeedQualificationResult
from .config import PREFIX as PROGRAMME


def feed_model_set(*, report: FeedQualificationResult, world_id: str) -> ModelSetSpec:
    """Preserve the exact qualified member and chosen-policy limitation."""
    qualification = report.laws[0].qualification
    law = qualification.response_law
    axes = report.laws[0].family.axis_map
    if law is None or len(axes.bindings) != 1:
        raise ValueError("control requires the supported frozen prepared-feed law")
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
        "Exact qualified frozen prepared-feed consumer; no online model selection.",
        "Three fixed-jacket feed words on the frozen prepared support; joint temperature and cooling envelopes.",
        (law.obligations.uncertainty.uncertainty_id,),
        law.obligations.validity,
        ModelIntersectionSemantics.QUALIFIED_INTERSECTION,
        EvidenceCeiling.ADMISSION,
        OutcomeAccess.OUTCOME_BLIND,
        (VisibilityCeiling.PROSPECTIVE,),
        VisibilityCeiling.PROSPECTIVE,
    )


@lru_cache(maxsize=4)
def feed_control_basis(
    system: SystemSpec, report: FeedQualificationResult
) -> tuple[LawQualificationBatch, BatchAtlasAssemblyResult, ModelSetSpec]:
    if report.admission_and_prospective_prerequisite != "QUALIFIED_PREPARED_FEED_RECEIVERS_CONTROL_ISSUE_REQUIRED":
        raise ValueError("prepared feed control prerequisite is not jointly supported")
    qualification, family = report.laws[0].qualification, report.laws[0].family
    law = qualification.response_law
    if law is None:
        raise ValueError("control requires supported useful prepared-feed qualification")
    axes = family.axis_map
    if len(axes.bindings) != 1 or len(axes.qualification_view_ids) != 2:
        raise ValueError("prepared-feed control changed its frozen member/version/view roster")
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
        raise ValueError("Qualified prepared-feed parent did not yield its exact atlas")
    models = feed_model_set(report=report, world_id=system.world.world_id)
    return batch, assembly, models
