"Exact local-law lineage into the existing atlas and admission model-set owners."

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
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.kernel.identification import LawQualificationResult
from empirical_lawhood.adapters.methods.contracts import LawCandidateAxisBinding


def qualified_local_basis(
    *,
    system: SystemSpec,
    qualification: LawQualificationResult,
    axis: LawCandidateAxisBinding,
    stem: str,
    scope_description: str,
    nontransport_description: str,
) -> tuple[LawQualificationBatch, BatchAtlasAssemblyResult, ModelSetSpec]:
    """One qualified local relation, with its own atlas and model intersection.

    Alternative actions/regimes are separate calls. This never promotes a gap,
    pools independent units or intersects laws for unrelated native words.
    """
    law = qualification.response_law
    if qualification.scientific_status is not ScientificStatus.SUPPORTED or law is None:
        raise ValueError("Admission local basis requires an independently supported law")
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
                tuple(link.link_id for link in qualification.evidence_links),
            ),
        ),
        (),
        qualification.evidence_links,
        EvidenceCeiling.LOCAL_LAW,
        VisibilityCeiling.PROSPECTIVE,
    )
    spec = QualificationBatchAtlasAssemblySpec(
        f"{stem}.assembly",
        ObjectIdentity.from_record(system.system_id, system),
        ObjectIdentity.from_record(batch.batch_id, batch),
        (
            AtlasDomainCell(
                declaration.domain_cell_id,
                law.chart_id,
                declaration.denominator_cell_id,
            ),
        ),
        (),
        EvidenceCeiling.LOCAL_LAW,
        VisibilityCeiling.PROSPECTIVE,
    )
    assembly = AtlasAssembler().assemble_batch(
        system=system, batch=batch, spec=spec, evidence_links=batch.evidence_links
    )
    if not isinstance(assembly, BatchAtlasAssemblyResult):
        raise ValueError("Admission atlas owner did not preserve the qualified member")
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
        scope_description,
        nontransport_description,
        (law.obligations.uncertainty.uncertainty_id,),
        law.obligations.validity,
        ModelIntersectionSemantics.QUALIFIED_INTERSECTION,
        EvidenceCeiling.ADMISSION,
        OutcomeAccess.OUTCOME_BLIND,
        (VisibilityCeiling.PROSPECTIVE,),
        VisibilityCeiling.PROSPECTIVE,
    )
    return batch, assembly, model_set
