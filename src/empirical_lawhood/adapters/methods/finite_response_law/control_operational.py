"Pre-parent operational operands from actual prefix and replayed authority.\n\nKnown prefix validity and qualified forecast support are not observations of\nfuture work, delivery, physical safety or preservation. Those remain controller-use gates.\n"

from decimal import Decimal as D

from empirical_lawhood.kernel.admission import AdmissionGateKind as G
from empirical_lawhood.kernel.provenance import EvidenceLink, ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity, NamedDecimal
from empirical_lawhood.planning.evidence_geometry import ReceiptAdmissionRawDisposition
from empirical_lawhood.planning.study_issue import StudyOperationAuthority, StudyAuthorityKind, require_study_authority
from empirical_lawhood.adapters.geometry.admission_receipts import AdmissionGateRawInput, LawMemberEvaluationBinder
from empirical_lawhood.adapters.methods.law_evaluation import LawEvaluationDisposition
from .control_plan import FiniteResponseLawControlLawContext
from .control_prediction import FiniteResponseLawControlPredictionTable
from .control_records import FiniteResponseLawRootForecast


def pre_parent_operational_inputs(
    *,
    context: FiniteResponseLawControlLawContext,
    forecast: FiniteResponseLawRootForecast,
    table: FiniteResponseLawControlPredictionTable,
    execution_authority: StudyOperationAuthority,
    issued_study: ObjectIdentity,
    prerequisite_authority: ObjectIdentity | None,
    grantee_id: str,
    at_utc: str,
    artifacts: tuple[ArtifactIdentity, ...],
    evidence_links: tuple[EvidenceLink, ...],
) -> tuple[AdmissionGateRawInput, ...]:
    """Require exact loaded authority and prefix evidence, with no passing defaults.

    The provider loads the authority from its existing authoritative port after
    runtime replay; constructing a record or reaching this helper grants none.
    """
    require_study_authority(
        execution_authority,
        kind=StudyAuthorityKind.EXPERIMENT_EXECUTION,
        subject=issued_study,
        prerequisite_authority=prerequisite_authority,
        grantee_id=grantee_id,
        at_utc=at_utc,
    )
    if (
        table not in forecast.tables
        or context.plan.authority_boundary
        != ObjectIdentity.from_record(execution_authority.authority_id, execution_authority)
        or not artifacts
        or not evidence_links
        or not {a for a in table.prefix_artifacts} <= set(artifacts)
        or not any(
            a.payload_schema == execution_authority.SCHEMA
            and a.sha256 == execution_authority.fingerprint()
            for a in artifacts
        )
        or any(
            e.information_cutoff_id != context.plan.information_cutoff.cutoff_id
            for e in evidence_links
        )
    ):
        raise ValueError(
            "Finite response-law operational operands lose actual prefix custody or execution authority"
        )
    # In this simulator, the known source condition is successful finite prefix
    # integration and its validated frame/checkpoint. A failed observation is
    # UNEVALUABLE, never evidence of a physical sink or a future prediction.
    prefix_available = (
        forecast.prefix.native_complete
        and forecast.prefix.frame is not None
        and forecast.primary_interface is not None
    )
    raw = []
    for coordinate in context.plan.coordinates:
        word = context.plan.action_fibre(coordinate.action_fibre).action_word
        for request, result in zip(table.requests, table.results, strict=True):
            if request.action_word != word:
                continue
            binding = LawMemberEvaluationBinder().bind(
                plan=context.plan,
                coordinate_id=coordinate.coordinate_id,
                request=request,
                result=result,
            )
            for kind in (G.PHYSICAL_SINK, G.OBSERVATION_VALIDITY, G.DYNAMICS, G.AUTHORITY):
                available = result.disposition is LawEvaluationDisposition.SUPPORTED and (
                    kind in (G.AUTHORITY, G.DYNAMICS) or prefix_available
                )
                disposition = (
                    ReceiptAdmissionRawDisposition.EVALUATED
                    if available
                    else ReceiptAdmissionRawDisposition.OUTSIDE_SUPPORT
                    if kind is G.DYNAMICS
                    and result.disposition is LawEvaluationDisposition.OUTSIDE_SUPPORT
                    else ReceiptAdmissionRawDisposition.EVIDENCE_UNAVAILABLE
                )
                stem = f"{forecast.root.stage_unit}.{table.boundary}.{coordinate.coordinate_id}.{request.qualification_view_id}.{kind.value.lower()}"
                raw.append(
                    AdmissionGateRawInput(
                        stem,
                        coordinate.coordinate_id,
                        kind,
                        disposition,
                        NamedDecimal(f"{stem}.known-condition", D(0), "1") if available else None,
                        None,
                        None,
                        binding,
                        artifacts,
                        evidence_links,
                        None,
                    )
                )
    return tuple(raw)
