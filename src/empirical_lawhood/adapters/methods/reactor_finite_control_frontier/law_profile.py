"Finite-action proof profile backed by the exact 72-family local law operands."

from dataclasses import dataclass
from decimal import Decimal as D

from empirical_lawhood.adapters.methods.contracts import LawCandidateEvidence, JointQualificationAssessment
from empirical_lawhood.adapters.methods.qualification_profiles import JointQualificationProfile, QualificationProofOwner
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity, NamedDecimal
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.adapters.methods.finite_qualification import finite_profile, finite_assessment
from .config import FrontierDesign
from .law_payload import FrontierLawPayload
from .qualification import QUALIFICATION_ROOTS, FrontierQualificationRow
from .science import METHOD, RECEIVERS, frontier_system


@dataclass(frozen=True)
class FrontierQualificationProfile:
    owner: QualificationProofOwner
    row: FrontierQualificationRow
    payload: FrontierLawPayload
    candidate: LawCandidateEvidence
    design_artifact: ArtifactIdentity

    @property
    def profile(self) -> JointQualificationProfile:
        return finite_profile(
            owner=self.owner,
            stem=self.payload.stem,
            method=METHOD,
            payload_schema=FrontierLawPayload.SCHEMA,
            semantics="Development-frozen word/window/interval; 96 assigned roots, both views, exact delivery and 120-second safety; CP alpha=.05/72 lower >=.90; no unsafe action/prefix; no bound refit.",
        )

    def evaluate_candidate(
        self, system: SystemSpec, candidate: LawCandidateEvidence, payload: CanonicalRecord
    ) -> JointQualificationAssessment:
        row, bound, stem = self.row, self.row.bound, self.payload.stem
        custody = (
            system == frontier_system()
            and candidate == self.candidate
            and payload == self.payload
            and self.design_artifact.sha256 == FrontierDesign().fingerprint()
            and self.payload.qualification == ObjectIdentity.from_record(row.record_id, row)
            and self.payload.bound == bound
            and len(candidate.method_receipts) == 96
            and candidate.physical_independent_unit_ids == QUALIFICATION_ROOTS
        )
        widths = (
            bound.thermal_allowance_K or D(0),
            D(0)
            if bound.lower_K is None or bound.upper_K is None
            else (bound.upper_K - bound.lower_K) / 2,
        )
        return finite_assessment(
            owner=self.owner,
            profile=self.profile,
            candidate=candidate,
            stem=stem,
            custody=custody,
            evaluable=row.evaluable,
            qualifies=row.qualifies,
            assigned_roots=QUALIFICATION_ROOTS,
            qualification=ObjectIdentity.from_record(row.record_id, row),
            design_artifact=self.design_artifact,
            metrics=(
                NamedDecimal("assigned-roots", D(96), "1"),
                NamedDecimal("coverage-family-lower", row.lower, "1"),
                NamedDecimal("successful-roots", D(row.successes), "1"),
            ),
            widths=tuple(
                NamedDecimal(key, width, "K") for key, width in zip(RECEIVERS, widths, strict=True)
            ),
            numerical_floors=(
                NamedDecimal(RECEIVERS[0], D(".01"), "K"),
                NamedDecimal(RECEIVERS[1], bound.coordinate.epsilon_K, "K"),
            ),
            failure_reason="FRONTIER_FROZEN_ROW_FAILED",
            unknown_reason="FRONTIER_ASSIGNED_EVIDENCE_UNEVALUABLE",
        )
