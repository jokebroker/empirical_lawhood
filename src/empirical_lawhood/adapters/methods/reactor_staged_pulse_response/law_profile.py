"""Adapter operands for the shared finite-action qualification profile."""

from dataclasses import dataclass
from decimal import Decimal as D

from empirical_lawhood.adapters.methods.contracts import LawCandidateEvidence, JointQualificationAssessment
from empirical_lawhood.adapters.methods.finite_qualification import finite_assessment, finite_profile
from empirical_lawhood.adapters.methods.qualification_profiles import JointQualificationProfile, QualificationProofOwner
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity, NamedDecimal
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.systems import SystemSpec
from .config import ClassicalDesign, roots
from .law_payload import ClassicalLawPayload
from .measurement import epsilon, receiver_ids
from .qualification import ClassicalQualificationRow
from .science import METHOD, staged_pulse_reactor_system


@dataclass(frozen=True)
class ClassicalQualificationProfile:
    owner: QualificationProofOwner
    row: ClassicalQualificationRow
    payload: ClassicalLawPayload
    candidate: LawCandidateEvidence
    design_artifact: ArtifactIdentity

    @property
    def profile(self) -> JointQualificationProfile:
        return finite_profile(
            owner=self.owner,
            stem=self.payload.stem,
            method=METHOD,
            payload_schema=ClassicalLawPayload.SCHEMA,
            semantics=f"Development-frozen native word/history and raw interval or saturated one-sided map; all 96 assigned roots, at least 95 successes, CP lower >=.90 at .05/{ {'base-menu-comparison': 72, 'expanded-menu-comparison': 36, 'staged-sequence-comparison': 5}[self.row.block] }; both raw numerical views, exact stages, every declared thermal guard and zero unsafe roots. No calibration/nonentry promotion or bound refit.",
        )

    def evaluate_candidate(
        self, system: SystemSpec, candidate: LawCandidateEvidence, payload: CanonicalRecord
    ) -> JointQualificationAssessment:
        row, recipe = self.row, self.row.recipe
        c = recipe.bound.coordinate
        assigned = roots(row.block, "qualification")
        custody = (
            system == staged_pulse_reactor_system(c)
            and candidate == self.candidate
            and payload == self.payload
            and self.design_artifact.sha256 == ClassicalDesign().fingerprint()
            and self.payload.qualification == ObjectIdentity.from_record(row.record_id, row)
            and self.payload.recipe == recipe
            and len(candidate.method_receipts) == 96
            and candidate.physical_independent_unit_ids == assigned
        )
        widths, floors = [], []
        for key in receiver_ids(c):
            interval = recipe.arm == "EL_INTERVAL" or key.startswith("s")
            width = (
                (recipe.bound.suffix_thermal_K if key == "s2" else recipe.bound.thermal_K) or D(0)
                if key.startswith("s")
                else (recipe.bound.response(key).upper - recipe.bound.response(key).lower) / 2
                if interval and recipe.bound.responses
                else D(0)
            )
            widths.append(NamedDecimal(key, width, "K"))
            # Raw correspondence is in the row's event. The point-box map has
            # zero numerical floor, not another epsilon of fictitious width.
            floors.append(NamedDecimal(key, epsilon(c, key) if interval else D(0), "K"))
        return finite_assessment(
            owner=self.owner,
            profile=self.profile,
            candidate=candidate,
            stem=self.payload.stem,
            custody=custody,
            evaluable=row.evaluable,
            qualifies=row.qualifies,
            assigned_roots=assigned,
            qualification=ObjectIdentity.from_record(row.record_id, row),
            design_artifact=self.design_artifact,
            metrics=(
                NamedDecimal("assigned-roots", D(96), "1"),
                NamedDecimal("coverage-family-lower", row.lower, "1"),
                NamedDecimal("successful-roots", D(row.successes), "1"),
            ),
            widths=tuple(widths),
            numerical_floors=tuple(floors),
            failure_reason="CLASSICAL_FROZEN_RELATION_FAILED",
            unknown_reason="CLASSICAL_ASSIGNED_EVIDENCE_OR_PREREQUISITE_UNEVALUABLE",
        )
