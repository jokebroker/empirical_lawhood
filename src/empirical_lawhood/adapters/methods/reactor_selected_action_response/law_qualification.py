"Selected response-bound predicates supplied to the sole measurement through local law owner."

from dataclasses import dataclass, fields, replace
from decimal import Decimal as D

from empirical_lawhood.adapters.methods.contracts import LawCandidateEvidence, JointQualificationAssessment
from empirical_lawhood.adapters.methods.qualification_profiles import MethodEquivalentProfileKind, ComponentQualificationProfile, JointQualificationProfile, QualificationProofOwner, assess_joint_predictive_candidate, build_method_equivalent_qualification_profile
from empirical_lawhood.kernel.identification import (
    AdequacyCheckKind as K,
    AdequacyCheckResult,
    UncertaintyClass,
    UncertaintyComponent,
    UncertaintyDecomposition,
)
from empirical_lawhood.kernel.obligations import ObligationStatus as O
from empirical_lawhood.kernel.predictive_uncertainty import JointPredictiveUncertainty
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity, NamedDecimal
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.systems import SystemSpec
from .config import PREFIX, ROOTS, ClassicalDesign
from .law_payload import ClassicalLawPayload
from .records import ClassicalQualification
from .science import METHOD, RECEIVERS, classical_system

QUALIFICATION_ROOTS = tuple(root for root, role, _, _ in ROOTS if role == "qualification")
STEM = f"{PREFIX}.selected-bound"


@dataclass(frozen=True)
class ClassicalQualificationProfile:
    owner: QualificationProofOwner
    design: ClassicalDesign
    qualification: ClassicalQualification
    expected_payload: ClassicalLawPayload
    expected_candidate: LawCandidateEvidence
    design_artifact: ArtifactIdentity

    @property
    def profile(self) -> JointQualificationProfile:
        base = build_method_equivalent_qualification_profile(
            self.owner, MethodEquivalentProfileKind.FINITE_ACTION
        )
        values = {f.name: getattr(base, f.name) for f in fields(ComponentQualificationProfile)}

        def rename(value: str) -> str:
            return value.replace("finite-action", STEM)

        values.update(
            profile_id=f"profile.{STEM}",
            applicable_method_keys=(METHOD,),
            applicable_payload_schemas=(ClassicalLawPayload.SCHEMA,),
            applicable_extension_schemas=(ClassicalLawPayload.SCHEMA,),
            proof_owner_bindings=tuple(
                sorted(
                    (
                        replace(
                            binding,
                            binding_id=rename(binding.binding_id),
                            output_id=rename(binding.output_id),
                            rule_id=rename(binding.rule_id),
                            input_schema=ClassicalLawPayload.SCHEMA
                            if binding.input_schema == base.applicable_payload_schemas[0]
                            else binding.input_schema,
                            rule_semantics="One development-frozen positive word and interval; all 64 fresh roots and two views; exact native delivery, paired response bound, coarse causal safety envelope and CP95 lower coverage at least .90. No coefficient identification or full-chart accuracy claim.",
                        )
                        for binding in base.proof_owner_bindings
                    ),
                    key=lambda v: v.binding_id,
                )
            ),
            facet_ids=tuple(sorted(rename(v) for v in base.facet_ids)),
            conformance_case_ids=tuple(sorted(rename(v) for v in base.conformance_case_ids)),
            allowed_not_applicable_output_ids=("uncertainty.transport",),
        )
        return JointQualificationProfile(**values)

    def evaluate_candidate(
        self, system: SystemSpec, candidate: LawCandidateEvidence, payload: CanonicalRecord
    ) -> JointQualificationAssessment:
        q = self.qualification
        custody = (
            system == classical_system()
            and candidate == self.expected_candidate
            and payload == self.expected_payload
            and self.design_artifact.sha256 == self.design.fingerprint()
            and self.expected_payload.qualification == ObjectIdentity.from_record(q.package_id, q)
            and tuple(row.root for row in q.roots) == QUALIFICATION_ROOTS
            and len(candidate.method_receipts) == 64
            and candidate.physical_independent_unit_ids == QUALIFICATION_ROOTS
        )
        status = (
            O.UNEVALUABLE
            if not custody or not q.evaluable
            else O.SATISFIED
            if q.qualifies
            else O.FAILED
        )
        reasons = (
            ()
            if status is O.SATISFIED
            else (
                "CLASSICAL_FRESH_BOUND_FAILED"
                if status is O.FAILED
                else "CLASSICAL_ASSIGNED_EVIDENCE_UNEVALUABLE",
            )
        )
        links = tuple(link.link_id for link in candidate.evidence_links) if custody else ()
        metrics = (
            NamedDecimal("assigned-roots", D(64), "1"),
            NamedDecimal("coverage-lower-95", q.lower_95, "1"),
            NamedDecimal("successful-roots", D(q.successes), "1"),
        )
        checks = tuple(
            sorted(
                (
                    AdequacyCheckResult(
                        f"check.{STEM}.{kind.value.lower().replace('_', '-')}",
                        kind,
                        status
                        if kind in (K.HELD_OUT_CALIBRATION, K.DECISIVE_FALSIFIER)
                        else O.SATISFIED
                        if custody and q.evaluable
                        else O.UNEVALUABLE,
                        True,
                        metrics if custody else (),
                        reasons
                        if (
                            kind in (K.HELD_OUT_CALIBRATION, K.DECISIVE_FALSIFIER)
                            or not custody
                            or not q.evaluable
                        )
                        else (),
                        links,
                    )
                    for kind in K
                ),
                key=lambda v: v.check_id,
            )
        )
        diagnostics = UncertaintyDecomposition(
            f"{STEM}.diagnostics",
            tuple(
                sorted(
                    (
                        UncertaintyComponent(
                            f"{STEM}.{kind.value.lower()}",
                            kind,
                            O.NOT_APPLICABLE
                            if kind is UncertaintyClass.TRANSPORT
                            else O.UNEVALUABLE,
                            (),
                            ()
                            if kind is UncertaintyClass.TRANSPORT
                            else ("FINITE_BOUND_DOES_NOT_DECOMPOSE_LATENT_UNCERTAINTY",),
                            (),
                        )
                        for kind in UncertaintyClass
                    ),
                    key=lambda v: v.component_id,
                )
            ),
        )
        template = candidate.obligation_template
        joint = JointPredictiveUncertainty(
            f"{STEM}.coverage-qualification",
            template.uncertainty_method_key,
            ObjectIdentity.from_record(q.package_id, q) if custody else None,
            self.design_artifact,
            template.independent_unit_id,
            QUALIFICATION_ROOTS,
            "all-assigned-selected-action-both-views-no-bound-refit",
            template.interval_quantity_ids,
            template.support_id,
            D(".90"),
            tuple(
                sorted(
                    (
                        NamedDecimal(RECEIVERS[0], D(1), "K"),
                        NamedDecimal(RECEIVERS[1], D(".0016"), "K"),
                    ),
                    key=lambda v: v.value_id,
                )
            )
            if custody
            else (),
            tuple(
                sorted(
                    (
                        NamedDecimal(RECEIVERS[0], D(".01"), "K"),
                        NamedDecimal(RECEIVERS[1], D(".000001"), "K"),
                    ),
                    key=lambda v: v.value_id,
                )
            ),
            template.assumption_ids,
            status,
            reasons,
            links,
        )
        return assess_joint_predictive_candidate(
            candidate=candidate,
            profile=self.profile,
            owner=self.owner,
            prefix=STEM,
            checks=checks,
            component_diagnostics=diagnostics,
            joint_uncertainty=joint,
        )
