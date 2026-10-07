"Task-specific predicates, reduced by the shared measurement through local law qualification owner."

from dataclasses import dataclass, fields, replace
from decimal import Decimal as D

from empirical_lawhood.adapters.methods.contracts import LawCandidateEvidence, JointQualificationAssessment
from empirical_lawhood.adapters.methods.qualification_profiles import MethodEquivalentProfileKind, ComponentQualificationProfile, JointQualificationProfile, QualificationProofOwner, QualificationProofOutputKind, assess_joint_predictive_candidate, build_method_equivalent_qualification_profile
from .records import DevelopmentBoundReactorQualificationOperands
from .science import RECEIVERS, UNITS

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
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
from .payload import DevelopmentBoundReactorResponsePayload
from .science import QUALIFICATION_UNITS, METHOD, PREFIX, empirical_system
from .config import EmpiricalRecipe

NUMERICAL_TOLERANCE = (D(".01"), D(".0002"))

RULES = {
    K.COORDINATE_ADEQUACY: "Pinned source, native clocks/units, post-run window maximum and endpoint conversion; 64 paired roots.",
    K.WITHIN_CELL_RECURRENCE: "32 assigned calibration roots and 32 distinct qualification roots; callbacks never increase n.",
    K.ONE_FACTOR_EXCHANGE: "Not applicable to chosen-path predictive qualification; development branch effects remain separate.",
    K.SUPPORT: "Training-only per-clock/action boxes, eight fit roots per cell, every chosen query supported.",
    K.CLOSURE_MEMORY: "Frozen causal 7/13/23 feature basis, prior requests and native available observations only.",
    K.HELD_OUT_CALIBRATION: "Joint root maxima, rank 32, q at most one; all 32 fresh A and one-sided lower bound at least .90.",
    K.STRUCTURAL_CONVERGENCE: "Both views and complete actual exposure profiles; finite numerical invalidity invalidates score.",
    K.DECISIVE_FALSIFIER: "Any invalid calibration root or false qualification A refuses predictive qualification.",
    K.COMPUTABILITY: "Bounded complete query census with binary64 inference and saved fit identities.",
    K.EVIDENCE_VISIBILITY: "Authenticate all 64 raw operand publications and task receipts before finalization.",
}


def calibration_metrics(calibration: DevelopmentBoundReactorQualificationOperands) -> tuple[NamedDecimal, ...]:
    return (
        NamedDecimal("assigned-calibration-units", D(32), "1"),
        NamedDecimal("assigned-qualification-units", D(32), "1"),
        NamedDecimal("qualification-adequate", D(sum(calibration.adequacy)), "1"),
        NamedDecimal("qualification-lower-bound", calibration.lower_bound, "1"),
    )


@dataclass(frozen=True)
class EmpiricalQualificationProfile:
    owner: QualificationProofOwner
    design: EmpiricalRecipe
    calibration: DevelopmentBoundReactorQualificationOperands
    expected_payload: DevelopmentBoundReactorResponsePayload
    expected_candidate: LawCandidateEvidence
    design_artifact: ArtifactIdentity

    @property
    def profile(self) -> JointQualificationProfile:
        base = build_method_equivalent_qualification_profile(
            self.owner, MethodEquivalentProfileKind.FINITE_ACTION
        )
        values = {f.name: getattr(base, f.name) for f in fields(ComponentQualificationProfile)}

        def rename(value: str) -> str:
            return value.replace("finite-action", PREFIX)

        bindings = tuple(
            replace(
                binding,
                binding_id=rename(binding.binding_id),
                output_id=rename(binding.output_id),
                rule_id=rename(binding.rule_id),
                input_schema=DevelopmentBoundReactorResponsePayload.SCHEMA
                if binding.input_schema == base.applicable_payload_schemas[0]
                else binding.input_schema,
                rule_semantics=RULES[K(binding.output_id.upper())]
                if binding.output_kind is QualificationProofOutputKind.ADEQUACY_CHECK
                else "Reduce these on-policy whole-episode predicates through the shared joint-predictive owner; do not promote latent components or off-policy support.",
            )
            for binding in base.proof_owner_bindings
        )
        values.update(
            profile_id=f"profile.{PREFIX}.joint-forecast",
            applicable_method_keys=(METHOD,),
            applicable_payload_schemas=(DevelopmentBoundReactorResponsePayload.SCHEMA,),
            applicable_extension_schemas=(DevelopmentBoundReactorResponsePayload.SCHEMA,),
            proof_owner_bindings=tuple(sorted(bindings, key=lambda b: b.binding_id)),
            facet_ids=tuple(sorted(rename(v) for v in base.facet_ids)),
            conformance_case_ids=tuple(sorted(rename(v) for v in base.conformance_case_ids)),
            allowed_not_applicable_output_ids=(
                "adequacy.one_factor_exchange",
                "uncertainty.transport",
            ),
        )
        return JointQualificationProfile(**values)

    def evaluate_candidate(
        self,
        system: SystemSpec,
        candidate: LawCandidateEvidence,
        payload: CanonicalRecord,
    ) -> JointQualificationAssessment:
        calibration = self.calibration
        calibration_id = ObjectIdentity.from_record(f"{PREFIX}.calibration", calibration)
        links = tuple(e.link_id for e in candidate.evidence_links)
        custody = (
            system == empirical_system(self.design)
            and candidate == self.expected_candidate
            and payload == self.expected_payload
            and self.expected_payload.calibration_sha256 == calibration.calibration_digest
            and self.expected_payload.q
            == (calibration.q if calibration.q is not None and calibration.q <= 1 else None)
            and self.expected_payload.policy_sha256 == self.design.fingerprint()
            and self.design_artifact.sha256 == self.design.fingerprint()
            and candidate.physical_independent_unit_ids == QUALIFICATION_UNITS
            and candidate.visibility_ceiling is VisibilityCeiling.PROSPECTIVE
            and candidate.outcome_access is OutcomeAccess.EVALUATOR_REVEAL
            and len(candidate.method_receipts) == 64
            and len(links) == 65
            and all(
                e.visibility_ceiling is VisibilityCeiling.PROSPECTIVE
                and e.outcome_access is OutcomeAccess.EVALUATOR_REVEAL
                for e in candidate.evidence_links
            )
        )
        bounds = calibration.bounds
        enough = (
            all(calibration.adequacy)
            and calibration.lower_bound >= D(".90")
            and calibration.q is not None
            and calibration.q <= 1
        )
        status = (
            O.UNEVALUABLE
            if not custody
            else O.SATISFIED
            if bounds is not None and enough
            else O.FAILED
        )
        reasons = (
            ()
            if status is O.SATISFIED
            else ("FORECAST_CUSTODY_UNACCOUNTED",)
            if not custody
            else (
                "NONFINITE_CALIBRATION_BOUND"
                if bounds is None
                else "QUALIFICATION_ADEQUACY_OR_PRECISION_FAILED",
            )
        )
        metrics = calibration_metrics(calibration) if custody else ()
        all_reasons = {
            r for group in (*calibration.invalidity, *calibration.adequacy_reasons) for r in group
        }
        cause_sets = {
            K.COORDINATE_ADEQUACY: {"INCOMPLETE_ASSIGNED_CENSUS", "DELIVERY_OR_CLOCK_INVALID"},
            K.SUPPORT: {"OUTSIDE_SUPPORT"},
            K.STRUCTURAL_CONVERGENCE: {
                "NUMERICAL_INVALID",
                "MATCHED_INPUT_EXPOSURE_DIFFERS",
                "DELIVERY_OR_CLOCK_INVALID",
                "REQUEST_PROJECTION_MISMATCH",
                "DOSE_EXPOSURE_MISMATCH",
                "INCOMPLETE_EXPOSURE",
                "REFINEMENT_TAPE_DIFFERS",
            },
            K.COMPUTABILITY: {"MISSING_OR_NONFINITE_LABEL", "NONFINITE_PREDICTION"},
        }
        checks = []
        for kind in K:
            disposition = (
                O.UNEVALUABLE
                if not custody
                else O.NOT_APPLICABLE
                if kind is K.ONE_FACTOR_EXCHANGE
                else status
                if kind in (K.HELD_OUT_CALIBRATION, K.DECISIVE_FALSIFIER)
                else O.FAILED
                if all_reasons & cause_sets.get(kind, set())
                else O.SATISFIED
            )
            checks.append(
                AdequacyCheckResult(
                    f"check.{PREFIX}.{kind.value.lower().replace('_', '-')}",
                    kind,
                    disposition,
                    True,
                    metrics,
                    (
                        ("MATCHED_INPUT_NUMERICAL_FIDELITY_FAILED",)
                        if kind is K.STRUCTURAL_CONVERGENCE and disposition is O.FAILED
                        else tuple(sorted(all_reasons & cause_sets.get(kind, set()))) or reasons
                    )
                    if disposition in (O.FAILED, O.UNEVALUABLE)
                    else (),
                    links if custody else (),
                )
            )
        diagnostics = UncertaintyDecomposition(
            f"{PREFIX}.component-diagnostics",
            tuple(
                sorted(
                    (
                        UncertaintyComponent(
                            f"{PREFIX}.{kind.value.lower()}",
                            kind,
                            O.NOT_APPLICABLE
                            if kind is UncertaintyClass.TRANSPORT
                            else O.UNEVALUABLE,
                            (),
                            ()
                            if kind is UncertaintyClass.TRANSPORT
                            else ("JOINT_ENVELOPE_DOES_NOT_IDENTIFY_COMPONENT_BOUND",),
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
            f"{PREFIX}.joint-calibration",
            template.uncertainty_method_key,
            calibration_id if custody else None,
            self.design_artifact,
            template.independent_unit_id,
            QUALIFICATION_UNITS,
            "two-receivers-all-2880-callbacks-two-views",
            template.interval_quantity_ids,
            template.support_id,
            D(".90"),
            tuple(
                sorted(
                    (
                        NamedDecimal(q, b, u)
                        for q, b, u in zip(RECEIVERS, bounds, UNITS, strict=True)
                    ),
                    key=lambda v: v.value_id,
                )
            )
            if custody and bounds is not None
            else (),
            tuple(
                sorted(
                    (
                        NamedDecimal(q, b, u)
                        for q, b, u in zip(RECEIVERS, NUMERICAL_TOLERANCE, UNITS, strict=True)
                    ),
                    key=lambda v: v.value_id,
                )
            ),
            template.assumption_ids,
            status,
            reasons,
            links if custody else (),
        )
        return assess_joint_predictive_candidate(
            candidate=candidate,
            profile=self.profile,
            owner=self.owner,
            prefix=PREFIX,
            checks=tuple(sorted(checks, key=lambda c: c.check_id)),
            component_diagnostics=diagnostics,
            joint_uncertainty=joint,
        )
