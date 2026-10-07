"Task-specific predicates, reduced by the shared measurement through local law qualification owner."

from dataclasses import dataclass, fields, replace
from decimal import Decimal as D

from empirical_lawhood.adapters.methods.contracts import LawCandidateEvidence, JointQualificationAssessment
from empirical_lawhood.adapters.methods.qualification_profiles import MethodEquivalentProfileKind, ComponentQualificationProfile, JointQualificationProfile, QualificationProofOwner, QualificationProofOutputKind, assess_joint_predictive_candidate, build_method_equivalent_qualification_profile
from empirical_lawhood.adapters.methods.reactor_prefix_response.batch_calibration import NUMERICAL_TOLERANCE, RECEIVERS, UNITS, ReactorForecastCalibration
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
from .law import ReactorForecastPayload
from .science import CALIBRATION_UNITS, METHOD, PREFIX, ReactorForecastDesign, forecast_system

RULES = {
    K.COORDINATE_ADEQUACY: "Exact pinned source, causal observer, native units, ten-second receiver and all 42 assigned episode pairs.",
    K.WITHIN_CELL_RECURRENCE: "Whole-episode residual recurrence under the same fixed reference policy; 32 calibration and ten independent heldout episodes, never 5760 independent callbacks.",
    K.ONE_FACTOR_EXCHANGE: "Not applicable to this singleton on-policy predictive claim; no action-exchange, off-policy response or controller-improvement claim is made.",
    K.SUPPORT: "Only the frozen public-envelope episode distribution and the reference policy's exact proposed command; no private-panel coverage or off-policy support.",
    K.CLOSURE_MEMORY: "Fixed public nominal model and reference observer with causal committed command history; no latent scenario, future faults or native state enters prediction.",
    K.HELD_OUT_CALIBRATION: "Per-receiver maximum of 32 calibration episode scores plus numerical allowance; at least nine of ten new heldout episodes jointly covered.",
    K.STRUCTURAL_CONVERGENCE: "Both coupled views enter each episode score. Any view discrepancy beyond its frozen allowance makes that episode's score infinite, with no exclusion.",
    K.DECISIVE_FALSIFIER: "A nonfinite calibration bound or fewer than nine covered heldout episodes refuses the claim; unavailable custody is unevaluable.",
    K.COMPUTABILITY: "Full trace accounting, finite bounded native receiver values and frozen source/operator/config identities.",
    K.EVIDENCE_VISIBILITY: "All 84 native artifacts and task receipts authenticated at the evaluator boundary; a separate task publishes terminal outcome-visible adjudication.",
}


def calibration_metrics(calibration: ReactorForecastCalibration) -> tuple[NamedDecimal, ...]:
    values = [
        NamedDecimal("assigned-calibration-units", D(32), "1"),
        NamedDecimal("assigned-heldout-units", D(10), "1"),
        NamedDecimal(
            "heldout-joint-covered", D(sum(c for _, c in calibration.heldout_coverage)), "1"
        ),
        NamedDecimal(
            "infinite-calibration-scores",
            D(
                sum(
                    s.maximum_absolute_errors is None
                    for s in calibration.unit_scores
                    if s.split == "calibration"
                )
            ),
            "1",
        ),
        NamedDecimal(
            "invalid-assigned-units",
            D(sum(bool(s.reason_codes) for s in calibration.unit_scores)),
            "1",
        ),
    ]
    if calibration.bounds is not None:
        values.extend(
            NamedDecimal(key, value, unit)
            for key, value, unit in zip(RECEIVERS, calibration.bounds, UNITS, strict=True)
        )
    return tuple(sorted(values, key=lambda v: v.value_id))


@dataclass(frozen=True)
class ReactorForecastQualificationProfile:
    owner: QualificationProofOwner
    design: ReactorForecastDesign
    calibration: ReactorForecastCalibration
    expected_payload: ReactorForecastPayload
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
                input_schema=ReactorForecastPayload.SCHEMA
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
            applicable_payload_schemas=(ReactorForecastPayload.SCHEMA,),
            applicable_extension_schemas=(ReactorForecastPayload.SCHEMA,),
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
            system == forecast_system(self.design)
            and candidate == self.expected_candidate
            and payload == self.expected_payload
            and self.expected_payload.calibration == calibration_id
            and self.expected_payload.bounds == calibration.bounds
            and self.expected_payload.design
            == ObjectIdentity.from_record(self.design.config_id, self.design)
            and self.design_artifact.sha256 == self.design.fingerprint()
            and candidate.physical_independent_unit_ids == CALIBRATION_UNITS
            and candidate.visibility_ceiling is VisibilityCeiling.PROSPECTIVE
            and candidate.outcome_access is OutcomeAccess.EVALUATOR_REVEAL
            and len(candidate.method_receipts) == 84
            and len(links) == 85
            and all(
                e.visibility_ceiling is VisibilityCeiling.PROSPECTIVE
                and e.outcome_access is OutcomeAccess.EVALUATOR_REVEAL
                for e in candidate.evidence_links
            )
        )
        bounds = calibration.bounds
        enough = (
            sum(c for _, c in calibration.heldout_coverage) >= self.design.minimum_heldout_covered
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
                else "HELDOUT_JOINT_COVERAGE_BELOW_NINE_OF_TEN",
            )
        )
        metrics = calibration_metrics(calibration) if custody else ()
        checks = []
        for kind in K:
            disposition = (
                O.UNEVALUABLE
                if not custody
                else (
                    O.NOT_APPLICABLE
                    if kind is K.ONE_FACTOR_EXCHANGE
                    else status
                    if kind is K.HELD_OUT_CALIBRATION
                    else O.SATISFIED
                )
            )
            checks.append(
                AdequacyCheckResult(
                    f"check.{PREFIX}.{kind.value.lower().replace('_', '-')}",
                    kind,
                    disposition,
                    True,
                    metrics,
                    reasons if disposition in (O.FAILED, O.UNEVALUABLE) else (),
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
            CALIBRATION_UNITS,
            "three-receivers-all-episode-callbacks-two-views",
            template.interval_quantity_ids,
            template.support_id,
            self.design.confidence_level,
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
