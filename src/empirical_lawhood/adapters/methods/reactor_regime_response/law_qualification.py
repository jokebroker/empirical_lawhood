"Reactor predicates supplied to the installed sole measurement through local law owner."

from __future__ import annotations

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

from .calibration_pipeline import RegimeCalibrationPackage
from .config import PREFIX, ROOTS, ReactorRegimeResponseDesign
from .law_payload import RegimeJointLawPayload
from .nomination_records import RegimeNominationPackage
from .qualification_pipeline import RegimeQualificationPackage
from .science import METHOD, RECEIVERS, regime_system

QUALIFICATION_ROOTS = tuple(root for root, role, _, _ in ROOTS if role == "qualification")
CALIBRATION_ROOTS = tuple(root for root, role, _, _ in ROOTS if role == "calibration")
STEM = f"{PREFIX}.selected-joint"


@dataclass(frozen=True)
class RegimeJointLawQualificationProfile:
    owner: QualificationProofOwner
    design: ReactorRegimeResponseDesign
    nomination: RegimeNominationPackage
    calibration: RegimeCalibrationPackage
    qualification: RegimeQualificationPackage
    expected_payload: RegimeJointLawPayload
    expected_candidate: LawCandidateEvidence
    design_artifact: ArtifactIdentity

    @property
    def profile(self) -> JointQualificationProfile:
        base = build_method_equivalent_qualification_profile(
            self.owner, MethodEquivalentProfileKind.FINITE_ACTION
        )
        values = {field.name: getattr(base, field.name) for field in fields(ComponentQualificationProfile)}

        def rename(value: str) -> str:
            return value.replace("finite-action", STEM)

        bindings = tuple(
            replace(
                binding,
                binding_id=rename(binding.binding_id),
                output_id=rename(binding.output_id),
                rule_id=rename(binding.rule_id),
                input_schema=RegimeJointLawPayload.SCHEMA
                if binding.input_schema == base.applicable_payload_schemas[0]
                else binding.input_schema,
                rule_semantics=(
                    "One frozen selected route and three-word scalar-feed chart; "
                    "32 calibration and all 64 fresh qualification roots, both numerical "
                    "views and whole-root maxima. Failed contact, support, source validity, "
                    "or interval coverage cannot be a success. law qualification law and preparation release "
                    "remain separate."
                ),
            )
            for binding in base.proof_owner_bindings
        )
        values.update(
            profile_id=f"profile.{STEM}",
            applicable_method_keys=(METHOD,),
            applicable_payload_schemas=(RegimeJointLawPayload.SCHEMA,),
            applicable_extension_schemas=(RegimeJointLawPayload.SCHEMA,),
            proof_owner_bindings=tuple(sorted(bindings, key=lambda value: value.binding_id)),
            facet_ids=tuple(sorted(rename(value) for value in base.facet_ids)),
            conformance_case_ids=tuple(sorted(rename(value) for value in base.conformance_case_ids)),
            allowed_not_applicable_output_ids=("uncertainty.transport",),
        )
        return JointQualificationProfile(**values)

    def evaluate_candidate(
        self, system: SystemSpec, candidate: LawCandidateEvidence, payload: CanonicalRecord
    ) -> JointQualificationAssessment:
        q = self.qualification
        cal = self.calibration
        expected_q = ObjectIdentity.from_record(q.package_id, q)
        custody = (
            system == regime_system()
            and candidate == self.expected_candidate
            and payload == self.expected_payload
            and self.design_artifact.sha256 == self.design.fingerprint()
            and self.expected_payload.qualification == expected_q
            and self.expected_payload.calibration == ObjectIdentity.from_record(cal.package_id, cal)
            and self.expected_payload.nomination
            == ObjectIdentity.from_record(self.nomination.package_id, self.nomination)
            and q.calibration == self.expected_payload.calibration
            and q.nomination == self.expected_payload.nomination
            and tuple(value.root for value in q.roots) == QUALIFICATION_ROOTS
            and len(candidate.method_receipts) == 64
            and candidate.physical_independent_unit_ids == QUALIFICATION_ROOTS
        )
        scientific_pass = (
            cal.precision_pass
            and q.law_pass
            and q.law_lower_95 >= D(".90")
            and self.expected_payload.route is not None
            and self.expected_payload.coefficient is not None
            and self.expected_payload.absolute is not None
            and self.expected_payload.q_temperature is not None
            and self.expected_payload.q_cooling is not None
        )
        status = O.UNEVALUABLE if not custody else O.SATISFIED if scientific_pass else O.FAILED
        reasons = (
            ("REACTOR_LAW_CUSTODY_UNACCOUNTED",)
            if not custody else () if scientific_pass else (
                "REACTOR_FROZEN_JOINT_LAW_PRECISION_OR_FRESH_ADEQUACY_FAILED",
            )
        )
        links = tuple(value.link_id for value in candidate.evidence_links) if custody else ()
        metrics = tuple(sorted((
            NamedDecimal("assigned-calibration-roots", D(32), "1"),
            NamedDecimal("assigned-qualification-roots", D(64), "1"),
            NamedDecimal("contacted-calibration-roots", D(len(cal.contacted_roots)), "1"),
            NamedDecimal("adequate-qualification-roots", D(q.law_successes), "1"),
            NamedDecimal("joint-law-lower-95", q.law_lower_95, "1"),
            NamedDecimal("law-and-preparation-lower-95", q.joint_lower_95, "1"),
        ), key=lambda value: value.value_id))
        checks = tuple(sorted((
            AdequacyCheckResult(
                f"check.{STEM}.{kind.value.lower().replace('_', '-')}",
                kind,
                status if kind in (K.HELD_OUT_CALIBRATION, K.DECISIVE_FALSIFIER)
                else O.UNEVALUABLE if not custody else O.SATISFIED,
                True,
                metrics if custody else (),
                reasons if kind in (K.HELD_OUT_CALIBRATION, K.DECISIVE_FALSIFIER) else (),
                links,
            )
            for kind in K
        ), key=lambda value: value.check_id))
        diagnostics = UncertaintyDecomposition(
            f"{STEM}.component-diagnostics",
            tuple(sorted((
                UncertaintyComponent(
                    f"{STEM}.{kind.value.lower()}", kind,
                    O.NOT_APPLICABLE if kind is UncertaintyClass.TRANSPORT else O.UNEVALUABLE,
                    (), () if kind is UncertaintyClass.TRANSPORT else (
                        "JOINT_ROOT_ENVELOPE_DOES_NOT_IDENTIFY_COMPONENT_BOUND",
                    ), (),
                )
                for kind in UncertaintyClass
            ), key=lambda value: value.component_id)),
        )
        template = candidate.obligation_template
        uncertainty = JointPredictiveUncertainty(
            f"{STEM}.joint-calibration",
            template.uncertainty_method_key,
            ObjectIdentity.from_record(cal.package_id, cal) if custody else None,
            self.design_artifact,
            template.independent_unit_id,
            CALIBRATION_ROOTS,
            "all-assigned-root-maxima-in-frozen-selected-route-and-native-chart",
            template.interval_quantity_ids,
            template.support_id,
            D(".90"),
            tuple(sorted((
                NamedDecimal(RECEIVERS[0], cal.q_temperature * D(".25") + D(".01"), "K"),
                NamedDecimal(RECEIVERS[1], cal.q_cooling * D(".00505") + D(".000001"), "K"),
            ), key=lambda value: value.value_id))
            if custody and cal.q_temperature is not None and cal.q_cooling is not None
            else (),
            tuple(sorted((
                NamedDecimal(RECEIVERS[0], D(".01"), "K"),
                NamedDecimal(RECEIVERS[1], D(".000001"), "K"),
            ), key=lambda value: value.value_id)),
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
            joint_uncertainty=uncertainty,
        )
