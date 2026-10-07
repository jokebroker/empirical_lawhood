"""Local receiver predicates delegated to the existing joint proof owner."""

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
from .config import LocalQualificationDesign, PREFIX
from .payload import LocalPayload
from .scoring import LocalCalibration, LocalQualificationCell
from .science import METHOD, RECEIVERS, UNITS, QUALIFICATION_UNITS, local_system


@dataclass(frozen=True)
class LocalQualificationProfile:
    owner: QualificationProofOwner
    design: LocalQualificationDesign
    calibration: LocalCalibration
    cell: LocalQualificationCell
    expected_payload: LocalPayload
    expected_candidate: LawCandidateEvidence
    design_artifact: ArtifactIdentity

    @property
    def stem(self) -> str:
        return f"{PREFIX}.{self.cell.domain}.r{self.cell.receiver}"

    @property
    def profile(self) -> JointQualificationProfile:
        base = build_method_equivalent_qualification_profile(
            self.owner, MethodEquivalentProfileKind.FINITE_ACTION
        )
        values = {f.name: getattr(base, f.name) for f in fields(ComponentQualificationProfile)}

        def rename(value: str) -> str:
            return value.replace("finite-action", self.stem)

        bindings = tuple(
            replace(
                binding,
                binding_id=rename(binding.binding_id),
                output_id=rename(binding.output_id),
                rule_id=rename(binding.rule_id),
                input_schema=LocalPayload.SCHEMA
                if binding.input_schema == base.applicable_payload_schemas[0]
                else binding.input_schema,
                rule_semantics="Frozen causal domain and measured action chart; assigned roots and missing/non-contact outcomes retained; receiver-specific root-max calibration and fresh conditional CP95 adequacy. No global or transition promotion.",
            )
            for binding in base.proof_owner_bindings
        )
        values.update(
            profile_id=f"profile.{self.stem}",
            applicable_method_keys=(METHOD,),
            applicable_payload_schemas=(LocalPayload.SCHEMA,),
            applicable_extension_schemas=(LocalPayload.SCHEMA,),
            proof_owner_bindings=tuple(sorted(bindings, key=lambda b: b.binding_id)),
            facet_ids=tuple(sorted(rename(v) for v in base.facet_ids)),
            conformance_case_ids=tuple(sorted(rename(v) for v in base.conformance_case_ids)),
            allowed_not_applicable_output_ids=("uncertainty.transport",),
        )
        return JointQualificationProfile(**values)

    def evaluate_candidate(
        self, system: SystemSpec, candidate: LawCandidateEvidence, payload: CanonicalRecord
    ) -> JointQualificationAssessment:
        cell = self.cell
        calibrated = next(
            c
            for c in self.calibration.cells
            if (c.domain, c.receiver) == (cell.domain, cell.receiver)
        )
        custody = (
            system == local_system(cell.receiver)
            and candidate == self.expected_candidate
            and payload == self.expected_payload
            and self.calibration.recipe
            == ObjectIdentity.from_record(self.design.config_id, self.design)
            and self.expected_payload.calibration_sha256 == self.calibration.fingerprint()
            and self.design_artifact.sha256 == self.design.fingerprint()
            and len(candidate.method_receipts) == 64
            and candidate.physical_independent_unit_ids == QUALIFICATION_UNITS
        )
        reasons = cell.reasons if custody else ("LOCAL_CUSTODY_UNACCOUNTED",)
        unknown = any("MISSING" in r or "INSUFFICIENT" in r for r in reasons)
        status = (
            O.SATISFIED if not reasons else O.UNEVALUABLE if not custody or unknown else O.FAILED
        )
        links = tuple(e.link_id for e in candidate.evidence_links) if custody else ()
        metrics = tuple(
            sorted(
                (
                    NamedDecimal("assigned-calibration-roots", D(32), "1"),
                    NamedDecimal("assigned-qualification-roots", D(32), "1"),
                    NamedDecimal("contacted-calibration-roots", D(len(calibrated.roots)), "1"),
                    NamedDecimal(
                        "contacted-qualification-roots", D(len(cell.contacted_roots)), "1"
                    ),
                    NamedDecimal("adequate-qualification-roots", D(len(cell.adequate_roots)), "1"),
                    NamedDecimal("conditional-coverage-lower-bound", cell.lower_bound, "1"),
                ),
                key=lambda m: m.value_id,
            )
        )
        checks = []
        for kind in K:
            disposition = O.SATISFIED
            if not custody:
                disposition = O.UNEVALUABLE
            elif kind in (K.HELD_OUT_CALIBRATION, K.DECISIVE_FALSIFIER):
                disposition = status
            elif kind in (K.WITHIN_CELL_RECURRENCE, K.ONE_FACTOR_EXCHANGE) and (
                len(calibrated.roots) < 29 or len(cell.contacted_roots) < 29
            ):
                disposition = O.UNEVALUABLE
            elif (
                kind is K.ONE_FACTOR_EXCHANGE
                and "LOCAL_ACTION_RESPONSE_PRECISION_FAILED" in reasons
            ):
                disposition = O.FAILED
            elif kind in (
                K.COORDINATE_ADEQUACY,
                K.STRUCTURAL_CONVERGENCE,
                K.COMPUTABILITY,
                K.CLOSURE_MEMORY,
            ) and any("INVALID" in r or "MISSING" in r for r in reasons):
                disposition = O.UNEVALUABLE if unknown else O.FAILED
            checks.append(
                AdequacyCheckResult(
                    f"check.{self.stem}.{kind.value.lower().replace('_', '-')}",
                    kind,
                    disposition,
                    True,
                    metrics if custody else (),
                    reasons if disposition in (O.UNEVALUABLE, O.FAILED) else (),
                    links,
                )
            )
        diagnostics = UncertaintyDecomposition(
            f"{self.stem}.component-diagnostics",
            tuple(
                sorted(
                    (
                        UncertaintyComponent(
                            f"{self.stem}.{kind.value.lower()}",
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
            f"{self.stem}.joint-calibration",
            template.uncertainty_method_key,
            ObjectIdentity.from_record(f"{PREFIX}.calibration", self.calibration)
            if custody
            else None,
            self.design_artifact,
            template.independent_unit_id,
            QUALIFICATION_UNITS,
            "root-maximum-in-frozen-domain-conditional-on-causal-assay-contact",
            template.interval_quantity_ids,
            template.support_id,
            D(".90"),
            (NamedDecimal(RECEIVERS[cell.receiver], cell.halfwidth, UNITS[cell.receiver]),)
            if custody and cell.halfwidth is not None
            else (),
            (
                NamedDecimal(
                    RECEIVERS[cell.receiver],
                    self.design.numerical_padding[cell.receiver],
                    UNITS[cell.receiver],
                ),
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
            prefix=self.stem,
            checks=tuple(sorted(checks, key=lambda c: c.check_id)),
            component_diagnostics=diagnostics,
            joint_uncertainty=joint,
        )
