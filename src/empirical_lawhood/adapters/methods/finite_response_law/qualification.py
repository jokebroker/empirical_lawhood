"""Plan-specific predicates reduced by the existing sole qualification owners."""

from dataclasses import dataclass, fields, replace
from decimal import Decimal as D

from empirical_lawhood.adapters.methods.contracts import LawCandidateEvidence, JointQualificationAssessment
from empirical_lawhood.adapters.methods.qualification_profiles import MethodEquivalentProfileKind, ComponentQualificationProfile, JointQualificationProfile, QualificationProofOutputKind, QualificationProofOwner, assess_joint_predictive_candidate, build_method_equivalent_qualification_profile
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.identification import (
    AdequacyCheckKind as K,
)
from empirical_lawhood.kernel.identification import (
    AdequacyCheckResult,
    UncertaintyClass,
    UncertaintyComponent,
    UncertaintyDecomposition,
)
from empirical_lawhood.kernel.obligations import ObligationStatus as O
from empirical_lawhood.kernel.predictive_uncertainty import JointPredictiveUncertainty
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.runtime.execution import DependencyReceiptBinding

from .calibration_operands import CalibrationOperands
from .calibration_panel import valid_calibration_root_ids
from .calibration_records import FiniteResponseLawBoundaryCalibration
from .law_binding import PARENT_QUANTITY, feature_quantities, output_quantities
from .law_family import EVIDENCE_KIND
from .law_payloads import MAXIMUM_LAW_BYTES, FiniteResponseLawConditionalPayload, FiniteResponseLawLowerPayload
from .science import PROGRAMME, FiniteResponseLawScienceSpec

RULES = {
    K.COORDINATE_ADEQUACY: "Exact source, native pair, receiver, units and +192 horizon; complete accounted 32-root roster.",
    K.WITHIN_CELL_RECURRENCE: "Two named independent future purposes per root; missing measurements remain explicit infinite scores, never new units.",
    K.ONE_FACTOR_EXCHANGE: "The complete fixed axial 8/16 signed-pair menu under one preassigned old parent per root; no invented actuator interpolation.",
    K.SUPPORT: "Frozen normalizers and six-standardized-unit support; unsupported roots remain infinite scores, not exclusions.",
    K.CLOSURE_MEMORY: "Exact pre-calibration coefficients/scales and causal prefix/handoff inputs; no calibration fitting or extra point-error gate.",
    K.HELD_OUT_CALIBRATION: "One complete-root maximum and rank 30 of all 32 scores, including infinity; finite nonnegative q required.",
    K.STRUCTURAL_CONVERGENCE: "Both coupled numerical views enter every root score; discrepancies yield infinity without an all-root numerical-success gate.",
    K.DECISIVE_FALSIFIER: "Predeclared nonfinite-q refusal and custody-unevaluable truth table; no additional population width, work or task-rate threshold.",
    K.COMPUTABILITY: "Bounded authenticated frozen finite coefficients, positive scales and fixed numerical allowance.",
    K.EVIDENCE_VISIBILITY: "Authenticated fresh evaluator-only calibration receipts and pre-calibration development dependencies.",
}


def calibration_metrics(
    calibration: FiniteResponseLawBoundaryCalibration,
) -> tuple[NamedDecimal, ...]:
    values = [
        NamedDecimal("accounted-roots", D(32), "1"),
        NamedDecimal(
            "infinite-root-scores", D(sum(s is None for s in calibration.scores)), "1"
        ),
        NamedDecimal("order-index", D(30), "1"),
    ]
    if calibration.q is not None:
        values.append(NamedDecimal("q", calibration.q, "1"))
    return tuple(sorted(values, key=lambda v: v.value_id))


@dataclass(frozen=True)
class FiniteResponseLawQualificationProfile:
    owner: QualificationProofOwner
    boundary: str
    calibration: FiniteResponseLawBoundaryCalibration
    expected_payload: FiniteResponseLawLowerPayload | FiniteResponseLawConditionalPayload
    operands: CalibrationOperands | None
    calibration_receipt: DependencyReceiptBinding | None
    calibration_materialization_id: str
    prediction_materialization_id: str

    def __post_init__(self) -> None:
        payload = self.expected_payload
        if (
            self.boundary != self.calibration.boundary
            or (self.boundary == "lower") != isinstance(payload, FiniteResponseLawLowerPayload)
            or isinstance(payload, FiniteResponseLawConditionalPayload)
            and payload.boundary != self.boundary
            or payload.calibration != self.calibration.identity
            or payload.q != self.calibration.q
            or self.calibration_materialization_id == self.prediction_materialization_id
        ):
            raise ValueError(
                "Finite response-law profile substitutes its boundary or committed calibration operands"
            )

    @property
    def prefix(self) -> str:
        return f"{PROGRAMME}.{self.boundary}"

    @property
    def profile(self) -> JointQualificationProfile:
        base = build_method_equivalent_qualification_profile(
            self.owner, MethodEquivalentProfileKind.FINITE_ACTION
        )
        values = {f.name: getattr(base, f.name) for f in fields(ComponentQualificationProfile)}

        def name(value: str) -> str:
            return value.replace("finite-action", self.prefix)

        bindings = []
        for binding in base.proof_owner_bindings:
            rule = (
                RULES[K(binding.output_id.upper())]
                if binding.output_kind is QualificationProofOutputKind.ADEQUACY_CHECK
                else "Reduce the declared FLH joint predictive basis and explicit diagnostic limitations through the shared owner; no latent-component promotion."
            )
            bindings.append(
                replace(
                    binding,
                    binding_id=name(binding.binding_id),
                    output_id=name(binding.output_id),
                    input_schema=self.expected_payload.SCHEMA
                    if binding.input_schema == base.applicable_payload_schemas[0]
                    else binding.input_schema,
                    rule_id=name(binding.rule_id),
                    rule_semantics=rule,
                )
            )
        values.update(
            profile_id=f"profile.{self.prefix}.fresh-joint",
            applicable_method_keys=(f"{PROGRAMME}.{self.boundary}-law",),
            applicable_payload_schemas=(self.expected_payload.SCHEMA,),
            applicable_extension_schemas=(self.expected_payload.SCHEMA,),
            proof_owner_bindings=tuple(sorted(bindings, key=lambda b: b.binding_id)),
            facet_ids=tuple(sorted(name(v) for v in base.facet_ids)),
            conformance_case_ids=tuple(
                sorted(name(v) for v in base.conformance_case_ids)
            ),
        )
        return JointQualificationProfile(**values)

    def evaluate_candidate(
        self,
        system: SystemSpec,
        candidate: LawCandidateEvidence,
        payload: CanonicalRecord,
    ) -> JointQualificationAssessment:
        if not isinstance(payload, FiniteResponseLawLowerPayload | FiniteResponseLawConditionalPayload):
            raise TypeError("Finite response-law profile received another payload family")
        calibration, operands, receipt = (
            self.calibration,
            self.operands,
            self.calibration_receipt,
        )
        records = tuple(
            r for r in candidate.method_receipts if r.evidence_kind_id == EVIDENCE_KIND
        )
        links = tuple(e.link_id for e in candidate.evidence_links)
        sources = {e.source for e in candidate.evidence_links}
        required_quantities = (*feature_quantities(self.boundary), *output_quantities())
        inputs = {q.quantity_id for q in feature_quantities(self.boundary)}
        if self.boundary != "lower":
            inputs.add(PARENT_QUANTITY)
        known = {q.quantity_id: q for q in system.quantities}
        custody = (
            operands is not None
            and receipt is not None
            and receipt.task_id == f"{PROGRAMME}.calibration.calibrate"
            and self.calibration_materialization_id
            in receipt.output_materialization_ids
            and self.prediction_materialization_id in receipt.output_materialization_ids
            and len(records) == 1
            and records[0].receipt_id == receipt.receipt_id
            and records[0].metrics == calibration_metrics(calibration)
            and records[0].evidence_link_ids == links
            and calibration in operands.boundaries
            and calibration.prediction_artifact == operands.prediction_artifact
            and calibration.native_evaluation
            == ObjectIdentity.from_record(
                operands.native_evaluation.evaluation_id, operands.native_evaluation
            )
            and {
                calibration.identity,
                calibration.native_evaluation,
                calibration.native_task_receipt,
            }
            <= sources
            and payload == self.expected_payload
            and payload.calibration == calibration.identity
            and payload.q == calibration.q
            and calibration.boundary == self.boundary
            and payload.development_manifest == calibration.development_manifest
            and payload.frozen_coefficients == calibration.frozen_coefficients
            and candidate.physical_independent_unit_ids == calibration.root_ids
            and valid_calibration_root_ids(calibration.root_ids)
            and candidate.dataset_or_projection == calibration.identity
            and candidate.obligation_template.uncertainty_confidence_level == D(".90")
            and candidate.obligation_template.interval_quantity_ids
            == tuple(sorted(q.quantity_id for q in output_quantities()))
            and all(known.get(q.quantity_id) == q for q in required_quantities)
            and candidate.claim_template.interface_input_quantity_ids
            == tuple(sorted((*inputs, *system.relation.action_quantity_ids)))
            and candidate.outcome_access is OutcomeAccess.EVALUATOR_REVEAL
            and candidate.visibility_ceiling is VisibilityCeiling.PROSPECTIVE
            and all(
                e.outcome_access is OutcomeAccess.EVALUATOR_REVEAL
                and e.visibility_ceiling is VisibilityCeiling.PROSPECTIVE
                for e in candidate.evidence_links
            )
            and len(payload.canonical_bytes()) <= MAXIMUM_LAW_BYTES
        )
        status = (
            O.UNEVALUABLE
            if not custody
            else O.FAILED
            if calibration.q is None
            else O.SATISFIED
        )
        metrics = calibration_metrics(calibration) if custody else ()
        checks = tuple(
            sorted(
                (
                    AdequacyCheckResult(
                        f"check.{self.prefix}.{kind.value.lower().replace('_', '-')}",
                        kind,
                        status
                        if kind is K.HELD_OUT_CALIBRATION
                        else O.SATISFIED
                        if custody
                        else O.UNEVALUABLE,
                        True,
                        metrics,
                        ("FLH_CUSTODY_UNACCOUNTED_OR_CONTRADICTORY",)
                        if not custody
                        else ("FLH_NONFINITE_CALIBRATION_QUANTILE",)
                        if kind is K.HELD_OUT_CALIBRATION and status is O.FAILED
                        else (),
                        links if custody else (),
                    )
                    for kind in K
                ),
                key=lambda c: c.check_id,
            )
        )
        diagnostics = UncertaintyDecomposition(
            f"{self.prefix}.component-diagnostics",
            tuple(
                sorted(
                    (
                        UncertaintyComponent(
                            f"{self.prefix}.{kind.value.lower()}",
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
                    key=lambda c: c.component_id,
                )
            ),
        )
        template = candidate.obligation_template
        joint = JointPredictiveUncertainty(
            f"{self.prefix}.joint-calibration",
            template.uncertainty_method_key,
            calibration.identity if custody else None,
            payload.frozen_coefficients,
            template.independent_unit_id,
            candidate.physical_independent_unit_ids,
            "four-native-pairs-eight-outputs-two-futures-two-views",
            template.interval_quantity_ids,
            template.support_id,
            D(".90"),
            (NamedDecimal("normalized-quantile", calibration.q, "1"),)
            if custody and calibration.q is not None
            else (),
            tuple(
                sorted(
                    (
                        NamedDecimal(q.quantity_id, d / 8, q.native_unit)
                        for q, d in zip(
                            output_quantities(), FiniteResponseLawScienceSpec().delta, strict=True
                        )
                    ),
                    key=lambda v: v.value_id,
                )
            ),
            template.assumption_ids,
            status,
            ()
            if status is O.SATISFIED
            else ("FLH_NONFINITE_CALIBRATION_QUANTILE",)
            if custody
            else ("FLH_CUSTODY_UNACCOUNTED_OR_CONTRADICTORY",),
            links if custody else (),
        )
        return assess_joint_predictive_candidate(
            candidate=candidate,
            profile=self.profile,
            owner=self.owner,
            prefix=self.prefix,
            checks=checks,
            component_diagnostics=diagnostics,
            joint_uncertainty=joint,
        )
