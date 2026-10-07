"""Finite empirical operands reduced by the established source/law qualification owners."""

from dataclasses import dataclass, replace
from decimal import Decimal

from empirical_lawhood.adapters.methods.contracts import CandidateQualificationEligibility, LawCandidateEvidence, MethodEvidenceAvailability, ComponentQualificationAssessment
from empirical_lawhood.adapters.methods.qualification_profiles import MethodEquivalentProfileKind, ComponentQualificationProfile, QualificationProofOwner, _method_trace, _property_qualifications, _terminal_dispositions, build_method_equivalent_qualification_profile
from empirical_lawhood.kernel.identification import (
    AdequacyCheckKind,
    AdequacyCheckResult,
    UncertaintyClass,
    UncertaintyComponent,
    UncertaintyDecomposition,
)
from empirical_lawhood.kernel.laws import LawRepresentationKind
from empirical_lawhood.kernel.obligations import ObligationStatus
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.status import ScientificStatus
from empirical_lawhood.kernel.systems import SystemSpec
from .law import FEATURE_QUANTITIES, LAW_KEY, LAW_MAXIMUM_BYTES, PreparationLawPayload


PROFILE_PREFIX = "matrix-preparation-finite-observable"
SCREEN_EVIDENCE = f"{PROFILE_PREFIX}.development-operands"


@dataclass(frozen=True, slots=True)
class PreparationQualificationProfile:
    owner: QualificationProofOwner

    @property
    def profile(self) -> ComponentQualificationProfile:
        base = build_method_equivalent_qualification_profile(
            self.owner, MethodEquivalentProfileKind.CONTROLLED_IO
        )

        def name(value: str) -> str:
            return value.replace("controlled-io", PROFILE_PREFIX)

        return replace(
            base,
            profile_id=f"profile.{PROFILE_PREFIX}.exposed-development",
            applicable_method_keys=(LAW_KEY,),
            applicable_representation_kinds=(LawRepresentationKind.FINITE_ACTION_OPERATOR,),
            applicable_payload_schemas=(PreparationLawPayload.SCHEMA,),
            applicable_extension_schemas=(PreparationLawPayload.SCHEMA,),
            proof_owner_bindings=tuple(
                replace(
                    b,
                    binding_id=name(b.binding_id),
                    output_id=name(b.output_id),
                    input_schema=PreparationLawPayload.SCHEMA
                    if b.input_schema == base.applicable_payload_schemas[0]
                    else b.input_schema,
                    rule_id=name(b.rule_id),
                    rule_semantics="Frozen five-readout three-action observable prediction, sixteen held exposed roots, paired native views and nonpromotable development visibility; empirical mixture intervals do not identify separate uncertainty components.",
                )
                for b in base.proof_owner_bindings
            ),
            facet_ids=tuple(name(v) for v in base.facet_ids),
            conformance_case_ids=tuple(name(v) for v in base.conformance_case_ids),
        )

    def evaluate_candidate(
        self, system: SystemSpec, candidate: LawCandidateEvidence, payload: CanonicalRecord
    ) -> ComponentQualificationAssessment:
        if (
            not isinstance(payload, PreparationLawPayload)
            or candidate.method_key != LAW_KEY
            or candidate.representation_kind is not LawRepresentationKind.FINITE_ACTION_OPERATOR
        ):
            raise TypeError("finite observable profile received another method/payload family")
        receipts = tuple(
            r for r in candidate.method_receipts if r.evidence_kind_id == SCREEN_EVIDENCE
        )
        if (
            len(receipts) != 1
            or receipts[0].availability is not MethodEvidenceAvailability.OBSERVED
        ):
            raise ValueError("finite observable profile requires its exact raw development receipt")
        metrics = {m.value_id: m for m in receipts[0].metrics}
        evidence = receipts[0].evidence_link_ids

        def number(key: str, unit: str = "1") -> Decimal | None:
            value = metrics.get(key)
            if value is None:
                return None
            if value.unit != unit:
                raise ValueError("finite observable qualification changes a native unit")
            return value.value

        def bound(
            key: str, threshold: str, *, minimum: bool = False, unit: str = "1"
        ) -> bool | None:
            value = number(key, unit)
            return (
                None
                if value is None
                else value >= Decimal(threshold)
                if minimum
                else value <= Decimal(threshold)
            )

        def intersection(*values: bool | None) -> bool | None:
            return False if False in values else None if None in values else True

        unit = "hilbert-schmidt-native"
        closure = intersection(
            bound("absolute-rmse", ".125", unit=unit), bound("odd-rmse", ".015625", unit=unit)
        )
        informative = number("informative-action-roots")
        decisions = {
            AdequacyCheckKind.COORDINATE_ADEQUACY: len(candidate.physical_independent_unit_ids)
            == 16
            and len(candidate.axis_map.qualification_view_ids) == 2
            and len(candidate.axis_map.denominator_member_ids) == 5
            and set(FEATURE_QUANTITIES).issubset(
                candidate.claim_template.interface_input_quantity_ids
            ),
            AdequacyCheckKind.WITHIN_CELL_RECURRENCE: bound(
                "minimum-audit-bundles", "4", minimum=True
            ),
            AdequacyCheckKind.ONE_FACTOR_EXCHANGE: bound(
                "minimum-parent-screen-roots", "16", minimum=True
            ),
            AdequacyCheckKind.SUPPORT: bound(
                "same-handoff-adequate-sharp-screen-roots", "8", minimum=True
            ),
            AdequacyCheckKind.CLOSURE_MEMORY: closure,
            AdequacyCheckKind.HELD_OUT_CALIBRATION: intersection(
                closure, bound("development-simultaneous-coverage", ".90", minimum=True)
            ),
            AdequacyCheckKind.STRUCTURAL_CONVERGENCE: intersection(
                bound("native-absolute-view-error-maximum", ".0078125", unit=unit),
                bound("native-odd-view-error-maximum", ".00390625", unit=unit),
                bound("native-numerical-semantics-qualified", "1", minimum=True),
            ),
            AdequacyCheckKind.DECISIVE_FALSIFIER: None
            if informative is None or informative < 8
            else bound("wrong-sign-loss-increase", ".0001", minimum=True),
            AdequacyCheckKind.COMPUTABILITY: bool(payload.coefficients)
            and len(payload.canonical_bytes()) <= LAW_MAXIMUM_BYTES,
            AdequacyCheckKind.EVIDENCE_VISIBILITY: candidate.visibility_ceiling.is_promotable,
        }
        checks = tuple(
            sorted(
                (
                    AdequacyCheckResult(
                        f"check.{PROFILE_PREFIX}.{kind.value.lower().replace('_', '-')}",
                        kind,
                        ObligationStatus.SATISFIED
                        if passed is True
                        else ObligationStatus.FAILED
                        if passed is False
                        else ObligationStatus.UNEVALUABLE,
                        True,
                        receipts[0].metrics,
                        ()
                        if passed is True
                        else (
                            f"{PROFILE_PREFIX}-{kind.value.lower().replace('_', '-')}-{'failed' if passed is False else 'unevaluable'}",
                        ),
                        evidence if passed is True else (),
                    )
                    for kind, passed in decisions.items()
                ),
                key=lambda c: c.check_id,
            )
        )
        components = []
        for kind in UncertaintyClass:
            value = (
                number("native-absolute-view-error-maximum", unit)
                if kind is UncertaintyClass.NUMERICAL
                else Decimal(0)
                if kind is UncertaintyClass.OBSERVATION
                else None
            )
            transport = kind is UncertaintyClass.TRANSPORT
            components.append(
                UncertaintyComponent(
                    f"uncertainty.{PROFILE_PREFIX}.{kind.value.lower()}",
                    kind,
                    ObligationStatus.NOT_APPLICABLE
                    if transport
                    else ObligationStatus.SATISFIED
                    if value is not None
                    else ObligationStatus.UNEVALUABLE,
                    (NamedDecimal(f"{kind.value.lower()}-operand", value, unit),)
                    if value is not None
                    else (),
                    ()
                    if transport or value is not None
                    else (
                        "EXPOSED_JOINT_RESIDUAL_ENVELOPE_DOES_NOT_SEPARATE_ALEATORIC_AND_EPISTEMIC_COMPONENTS",
                    ),
                    evidence if value is not None else (),
                )
            )
        uncertainty = UncertaintyDecomposition(
            f"uncertainty.{candidate.candidate_id}",
            tuple(sorted(components, key=lambda c: c.component_id)),
        )
        profile = self.profile
        terminal = _terminal_dispositions(
            candidate_id=candidate.candidate_id,
            checks=checks,
            uncertainty=uncertainty,
            profile=profile,
            fallback_evidence_ids=evidence,
        )
        trace = _method_trace(
            candidate=candidate,
            prefix=PROFILE_PREFIX,
            profile=profile,
            checks=checks,
            uncertainty=uncertainty,
        )
        status = (
            ScientificStatus.SUPPORTED
            if terminal.claim_ready
            else ScientificStatus.NOT_SUPPORTED
            if False in decisions.values()
            else ScientificStatus.UNEVALUABLE
        )
        properties = _property_qualifications(
            candidate=candidate,
            owner=self.owner,
            status=status,
            reason=""
            if status is ScientificStatus.SUPPORTED
            else "FINITE_OBSERVABLE_DEVELOPMENT_DOES_NOT_QUALIFY_PROTECTED_LAW",
        )
        eligibility = (
            CandidateQualificationEligibility.NONPROMOTABLE
            if not candidate.visibility_ceiling.is_promotable
            else CandidateQualificationEligibility.PASSING
            if terminal.claim_ready
            else CandidateQualificationEligibility.NONPASSING
            if status is ScientificStatus.NOT_SUPPORTED
            else CandidateQualificationEligibility.UNEVALUABLE
        )
        return ComponentQualificationAssessment(
            f"profile-assessment.{candidate.candidate_id}",
            ObjectIdentity.from_record(profile.profile_id, profile),
            ObjectIdentity.from_record(candidate.evidence_id, candidate),
            checks,
            uncertainty,
            properties,
            terminal,
            trace,
            candidate.visibility_ceiling.is_promotable,
            eligibility,
        )
