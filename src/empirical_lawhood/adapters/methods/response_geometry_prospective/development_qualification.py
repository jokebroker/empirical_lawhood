"development's finite empirical proof operands, reduced by the existing measurement through law qualification owners.\n\nThis profile supplies the scientific checks for the complete affine forecast.\nIt reuses the established facet prerequisites and terminal-obligation reducer;\nResponseLawQualificationService remains the only law finalizer.\n"

from dataclasses import dataclass, replace
from decimal import Decimal

import numpy as np

from empirical_lawhood.adapters.methods.contracts import CandidateQualificationEligibility, LawCandidateEvidence, MethodEvidenceAvailability, ComponentQualificationAssessment
from empirical_lawhood.adapters.methods.qualification_profiles import MethodEquivalentProfileKind, ComponentQualificationProfile, QualificationProofOwner, _method_trace, _property_qualifications, _terminal_dispositions, build_method_equivalent_qualification_profile
from empirical_lawhood.kernel.evidence import EvidenceRung
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

from .development_law import DEVELOPMENT_LAW_KEY, DEVELOPMENT_LAW_MAXIMUM_BYTES, DEVELOPMENT_LATENT_QUANTITIES, DEVELOPMENT_INVOCATION_QUANTITY, ResponseGeometryDevelopmentAffineLawPayload
from .development_models import propagate_response_geometry_development_latent


DEVELOPMENT_PROFILE_PREFIX = "response-geometry-development-affine"
DEVELOPMENT_SCREEN_EVIDENCE = f"{DEVELOPMENT_PROFILE_PREFIX}.validation-operands"


@dataclass(frozen=True, slots=True)
class ResponseGeometryDevelopmentAffineQualificationProfile:
    owner: QualificationProofOwner

    @property
    def profile(self) -> ComponentQualificationProfile:
        # Compatibility is the established generic check/facet/obligation graph,
        # not an assertion that development's stochastic payload is a Bu-only version set.
        base = build_method_equivalent_qualification_profile(
            self.owner, MethodEquivalentProfileKind.CONTROLLED_IO
        )

        def name(value: str) -> str:
            return value.replace("controlled-io", DEVELOPMENT_PROFILE_PREFIX)

        return replace(
            base,
            profile_id=f"profile.{DEVELOPMENT_PROFILE_PREFIX}.finite-empirical",
            applicable_method_keys=(DEVELOPMENT_LAW_KEY,),
            applicable_representation_kinds=(LawRepresentationKind.LOCAL_STATE_SPACE,),
            applicable_payload_schemas=(ResponseGeometryDevelopmentAffineLawPayload.SCHEMA,),
            applicable_extension_schemas=(ResponseGeometryDevelopmentAffineLawPayload.SCHEMA,),
            proof_owner_bindings=tuple(
                replace(
                    binding,
                    binding_id=name(binding.binding_id),
                    output_id=name(binding.output_id),
                    input_schema=ResponseGeometryDevelopmentAffineLawPayload.SCHEMA
                    if binding.input_schema == base.applicable_payload_schemas[0]
                    else binding.input_schema,
                    rule_id=name(binding.rule_id),
                    rule_semantics="Apply the frozen development finite empirical response, complete initial/affine/noise, whole-root and paired-native-view rules; reuse common prerequisite and terminal reductions.",
                )
                for binding in base.proof_owner_bindings
            ),
            facet_ids=tuple(name(value) for value in base.facet_ids),
            conformance_case_ids=tuple(name(value) for value in base.conformance_case_ids),
        )

    def evaluate_candidate(
        self,
        system: SystemSpec,
        candidate: LawCandidateEvidence,
        payload: CanonicalRecord,
    ) -> ComponentQualificationAssessment:
        if (
            not isinstance(payload, ResponseGeometryDevelopmentAffineLawPayload)
            or candidate.method_key != DEVELOPMENT_LAW_KEY
            or candidate.representation_kind is not LawRepresentationKind.LOCAL_STATE_SPACE
        ):
            raise TypeError("development affine profile received another method/payload family")
        receipts = tuple(
            r for r in candidate.method_receipts if r.evidence_kind_id == DEVELOPMENT_SCREEN_EVIDENCE
        )
        if (
            len(receipts) != 1
            or receipts[0].availability is not MethodEvidenceAvailability.OBSERVED
        ):
            raise ValueError("development profile requires its exact raw validation-operands receipt")
        metrics = {value.value_id: value for value in receipts[0].metrics}
        evidence_ids = receipts[0].evidence_link_ids

        def number(key: str, unit: str = "1") -> Decimal | None:
            item = metrics.get(key)
            if item is None:
                return None
            if item.unit != unit:
                raise ValueError("development qualification operand changes its native unit")
            return item.value

        def bound(
            key: str,
            threshold: Decimal,
            *,
            minimum: bool = False,
            unit: str = "1",
            absolute: bool = False,
        ) -> bool | None:
            value = number(key, unit)
            if value is None:
                return None
            value = abs(value) if absolute else value
            return value >= threshold if minimum else value <= threshold

        def intersection(*values: bool | None) -> bool | None:
            return False if False in values else None if None in values else True

        unit = "hilbert-schmidt-native"
        dimension = payload.members[0].dimension if payload.members else 0
        coordinates = (
            bool(payload.members)
            and len(candidate.physical_independent_unit_ids) == 32
            and len(candidate.axis_map.qualification_view_ids) == 2
            and len(system.relation.receiver_quantity_ids) == 1
            and {*DEVELOPMENT_LATENT_QUANTITIES[:dimension], DEVELOPMENT_INVOCATION_QUANTITY}.issubset(
                candidate.claim_template.interface_input_quantity_ids
            )
            and len({m.parent for m in payload.members}) == 5
        )
        closure = intersection(
            bound("mean-error", Decimal(".03125"), unit=unit, absolute=True),
            bound("rmse", Decimal(".125"), unit=unit),
        )
        calibration = intersection(
            closure,
            bound("joint-coverage", Decimal(".90"), minimum=True),
            bound("median-root-maximum-halfwidth", Decimal(".25"), unit=unit),
        )
        informative = number("informative-action-roots")
        falsifier = (
            None
            if informative is None or informative < 8
            else bound("wrong-sign-loss-increase", Decimal(".0001"), minimum=True)
        )
        decisions = {
            AdequacyCheckKind.COORDINATE_ADEQUACY: coordinates,
            AdequacyCheckKind.SUPPORT: bound("usable-roots", Decimal(16), minimum=True),
            AdequacyCheckKind.WITHIN_CELL_RECURRENCE: bound(
                "minimum-invocation-repeats", Decimal(2), minimum=True
            ),
            AdequacyCheckKind.ONE_FACTOR_EXCHANGE: bound(
                "minimum-parent-contact-roots", Decimal(8), minimum=True
            ),
            AdequacyCheckKind.DECISIVE_FALSIFIER: falsifier,
            AdequacyCheckKind.HELD_OUT_CALIBRATION: calibration,
            AdequacyCheckKind.CLOSURE_MEMORY: closure,
            AdequacyCheckKind.STRUCTURAL_CONVERGENCE: bound(
                "maximum-paired-numerical-error", Decimal(".03125"), unit=unit
            ),
            AdequacyCheckKind.COMPUTABILITY: bool(payload.members)
            and len(payload.canonical_bytes()) <= DEVELOPMENT_LAW_MAXIMUM_BYTES,
            AdequacyCheckKind.EVIDENCE_VISIBILITY: candidate.visibility_ceiling.is_promotable,
        }
        checks = tuple(
            sorted(
                (
                    AdequacyCheckResult(
                        f"check.{DEVELOPMENT_PROFILE_PREFIX}.{kind.value.lower().replace('_', '-')}",
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
                            f"{DEVELOPMENT_PROFILE_PREFIX}-{kind.value.lower().replace('_', '-')}-{'failed' if passed is False else 'unevaluable'}",
                        ),
                        evidence_ids if passed is True else (),
                    )
                    for kind, passed in decisions.items()
                ),
                key=lambda value: value.check_id,
            )
        )
        sigma = None
        if payload.members:
            try:
                with np.errstate(over="raise", invalid="raise"):
                    sigma = max(
                        float(propagate_response_geometry_development_latent(np.zeros(m.dimension), *m.arrays())[1][-1])
                        for m in payload.members
                    )
            except (ValueError, FloatingPointError):
                pass
        components = []
        for kind in UncertaintyClass:
            value = (
                number("maximum-paired-numerical-error", unit)
                if kind is UncertaintyClass.NUMERICAL
                else Decimal(format(sigma, ".17g"))
                if kind is UncertaintyClass.ALEATORIC and sigma is not None
                else payload.calibration_quantile * Decimal(format(sigma, ".17g"))
                if kind is UncertaintyClass.EPISTEMIC
                and sigma is not None
                and payload.calibration_quantile is not None
                else Decimal(0)
                if kind is UncertaintyClass.OBSERVATION
                else None
            )
            available = value is not None and value.is_finite()
            transport = kind is UncertaintyClass.TRANSPORT
            components.append(
                UncertaintyComponent(
                    f"uncertainty.{DEVELOPMENT_PROFILE_PREFIX}.{kind.value.lower()}",
                    kind,
                    ObligationStatus.NOT_APPLICABLE
                    if transport
                    else ObligationStatus.SATISFIED
                    if available
                    else ObligationStatus.UNEVALUABLE,
                    (NamedDecimal(f"{kind.value.lower()}-bound", value, unit),)
                    if available and value is not None
                    else (),
                    () if transport or available else ("DEVELOPMENT_UNCERTAINTY_OPERAND_UNAVAILABLE",),
                    evidence_ids if available else (),
                )
            )
        uncertainty = UncertaintyDecomposition(
            f"uncertainty.{candidate.candidate_id}",
            tuple(sorted(components, key=lambda value: value.component_id)),
        )
        profile = self.profile
        terminal = _terminal_dispositions(
            candidate_id=candidate.candidate_id,
            checks=checks,
            uncertainty=uncertainty,
            profile=profile,
            fallback_evidence_ids=evidence_ids,
        )
        trace = _method_trace(
            candidate=candidate,
            prefix=DEVELOPMENT_PROFILE_PREFIX,
            profile=profile,
            checks=checks,
            uncertainty=uncertainty,
        )
        status = (
            ScientificStatus.SUPPORTED
            if terminal.claim_ready
            else ScientificStatus.UNEVALUABLE
            if any(v is None for v in decisions.values())
            else ScientificStatus.NOT_SUPPORTED
        )
        properties = _property_qualifications(
            candidate=candidate,
            owner=self.owner,
            status=status,
            reason=""
            if status is ScientificStatus.SUPPORTED
            else "DEVELOPMENT_FINITE_AFFINE_RESPONSE_UNQUALIFIED",
        )
        eligibility = (
            CandidateQualificationEligibility.NONPROMOTABLE
            if not candidate.visibility_ceiling.is_promotable
            else CandidateQualificationEligibility.PASSING
            if terminal.claim_ready and trace.highest_supported_rung is EvidenceRung.LOCAL_LAW
            else CandidateQualificationEligibility.UNEVALUABLE
            if status is ScientificStatus.UNEVALUABLE
            else CandidateQualificationEligibility.NONPASSING
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
