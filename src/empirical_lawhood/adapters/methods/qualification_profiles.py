"""Registered proof-owner profiles for generic response-law qualification."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import EvidenceRung
from empirical_lawhood.kernel.identification import AdequacyCheckKind, AdequacyCheckResult, ResponseQualificationFacetAssessment, ResponseQualificationTrace, StructuralConvergenceResult, TerminalLawObligationAssessment, TerminalObligationDisposition, TerminalObligationKind, UncertaintyComponent, UncertaintyClass, UncertaintyDecomposition
from empirical_lawhood.kernel.laws import LawRepresentationKind
from empirical_lawhood.kernel.obligations import ObligationStatus
from empirical_lawhood.kernel.provenance import EvidenceLink, ObjectIdentity
from empirical_lawhood.kernel.predictive_uncertainty import JointPredictiveUncertainty
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_schema,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.status import ScientificStatus
from empirical_lawhood.kernel.systems import SystemSpec

from .contracts import CandidateFit, CandidateQualificationEligibility, CandidateMethodEvidenceReceipt, FiniteActionCellDisposition, FiniteActionCompatibilitySetExtension, IdentificationDataset, LawCandidateEvidence, LawIdentificationConfig, MethodEvidenceAvailability, ParametricLawModel, PropertyQualification, ComponentQualificationAssessment, JointQualificationAssessment
from .receiver_conditioned_io.contracts import (
    ControlledIOProductDisposition,
    ControlledIOVersionSet,
)
from .receiver_conditioned_io.controlled_io import ControlledIOEvaluator
from .services import AdequacyServices, NumericalQualifier, UncertaintyService


class QualificationProofOutputKind(StrEnum):
    ADEQUACY_CHECK = "ADEQUACY_CHECK"
    UNCERTAINTY_CLASS = "UNCERTAINTY_CLASS"
    TERMINAL_OBLIGATION = "TERMINAL_OBLIGATION"
    TRACE_FACET = "TRACE_FACET"


@dataclass(frozen=True, slots=True)
class QualificationProofOwner(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/qualification-proof-owner'

    owner_id: str
    capability_key: str
    capability_version: str
    implementation_sha256: str

    def __post_init__(self) -> None:
        validate_stable_id(self.owner_id, field_name="owner_id")
        validate_stable_id(self.capability_key, field_name="capability_key")
        validate_semantic_version(self.capability_version)
        validate_sha256(self.implementation_sha256, field_name="implementation_sha256")


@dataclass(frozen=True, slots=True)
class QualificationProofOwnerBinding(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/qualification-proof-owner-binding'

    binding_id: str
    output_kind: QualificationProofOutputKind
    output_id: str
    proof_owner: ObjectIdentity
    input_schema: str
    output_schema: str
    rule_id: str
    rule_semantics: str

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")
        validate_stable_id(self.output_id, field_name="output_id")
        validate_stable_id(self.rule_id, field_name="rule_id")
        validate_schema(self.input_schema)
        validate_schema(self.output_schema)
        if not self.rule_semantics.strip():
            raise ValueError("qualification proof binding requires frozen rule semantics")


@dataclass(frozen=True, slots=True)
class ComponentQualificationProfile(CanonicalRecord):
    """Closed compatibility and proof-ownership contract for one method family."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/component-qualification-profile'

    profile_id: str
    profile_version: str
    applicable_method_keys: tuple[str, ...]
    applicable_representation_kinds: tuple[LawRepresentationKind, ...]
    applicable_payload_schemas: tuple[str, ...]
    applicable_extension_schemas: tuple[str, ...]
    proof_owner_bindings: tuple[QualificationProofOwnerBinding, ...]
    facet_ids: tuple[str, ...]
    allowed_not_applicable_output_ids: tuple[str, ...]
    status_precedence_ids: tuple[str, ...]
    conformance_case_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.profile_id, field_name="profile_id")
        validate_semantic_version(self.profile_version)
        require_sorted_unique_strings(
            self.applicable_method_keys,
            field_name="applicable_method_keys",
            allow_empty=False,
        )
        if tuple(sorted(set(self.applicable_representation_kinds))) != tuple(
            sorted(self.applicable_representation_kinds)
        ):
            raise ValueError("profile representation kinds must be sorted and unique")
        if not self.applicable_representation_kinds:
            raise ValueError("qualification profile requires representation kinds")
        require_sorted_unique_strings(
            self.applicable_payload_schemas,
            field_name="applicable_payload_schemas",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.applicable_extension_schemas,
            field_name="applicable_extension_schemas",
        )
        for schema in (*self.applicable_payload_schemas, *self.applicable_extension_schemas):
            validate_schema(schema)
        if not set(self.applicable_extension_schemas).issubset(self.applicable_payload_schemas):
            raise ValueError("profile extension schemas must be registered payload schemas")
        require_sorted_unique_ids(
            self.proof_owner_bindings,
            attribute="binding_id",
            field_name="proof_owner_bindings",
        )
        require_sorted_unique_strings(self.facet_ids, field_name="facet_ids", allow_empty=False)
        for name, values in (
            ("allowed_not_applicable_output_ids", self.allowed_not_applicable_output_ids),
            ("status_precedence_ids", self.status_precedence_ids),
            ("conformance_case_ids", self.conformance_case_ids),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)
        keys = {(value.output_kind, value.output_id) for value in self.proof_owner_bindings}
        if len(keys) != len(self.proof_owner_bindings):
            raise ValueError("qualification output has more than one proof owner")
        required = {
            *(
                (QualificationProofOutputKind.ADEQUACY_CHECK, value.value.lower())
                for value in AdequacyCheckKind
            ),
            *(
                (QualificationProofOutputKind.UNCERTAINTY_CLASS, value.value.lower())
                for value in UncertaintyClass
            ),
            *(
                (QualificationProofOutputKind.TERMINAL_OBLIGATION, value.value.lower())
                for value in TerminalObligationKind
            ),
            *((QualificationProofOutputKind.TRACE_FACET, value) for value in self.facet_ids),
        }
        if keys != required:
            raise ValueError("qualification profile proof ownership is incomplete")

    def proof_owner(
        self,
        output_kind: QualificationProofOutputKind,
        output_id: str,
    ) -> ObjectIdentity:
        return next(
            value.proof_owner
            for value in self.proof_owner_bindings
            if value.output_kind is output_kind and value.output_id == output_id
        )


_RESPONSE_METHOD_FACET_RUNG = {
    AdequacyCheckKind.COORDINATE_ADEQUACY: EvidenceRung.MEASUREMENT,
    AdequacyCheckKind.SUPPORT: EvidenceRung.MEASUREMENT,
    AdequacyCheckKind.EVIDENCE_VISIBILITY: EvidenceRung.MEASUREMENT,
    AdequacyCheckKind.WITHIN_CELL_RECURRENCE: EvidenceRung.ORDER_RELATION,
    AdequacyCheckKind.ONE_FACTOR_EXCHANGE: EvidenceRung.ORDER_RELATION,
    AdequacyCheckKind.DECISIVE_FALSIFIER: EvidenceRung.ORDER_RELATION,
    AdequacyCheckKind.HELD_OUT_CALIBRATION: EvidenceRung.RESPONSE,
    AdequacyCheckKind.CLOSURE_MEMORY: EvidenceRung.RESPONSE,
    AdequacyCheckKind.STRUCTURAL_CONVERGENCE: EvidenceRung.LOCAL_LAW,
    AdequacyCheckKind.COMPUTABILITY: EvidenceRung.LOCAL_LAW,
}


def _facet_id(kind: AdequacyCheckKind) -> str:
    return f"response-method.{kind.value.lower().replace('_', '-')}"


@dataclass(frozen=True, slots=True)
class JointQualificationProfile(ComponentQualificationProfile):
    """Explicit joint-predictive basis, retaining component diagnostics separately."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/joint-qualification-profile'
    joint_uncertainty_schema: str = JointPredictiveUncertainty.SCHEMA

    def __post_init__(self) -> None:
        ComponentQualificationProfile.__post_init__(self)
        if self.joint_uncertainty_schema != JointPredictiveUncertainty.SCHEMA:
            raise ValueError("joint profile requires its explicit supported uncertainty basis")


_RESPONSE_METHOD_FACET_IDS = tuple(
    sorted((*(_facet_id(value) for value in AdequacyCheckKind), "response-method.uncertainty"))
)


def build_response_method_qualification_profile(owner: QualificationProofOwner) -> ComponentQualificationProfile:
    owner_identity = ObjectIdentity.from_record(owner.owner_id, owner)
    input_schema = 'empirical-lawhood/methods/law-candidate-evidence'
    bindings: list[QualificationProofOwnerBinding] = []
    for kind in AdequacyCheckKind:
        output_id = kind.value.lower()
        bindings.append(
            QualificationProofOwnerBinding(
                binding_id=f"proof.adequacy.{output_id.replace('_', '-')}",
                output_kind=QualificationProofOutputKind.ADEQUACY_CHECK,
                output_id=output_id,
                proof_owner=owner_identity,
                input_schema=input_schema,
                output_schema='empirical-lawhood/kernel/adequacy-check-result',
                rule_id=f"response-method-rule.adequacy.{output_id.replace('_', '-')}",
                rule_semantics=(
                    "Apply the frozen parametric response-method adequacy service rule for this exact check to the reference fit, independent-unit evidence and structural qualifier."
                ),
            )
        )
    for uncertainty_class in UncertaintyClass:
        output_id = uncertainty_class.value.lower()
        bindings.append(
            QualificationProofOwnerBinding(
                binding_id=f"proof.uncertainty.{output_id}",
                output_kind=QualificationProofOutputKind.UNCERTAINTY_CLASS,
                output_id=output_id,
                proof_owner=owner_identity,
                input_schema=input_schema,
                output_schema='empirical-lawhood/kernel/uncertainty-component',
                rule_id=f"response-method-rule.uncertainty.{output_id}",
                rule_semantics=(
                    "Apply the frozen parametric response-method uncertainty decomposition rule to the reference fit without pooling numerical, aleatoric, epistemic, transport or observation components."
                ),
            )
        )
    for obligation in TerminalObligationKind:
        output_id = obligation.value.lower()
        bindings.append(
            QualificationProofOwnerBinding(
                binding_id=f"proof.obligation.{output_id.replace('_', '-')}",
                output_kind=QualificationProofOutputKind.TERMINAL_OBLIGATION,
                output_id=output_id,
                proof_owner=owner_identity,
                input_schema=input_schema,
                output_schema='empirical-lawhood/kernel/terminal-obligation-disposition',
                rule_id=f"response-method-rule.obligation.{output_id.replace('_', '-')}",
                rule_semantics=(
                    "Reduce only the exact frozen parametric response-method proof operands mapped to this terminal obligation under noncompensating precedence."
                ),
            )
        )
    for facet_id in _RESPONSE_METHOD_FACET_IDS:
        bindings.append(
            QualificationProofOwnerBinding(
                binding_id=f"proof.facet.{facet_id.replace('.', '-')}",
                output_kind=QualificationProofOutputKind.TRACE_FACET,
                output_id=facet_id,
                proof_owner=owner_identity,
                input_schema=input_schema,
                output_schema='empirical-lawhood/kernel/response-qualification-facet-assessment',
                rule_id=f"response-method-rule.facet.{facet_id.replace('.', '-')}",
                rule_semantics=(
                    "Project the owned parametric response-method check or uncertainty operand to its declared rung without changing prerequisite or evidence identity."
                ),
            )
        )
    return ComponentQualificationProfile(
        profile_id="profile.response-method.point-parametric",
        profile_version="1.0.0",
        applicable_method_keys=(
            "baseline.local-linear",
            "baseline.local-state-space",
            "baseline.nonlinear-local",
        ),
        applicable_representation_kinds=(
            LawRepresentationKind.FINITE_ACTION_OPERATOR,
            LawRepresentationKind.LOCAL_STATE_SPACE,
        ),
        applicable_payload_schemas=(ParametricLawModel.SCHEMA,),
        applicable_extension_schemas=(),
        proof_owner_bindings=tuple(sorted(bindings, key=lambda value: value.binding_id)),
        facet_ids=_RESPONSE_METHOD_FACET_IDS,
        allowed_not_applicable_output_ids=("observation", "transport"),
        status_precedence_ids=(
            "complete-mixed-partial",
            "complete-not-supported",
            "complete-supported",
            "complete-unevaluable-not-tested",
            "contract-invalid-no-assessment",
            "operational-blocked-no-assessment",
        ),
        conformance_case_ids=(
            "response-method-missing-exchange",
            "response-method-non-closing",
            "response-method-outcome-visible",
            "response-method-stable-linear",
            "response-method-structurally-unstable",
            "response-method-wrong-action-rejection",
        ),
    )


def _status_from_obligations(statuses: tuple[ObligationStatus, ...]) -> ObligationStatus:
    if all(
        value in {ObligationStatus.SATISFIED, ObligationStatus.NOT_APPLICABLE} for value in statuses
    ):
        return ObligationStatus.SATISFIED
    if any(value is ObligationStatus.FAILED for value in statuses):
        return ObligationStatus.FAILED
    if any(value is ObligationStatus.UNEVALUABLE for value in statuses):
        return ObligationStatus.UNEVALUABLE
    return ObligationStatus.REQUIRED


def _uncertainty_operands(
    uncertainty: UncertaintyDecomposition,
    profile: ComponentQualificationProfile,
    joint: JointPredictiveUncertainty | None,
) -> tuple[ObligationStatus, tuple[str, ...], tuple[str, ...]]:
    if isinstance(profile, JointQualificationProfile):
        if joint is None:
            raise ValueError("joint qualification profile requires its joint uncertainty operand")
        return joint.status, joint.reason_codes, joint.evidence_link_ids
    if joint is not None:
        raise ValueError("Qualification cannot replace its component-decomposition basis")
    return (
        _status_from_obligations(tuple(v.status for v in uncertainty.components)),
        tuple(
            sorted(
                {
                    r
                    for v in uncertainty.components
                    for r in v.limitation_codes
                    if v.status not in {ObligationStatus.SATISFIED, ObligationStatus.NOT_APPLICABLE}
                }
            )
        ),
        tuple(sorted({e for v in uncertainty.components for e in v.evidence_link_ids})),
    )


def _terminal_dispositions(
    *,
    candidate_id: str,
    checks: tuple[AdequacyCheckResult, ...],
    uncertainty: UncertaintyDecomposition,
    profile: ComponentQualificationProfile,
    fallback_evidence_ids: tuple[str, ...],
    joint_uncertainty: JointPredictiveUncertainty | None = None,
) -> TerminalLawObligationAssessment:
    by_kind = {value.kind: value for value in checks}
    mapping = {
        TerminalObligationKind.SUPPORT: (by_kind[AdequacyCheckKind.SUPPORT],),
        TerminalObligationKind.VALIDITY: (
            by_kind[AdequacyCheckKind.COORDINATE_ADEQUACY],
            by_kind[AdequacyCheckKind.WITHIN_CELL_RECURRENCE],
            by_kind[AdequacyCheckKind.HELD_OUT_CALIBRATION],
        ),
        TerminalObligationKind.FALSIFIER: (
            by_kind[AdequacyCheckKind.DECISIVE_FALSIFIER],
            by_kind[AdequacyCheckKind.ONE_FACTOR_EXCHANGE],
        ),
        TerminalObligationKind.CLOSURE: (
            by_kind[AdequacyCheckKind.WITHIN_CELL_RECURRENCE],
            by_kind[AdequacyCheckKind.CLOSURE_MEMORY],
        ),
        TerminalObligationKind.STRUCTURAL_QUALIFICATION: (
            by_kind[AdequacyCheckKind.STRUCTURAL_CONVERGENCE],
        ),
        TerminalObligationKind.COMPUTABILITY: (by_kind[AdequacyCheckKind.COMPUTABILITY],),
        TerminalObligationKind.EVIDENCE_VISIBILITY: (
            by_kind[AdequacyCheckKind.EVIDENCE_VISIBILITY],
        ),
    }
    dispositions: list[TerminalObligationDisposition] = []
    for obligation, outputs in mapping.items():
        status = _status_from_obligations(tuple(value.status for value in outputs))
        reasons = tuple(sorted({reason for value in outputs for reason in value.reason_codes}))
        evidence_ids = tuple(
            sorted({link for value in outputs for link in value.evidence_link_ids})
        )
        if status is ObligationStatus.SATISFIED and not evidence_ids:
            evidence_ids = fallback_evidence_ids
        dispositions.append(
            TerminalObligationDisposition(
                disposition_id=f"obligation.{candidate_id}.{obligation.value.lower().replace('_', '-')}",
                kind=obligation,
                status=status,
                proof_owner=profile.proof_owner(
                    QualificationProofOutputKind.TERMINAL_OBLIGATION,
                    obligation.value.lower(),
                ),
                evidence_link_ids=evidence_ids,
                reason_codes=reasons,
            )
        )
    uncertainty_status, uncertainty_reasons, uncertainty_evidence = _uncertainty_operands(
        uncertainty, profile, joint_uncertainty
    )
    if uncertainty_status is ObligationStatus.SATISFIED and not uncertainty_evidence:
        uncertainty_evidence = fallback_evidence_ids
    dispositions.append(
        TerminalObligationDisposition(
            disposition_id=f"obligation.{candidate_id}.uncertainty",
            kind=TerminalObligationKind.UNCERTAINTY,
            status=uncertainty_status,
            proof_owner=profile.proof_owner(
                QualificationProofOutputKind.TERMINAL_OBLIGATION,
                TerminalObligationKind.UNCERTAINTY.value.lower(),
            ),
            evidence_link_ids=uncertainty_evidence,
            reason_codes=uncertainty_reasons,
        )
    )
    return TerminalLawObligationAssessment(
        assessment_id=f"terminal-obligations.{candidate_id}",
        dispositions=tuple(sorted(dispositions, key=lambda value: value.disposition_id)),
    )


def _facet_status(status: ObligationStatus) -> ScientificStatus:
    if status in {ObligationStatus.SATISFIED, ObligationStatus.NOT_APPLICABLE}:
        return ScientificStatus.SUPPORTED
    if status is ObligationStatus.UNEVALUABLE:
        return ScientificStatus.UNEVALUABLE
    if status is ObligationStatus.REQUIRED:
        return ScientificStatus.NOT_TESTED
    return ScientificStatus.NOT_SUPPORTED


def _trace(
    *,
    candidate: LawCandidateEvidence,
    checks: tuple[AdequacyCheckResult, ...],
    uncertainty: UncertaintyDecomposition,
    profile: ComponentQualificationProfile,
    physical_unit_ids: tuple[str, ...],
    information_cutoff_id: str,
) -> ResponseQualificationTrace:
    check_by_kind = {value.kind: value for value in checks}
    scientific_prerequisite_ids = tuple(
        sorted(
            {
                _facet_id(AdequacyCheckKind.COORDINATE_ADEQUACY),
                _facet_id(AdequacyCheckKind.SUPPORT),
            }
        )
    )
    facets: list[ResponseQualificationFacetAssessment] = []
    fallback_evidence = tuple(value.link_id for value in candidate.evidence_links)
    for kind in AdequacyCheckKind:
        check = check_by_kind[kind]
        facet_id = _facet_id(kind)
        if _RESPONSE_METHOD_FACET_RUNG[kind] is EvidenceRung.MEASUREMENT:
            prerequisites: tuple[str, ...] = ()
        else:
            prerequisites = scientific_prerequisite_ids
        status = _facet_status(check.status)
        evidence_ids = check.evidence_link_ids or fallback_evidence
        reasons = check.reason_codes
        if status is ScientificStatus.SUPPORTED:
            reasons = ()
        elif not reasons:
            reasons = (f"{kind.value.lower().replace('_', '-')}-unresolved",)
        facets.append(
            ResponseQualificationFacetAssessment(
                facet_id=facet_id,
                rung=_RESPONSE_METHOD_FACET_RUNG[kind],
                status=status,
                proof_owner=profile.proof_owner(
                    QualificationProofOutputKind.TRACE_FACET,
                    facet_id,
                ),
                prerequisite_facet_ids=prerequisites,
                evidence_link_ids=evidence_ids if status is ScientificStatus.SUPPORTED else (),
                reason_codes=reasons,
                maximum_supported_object_id=(
                    f"object.{facet_id}.{candidate.candidate_id}"
                    if status is ScientificStatus.SUPPORTED
                    else None
                ),
                required_for_public_rung=True,
            )
        )
    uncertainty_status = _facet_status(
        _status_from_obligations(tuple(value.status for value in uncertainty.components))
    )
    uncertainty_reasons = tuple(
        sorted(
            {
                reason
                for value in uncertainty.components
                for reason in value.limitation_codes
                if value.status not in {ObligationStatus.SATISFIED, ObligationStatus.NOT_APPLICABLE}
            }
        )
    )
    facets.append(
        ResponseQualificationFacetAssessment(
            facet_id="response-method.uncertainty",
            rung=EvidenceRung.LOCAL_LAW,
            status=uncertainty_status,
            proof_owner=profile.proof_owner(
                QualificationProofOutputKind.TRACE_FACET,
                "response-method.uncertainty",
            ),
            prerequisite_facet_ids=scientific_prerequisite_ids,
            evidence_link_ids=fallback_evidence
            if uncertainty_status is ScientificStatus.SUPPORTED
            else (),
            reason_codes=(
                ()
                if uncertainty_status is ScientificStatus.SUPPORTED
                else uncertainty_reasons or ("uncertainty-unresolved",)
            ),
            maximum_supported_object_id=(
                f"object.response-method.uncertainty.{candidate.candidate_id}"
                if uncertainty_status is ScientificStatus.SUPPORTED
                else None
            ),
            required_for_public_rung=True,
        )
    )
    ordered = tuple(sorted(facets, key=lambda value: value.facet_id))
    required = tuple(value for value in ordered if value.required_for_public_rung)
    highest = None
    for rung in (
        EvidenceRung.MEASUREMENT,
        EvidenceRung.ORDER_RELATION,
        EvidenceRung.RESPONSE,
        EvidenceRung.LOCAL_LAW,
    ):
        rank = {
            EvidenceRung.MEASUREMENT: 0,
            EvidenceRung.ORDER_RELATION: 1,
            EvidenceRung.RESPONSE: 2,
            EvidenceRung.LOCAL_LAW: 3,
        }
        if all(
            value.status is ScientificStatus.SUPPORTED
            for value in required
            if rank[value.rung] <= rank[rung]
        ):
            highest = rung
        else:
            break
    return ResponseQualificationTrace(
        trace_id=f"trace.{candidate.candidate_id}",
        candidate_id=candidate.candidate_id,
        facets=ordered,
        physical_independent_unit_ids=physical_unit_ids,
        candidate_version_member_ids=candidate.axis_map.candidate_version_member_ids,
        denominator_member_ids=candidate.axis_map.denominator_member_ids,
        qualification_view_ids=candidate.axis_map.qualification_view_ids,
        information_cutoff_id=information_cutoff_id,
        outcome_access=candidate.outcome_access,
        highest_supported_rung=highest,
    )


class MethodEquivalentProfileKind(StrEnum):
    FINITE_ACTION = "FINITE_ACTION"
    CONFIRMATORY_FINITE_ACTION = "CONFIRMATORY_FINITE_ACTION"
    CONFIRMATORY_FINITE_ACTION_LOCAL_SUPPORT = "CONFIRMATORY_FINITE_ACTION_LOCAL_SUPPORT"
    CONTROLLED_IO = "CONTROLLED_IO"


_METHOD_PROFILE_SPEC = {
    MethodEquivalentProfileKind.FINITE_ACTION: (
        "finite-action",
        "finite-action.compatibility-set",
        LawRepresentationKind.FINITE_ACTION_OPERATOR,
        FiniteActionCompatibilitySetExtension.SCHEMA,
    ),
    MethodEquivalentProfileKind.CONFIRMATORY_FINITE_ACTION: (
        "confirmatory-finite-action",
        "finite-action.confirmatory-prefix",
        LawRepresentationKind.FINITE_ACTION_OPERATOR,
        FiniteActionCompatibilitySetExtension.SCHEMA,
    ),
    MethodEquivalentProfileKind.CONFIRMATORY_FINITE_ACTION_LOCAL_SUPPORT: (
        "confirmatory-finite-action-local-support",
        "finite-action.confirmatory-prefix",
        LawRepresentationKind.FINITE_ACTION_OPERATOR,
        FiniteActionCompatibilitySetExtension.SCHEMA,
    ),
    MethodEquivalentProfileKind.CONTROLLED_IO: (
        "controlled-io",
        "controlled-io.version-set",
        LawRepresentationKind.LOCAL_STATE_SPACE,
        ControlledIOVersionSet.SCHEMA,
    ),
}


def _method_facet_id(prefix: str, kind: AdequacyCheckKind) -> str:
    return f"{prefix}.{kind.value.lower().replace('_', '-')}"


def build_method_equivalent_qualification_profile(
    owner: QualificationProofOwner,
    kind: MethodEquivalentProfileKind,
) -> ComponentQualificationProfile:
    prefix, method_key, representation, payload_schema = _METHOD_PROFILE_SPEC[kind]
    owner_identity = ObjectIdentity.from_record(owner.owner_id, owner)
    facet_ids = tuple(
        sorted(
            (
                *(_method_facet_id(prefix, value) for value in AdequacyCheckKind),
                f"{prefix}.uncertainty",
            )
        )
    )
    bindings: list[QualificationProofOwnerBinding] = []
    for check_kind in AdequacyCheckKind:
        output_id = check_kind.value.lower()
        bindings.append(
            QualificationProofOwnerBinding(
                binding_id=f"proof.{prefix}.adequacy.{output_id.replace('_', '-')}",
                output_kind=QualificationProofOutputKind.ADEQUACY_CHECK,
                output_id=output_id,
                proof_owner=owner_identity,
                input_schema=payload_schema,
                output_schema=AdequacyCheckResult.SCHEMA,
                rule_id=f"{prefix}-rule.adequacy.{output_id.replace('_', '-')}",
                rule_semantics=(
                    "Derive this generic check from the exact method-local payload and "
                    "profile-named raw receipt; a producer disposition is never accepted "
                    "as generic qualification truth."
                ),
            )
        )
    for uncertainty_class in UncertaintyClass:
        output_id = uncertainty_class.value.lower()
        bindings.append(
            QualificationProofOwnerBinding(
                binding_id=f"proof.{prefix}.uncertainty.{output_id}",
                output_kind=QualificationProofOutputKind.UNCERTAINTY_CLASS,
                output_id=output_id,
                proof_owner=owner_identity,
                input_schema=payload_schema,
                output_schema=UncertaintyComponent.SCHEMA,
                rule_id=f"{prefix}-rule.uncertainty.{output_id}",
                rule_semantics=(
                    "Require the exact profile-named independent-unit uncertainty operand; "
                    "retain unavailable and explicitly inapplicable classes without substitution."
                ),
            )
        )
    for obligation in TerminalObligationKind:
        output_id = obligation.value.lower()
        bindings.append(
            QualificationProofOwnerBinding(
                binding_id=f"proof.{prefix}.obligation.{output_id.replace('_', '-')}",
                output_kind=QualificationProofOutputKind.TERMINAL_OBLIGATION,
                output_id=output_id,
                proof_owner=owner_identity,
                input_schema=ComponentQualificationAssessment.SCHEMA,
                output_schema='empirical-lawhood/kernel/terminal-obligation-disposition',
                rule_id=f"{prefix}-rule.obligation.{output_id.replace('_', '-')}",
                rule_semantics=(
                    "Reduce the exact owned checks and uncertainty classes through the common "
                    "noncompensating terminal-obligation map."
                ),
            )
        )
    for facet_id in facet_ids:
        bindings.append(
            QualificationProofOwnerBinding(
                binding_id=f"proof.{prefix}.facet.{facet_id.replace('.', '-')}",
                output_kind=QualificationProofOutputKind.TRACE_FACET,
                output_id=facet_id,
                proof_owner=owner_identity,
                input_schema=ComponentQualificationAssessment.SCHEMA,
                output_schema=ResponseQualificationFacetAssessment.SCHEMA,
                rule_id=f"{prefix}-rule.facet.{facet_id.replace('.', '-')}",
                rule_semantics=(
                    'Project the owned operand to the declared measurement-through-law-qualification rung with the frozen method-equivalent prerequisite graph.'
                ),
            )
        )
    return ComponentQualificationProfile(
        profile_id=f"profile.{prefix}.method-equivalent",
        profile_version="1.0.0",
        applicable_method_keys=(method_key,),
        applicable_representation_kinds=(representation,),
        applicable_payload_schemas=(payload_schema,),
        applicable_extension_schemas=(payload_schema,),
        proof_owner_bindings=tuple(sorted(bindings, key=lambda value: value.binding_id)),
        facet_ids=facet_ids,
        allowed_not_applicable_output_ids=("uncertainty.transport",),
        status_precedence_ids=(
            "complete-mixed-partial",
            "complete-not-supported",
            "complete-supported",
            "complete-unevaluable-not-tested",
            "contract-invalid-no-assessment",
            "operational-blocked-no-assessment",
        ),
        conformance_case_ids=(
            f"{prefix}-negative",
            f"{prefix}-outside-support",
            f"{prefix}-positive",
            f"{prefix}-unevaluable",
            f"{prefix}-wrong-action-order",
        ),
    )


def _method_receipt(
    candidate: LawCandidateEvidence,
    evidence_kind_id: str,
) -> CandidateMethodEvidenceReceipt | None:
    matches = tuple(
        value for value in candidate.method_receipts if value.evidence_kind_id == evidence_kind_id
    )
    if len(matches) > 1:
        raise ValueError("method-equivalent profile received duplicate proof operands")
    return matches[0] if matches else None


def _binary_method_check(
    *,
    candidate: LawCandidateEvidence,
    prefix: str,
    kind: AdequacyCheckKind,
    evidence_kind_id: str,
    metric_id: str,
) -> AdequacyCheckResult:
    receipt = _method_receipt(candidate, evidence_kind_id)
    reasons: tuple[str, ...]
    if receipt is None:
        status = ObligationStatus.UNEVALUABLE
        metrics: tuple[NamedDecimal, ...] = ()
        evidence_ids: tuple[str, ...] = ()
        reasons = (f"{prefix}-{kind.value.lower().replace('_', '-')}-operand-absent",)
    elif receipt.availability is MethodEvidenceAvailability.UNAVAILABLE:
        status = ObligationStatus.UNEVALUABLE
        metrics = ()
        evidence_ids = ()
        reasons = receipt.method_reason_codes
    elif receipt.availability is MethodEvidenceAvailability.NOT_APPLICABLE:
        status = ObligationStatus.FAILED
        metrics = ()
        evidence_ids = ()
        reasons = (f"{prefix}-{kind.value.lower().replace('_', '-')}-required",)
    else:
        matches = tuple(value for value in receipt.metrics if value.value_id == metric_id)
        if len(matches) != 1 or len(receipt.metrics) != 1 or matches[0].unit != "1":
            raise ValueError("method-equivalent binary proof operand is malformed")
        metrics = receipt.metrics
        if matches[0].value >= Decimal(1):
            status = ObligationStatus.SATISFIED
            evidence_ids = receipt.evidence_link_ids
            reasons = ()
        else:
            status = ObligationStatus.FAILED
            evidence_ids = ()
            reasons = (f"{prefix}-{kind.value.lower().replace('_', '-')}-failed",)
    return AdequacyCheckResult(
        check_id=f"check.{prefix}.{kind.value.lower().replace('_', '-')}",
        kind=kind,
        status=status,
        decisive=True,
        metrics=metrics,
        reason_codes=reasons,
        evidence_link_ids=evidence_ids,
    )


def _derived_check(
    *,
    candidate: LawCandidateEvidence,
    prefix: str,
    kind: AdequacyCheckKind,
    passed: bool,
    metrics: tuple[NamedDecimal, ...] = (),
    failure_reason: str,
) -> AdequacyCheckResult:
    return AdequacyCheckResult(
        check_id=f"check.{prefix}.{kind.value.lower().replace('_', '-')}",
        kind=kind,
        status=ObligationStatus.SATISFIED if passed else ObligationStatus.FAILED,
        decisive=True,
        metrics=tuple(sorted(metrics, key=lambda value: value.value_id)),
        reason_codes=() if passed else (failure_reason,),
        evidence_link_ids=(
            tuple(value.link_id for value in candidate.evidence_links) if passed else ()
        ),
    )


def _confirmatory_local_support_check(
    *,
    candidate: LawCandidateEvidence,
    prefix: str,
    structural_passed: bool,
) -> AdequacyCheckResult:
    """Intersect exact prepared/local topology with the G1 and G1R operands."""

    metrics: list[NamedDecimal] = []
    evidence_ids: set[str] = set()
    reasons: set[str] = set()
    statuses: list[ObligationStatus] = []
    for suffix in (
        "g1-prefix-support",
        "g1r-region-support",
        "region-support-compatibility",
    ):
        evidence_kind_id = f"confirmatory-finite-action.{suffix}"
        metric_id = f"confirmatory-finite-action-{suffix}-criterion"
        receipt = _method_receipt(candidate, evidence_kind_id)
        if receipt is None:
            statuses.append(ObligationStatus.UNEVALUABLE)
            reasons.add(f"{prefix}-{suffix}-operand-absent")
        elif receipt.availability is MethodEvidenceAvailability.UNAVAILABLE:
            statuses.append(ObligationStatus.UNEVALUABLE)
            reasons.update(receipt.method_reason_codes)
        elif receipt.availability is MethodEvidenceAvailability.NOT_APPLICABLE:
            statuses.append(ObligationStatus.FAILED)
            reasons.add(f"{prefix}-{suffix}-required")
        else:
            matches = tuple(value for value in receipt.metrics if value.value_id == metric_id)
            if len(matches) != 1 or len(receipt.metrics) != 1 or matches[0].unit != "1":
                raise ValueError("confirmatory local-support proof operand is malformed")
            metrics.extend(receipt.metrics)
            if matches[0].value >= Decimal(1):
                statuses.append(ObligationStatus.SATISFIED)
                evidence_ids.update(receipt.evidence_link_ids)
            else:
                statuses.append(ObligationStatus.FAILED)
                reasons.add(f"{prefix}-{suffix}-failed")
    if not structural_passed:
        status = ObligationStatus.FAILED
        reasons.add("finite-action-local-support-roster-failed")
    elif any(value is ObligationStatus.FAILED for value in statuses):
        status = ObligationStatus.FAILED
    elif any(value is ObligationStatus.UNEVALUABLE for value in statuses):
        status = ObligationStatus.UNEVALUABLE
    else:
        status = ObligationStatus.SATISFIED
    return AdequacyCheckResult(
        check_id=f"check.{prefix}.support",
        kind=AdequacyCheckKind.SUPPORT,
        status=status,
        decisive=True,
        metrics=tuple(sorted(metrics, key=lambda value: value.value_id)),
        reason_codes=() if status is ObligationStatus.SATISFIED else tuple(sorted(reasons)),
        evidence_link_ids=(
            tuple(sorted(evidence_ids)) if status is ObligationStatus.SATISFIED else ()
        ),
    )


def _method_uncertainty(
    *,
    candidate: LawCandidateEvidence,
    prefix: str,
    evidence_prefix: str | None = None,
) -> UncertaintyDecomposition:
    source_prefix = evidence_prefix or prefix
    components: list[UncertaintyComponent] = []
    for uncertainty_class in UncertaintyClass:
        limitations: tuple[str, ...]
        receipt = _method_receipt(
            candidate,
            f"{source_prefix}.uncertainty.{uncertainty_class.value.lower()}",
        )
        if receipt is None:
            status = ObligationStatus.UNEVALUABLE
            bounds: tuple[NamedDecimal, ...] = ()
            evidence_ids: tuple[str, ...] = ()
            limitations = (f"{prefix}-{uncertainty_class.value.lower()}-uncertainty-absent",)
        elif receipt.availability is MethodEvidenceAvailability.OBSERVED:
            status = ObligationStatus.SATISFIED
            bounds = receipt.metrics
            evidence_ids = receipt.evidence_link_ids
            limitations = ()
        elif receipt.availability is MethodEvidenceAvailability.NOT_APPLICABLE:
            if uncertainty_class is not UncertaintyClass.TRANSPORT:
                raise ValueError("method profile permits only transport uncertainty as N/A")
            status = ObligationStatus.NOT_APPLICABLE
            bounds = ()
            evidence_ids = ()
            limitations = ()
        else:
            status = ObligationStatus.UNEVALUABLE
            bounds = ()
            evidence_ids = ()
            limitations = receipt.method_reason_codes
        components.append(
            UncertaintyComponent(
                component_id=(f"uncertainty.{prefix}.{uncertainty_class.value.lower()}"),
                uncertainty_class=uncertainty_class,
                status=status,
                bounds=bounds,
                limitation_codes=limitations,
                evidence_link_ids=evidence_ids,
            )
        )
    return UncertaintyDecomposition(
        decomposition_id=f"uncertainty-decomposition.{prefix}.{candidate.candidate_id}",
        components=tuple(sorted(components, key=lambda value: value.component_id)),
    )


_METHOD_PREREQUISITES = {
    AdequacyCheckKind.COORDINATE_ADEQUACY: (),
    AdequacyCheckKind.SUPPORT: (),
    AdequacyCheckKind.EVIDENCE_VISIBILITY: (),
    AdequacyCheckKind.WITHIN_CELL_RECURRENCE: (
        AdequacyCheckKind.COORDINATE_ADEQUACY,
        AdequacyCheckKind.SUPPORT,
    ),
    AdequacyCheckKind.ONE_FACTOR_EXCHANGE: (
        AdequacyCheckKind.COORDINATE_ADEQUACY,
        AdequacyCheckKind.SUPPORT,
    ),
    AdequacyCheckKind.DECISIVE_FALSIFIER: (
        AdequacyCheckKind.COORDINATE_ADEQUACY,
        AdequacyCheckKind.SUPPORT,
    ),
    AdequacyCheckKind.HELD_OUT_CALIBRATION: (
        AdequacyCheckKind.WITHIN_CELL_RECURRENCE,
        AdequacyCheckKind.ONE_FACTOR_EXCHANGE,
        AdequacyCheckKind.DECISIVE_FALSIFIER,
    ),
    AdequacyCheckKind.CLOSURE_MEMORY: (AdequacyCheckKind.WITHIN_CELL_RECURRENCE,),
    AdequacyCheckKind.STRUCTURAL_CONVERGENCE: (
        AdequacyCheckKind.COORDINATE_ADEQUACY,
        AdequacyCheckKind.SUPPORT,
    ),
    AdequacyCheckKind.COMPUTABILITY: (
        AdequacyCheckKind.COORDINATE_ADEQUACY,
        AdequacyCheckKind.SUPPORT,
    ),
}


def _method_trace(
    *,
    candidate: LawCandidateEvidence,
    prefix: str,
    profile: ComponentQualificationProfile,
    checks: tuple[AdequacyCheckResult, ...],
    uncertainty: UncertaintyDecomposition,
    joint_uncertainty: JointPredictiveUncertainty | None = None,
) -> ResponseQualificationTrace:
    by_kind = {value.kind: value for value in checks}
    resolved_statuses: dict[AdequacyCheckKind, ScientificStatus] = {}

    def resolved_status(kind: AdequacyCheckKind) -> ScientificStatus:
        existing = resolved_statuses.get(kind)
        if existing is not None:
            return existing
        status = _facet_status(by_kind[kind].status)
        if status is ScientificStatus.SUPPORTED and any(
            resolved_status(parent) is not ScientificStatus.SUPPORTED
            for parent in _METHOD_PREREQUISITES[kind]
        ):
            status = ScientificStatus.NOT_TESTED
        resolved_statuses[kind] = status
        return status

    facets: list[ResponseQualificationFacetAssessment] = []
    for kind in AdequacyCheckKind:
        check = by_kind[kind]
        facet_id = _method_facet_id(prefix, kind)
        status = resolved_status(kind)
        prerequisite_ids = tuple(
            sorted(_method_facet_id(prefix, value) for value in _METHOD_PREREQUISITES[kind])
        )
        facets.append(
            ResponseQualificationFacetAssessment(
                facet_id=facet_id,
                rung=_RESPONSE_METHOD_FACET_RUNG[kind],
                status=status,
                proof_owner=profile.proof_owner(
                    QualificationProofOutputKind.TRACE_FACET,
                    facet_id,
                ),
                prerequisite_facet_ids=prerequisite_ids,
                evidence_link_ids=(
                    check.evidence_link_ids if status is ScientificStatus.SUPPORTED else ()
                ),
                reason_codes=(
                    ()
                    if status is ScientificStatus.SUPPORTED
                    else check.reason_codes
                    or (f"{prefix}-{kind.value.lower().replace('_', '-')}-prerequisite",)
                ),
                maximum_supported_object_id=(
                    f"object.{facet_id}.{candidate.candidate_id}"
                    if status is ScientificStatus.SUPPORTED
                    else None
                ),
                required_for_public_rung=True,
            )
        )
    uncertainty_obligation, uncertainty_reasons, joint_evidence = _uncertainty_operands(
        uncertainty, profile, joint_uncertainty
    )
    uncertainty_status = _facet_status(uncertainty_obligation)
    uncertainty_prerequisites = tuple(
        sorted(
            (
                _method_facet_id(prefix, AdequacyCheckKind.HELD_OUT_CALIBRATION),
                _method_facet_id(prefix, AdequacyCheckKind.SUPPORT),
            )
        )
    )
    if uncertainty_status is ScientificStatus.SUPPORTED and any(
        next(value for value in facets if value.facet_id == prerequisite).status
        is not ScientificStatus.SUPPORTED
        for prerequisite in uncertainty_prerequisites
    ):
        uncertainty_status = ScientificStatus.NOT_TESTED
    facets.append(
        ResponseQualificationFacetAssessment(
            facet_id=f"{prefix}.uncertainty",
            rung=EvidenceRung.LOCAL_LAW,
            status=uncertainty_status,
            proof_owner=profile.proof_owner(
                QualificationProofOutputKind.TRACE_FACET,
                f"{prefix}.uncertainty",
            ),
            prerequisite_facet_ids=uncertainty_prerequisites,
            evidence_link_ids=(
                (
                    joint_evidence
                    if joint_uncertainty is not None
                    else tuple(value.link_id for value in candidate.evidence_links)
                )
                if uncertainty_status is ScientificStatus.SUPPORTED
                else ()
            ),
            reason_codes=(
                ()
                if uncertainty_status is ScientificStatus.SUPPORTED
                else uncertainty_reasons or (f"{prefix}-uncertainty-prerequisite",)
            ),
            maximum_supported_object_id=(
                f"object.{prefix}.uncertainty.{candidate.candidate_id}"
                if uncertainty_status is ScientificStatus.SUPPORTED
                else None
            ),
            required_for_public_rung=True,
        )
    )
    ordered = tuple(sorted(facets, key=lambda value: value.facet_id))
    rung_order = (
        EvidenceRung.MEASUREMENT,
        EvidenceRung.ORDER_RELATION,
        EvidenceRung.RESPONSE,
        EvidenceRung.LOCAL_LAW,
    )
    rung_rank = {value: index for index, value in enumerate(rung_order)}
    highest: EvidenceRung | None = None
    for rung in rung_order:
        applicable = tuple(
            value
            for value in ordered
            if value.required_for_public_rung and rung_rank[value.rung] <= rung_rank[rung]
        )
        if applicable and all(value.status is ScientificStatus.SUPPORTED for value in applicable):
            highest = rung
        else:
            break
    return ResponseQualificationTrace(
        trace_id=f"trace.{candidate.candidate_id}",
        candidate_id=candidate.candidate_id,
        facets=ordered,
        physical_independent_unit_ids=candidate.physical_independent_unit_ids,
        candidate_version_member_ids=candidate.axis_map.candidate_version_member_ids,
        denominator_member_ids=candidate.axis_map.denominator_member_ids,
        qualification_view_ids=candidate.axis_map.qualification_view_ids,
        information_cutoff_id=candidate.obligation_template.information_cutoff_id,
        outcome_access=candidate.outcome_access,
        highest_supported_rung=highest,
    )


def _property_qualifications(
    *,
    candidate: LawCandidateEvidence,
    owner: QualificationProofOwner,
    status: ScientificStatus,
    reason: str,
) -> tuple[PropertyQualification, ...]:
    owner_identity = ObjectIdentity.from_record(owner.owner_id, owner)
    result = []
    for property_id in sorted(
        {value for binding in candidate.axis_map.bindings for value in binding.claimed_property_ids}
    ):
        bindings = tuple(
            value
            for value in candidate.axis_map.bindings
            if property_id in value.claimed_property_ids
        )
        result.append(
            PropertyQualification(
                qualification_id=f"qualification.{candidate.candidate_id}.{property_id}",
                property_id=property_id,
                candidate_version_member_ids=tuple(
                    sorted({value.candidate_version_member_id for value in bindings})
                ),
                denominator_member_ids=tuple(
                    sorted({value.denominator_member_id for value in bindings})
                ),
                qualification_view_ids=tuple(
                    sorted({view for value in bindings for view in value.qualification_view_ids})
                ),
                status=status,
                proof_owner=owner_identity,
                evidence_link_ids=(
                    tuple(value.link_id for value in candidate.evidence_links)
                    if status is ScientificStatus.SUPPORTED
                    else ()
                ),
                reason_codes=() if status is ScientificStatus.SUPPORTED else (reason,),
            )
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class MethodEquivalentQualificationProfileEvaluator:
    "Proof owner for finite-action or controlled-IO method-equivalent measurement through local law facts."

    owner: QualificationProofOwner
    kind: MethodEquivalentProfileKind

    @property
    def profile(self) -> ComponentQualificationProfile:
        return build_method_equivalent_qualification_profile(self.owner, self.kind)

    def evaluate_candidate(
        self,
        system: SystemSpec,
        candidate: LawCandidateEvidence,
        payload: CanonicalRecord,
    ) -> ComponentQualificationAssessment:
        prefix, method_key, representation, _payload_schema = _METHOD_PROFILE_SPEC[self.kind]
        profile = self.profile
        if (
            candidate.method_key != method_key
            or candidate.representation_kind is not representation
        ):
            raise ValueError("method-equivalent profile does not own this candidate family")
        if self.kind in {
            MethodEquivalentProfileKind.FINITE_ACTION,
            MethodEquivalentProfileKind.CONFIRMATORY_FINITE_ACTION,
            MethodEquivalentProfileKind.CONFIRMATORY_FINITE_ACTION_LOCAL_SUPPORT,
        }:
            if not isinstance(payload, FiniteActionCompatibilitySetExtension):
                raise TypeError("finite-action profile received another payload type")
            checks, property_status, property_reason = self._finite_checks(
                system,
                candidate,
                payload,
                prefix=prefix,
            )
        else:
            if not isinstance(payload, ControlledIOVersionSet):
                raise TypeError("controlled-IO profile received another payload type")
            checks, property_status, property_reason = self._controlled_checks(
                system,
                candidate,
                payload,
            )
        evidence_prefix = (
            "confirmatory-finite-action"
            if self.kind is MethodEquivalentProfileKind.CONFIRMATORY_FINITE_ACTION_LOCAL_SUPPORT
            else prefix
        )
        uncertainty = _method_uncertainty(
            candidate=candidate,
            prefix=prefix,
            evidence_prefix=evidence_prefix,
        )
        terminal = _terminal_dispositions(
            candidate_id=candidate.candidate_id,
            checks=checks,
            uncertainty=uncertainty,
            profile=profile,
            fallback_evidence_ids=tuple(value.link_id for value in candidate.evidence_links),
        )
        trace = _method_trace(
            candidate=candidate,
            prefix=prefix,
            profile=profile,
            checks=checks,
            uncertainty=uncertainty,
        )
        properties = _property_qualifications(
            candidate=candidate,
            owner=self.owner,
            status=property_status,
            reason=property_reason,
        )
        scientific_statuses = tuple(value.status for value in properties) + tuple(
            value.status for value in trace.facets
        )
        if not candidate.visibility_ceiling.is_promotable:
            eligibility = CandidateQualificationEligibility.NONPROMOTABLE
        elif (
            all(value.passed for value in checks)
            and uncertainty.claim_ready
            and terminal.claim_ready
            and trace.highest_supported_rung is EvidenceRung.LOCAL_LAW
            and all(value.status is ScientificStatus.SUPPORTED for value in properties)
        ):
            eligibility = CandidateQualificationEligibility.PASSING
        elif (
            any(value.status is ObligationStatus.FAILED for value in checks)
            or any(value.status is ObligationStatus.FAILED for value in uncertainty.components)
            or any(value.status is ObligationStatus.FAILED for value in terminal.dispositions)
            or any(
                status in {
                    ScientificStatus.NOT_SUPPORTED,
                    ScientificStatus.MIXED,
                    ScientificStatus.PARTIAL,
                }
                for status in scientific_statuses
            )
        ):
            # Match the canonical assessment: a known failure takes precedence
            # over a simultaneous unevaluable operand. Both operands are retained.
            eligibility = CandidateQualificationEligibility.NONPASSING
        elif (
            any(value.status is ObligationStatus.UNEVALUABLE for value in checks)
            or any(value.status is ObligationStatus.UNEVALUABLE for value in uncertainty.components)
            or any(value.status is ObligationStatus.UNEVALUABLE for value in terminal.dispositions)
            or any(status is ScientificStatus.UNEVALUABLE for status in scientific_statuses)
        ):
            eligibility = CandidateQualificationEligibility.UNEVALUABLE
        else:
            eligibility = CandidateQualificationEligibility.NONPASSING
        return ComponentQualificationAssessment(
            assessment_id=f"profile-assessment.{candidate.candidate_id}",
            profile=ObjectIdentity.from_record(profile.profile_id, profile),
            candidate_evidence=ObjectIdentity.from_record(candidate.evidence_id, candidate),
            checks=checks,
            uncertainty=uncertainty,
            property_qualifications=properties,
            terminal_obligations=terminal,
            qualification_trace=trace,
            visibility_promotable=candidate.visibility_ceiling.is_promotable,
            eligibility=eligibility,
        )

    def _common_receipt_checks(
        self,
        candidate: LawCandidateEvidence,
        prefix: str,
    ) -> dict[AdequacyCheckKind, AdequacyCheckResult]:
        evidence_prefix = (
            "confirmatory-finite-action"
            if self.kind is MethodEquivalentProfileKind.CONFIRMATORY_FINITE_ACTION_LOCAL_SUPPORT
            else prefix
        )
        result = {}
        suffixes = (
            (
                AdequacyCheckKind.WITHIN_CELL_RECURRENCE,
                "recurrence",
            ),
            (
                AdequacyCheckKind.ONE_FACTOR_EXCHANGE,
                "declared-word-exchange"
                if self.kind
                in {
                    MethodEquivalentProfileKind.CONFIRMATORY_FINITE_ACTION,
                    MethodEquivalentProfileKind.CONFIRMATORY_FINITE_ACTION_LOCAL_SUPPORT,
                }
                else "one-factor-exchange",
            ),
            (AdequacyCheckKind.DECISIVE_FALSIFIER, "causal-falsifiers"),
        )
        for kind, suffix in suffixes:
            result[kind] = _binary_method_check(
                candidate=candidate,
                prefix=prefix,
                kind=kind,
                evidence_kind_id=f"{evidence_prefix}.{suffix}",
                metric_id=f"{evidence_prefix}-{suffix}-criterion",
            )
        return result

    def _finite_checks(
        self,
        system: SystemSpec,
        candidate: LawCandidateEvidence,
        payload: FiniteActionCompatibilitySetExtension,
        *,
        prefix: str = "finite-action",
    ) -> tuple[tuple[AdequacyCheckResult, ...], ScientificStatus, str]:
        quantities = {value.quantity_id: value for value in system.quantities}
        axes = {
            (value.denominator_member_id, value.candidate_version_member_id): value
            for value in candidate.axis_map.bindings
        }
        coordinate_ok = (
            payload.horizon
            == ObjectIdentity.from_record(
                system.relation.horizon.horizon_id,
                system.relation.horizon,
            )
            and payload.physical_independent_unit_ids == candidate.physical_independent_unit_ids
            and all(
                word.denominator_id == payload.prepared_denominator_id
                and word.horizon_id == system.relation.horizon.horizon_id
                and word.retained_history_id in payload.retained_history_ids
                and all(
                    occurrence.channel.controller_quantity_id in system.relation.action_quantity_ids
                    and occurrence.realized.native_unit
                    == quantities[occurrence.channel.controller_quantity_id].native_unit
                    and occurrence.realized.native_action_frame
                    == quantities[occurrence.channel.controller_quantity_id].coordinate_frame
                    and occurrence.realized.coordinate.clock_id
                    == quantities[occurrence.channel.controller_quantity_id].clock_id
                    for occurrence in word.occurrences
                )
                for word in payload.action_words
            )
        )
        support_ids = set(candidate.obligation_template.denominator_cell_ids)
        supported_entries = tuple(
            value
            for value in payload.entries
            if value.disposition is FiniteActionCellDisposition.SUPPORTED
        )
        local_support = (
            self.kind is MethodEquivalentProfileKind.CONFIRMATORY_FINITE_ACTION_LOCAL_SUPPORT
        )
        if local_support:
            local_ids = support_ids - {payload.prepared_denominator_id}
            local_unit_rosters = {
                local_id: {
                    value.physical_independent_unit_ids
                    for value in payload.entries
                    if value.support_cell_id == local_id
                }
                for local_id in local_ids
            }
            exact_local_units = (
                all(
                    len(rosters) == 1 and len(next(iter(rosters))) == 6
                    for rosters in local_unit_rosters.values()
                )
                if local_unit_rosters
                else False
            )
            roster_values = (
                tuple(next(iter(value)) for value in local_unit_rosters.values())
                if exact_local_units
                else ()
            )
            expected_coordinates = {
                (
                    axis.denominator_member_id,
                    axis.candidate_version_member_id,
                    local_id,
                    word.word_id,
                )
                for axis in candidate.axis_map.bindings
                for local_id in local_ids
                for word in payload.action_words
            }
            actual_coordinates = {
                (
                    value.denominator_member_id,
                    value.candidate_version_id,
                    value.support_cell_id,
                    value.action_word.object_id,
                )
                for value in payload.entries
            }
            support_ok = (
                len(support_ids) == 4
                and payload.prepared_denominator_id in support_ids
                and len(local_ids) == 3
                and not (support_ids & set(candidate.obligation_template.recurrence_cell_ids))
                and len(candidate.obligation_template.recurrence_cell_ids) == 3
                and actual_coordinates == expected_coordinates
                and all(value.support_cell_id in local_ids for value in payload.entries)
                and exact_local_units
                and not any(
                    set(left) & set(right)
                    for index, left in enumerate(roster_values)
                    for right in roster_values[index + 1 :]
                )
                and {unit_id for roster in roster_values for unit_id in roster}
                == set(payload.physical_independent_unit_ids)
                and all(
                    (value.denominator_member_id, value.candidate_version_id) in axes
                    and value.qualification_view_ids
                    == axes[
                        (value.denominator_member_id, value.candidate_version_id)
                    ].qualification_view_ids
                    for value in payload.entries
                )
            )
        else:
            support_ok = (
                bool(supported_entries)
                and all(
                    value.support_cell_id in support_ids
                    and (value.denominator_member_id, value.candidate_version_id) in axes
                    and value.qualification_view_ids
                    == axes[
                        (value.denominator_member_id, value.candidate_version_id)
                    ].qualification_view_ids
                    for value in supported_entries
                )
                and all(
                    any(value.support_cell_id == support_id for value in supported_entries)
                    for support_id in support_ids
                )
            )
        checks = self._common_receipt_checks(candidate, prefix)
        checks[AdequacyCheckKind.COORDINATE_ADEQUACY] = _derived_check(
            candidate=candidate,
            prefix=prefix,
            kind=AdequacyCheckKind.COORDINATE_ADEQUACY,
            passed=coordinate_ok,
            failure_reason="finite-action-coordinate-contract-failed",
        )
        checks[AdequacyCheckKind.SUPPORT] = (
            _confirmatory_local_support_check(
                candidate=candidate,
                prefix=prefix,
                structural_passed=support_ok,
            )
            if local_support
            else _derived_check(
                candidate=candidate,
                prefix=prefix,
                kind=AdequacyCheckKind.SUPPORT,
                passed=support_ok,
                failure_reason="finite-action-support-roster-failed",
            )
        )
        if self.kind in {
            MethodEquivalentProfileKind.CONFIRMATORY_FINITE_ACTION,
            MethodEquivalentProfileKind.CONFIRMATORY_FINITE_ACTION_LOCAL_SUPPORT,
        }:
            finite_operands = (
                (AdequacyCheckKind.CLOSURE_MEMORY, "history-prefix-closure"),
                (AdequacyCheckKind.HELD_OUT_CALIBRATION, "fresh-confirmatory-evaluation"),
                (
                    AdequacyCheckKind.STRUCTURAL_CONVERGENCE,
                    "member-structural-agreement",
                ),
            )
        else:
            finite_operands = (
                (AdequacyCheckKind.CLOSURE_MEMORY, "history-order-closure"),
                (AdequacyCheckKind.HELD_OUT_CALIBRATION, "heldout-prediction"),
                (
                    AdequacyCheckKind.STRUCTURAL_CONVERGENCE,
                    "member-refinement-stability",
                ),
            )
        evidence_prefix = "confirmatory-finite-action" if local_support else prefix
        for kind, suffix in finite_operands:
            checks[kind] = _binary_method_check(
                candidate=candidate,
                prefix=prefix,
                kind=kind,
                evidence_kind_id=f"{evidence_prefix}.{suffix}",
                metric_id=f"{evidence_prefix}-{suffix}-criterion",
            )
        checks[AdequacyCheckKind.COMPUTABILITY] = _derived_check(
            candidate=candidate,
            prefix=prefix,
            kind=AdequacyCheckKind.COMPUTABILITY,
            passed=True,
            failure_reason="finite-action-bounded-lookup-unavailable",
        )
        checks[AdequacyCheckKind.EVIDENCE_VISIBILITY] = _derived_check(
            candidate=candidate,
            prefix=prefix,
            kind=AdequacyCheckKind.EVIDENCE_VISIBILITY,
            passed=candidate.visibility_ceiling.is_promotable,
            failure_reason="finite-action-evidence-not-promotable",
        )
        if any(
            value.disposition is FiniteActionCellDisposition.UNEVALUABLE
            and value.support_cell_id in support_ids
            for value in payload.entries
        ):
            property_status = ScientificStatus.UNEVALUABLE
            property_reason = "finite-action-qualified-cell-unevaluable"
        elif any(
            value.disposition is not FiniteActionCellDisposition.SUPPORTED
            and value.support_cell_id in support_ids
            for value in payload.entries
        ):
            property_status = ScientificStatus.NOT_SUPPORTED
            property_reason = "finite-action-qualified-cell-not-supported"
        else:
            property_status = ScientificStatus.SUPPORTED
            property_reason = ""
        return (
            tuple(sorted(checks.values(), key=lambda value: value.check_id)),
            property_status,
            property_reason,
        )

    def _controlled_checks(
        self,
        system: SystemSpec,
        candidate: LawCandidateEvidence,
        payload: ControlledIOVersionSet,
    ) -> tuple[tuple[AdequacyCheckResult, ...], ScientificStatus, str]:
        prefix = "controlled-io"
        axes = {
            (value.denominator_member_id, value.candidate_version_member_id): value
            for value in candidate.axis_map.bindings
        }
        coordinate_ok = (
            payload.physical_independent_unit_ids == candidate.physical_independent_unit_ids
        ) and all(
            (member.denominator_member_id, member.candidate_version_id) in axes
            and member.qualification_view_ids
            == axes[
                (member.denominator_member_id, member.candidate_version_id)
            ].qualification_view_ids
            and member.input_basis.coordinate_ids == system.relation.action_quantity_ids
            and member.receiver_basis.coordinate_ids == system.relation.receiver_quantity_ids
            and member.horizon_id == system.relation.horizon.horizon_id
            for member in payload.members
        )
        support_ids = set(candidate.obligation_template.denominator_cell_ids)
        support_ok = all(set(member.support_cell_ids) == support_ids for member in payload.members)
        checks = self._common_receipt_checks(candidate, prefix)
        checks[AdequacyCheckKind.COORDINATE_ADEQUACY] = _derived_check(
            candidate=candidate,
            prefix=prefix,
            kind=AdequacyCheckKind.COORDINATE_ADEQUACY,
            passed=coordinate_ok,
            failure_reason="controlled-io-coordinate-contract-failed",
        )
        checks[AdequacyCheckKind.SUPPORT] = _derived_check(
            candidate=candidate,
            prefix=prefix,
            kind=AdequacyCheckKind.SUPPORT,
            passed=support_ok,
            failure_reason="controlled-io-support-roster-failed",
        )
        heldout_metrics = tuple(
            member.diagnostics.held_out_prediction_error for member in payload.members
        )
        heldout_ok = all(
            value.diagnostics.held_out_prediction_error.value
            <= value.qualification_config.maximum_held_out_prediction_error
            for value in payload.members
        )
        checks[AdequacyCheckKind.HELD_OUT_CALIBRATION] = _derived_check(
            candidate=candidate,
            prefix=prefix,
            kind=AdequacyCheckKind.HELD_OUT_CALIBRATION,
            passed=heldout_ok,
            metrics=heldout_metrics,
            failure_reason="controlled-io-heldout-prediction-failed",
        )
        closure_metrics = tuple(member.diagnostics.residual_norm for member in payload.members)
        closure_ok = all(
            value.diagnostics.residual_norm.value
            <= value.qualification_config.maximum_residual_norm
            for value in payload.members
        )
        checks[AdequacyCheckKind.CLOSURE_MEMORY] = _derived_check(
            candidate=candidate,
            prefix=prefix,
            kind=AdequacyCheckKind.CLOSURE_MEMORY,
            passed=closure_ok,
            metrics=closure_metrics,
            failure_reason="controlled-io-history-closure-failed",
        )
        structural_metrics = tuple(
            sorted(
                (
                    *(member.diagnostics.maximum_spectral_radius for member in payload.members),
                    *(member.diagnostics.maximum_condition_number for member in payload.members),
                ),
                key=lambda value: value.value_id,
            )
        )
        structural_ok = all(
            member.diagnostics.maximum_spectral_radius.value
            <= member.qualification_config.maximum_spectral_radius
            and member.diagnostics.maximum_condition_number.value
            <= member.qualification_config.maximum_condition_number
            and member.diagnostics.observability_rank
            >= member.qualification_config.minimum_observability_rank
            for member in payload.members
        )
        checks[AdequacyCheckKind.STRUCTURAL_CONVERGENCE] = _derived_check(
            candidate=candidate,
            prefix=prefix,
            kind=AdequacyCheckKind.STRUCTURAL_CONVERGENCE,
            passed=structural_ok,
            metrics=structural_metrics,
            failure_reason="controlled-io-member-stability-failed",
        )
        computable = all(
            ControlledIOEvaluator().evaluate(member, member.qualification_config).finite_horizon_map
            is not None
            for member in payload.members
            if member.disposition is ControlledIOProductDisposition.SUPPORTED
        )
        checks[AdequacyCheckKind.COMPUTABILITY] = _derived_check(
            candidate=candidate,
            prefix=prefix,
            kind=AdequacyCheckKind.COMPUTABILITY,
            passed=computable,
            failure_reason="controlled-io-finite-map-unavailable",
        )
        checks[AdequacyCheckKind.EVIDENCE_VISIBILITY] = _derived_check(
            candidate=candidate,
            prefix=prefix,
            kind=AdequacyCheckKind.EVIDENCE_VISIBILITY,
            passed=candidate.visibility_ceiling.is_promotable,
            failure_reason="controlled-io-evidence-not-promotable",
        )
        if any(
            value.disposition is ControlledIOProductDisposition.UNEVALUABLE
            for value in payload.members
        ):
            property_status = ScientificStatus.UNEVALUABLE
            property_reason = "controlled-io-member-unevaluable"
        elif (
            any(
                value.disposition is not ControlledIOProductDisposition.SUPPORTED
                for value in payload.members
            )
            or not structural_ok
        ):
            property_status = ScientificStatus.NOT_SUPPORTED
            property_reason = "controlled-io-member-not-supported"
        else:
            property_status = ScientificStatus.SUPPORTED
            property_reason = ""
        return (
            tuple(sorted(checks.values(), key=lambda value: value.check_id)),
            property_status,
            property_reason,
        )


def assess_joint_predictive_candidate(
    *,
    candidate: LawCandidateEvidence,
    profile: JointQualificationProfile,
    owner: QualificationProofOwner,
    prefix: str,
    checks: tuple[AdequacyCheckResult, ...],
    component_diagnostics: UncertaintyDecomposition,
    joint_uncertainty: JointPredictiveUncertainty,
) -> JointQualificationAssessment:
    """Shared reduction for a registered joint profile, without latent promotion."""
    terminal = _terminal_dispositions(
        candidate_id=candidate.candidate_id,
        checks=checks,
        uncertainty=component_diagnostics,
        profile=profile,
        fallback_evidence_ids=tuple(v.link_id for v in candidate.evidence_links),
        joint_uncertainty=joint_uncertainty,
    )
    trace = _method_trace(
        candidate=candidate,
        prefix=prefix,
        profile=profile,
        checks=checks,
        uncertainty=component_diagnostics,
        joint_uncertainty=joint_uncertainty,
    )
    failed = any(v.status is ObligationStatus.FAILED for v in terminal.dispositions)
    status = (
        ScientificStatus.SUPPORTED
        if terminal.claim_ready and trace.highest_supported_rung is EvidenceRung.LOCAL_LAW
        else ScientificStatus.NOT_SUPPORTED
        if failed
        else ScientificStatus.UNEVALUABLE
    )
    properties = _property_qualifications(
        candidate=candidate,
        owner=owner,
        status=status,
        reason=""
        if status is ScientificStatus.SUPPORTED
        else "JOINT_PREDICTIVE_PROFILE_NOT_QUALIFIED",
    )
    eligibility = (
        CandidateQualificationEligibility.NONPROMOTABLE
        if not candidate.visibility_ceiling.is_promotable
        else CandidateQualificationEligibility.PASSING
        if status is ScientificStatus.SUPPORTED
        else CandidateQualificationEligibility.NONPASSING
        if failed
        else CandidateQualificationEligibility.UNEVALUABLE
    )
    return JointQualificationAssessment(
        assessment_id=f"profile-assessment.{candidate.candidate_id}",
        profile=ObjectIdentity.from_record(profile.profile_id, profile),
        candidate_evidence=ObjectIdentity.from_record(candidate.evidence_id, candidate),
        checks=checks,
        uncertainty=component_diagnostics,
        property_qualifications=properties,
        terminal_obligations=terminal,
        qualification_trace=trace,
        visibility_promotable=candidate.visibility_ceiling.is_promotable,
        eligibility=eligibility,
        joint_uncertainty=joint_uncertainty,
    )


@dataclass(frozen=True, slots=True)
class ResponseQualificationProfileEvaluator:
    "Sole parametric response-method proof owner; fitters never author generic check truth."

    owner: QualificationProofOwner
    adequacy: AdequacyServices = AdequacyServices()
    numerical: NumericalQualifier = NumericalQualifier()
    uncertainty: UncertaintyService = UncertaintyService()

    @property
    def profile(self) -> ComponentQualificationProfile:
        return build_response_method_qualification_profile(self.owner)

    def evaluate(
        self,
        *,
        system: SystemSpec,
        dataset: IdentificationDataset,
        config: LawIdentificationConfig,
        fits: tuple[CandidateFit, ...],
        reference_fit: CandidateFit,
        candidate: LawCandidateEvidence,
        evidence_links: tuple[EvidenceLink, ...],
    ) -> tuple[ComponentQualificationAssessment, StructuralConvergenceResult]:
        profile = self.profile
        if config.method_key not in profile.applicable_method_keys:
            raise ValueError("Parametric response-method qualification profile does not own this method")
        if config.representation_kind not in profile.applicable_representation_kinds:
            raise ValueError("Parametric response-method qualification profile does not own this representation")
        if candidate.payload_publication.artifact.payload_schema not in (
            profile.applicable_payload_schemas
        ):
            raise ValueError("Parametric response-method qualification profile does not own this evaluator payload")
        if candidate.candidate_extension is not None:
            raise ValueError("Parametric response-method profile cannot accept a candidate extension")
        convergence = self.numerical.qualify(
            system,
            dataset,
            config,
            fits,
            evidence_links=evidence_links,
        )
        uncertainty = self.uncertainty.decompose(
            config,
            reference_fit,
            convergence,
            evidence_link_id=evidence_links[0].link_id,
        )
        checks = self.adequacy.evaluate(
            system,
            dataset,
            config,
            reference_fit,
            convergence,
            evidence_link_id=evidence_links[0].link_id,
        )
        terminal = _terminal_dispositions(
            candidate_id=candidate.candidate_id,
            checks=checks,
            uncertainty=uncertainty,
            profile=profile,
            fallback_evidence_ids=tuple(value.link_id for value in evidence_links),
        )
        trace = _trace(
            candidate=candidate,
            checks=checks,
            uncertainty=uncertainty,
            profile=profile,
            physical_unit_ids=dataset.physical_unit_instance_ids,
            information_cutoff_id=dataset.information_cutoff_id,
        )
        property_status = convergence.status
        property_reasons = convergence.reason_codes
        property_qualification = PropertyQualification(
            qualification_id=f"qualification.{candidate.candidate_id}.response-method-structure",
            property_id="response-method-structural-qualification",
            candidate_version_member_ids=candidate.axis_map.candidate_version_member_ids,
            denominator_member_ids=candidate.axis_map.denominator_member_ids,
            qualification_view_ids=candidate.axis_map.qualification_view_ids,
            status=property_status,
            proof_owner=ObjectIdentity.from_record(self.owner.owner_id, self.owner),
            evidence_link_ids=(
                tuple(value.link_id for value in evidence_links)
                if property_status is ScientificStatus.SUPPORTED
                else ()
            ),
            reason_codes=property_reasons,
        )
        if not candidate.visibility_ceiling.is_promotable:
            eligibility = CandidateQualificationEligibility.NONPROMOTABLE
        elif terminal.claim_ready and trace.highest_supported_rung is EvidenceRung.LOCAL_LAW:
            eligibility = CandidateQualificationEligibility.PASSING
        elif any(value.status is ObligationStatus.UNEVALUABLE for value in terminal.dispositions):
            eligibility = CandidateQualificationEligibility.UNEVALUABLE
        else:
            eligibility = CandidateQualificationEligibility.NONPASSING
        return (
            ComponentQualificationAssessment(
                assessment_id=f"profile-assessment.{candidate.candidate_id}",
                profile=ObjectIdentity.from_record(profile.profile_id, profile),
                candidate_evidence=ObjectIdentity.from_record(candidate.evidence_id, candidate),
                checks=checks,
                uncertainty=uncertainty,
                property_qualifications=(property_qualification,),
                terminal_obligations=terminal,
                qualification_trace=trace,
                visibility_promotable=candidate.visibility_ceiling.is_promotable,
                eligibility=eligibility,
            ),
            convergence,
        )
