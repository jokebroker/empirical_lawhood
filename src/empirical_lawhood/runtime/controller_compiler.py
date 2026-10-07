"""Fourteen-pass compiler for the sole current ControllerProgramme route."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar, NoReturn, Protocol, cast, overload

from empirical_lawhood.kernel.action_contracts import ActionWordSupportStatus
from empirical_lawhood.kernel.admission import AdmissionGateKind, GateStatus
from empirical_lawhood.kernel.causal_contracts import CausalCompositionDisposition
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import EvidenceCeiling, EvidenceRung, OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.receiver_geometry_control import (
    AmbiguityActionCertificate,
    AmbiguityCertificateDisposition,
    DecisionEquivalenceClass,
    ReceiverGeometryControlBinding,
)
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    require_unique_ids,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.planning.controller_study import AtlasActionCandidate, AdmissionActionCandidate, ActHoldActionCandidate, NativeCandidateKind, CandidateGateReceipt, CandidateReachabilityReceipt, AtlasControllerStudy, AdmissionControllerStudy, DeliveryControllerStudy, LeastMagnitudeControllerStudy, ImplementationBinding, ImplementationRole, AtlasMeasuredHoldFibre, AdmissionMeasuredHoldFibre, ControllerActionBinding, decode_atlas_controller_study, decode_admission_controller_study, decode_delivery_controller_study, decode_least_magnitude_controller_study
from empirical_lawhood.planning.evidence_geometry import AtlasAdmissionSpec, ReceiptAdmissionSpec, AtlasReachabilitySpec, ControlledMapReachabilitySpec, AdmissionCoordinateGateReceipt, LawMemberEvaluationBinding, ReceiptAdmissionAdmissionCandidateCell, ReceiptAdmissionPlannedCoordinate, ReceiptAdmissionUtilityStatus, ReachabilityCellDisposition, ControlledMapReachabilityComparison, ControlledMapReachabilityReceipt, UtilityEvaluationReceipt
from empirical_lawhood.planning.finite_admission import FiniteCertificateAdmissionSpec, FiniteCertificateReachabilitySpec, FiniteCertificateReachabilityComparison, derive_finite_certificate_admission_comparison, derive_finite_certificate_reachability_comparison
from empirical_lawhood.planning.finite_response_geometry import FiniteCertificateStatus, FiniteReachabilityMethodReceipt
from empirical_lawhood.planning.geometry import AdmissionComparison, ReachabilityComparison


MAX_COMPILED_CONTROLLER_BYTES = 16 * 1024 * 1024
MAX_STREAMING_COMPILED_CONTROLLER_BYTES = 64 * 1024 * 1024


class AtlasAdmissionDeriverPort(Protocol):
    implementation_binding: ImplementationBinding

    def derive(self, spec: AtlasAdmissionSpec) -> AdmissionComparison: ...


class AtlasReachabilityDeriverPort(Protocol):
    implementation_binding: ImplementationBinding

    def derive(self, spec: AtlasReachabilitySpec) -> ReachabilityComparison: ...


class ReceiptAdmissionDeriverPort(Protocol):
    implementation_binding: ImplementationBinding

    def derive(self, spec: ReceiptAdmissionSpec) -> AdmissionComparison: ...


class ReceiptReachabilityDeriverPort(Protocol):
    implementation_binding: ImplementationBinding

    def derive(self, spec: ControlledMapReachabilitySpec) -> ControlledMapReachabilityComparison: ...


class FiniteCertificateAdmissionDeriverPort(Protocol):
    implementation_binding: ImplementationBinding

    def derive(self, spec: FiniteCertificateAdmissionSpec) -> AdmissionComparison: ...


class FiniteCertificateReachabilityDeriverPort(Protocol):
    implementation_binding: ImplementationBinding

    def derive(self, spec: FiniteCertificateReachabilitySpec) -> FiniteCertificateReachabilityComparison: ...


class CompilerPass(StrEnum):
    SCHEMA_AND_IDENTITY = "SCHEMA_AND_IDENTITY"
    WORLD_AND_LAW_BINDING = "WORLD_AND_LAW_BINDING"
    EVIDENCE_LINEAGE = "EVIDENCE_LINEAGE"
    ACTION_OCCURRENCE_AND_CLOCK_TRANSPORT = "ACTION_OCCURRENCE_AND_CLOCK_TRANSPORT"
    ALGEBRA_AND_CAUSAL_SUPPORT_BINDING = "ALGEBRA_AND_CAUSAL_SUPPORT_BINDING"
    RECEIVER_INFORMATION_BINDING = "RECEIVER_INFORMATION_BINDING"
    ACTION_RECEIVER_BINDING = "ACTION_RECEIVER_BINDING"
    PROFILE_QUALIFICATION = "PROFILE_QUALIFICATION"
    EVIDENCE_DERIVED_GEOMETRY = "EVIDENCE_DERIVED_GEOMETRY"
    AMBIGUITY_CONTROL_QUOTIENT = "AMBIGUITY_CONTROL_QUOTIENT"
    CONSTRUCTION = "CONSTRUCTION"
    RUNTIME_COMPLETENESS = "RUNTIME_COMPLETENESS"
    PROSPECTIVE_EVALUATION_BINDING = "PROSPECTIVE_EVALUATION_BINDING"
    LOWERING_AND_REPLAY = "LOWERING_AND_REPLAY"


class CompiledDisposition(StrEnum):
    CONTROLLER = "CONTROLLER"
    HOLD_ONLY = "HOLD_ONLY"
    NONATTEMPT = "NONATTEMPT"


class CandidateEligibility(StrEnum):
    ELIGIBLE = "ELIGIBLE"
    INELIGIBLE = "INELIGIBLE"
    UNEVALUABLE = "UNEVALUABLE"


class HoldFibreDisposition(StrEnum):
    QUALIFIED = "QUALIFIED"
    UNSUPPORTED = "UNSUPPORTED"
    RECEIVER_INADMISSIBLE = "RECEIVER_INADMISSIBLE"
    UNVIABLE = "UNVIABLE"
    UNEVALUABLE = "UNEVALUABLE"


class ProspectiveBindingDisposition(StrEnum):
    BOUND = "BOUND"
    NOT_DECLARED = "NOT_DECLARED"


@dataclass(frozen=True, slots=True)
class CompilerPassProduct(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/compiler-pass-product'

    pass_id: str
    pass_name: CompilerPass
    input_identities: tuple[ObjectIdentity, ...]
    output_identities: tuple[ObjectIdentity, ...]
    highest_valid_rung: EvidenceRung
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_stable_id(self.pass_id, field_name="pass_id")
        require_unique_ids(
            self.input_identities, attribute="object_id", field_name="input_identities"
        )
        require_unique_ids(
            self.output_identities, attribute="object_id", field_name="output_identities"
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")


@dataclass(frozen=True, slots=True)
class CompilerFailure(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/compiler-failure'

    failure_id: str
    pass_name: CompilerPass
    affected_identity: str
    highest_valid_rung: EvidenceRung
    reason_codes: tuple[str, ...]
    retained_obstruction_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.failure_id, field_name="failure_id")
        validate_stable_id(self.affected_identity, field_name="affected_identity")
        require_sorted_unique_strings(
            self.reason_codes, field_name="reason_codes", allow_empty=False
        )
        require_sorted_unique_strings(
            self.retained_obstruction_codes, field_name="retained_obstruction_codes"
        )


class ControllerCompilationError(ValueError):
    def __init__(self, failure: CompilerFailure) -> None:
        self.failure = failure
        super().__init__(",".join(failure.reason_codes))


@dataclass(frozen=True, slots=True)
class CandidateMemberUtility(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/candidate-member-utility'

    model_member_id: str
    utility: NamedDecimal

    def __post_init__(self) -> None:
        validate_stable_id(self.model_member_id, field_name="model_member_id")


@dataclass(frozen=True, slots=True)
class AtlasCandidateAudit(CanonicalRecord):
    """Receipt-derived candidate disposition retained without an online selector."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/atlas-candidate-audit'

    audit_id: str
    candidate_id: str
    decision_cell_id: str
    action_binding_id: str
    member_utilities: tuple[CandidateMemberUtility, ...]
    robust_utility: NamedDecimal
    nominal_utility: NamedDecimal
    eligibility: CandidateEligibility
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("audit_id", self.audit_id),
            ("candidate_id", self.candidate_id),
            ("decision_cell_id", self.decision_cell_id),
            ("action_binding_id", self.action_binding_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_ids(
            self.member_utilities,
            attribute="model_member_id",
            field_name="member_utilities",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.eligibility is CandidateEligibility.ELIGIBLE:
            if self.reason_codes:
                raise ValueError("eligible candidate cannot carry refusal reasons")
        elif not self.reason_codes:
            raise ValueError("noneligible candidate requires reasons")
        values = tuple(value.utility.value for value in self.member_utilities)
        units = {value.utility.unit for value in self.member_utilities}
        if not values or len(units) != 1:
            raise ValueError("candidate audit requires a complete common-unit utility roster")
        if self.robust_utility.value != min(values) or self.robust_utility.unit not in units:
            raise ValueError("robust utility is not the memberwise minimum")


@dataclass(frozen=True, slots=True)
class CompiledCellAction(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/compiled-cell-action'

    selection_id: str
    decision_cell_id: str
    admission_cell_id: str
    candidate_id: str
    action_binding: ControllerActionBinding
    robust_utility: NamedDecimal
    nominal_utility: NamedDecimal
    priority_rank: int

    def __post_init__(self) -> None:
        for name, value in (
            ("selection_id", self.selection_id),
            ("decision_cell_id", self.decision_cell_id),
            ("admission_cell_id", self.admission_cell_id),
            ("candidate_id", self.candidate_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.priority_rank < 0:
            raise ValueError("compiled priority rank must be nonnegative")
        if self.robust_utility.unit != self.nominal_utility.unit:
            raise ValueError("compiled selection utilities use different units")


@dataclass(frozen=True, slots=True)
class CompiledAtlasHoldFibre(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/compiled-atlas-hold-fibre'

    qualification_id: str
    source: AtlasMeasuredHoldFibre
    action_binding: ControllerActionBinding
    disposition: HoldFibreDisposition
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.qualification_id, field_name="qualification_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.action_binding.action_binding_id != self.source.action_binding_id:
            raise ValueError("compiled HOLD uses another native action binding")
        if self.disposition is HoldFibreDisposition.QUALIFIED:
            if self.reason_codes:
                raise ValueError("qualified HOLD cannot carry refusal reasons")
        elif not self.reason_codes:
            raise ValueError("unqualified HOLD requires reasons")


@dataclass(frozen=True, slots=True)
class ProspectiveCompilationBinding(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/prospective-compilation-binding'

    binding_id: str
    disposition: ProspectiveBindingDisposition
    evaluation_plan: ObjectIdentity | None
    evaluator: ObjectIdentity | None

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")
        if self.disposition is ProspectiveBindingDisposition.BOUND:
            if self.evaluation_plan is None or self.evaluator is None:
                raise ValueError("bound controller use plan requires plan and evaluator identities")
        elif self.evaluation_plan is not None or self.evaluator is not None:
            raise ValueError("CONTROLLER_USE_NOT_DECLARED cannot bind plan or evaluator")


@dataclass(frozen=True, slots=True)
class CompiledAtlasControllerStudy(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/compiled-atlas-controller-study'

    compiled_study_id: str
    study: AtlasControllerStudy
    specification_fingerprint: str
    pass_products: tuple[CompilerPassProduct, ...]
    admission: AdmissionComparison
    reachability: ReachabilityComparison
    action_table: tuple[CompiledCellAction, ...]
    candidate_audit: tuple[AtlasCandidateAudit, ...]
    implementations: tuple[ImplementationBinding, ...]
    ambiguity_certificate: AmbiguityActionCertificate | None
    receiver_geometry_binding: ReceiverGeometryControlBinding | None
    hold_fibre: CompiledAtlasHoldFibre | None
    prospective_evaluation_binding: ProspectiveCompilationBinding
    disposition: CompiledDisposition
    reason_codes: tuple[str, ...]
    geometry_diagnostic_codes: tuple[str, ...]
    compilation_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    new_scientific_result: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.compiled_study_id, field_name='compiled_study_id')
        validate_sha256(self.specification_fingerprint, field_name="specification_fingerprint")
        require_unique_ids(self.pass_products, attribute="pass_id", field_name="pass_products")
        if tuple(value.pass_name for value in self.pass_products) != tuple(CompilerPass):
            raise ValueError("compiled pass products differ from the fourteen-pass pipeline")
        require_sorted_unique_ids(
            self.action_table, attribute="decision_cell_id", field_name="action_table"
        )
        require_sorted_unique_ids(
            self.candidate_audit, attribute="candidate_id", field_name="candidate_audit"
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        require_sorted_unique_strings(
            self.geometry_diagnostic_codes, field_name="geometry_diagnostic_codes"
        )
        if self.specification_fingerprint != self.study.fingerprint():
            raise ValueError("compiled programme fingerprint differs from source bytes")
        if self.implementations != self.study.implementations:
            raise ValueError("compiled implementations differ from programme")
        action_candidates = {
            value.candidate_id: value
            for value in self.study.synthesis.candidate_chart.candidates
        }
        action_bindings = {
            value.action_binding_id: value for value in self.study.action_bindings
        }
        audits = {value.candidate_id: value for value in self.candidate_audit}
        for selection in self.action_table:
            candidate = action_candidates.get(selection.candidate_id)
            audit = audits.get(selection.candidate_id)
            if (
                candidate is None
                or audit is None
                or audit.eligibility is not CandidateEligibility.ELIGIBLE
                or selection.action_binding != action_bindings.get(candidate.action_binding_id)
                or selection.decision_cell_id != candidate.decision_cell_id
                or selection.admission_cell_id != candidate.admission_cell_id
                or selection.robust_utility != audit.robust_utility
                or selection.nominal_utility != audit.nominal_utility
                or selection.priority_rank != candidate.priority_rank
            ):
                raise ValueError("compiled action table is not derived from candidate audit")
        if self.study.receiver_quotient is None:
            if self.ambiguity_certificate is not None or self.receiver_geometry_binding is not None:
                raise ValueError("ordinary compilation fabricates receiver ambiguity")
        elif self.ambiguity_certificate is None or self.receiver_geometry_binding is None:
            raise ValueError("receiver-quotient compilation lacks exact certificate/binding")
        if self.hold_fibre is not None:
            if self.study.measured_hold_fibre is None or (
                self.hold_fibre.source != self.study.measured_hold_fibre
            ):
                raise ValueError("compiled HOLD differs from programme")
        expected_disposition = (
            CompiledDisposition.CONTROLLER
            if self.action_table
            else (
                CompiledDisposition.HOLD_ONLY
                if self.hold_fibre is not None
                and self.hold_fibre.disposition is HoldFibreDisposition.QUALIFIED
                else CompiledDisposition.NONATTEMPT
            )
        )
        if self.disposition is not expected_disposition:
            raise ValueError("compiled disposition differs from frozen action/HOLD tables")
        if self.disposition is CompiledDisposition.CONTROLLER:
            if self.reason_codes:
                raise ValueError("active compiled controller cannot carry terminal reasons")
        elif not self.reason_codes:
            raise ValueError("nonactive compiled programme requires reasons")
        if self.compilation_ceiling is not EvidenceCeiling.ADMISSION:
            raise ValueError("compiled controller is capped at admission")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("compiled controller must remain outcome-blind")
        if self.new_scientific_result:
            raise ValueError("compilation is construction, not validation")

    def implementation(self, role: ImplementationRole) -> ImplementationBinding:
        return next(value for value in self.implementations if value.role is role)

    def action_for_cell(self, decision_cell_id: str) -> CompiledCellAction | None:
        return next(
            (value for value in self.action_table if value.decision_cell_id == decision_cell_id),
            None,
        )


@dataclass(frozen=True, slots=True)
class AdmissionCandidateConstituentAudit(CanonicalRecord):
    "Compiler audit of one exact member/version/action/support admission coordinate."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/admission-candidate-constituent-audit'

    audit_id: str
    planned_coordinate: ReceiptAdmissionPlannedCoordinate
    receipt_ids: tuple[str, ...]
    law_evaluation_bindings: tuple[ObjectIdentity, ...]
    utility: NamedDecimal | None
    eligibility: CandidateEligibility
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.audit_id, field_name="audit_id")
        require_sorted_unique_strings(
            self.receipt_ids,
            field_name="receipt_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(
            self.law_evaluation_bindings,
            attribute="object_id",
            field_name="law_evaluation_bindings",
        )
        if not self.law_evaluation_bindings:
            raise ValueError("compiled constituent requires exact law-evaluation bindings")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.eligibility is CandidateEligibility.ELIGIBLE:
            if self.utility is None or self.reason_codes:
                raise ValueError("eligible constituent requires utility and no refusal reasons")
        elif not self.reason_codes:
            raise ValueError("noneligible constituent requires reasons")


@dataclass(frozen=True, slots=True)
class AdmissionCandidateAudit(CanonicalRecord):
    """Complete robust audit; no member/version constituent may be omitted."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/admission-candidate-audit'

    audit_id: str
    candidate_id: str
    decision_cell_id: str
    admission_candidate_cell: ObjectIdentity
    action_binding_id: str
    nominal_denominator_member_id: str
    constituents: tuple[AdmissionCandidateConstituentAudit, ...]
    robust_utility: NamedDecimal | None
    nominal_utility: NamedDecimal | None
    eligibility: CandidateEligibility
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("audit_id", self.audit_id),
            ("candidate_id", self.candidate_id),
            ("decision_cell_id", self.decision_cell_id),
            ("action_binding_id", self.action_binding_id),
            ("nominal_denominator_member_id", self.nominal_denominator_member_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.admission_candidate_cell.object_schema != ReceiptAdmissionAdmissionCandidateCell.SCHEMA:
            raise ValueError("compiled admission candidate audit requires an exact admission candidate cell")
        require_sorted_unique_ids(
            self.constituents,
            attribute="audit_id",
            field_name="constituents",
        )
        if not self.constituents:
            raise ValueError("compiled candidate audit requires every constituent")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        utilities = tuple(value.utility for value in self.constituents if value.utility is not None)
        complete_utilities = len(utilities) == len(self.constituents)
        if complete_utilities:
            units = {value.unit for value in utilities}
            if len(units) != 1:
                raise ValueError("compiled candidate constituents mix utility units")
            expected_robust = min(value.value for value in utilities)
            nominal = tuple(
                value.utility
                for value in self.constituents
                if value.planned_coordinate.denominator_member_id
                == self.nominal_denominator_member_id
                and value.utility is not None
            )
            if not nominal:
                raise ValueError("compiled candidate lacks nominal-member constituents")
            expected_nominal = min(value.value for value in nominal)
            if (
                self.robust_utility is None
                or self.nominal_utility is None
                or self.robust_utility.value != expected_robust
                or self.nominal_utility.value != expected_nominal
                or self.robust_utility.unit not in units
                or self.nominal_utility.unit not in units
            ):
                raise ValueError("compiled robust/nominal utility is not constituent-derived")
        elif self.robust_utility is not None or self.nominal_utility is not None:
            raise ValueError("incomplete constituent utilities cannot produce aggregate utility")
        if self.eligibility is CandidateEligibility.ELIGIBLE:
            if (
                not complete_utilities
                or any(
                    value.eligibility is not CandidateEligibility.ELIGIBLE
                    for value in self.constituents
                )
                or self.reason_codes
            ):
                raise ValueError("eligible candidate is not the complete constituent intersection")
        elif not self.reason_codes:
            raise ValueError("noneligible candidate requires reasons")


@dataclass(frozen=True, slots=True)
class CompiledAdmissionHoldFibre(CanonicalRecord):
    "Qualified HOLD retaining every exact admission/Gate-5 constituent receipt."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/compiled-admission-hold-fibre'

    qualification_id: str
    source: AdmissionMeasuredHoldFibre
    action_binding: ControllerActionBinding
    planned_coordinate_ids: tuple[str, ...]
    gate_receipt_ids: tuple[str, ...]
    reachability_receipt_ids: tuple[str, ...]
    law_evaluation_bindings: tuple[ObjectIdentity, ...]
    disposition: HoldFibreDisposition
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.qualification_id, field_name="qualification_id")
        for name, values in (
            ("planned_coordinate_ids", self.planned_coordinate_ids),
            ("gate_receipt_ids", self.gate_receipt_ids),
            ("reachability_receipt_ids", self.reachability_receipt_ids),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)
        require_sorted_unique_ids(
            self.law_evaluation_bindings,
            attribute="object_id",
            field_name="law_evaluation_bindings",
        )
        if not self.law_evaluation_bindings:
            raise ValueError("compiled HOLD requires exact law-evaluation bindings")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.disposition is HoldFibreDisposition.QUALIFIED:
            if self.reason_codes:
                raise ValueError("qualified HOLD cannot carry refusal reasons")
        elif not self.reason_codes:
            raise ValueError("unqualified HOLD requires reasons")


@dataclass(frozen=True, slots=True)
class CompiledAdmissionControllerStudy(CanonicalRecord):
    "Compiled admission controller study retaining the complete member-local admission topology."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/compiled-admission-controller-study'
    VERSION: ClassVar[str] = '1.0.0'

    compiled_study_id: str
    study: AdmissionControllerStudy
    specification_fingerprint: str
    pass_products: tuple[CompilerPassProduct, ...]
    admission: AdmissionComparison
    reachability: ControlledMapReachabilityComparison
    action_table: tuple[CompiledCellAction, ...]
    candidate_audit: tuple[AdmissionCandidateAudit, ...]
    implementations: tuple[ImplementationBinding, ...]
    hold_fibre: CompiledAdmissionHoldFibre | None
    prospective_evaluation_binding: ProspectiveCompilationBinding
    disposition: CompiledDisposition
    reason_codes: tuple[str, ...]
    geometry_diagnostic_codes: tuple[str, ...]
    compilation_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    new_scientific_result: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.compiled_study_id, field_name='compiled_study_id')
        validate_sha256(self.specification_fingerprint, field_name="specification_fingerprint")
        require_unique_ids(self.pass_products, attribute="pass_id", field_name="pass_products")
        if tuple(value.pass_name for value in self.pass_products) != tuple(CompilerPass):
            raise ValueError("compiled admission controller products differ from the fourteen-pass pipeline")
        require_sorted_unique_ids(
            self.action_table,
            attribute="decision_cell_id",
            field_name="action_table",
        )
        require_sorted_unique_ids(
            self.candidate_audit,
            attribute="candidate_id",
            field_name="candidate_audit",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        require_sorted_unique_strings(
            self.geometry_diagnostic_codes,
            field_name="geometry_diagnostic_codes",
        )
        if self.specification_fingerprint != self.study.fingerprint():
            raise ValueError("compiled admission controller fingerprint differs from source bytes")
        if self.implementations != self.study.implementations:
            raise ValueError("compiled admission controller implementations differ from programme")
        candidates = {
            value.candidate_id: value
            for value in self.study.synthesis.candidate_chart.candidates
        }
        actions = {value.action_binding_id: value for value in self.study.action_bindings}
        audits = {value.candidate_id: value for value in self.candidate_audit}
        if set(audits) != set(candidates):
            raise ValueError("compiled admission controller candidate audit omits or adds candidates")
        for audit in self.candidate_audit:
            candidate = candidates[audit.candidate_id]
            cell = self.study.admission.corpus.plan.action_fibre(
                next(
                    value.action_fibre
                    for value in self.study.admission.candidate_cells
                    if ObjectIdentity.from_record(value.candidate_cell_id, value)
                    == candidate.admission_candidate_cell
                )
            )
            expected_coordinates = tuple(
                sorted(
                    value.coordinate_id
                    for value in self.study.admission.corpus.plan.coordinates
                    if value.coordinate_id
                    in next(
                        item.planned_coordinate_ids
                        for item in self.study.admission.candidate_cells
                        if ObjectIdentity.from_record(item.candidate_cell_id, item)
                        == candidate.admission_candidate_cell
                    )
                )
            )
            if (
                audit.decision_cell_id != candidate.decision_cell_id
                or audit.admission_candidate_cell != candidate.admission_candidate_cell
                or audit.action_binding_id != cell.action_binding_id
                or tuple(value.planned_coordinate.coordinate_id for value in audit.constituents)
                != expected_coordinates
            ):
                raise ValueError("compiled admission controller audit rewrites or omits admission constituents")
        for selection in self.action_table:
            selected_candidate = candidates.get(selection.candidate_id)
            selected_audit = audits.get(selection.candidate_id)
            if (
                selected_candidate is None
                or selected_audit is None
                or selected_audit.eligibility is not CandidateEligibility.ELIGIBLE
                or selection.action_binding != actions.get(selected_audit.action_binding_id)
                or selection.decision_cell_id != selected_candidate.decision_cell_id
                or selection.admission_cell_id != selected_candidate.admission_candidate_cell.object_id
                or selection.robust_utility != selected_audit.robust_utility
                or selection.nominal_utility != selected_audit.nominal_utility
                or selection.priority_rank != selected_candidate.priority_rank
            ):
                raise ValueError("compiled admission controller action table is not candidate-audit-derived")
        if self.hold_fibre is not None:
            if (
                self.study.measured_hold_fibre is None
                or self.hold_fibre.source != self.study.measured_hold_fibre
            ):
                raise ValueError("compiled admission controller HOLD differs from programme")
            hold_cell = next(
                value
                for value in self.study.admission.candidate_cells
                if ObjectIdentity.from_record(value.candidate_cell_id, value)
                == self.hold_fibre.source.admission_candidate_cell
            )
            hold_action = self.study.admission.corpus.plan.action_fibre(hold_cell.action_fibre)
            hold_coordinates = set(hold_cell.planned_coordinate_ids)
            hold_gates = tuple(
                value
                for value in self.study.admission.corpus.gate_receipts
                if value.planned_coordinate.coordinate_id in hold_coordinates
            )
            hold_reaches = tuple(
                value
                for value in self.study.admission.corpus.reachability_receipts
                if value.planned_coordinate.coordinate_id in hold_coordinates
            )
            bindings: dict[str, LawMemberEvaluationBinding] = {}
            hold_receipts: tuple[
                AdmissionCoordinateGateReceipt | ControlledMapReachabilityReceipt,
                ...,
            ] = (*hold_gates, *hold_reaches)
            for receipt in hold_receipts:
                binding = receipt.law_evaluation_binding
                prior = bindings.setdefault(binding.binding_id, binding)
                if prior != binding:
                    raise ValueError("compiled admission controller HOLD reuses a binding ID inconsistently")
            expected_bindings = tuple(
                ObjectIdentity.from_record(binding_id, bindings[binding_id])
                for binding_id in sorted(bindings)
            )
            if (
                self.hold_fibre.action_binding != actions.get(hold_action.action_binding_id)
                or self.hold_fibre.planned_coordinate_ids != tuple(sorted(hold_coordinates))
                or self.hold_fibre.gate_receipt_ids
                != tuple(sorted(value.receipt_id for value in hold_gates))
                or self.hold_fibre.reachability_receipt_ids
                != tuple(sorted(value.receipt_id for value in hold_reaches))
                or self.hold_fibre.law_evaluation_bindings != expected_bindings
            ):
                raise ValueError("compiled admission controller HOLD is not exact raw admission-derived evidence")
        expected_disposition = (
            CompiledDisposition.CONTROLLER
            if self.action_table
            else (
                CompiledDisposition.HOLD_ONLY
                if self.hold_fibre is not None
                and self.hold_fibre.disposition is HoldFibreDisposition.QUALIFIED
                else CompiledDisposition.NONATTEMPT
            )
        )
        if self.disposition is not expected_disposition:
            raise ValueError("compiled admission controller disposition differs from action/HOLD tables")
        if self.disposition is CompiledDisposition.CONTROLLER:
            if self.reason_codes:
                raise ValueError("active compiled admission controller controller cannot carry terminal reasons")
        elif not self.reason_codes:
            raise ValueError("nonactive compiled admission controller programme requires reasons")
        if self.compilation_ceiling is not EvidenceCeiling.ADMISSION:
            raise ValueError("compiled admission controller controller is capped at admission")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("compiled admission controller controller must remain outcome-blind")
        if self.new_scientific_result:
            raise ValueError("compiled admission controller construction is not validation")

    def implementation(self, role: ImplementationRole) -> ImplementationBinding:
        return next(value for value in self.implementations if value.role is role)

    def action_for_cell(self, decision_cell_id: str) -> CompiledCellAction | None:
        return next(
            (value for value in self.action_table if value.decision_cell_id == decision_cell_id),
            None,
        )


@dataclass(frozen=True, slots=True)
class ActHoldCompiledCellAction(CompiledCellAction):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/act-hold-compiled-cell-action'

    kind: NativeCandidateKind

    def __post_init__(self) -> None:
        CompiledCellAction.__post_init__(self)
        if not isinstance(self.kind, NativeCandidateKind):
            raise ValueError("compiled finite choice must retain ACT/HOLD kind")


def _finite_compiled_disposition(
    study: DeliveryControllerStudy,
    actions: tuple[ActHoldCompiledCellAction, ...],
    hold: CompiledAdmissionHoldFibre | None,
) -> CompiledDisposition:
    if any(action.kind is NativeCandidateKind.ACT for action in actions):
        return CompiledDisposition.CONTROLLER
    if actions or (
        study.allow_fallback_hold
        and hold is not None
        and hold.disposition is HoldFibreDisposition.QUALIFIED
    ):
        return CompiledDisposition.HOLD_ONLY
    return CompiledDisposition.NONATTEMPT


def _finite_compiled_reasons(
    disposition: CompiledDisposition,
    actions: tuple[ActHoldCompiledCellAction, ...],
    audits: tuple[AdmissionCandidateAudit, ...],
    hold: CompiledAdmissionHoldFibre | None,
) -> tuple[str, ...]:
    if disposition is CompiledDisposition.CONTROLLER:
        return ()
    if disposition is CompiledDisposition.HOLD_ONLY:
        return (
            ("FINITE_HOLD_SELECTED",)
            if actions
            else ("FINITE_FALLBACK_HOLD_ONLY", "NO_ELIGIBLE_FINITE_CANDIDATE")
        )
    return tuple(
        sorted(
            {
                "NO_ELIGIBLE_FINITE_CANDIDATE",
                *(reason for audit in audits for reason in audit.reason_codes),
                *(("MEASURED_HOLD_FIBRE_MISSING",) if hold is None else hold.reason_codes),
            }
        )
    )


@dataclass(frozen=True, slots=True)
class CompiledDeliveryControllerStudy(CanonicalRecord):
    """Replayable finite decisions from the sole fourteen-pass compiler."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/compiled-delivery-controller-study'
    VERSION: ClassVar[str] = '1.0.0'
    PROGRAMME_TYPE: ClassVar[type[DeliveryControllerStudy]] = DeliveryControllerStudy

    compiled_study_id: str
    study: DeliveryControllerStudy
    specification_fingerprint: str
    pass_products: tuple[CompilerPassProduct, ...]
    admission: AdmissionComparison
    reachability: FiniteCertificateReachabilityComparison
    action_table: tuple[ActHoldCompiledCellAction, ...]
    candidate_audit: tuple[AdmissionCandidateAudit, ...]
    implementations: tuple[ImplementationBinding, ...]
    hold_fibre: CompiledAdmissionHoldFibre | None
    prospective_evaluation_binding: ProspectiveCompilationBinding
    disposition: CompiledDisposition
    reason_codes: tuple[str, ...]
    geometry_diagnostic_codes: tuple[str, ...]
    compilation_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    new_scientific_result: bool

    def __post_init__(self) -> None:
        if type(self.study) is not self.PROGRAMME_TYPE:
            raise ValueError("compiled controller requires its exact programme schema")
        validate_stable_id(self.compiled_study_id, field_name='compiled_study_id')
        validate_sha256(self.specification_fingerprint, field_name="specification_fingerprint")
        require_unique_ids(self.pass_products, attribute="pass_id", field_name="pass_products")
        if tuple(p.pass_name for p in self.pass_products) != tuple(CompilerPass):
            raise ValueError("compiled finite programme must retain the fourteen-pass pipeline")
        require_sorted_unique_ids(
            self.action_table, attribute="decision_cell_id", field_name="action_table"
        )
        require_sorted_unique_ids(
            self.candidate_audit, attribute="candidate_id", field_name="candidate_audit"
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        require_sorted_unique_strings(
            self.geometry_diagnostic_codes, field_name="geometry_diagnostic_codes"
        )
        if (
            self.specification_fingerprint != self.study.fingerprint()
            or self.implementations != self.study.implementations
        ):
            raise ValueError(
                "compiled finite programme rewrites its exact source or implementations"
            )
        if self.admission != derive_finite_certificate_admission_comparison(
            self.study.admission
        ) or self.reachability != derive_finite_certificate_reachability_comparison(self.study.reachability):
            raise ValueError("compiled finite geometry fails exact raw-corpus replay")
        expected_audits = tuple(
            ControllerCompiler._audit_delivery_candidate(
                self.study,
                candidate,
                admission=self.admission,
                reachability=self.reachability,
            )
            for candidate in self.study.synthesis.candidate_chart.candidates
        )
        if self.candidate_audit != expected_audits:
            raise ValueError("compiled finite candidate audit omits or rewrites a constituent")
        if self.action_table != ControllerCompiler._select_delivery_action_table(
            self.study, expected_audits
        ):
            raise ValueError(
                "compiled finite choice differs from the frozen robust utility/tie rule"
            )
        expected_hold = (
            None
            if self.study.measured_hold_fibre is None
            else ControllerCompiler._qualify_delivery_hold_fibre(
                self.study,
                self.study.measured_hold_fibre,
                admission=self.admission,
                reachability=self.reachability,
            )
        )
        if self.hold_fibre != expected_hold:
            raise ValueError("compiled finite HOLD differs from its full measured-fibre evidence")
        if self.disposition is not _finite_compiled_disposition(
            self.study, self.action_table, self.hold_fibre
        ):
            raise ValueError(
                "compiled finite disposition differs from its ACT/HOLD/fallback tables"
            )
        if self.reason_codes != _finite_compiled_reasons(
            self.disposition, self.action_table, self.candidate_audit, self.hold_fibre
        ):
            raise ValueError(
                "compiled finite terminal reasons are not derived from retained decisions"
            )
        if (
            self.compilation_ceiling is not EvidenceCeiling.ADMISSION
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
        ):
            raise ValueError("compiled finite controller must retain its outcome-blind admission ceiling")
        if self.new_scientific_result:
            raise ValueError("finite compilation is not prospective scientific validation")
        if self.prospective_evaluation_binding.evaluation_plan != self.study.prospective_evaluation:
            raise ValueError("compiled finite programme changes its controller use evaluation identity")

    def implementation(self, role: ImplementationRole) -> ImplementationBinding:
        return next(value for value in self.implementations if value.role is role)

    def action_for_cell(self, decision_cell_id: str) -> ActHoldCompiledCellAction | None:
        return next((a for a in self.action_table if a.decision_cell_id == decision_cell_id), None)


@dataclass(frozen=True, slots=True)
class CompiledLeastMagnitudeControllerStudy(CompiledDeliveryControllerStudy):
    "Same finite pipeline with the explicit native-magnitude ordering."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/compiled-least-magnitude-controller-study'
    VERSION: ClassVar[str] = '1.0.0'
    PROGRAMME_TYPE: ClassVar[type[DeliveryControllerStudy]] = LeastMagnitudeControllerStudy

    study: LeastMagnitudeControllerStudy


# Both carry the same finite commitment/delivery/prospective controller evaluation operands.
# The least-magnitude programme adds only native ordering; decoder choice follows identity.
FINITE_COMPILED_COMMITMENT_SCHEMAS = (
    CompiledDeliveryControllerStudy.SCHEMA,
    CompiledLeastMagnitudeControllerStudy.SCHEMA,
)


class ControllerCompiler:
    """The sole public controller compiler; no compact or compatibility entry point."""

    def __init__(
        self,
        *,
        admission_deriver: AtlasAdmissionDeriverPort | ReceiptAdmissionDeriverPort | FiniteCertificateAdmissionDeriverPort,
        reachability_deriver: AtlasReachabilityDeriverPort
        | ReceiptReachabilityDeriverPort
        | FiniteCertificateReachabilityDeriverPort,
    ) -> None:
        self._admission_deriver = admission_deriver
        self._reachability_deriver = reachability_deriver

    @overload
    def compile(self, study: AtlasControllerStudy) -> CompiledAtlasControllerStudy: ...

    @overload
    def compile(self, study: AdmissionControllerStudy) -> CompiledAdmissionControllerStudy: ...

    @overload
    def compile(self, study: DeliveryControllerStudy) -> CompiledDeliveryControllerStudy: ...

    def compile(
        self,
        study: AtlasControllerStudy | AdmissionControllerStudy | DeliveryControllerStudy,
    ) -> (
        CompiledAtlasControllerStudy | CompiledAdmissionControllerStudy | CompiledDeliveryControllerStudy
    ):
        if isinstance(study, (AdmissionControllerStudy, DeliveryControllerStudy)):
            return self._compile_admission_controller_study(study)
        return self._compile_atlas_controller_study(study)

    def _compile_atlas_controller_study(self, study: AtlasControllerStudy) -> CompiledAtlasControllerStudy:
        admission_deriver = cast(AtlasAdmissionDeriverPort, self._admission_deriver)
        reachability_deriver = cast(AtlasReachabilityDeriverPort, self._reachability_deriver)
        products: list[CompilerPassProduct] = []

        def fail(
            *,
            pass_name: CompilerPass,
            reason_codes: tuple[str, ...],
            highest_valid_rung: EvidenceRung,
            obstruction_codes: tuple[str, ...] = (),
        ) -> NoReturn:
            self._fail(
                study,
                pass_name=pass_name,
                reason_codes=reason_codes,
                highest_valid_rung=highest_valid_rung,
                obstruction_codes=obstruction_codes,
            )

        programme_identity = ObjectIdentity.from_record(study.study_id, study)
        implementations = {value.role: value for value in study.implementations}
        for role, observed in (
            (
                ImplementationRole.ADMISSION_DERIVER,
                admission_deriver.implementation_binding,
            ),
            (
                ImplementationRole.REACHABILITY_DERIVER,
                reachability_deriver.implementation_binding,
            ),
        ):
            if implementations[role] != observed:
                fail(
                    pass_name=CompilerPass.SCHEMA_AND_IDENTITY,
                    reason_codes=("IMPLEMENTATION_BINDING_MISMATCH",),
                    highest_valid_rung=EvidenceRung.MEASUREMENT,
                )
        products.append(
            self._product(
                study,
                CompilerPass.SCHEMA_AND_IDENTITY,
                (programme_identity,),
                tuple(
                    ObjectIdentity.from_record(value.binding_id, value)
                    for value in study.implementations
                ),
                EvidenceRung.MEASUREMENT,
            )
        )

        law_identity = ObjectIdentity.from_record(study.law.law_id, study.law)
        system_identity = ObjectIdentity.from_record(study.system.system_id, study.system)
        products.append(
            self._product(
                study,
                CompilerPass.WORLD_AND_LAW_BINDING,
                (system_identity, law_identity),
                (law_identity,),
                EvidenceRung.RESPONSE,
            )
        )

        atlas_identity = ObjectIdentity.from_record(
            study.admission.atlas.atlas_id, study.admission.atlas
        )
        model_set_identity = ObjectIdentity.from_record(
            study.admission.model_set.model_set_id, study.admission.model_set
        )
        chart_identity = ObjectIdentity.from_record(
            study.synthesis.candidate_chart.chart_id,
            study.synthesis.candidate_chart,
        )
        try:
            self._validate_candidate_receipt_corpora(study)
        except ValueError as error:
            fail(
                pass_name=CompilerPass.EVIDENCE_LINEAGE,
                reason_codes=("CANDIDATE_EVIDENCE_LINEAGE_FAILED", self._exception_reason(error)),
                highest_valid_rung=EvidenceRung.LOCAL_LAW,
            )
        products.append(
            self._product(
                study,
                CompilerPass.EVIDENCE_LINEAGE,
                (law_identity, atlas_identity, model_set_identity),
                (atlas_identity, model_set_identity, chart_identity),
                EvidenceRung.LOCAL_LAW,
            )
        )

        action_identities = tuple(
            ObjectIdentity.from_record(value.action_binding_id, value)
            for value in study.action_bindings
        )
        action_words = tuple(
            ObjectIdentity.from_record(value.action_word.word_id, value.action_word)
            for value in study.action_bindings
        )
        products.append(
            self._product(
                study,
                CompilerPass.ACTION_OCCURRENCE_AND_CLOCK_TRANSPORT,
                action_identities,
                action_words,
                EvidenceRung.LOCAL_LAW,
            )
        )

        blocked = tuple(
            value
            for value in study.action_bindings
            if value.causal_prefix.disposition is not CausalCompositionDisposition.DEFINED
        )
        if blocked:
            fail(
                pass_name=CompilerPass.ALGEBRA_AND_CAUSAL_SUPPORT_BINDING,
                reason_codes=("CAUSAL_ACTION_BINDING_BLOCKED",),
                highest_valid_rung=EvidenceRung.RESPONSE,
                obstruction_codes=tuple(
                    sorted(
                        {
                            obstruction.value
                            for binding in blocked
                            for obstruction in binding.causal_prefix.obstructions
                        }
                    )
                ),
            )
        causal_identities = tuple(
            ObjectIdentity.from_record(value.causal_prefix.assessment_id, value.causal_prefix)
            for value in study.action_bindings
        )
        products.append(
            self._product(
                study,
                CompilerPass.ALGEBRA_AND_CAUSAL_SUPPORT_BINDING,
                action_identities,
                causal_identities,
                EvidenceRung.LOCAL_LAW,
            )
        )

        quotient = study.receiver_quotient
        quotient_identity = (
            ObjectIdentity.from_record(quotient.plan_id, quotient)
            if quotient is not None
            else programme_identity
        )
        if quotient is not None:
            try:
                self._validate_receiver_information(study)
            except ValueError as error:
                fail(
                    pass_name=CompilerPass.RECEIVER_INFORMATION_BINDING,
                    reason_codes=(
                        "RECEIVER_INFORMATION_BINDING_FAILED",
                        self._exception_reason(error),
                    ),
                    highest_valid_rung=EvidenceRung.LOCAL_LAW,
                )
        products.append(
            self._product(
                study,
                CompilerPass.RECEIVER_INFORMATION_BINDING,
                (quotient_identity,),
                (quotient_identity,),
                EvidenceRung.LOCAL_LAW,
            )
        )
        products.append(
            self._product(
                study,
                CompilerPass.ACTION_RECEIVER_BINDING,
                (*action_identities, law_identity),
                action_identities,
                EvidenceRung.LOCAL_LAW,
            )
        )
        products.append(
            self._product(
                study,
                CompilerPass.PROFILE_QUALIFICATION,
                (system_identity, model_set_identity),
                (model_set_identity,),
                EvidenceRung.LOCAL_LAW,
            )
        )

        try:
            admission = admission_deriver.derive(study.admission)
            if admission != study.reachability.admission:
                raise ValueError("admission replay differs from reachability input")
            admission_identity = ObjectIdentity.from_record(admission.comparison_id, admission)
            if study.synthesis.admission_comparison != admission_identity:
                raise ValueError("synthesis admission identity differs")
            reachability = reachability_deriver.derive(study.reachability)
            reachability_identity = ObjectIdentity.from_record(
                reachability.comparison_id, reachability
            )
            if study.synthesis.reachability_comparison != reachability_identity:
                raise ValueError("synthesis reachability identity differs")
        except ValueError as error:
            fail(
                pass_name=CompilerPass.EVIDENCE_DERIVED_GEOMETRY,
                reason_codes=("EVIDENCE_GEOMETRY_DERIVATION_FAILED", self._exception_reason(error)),
                highest_valid_rung=EvidenceRung.LOCAL_LAW,
            )
        products.append(
            self._product(
                study,
                CompilerPass.EVIDENCE_DERIVED_GEOMETRY,
                (
                    ObjectIdentity.from_record(
                        study.admission.evaluation_id, study.admission
                    ),
                    ObjectIdentity.from_record(
                        study.reachability.evaluation_id, study.reachability
                    ),
                ),
                (admission_identity, reachability_identity),
                EvidenceRung.ADMISSION,
            )
        )

        ambiguity_certificate: AmbiguityActionCertificate | None = None
        receiver_binding: ReceiverGeometryControlBinding | None = None
        ambiguity_outputs: tuple[ObjectIdentity, ...] = (quotient_identity,)
        if quotient is not None:
            try:
                ambiguity_certificate, receiver_binding = self._compile_receiver_quotient(
                    study,
                    admission_identity=admission_identity,
                    reachability_identity=reachability_identity,
                )
            except ValueError as error:
                fail(
                    pass_name=CompilerPass.AMBIGUITY_CONTROL_QUOTIENT,
                    reason_codes=(
                        "AMBIGUITY_CONTROL_QUOTIENT_FAILED",
                        self._exception_reason(error),
                    ),
                    highest_valid_rung=EvidenceRung.ADMISSION,
                )
            ambiguity_outputs = (
                ObjectIdentity.from_record(
                    ambiguity_certificate.certificate_id, ambiguity_certificate
                ),
                ObjectIdentity.from_record(receiver_binding.binding_id, receiver_binding),
            )
        products.append(
            self._product(
                study,
                CompilerPass.AMBIGUITY_CONTROL_QUOTIENT,
                (quotient_identity, admission_identity, reachability_identity),
                ambiguity_outputs,
                EvidenceRung.ADMISSION,
            )
        )

        try:
            candidate_audit = tuple(
                sorted(
                    (
                        self._audit_atlas_candidate(
                            study,
                            value,
                            admission=admission,
                            reachability=reachability,
                        )
                        for value in study.synthesis.candidate_chart.candidates
                    ),
                    key=lambda value: value.candidate_id,
                )
            )
            action_table = self._select_atlas_action_table(study, candidate_audit)
        except ValueError as error:
            fail(
                pass_name=CompilerPass.CONSTRUCTION,
                reason_codes=("CONTROLLER_CONSTRUCTION_FAILED", self._exception_reason(error)),
                highest_valid_rung=EvidenceRung.ADMISSION,
            )
        audit_identities = tuple(
            ObjectIdentity.from_record(value.audit_id, value) for value in candidate_audit
        )
        action_table_identity = ObjectIdentity.from_record(
            f"lookup.{study.study_id}",
            _CompiledActionLookupTable(table_id=f"lookup.{study.study_id}", entries=action_table),
        )
        products.append(
            self._product(
                study,
                CompilerPass.CONSTRUCTION,
                (chart_identity, admission_identity, reachability_identity),
                (*audit_identities, action_table_identity),
                EvidenceRung.ADMISSION,
            )
        )

        compiled_hold: CompiledAtlasHoldFibre | None = None
        if study.measured_hold_fibre is not None:
            try:
                compiled_hold = self._qualify_atlas_hold_fibre(
                    study,
                    study.measured_hold_fibre,
                    admission=admission,
                    reachability=reachability,
                )
            except ValueError as error:
                fail(
                    pass_name=CompilerPass.RUNTIME_COMPLETENESS,
                    reason_codes=("HOLD_FIBRE_BINDING_FAILED", self._exception_reason(error)),
                    highest_valid_rung=EvidenceRung.ADMISSION,
                )
        runtime_bindings = tuple(
            ObjectIdentity.from_record(value.binding_id, value)
            for value in study.implementations
            if value.role
            in {
                ImplementationRole.OBSERVER,
                ImplementationRole.ONLINE_GATE_EVALUATOR,
                ImplementationRole.DELIVERY,
            }
        )
        products.append(
            self._product(
                study,
                CompilerPass.RUNTIME_COMPLETENESS,
                (action_table_identity,),
                (
                    *runtime_bindings,
                    *(
                        (ObjectIdentity.from_record(compiled_hold.qualification_id, compiled_hold),)
                        if compiled_hold is not None
                        else ()
                    ),
                ),
                EvidenceRung.ADMISSION,
            )
        )

        evaluation = study.prospective_evaluation
        if evaluation is None:
            prospective_evaluation_binding = ProspectiveCompilationBinding(
                binding_id=f"prospective-evaluation-binding.{study.study_id}",
                disposition=ProspectiveBindingDisposition.NOT_DECLARED,
                evaluation_plan=None,
                evaluator=None,
            )
        else:
            evaluator = implementations[ImplementationRole.OUTCOME_EVALUATOR]
            prospective_evaluation_binding = ProspectiveCompilationBinding(
                binding_id=f"prospective-evaluation-binding.{study.study_id}",
                disposition=ProspectiveBindingDisposition.BOUND,
                evaluation_plan=ObjectIdentity.from_record(
                    evaluation.evaluation_plan_id, evaluation
                ),
                evaluator=ObjectIdentity.from_record(evaluator.binding_id, evaluator),
            )
        prospective_evaluation_binding_identity = ObjectIdentity.from_record(prospective_evaluation_binding.binding_id, prospective_evaluation_binding)
        products.append(
            self._product(
                study,
                CompilerPass.PROSPECTIVE_EVALUATION_BINDING,
                (programme_identity,),
                (prospective_evaluation_binding_identity,),
                EvidenceRung.ADMISSION,
                reason_codes=(
                    ("CONTROLLER_USE_NOT_DECLARED",)
                    if prospective_evaluation_binding.disposition is ProspectiveBindingDisposition.NOT_DECLARED
                    else ()
                ),
            )
        )

        try:
            replayed = decode_atlas_controller_study(study.canonical_bytes())
        except (TypeError, ValueError) as error:
            fail(
                pass_name=CompilerPass.LOWERING_AND_REPLAY,
                reason_codes=("CANONICAL_REPLAY_FAILED", self._exception_reason(error)),
                highest_valid_rung=EvidenceRung.ADMISSION,
            )
        if replayed != study or replayed.canonical_bytes() != study.canonical_bytes():
            fail(
                pass_name=CompilerPass.LOWERING_AND_REPLAY,
                reason_codes=("CANONICAL_REPLAY_MISMATCH",),
                highest_valid_rung=EvidenceRung.ADMISSION,
            )
        products.append(
            self._product(
                study,
                CompilerPass.LOWERING_AND_REPLAY,
                (programme_identity,),
                (programme_identity,),
                EvidenceRung.ADMISSION,
            )
        )

        disposition = (
            CompiledDisposition.CONTROLLER
            if action_table
            else (
                CompiledDisposition.HOLD_ONLY
                if compiled_hold is not None
                and compiled_hold.disposition is HoldFibreDisposition.QUALIFIED
                else CompiledDisposition.NONATTEMPT
            )
        )
        if disposition is CompiledDisposition.CONTROLLER:
            reason_codes: tuple[str, ...] = ()
        elif disposition is CompiledDisposition.HOLD_ONLY:
            reason_codes = ("NO_ELIGIBLE_ACTIVE_ACTION",)
        else:
            reason_codes = tuple(
                sorted(
                    {
                        "NO_ELIGIBLE_ACTIVE_ACTION",
                        *(
                            ("MEASURED_HOLD_FIBRE_MISSING",)
                            if compiled_hold is None
                            else compiled_hold.reason_codes
                        ),
                    }
                )
            )
        geometry_diagnostics = tuple(
            sorted(
                {
                    *(
                        ("ADMISSION_MODEL_SET_DISAGREEMENT",)
                        if admission.disagreement_cell_ids
                        else ()
                    ),
                    *(
                        ("REACHABILITY_MODEL_SET_DISAGREEMENT",)
                        if reachability.disagreement_cell_ids
                        else ()
                    ),
                }
            )
        )
        return CompiledAtlasControllerStudy(
            compiled_study_id=f"compiled.{study.study_id}",
            study=study,
            specification_fingerprint=study.fingerprint(),
            pass_products=tuple(products),
            admission=admission,
            reachability=reachability,
            action_table=action_table,
            candidate_audit=candidate_audit,
            implementations=study.implementations,
            ambiguity_certificate=ambiguity_certificate,
            receiver_geometry_binding=receiver_binding,
            hold_fibre=compiled_hold,
            prospective_evaluation_binding=prospective_evaluation_binding,
            disposition=disposition,
            reason_codes=reason_codes,
            geometry_diagnostic_codes=geometry_diagnostics,
            compilation_ceiling=EvidenceCeiling.ADMISSION,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            new_scientific_result=False,
        )

    @classmethod
    def assess_finite_candidates(
        cls, study: DeliveryControllerStudy
    ) -> tuple[AdmissionCandidateAudit, ...]:
        """Assess independent local alternatives before compiling the selected use.

        This is the same receipt/utility/action audit used by compilation, not
        execution authority or a compiled controller. Each programme retains its
        own qualified model intersection; unrelated local laws need not be put
        into a fictitious shared intersection to compare their eligible actions.
        The chosen programme must still pass the complete compiler and runtime.
        """
        admission = derive_finite_certificate_admission_comparison(study.admission)
        reachability = derive_finite_certificate_reachability_comparison(study.reachability)
        if (
            admission != study.reachability.admission
            or study.synthesis.admission_comparison
            != ObjectIdentity.from_record(admission.comparison_id, admission)
            or study.synthesis.reachability_comparison
            != ObjectIdentity.from_record(reachability.comparison_id, reachability)
        ):
            raise ValueError("finite assessment changed its exact evidence geometry")
        return cls._finite_candidate_audits(study, admission, reachability)

    @classmethod
    def _finite_candidate_audits(
        cls,
        study: DeliveryControllerStudy,
        admission: AdmissionComparison,
        reachability: FiniteCertificateReachabilityComparison,
    ) -> tuple[AdmissionCandidateAudit, ...]:
        return tuple(
            cls._audit_delivery_candidate(
                study, candidate, admission=admission, reachability=reachability
            )
            for candidate in study.synthesis.candidate_chart.candidates
        )

    def _compile_admission_controller_study(
        self,
        study: AdmissionControllerStudy | DeliveryControllerStudy,
    ) -> CompiledAdmissionControllerStudy | CompiledDeliveryControllerStudy:
        "Run the same fourteen semantic passes over the additive topology."

        admission_deriver = self._admission_deriver
        reachability_deriver = self._reachability_deriver
        products: list[CompilerPassProduct] = []

        def fail(
            *,
            pass_name: CompilerPass,
            reason_codes: tuple[str, ...],
            highest_valid_rung: EvidenceRung,
            obstruction_codes: tuple[str, ...] = (),
        ) -> NoReturn:
            self._fail(
                study,
                pass_name=pass_name,
                reason_codes=reason_codes,
                highest_valid_rung=highest_valid_rung,
                obstruction_codes=obstruction_codes,
            )

        programme_identity = ObjectIdentity.from_record(study.study_id, study)
        implementations = {value.role: value for value in study.implementations}
        for role, observed in (
            (
                ImplementationRole.ADMISSION_DERIVER,
                admission_deriver.implementation_binding,
            ),
            (
                ImplementationRole.REACHABILITY_DERIVER,
                reachability_deriver.implementation_binding,
            ),
        ):
            if implementations[role] != observed:
                fail(
                    pass_name=CompilerPass.SCHEMA_AND_IDENTITY,
                    reason_codes=("IMPLEMENTATION_BINDING_MISMATCH",),
                    highest_valid_rung=EvidenceRung.MEASUREMENT,
                )
        products.append(
            self._product(
                study,
                CompilerPass.SCHEMA_AND_IDENTITY,
                (programme_identity,),
                tuple(
                    ObjectIdentity.from_record(value.binding_id, value)
                    for value in study.implementations
                ),
                EvidenceRung.MEASUREMENT,
            )
        )

        plan = study.admission.corpus.plan
        system_identity = ObjectIdentity.from_record(
            study.system.system_id,
            study.system,
        )
        law_identities = tuple(
            ObjectIdentity.from_record(value.law_id, value) for value in plan.atlas.laws
        )
        products.append(
            self._product(
                study,
                CompilerPass.WORLD_AND_LAW_BINDING,
                (system_identity, *law_identities),
                law_identities,
                EvidenceRung.RESPONSE,
            )
        )

        atlas_identity = ObjectIdentity.from_record(plan.atlas.atlas_id, plan.atlas)
        model_set_identity = ObjectIdentity.from_record(
            plan.model_set.model_set_id,
            plan.model_set,
        )
        corpus_identity = ObjectIdentity.from_record(
            study.admission.corpus.corpus_id,
            study.admission.corpus,
        )
        chart_identity = ObjectIdentity.from_record(
            study.synthesis.candidate_chart.chart_id,
            study.synthesis.candidate_chart,
        )
        products.append(
            self._product(
                study,
                CompilerPass.EVIDENCE_LINEAGE,
                (*law_identities, atlas_identity, model_set_identity, corpus_identity),
                (corpus_identity, chart_identity),
                EvidenceRung.LOCAL_LAW,
            )
        )

        action_identities = tuple(
            ObjectIdentity.from_record(value.action_binding_id, value)
            for value in study.action_bindings
        )
        action_words = tuple(
            ObjectIdentity.from_record(value.action_word.word_id, value.action_word)
            for value in study.action_bindings
        )
        products.append(
            self._product(
                study,
                CompilerPass.ACTION_OCCURRENCE_AND_CLOCK_TRANSPORT,
                action_identities,
                action_words,
                EvidenceRung.LOCAL_LAW,
            )
        )
        blocked = tuple(
            value
            for value in study.action_bindings
            if value.causal_prefix.disposition is not CausalCompositionDisposition.DEFINED
        )
        if blocked:
            fail(
                pass_name=CompilerPass.ALGEBRA_AND_CAUSAL_SUPPORT_BINDING,
                reason_codes=("CAUSAL_ACTION_BINDING_BLOCKED",),
                highest_valid_rung=EvidenceRung.RESPONSE,
                obstruction_codes=tuple(
                    sorted(
                        {
                            obstruction.value
                            for binding in blocked
                            for obstruction in binding.causal_prefix.obstructions
                        }
                    )
                ),
            )
        causal_identities = tuple(
            ObjectIdentity.from_record(value.causal_prefix.assessment_id, value.causal_prefix)
            for value in study.action_bindings
        )
        products.append(
            self._product(
                study,
                CompilerPass.ALGEBRA_AND_CAUSAL_SUPPORT_BINDING,
                action_identities,
                causal_identities,
                EvidenceRung.LOCAL_LAW,
            )
        )
        products.append(
            self._product(
                study,
                CompilerPass.RECEIVER_INFORMATION_BINDING,
                (programme_identity,),
                (programme_identity,),
                EvidenceRung.LOCAL_LAW,
            )
        )
        products.append(
            self._product(
                study,
                CompilerPass.ACTION_RECEIVER_BINDING,
                (*action_identities, *law_identities),
                action_identities,
                EvidenceRung.LOCAL_LAW,
            )
        )
        qualification_identities = tuple(
            value.qualification_result for value in plan.model_set.members
        )
        products.append(
            self._product(
                study,
                CompilerPass.PROFILE_QUALIFICATION,
                (model_set_identity, *qualification_identities),
                qualification_identities,
                EvidenceRung.LOCAL_LAW,
            )
        )

        try:
            if isinstance(study, DeliveryControllerStudy):
                admission = cast(FiniteCertificateAdmissionDeriverPort, admission_deriver).derive(
                    study.admission
                )
            else:
                admission = cast(ReceiptAdmissionDeriverPort, admission_deriver).derive(
                    study.admission
                )
            if admission != study.reachability.admission:
                raise ValueError("admission replay differs from reachability input")
            admission_identity = ObjectIdentity.from_record(
                admission.comparison_id,
                admission,
            )
            if study.synthesis.admission_comparison != admission_identity:
                raise ValueError("synthesis admission identity differs")
            reachability: ControlledMapReachabilityComparison | FiniteCertificateReachabilityComparison
            if isinstance(study, DeliveryControllerStudy):
                reachability = cast(FiniteCertificateReachabilityDeriverPort, reachability_deriver).derive(
                    study.reachability
                )
            else:
                reachability = cast(ReceiptReachabilityDeriverPort, reachability_deriver).derive(
                    study.reachability
                )
            reachability_identity = ObjectIdentity.from_record(
                reachability.comparison_id,
                reachability,
            )
            if study.synthesis.reachability_comparison != reachability_identity:
                raise ValueError("synthesis reachability identity differs")
        except ValueError as error:
            fail(
                pass_name=CompilerPass.EVIDENCE_DERIVED_GEOMETRY,
                reason_codes=("EVIDENCE_GEOMETRY_DERIVATION_FAILED", self._exception_reason(error)),
                highest_valid_rung=EvidenceRung.LOCAL_LAW,
            )
        products.append(
            self._product(
                study,
                CompilerPass.EVIDENCE_DERIVED_GEOMETRY,
                (
                    ObjectIdentity.from_record(
                        study.admission.evaluation_id,
                        study.admission,
                    ),
                    ObjectIdentity.from_record(
                        study.reachability.evaluation_id,
                        study.reachability,
                    ),
                ),
                (admission_identity, reachability_identity),
                EvidenceRung.ADMISSION,
            )
        )
        products.append(
            self._product(
                study,
                CompilerPass.AMBIGUITY_CONTROL_QUOTIENT,
                (admission_identity, reachability_identity),
                (admission_identity, reachability_identity),
                EvidenceRung.ADMISSION,
            )
        )

        try:
            action_table: tuple[CompiledCellAction, ...] | tuple[ActHoldCompiledCellAction, ...]
            if isinstance(study, DeliveryControllerStudy):
                finite_reachability = cast(FiniteCertificateReachabilityComparison, reachability)
                candidate_audit = self._finite_candidate_audits(
                    study, admission, finite_reachability
                )
                action_table = self._select_delivery_action_table(study, candidate_audit)
            else:
                candidate_audit = tuple(
                    self._audit_admission_candidate(
                        study,
                        candidate,
                        admission=admission,
                        reachability=cast(ControlledMapReachabilityComparison, reachability),
                    )
                    for candidate in study.synthesis.candidate_chart.candidates
                )
                action_table = self._select_admission_action_table(study, candidate_audit)
        except ValueError as error:
            fail(
                pass_name=CompilerPass.CONSTRUCTION,
                reason_codes=("CONTROLLER_CONSTRUCTION_FAILED", self._exception_reason(error)),
                highest_valid_rung=EvidenceRung.ADMISSION,
            )
        audit_identities = tuple(
            ObjectIdentity.from_record(value.audit_id, value) for value in candidate_audit
        )
        action_table_identity = ObjectIdentity.from_record(
            f"lookup.{study.study_id}",
            (
                _ActHoldCompiledActionLookupTable(
                    table_id=f"lookup.{study.study_id}",
                    entries=cast(tuple[ActHoldCompiledCellAction, ...], action_table),
                )
                if isinstance(study, DeliveryControllerStudy)
                else _CompiledActionLookupTable(
                    table_id=f"lookup.{study.study_id}",
                    entries=action_table,
                )
            ),
        )
        products.append(
            self._product(
                study,
                CompilerPass.CONSTRUCTION,
                (chart_identity, admission_identity, reachability_identity),
                (*audit_identities, action_table_identity),
                EvidenceRung.ADMISSION,
            )
        )

        compiled_hold: CompiledAdmissionHoldFibre | None = None
        if study.measured_hold_fibre is not None:
            try:
                if isinstance(study, DeliveryControllerStudy):
                    compiled_hold = self._qualify_delivery_hold_fibre(
                        study,
                        study.measured_hold_fibre,
                        admission=admission,
                        reachability=cast(FiniteCertificateReachabilityComparison, reachability),
                    )
                else:
                    compiled_hold = self._qualify_admission_hold_fibre(
                        study,
                        study.measured_hold_fibre,
                        admission=admission,
                        reachability=cast(ControlledMapReachabilityComparison, reachability),
                    )
            except ValueError as error:
                fail(
                    pass_name=CompilerPass.RUNTIME_COMPLETENESS,
                    reason_codes=("HOLD_FIBRE_BINDING_FAILED", self._exception_reason(error)),
                    highest_valid_rung=EvidenceRung.ADMISSION,
                )
        runtime_bindings = tuple(
            ObjectIdentity.from_record(value.binding_id, value)
            for value in study.implementations
            if value.role
            in {
                ImplementationRole.OBSERVER,
                ImplementationRole.ONLINE_GATE_EVALUATOR,
                ImplementationRole.DELIVERY,
            }
        )
        products.append(
            self._product(
                study,
                CompilerPass.RUNTIME_COMPLETENESS,
                (action_table_identity,),
                (
                    *runtime_bindings,
                    *(
                        (ObjectIdentity.from_record(compiled_hold.qualification_id, compiled_hold),)
                        if compiled_hold is not None
                        else ()
                    ),
                ),
                EvidenceRung.ADMISSION,
            )
        )

        evaluation = study.prospective_evaluation
        if evaluation is None:
            prospective_evaluation_binding = ProspectiveCompilationBinding(
                binding_id=f"prospective-evaluation-binding.{study.study_id}",
                disposition=ProspectiveBindingDisposition.NOT_DECLARED,
                evaluation_plan=None,
                evaluator=None,
            )
        else:
            evaluator = implementations[ImplementationRole.OUTCOME_EVALUATOR]
            prospective_evaluation_binding = ProspectiveCompilationBinding(
                binding_id=f"prospective-evaluation-binding.{study.study_id}",
                disposition=ProspectiveBindingDisposition.BOUND,
                evaluation_plan=evaluation,
                evaluator=ObjectIdentity.from_record(evaluator.binding_id, evaluator),
            )
        prospective_evaluation_binding_identity = ObjectIdentity.from_record(prospective_evaluation_binding.binding_id, prospective_evaluation_binding)
        products.append(
            self._product(
                study,
                CompilerPass.PROSPECTIVE_EVALUATION_BINDING,
                (programme_identity,),
                (prospective_evaluation_binding_identity,),
                EvidenceRung.ADMISSION,
                reason_codes=(
                    ("CONTROLLER_USE_NOT_DECLARED",)
                    if prospective_evaluation_binding.disposition is ProspectiveBindingDisposition.NOT_DECLARED
                    else ()
                ),
            )
        )

        try:
            replayed = (
                decode_least_magnitude_controller_study(study.canonical_bytes())
                if isinstance(study, LeastMagnitudeControllerStudy)
                else decode_delivery_controller_study(study.canonical_bytes())
                if isinstance(study, DeliveryControllerStudy)
                else decode_admission_controller_study(study.canonical_bytes())
            )
        except (TypeError, ValueError) as error:
            fail(
                pass_name=CompilerPass.LOWERING_AND_REPLAY,
                reason_codes=("CANONICAL_REPLAY_FAILED", self._exception_reason(error)),
                highest_valid_rung=EvidenceRung.ADMISSION,
            )
        if replayed != study or replayed.canonical_bytes() != study.canonical_bytes():
            fail(
                pass_name=CompilerPass.LOWERING_AND_REPLAY,
                reason_codes=("CANONICAL_REPLAY_MISMATCH",),
                highest_valid_rung=EvidenceRung.ADMISSION,
            )
        products.append(
            self._product(
                study,
                CompilerPass.LOWERING_AND_REPLAY,
                (programme_identity,),
                (programme_identity,),
                EvidenceRung.ADMISSION,
            )
        )

        disposition = (
            CompiledDisposition.CONTROLLER
            if action_table
            else (
                CompiledDisposition.HOLD_ONLY
                if compiled_hold is not None
                and compiled_hold.disposition is HoldFibreDisposition.QUALIFIED
                else CompiledDisposition.NONATTEMPT
            )
        )
        if isinstance(study, DeliveryControllerStudy):
            disposition = _finite_compiled_disposition(
                study,
                cast(tuple[ActHoldCompiledCellAction, ...], action_table),
                compiled_hold,
            )
        if disposition is CompiledDisposition.CONTROLLER:
            reason_codes: tuple[str, ...] = ()
        elif disposition is CompiledDisposition.HOLD_ONLY:
            reason_codes = (
                ("FINITE_HOLD_SELECTED",)
                if isinstance(study, DeliveryControllerStudy) and action_table
                else ("NO_ELIGIBLE_ACTIVE_ACTION",)
            )
        else:
            reason_codes = tuple(
                sorted(
                    {
                        "NO_ELIGIBLE_ACTIVE_ACTION",
                        *(
                            ("MEASURED_HOLD_FIBRE_MISSING",)
                            if compiled_hold is None
                            else compiled_hold.reason_codes
                        ),
                    }
                )
            )
        geometry_diagnostics = tuple(
            sorted(
                {
                    *(
                        ("ADMISSION_MODEL_SET_DISAGREEMENT",)
                        if admission.disagreement_cell_ids
                        else ()
                    ),
                    *(
                        ("REACHABILITY_MODEL_SET_DISAGREEMENT",)
                        if reachability.disagreement_cell_ids
                        else ()
                    ),
                }
            )
        )
        if isinstance(study, DeliveryControllerStudy):
            reason_codes = _finite_compiled_reasons(
                disposition,
                cast(tuple[ActHoldCompiledCellAction, ...], action_table),
                candidate_audit,
                compiled_hold,
            )
            compiled_type: type[CompiledDeliveryControllerStudy] = (
                CompiledLeastMagnitudeControllerStudy
                if isinstance(study, LeastMagnitudeControllerStudy)
                else CompiledDeliveryControllerStudy
            )
            return compiled_type(
                compiled_study_id=f"compiled.{study.study_id}",
                study=study,
                specification_fingerprint=study.fingerprint(),
                pass_products=tuple(products),
                admission=admission,
                reachability=cast(FiniteCertificateReachabilityComparison, reachability),
                action_table=cast(tuple[ActHoldCompiledCellAction, ...], action_table),
                candidate_audit=candidate_audit,
                implementations=study.implementations,
                hold_fibre=compiled_hold,
                prospective_evaluation_binding=prospective_evaluation_binding,
                disposition=disposition,
                reason_codes=reason_codes,
                geometry_diagnostic_codes=geometry_diagnostics,
                compilation_ceiling=EvidenceCeiling.ADMISSION,
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
                new_scientific_result=False,
            )

        return CompiledAdmissionControllerStudy(
            compiled_study_id=f"compiled.{study.study_id}",
            study=study,
            specification_fingerprint=study.fingerprint(),
            pass_products=tuple(products),
            admission=admission,
            reachability=cast(ControlledMapReachabilityComparison, reachability),
            action_table=action_table,
            candidate_audit=candidate_audit,
            implementations=study.implementations,
            hold_fibre=compiled_hold,
            prospective_evaluation_binding=prospective_evaluation_binding,
            disposition=disposition,
            reason_codes=reason_codes,
            geometry_diagnostic_codes=geometry_diagnostics,
            compilation_ceiling=EvidenceCeiling.ADMISSION,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            new_scientific_result=False,
        )

    @staticmethod
    def _controlled_map_candidate_cell(
        study: AdmissionControllerStudy | DeliveryControllerStudy,
        candidate: AdmissionActionCandidate,
    ) -> ReceiptAdmissionAdmissionCandidateCell:
        cell = next(
            (
                value
                for value in study.admission.candidate_cells
                if ObjectIdentity.from_record(value.candidate_cell_id, value)
                == candidate.admission_candidate_cell
            ),
            None,
        )
        if cell is None:
            raise ValueError("controller candidate admission cell is absent or stale")
        return cell

    @staticmethod
    def _exact_binding_identities(
        receipts: tuple[
            AdmissionCoordinateGateReceipt
            | ControlledMapReachabilityReceipt
            | UtilityEvaluationReceipt
            | FiniteReachabilityMethodReceipt,
            ...,
        ],
    ) -> tuple[ObjectIdentity, ...]:
        by_id: dict[str, LawMemberEvaluationBinding] = {}
        for receipt in receipts:
            bindings = (
                receipt.request.response_set.evaluation_bindings
                if isinstance(receipt, FiniteReachabilityMethodReceipt)
                else (receipt.law_evaluation_binding,)
            )
            for binding in bindings:
                prior = by_id.setdefault(binding.binding_id, binding)
                if prior != binding:
                    raise ValueError("one law-evaluation binding ID has several exact records")
        return tuple(
            ObjectIdentity.from_record(binding_id, by_id[binding_id])
            for binding_id in sorted(by_id)
        )

    @classmethod
    def _audit_admission_candidate(
        cls,
        study: AdmissionControllerStudy,
        candidate: AdmissionActionCandidate,
        *,
        admission: AdmissionComparison,
        reachability: ControlledMapReachabilityComparison,
    ) -> AdmissionCandidateAudit:
        cell = cls._controlled_map_candidate_cell(study, candidate)
        corpus = study.admission.corpus
        coordinate_ids = set(cell.planned_coordinate_ids)
        constituents: list[AdmissionCandidateConstituentAudit] = []
        for coordinate_id in sorted(coordinate_ids):
            coordinate = next(
                value for value in corpus.plan.coordinates if value.coordinate_id == coordinate_id
            )
            gates = tuple(
                value
                for value in corpus.gate_receipts
                if value.planned_coordinate.coordinate_id == coordinate_id
            )
            reach = next(
                value
                for value in corpus.reachability_receipts
                if value.planned_coordinate.coordinate_id == coordinate_id
            )
            utility = next(
                value
                for value in corpus.utility_receipts
                if value.planned_coordinate.coordinate_id == coordinate_id
            )
            if len(gates) != len(AdmissionGateKind):
                raise ValueError("candidate gate constituent grid is incomplete")
            reasons: set[str] = set()
            unevaluable = False
            for gate in gates:
                if gate.status is GateStatus.FAIL:
                    reasons.add(f"{gate.predicate.gate_kind.value}_FAILED")
                elif gate.status is GateStatus.UNEVALUABLE:
                    reasons.add(f"{gate.predicate.gate_kind.value}_UNEVALUABLE")
                    unevaluable = True
            if reach.status is ReachabilityCellDisposition.UNREACHABLE:
                reasons.add("REACHABILITY_BLOCKED")
            elif reach.status is ReachabilityCellDisposition.UNEVALUABLE:
                reasons.add("REACHABILITY_UNEVALUABLE")
                unevaluable = True
            if utility.status is ReceiptAdmissionUtilityStatus.NOT_VIABLE:
                reasons.update(utility.reason_codes or ("UTILITY_NOT_VIABLE",))
            elif utility.status is ReceiptAdmissionUtilityStatus.UNEVALUABLE:
                reasons.update(utility.reason_codes or ("UTILITY_UNEVALUABLE",))
                unevaluable = True
            eligibility = (
                CandidateEligibility.UNEVALUABLE
                if unevaluable
                else (CandidateEligibility.INELIGIBLE if reasons else CandidateEligibility.ELIGIBLE)
            )
            all_receipts: tuple[
                AdmissionCoordinateGateReceipt | ControlledMapReachabilityReceipt | UtilityEvaluationReceipt,
                ...,
            ] = (*gates, reach, utility)
            constituents.append(
                AdmissionCandidateConstituentAudit(
                    audit_id=f"constituent-audit.{candidate.candidate_id}.{coordinate_id}",
                    planned_coordinate=coordinate,
                    receipt_ids=tuple(sorted(value.receipt_id for value in all_receipts)),
                    law_evaluation_bindings=cls._exact_binding_identities(all_receipts),
                    utility=utility.derived_utility,
                    eligibility=eligibility,
                    reason_codes=tuple(sorted(reasons)),
                )
            )
        return cls._finish_candidate_audit(
            study,
            candidate,
            constituents=tuple(constituents),
            cell=cell,
            admission=admission,
            certified_cell_ids=reachability.robust.reachable_cell_ids,
        )

    @staticmethod
    def _finish_candidate_audit(
        study: AdmissionControllerStudy | DeliveryControllerStudy,
        candidate: AdmissionActionCandidate,
        *,
        constituents: tuple[AdmissionCandidateConstituentAudit, ...],
        cell: ReceiptAdmissionAdmissionCandidateCell,
        admission: AdmissionComparison,
        certified_cell_ids: tuple[str, ...],
        extra_reasons: tuple[str, ...] = (),
        extra_unevaluable: bool = False,
    ) -> AdmissionCandidateAudit:
        corpus = study.admission.corpus
        constituent_tuple = tuple(sorted(constituents, key=lambda value: value.audit_id))
        candidate_reasons = {reason for value in constituent_tuple for reason in value.reason_codes}
        candidate_reasons.update(extra_reasons)
        if cell.candidate_cell_id not in admission.robust.admitted_cell_ids:
            candidate_reasons.add("OUTSIDE_ROBUST_ADMISSION")
        if cell.candidate_cell_id not in certified_cell_ids:
            candidate_reasons.add(
                "OUTSIDE_ROBUST_FINITE_CERTIFICATION"
                if isinstance(study, DeliveryControllerStudy)
                else "OUTSIDE_ROBUST_REACHABILITY"
            )
        action = corpus.plan.action_fibre(cell.action_fibre)
        action_binding = next(
            value
            for value in study.action_bindings
            if value.action_binding_id == action.action_binding_id
        )
        action_unevaluable = (
            action_binding.action_word.support_status is ActionWordSupportStatus.UNEVALUABLE
            or action_binding.causal_prefix.disposition is CausalCompositionDisposition.UNEVALUABLE
        )
        if action_unevaluable:
            candidate_reasons.add("ACTION_OR_CAUSAL_SUPPORT_UNEVALUABLE")
        elif (
            action_binding.action_word.support_status is not ActionWordSupportStatus.SUPPORTED
            or action_binding.causal_prefix.disposition is not CausalCompositionDisposition.DEFINED
        ):
            candidate_reasons.add("ACTION_OR_CAUSAL_SUPPORT_BLOCKED")
        utilities = tuple(value.utility for value in constituent_tuple if value.utility is not None)
        complete_utilities = len(utilities) == len(constituent_tuple)
        if complete_utilities:
            robust = NamedDecimal(
                value_id=f"robust-utility.{candidate.candidate_id}",
                value=min(value.value for value in utilities),
                unit=utilities[0].unit,
            )
            nominal_values = tuple(
                value.utility
                for value in constituent_tuple
                if value.planned_coordinate.denominator_member_id
                == study.synthesis.candidate_chart.nominal_denominator_member_id
                and value.utility is not None
            )
            if not nominal_values:
                raise ValueError("candidate lacks nominal-member utility")
            nominal = NamedDecimal(
                value_id=f"nominal-utility.{candidate.candidate_id}",
                value=min(value.value for value in nominal_values),
                unit=nominal_values[0].unit,
            )
        else:
            robust = None
            nominal = None
        unevaluable = (
            extra_unevaluable
            or action_unevaluable
            or any(
                value.eligibility is CandidateEligibility.UNEVALUABLE for value in constituent_tuple
            )
        )
        eligibility = (
            CandidateEligibility.UNEVALUABLE
            if unevaluable
            else (
                CandidateEligibility.INELIGIBLE
                if candidate_reasons
                else CandidateEligibility.ELIGIBLE
            )
        )
        return AdmissionCandidateAudit(
            audit_id=f"candidate-audit.{candidate.candidate_id}",
            candidate_id=candidate.candidate_id,
            decision_cell_id=candidate.decision_cell_id,
            admission_candidate_cell=candidate.admission_candidate_cell,
            action_binding_id=action.action_binding_id,
            nominal_denominator_member_id=(
                study.synthesis.candidate_chart.nominal_denominator_member_id
            ),
            constituents=constituent_tuple,
            robust_utility=robust,
            nominal_utility=nominal,
            eligibility=eligibility,
            reason_codes=tuple(sorted(candidate_reasons)),
        )

    @staticmethod
    def _candidate_winners(
        study: AdmissionControllerStudy | DeliveryControllerStudy,
        audits: tuple[AdmissionCandidateAudit, ...],
    ) -> tuple[AdmissionCandidateAudit, ...]:
        candidates = {c.candidate_id: c for c in study.synthesis.candidate_chart.candidates}
        by_cell: dict[str, list[AdmissionCandidateAudit]] = {}
        for audit in audits:
            if audit.eligibility is CandidateEligibility.ELIGIBLE:
                by_cell.setdefault(audit.decision_cell_id, []).append(audit)
        winners = []
        for _, eligible in sorted(by_cell.items()):

            def score(value: AdmissionCandidateAudit) -> tuple[Decimal, Decimal, int]:
                if value.robust_utility is None or value.nominal_utility is None:
                    raise ValueError("eligible candidate lost its complete utility")
                if isinstance(study, LeastMagnitudeControllerStudy):
                    magnitude = next(
                        m.magnitude.value
                        for m in study.native_magnitudes
                        if m.action_binding_id == value.action_binding_id
                    )
                    return (-magnitude, Decimal(0), -candidates[value.candidate_id].priority_rank)
                return (
                    value.robust_utility.value,
                    value.nominal_utility.value,
                    -candidates[value.candidate_id].priority_rank,
                )

            selected = max(eligible, key=score)
            if sum(score(value) == score(selected) for value in eligible) != 1:
                raise ValueError("scientific candidate tie remains unresolved")
            winners.append(selected)
        return tuple(winners)

    @classmethod
    def _select_admission_action_table(
        cls,
        study: AdmissionControllerStudy,
        audits: tuple[AdmissionCandidateAudit, ...],
    ) -> tuple[CompiledCellAction, ...]:
        candidates = {c.candidate_id: c for c in study.synthesis.candidate_chart.candidates}
        actions = {a.action_binding_id: a for a in study.action_bindings}
        selections = []
        for selected in cls._candidate_winners(study, audits):
            candidate = candidates[selected.candidate_id]
            if selected.robust_utility is None or selected.nominal_utility is None:
                raise AssertionError("selected candidate lost utility")
            selections.append(
                CompiledCellAction(
                    selection_id=f"cell-selection.{selected.decision_cell_id}",
                    decision_cell_id=selected.decision_cell_id,
                    admission_cell_id=candidate.admission_candidate_cell.object_id,
                    candidate_id=candidate.candidate_id,
                    action_binding=actions[selected.action_binding_id],
                    robust_utility=selected.robust_utility,
                    nominal_utility=selected.nominal_utility,
                    priority_rank=candidate.priority_rank,
                )
            )
        return tuple(selections)

    @classmethod
    def _select_delivery_action_table(
        cls,
        study: DeliveryControllerStudy,
        audits: tuple[AdmissionCandidateAudit, ...],
    ) -> tuple[ActHoldCompiledCellAction, ...]:
        candidates = {c.candidate_id: c for c in study.synthesis.candidate_chart.candidates}
        actions = {a.action_binding_id: a for a in study.action_bindings}
        selections = []
        for selected in cls._candidate_winners(study, audits):
            candidate = candidates[selected.candidate_id]
            if selected.robust_utility is None or selected.nominal_utility is None:
                raise AssertionError("selected finite candidate lost utility")
            selections.append(
                ActHoldCompiledCellAction(
                    selection_id=f"cell-selection.{selected.decision_cell_id}",
                    decision_cell_id=selected.decision_cell_id,
                    admission_cell_id=candidate.admission_candidate_cell.object_id,
                    candidate_id=candidate.candidate_id,
                    action_binding=actions[selected.action_binding_id],
                    robust_utility=selected.robust_utility,
                    nominal_utility=selected.nominal_utility,
                    priority_rank=candidate.priority_rank,
                    kind=candidate.kind,
                )
            )
        return tuple(selections)

    @classmethod
    def _audit_delivery_candidate(
        cls,
        study: DeliveryControllerStudy,
        candidate: ActHoldActionCandidate,
        *,
        admission: AdmissionComparison,
        reachability: FiniteCertificateReachabilityComparison,
    ) -> AdmissionCandidateAudit:
        cell = cls._controlled_map_candidate_cell(study, candidate)
        corpus = study.admission.corpus
        coordinates = {c.coordinate_id: c for c in corpus.plan.coordinates}
        constituents = []
        for coordinate_id in cell.planned_coordinate_ids:
            coordinate = coordinates[coordinate_id]
            gates = tuple(g for g in corpus.gate_receipts if g.planned_coordinate == coordinate)
            reach = next(
                r for r in corpus.reachability_receipts if r.planned_coordinate == coordinate
            )
            utilities = tuple(
                u for u in corpus.utility_receipts if u.planned_coordinate == coordinate
            )
            if len(gates) != len(coordinate.qualification_view_ids) * len(AdmissionGateKind) or len(
                utilities
            ) != len(coordinate.qualification_view_ids):
                raise ValueError("finite candidate omits a numerical-view gate/utility constituent")
            reasons: set[str] = set()
            unevaluable = False
            for gate in gates:
                if gate.status is GateStatus.FAIL:
                    reasons.add(f"{gate.predicate.gate_kind.value}_FAILED")
                elif gate.status is GateStatus.UNEVALUABLE:
                    reasons.add(f"{gate.predicate.gate_kind.value}_UNEVALUABLE")
                    unevaluable = True
                reasons.update(gate.reason_codes)
            if reach.status is not FiniteCertificateStatus.CERTIFIED:
                reasons.add(f"FINITE_{reach.status.value}")
                unevaluable = unevaluable or reach.status is FiniteCertificateStatus.UNEVALUABLE
            for utility in utilities:
                if utility.status is not ReceiptAdmissionUtilityStatus.VIABLE:
                    reasons.update(utility.reason_codes or (f"UTILITY_{utility.status.value}",))
                    unevaluable = unevaluable or utility.status is ReceiptAdmissionUtilityStatus.UNEVALUABLE
            values = tuple(u.derived_utility for u in utilities if u.derived_utility is not None)
            if len({v.unit for v in values}) > 1:
                raise ValueError("finite numerical views mix native utility units")
            utility_value = (
                NamedDecimal(
                    f"view-robust-utility.{candidate.candidate_id}.{coordinate_id}",
                    min(v.value for v in values),
                    values[0].unit,
                )
                if len(values) == len(utilities)
                else None
            )
            all_receipts: tuple[
                AdmissionCoordinateGateReceipt
                | FiniteReachabilityMethodReceipt
                | UtilityEvaluationReceipt,
                ...,
            ] = (*gates, reach, *utilities)
            constituents.append(
                AdmissionCandidateConstituentAudit(
                    audit_id=f"constituent-audit.{candidate.candidate_id}.{coordinate_id}",
                    planned_coordinate=coordinate,
                    receipt_ids=tuple(sorted(r.receipt_id for r in all_receipts)),
                    law_evaluation_bindings=cls._exact_binding_identities(all_receipts),
                    utility=utility_value,
                    eligibility=(
                        CandidateEligibility.UNEVALUABLE
                        if unevaluable
                        else CandidateEligibility.INELIGIBLE
                        if reasons
                        else CandidateEligibility.ELIGIBLE
                    ),
                    reason_codes=tuple(sorted(reasons)),
                )
            )
        hold_missing = candidate.kind is NativeCandidateKind.HOLD and (
            study.measured_hold_fibre is None
            or study.measured_hold_fibre.admission_candidate_cell != candidate.admission_candidate_cell
        )
        return cls._finish_candidate_audit(
            study,
            candidate,
            constituents=tuple(constituents),
            cell=cell,
            admission=admission,
            certified_cell_ids=reachability.robust.certified_cell_ids,
            extra_reasons=(("MEASURED_HOLD_FIBRE_MISSING",) if hold_missing else ()),
            extra_unevaluable=hold_missing,
        )

    @classmethod
    def _qualify_delivery_hold_fibre(
        cls,
        study: DeliveryControllerStudy,
        source: AdmissionMeasuredHoldFibre,
        *,
        admission: AdmissionComparison,
        reachability: FiniteCertificateReachabilityComparison,
    ) -> CompiledAdmissionHoldFibre:
        return cls._qualify_admission_hold_fibre(
            study, source, admission=admission, reachability=reachability
        )

    @classmethod
    def _qualify_admission_hold_fibre(
        cls,
        study: AdmissionControllerStudy | DeliveryControllerStudy,
        source: AdmissionMeasuredHoldFibre,
        *,
        admission: AdmissionComparison,
        reachability: ControlledMapReachabilityComparison | FiniteCertificateReachabilityComparison,
    ) -> CompiledAdmissionHoldFibre:
        cell = next(
            value
            for value in study.admission.candidate_cells
            if ObjectIdentity.from_record(value.candidate_cell_id, value)
            == source.admission_candidate_cell
        )
        corpus = study.admission.corpus
        coordinate_ids = set(cell.planned_coordinate_ids)
        gates = tuple(
            value
            for value in corpus.gate_receipts
            if value.planned_coordinate.coordinate_id in coordinate_ids
        )
        reaches = tuple(
            value
            for value in corpus.reachability_receipts
            if value.planned_coordinate.coordinate_id in coordinate_ids
        )
        expected_gate_count = len(AdmissionGateKind) * (
            sum(
                len(c.qualification_view_ids)
                for c in corpus.plan.coordinates
                if c.coordinate_id in coordinate_ids
            )
            if isinstance(study, DeliveryControllerStudy)
            else len(coordinate_ids)
        )
        if len(gates) != expected_gate_count or len(reaches) != len(coordinate_ids):
            raise ValueError("compiled HOLD receipt grid is incomplete")
        reasons: set[str] = set()
        disposition = HoldFibreDisposition.QUALIFIED
        if any(value.status is GateStatus.UNEVALUABLE for value in gates):
            disposition = HoldFibreDisposition.UNEVALUABLE
            reasons.add("HOLD_GATE_UNEVALUABLE")
        elif any(value.status is GateStatus.FAIL for value in gates):
            disposition = HoldFibreDisposition.RECEIVER_INADMISSIBLE
            reasons.add("HOLD_GATE_FAILED")
        if any(
            value.status
            in (ReachabilityCellDisposition.UNEVALUABLE, FiniteCertificateStatus.UNEVALUABLE)
            for value in reaches
        ):
            disposition = HoldFibreDisposition.UNEVALUABLE
            reasons.add("HOLD_REACHABILITY_UNEVALUABLE")
        elif any(
            value.status
            not in (ReachabilityCellDisposition.REACHABLE, FiniteCertificateStatus.CERTIFIED)
            for value in reaches
        ):
            disposition = HoldFibreDisposition.UNVIABLE
            reasons.add("HOLD_REACHABILITY_FAILED")
        if cell.candidate_cell_id not in admission.robust.admitted_cell_ids:
            disposition = HoldFibreDisposition.RECEIVER_INADMISSIBLE
            reasons.add("HOLD_OUTSIDE_ROBUST_ADMISSION")
        certified_cells = (
            reachability.robust.certified_cell_ids
            if isinstance(reachability, FiniteCertificateReachabilityComparison)
            else reachability.robust.reachable_cell_ids
        )
        if cell.candidate_cell_id not in certified_cells:
            disposition = HoldFibreDisposition.UNVIABLE
            reasons.add("HOLD_OUTSIDE_ROBUST_REACHABILITY")
        admission_action_fibre = corpus.plan.action_fibre(cell.action_fibre)
        action = next(
            value
            for value in study.action_bindings
            if value.action_binding_id == admission_action_fibre.action_binding_id
        )
        if (
            action.action_word.support_status is ActionWordSupportStatus.UNEVALUABLE
            or action.causal_prefix.disposition is CausalCompositionDisposition.UNEVALUABLE
        ):
            disposition = HoldFibreDisposition.UNEVALUABLE
            reasons.add("HOLD_ACTION_UNEVALUABLE")
        elif (
            action.action_word.support_status is not ActionWordSupportStatus.SUPPORTED
            or action.causal_prefix.disposition is not CausalCompositionDisposition.DEFINED
        ):
            disposition = HoldFibreDisposition.UNSUPPORTED
            reasons.add("HOLD_ACTION_UNSUPPORTED")
        if (
            any(value.status is GateStatus.UNEVALUABLE for value in gates)
            or any(
                value.status
                in (ReachabilityCellDisposition.UNEVALUABLE, FiniteCertificateStatus.UNEVALUABLE)
                for value in reaches
            )
            or action.action_word.support_status is ActionWordSupportStatus.UNEVALUABLE
            or action.causal_prefix.disposition is CausalCompositionDisposition.UNEVALUABLE
        ):
            disposition = HoldFibreDisposition.UNEVALUABLE
        all_receipts: tuple[
            AdmissionCoordinateGateReceipt
            | ControlledMapReachabilityReceipt
            | FiniteReachabilityMethodReceipt,
            ...,
        ] = (
            *gates,
            *reaches,
        )
        reasons.update(
            f"HOLD_FINITE_{r.status.value}"
            for r in reaches
            if isinstance(r, FiniteReachabilityMethodReceipt)
            and r.status is not FiniteCertificateStatus.CERTIFIED
        )
        return CompiledAdmissionHoldFibre(
            qualification_id=f"hold-qualification.{source.hold_fibre_id}",
            source=source,
            action_binding=action,
            planned_coordinate_ids=tuple(sorted(coordinate_ids)),
            gate_receipt_ids=tuple(sorted(value.receipt_id for value in gates)),
            reachability_receipt_ids=tuple(sorted(value.receipt_id for value in reaches)),
            law_evaluation_bindings=cls._exact_binding_identities(all_receipts),
            disposition=disposition,
            reason_codes=tuple(sorted(reasons)),
        )

    @staticmethod
    def _validate_candidate_receipt_corpora(study: AtlasControllerStudy) -> None:
        gate_corpus = {value.receipt_id: value for value in study.admission.receipts}
        reachability_corpus = {value.receipt_id: value for value in study.reachability.receipts}
        candidate_receipts: list[CandidateGateReceipt | CandidateReachabilityReceipt] = []
        for candidate in study.synthesis.candidate_chart.candidates:
            candidate_receipts.extend(candidate.gate_receipts)
            candidate_receipts.extend(candidate.reachability_receipts)
        hold = study.measured_hold_fibre
        if hold is not None:
            candidate_receipts.extend(hold.gate_receipts)
            candidate_receipts.extend(hold.reachability_receipts)
        for wrapped in candidate_receipts:
            receipt = wrapped.receipt
            corpus = (
                gate_corpus if isinstance(wrapped, CandidateGateReceipt) else reachability_corpus
            )
            if corpus.get(receipt.receipt_id) != receipt:
                raise ValueError("candidate receipt is orphaned, foreign or stale")

    @staticmethod
    def _audit_atlas_candidate(
        study: AtlasControllerStudy,
        candidate: AtlasActionCandidate,
        *,
        admission: AdmissionComparison,
        reachability: ReachabilityComparison,
    ) -> AtlasCandidateAudit:
        chart = study.synthesis.candidate_chart
        actions = {value.action_binding_id: value for value in study.action_bindings}
        action = actions[candidate.action_binding_id]
        reasons: set[str] = set()
        unevaluable = False
        if (
            action.action_word.support_status is ActionWordSupportStatus.UNEVALUABLE
            or action.causal_prefix.disposition is CausalCompositionDisposition.UNEVALUABLE
        ):
            reasons.add("ACTION_OR_CAUSAL_SUPPORT_UNEVALUABLE")
            unevaluable = True
        elif action.action_word.support_status is not ActionWordSupportStatus.SUPPORTED:
            reasons.add("ACTION_SUPPORT_BLOCKED")
        for wrapped in candidate.gate_receipts:
            gate_receipt = wrapped.receipt
            if gate_receipt.status is GateStatus.FAIL:
                reasons.add(f"{gate_receipt.predicate.gate_kind.value}_FAILED")
            elif gate_receipt.status is GateStatus.UNEVALUABLE:
                reasons.add(f"{gate_receipt.predicate.gate_kind.value}_UNEVALUABLE")
                unevaluable = True
        for reach_wrapped in candidate.reachability_receipts:
            reachability_receipt = reach_wrapped.receipt
            if reachability_receipt.status is ReachabilityCellDisposition.UNREACHABLE:
                reasons.add("REACHABILITY_BLOCKED")
            elif reachability_receipt.status is ReachabilityCellDisposition.UNEVALUABLE:
                reasons.add("REACHABILITY_UNEVALUABLE")
                unevaluable = True
        if candidate.admission_cell_id not in admission.robust.admitted_cell_ids:
            reasons.add("OUTSIDE_ROBUST_ADMISSION")
        if candidate.admission_cell_id not in reachability.robust.reachable_cell_ids:
            reasons.add("OUTSIDE_ROBUST_REACHABILITY")
        utilities = tuple(
            sorted(
                (
                    CandidateMemberUtility(
                        model_member_id=value.model_member_id,
                        utility=value.derived_utility,
                    )
                    for value in candidate.utility_receipts
                ),
                key=lambda value: value.model_member_id,
            )
        )
        by_member = {value.model_member_id: value.utility for value in utilities}
        robust = NamedDecimal(
            value_id=f"robust-utility.{candidate.candidate_id}",
            value=min(value.utility.value for value in utilities),
            unit=chart.utility_definition.native_unit,
        )
        nominal_source = by_member[chart.nominal_member_id]
        nominal = NamedDecimal(
            value_id=f"nominal-utility.{candidate.candidate_id}",
            value=nominal_source.value,
            unit=nominal_source.unit,
        )
        if robust.value < chart.utility_definition.minimum_robust_utility.value:
            reasons.add("ROBUST_UTILITY_BELOW_FLOOR")
        eligibility = (
            CandidateEligibility.UNEVALUABLE
            if unevaluable
            else (CandidateEligibility.INELIGIBLE if reasons else CandidateEligibility.ELIGIBLE)
        )
        return AtlasCandidateAudit(
            audit_id=f"candidate-audit.{candidate.candidate_id}",
            candidate_id=candidate.candidate_id,
            decision_cell_id=candidate.decision_cell_id,
            action_binding_id=candidate.action_binding_id,
            member_utilities=utilities,
            robust_utility=robust,
            nominal_utility=nominal,
            eligibility=eligibility,
            reason_codes=tuple(sorted(reasons)),
        )

    @staticmethod
    def _select_atlas_action_table(
        study: AtlasControllerStudy,
        audits: tuple[AtlasCandidateAudit, ...],
    ) -> tuple[CompiledCellAction, ...]:
        candidates = {
            value.candidate_id: value for value in study.synthesis.candidate_chart.candidates
        }
        actions = {value.action_binding_id: value for value in study.action_bindings}
        by_cell: dict[str, list[AtlasCandidateAudit]] = {}
        for audit in audits:
            if audit.eligibility is CandidateEligibility.ELIGIBLE:
                by_cell.setdefault(audit.decision_cell_id, []).append(audit)
        selections: list[CompiledCellAction] = []
        for cell_id, eligible in sorted(by_cell.items()):

            def score(value: AtlasCandidateAudit) -> tuple[Decimal, Decimal, int]:
                candidate = candidates[value.candidate_id]
                return (
                    value.robust_utility.value,
                    value.nominal_utility.value,
                    -candidate.priority_rank,
                )

            selected = max(eligible, key=score)
            selected_score = score(selected)
            if sum(score(value) == selected_score for value in eligible) != 1:
                raise ValueError("scientific candidate tie remains unresolved")
            candidate = candidates[selected.candidate_id]
            selections.append(
                CompiledCellAction(
                    selection_id=f"cell-selection.{cell_id}",
                    decision_cell_id=cell_id,
                    admission_cell_id=candidate.admission_cell_id,
                    candidate_id=candidate.candidate_id,
                    action_binding=actions[candidate.action_binding_id],
                    robust_utility=selected.robust_utility,
                    nominal_utility=selected.nominal_utility,
                    priority_rank=candidate.priority_rank,
                )
            )
        return tuple(selections)

    @staticmethod
    def _qualify_atlas_hold_fibre(
        study: AtlasControllerStudy,
        source: AtlasMeasuredHoldFibre,
        *,
        admission: AdmissionComparison,
        reachability: ReachabilityComparison,
    ) -> CompiledAtlasHoldFibre:
        actions = {value.action_binding_id: value for value in study.action_bindings}
        action = actions[source.action_binding_id]
        reasons: set[str] = set()
        disposition = HoldFibreDisposition.QUALIFIED
        if (
            action.action_word.support_status is ActionWordSupportStatus.UNEVALUABLE
            or action.causal_prefix.disposition is CausalCompositionDisposition.UNEVALUABLE
        ):
            disposition = HoldFibreDisposition.UNEVALUABLE
            reasons.add("HOLD_ACTION_UNEVALUABLE")
        elif (
            action.action_word.support_status is not ActionWordSupportStatus.SUPPORTED
            or action.causal_prefix.disposition is not CausalCompositionDisposition.DEFINED
        ):
            disposition = HoldFibreDisposition.UNSUPPORTED
            reasons.add("HOLD_ACTION_UNSUPPORTED")
        gate_statuses = {value.receipt.status for value in source.gate_receipts}
        if GateStatus.UNEVALUABLE in gate_statuses:
            disposition = HoldFibreDisposition.UNEVALUABLE
            reasons.add("HOLD_GATE_UNEVALUABLE")
        elif GateStatus.FAIL in gate_statuses:
            disposition = HoldFibreDisposition.RECEIVER_INADMISSIBLE
            reasons.add("HOLD_GATE_FAILED")
        reach_receipts = tuple(value.receipt for value in source.reachability_receipts)
        if any(value.status is ReachabilityCellDisposition.UNEVALUABLE for value in reach_receipts):
            disposition = HoldFibreDisposition.UNEVALUABLE
            reasons.add("HOLD_REACHABILITY_UNEVALUABLE")
        elif not all(
            value.status is ReachabilityCellDisposition.REACHABLE
            and source.viable_direction_id in value.candidate_direction_ids
            and any(
                direction.direction_id == source.viable_direction_id
                and direction.status is GateStatus.PASS
                for direction in value.direction_receipts
            )
            for value in reach_receipts
        ):
            disposition = HoldFibreDisposition.UNVIABLE
            reasons.add("HOLD_DIRECTION_NOT_VIABLE")
        if source.admission_cell_id not in admission.robust.admitted_cell_ids:
            disposition = HoldFibreDisposition.RECEIVER_INADMISSIBLE
            reasons.add("HOLD_OUTSIDE_ROBUST_ADMISSION")
        if source.admission_cell_id not in reachability.robust.reachable_cell_ids:
            disposition = HoldFibreDisposition.UNVIABLE
            reasons.add("HOLD_OUTSIDE_ROBUST_REACHABILITY")
        return CompiledAtlasHoldFibre(
            qualification_id=f"hold-qualification.{source.hold_fibre_id}",
            source=source,
            action_binding=action,
            disposition=disposition,
            reason_codes=tuple(sorted(reasons)),
        )

    @staticmethod
    def _validate_receiver_information(study: AtlasControllerStudy) -> None:
        plan = study.receiver_quotient
        if plan is None:
            return
        assessment = plan.equivalence_assessment
        action_words = {
            (value.action_word.word_id, value.action_word.fingerprint())
            for value in study.action_bindings
        }
        causal_prefixes = {
            (value.causal_prefix.assessment_id, value.causal_prefix.fingerprint())
            for value in study.action_bindings
        }
        response_laws = {
            (value.law_id, value.fingerprint()) for value in study.admission.atlas.laws
        }
        gate_receipts = {
            (value.receipt_id, value.fingerprint()) for value in study.admission.receipts
        }
        reachability_receipts = {
            (value.receipt_id, value.fingerprint()) for value in study.reachability.receipts
        }
        for decision in assessment.decisions:
            for evidence in decision.action_evidence:
                if (
                    (evidence.action_word.word_id, evidence.action_word.fingerprint())
                    not in action_words
                    or (evidence.causal_prefix.object_id, evidence.causal_prefix.object_fingerprint)
                    not in causal_prefixes
                    or (evidence.response_law.object_id, evidence.response_law.object_fingerprint)
                    not in response_laws
                    or {
                        (value.object_id, value.object_fingerprint)
                        for value in evidence.gate_receipts
                    }
                    - gate_receipts
                    or (
                        evidence.reachability_receipt.object_id,
                        evidence.reachability_receipt.object_fingerprint,
                    )
                    not in reachability_receipts
                ):
                    raise ValueError(
                        "receiver candidate evidence differs from loaded programme operands"
                    )
        if assessment.decision_class is DecisionEquivalenceClass.SHARED_ACTION and (
            assessment.shared_action_word is None
            or (
                assessment.shared_action_word.word_id,
                assessment.shared_action_word.fingerprint(),
            )
            not in action_words
        ):
            raise ValueError("receiver quotient selects an unbound shared action")
        if plan.fallback_word is not None:
            hold = study.measured_hold_fibre
            actions = {
                value.action_binding_id: value.action_word for value in study.action_bindings
            }
            if hold is None or actions.get(hold.action_binding_id) != plan.fallback_word:
                raise ValueError("receiver quotient fallback differs from the measured HOLD fibre")

    @staticmethod
    def _compile_receiver_quotient(
        study: AtlasControllerStudy,
        *,
        admission_identity: ObjectIdentity,
        reachability_identity: ObjectIdentity,
    ) -> tuple[AmbiguityActionCertificate, ReceiverGeometryControlBinding]:
        plan = study.receiver_quotient
        if plan is None:
            raise ValueError("receiver quotient plan is absent")
        decision_class = plan.equivalence_assessment.decision_class
        disposition = {
            DecisionEquivalenceClass.SHARED_ACTION: (
                AmbiguityCertificateDisposition.SHARED_ROBUST_ACTION
            ),
            DecisionEquivalenceClass.SHARED_QUALIFIED_HOLD: (
                AmbiguityCertificateDisposition.SHARED_QUALIFIED_HOLD
            ),
            DecisionEquivalenceClass.SHARED_TERMINATION: (
                AmbiguityCertificateDisposition.SHARED_TERMINATION
            ),
            DecisionEquivalenceClass.DECISION_DISTINCT: (
                AmbiguityCertificateDisposition.DECISION_DISTINCT
            ),
            DecisionEquivalenceClass.UNEVALUABLE: (AmbiguityCertificateDisposition.UNEVALUABLE),
        }[decision_class]
        reasons = (
            ()
            if decision_class
            in {
                DecisionEquivalenceClass.SHARED_ACTION,
                DecisionEquivalenceClass.SHARED_QUALIFIED_HOLD,
                DecisionEquivalenceClass.SHARED_TERMINATION,
            }
            else (
                ("DECISION_DISTINCT_SAFE_ABSTENTION",)
                if decision_class is DecisionEquivalenceClass.DECISION_DISTINCT
                else ("DECISION_UNEVALUABLE_SAFE_ABSTENTION",)
            )
        )
        observer = next(
            value
            for value in study.implementations
            if value.binding_id == plan.observer_binding_id
        )
        certificate = AmbiguityActionCertificate(
            certificate_id=f"certificate.{plan.plan_id}",
            assessment=plan.equivalence_assessment,
            observer=ObjectIdentity.from_record(observer.binding_id, observer),
            compiled_execution=ObjectIdentity.from_record(
                study.synthesis.synthesis_id, study.synthesis
            ),
            disposition=disposition,
            shared_action_word=plan.equivalence_assessment.shared_action_word,
            fallback_word=plan.fallback_word,
            termination_contract=plan.termination_contract,
            evidence_ceiling=EvidenceCeiling.ADMISSION,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            parent_visibility_ceilings=(plan.visibility_ceiling,),
            visibility_ceiling=plan.visibility_ceiling,
            reason_codes=reasons,
        )
        binding = ReceiverGeometryControlBinding(
            binding_id=f"receiver-binding.{plan.plan_id}",
            relation=ObjectIdentity.from_record(
                study.law.relation.relation_id, study.law.relation
            ),
            response_laws=tuple(
                sorted(
                    (
                        ObjectIdentity.from_record(value.law_id, value)
                        for value in study.admission.atlas.laws
                    ),
                    key=lambda value: value.object_id,
                )
            ),
            atlas=ObjectIdentity.from_record(
                study.admission.atlas.atlas_id, study.admission.atlas
            ),
            admission=admission_identity,
            reachability=reachability_identity,
            observation_experiment=ObjectIdentity.from_record(
                plan.observation_experiment.experiment_id, plan.observation_experiment
            ),
            fiber_assessment=ObjectIdentity.from_record(
                plan.fiber_assessment.assessment_id, plan.fiber_assessment
            ),
            equivalence_assessment=ObjectIdentity.from_record(
                plan.equivalence_assessment.assessment_id, plan.equivalence_assessment
            ),
            certificate=ObjectIdentity.from_record(certificate.certificate_id, certificate),
            controller_study=ObjectIdentity.from_record(study.study_id, study),
        )
        return certificate, binding

    @staticmethod
    def _product(
        study: AtlasControllerStudy | AdmissionControllerStudy | DeliveryControllerStudy,
        pass_name: CompilerPass,
        inputs: tuple[ObjectIdentity, ...],
        outputs: tuple[ObjectIdentity, ...],
        rung: EvidenceRung,
        *,
        reason_codes: tuple[str, ...] = (),
    ) -> CompilerPassProduct:
        return CompilerPassProduct(
            pass_id=f"pass.{study.study_id}.{tuple(CompilerPass).index(pass_name):02d}",
            pass_name=pass_name,
            input_identities=inputs,
            output_identities=outputs,
            highest_valid_rung=rung,
            reason_codes=reason_codes,
        )

    @staticmethod
    def _exception_reason(error: TypeError | ValueError) -> str:
        message = str(error).upper()
        normalized = "".join(value if value.isalnum() else "_" for value in message)
        return f"DETAIL_{normalized.strip('_')[:160] or 'VALUE_ERROR'}"

    @staticmethod
    def _fail(
        study: AtlasControllerStudy | AdmissionControllerStudy | DeliveryControllerStudy,
        *,
        pass_name: CompilerPass,
        reason_codes: tuple[str, ...],
        highest_valid_rung: EvidenceRung,
        obstruction_codes: tuple[str, ...] = (),
    ) -> NoReturn:
        raise ControllerCompilationError(
            CompilerFailure(
                failure_id=f"failure.{study.study_id}.{pass_name.value.lower()}",
                pass_name=pass_name,
                affected_identity=study.study_id,
                highest_valid_rung=highest_valid_rung,
                reason_codes=tuple(sorted(set(reason_codes))),
                retained_obstruction_codes=tuple(sorted(set(obstruction_codes))),
            )
        )


@dataclass(frozen=True, slots=True)
class _CompiledActionLookupTable(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/compiled-action-lookup-table'

    table_id: str
    entries: tuple[CompiledCellAction, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.table_id, field_name="table_id")
        require_sorted_unique_ids(self.entries, attribute="decision_cell_id", field_name="entries")


@dataclass(frozen=True, slots=True)
class _ActHoldCompiledActionLookupTable(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/act-hold-compiled-action-lookup-table'

    table_id: str
    entries: tuple[ActHoldCompiledCellAction, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.table_id, field_name="table_id")
        require_sorted_unique_ids(self.entries, attribute="decision_cell_id", field_name="entries")


def decode_compiled_delivery_controller_study(payload: bytes) -> CompiledDeliveryControllerStudy:
    return decode_canonical_bytes(
        payload,
        CompiledDeliveryControllerStudy,
        maximum_bytes=MAX_STREAMING_COMPILED_CONTROLLER_BYTES,
    )


def decode_compiled_least_magnitude_controller_study(payload: bytes) -> CompiledLeastMagnitudeControllerStudy:
    return decode_canonical_bytes(
        payload, CompiledLeastMagnitudeControllerStudy, maximum_bytes=MAX_COMPILED_CONTROLLER_BYTES
    )


def decode_compiled_atlas_controller_study(payload: bytes) -> CompiledAtlasControllerStudy:
    return decode_canonical_bytes(
        payload,
        CompiledAtlasControllerStudy,
        maximum_bytes=MAX_COMPILED_CONTROLLER_BYTES,
    )


def decode_compiled_admission_controller_study(
    payload: bytes,
) -> CompiledAdmissionControllerStudy:
    return decode_canonical_bytes(
        payload,
        CompiledAdmissionControllerStudy,
        maximum_bytes=MAX_COMPILED_CONTROLLER_BYTES,
    )


__all__ = [
    'AtlasAdmissionDeriverPort',
    'ReceiptAdmissionDeriverPort',
    "CandidateEligibility",
    "CandidateMemberUtility",
    'AtlasCandidateAudit',
    'AdmissionCandidateAudit',
    'AdmissionCandidateConstituentAudit',
    "CompiledCellAction",
    'CompiledAtlasControllerStudy',
    'CompiledAdmissionControllerStudy',
    "CompiledDisposition",
    'CompiledAtlasHoldFibre',
    'CompiledAdmissionHoldFibre',
    "CompilerFailure",
    "CompilerPass",
    "CompilerPassProduct",
    "ControllerCompilationError",
    "ControllerCompiler",
    "HoldFibreDisposition",
    'ProspectiveBindingDisposition',
    'ProspectiveCompilationBinding',
    'AtlasReachabilityDeriverPort',
    'ReceiptReachabilityDeriverPort',
    'decode_compiled_atlas_controller_study',
    'decode_compiled_admission_controller_study',
]
