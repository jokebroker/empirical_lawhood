"""Method-independent law-identification, adequacy and uncertainty results."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from .evidence import EvidenceRung, OutcomeAccess, VisibilityCeiling, inherited_visibility
from .laws import LawRepresentationKind, ResponseLaw
from .obligations import ObligationStatus
from .provenance import Claim, EvidenceLink, ObjectIdentity
from .references import ExecutableReference, NamedDecimal
from .serialization import (
    CanonicalRecord,
    ExtensionBinding,
    require_extensions,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_semantic_version,
    validate_stable_id,
)
from .status import ScientificStatus
from .systems import RelationalIdentity


class LawMethodKind(StrEnum):
    LOCAL_LINEAR = "LOCAL_LINEAR"
    LOCAL_STATE_SPACE = "LOCAL_STATE_SPACE"
    NONLINEAR_LOCAL = "NONLINEAR_LOCAL"


class AdequacyCheckKind(StrEnum):
    COORDINATE_ADEQUACY = "COORDINATE_ADEQUACY"
    WITHIN_CELL_RECURRENCE = "WITHIN_CELL_RECURRENCE"
    ONE_FACTOR_EXCHANGE = "ONE_FACTOR_EXCHANGE"
    SUPPORT = "SUPPORT"
    CLOSURE_MEMORY = "CLOSURE_MEMORY"
    HELD_OUT_CALIBRATION = "HELD_OUT_CALIBRATION"
    STRUCTURAL_CONVERGENCE = "STRUCTURAL_CONVERGENCE"
    DECISIVE_FALSIFIER = "DECISIVE_FALSIFIER"
    COMPUTABILITY = "COMPUTABILITY"
    EVIDENCE_VISIBILITY = "EVIDENCE_VISIBILITY"


class UncertaintyClass(StrEnum):
    NUMERICAL = "NUMERICAL"
    ALEATORIC = "ALEATORIC"
    EPISTEMIC = "EPISTEMIC"
    TRANSPORT = "TRANSPORT"
    OBSERVATION = "OBSERVATION"


class QualificationOperationalDisposition(StrEnum):
    "Operational axis for a valid terminal qualification record.\n\n    Contract-invalid and operationally blocked attempts do not construct a\n    :class:`LawQualificationResult`; the service reports those through its\n    typed failure boundary.  A constructed terminal result is therefore a\n    scientifically interpretable, operationally complete assessment.\n    "

    COMPLETE = "COMPLETE"


class TerminalObligationKind(StrEnum):
    SUPPORT = "SUPPORT"
    VALIDITY = "VALIDITY"
    UNCERTAINTY = "UNCERTAINTY"
    FALSIFIER = "FALSIFIER"
    CLOSURE = "CLOSURE"
    STRUCTURAL_QUALIFICATION = "STRUCTURAL_QUALIFICATION"
    COMPUTABILITY = "COMPUTABILITY"
    EVIDENCE_VISIBILITY = "EVIDENCE_VISIBILITY"


_REQUIRED_TERMINAL_OBLIGATIONS = frozenset(TerminalObligationKind)


@dataclass(frozen=True, slots=True)
class TerminalObligationDisposition(CanonicalRecord):
    """One final obligation disposition, present even when no law exists."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/terminal-obligation-disposition'

    disposition_id: str
    kind: TerminalObligationKind
    status: ObligationStatus
    proof_owner: ObjectIdentity
    evidence_link_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.disposition_id, field_name="disposition_id")
        require_sorted_unique_strings(
            self.evidence_link_ids,
            field_name="evidence_link_ids",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.status is ObligationStatus.SATISFIED:
            if not self.evidence_link_ids or self.reason_codes:
                raise ValueError("satisfied terminal obligation requires evidence and no reasons")
        elif self.status is ObligationStatus.NOT_APPLICABLE:
            if self.reason_codes:
                raise ValueError("not-applicable terminal obligation cannot carry reasons")
        elif not self.reason_codes:
            raise ValueError("unresolved terminal obligation requires reason codes")

    @property
    def passed(self) -> bool:
        return self.status in {
            ObligationStatus.SATISFIED,
            ObligationStatus.NOT_APPLICABLE,
        }


@dataclass(frozen=True, slots=True)
class TerminalLawObligationAssessment(CanonicalRecord):
    "Complete local-law obligation plane independent of supported-law emission."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/terminal-law-obligation-assessment'

    assessment_id: str
    dispositions: tuple[TerminalObligationDisposition, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.assessment_id, field_name="assessment_id")
        require_sorted_unique_ids(
            self.dispositions,
            attribute="disposition_id",
            field_name="dispositions",
        )
        observed = {value.kind for value in self.dispositions}
        if observed != _REQUIRED_TERMINAL_OBLIGATIONS:
            raise ValueError("terminal assessment requires every obligation family")
        if len(observed) != len(self.dispositions):
            raise ValueError("terminal obligation families cannot be duplicated")

    @property
    def claim_ready(self) -> bool:
        return all(value.passed for value in self.dispositions)

    def disposition(self, kind: TerminalObligationKind) -> TerminalObligationDisposition:
        return next(value for value in self.dispositions if value.kind is kind)


@dataclass(frozen=True, slots=True)
class ResponseQualificationFacetAssessment(CanonicalRecord):
    "One profile-derived facet in the acyclic measurement through local law proof graph."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/response-qualification-facet-assessment'

    facet_id: str
    rung: EvidenceRung
    status: ScientificStatus
    proof_owner: ObjectIdentity
    prerequisite_facet_ids: tuple[str, ...]
    evidence_link_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    maximum_supported_object_id: str | None
    required_for_public_rung: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.facet_id, field_name="facet_id")
        if self.rung not in {
            EvidenceRung.MEASUREMENT,
            EvidenceRung.ORDER_RELATION,
            EvidenceRung.RESPONSE,
            EvidenceRung.LOCAL_LAW,
        }:
            raise ValueError("qualification facets are restricted to measurement through local law")
        require_sorted_unique_strings(
            self.prerequisite_facet_ids,
            field_name="prerequisite_facet_ids",
        )
        if self.facet_id in self.prerequisite_facet_ids:
            raise ValueError("qualification facet cannot depend on itself")
        require_sorted_unique_strings(
            self.evidence_link_ids,
            field_name="evidence_link_ids",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.maximum_supported_object_id is not None:
            validate_stable_id(
                self.maximum_supported_object_id,
                field_name="maximum_supported_object_id",
            )
        if self.status is ScientificStatus.SUPPORTED:
            if not self.evidence_link_ids or self.reason_codes:
                raise ValueError("supported qualification facet requires evidence and no reasons")
            if self.maximum_supported_object_id is None:
                raise ValueError("supported qualification facet requires a supported object")
        else:
            if not self.reason_codes:
                raise ValueError("non-supported qualification facet requires reasons")
            if self.maximum_supported_object_id is not None:
                raise ValueError("non-supported qualification facet cannot claim an object")


_PRE_ADMISSION_EVIDENCE_RANK = {
    EvidenceRung.MEASUREMENT: 0,
    EvidenceRung.ORDER_RELATION: 1,
    EvidenceRung.RESPONSE: 2,
    EvidenceRung.LOCAL_LAW: 3,
}


@dataclass(frozen=True, slots=True)
class ResponseQualificationTrace(CanonicalRecord):
    """Finite proof graph retaining partial products and the public ceiling."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/response-qualification-trace'

    trace_id: str
    candidate_id: str
    facets: tuple[ResponseQualificationFacetAssessment, ...]
    physical_independent_unit_ids: tuple[str, ...]
    candidate_version_member_ids: tuple[str, ...]
    denominator_member_ids: tuple[str, ...]
    qualification_view_ids: tuple[str, ...]
    information_cutoff_id: str
    outcome_access: OutcomeAccess
    highest_supported_rung: EvidenceRung | None

    def __post_init__(self) -> None:
        for name, value in (
            ("trace_id", self.trace_id),
            ("candidate_id", self.candidate_id),
            ("information_cutoff_id", self.information_cutoff_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_ids(self.facets, attribute="facet_id", field_name="facets")
        if not self.facets:
            raise ValueError("qualification trace requires facets")
        for name, values in (
            ("physical_independent_unit_ids", self.physical_independent_unit_ids),
            ("candidate_version_member_ids", self.candidate_version_member_ids),
            ("denominator_member_ids", self.denominator_member_ids),
            ("qualification_view_ids", self.qualification_view_ids),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)
        if (
            self.highest_supported_rung is not None
            and self.highest_supported_rung not in _PRE_ADMISSION_EVIDENCE_RANK
        ):
            raise ValueError("highest supported qualification rung is outside measurement through local law")
        self._validate_graph()
        if self.highest_supported_rung is not self._derived_highest_rung():
            raise ValueError("qualification trace highest rung is not mechanically derived")

    def _validate_graph(self) -> None:
        facets = {value.facet_id: value for value in self.facets}
        for facet in self.facets:
            unknown = set(facet.prerequisite_facet_ids) - set(facets)
            if unknown:
                raise ValueError("qualification facet names an unknown prerequisite")
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(facet_id: str) -> None:
            if facet_id in visiting:
                raise ValueError("qualification facet graph contains a cycle")
            if facet_id in visited:
                return
            visiting.add(facet_id)
            for parent in facets[facet_id].prerequisite_facet_ids:
                visit(parent)
            visiting.remove(facet_id)
            visited.add(facet_id)

        for facet_id in facets:
            visit(facet_id)
        for facet in self.facets:
            if facet.status is ScientificStatus.SUPPORTED and any(
                facets[parent].status is not ScientificStatus.SUPPORTED
                for parent in facet.prerequisite_facet_ids
            ):
                raise ValueError("supported qualification facet has an unsupported prerequisite")

    def _derived_highest_rung(self) -> EvidenceRung | None:
        required = tuple(value for value in self.facets if value.required_for_public_rung)
        highest: EvidenceRung | None = None
        for rung in _PRE_ADMISSION_EVIDENCE_RANK:
            applicable = tuple(
                value for value in required if _PRE_ADMISSION_EVIDENCE_RANK[value.rung] <= _PRE_ADMISSION_EVIDENCE_RANK[rung]
            )
            if applicable and all(
                value.status is ScientificStatus.SUPPORTED for value in applicable
            ):
                highest = rung
            else:
                break
        return highest


@dataclass(frozen=True, slots=True)
class AdequacyCheckResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/adequacy-check-result'

    check_id: str
    kind: AdequacyCheckKind
    status: ObligationStatus
    decisive: bool
    metrics: tuple[NamedDecimal, ...]
    reason_codes: tuple[str, ...]
    evidence_link_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.check_id, field_name="check_id")
        require_sorted_unique_ids(self.metrics, attribute="value_id", field_name="metrics")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        require_sorted_unique_strings(
            self.evidence_link_ids,
            field_name="evidence_link_ids",
        )
        if self.status is ObligationStatus.SATISFIED:
            if self.reason_codes or not self.evidence_link_ids:
                raise ValueError("satisfied adequacy check requires evidence and no reasons")
        elif self.status is ObligationStatus.NOT_APPLICABLE:
            if self.reason_codes:
                raise ValueError("not-applicable adequacy check cannot carry failure reasons")
        elif not self.reason_codes:
            raise ValueError("non-passing adequacy check requires reason codes")

    @property
    def passed(self) -> bool:
        return self.status in {
            ObligationStatus.SATISFIED,
            ObligationStatus.NOT_APPLICABLE,
        }


@dataclass(frozen=True, slots=True)
class UncertaintyComponent(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/uncertainty-component'

    component_id: str
    uncertainty_class: UncertaintyClass
    status: ObligationStatus
    bounds: tuple[NamedDecimal, ...]
    limitation_codes: tuple[str, ...]
    evidence_link_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.component_id, field_name="component_id")
        require_sorted_unique_ids(self.bounds, attribute="value_id", field_name="bounds")
        require_sorted_unique_strings(
            self.limitation_codes,
            field_name="limitation_codes",
        )
        require_sorted_unique_strings(
            self.evidence_link_ids,
            field_name="evidence_link_ids",
        )
        if self.status is ObligationStatus.SATISFIED:
            if not self.bounds or not self.evidence_link_ids or self.limitation_codes:
                raise ValueError("satisfied uncertainty component requires bounded evidence")
        elif self.status is ObligationStatus.NOT_APPLICABLE:
            if self.bounds or self.limitation_codes:
                raise ValueError(
                    "not-applicable uncertainty component cannot carry bounds or limitations"
                )
        elif not self.limitation_codes:
            raise ValueError("unresolved uncertainty component requires limitations")


@dataclass(frozen=True, slots=True)
class UncertaintyDecomposition(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/uncertainty-decomposition'

    decomposition_id: str
    components: tuple[UncertaintyComponent, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.decomposition_id, field_name="decomposition_id")
        require_sorted_unique_ids(
            self.components,
            attribute="component_id",
            field_name="components",
        )
        observed = {component.uncertainty_class for component in self.components}
        if observed != set(UncertaintyClass):
            raise ValueError("uncertainty decomposition requires all five distinct classes")
        if len(observed) != len(self.components):
            raise ValueError("uncertainty classes cannot be duplicated")

    @property
    def claim_ready(self) -> bool:
        return all(
            component.status in {ObligationStatus.SATISFIED, ObligationStatus.NOT_APPLICABLE}
            for component in self.components
        )


@dataclass(frozen=True, slots=True)
class StructuralSignature(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/structural-signature'

    signature_id: str
    numerical_view_id: str
    response_rank: int
    response_direction_ids: tuple[str, ...]
    curved_term_ids: tuple[str, ...]
    retained_history_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.signature_id, field_name="signature_id")
        validate_stable_id(self.numerical_view_id, field_name="numerical_view_id")
        if self.response_rank < 0:
            raise ValueError("response rank must be nonnegative")
        for name, values in (
            ("response_direction_ids", self.response_direction_ids),
            ("curved_term_ids", self.curved_term_ids),
            ("retained_history_ids", self.retained_history_ids),
        ):
            require_sorted_unique_strings(values, field_name=name)


@dataclass(frozen=True, slots=True)
class StructuralConvergenceResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/structural-convergence-result'

    result_id: str
    system_id: str
    relation_id: str
    signatures: tuple[StructuralSignature, ...]
    stable_structure_ids: tuple[str, ...]
    unstable_structure_ids: tuple[str, ...]
    metrics: tuple[NamedDecimal, ...]
    status: ScientificStatus
    evidence_links: tuple[EvidenceLink, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("result_id", self.result_id),
            ("system_id", self.system_id),
            ("relation_id", self.relation_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_ids(
            self.signatures,
            attribute="signature_id",
            field_name="signatures",
        )
        if len(self.signatures) < 2:
            raise ValueError("structural convergence requires at least two numerical views")
        view_ids = {signature.numerical_view_id for signature in self.signatures}
        if len(view_ids) != len(self.signatures):
            raise ValueError("structural convergence requires one signature per view")
        require_sorted_unique_strings(
            self.stable_structure_ids,
            field_name="stable_structure_ids",
        )
        require_sorted_unique_strings(
            self.unstable_structure_ids,
            field_name="unstable_structure_ids",
        )
        if set(self.stable_structure_ids) & set(self.unstable_structure_ids):
            raise ValueError("structure cannot be both stable and unstable")
        require_sorted_unique_ids(self.metrics, attribute="value_id", field_name="metrics")
        require_sorted_unique_ids(
            self.evidence_links,
            attribute="link_id",
            field_name="evidence_links",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.status is ScientificStatus.SUPPORTED:
            if self.unstable_structure_ids or self.reason_codes or not self.evidence_links:
                raise ValueError("supported structural convergence cannot retain instability")
        elif not self.unstable_structure_ids or not self.reason_codes:
            raise ValueError("non-supported convergence requires unstable structures and reasons")


_REQUIRED_ADEQUACY_KINDS = frozenset(AdequacyCheckKind)


@dataclass(frozen=True, slots=True)
class LawIdentificationResult(CanonicalRecord):
    """Terminal candidate assessment; only a fully passing result may carry a law."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/law-identification-result'

    result_id: str
    dataset: ObjectIdentity
    config: ObjectIdentity
    system_id: str
    world_id: str
    relation: RelationalIdentity
    chart_id: str
    method_key: str
    method_version: str
    method_kind: LawMethodKind
    representation_kind: LawRepresentationKind
    candidate_evaluator: ExecutableReference
    checks: tuple[AdequacyCheckResult, ...]
    uncertainty: UncertaintyDecomposition
    convergence: StructuralConvergenceResult
    metrics: tuple[NamedDecimal, ...]
    claim: Claim
    response_law: ResponseLaw | None
    scientific_status: ScientificStatus
    reason_codes: tuple[str, ...]
    evidence_links: tuple[EvidenceLink, ...]
    outcome_access: OutcomeAccess
    parent_visibility_ceilings: tuple[VisibilityCeiling, ...]
    visibility_ceiling: VisibilityCeiling
    extensions: tuple[ExtensionBinding, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("result_id", self.result_id),
            ("system_id", self.system_id),
            ("world_id", self.world_id),
            ("chart_id", self.chart_id),
            ("method_key", self.method_key),
        ):
            validate_stable_id(value, field_name=name)
        validate_semantic_version(self.method_version)
        require_sorted_unique_ids(self.checks, attribute="check_id", field_name="checks")
        observed_kinds = {check.kind for check in self.checks}
        if observed_kinds != _REQUIRED_ADEQUACY_KINDS:
            raise ValueError("identification result lacks the complete adequacy family")
        require_sorted_unique_ids(self.metrics, attribute="value_id", field_name="metrics")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        require_sorted_unique_ids(
            self.evidence_links,
            attribute="link_id",
            field_name="evidence_links",
        )
        inherited = inherited_visibility(self.parent_visibility_ceilings, self.outcome_access)
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(inherited):
            raise ValueError("identification-result visibility cannot be lowered")
        self._validate_terminal_status()
        require_extensions(self.extensions)

    def _validate_terminal_status(self) -> None:
        all_checks_pass = all(check.passed for check in self.checks)
        claim_ready = (
            all_checks_pass
            and self.uncertainty.claim_ready
            and self.visibility_ceiling.is_promotable
        )
        convergence_passed = self.convergence.status is ScientificStatus.SUPPORTED
        if self.scientific_status is ScientificStatus.SUPPORTED:
            if not claim_ready or not convergence_passed:
                raise ValueError("supported identification bypasses an adequacy obligation")
            if self.response_law is None or self.reason_codes or not self.evidence_links:
                raise ValueError("supported identification requires a law and evidence")
            if self.response_law.claim != self.claim:
                raise ValueError("identified law and terminal claim differ")
            if self.response_law.evaluator != self.candidate_evaluator:
                raise ValueError("identified law differs from the assessed evaluator")
        else:
            if self.response_law is not None:
                raise ValueError("a non-supported candidate cannot emit a ResponseLaw")
            if not self.reason_codes:
                raise ValueError("a non-supported candidate requires terminal reasons")
            if claim_ready and convergence_passed:
                raise ValueError("a rejected candidate must retain a failed scientific gate")
        if self.claim.scientific_status is not self.scientific_status:
            raise ValueError("terminal identification and claim statuses differ")


@dataclass(frozen=True, slots=True)
class LawQualificationResult(CanonicalRecord):
    "Authoritative family-first measurement through local law terminal qualification result.\n\n    A valid result is always operationally complete.  Malformed inputs and\n    failures to construct a trustworthy evidence/evaluator projection are\n    refused by the qualification service before this record exists.\n    "

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/law-qualification-result'
    VERSION: ClassVar[str] = '1.0.0'

    result_id: str
    dataset_or_projection: ObjectIdentity
    config: ObjectIdentity
    candidate_family_assessment: ObjectIdentity
    selection_receipt: ObjectIdentity
    axis_map: ObjectIdentity
    claim_unit_binding: ObjectIdentity
    payload_publication: ObjectIdentity | None
    system_id: str
    world_id: str
    relation: RelationalIdentity
    chart_id: str
    method_key: str
    method_version: str
    method_kind: LawMethodKind
    representation_kind: LawRepresentationKind
    selected_candidate_id: str | None
    candidate_evaluator: ExecutableReference | None
    qualification_trace: ResponseQualificationTrace
    terminal_obligations: TerminalLawObligationAssessment
    metrics: tuple[NamedDecimal, ...]
    claim: Claim
    response_law: ResponseLaw | None
    operational_disposition: QualificationOperationalDisposition
    scientific_status: ScientificStatus
    highest_supported_rung: EvidenceRung | None
    reason_codes: tuple[str, ...]
    evidence_links: tuple[EvidenceLink, ...]
    outcome_access: OutcomeAccess
    parent_visibility_ceilings: tuple[VisibilityCeiling, ...]
    visibility_ceiling: VisibilityCeiling
    response_method_projection: LawIdentificationResult | None = None
    extensions: tuple[ExtensionBinding, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("result_id", self.result_id),
            ("system_id", self.system_id),
            ("world_id", self.world_id),
            ("chart_id", self.chart_id),
            ("method_key", self.method_key),
        ):
            validate_stable_id(value, field_name=name)
        validate_semantic_version(self.method_version)
        if self.selected_candidate_id is not None:
            validate_stable_id(self.selected_candidate_id, field_name="selected_candidate_id")
        require_sorted_unique_ids(self.metrics, attribute="value_id", field_name="metrics")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        require_sorted_unique_ids(
            self.evidence_links,
            attribute="link_id",
            field_name="evidence_links",
        )
        inherited = inherited_visibility(self.parent_visibility_ceilings, self.outcome_access)
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(inherited):
            raise ValueError("qualification-result visibility cannot be lowered")
        if self.operational_disposition is not QualificationOperationalDisposition.COMPLETE:
            raise ValueError("a terminal qualification result must be operationally complete")
        if self.qualification_trace.candidate_id != (
            self.selected_candidate_id or self.selection_receipt.object_id
        ):
            raise ValueError("qualification trace does not bind the selected/no-selection object")
        if self.highest_supported_rung is not self.qualification_trace.highest_supported_rung:
            raise ValueError("qualification result and trace highest rung differ")
        self._validate_terminal_status()
        self._validate_response_method_projection()
        require_extensions(self.extensions)

    def _validate_terminal_status(self) -> None:
        if self.scientific_status is ScientificStatus.SUPPORTED:
            if self.selected_candidate_id is None or self.candidate_evaluator is None:
                raise ValueError("supported qualification requires one selected candidate")
            if self.payload_publication is None:
                raise ValueError("supported qualification requires published evaluator payload")
            if self.highest_supported_rung is not EvidenceRung.LOCAL_LAW:
                raise ValueError("supported qualification requires a complete local-law trace")
            if not self.terminal_obligations.claim_ready:
                raise ValueError("supported qualification bypasses terminal obligations")
            if self.response_law is None or self.reason_codes or not self.evidence_links:
                raise ValueError("supported qualification requires a law and evidence")
            if self.response_law.claim != self.claim:
                raise ValueError("qualified law and terminal claim differ")
            if self.response_law.evaluator != self.candidate_evaluator:
                raise ValueError("qualified law differs from the selected evaluator")
        else:
            if self.response_law is not None:
                raise ValueError("a non-supported qualification cannot emit a ResponseLaw")
            if not self.reason_codes:
                raise ValueError("a non-supported qualification requires terminal reasons")
            if self.terminal_obligations.claim_ready and self.visibility_ceiling.is_promotable:
                raise ValueError("a rejected qualification must retain a failed gate")
            if self.selected_candidate_id is None:
                if self.candidate_evaluator is not None or self.payload_publication is not None:
                    raise ValueError("NO_SELECTION cannot bind an evaluator payload")
        if self.claim.scientific_status is not self.scientific_status:
            raise ValueError("qualification result and claim statuses differ")
        if self.scientific_status is ScientificStatus.SUPPORTED:
            if self.claim.observed_rung is not EvidenceRung.LOCAL_LAW:
                raise ValueError("supported qualification claim must observe local law")
        elif self.claim.observed_rung != self.highest_supported_rung:
            raise ValueError("non-supported qualification claim must retain its trace ceiling")

    def _validate_response_method_projection(self) -> None:
        projection = self.response_method_projection
        if projection is None:
            return
        if projection.dataset != self.dataset_or_projection or projection.config != self.config:
            raise ValueError("LawIdentificationResult compatibility projection binds different inputs")
        for name in (
            "system_id",
            "world_id",
            "relation",
            "chart_id",
            "method_key",
            "method_version",
            "method_kind",
            "representation_kind",
            "response_law",
            "scientific_status",
            "evidence_links",
            "outcome_access",
            "parent_visibility_ceilings",
            "visibility_ceiling",
        ):
            if getattr(projection, name) != getattr(self, name):
                raise ValueError(f'LawIdentificationResult compatibility projection differs at {name}')
        if (
            self.selected_candidate_id is not None
            and projection.candidate_evaluator != self.candidate_evaluator
        ):
            raise ValueError("LawIdentificationResult compatibility projection evaluator differs")
        if self.scientific_status is ScientificStatus.SUPPORTED and projection.claim != self.claim:
            raise ValueError("supported LawIdentificationResult compatibility projection claim differs")
        if projection.metrics != self.metrics:
            raise ValueError("LawIdentificationResult compatibility projection metrics differ")
        if self.selected_candidate_id is not None and projection.reason_codes != self.reason_codes:
            raise ValueError("LawIdentificationResult compatibility projection reasons differ")
