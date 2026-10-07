"""Pure, outcome-access-bounded selection between two preissued branch designs.

The selector consumes compact readiness and excluded-feasibility dispositions.
It does not read source or outcome payloads, issue either parent, create
authority, or permit reselection after parent issue.  Its receipt contains
enough frozen information to reconstruct the deterministic decision.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_schema,
    validate_sha256,
    validate_stable_id,
)

_PARTIAL_APPLICATION_EDGE_HANDOFF_RECEIPT_SCHEMA = (
    'empirical-lawhood/planning/partial-application-edge-handoff-receipt'
)


_BRANCH_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_REASON_CODE = re.compile(r"^[A-Z][A-Z0-9_]*$")

TOKAMAK_PRIMARY_BRANCH_ID = 'TOKAMAK-ORDINARY-CONTROLLED-RESPONSE'
TOKAMAK_FINITE_ACTION_RECURRENCE_BRANCH_ID = 'TOKAMAK-FINITE-ACTION-RECURRENCE'
TOKAMAK_PRIMARY_SELECTED_TRANSITION_CODE = "PRIMARY_CONTROLLED_IO_SELECTED"
TOKAMAK_FINITE_ACTION_RECURRENCE_SELECTED_TRANSITION_CODE = "FINITE_ACTION_RECURRENCE_SELECTED"
TOKAMAK_FINITE_ACTION_RECURRENCE_CONTRACTS_UNAVAILABLE_REASON_CODE = "FINITE_ACTION_RECURRENCE_GENERIC_CONTRACTS_UNAVAILABLE"
TOKAMAK_FINITE_ACTION_RECURRENCE_REASON_CODE_PRECEDENCE = (
    "CONTROLLED_IO_OPERATOR_API_UNAVAILABLE",
    "CONTROLLED_IO_DERIVATIVE_SURFACE_UNAVAILABLE",
    "CONTROLLED_IO_STATE_OR_OBSERVATION_REPRESENTATION_UNAVAILABLE",
    "CONTROLLED_IO_REPRESENTATION_BOUND_EXCEEDED",
    "CONTROLLED_IO_EXCLUDED_NUMERICAL_FEASIBILITY_NOT_SUPPORTED",
    "CONTROLLED_IO_EXCLUDED_NUMERICAL_FEASIBILITY_UNEVALUABLE",
)


def _validate_branch_label(value: str, *, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value.encode("utf-8")) > 128
        or _BRANCH_LABEL.fullmatch(value) is None
    ):
        raise ValueError(f"{field_name} must be a bounded canonical branch label")


def _validate_reason_code(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or _REASON_CODE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a canonical UPPER_SNAKE_CASE code")


def _validate_reason_codes(
    values: tuple[str, ...],
    *,
    field_name: str,
    allow_empty: bool,
) -> None:
    require_sorted_unique_strings(values, field_name=field_name, allow_empty=allow_empty)
    for value in values:
        _validate_reason_code(value, field_name=field_name)


class PreissueBranchRole(StrEnum):
    PRIMARY = "PRIMARY"
    ALTERNATE = "ALTERNATE"


class PreissueReadinessDisposition(StrEnum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    UNEVALUABLE = "UNEVALUABLE"


class PreissueOperatorFeasibilityDisposition(StrEnum):
    FEASIBLE = "FEASIBLE"
    OBSTRUCTED = "OBSTRUCTED"


class PreissueContractComponentKind(StrEnum):
    SCHEMA = "SCHEMA"
    CODEC = "CODEC"
    IMPLEMENTATION = "IMPLEMENTATION"
    FOCUSED_CONFORMANCE = "FOCUSED_CONFORMANCE"


class PreissueAlternateContractClosureDisposition(StrEnum):
    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"
    UNFINGERPRINTED = "UNFINGERPRINTED"


class PreissueBranchSelectionDisposition(StrEnum):
    PRIMARY_SELECTED = "PRIMARY_SELECTED"
    ALTERNATE_SELECTED = "ALTERNATE_SELECTED"
    ALTERNATE_CONTRACTS_UNAVAILABLE = "ALTERNATE_CONTRACTS_UNAVAILABLE"
    NO_BRANCH_SELECTED = "NO_BRANCH_SELECTED"


@dataclass(frozen=True, slots=True)
class ProspectiveBranchParentTemplate(CanonicalRecord):
    """Pre-inventory branch root without a future package fingerprint."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/prospective-branch-parent-template'

    template_id: str
    branch_id: str
    role: PreissueBranchRole
    expected_parent_package_id: str
    expected_parent_package_schema: str
    config_subinventory_id: str
    requires_exact_inventory_design_input: bool
    issue_started: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    grants_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.template_id, field_name="template_id")
        _validate_branch_label(self.branch_id, field_name="branch_id")
        validate_stable_id(
            self.expected_parent_package_id,
            field_name="expected_parent_package_id",
        )
        validate_schema(self.expected_parent_package_schema)
        validate_stable_id(
            self.config_subinventory_id,
            field_name="config_subinventory_id",
        )
        if self.template_id != f"branch-parent-template.{self.branch_id.lower()}":
            raise ValueError("prospective branch-parent template ID derivation changed")
        if self.expected_parent_package_schema != (
            'empirical-lawhood/planning/study-definition'
        ):
            raise ValueError("prospective branch parent requires the study-definition package schema")
        if not self.requires_exact_inventory_design_input:
            raise ValueError("prospective branch parent must require exact inventory custody")
        if self.issue_started:
            raise ValueError("prospective branch parent cannot exist after issue starts")
        if (
            self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("prospective branch parent must remain blind/prospective")
        if self.grants_authority:
            raise ValueError("prospective branch parent cannot grant authority")


@dataclass(frozen=True, slots=True)
class PreissueBranchDefinition(CanonicalRecord):
    """One frozen branch inventory and its not-yet-issued parent identity."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/preissue-branch-definition'

    branch_id: str
    role: PreissueBranchRole
    config_inventory_sha256: str
    parent: ObjectIdentity

    def __post_init__(self) -> None:
        _validate_branch_label(self.branch_id, field_name="branch_id")
        validate_sha256(self.config_inventory_sha256, field_name="config_inventory_sha256")
        if self.parent.object_schema != ProspectiveBranchParentTemplate.SCHEMA:
            raise ValueError("preissue branch parent must be a prospective template")


@dataclass(frozen=True, slots=True)
class PreissueContractFingerprint(CanonicalRecord):
    """One predeclared schema, codec, implementation, or conformance digest."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/preissue-contract-fingerprint'

    component_id: str
    kind: PreissueContractComponentKind
    component_sha256: str

    def __post_init__(self) -> None:
        validate_stable_id(self.component_id, field_name="component_id")
        validate_sha256(self.component_sha256, field_name="component_sha256")


@dataclass(frozen=True, slots=True)
class PreissueReadinessPredicateRequirement(CanonicalRecord):
    """Predeclared identity and schema for one late-bound readiness receipt."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/preissue-readiness-predicate-requirement'

    predicate_id: str
    expected_receipt_id: str
    expected_receipt_schema: str

    def __post_init__(self) -> None:
        validate_stable_id(self.predicate_id, field_name="predicate_id")
        validate_stable_id(
            self.expected_receipt_id,
            field_name="expected_receipt_id",
        )
        validate_schema(self.expected_receipt_schema)


@dataclass(frozen=True, slots=True)
class PreissueReadinessPredicateReceipt(CanonicalRecord):
    """Compact outcome-blind result for one ordinary readiness predicate."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/preissue-readiness-predicate-receipt'

    predicate_id: str
    receipt: ObjectIdentity
    disposition: PreissueReadinessDisposition
    reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess = OutcomeAccess.OUTCOME_BLIND

    def __post_init__(self) -> None:
        validate_stable_id(self.predicate_id, field_name="predicate_id")
        passed = self.disposition is PreissueReadinessDisposition.PASSED
        _validate_reason_codes(
            self.reason_codes,
            field_name="reason_codes",
            allow_empty=passed,
        )
        if passed and self.reason_codes:
            raise ValueError("a passed readiness predicate cannot contain stop reasons")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("an ordinary readiness predicate cannot read outcomes")


@dataclass(frozen=True, slots=True)
class PreissueOperatorFeasibility(CanonicalRecord):
    """Excluded operator-feasibility metadata, never final member evidence."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/preissue-operator-feasibility'

    assessment_id: str
    specification: ObjectIdentity
    receipt: ObjectIdentity
    evaluator_implementation: ObjectIdentity
    disposition: PreissueOperatorFeasibilityDisposition
    reason_codes: tuple[str, ...]
    evidence_identities: tuple[ObjectIdentity, ...]
    outcome_access: OutcomeAccess = OutcomeAccess.OUTCOME_BLIND
    raw_outcome_values_present: bool = False
    measurement_through_law_qualification_outcome_used: bool = False
    final_member_qualification_used: bool = False

    def __post_init__(self) -> None:
        validate_stable_id(self.assessment_id, field_name="assessment_id")
        feasible = self.disposition is PreissueOperatorFeasibilityDisposition.FEASIBLE
        _validate_reason_codes(
            self.reason_codes,
            field_name="reason_codes",
            allow_empty=feasible,
        )
        if feasible and self.reason_codes:
            raise ValueError("a feasible operator assessment cannot contain obstruction codes")
        require_sorted_unique_ids(
            self.evidence_identities,
            attribute="object_id",
            field_name="evidence_identities",
        )
        if not self.evidence_identities:
            raise ValueError("operator feasibility requires exact evidence identities")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("branch selection accepts only outcome-blind feasibility metadata")
        if self.raw_outcome_values_present:
            raise ValueError("operator-feasibility metadata cannot contain raw outcome values")
        if self.measurement_through_law_qualification_outcome_used:
            raise ValueError("operator-feasibility metadata cannot use response qualification outcomes")
        if self.final_member_qualification_used:
            raise ValueError("excluded feasibility cannot use final member qualification")


@dataclass(frozen=True, slots=True)
class PreissueAlternateContractClosure(CanonicalRecord):
    """Observed alternate-branch fingerprints and their compact closure evidence."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/preissue-alternate-contract-closure'

    closure_id: str
    disposition: PreissueAlternateContractClosureDisposition
    fingerprints: tuple[PreissueContractFingerprint, ...]
    evidence_identities: tuple[ObjectIdentity, ...]
    reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess = OutcomeAccess.OUTCOME_BLIND

    def __post_init__(self) -> None:
        validate_stable_id(self.closure_id, field_name="closure_id")
        require_sorted_unique_ids(
            self.fingerprints,
            attribute="component_id",
            field_name="fingerprints",
        )
        require_sorted_unique_ids(
            self.evidence_identities,
            attribute="object_id",
            field_name="evidence_identities",
        )
        complete = self.disposition is PreissueAlternateContractClosureDisposition.COMPLETE
        _validate_reason_codes(
            self.reason_codes,
            field_name="reason_codes",
            allow_empty=complete,
        )
        if complete:
            kinds = frozenset(value.kind for value in self.fingerprints)
            if kinds != frozenset(PreissueContractComponentKind):
                raise ValueError("complete alternate closure requires all fingerprint kinds")
            if not self.evidence_identities:
                raise ValueError("complete alternate closure requires exact evidence identities")
            if self.reason_codes:
                raise ValueError("complete alternate closure cannot contain stop reasons")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("alternate-contract closure cannot read outcomes")


@dataclass(frozen=True, slots=True)
class PreissueBranchSelectionSpec(CanonicalRecord):
    """Frozen two-branch selector and its closed alternate-obstruction codebook."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/preissue-branch-selection-spec'

    selector_id: str
    branches: tuple[PreissueBranchDefinition, ...]
    operator_feasibility_specification: ObjectIdentity
    operator_feasibility_receipt_schema: str
    operator_feasibility_implementation: ObjectIdentity
    ordinary_readiness_requirements: tuple[PreissueReadinessPredicateRequirement, ...]
    alternate_contract_fingerprints: tuple[PreissueContractFingerprint, ...]
    alternate_reason_code_precedence: tuple[str, ...]
    primary_selected_transition_code: str
    alternate_selected_transition_code: str
    alternate_contracts_unavailable_reason_code: str
    partial_application_edge_handoff: ObjectIdentity
    protected_source_extraction_manifest: ObjectIdentity
    outcome_access: OutcomeAccess = OutcomeAccess.OUTCOME_BLIND
    frozen_before_excluded_qualification: bool = True
    allows_postissue_reselection: bool = False
    grants_authority: bool = False

    def __post_init__(self) -> None:
        validate_stable_id(self.selector_id, field_name="selector_id")
        require_sorted_unique_ids(self.branches, attribute="branch_id", field_name="branches")
        if len(self.branches) != 2 or {value.role for value in self.branches} != {
            PreissueBranchRole.PRIMARY,
            PreissueBranchRole.ALTERNATE,
        }:
            raise ValueError("a preissue selector requires one primary and one alternate branch")
        if len({value.config_inventory_sha256 for value in self.branches}) != 2:
            raise ValueError("primary and alternate branches require distinct config inventories")
        if len({value.parent.object_id for value in self.branches}) != 2:
            raise ValueError("primary and alternate branches require distinct parent identities")
        validate_schema(self.operator_feasibility_receipt_schema)
        require_sorted_unique_ids(
            self.ordinary_readiness_requirements,
            attribute="predicate_id",
            field_name="ordinary_readiness_requirements",
        )
        if not self.ordinary_readiness_requirements:
            raise ValueError("ordinary readiness requirements must not be empty")
        expected_receipt_ids = tuple(
            value.expected_receipt_id for value in self.ordinary_readiness_requirements
        )
        if len(set(expected_receipt_ids)) != len(expected_receipt_ids):
            raise ValueError("ordinary readiness requires distinct receipt identities")
        require_sorted_unique_ids(
            self.alternate_contract_fingerprints,
            attribute="component_id",
            field_name="alternate_contract_fingerprints",
        )
        kinds = frozenset(value.kind for value in self.alternate_contract_fingerprints)
        if kinds != frozenset(PreissueContractComponentKind):
            raise ValueError("alternate branch requires every contract fingerprint kind")
        if not self.alternate_reason_code_precedence:
            raise ValueError("alternate reason-code precedence must not be empty")
        if len(set(self.alternate_reason_code_precedence)) != len(
            self.alternate_reason_code_precedence
        ):
            raise ValueError("alternate reason-code precedence must be unique")
        for value in self.alternate_reason_code_precedence:
            _validate_reason_code(value, field_name="alternate_reason_code_precedence")
        transition_codes = (
            self.primary_selected_transition_code,
            self.alternate_selected_transition_code,
            self.alternate_contracts_unavailable_reason_code,
        )
        for value in transition_codes:
            _validate_reason_code(value, field_name="transition_code")
        if len(set(transition_codes)) != len(transition_codes):
            raise ValueError("branch transition and stop codes must be distinct")
        if self.alternate_contracts_unavailable_reason_code in set(
            self.alternate_reason_code_precedence
        ):
            raise ValueError("alternate-contract stop cannot be an operator obstruction")
        if (
            self.partial_application_edge_handoff.object_schema
            != _PARTIAL_APPLICATION_EDGE_HANDOFF_RECEIPT_SCHEMA
        ):
            raise ValueError("selector requires the exact partial application-edge handoff schema")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("a preissue branch-selection spec must be outcome-blind")
        if not self.frozen_before_excluded_qualification:
            raise ValueError("branch-selection spec must freeze before excluded qualification")
        if self.allows_postissue_reselection:
            raise ValueError("a frozen selector cannot permit postissue reselection")
        if self.grants_authority:
            raise ValueError("a branch-selection spec cannot grant authority")

    def branch(self, role: PreissueBranchRole) -> PreissueBranchDefinition:
        return next(value for value in self.branches if value.role is role)

    @property
    def ordinary_readiness_predicate_ids(self) -> tuple[str, ...]:
        """Compatibility projection of the stronger typed requirement roster."""

        return tuple(value.predicate_id for value in self.ordinary_readiness_requirements)


@dataclass(frozen=True, slots=True)
class _SelectionDecision:
    disposition: PreissueBranchSelectionDisposition
    transition_code: str | None
    selected_branch_id: str | None
    unselected_branch_ids: tuple[str, ...]
    selected_parent: ObjectIdentity | None
    unselected_parents: tuple[ObjectIdentity, ...]
    controlling_reason_code: str | None
    stop_reason_codes: tuple[str, ...]


def _unselected(
    spec: PreissueBranchSelectionSpec,
    selected: PreissueBranchDefinition | None,
) -> tuple[tuple[str, ...], tuple[ObjectIdentity, ...]]:
    branches = tuple(value for value in spec.branches if value is not selected)
    return (
        tuple(sorted(value.branch_id for value in branches)),
        tuple(sorted((value.parent for value in branches), key=lambda value: value.object_id)),
    )


def _selected_decision(
    *,
    spec: PreissueBranchSelectionSpec,
    role: PreissueBranchRole,
    transition_code: str,
    controlling_reason_code: str | None,
    stop_reason_codes: tuple[str, ...],
) -> _SelectionDecision:
    selected = spec.branch(role)
    unselected_branch_ids, unselected_parents = _unselected(spec, selected)
    return _SelectionDecision(
        disposition=(
            PreissueBranchSelectionDisposition.PRIMARY_SELECTED
            if role is PreissueBranchRole.PRIMARY
            else PreissueBranchSelectionDisposition.ALTERNATE_SELECTED
        ),
        transition_code=transition_code,
        selected_branch_id=selected.branch_id,
        unselected_branch_ids=unselected_branch_ids,
        selected_parent=selected.parent,
        unselected_parents=unselected_parents,
        controlling_reason_code=controlling_reason_code,
        stop_reason_codes=stop_reason_codes,
    )


def _no_branch_decision(
    *,
    spec: PreissueBranchSelectionSpec,
    disposition: PreissueBranchSelectionDisposition,
    transition_code: str | None,
    controlling_reason_code: str,
    stop_reason_codes: tuple[str, ...],
) -> _SelectionDecision:
    unselected_branch_ids, unselected_parents = _unselected(spec, None)
    return _SelectionDecision(
        disposition=disposition,
        transition_code=transition_code,
        selected_branch_id=None,
        unselected_branch_ids=unselected_branch_ids,
        selected_parent=None,
        unselected_parents=unselected_parents,
        controlling_reason_code=controlling_reason_code,
        stop_reason_codes=stop_reason_codes,
    )


def _evaluate_selection(
    *,
    spec: PreissueBranchSelectionSpec,
    operator_feasibility: PreissueOperatorFeasibility,
    ordinary_readiness: tuple[PreissueReadinessPredicateReceipt, ...],
    alternate_contract_closure: PreissueAlternateContractClosure,
) -> _SelectionDecision:
    require_sorted_unique_ids(
        ordinary_readiness,
        attribute="predicate_id",
        field_name="ordinary_readiness",
    )
    if tuple(value.predicate_id for value in ordinary_readiness) != (
        spec.ordinary_readiness_predicate_ids
    ):
        raise ValueError("ordinary readiness receipts differ from the frozen predicate roster")
    requirements = {value.predicate_id: value for value in spec.ordinary_readiness_requirements}
    for receipt in ordinary_readiness:
        requirement = requirements[receipt.predicate_id]
        if (
            receipt.receipt.object_id != requirement.expected_receipt_id
            or receipt.receipt.object_schema != requirement.expected_receipt_schema
        ):
            raise ValueError(
                "ordinary readiness receipt identity/schema differs from its frozen requirement"
            )
    if operator_feasibility.specification != spec.operator_feasibility_specification:
        raise ValueError("operator feasibility uses another frozen specification")
    if operator_feasibility.receipt.object_schema != spec.operator_feasibility_receipt_schema:
        raise ValueError("operator feasibility uses another receipt schema")
    if operator_feasibility.evaluator_implementation != spec.operator_feasibility_implementation:
        raise ValueError("operator feasibility uses another evaluator implementation")

    failed = tuple(
        value
        for value in ordinary_readiness
        if value.disposition is not PreissueReadinessDisposition.PASSED
    )
    if failed:
        stop_reason_codes = tuple(
            sorted({reason for value in failed for reason in value.reason_codes})
        )
        return _no_branch_decision(
            spec=spec,
            disposition=PreissueBranchSelectionDisposition.NO_BRANCH_SELECTED,
            transition_code=None,
            controlling_reason_code=failed[0].reason_codes[0],
            stop_reason_codes=stop_reason_codes,
        )

    if operator_feasibility.disposition is PreissueOperatorFeasibilityDisposition.FEASIBLE:
        return _selected_decision(
            spec=spec,
            role=PreissueBranchRole.PRIMARY,
            transition_code=spec.primary_selected_transition_code,
            controlling_reason_code=None,
            stop_reason_codes=(),
        )

    obstruction_codes = frozenset(operator_feasibility.reason_codes)
    allowed_codes = frozenset(spec.alternate_reason_code_precedence)
    unlisted_codes = tuple(sorted(obstruction_codes - allowed_codes))
    if unlisted_codes:
        return _no_branch_decision(
            spec=spec,
            disposition=PreissueBranchSelectionDisposition.NO_BRANCH_SELECTED,
            transition_code=None,
            controlling_reason_code=unlisted_codes[0],
            stop_reason_codes=operator_feasibility.reason_codes,
        )

    controlling_code = next(
        value for value in spec.alternate_reason_code_precedence if value in obstruction_codes
    )
    alternate_complete = (
        alternate_contract_closure.disposition
        is PreissueAlternateContractClosureDisposition.COMPLETE
        and alternate_contract_closure.fingerprints == spec.alternate_contract_fingerprints
    )
    if alternate_complete:
        return _selected_decision(
            spec=spec,
            role=PreissueBranchRole.ALTERNATE,
            transition_code=spec.alternate_selected_transition_code,
            controlling_reason_code=controlling_code,
            stop_reason_codes=operator_feasibility.reason_codes,
        )

    stop_reason_codes = tuple(
        sorted(
            {
                *operator_feasibility.reason_codes,
                *alternate_contract_closure.reason_codes,
                spec.alternate_contracts_unavailable_reason_code,
            }
        )
    )
    return _no_branch_decision(
        spec=spec,
        disposition=PreissueBranchSelectionDisposition.ALTERNATE_CONTRACTS_UNAVAILABLE,
        transition_code=spec.alternate_contracts_unavailable_reason_code,
        controlling_reason_code=controlling_code,
        stop_reason_codes=stop_reason_codes,
    )


@dataclass(frozen=True, slots=True)
class PreissueBranchSelectionReceipt(CanonicalRecord):
    """Sealed deterministic transition metadata; it does not issue a parent."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/preissue-branch-selection-receipt'

    selection_receipt_id: str
    spec: PreissueBranchSelectionSpec
    operator_feasibility: PreissueOperatorFeasibility
    ordinary_readiness: tuple[PreissueReadinessPredicateReceipt, ...]
    alternate_contract_closure: PreissueAlternateContractClosure
    disposition: PreissueBranchSelectionDisposition
    transition_code: str | None
    selected_branch_id: str | None
    unselected_branch_ids: tuple[str, ...]
    selected_parent: ObjectIdentity | None
    unselected_parents: tuple[ObjectIdentity, ...]
    controlling_reason_code: str | None
    stop_reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess = OutcomeAccess.OUTCOME_BLIND
    sealed_before_parent_issue: bool = True
    parent_issue_started: bool = False
    measurement_through_law_qualification_outcome_used: bool = False
    final_member_qualification_used: bool = False
    prospective_outcome_used: bool = False
    issues_parent: bool = False
    grants_authority: bool = False

    def __post_init__(self) -> None:
        validate_stable_id(self.selection_receipt_id, field_name="selection_receipt_id")
        _validate_reason_codes(
            self.stop_reason_codes,
            field_name="stop_reason_codes",
            allow_empty=True,
        )
        if self.transition_code is not None:
            _validate_reason_code(self.transition_code, field_name="transition_code")
        if self.controlling_reason_code is not None:
            _validate_reason_code(
                self.controlling_reason_code,
                field_name="controlling_reason_code",
            )
        expected = _evaluate_selection(
            spec=self.spec,
            operator_feasibility=self.operator_feasibility,
            ordinary_readiness=self.ordinary_readiness,
            alternate_contract_closure=self.alternate_contract_closure,
        )
        observed = _SelectionDecision(
            disposition=self.disposition,
            transition_code=self.transition_code,
            selected_branch_id=self.selected_branch_id,
            unselected_branch_ids=self.unselected_branch_ids,
            selected_parent=self.selected_parent,
            unselected_parents=self.unselected_parents,
            controlling_reason_code=self.controlling_reason_code,
            stop_reason_codes=self.stop_reason_codes,
        )
        if observed != expected:
            raise ValueError("branch-selection receipt differs from the frozen selector")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("branch-selection receipt cannot read outcomes")
        if not self.sealed_before_parent_issue or self.parent_issue_started:
            raise ValueError("branch selection must seal before either parent issues")
        if self.measurement_through_law_qualification_outcome_used:
            raise ValueError("branch selection cannot use response qualification outcomes")
        if self.final_member_qualification_used:
            raise ValueError("branch selection cannot use final member qualification")
        if self.prospective_outcome_used:
            raise ValueError("branch selection cannot use prospective outcomes")
        if self.issues_parent:
            raise ValueError("branch selection cannot issue a parent")
        if self.grants_authority:
            raise ValueError("branch selection cannot grant authority")


def select_preissue_branch(
    *,
    selection_receipt_id: str,
    spec: PreissueBranchSelectionSpec,
    operator_feasibility: PreissueOperatorFeasibility,
    ordinary_readiness: tuple[PreissueReadinessPredicateReceipt, ...],
    alternate_contract_closure: PreissueAlternateContractClosure,
    parent_issue_started: bool,
) -> PreissueBranchSelectionReceipt:
    """Apply the frozen selector without I/O, issue, authority, or outcome access."""

    if parent_issue_started:
        raise ValueError("a controlled-I/O feasibility obstruction discovered after parent issue cannot reselect a branch")
    decision = _evaluate_selection(
        spec=spec,
        operator_feasibility=operator_feasibility,
        ordinary_readiness=ordinary_readiness,
        alternate_contract_closure=alternate_contract_closure,
    )
    return PreissueBranchSelectionReceipt(
        selection_receipt_id=selection_receipt_id,
        spec=spec,
        operator_feasibility=operator_feasibility,
        ordinary_readiness=ordinary_readiness,
        alternate_contract_closure=alternate_contract_closure,
        disposition=decision.disposition,
        transition_code=decision.transition_code,
        selected_branch_id=decision.selected_branch_id,
        unselected_branch_ids=decision.unselected_branch_ids,
        selected_parent=decision.selected_parent,
        unselected_parents=decision.unselected_parents,
        controlling_reason_code=decision.controlling_reason_code,
        stop_reason_codes=decision.stop_reason_codes,
    )


def _validate_gym_torax_spec(spec: PreissueBranchSelectionSpec) -> None:
    branch_ids = {value.role: value.branch_id for value in spec.branches}
    if branch_ids != {
        PreissueBranchRole.PRIMARY: TOKAMAK_PRIMARY_BRANCH_ID,
        PreissueBranchRole.ALTERNATE: TOKAMAK_FINITE_ACTION_RECURRENCE_BRANCH_ID,
    }:
        raise ValueError("tokamak-control replication selector requires the exact primary and finite-action recurrence branch identities")
    if spec.alternate_reason_code_precedence != TOKAMAK_FINITE_ACTION_RECURRENCE_REASON_CODE_PRECEDENCE:
        raise ValueError("tokamak-control replication selector requires the exact closed controlled-I/O feasibility reason-code precedence")
    if spec.primary_selected_transition_code != TOKAMAK_PRIMARY_SELECTED_TRANSITION_CODE:
        raise ValueError("tokamak-control replication selector requires the exact primary transition code")
    if spec.alternate_selected_transition_code != TOKAMAK_FINITE_ACTION_RECURRENCE_SELECTED_TRANSITION_CODE:
        raise ValueError("tokamak-control replication selector requires the exact finite-action recurrence transition code")
    if (
        spec.alternate_contracts_unavailable_reason_code
        != TOKAMAK_FINITE_ACTION_RECURRENCE_CONTRACTS_UNAVAILABLE_REASON_CODE
    ):
        raise ValueError("tokamak-control replication selector requires the exact finite-action recurrence contract stop code")


def select_gym_torax_preissue_branch(
    *,
    selection_receipt_id: str,
    spec: PreissueBranchSelectionSpec,
    operator_feasibility: PreissueOperatorFeasibility,
    ordinary_readiness: tuple[PreissueReadinessPredicateReceipt, ...],
    alternate_contract_closure: PreissueAlternateContractClosure,
    parent_issue_started: bool,
) -> PreissueBranchSelectionReceipt:
    "Apply the generic selector after enforcing tokamak-control replication's exact frozen codebook."

    _validate_gym_torax_spec(spec)
    return select_preissue_branch(
        selection_receipt_id=selection_receipt_id,
        spec=spec,
        operator_feasibility=operator_feasibility,
        ordinary_readiness=ordinary_readiness,
        alternate_contract_closure=alternate_contract_closure,
        parent_issue_started=parent_issue_started,
    )


# Plan-facing aliases retain a reusable schema and implementation.


__all__ = [
    'TOKAMAK_FINITE_ACTION_RECURRENCE_BRANCH_ID',
    'TOKAMAK_FINITE_ACTION_RECURRENCE_CONTRACTS_UNAVAILABLE_REASON_CODE',
    'TOKAMAK_FINITE_ACTION_RECURRENCE_REASON_CODE_PRECEDENCE',
    'TOKAMAK_FINITE_ACTION_RECURRENCE_SELECTED_TRANSITION_CODE',
    'TOKAMAK_PRIMARY_BRANCH_ID',
    'TOKAMAK_PRIMARY_SELECTED_TRANSITION_CODE',
    'PreissueAlternateContractClosureDisposition',
    'PreissueAlternateContractClosure',
    'PreissueBranchDefinition',
    'PreissueBranchRole',
    'PreissueBranchSelectionDisposition',
    'PreissueBranchSelectionReceipt',
    'PreissueBranchSelectionSpec',
    'PreissueContractComponentKind',
    'PreissueContractFingerprint',
    'PreissueOperatorFeasibilityDisposition',
    'PreissueOperatorFeasibility',
    'PreissueReadinessDisposition',
    'PreissueReadinessPredicateRequirement',
    'PreissueReadinessPredicateReceipt',
    'ProspectiveBranchParentTemplate',
    'select_gym_torax_preissue_branch',
    'select_preissue_branch',
]
