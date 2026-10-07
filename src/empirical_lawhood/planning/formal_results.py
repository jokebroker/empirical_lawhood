"""Typed inputs and outputs for standard reusable formal analysis."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_stable_id,
)

from .formal_gaps import FormalDomain, FormalGapRegister


class FormalGapResultStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    MIXED = "MIXED"
    STOPPED = "STOPPED"
    UNEVALUABLE = "UNEVALUABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True, slots=True)
class FormalNamedMatrix(CanonicalRecord):
    """Small declared matrix used only when a formal question requires one."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/formal-named-matrix'

    matrix_id: str
    role_id: str
    row_ids: tuple[str, ...]
    column_ids: tuple[str, ...]
    values: tuple[Decimal, ...]
    unit: str

    def __post_init__(self) -> None:
        validate_stable_id(self.matrix_id, field_name="matrix_id")
        validate_stable_id(self.role_id, field_name="role_id")
        require_sorted_unique_strings(
            self.row_ids,
            field_name="row_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.column_ids,
            field_name="column_ids",
            allow_empty=False,
        )
        if len(self.values) != len(self.row_ids) * len(self.column_ids):
            raise ValueError("formal matrix shape differs from its values")
        for value in self.values:
            validate_decimal(value, field_name="matrix_value")
        validate_nonempty(self.unit, field_name="unit")


@dataclass(frozen=True, slots=True)
class FormalAnalysisSample(CanonicalRecord):
    """One preparation/view/clock observation with full action-stage identity."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/formal-analysis-sample'

    sample_id: str
    independent_unit_id: str
    numerical_view_id: str
    action_word_id: str
    clock_id: str
    clock_coordinate: Decimal
    operand_ids: tuple[str, ...]
    prerequisite_ids: tuple[str, ...]
    control_ids: tuple[str, ...]
    requested_actions: tuple[NamedDecimal, ...]
    accepted_actions: tuple[NamedDecimal, ...]
    applied_actions: tuple[NamedDecimal, ...]
    realized_actions: tuple[NamedDecimal, ...]
    receiver_values: tuple[NamedDecimal, ...]
    state_values: tuple[NamedDecimal, ...]
    history_values: tuple[NamedDecimal, ...]
    valid: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name, value in (
            ("sample_id", self.sample_id),
            ("independent_unit_id", self.independent_unit_id),
            ("numerical_view_id", self.numerical_view_id),
            ("action_word_id", self.action_word_id),
            ("clock_id", self.clock_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_decimal(self.clock_coordinate, field_name="clock_coordinate")
        for name, values in (
            ("operand_ids", self.operand_ids),
            ("prerequisite_ids", self.prerequisite_ids),
            ("control_ids", self.control_ids),
        ):
            require_sorted_unique_strings(values, field_name=name)
        for record_name, records in (
            ("requested_actions", self.requested_actions),
            ("accepted_actions", self.accepted_actions),
            ("applied_actions", self.applied_actions),
            ("realized_actions", self.realized_actions),
            ("receiver_values", self.receiver_values),
            ("state_values", self.state_values),
            ("history_values", self.history_values),
        ):
            require_sorted_unique_ids(
                records,
                attribute="value_id",
                field_name=record_name,
            )
        if not self.realized_actions or not self.receiver_values:
            raise ValueError("formal sample requires realized action and receiver values")
        if self.outcome_access not in {
            OutcomeAccess.DEVELOPMENT_VISIBLE,
            OutcomeAccess.EVALUATION_SEALED,
        }:
            raise ValueError("formal samples must remain development-visible or sealed")


@dataclass(frozen=True, slots=True)
class FormalAnalysisInputManifest(CanonicalRecord):
    """Path-free standard method input for one formal domain."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/formal-analysis-input-manifest'

    input_id: str
    domain: FormalDomain
    denominator: ObjectIdentity
    formal_gap_register: ObjectIdentity
    source_materializations: tuple[ObjectIdentity, ...]
    preparation_unit_ids: tuple[str, ...]
    numerical_view_ids: tuple[str, ...]
    action_coordinate_ids: tuple[str, ...]
    receiver_coordinate_ids: tuple[str, ...]
    state_coordinate_ids: tuple[str, ...]
    clock_ids: tuple[str, ...]
    samples: tuple[FormalAnalysisSample, ...]
    declared_matrices: tuple[FormalNamedMatrix, ...]
    declared_structure_ids: tuple[str, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.input_id, field_name="input_id")
        if self.formal_gap_register.object_schema != FormalGapRegister.SCHEMA:
            raise ValueError("formal analysis input binds another gap register")
        require_sorted_unique_ids(
            self.source_materializations,
            attribute="object_id",
            field_name="source_materializations",
        )
        for name, values in (
            ("preparation_unit_ids", self.preparation_unit_ids),
            ("numerical_view_ids", self.numerical_view_ids),
            ("action_coordinate_ids", self.action_coordinate_ids),
            ("receiver_coordinate_ids", self.receiver_coordinate_ids),
            ("state_coordinate_ids", self.state_coordinate_ids),
            ("clock_ids", self.clock_ids),
            ("declared_structure_ids", self.declared_structure_ids),
        ):
            require_sorted_unique_strings(values, field_name=name)
        require_sorted_unique_ids(
            self.samples,
            attribute="sample_id",
            field_name="samples",
        )
        require_sorted_unique_ids(
            self.declared_matrices,
            attribute="matrix_id",
            field_name="declared_matrices",
        )
        if not self.source_materializations or not self.samples:
            raise ValueError("formal analysis input requires source and samples")
        if any(
            value.independent_unit_id not in self.preparation_unit_ids
            or value.numerical_view_id not in self.numerical_view_ids
            or value.clock_id not in self.clock_ids
            or value.outcome_access is not self.outcome_access
            for value in self.samples
        ):
            raise ValueError("formal sample lies outside the input manifest")
        if self.outcome_access not in {
            OutcomeAccess.DEVELOPMENT_VISIBLE,
            OutcomeAccess.EVALUATION_SEALED,
        }:
            raise ValueError("formal analysis input has incompatible visibility")


@dataclass(frozen=True, slots=True)
class FormalDomainMethodConfig(CanonicalRecord):
    """Closed scientific configuration shared by one formal-domain method."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/formal-domain-method-config'

    config_id: str
    domain: FormalDomain
    gap_ids: tuple[str, ...]
    zero_tolerance: Decimal
    materiality_threshold: Decimal
    convergence_tolerance: Decimal
    familywise_alpha: Decimal
    method_version: str
    maximum_claim_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        require_sorted_unique_strings(
            self.gap_ids,
            field_name="gap_ids",
            allow_empty=False,
        )
        for name, value in (
            ("zero_tolerance", self.zero_tolerance),
            ("materiality_threshold", self.materiality_threshold),
            ("convergence_tolerance", self.convergence_tolerance),
            ("familywise_alpha", self.familywise_alpha),
        ):
            validate_decimal(value, field_name=name, minimum=Decimal(0))
        if (
            self.materiality_threshold <= self.zero_tolerance
            or self.convergence_tolerance <= 0
            or not Decimal(0) < self.familywise_alpha < Decimal(1)
        ):
            raise ValueError("formal method thresholds are inconsistent")
        if self.method_version != "1.0.0":
            raise ValueError("unsupported formal method version")
        if self.maximum_claim_ceiling is not EvidenceCeiling.LOCAL_LAW:
            raise ValueError("formal domain methods are capped at local law")


@dataclass(frozen=True, slots=True)
class FormalPanelEvaluatorConfig(CanonicalRecord):
    """Frozen evaluator binding for the complete standard formal panel."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/formal-panel-evaluator-config'

    config_id: str
    formal_gap_register: ObjectIdentity
    formal_gap_coverage: ObjectIdentity
    domain_config_ids: tuple[str, ...]
    evaluator_version: str
    maximum_claim_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        if self.formal_gap_register.object_schema != FormalGapRegister.SCHEMA:
            raise ValueError("formal panel config binds another register")
        require_sorted_unique_strings(
            self.domain_config_ids,
            field_name="domain_config_ids",
            allow_empty=False,
        )
        if len(self.domain_config_ids) != 4:
            raise ValueError("formal panel config requires four domain configs")
        if self.evaluator_version != "1.0.0":
            raise ValueError("unsupported formal panel evaluator version")
        if self.maximum_claim_ceiling is not EvidenceCeiling.LOCAL_LAW:
            raise ValueError("formal panel evaluator cannot exceed local law")


@dataclass(frozen=True, slots=True)
class FormalGapAdjudication(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/formal-gap-adjudication'

    gap_id: str
    domain: FormalDomain
    status: FormalGapResultStatus
    estimator_family_id: str | None
    metrics: tuple[NamedDecimal, ...]
    independent_unit_ids: tuple[str, ...]
    numerical_view_ids: tuple[str, ...]
    decisive_falsifier_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    maximum_claim_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.gap_id, field_name="gap_id")
        if self.estimator_family_id is not None:
            validate_stable_id(
                self.estimator_family_id,
                field_name="estimator_family_id",
            )
        require_sorted_unique_ids(
            self.metrics,
            attribute="value_id",
            field_name="metrics",
        )
        for name, values in (
            ("independent_unit_ids", self.independent_unit_ids),
            ("numerical_view_ids", self.numerical_view_ids),
            ("decisive_falsifier_ids", self.decisive_falsifier_ids),
            ("reason_codes", self.reason_codes),
        ):
            require_sorted_unique_strings(values, field_name=name)
        if (
            self.status
            in {
                FormalGapResultStatus.UNEVALUABLE,
                FormalGapResultStatus.NOT_APPLICABLE,
                FormalGapResultStatus.STOPPED,
            }
            and not self.reason_codes
        ):
            raise ValueError("non-evaluable formal result requires a reason")
        if self.maximum_claim_ceiling is not EvidenceCeiling.LOCAL_LAW:
            raise ValueError("formal gap adjudication cannot exceed local law")
        if self.outcome_access not in {
            OutcomeAccess.DEVELOPMENT_VISIBLE,
            OutcomeAccess.EVALUATION_SEALED,
            OutcomeAccess.EVALUATION_REVEALED,
        }:
            raise ValueError("formal result has incompatible outcome access")


@dataclass(frozen=True, slots=True)
class FormalDomainAnalysisResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/formal-domain-analysis-result'

    result_id: str
    domain: FormalDomain
    input_manifest: ObjectIdentity
    config: ObjectIdentity
    rows: tuple[FormalGapAdjudication, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        require_sorted_unique_ids(
            self.rows,
            attribute="gap_id",
            field_name="rows",
        )
        if not self.rows or any(
            value.domain is not self.domain or value.outcome_access is not self.outcome_access
            for value in self.rows
        ):
            raise ValueError("formal domain result rows differ from their envelope")


@dataclass(frozen=True, slots=True)
class FormalMultiplicityResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/formal-multiplicity-result'

    family_id: str
    domain: FormalDomain
    hypothesis_count: int
    familywise_alpha: Decimal
    per_hypothesis_alpha: Decimal
    method_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.family_id, field_name="family_id")
        validate_stable_id(self.method_id, field_name="method_id")
        if self.hypothesis_count <= 0:
            raise ValueError("multiplicity family must contain hypotheses")
        for name, value in (
            ("familywise_alpha", self.familywise_alpha),
            ("per_hypothesis_alpha", self.per_hypothesis_alpha),
        ):
            validate_decimal(value, field_name=name, minimum=Decimal(0))
        if self.per_hypothesis_alpha != (self.familywise_alpha / Decimal(self.hypothesis_count)):
            raise ValueError("multiplicity threshold is not exact Bonferroni")


@dataclass(frozen=True, slots=True)
class FormalGapAdjudicationPanel(CanonicalRecord):
    """Complete evaluator-revealed panel; not a second scientific verdict."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/formal-gap-adjudication-panel'

    panel_id: str
    formal_gap_register: ObjectIdentity
    formal_gap_coverage: ObjectIdentity
    domain_results: tuple[ObjectIdentity, ...]
    rows: tuple[FormalGapAdjudication, ...]
    multiplicity: tuple[FormalMultiplicityResult, ...]
    maximum_claim_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.panel_id, field_name="panel_id")
        if self.formal_gap_register.object_schema != FormalGapRegister.SCHEMA:
            raise ValueError("formal panel binds another register")
        require_sorted_unique_ids(
            self.domain_results,
            attribute="object_id",
            field_name="domain_results",
        )
        require_sorted_unique_ids(
            self.rows,
            attribute="gap_id",
            field_name="rows",
        )
        require_sorted_unique_ids(
            self.multiplicity,
            attribute="family_id",
            field_name="multiplicity",
        )
        counts = {
            domain: sum(value.domain is domain for value in self.rows) for domain in FormalDomain
        }
        if counts != {
            FormalDomain.ALGEBRA: 12,
            FormalDomain.CALCULUS: 11,
            FormalDomain.GEOMETRY: 12,
            FormalDomain.DYNAMICS: 13,
        }:
            raise ValueError("formal panel must contain the complete 48-row register shape")
        if {value.domain for value in self.multiplicity} != set(FormalDomain):
            raise ValueError("formal panel requires four multiplicity families")
        if self.maximum_claim_ceiling is not EvidenceCeiling.LOCAL_LAW:
            raise ValueError("formal panel cannot exceed local law")
        if self.outcome_access is not OutcomeAccess.EVALUATION_REVEALED:
            raise ValueError("formal panel is revealed only by the evaluator")
        if any(
            value.outcome_access is not OutcomeAccess.EVALUATION_REVEALED for value in self.rows
        ):
            raise ValueError("formal panel rows differ from evaluator reveal state")


__all__ = [
    "FormalAnalysisInputManifest",
    "FormalAnalysisSample",
    "FormalDomainAnalysisResult",
    "FormalDomainMethodConfig",
    "FormalGapAdjudication",
    "FormalGapAdjudicationPanel",
    "FormalGapResultStatus",
    "FormalMultiplicityResult",
    "FormalNamedMatrix",
    "FormalPanelEvaluatorConfig",
]
