"""Reusable support, validity and scientific-evidence obligations."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from .evidence import EvidenceCeiling
from .references import NamedDecimal, QuantityBound
from .serialization import (
    CanonicalRecord,
    ExtensionBinding,
    require_extensions,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_stable_id,
)
from .status import ReadinessStatus


class ObligationStatus(StrEnum):
    REQUIRED = "REQUIRED"
    SATISFIED = "SATISFIED"
    FAILED = "FAILED"
    UNEVALUABLE = "UNEVALUABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


def _require_evidence_for_satisfied(
    status: ObligationStatus, evidence_link_ids: tuple[str, ...], name: str
) -> None:
    require_sorted_unique_strings(evidence_link_ids, field_name=f"{name}.evidence_link_ids")
    if status is ObligationStatus.SATISFIED and not evidence_link_ids:
        raise ValueError(f"satisfied {name} requires evidence links")


@dataclass(frozen=True, slots=True)
class SupportSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/support-spec'

    support_id: str
    relation_id: str
    independent_unit_id: str
    physical_unit_count: int
    nested_numerical_view_count: int
    information_cutoff_id: str
    chart_ids: tuple[str, ...]
    denominator_cell_ids: tuple[str, ...]
    action_bounds: tuple[QuantityBound, ...]
    status: ObligationStatus
    evidence_link_ids: tuple[str, ...] = ()
    extensions: tuple[ExtensionBinding, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("support_id", self.support_id),
            ("relation_id", self.relation_id),
            ("independent_unit_id", self.independent_unit_id),
            ("information_cutoff_id", self.information_cutoff_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.physical_unit_count < 0:
            raise ValueError("physical_unit_count must be nonnegative")
        if self.nested_numerical_view_count < 0:
            raise ValueError("nested_numerical_view_count must be nonnegative")
        require_sorted_unique_strings(self.chart_ids, field_name="chart_ids")
        require_sorted_unique_strings(
            self.denominator_cell_ids,
            field_name="denominator_cell_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(
            self.action_bounds, attribute="bound_id", field_name="action_bounds"
        )
        _require_evidence_for_satisfied(self.status, self.evidence_link_ids, "support")
        if self.status is ObligationStatus.SATISFIED and self.physical_unit_count == 0:
            raise ValueError("satisfied support requires physical independent units")
        require_extensions(self.extensions)


@dataclass(frozen=True, slots=True)
class ValiditySpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/validity-spec'

    validity_id: str
    validity_domain_ids: tuple[str, ...]
    assumption_ids: tuple[str, ...]
    exclusion_reason_codes: tuple[str, ...]
    status: ObligationStatus
    evidence_link_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_stable_id(self.validity_id, field_name="validity_id")
        require_sorted_unique_strings(
            self.validity_domain_ids,
            field_name="validity_domain_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.assumption_ids, field_name="assumption_ids", allow_empty=False
        )
        require_sorted_unique_strings(
            self.exclusion_reason_codes, field_name="exclusion_reason_codes"
        )
        _require_evidence_for_satisfied(self.status, self.evidence_link_ids, "validity")


@dataclass(frozen=True, slots=True)
class UncertaintySpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/uncertainty-spec'

    uncertainty_id: str
    method_key: str
    independent_unit_id: str
    confidence_level: Decimal
    interval_quantity_ids: tuple[str, ...]
    limitation_codes: tuple[str, ...]
    status: ObligationStatus
    evidence_link_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("uncertainty_id", self.uncertainty_id),
            ("method_key", self.method_key),
            ("independent_unit_id", self.independent_unit_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_decimal(
            self.confidence_level,
            field_name="confidence_level",
            minimum=Decimal("0"),
        )
        if not Decimal("0") < self.confidence_level < Decimal("1"):
            raise ValueError("confidence_level must be strictly between zero and one")
        require_sorted_unique_strings(
            self.interval_quantity_ids,
            field_name="interval_quantity_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(self.limitation_codes, field_name="limitation_codes")
        _require_evidence_for_satisfied(self.status, self.evidence_link_ids, "uncertainty")


class FalsifierKind(StrEnum):
    WRONG_ACTION = "WRONG_ACTION"
    TEMPORAL_SUPPORT = "TEMPORAL_SUPPORT"
    WITHIN_CELL_RECURRENCE = "WITHIN_CELL_RECURRENCE"
    ONE_FACTOR_EXCHANGE = "ONE_FACTOR_EXCHANGE"
    NEGATIVE_CONTROL = "NEGATIVE_CONTROL"
    BASELINE_COMPARATOR = "BASELINE_COMPARATOR"
    STRUCTURAL_CONVERGENCE = "STRUCTURAL_CONVERGENCE"
    RECEIVER_GATE = "RECEIVER_GATE"


@dataclass(frozen=True, slots=True)
class FalsifierSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/falsifier-spec'

    falsifier_id: str
    kind: FalsifierKind
    capability_key: str
    description: str
    decisive_rule: str
    status: ObligationStatus
    evidence_link_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_stable_id(self.falsifier_id, field_name="falsifier_id")
        validate_stable_id(self.capability_key, field_name="capability_key")
        validate_nonempty(self.description, field_name="description")
        validate_nonempty(self.decisive_rule, field_name="decisive_rule")
        _require_evidence_for_satisfied(self.status, self.evidence_link_ids, "falsifier")


@dataclass(frozen=True, slots=True)
class ClosureSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/closure-spec'

    closure_id: str
    recurrence_cell_ids: tuple[str, ...]
    exchange_factor_ids: tuple[str, ...]
    retained_history_ids: tuple[str, ...]
    status: ObligationStatus
    evidence_link_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_stable_id(self.closure_id, field_name="closure_id")
        require_sorted_unique_strings(
            self.recurrence_cell_ids,
            field_name="recurrence_cell_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.exchange_factor_ids,
            field_name="exchange_factor_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(self.retained_history_ids, field_name="retained_history_ids")
        _require_evidence_for_satisfied(self.status, self.evidence_link_ids, "closure")


@dataclass(frozen=True, slots=True)
class StructuralConvergenceSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/structural-convergence-spec'

    convergence_id: str
    required_structure_ids: tuple[str, ...]
    numerical_view_ids: tuple[str, ...]
    tolerances: tuple[NamedDecimal, ...]
    status: ObligationStatus
    evidence_link_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_stable_id(self.convergence_id, field_name="convergence_id")
        require_sorted_unique_strings(
            self.required_structure_ids,
            field_name="required_structure_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(self.numerical_view_ids, field_name="numerical_view_ids")
        require_sorted_unique_ids(self.tolerances, attribute="value_id", field_name="tolerances")
        if self.status is ObligationStatus.SATISFIED:
            if len(self.numerical_view_ids) < 2:
                raise ValueError("satisfied structural convergence requires at least two views")
            if not self.tolerances:
                raise ValueError("satisfied structural convergence needs tolerances")
        _require_evidence_for_satisfied(
            self.status, self.evidence_link_ids, "structural convergence"
        )


@dataclass(frozen=True, slots=True)
class DiscrepancySpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/discrepancy-spec'

    discrepancy_id: str
    source_world_id: str
    target_world_id: str
    aligned_quantity_ids: tuple[str, ...]
    aligned_clock_relation_ids: tuple[str, ...]
    calibration_unit_ids: tuple[str, ...]
    held_out_validation_unit_ids: tuple[str, ...]
    discrepancy_bound_ids: tuple[str, ...]
    failed_test_ids: tuple[str, ...]
    evidence_ceiling: EvidenceCeiling
    status: ObligationStatus
    evidence_link_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("discrepancy_id", self.discrepancy_id),
            ("source_world_id", self.source_world_id),
            ("target_world_id", self.target_world_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.source_world_id == self.target_world_id:
            raise ValueError("discrepancy requires distinct source and target worlds")
        for field_name, values, allow_empty in (
            ("aligned_quantity_ids", self.aligned_quantity_ids, False),
            (
                "aligned_clock_relation_ids",
                self.aligned_clock_relation_ids,
                False,
            ),
            ("calibration_unit_ids", self.calibration_unit_ids, False),
            (
                "held_out_validation_unit_ids",
                self.held_out_validation_unit_ids,
                False,
            ),
            ("discrepancy_bound_ids", self.discrepancy_bound_ids, False),
            ("failed_test_ids", self.failed_test_ids, True),
        ):
            require_sorted_unique_strings(values, field_name=field_name, allow_empty=allow_empty)
        if set(self.calibration_unit_ids) & set(self.held_out_validation_unit_ids):
            raise ValueError("calibration and held-out discrepancy units overlap")
        _require_evidence_for_satisfied(self.status, self.evidence_link_ids, "discrepancy")


@dataclass(frozen=True, slots=True)
class ComputabilityEvidence(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/computability-evidence'

    computability_id: str
    envelope_id: str
    numerical_view_ids: tuple[str, ...]
    readiness: ReadinessStatus
    unresolved_reason_codes: tuple[str, ...]
    evidence_link_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_stable_id(self.computability_id, field_name="computability_id")
        validate_stable_id(self.envelope_id, field_name="envelope_id")
        require_sorted_unique_strings(
            self.numerical_view_ids,
            field_name="numerical_view_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.unresolved_reason_codes, field_name="unresolved_reason_codes"
        )
        require_sorted_unique_strings(self.evidence_link_ids, field_name="evidence_link_ids")
        if self.readiness is ReadinessStatus.READY:
            if self.unresolved_reason_codes:
                raise ValueError("ready computability cannot have unresolved reasons")
            if not self.evidence_link_ids:
                raise ValueError("ready computability requires evidence links")
        elif not self.unresolved_reason_codes:
            raise ValueError("non-ready computability requires reason codes")


@dataclass(frozen=True, slots=True)
class ScientificObligations(CanonicalRecord):
    """Typed evidence obligations shared by laws, transport and control."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/scientific-obligations'

    obligations_id: str
    support: SupportSpec
    validity: ValiditySpec
    uncertainty: UncertaintySpec
    falsifiers: tuple[FalsifierSpec, ...]
    closure: ClosureSpec
    structural_convergence: StructuralConvergenceSpec
    computability: ComputabilityEvidence
    discrepancy: DiscrepancySpec | None = None

    def __post_init__(self) -> None:
        validate_stable_id(self.obligations_id, field_name="obligations_id")
        require_sorted_unique_ids(
            self.falsifiers, attribute="falsifier_id", field_name="falsifiers"
        )
        if not self.falsifiers:
            raise ValueError("scientific obligations require decisive falsifiers")

    def blocking_reason_codes(self) -> tuple[str, ...]:
        reasons: list[str] = []
        statuses = (
            ("support", self.support.status),
            ("validity", self.validity.status),
            ("uncertainty", self.uncertainty.status),
            ("closure", self.closure.status),
            ("structural-convergence", self.structural_convergence.status),
        )
        for name, status in statuses:
            if status not in {ObligationStatus.SATISFIED, ObligationStatus.NOT_APPLICABLE}:
                reasons.append(f"{name}-{status.value.lower()}")
        for falsifier in self.falsifiers:
            if falsifier.status is not ObligationStatus.SATISFIED:
                reasons.append(
                    f"falsifier-{falsifier.falsifier_id}-{falsifier.status.value.lower()}"
                )
        if self.computability.readiness is not ReadinessStatus.READY:
            reasons.append("computability-boundary")
        if self.discrepancy is not None and self.discrepancy.status not in {
            ObligationStatus.SATISFIED,
            ObligationStatus.NOT_APPLICABLE,
        }:
            reasons.append(f"discrepancy-{self.discrepancy.status.value.lower()}")
        return tuple(sorted(reasons))

    @property
    def claim_ready(self) -> bool:
        return not self.blocking_reason_codes()
