"""Canonical contracts shared by all truth-known reference worlds."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import VisibilityCeiling
from empirical_lawhood.kernel.references import ControllerDecisionKind, NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.status import AdmissionStatus, ScientificStatus
from empirical_lawhood.kernel.systems import SystemSpec


class ReferenceWorldKind(StrEnum):
    STABLE_LINEAR = "w01-stable-linear"
    NONLINEAR_CHARTED = "w02-nonlinear-charted"
    HYSTERETIC_MEMORY = "w03-hysteretic-memory"
    MULTIRATE_DELAYED = "w04-multirate-delayed"
    HYBRID_PARTIAL = "w05-hybrid-partial"
    STRONG_EMPTY_ADMISSION = "w06-strong-empty-admission"
    COUPLED_INTERFACES = "w07-coupled-interfaces"
    DRIFTING_DEGRADING = "w08-drifting-degrading"
    OBSERVATIONAL_EQUIVALENCE = "w09-observational-equivalence"
    NUMERICAL_FALSE_STRUCTURE = "w10-numerical-false-structure"
    RARE_DECISIVE_SINK = "w11-rare-decisive-sink"
    DISCREPANCY_EXPLOITATION = "w12-discrepancy-exploitation"
    LATENCY_BOUNDARY = "w13-latency-boundary"
    PLANTED_RELATIONAL_ANOMALY = "w14-planted-relational-anomaly"
    NULL_SEARCH_FAMILY = "w15-null-search-family"
    RETROSPECTIVE_DEFEATED = "w16-retrospective-defeated"


REQUIRED_CONTROL_IDS = (
    "controller-exploitation",
    "exploration-to-confirmation",
    "leakage",
    "model-discrepancy",
    "multiplicity",
    "negative",
    "no-admission",
    "numerical-refinement",
    "positive",
    "search-family-completeness",
    "unsupported-action",
    "wrong-action",
    "wrong-time",
)


@dataclass(frozen=True, slots=True)
class ReferenceCase(CanonicalRecord):
    """One compact independent-unit/numerical-view reference observation."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/reference-case'

    case_id: str
    independent_unit_id: str
    denominator_cell_id: str
    action_id: str
    gauge_id: str
    horizon_id: str
    model_id: str
    numerical_view_id: str
    values: tuple[NamedDecimal, ...]
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("case_id", self.case_id),
            ("independent_unit_id", self.independent_unit_id),
            ("denominator_cell_id", self.denominator_cell_id),
            ("action_id", self.action_id),
            ("gauge_id", self.gauge_id),
            ("horizon_id", self.horizon_id),
            ("model_id", self.model_id),
            ("numerical_view_id", self.numerical_view_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_ids(self.values, attribute="value_id", field_name="values")
        if not self.values:
            raise ValueError("a reference case requires named values")
        require_sorted_unique_strings(self.tags, field_name="tags")

    def value(self, value_id: str) -> Decimal:
        validate_stable_id(value_id, field_name="value_id")
        for value in self.values:
            if value.value_id == value_id:
                return value.value
        raise KeyError(value_id)


@dataclass(frozen=True, slots=True)
class ReferenceControlSuite(CanonicalRecord):
    """Shared falsification contract exercised independently in every world."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/reference-control-suite'

    control_ids: tuple[str, ...]
    supported_action_lower: Decimal
    supported_action_upper: Decimal
    wrong_action_value: Decimal
    wrong_action_effect: Decimal
    minimum_response_effect: Decimal
    unsupported_action_value: Decimal
    registered_analysis_ids: tuple[str, ...]
    executed_analysis_ids: tuple[str, ...]
    raw_signal_p: Decimal
    multiplicity_cutoff: Decimal
    nominal_model_safe: bool
    discrepant_model_safe: bool
    outside_admission_gate_passes: tuple[bool, ...]
    parent_visibility: VisibilityCeiling
    fresh_visibility: VisibilityCeiling

    def __post_init__(self) -> None:
        if self.control_ids != REQUIRED_CONTROL_IDS:
            raise ValueError("reference world must bind the complete shared control suite")
        for name, value in (
            ("supported_action_lower", self.supported_action_lower),
            ("supported_action_upper", self.supported_action_upper),
            ("wrong_action_value", self.wrong_action_value),
            ("wrong_action_effect", self.wrong_action_effect),
            ("minimum_response_effect", self.minimum_response_effect),
            ("unsupported_action_value", self.unsupported_action_value),
            ("raw_signal_p", self.raw_signal_p),
            ("multiplicity_cutoff", self.multiplicity_cutoff),
        ):
            validate_decimal(value, field_name=name)
        if self.supported_action_lower >= self.supported_action_upper:
            raise ValueError("reference action support must have positive width")
        if not (
            self.supported_action_lower <= self.wrong_action_value <= self.supported_action_upper
        ):
            raise ValueError("wrong-action control must remain inside observed support")
        if (
            self.supported_action_lower
            <= self.unsupported_action_value
            <= self.supported_action_upper
        ):
            raise ValueError("unsupported-action control must lie outside support")
        if self.wrong_action_effect >= self.minimum_response_effect:
            raise ValueError("wrong-action control must fail the response threshold")
        require_sorted_unique_strings(
            self.registered_analysis_ids,
            field_name="registered_analysis_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.executed_analysis_ids,
            field_name="executed_analysis_ids",
            allow_empty=False,
        )
        for name, value in (
            ("raw_signal_p", self.raw_signal_p),
            ("multiplicity_cutoff", self.multiplicity_cutoff),
        ):
            if not Decimal(0) <= value <= Decimal(1):
                raise ValueError(f"{name} must be a probability")
        if self.raw_signal_p <= self.multiplicity_cutoff:
            raise ValueError("shared chance signal must fail the multiplicity threshold")
        if not self.nominal_model_safe or self.discrepant_model_safe:
            raise ValueError("shared discrepancy control must defeat nominal-only safety")
        if not self.outside_admission_gate_passes or all(self.outside_admission_gate_passes):
            raise ValueError("outside-admission control requires a failed gate")
        if self.parent_visibility is not VisibilityCeiling.OUTCOME_VISIBLE:
            raise ValueError("shared exploratory parent must be outcome-visible")
        if self.fresh_visibility is not VisibilityCeiling.PROSPECTIVE:
            raise ValueError("shared fresh evidence must begin prospectively")


@dataclass(frozen=True, slots=True)
class ReferenceCheck(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/reference-check'

    check_id: str
    passed: bool
    reason_codes: tuple[str, ...]
    observed: Decimal | None = None
    expected: Decimal | None = None
    tolerance: Decimal | None = None
    unit: str | None = None

    def __post_init__(self) -> None:
        validate_stable_id(self.check_id, field_name="check_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        metric_values = (self.observed, self.expected, self.tolerance, self.unit)
        if any(value is not None for value in metric_values):
            if any(value is None for value in metric_values):
                raise ValueError("quantitative check fields must be supplied together")
            assert self.observed is not None
            assert self.expected is not None
            assert self.tolerance is not None
            assert self.unit is not None
            validate_decimal(self.observed, field_name="observed")
            validate_decimal(self.expected, field_name="expected")
            validate_decimal(self.tolerance, field_name="tolerance", minimum=Decimal(0))
            validate_nonempty(self.unit, field_name="unit")
            within = abs(self.observed - self.expected) <= self.tolerance
            if within is not self.passed:
                raise ValueError("quantitative check disposition differs from its tolerance")
        if self.passed and self.reason_codes:
            raise ValueError("a passing reference check cannot retain failure reasons")
        if not self.passed and not self.reason_codes:
            raise ValueError("a failing reference check requires reason codes")


@dataclass(frozen=True, slots=True)
class ReferenceOracle(CanonicalRecord):
    """Ground-truth result declared independently of the recovery evaluator."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/reference-oracle'

    expected_check_ids: tuple[str, ...]
    expected_metrics: tuple[NamedDecimal, ...]
    recovered_structure_ids: tuple[str, ...]
    rejected_law_ids: tuple[str, ...]
    admission_status: AdmissionStatus
    controller_decision: ControllerDecisionKind
    nominated_hypothesis_ids: tuple[str, ...]
    supported_hypothesis_ids: tuple[str, ...]
    scientific_status: ScientificStatus
    parent_claim_preserved: bool

    def __post_init__(self) -> None:
        require_sorted_unique_strings(
            self.expected_check_ids,
            field_name="expected_check_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(
            self.expected_metrics,
            attribute="value_id",
            field_name="expected_metrics",
        )
        for name, values in (
            ("recovered_structure_ids", self.recovered_structure_ids),
            ("rejected_law_ids", self.rejected_law_ids),
            ("nominated_hypothesis_ids", self.nominated_hypothesis_ids),
            ("supported_hypothesis_ids", self.supported_hypothesis_ids),
        ):
            require_sorted_unique_strings(values, field_name=name)
        if not set(self.supported_hypothesis_ids).issubset(self.nominated_hypothesis_ids):
            raise ValueError("supported hypotheses must have been nominated")


@dataclass(frozen=True, slots=True)
class ReferenceWorldSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/reference-world-spec'

    reference_id: str
    kind: ReferenceWorldKind
    description: str
    system: SystemSpec
    cases: tuple[ReferenceCase, ...]
    controls: ReferenceControlSuite
    oracle: ReferenceOracle

    def __post_init__(self) -> None:
        validate_stable_id(self.reference_id, field_name="reference_id")
        validate_nonempty(self.description, field_name="description")
        if self.reference_id != self.kind.value:
            raise ValueError("reference ID must equal its registered world-kind value")
        if self.system.world.world_id != self.reference_id:
            raise ValueError("reference system world identity differs")
        require_sorted_unique_ids(self.cases, attribute="case_id", field_name="cases")
        if not self.cases:
            raise ValueError("a reference world requires observations")
        known_actions = set(self.system.relation.action_quantity_ids)
        known_views = {view.view_id for view in self.system.numerical_views}
        for case in self.cases:
            if case.independent_unit_id != self.system.independent_unit.unit_id:
                raise ValueError("reference case inflates the physical independent unit")
            if case.action_id not in known_actions:
                raise ValueError("reference case uses an unknown action coordinate")
            if case.horizon_id != self.system.relation.horizon.horizon_id:
                raise ValueError("reference case uses a different response horizon")
            if case.numerical_view_id not in known_views:
                raise ValueError("reference case uses an unknown numerical view")

    def accepts_identity(self, object_id: str, fingerprint: str) -> bool:
        validate_stable_id(object_id, field_name="object_id")
        validate_sha256(fingerprint, field_name="fingerprint")
        return object_id == self.reference_id and fingerprint == self.fingerprint()


@dataclass(frozen=True, slots=True)
class ReferenceEvaluation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/reference-evaluation'

    evaluation_id: str
    reference_id: str
    reference_fingerprint: str
    checks: tuple[ReferenceCheck, ...]
    metrics: tuple[NamedDecimal, ...]
    recovered_structure_ids: tuple[str, ...]
    rejected_law_ids: tuple[str, ...]
    admission_status: AdmissionStatus
    controller_decision: ControllerDecisionKind
    nominated_hypothesis_ids: tuple[str, ...]
    supported_hypothesis_ids: tuple[str, ...]
    scientific_status: ScientificStatus
    parent_claim_fingerprint_before: str
    parent_claim_fingerprint_after: str
    exploration_visibility: VisibilityCeiling
    fresh_evidence_visibility: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.evaluation_id, field_name="evaluation_id")
        validate_stable_id(self.reference_id, field_name="reference_id")
        for name, value in (
            ("reference_fingerprint", self.reference_fingerprint),
            ("parent_claim_fingerprint_before", self.parent_claim_fingerprint_before),
            ("parent_claim_fingerprint_after", self.parent_claim_fingerprint_after),
        ):
            validate_sha256(value, field_name=name)
        require_sorted_unique_ids(self.checks, attribute="check_id", field_name="checks")
        require_sorted_unique_ids(self.metrics, attribute="value_id", field_name="metrics")
        for name, values in (
            ("recovered_structure_ids", self.recovered_structure_ids),
            ("rejected_law_ids", self.rejected_law_ids),
            ("nominated_hypothesis_ids", self.nominated_hypothesis_ids),
            ("supported_hypothesis_ids", self.supported_hypothesis_ids),
        ):
            require_sorted_unique_strings(values, field_name=name)
        if not set(self.supported_hypothesis_ids).issubset(self.nominated_hypothesis_ids):
            raise ValueError("supported hypotheses must have been nominated")
        if self.exploration_visibility is not VisibilityCeiling.OUTCOME_VISIBLE:
            raise ValueError("reference exploration must remain outcome-visible")
        if self.fresh_evidence_visibility is not VisibilityCeiling.PROSPECTIVE:
            raise ValueError("fresh reference evidence must remain prospective")

    @property
    def all_checks_passed(self) -> bool:
        return all(check.passed for check in self.checks)


def assert_matches_oracle(world: ReferenceWorldSpec, evaluation: ReferenceEvaluation) -> None:
    """Fail closed when recovered structure differs from truth-known expectations."""

    differences: list[str] = []
    if evaluation.reference_id != world.reference_id:
        differences.append("reference-id")
    if evaluation.reference_fingerprint != world.fingerprint():
        differences.append("reference-fingerprint")
    if tuple(check.check_id for check in evaluation.checks) != (world.oracle.expected_check_ids):
        differences.append("check-family")
    if not evaluation.all_checks_passed:
        differences.append("failed-check")
    for attribute in (
        "metrics",
        "recovered_structure_ids",
        "rejected_law_ids",
        "admission_status",
        "controller_decision",
        "nominated_hypothesis_ids",
        "supported_hypothesis_ids",
        "scientific_status",
    ):
        expected_attribute = "expected_metrics" if attribute == "metrics" else attribute
        if getattr(evaluation, attribute) != getattr(world.oracle, expected_attribute):
            differences.append(attribute)
    preserved = (
        evaluation.parent_claim_fingerprint_before == evaluation.parent_claim_fingerprint_after
    )
    if preserved is not world.oracle.parent_claim_preserved:
        differences.append("parent-claim-preservation")
    if differences:
        raise ValueError(
            "reference evaluation differs from its truth oracle: " + ", ".join(sorted(differences))
        )
