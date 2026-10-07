"""Prospective structural prediction, sealed target and adjudication records."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_stable_id,
)
from empirical_lawhood.planning.evidence_lineage import EvidenceDependenceAssessment, EvidenceDependenceSpec
from empirical_lawhood.planning.metatheory import EvidenceDependenceClass, MetatheoryAggregateDisposition, MetatheoryCellDisposition, MetatheoryClaimKind, MetatheoryEvidenceCeiling, MetatheoryMethodSelection, MetatheoryPredictiveLevel


class MetatheoryMissingTargetPolicy(StrEnum):
    PERMIT_UNEVALUABLE = "PERMIT_UNEVALUABLE"
    REQUIRE_COMPLETE = "REQUIRE_COMPLETE"


class MetatheoryAdjudicationStopKind(StrEnum):
    REVEAL_AUTHORITY_REQUIRED = "REVEAL_AUTHORITY_REQUIRED"
    ACQUISITION_INCOMPLETE = "ACQUISITION_INCOMPLETE"
    REQUIRED_TARGET_MISSING = "REQUIRED_TARGET_MISSING"
    PREREQUISITE_NONATTEMPT = "PREREQUISITE_NONATTEMPT"
    INPUT_IDENTITY_MISMATCH = "INPUT_IDENTITY_MISMATCH"


@dataclass(frozen=True, slots=True)
class MetatheoryTargetSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/metatheory-target-spec'

    target_id: str
    evidence_world_profile: ObjectIdentity
    source_qualification_requirement: ObjectIdentity | None
    preparation_requirement: ObjectIdentity | None
    physical_unit_ids: tuple[str, ...]
    complete_unit_role_id: str
    missing_target_policy: MetatheoryMissingTargetPolicy
    target_visibility: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.target_id, field_name="target_id")
        require_sorted_unique_strings(
            self.physical_unit_ids,
            field_name="physical_unit_ids",
            allow_empty=False,
        )
        validate_stable_id(self.complete_unit_role_id, field_name="complete_unit_role_id")


@dataclass(frozen=True, slots=True)
class CategoricalForecast(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/categorical-forecast'

    forecast_id: str
    category_ids: tuple[str, ...]
    predicted_category_id: str
    category_probabilities: tuple[NamedDecimal, ...]
    unsafe_observed_category_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.forecast_id, field_name="forecast_id")
        require_sorted_unique_strings(
            self.category_ids,
            field_name="category_ids",
            allow_empty=False,
        )
        if self.predicted_category_id not in self.category_ids:
            raise ValueError("categorical forecast predicts outside its roster")
        require_sorted_unique_ids(
            self.category_probabilities,
            attribute="value_id",
            field_name="category_probabilities",
        )
        if self.category_probabilities:
            probabilities = {value.value_id: value for value in self.category_probabilities}
            if set(probabilities) != set(self.category_ids):
                raise ValueError("categorical probabilities differ from category roster")
            if any(value.unit != "1" or value.value < 0 for value in probabilities.values()):
                raise ValueError("categorical probabilities must be nonnegative fractions")
            if sum(value.value for value in probabilities.values()) != Decimal(1):
                raise ValueError("categorical probabilities must sum exactly to one")
        require_sorted_unique_strings(
            self.unsafe_observed_category_ids,
            field_name="unsafe_observed_category_ids",
        )
        if not set(self.unsafe_observed_category_ids).issubset(self.category_ids):
            raise ValueError("unsafe categories differ from the closed category roster")


@dataclass(frozen=True, slots=True)
class MetricForecast(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/metric-forecast'

    forecast_id: str
    quantity_id: str
    native_unit: str
    interval_lower: NamedDecimal
    interval_upper: NamedDecimal
    calibration_method: MetatheoryMethodSelection
    uncertainty_operation_id: str
    acceptance_rule_id: str

    def __post_init__(self) -> None:
        for name in (
            "forecast_id",
            "quantity_id",
            "uncertainty_operation_id",
            "acceptance_rule_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if (
            self.interval_lower.unit != self.native_unit
            or self.interval_upper.unit != self.native_unit
            or self.interval_lower.value > self.interval_upper.value
        ):
            raise ValueError("metric forecast interval is inconsistent")


@dataclass(frozen=True, slots=True)
class DynamicalForecast(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/dynamical-forecast'

    forecast_id: str
    state_ids: tuple[str, ...]
    transition_id: str
    time_origin: ObjectIdentity
    horizon: NamedDecimal
    first_passage_lower: NamedDecimal
    first_passage_upper: NamedDecimal
    recovery_or_reentry_rule_id: str
    censoring_rule_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.forecast_id, field_name="forecast_id")
        require_sorted_unique_strings(self.state_ids, field_name="state_ids", allow_empty=False)
        for name in (
            "transition_id",
            "recovery_or_reentry_rule_id",
            "censoring_rule_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        unit = self.horizon.unit
        if (
            self.first_passage_lower.unit != unit
            or self.first_passage_upper.unit != unit
            or self.first_passage_lower.value < 0
            or self.first_passage_lower.value > self.first_passage_upper.value
            or self.first_passage_upper.value > self.horizon.value
        ):
            raise ValueError("dynamical first-passage interval exceeds its horizon")


@dataclass(frozen=True, slots=True)
class MetatheoryForecastCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/metatheory-forecast-cell'

    cell_id: str
    claim_kind: MetatheoryClaimKind
    predictive_level: MetatheoryPredictiveLevel
    target_id: str
    physical_unit_id: str
    coordinate_id: str
    property_id: str
    action_fibre_id: str
    face_id: str
    categorical: CategoricalForecast | None
    metric: MetricForecast | None
    dynamical: DynamicalForecast | None
    falsifier_ids: tuple[str, ...]
    unevaluable_condition_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "cell_id",
            "target_id",
            "physical_unit_id",
            "coordinate_id",
            "property_id",
            "action_fibre_id",
            "face_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        selected = tuple(
            value for value in (self.categorical, self.metric, self.dynamical) if value is not None
        )
        if len(selected) != 1:
            raise ValueError("forecast cell requires exactly one predictive payload")
        expected = {
            MetatheoryPredictiveLevel.CATEGORICAL: self.categorical,
            MetatheoryPredictiveLevel.METRIC: self.metric,
            MetatheoryPredictiveLevel.DYNAMICAL: self.dynamical,
        }
        if expected[self.predictive_level] is None:
            raise ValueError("forecast payload differs from its predictive level")
        require_sorted_unique_strings(
            self.falsifier_ids,
            field_name="falsifier_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.unevaluable_condition_ids,
            field_name="unevaluable_condition_ids",
            allow_empty=False,
        )


@dataclass(frozen=True, slots=True)
class MetatheoryScoringMethodBinding(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/metatheory-scoring-method-binding'

    binding_id: str
    predictive_level: MetatheoryPredictiveLevel
    method: MetatheoryMethodSelection

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")


@dataclass(frozen=True, slots=True)
class MetatheoryPredictionPackage(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/metatheory-prediction-package'

    package_id: str
    development_parents: tuple[ObjectIdentity, ...]
    targets: tuple[MetatheoryTargetSpec, ...]
    forecast_cells: tuple[MetatheoryForecastCell, ...]
    requested_dependence_class: EvidenceDependenceClass
    dependence_spec: ObjectIdentity
    complete_unit_rule_id: str
    aggregation_rule_id: str
    scoring_methods: tuple[MetatheoryScoringMethodBinding, ...]
    decisive_falsifier_ids: tuple[str, ...]
    planned_issue_id: str
    information_cutoff: ObjectIdentity
    target_outcome_access_count: int
    maximum_structural_evidence_ceiling: MetatheoryEvidenceCeiling
    ordinary_parent_identities: tuple[ObjectIdentity, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.package_id, field_name="package_id")
        require_sorted_unique_ids(
            self.development_parents,
            attribute="object_id",
            field_name="development_parents",
        )
        require_sorted_unique_ids(self.targets, attribute="target_id", field_name="targets")
        require_sorted_unique_ids(
            self.forecast_cells,
            attribute="cell_id",
            field_name="forecast_cells",
        )
        if not self.targets or not self.forecast_cells:
            raise ValueError("prediction package requires targets and forecasts")
        targets = {value.target_id: value for value in self.targets}
        if any(
            value.target_id not in targets
            or value.physical_unit_id not in targets[value.target_id].physical_unit_ids
            for value in self.forecast_cells
        ):
            raise ValueError("forecast cell falls outside its target/unit roster")
        if self.dependence_spec.object_schema != EvidenceDependenceSpec.SCHEMA:
            raise ValueError("prediction package requires a dependence spec")
        for name in ("complete_unit_rule_id", "aggregation_rule_id", "planned_issue_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_ids(
            self.scoring_methods,
            attribute="binding_id",
            field_name="scoring_methods",
        )
        levels = {value.predictive_level for value in self.forecast_cells}
        if {value.predictive_level for value in self.scoring_methods} != levels:
            raise ValueError("prediction package scoring-method roster differs from levels")
        require_sorted_unique_strings(
            self.decisive_falsifier_ids,
            field_name="decisive_falsifier_ids",
            allow_empty=False,
        )
        if type(self.target_outcome_access_count) is not int or self.target_outcome_access_count:
            raise ValueError("prediction package crossed the target-outcome cutoff")
        require_sorted_unique_ids(
            self.ordinary_parent_identities,
            attribute="object_id",
            field_name="ordinary_parent_identities",
        )


@dataclass(frozen=True, slots=True)
class MetatheoryPredictionIssueReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/metatheory-prediction-issue-receipt'

    receipt_id: str
    prediction_package: ObjectIdentity
    issue_clock_id: str
    issued_before_target_contact: bool
    target_contact_count: int
    target_outcome_access_count: int
    publication: ObjectIdentity
    recovery: ObjectIdentity
    grants_execution_or_reveal_authority: bool

    def __post_init__(self) -> None:
        for name in ("receipt_id", "issue_clock_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.prediction_package.object_schema != MetatheoryPredictionPackage.SCHEMA:
            raise ValueError("prediction issue binds another package schema")
        if (
            not self.issued_before_target_contact
            or self.target_contact_count
            or self.target_outcome_access_count
            or self.grants_execution_or_reveal_authority
        ):
            raise ValueError("prediction issue crossed target contact or granted authority")


@dataclass(frozen=True, slots=True)
class MetatheorySealedTarget(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/metatheory-sealed-target'

    target_id: str
    acquired_physical_unit_ids: tuple[str, ...]
    sealed_artifact: ObjectIdentity
    acquisition_complete: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.target_id, field_name="target_id")
        require_sorted_unique_strings(
            self.acquired_physical_unit_ids,
            field_name="acquired_physical_unit_ids",
        )


@dataclass(frozen=True, slots=True)
class MetatheoryTargetAcquisitionIndex(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/metatheory-target-acquisition-index'

    index_id: str
    prediction_issue: ObjectIdentity
    sealed_targets: tuple[MetatheorySealedTarget, ...]
    acquired_after_issue: bool
    outcomes_exposed: bool
    visibility: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.index_id, field_name="index_id")
        if self.prediction_issue.object_schema != MetatheoryPredictionIssueReceipt.SCHEMA:
            raise ValueError("target index binds another issue schema")
        require_sorted_unique_ids(
            self.sealed_targets,
            attribute="target_id",
            field_name="sealed_targets",
        )
        if not self.acquired_after_issue or self.outcomes_exposed:
            raise ValueError("sealed target index violates issue/reveal ordering")


@dataclass(frozen=True, slots=True)
class MetatheoryTargetOutcome(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/metatheory-target-outcome'

    outcome_id: str
    forecast_cell_id: str
    target_id: str
    physical_unit_id: str
    predictive_level: MetatheoryPredictiveLevel
    observed_category_id: str | None
    observed_metric: NamedDecimal | None
    observed_transition_id: str | None
    observed_first_passage: NamedDecimal | None
    censored: bool
    evidence_links: tuple[ObjectIdentity, ...]

    def __post_init__(self) -> None:
        for name in ("outcome_id", "forecast_cell_id", "target_id", "physical_unit_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        payloads = {
            MetatheoryPredictiveLevel.CATEGORICAL: self.observed_category_id is not None,
            MetatheoryPredictiveLevel.METRIC: self.observed_metric is not None,
            MetatheoryPredictiveLevel.DYNAMICAL: (
                self.observed_transition_id is not None or self.censored
            ),
        }
        if not payloads[self.predictive_level]:
            raise ValueError("target outcome lacks its predictive-level payload")
        if (
            self.predictive_level is not MetatheoryPredictiveLevel.CATEGORICAL
            and self.observed_category_id is not None
        ):
            raise ValueError("noncategorical outcome carries a category")
        if (
            self.predictive_level is not MetatheoryPredictiveLevel.METRIC
            and self.observed_metric is not None
        ):
            raise ValueError("nonmetric outcome carries a metric")
        if self.predictive_level is not MetatheoryPredictiveLevel.DYNAMICAL and (
            self.observed_transition_id is not None
            or self.observed_first_passage is not None
            or self.censored
        ):
            raise ValueError("nondynamical outcome carries transition data")
        if self.predictive_level is MetatheoryPredictiveLevel.DYNAMICAL:
            if self.censored:
                if self.observed_first_passage is not None:
                    raise ValueError("censored dynamical outcome cannot invent passage time")
            elif self.observed_transition_id is None or self.observed_first_passage is None:
                raise ValueError("uncensored dynamical outcome requires transition and passage")
        require_sorted_unique_ids(
            self.evidence_links,
            attribute="object_id",
            field_name="evidence_links",
        )


@dataclass(frozen=True, slots=True)
class RevealedTargetOutcomeBinding(CanonicalRecord):
    """Custody join from one evaluator-visible outcome to its sealed target."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/revealed-target-outcome-binding'

    binding_id: str
    prediction_issue: ObjectIdentity
    acquisition_index: ObjectIdentity
    sealed_target: ObjectIdentity
    sealed_artifact: ObjectIdentity
    forecast_cell: ObjectIdentity
    outcome: ObjectIdentity
    target_id: str
    physical_unit_id: str
    predictive_level: MetatheoryPredictiveLevel
    reveal_authorization: ObjectIdentity
    revealed_view_publication: ObjectIdentity
    revealed_view_recovery: ObjectIdentity
    outcome_access: OutcomeAccess
    visibility: VisibilityCeiling

    def __post_init__(self) -> None:
        for name in ("binding_id", "target_id", "physical_unit_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        expected_schemas = (
            (self.prediction_issue, MetatheoryPredictionIssueReceipt.SCHEMA),
            (self.acquisition_index, MetatheoryTargetAcquisitionIndex.SCHEMA),
            (self.sealed_target, MetatheorySealedTarget.SCHEMA),
            (self.forecast_cell, MetatheoryForecastCell.SCHEMA),
            (self.outcome, MetatheoryTargetOutcome.SCHEMA),
            (self.reveal_authorization, MetatheoryRevealAuthorizationBinding.SCHEMA),
        )
        if any(identity.object_schema != schema for identity, schema in expected_schemas):
            raise ValueError("revealed target outcome binding crosses schemas")
        if (
            self.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL
            or self.visibility is not VisibilityCeiling.OUTCOME_VISIBLE
        ):
            raise ValueError("revealed target outcome binding lacks evaluator-only visibility")


@dataclass(frozen=True, slots=True)
class MetatheoryRevealAuthorizationBinding(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/metatheory-reveal-authorization-binding'

    binding_id: str
    reveal_authority: ObjectIdentity
    prediction_issue: ObjectIdentity
    acquisition_index: ObjectIdentity
    reveal_permitted: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")
        if self.prediction_issue.object_schema != MetatheoryPredictionIssueReceipt.SCHEMA:
            raise ValueError("reveal binding names another issue schema")
        if self.acquisition_index.object_schema != MetatheoryTargetAcquisitionIndex.SCHEMA:
            raise ValueError("reveal binding names another acquisition schema")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.reveal_permitted == bool(self.reason_codes):
            raise ValueError("reveal binding permission and reasons disagree")


@dataclass(frozen=True, slots=True)
class MetatheoryAdjudicationSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/metatheory-adjudication-spec'

    spec_id: str
    prediction_package: ObjectIdentity
    scoring_methods: tuple[MetatheoryScoringMethodBinding, ...]
    multiplicity_rule_id: str
    complete_unit_aggregation_rule_id: str
    terminal_rule_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.spec_id, field_name="spec_id")
        if self.prediction_package.object_schema != MetatheoryPredictionPackage.SCHEMA:
            raise ValueError("adjudication spec names another prediction package schema")
        require_sorted_unique_ids(
            self.scoring_methods,
            attribute="binding_id",
            field_name="scoring_methods",
        )
        for name in (
            "multiplicity_rule_id",
            "complete_unit_aggregation_rule_id",
            "terminal_rule_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)


@dataclass(frozen=True, slots=True)
class MetatheoryAdjudicationCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/metatheory-adjudication-cell'

    cell_id: str
    forecast_cell: ObjectIdentity
    outcome: ObjectIdentity | None
    predictive_level: MetatheoryPredictiveLevel
    score: NamedDecimal | None
    decisive_falsifier_ids: tuple[str, ...]
    disposition: MetatheoryCellDisposition
    reason_codes: tuple[str, ...]
    evidence_links: tuple[ObjectIdentity, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.cell_id, field_name="cell_id")
        require_sorted_unique_strings(
            self.decisive_falsifier_ids,
            field_name="decisive_falsifier_ids",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        require_sorted_unique_ids(
            self.evidence_links,
            attribute="object_id",
            field_name="evidence_links",
        )
        if self.outcome is None and self.disposition is not MetatheoryCellDisposition.UNEVALUABLE:
            raise ValueError("missing outcome can only be unevaluable")


@dataclass(frozen=True, slots=True)
class MetatheoryAdjudicationResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/metatheory-adjudication-result'

    result_id: str
    prediction_issue: ObjectIdentity
    reveal_authorization: ObjectIdentity
    acquisition_index: ObjectIdentity
    cells: tuple[MetatheoryAdjudicationCell, ...]
    complete_physical_unit_count: int
    dependence_assessment: EvidenceDependenceAssessment
    achieved_dependence_class: EvidenceDependenceClass | None
    categorical_disposition: MetatheoryAggregateDisposition | None
    metric_disposition: MetatheoryAggregateDisposition | None
    dynamical_disposition: MetatheoryAggregateDisposition | None
    disposition: MetatheoryAggregateDisposition
    maximum_ordinary_evidence_ceiling: EvidenceCeiling
    maximum_structural_evidence_ceiling: MetatheoryEvidenceCeiling
    obstruction_refs: tuple[ObjectIdentity, ...]
    parent_promotion_permitted: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        require_sorted_unique_ids(self.cells, attribute="cell_id", field_name="cells")
        if (
            type(self.complete_physical_unit_count) is not int
            or self.complete_physical_unit_count < 0
        ):
            raise ValueError("adjudication complete-unit count must be nonnegative")
        require_sorted_unique_ids(
            self.obstruction_refs,
            attribute="object_id",
            field_name="obstruction_refs",
        )
        if self.parent_promotion_permitted:
            raise ValueError("structural adjudication cannot promote a parent claim")


@dataclass(frozen=True, slots=True)
class CustodyBoundMetatheoryAdjudicationResult(CanonicalRecord):
    """Additive adjudication result carrying exact revealed-outcome custody."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/custody-bound-metatheory-adjudication-result'

    result_id: str
    adjudication_result: MetatheoryAdjudicationResult
    outcome_bindings: tuple[RevealedTargetOutcomeBinding, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        require_sorted_unique_ids(
            self.outcome_bindings,
            attribute="binding_id",
            field_name="outcome_bindings",
        )
        expected_outcomes = {
            value.outcome for value in self.adjudication_result.cells if value.outcome is not None
        }
        bound_outcomes = {value.outcome for value in self.outcome_bindings}
        if bound_outcomes != expected_outcomes:
            raise ValueError("metatheory adjudication custody differs from scored outcomes")


@dataclass(frozen=True, slots=True)
class MetatheoryAdjudicationStop(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/metatheory-adjudication-stop'

    stop_id: str
    prediction_issue: ObjectIdentity
    stop_kind: MetatheoryAdjudicationStopKind
    reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess
    scientific_result_constructed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.stop_id, field_name="stop_id")
        require_sorted_unique_strings(
            self.reason_codes,
            field_name="reason_codes",
            allow_empty=False,
        )
        if self.scientific_result_constructed:
            raise ValueError("adjudication stop cannot construct a scientific result")


__all__ = [
    'CategoricalForecast',
    'DynamicalForecast',
    'MetatheoryAdjudicationCell',
    'MetatheoryAdjudicationResult',
    'CustodyBoundMetatheoryAdjudicationResult',
    'MetatheoryAdjudicationSpec',
    'MetatheoryAdjudicationStopKind',
    'MetatheoryAdjudicationStop',
    'MetatheoryForecastCell',
    'MetatheoryMissingTargetPolicy',
    'MetatheoryPredictionIssueReceipt',
    'MetatheoryPredictionPackage',
    'MetatheoryRevealAuthorizationBinding',
    'MetatheoryScoringMethodBinding',
    'MetatheorySealedTarget',
    'MetatheoryTargetAcquisitionIndex',
    'MetatheoryTargetOutcome',
    'MetatheoryTargetSpec',
    'MetricForecast',
    'RevealedTargetOutcomeBinding',
]
