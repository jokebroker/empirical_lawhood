"""Prospective package, sealed fan-in and target-local evaluation for selective dependence response.

The evaluator is deliberately target-neutral.  Target adapters supply only the
frozen receiver roster, hold identity, neutral margin and selective-exchange
reducer.  No target may add a post-reveal threshold or rescue rule here.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from math import sqrt
from typing import ClassVar

from scipy.stats import t as student_t

from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_sha256,
    validate_stable_id,
)

from .analysis import SelectiveDependenceResponsePredictiveScore, SelectiveDependenceResponseSelectiveDependenceSignature, bonferroni_confidence_level, condition_cell_id, score_categorical_predictions
from .analysis_design import SelectiveDependenceResponseTargetAnalysisFreeze, reduce_selective_signature
from .comparators import SelectiveDependenceResponseCategoricalPrediction, SelectiveDependenceResponseComparatorEncoding
from .contracts import SelectiveDependenceResponseAxisAdjudication, SelectiveDependenceResponseAxisName, SelectiveDependenceResponseAxisState, SelectiveDependenceResponseConstructReviewAttestation, SelectiveDependenceResponseDisposition, SelectiveDependenceResponseExchangeExpectation, SelectiveDependenceResponseExchangeForecast, SelectiveDependenceResponsePhase, SelectiveDependenceResponseTargetAdjudication, SelectiveDependenceResponseTargetHandoff, SelectiveDependenceResponseTargetPanel, digest_ids
from .forecast import SelectiveDependenceResponseDevelopmentBundle
from .development_completion import SelectiveDependenceResponseDevelopmentCompletionEnvelope
from .study_completion import SelectiveDependenceResponseStudyForecastCompletionEnvelope
from .inference import SelectiveDependenceResponseExchangeState, assess_exchange
from .target_adjudication import adjudicate_target, axis, compact_handoff


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseEvaluationPackage(CanonicalRecord):
    """Outcome-free conditional follow-up of one eligible development parent."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-evaluation-package'

    package_id: str
    target_id: str
    study_forecast_completion: ObjectIdentity
    study_forecast_qualification: ObjectIdentity
    development_completion: ObjectIdentity
    development_bundle: ObjectIdentity
    development_parent: ObjectIdentity
    forecast: ObjectIdentity
    design: ObjectIdentity
    analysis_freeze: ObjectIdentity
    construct_review: ObjectIdentity
    source_qualification: ObjectIdentity
    comparator_encodings: tuple[ObjectIdentity, ...]
    development_implementation_sha256: str
    evaluation_implementation_sha256: str
    development_receipt_ids: tuple[str, ...]
    evaluation_complete_unit_ids: tuple[str, ...]
    evaluation_complete_unit_ids_sha256: str
    minimum_predictive_accuracy: Decimal
    scientific_override_count: int
    evaluation_outcome_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("package_id", "target_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_ids(
            self.comparator_encodings,
            attribute="object_id",
            field_name="comparator_encodings",
        )
        if len(self.comparator_encodings) != 10:
            raise ValueError("evaluation package requires the exact ten comparators")
        validate_sha256(
            self.development_implementation_sha256,
            field_name="development_implementation_sha256",
        )
        validate_sha256(
            self.evaluation_implementation_sha256,
            field_name="evaluation_implementation_sha256",
        )
        require_sorted_unique_strings(
            self.development_receipt_ids,
            field_name="development_receipt_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.evaluation_complete_unit_ids,
            field_name="evaluation_complete_unit_ids",
            allow_empty=False,
        )
        validate_sha256(
            self.evaluation_complete_unit_ids_sha256,
            field_name="evaluation_complete_unit_ids_sha256",
        )
        if self.evaluation_complete_unit_ids_sha256 != digest_ids(
            self.evaluation_complete_unit_ids
        ):
            raise ValueError("evaluation-package roster digest differs")
        validate_decimal(
            self.minimum_predictive_accuracy,
            field_name="minimum_predictive_accuracy",
            minimum=Decimal(0),
        )
        if self.minimum_predictive_accuracy > 1:
            raise ValueError("minimum predictive accuracy exceeds one")
        if self.scientific_override_count or self.evaluation_outcome_count:
            raise ValueError("evaluation package cannot override science or inspect evaluation")
        if self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
            raise ValueError("evaluation package must remain development visible")


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseSealedIndex(CanonicalRecord):
    """Receipt-bound exact index of one sealed bounded-shard panel."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-sealed-index'

    index_id: str
    target_id: str
    evaluation_package: ObjectIdentity
    sealed_panel: ObjectIdentity
    expected_complete_unit_ids: tuple[str, ...]
    observed_complete_unit_ids: tuple[str, ...]
    complete_unit_ids_sha256: str
    generator_receipt_ids: tuple[str, ...]
    generator_materialization_ids: tuple[str, ...]
    duplicate_complete_unit_count: int
    surplus_complete_unit_count: int
    full_fan_in: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("index_id", "target_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in ("expected_complete_unit_ids", "observed_complete_unit_ids"):
            require_sorted_unique_strings(getattr(self, name), field_name=name, allow_empty=False)
        validate_sha256(self.complete_unit_ids_sha256, field_name="complete_unit_ids_sha256")
        if self.complete_unit_ids_sha256 != digest_ids(self.expected_complete_unit_ids):
            raise ValueError("sealed-index roster digest differs")
        for name in ("generator_receipt_ids", "generator_materialization_ids"):
            require_sorted_unique_strings(getattr(self, name), field_name=name, allow_empty=False)
        if self.duplicate_complete_unit_count < 0 or self.surplus_complete_unit_count < 0:
            raise ValueError("sealed-index adverse counts must be nonnegative")
        expected_full = (
            self.expected_complete_unit_ids == self.observed_complete_unit_ids
            and self.duplicate_complete_unit_count == 0
            and self.surplus_complete_unit_count == 0
        )
        if self.full_fan_in != expected_full:
            raise ValueError("sealed-index full-fan-in flag is not roster-derived")
        if not self.full_fan_in:
            raise ValueError("an incomplete sealed index cannot cross the reveal barrier")
        if self.outcome_access is not OutcomeAccess.EVALUATION_SEALED:
            raise ValueError("sealed index must remain evaluation sealed")


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseSealedShardManifest(CanonicalRecord):
    """Value-free metadata emitted beside one sealed bounded shard."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-sealed-shard-manifest'

    manifest_id: str
    target_id: str
    evaluation_package: ObjectIdentity
    sealed_panel: ObjectIdentity
    complete_unit_ids: tuple[str, ...]
    complete_unit_ids_sha256: str
    native_receiver_value_count: int
    prediction_correctness_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("manifest_id", "target_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_strings(
            self.complete_unit_ids, field_name="complete_unit_ids", allow_empty=False
        )
        validate_sha256(self.complete_unit_ids_sha256, field_name="complete_unit_ids_sha256")
        if self.complete_unit_ids_sha256 != digest_ids(self.complete_unit_ids):
            raise ValueError("sealed-shard roster digest differs")
        if self.native_receiver_value_count or self.prediction_correctness_count:
            raise ValueError("sealed progress metadata cannot expose outcomes or correctness")
        if self.outcome_access is not OutcomeAccess.EVALUATION_SEALED:
            raise ValueError("sealed-shard manifest must retain sealed custody")


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseRevealRecord(CanonicalRecord):
    """Immutable audit of the one logical full-panel reveal act."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-reveal-record'

    reveal_id: str
    target_id: str
    evaluation_package: ObjectIdentity
    sealed_index: ObjectIdentity
    sealed_panel: ObjectIdentity
    reveal_authority: ObjectIdentity
    evaluator_task_id: str
    evaluator_attempt_id: str
    input_receipt_ids: tuple[str, ...]
    input_materialization_ids: tuple[str, ...]
    revealed_complete_unit_count: int
    partial_score_count_before_reveal: int
    atomic_full_panel_open: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in (
            "reveal_id",
            "target_id",
            "evaluator_task_id",
            "evaluator_attempt_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in ("input_receipt_ids", "input_materialization_ids"):
            require_sorted_unique_strings(getattr(self, name), field_name=name, allow_empty=False)
        if self.revealed_complete_unit_count < 1:
            raise ValueError("reveal record must open at least one complete unit")
        if self.partial_score_count_before_reveal:
            raise ValueError("partial scoring before reveal is forbidden")
        if not self.atomic_full_panel_open:
            raise ValueError("evaluation reveal must open the exact panel in one logical act")
        if self.outcome_access is not OutcomeAccess.EVALUATION_REVEALED:
            raise ValueError("reveal record must retain revealed visibility")


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseDenominatorReadout(CanonicalRecord):
    """Explicit issued/source-valid/evaluable alternatives for one model score."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-denominator-readout'

    readout_id: str
    target_id: str
    model_id: str
    issued_complete_unit_count: int
    source_valid_complete_unit_count: int
    evaluable_complete_unit_count: int
    stopped_complete_unit_count: int
    missing_complete_unit_count: int
    evaluable_mean_accuracy: Decimal
    issued_worst_case_mean_accuracy: Decimal
    issued_best_case_mean_accuracy: Decimal
    stopped_or_missing_count_as_failures_in_primary: bool
    nested_cases_count_as_units: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("readout_id", "target_id", "model_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in (
            "issued_complete_unit_count",
            "source_valid_complete_unit_count",
            "evaluable_complete_unit_count",
            "stopped_complete_unit_count",
            "missing_complete_unit_count",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.issued_complete_unit_count != (
            self.evaluable_complete_unit_count
            + self.stopped_complete_unit_count
            + self.missing_complete_unit_count
        ):
            raise ValueError("denominator alternatives do not close")
        if self.source_valid_complete_unit_count != (
            self.evaluable_complete_unit_count + self.stopped_complete_unit_count
        ):
            raise ValueError("source-valid denominator does not close")
        for name in (
            "evaluable_mean_accuracy",
            "issued_worst_case_mean_accuracy",
            "issued_best_case_mean_accuracy",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
            if getattr(self, name) > 1:
                raise ValueError(f"{name} exceeds one")
        if not (self.issued_worst_case_mean_accuracy <= self.issued_best_case_mean_accuracy):
            raise ValueError("denominator bounds are reversed")
        if not self.stopped_or_missing_count_as_failures_in_primary:
            raise ValueError("issued worst-case accuracy is the frozen primary denominator")
        if self.nested_cases_count_as_units:
            raise ValueError("denominator readout cannot inflate nested cases")
        if self.outcome_access is not OutcomeAccess.EVALUATION_REVEALED:
            raise ValueError("denominator readout requires revealed evaluation outcomes")


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseComparatorFamilyAdvantage(CanonicalRecord):
    """Paired complete-unit advantage over the best comparator on each unit."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-comparator-family-advantage'

    advantage_id: str
    target_id: str
    comparator_model_ids: tuple[str, ...]
    complete_unit_advantages: tuple[NamedDecimal, ...]
    complete_unit_ids_sha256: str
    mean_advantage: Decimal
    simultaneous_lower: Decimal
    one_sided_alpha: Decimal
    comparator_advantages: tuple[SelectiveDependenceResponseComparatorSpecificAdvantage, ...]
    comparator_better_ids: tuple[str, ...]
    distinguished: bool
    nested_cases_count_as_units: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("advantage_id", "target_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_strings(
            self.comparator_model_ids,
            field_name="comparator_model_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(
            self.complete_unit_advantages,
            attribute="value_id",
            field_name="complete_unit_advantages",
        )
        if len(self.complete_unit_advantages) < 2:
            raise ValueError("paired comparator inference needs two complete units")
        validate_sha256(self.complete_unit_ids_sha256, field_name="complete_unit_ids_sha256")
        if self.complete_unit_ids_sha256 != digest_ids(
            tuple(value.value_id for value in self.complete_unit_advantages)
        ):
            raise ValueError("paired comparator roster digest differs")
        for name in ("mean_advantage", "simultaneous_lower", "one_sided_alpha"):
            validate_decimal(getattr(self, name), field_name=name)
        if not Decimal(0) < self.one_sided_alpha < Decimal(1):
            raise ValueError("comparator alpha must lie inside (0, 1)")
        require_sorted_unique_ids(
            self.comparator_advantages,
            attribute="advantage_id",
            field_name="comparator_advantages",
        )
        if (
            tuple(sorted(value.comparator_model_id for value in self.comparator_advantages))
            != self.comparator_model_ids
        ):
            raise ValueError("comparator-specific advantage roster differs")
        require_sorted_unique_strings(
            self.comparator_better_ids,
            field_name="comparator_better_ids",
        )
        expected_better = tuple(
            sorted(
                value.comparator_model_id
                for value in self.comparator_advantages
                if value.comparator_better
            )
        )
        if self.comparator_better_ids != expected_better:
            raise ValueError("comparator-better roster is not interval-derived")
        if self.distinguished != all(value.distinguished for value in self.comparator_advantages):
            raise ValueError("comparator family distinctiveness is not question-derived")
        if self.nested_cases_count_as_units:
            raise ValueError("comparator inference cannot inflate nested cases")
        if self.outcome_access is not OutcomeAccess.EVALUATION_REVEALED:
            raise ValueError("comparator advantage requires revealed outcomes")


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseComparatorSpecificAdvantage(CanonicalRecord):
    """Held-out cell and exchange advantage against one frozen comparator."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-comparator-specific-advantage'

    advantage_id: str
    target_id: str
    comparator_model_id: str
    complete_unit_advantages: tuple[NamedDecimal, ...]
    mean_advantage: Decimal
    simultaneous_lower: Decimal
    simultaneous_upper: Decimal
    relevant_exchange_ids: tuple[str, ...]
    comparator_opposed_exchange_ids: tuple[str, ...]
    distinguished: bool
    comparator_better: bool
    nested_cases_count_as_units: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("advantage_id", "target_id", "comparator_model_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_ids(
            self.complete_unit_advantages,
            attribute="value_id",
            field_name="complete_unit_advantages",
        )
        if len(self.complete_unit_advantages) < 2:
            raise ValueError("comparator-specific inference needs two complete units")
        for name in ("mean_advantage", "simultaneous_lower", "simultaneous_upper"):
            validate_decimal(getattr(self, name), field_name=name)
        if not self.simultaneous_lower <= self.mean_advantage <= self.simultaneous_upper:
            raise ValueError("comparator-specific interval does not contain its mean")
        for name in ("relevant_exchange_ids", "comparator_opposed_exchange_ids"):
            require_sorted_unique_strings(getattr(self, name), field_name=name)
        if not set(self.comparator_opposed_exchange_ids).issubset(self.relevant_exchange_ids):
            raise ValueError("opposed comparator exchange lies outside its challenge")
        expected_distinguished = self.simultaneous_lower > 0 or bool(
            self.comparator_opposed_exchange_ids
        )
        if self.distinguished != expected_distinguished:
            raise ValueError("comparator-specific distinctiveness is not evidence-derived")
        if self.comparator_better != (
            self.simultaneous_upper < 0 and not self.comparator_opposed_exchange_ids
        ):
            raise ValueError("comparator-specific opposition is not evidence-derived")
        if self.nested_cases_count_as_units:
            raise ValueError("comparator-specific inference cannot inflate nested cases")
        if self.outcome_access is not OutcomeAccess.EVALUATION_REVEALED:
            raise ValueError("comparator-specific advantage requires revealed outcomes")


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseTargetEvaluationBundle(CanonicalRecord):
    """Single revealed target result; it does not contain the sealed panel bytes."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-target-evaluation-bundle'

    bundle_id: str
    target_id: str
    evaluation_package: ObjectIdentity
    sealed_index: ObjectIdentity
    reveal_record: SelectiveDependenceResponseRevealRecord
    forecast_score: SelectiveDependenceResponsePredictiveScore
    comparator_scores: tuple[SelectiveDependenceResponsePredictiveScore, ...]
    comparator_family_advantage: SelectiveDependenceResponseComparatorFamilyAdvantage
    denominator_readouts: tuple[SelectiveDependenceResponseDenominatorReadout, ...]
    selective_signature: SelectiveDependenceResponseSelectiveDependenceSignature
    axes: tuple[SelectiveDependenceResponseAxisAdjudication, ...]
    adjudication: SelectiveDependenceResponseTargetAdjudication
    handoff: SelectiveDependenceResponseTargetHandoff
    maximum_evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("bundle_id", "target_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_ids(
            self.comparator_scores, attribute="score_id", field_name="comparator_scores"
        )
        if len(self.comparator_scores) != 10:
            raise ValueError("target evaluation lacks the exact comparator score family")
        require_sorted_unique_ids(
            self.denominator_readouts,
            attribute="readout_id",
            field_name="denominator_readouts",
        )
        if len(self.denominator_readouts) != 11:
            raise ValueError("target evaluation lacks model denominator alternatives")
        require_sorted_unique_ids(self.axes, attribute="adjudication_id", field_name="axes")
        if tuple(sorted(value.axis for value in self.axes)) != tuple(sorted(SelectiveDependenceResponseAxisName)):
            raise ValueError("target evaluation lacks an independent axis")
        fixed_target_ids = (
            self.forecast_score.target_id,
            self.selective_signature.target_id,
            self.adjudication.target_id,
            self.handoff.target_id,
            self.comparator_family_advantage.target_id,
        )
        if (
            any(value != self.target_id for value in fixed_target_ids)
            or any(value.target_id != self.target_id for value in self.comparator_scores)
            or any(value.target_id != self.target_id for value in self.denominator_readouts)
        ):
            raise ValueError("target evaluation bundle crosses targets")
        if self.maximum_evidence_ceiling is not EvidenceCeiling.LOCAL_LAW:
            raise ValueError("selective dependence response target evaluation is capped at local law")
        if self.outcome_access is not OutcomeAccess.EVALUATION_REVEALED:
            raise ValueError("target evaluation bundle must remain outcome visible")


def derive_evaluation_package(
    *,
    development_completion: SelectiveDependenceResponseDevelopmentCompletionEnvelope,
    study_completion: SelectiveDependenceResponseStudyForecastCompletionEnvelope,
    design: CanonicalRecord,
    design_id: str,
    analysis_freeze: SelectiveDependenceResponseTargetAnalysisFreeze,
    construct_review: SelectiveDependenceResponseConstructReviewAttestation,
    source_qualification: CanonicalRecord,
    source_qualification_id: str,
    implementation_sha256: str,
) -> SelectiveDependenceResponseEvaluationPackage:
    """Derive the follow-up without permitting any scientific override."""

    bundle = development_completion.development_bundle
    programme_qualification = study_completion.study_qualification
    if not bundle.evaluation_eligible or bundle.forecast is None:
        raise ValueError("development parent is not eligible for evaluation")
    if not development_completion.evaluation_eligible:
        raise ValueError("development completion is not eligible for evaluation")
    if not study_completion.qualified:
        raise ValueError("programme completion did not qualify evaluation issue")
    if (
        development_completion.development_implementation_sha256
        != bundle.parent.implementation_sha256
        or study_completion.study_implementation_sha256 != implementation_sha256
        or development_completion.development_implementation_sha256 != implementation_sha256
    ):
        raise ValueError("evaluation package implementation lineage differs")
    parent = bundle.parent
    forecast = bundle.forecast
    if not programme_qualification.qualified or bundle.target_id not in (
        programme_qualification.target_ids
    ):
        raise ValueError("target is outside the qualified two-target programme forecast")
    forecast_identity = ObjectIdentity.from_record(forecast.forecast_id, forecast)
    if forecast_identity not in programme_qualification.target_forecasts:
        raise ValueError("target forecast differs from the programme qualification")
    if forecast.evaluation_unit_ids_sha256 != parent.evaluation_complete_unit_ids_sha256:
        raise ValueError("forecast and development parent bind different evaluation rosters")
    if (
        forecast.method_question != parent.method_question
        or forecast.contamination_ledger != parent.contamination_ledger
        or forecast.preparation_freeze != parent.preparation_freeze
        or forecast.analysis_freeze != parent.analysis_freeze
        or forecast.design != parent.design
        or forecast.construct_review != parent.construct_review
        or forecast.selected_role_map != parent.selected_role_map
        or forecast.role_map_selection != parent.role_map_selection
        or forecast.denominator_selection != parent.denominator_selection
        or forecast.target_relation_id != parent.target_relation_id
    ):
        raise ValueError("forecast scientific lineage differs from the development parent")
    if parent.denominator_selection != ObjectIdentity.from_record(
        bundle.denominator.selection_id, bundle.denominator
    ):
        raise ValueError("development bundle denominator differs from its parent")
    if construct_review.target_id != bundle.target_id:
        raise ValueError("evaluation package construct review crosses targets")
    if construct_review.role_map != parent.selected_role_map:
        raise ValueError("evaluation package role map differs from development")
    if construct_review.role_map_selection != parent.role_map_selection:
        raise ValueError("evaluation package role-map selection differs from development")
    if ObjectIdentity.from_record(design_id, design) != parent.design:
        raise ValueError("evaluation package design differs from development")
    analysis_identity = ObjectIdentity.from_record(analysis_freeze.freeze_id, analysis_freeze)
    if (
        analysis_freeze.target_id != bundle.target_id
        or analysis_identity != parent.analysis_freeze
        or analysis_identity != forecast.analysis_freeze
        or analysis_freeze.design != parent.design
        or analysis_freeze.preparation_freeze != parent.preparation_freeze
        or analysis_freeze.minimum_law_accuracy != bundle.law.minimum_required_accuracy
    ):
        raise ValueError("evaluation package analysis freeze differs from development")
    source_identity = ObjectIdentity.from_record(source_qualification_id, source_qualification)
    if source_identity != parent.source_qualification:
        raise ValueError("evaluation package source differs from the development parent")
    return SelectiveDependenceResponseEvaluationPackage(
        package_id=f"package.{bundle.target_id}.evaluation",
        target_id=bundle.target_id,
        study_forecast_completion=ObjectIdentity.from_record(
            study_completion.envelope_id,
            study_completion,
        ),
        study_forecast_qualification=ObjectIdentity.from_record(
            programme_qualification.qualification_id, programme_qualification
        ),
        development_completion=ObjectIdentity.from_record(
            development_completion.envelope_id,
            development_completion,
        ),
        development_bundle=ObjectIdentity.from_record(bundle.bundle_id, bundle),
        development_parent=ObjectIdentity.from_record(parent.parent_id, parent),
        forecast=ObjectIdentity.from_record(forecast.forecast_id, forecast),
        design=ObjectIdentity.from_record(design_id, design),
        analysis_freeze=analysis_identity,
        construct_review=ObjectIdentity.from_record(
            construct_review.attestation_id, construct_review
        ),
        source_qualification=source_identity,
        comparator_encodings=tuple(
            sorted(
                (
                    ObjectIdentity.from_record(value.encoding_id, value)
                    for value in bundle.comparators
                ),
                key=lambda value: value.object_id,
            )
        ),
        development_implementation_sha256=parent.implementation_sha256,
        evaluation_implementation_sha256=implementation_sha256,
        development_receipt_ids=parent.development_receipt_ids,
        evaluation_complete_unit_ids=parent.evaluation_complete_unit_ids,
        evaluation_complete_unit_ids_sha256=parent.evaluation_complete_unit_ids_sha256,
        minimum_predictive_accuracy=bundle.law.minimum_required_accuracy,
        scientific_override_count=0,
        evaluation_outcome_count=0,
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
    )


def build_sealed_index(
    *,
    package: SelectiveDependenceResponseEvaluationPackage,
    shard_manifest: SelectiveDependenceResponseSealedShardManifest,
    generator_receipt_ids: tuple[str, ...],
    generator_materialization_ids: tuple[str, ...],
) -> SelectiveDependenceResponseSealedIndex:
    if shard_manifest.target_id != package.target_id:
        raise ValueError("sealed index crosses targets")
    observed = shard_manifest.complete_unit_ids
    expected = package.evaluation_complete_unit_ids
    surplus = len(set(observed) - set(expected))
    return SelectiveDependenceResponseSealedIndex(
        index_id=f"index.{package.target_id}.evaluation-sealed",
        target_id=package.target_id,
        evaluation_package=ObjectIdentity.from_record(package.package_id, package),
        sealed_panel=shard_manifest.sealed_panel,
        expected_complete_unit_ids=expected,
        observed_complete_unit_ids=observed,
        complete_unit_ids_sha256=package.evaluation_complete_unit_ids_sha256,
        generator_receipt_ids=tuple(sorted(generator_receipt_ids)),
        generator_materialization_ids=tuple(sorted(generator_materialization_ids)),
        duplicate_complete_unit_count=0,
        surplus_complete_unit_count=surplus,
        full_fan_in=observed == expected and surplus == 0,
        outcome_access=OutcomeAccess.EVALUATION_SEALED,
    )


def _forecast_predictions(
    bundle: SelectiveDependenceResponseDevelopmentBundle,
) -> tuple[SelectiveDependenceResponseCategoricalPrediction, ...]:
    if bundle.forecast is None:
        raise ValueError("evaluation requires a frozen forecast")
    return tuple(
        sorted(
            (
                SelectiveDependenceResponseCategoricalPrediction(
                    prediction_id=f"prediction.{bundle.target_id}.selective-dependence-response.{value.cell_id}",
                    cell_id=value.cell_id,
                    response_state=value.expected_state,
                    fibre_admitted=value.expected_fibre_admitted,
                    disposition=value.expected_disposition,
                )
                for value in bundle.forecast.cell_forecasts
            ),
            key=lambda value: value.prediction_id,
        )
    )


def _denominator_readout(
    score: SelectiveDependenceResponsePredictiveScore,
    panel: SelectiveDependenceResponseTargetPanel,
) -> SelectiveDependenceResponseDenominatorReadout:
    issued = len(panel.expected_complete_unit_ids)
    source_valid = len(panel.complete_units)
    evaluable = score.evaluable_complete_unit_count
    stopped = source_valid - evaluable
    missing = issued - source_valid
    total_accuracy = sum((value.value for value in score.complete_unit_accuracies), Decimal(0))
    worst = total_accuracy / Decimal(issued)
    best = (total_accuracy + Decimal(stopped + missing)) / Decimal(issued)
    return SelectiveDependenceResponseDenominatorReadout(
        readout_id=f"denominator.{panel.target_id}.{score.model_id}",
        target_id=panel.target_id,
        model_id=score.model_id,
        issued_complete_unit_count=issued,
        source_valid_complete_unit_count=source_valid,
        evaluable_complete_unit_count=evaluable,
        stopped_complete_unit_count=stopped,
        missing_complete_unit_count=missing,
        evaluable_mean_accuracy=score.mean_complete_unit_accuracy,
        issued_worst_case_mean_accuracy=worst,
        issued_best_case_mean_accuracy=best,
        stopped_or_missing_count_as_failures_in_primary=True,
        nested_cases_count_as_units=False,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
    )


def _comparator_family_advantage(
    forecast: SelectiveDependenceResponsePredictiveScore,
    comparators: tuple[SelectiveDependenceResponsePredictiveScore, ...],
    *,
    comparator_encodings: tuple[SelectiveDependenceResponseComparatorEncoding, ...],
    selective_signature: SelectiveDependenceResponseSelectiveDependenceSignature,
    exchange_forecasts: tuple[SelectiveDependenceResponseExchangeForecast, ...],
    one_sided_alpha: Decimal,
) -> SelectiveDependenceResponseComparatorFamilyAdvantage:
    primary = {value.value_id: value.value for value in forecast.complete_unit_accuracies}
    alternatives = [
        {value.value_id: value.value for value in score.complete_unit_accuracies}
        for score in comparators
    ]
    if not primary or any(set(value) != set(primary) for value in alternatives):
        raise ValueError("paired comparator complete-unit rosters differ")
    values = tuple(
        NamedDecimal(
            value_id=unit_id,
            value=primary[unit_id] - max(value[unit_id] for value in alternatives),
            unit="paired-fraction-correct-advantage",
        )
        for unit_id in sorted(primary)
    )
    numeric = tuple(float(value.value) for value in values)
    mean = sum(numeric) / len(numeric)
    variance = sum((value - mean) ** 2 for value in numeric) / (len(numeric) - 1)
    critical = float(student_t.ppf(1 - float(one_sided_alpha), df=len(numeric) - 1))
    lower = mean - critical * sqrt(variance / len(numeric))
    mean_decimal = Decimal(format(mean, ".12g"))
    lower_decimal = Decimal(format(lower, ".12g"))
    scores_by_model = {value.model_id: value for value in comparators}
    encodings_by_model = {
        value.kind.value.lower().replace("_", "-"): value for value in comparator_encodings
    }
    if set(scores_by_model) != set(encodings_by_model):
        raise ValueError("comparator score and encoding families differ")
    primary_exchange_expectations = {
        value.exchange_id: value.expectation for value in exchange_forecasts
    }
    assessments = {value.exchange_id: value for value in selective_signature.exchange_assessments}
    comparisons = []
    for model_id in sorted(scores_by_model):
        score = scores_by_model[model_id]
        encoding = encodings_by_model[model_id]
        alternate_by_unit = {
            value.value_id: value.value for value in score.complete_unit_accuracies
        }
        paired = tuple(
            NamedDecimal(
                value_id=unit_id,
                value=primary[unit_id] - alternate_by_unit[unit_id],
                unit="paired-fraction-correct-advantage",
            )
            for unit_id in sorted(primary)
        )
        paired_numeric = tuple(float(value.value) for value in paired)
        paired_mean = sum(paired_numeric) / len(paired_numeric)
        paired_variance = sum((value - paired_mean) ** 2 for value in paired_numeric) / (
            len(paired_numeric) - 1
        )
        paired_half = critical * sqrt(paired_variance / len(paired_numeric))
        paired_lower = paired_mean - paired_half
        paired_upper = paired_mean + paired_half
        alternate_exchange = {
            value.exchange_id: value.expectation for value in encoding.exchange_predictions
        }
        relevant_exchange_ids = tuple(
            sorted(
                exchange_id
                for exchange_id, expectation in alternate_exchange.items()
                if exchange_id in assessments
                and primary_exchange_expectations.get(exchange_id) != expectation
            )
        )
        opposed_exchange_ids = []
        for exchange_id in relevant_exchange_ids:
            primary_assessment = assessments[exchange_id]
            alternate_expectation = alternate_exchange[exchange_id]
            if primary_assessment.state is not SelectiveDependenceResponseExchangeState.SUPPORTED:
                continue
            if alternate_expectation is None:
                opposed_exchange_ids.append(exchange_id)
                continue
            alternate_assessment = assess_exchange(
                assessment_id=f"assessment.{exchange_id}.{model_id}",
                exchange_id=exchange_id,
                expectation=alternate_expectation,
                point=primary_assessment.point,
                lower=primary_assessment.lower,
                upper=primary_assessment.upper,
                equivalence_margin=primary_assessment.equivalence_margin,
                minimum_active_difference=primary_assessment.minimum_active_difference,
                complete_unit_count=primary_assessment.complete_unit_count,
            )
            if alternate_assessment.state is SelectiveDependenceResponseExchangeState.OPPOSED:
                opposed_exchange_ids.append(exchange_id)
        comparison_lower = Decimal(format(paired_lower, ".12g"))
        comparison_upper = Decimal(format(paired_upper, ".12g"))
        comparison_mean = Decimal(format(paired_mean, ".12g"))
        opposed = tuple(sorted(opposed_exchange_ids))
        comparisons.append(
            SelectiveDependenceResponseComparatorSpecificAdvantage(
                advantage_id=f"advantage.{forecast.target_id}.{model_id}",
                target_id=forecast.target_id,
                comparator_model_id=model_id,
                complete_unit_advantages=paired,
                mean_advantage=comparison_mean,
                simultaneous_lower=comparison_lower,
                simultaneous_upper=comparison_upper,
                relevant_exchange_ids=relevant_exchange_ids,
                comparator_opposed_exchange_ids=opposed,
                distinguished=comparison_lower > 0 or bool(opposed),
                comparator_better=comparison_upper < 0 and not opposed,
                nested_cases_count_as_units=False,
                outcome_access=OutcomeAccess.EVALUATION_REVEALED,
            )
        )
    ordered_comparisons = tuple(sorted(comparisons, key=lambda value: value.advantage_id))
    return SelectiveDependenceResponseComparatorFamilyAdvantage(
        advantage_id=f"advantage.{forecast.target_id}.best-comparator-family",
        target_id=forecast.target_id,
        comparator_model_ids=tuple(sorted(value.model_id for value in comparators)),
        complete_unit_advantages=values,
        complete_unit_ids_sha256=digest_ids(tuple(value.value_id for value in values)),
        mean_advantage=mean_decimal,
        simultaneous_lower=lower_decimal,
        one_sided_alpha=one_sided_alpha,
        comparator_advantages=ordered_comparisons,
        comparator_better_ids=tuple(
            sorted(
                value.comparator_model_id
                for value in ordered_comparisons
                if value.comparator_better
            )
        ),
        distinguished=all(value.distinguished for value in ordered_comparisons),
        nested_cases_count_as_units=False,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
    )


def _measurement_action_axis(panel: SelectiveDependenceResponseTargetPanel) -> SelectiveDependenceResponseAxisAdjudication:
    conditions = tuple(condition for unit in panel.complete_units for condition in unit.conditions)
    invalid = []
    for unit in panel.complete_units:
        for condition in unit.conditions:
            action = condition.action
            adverse = (
                action.action_id != condition.action_id
                or action.acceptance_state not in {"accepted", "rejected-to-hold"}
                or not (
                    action.requested_clock
                    <= action.accepted_clock
                    <= action.applied_clock
                    <= action.realized_clock
                )
                or action.applied_unit == action.realized_unit
            )
            if action.acceptance_state == "rejected-to-hold":
                holds = [
                    value
                    for value in unit.conditions
                    if value.denominator_id == condition.denominator_id
                    and value.history_id == condition.history_id
                    and value.horizon_id == condition.horizon_id
                    and value.action.acceptance_state == "accepted"
                    and value.action.action_id.endswith("hold")
                ]
                adverse |= len(holds) != 1 or any(
                    (
                        action.accepted_value != hold.action.accepted_value
                        or action.applied_value != hold.action.applied_value
                        or action.realized_value != hold.action.realized_value
                    )
                    for hold in holds
                )
            if adverse:
                invalid.append(f"case.{unit.complete_unit_id}.{condition.condition_id}")
    invalid_ids = tuple(sorted(invalid))
    if invalid_ids:
        state = SelectiveDependenceResponseAxisState.INVALID
        reason = "At least one requested/accepted/applied/realized action chain was invalid."
    elif any(condition.stopped for condition in conditions):
        state = SelectiveDependenceResponseAxisState.PARTIAL
        reason = "The action chain was stage-resolved, but at least one branch stopped."
    else:
        state = SelectiveDependenceResponseAxisState.QUALIFIED
        reason = "Every branch retained a causal requested/accepted/applied/realized chain."
    return axis(
        panel.target_id,
        SelectiveDependenceResponseAxisName.MEASUREMENT_ACTION_CHAIN,
        state,
        decisive_case_ids=invalid_ids,
        reason=reason,
    )


def _outside_support_counterexamples(panel: SelectiveDependenceResponseTargetPanel) -> tuple[str, ...]:
    return tuple(
        sorted(
            f"case.{unit.complete_unit_id}.{condition_cell_id(condition).removeprefix('cell.')}"
            for unit in panel.complete_units
            for condition in unit.conditions
            if condition.support_state == "outside-support"
            and (
                condition.disposition is not SelectiveDependenceResponseDisposition.NONATTEMPT
                or condition.action.acceptance_state != "rejected-to-hold"
            )
        )
    )


def _validate_selective_signature_against_forecast(
    development: SelectiveDependenceResponseDevelopmentBundle,
    signature: SelectiveDependenceResponseSelectiveDependenceSignature,
) -> None:
    """Reject reducer drift from the frozen exchange family and thresholds."""

    forecast = development.forecast
    if forecast is None:
        raise ValueError("selective signature validation requires a frozen forecast")
    if forecast.exchange_interval_method != "bonferroni-complete-unit-student-t":
        raise ValueError("unknown frozen exchange interval method")
    expected = {
        value.exchange_id: value
        for value in forecast.exchange_forecasts
        if value.expectation is not SelectiveDependenceResponseExchangeExpectation.SUPPORT_BOUNDARY
    }
    observed = {value.exchange_id: value for value in signature.exchange_assessments}
    if set(expected) != set(observed):
        raise ValueError("evaluation exchange roster differs from the frozen forecast")
    for exchange_id, specification in expected.items():
        assessment = observed[exchange_id]
        if (
            assessment.expectation is not specification.expectation
            or assessment.equivalence_margin != specification.equivalence_margin
            or assessment.minimum_active_difference != specification.minimum_active_difference
        ):
            raise ValueError("evaluation exchange rule differs from the frozen forecast")
    expected_confidence = bonferroni_confidence_level(
        familywise_alpha=forecast.exchange_familywise_alpha,
        family_size=len(expected),
    )
    estimates = {
        value.estimate_id.removeprefix("estimate."): value for value in signature.estimates
    }
    if set(estimates) != set(expected):
        raise ValueError("evaluation estimate roster differs from the frozen forecast")
    if any(
        value.confidence_level != expected_confidence
        or value.native_unit != expected[exchange_id].native_unit
        for exchange_id, value in estimates.items()
    ):
        raise ValueError("evaluation exchange interval or native unit drifted")


def evaluate_target(
    *,
    package: SelectiveDependenceResponseEvaluationPackage,
    development: SelectiveDependenceResponseDevelopmentBundle,
    construct_review: SelectiveDependenceResponseConstructReviewAttestation,
    sealed_index: SelectiveDependenceResponseSealedIndex,
    panel: SelectiveDependenceResponseTargetPanel,
    reveal_authority: ObjectIdentity,
    evaluator_task_id: str,
    evaluator_attempt_id: str,
    input_receipt_ids: tuple[str, ...],
    input_materialization_ids: tuple[str, ...],
    analysis_freeze: SelectiveDependenceResponseTargetAnalysisFreeze,
    source_family_id: str,
    solver_family_id: str,
) -> SelectiveDependenceResponseTargetEvaluationBundle:
    """Open one exact panel and apply only frozen, target-neutral rules."""

    if panel.target_id != package.target_id or development.target_id != package.target_id:
        raise ValueError("evaluation inputs cross targets")
    if (
        analysis_freeze.target_id != package.target_id
        or package.analysis_freeze
        != ObjectIdentity.from_record(analysis_freeze.freeze_id, analysis_freeze)
    ):
        raise ValueError("evaluation analysis freeze identity differs")
    if panel.phase is not SelectiveDependenceResponsePhase.EVALUATION:
        raise ValueError("target evaluator received a non-evaluation panel")
    if panel.expected_complete_unit_ids != package.evaluation_complete_unit_ids:
        raise ValueError("evaluation panel differs from the issued roster")
    if not sealed_index.full_fan_in or sealed_index.sealed_panel != ObjectIdentity.from_record(
        panel.panel_id, panel
    ):
        raise ValueError("evaluation panel differs from the verified sealed index")
    if construct_review.target_id != package.target_id or package.construct_review != (
        ObjectIdentity.from_record(construct_review.attestation_id, construct_review)
    ):
        raise ValueError("evaluation construct review identity differs")
    if development.forecast is None or package.forecast != ObjectIdentity.from_record(
        development.forecast.forecast_id, development.forecast
    ):
        raise ValueError("evaluation forecast identity differs")

    forecast_score = score_categorical_predictions(
        panel,
        model_id="selective-dependence-response-law",
        predictions=_forecast_predictions(development),
        hold_action_id=analysis_freeze.hold_action_id,
        receiver_ids=analysis_freeze.scored_receiver_ids,
        neutral_margin=analysis_freeze.neutral_margin,
    )
    comparator_scores = tuple(
        sorted(
            (
                score_categorical_predictions(
                    panel,
                    model_id=value.kind.value.lower().replace("_", "-"),
                    predictions=value.predictions,
                    hold_action_id=analysis_freeze.hold_action_id,
                    receiver_ids=analysis_freeze.scored_receiver_ids,
                    neutral_margin=analysis_freeze.neutral_margin,
                )
                for value in development.comparators
            ),
            key=lambda value: value.score_id,
        )
    )
    signature = reduce_selective_signature(
        panel,
        analysis_freeze,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
    )
    _validate_selective_signature_against_forecast(development, signature)
    comparator_advantage = _comparator_family_advantage(
        forecast_score,
        comparator_scores,
        comparator_encodings=development.comparators,
        selective_signature=signature,
        exchange_forecasts=development.forecast.exchange_forecasts,
        one_sided_alpha=(
            development.power.familywise_alpha / Decimal(len(development.comparators))
        ),
    )
    issued = len(package.evaluation_complete_unit_ids)
    evaluable = forecast_score.evaluable_complete_unit_count
    source_valid = len(panel.complete_units)
    stopped = source_valid - evaluable
    missing = issued - source_valid
    complete = evaluable == issued and not stopped and not missing

    active_assessments = tuple(
        value
        for value in signature.exchange_assessments
        if value.exchange_id in signature.active_exchange_ids
    )
    invariant_assessments = tuple(
        value
        for value in signature.exchange_assessments
        if value.exchange_id in signature.invariant_exchange_ids
    )
    exchange_states = tuple(value.state for value in (*active_assessments, *invariant_assessments))
    if any(value is SelectiveDependenceResponseExchangeState.OPPOSED for value in exchange_states):
        selective_state = SelectiveDependenceResponseAxisState.OPPOSED
    elif any(value is SelectiveDependenceResponseExchangeState.UNEVALUABLE for value in exchange_states):
        selective_state = SelectiveDependenceResponseAxisState.UNEVALUABLE
    else:
        selective_state = SelectiveDependenceResponseAxisState.SUPPORTED

    if not complete:
        law_state = SelectiveDependenceResponseAxisState.UNEVALUABLE
        admission_state = SelectiveDependenceResponseAxisState.UNEVALUABLE
        hold_state = SelectiveDependenceResponseAxisState.UNEVALUABLE
    else:
        law_state = (
            SelectiveDependenceResponseAxisState.SUPPORTED
            if not forecast_score.incorrect_case_ids
            and not forecast_score.incorrect_context_case_ids
            and forecast_score.mean_complete_unit_accuracy >= package.minimum_predictive_accuracy
            else SelectiveDependenceResponseAxisState.OPPOSED
        )
        disposition_accuracy = Decimal(forecast_score.correct_disposition_count) / Decimal(
            forecast_score.disposition_case_count
        )
        fibre_accuracy = Decimal(forecast_score.correct_fibre_count) / Decimal(
            forecast_score.fibre_case_count
        )
        context_accuracy = Decimal(forecast_score.correct_context_count) / Decimal(
            forecast_score.context_case_count
        )
        admission_state = (
            SelectiveDependenceResponseAxisState.SUPPORTED
            if forecast_score.unsafe_false_admission_count == 0
            and forecast_score.correct_fibre_count == forecast_score.fibre_case_count
            and forecast_score.correct_disposition_count == forecast_score.disposition_case_count
            and forecast_score.correct_context_count == forecast_score.context_case_count
            and fibre_accuracy >= package.minimum_predictive_accuracy
            and disposition_accuracy >= package.minimum_predictive_accuracy
            and context_accuracy >= package.minimum_predictive_accuracy
            else SelectiveDependenceResponseAxisState.OPPOSED
        )
        forecast_hold_states = {
            value.hold_viable for value in development.forecast.context_forecasts
        }
        if None in forecast_hold_states or forecast_score.incorrect_context_case_ids:
            hold_state = SelectiveDependenceResponseAxisState.UNEVALUABLE
        elif forecast_hold_states == {True}:
            hold_state = SelectiveDependenceResponseAxisState.VIABLE
        elif forecast_hold_states == {False}:
            hold_state = SelectiveDependenceResponseAxisState.NOT_VIABLE
        else:
            hold_state = SelectiveDependenceResponseAxisState.MIXED

    if not complete or any(
        value.evaluable_complete_unit_count != issued for value in comparator_scores
    ):
        distinctive_state = SelectiveDependenceResponseAxisState.UNEVALUABLE
    elif comparator_advantage.distinguished:
        distinctive_state = SelectiveDependenceResponseAxisState.DISTINGUISHED
    elif comparator_advantage.comparator_better_ids:
        distinctive_state = SelectiveDependenceResponseAxisState.OPPOSED
    else:
        distinctive_state = SelectiveDependenceResponseAxisState.NOT_DISTINGUISHED

    outside_counterexamples = _outside_support_counterexamples(panel)
    support_state = (
        SelectiveDependenceResponseAxisState.SUPPORTED
        if signature.support_boundary_refused
        else SelectiveDependenceResponseAxisState.OPPOSED
    )
    axes = (
        axis(
            package.target_id,
            SelectiveDependenceResponseAxisName.CONSTRUCT,
            SelectiveDependenceResponseAxisState.VALID,
            reason="An accountable reviewer distinct from the forecast author bound the exact dossier and role map.",
        ),
        _measurement_action_axis(panel),
        axis(
            package.target_id,
            SelectiveDependenceResponseAxisName.LOCAL_RESPONSE_LAW,
            law_state,
            decisive_case_ids=forecast_score.incorrect_case_ids,
            reason="Frozen categorical law accuracy was evaluated on complete preparation units with adverse units retained.",
        ),
        axis(
            package.target_id,
            SelectiveDependenceResponseAxisName.SELECTIVE_DEPENDENCE,
            selective_state,
            decisive_case_ids=tuple(
                sorted(
                    value.assessment_id
                    for value in signature.exchange_assessments
                    if value.state is not SelectiveDependenceResponseExchangeState.SUPPORTED
                )
            ),
            reason="Every predeclared active and invariant exchange was assessed without pooling targets.",
        ),
        axis(
            package.target_id,
            SelectiveDependenceResponseAxisName.SUPPORT_BOUNDARY,
            support_state,
            decisive_case_ids=outside_counterexamples,
            reason="Outside-support actions were required to be rejected to measured hold.",
        ),
        axis(
            package.target_id,
            SelectiveDependenceResponseAxisName.RECEIVER_ADMISSION,
            admission_state,
            decisive_case_ids=forecast_score.unsafe_false_admission_case_ids,
            reason="Admission used the noncompensating target/sink/effort/validity intersection.",
        ),
        axis(
            package.target_id,
            SelectiveDependenceResponseAxisName.HOLD_VIABILITY,
            hold_state,
            decisive_case_ids=forecast_score.false_safe_hold_case_ids,
            reason=(
                "Measured hold-fibre viability and the resulting active/hold/nonattempt "
                "context decisions were scored separately."
            ),
        ),
        axis(
            package.target_id,
            SelectiveDependenceResponseAxisName.PREDICTIVE_DISTINCTIVENESS,
            distinctive_state,
            decisive_case_ids=tuple(
                ()
                if comparator_advantage.distinguished
                else ("comparator-family-best-per-complete-unit",)
            ),
            reason=(
                "A one-sided paired complete-unit interval had to remain above zero "
                "against the best frozen comparator on each unit."
            ),
        ),
    )
    exact_counterexamples = tuple(
        sorted(
            set(
                (
                    *outside_counterexamples,
                    *forecast_score.incorrect_case_ids,
                    *forecast_score.incorrect_context_case_ids,
                    *forecast_score.unsafe_false_admission_case_ids,
                    *forecast_score.false_safe_hold_case_ids,
                )
            )
        )
    )
    forecast_identity = ObjectIdentity.from_record(
        development.forecast.forecast_id, development.forecast
    )
    adjudication = adjudicate_target(
        target_id=package.target_id,
        forecast=forecast_identity,
        evaluation_unit_ids_sha256=package.evaluation_complete_unit_ids_sha256,
        issued_unit_count=issued,
        source_valid_unit_count=source_valid,
        evaluable_unit_count=evaluable,
        stopped_unit_count=stopped,
        missing_unit_count=missing,
        axes=axes,
        exact_counterexample_ids=exact_counterexamples,
    )
    handoff = compact_handoff(
        adjudication,
        source_family_id=source_family_id,
        solver_family_id=solver_family_id,
        forecast_policy_dispositions=tuple(
            sorted(
                {value.disposition for value in development.forecast.context_forecasts},
                key=lambda value: value.value,
            )
        ),
        observed_policy_dispositions=tuple(
            sorted(
                {
                    value.disposition
                    for unit in panel.complete_units
                    for value in unit.context_decisions
                },
                key=lambda value: value.value,
            )
        ),
    )
    reveal = SelectiveDependenceResponseRevealRecord(
        reveal_id=f"reveal.{package.target_id}.evaluation",
        target_id=package.target_id,
        evaluation_package=ObjectIdentity.from_record(package.package_id, package),
        sealed_index=ObjectIdentity.from_record(sealed_index.index_id, sealed_index),
        sealed_panel=ObjectIdentity.from_record(panel.panel_id, panel),
        reveal_authority=reveal_authority,
        evaluator_task_id=evaluator_task_id,
        evaluator_attempt_id=evaluator_attempt_id,
        input_receipt_ids=tuple(sorted(input_receipt_ids)),
        input_materialization_ids=tuple(sorted(input_materialization_ids)),
        revealed_complete_unit_count=issued,
        partial_score_count_before_reveal=0,
        atomic_full_panel_open=True,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
    )
    scores = (forecast_score, *comparator_scores)
    readouts = tuple(
        sorted(
            (_denominator_readout(score, panel) for score in scores),
            key=lambda value: value.readout_id,
        )
    )
    return SelectiveDependenceResponseTargetEvaluationBundle(
        bundle_id=f"bundle.{package.target_id}.evaluation",
        target_id=package.target_id,
        evaluation_package=ObjectIdentity.from_record(package.package_id, package),
        sealed_index=ObjectIdentity.from_record(sealed_index.index_id, sealed_index),
        reveal_record=reveal,
        forecast_score=forecast_score,
        comparator_scores=comparator_scores,
        comparator_family_advantage=comparator_advantage,
        denominator_readouts=readouts,
        selective_signature=signature,
        axes=tuple(sorted(axes, key=lambda value: value.adjudication_id)),
        adjudication=adjudication,
        handoff=handoff,
        maximum_evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
    )


__all__ = [
    'SelectiveDependenceResponseComparatorFamilyAdvantage',
    'SelectiveDependenceResponseComparatorSpecificAdvantage',
    'SelectiveDependenceResponseDenominatorReadout',
    'SelectiveDependenceResponseEvaluationPackage',
    'SelectiveDependenceResponseRevealRecord',
    'SelectiveDependenceResponseSealedIndex',
    'SelectiveDependenceResponseSealedShardManifest',
    'SelectiveDependenceResponseTargetEvaluationBundle',
    "build_sealed_index",
    "derive_evaluation_package",
    "evaluate_target",
]
