"""Canonical in-memory contracts shared by registered law methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar, Protocol

from empirical_lawhood.kernel.evidence import (
    EvidenceCeiling,
    EvidenceRung,
    OutcomeAccess,
    VisibilityCeiling,
    inherited_visibility,
)
from empirical_lawhood.kernel.identification import AdequacyCheckKind, AdequacyCheckResult, LawQualificationResult, LawMethodKind, ResponseQualificationTrace, StructuralConvergenceResult, TerminalObligationKind, TerminalLawObligationAssessment, UncertaintyDecomposition
from empirical_lawhood.kernel.laws import CausalStrength, LawRepresentationKind, ResponseLaw
from empirical_lawhood.kernel.predictive_uncertainty import JointPredictiveUncertainty
from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord
from empirical_lawhood.kernel.obligations import FalsifierKind, ObligationStatus
from empirical_lawhood.kernel.provenance import EvidenceLink, ObjectIdentity
from empirical_lawhood.kernel.references import (
    ArtifactIdentity,
    ExecutableReference,
    NamedDecimal,
    QuantityBound,
)
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    ExtensionBinding,
    require_extensions,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.status import ScientificStatus
from empirical_lawhood.kernel.systems import RelationalIdentity
from empirical_lawhood.kernel.time import HorizonSpec
from empirical_lawhood.runtime.candidate_payloads import (
    CANDIDATE_EVALUATOR_IMPLEMENTATION_SCHEMA,
    CandidatePayloadPublicationReceipt,
)


class DataSplit(StrEnum):
    CALIBRATION = "CALIBRATION"
    HELD_OUT = "HELD_OUT"


class ObservationRole(StrEnum):
    PRIMARY = "PRIMARY"
    WRONG_ACTION = "WRONG_ACTION"
    NEGATIVE_CONTROL = "NEGATIVE_CONTROL"
    WRONG_TIME = "WRONG_TIME"


@dataclass(frozen=True, slots=True)
class ReceiverCriterion(CanonicalRecord):
    """Native-unit gates and irreducible uncertainty for one receiver."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-criterion'

    criterion_id: str
    receiver_quantity_id: str
    native_unit: str
    maximum_held_out_rmse: Decimal
    maximum_wrong_action_effect: Decimal
    minimum_baseline_improvement: Decimal
    aleatoric_uncertainty_bound: Decimal
    aleatoric_bound_method_id: str
    observation_uncertainty_bound: Decimal
    observation_bound_method_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.criterion_id, field_name="criterion_id")
        validate_stable_id(self.receiver_quantity_id, field_name="receiver_quantity_id")
        if not self.native_unit.strip():
            raise ValueError("receiver criterion native_unit must not be empty")
        validate_stable_id(
            self.aleatoric_bound_method_id,
            field_name="aleatoric_bound_method_id",
        )
        validate_stable_id(
            self.observation_bound_method_id,
            field_name="observation_bound_method_id",
        )
        for name, threshold in (
            ("maximum_held_out_rmse", self.maximum_held_out_rmse),
            ("maximum_wrong_action_effect", self.maximum_wrong_action_effect),
            ("minimum_baseline_improvement", self.minimum_baseline_improvement),
            ("aleatoric_uncertainty_bound", self.aleatoric_uncertainty_bound),
            ("observation_uncertainty_bound", self.observation_uncertainty_bound),
        ):
            validate_decimal(threshold, field_name=name, minimum=Decimal(0))


@dataclass(frozen=True, slots=True)
class ResponseSlopeTolerance(CanonicalRecord):
    """One native-unit one-factor-exchange response-slope tolerance."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-slope-tolerance'

    tolerance_id: str
    action_quantity_id: str
    receiver_quantity_id: str
    native_unit: str
    maximum_absolute_difference: Decimal

    def __post_init__(self) -> None:
        for name, value in (
            ("tolerance_id", self.tolerance_id),
            ("action_quantity_id", self.action_quantity_id),
            ("receiver_quantity_id", self.receiver_quantity_id),
        ):
            validate_stable_id(value, field_name=name)
        if not self.native_unit.strip():
            raise ValueError("response-slope tolerance native_unit must not be empty")
        validate_decimal(
            self.maximum_absolute_difference,
            field_name="maximum_absolute_difference",
            minimum=Decimal(0),
        )


@dataclass(frozen=True, slots=True)
class StructuralCoefficientTolerance(CanonicalRecord):
    """Native-unit zero and refinement tolerances for one model coefficient."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/structural-coefficient-tolerance'

    tolerance_id: str
    receiver_quantity_id: str
    term_id: str
    native_unit: str
    zero_absolute_tolerance: Decimal
    maximum_refinement_difference: Decimal

    def __post_init__(self) -> None:
        for name, value in (
            ("tolerance_id", self.tolerance_id),
            ("receiver_quantity_id", self.receiver_quantity_id),
            ("term_id", self.term_id),
        ):
            validate_stable_id(value, field_name=name)
        if not self.native_unit.strip():
            raise ValueError("structural coefficient tolerance native_unit must not be empty")
        for name, threshold in (
            ("zero_absolute_tolerance", self.zero_absolute_tolerance),
            ("maximum_refinement_difference", self.maximum_refinement_difference),
        ):
            validate_decimal(threshold, field_name=name, minimum=Decimal(0))


@dataclass(frozen=True, slots=True)
class LawObservation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/law-observation'

    observation_id: str
    physical_unit_instance_id: str
    denominator_cell_id: str
    chart_id: str
    numerical_view_id: str
    split: DataSplit
    role: ObservationRole
    denominator_values: tuple[NamedDecimal, ...]
    history_values: tuple[NamedDecimal, ...]
    action_values: tuple[NamedDecimal, ...]
    receiver_values: tuple[NamedDecimal, ...]
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("observation_id", self.observation_id),
            ("physical_unit_instance_id", self.physical_unit_instance_id),
            ("denominator_cell_id", self.denominator_cell_id),
            ("chart_id", self.chart_id),
            ("numerical_view_id", self.numerical_view_id),
        ):
            validate_stable_id(value, field_name=name)
        for name, values, allow_empty in (
            ("denominator_values", self.denominator_values, False),
            ("history_values", self.history_values, True),
            ("action_values", self.action_values, False),
            ("receiver_values", self.receiver_values, False),
        ):
            require_sorted_unique_ids(
                values,
                attribute="value_id",
                field_name=name,
            )
            if not allow_empty and not values:
                raise ValueError(f"{name} must not be empty")
        require_sorted_unique_strings(self.tags, field_name="tags")

    def value(self, value_id: str) -> Decimal:
        validate_stable_id(value_id, field_name="value_id")
        for value in (
            *self.denominator_values,
            *self.history_values,
            *self.action_values,
            *self.receiver_values,
        ):
            if value.value_id == value_id:
                return value.value
        raise KeyError(value_id)


@dataclass(frozen=True, slots=True)
class OneFactorExchange(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/one-factor-exchange'

    exchange_id: str
    left_denominator_cell_id: str
    right_denominator_cell_id: str
    exchanged_factor_id: str
    slope_tolerances: tuple[ResponseSlopeTolerance, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("exchange_id", self.exchange_id),
            ("left_denominator_cell_id", self.left_denominator_cell_id),
            ("right_denominator_cell_id", self.right_denominator_cell_id),
            ("exchanged_factor_id", self.exchanged_factor_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.left_denominator_cell_id == self.right_denominator_cell_id:
            raise ValueError("one-factor exchange requires distinct denominator cells")
        require_sorted_unique_ids(
            self.slope_tolerances,
            attribute="tolerance_id",
            field_name="slope_tolerances",
        )
        if not self.slope_tolerances:
            raise ValueError("one-factor exchange requires response-slope tolerances")


@dataclass(frozen=True, slots=True)
class IdentificationDataset(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/identification-dataset'

    dataset_id: str
    system: ObjectIdentity
    relation: RelationalIdentity
    information_cutoff_id: str
    observations: tuple[LawObservation, ...]
    one_factor_exchanges: tuple[OneFactorExchange, ...]
    evidence_artifacts: tuple[ArtifactIdentity, ...]
    outcome_access: OutcomeAccess
    parent_visibility_ceilings: tuple[VisibilityCeiling, ...]
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.dataset_id, field_name="dataset_id")
        validate_stable_id(self.information_cutoff_id, field_name="information_cutoff_id")
        require_sorted_unique_ids(
            self.observations,
            attribute="observation_id",
            field_name="observations",
        )
        if not self.observations:
            raise ValueError("identification dataset requires observations")
        require_sorted_unique_ids(
            self.one_factor_exchanges,
            attribute="exchange_id",
            field_name="one_factor_exchanges",
        )
        require_sorted_unique_ids(
            self.evidence_artifacts,
            attribute="artifact_id",
            field_name="evidence_artifacts",
        )
        if not self.evidence_artifacts:
            raise ValueError("identification dataset requires external evidence identities")
        if {observation.split for observation in self.observations} != set(DataSplit):
            raise ValueError("identification dataset requires calibration and held-out splits")
        if not any(
            observation.role is ObservationRole.PRIMARY for observation in self.observations
        ):
            raise ValueError("identification dataset requires primary observations")
        inherited = inherited_visibility(self.parent_visibility_ceilings, self.outcome_access)
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(inherited):
            raise ValueError("identification dataset visibility cannot be lowered")

    def selected(
        self,
        *,
        chart_id: str,
        split: DataSplit | None = None,
        role: ObservationRole | None = None,
        numerical_view_id: str | None = None,
    ) -> tuple[LawObservation, ...]:
        validate_stable_id(chart_id, field_name="chart_id")
        if numerical_view_id is not None:
            validate_stable_id(numerical_view_id, field_name="numerical_view_id")
        return tuple(
            observation
            for observation in self.observations
            if observation.chart_id == chart_id
            and (split is None or observation.split is split)
            and (role is None or observation.role is role)
            and (numerical_view_id is None or observation.numerical_view_id == numerical_view_id)
        )

    @property
    def physical_unit_instance_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    observation.physical_unit_instance_id
                    for observation in self.observations
                    if observation.role is ObservationRole.PRIMARY
                }
            )
        )


@dataclass(frozen=True, slots=True)
class LawIdentificationConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/law-identification-config'

    config_id: str
    method_key: str
    method_version: str
    method_kind: LawMethodKind
    chart_id: str
    feature_quantity_ids: tuple[str, ...]
    receiver_quantity_ids: tuple[str, ...]
    receiver_criteria: tuple[ReceiverCriterion, ...]
    action_bounds: tuple[QuantityBound, ...]
    structural_tolerances: tuple[StructuralCoefficientTolerance, ...]
    mapping_assumption_ids: tuple[str, ...]
    polynomial_degree: int
    minimum_recurrence_units: int
    minimum_action_levels: int
    uncertainty_confidence_level: Decimal
    maximum_residual_history_correlation: Decimal
    causal_strength: CausalStrength
    representation_kind: LawRepresentationKind

    def __post_init__(self) -> None:
        self._validate_registries()
        self._validate_model_shape()
        self._validate_numeric_gates()

    def _validate_registries(self) -> None:
        for name, value in (
            ("config_id", self.config_id),
            ("method_key", self.method_key),
            ("chart_id", self.chart_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_semantic_version(self.method_version)
        require_sorted_unique_strings(
            self.feature_quantity_ids,
            field_name="feature_quantity_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.receiver_quantity_ids,
            field_name="receiver_quantity_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(
            self.receiver_criteria,
            attribute="criterion_id",
            field_name="receiver_criteria",
        )
        if {criterion.receiver_quantity_id for criterion in self.receiver_criteria} != set(
            self.receiver_quantity_ids
        ):
            raise ValueError("each configured receiver requires exactly one native-unit criterion")
        if len(self.receiver_criteria) != len(self.receiver_quantity_ids):
            raise ValueError("receiver criteria cannot duplicate a receiver")
        require_sorted_unique_ids(
            self.action_bounds,
            attribute="bound_id",
            field_name="action_bounds",
        )
        if not self.action_bounds:
            raise ValueError("law identification requires action bounds")
        require_sorted_unique_ids(
            self.structural_tolerances,
            attribute="tolerance_id",
            field_name="structural_tolerances",
        )
        if not self.structural_tolerances:
            raise ValueError("law identification requires native-unit structural tolerances")
        require_sorted_unique_strings(
            self.mapping_assumption_ids,
            field_name="mapping_assumption_ids",
            allow_empty=False,
        )

    def _validate_model_shape(self) -> None:
        if self.polynomial_degree not in {1, 2}:
            raise ValueError("baseline law methods support polynomial degree one or two")
        expected_terms = {"intercept", *(f"linear.{value}" for value in self.feature_quantity_ids)}
        if self.polynomial_degree == 2:
            expected_terms.update(f"quadratic.{value}" for value in self.feature_quantity_ids)
        expected_tolerances = {
            (receiver_id, term_id)
            for receiver_id in self.receiver_quantity_ids
            for term_id in expected_terms
        }
        observed_tolerances = {
            (value.receiver_quantity_id, value.term_id) for value in self.structural_tolerances
        }
        if observed_tolerances != expected_tolerances or len(self.structural_tolerances) != len(
            expected_tolerances
        ):
            raise ValueError("structural tolerances must cover every receiver coefficient once")

    def _validate_numeric_gates(self) -> None:
        if self.minimum_recurrence_units < 2:
            raise ValueError("recurrence requires at least two independent units")
        if self.minimum_action_levels < 2:
            raise ValueError("response identification requires action variation")
        for name, threshold in (
            ("uncertainty_confidence_level", self.uncertainty_confidence_level),
            (
                "maximum_residual_history_correlation",
                self.maximum_residual_history_correlation,
            ),
        ):
            validate_decimal(threshold, field_name=name, minimum=Decimal(0))
        if not Decimal(0) < self.uncertainty_confidence_level < Decimal(1):
            raise ValueError("uncertainty confidence level must be strictly between zero and one")
        if self.maximum_residual_history_correlation > Decimal(1):
            raise ValueError("history correlation tolerance cannot exceed one")


@dataclass(frozen=True, slots=True)
class ModelCoefficient(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/model-coefficient'

    coefficient_id: str
    receiver_quantity_id: str
    term_id: str
    value: Decimal
    native_unit: str

    def __post_init__(self) -> None:
        for name, value in (
            ("coefficient_id", self.coefficient_id),
            ("receiver_quantity_id", self.receiver_quantity_id),
            ("term_id", self.term_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_decimal(self.value, field_name="value")
        if not self.native_unit.strip():
            raise ValueError("model coefficient native_unit must not be empty")


@dataclass(frozen=True, slots=True)
class ParametricLawModel(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/parametric-law-model'

    model_id: str
    method_key: str
    method_version: str
    method_kind: LawMethodKind
    chart_id: str
    feature_quantity_ids: tuple[str, ...]
    receiver_quantity_ids: tuple[str, ...]
    term_ids: tuple[str, ...]
    coefficients: tuple[ModelCoefficient, ...]
    numerical_view_id: str

    def __post_init__(self) -> None:
        for name, value in (
            ("model_id", self.model_id),
            ("method_key", self.method_key),
            ("chart_id", self.chart_id),
            ("numerical_view_id", self.numerical_view_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_semantic_version(self.method_version)
        for name, values in (
            ("feature_quantity_ids", self.feature_quantity_ids),
            ("receiver_quantity_ids", self.receiver_quantity_ids),
            ("term_ids", self.term_ids),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)
        require_sorted_unique_ids(
            self.coefficients,
            attribute="coefficient_id",
            field_name="coefficients",
        )
        expected = {
            (receiver_id, term_id)
            for receiver_id in self.receiver_quantity_ids
            for term_id in self.term_ids
        }
        observed = {
            (coefficient.receiver_quantity_id, coefficient.term_id)
            for coefficient in self.coefficients
        }
        if observed != expected:
            raise ValueError("parametric model lacks a complete coefficient matrix")


@dataclass(frozen=True, slots=True)
class PredictionRecord(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/prediction-record'

    observation_id: str
    predicted_receiver_values: tuple[NamedDecimal, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.observation_id, field_name="observation_id")
        require_sorted_unique_ids(
            self.predicted_receiver_values,
            attribute="value_id",
            field_name="predicted_receiver_values",
        )
        if not self.predicted_receiver_values:
            raise ValueError("prediction record requires receiver values")


@dataclass(frozen=True, slots=True)
class CandidateFit(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/candidate-fit'

    fit_id: str
    model: ParametricLawModel
    calibration_predictions: tuple[PredictionRecord, ...]
    held_out_predictions: tuple[PredictionRecord, ...]
    metrics: tuple[NamedDecimal, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.fit_id, field_name="fit_id")
        for name, values in (
            ("calibration_predictions", self.calibration_predictions),
            ("held_out_predictions", self.held_out_predictions),
        ):
            require_sorted_unique_ids(
                values,
                attribute="observation_id",
                field_name=name,
            )
            if not values:
                raise ValueError(f"{name} must not be empty")
        require_sorted_unique_ids(self.metrics, attribute="value_id", field_name="metrics")


class LawIdentifier(Protocol):
    method_key: str
    method_version: str
    method_kind: LawMethodKind

    def fit(
        self,
        dataset: IdentificationDataset,
        config: LawIdentificationConfig,
        *,
        numerical_view_id: str,
    ) -> CandidateFit: ...


class CandidateRosterDisposition(StrEnum):
    ASSESS_REQUIRED = "ASSESS_REQUIRED"
    PREDECLARED_INAPPLICABLE = "PREDECLARED_INAPPLICABLE"


class CandidateQualificationEligibility(StrEnum):
    PASSING = "PASSING"
    NONPASSING = "NONPASSING"
    UNEVALUABLE = "UNEVALUABLE"
    NONPROMOTABLE = "NONPROMOTABLE"


class CandidateSelectionDisposition(StrEnum):
    SELECTED = "SELECTED"
    NO_SELECTION = "NO_SELECTION"


@dataclass(frozen=True, slots=True)
class LawCandidateAxisBinding(CanonicalRecord):
    """One candidate-version to denominator-member/refinement mapping."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/law-candidate-axis-binding'

    binding_id: str
    candidate_version_member_id: str
    denominator_member_id: str
    qualification_view_ids: tuple[str, ...]
    claimed_property_ids: tuple[str, ...]
    nontransported_property_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("binding_id", self.binding_id),
            ("candidate_version_member_id", self.candidate_version_member_id),
            ("denominator_member_id", self.denominator_member_id),
        ):
            validate_stable_id(value, field_name=name)
        for name, values in (
            ("qualification_view_ids", self.qualification_view_ids),
            ("claimed_property_ids", self.claimed_property_ids),
            ("nontransported_property_ids", self.nontransported_property_ids),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)
        if set(self.claimed_property_ids) & set(self.nontransported_property_ids):
            raise ValueError("axis property cannot be claimed and nontransported")


@dataclass(frozen=True, slots=True)
class LawCandidateAxisMap(CanonicalRecord):
    "Closed three-axis topology used by qualification and downstream admission."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/law-candidate-axis-map'

    axis_map_id: str
    bindings: tuple[LawCandidateAxisBinding, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.axis_map_id, field_name="axis_map_id")
        require_sorted_unique_ids(
            self.bindings,
            attribute="candidate_version_member_id",
            field_name="bindings",
        )
        if not self.bindings:
            raise ValueError("law candidate axis map requires bindings")
        denominator_views: dict[str, set[str]] = {}
        for binding in self.bindings:
            views = denominator_views.setdefault(binding.denominator_member_id, set())
            if views & set(binding.qualification_view_ids):
                raise ValueError("qualification view is duplicated within a denominator member")
            views.update(binding.qualification_view_ids)

    @property
    def candidate_version_member_ids(self) -> tuple[str, ...]:
        return tuple(value.candidate_version_member_id for value in self.bindings)

    @property
    def denominator_member_ids(self) -> tuple[str, ...]:
        return tuple(sorted({value.denominator_member_id for value in self.bindings}))

    @property
    def qualification_view_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted({view for value in self.bindings for view in value.qualification_view_ids})
        )


@dataclass(frozen=True, slots=True)
class CandidateEvaluatorImplementation(CanonicalRecord):
    SCHEMA: ClassVar[str] = CANDIDATE_EVALUATOR_IMPLEMENTATION_SCHEMA

    implementation_id: str
    capability_key: str
    capability_version: str
    evaluator_key: str
    source_sha256: str

    def __post_init__(self) -> None:
        for name, value in (
            ("implementation_id", self.implementation_id),
            ("capability_key", self.capability_key),
            ("evaluator_key", self.evaluator_key),
        ):
            validate_stable_id(value, field_name=name)
        validate_semantic_version(self.capability_version)
        validate_sha256(self.source_sha256, field_name="source_sha256")


class MethodEvidenceAvailability(StrEnum):
    OBSERVED = "OBSERVED"
    UNAVAILABLE = "UNAVAILABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True, slots=True)
class CandidateMethodEvidenceReceipt(CanonicalRecord):
    """Method evidence only; deliberately has no generic adequacy disposition."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/candidate-method-evidence-receipt'

    receipt_id: str
    evidence_kind_id: str
    metrics: tuple[NamedDecimal, ...]
    artifact_ids: tuple[str, ...]
    evidence_link_ids: tuple[str, ...]
    method_reason_codes: tuple[str, ...]
    availability: MethodEvidenceAvailability = MethodEvidenceAvailability.OBSERVED

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        validate_stable_id(self.evidence_kind_id, field_name="evidence_kind_id")
        require_sorted_unique_ids(self.metrics, attribute="value_id", field_name="metrics")
        require_sorted_unique_strings(self.artifact_ids, field_name="artifact_ids")
        require_sorted_unique_strings(
            self.evidence_link_ids,
            field_name="evidence_link_ids",
        )
        require_sorted_unique_strings(
            self.method_reason_codes,
            field_name="method_reason_codes",
        )
        if self.availability is MethodEvidenceAvailability.OBSERVED:
            if not self.metrics or self.method_reason_codes:
                raise ValueError("observed method evidence requires metrics and no reasons")
        elif self.availability is MethodEvidenceAvailability.UNAVAILABLE:
            if self.metrics or not self.method_reason_codes:
                raise ValueError("unavailable method evidence requires reasons and no metrics")
        elif self.metrics or self.method_reason_codes:
            raise ValueError("not-applicable method evidence cannot carry metrics or reasons")
        if not self.evidence_link_ids:
            raise ValueError("candidate method evidence requires evidence links")


@dataclass(frozen=True, slots=True)
class CandidateClaimTemplate(CanonicalRecord):
    """Inputs from which the terminal reducer, not the producer, builds a claim."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/candidate-claim-template'

    template_id: str
    terminal_result_id: str
    compatibility_result_id: str | None
    claim_id: str
    law_id: str
    proposition: str
    estimand: str
    promotion_rule: str
    causal_strength: CausalStrength
    requested_evidence_ceiling: EvidenceCeiling
    interface_input_quantity_ids: tuple[str, ...]
    interface_output_quantity_ids: tuple[str, ...]
    mapping_assumption_ids: tuple[str, ...]
    joint_response_sink_effort_distribution_identified: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.template_id, field_name="template_id")
        validate_stable_id(self.terminal_result_id, field_name="terminal_result_id")
        if self.compatibility_result_id is not None:
            validate_stable_id(
                self.compatibility_result_id,
                field_name="compatibility_result_id",
            )
        validate_stable_id(self.claim_id, field_name="claim_id")
        validate_stable_id(self.law_id, field_name="law_id")
        validate_nonempty(self.proposition, field_name="proposition")
        validate_nonempty(self.estimand, field_name="estimand")
        validate_nonempty(self.promotion_rule, field_name="promotion_rule")
        for name, values in (
            ("interface_input_quantity_ids", self.interface_input_quantity_ids),
            ("interface_output_quantity_ids", self.interface_output_quantity_ids),
            ("mapping_assumption_ids", self.mapping_assumption_ids),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)


@dataclass(frozen=True, slots=True)
class FalsifierObligationTemplate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/falsifier-obligation-template'

    falsifier_id: str
    kind: FalsifierKind
    capability_key: str
    description: str
    decisive_rule: str

    def __post_init__(self) -> None:
        validate_stable_id(self.falsifier_id, field_name="falsifier_id")
        validate_stable_id(self.capability_key, field_name="capability_key")
        validate_nonempty(self.description, field_name="description")
        validate_nonempty(self.decisive_rule, field_name="decisive_rule")


@dataclass(frozen=True, slots=True)
class LawObligationTemplate(CanonicalRecord):
    "Status-free facts from which the terminal reducer builds local-law obligations."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/law-obligation-template'

    template_id: str
    obligations_id: str
    support_id: str
    validity_id: str
    uncertainty_id: str
    closure_id: str
    structural_convergence_id: str
    computability_id: str
    independent_unit_id: str
    physical_unit_count: int
    nested_numerical_view_count: int
    information_cutoff_id: str
    chart_ids: tuple[str, ...]
    denominator_cell_ids: tuple[str, ...]
    action_bounds: tuple[QuantityBound, ...]
    validity_domain_ids: tuple[str, ...]
    assumption_ids: tuple[str, ...]
    uncertainty_method_key: str
    uncertainty_confidence_level: Decimal
    interval_quantity_ids: tuple[str, ...]
    uncertainty_limitation_codes: tuple[str, ...]
    falsifiers: tuple[FalsifierObligationTemplate, ...]
    recurrence_cell_ids: tuple[str, ...]
    exchange_factor_ids: tuple[str, ...]
    retained_history_ids: tuple[str, ...]
    required_structure_ids: tuple[str, ...]
    numerical_view_ids: tuple[str, ...]
    structural_tolerances: tuple[NamedDecimal, ...]
    computability_envelope_id: str

    def __post_init__(self) -> None:
        for name, value in (
            ("template_id", self.template_id),
            ("obligations_id", self.obligations_id),
            ("support_id", self.support_id),
            ("validity_id", self.validity_id),
            ("uncertainty_id", self.uncertainty_id),
            ("closure_id", self.closure_id),
            ("structural_convergence_id", self.structural_convergence_id),
            ("computability_id", self.computability_id),
            ("independent_unit_id", self.independent_unit_id),
            ("information_cutoff_id", self.information_cutoff_id),
            ("uncertainty_method_key", self.uncertainty_method_key),
            ("computability_envelope_id", self.computability_envelope_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.physical_unit_count <= 0:
            raise ValueError("law-obligation template requires physical units")
        if self.nested_numerical_view_count < 0:
            raise ValueError("nested numerical view count must be nonnegative")
        for name, values in (
            ("chart_ids", self.chart_ids),
            ("denominator_cell_ids", self.denominator_cell_ids),
            ("validity_domain_ids", self.validity_domain_ids),
            ("assumption_ids", self.assumption_ids),
            ("interval_quantity_ids", self.interval_quantity_ids),
            ("recurrence_cell_ids", self.recurrence_cell_ids),
            ("required_structure_ids", self.required_structure_ids),
            ("numerical_view_ids", self.numerical_view_ids),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)
        require_sorted_unique_strings(
            self.uncertainty_limitation_codes,
            field_name="uncertainty_limitation_codes",
        )
        require_sorted_unique_strings(
            self.exchange_factor_ids,
            field_name="exchange_factor_ids",
        )
        require_sorted_unique_strings(
            self.retained_history_ids,
            field_name="retained_history_ids",
        )
        require_sorted_unique_ids(
            self.action_bounds,
            attribute="bound_id",
            field_name="action_bounds",
        )
        require_sorted_unique_ids(
            self.falsifiers,
            attribute="falsifier_id",
            field_name="falsifiers",
        )
        if not self.falsifiers:
            raise ValueError("law-obligation template requires decisive falsifiers")
        require_sorted_unique_ids(
            self.structural_tolerances,
            attribute="value_id",
            field_name="structural_tolerances",
        )
        validate_decimal(
            self.uncertainty_confidence_level,
            field_name="uncertainty_confidence_level",
            minimum=Decimal(0),
        )
        if not Decimal(0) < self.uncertainty_confidence_level < Decimal(1):
            raise ValueError("uncertainty confidence level must be between zero and one")


@dataclass(frozen=True, slots=True)
class LawCandidateEvidence(CanonicalRecord):
    """Candidate-producer output with no producer-authored generic PASS."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/law-candidate-evidence'

    evidence_id: str
    candidate_id: str
    system: ObjectIdentity
    dataset_or_projection: ObjectIdentity
    config: ObjectIdentity
    method_key: str
    method_version: str
    method_kind: LawMethodKind
    representation_kind: LawRepresentationKind
    candidate_evaluator: ExecutableReference
    axis_map: LawCandidateAxisMap
    claim_unit_binding: ObjectIdentity
    physical_independent_unit_ids: tuple[str, ...]
    claim_template: CandidateClaimTemplate
    obligation_template: LawObligationTemplate
    method_receipts: tuple[CandidateMethodEvidenceReceipt, ...]
    payload_publication: CandidatePayloadPublicationReceipt
    evidence_links: tuple[EvidenceLink, ...]
    outcome_access: OutcomeAccess
    parent_visibility_ceilings: tuple[VisibilityCeiling, ...]
    visibility_ceiling: VisibilityCeiling
    candidate_extension: ExtensionBinding | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("evidence_id", self.evidence_id),
            ("candidate_id", self.candidate_id),
            ("method_key", self.method_key),
        ):
            validate_stable_id(value, field_name=name)
        validate_semantic_version(self.method_version)
        require_sorted_unique_strings(
            self.physical_independent_unit_ids,
            field_name="physical_independent_unit_ids",
            allow_empty=False,
        )
        if len(self.physical_independent_unit_ids) != self.obligation_template.physical_unit_count:
            raise ValueError("candidate physical-unit roster differs from its obligation scope")
        require_sorted_unique_ids(
            self.method_receipts,
            attribute="receipt_id",
            field_name="method_receipts",
        )
        if not self.method_receipts:
            raise ValueError("law candidate evidence requires method receipts")
        require_sorted_unique_ids(
            self.evidence_links,
            attribute="link_id",
            field_name="evidence_links",
        )
        if not self.evidence_links:
            raise ValueError("law candidate evidence requires evidence links")
        evidence_ids = {value.link_id for value in self.evidence_links}
        artifact_ids = {
            artifact_id for value in self.evidence_links for artifact_id in value.artifact_ids
        }
        if any(
            not set(value.evidence_link_ids).issubset(evidence_ids)
            or not set(value.artifact_ids).issubset(artifact_ids)
            for value in self.method_receipts
        ):
            raise ValueError("candidate method evidence escapes the candidate evidence corpus")
        if self.candidate_evaluator != self.payload_publication.candidate_evaluator:
            raise ValueError("candidate evidence and publication evaluator differ")
        payload_schema = self.payload_publication.artifact.payload_schema
        extension_required = payload_schema != ParametricLawModel.SCHEMA
        if extension_required != (self.candidate_extension is not None):
            raise ValueError(
                "candidate extension presence differs from the evaluator payload representation"
            )
        if self.candidate_extension is not None:
            require_extensions((self.candidate_extension,))
            if self.candidate_extension.schema != payload_schema:
                raise ValueError("candidate extension and published payload schemas differ")
            if self.candidate_extension.payload_sha256 != self.payload_publication.content_sha256:
                raise ValueError("candidate extension and published payload digests differ")
        inherited = inherited_visibility(self.parent_visibility_ceilings, self.outcome_access)
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(inherited):
            raise ValueError("candidate-evidence visibility cannot be lowered")


@dataclass(frozen=True, slots=True)
class PropertyQualification(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/property-qualification'

    qualification_id: str
    property_id: str
    candidate_version_member_ids: tuple[str, ...]
    denominator_member_ids: tuple[str, ...]
    qualification_view_ids: tuple[str, ...]
    status: ScientificStatus
    proof_owner: ObjectIdentity
    evidence_link_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.qualification_id, field_name="qualification_id")
        validate_stable_id(self.property_id, field_name="property_id")
        for name, values in (
            ("candidate_version_member_ids", self.candidate_version_member_ids),
            ("denominator_member_ids", self.denominator_member_ids),
            ("qualification_view_ids", self.qualification_view_ids),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)
        require_sorted_unique_strings(
            self.evidence_link_ids,
            field_name="evidence_link_ids",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.status is ScientificStatus.SUPPORTED:
            if not self.evidence_link_ids or self.reason_codes:
                raise ValueError("supported property qualification requires evidence")
        elif not self.reason_codes:
            raise ValueError("non-supported property qualification requires reasons")


@dataclass(frozen=True, slots=True)
class ComponentQualificationAssessment(CanonicalRecord):
    """Generic truth derived only by registered profile proof owners."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/component-qualification-assessment'

    assessment_id: str
    profile: ObjectIdentity
    candidate_evidence: ObjectIdentity
    checks: tuple[AdequacyCheckResult, ...]
    uncertainty: UncertaintyDecomposition
    property_qualifications: tuple[PropertyQualification, ...]
    terminal_obligations: TerminalLawObligationAssessment
    qualification_trace: ResponseQualificationTrace
    visibility_promotable: bool
    eligibility: CandidateQualificationEligibility

    def _profile_schema(self) -> str:
        return 'empirical-lawhood/methods/component-qualification-profile'

    def _uncertainty_gate(self) -> tuple[bool, bool, bool]:
        return (
            self.uncertainty.claim_ready,
            any(v.status is ObligationStatus.FAILED for v in self.uncertainty.components),
            any(v.status is ObligationStatus.UNEVALUABLE for v in self.uncertainty.components),
        )

    def __post_init__(self) -> None:
        validate_stable_id(self.assessment_id, field_name="assessment_id")
        if self.profile.object_schema != self._profile_schema():
            raise ValueError("profile assessment requires an exact QualificationProfile")
        require_sorted_unique_ids(self.checks, attribute="check_id", field_name="checks")
        if len(self.checks) != len(AdequacyCheckKind) or {
            value.kind for value in self.checks
        } != set(AdequacyCheckKind):
            raise ValueError("profile assessment requires the complete adequacy family")
        require_sorted_unique_ids(
            self.property_qualifications,
            attribute="qualification_id",
            field_name="property_qualifications",
        )
        if not self.property_qualifications:
            raise ValueError("profile assessment requires property qualifications")
        uncertainty_ready, uncertainty_failed, uncertainty_unevaluable = self._uncertainty_gate()
        passing = (
            all(value.passed for value in self.checks)
            and uncertainty_ready
            and all(
                value.status is ScientificStatus.SUPPORTED for value in self.property_qualifications
            )
            and self.terminal_obligations.claim_ready
            and self.qualification_trace.highest_supported_rung is EvidenceRung.LOCAL_LAW
        )
        failed = any(
            (
                any(value.status is ObligationStatus.FAILED for value in self.checks),
                uncertainty_failed,
                any(
                    value.status
                    in {
                        ScientificStatus.NOT_SUPPORTED,
                        ScientificStatus.MIXED,
                        ScientificStatus.PARTIAL,
                    }
                    for value in self.property_qualifications
                ),
                any(
                    value.status is ObligationStatus.FAILED
                    for value in self.terminal_obligations.dispositions
                ),
                any(
                    value.status
                    in {
                        ScientificStatus.NOT_SUPPORTED,
                        ScientificStatus.MIXED,
                        ScientificStatus.PARTIAL,
                    }
                    for value in self.qualification_trace.facets
                ),
            )
        )
        unevaluable = any(
            (
                any(value.status is ObligationStatus.UNEVALUABLE for value in self.checks),
                uncertainty_unevaluable,
                any(
                    value.status is ScientificStatus.UNEVALUABLE
                    for value in self.property_qualifications
                ),
                any(
                    value.status is ObligationStatus.UNEVALUABLE
                    for value in self.terminal_obligations.dispositions
                ),
                any(
                    value.status is ScientificStatus.UNEVALUABLE
                    for value in self.qualification_trace.facets
                ),
            )
        )
        if not self.visibility_promotable:
            expected = CandidateQualificationEligibility.NONPROMOTABLE
        elif passing:
            expected = CandidateQualificationEligibility.PASSING
        elif failed:
            expected = CandidateQualificationEligibility.NONPASSING
        elif unevaluable:
            expected = CandidateQualificationEligibility.UNEVALUABLE
        else:
            expected = CandidateQualificationEligibility.NONPASSING
        if self.eligibility is not expected:
            raise ValueError("profile eligibility is not mechanically derived")


@dataclass(frozen=True, slots=True)
class JointQualificationAssessment(ComponentQualificationAssessment):
    "Explicit joint uncertainty basis; component diagnostics retain their own meaning.\n\n    Component and joint assessments have distinct declared schemas. The added\n    joint operand controls this uncertainty gate.\n    "

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/joint-qualification-assessment'
    joint_uncertainty: JointPredictiveUncertainty = field(kw_only=True)

    def _profile_schema(self) -> str:
        return 'empirical-lawhood/methods/joint-qualification-profile'

    def _uncertainty_gate(self) -> tuple[bool, bool, bool]:
        return (
            self.joint_uncertainty.claim_ready,
            self.joint_uncertainty.status is ObligationStatus.FAILED,
            self.joint_uncertainty.status is ObligationStatus.UNEVALUABLE,
        )

    def __post_init__(self) -> None:
        ComponentQualificationAssessment.__post_init__(self)
        calibration_ids = set(self.joint_uncertainty.calibration_unit_ids)
        qualification_ids = set(self.qualification_trace.physical_independent_unit_ids)
        if not (
            calibration_ids == qualification_ids
            or (
                self.joint_uncertainty.calibration is not None
                and calibration_ids.isdisjoint(qualification_ids)
            )
        ):
            raise ValueError("joint uncertainty changes or overlaps the independent calibration units")
        gate = self.terminal_obligations.disposition(TerminalObligationKind.UNCERTAINTY)
        if gate.status is not self.joint_uncertainty.status:
            raise ValueError("terminal uncertainty changes the declared joint basis")


@dataclass(frozen=True, slots=True)
class ComponentUncertaintyCandidateAssessment(CanonicalRecord):
    """Exact candidate evidence plus profile-derived generic assessment."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/component-uncertainty-candidate-assessment'

    assessment_id: str
    candidate_id: str
    candidate_evidence: LawCandidateEvidence
    profile_assessment: ComponentQualificationAssessment
    metrics: tuple[NamedDecimal, ...]
    response_method_structural_convergence: StructuralConvergenceResult | None = None

    def __post_init__(self) -> None:
        if (
            type(self) is ComponentUncertaintyCandidateAssessment
            and type(self.profile_assessment) is not ComponentQualificationAssessment
        ):
            raise ValueError("ComponentUncertaintyCandidateAssessment requires ComponentQualificationAssessment")
        validate_stable_id(self.assessment_id, field_name="assessment_id")
        validate_stable_id(self.candidate_id, field_name="candidate_id")
        if self.candidate_id != self.candidate_evidence.candidate_id:
            raise ValueError("candidate assessment binds different candidate evidence")
        expected = ObjectIdentity.from_record(
            self.candidate_evidence.evidence_id,
            self.candidate_evidence,
        )
        if self.profile_assessment.candidate_evidence != expected:
            raise ValueError("profile assessment binds different candidate evidence")
        if self.profile_assessment.qualification_trace.candidate_id != self.candidate_id:
            raise ValueError("candidate assessment trace binds another candidate")
        evidence = self.candidate_evidence
        profile = self.profile_assessment
        trace = profile.qualification_trace
        if profile.visibility_promotable != evidence.visibility_ceiling.is_promotable:
            raise ValueError("profile promotability differs from candidate visibility")
        if (
            trace.candidate_version_member_ids != evidence.axis_map.candidate_version_member_ids
            or trace.denominator_member_ids != evidence.axis_map.denominator_member_ids
            or trace.qualification_view_ids != evidence.axis_map.qualification_view_ids
            or trace.information_cutoff_id != evidence.obligation_template.information_cutoff_id
            or trace.outcome_access is not evidence.outcome_access
            or len(trace.physical_independent_unit_ids)
            != evidence.obligation_template.physical_unit_count
        ):
            raise ValueError(
                "candidate trace changes axis, cutoff, outcome, or physical-unit scope"
            )
        expected_properties = {
            property_id
            for binding in evidence.axis_map.bindings
            for property_id in binding.claimed_property_ids
        }
        if {value.property_id for value in profile.property_qualifications} != expected_properties:
            raise ValueError("profile property roster differs from the candidate axis map")
        for qualification in profile.property_qualifications:
            bindings = tuple(
                value
                for value in evidence.axis_map.bindings
                if qualification.property_id in value.claimed_property_ids
            )
            if (
                qualification.candidate_version_member_ids
                != tuple(sorted(value.candidate_version_member_id for value in bindings))
                or qualification.denominator_member_ids
                != tuple(sorted({value.denominator_member_id for value in bindings}))
                or qualification.qualification_view_ids
                != tuple(
                    sorted({view for value in bindings for view in value.qualification_view_ids})
                )
            ):
                raise ValueError("property qualification changes its declared candidate axes")
        evidence_ids = {value.link_id for value in evidence.evidence_links}
        nested_evidence_ids = {
            link_id
            for link_ids in (
                *(value.evidence_link_ids for value in profile.checks),
                *(value.evidence_link_ids for value in profile.uncertainty.components),
                *(value.evidence_link_ids for value in profile.property_qualifications),
                *(value.evidence_link_ids for value in profile.terminal_obligations.dispositions),
                *(value.evidence_link_ids for value in trace.facets),
            )
            for link_id in link_ids
        }
        if not nested_evidence_ids <= evidence_ids:
            raise ValueError("profile assessment cites evidence outside the candidate corpus")
        require_sorted_unique_ids(self.metrics, attribute="value_id", field_name="metrics")

    @property
    def qualification_eligibility(self) -> CandidateQualificationEligibility:
        return self.profile_assessment.eligibility


@dataclass(frozen=True, slots=True)
class JointUncertaintyCandidateAssessment(ComponentUncertaintyCandidateAssessment):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/joint-uncertainty-candidate-assessment'
    profile_assessment: JointQualificationAssessment

    def __post_init__(self) -> None:
        ComponentUncertaintyCandidateAssessment.__post_init__(self)
        joint = self.profile_assessment.joint_uncertainty
        evidence = self.candidate_evidence
        template = evidence.obligation_template
        if (
            evidence.claim_template.compatibility_result_id is not None
            or not set(joint.evidence_link_ids) <= {v.link_id for v in evidence.evidence_links}
            or joint.independent_unit_id != template.independent_unit_id
            or not (
                joint.calibration_unit_ids == evidence.physical_independent_unit_ids
                or (
                    joint.calibration is not None
                    and joint.calibration != evidence.dataset_or_projection
                    and set(joint.calibration_unit_ids).isdisjoint(
                        evidence.physical_independent_unit_ids
                    )
                    and any(
                        link.source == joint.calibration
                        and link.link_id in joint.evidence_link_ids
                        for link in evidence.evidence_links
                    )
                )
            )
            or joint.interval_quantity_ids != template.interval_quantity_ids
            or joint.confidence_level != template.uncertainty_confidence_level
            or joint.method_key != template.uncertainty_method_key
            or joint.support_id != template.support_id
        ):
            raise ValueError("joint uncertainty changes the candidate evidence or claim scope")


@dataclass(frozen=True, slots=True)
class CandidateFamilyMember(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/candidate-family-member'

    candidate_id: str
    config: ObjectIdentity
    complexity_rank: int
    disposition: CandidateRosterDisposition
    predeclared_reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.candidate_id, field_name="candidate_id")
        if self.complexity_rank < 0:
            raise ValueError("candidate complexity rank must be nonnegative")
        require_sorted_unique_strings(
            self.predeclared_reason_codes,
            field_name="predeclared_reason_codes",
        )
        if self.disposition is CandidateRosterDisposition.ASSESS_REQUIRED:
            if self.predeclared_reason_codes:
                raise ValueError("assess-required candidate cannot be predeclared inapplicable")
        elif not self.predeclared_reason_codes:
            raise ValueError("predeclared-inapplicable candidate requires reasons")


@dataclass(frozen=True, slots=True)
class CandidateFamilyLedger(CanonicalRecord):
    """Frozen development-only candidate roster and selector lineage."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/candidate-family-ledger'

    family_id: str
    system: ObjectIdentity
    dataset_or_projection: ObjectIdentity
    method_key: str
    method_version: str
    method_kind: LawMethodKind
    representation_kind: LawRepresentationKind
    chart_id: str
    axis_map: LawCandidateAxisMap
    claim_unit_binding: ObjectIdentity
    claim_template: CandidateClaimTemplate
    obligation_template: LawObligationTemplate
    members: tuple[CandidateFamilyMember, ...]
    candidate_generation_rule_ids: tuple[str, ...]
    selection_threshold_ids: tuple[str, ...]
    development_input_ids: tuple[str, ...]
    selector_capability: ObjectIdentity
    selector_config: ObjectIdentity
    selector_implementation: ObjectIdentity
    multiplicity_family_id: str
    multiplicity_rule_id: str
    randomness_seed_ids: tuple[str, ...]
    tie_break_rule: str
    outcome_access: OutcomeAccess
    parent_visibility_ceilings: tuple[VisibilityCeiling, ...]
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.family_id, field_name="family_id")
        validate_stable_id(self.method_key, field_name="method_key")
        validate_semantic_version(self.method_version)
        validate_stable_id(self.chart_id, field_name="chart_id")
        validate_stable_id(self.multiplicity_family_id, field_name="multiplicity_family_id")
        validate_stable_id(self.multiplicity_rule_id, field_name="multiplicity_rule_id")
        if not self.members:
            raise ValueError("candidate family requires a roster")
        candidate_ids = tuple(value.candidate_id for value in self.members)
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate family candidate IDs must be unique")
        if self.members != tuple(
            sorted(
                self.members,
                key=lambda value: (value.complexity_rank, value.candidate_id),
            )
        ):
            raise ValueError(
                "candidate family must be ordered by complexity rank then candidate ID"
            )
        for name, values in (
            ("candidate_generation_rule_ids", self.candidate_generation_rule_ids),
            ("selection_threshold_ids", self.selection_threshold_ids),
            ("development_input_ids", self.development_input_ids),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)
        require_sorted_unique_strings(
            self.randomness_seed_ids,
            field_name="randomness_seed_ids",
        )
        validate_nonempty(self.tie_break_rule, field_name="tie_break_rule")
        inherited = inherited_visibility(self.parent_visibility_ceilings, self.outcome_access)
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(inherited):
            raise ValueError("candidate-family visibility cannot be lowered")
        allowed_access = {
            OutcomeAccess.OUTCOME_BLIND,
            OutcomeAccess.DEVELOPMENT_VISIBLE,
        }
        if len(self.members) > 1 and self.outcome_access not in allowed_access:
            raise ValueError("candidate selection cannot consume protected evaluation outcomes")


@dataclass(frozen=True, slots=True)
class CandidateSelectionReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/candidate-selection-receipt'

    receipt_id: str
    family: ObjectIdentity
    assessed_candidate_ids: tuple[str, ...]
    selected_candidate_id: str | None
    disposition: CandidateSelectionDisposition
    candidate_generation_rule_ids: tuple[str, ...]
    selection_threshold_ids: tuple[str, ...]
    development_input_ids: tuple[str, ...]
    selector_capability: ObjectIdentity
    selector_config: ObjectIdentity
    selector_implementation: ObjectIdentity
    multiplicity_family_id: str
    multiplicity_rule_id: str
    randomness_seed_ids: tuple[str, ...]
    tie_break_rule: str

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        validate_stable_id(self.multiplicity_family_id, field_name="multiplicity_family_id")
        validate_stable_id(self.multiplicity_rule_id, field_name="multiplicity_rule_id")
        require_sorted_unique_strings(
            self.assessed_candidate_ids,
            field_name="assessed_candidate_ids",
        )
        for name, values in (
            ("candidate_generation_rule_ids", self.candidate_generation_rule_ids),
            ("selection_threshold_ids", self.selection_threshold_ids),
            ("development_input_ids", self.development_input_ids),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)
        require_sorted_unique_strings(
            self.randomness_seed_ids,
            field_name="randomness_seed_ids",
        )
        validate_nonempty(self.tie_break_rule, field_name="tie_break_rule")
        if self.selected_candidate_id is not None:
            validate_stable_id(self.selected_candidate_id, field_name="selected_candidate_id")
        if self.disposition is CandidateSelectionDisposition.SELECTED:
            if self.selected_candidate_id not in self.assessed_candidate_ids:
                raise ValueError("selected candidate was not assessed")
        elif self.selected_candidate_id is not None:
            raise ValueError("NO_SELECTION cannot name a selected candidate")


@dataclass(frozen=True, slots=True)
class ComponentUncertaintyFamilyAssessment(CanonicalRecord):
    """Complete family assessment and deterministic selection result."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/component-uncertainty-family-assessment'

    assessment_id: str
    ledger: CandidateFamilyLedger
    assessments: tuple[ComponentUncertaintyCandidateAssessment, ...]
    selection: CandidateSelectionReceipt

    def __post_init__(self) -> None:
        if type(self) is ComponentUncertaintyFamilyAssessment and any(
            type(v) is not ComponentUncertaintyCandidateAssessment for v in self.assessments
        ):
            raise ValueError("ComponentUncertaintyFamilyAssessment requires ComponentUncertaintyCandidateAssessment")
        validate_stable_id(self.assessment_id, field_name="assessment_id")
        require_sorted_unique_ids(
            self.assessments,
            attribute="candidate_id",
            field_name="assessments",
        )
        required = {
            value.candidate_id
            for value in self.ledger.members
            if value.disposition is CandidateRosterDisposition.ASSESS_REQUIRED
        }
        observed = {value.candidate_id for value in self.assessments}
        if observed != required:
            raise ValueError("candidate-family assessment does not cover the frozen roster")
        for assessment in self.assessments:
            evidence = assessment.candidate_evidence
            member = next(
                value
                for value in self.ledger.members
                if value.candidate_id == assessment.candidate_id
            )
            if evidence.config != member.config:
                raise ValueError("candidate assessment changes its frozen family config")
            for name in (
                "system",
                "dataset_or_projection",
                "method_key",
                "method_version",
                "method_kind",
                "representation_kind",
                "axis_map",
                "claim_unit_binding",
                "claim_template",
                "obligation_template",
            ):
                if getattr(evidence, name) != getattr(self.ledger, name):
                    raise ValueError(f"candidate family changes common {name}")
            if (
                evidence.outcome_access is not self.ledger.outcome_access
                or evidence.parent_visibility_ceilings != self.ledger.parent_visibility_ceilings
                or evidence.visibility_ceiling is not self.ledger.visibility_ceiling
            ):
                raise ValueError("candidate family changes common outcome/visibility custody")
        profile_identities = {value.profile_assessment.profile for value in self.assessments}
        if len(profile_identities) != 1:
            raise ValueError("candidate family must use one exact qualification profile")
        if self.selection.family != ObjectIdentity.from_record(
            self.ledger.family_id,
            self.ledger,
        ):
            raise ValueError("candidate selection binds another family ledger")
        if set(self.selection.assessed_candidate_ids) != observed:
            raise ValueError("candidate selection omits assessed candidates")
        if self.selection.multiplicity_family_id != self.ledger.multiplicity_family_id:
            raise ValueError("candidate selection multiplicity family changed")
        for name in (
            "candidate_generation_rule_ids",
            "selection_threshold_ids",
            "development_input_ids",
            "selector_capability",
            "selector_config",
            "selector_implementation",
            "multiplicity_rule_id",
            "randomness_seed_ids",
        ):
            if getattr(self.selection, name) != getattr(self.ledger, name):
                raise ValueError(f"candidate selection changes frozen {name}")
        if self.selection.tie_break_rule != self.ledger.tie_break_rule:
            raise ValueError("candidate selection tie-break changed")
        passing = tuple(
            value
            for value in self.assessments
            if value.qualification_eligibility is CandidateQualificationEligibility.PASSING
        )
        expected = None
        if passing:
            ranks = {member.candidate_id: member.complexity_rank for member in self.ledger.members}
            expected = min(
                passing,
                key=lambda value: (ranks[value.candidate_id], value.candidate_id),
            ).candidate_id
        if self.selection.selected_candidate_id != expected:
            raise ValueError("candidate selection is not the least-complex passing member")

    def selected_assessment(self) -> ComponentUncertaintyCandidateAssessment | None:
        selected = self.selection.selected_candidate_id
        if selected is None:
            return None
        return next(value for value in self.assessments if value.candidate_id == selected)


@dataclass(frozen=True, slots=True)
class JointUncertaintyFamilyAssessment(ComponentUncertaintyFamilyAssessment):
    """Same selection owner and rule, with explicitly versioned joint profiles."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/joint-uncertainty-family-assessment'
    assessments: tuple[JointUncertaintyCandidateAssessment, ...]

    def __post_init__(self) -> None:
        if any(type(v) is not JointUncertaintyCandidateAssessment for v in self.assessments):
            raise ValueError("JointUncertaintyFamilyAssessment requires JointUncertaintyCandidateAssessment")
        ComponentUncertaintyFamilyAssessment.__post_init__(self)


class FiniteActionCellDisposition(StrEnum):
    SUPPORTED = "SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    UNEVALUABLE = "UNEVALUABLE"
    OUTSIDE_SUPPORT = "OUTSIDE_SUPPORT"


@dataclass(frozen=True, slots=True)
class FiniteActionNativeValue(CanonicalRecord):
    """One cell value with an explicit quantity, native unit, frame and clock."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-action-native-value'

    value_id: str
    quantity_id: str
    value: Decimal
    native_unit: str
    native_frame_id: str
    clock_id: str

    def __post_init__(self) -> None:
        for name, identifier in (
            ("value_id", self.value_id),
            ("quantity_id", self.quantity_id),
            ("native_frame_id", self.native_frame_id),
            ("clock_id", self.clock_id),
        ):
            validate_stable_id(identifier, field_name=name)
        validate_decimal(self.value, field_name="value")
        validate_nonempty(self.native_unit, field_name="native_unit")


@dataclass(frozen=True, slots=True)
class FiniteActionResponseEntry(CanonicalRecord):
    """One exact member-local response cell; no pooled coefficients or means."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-action-response-entry'

    entry_id: str
    denominator_member_id: str
    candidate_version_id: str
    qualification_view_ids: tuple[str, ...]
    support_cell_id: str
    action_word: ObjectIdentity
    comparator_action_word_ids: tuple[str, ...]
    physical_independent_unit_ids: tuple[str, ...]
    response_values: tuple[FiniteActionNativeValue, ...]
    sink_values: tuple[FiniteActionNativeValue, ...]
    effort_values: tuple[FiniteActionNativeValue, ...]
    uncertainty_values: tuple[FiniteActionNativeValue, ...]
    disposition: FiniteActionCellDisposition
    evidence_link_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("entry_id", self.entry_id),
            ("denominator_member_id", self.denominator_member_id),
            ("candidate_version_id", self.candidate_version_id),
            ("support_cell_id", self.support_cell_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.action_word.object_schema != OccurrenceActionWord.SCHEMA:
            raise ValueError("finite-action entry requires current exact ActionWord identity")
        require_sorted_unique_strings(
            self.qualification_view_ids,
            field_name="qualification_view_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.comparator_action_word_ids,
            field_name="comparator_action_word_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.physical_independent_unit_ids,
            field_name="physical_independent_unit_ids",
            allow_empty=False,
        )
        for name, values in (
            ("response_values", self.response_values),
            ("sink_values", self.sink_values),
            ("effort_values", self.effort_values),
            ("uncertainty_values", self.uncertainty_values),
        ):
            require_sorted_unique_ids(values, attribute="value_id", field_name=name)
        require_sorted_unique_strings(
            self.evidence_link_ids,
            field_name="evidence_link_ids",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.disposition is FiniteActionCellDisposition.SUPPORTED:
            if not self.response_values or not self.evidence_link_ids or self.reason_codes:
                raise ValueError("supported finite-action entry requires response evidence")
        elif not self.reason_codes:
            raise ValueError("non-supported finite-action entry requires reasons")


@dataclass(frozen=True, slots=True)
class FiniteActionCompatibilitySetExtension(CanonicalRecord):
    """Exact finite-word compatibility table with explicit nonlinearity disclaimer."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-action-compatibility-set-extension'

    extension_id: str
    method_semantics: str
    prepared_denominator_id: str
    horizon: ObjectIdentity
    retained_history_ids: tuple[str, ...]
    action_words: tuple[OccurrenceActionWord, ...]
    physical_independent_unit_ids: tuple[str, ...]
    entries: tuple[FiniteActionResponseEntry, ...]
    rejection_entry_ids: tuple[str, ...]
    refusal_reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.extension_id, field_name="extension_id")
        validate_stable_id(
            self.prepared_denominator_id,
            field_name="prepared_denominator_id",
        )
        validate_nonempty(self.method_semantics, field_name="method_semantics")
        if "no linearity inference" not in self.method_semantics.lower():
            raise ValueError("finite compatibility extension must prohibit linearity inference")
        if self.horizon.object_schema != HorizonSpec.SCHEMA:
            raise ValueError("finite compatibility set requires the exact response horizon")
        require_sorted_unique_strings(
            self.retained_history_ids,
            field_name="retained_history_ids",
        )
        require_sorted_unique_ids(
            self.action_words,
            attribute="word_id",
            field_name="action_words",
        )
        if not self.action_words:
            raise ValueError("finite compatibility set requires current ActionWords")
        if any(
            value.denominator_id != self.prepared_denominator_id
            or value.horizon_id != self.horizon.object_id
            or value.retained_history_id not in self.retained_history_ids
            for value in self.action_words
        ):
            raise ValueError("finite compatibility action changes denominator/history/horizon")
        require_sorted_unique_strings(
            self.physical_independent_unit_ids,
            field_name="physical_independent_unit_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(self.entries, attribute="entry_id", field_name="entries")
        if not self.entries:
            raise ValueError("finite compatibility set requires response entries")
        word_identities = tuple(
            ObjectIdentity.from_record(value.word_id, value) for value in self.action_words
        )
        word_ids = {value.word_id for value in self.action_words}
        if {value.action_word.object_id for value in self.entries} != word_ids:
            raise ValueError("finite compatibility entries do not cover the exact word roster")
        if any(value.action_word not in word_identities for value in self.entries):
            raise ValueError("finite compatibility entry changes an action-word identity")
        if any(
            not set(value.comparator_action_word_ids).issubset(word_ids) for value in self.entries
        ):
            raise ValueError("finite compatibility entry names a foreign comparator word")
        if any(
            not set(value.physical_independent_unit_ids).issubset(
                self.physical_independent_unit_ids
            )
            for value in self.entries
        ):
            raise ValueError("finite compatibility entry names a foreign independent unit")
        if {
            unit_id for value in self.entries for unit_id in value.physical_independent_unit_ids
        } != set(self.physical_independent_unit_ids):
            raise ValueError("finite compatibility entries omit an independent unit")
        coordinates = tuple(
            (
                value.denominator_member_id,
                value.candidate_version_id,
                value.qualification_view_ids,
                value.support_cell_id,
                value.action_word.object_id,
            )
            for value in self.entries
        )
        if len(set(coordinates)) != len(coordinates):
            raise ValueError("finite compatibility set duplicates an exact response cell")
        require_sorted_unique_strings(
            self.rejection_entry_ids,
            field_name="rejection_entry_ids",
        )
        if set(self.rejection_entry_ids) != {
            value.entry_id
            for value in self.entries
            if value.disposition is not FiniteActionCellDisposition.SUPPORTED
        }:
            raise ValueError("finite compatibility rejection ledger is incomplete")
        require_sorted_unique_strings(
            self.refusal_reason_codes,
            field_name="refusal_reason_codes",
            allow_empty=False,
        )


class LawQualificationCoordinateDisposition(StrEnum):
    "Typed local-law source retained at one declared batch coordinate."

    SUPPORTED_LAW = "SUPPORTED_LAW"
    SUPPORTED_CONSTITUENT = "SUPPORTED_CONSTITUENT"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    MIXED = "MIXED"
    PARTIAL = "PARTIAL"
    UNEVALUABLE = "UNEVALUABLE"
    NO_DATA = "NO_DATA"
    COMPUTABILITY_BOUNDARY = "COMPUTABILITY_BOUNDARY"
    PREREQUISITE_OBSTRUCTION = "PREREQUISITE_OBSTRUCTION"
    CONTRACT_OBSTRUCTION = "CONTRACT_OBSTRUCTION"


_SUPPORTED_BATCH_DISPOSITIONS = {
    LawQualificationCoordinateDisposition.SUPPORTED_LAW,
    LawQualificationCoordinateDisposition.SUPPORTED_CONSTITUENT,
}


@dataclass(frozen=True, slots=True)
class LawQualificationBatchCoordinateDeclaration(CanonicalRecord):
    """Frozen coordinate roster entry, independent of its terminal result."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/law-qualification-batch-coordinate-declaration'

    coordinate_id: str
    domain_cell_id: str
    chart_id: str
    denominator_cell_id: str
    denominator_member_id: str

    def __post_init__(self) -> None:
        for name, value in (
            ("coordinate_id", self.coordinate_id),
            ("domain_cell_id", self.domain_cell_id),
            ("chart_id", self.chart_id),
            ("denominator_cell_id", self.denominator_cell_id),
            ("denominator_member_id", self.denominator_member_id),
        ):
            validate_stable_id(value, field_name=name)


@dataclass(frozen=True, slots=True)
class LawQualificationBatchCoordinate(CanonicalRecord):
    """One exact chart/cell/member terminal source for atlas assembly."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/law-qualification-batch-coordinate'

    coordinate_id: str
    domain_cell_id: str
    chart_id: str
    denominator_cell_id: str
    denominator_member_id: str
    qualification_result: LawQualificationResult
    disposition: LawQualificationCoordinateDisposition
    response_law: ObjectIdentity | None
    law_constituent_id: str | None
    reason_codes: tuple[str, ...]
    evidence_link_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("coordinate_id", self.coordinate_id),
            ("domain_cell_id", self.domain_cell_id),
            ("chart_id", self.chart_id),
            ("denominator_cell_id", self.denominator_cell_id),
            ("denominator_member_id", self.denominator_member_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.law_constituent_id is not None:
            validate_stable_id(self.law_constituent_id, field_name="law_constituent_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        require_sorted_unique_strings(
            self.evidence_link_ids,
            field_name="evidence_link_ids",
        )
        result = self.qualification_result
        if result.chart_id != self.chart_id:
            raise ValueError("batch coordinate and qualification result use different charts")
        if self.denominator_member_id not in result.qualification_trace.denominator_member_ids:
            raise ValueError("batch coordinate member is absent from the qualification trace")
        expected_evidence = tuple(link.link_id for link in result.evidence_links)
        if self.evidence_link_ids != expected_evidence:
            raise ValueError("batch coordinate evidence differs from its terminal result")
        if self.disposition in _SUPPORTED_BATCH_DISPOSITIONS:
            self._validate_supported()
        else:
            self._validate_gap_source()

    def _validate_supported(self) -> None:
        result = self.qualification_result
        if result.scientific_status is not ScientificStatus.SUPPORTED:
            raise ValueError("supported batch coordinate requires a supported terminal result")
        law = result.response_law
        if law is None:  # pragma: no cover - qualification-result invariant
            raise AssertionError("supported qualification result lost its ResponseLaw")
        expected_law = ObjectIdentity.from_record(law.law_id, law)
        if self.response_law != expected_law:
            raise ValueError("batch coordinate rewrites its terminal ResponseLaw identity")
        if self.denominator_cell_id not in law.obligations.support.denominator_cell_ids:
            raise ValueError("batch coordinate lies outside its terminal law support")
        if self.reason_codes:
            raise ValueError("supported batch coordinate cannot carry gap reasons")
        if self.disposition is LawQualificationCoordinateDisposition.SUPPORTED_LAW:
            if self.law_constituent_id is not None:
                raise ValueError("ordinary supported law cannot name a set-valued constituent")
        elif self.law_constituent_id is None:
            raise ValueError("supported constituent requires an exact constituent identity")

    def _validate_gap_source(self) -> None:
        result = self.qualification_result
        if result.scientific_status is ScientificStatus.SUPPORTED:
            raise ValueError("supported terminal result cannot be relabelled as an atlas gap")
        if self.response_law is not None or self.law_constituent_id is not None:
            raise ValueError("non-supported batch coordinate cannot bind a law")
        if self.reason_codes != result.reason_codes:
            raise ValueError("batch gap reasons must be the exact terminal result reasons")
        expected: dict[ScientificStatus, set[LawQualificationCoordinateDisposition]] = {
            ScientificStatus.NOT_TESTED: {
                LawQualificationCoordinateDisposition.NO_DATA,
                LawQualificationCoordinateDisposition.PREREQUISITE_OBSTRUCTION,
                LawQualificationCoordinateDisposition.CONTRACT_OBSTRUCTION,
            },
            ScientificStatus.NOT_SUPPORTED: {
                LawQualificationCoordinateDisposition.NOT_SUPPORTED,
            },
            ScientificStatus.MIXED: {LawQualificationCoordinateDisposition.MIXED},
            ScientificStatus.PARTIAL: {LawQualificationCoordinateDisposition.PARTIAL},
            ScientificStatus.UNEVALUABLE: {
                LawQualificationCoordinateDisposition.UNEVALUABLE,
                LawQualificationCoordinateDisposition.NO_DATA,
                LawQualificationCoordinateDisposition.COMPUTABILITY_BOUNDARY,
                LawQualificationCoordinateDisposition.PREREQUISITE_OBSTRUCTION,
                LawQualificationCoordinateDisposition.CONTRACT_OBSTRUCTION,
            },
        }
        if self.disposition not in expected[result.scientific_status]:
            raise ValueError("batch gap disposition conflicts with terminal scientific status")
        if (
            self.disposition is LawQualificationCoordinateDisposition.COMPUTABILITY_BOUNDARY
            and result.terminal_obligations.disposition(TerminalObligationKind.COMPUTABILITY).passed
        ):
            raise ValueError("computability-boundary gap requires failed computability")


class LawQualificationOverlapKind(StrEnum):
    SELECTED_COMPATIBLE_LAW = "SELECTED_COMPATIBLE_LAW"
    SET_VALUED_COEXISTENCE = "SET_VALUED_COEXISTENCE"
    REJECTED_COMPOSITION = "REJECTED_COMPOSITION"


@dataclass(frozen=True, slots=True)
class LawQualificationOverlapDisposition(CanonicalRecord):
    """Predeclared handling for every multiply covered chart/cell coordinate."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/law-qualification-overlap-disposition'

    overlap_id: str
    chart_id: str
    denominator_cell_id: str
    kind: LawQualificationOverlapKind
    constituent_laws: tuple[ObjectIdentity, ...]
    selected_law: ObjectIdentity | None
    coexistence_set_id: str | None
    adjudicator: ObjectIdentity
    evidence_link_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("overlap_id", self.overlap_id),
            ("chart_id", self.chart_id),
            ("denominator_cell_id", self.denominator_cell_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.coexistence_set_id is not None:
            validate_stable_id(self.coexistence_set_id, field_name="coexistence_set_id")
        require_sorted_unique_ids(
            self.constituent_laws,
            attribute="object_id",
            field_name="constituent_laws",
        )
        if len(self.constituent_laws) < 2:
            raise ValueError("overlap disposition requires at least two laws")
        if any(value.object_schema != ResponseLaw.SCHEMA for value in self.constituent_laws):
            raise ValueError("overlap constituents must be ResponseLaw identities")
        require_sorted_unique_strings(
            self.evidence_link_ids,
            field_name="evidence_link_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.kind is LawQualificationOverlapKind.SELECTED_COMPATIBLE_LAW:
            if self.selected_law not in self.constituent_laws:
                raise ValueError("selected overlap law is absent from the constituent roster")
            if self.coexistence_set_id is not None or self.reason_codes:
                raise ValueError("selected overlap cannot claim coexistence or rejection")
        elif self.kind is LawQualificationOverlapKind.SET_VALUED_COEXISTENCE:
            if self.selected_law is not None or self.coexistence_set_id is None:
                raise ValueError("set-valued overlap requires a coexistence set only")
            if self.reason_codes:
                raise ValueError("supported coexistence cannot carry rejection reasons")
        else:
            if self.selected_law is not None or self.coexistence_set_id is not None:
                raise ValueError("rejected overlap cannot select or compose laws")
            if not self.reason_codes:
                raise ValueError("rejected overlap requires exact composition reasons")

    @property
    def coverage_id(self) -> str:
        return f"{self.chart_id}.{self.denominator_cell_id}"


@dataclass(frozen=True, slots=True)
class LawQualificationBatch(CanonicalRecord):
    "Complete chart/cell/member roster of authoritative terminal local-law results."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/law-qualification-batch'

    batch_id: str
    system_id: str
    world_id: str
    declared_coordinates: tuple[LawQualificationBatchCoordinateDeclaration, ...]
    coordinates: tuple[LawQualificationBatchCoordinate, ...]
    overlaps: tuple[LawQualificationOverlapDisposition, ...]
    evidence_links: tuple[EvidenceLink, ...]
    evidence_ceiling: EvidenceCeiling
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        for name, value in (
            ("batch_id", self.batch_id),
            ("system_id", self.system_id),
            ("world_id", self.world_id),
        ):
            validate_stable_id(value, field_name=name)
        for name, values in (
            ("declared_coordinates", self.declared_coordinates),
            ("coordinates", self.coordinates),
        ):
            require_sorted_unique_ids(
                values,
                attribute="coordinate_id",
                field_name=name,
            )
        if not self.declared_coordinates or not self.coordinates:
            raise ValueError("law qualification batch requires declared coordinates")
        declared = {
            value.coordinate_id: (
                value.domain_cell_id,
                value.chart_id,
                value.denominator_cell_id,
                value.denominator_member_id,
            )
            for value in self.declared_coordinates
        }
        observed = {
            value.coordinate_id: (
                value.domain_cell_id,
                value.chart_id,
                value.denominator_cell_id,
                value.denominator_member_id,
            )
            for value in self.coordinates
        }
        if observed != declared:
            raise ValueError("law qualification batch does not cover its frozen coordinate roster")
        coordinate_keys = {
            (
                value.chart_id,
                value.denominator_cell_id,
                value.denominator_member_id,
            )
            for value in self.coordinates
        }
        if len(coordinate_keys) != len(self.coordinates):
            raise ValueError("law qualification batch repeats a chart/cell/member coordinate")
        if len({value.qualification_result.result_id for value in self.coordinates}) != len(
            self.coordinates
        ):
            raise ValueError("each batch coordinate requires its own terminal result identity")
        for coordinate in self.coordinates:
            result = coordinate.qualification_result
            if result.system_id != self.system_id or result.world_id != self.world_id:
                raise ValueError("batch coordinate uses another system or evidence world")
        self._validate_supported_law_locality()
        require_sorted_unique_ids(
            self.overlaps,
            attribute="overlap_id",
            field_name="overlaps",
        )
        require_sorted_unique_ids(
            self.evidence_links,
            attribute="link_id",
            field_name="evidence_links",
        )
        expected_evidence: dict[str, EvidenceLink] = {}
        for coordinate in self.coordinates:
            for link in coordinate.qualification_result.evidence_links:
                prior = expected_evidence.setdefault(link.link_id, link)
                if prior != link:
                    raise ValueError(
                        "batch terminal results reuse one evidence ID for different exact evidence"
                    )
        observed_evidence = {link.link_id: link for link in self.evidence_links}
        if observed_evidence != expected_evidence:
            raise ValueError("batch evidence differs from its terminal result corpus")
        self._validate_overlaps()
        if self.evidence_ceiling is not EvidenceCeiling.LOCAL_LAW:
            raise ValueError("law qualification batch ceiling must be exactly local law")
        if not self.visibility_ceiling.is_promotable:
            raise ValueError("outcome-visible terminal results cannot form an atlas batch")
        inherited = VisibilityCeiling.most_restrictive(
            *(value.qualification_result.visibility_ceiling for value in self.coordinates)
        )
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(inherited):
            raise ValueError("law qualification batch visibility cannot be lowered")

    def _validate_supported_law_locality(self) -> None:
        supported = tuple(value for value in self.coordinates if value.response_law is not None)
        law_identities: dict[str, ObjectIdentity] = {}
        law_members: dict[ObjectIdentity, set[str]] = {}
        for coordinate in supported:
            identity = coordinate.response_law
            if identity is None:  # pragma: no cover - filter invariant
                raise AssertionError("supported coordinate lost its law identity")
            prior = law_identities.setdefault(identity.object_id, identity)
            if prior != identity:
                raise ValueError("one law ID resolves to different exact law bytes in the batch")
            law_members.setdefault(identity, set()).add(coordinate.denominator_member_id)
        if any(len(members) != 1 for members in law_members.values()):
            raise ValueError("one qualified law cannot pool several denominator members")
        for identity, members in law_members.items():
            member = next(iter(members))
            law_coordinates = tuple(
                value
                for value in supported
                if value.response_law == identity and value.denominator_member_id == member
            )
            law = law_coordinates[0].qualification_result.response_law
            if law is None:  # pragma: no cover - coordinate invariant
                raise AssertionError("supported coordinate lost its terminal law")
            expected = {
                (law.chart_id, denominator_id)
                for denominator_id in law.obligations.support.denominator_cell_ids
            }
            observed = {(value.chart_id, value.denominator_cell_id) for value in law_coordinates}
            if observed != expected:
                raise ValueError(
                    "batch coordinates do not exactly cover each supported law's local support"
                )

    def _validate_overlaps(self) -> None:
        coverage: dict[str, set[ObjectIdentity]] = {}
        for coordinate in self.coordinates:
            if coordinate.response_law is None:
                continue
            coverage.setdefault(
                f"{coordinate.chart_id}.{coordinate.denominator_cell_id}", set()
            ).add(coordinate.response_law)
        expected = {key: values for key, values in coverage.items() if len(values) > 1}
        observed = {value.coverage_id: value for value in self.overlaps}
        if set(observed) != set(expected):
            raise ValueError("batch overlap ledger is incomplete or contains extras")
        evidence_ids = {link.link_id for link in self.evidence_links}
        for key, disposition in observed.items():
            expected_laws = tuple(sorted(expected[key], key=lambda value: value.object_id))
            if disposition.constituent_laws != expected_laws:
                raise ValueError("overlap disposition changes the covering law roster")
            if not set(disposition.evidence_link_ids) <= evidence_ids:
                raise ValueError("overlap disposition references unknown evidence")

    @property
    def supported_laws(self) -> tuple[ResponseLaw, ...]:
        by_id = {
            coordinate.qualification_result.response_law.law_id: (
                coordinate.qualification_result.response_law
            )
            for coordinate in self.coordinates
            if coordinate.qualification_result.response_law is not None
        }
        return tuple(by_id[value] for value in sorted(by_id))
