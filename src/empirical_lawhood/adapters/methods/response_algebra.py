"""Transparent response-algebra estimators for bounded word experiments.

The implementation deliberately uses direct contrasts, affine least squares,
finite transition counts and complete-independent-unit bootstrap intervals. It
contains no learned representation, neural model, LLM or adaptive search.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.response_algebra import ActionLetter, ActionQuotient, LetterActionWord, AdmissionAnnotation, AlgebraClosure, AlgebraRepresentation, PairScientificLabel, PortMateriality, ReceiverVisibility, ResponseAlgebraSignature, SequentialComposition, SimultaneousComposition, StateSufficiency, TemporalComposition
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_semantic_version,
    validate_stable_id,
)

from .contracts import DataSplit


class StateViewRole(StrEnum):
    PRIMARY_RECEIVER = "PRIMARY_RECEIVER"
    AUGMENTED_STATE = "AUGMENTED_STATE"
    FULL_STATE = "FULL_STATE"


@dataclass(frozen=True, slots=True)
class StateViewSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/state-view-spec'

    view_id: str
    role: StateViewRole
    coordinate_ids: tuple[str, ...]
    native_units: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.view_id, field_name="view_id")
        require_sorted_unique_strings(
            self.coordinate_ids,
            field_name="coordinate_ids",
            allow_empty=False,
        )
        if len(self.native_units) != len(self.coordinate_ids):
            raise ValueError("state-view units and coordinates differ in length")
        for unit in self.native_units:
            validate_nonempty(unit, field_name="native_units")


@dataclass(frozen=True, slots=True)
class StateVector(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/state-vector'

    vector_id: str
    view_id: str
    values: tuple[NamedDecimal, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.vector_id, field_name="vector_id")
        validate_stable_id(self.view_id, field_name="view_id")
        require_sorted_unique_ids(self.values, attribute="value_id", field_name="values")
        if not self.values:
            raise ValueError("state vector must not be empty")


@dataclass(frozen=True, slots=True)
class WordResponse(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/word-response'

    observation_id: str
    independent_unit_id: str
    denominator_id: str
    split: DataSplit
    numerical_view_id: str
    state_view_id: str
    word_id: str
    dose: Decimal
    delivered_word_record_id: str
    initial_state: StateVector
    final_state: StateVector

    def __post_init__(self) -> None:
        for name, value in (
            ("observation_id", self.observation_id),
            ("independent_unit_id", self.independent_unit_id),
            ("denominator_id", self.denominator_id),
            ("numerical_view_id", self.numerical_view_id),
            ("state_view_id", self.state_view_id),
            ("word_id", self.word_id),
            ("delivered_word_record_id", self.delivered_word_record_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_decimal(self.dose, field_name="dose", minimum=Decimal("0"))
        if self.initial_state.view_id != self.state_view_id:
            raise ValueError("word initial state uses another receiver/state view")
        if self.final_state.view_id != self.state_view_id:
            raise ValueError("word final state uses another receiver/state view")
        if tuple(value.value_id for value in self.initial_state.values) != tuple(
            value.value_id for value in self.final_state.values
        ):
            raise ValueError("word initial/final coordinates differ")
        if tuple(value.unit for value in self.initial_state.values) != tuple(
            value.unit for value in self.final_state.values
        ):
            raise ValueError("word initial/final native units differ")


@dataclass(frozen=True, slots=True)
class DeliveredWordRecord(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/delivered-word-record'

    record_id: str
    independent_unit_id: str
    split: DataSplit
    family_word_id: str
    dose: Decimal
    delivered_word: LetterActionWord

    def __post_init__(self) -> None:
        for name, value in (
            ("record_id", self.record_id),
            ("independent_unit_id", self.independent_unit_id),
            ("family_word_id", self.family_word_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_decimal(self.dose, field_name="dose", minimum=Decimal("0"))


@dataclass(frozen=True, slots=True)
class DeliveredGeneratorRecord(CanonicalRecord):
    """Exact four-stage delivery evidence for one signed local action."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/delivered-generator-record'

    record_id: str
    independent_unit_id: str
    split: DataSplit
    action_id: str
    signed_dose: Decimal
    delivered_letter: ActionLetter

    def __post_init__(self) -> None:
        for name, value in (
            ("record_id", self.record_id),
            ("independent_unit_id", self.independent_unit_id),
            ("action_id", self.action_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_decimal(self.signed_dose, field_name="signed_dose")
        if self.signed_dose == 0:
            raise ValueError("generator delivery requires a nonzero signed dose")
        if self.delivered_letter.letter_id != self.action_id:
            raise ValueError("generator delivery action and letter identities differ")


@dataclass(frozen=True, slots=True)
class AffineTransition(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/affine-transition'

    transition_id: str
    independent_unit_id: str
    denominator_id: str
    split: DataSplit
    numerical_view_id: str
    state_view_id: str
    start_anchor_id: str
    end_anchor_id: str
    duration: Decimal
    duration_unit: str
    start_state: StateVector
    end_state: StateVector

    def __post_init__(self) -> None:
        for name, value in (
            ("transition_id", self.transition_id),
            ("independent_unit_id", self.independent_unit_id),
            ("denominator_id", self.denominator_id),
            ("numerical_view_id", self.numerical_view_id),
            ("state_view_id", self.state_view_id),
            ("start_anchor_id", self.start_anchor_id),
            ("end_anchor_id", self.end_anchor_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.start_anchor_id == self.end_anchor_id:
            raise ValueError("affine transition requires distinct clock anchors")
        validate_decimal(self.duration, field_name="duration", minimum=Decimal("0"))
        if self.duration == 0:
            raise ValueError("affine transition duration must be positive")
        validate_nonempty(self.duration_unit, field_name="duration_unit")
        if self.start_state.view_id != self.state_view_id:
            raise ValueError("transition start state uses another state view")
        if self.end_state.view_id != self.state_view_id:
            raise ValueError("transition end state uses another state view")


@dataclass(frozen=True, slots=True)
class DiscreteTransition(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/discrete-transition'

    transition_id: str
    independent_unit_id: str
    split: DataSplit
    action_id: str
    start_state_id: str
    end_state_id: str

    def __post_init__(self) -> None:
        for name, value in (
            ("transition_id", self.transition_id),
            ("independent_unit_id", self.independent_unit_id),
            ("action_id", self.action_id),
            ("start_state_id", self.start_state_id),
            ("end_state_id", self.end_state_id),
        ):
            validate_stable_id(value, field_name=name)


@dataclass(frozen=True, slots=True)
class GeneratorTransition(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/generator-transition'

    transition_id: str
    independent_unit_id: str
    split: DataSplit
    action_id: str
    signed_dose: Decimal
    delivered_generator_record_id: str
    state_view_id: str
    start_state: StateVector
    end_state: StateVector

    def __post_init__(self) -> None:
        for name, value in (
            ("transition_id", self.transition_id),
            ("independent_unit_id", self.independent_unit_id),
            ("action_id", self.action_id),
            ("delivered_generator_record_id", self.delivered_generator_record_id),
            ("state_view_id", self.state_view_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_decimal(self.signed_dose, field_name="signed_dose")
        if self.signed_dose == 0:
            raise ValueError("generator transition requires a nonzero signed dose")
        if self.start_state.view_id != self.state_view_id:
            raise ValueError("generator start state uses another state view")
        if self.end_state.view_id != self.state_view_id:
            raise ValueError("generator end state uses another state view")


@dataclass(frozen=True, slots=True)
class ResponseAlgebraMethodInput(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-algebra-method-input'

    case_token: str
    state_views: tuple[StateViewSpec, ...]
    delivered_words: tuple[DeliveredWordRecord, ...]
    delivered_generators: tuple[DeliveredGeneratorRecord, ...]
    word_responses: tuple[WordResponse, ...]
    affine_transitions: tuple[AffineTransition, ...]
    discrete_transitions: tuple[DiscreteTransition, ...]
    generator_transitions: tuple[GeneratorTransition, ...]
    two_sided_actions_delivered: bool
    local_state_closed: bool
    hybrid_switching_observed: bool
    evaluator_reveal_attestation_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.case_token, field_name="case_token")
        validate_stable_id(
            self.evaluator_reveal_attestation_id,
            field_name="evaluator_reveal_attestation_id",
        )
        require_sorted_unique_ids(self.state_views, attribute="view_id", field_name="state_views")
        require_sorted_unique_ids(
            self.delivered_words,
            attribute="record_id",
            field_name="delivered_words",
        )
        if not self.delivered_words:
            raise ValueError("response-algebra input requires delivered-word records")
        require_sorted_unique_ids(
            self.delivered_generators,
            attribute="record_id",
            field_name="delivered_generators",
        )
        require_sorted_unique_ids(
            self.word_responses,
            attribute="observation_id",
            field_name="word_responses",
        )
        if not self.word_responses:
            raise ValueError("response-algebra input requires word responses")
        require_sorted_unique_ids(
            self.affine_transitions,
            attribute="transition_id",
            field_name="affine_transitions",
        )
        require_sorted_unique_ids(
            self.discrete_transitions,
            attribute="transition_id",
            field_name="discrete_transitions",
        )
        require_sorted_unique_ids(
            self.generator_transitions,
            attribute="transition_id",
            field_name="generator_transitions",
        )
        view_ids = {view.view_id for view in self.state_views}
        if any(value.state_view_id not in view_ids for value in self.word_responses):
            raise ValueError("word response references an undeclared state view")
        if any(value.state_view_id not in view_ids for value in self.affine_transitions):
            raise ValueError("affine transition references an undeclared state view")
        if any(value.state_view_id not in view_ids for value in self.generator_transitions):
            raise ValueError("generator transition references an undeclared state view")
        units_by_view = {
            view.view_id: dict(zip(view.coordinate_ids, view.native_units, strict=True))
            for view in self.state_views
        }
        vectors = (
            *(
                state
                for response in self.word_responses
                for state in (response.initial_state, response.final_state)
            ),
            *(
                state
                for transition in self.affine_transitions
                for state in (transition.start_state, transition.end_state)
            ),
            *(
                state
                for transition in self.generator_transitions
                for state in (transition.start_state, transition.end_state)
            ),
        )
        for state in vectors:
            observed = {value.value_id: value.unit for value in state.values}
            if observed != units_by_view[state.view_id]:
                raise ValueError(
                    "state-vector coordinates/units differ from its declared state view"
                )
        delivered_by_id = {value.record_id: value for value in self.delivered_words}
        for response in self.word_responses:
            delivered = delivered_by_id.get(response.delivered_word_record_id)
            if delivered is None:
                raise ValueError("word response lacks its delivered-word record")
            if (
                delivered.independent_unit_id != response.independent_unit_id
                or delivered.split is not response.split
                or delivered.family_word_id != response.word_id
                or delivered.dose != response.dose
                or delivered.delivered_word.denominator_id != response.denominator_id
            ):
                raise ValueError("word response and delivered-word semantics differ")
        delivered_generators_by_id = {value.record_id: value for value in self.delivered_generators}
        for transition in self.generator_transitions:
            generator_delivery = delivered_generators_by_id.get(
                transition.delivered_generator_record_id
            )
            if generator_delivery is None:
                raise ValueError("generator transition lacks its delivery record")
            if (
                generator_delivery.independent_unit_id != transition.independent_unit_id
                or generator_delivery.split is not transition.split
                or generator_delivery.action_id != transition.action_id
                or generator_delivery.signed_dose != transition.signed_dose
            ):
                raise ValueError("generator transition and delivery semantics differ")
        if self.two_sided_actions_delivered and not self.delivered_generators:
            raise ValueError("two-sided action eligibility requires exact generator deliveries")


@dataclass(frozen=True, slots=True)
class ReceiverEquivalenceCriterion(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-equivalence-criterion'

    criterion_id: str
    coordinate_id: str
    native_unit: str
    floor_upper: Decimal
    equivalence_width: Decimal
    materiality_lower: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.criterion_id, field_name="criterion_id")
        validate_stable_id(self.coordinate_id, field_name="coordinate_id")
        validate_nonempty(self.native_unit, field_name="native_unit")
        for name, value in (
            ("floor_upper", self.floor_upper),
            ("equivalence_width", self.equivalence_width),
            ("materiality_lower", self.materiality_lower),
        ):
            validate_decimal(value, field_name=name, minimum=Decimal("0"))
        if not self.floor_upper < self.equivalence_width <= self.materiality_lower:
            raise ValueError("criterion requires floor < equivalence <= materiality")


@dataclass(frozen=True, slots=True)
class ResponseAlgebraMethodConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-algebra-method-config'

    config_id: str
    method_key: str
    method_version: str
    primary_numerical_view_id: str
    primary_state_view_id: str
    augmented_state_view_id: str | None
    full_state_view_id: str | None
    receiver_criteria: tuple[ReceiverEquivalenceCriterion, ...]
    confidence_level: Decimal
    bootstrap_replicates: int
    bootstrap_seed: int
    affine_ridge: Decimal
    maximum_affine_closure_error: Decimal
    maximum_stationarity_error: Decimal
    minimum_wrong_horizon_ratio: Decimal
    minimum_stochastic_commutator: Decimal
    maximum_dose_scaling_relative_error: Decimal
    minimum_independent_units: int
    require_exact_stage_value_equality: bool
    require_exact_stage_clock_equality: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("config_id", self.config_id),
            ("method_key", self.method_key),
            ("primary_numerical_view_id", self.primary_numerical_view_id),
            ("primary_state_view_id", self.primary_state_view_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.method_key != "response-algebra.direct-affine-finite":
            raise ValueError("response-algebra config selects another static method")
        validate_semantic_version(self.method_version)
        if self.method_version != "1.0.0":
            raise ValueError("unsupported response-algebra method version")
        for view_name, view_id in (
            ("augmented_state_view_id", self.augmented_state_view_id),
            ("full_state_view_id", self.full_state_view_id),
        ):
            if view_id is not None:
                validate_stable_id(view_id, field_name=view_name)
        require_sorted_unique_ids(
            self.receiver_criteria,
            attribute="criterion_id",
            field_name="receiver_criteria",
        )
        if not self.receiver_criteria:
            raise ValueError("method config requires native-unit receiver criteria")
        validate_decimal(self.confidence_level, field_name="confidence_level")
        if not Decimal("0") < self.confidence_level < Decimal("1"):
            raise ValueError("confidence level must be strictly between zero and one")
        if not 100 <= self.bootstrap_replicates <= 100_000:
            raise ValueError("bootstrap replicate count must be in [100, 100000]")
        if self.bootstrap_seed < 0:
            raise ValueError("bootstrap seed must be nonnegative")
        for threshold_name, threshold in (
            ("affine_ridge", self.affine_ridge),
            ("maximum_affine_closure_error", self.maximum_affine_closure_error),
            ("maximum_stationarity_error", self.maximum_stationarity_error),
            ("minimum_wrong_horizon_ratio", self.minimum_wrong_horizon_ratio),
            ("minimum_stochastic_commutator", self.minimum_stochastic_commutator),
            ("maximum_dose_scaling_relative_error", self.maximum_dose_scaling_relative_error),
        ):
            validate_decimal(threshold, field_name=threshold_name, minimum=Decimal("0"))
        if self.minimum_independent_units < 2:
            raise ValueError("method requires at least two independent units")


@dataclass(frozen=True, slots=True)
class ResponseAlgebraMethodResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-algebra-method-result'

    result_id: str
    case_token: str
    method_key: str
    method_version: str
    signature: ResponseAlgebraSignature
    pair_label: PairScientificLabel
    smooth_lie_eligible: bool
    metrics: tuple[NamedDecimal, ...]
    independent_unit_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("result_id", self.result_id),
            ("case_token", self.case_token),
            ("method_key", self.method_key),
        ):
            validate_stable_id(value, field_name=name)
        validate_semantic_version(self.method_version)
        require_sorted_unique_ids(self.metrics, attribute="value_id", field_name="metrics")
        require_sorted_unique_strings(
            self.independent_unit_ids,
            field_name="independent_unit_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")


@dataclass(frozen=True, slots=True)
class ResponseAlgebraUnitContrast(CanonicalRecord):
    """One complete held-out preparation's native-coordinate word contrasts."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-algebra-unit-contrast'

    contrast_id: str
    case_token: str
    independent_unit_id: str
    state_view_id: str
    coordinate_id: str
    native_unit: str
    port_a_effect: Decimal
    port_b_effect: Decimal
    simultaneous_interaction: Decimal
    raw_order: Decimal
    lti_timing_prediction: Decimal
    controlled_order: Decimal

    def __post_init__(self) -> None:
        for name, value in (
            ("contrast_id", self.contrast_id),
            ("case_token", self.case_token),
            ("independent_unit_id", self.independent_unit_id),
            ("state_view_id", self.state_view_id),
            ("coordinate_id", self.coordinate_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_nonempty(self.native_unit, field_name="native_unit")
        for name, decimal_value in (
            ("port_a_effect", self.port_a_effect),
            ("port_b_effect", self.port_b_effect),
            ("simultaneous_interaction", self.simultaneous_interaction),
            ("raw_order", self.raw_order),
            ("lti_timing_prediction", self.lti_timing_prediction),
            ("controlled_order", self.controlled_order),
        ):
            validate_decimal(decimal_value, field_name=name)
        if self.port_a_effect < 0 or self.port_b_effect < 0:
            raise ValueError("constituent-port effect magnitudes cannot be negative")


@dataclass(frozen=True, slots=True)
class ResponseAlgebraUnitContrastTable(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-algebra-unit-contrast-table'

    table_id: str
    run_id: str
    contrasts: tuple[ResponseAlgebraUnitContrast, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.table_id, field_name="table_id")
        validate_stable_id(self.run_id, field_name="run_id")
        require_sorted_unique_ids(
            self.contrasts,
            attribute="contrast_id",
            field_name="contrasts",
        )
        if not self.contrasts:
            raise ValueError("unit contrast table must not be empty")


@dataclass(frozen=True, slots=True)
class _Interval:
    mean: float
    lower: float
    upper: float


def _vector(state: StateVector) -> npt.NDArray[np.float64]:
    return np.asarray(tuple(float(value.value) for value in state.values), dtype=np.float64)


def _decimal(value: float) -> Decimal:
    if not np.isfinite(value):
        raise ValueError("response-algebra method produced a non-finite value")
    return Decimal(f"{value:.15g}")


def _bootstrap_interval(
    values: npt.NDArray[np.float64],
    *,
    confidence_level: float,
    replicates: int,
    seed: int,
) -> _Interval:
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("bootstrap requires a finite per-unit scalar vector")
    if values.size == 1:
        value = float(values[0])
        return _Interval(value, value, value)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, values.size, size=(replicates, values.size))
    boot = np.mean(values[indices], axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    return _Interval(
        mean=float(np.mean(values)),
        lower=float(np.quantile(boot, alpha)),
        upper=float(np.quantile(boot, 1.0 - alpha)),
    )


def _homogeneous_fit(
    starts: npt.NDArray[np.float64],
    ends: npt.NDArray[np.float64],
    ridge: float,
) -> npt.NDArray[np.float64]:
    if starts.ndim != 2 or ends.ndim != 2 or starts.shape != ends.shape:
        raise ValueError("affine fit requires matched finite state matrices")
    design = np.column_stack((starts, np.ones(starts.shape[0], dtype=np.float64)))
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
    penalty[-1, -1] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ ends)
    dimension = starts.shape[1]
    result = np.eye(dimension + 1, dtype=np.float64)
    result[:dimension, :dimension] = coefficients[:dimension, :].T
    result[:dimension, dimension] = coefficients[dimension, :]
    return result


def _apply_homogeneous(
    operator: npt.NDArray[np.float64],
    states: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    augmented = np.column_stack((states, np.ones(states.shape[0], dtype=np.float64)))
    return (operator @ augmented.T).T[:, :-1]


def _word_table(
    method_input: ResponseAlgebraMethodInput,
    config: ResponseAlgebraMethodConfig,
    view_id: str,
) -> dict[str, dict[str, npt.NDArray[np.float64]]]:
    selected = (
        value
        for value in method_input.word_responses
        if value.split is DataSplit.HELD_OUT
        and value.numerical_view_id == config.primary_numerical_view_id
        and value.state_view_id == view_id
        and value.dose == Decimal("1")
    )
    table: dict[str, dict[str, npt.NDArray[np.float64]]] = defaultdict(dict)
    for value in selected:
        if value.word_id in table[value.independent_unit_id]:
            raise ValueError("word table contains a duplicate unit/word observation")
        table[value.independent_unit_id][value.word_id] = _vector(value.final_state)
    return dict(table)


_REQUIRED_WORDS = frozenset(
    {
        "identity",
        "a-early",
        "a-late",
        "b-early",
        "b-late",
        "a-repeat",
        "b-repeat",
        "a-then-b",
        "b-then-a",
        "simultaneous-a-b",
    }
)


def response_algebra_unit_contrasts(
    method_input: ResponseAlgebraMethodInput,
    config: ResponseAlgebraMethodConfig,
) -> tuple[ResponseAlgebraUnitContrast, ...]:
    """Expose signed, complete-unit direct estimands without nested-row pooling."""

    table = _word_table(method_input, config, config.primary_state_view_id)
    if any(set(words) != _REQUIRED_WORDS for words in table.values()):
        raise ValueError("unit contrast table lacks the complete frozen word family")
    view = next(
        value for value in method_input.state_views if value.view_id == config.primary_state_view_id
    )
    values = []
    for independent_unit_id in sorted(table):
        words = table[independent_unit_id]
        for coordinate_index, (coordinate_id, native_unit) in enumerate(
            zip(view.coordinate_ids, view.native_units, strict=True)
        ):
            identity = words["identity"][coordinate_index]
            a_early = words["a-early"][coordinate_index]
            a_late = words["a-late"][coordinate_index]
            b_early = words["b-early"][coordinate_index]
            b_late = words["b-late"][coordinate_index]
            simultaneous = words["simultaneous-a-b"][coordinate_index]
            ab = words["a-then-b"][coordinate_index]
            ba = words["b-then-a"][coordinate_index]
            raw = ab - ba
            lti = a_early + b_late - b_early - a_late
            values.append(
                ResponseAlgebraUnitContrast(
                    contrast_id=(
                        f"unit-contrast.{method_input.case_token}."
                        f"{independent_unit_id}.{coordinate_id}"
                    ),
                    case_token=method_input.case_token,
                    independent_unit_id=independent_unit_id,
                    state_view_id=view.view_id,
                    coordinate_id=coordinate_id,
                    native_unit=native_unit,
                    port_a_effect=_decimal(
                        (abs(a_early - identity) + abs(a_late - identity)) / 2.0
                    ),
                    port_b_effect=_decimal(
                        (abs(b_early - identity) + abs(b_late - identity)) / 2.0
                    ),
                    simultaneous_interaction=_decimal(simultaneous - a_early - b_early + identity),
                    raw_order=_decimal(raw),
                    lti_timing_prediction=_decimal(lti),
                    controlled_order=_decimal(raw - lti),
                )
            )
    return tuple(sorted(values, key=lambda value: value.contrast_id))


def _direct_assessment(
    method_input: ResponseAlgebraMethodInput,
    config: ResponseAlgebraMethodConfig,
    view_id: str,
    *,
    seed_offset: int,
) -> tuple[
    PortMateriality,
    PortMateriality,
    SimultaneousComposition,
    SequentialComposition,
    PairScientificLabel,
    tuple[NamedDecimal, ...],
]:
    table = _word_table(method_input, config, view_id)
    if len(table) < config.minimum_independent_units:
        raise ValueError("direct assessment has too few independent units")
    if any(set(words) != _REQUIRED_WORDS for words in table.values()):
        raise ValueError("direct assessment lacks the complete frozen word family")
    units = tuple(sorted(table))
    dimension = len(next(iter(table.values()))["identity"])
    criteria = {value.coordinate_id: value for value in config.receiver_criteria}
    view = next(value for value in method_input.state_views if value.view_id == view_id)
    if not set(view.coordinate_ids) <= set(criteria):
        raise ValueError("receiver criteria omit state-view coordinates")
    for coordinate_id, unit in zip(view.coordinate_ids, view.native_units, strict=True):
        if criteria[coordinate_id].native_unit != unit:
            raise ValueError("receiver criterion unit differs from its state-view unit")

    port_a_intervals: list[_Interval] = []
    port_b_intervals: list[_Interval] = []
    simultaneous_intervals: list[_Interval] = []
    raw_intervals: list[_Interval] = []
    controlled_intervals: list[_Interval] = []
    metrics: list[NamedDecimal] = []
    for coordinate_index in range(dimension):
        coordinate_id = view.coordinate_ids[coordinate_index]
        criterion = criteria[coordinate_id]
        identity = np.asarray(
            [table[unit]["identity"][coordinate_index] for unit in units],
            dtype=np.float64,
        )
        a_early = np.asarray(
            [table[unit]["a-early"][coordinate_index] for unit in units],
            dtype=np.float64,
        )
        a_late = np.asarray(
            [table[unit]["a-late"][coordinate_index] for unit in units],
            dtype=np.float64,
        )
        b_early = np.asarray(
            [table[unit]["b-early"][coordinate_index] for unit in units],
            dtype=np.float64,
        )
        b_late = np.asarray(
            [table[unit]["b-late"][coordinate_index] for unit in units],
            dtype=np.float64,
        )
        simultaneous = np.asarray(
            [table[unit]["simultaneous-a-b"][coordinate_index] for unit in units],
            dtype=np.float64,
        )
        ab = np.asarray(
            [table[unit]["a-then-b"][coordinate_index] for unit in units],
            dtype=np.float64,
        )
        ba = np.asarray(
            [table[unit]["b-then-a"][coordinate_index] for unit in units],
            dtype=np.float64,
        )
        a_effect = (np.abs(a_early - identity) + np.abs(a_late - identity)) / 2.0
        b_effect = (np.abs(b_early - identity) + np.abs(b_late - identity)) / 2.0
        interaction = np.abs(simultaneous - a_early - b_early + identity)
        raw = np.abs(ab - ba)
        lti = a_early + b_late - b_early - a_late
        controlled = np.abs((ab - ba) - lti)
        vectors = (a_effect, b_effect, interaction, raw, controlled)
        intervals = tuple(
            _bootstrap_interval(
                values,
                confidence_level=float(config.confidence_level),
                replicates=config.bootstrap_replicates,
                seed=config.bootstrap_seed + seed_offset + coordinate_index * 17 + index,
            )
            for index, values in enumerate(vectors)
        )
        port_a_intervals.append(intervals[0])
        port_b_intervals.append(intervals[1])
        simultaneous_intervals.append(intervals[2])
        raw_intervals.append(intervals[3])
        controlled_intervals.append(intervals[4])
        for metric_name, interval in zip(
            ("port-a", "port-b", "simultaneous", "raw-order", "controlled-order"),
            intervals,
            strict=True,
        ):
            for bound_name, bound in (
                ("mean", interval.mean),
                ("lower", interval.lower),
                ("upper", interval.upper),
            ):
                metrics.append(
                    NamedDecimal(
                        value_id=f"{metric_name}.{coordinate_id}.{bound_name}",
                        value=_decimal(bound),
                        unit=criterion.native_unit,
                    )
                )

    def materiality(intervals: list[_Interval]) -> PortMateriality:
        if any(
            interval.lower > float(criteria[coordinate].materiality_lower)
            for interval, coordinate in zip(intervals, view.coordinate_ids, strict=True)
        ):
            return PortMateriality.MATERIAL
        if all(
            interval.upper < float(criteria[coordinate].equivalence_width)
            for interval, coordinate in zip(intervals, view.coordinate_ids, strict=True)
        ):
            return PortMateriality.NULL
        return PortMateriality.MIXED

    def equivalent(intervals: list[_Interval]) -> bool:
        return all(
            interval.upper < float(criteria[coordinate].equivalence_width)
            for interval, coordinate in zip(intervals, view.coordinate_ids, strict=True)
        )

    def effect_material(intervals: list[_Interval]) -> bool:
        return any(
            interval.lower > float(criteria[coordinate].materiality_lower)
            for interval, coordinate in zip(intervals, view.coordinate_ids, strict=True)
        )

    material_a = materiality(port_a_intervals)
    material_b = materiality(port_b_intervals)
    both_material = (
        material_a is PortMateriality.MATERIAL and material_b is PortMateriality.MATERIAL
    )
    if equivalent(simultaneous_intervals):
        simultaneous_status = (
            SimultaneousComposition.ADDITIVE_EQUIVALENT
            if both_material
            else SimultaneousComposition.BELOW_RESOLUTION
        )
    elif effect_material(simultaneous_intervals) and both_material:
        simultaneous_status = SimultaneousComposition.NONLINEAR_INTERACTION
    else:
        simultaneous_status = SimultaneousComposition.UNEVALUABLE

    raw_material = effect_material(raw_intervals)
    if equivalent(controlled_intervals):
        if both_material:
            sequential_status = (
                SequentialComposition.LTI_OR_DRIFT_EXPLAINED
                if raw_material
                else SequentialComposition.COMMUTATOR_EQUIVALENT
            )
        else:
            sequential_status = SequentialComposition.BELOW_RESOLUTION
    elif effect_material(controlled_intervals) and both_material:
        sequential_status = SequentialComposition.MATERIAL_NONCOMMUTATIVE
    else:
        sequential_status = SequentialComposition.UNEVALUABLE

    if not both_material:
        pair_label = (
            PairScientificLabel.NO_MATERIAL_RESPONSE
            if material_a is PortMateriality.NULL and material_b is PortMateriality.NULL
            else PairScientificLabel.COMPOSITION_NULL_AT_RESOLUTION
        )
    elif sequential_status is SequentialComposition.LTI_OR_DRIFT_EXPLAINED:
        pair_label = PairScientificLabel.PATH_ORDER_EXPLAINED_BY_LTI_OR_DRIFT
    elif sequential_status is SequentialComposition.MATERIAL_NONCOMMUTATIVE:
        pair_label = PairScientificLabel.FINITE_NONCOMMUTATIVE_CLOSED
    elif simultaneous_status is SimultaneousComposition.NONLINEAR_INTERACTION:
        pair_label = PairScientificLabel.NONLINEAR_COMMUTATIVE
    elif (
        simultaneous_status is SimultaneousComposition.ADDITIVE_EQUIVALENT
        and sequential_status is SequentialComposition.COMMUTATOR_EQUIVALENT
    ):
        pair_label = PairScientificLabel.MATERIAL_AFFINE_COMMUTATIVE
    else:
        pair_label = PairScientificLabel.UNEVALUABLE
    return (
        material_a,
        material_b,
        simultaneous_status,
        sequential_status,
        pair_label,
        tuple(sorted(metrics, key=lambda value: value.value_id)),
    )


@dataclass(frozen=True, slots=True)
class _TemporalResult:
    status: TemporalComposition
    closure_error: float
    stationarity_error: float
    wrong_horizon_error: float
    denominator_stable: bool


def _temporal_assessment(
    method_input: ResponseAlgebraMethodInput,
    config: ResponseAlgebraMethodConfig,
    view_id: str,
) -> _TemporalResult:
    selected = tuple(
        value
        for value in method_input.affine_transitions
        if value.numerical_view_id == config.primary_numerical_view_id
        and value.state_view_id == view_id
    )
    if not selected:
        return _TemporalResult(TemporalComposition.UNEVALUABLE, np.nan, np.nan, np.nan, True)
    by_edge_split: dict[tuple[str, str, DataSplit], list[AffineTransition]] = defaultdict(list)
    for transition in selected:
        by_edge_split[
            (transition.start_anchor_id, transition.end_anchor_id, transition.split)
        ].append(transition)
    edges = (("t0", "t1"), ("t1", "t2"), ("t0", "t2"))
    if any((start, end, split) not in by_edge_split for start, end in edges for split in DataSplit):
        return _TemporalResult(TemporalComposition.UNEVALUABLE, np.nan, np.nan, np.nan, True)

    operators: dict[tuple[str, str], npt.NDArray[np.float64]] = {}
    for start, end in edges:
        development = sorted(
            by_edge_split[(start, end, DataSplit.CALIBRATION)],
            key=lambda value: value.transition_id,
        )
        operators[(start, end)] = _homogeneous_fit(
            np.asarray([_vector(value.start_state) for value in development]),
            np.asarray([_vector(value.end_state) for value in development]),
            float(config.affine_ridge),
        )
    evaluation = sorted(
        by_edge_split[("t0", "t2", DataSplit.HELD_OUT)],
        key=lambda value: value.transition_id,
    )
    starts = np.asarray([_vector(value.start_state) for value in evaluation])
    observed = np.asarray([_vector(value.end_state) for value in evaluation])
    composed = operators[("t1", "t2")] @ operators[("t0", "t1")]
    closure_error = float(np.sqrt(np.mean((_apply_homogeneous(composed, starts) - observed) ** 2)))
    wrong = _apply_homogeneous(operators[("t0", "t1")], starts)
    wrong_horizon_error = float(np.sqrt(np.mean((wrong - observed) ** 2)))
    stationarity_error = float(
        np.linalg.norm(operators[("t0", "t1")] - operators[("t1", "t2")], ord="fro")
    )
    denominator_stable = len({value.denominator_id for value in selected}) == 1
    closure_pass = closure_error <= float(config.maximum_affine_closure_error)
    if not closure_pass:
        status = TemporalComposition.NONCLOSED
    elif not denominator_stable:
        status = TemporalComposition.NONSTATIONARY_COCYCLE
    elif stationarity_error <= float(config.maximum_stationarity_error):
        specificity = wrong_horizon_error >= (
            max(closure_error, 1e-15) * float(config.minimum_wrong_horizon_ratio)
        )
        status = (
            TemporalComposition.STATIONARY_SEMIGROUP
            if specificity
            else TemporalComposition.UNEVALUABLE
        )
    else:
        status = TemporalComposition.NONSTATIONARY_COCYCLE
    return _TemporalResult(
        status,
        closure_error,
        stationarity_error,
        wrong_horizon_error,
        denominator_stable,
    )


def _finite_stochastic_commutator(
    method_input: ResponseAlgebraMethodInput,
) -> float | None:
    development = tuple(
        value for value in method_input.discrete_transitions if value.split is DataSplit.CALIBRATION
    )
    if not development:
        return None
    state_ids = tuple(
        sorted(
            {value.start_state_id for value in development}
            | {value.end_state_id for value in development}
        )
    )
    index = {value: position for position, value in enumerate(state_ids)}

    def matrix(action_id: str) -> npt.NDArray[np.float64]:
        counts = np.zeros((len(state_ids), len(state_ids)), dtype=np.float64)
        for value in development:
            if value.action_id == action_id:
                counts[index[value.start_state_id], index[value.end_state_id]] += 1.0
        row_sums = counts.sum(axis=1)
        if np.any(row_sums == 0):
            raise ValueError("stochastic transition support omits a source state")
        return np.asarray(counts / row_sums[:, None], dtype=np.float64)

    action_ids = tuple(sorted({value.action_id for value in development}))
    if action_ids != ("a", "b"):
        raise ValueError("stochastic response algebra requires exact a/b actions")
    a = matrix("a")
    b = matrix("b")
    return float(np.linalg.norm(a @ b - b @ a, ord="fro"))


def _smooth_bracket_diagnostics(
    method_input: ResponseAlgebraMethodInput,
    config: ResponseAlgebraMethodConfig,
) -> tuple[bool, float, float]:
    if not method_input.two_sided_actions_delivered or not method_input.local_state_closed:
        return False, 0.0, 0.0
    transitions = tuple(
        value
        for value in method_input.generator_transitions
        if value.split is DataSplit.CALIBRATION
        and value.state_view_id == (config.full_state_view_id or config.primary_state_view_id)
    )
    positive_doses = tuple(
        sorted({value.signed_dose for value in transitions if value.signed_dose > 0})
    )
    if len(positive_doses) < 3:
        return False, 0.0, 0.0
    smallest = positive_doses[0]
    generators: dict[str, npt.NDArray[np.float64]] = {}
    for action_id in ("a", "b"):
        operators: dict[int, npt.NDArray[np.float64]] = {}
        for sign in (-1, 1):
            selected = sorted(
                (
                    value
                    for value in transitions
                    if value.action_id == action_id and value.signed_dose == smallest * sign
                ),
                key=lambda value: value.transition_id,
            )
            if not selected:
                return False, 0.0, 0.0
            operators[sign] = _homogeneous_fit(
                np.asarray([_vector(value.start_state) for value in selected]),
                np.asarray([_vector(value.end_state) for value in selected]),
                float(config.affine_ridge),
            )
        generators[action_id] = (operators[1] - operators[-1]) / (2.0 * float(smallest))
    bracket = generators["b"] @ generators["a"] - generators["a"] @ generators["b"]
    bracket_norm = float(np.linalg.norm(bracket[:-1, :], ord="fro"))

    word_by_dose: dict[Decimal, list[float]] = defaultdict(list)
    full_view = config.full_state_view_id or config.primary_state_view_id
    grouped: dict[tuple[Decimal, str], dict[str, npt.NDArray[np.float64]]] = defaultdict(dict)
    for value in method_input.word_responses:
        if value.split is DataSplit.HELD_OUT and value.state_view_id == full_view:
            if value.word_id in {"a-then-b", "b-then-a"} and value.dose in positive_doses:
                grouped[(value.dose, value.independent_unit_id)][value.word_id] = _vector(
                    value.final_state
                )
    for (dose, _unit), words in grouped.items():
        if set(words) == {"a-then-b", "b-then-a"}:
            word_by_dose[dose].append(float(np.linalg.norm(words["a-then-b"] - words["b-then-a"])))
    if any(dose not in word_by_dose for dose in positive_doses):
        return False, bracket_norm, 0.0
    scaled = np.asarray(
        [np.mean(word_by_dose[dose]) / float(dose * dose) for dose in positive_doses],
        dtype=np.float64,
    )
    scaled_mean = float(np.mean(scaled))
    relative_error = float(
        (float(np.max(scaled)) - float(np.min(scaled))) / max(scaled_mean, 1e-15)
    )
    eligible = relative_error <= float(config.maximum_dose_scaling_relative_error)
    return eligible, bracket_norm, relative_error


def identify_response_algebra(
    method_input: ResponseAlgebraMethodInput,
    config: ResponseAlgebraMethodConfig,
) -> ResponseAlgebraMethodResult:
    """Identify one truth-blind local signature under a frozen method config."""

    view_by_role = {value.role: value for value in method_input.state_views}
    primary = view_by_role.get(StateViewRole.PRIMARY_RECEIVER)
    if primary is None or primary.view_id != config.primary_state_view_id:
        raise ValueError("method config primary view differs from the declared receiver")
    held_out_delivery = tuple(
        value
        for value in method_input.delivered_words
        if value.split is DataSplit.HELD_OUT and value.dose == Decimal("1")
    )
    if not held_out_delivery or any(
        not value.delivered_word.supported for value in held_out_delivery
    ):
        raise ValueError("held-out action delivery or prefix support is incomplete")
    audited_letters = tuple(
        letter for delivered in held_out_delivery for letter in delivered.delivered_word.letters
    ) + tuple(
        delivered.delivered_letter
        for delivered in method_input.delivered_generators
        if delivered.split is DataSplit.CALIBRATION
    )
    if config.require_exact_stage_value_equality:
        for letter in audited_letters:
            stage_values = {
                letter.requested.value,
                letter.accepted.value,
                letter.applied.value,
                letter.realized.value,
            }
            if len(stage_values) != 1:
                raise ValueError("held-out requested/accepted/applied/realized values differ")
        for word_delivery in held_out_delivery:
            if any(
                abs(letter.realized.value) != word_delivery.dose
                for letter in word_delivery.delivered_word.letters
            ):
                raise ValueError("held-out realized word dose differs from its frozen dose")
        for generator_delivery in method_input.delivered_generators:
            if (
                generator_delivery.split is DataSplit.CALIBRATION
                and generator_delivery.delivered_letter.realized.value
                != generator_delivery.signed_dose
            ):
                raise ValueError("generator realized dose differs from its frozen signed dose")
    if config.require_exact_stage_clock_equality:
        for letter in audited_letters:
            stage_clocks = {
                (letter.requested.clock_id, letter.requested.clock_coordinate),
                (letter.accepted.clock_id, letter.accepted.clock_coordinate),
                (letter.applied.clock_id, letter.applied.clock_coordinate),
                (letter.realized.clock_id, letter.realized.clock_coordinate),
            }
            if len(stage_clocks) != 1:
                raise ValueError("requested/accepted/applied/realized action clocks differ")
    unit_ids = tuple(
        sorted(
            {
                value.independent_unit_id
                for value in method_input.word_responses
                if value.split is DataSplit.HELD_OUT
                and value.state_view_id == config.primary_state_view_id
            }
        )
    )
    if len(unit_ids) < config.minimum_independent_units:
        raise ValueError("method input has too few held-out independent units")
    (
        material_a,
        material_b,
        simultaneous,
        sequential,
        pair_label,
        direct_metrics,
    ) = _direct_assessment(method_input, config, config.primary_state_view_id, seed_offset=0)
    primary_temporal = _temporal_assessment(method_input, config, config.primary_state_view_id)
    augmented_temporal = (
        _temporal_assessment(method_input, config, config.augmented_state_view_id)
        if config.augmented_state_view_id is not None
        else None
    )
    full_temporal = (
        _temporal_assessment(method_input, config, config.full_state_view_id)
        if config.full_state_view_id is not None
        else None
    )

    full_sequential = sequential
    if config.full_state_view_id is not None:
        full_sequential = _direct_assessment(
            method_input,
            config,
            config.full_state_view_id,
            seed_offset=10_000,
        )[3]
    if full_sequential is SequentialComposition.MATERIAL_NONCOMMUTATIVE and sequential in {
        SequentialComposition.BELOW_RESOLUTION,
        SequentialComposition.COMMUTATOR_EQUIVALENT,
    }:
        receiver_visibility = ReceiverVisibility.BRACKET_HIDDEN_BY_PROJECTION
    elif (
        primary_temporal.status is TemporalComposition.NONCLOSED
        and augmented_temporal is not None
        and augmented_temporal.status
        in {TemporalComposition.STATIONARY_SEMIGROUP, TemporalComposition.NONSTATIONARY_COCYCLE}
    ):
        receiver_visibility = ReceiverVisibility.GAUGE_DEPENDENT
    else:
        receiver_visibility = ReceiverVisibility.FAITHFUL_AT_TESTED_RESOLUTION

    if not primary_temporal.denominator_stable:
        state_sufficiency = StateSufficiency.TIME_VARYING_DENOMINATOR
    elif primary_temporal.status is TemporalComposition.NONCLOSED:
        if augmented_temporal is not None and augmented_temporal.status in {
            TemporalComposition.STATIONARY_SEMIGROUP,
            TemporalComposition.NONSTATIONARY_COCYCLE,
        }:
            state_sufficiency = StateSufficiency.FINITE_HISTORY_SUFFICIENT
        elif full_temporal is not None and full_temporal.status in {
            TemporalComposition.STATIONARY_SEMIGROUP,
            TemporalComposition.NONSTATIONARY_COCYCLE,
        }:
            state_sufficiency = StateSufficiency.HIDDEN_STATE_LIMITED
        else:
            state_sufficiency = StateSufficiency.UNEVALUABLE
    else:
        state_sufficiency = StateSufficiency.MARKOV_SUFFICIENT

    temporal = primary_temporal.status
    if (
        temporal is TemporalComposition.NONCLOSED
        and augmented_temporal is not None
        and augmented_temporal.status
        in {TemporalComposition.STATIONARY_SEMIGROUP, TemporalComposition.NONSTATIONARY_COCYCLE}
    ):
        temporal = TemporalComposition.HISTORY_AUGMENTED_CLOSURE

    stochastic_commutator = _finite_stochastic_commutator(method_input)
    smooth_eligible, bracket_norm, dose_scaling_error = _smooth_bracket_diagnostics(
        method_input,
        config,
    )
    if stochastic_commutator is not None:
        representation = AlgebraRepresentation.STOCHASTIC_KERNEL
        if stochastic_commutator > float(config.minimum_stochastic_commutator):
            sequential = SequentialComposition.MATERIAL_NONCOMMUTATIVE
            pair_label = PairScientificLabel.STOCHASTIC_NONCOMMUTATIVE
    elif smooth_eligible:
        representation = AlgebraRepresentation.DETERMINISTIC_SMOOTH
        if sequential is SequentialComposition.MATERIAL_NONCOMMUTATIVE:
            pair_label = PairScientificLabel.SMOOTH_NONCOMMUTATIVE_LIE_LOCAL
    elif method_input.hybrid_switching_observed and (
        sequential is SequentialComposition.MATERIAL_NONCOMMUTATIVE
    ):
        representation = AlgebraRepresentation.HYBRID_SWITCHING
        pair_label = PairScientificLabel.HYBRID_NONCOMMUTATIVE
    else:
        representation = AlgebraRepresentation.DETERMINISTIC_FINITE

    if sequential is SequentialComposition.MATERIAL_NONCOMMUTATIVE:
        if state_sufficiency in {
            StateSufficiency.HIDDEN_STATE_LIMITED,
            StateSufficiency.TIME_VARYING_DENOMINATOR,
            StateSufficiency.UNEVALUABLE,
        }:
            closure = AlgebraClosure.PROJECTED_OR_STATE_NONCLOSED
            if pair_label not in {
                PairScientificLabel.STOCHASTIC_NONCOMMUTATIVE,
                PairScientificLabel.HYBRID_NONCOMMUTATIVE,
            }:
                pair_label = PairScientificLabel.APPARENT_NONCOMMUTATIVITY_STATE_NOT_CLOSED
        else:
            closure = AlgebraClosure.CLOSED_IN_GENERATOR_SPAN
    elif sequential is SequentialComposition.UNEVALUABLE:
        closure = AlgebraClosure.UNEVALUABLE
    else:
        closure = AlgebraClosure.NOT_APPLICABLE

    port_materiality = (
        PortMateriality.MATERIAL
        if material_a is PortMateriality.MATERIAL and material_b is PortMateriality.MATERIAL
        else PortMateriality.NULL
        if material_a is PortMateriality.NULL and material_b is PortMateriality.NULL
        else PortMateriality.MIXED
    )
    reason_codes: tuple[str, ...] = ()
    if "UNEVALUABLE" in {
        temporal.value,
        simultaneous.value,
        sequential.value,
        closure.value,
        state_sufficiency.value,
    }:
        reason_codes = ("TEMPORAL_OR_COMPOSITION_UNEVALUABLE",)
    signature = ResponseAlgebraSignature(
        signature_id=f"signature.{method_input.case_token}",
        port_materiality=port_materiality,
        temporal_composition=temporal,
        simultaneous_composition=simultaneous,
        sequential_composition=sequential,
        algebra_closure=closure,
        state_sufficiency=state_sufficiency,
        representation=representation,
        receiver_visibility=receiver_visibility,
        action_quotient=(
            ActionQuotient.FAITHFUL_PORTS
            if port_materiality is PortMateriality.MATERIAL
            else ActionQuotient.PARTIALLY_COLLAPSED
        ),
        admission_annotation=AdmissionAnnotation.NOT_TESTED,
        metrics=(),
        evidence_link_ids=(),
        reason_codes=reason_codes,
    )
    temporal_metrics = tuple(
        NamedDecimal(value_id=metric_id, value=_decimal(value), unit="1")
        for metric_id, value in (
            ("temporal.affine-closure-rmse", primary_temporal.closure_error),
            ("temporal.stationarity-frobenius", primary_temporal.stationarity_error),
            ("temporal.wrong-horizon-rmse", primary_temporal.wrong_horizon_error),
        )
        if np.isfinite(value)
    )
    optional_metrics: list[NamedDecimal] = [
        NamedDecimal("generator.bracket-frobenius", _decimal(bracket_norm), "1"),
        NamedDecimal("generator.dose-scaling-relative-error", _decimal(dose_scaling_error), "1"),
    ]
    if stochastic_commutator is not None:
        optional_metrics.append(
            NamedDecimal(
                "stochastic.commutator-frobenius",
                _decimal(stochastic_commutator),
                "1",
            )
        )
    metrics = tuple(
        sorted(
            (*direct_metrics, *temporal_metrics, *optional_metrics),
            key=lambda value: value.value_id,
        )
    )
    return ResponseAlgebraMethodResult(
        result_id=f"method-result.{method_input.case_token}",
        case_token=method_input.case_token,
        method_key=config.method_key,
        method_version=config.method_version,
        signature=signature,
        pair_label=pair_label,
        smooth_lie_eligible=smooth_eligible,
        metrics=metrics,
        independent_unit_ids=unit_ids,
        reason_codes=reason_codes,
    )
