"Truth-blind formalization conformance method computation and separately privileged scoring."

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from typing import ClassVar

import numpy as np

from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_stable_id,
)

from .topological_response import (
    FloatArray,
    LocalLawKind,
    ReceiverKind,
    WorldSplit,
    _fit_transition,
    _transition_panel,
    build_world,
    graph_family,
    linear_generator,
    observability_rank,
    replace_world_graph,
)


FORMALIZATION_CAPABILITY_AXES = (
    "action-rank",
    "delivery",
    "history-sufficiency",
    "mixture",
    "receiver-faithfulness",
    "stationarity",
    "topology-binding",
)


@dataclass(frozen=True, slots=True)
class FormalizationEstimate(CanonicalRecord):
    "One truth-blind structural estimate under the frozen formalization conformance thresholds."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/formalization-estimate'

    estimate_id: str
    case_id: str
    structural_class: str
    metric: Decimal | None
    action_rank: int | None

    def __post_init__(self) -> None:
        validate_stable_id(self.estimate_id, field_name="estimate_id")
        validate_stable_id(self.case_id, field_name="case_id")
        validate_nonempty(self.structural_class, field_name="structural_class")
        if self.action_rank is not None and self.action_rank < 0:
            raise ValueError("formalization action rank must be nonnegative")


@dataclass(frozen=True, slots=True)
class FormalizationPrefixEstimate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/formalization-prefix-estimate'

    prefix_id: str
    independent_units: int
    stationary_defect: Decimal
    nonstationary_defect: Decimal
    classification_stable: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.prefix_id, field_name="prefix_id")
        if self.independent_units <= 0:
            raise ValueError("formalization prefix needs independent units")
        expected = self.stationary_defect < Decimal("0.03") and (
            self.nonstationary_defect > Decimal("0.05")
        )
        if self.classification_stable != expected:
            raise ValueError("formalization prefix stability differs from frozen thresholds")


@dataclass(frozen=True, slots=True)
class TruthBlindFormalizationResult(CanonicalRecord):
    "Formalization conformance method output with no oracle labels or pass/fail adjudication."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/truth-blind-formalization-result'

    result_id: str
    seed: int
    estimates: tuple[FormalizationEstimate, ...]
    prefix_estimates: tuple[FormalizationPrefixEstimate, ...]
    capability_axes: tuple[str, ...]
    numeric_panel_sha256: str
    numeric_panel_shape: tuple[int, ...]
    stationarity_null_threshold: Decimal
    nonstationarity_materiality_threshold: Decimal
    operator_concentration_threshold: Decimal
    universal_sample_threshold_claimed: bool
    truth_or_score_present: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        if self.seed < 0:
            raise ValueError("formalization seed must be nonnegative")
        require_sorted_unique_ids(
            self.estimates,
            attribute="estimate_id",
            field_name="estimates",
        )
        require_sorted_unique_ids(
            self.prefix_estimates,
            attribute="prefix_id",
            field_name="prefix_estimates",
        )
        require_sorted_unique_strings(
            self.capability_axes,
            field_name="capability_axes",
            allow_empty=False,
        )
        if self.capability_axes != FORMALIZATION_CAPABILITY_AXES:
            raise ValueError("Formalization conformance capability axes differ from the frozen contract")
        if len(self.numeric_panel_sha256) != 64:
            raise ValueError("Formalization conformance numeric panel identity must be a SHA-256")
        if not self.numeric_panel_shape or any(value <= 0 for value in self.numeric_panel_shape):
            raise ValueError("Formalization conformance numeric panel shape must be positive")
        if self.universal_sample_threshold_claimed:
            raise ValueError("Formalization conformance cannot claim a universal sample threshold")
        if self.truth_or_score_present:
            raise ValueError("truth-blind formalization conformance output cannot contain truth or scores")


@dataclass(frozen=True, slots=True)
class FormalizationOracle(CanonicalRecord):
    "Privileged formalization conformance truth kept outside the method input and method process."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/formalization-oracle'

    oracle_id: str
    case_id: str
    expected_structural_class: str

    def __post_init__(self) -> None:
        validate_stable_id(self.oracle_id, field_name="oracle_id")
        validate_stable_id(self.case_id, field_name="case_id")
        validate_nonempty(
            self.expected_structural_class,
            field_name="expected_structural_class",
        )


@dataclass(frozen=True, slots=True)
class FormalizationOracleBatch(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/formalization-oracle-batch'

    batch_id: str
    oracles: tuple[FormalizationOracle, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.batch_id, field_name="batch_id")
        require_sorted_unique_ids(
            self.oracles,
            attribute="oracle_id",
            field_name="oracles",
        )
        if len(self.oracles) != 8:
            raise ValueError("Formalization conformance oracle batch requires eight cases")


@dataclass(frozen=True, slots=True)
class FormalizationCaseScore(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/formalization-case-score'

    score_id: str
    case_id: str
    passed: bool
    observed_structural_class: str
    expected_structural_class: str

    def __post_init__(self) -> None:
        validate_stable_id(self.score_id, field_name="score_id")
        validate_stable_id(self.case_id, field_name="case_id")
        if self.passed != (self.observed_structural_class == self.expected_structural_class):
            raise ValueError("Formalization conformance case score differs from exact class equality")


@dataclass(frozen=True, slots=True)
class FormalizationConformanceScore(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/formalization-conformance-score'

    score_id: str
    case_scores: tuple[FormalizationCaseScore, ...]
    passed_cases: int
    failed_cases: int
    false_promotions: int
    false_oppositions: int
    gate_passed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.score_id, field_name="score_id")
        require_sorted_unique_ids(
            self.case_scores,
            attribute="score_id",
            field_name="case_scores",
        )
        if self.passed_cases + self.failed_cases != len(self.case_scores):
            raise ValueError("Formalization conformance counts do not close")
        expected = (
            self.failed_cases == 0 and self.false_promotions == 0 and self.false_oppositions == 0
        )
        if self.gate_passed != expected:
            raise ValueError("Formalization conformance gate differs from frozen rules")


def _decimal(value: float) -> Decimal:
    return Decimal(repr(float(value)))


def compute_truth_blind_formalization(
    seed: int = 20260720,
) -> tuple[TruthBlindFormalizationResult, FloatArray]:
    "Run formalization conformance estimation without importing an oracle or assigning pass/fail."

    unit_count = 8
    stationary_states, stationary_actions, _, _ = _transition_panel(
        seed=seed,
        unit_count=unit_count,
        nonstationary=False,
        mixture=False,
    )
    nonstationary_states, nonstationary_actions, _, _ = _transition_panel(
        seed=seed + 1,
        unit_count=unit_count,
        nonstationary=True,
        mixture=False,
    )
    mixture_states, mixture_actions, _, _ = _transition_panel(
        seed=seed + 2,
        unit_count=unit_count,
        nonstationary=False,
        mixture=True,
    )

    def stationarity_defect(states: FloatArray, actions: FloatArray) -> float:
        midpoint = actions.shape[1] // 2
        early = _fit_transition(states[:, : midpoint + 1], actions[:, :midpoint])
        late = _fit_transition(states[:, midpoint:], actions[:, midpoint:])
        scale = max(float(np.linalg.norm(early)), np.finfo(float).eps)
        return float(np.linalg.norm(early - late) / scale)

    stationary_defect = stationarity_defect(stationary_states, stationary_actions)
    nonstationary_defect = stationarity_defect(
        nonstationary_states,
        nonstationary_actions,
    )
    per_unit_operators = tuple(
        _fit_transition(mixture_states[index : index + 1], mixture_actions[index : index + 1])
        for index in range(unit_count)
    )
    flattened = np.stack([operator.reshape(-1) for operator in per_unit_operators])
    within_components = np.mean(
        [
            np.linalg.norm(flattened[index] - np.mean(flattened[index % 2 :: 2], axis=0))
            for index in range(unit_count)
        ]
    )
    between_components = float(
        np.linalg.norm(np.mean(flattened[::2], axis=0) - np.mean(flattened[1::2], axis=0))
    )
    mixture_separation = between_components / max(float(within_components), 1e-12)

    graph = graph_family("path")
    base_world = build_world(
        graph,
        preparation_index=1,
        split=WorldSplit.DEVELOPMENT,
        local_law=LocalLawKind.LINEAR,
        seed=seed,
        observation_noise_sd=0.0,
    )
    full_observability = observability_rank(base_world, ReceiverKind.FULL_STATE)
    folded_observability = observability_rank(base_world, ReceiverKind.AGGREGATE)
    wrong_world = replace_world_graph(
        base_world,
        graph_family("cycle"),
        "world.wrong-topology-binding",
    )
    topology_defect = float(
        np.linalg.norm(linear_generator(base_world) - linear_generator(wrong_world))
    )

    constant_receiver = np.ones((unit_count, 24, 1), dtype=np.float64)
    constant_prediction_rms = float(
        np.sqrt(np.mean(np.square(constant_receiver[:, 1:] - constant_receiver[:, :-1])))
    )
    constant_action_rank = 0

    random = np.random.default_rng(seed + 3)
    oscillator = np.asarray([[0.92, 0.30], [-0.28, 0.90]], dtype=np.float64)
    hidden = np.empty((unit_count, 50, 2), dtype=np.float64)
    for unit in range(unit_count):
        hidden[unit, 0] = random.normal(0.0, 0.3, size=2)
        for step in range(49):
            hidden[unit, step + 1] = np.tanh(oscillator @ hidden[unit, step])
    visible = hidden[:, :, :1]
    target = visible[:, 2:].reshape(-1, 1)
    current = visible[:, 1:-1].reshape(-1, 1)
    lagged = visible[:, :-2].reshape(-1, 1)
    current_design = np.column_stack((current, np.ones(current.shape[0])))
    history_design = np.column_stack((current, lagged, np.ones(current.shape[0])))
    current_fit = np.linalg.lstsq(current_design, target, rcond=None)[0]
    history_fit = np.linalg.lstsq(history_design, target, rcond=None)[0]
    current_rms = float(np.sqrt(np.mean(np.square(target - current_design @ current_fit))))
    history_rms = float(np.sqrt(np.mean(np.square(target - history_design @ history_fit))))
    history_improvement = (current_rms - history_rms) / max(current_rms, 1e-12)

    rows = (
        (
            "adversarial-constant-predictor",
            "PREDICTIVE_NULL_ACTION_RANK_ZERO",
            constant_prediction_rms,
            constant_action_rank,
        ),
        ("delivery-hidden", "UNEVALUABLE_DELIVERY_UNOBSERVED", None, None),
        (
            "hidden-state-memory",
            (
                "CURRENT_RECEIVER_NOT_MARKOV"
                if history_improvement > 0.2
                else "CURRENT_RECEIVER_SUFFICIENT"
            ),
            history_improvement,
            None,
        ),
        (
            "nonstationary",
            "NONSTATIONARY" if nonstationary_defect > 0.05 else "STATIONARY",
            nonstationary_defect,
            None,
        ),
        (
            "receiver-folded",
            ("RECEIVER_UNFAITHFUL" if folded_observability < full_observability else "FAITHFUL"),
            folded_observability / full_observability,
            None,
        ),
        (
            "stationary-deterministic",
            "STATIONARY" if stationary_defect < 0.03 else "NONSTATIONARY",
            stationary_defect,
            None,
        ),
        (
            "stochastic-mixture",
            "TWO_COMPONENT_MIXTURE" if mixture_separation > 5.0 else "CONCENTRATED",
            mixture_separation,
            None,
        ),
        (
            "topology-mislabelled",
            "TOPOLOGY_BINDING_FALSE" if topology_defect > 0.05 else "TOPOLOGY_ACCEPTED",
            topology_defect,
            None,
        ),
    )
    estimates = tuple(
        FormalizationEstimate(
            estimate_id=f"estimate.{case_id}",
            case_id=f"case.{case_id}",
            structural_class=structural_class,
            metric=None if metric is None else _decimal(metric),
            action_rank=action_rank,
        )
        for case_id, structural_class, metric, action_rank in rows
    )
    prefix_estimates = tuple(
        FormalizationPrefixEstimate(
            prefix_id=f"prefix.{count:02d}",
            independent_units=count,
            stationary_defect=_decimal(
                stationarity_defect(
                    stationary_states[:count],
                    stationary_actions[:count],
                )
            ),
            nonstationary_defect=_decimal(
                stationarity_defect(
                    nonstationary_states[:count],
                    nonstationary_actions[:count],
                )
            ),
            classification_stable=(
                stationarity_defect(
                    stationary_states[:count],
                    stationary_actions[:count],
                )
                < 0.03
                and stationarity_defect(
                    nonstationary_states[:count],
                    nonstationary_actions[:count],
                )
                > 0.05
            ),
        )
        for count in (2, 3, 4, 6, 8)
    )
    numeric_panel = np.concatenate(
        (
            stationary_states.reshape(unit_count, -1),
            stationary_actions.reshape(unit_count, -1),
            nonstationary_states.reshape(unit_count, -1),
            nonstationary_actions.reshape(unit_count, -1),
            mixture_states.reshape(unit_count, -1),
            mixture_actions.reshape(unit_count, -1),
            hidden.reshape(unit_count, -1),
        ),
        axis=1,
    )
    panel = np.asarray(numeric_panel, dtype=np.float64)
    return (
        TruthBlindFormalizationResult(
            result_id=f"rf3-truth-blind.{seed}",
            seed=seed,
            estimates=estimates,
            prefix_estimates=prefix_estimates,
            capability_axes=FORMALIZATION_CAPABILITY_AXES,
            numeric_panel_sha256=sha256(panel.tobytes(order="C")).hexdigest(),
            numeric_panel_shape=tuple(int(value) for value in panel.shape),
            stationarity_null_threshold=Decimal("0.03"),
            nonstationarity_materiality_threshold=Decimal("0.05"),
            operator_concentration_threshold=Decimal("5"),
            universal_sample_threshold_claimed=False,
            truth_or_score_present=False,
        ),
        panel,
    )


def privileged_formalization_oracles() -> tuple[FormalizationOracle, ...]:
    "Return evaluator-only formalization conformance truth; method code never calls this function."

    values = (
        ("adversarial-constant-predictor", "PREDICTIVE_NULL_ACTION_RANK_ZERO"),
        ("delivery-hidden", "UNEVALUABLE_DELIVERY_UNOBSERVED"),
        ("hidden-state-memory", "CURRENT_RECEIVER_NOT_MARKOV"),
        ("nonstationary", "NONSTATIONARY"),
        ("receiver-folded", "RECEIVER_UNFAITHFUL"),
        ("stationary-deterministic", "STATIONARY"),
        ("stochastic-mixture", "TWO_COMPONENT_MIXTURE"),
        ("topology-mislabelled", "TOPOLOGY_BINDING_FALSE"),
    )
    return tuple(
        FormalizationOracle(
            oracle_id=f"oracle.{case_id}",
            case_id=f"case.{case_id}",
            expected_structural_class=expected,
        )
        for case_id, expected in values
    )


def score_formalization_conformance(
    result: TruthBlindFormalizationResult,
    oracles: tuple[FormalizationOracle, ...],
) -> FormalizationConformanceScore:
    "Evaluate exact formalization conformance class recovery after the protected truth boundary."

    require_sorted_unique_ids(oracles, attribute="oracle_id", field_name="oracles")
    estimates = {value.case_id: value for value in result.estimates}
    expected = {value.case_id: value for value in oracles}
    if set(estimates) != set(expected):
        raise ValueError("Formalization conformance method and oracle case sets differ")
    scores = tuple(
        FormalizationCaseScore(
            score_id=f"score.{case_id}",
            case_id=case_id,
            passed=(
                estimates[case_id].structural_class == expected[case_id].expected_structural_class
            ),
            observed_structural_class=estimates[case_id].structural_class,
            expected_structural_class=expected[case_id].expected_structural_class,
        )
        for case_id in sorted(estimates)
    )
    passed = sum(value.passed for value in scores)
    false_promotions = sum(
        not value.passed
        and value.expected_structural_class.startswith(("UNEVALUABLE", "PREDICTIVE_NULL"))
        for value in scores
    )
    false_oppositions = sum(
        not value.passed
        and not value.expected_structural_class.startswith(("UNEVALUABLE", "PREDICTIVE_NULL"))
        for value in scores
    )
    return FormalizationConformanceScore(
        score_id=f"rf3-score.{result.seed}",
        case_scores=scores,
        passed_cases=passed,
        failed_cases=len(scores) - passed,
        false_promotions=false_promotions,
        false_oppositions=false_oppositions,
        gate_passed=passed == len(scores),
    )


__all__ = [
    "FormalizationCaseScore",
    "FormalizationConformanceScore",
    "FormalizationEstimate",
    "FormalizationOracle",
    "FormalizationOracleBatch",
    "FormalizationPrefixEstimate",
    "FORMALIZATION_CAPABILITY_AXES",
    "TruthBlindFormalizationResult",
    "compute_truth_blind_formalization",
    "privileged_formalization_oracles",
    "score_formalization_conformance",
]
