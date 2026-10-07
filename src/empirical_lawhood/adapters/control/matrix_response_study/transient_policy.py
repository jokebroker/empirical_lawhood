"""Low-capacity phase rules and comparator schedules for the Matrix transient response tranche."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from itertools import combinations, product
from typing import ClassVar, Mapping, cast

import numpy as np

from .scientific_inputs import MatrixResponsePolicyAllocationScientificInputs, require_policy_allocation_scientific_inputs

from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    validate_decimal,
    validate_sha256,
    validate_stable_id,
)


PHASE_FEATURE_NAMES = (
    "closure_y",
    "closure_y_velocity",
    "gap_y",
    "gap_y_velocity",
    "radius_y",
    "radius_y_velocity",
)
ACTIVE_ACTION_WORDS = ("x-negative", "x-positive", "y-negative", "y-positive")
POLICY_WORDS = (
    "hold",
    "open-loop-energy-matched",
    "phase-aware",
    "phase-shuffled-feedback",
    "radius-only-feedback",
    "stratified-random",
)


@dataclass(frozen=True, slots=True)
class MatrixResponseTransientControlledInvariancePhaseFeature(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/control/matrix-response-study/matrix-response-transient-controlled-invariance-phase-feature'

    feature_id: str
    parent_step: int
    radius_y: Decimal
    radius_y_velocity: Decimal
    closure_y: Decimal
    closure_y_velocity: Decimal
    gap_y: Decimal
    gap_y_velocity: Decimal
    radius_x: Decimal
    closure_x: Decimal
    kernel_x: Decimal
    cross_commutator_ratio: Decimal
    valid: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.feature_id, field_name="feature_id")
        if self.parent_step < 0:
            raise ValueError("Matrix transient response phase-feature step cannot be negative")
        for name in (
            "radius_y",
            "radius_y_velocity",
            "closure_y",
            "closure_y_velocity",
            "gap_y",
            "gap_y_velocity",
            "radius_x",
            "closure_x",
            "kernel_x",
            "cross_commutator_ratio",
        ):
            validate_decimal(getattr(self, name), field_name=name)

    def primary(self) -> Mapping[str, Decimal]:
        return {name: getattr(self, name) for name in PHASE_FEATURE_NAMES}


@dataclass(frozen=True, slots=True)
class MatrixResponseTransientControlledInvariancePhasePredicate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/control/matrix-response-study/matrix-response-transient-controlled-invariance-phase-predicate'

    predicate_id: str
    feature_name: str
    direction: str
    threshold: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.predicate_id, field_name="predicate_id")
        if self.feature_name not in PHASE_FEATURE_NAMES or self.direction not in {"ge", "le"}:
            raise ValueError("Matrix transient response phase predicate differs")
        validate_decimal(self.threshold, field_name="threshold")

    def accepts(self, feature: MatrixResponseTransientControlledInvariancePhaseFeature) -> bool:
        if not feature.valid:
            return False
        value = feature.primary()[self.feature_name]
        return value >= self.threshold if self.direction == "ge" else value <= self.threshold


@dataclass(frozen=True, slots=True)
class MatrixResponseTransientControlledInvariancePhaseRule(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/control/matrix-response-study/matrix-response-transient-controlled-invariance-phase-rule'

    rule_id: str
    action_word: str
    development_step: int
    development_effect_margin: Decimal
    predicates: tuple[MatrixResponseTransientControlledInvariancePhasePredicate, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.rule_id, field_name="rule_id")
        if self.action_word not in ACTIVE_ACTION_WORDS or self.development_step < 0:
            raise ValueError("Matrix transient response phase rule action/development differs")
        validate_decimal(self.development_effect_margin, field_name="development_effect_margin")
        require_sorted_unique_ids(
            self.predicates, attribute="predicate_id", field_name="predicates"
        )
        if not 1 <= len(self.predicates) <= 3:
            raise ValueError("Matrix transient response phase rule requires one to three predicates")
        feature_names = tuple(value.feature_name for value in self.predicates)
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("Matrix transient response phase rule repeats a feature")

    def accepts(self, feature: MatrixResponseTransientControlledInvariancePhaseFeature) -> bool:
        return all(value.accepts(feature) for value in self.predicates)

    def first_trigger(
        self,
        features: tuple[MatrixResponseTransientControlledInvariancePhaseFeature, ...],
        *,
        first_step: int = 816,
        last_step: int = 1024,
    ) -> int | None:
        eligible = tuple(
            value
            for value in features
            if first_step <= value.parent_step <= last_step and self.accepts(value)
        )
        return eligible[0].parent_step if eligible else None


def enumerate_phase_predicates(
    features: tuple[MatrixResponseTransientControlledInvariancePhaseFeature, ...],
) -> tuple[MatrixResponseTransientControlledInvariancePhasePredicate, ...]:
    if len(features) != 7 or tuple(value.parent_step for value in features) != (
        816,
        848,
        880,
        896,
        928,
        960,
        1024,
    ):
        raise ValueError("Matrix transient response development feature roster differs")
    outputs: list[MatrixResponseTransientControlledInvariancePhasePredicate] = []
    for feature_name in PHASE_FEATURE_NAMES:
        values = tuple(sorted(set(value.primary()[feature_name] for value in features)))
        thresholds = tuple((left + right) / Decimal(2) for left, right in zip(values, values[1:]))
        for direction, threshold in product(("ge", "le"), thresholds):
            threshold_id = sha256(str(threshold).encode()).hexdigest()[:12]
            outputs.append(
                MatrixResponseTransientControlledInvariancePhasePredicate(
                    predicate_id=(f"matrix-transient-response.predicate.{feature_name}.{direction}.{threshold_id}"),
                    feature_name=feature_name,
                    direction=direction,
                    threshold=threshold,
                )
            )
    return tuple(sorted(outputs, key=lambda value: value.predicate_id))


def enumerate_phase_rules(
    *,
    features: tuple[MatrixResponseTransientControlledInvariancePhaseFeature, ...],
    action_word: str,
    development_step: int,
    development_effect_margin: Decimal,
) -> tuple[MatrixResponseTransientControlledInvariancePhaseRule, ...]:
    if action_word not in ACTIVE_ACTION_WORDS:
        raise ValueError("Matrix transient response phase-rule action differs")
    predicates = enumerate_phase_predicates(features)
    outputs: list[MatrixResponseTransientControlledInvariancePhaseRule] = []
    for count in range(1, 4):
        for selected in combinations(predicates, count):
            if len({value.feature_name for value in selected}) != count:
                continue
            canonical = tuple(sorted(selected, key=lambda value: value.predicate_id))
            digest = sha256(
                "\0".join(value.predicate_id for value in canonical).encode()
            ).hexdigest()[:16]
            outputs.append(
                MatrixResponseTransientControlledInvariancePhaseRule(
                    rule_id=f"matrix-transient-response.rule.{action_word}.{development_step}.{digest}",
                    action_word=action_word,
                    development_step=development_step,
                    development_effect_margin=development_effect_margin,
                    predicates=canonical,
                )
            )
    return tuple(sorted(outputs, key=lambda value: value.rule_id))


@dataclass(frozen=True, slots=True)
class MatrixResponseTransientControlledInvarianceFrozenInterventionRule(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/control/matrix-response-study/matrix-response-transient-controlled-invariance-frozen-intervention-rule'

    frozen_rule_id: str
    config_fingerprint: str
    phase_rule: MatrixResponseTransientControlledInvariancePhaseRule
    candidate_sha256: str
    confirmation_terminal_sha256: str
    pulse_delta: Decimal
    pulse_ramp_steps: int
    pulse_dwell_steps: int
    pulse_return_steps: int
    trigger_first_step: int
    trigger_last_step: int
    radius_only_direction: str
    radius_only_threshold: Decimal
    development_median_trigger_step: int
    grants_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.frozen_rule_id, field_name="frozen_rule_id")
        for name in ("config_fingerprint", "candidate_sha256", "confirmation_terminal_sha256"):
            validate_sha256(getattr(self, name), field_name=name)
        validate_decimal(self.pulse_delta, field_name="pulse_delta", minimum=Decimal(0))
        validate_decimal(self.radius_only_threshold, field_name="radius_only_threshold")
        if (
            self.pulse_delta != Decimal("0.125")
            or (self.pulse_ramp_steps, self.pulse_dwell_steps, self.pulse_return_steps)
            != (48, 32, 48)
            or (self.trigger_first_step, self.trigger_last_step) != (816, 1024)
            or self.radius_only_direction not in {"ge", "le"}
            or not 816 <= self.development_median_trigger_step <= 1024
            or self.grants_authority
        ):
            raise ValueError("Matrix transient response frozen intervention contract differs")

    def phase_trigger(self, features: tuple[MatrixResponseTransientControlledInvariancePhaseFeature, ...]) -> int | None:
        return self.phase_rule.first_trigger(
            features,
            first_step=self.trigger_first_step,
            last_step=self.trigger_last_step,
        )

    def radius_trigger(self, features: tuple[MatrixResponseTransientControlledInvariancePhaseFeature, ...]) -> int | None:
        for feature in features:
            if not self.trigger_first_step <= feature.parent_step <= self.trigger_last_step:
                continue
            accepted = (
                feature.radius_y >= self.radius_only_threshold
                if self.radius_only_direction == "ge"
                else feature.radius_y <= self.radius_only_threshold
            )
            if feature.valid and accepted:
                return feature.parent_step
        return None


@dataclass(frozen=True, slots=True)
class MatrixResponseTransientControlledInvariancePolicyDecision(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/control/matrix-response-study/matrix-response-transient-controlled-invariance-policy-decision'

    decision_id: str
    policy_word: str
    block_index: int
    action_word: str
    trigger_parent_step: int | None
    source_block_index: int | None

    def __post_init__(self) -> None:
        validate_stable_id(self.decision_id, field_name="decision_id")
        if (
            self.policy_word not in POLICY_WORDS
            or self.block_index < 0
            or self.action_word not in (*ACTIVE_ACTION_WORDS, "hold")
        ):
            raise ValueError("Matrix transient response policy decision identity differs")
        active = self.action_word != "hold"
        if active != (self.trigger_parent_step is not None):
            raise ValueError("Matrix transient response policy action/trigger differs")


def _decision(
    *,
    policy: str,
    block_index: int,
    action_word: str,
    trigger: int | None,
    source_block_index: int | None = None,
) -> MatrixResponseTransientControlledInvariancePolicyDecision:
    return MatrixResponseTransientControlledInvariancePolicyDecision(
        decision_id=f"matrix-transient-response.decision.{policy}.block-{block_index:03d}",
        policy_word=policy,
        block_index=block_index,
        action_word=action_word if trigger is not None else "hold",
        trigger_parent_step=trigger,
        source_block_index=source_block_index,
    )


def build_policy_decisions(
    *,
    rule: MatrixResponseTransientControlledInvarianceFrozenInterventionRule,
    feature_series: tuple[tuple[MatrixResponseTransientControlledInvariancePhaseFeature, ...], ...],
    scientific_inputs: MatrixResponsePolicyAllocationScientificInputs,
) -> Mapping[str, tuple[MatrixResponseTransientControlledInvariancePolicyDecision, ...]]:
    """Freeze all six C schedules from HOLD prefixes before controlled outcomes."""

    if len(feature_series) != 128:
        raise ValueError("Matrix transient response policy schedule requires 128 feature prefixes")
    require_policy_allocation_scientific_inputs(scientific_inputs, rule=rule, feature_series=feature_series)
    block_count = len(feature_series)
    action = rule.phase_rule.action_word
    phase_triggers = tuple(rule.phase_trigger(values) for values in feature_series)
    active_count = sum(value is not None for value in phase_triggers)
    outputs: dict[str, tuple[MatrixResponseTransientControlledInvariancePolicyDecision, ...]] = {}
    outputs["hold"] = tuple(
        _decision(policy="hold", block_index=index, action_word="hold", trigger=None)
        for index in range(block_count)
    )
    outputs["phase-aware"] = tuple(
        _decision(
            policy="phase-aware",
            block_index=index,
            action_word=action,
            trigger=trigger,
            source_block_index=index,
        )
        for index, trigger in enumerate(phase_triggers)
    )
    digest = bytes.fromhex(scientific_inputs.open_loop_seed_sha256)
    rng = np.random.Generator(np.random.PCG64DXSM(int.from_bytes(digest[:16], "big")))
    permutation = tuple(int(value) for value in rng.permutation(block_count))
    open_active = set(permutation[:active_count])
    outputs["open-loop-energy-matched"] = tuple(
        _decision(
            policy="open-loop-energy-matched",
            block_index=index,
            action_word=action,
            trigger=(rule.development_median_trigger_step if index in open_active else None),
        )
        for index in range(block_count)
    )
    outputs["phase-shuffled-feedback"] = tuple(
        _decision(
            policy="phase-shuffled-feedback",
            block_index=index,
            action_word=action,
            trigger=phase_triggers[(index + 1) % block_count],
            source_block_index=(index + 1) % block_count,
        )
        for index in range(block_count)
    )
    outputs["radius-only-feedback"] = tuple(
        _decision(
            policy="radius-only-feedback",
            block_index=index,
            action_word=action,
            trigger=rule.radius_trigger(values),
            source_block_index=index,
        )
        for index, values in enumerate(feature_series)
    )
    median_step = rule.development_median_trigger_step
    amplitude = np.asarray(
        [
            float(next(value.radius_y for value in values if value.parent_step == median_step))
            for values in feature_series
        ],
        dtype=np.float64,
    )
    boundaries = np.quantile(amplitude, (0.25, 0.5, 0.75), method="linear")
    strata = np.searchsorted(boundaries, amplitude, side="right")
    random_triggers: list[int | None] = [None] * block_count
    for stratum in range(4):
        members = np.flatnonzero(strata == stratum)
        source_times = [
            cast(int, phase_triggers[index])
            for index in members
            if phase_triggers[index] is not None
        ]
        stratum_digest = bytes.fromhex(scientific_inputs.stratum_seed_sha256s[stratum])
        stratum_rng = np.random.Generator(
            np.random.PCG64DXSM(int.from_bytes(stratum_digest[:16], "big"))
        )
        targets = tuple(int(value) for value in stratum_rng.permutation(members))[
            : len(source_times)
        ]
        times = tuple(int(value) for value in stratum_rng.permutation(source_times))
        for target, trigger in zip(targets, times, strict=True):
            random_triggers[target] = trigger
    outputs["stratified-random"] = tuple(
        _decision(
            policy="stratified-random",
            block_index=index,
            action_word=action,
            trigger=trigger,
        )
        for index, trigger in enumerate(random_triggers)
    )
    if set(outputs) != set(POLICY_WORDS):
        raise AssertionError("Matrix transient response policy family is incomplete")
    for policy in ("open-loop-energy-matched", "phase-shuffled-feedback", "stratified-random"):
        if sum(value.trigger_parent_step is not None for value in outputs[policy]) != active_count:
            raise AssertionError("Matrix transient response comparator pulse count differs")
    return outputs


__all__ = [
    "ACTIVE_ACTION_WORDS",
    'MatrixResponseTransientControlledInvarianceFrozenInterventionRule',
    'MatrixResponseTransientControlledInvariancePhaseFeature',
    'MatrixResponseTransientControlledInvariancePhasePredicate',
    'MatrixResponseTransientControlledInvariancePhaseRule',
    'MatrixResponseTransientControlledInvariancePolicyDecision',
    "PHASE_FEATURE_NAMES",
    "POLICY_WORDS",
    'build_policy_decisions',
    'enumerate_phase_predicates',
    'enumerate_phase_rules',
]
