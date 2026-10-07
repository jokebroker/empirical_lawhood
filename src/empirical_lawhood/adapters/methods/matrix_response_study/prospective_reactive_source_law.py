"""Prospective Six-matrix response source features, frozen score, direct outcomes, and terminal.

Development fits one low-capacity past-only rule.  Confirmation only applies
that frozen rule and reduces wholly fresh complete histories.  Branches and
same-driver numerical views never contribute independent units.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
from typing import ClassVar, Sequence

import numpy as np
from scipy.optimize import minimize

from empirical_lawhood.adapters.composition.matrix_response_study.causal_intersection_residence_design import MatrixResponseCausalIntersectionResidenceStudyConfig
from empirical_lawhood.adapters.simulators.six_matrix_response.reactive_source_scientific_inputs import MatrixReactiveSourceAnalysisScientificInput
from empirical_lawhood.adapters.simulators.six_matrix_response.controlled_branch import SixMatrixResponseTransientControlledInvarianceActionLedger, SixMatrixResponseTransientControlledInvarianceBranchTrace, build_action_schedule
from empirical_lawhood.adapters.simulators.six_matrix_response.model import SixMatrixState
from empirical_lawhood.adapters.simulators.six_matrix_response.passive_probe import derive_probe_roster
from empirical_lawhood.adapters.simulators.six_matrix_response.prospective_reactive_source import SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_CANDIDATE_STEPS, SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_PROSPECTIVE_CONTROL_FUTURES, SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_PROSPECTIVE_SOURCE_FUTURES, SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_FAMILIES, SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_FINE_COUNT, SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_FUTURE_STEPS, SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HISTORY_COUNT, SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HOLD_STEPS, SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_PREPARATION_STEPS, SixMatrixResponseProspectiveReactiveSourceLawArm, SixMatrixResponseProspectiveReactiveSourceLawBranchSpec, SixMatrixResponseProspectiveReactiveSourceLawHistorySlot, SixMatrixResponseProspectiveReactiveSourceLawNumericalView, SixMatrixResponseProspectiveReactiveSourceLawPersistedHistory, SixMatrixResponseProspectiveReactiveSourceLawRouting, SixMatrixResponseProspectiveReactiveSourceLawSourceConfig, SixMatrixResponseProspectiveReactiveSourceLawTranche, development_routing
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    canonical_json_bytes,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)

from .full_intersection_causal_authority import MatrixResponseCausalIntersectionResidenceResponseFactorSample, _first_nonkernel_gap, evaluate_full_intersection_branch, extract_online_phase_features
from .reactive_entrance_source import ENTRY_CADENCE_STEPS, ENTRY_END_STEP, ENTRY_START_STEP
from .shooting_committor import observe_state, rolling_labels


PROSPECTIVE_REACTIVE_SOURCE_LAW_FEATURE_NAMES = (
    "radius_y", "radius_y_velocity", "closure_y", "closure_y_velocity",
    "gap_y", "gap_y_velocity", "radius_x", "closure_x", "kernel_x",
    "cross_commutator_ratio", "radius_closure_pass_count_16",
    "strict_kernel_pass_count_16", "closure_y_min_16", "closure_y_mean_16",
    "gap_y_min_16", "closure_y_sign_changes_16",
)
_MODEL_FEATURES: dict[str, tuple[int, ...]] = {
    "FULL_HISTORY": tuple(range(16)),
    "CURRENT_STATE": (0, 1, 2, 4, 6, 7, 8, 9),
    "NO_MOMENTUM": tuple(index for index in range(16) if index != 1),
    "SHALLOW_HISTORY": tuple(range(16)),
    "PHASE_TIME_NULL": (0,),
}


class MatrixResponseProspectiveReactiveSourceLawProjectionTerminal(StrEnum):
    COMPLETE = "COMPLETE"
    CUSTODY_INVALID = "CUSTODY_INVALID"
    NUMERICAL_INVALID = "NUMERICAL_INVALID"


class MatrixResponseProspectiveReactiveSourceLawTerminal(StrEnum):
    DEVELOPMENT_SCORE_FROZEN = "DEVELOPMENT_SCORE_FROZEN"
    NO_DEVELOPMENT_SOURCE_RULE = "NO_DEVELOPMENT_SOURCE_RULE"
    PREREQUISITE_NONATTEMPT = "PREREQUISITE_NONATTEMPT"
    CUSTODY_INVALID = "CUSTODY_INVALID"
    NUMERICAL_INVALID = "NUMERICAL_INVALID"
    SOURCE_EVALUATION_UNEVALUABLE = "SOURCE_EVALUATION_UNEVALUABLE"
    SUPPORTED = "PROSPECTIVE_REACTIVE_SOURCE_SUPPORTED"
    CONDITIONAL = "CONDITIONAL_SOURCE_ONLY"
    NO_SOURCE = "NO_PROSPECTIVE_SOURCE"


def _decimal(value: float) -> Decimal:
    if not np.isfinite(value):
        raise FloatingPointError("matrix response prospective reactive source law method value is nonfinite")
    return Decimal(repr(float(value)))


@dataclass(frozen=True, slots=True)
class MatrixResponseProspectiveReactiveSourceLawFeature(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-prospective-reactive-source-law-feature'

    feature_id: str
    checkpoint_step: int
    values: tuple[Decimal, ...]
    shallow_values: tuple[Decimal, ...]
    eligible: bool
    full_intersection_at_cutoff: bool
    robust_00_at_cutoff: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.feature_id, field_name="feature_id")
        if self.checkpoint_step not in SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_CANDIDATE_STEPS:
            raise ValueError("matrix response prospective reactive source law feature checkpoint differs")
        if len(self.values) != 16 or len(self.shallow_values) != 16:
            raise ValueError("matrix response prospective reactive source law feature dimension differs")
        for index, value in enumerate((*self.values, *self.shallow_values)):
            validate_decimal(value, field_name=f"feature[{index}]")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        expected = not self.reason_codes and not self.full_intersection_at_cutoff and not self.robust_00_at_cutoff
        if self.eligible != expected:
            raise ValueError("matrix response prospective reactive source law causal eligibility differs")


@dataclass(frozen=True, slots=True)
class MatrixResponseProspectiveReactiveSourceLawFrozenModel(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-prospective-reactive-source-law-frozen-model'

    model_id: str
    model_kind: str
    feature_indices: tuple[int, ...]
    means: tuple[Decimal, ...]
    scales: tuple[Decimal, ...]
    coefficients: tuple[Decimal, ...]
    intercept: Decimal
    lambda_l2: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.model_id, field_name="model_id")
        if self.model_kind not in _MODEL_FEATURES or self.feature_indices != _MODEL_FEATURES[self.model_kind]:
            raise ValueError("matrix response prospective reactive source law model feature roster differs")
        if not (len(self.means) == len(self.scales) == len(self.coefficients) == len(self.feature_indices)):
            raise ValueError("matrix response prospective reactive source law model geometry differs")
        for name, values in (("means", self.means), ("scales", self.scales), ("coefficients", self.coefficients)):
            for index, value in enumerate(values):
                validate_decimal(value, field_name=f"{name}[{index}]")
        validate_decimal(self.intercept, field_name="intercept")
        if self.lambda_l2 != Decimal("1") or any(value <= 0 for value in self.scales):
            raise ValueError("matrix response prospective reactive source law model regularization/scaling differs")


@dataclass(frozen=True, slots=True)
class MatrixResponseProspectiveReactiveSourceLawFrozenScore(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-prospective-reactive-source-law-frozen-score'

    score_id: str
    scientific_fingerprint: str
    development_source_config: ObjectIdentity
    fit_panel_sha256: str
    models: tuple[MatrixResponseProspectiveReactiveSourceLawFrozenModel, ...]
    threshold: Decimal
    threshold_grid: tuple[Decimal, ...]
    fold_rule_id: str
    checkpoint_grid: tuple[int, ...]
    grants_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.score_id, field_name="score_id")
        validate_stable_id(self.fold_rule_id, field_name="fold_rule_id")
        validate_sha256(self.scientific_fingerprint, field_name="scientific_fingerprint")
        validate_sha256(self.fit_panel_sha256, field_name="fit_panel_sha256")
        require_sorted_unique_ids(self.models, attribute="model_id", field_name="models")
        if (
            tuple(value.model_kind for value in self.models) != tuple(sorted(_MODEL_FEATURES))
            or self.threshold_grid != tuple(Decimal(value) / Decimal(100) for value in range(5, 100, 5))
            or self.threshold not in self.threshold_grid
            or self.checkpoint_grid != SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_CANDIDATE_STEPS
            or self.fold_rule_id != "matrix-response-prospective-reactive-source-law.eight-fold-history-grouped-family-stratified"
            or self.grants_authority
        ):
            raise ValueError("matrix response prospective reactive source law frozen score differs")

    def model(self, kind: str) -> MatrixResponseProspectiveReactiveSourceLawFrozenModel:
        return next(value for value in self.models if value.model_kind == kind)


@dataclass(frozen=True, slots=True)
class MatrixResponseProspectiveReactiveSourceLawMethodConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-prospective-reactive-source-law-method-config'

    config_id: str
    config_version: str
    scientific_fingerprint: str
    source_config: ObjectIdentity
    response_factor_config: MatrixResponseCausalIntersectionResidenceStudyConfig
    tranche: SixMatrixResponseProspectiveReactiveSourceLawTranche
    analysis_scientific_input: MatrixReactiveSourceAnalysisScientificInput
    frozen_score: MatrixResponseProspectiveReactiveSourceLawFrozenScore | None
    projection_task_prefix: str
    aggregate_task_id: str
    bootstrap_replicates: int
    grants_authority: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        for name in ("config_id", "projection_task_prefix", "aggregate_task_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_semantic_version(self.config_version)
        validate_sha256(self.scientific_fingerprint, field_name="scientific_fingerprint")
        if not isinstance(self.analysis_scientific_input, MatrixReactiveSourceAnalysisScientificInput):
            raise ValueError("reactive-source method requires an explicit analysis scientific input")
        is_prospective_evaluation = self.tranche is SixMatrixResponseProspectiveReactiveSourceLawTranche.PROSPECTIVE_EVALUATION
        if (
            (self.frozen_score is None) == is_prospective_evaluation
            or self.bootstrap_replicates != 10_000
            or self.response_factor_config.q != 2
            or self.grants_authority
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("matrix response prospective reactive source law method config differs")


@dataclass(frozen=True, slots=True)
class MatrixResponseProspectiveReactiveSourceLawBranchOutcome(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-prospective-reactive-source-law-branch-outcome'

    outcome_id: str
    branch: SixMatrixResponseProspectiveReactiveSourceLawBranchSpec
    entrance_hit: bool
    residence_16_steps: bool
    residence_32_steps: bool
    residence_128_steps: bool
    first_entry_step: int | None
    robust_00_step: int | None
    technically_valid: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.outcome_id, field_name="outcome_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.entrance_hit != (self.first_entry_step is not None) or self.residence_128_steps and not self.residence_32_steps or self.residence_32_steps and not self.residence_16_steps or self.residence_16_steps and not self.entrance_hit:
            raise ValueError("matrix response prospective reactive source law outcome nesting differs")
        if self.technically_valid != (not self.reason_codes):
            raise ValueError("matrix response prospective reactive source law outcome validity differs")


@dataclass(frozen=True, slots=True)
class MatrixResponseProspectiveReactiveSourceLawProjection(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-prospective-reactive-source-law-projection'

    projection_id: str
    source_result: ObjectIdentity
    source_config: ObjectIdentity
    method_config: ObjectIdentity
    history_index: int
    family_id: str
    half: str
    physical_independent_unit_id: str
    numerical_view: SixMatrixResponseProspectiveReactiveSourceLawNumericalView
    features: tuple[MatrixResponseProspectiveReactiveSourceLawFeature, ...]
    routing: SixMatrixResponseProspectiveReactiveSourceLawRouting
    outcomes: tuple[MatrixResponseProspectiveReactiveSourceLawBranchOutcome, ...]
    terminal: MatrixResponseProspectiveReactiveSourceLawProjectionTerminal
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("projection_id", "family_id", "physical_independent_unit_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_ids(self.features, attribute="feature_id", field_name="features")
        require_sorted_unique_ids(self.outcomes, attribute="outcome_id", field_name="outcomes")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if (
            tuple(value.checkpoint_step for value in self.features) != SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_CANDIDATE_STEPS
            or tuple(value.branch for value in self.outcomes) != self.routing.branches
            or (self.terminal is MatrixResponseProspectiveReactiveSourceLawProjectionTerminal.COMPLETE) != (not self.reason_codes)
        ):
            raise ValueError("matrix response prospective reactive source law projection roster differs")


@dataclass(frozen=True, slots=True)
class MatrixResponseProspectiveReactiveSourceLawInterval(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-prospective-reactive-source-law-interval'

    metric_id: str
    estimate: Decimal
    lower_95: Decimal
    upper_95: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.metric_id, field_name="metric_id")
        for name in ("estimate", "lower_95", "upper_95"):
            validate_decimal(getattr(self, name), field_name=name)
        if self.lower_95 > self.upper_95:
            raise ValueError("matrix response prospective reactive source law interval is reversed")


@dataclass(frozen=True, slots=True)
class MatrixResponseProspectiveReactiveSourceLawAggregate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-prospective-reactive-source-law-aggregate'

    aggregate_id: str
    tranche: SixMatrixResponseProspectiveReactiveSourceLawTranche
    source_config: ObjectIdentity
    method_config: ObjectIdentity
    physical_history_count: int
    valid_primary_count: int
    fine_projection_count: int
    source_history_count: int
    entrance_history_count: int
    residence_16_steps_history_count: int
    represented_entrance_families: int
    intervals: tuple[MatrixResponseProspectiveReactiveSourceLawInterval, ...]
    ablation_auc: tuple[tuple[str, Decimal], ...]
    numerical_decision_agreement: Decimal
    numerical_entry_agreement: Decimal
    maximum_dual_entry_time_difference: Decimal
    frozen_score: MatrixResponseProspectiveReactiveSourceLawFrozenScore | None
    terminal: MatrixResponseProspectiveReactiveSourceLawTerminal
    reason_codes: tuple[str, ...]
    condition_false_descendants: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.aggregate_id, field_name="aggregate_id")
        require_sorted_unique_ids(self.intervals, attribute="metric_id", field_name="intervals")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        require_sorted_unique_strings(self.condition_false_descendants, field_name="condition_false_descendants")
        for name in ("numerical_decision_agreement", "numerical_entry_agreement", "maximum_dual_entry_time_difference"):
            validate_decimal(getattr(self, name), field_name=name)
        if self.physical_history_count != SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HISTORY_COUNT or self.fine_projection_count != SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_FINE_COUNT:
            raise ValueError("matrix response prospective reactive source law aggregate denominator differs")
        if self.tranche is SixMatrixResponseProspectiveReactiveSourceLawTranche.DEVELOPMENT:
            if (self.frozen_score is None) != (self.terminal is not MatrixResponseProspectiveReactiveSourceLawTerminal.DEVELOPMENT_SCORE_FROZEN):
                raise ValueError("matrix response prospective reactive source law DEVELOPMENT score terminal differs")
        elif self.frozen_score is not None:
            raise ValueError("matrix response prospective reactive source law PROSPECTIVE_EVALUATION aggregate cannot refit a score")


def _relative_history(states: tuple[SixMatrixState, ...]) -> tuple[SixMatrixState, ...]:
    post = states[SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_PREPARATION_STEPS :]
    return tuple(
        SixMatrixState(
            q=value.q, positions=value.positions, momenta=value.momenta,
            step_index=index, alpha_tilde_x=value.alpha_tilde_x, alpha_tilde_y=value.alpha_tilde_y,
        )
        for index, value in enumerate(post)
    )


def _hold_trace(
    states: tuple[SixMatrixState, ...],
    *,
    source: SixMatrixResponseProspectiveReactiveSourceLawSourceConfig,
    config: MatrixResponseCausalIntersectionResidenceStudyConfig,
    trace_id: str,
    multiplier: int,
) -> SixMatrixResponseTransientControlledInvarianceBranchTrace:
    total_primary = (len(states) - 1) // multiplier
    schedule = build_action_schedule(
        action_word="hold", branch_start_step=0, trigger_parent_step=None,
        total_primary_steps=total_primary, baseline_x=float(states[0].alpha_tilde_x),
        baseline_y=float(states[0].alpha_tilde_y), timestep=float(config.primary_timestep) / multiplier,
        multiplier=multiplier,
    )
    ledger = SixMatrixResponseTransientControlledInvarianceActionLedger(
        ledger_id=f"ledger.{trace_id}", action_word="hold", trigger_parent_step=None,
        requested_sha256=schedule.requested_sha256, accepted_sha256=schedule.requested_sha256,
        applied_sha256=schedule.requested_sha256, realized_sha256=schedule.requested_sha256,
        maximum_excursion=Decimal(0), maximum_increment=Decimal(0), total_variation=Decimal(0),
        squared_action_energy=Decimal(0), generalized_absolute_work=Decimal(0), pulse_count=0,
        exact_baseline_return=True, clipped=False, valid=True, reason_codes=(),
    )
    cadence = ENTRY_CADENCE_STEPS * multiplier
    y_path = np.ascontiguousarray(np.stack([value.positions[1] for value in states]), dtype="<c16")
    digest = sha256(y_path.tobytes()).hexdigest()
    return SixMatrixResponseTransientControlledInvarianceBranchTrace(
        branch_id=trace_id, block_index=0,
        numerical_view_id=source.primary_view.view_id if multiplier == 1 else source.fine_view.view_id,
        parent_step_multiplier=multiplier, start_state=states[0], final_state=states[-1],
        receiver_states=tuple(states[index] for index in range(cadence, len(states), cadence)),
        y_path=y_path, noise_seed_sha256=digest, noise_block_sha256=digest,
        action_ledger=ledger, valid=True, reason_codes=(),
    )


def _history_operands(
    *,
    source: SixMatrixResponseProspectiveReactiveSourceLawSourceConfig,
    config: MatrixResponseCausalIntersectionResidenceStudyConfig,
    routing_states: tuple[SixMatrixState, ...],
    history_index: int,
) -> tuple[tuple[MatrixResponseProspectiveReactiveSourceLawFeature, ...], dict[int, MatrixResponseCausalIntersectionResidenceResponseFactorSample]]:
    states = _relative_history(routing_states)
    trace = _hold_trace(states, source=source, config=config, trace_id=f"matrix-response-prospective-reactive-source-law.history.h{history_index:03d}", multiplier=1)
    roster = derive_probe_roster(
        config_fingerprint=config.probe_roster_config_fingerprint,
        rule_id=config.probe_seed_rule_id,
        scientific_seed=int(config.probe_scientific_input.scientific_seed_sha256, 16),
    )
    evaluation = evaluate_full_intersection_branch(
        trace=trace, intent_word="hold", member=source.member, roster=roster, config=config,
        assessment_start_step=256, assessment_end_step=SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HOLD_STEPS,
    )
    factors = {value.parent_step: value for value in evaluation.samples}
    base = extract_online_phase_features(
        states=states, decision_steps=SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_CANDIDATE_STEPS,
        member=source.member, config=config, parent_step_multiplier=1,
    )
    needed = tuple(sorted({step for candidate in SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_CANDIDATE_STEPS for step in range(candidate - 240, candidate + 1, 16)}))
    observations = {
        step: observe_state(
            observation_id=f"matrix-response-prospective-reactive-source-law.history-observation.h{history_index:03d}.s{step:04d}",
            local_step=step, local_time=step * float(config.primary_timestep),
            state=states[step], member=source.member, config=config,
        )
        for step in needed
    }
    label_observations = tuple(observations[step] for step in needed)
    labels = {value.endpoint_step: value for value in rolling_labels(label_observations, config=config)}
    outputs = []
    for feature in base:
        step = feature.parent_step
        window_steps = tuple(range(step - 240, step + 1, 16))
        window = tuple(observations[value] for value in window_steps)
        closures = np.asarray([float(value.factor_y.closure_ratio) for value in window])
        gaps = np.asarray([_first_nonkernel_gap(states[value]) for value in window_steps])
        full = bool(factors.get(step) and factors[step].full_intersection_pass)
        robust00 = bool(labels.get(step) and labels[step].label == "00")
        summaries = (
            float(sum(value.factor_y.radius_closure_pass for value in window)),
            float(sum(bool(factors.get(value) and factors[value].strict_kernel_window_pass) for value in window_steps)),
            float(np.min(closures)), float(np.mean(closures)), float(np.min(gaps)),
            float(np.sum(np.sign(np.diff(closures))[1:] != np.sign(np.diff(closures))[:-1])),
        )
        shallow_window = window[-4:]
        shallow_steps = window_steps[-4:]
        shallow_closures = np.asarray([float(value.factor_y.closure_ratio) for value in shallow_window])
        shallow_gaps = gaps[-4:]
        shallow = (
            float(sum(value.factor_y.radius_closure_pass for value in shallow_window)),
            float(sum(bool(factors.get(value) and factors[value].strict_kernel_window_pass) for value in shallow_steps)),
            float(np.min(shallow_closures)), float(np.mean(shallow_closures)), float(np.min(shallow_gaps)),
            float(np.sum(np.sign(np.diff(shallow_closures))[1:] != np.sign(np.diff(shallow_closures))[:-1])),
        )
        base_values = tuple(float(getattr(feature, name)) for name in (
            "radius_y", "radius_y_velocity", "closure_y", "closure_y_velocity", "gap_y",
            "gap_y_velocity", "radius_x", "closure_x", "kernel_x", "cross_commutator_ratio",
        ))
        reasons = set()
        if not feature.valid or not np.isfinite((*base_values, *summaries, *shallow)).all():
            reasons.add("INVALID_PREFIX_OPERAND")
        if full:
            reasons.add("ALREADY_FULL_INTERSECTION")
        if robust00:
            reasons.add("ROBUST_00_AT_CUTOFF")
        outputs.append(MatrixResponseProspectiveReactiveSourceLawFeature(
            feature_id=f"feature.h{history_index:03d}.s{step:04d}", checkpoint_step=step,
            values=tuple(_decimal(value) for value in (*base_values, *summaries)),
            shallow_values=tuple(_decimal(value) for value in (*base_values, *shallow)),
            eligible=not reasons, full_intersection_at_cutoff=full, robust_00_at_cutoff=robust00,
            reason_codes=tuple(sorted(reasons)),
        ))
    return tuple(outputs), factors


def _predict(model: MatrixResponseProspectiveReactiveSourceLawFrozenModel, values: Sequence[Decimal]) -> float:
    raw = np.asarray([float(values[index]) for index in model.feature_indices])
    means = np.asarray(tuple(map(float, model.means)))
    scales = np.asarray(tuple(map(float, model.scales)))
    coefficients = np.asarray(tuple(map(float, model.coefficients)))
    linear = float(model.intercept) + float(np.dot((raw - means) / scales, coefficients))
    return float(1.0 / (1.0 + np.exp(-np.clip(linear, -40.0, 40.0))))


def route_matrix_response_study_reactive_source_evaluation(
    *,
    source: SixMatrixResponseProspectiveReactiveSourceLawSourceConfig,
    method: MatrixResponseProspectiveReactiveSourceLawMethodConfig,
    slot: SixMatrixResponseProspectiveReactiveSourceLawHistorySlot,
    numerical_view: SixMatrixResponseProspectiveReactiveSourceLawNumericalView,
    routing_states: tuple[SixMatrixState, ...],
) -> SixMatrixResponseProspectiveReactiveSourceLawRouting:
    """Apply the issued FULL_HISTORY score before any future seed exists."""

    score = method.frozen_score
    if source.tranche is not SixMatrixResponseProspectiveReactiveSourceLawTranche.PROSPECTIVE_EVALUATION or score is None or source.frozen_score != ObjectIdentity.from_record(score.score_id, score):
        raise ValueError("matrix response prospective reactive source law PROSPECTIVE_EVALUATION routing score custody differs")
    features, _ = _history_operands(source=source, config=method.response_factor_config, routing_states=routing_states, history_index=slot.history_index)
    model = score.model("FULL_HISTORY")
    predictions = {value.checkpoint_step: _predict(model, value.values) for value in features if value.eligible}
    source_step = next((step for step in SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_CANDIDATE_STEPS if predictions.get(step, -1.0) >= float(score.threshold)), None)
    start = source.scientific_history_inputs[slot.history_index].control_start_index
    cyclic = SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_CANDIDATE_STEPS[start:] + SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_CANDIDATE_STEPS[:start]
    control_step = next((step for step in cyclic if step != source_step and step in predictions and predictions[step] < float(score.threshold)), None)
    per_arm = 1 if numerical_view is SixMatrixResponseProspectiveReactiveSourceLawNumericalView.SAME_DRIVER_HALF else None
    specs = []
    if source_step is not None:
        for future in range(per_arm or SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_PROSPECTIVE_SOURCE_FUTURES):
            specs.append(SixMatrixResponseProspectiveReactiveSourceLawBranchSpec(f"branch.prospective-evaluation.h{slot.history_index:03d}.source.f{future:02d}", SixMatrixResponseProspectiveReactiveSourceLawArm.SOURCE, source_step, future))
    if control_step is not None:
        for future in range(per_arm or SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_PROSPECTIVE_CONTROL_FUTURES):
            specs.append(SixMatrixResponseProspectiveReactiveSourceLawBranchSpec(f"branch.prospective-evaluation.h{slot.history_index:03d}.control.f{future:02d}", SixMatrixResponseProspectiveReactiveSourceLawArm.CONTROL, control_step, future))
    reasons = []
    if source_step is None:
        reasons.append("NO_SOURCE_THRESHOLD_CROSSING")
    if control_step is None:
        reasons.append("CONTROL_UNAVAILABLE")
    return SixMatrixResponseProspectiveReactiveSourceLawRouting(
        routing_id=f"routing.prospective-evaluation.h{slot.history_index:03d}.{numerical_view.value.lower()}",
        tranche=source.tranche, history_index=slot.history_index, numerical_view=numerical_view,
        frozen_score=ObjectIdentity.from_record(score.score_id, score),
        eligible_steps=tuple(value.checkpoint_step for value in features if value.eligible),
        selected_development_steps=(), source_step=source_step, control_step=control_step,
        branches=tuple(sorted(specs, key=lambda value: (value.arm.value, value.checkpoint_step, value.future_index))),
        reason_codes=tuple(sorted(reasons)),
    )


def _outcome(
    *,
    source: SixMatrixResponseProspectiveReactiveSourceLawSourceConfig,
    config: MatrixResponseCausalIntersectionResidenceStudyConfig,
    spec: SixMatrixResponseProspectiveReactiveSourceLawBranchSpec,
    states: tuple[SixMatrixState, ...],
    multiplier: int,
) -> MatrixResponseProspectiveReactiveSourceLawBranchOutcome:
    trace = _hold_trace(states, source=source, config=config, trace_id=f"trace.{spec.branch_id}.{multiplier}", multiplier=multiplier)
    roster = derive_probe_roster(
        config_fingerprint=config.probe_roster_config_fingerprint,
        rule_id=config.probe_seed_rule_id,
        scientific_seed=int(config.probe_scientific_input.scientific_seed_sha256, 16),
    )
    evaluation = evaluate_full_intersection_branch(
        trace=trace, intent_word="hold", member=source.member, roster=roster, config=config,
        assessment_start_step=ENTRY_START_STEP, assessment_end_step=SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_FUTURE_STEPS,
    )
    samples = evaluation.samples
    observations = tuple(
        observe_state(
            observation_id=f"observation.{spec.branch_id}.{state.step_index // multiplier}",
            local_step=state.step_index // multiplier,
            local_time=(state.step_index // multiplier) * float(config.primary_timestep),
            state=state, member=source.member, config=config,
        )
        for state in trace.receiver_states
    )
    labels = rolling_labels(observations, config=config)
    robust00 = next((value.endpoint_step for value in labels if value.label == "00"), None)
    candidate = next((value.parent_step for value in samples if value.parent_step <= ENTRY_END_STEP and value.full_intersection_pass), None)
    entry = candidate if candidate is not None and (robust00 is None or candidate < robust00) else None
    by_step = {value.parent_step: value.full_intersection_pass for value in samples}
    def duration(steps: int) -> bool:
        return bool(
            entry is not None
            and all(
                by_step.get(step, False)
                for step in range(entry, entry + steps + 1, ENTRY_CADENCE_STEPS)
            )
        )
    reasons = tuple(sorted(evaluation.outcome.reason_codes)) if not evaluation.outcome.technically_valid else ()
    return MatrixResponseProspectiveReactiveSourceLawBranchOutcome(
        outcome_id=f"outcome.{spec.branch_id}.{multiplier}", branch=spec,
        entrance_hit=entry is not None, residence_16_steps=duration(16), residence_32_steps=duration(32), residence_128_steps=duration(128),
        first_entry_step=entry, robust_00_step=robust00,
        technically_valid=not reasons, reason_codes=reasons,
    )


def project_matrix_response_study_reactive_source_history(
    *, persisted: SixMatrixResponseProspectiveReactiveSourceLawPersistedHistory, source: SixMatrixResponseProspectiveReactiveSourceLawSourceConfig, method: MatrixResponseProspectiveReactiveSourceLawMethodConfig,
) -> MatrixResponseProspectiveReactiveSourceLawProjection:
    result = persisted.result
    slot = source.slot(result.history_index)
    reasons: set[str] = set()
    if result.source_config != ObjectIdentity.from_record(source.config_id, source) or method.source_config != ObjectIdentity.from_record(source.config_id, source):
        reasons.add("SOURCE_CUSTODY_MISMATCH")
    features, _ = _history_operands(source=source, config=method.response_factor_config, routing_states=persisted.routing_states, history_index=slot.history_index)
    expected = (
        development_routing(source, slot, result.numerical_view, persisted.routing_states)
        if source.tranche is SixMatrixResponseProspectiveReactiveSourceLawTranche.DEVELOPMENT
        else route_matrix_response_study_reactive_source_evaluation(source=source, method=method, slot=slot, numerical_view=result.numerical_view, routing_states=persisted.routing_states)
    )
    if expected != result.routing:
        reasons.add("ROUTING_RECOMPUTATION_MISMATCH")
    multiplier = 1 if result.numerical_view is SixMatrixResponseProspectiveReactiveSourceLawNumericalView.PRIMARY else 2
    outcomes = tuple(
        _outcome(source=source, config=method.response_factor_config, spec=spec, states=states, multiplier=multiplier)
        for spec, states in zip(result.routing.branches, persisted.branch_states, strict=True)
    )
    if any(not value.technically_valid for value in outcomes):
        reasons.add("FUTURE_FACTOR_INVALID")
    terminal = MatrixResponseProspectiveReactiveSourceLawProjectionTerminal.COMPLETE if not reasons else (
        MatrixResponseProspectiveReactiveSourceLawProjectionTerminal.CUSTODY_INVALID if any("MISMATCH" in value for value in reasons) else MatrixResponseProspectiveReactiveSourceLawProjectionTerminal.NUMERICAL_INVALID
    )
    return MatrixResponseProspectiveReactiveSourceLawProjection(
        projection_id=f"projection.{result.scientific_view_id}", source_result=ObjectIdentity.from_record(result.result_id, result),
        source_config=ObjectIdentity.from_record(source.config_id, source), method_config=ObjectIdentity.from_record(method.config_id, method),
        history_index=slot.history_index, family_id=slot.family_id, half=slot.half.value,
        physical_independent_unit_id=slot.physical_independent_unit_id, numerical_view=result.numerical_view,
        features=features, routing=result.routing, outcomes=outcomes, terminal=terminal, reason_codes=tuple(sorted(reasons)),
    )


def _fit_model(kind: str, matrix: np.ndarray, labels: np.ndarray, *, model_id: str) -> MatrixResponseProspectiveReactiveSourceLawFrozenModel:
    indices = _MODEL_FEATURES[kind]
    data = matrix[:, indices]
    means = np.mean(data, axis=0)
    scales = np.std(data, axis=0)
    scales[scales <= np.finfo(float).eps] = 1.0
    z = (data - means) / scales
    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        linear = theta[0] + z @ theta[1:]
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(linear, -40.0, 40.0)))
        loss = float(np.mean(np.logaddexp(0.0, linear) - labels * linear) + 0.5 * np.dot(theta[1:], theta[1:]))
        residual = probabilities - labels
        gradient = np.concatenate(([np.mean(residual)], z.T @ residual / len(labels) + theta[1:]))
        return loss, gradient
    fitted = minimize(lambda value: objective(value), np.zeros(1 + z.shape[1]), jac=True, method="L-BFGS-B", options={"ftol": 1e-12, "gtol": 1e-9, "maxiter": 1000})
    if not fitted.success and np.linalg.norm(objective(fitted.x)[1]) > 1e-5:
        raise ValueError("matrix response prospective reactive source law logistic fit did not converge")
    return MatrixResponseProspectiveReactiveSourceLawFrozenModel(
        model_id=model_id, model_kind=kind, feature_indices=indices,
        means=tuple(_decimal(value) for value in means), scales=tuple(_decimal(value) for value in scales),
        coefficients=tuple(_decimal(value) for value in fitted.x[1:]), intercept=_decimal(fitted.x[0]), lambda_l2=Decimal(1),
    )


def _auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    positive = [score for score, label in zip(scores, labels, strict=True) if label]
    negative = [score for score, label in zip(scores, labels, strict=True) if not label]
    if not positive or not negative:
        return 0.5
    return float(np.mean([float(left > right) + 0.5 * float(left == right) for left in positive for right in negative]))


def _percentile(metric_id: str, estimate: float, samples: np.ndarray) -> MatrixResponseProspectiveReactiveSourceLawInterval:
    lower, upper = np.quantile(samples, (0.025, 0.975)) if len(samples) else (estimate, estimate)
    return MatrixResponseProspectiveReactiveSourceLawInterval(metric_id, _decimal(estimate), _decimal(float(lower)), _decimal(float(upper)))


def _wilson(metric_id: str, successes: int, trials: int) -> MatrixResponseProspectiveReactiveSourceLawInterval:
    if trials <= 0:
        return MatrixResponseProspectiveReactiveSourceLawInterval(metric_id, Decimal(0), Decimal(0), Decimal(1))
    z = 1.959963984540054
    estimate = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (estimate + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * np.sqrt(estimate * (1.0 - estimate) / trials + z * z / (4.0 * trials * trials))
        / denominator
    )
    return MatrixResponseProspectiveReactiveSourceLawInterval(
        metric_id,
        _decimal(estimate),
        _decimal(max(0.0, centre - radius)),
        _decimal(min(1.0, centre + radius)),
    )


def _development_accessible_lower_bound(
    *,
    selected: tuple[int, ...],
    rows: Sequence[tuple[int, str, MatrixResponseProspectiveReactiveSourceLawFeature, int, int, int]],
    scientific_seed_sha256: str,
    threshold: float,
    replicates: int,
) -> float:
    """History-clustered, family-stratified DEVELOPMENT accessible-yield lower bound."""

    by_family: dict[str, list[tuple[int, int]]] = {
        family: [] for family in SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_FAMILIES
    }
    selected_by_history = {rows[index][0]: rows[index] for index in selected}
    family_by_history = {row[0]: row[1] for row in rows}
    for history_index, family in sorted(family_by_history.items()):
        chosen = selected_by_history.get(history_index)
        by_family[family].append(
            (0, 0) if chosen is None else (chosen[4], chosen[5])
        )
    validate_sha256(scientific_seed_sha256, field_name="scientific_seed_sha256")
    seed = int.from_bytes(bytes.fromhex(scientific_seed_sha256)[:16], "big")
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    samples = np.empty(replicates)
    for replicate in range(replicates):
        hits = 0
        trials = 0
        for values in by_family.values():
            draws = rng.integers(0, len(values), size=len(values))
            for draw in draws:
                history_hits, history_trials = values[int(draw)]
                hits += history_hits
                trials += history_trials
        # Every sampled history remains in the availability denominator.  D0
        # has exactly four primary futures per issued SOURCE checkpoint.
        samples[replicate] = hits / max(1, SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HISTORY_COUNT * 4)
    return float(np.quantile(samples, 0.025))


def _numerical(projections: tuple[MatrixResponseProspectiveReactiveSourceLawProjection, ...]) -> tuple[float, float, float]:
    primary = {value.history_index: value for value in projections if value.numerical_view is SixMatrixResponseProspectiveReactiveSourceLawNumericalView.PRIMARY}
    fine = {value.history_index: value for value in projections if value.numerical_view is SixMatrixResponseProspectiveReactiveSourceLawNumericalView.SAME_DRIVER_HALF}
    decisions, agreements, differences = [], [], []
    for index, right in fine.items():
        left = primary[index]
        decisions.append((left.routing.source_step, left.routing.control_step) == (right.routing.source_step, right.routing.control_step))
        left_outcomes = {(value.branch.arm, value.branch.checkpoint_step): value for value in left.outcomes if value.branch.future_index == 0}
        right_outcomes = {(value.branch.arm, value.branch.checkpoint_step): value for value in right.outcomes}
        for coordinate in set(left_outcomes) & set(right_outcomes):
            a, b = left_outcomes[coordinate], right_outcomes[coordinate]
            agreements.append(a.entrance_hit == b.entrance_hit)
            if a.first_entry_step is not None and b.first_entry_step is not None:
                differences.append(abs(a.first_entry_step - b.first_entry_step) * 0.001)
    return float(np.mean(decisions)), float(np.mean(agreements) if agreements else 1.0), max(differences, default=0.0)


def _evaluation_ablation_auc(
    projections: tuple[MatrixResponseProspectiveReactiveSourceLawProjection, ...],
    score: MatrixResponseProspectiveReactiveSourceLawFrozenScore,
) -> tuple[tuple[str, Decimal], ...]:
    values: dict[str, list[float]] = {kind: [] for kind in _MODEL_FEATURES}
    labels: list[int] = []
    for projection in projections:
        feature_by_step = {value.checkpoint_step: value for value in projection.features}
        outcomes: dict[int, list[MatrixResponseProspectiveReactiveSourceLawBranchOutcome]] = defaultdict(list)
        for outcome in projection.outcomes:
            outcomes[outcome.branch.checkpoint_step].append(outcome)
        for step in sorted(outcomes):
            feature = feature_by_step.get(step)
            if feature is None or not feature.eligible:
                continue
            labels.append(int(any(value.entrance_hit for value in outcomes[step])))
            for kind in _MODEL_FEATURES:
                model = score.model(kind)
                if kind == "SHALLOW_HISTORY":
                    operands = feature.shallow_values
                elif kind == "PHASE_TIME_NULL":
                    phase = _decimal(step / SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HOLD_STEPS)
                    operands = (phase,) * len(PROSPECTIVE_REACTIVE_SOURCE_LAW_FEATURE_NAMES)
                else:
                    operands = feature.values
                values[kind].append(_predict(model, operands))
    return tuple(
        sorted(
            (kind, _decimal(_auc(kind_values, labels)))
            for kind, kind_values in values.items()
        )
    )


def reduce_matrix_response_study_reactive_source(
    *, source: SixMatrixResponseProspectiveReactiveSourceLawSourceConfig, method: MatrixResponseProspectiveReactiveSourceLawMethodConfig, projections: tuple[MatrixResponseProspectiveReactiveSourceLawProjection, ...], analysis_identity: str,
) -> MatrixResponseProspectiveReactiveSourceLawAggregate:
    """Fit DEVELOPMENT once or emit exactly one fresh PROSPECTIVE_EVALUATION terminal."""

    validate_stable_id(analysis_identity, field_name="analysis_identity")
    method.analysis_scientific_input.require_target_analysis(analysis_identity)
    if method.tranche is not source.tranche:
        raise ValueError("reactive-source method and source tranche differ")
    primary = tuple(sorted((value for value in projections if value.numerical_view is SixMatrixResponseProspectiveReactiveSourceLawNumericalView.PRIMARY), key=lambda value: value.history_index))
    fine = tuple(value for value in projections if value.numerical_view is SixMatrixResponseProspectiveReactiveSourceLawNumericalView.SAME_DRIVER_HALF)
    if len(primary) != SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HISTORY_COUNT or len(fine) != SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_FINE_COUNT or {value.history_index for value in primary} != set(range(SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HISTORY_COUNT)):
        raise ValueError("matrix response prospective reactive source law finalizer lacks the complete projection roster")
    decision_agreement, entry_agreement, max_difference = _numerical(projections)
    valid = tuple(value for value in primary if value.terminal is MatrixResponseProspectiveReactiveSourceLawProjectionTerminal.COMPLETE)
    method_identity = ObjectIdentity.from_record(method.config_id, method)
    source_identity = ObjectIdentity.from_record(source.config_id, source)
    descendants = tuple(sorted(("LOWER_LAW", "MAINTAINED_DURATION", "programme admission", "runtime owner", "prospective evaluation", "PROTECTED_NESTING")))
    if source.tranche is SixMatrixResponseProspectiveReactiveSourceLawTranche.DEVELOPMENT:
        rows = []
        for projection in valid:
            outcomes = defaultdict(list)
            for value in projection.outcomes:
                outcomes[value.branch.checkpoint_step].append(value)
            for feature in projection.features:
                if feature.checkpoint_step in projection.routing.selected_development_steps and feature.eligible:
                    branch = outcomes[feature.checkpoint_step]
                    rows.append((projection.history_index, projection.family_id, feature, int(any(value.entrance_hit for value in branch)), sum(value.entrance_hit for value in branch), len(branch)))
        score: MatrixResponseProspectiveReactiveSourceLawFrozenScore | None = None
        reasons: list[str] = []
        ablation_auc: list[tuple[str, Decimal]] = []
        if len(valid) < 240 or len(rows) < 512 or not any(value[3] for value in rows):
            reasons.append("DEVELOPMENT_PANEL_INSUFFICIENT")
        else:
            matrix = np.asarray([[float(value) for value in row[2].values] for row in rows])
            shallow = np.asarray([[float(value) for value in row[2].shallow_values] for row in rows])
            labels = np.asarray([row[3] for row in rows], dtype=float)
            oof = {kind: np.zeros(len(rows)) for kind in _MODEL_FEATURES}
            for fold in range(8):
                train = np.asarray([(row[0] // 4) % 8 != fold for row in rows])
                test = ~train
                for kind in _MODEL_FEATURES:
                    data = shallow if kind == "SHALLOW_HISTORY" else matrix
                    if kind == "PHASE_TIME_NULL":
                        data = np.asarray([[row[2].checkpoint_step / 4096.0] * 16 for row in rows])
                    fitted = _fit_model(kind, data[train], labels[train], model_id=f"model.oof.{kind.lower()}.f{fold}")
                    values = data[test]
                    oof[kind][test] = [_predict(fitted, tuple(map(_decimal, item))) for item in values]
            models = []
            for kind in sorted(_MODEL_FEATURES):
                data = shallow if kind == "SHALLOW_HISTORY" else matrix
                if kind == "PHASE_TIME_NULL":
                    data = np.asarray([[row[2].checkpoint_step / 4096.0] * 16 for row in rows])
                models.append(_fit_model(kind, data, labels, model_id=f"model.matrix-response-prospective-reactive-source-law.{kind.lower()}"))
                ablation_auc.append(
                    (
                        kind,
                        _decimal(
                            _auc(
                                oof[kind].tolist(),
                                labels.astype(int).tolist(),
                            )
                        ),
                    )
                )
            eligible_thresholds: list[tuple[float, float, str]] = []
            for threshold in (value / 100 for value in range(5, 100, 5)):
                by_history: dict[int, list[int]] = defaultdict(list)
                for index, row in enumerate(rows):
                    by_history[row[0]].append(index)
                selected = [min((index for index in indices if oof["FULL_HISTORY"][index] >= threshold), key=lambda index: rows[index][2].checkpoint_step, default=-1) for indices in by_history.values()]
                selected = [index for index in selected if index >= 0]
                families = [rows[index][1] for index in selected]
                counts = {family: families.count(family) for family in set(families)}
                hits = sum(rows[index][4] for index in selected)
                trials = sum(rows[index][5] for index in selected)
                below = [index for index in range(len(rows)) if oof["FULL_HISTORY"][index] < threshold]
                control_rate = sum(rows[index][4] for index in below) / max(1, sum(rows[index][5] for index in below))
                source_rate = hits / max(1, trials)
                accessible = len(selected) / 256 * source_rate
                if len(selected) >= 48 and len(counts) >= 3 and min(counts.values()) >= 8 and max(counts.values()) / len(selected) <= 0.60 and source_rate > control_rate and accessible > 0:
                    rule_identity = sha256(
                        canonical_json_bytes(
                            (
                                method.scientific_fingerprint,
                                f"{threshold:.2f}",
                                tuple(rows[index][0] for index in selected),
                            )
                        )
                    ).hexdigest()
                    accessible_lower = _development_accessible_lower_bound(
                        selected=tuple(selected),
                        rows=rows,
                        scientific_seed_sha256=method.analysis_scientific_input.threshold_seed(threshold),
                        threshold=threshold,
                        replicates=method.bootstrap_replicates,
                    )
                    eligible_thresholds.append(
                        (accessible_lower, threshold, rule_identity)
                    )
            if eligible_thresholds:
                _, threshold, _ = sorted(
                    eligible_thresholds,
                    key=lambda value: (-value[0], -value[1], value[2]),
                )[0]
                panel_digest = sha256(canonical_json_bytes([(row[0], row[2].checkpoint_step, row[3], row[4], row[5]) for row in rows])).hexdigest()
                score = MatrixResponseProspectiveReactiveSourceLawFrozenScore(
                    score_id=f"matrix-response-prospective-reactive-source-law.frozen-score.{panel_digest[:16]}", scientific_fingerprint=method.scientific_fingerprint,
                    development_source_config=source_identity, fit_panel_sha256=panel_digest,
                    models=tuple(sorted(models, key=lambda value: value.model_id)), threshold=_decimal(threshold),
                    threshold_grid=tuple(Decimal(value) / Decimal(100) for value in range(5, 100, 5)),
                    fold_rule_id="matrix-response-prospective-reactive-source-law.eight-fold-history-grouped-family-stratified",
                    checkpoint_grid=SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_CANDIDATE_STEPS, grants_authority=False,
                )
            else:
                reasons.append("NO_ELIGIBLE_DEVELOPMENT_THRESHOLD")
        terminal = MatrixResponseProspectiveReactiveSourceLawTerminal.DEVELOPMENT_SCORE_FROZEN if score is not None else MatrixResponseProspectiveReactiveSourceLawTerminal.NO_DEVELOPMENT_SOURCE_RULE
        return MatrixResponseProspectiveReactiveSourceLawAggregate(
            aggregate_id=f"aggregate.matrix-response-prospective-reactive-source-law.development.{analysis_identity}", tranche=source.tranche,
            source_config=source_identity, method_config=method_identity, physical_history_count=256,
            valid_primary_count=len(valid), fine_projection_count=64, source_history_count=0,
            entrance_history_count=sum(any(value.entrance_hit for value in projection.outcomes) for projection in primary),
            residence_16_steps_history_count=sum(any(value.residence_16_steps for value in projection.outcomes) for projection in primary),
            represented_entrance_families=len({value.family_id for value in primary if any(outcome.entrance_hit for outcome in value.outcomes)}),
            intervals=(), ablation_auc=tuple(sorted(ablation_auc)), numerical_decision_agreement=_decimal(decision_agreement),
            numerical_entry_agreement=_decimal(entry_agreement), maximum_dual_entry_time_difference=_decimal(max_difference),
            frozen_score=score, terminal=terminal, reason_codes=tuple(sorted(reasons)), condition_false_descendants=descendants,
        )

    reasons = []
    if len(valid) < 240 or any(sum(value.family_id == family for value in valid) < 56 for family in SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_FAMILIES):
        reasons.append("MINIMUM_EVALUABILITY_FAILED")
    if decision_agreement < 0.95 or entry_agreement < 0.90 or max_difference > 0.032:
        reasons.append("NUMERICAL_AUDIT_FAILED")
    source_histories = tuple(value for value in valid if value.routing.source_step is not None)
    control_histories = tuple(value for value in valid if value.routing.control_step is not None)
    if len(control_histories) < 224:
        reasons.append("CONTROL_PANEL_EVALUABILITY_FAILED")
    source_outcomes = tuple(outcome for value in source_histories for outcome in value.outcomes if outcome.branch.arm is SixMatrixResponseProspectiveReactiveSourceLawArm.SOURCE)
    control_outcomes = tuple(outcome for value in control_histories for outcome in value.outcomes if outcome.branch.arm is SixMatrixResponseProspectiveReactiveSourceLawArm.CONTROL)
    p_source = len(source_histories) / 256
    p_entry_source = sum(value.entrance_hit for value in source_outcomes) / max(1, len(source_outcomes))
    p_entry_control = sum(value.entrance_hit for value in control_outcomes) / max(1, len(control_outcomes))
    uplift = p_entry_source - p_entry_control
    accessible = p_source * p_entry_source
    seed = int.from_bytes(bytes.fromhex(method.analysis_scientific_input.evaluation_seed_sha256)[:16], "big")
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    bootstrap = np.empty((method.bootstrap_replicates, 5))
    by_family = {family: tuple(value for value in valid if value.family_id == family) for family in SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_FAMILIES}
    for replicate in range(method.bootstrap_replicates):
        sampled = [
            by_family[family][index]
            for family in SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_FAMILIES
            for index in rng.integers(
                0,
                len(by_family[family]),
                size=len(by_family[family]),
            )
        ]
        sources = [value for value in sampled if value.routing.source_step is not None]
        controls = [value for value in sampled if value.routing.control_step is not None]
        source_values = [outcome.entrance_hit for value in sources for outcome in value.outcomes if outcome.branch.arm is SixMatrixResponseProspectiveReactiveSourceLawArm.SOURCE]
        control_values = [outcome.entrance_hit for value in controls for outcome in value.outcomes if outcome.branch.arm is SixMatrixResponseProspectiveReactiveSourceLawArm.CONTROL]
        ps = len(sources) / len(sampled)
        pe = float(np.mean(source_values)) if source_values else 0.0
        pc = float(np.mean(control_values)) if control_values else 0.0
        bootstrap[replicate] = (ps, pe, pc, pe - pc, ps * pe)
    bootstrap_intervals = tuple(
        _percentile(name, estimate, bootstrap[:, index])
        for index, (name, estimate) in enumerate((
            ("p-source", p_source), ("p-entry-source", p_entry_source),
            ("p-entry-control", p_entry_control), ("source-uplift", uplift),
            ("accessible-yield", accessible),
        ))
    )
    source_entrance_history_count = sum(
        any(
            outcome.entrance_hit
            for outcome in value.outcomes
            if outcome.branch.arm is SixMatrixResponseProspectiveReactiveSourceLawArm.SOURCE
        )
        for value in source_histories
    )
    control_entrance_history_count = sum(
        any(
            outcome.entrance_hit
            for outcome in value.outcomes
            if outcome.branch.arm is SixMatrixResponseProspectiveReactiveSourceLawArm.CONTROL
        )
        for value in control_histories
    )
    intervals = tuple(
        sorted(
            (
                *bootstrap_intervals,
                _wilson("wilson-p-source", len(source_histories), SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HISTORY_COUNT),
                _wilson(
                    "wilson-source-history-entry",
                    source_entrance_history_count,
                    len(source_histories),
                ),
                _wilson(
                    "wilson-control-history-entry",
                    control_entrance_history_count,
                    len(control_histories),
                ),
            ),
            key=lambda value: value.metric_id,
        )
    )
    interval = {value.metric_id: value for value in intervals}
    entrance_histories = tuple(value for value in source_histories if any(outcome.entrance_hit for outcome in value.outcomes if outcome.branch.arm is SixMatrixResponseProspectiveReactiveSourceLawArm.SOURCE))
    residence_16_steps_histories = tuple(value for value in source_histories if any(outcome.residence_16_steps for outcome in value.outcomes if outcome.branch.arm is SixMatrixResponseProspectiveReactiveSourceLawArm.SOURCE))
    family_counts = {family: sum(value.family_id == family for value in entrance_histories) for family in SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_FAMILIES}
    represented = {family: count for family, count in family_counts.items() if count >= 3}
    half_uplifts = []
    for half in ("FIRST", "SECOND"):
        left = [outcome.entrance_hit for value in source_histories if value.half == half for outcome in value.outcomes if outcome.branch.arm is SixMatrixResponseProspectiveReactiveSourceLawArm.SOURCE]
        right = [outcome.entrance_hit for value in control_histories if value.half == half for outcome in value.outcomes if outcome.branch.arm is SixMatrixResponseProspectiveReactiveSourceLawArm.CONTROL]
        half_uplifts.append((float(np.mean(left)) if left else 0.0) - (float(np.mean(right)) if right else 0.0))
    family_uplifts = []
    for family in SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_FAMILIES:
        left = [outcome.entrance_hit for value in source_histories if value.family_id == family for outcome in value.outcomes if outcome.branch.arm is SixMatrixResponseProspectiveReactiveSourceLawArm.SOURCE]
        right = [outcome.entrance_hit for value in control_histories if value.family_id == family for outcome in value.outcomes if outcome.branch.arm is SixMatrixResponseProspectiveReactiveSourceLawArm.CONTROL]
        family_uplifts.append((float(np.mean(left)) if left else 0.0) - (float(np.mean(right)) if right else 0.0))
    if method.frozen_score is None:
        raise ValueError("matrix response prospective reactive source law PROSPECTIVE_EVALUATION finalization lacks its issued frozen score")
    evaluation_ablation_auc = _evaluation_ablation_auc(valid, method.frozen_score)
    auc_by_kind = {kind: float(value) for kind, value in evaluation_ablation_auc}
    phase_null_dominates = (
        auc_by_kind["FULL_HISTORY"] <= 0.5
        or auc_by_kind["FULL_HISTORY"] <= auc_by_kind["PHASE_TIME_NULL"]
    )
    positive = (
        not reasons and len(source_histories) >= 48
        and float(interval["p-source"].lower_95) > 0.10
        and float(interval["p-entry-source"].lower_95) >= 0.05
        and float(interval["source-uplift"].lower_95) > 0
        and float(interval["accessible-yield"].lower_95) >= 0.01
        and len(entrance_histories) >= 16 and len(represented) >= 3
        and max(family_counts.values(), default=0) / max(1, len(entrance_histories)) <= 0.60
        and len(residence_16_steps_histories) >= 8 and len({value.family_id for value in residence_16_steps_histories}) >= 2
        and all(value > 0 for value in half_uplifts) and sum(value > 0 for value in family_uplifts) >= 3
        and not phase_null_dominates
    )
    any_signal = bool(entrance_histories or uplift > 0)
    if reasons:
        terminal = MatrixResponseProspectiveReactiveSourceLawTerminal.NUMERICAL_INVALID if "NUMERICAL_AUDIT_FAILED" in reasons else MatrixResponseProspectiveReactiveSourceLawTerminal.SOURCE_EVALUATION_UNEVALUABLE
    elif positive:
        terminal = MatrixResponseProspectiveReactiveSourceLawTerminal.SUPPORTED
    elif any_signal:
        terminal = MatrixResponseProspectiveReactiveSourceLawTerminal.CONDITIONAL
        reasons.append("POSITIVE_GATE_CONJUNCTION_FAILED")
        if phase_null_dominates:
            reasons.append("PHASE_TIME_NULL_NOT_OUTPERFORMED")
    else:
        terminal = MatrixResponseProspectiveReactiveSourceLawTerminal.NO_SOURCE
        reasons.append("NO_SOURCE_ENTRANCE_OR_POSITIVE_UPLIFT")
    return MatrixResponseProspectiveReactiveSourceLawAggregate(
        aggregate_id=f"aggregate.matrix-response-prospective-reactive-source-law.prospective-evaluation.{analysis_identity}", tranche=source.tranche,
        source_config=source_identity, method_config=method_identity, physical_history_count=256,
        valid_primary_count=len(valid), fine_projection_count=64, source_history_count=len(source_histories),
        entrance_history_count=len(entrance_histories), residence_16_steps_history_count=len(residence_16_steps_histories),
        represented_entrance_families=len(represented), intervals=intervals, ablation_auc=evaluation_ablation_auc,
        numerical_decision_agreement=_decimal(decision_agreement), numerical_entry_agreement=_decimal(entry_agreement),
        maximum_dual_entry_time_difference=_decimal(max_difference), frozen_score=None, terminal=terminal,
        reason_codes=tuple(sorted(reasons)), condition_false_descendants=descendants,
    )


__all__ = [name for name in globals() if name.startswith("PROSPECTIVE_REACTIVE_SOURCE_LAW") or name.startswith("MatrixResponseProspectiveReactiveSourceLaw") or name.startswith("project_matrix_response_study_reactive_source") or name.startswith("reduce_matrix_response_study_reactive_source") or name.startswith("route_matrix_response_study_reactive_source")]
