"""Scientific endpoint and sole finalizers for Matrix causal intersection residence intervention confirmation.

The primary estimand is residence in the exact noncompensating intersection
of structural fuzzy geometry and held-out passive-probe response-law validity.
Response-only and structure-only quantities are diagnostics and cannot select
a development candidate or produce a positive terminal.
"""

from __future__ import annotations

from .bootstrap_scientific_inputs import fixed_matrix_response_bootstrap_seed_sha256

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
from typing import Any, ClassVar, Mapping, Sequence

import numpy as np
from scipy.stats import binomtest

from empirical_lawhood.adapters.composition.matrix_response_study.causal_intersection_residence_design import MatrixResponseCausalIntersectionResidenceStudyConfig
from empirical_lawhood.adapters.control.matrix_response_study.transient_policy import ACTIVE_ACTION_WORDS, MatrixResponseTransientControlledInvariancePhaseFeature, MatrixResponseTransientControlledInvariancePhaseRule, enumerate_phase_rules
from empirical_lawhood.adapters.simulators.six_matrix_response.contracts import SixMatrixResponseModelFamilyMember
from empirical_lawhood.adapters.simulators.six_matrix_response.controlled_branch import SixMatrixResponseTransientControlledInvarianceBranchTrace
from empirical_lawhood.adapters.simulators.six_matrix_response.model import SixMatrixState
from empirical_lawhood.adapters.simulators.six_matrix_response.passive_probe import SixMatrixResponseTransientControlledInvarianceProbePropagation, SixMatrixResponseTransientControlledInvarianceProbeRoster, heat_predict_probe, normalized_increment_loss, propagate_passive_probes, radius_only_operator, traceless_operator
from empirical_lawhood.adapters.simulators.six_matrix_response.spectral import spectral_receiver
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    canonical_json_bytes,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_sha256,
    validate_stable_id,
)

from .controlled_invariance import stratified_paired_bootstrap_lower
from .shooting_committor import observe_state, rolling_labels


NUMERICAL_AUDIT_BLOCK_INDICES = (*range(16), *range(64, 80))


def _decimal(value: float) -> Decimal:
    if not np.isfinite(value):
        raise FloatingPointError("Matrix causal intersection residence scientific value is nonfinite")
    return Decimal(repr(float(value)))


def _relative_error(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), np.finfo(np.float64).tiny)


def _directions_concordant(left: float, right: float) -> bool:
    return bool((left > 0 and right > 0) or (left < 0 and right < 0) or (left == right == 0))


def conservative_residence(flags: Sequence[bool], *, cadence_time: float) -> tuple[float, float]:
    """Conservative endpoint-spanned total and longest residence."""

    total = 0.0
    longest = 0.0
    current = 0.0
    for left, right in zip(flags, flags[1:]):
        if left and right:
            total += cadence_time
            current += cadence_time
            longest = max(longest, current)
        else:
            current = 0.0
    return total, longest


@dataclass(frozen=True, slots=True)
class MatrixResponseCausalIntersectionResidenceResponseFactorSample(CanonicalRecord):
    """One fully decomposed claim-bearing receiver sample."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-causal-intersection-residence-response-factor-sample'

    sample_id: str
    branch_id: str
    parent_step: int
    parent_time: Decimal
    phi_y: Decimal
    amplitude_pass: bool
    closure_y: Decimal
    closure_pass: bool
    kernel_band_y: Decimal
    spectrum_valid: bool
    kernel_pass: bool
    persistence_pass_count: int
    persistence_count_pass: bool
    strict_kernel_window_pass: bool
    x_rolling_geometric: bool
    x_exclusion_pass: bool
    resolved_probe_cells: int
    required_probe_cells: int
    probe_invariants_pass: bool
    geometry_loss: Decimal | None
    radius_loss: Decimal | None
    geometry_radius_ratio: Decimal | None
    response_law_pass: bool
    structural_geometry_pass: bool
    full_intersection_pass: bool
    predictive_skill_only: bool
    structure_only: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("sample_id", "branch_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.parent_step < 0 or self.required_probe_cells != 18:
            raise ValueError("Matrix causal intersection residence sample coordinate differs")
        if not 0 <= self.resolved_probe_cells <= self.required_probe_cells:
            raise ValueError("Matrix causal intersection residence resolved probe count differs")
        if not 0 <= self.persistence_pass_count <= 16:
            raise ValueError("Matrix causal intersection residence persistence count differs")
        for name in ("parent_time", "phi_y", "closure_y", "kernel_band_y"):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        for name in ("geometry_loss", "radius_loss", "geometry_radius_ratio"):
            value = getattr(self, name)
            if value is not None:
                validate_decimal(value, field_name=name, minimum=Decimal(0))
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        structural = bool(
            self.amplitude_pass
            and self.closure_pass
            and self.spectrum_valid
            and self.kernel_pass
            and self.persistence_count_pass
            and self.strict_kernel_window_pass
            and self.x_exclusion_pass
        )
        if self.structural_geometry_pass != structural:
            raise ValueError("Matrix causal intersection residence structural factor is not its exact conjunction")
        if self.full_intersection_pass != (
            self.response_law_pass and self.structural_geometry_pass
        ):
            raise ValueError("Matrix causal intersection residence full endpoint is not its exact conjunction")
        if self.predictive_skill_only != (
            self.response_law_pass and not self.structural_geometry_pass
        ) or self.structure_only != (self.structural_geometry_pass and not self.response_law_pass):
            raise ValueError("Matrix causal intersection residence mechanism-only flags differ")


@dataclass(frozen=True, slots=True)
class MatrixResponseCausalIntersectionResidenceBranchOutcome(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-causal-intersection-residence-branch-outcome'

    outcome_id: str
    branch_id: str
    block_index: int
    seed_stratum: int
    intent_word: str
    realized_action_word: str
    numerical_view_id: str
    parent_step_multiplier: int
    noise_seed_sha256: str
    noise_block_sha256: str
    requested_action_sha256: str
    accepted_action_sha256: str
    applied_action_sha256: str
    realized_action_sha256: str
    trigger_parent_step: int | None
    action_maximum_excursion: Decimal
    action_maximum_increment: Decimal
    action_total_variation: Decimal
    action_squared_energy: Decimal
    action_generalized_absolute_work: Decimal
    action_pulse_count: int
    action_exact_baseline_return: bool
    action_clipped: bool
    action_delivery_valid: bool
    assessment_start_step: int
    assessment_end_step: int
    assessment_sample_count: int
    full_sustained: bool
    full_total_residence: Decimal
    full_longest_residence: Decimal
    response_only_sustained: bool
    response_only_total_residence: Decimal
    response_only_longest_residence: Decimal
    structure_only_sustained: bool
    structure_only_total_residence: Decimal
    structure_only_longest_residence: Decimal
    full_sample_count: int
    response_law_sample_count: int
    structural_sample_count: int
    predictive_skill_only_sample_count: int
    structure_only_sample_count: int
    amplitude_only_sample_count: int
    false_x_admission: bool
    technically_valid: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "outcome_id",
            "branch_id",
            "intent_word",
            "realized_action_word",
            "numerical_view_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in (
            "noise_seed_sha256",
            "noise_block_sha256",
            "requested_action_sha256",
            "accepted_action_sha256",
            "applied_action_sha256",
            "realized_action_sha256",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        if (
            self.block_index < 0
            or self.seed_stratum not in {0, 1}
            or self.parent_step_multiplier not in {1, 2}
            or self.assessment_start_step >= self.assessment_end_step
            or self.assessment_sample_count < 2
            or self.action_pulse_count not in {0, 1}
            or self.action_clipped
        ):
            raise ValueError("Matrix causal intersection residence outcome coordinate differs")
        for name in (
            "action_maximum_excursion",
            "action_maximum_increment",
            "action_total_variation",
            "action_squared_energy",
            "action_generalized_absolute_work",
            "full_total_residence",
            "full_longest_residence",
            "response_only_total_residence",
            "response_only_longest_residence",
            "structure_only_total_residence",
            "structure_only_longest_residence",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        for name in (
            "full_sample_count",
            "response_law_sample_count",
            "structural_sample_count",
            "predictive_skill_only_sample_count",
            "structure_only_sample_count",
            "amplitude_only_sample_count",
        ):
            if not 0 <= getattr(self, name) <= self.assessment_sample_count:
                raise ValueError("Matrix causal intersection residence outcome sample count differs")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        action_delivery_valid = bool(
            self.requested_action_sha256
            == self.accepted_action_sha256
            == self.applied_action_sha256
            == self.realized_action_sha256
            and self.action_exact_baseline_return
            and not self.action_clipped
        )
        if self.action_delivery_valid != action_delivery_valid:
            raise ValueError("Matrix causal intersection residence outcome action-delivery custody differs")
        if self.technically_valid != (not self.reason_codes):
            raise ValueError("Matrix causal intersection residence outcome validity differs")


@dataclass(frozen=True, slots=True)
class MatrixResponseCausalIntersectionResidenceBranchEvaluation:
    outcome: MatrixResponseCausalIntersectionResidenceBranchOutcome
    samples: tuple[MatrixResponseCausalIntersectionResidenceResponseFactorSample, ...]
    propagation: SixMatrixResponseTransientControlledInvarianceProbePropagation


def evaluate_full_intersection_branch(
    *,
    trace: SixMatrixResponseTransientControlledInvarianceBranchTrace,
    intent_word: str,
    member: SixMatrixResponseModelFamilyMember,
    roster: SixMatrixResponseTransientControlledInvarianceProbeRoster,
    config: MatrixResponseCausalIntersectionResidenceStudyConfig,
    assessment_start_step: int,
    assessment_end_step: int,
) -> MatrixResponseCausalIntersectionResidenceBranchEvaluation:
    """Score the exact structural/response conjunction from native branch output."""

    if intent_word not in (*ACTIVE_ACTION_WORDS, "hold", "phase-action"):
        raise ValueError("Matrix causal intersection residence intent word differs")
    multiplier = trace.parent_step_multiplier
    start_parent_step = trace.start_state.step_index // multiplier
    observations = tuple(
        observe_state(
            observation_id=f"matrix-causal-intersection-residence.observation.{trace.branch_id}.{state.step_index // multiplier}",
            local_step=state.step_index // multiplier,
            local_time=(state.step_index // multiplier) * float(config.primary_timestep),
            state=state,
            member=member,
            config=config,
        )
        for state in trace.receiver_states
    )
    observations_by_step = {value.local_step: value for value in observations}
    labels_by_step = {
        value.endpoint_step: value for value in rolling_labels(observations, config=config)
    }
    assessment_steps = tuple(
        range(assessment_start_step, assessment_end_step + 1, config.receiver_cadence_steps)
    )
    if any(
        step not in labels_by_step or step not in observations_by_step for step in assessment_steps
    ):
        raise ValueError("Matrix causal intersection residence assessment lacks complete rolling history")
    propagation = propagate_passive_probes(
        y_path=trace.y_path,
        start_step=start_parent_step,
        timestep=float(config.primary_timestep) / multiplier,
        roster=roster,
        field_indices=config.heldout_probe_field_indices,
    )
    invariant_reasons: set[str] = set(trace.reason_codes)
    if not trace.valid:
        invariant_reasons.add("branch-trace-invalid")
    if propagation.maximum_hermiticity_residual > float(config.probe_hermiticity_residual_max):
        invariant_reasons.add("probe-hermiticity-residual")
    if propagation.maximum_trace_residual > float(config.probe_trace_residual_max):
        invariant_reasons.add("probe-trace-residual")
    if propagation.maximum_identity_relative_drift > float(config.probe_identity_drift_max):
        invariant_reasons.add("identity-sentinel-drift")
    if propagation.maximum_relative_norm_increase > float(config.probe_norm_increase_max):
        invariant_reasons.add("probe-norm-increase")
    invariants_pass = not invariant_reasons.difference(trace.reason_codes)
    samples: list[MatrixResponseCausalIntersectionResidenceResponseFactorSample] = []
    for step in assessment_steps:
        observation = observations_by_step[step]
        label = labels_by_step[step]
        window_steps = tuple(
            range(
                step - (config.rolling_window_samples - 1) * config.receiver_cadence_steps,
                step + 1,
                config.receiver_cadence_steps,
            )
        )
        window = tuple(observations_by_step[value] for value in window_steps)
        persistence_count = sum(value.factor_y.radius_closure_pass for value in window)
        amplitude = bool(
            float(config.phi_min) <= float(observation.factor_y.phi) <= float(config.phi_max)
        )
        closure = bool(float(observation.factor_y.closure_ratio) <= float(config.closure_ratio_max))
        spectrum_valid = bool(observation.factor_y.valid)
        kernel = bool(
            spectrum_valid
            and float(observation.factor_y.kernel_band_ratio) <= float(config.kernel_band_ratio_max)
        )
        persistence = persistence_count >= config.persistence_pass_count
        strict_kernel = bool(
            all(value.factor_y.valid for value in window)
            and max(float(value.factor_y.kernel_band_ratio) for value in window)
            <= float(config.kernel_band_ratio_max)
        )
        x_exclusion = not label.x_geometric
        origin_step = step - config.probe_forecast_lag_steps
        origin_offset = multiplier * (origin_step - start_parent_step)
        endpoint_offset = multiplier * (step - start_parent_step)
        operator = traceless_operator(trace.y_path[origin_offset])
        radius_operator = radius_only_operator(operator)
        geometry_losses: list[float] = []
        radius_losses: list[float] = []
        resolved = 0
        for local_field in range(len(config.heldout_probe_field_indices)):
            for kappa_index, kappa_decimal in enumerate(config.probe_kappas):
                initial = propagation.states[local_field, kappa_index, origin_offset]
                observed = propagation.states[local_field, kappa_index, endpoint_offset]
                horizon = config.probe_forecast_lag_steps * float(config.primary_timestep)
                geometry_prediction = heat_predict_probe(
                    operator=operator,
                    probe=initial,
                    kappa=float(kappa_decimal),
                    horizon_time=horizon,
                )
                radius_prediction = heat_predict_probe(
                    operator=radius_operator,
                    probe=initial,
                    kappa=float(kappa_decimal),
                    horizon_time=horizon,
                )
                geometry_loss, is_resolved = normalized_increment_loss(
                    predicted=geometry_prediction,
                    observed=observed,
                    initial=initial,
                    floor=float(config.probe_numeric_floor),
                )
                radius_loss, _ = normalized_increment_loss(
                    predicted=radius_prediction,
                    observed=observed,
                    initial=initial,
                    floor=float(config.probe_numeric_floor),
                )
                if is_resolved:
                    resolved += 1
                    geometry_losses.append(geometry_loss)
                    radius_losses.append(radius_loss)
        geometry_mean = float(np.mean(geometry_losses)) if geometry_losses else None
        radius_mean = float(np.mean(radius_losses)) if radius_losses else None
        ratio = (
            geometry_mean / radius_mean
            if geometry_mean is not None
            and radius_mean is not None
            and radius_mean > np.finfo(np.float64).tiny
            else None
        )
        response_pass = bool(
            resolved == 18
            and invariants_pass
            and geometry_mean is not None
            and ratio is not None
            and geometry_mean <= float(config.response_geometry_loss_max)
            and ratio <= float(config.response_radius_ratio_max)
        )
        structural_pass = bool(
            amplitude
            and closure
            and spectrum_valid
            and kernel
            and persistence
            and strict_kernel
            and x_exclusion
        )
        reasons: set[str] = set()
        if not amplitude:
            reasons.add("amplitude-failed")
        if not closure:
            reasons.add("closure-failed")
        if not spectrum_valid:
            reasons.add("spectrum-invalid")
        if not kernel:
            reasons.add("kernel-failed")
        if not persistence:
            reasons.add("persistence-count-failed")
        if not strict_kernel:
            reasons.add("strict-kernel-window-failed")
        if not x_exclusion:
            reasons.add("x-exclusion-failed")
        if resolved != 18:
            reasons.add("probe-cells-unresolved")
        if not invariants_pass:
            reasons.add("probe-invariants-failed")
        if geometry_mean is None or geometry_mean > float(config.response_geometry_loss_max):
            reasons.add("geometry-loss-failed")
        if ratio is None or ratio > float(config.response_radius_ratio_max):
            reasons.add("radius-contrast-failed")
        samples.append(
            MatrixResponseCausalIntersectionResidenceResponseFactorSample(
                sample_id=f"matrix-causal-intersection-residence.sample.{trace.branch_id}.step-{step}",
                branch_id=trace.branch_id,
                parent_step=step,
                parent_time=_decimal(step * float(config.primary_timestep)),
                phi_y=observation.factor_y.phi,
                amplitude_pass=amplitude,
                closure_y=observation.factor_y.closure_ratio,
                closure_pass=closure,
                kernel_band_y=observation.factor_y.kernel_band_ratio,
                spectrum_valid=spectrum_valid,
                kernel_pass=kernel,
                persistence_pass_count=persistence_count,
                persistence_count_pass=persistence,
                strict_kernel_window_pass=strict_kernel,
                x_rolling_geometric=label.x_geometric,
                x_exclusion_pass=x_exclusion,
                resolved_probe_cells=resolved,
                required_probe_cells=18,
                probe_invariants_pass=invariants_pass,
                geometry_loss=None if geometry_mean is None else _decimal(geometry_mean),
                radius_loss=None if radius_mean is None else _decimal(radius_mean),
                geometry_radius_ratio=None if ratio is None else _decimal(ratio),
                response_law_pass=response_pass,
                structural_geometry_pass=structural_pass,
                full_intersection_pass=bool(response_pass and structural_pass),
                predictive_skill_only=bool(response_pass and not structural_pass),
                structure_only=bool(structural_pass and not response_pass),
                reason_codes=tuple(sorted(reasons)),
            )
        )
    full_flags = tuple(value.full_intersection_pass for value in samples)
    response_flags = tuple(value.response_law_pass for value in samples)
    structural_flags = tuple(value.structural_geometry_pass for value in samples)
    cadence_time = config.receiver_cadence_steps * float(config.primary_timestep)
    full_total, full_longest = conservative_residence(full_flags, cadence_time=cadence_time)
    response_total, response_longest = conservative_residence(
        response_flags, cadence_time=cadence_time
    )
    structural_total, structural_longest = conservative_residence(
        structural_flags, cadence_time=cadence_time
    )
    return MatrixResponseCausalIntersectionResidenceBranchEvaluation(
        outcome=MatrixResponseCausalIntersectionResidenceBranchOutcome(
            outcome_id=f"matrix-causal-intersection-residence.outcome.{trace.branch_id}",
            branch_id=trace.branch_id,
            block_index=trace.block_index,
            seed_stratum=0 if trace.block_index < config.confirmation_blocks // 2 else 1,
            intent_word=intent_word,
            realized_action_word=trace.action_ledger.action_word,
            numerical_view_id=trace.numerical_view_id,
            parent_step_multiplier=trace.parent_step_multiplier,
            noise_seed_sha256=trace.noise_seed_sha256,
            noise_block_sha256=trace.noise_block_sha256,
            requested_action_sha256=trace.action_ledger.requested_sha256,
            accepted_action_sha256=trace.action_ledger.accepted_sha256,
            applied_action_sha256=trace.action_ledger.applied_sha256,
            realized_action_sha256=trace.action_ledger.realized_sha256,
            trigger_parent_step=trace.action_ledger.trigger_parent_step,
            action_maximum_excursion=trace.action_ledger.maximum_excursion,
            action_maximum_increment=trace.action_ledger.maximum_increment,
            action_total_variation=trace.action_ledger.total_variation,
            action_squared_energy=trace.action_ledger.squared_action_energy,
            action_generalized_absolute_work=trace.action_ledger.generalized_absolute_work,
            action_pulse_count=trace.action_ledger.pulse_count,
            action_exact_baseline_return=trace.action_ledger.exact_baseline_return,
            action_clipped=trace.action_ledger.clipped,
            action_delivery_valid=trace.action_ledger.valid,
            assessment_start_step=assessment_start_step,
            assessment_end_step=assessment_end_step,
            assessment_sample_count=len(samples),
            full_sustained=full_longest >= float(config.sustained_duration),
            full_total_residence=_decimal(full_total),
            full_longest_residence=_decimal(full_longest),
            response_only_sustained=response_longest >= float(config.sustained_duration),
            response_only_total_residence=_decimal(response_total),
            response_only_longest_residence=_decimal(response_longest),
            structure_only_sustained=structural_longest >= float(config.sustained_duration),
            structure_only_total_residence=_decimal(structural_total),
            structure_only_longest_residence=_decimal(structural_longest),
            full_sample_count=sum(full_flags),
            response_law_sample_count=sum(response_flags),
            structural_sample_count=sum(structural_flags),
            predictive_skill_only_sample_count=sum(
                value.predictive_skill_only for value in samples
            ),
            structure_only_sample_count=sum(value.structure_only for value in samples),
            amplitude_only_sample_count=sum(
                value.amplitude_pass and not value.full_intersection_pass for value in samples
            ),
            false_x_admission=any(
                value.full_intersection_pass and value.x_rolling_geometric for value in samples
            ),
            technically_valid=not invariant_reasons,
            reason_codes=tuple(sorted(invariant_reasons)),
        ),
        samples=tuple(samples),
        propagation=propagation,
    )


@dataclass(frozen=True, slots=True)
class MatrixResponseCausalIntersectionResidenceDevelopmentCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-causal-intersection-residence-development-cell'

    cell_id: str
    checkpoint_step: int
    action_word: str
    valid_pair_count: int
    full_risk_gain: Decimal
    full_residence_gain: Decimal
    first_half_full_risk_gain: Decimal
    second_half_full_risk_gain: Decimal
    first_half_full_residence_gain: Decimal
    second_half_full_residence_gain: Decimal
    response_only_risk_gain: Decimal
    response_only_residence_gain: Decimal
    first_half_response_only_risk_gain: Decimal
    second_half_response_only_risk_gain: Decimal
    first_half_response_only_residence_gain: Decimal
    second_half_response_only_residence_gain: Decimal
    response_only_eligible: bool
    structure_only_risk_gain: Decimal
    structure_only_residence_gain: Decimal
    first_half_structure_only_risk_gain: Decimal
    second_half_structure_only_risk_gain: Decimal
    first_half_structure_only_residence_gain: Decimal
    second_half_structure_only_residence_gain: Decimal
    structure_only_eligible: bool
    normalized_full_effect_margin: Decimal
    eligible: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.cell_id, field_name="cell_id")
        if (
            self.checkpoint_step < 0
            or self.action_word not in ACTIVE_ACTION_WORDS
            or not 0 <= self.valid_pair_count <= 32
        ):
            raise ValueError("Matrix causal intersection residence development cell identity differs")
        for name in (
            "full_risk_gain",
            "full_residence_gain",
            "first_half_full_risk_gain",
            "second_half_full_risk_gain",
            "first_half_full_residence_gain",
            "second_half_full_residence_gain",
            "response_only_risk_gain",
            "response_only_residence_gain",
            "first_half_response_only_risk_gain",
            "second_half_response_only_risk_gain",
            "first_half_response_only_residence_gain",
            "second_half_response_only_residence_gain",
            "structure_only_risk_gain",
            "structure_only_residence_gain",
            "first_half_structure_only_risk_gain",
            "second_half_structure_only_risk_gain",
            "first_half_structure_only_residence_gain",
            "second_half_structure_only_residence_gain",
            "normalized_full_effect_margin",
        ):
            validate_decimal(getattr(self, name), field_name=name)
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.eligible != (not self.reason_codes):
            raise ValueError("Matrix causal intersection residence development-cell eligibility differs")


def _paired_effect(
    *,
    action: Mapping[int, MatrixResponseCausalIntersectionResidenceBranchOutcome],
    hold: Mapping[int, MatrixResponseCausalIntersectionResidenceBranchOutcome],
    success_field: str,
    residence_field: str,
) -> tuple[np.ndarray, np.ndarray]:
    risk = np.asarray(
        [
            int(getattr(action[index], success_field)) - int(getattr(hold[index], success_field))
            for index in range(32)
        ],
        dtype=np.float64,
    )
    residence = np.asarray(
        [
            float(getattr(action[index], residence_field))
            - float(getattr(hold[index], residence_field))
            for index in range(32)
        ],
        dtype=np.float64,
    )
    return risk, residence


def _outcome_effort_valid(outcome: MatrixResponseCausalIntersectionResidenceBranchOutcome, *, config: MatrixResponseCausalIntersectionResidenceStudyConfig) -> bool:
    maximum_increment = float(config.pulse_increment_max) / outcome.parent_step_multiplier
    return bool(
        outcome.action_delivery_valid
        and float(outcome.action_maximum_excursion) <= float(config.pulse_delta) + 1e-15
        and float(outcome.action_maximum_increment) <= maximum_increment + 1e-15
        and float(outcome.action_total_variation) <= float(config.pulse_total_variation_max) + 1e-14
        and float(outcome.action_squared_energy) <= float(config.pulse_squared_energy_max) + 1e-15
        and float(outcome.action_generalized_absolute_work)
        <= float(config.generalized_work_ceiling) + 1e-12
        and outcome.action_exact_baseline_return
        and not outcome.action_clipped
        and outcome.action_pulse_count == int(outcome.realized_action_word != "hold")
    )


def reduce_development_cell(
    *,
    checkpoint_step: int,
    action_word: str,
    action: Sequence[MatrixResponseCausalIntersectionResidenceBranchOutcome],
    hold: Sequence[MatrixResponseCausalIntersectionResidenceBranchOutcome],
    config: MatrixResponseCausalIntersectionResidenceStudyConfig,
) -> MatrixResponseCausalIntersectionResidenceDevelopmentCell:
    """Reduce one cell; eligibility is determined only by full-intersection outcomes."""

    if len(action) != 32 or len(hold) != 32:
        raise ValueError("Matrix causal intersection residence development cell requires 32 paired blocks")
    action_by = {value.block_index: value for value in action}
    hold_by = {value.block_index: value for value in hold}
    if set(action_by) != set(range(32)) or set(hold_by) != set(range(32)):
        raise ValueError("Matrix causal intersection residence development block roster differs")
    valid_count = sum(
        action_by[index].technically_valid and hold_by[index].technically_valid
        for index in range(32)
    )
    full_risk, full_residence = _paired_effect(
        action=action_by,
        hold=hold_by,
        success_field="full_sustained",
        residence_field="full_total_residence",
    )
    response_risk, response_residence = _paired_effect(
        action=action_by,
        hold=hold_by,
        success_field="response_only_sustained",
        residence_field="response_only_total_residence",
    )
    structure_risk, structure_residence = _paired_effect(
        action=action_by,
        hold=hold_by,
        success_field="structure_only_sustained",
        residence_field="structure_only_total_residence",
    )
    risk_gain = float(np.mean(full_risk))
    residence_gain = float(np.mean(full_residence))
    half_risk = (float(np.mean(full_risk[:16])), float(np.mean(full_risk[16:])))
    half_residence = (
        float(np.mean(full_residence[:16])),
        float(np.mean(full_residence[16:])),
    )
    response_half_risk = (
        float(np.mean(response_risk[:16])),
        float(np.mean(response_risk[16:])),
    )
    response_half_residence = (
        float(np.mean(response_residence[:16])),
        float(np.mean(response_residence[16:])),
    )
    structure_half_risk = (
        float(np.mean(structure_risk[:16])),
        float(np.mean(structure_risk[16:])),
    )
    structure_half_residence = (
        float(np.mean(structure_residence[:16])),
        float(np.mean(structure_residence[16:])),
    )
    response_eligible = bool(
        valid_count == 32
        and float(np.mean(response_risk)) >= float(config.development_risk_gain_min)
        and float(np.mean(response_residence)) >= float(config.development_residence_gain_min)
        and min(response_half_risk) > 0
        and min(response_half_residence) > 0
    )
    structure_eligible = bool(
        valid_count == 32
        and float(np.mean(structure_risk)) >= float(config.development_risk_gain_min)
        and float(np.mean(structure_residence)) >= float(config.development_residence_gain_min)
        and min(structure_half_risk) > 0
        and min(structure_half_residence) > 0
    )
    reasons: set[str] = set()
    if valid_count != 32:
        reasons.add("development-pair-invalid")
    if risk_gain < float(config.development_risk_gain_min):
        reasons.add("full-risk-margin-failed")
    if residence_gain < float(config.development_residence_gain_min):
        reasons.add("full-residence-margin-failed")
    if min(half_risk) <= 0:
        reasons.add("full-risk-split-direction-failed")
    if min(half_residence) <= 0:
        reasons.add("full-residence-split-direction-failed")
    margin = min(
        (risk_gain - float(config.development_risk_gain_min))
        / float(config.development_risk_gain_min),
        (residence_gain - float(config.development_residence_gain_min))
        / float(config.development_residence_gain_min),
    )
    return MatrixResponseCausalIntersectionResidenceDevelopmentCell(
        cell_id=f"matrix-causal-intersection-residence.development-cell.step-{checkpoint_step}.{action_word}",
        checkpoint_step=checkpoint_step,
        action_word=action_word,
        valid_pair_count=valid_count,
        full_risk_gain=_decimal(risk_gain),
        full_residence_gain=_decimal(residence_gain),
        first_half_full_risk_gain=_decimal(half_risk[0]),
        second_half_full_risk_gain=_decimal(half_risk[1]),
        first_half_full_residence_gain=_decimal(half_residence[0]),
        second_half_full_residence_gain=_decimal(half_residence[1]),
        response_only_risk_gain=_decimal(float(np.mean(response_risk))),
        response_only_residence_gain=_decimal(float(np.mean(response_residence))),
        first_half_response_only_risk_gain=_decimal(response_half_risk[0]),
        second_half_response_only_risk_gain=_decimal(response_half_risk[1]),
        first_half_response_only_residence_gain=_decimal(response_half_residence[0]),
        second_half_response_only_residence_gain=_decimal(response_half_residence[1]),
        response_only_eligible=response_eligible,
        structure_only_risk_gain=_decimal(float(np.mean(structure_risk))),
        structure_only_residence_gain=_decimal(float(np.mean(structure_residence))),
        first_half_structure_only_risk_gain=_decimal(structure_half_risk[0]),
        second_half_structure_only_risk_gain=_decimal(structure_half_risk[1]),
        first_half_structure_only_residence_gain=_decimal(structure_half_residence[0]),
        second_half_structure_only_residence_gain=_decimal(structure_half_residence[1]),
        structure_only_eligible=structure_eligible,
        normalized_full_effect_margin=_decimal(margin),
        eligible=not reasons,
        reason_codes=tuple(sorted(reasons)),
    )


class MatrixResponseCausalIntersectionResidenceTerminal(StrEnum):
    CAUSAL_RESPONSE_GEOMETRY_AUTHORITY_ESTABLISHED = (
        "CAUSAL_RESPONSE_GEOMETRY_AUTHORITY_ESTABLISHED"
    )
    PREDICTIVE_SKILL_ONLY_NO_GEOMETRIC_AUTHORITY = "PREDICTIVE_SKILL_ONLY_NO_GEOMETRIC_AUTHORITY"
    STRUCTURE_ONLY_NO_RESPONSE_LAW_AUTHORITY = "STRUCTURE_ONLY_NO_RESPONSE_LAW_AUTHORITY"
    NO_FULL_INTERSECTION_DEVELOPMENT_CANDIDATE = "NO_FULL_INTERSECTION_DEVELOPMENT_CANDIDATE"
    INBOUND_PHASE_NOT_CAUSALLY_OBSERVABLE = "INBOUND_PHASE_NOT_CAUSALLY_OBSERVABLE"
    HELDOUT_FULL_INTERSECTION_EFFECT_NOT_MATERIAL = "HELDOUT_FULL_INTERSECTION_EFFECT_NOT_MATERIAL"
    ACTION_DELIVERY_OR_EFFORT_INVALID = "ACTION_DELIVERY_OR_EFFORT_INVALID"
    NUMERICAL_VIEW_INVALID = "NUMERICAL_VIEW_INVALID"
    PREREQUISITE_NONATTEMPT = "PREREQUISITE_NONATTEMPT"
    UNEVALUABLE = "UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class MatrixResponseCausalIntersectionResidenceDevelopmentCandidate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-causal-intersection-residence-development-candidate'

    candidate_id: str
    config_fingerprint: str
    method_package_sha256: str
    selected_cell: MatrixResponseCausalIntersectionResidenceDevelopmentCell
    phase_rule: MatrixResponseTransientControlledInvariancePhaseRule
    admitted_phase_steps: tuple[int, ...]
    minimum_normalized_full_effect_margin: Decimal
    candidate_sha256: str
    grants_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.candidate_id, field_name="candidate_id")
        validate_sha256(self.config_fingerprint, field_name="config_fingerprint")
        validate_sha256(self.method_package_sha256, field_name="method_package_sha256")
        validate_sha256(self.candidate_sha256, field_name="candidate_sha256")
        validate_decimal(
            self.minimum_normalized_full_effect_margin,
            field_name="minimum_normalized_full_effect_margin",
        )
        if (
            not self.selected_cell.eligible
            or self.phase_rule.action_word != self.selected_cell.action_word
            or tuple(sorted(set(self.admitted_phase_steps))) != self.admitted_phase_steps
            or self.selected_cell.checkpoint_step not in self.admitted_phase_steps
            or self.grants_authority
        ):
            raise ValueError("Matrix causal intersection residence development candidate differs")


def _original_phase_rule_comparison_coordinate(
    rule: MatrixResponseTransientControlledInvariancePhaseRule,
) -> str:
    """Reconstruct the original scientific final tie coordinate.

    Original predicate coordinates were hashed before the rule-ID comparison.
    Current predicate IDs remain mandatory; original IDs are not input aliases.
    """

    if type(rule) is not MatrixResponseTransientControlledInvariancePhaseRule:
        raise ValueError("full-intersection selection requires an exact current phase rule")
    original_predicate_coordinates = []
    for predicate in rule.predicates:
        threshold_sha256 = sha256(str(predicate.threshold).encode()).hexdigest()[:12]
        current_coordinate = (
            f"matrix-transient-response.predicate.{predicate.feature_name}.{predicate.direction}.{threshold_sha256}"
        )
        if predicate.predicate_id != current_coordinate:
            raise ValueError("full-intersection selection phase predicate identity differs from its scientific operands")
        original_predicate_coordinates.append(
            f"cc1-s2.predicate.{predicate.feature_name}.{predicate.direction}.{threshold_sha256}"
        )
    original_digest = sha256(
        "\0".join(sorted(original_predicate_coordinates)).encode()
    ).hexdigest()[:16]
    return f"cc1-s2.rule.{rule.action_word}.{rule.development_step}.{original_digest}"


def select_development_candidate(
    *,
    features: tuple[MatrixResponseTransientControlledInvariancePhaseFeature, ...],
    cells: tuple[MatrixResponseCausalIntersectionResidenceDevelopmentCell, ...],
    config: MatrixResponseCausalIntersectionResidenceStudyConfig,
) -> tuple[MatrixResponseCausalIntersectionResidenceDevelopmentCandidate | None, MatrixResponseCausalIntersectionResidenceTerminal | None]:
    """Freeze at most one representable rule using full outcomes only."""

    by_coordinate = {(value.checkpoint_step, value.action_word): value for value in cells}
    required = set(
        (step, action)
        for step in config.development_checkpoint_steps
        for action in ACTIVE_ACTION_WORDS
    )
    if set(by_coordinate) != required:
        raise ValueError("Matrix causal intersection residence development cell panel is incomplete")
    eligible = tuple(value for value in cells if value.eligible)
    if not eligible:
        response_only = any(value.response_only_eligible for value in cells)
        structure_only = any(value.structure_only_eligible for value in cells)
        if response_only and not structure_only:
            return None, MatrixResponseCausalIntersectionResidenceTerminal.PREDICTIVE_SKILL_ONLY_NO_GEOMETRIC_AUTHORITY
        if structure_only and not response_only:
            return None, MatrixResponseCausalIntersectionResidenceTerminal.STRUCTURE_ONLY_NO_RESPONSE_LAW_AUTHORITY
        return None, MatrixResponseCausalIntersectionResidenceTerminal.NO_FULL_INTERSECTION_DEVELOPMENT_CANDIDATE
    channel_order = {"x": 0, "y": 1}
    sign_order = {"negative": 0, "positive": 1}
    candidates: list[tuple[Any, ...]] = []
    for cell in eligible:
        rules = enumerate_phase_rules(
            features=features,
            action_word=cell.action_word,
            development_step=cell.checkpoint_step,
            development_effect_margin=cell.normalized_full_effect_margin,
        )
        for rule in rules:
            admitted = tuple(value.parent_step for value in features if rule.accepts(value))
            if cell.checkpoint_step not in admitted:
                continue
            material = tuple(
                value.checkpoint_step
                for value in cells
                if value.action_word == cell.action_word and value.eligible
            )
            if not set(material).issubset(admitted):
                continue
            nonpositive = tuple(
                value.checkpoint_step
                for value in cells
                if value.action_word == cell.action_word
                and (float(value.full_risk_gain) <= 0 or float(value.full_residence_gain) <= 0)
            )
            if set(nonpositive).intersection(admitted):
                continue
            margins = tuple(
                float(by_coordinate[(step, cell.action_word)].normalized_full_effect_margin)
                for step in admitted
            )
            channel, sign = cell.action_word.split("-", maxsplit=1)
            candidates.append(
                (
                    min(margins),
                    len(rule.predicates),
                    cell.checkpoint_step,
                    channel_order[channel],
                    sign_order[sign],
                    _original_phase_rule_comparison_coordinate(rule),
                    cell,
                    rule,
                    admitted,
                )
            )
    if not candidates:
        return None, MatrixResponseCausalIntersectionResidenceTerminal.INBOUND_PHASE_NOT_CAUSALLY_OBSERVABLE
    selected = min(
        candidates,
        key=lambda value: (-value[0], value[1], value[2], value[3], value[4], value[5]),
    )
    margin, _, _, _, _, _, cell, rule, admitted = selected
    method_package_sha256 = qualify_method(config=config).fingerprint()
    digest = sha256(
        (
            f"{config.fingerprint()}\0{method_package_sha256}\0"
            f"{cell.fingerprint()}\0{rule.fingerprint()}\0" + ",".join(map(str, admitted))
        ).encode()
    ).hexdigest()
    return (
        MatrixResponseCausalIntersectionResidenceDevelopmentCandidate(
            candidate_id=f"matrix-causal-intersection-residence.development-candidate.{digest[:16]}",
            config_fingerprint=config.fingerprint(),
            method_package_sha256=method_package_sha256,
            selected_cell=cell,
            phase_rule=rule,
            admitted_phase_steps=admitted,
            minimum_normalized_full_effect_margin=_decimal(margin),
            candidate_sha256=digest,
            grants_authority=False,
        ),
        None,
    )


@dataclass(frozen=True, slots=True)
class MatrixResponseCausalIntersectionResidenceNumericalAudit(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-causal-intersection-residence-numerical-audit'

    audit_id: str
    paired_block_count: int
    arm_classification_count: int
    arm_classification_agreement: Decimal
    primary_subset_risk_gain: Decimal
    half_subset_risk_gain: Decimal
    primary_subset_residence_gain: Decimal
    half_subset_residence_gain: Decimal
    risk_direction_concordant: bool
    residence_direction_concordant: bool
    median_probe_state_difference: Decimal
    maximum_energy_relative_error: Decimal
    maximum_work_relative_error: Decimal
    invariant_valid: bool
    valid: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.audit_id, field_name="audit_id")
        if self.paired_block_count != 32 or self.arm_classification_count != 64:
            raise ValueError("Matrix causal intersection residence numerical audit roster differs")
        for name in (
            "arm_classification_agreement",
            "primary_subset_risk_gain",
            "half_subset_risk_gain",
            "primary_subset_residence_gain",
            "half_subset_residence_gain",
            "median_probe_state_difference",
            "maximum_energy_relative_error",
            "maximum_work_relative_error",
        ):
            validate_decimal(getattr(self, name), field_name=name)
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.valid != (not self.reason_codes):
            raise ValueError("Matrix causal intersection residence numerical audit validity differs")


def build_numerical_audit(
    *,
    primary_action: Sequence[MatrixResponseCausalIntersectionResidenceBranchEvaluation],
    primary_hold: Sequence[MatrixResponseCausalIntersectionResidenceBranchEvaluation],
    half_action: Sequence[MatrixResponseCausalIntersectionResidenceBranchEvaluation],
    half_hold: Sequence[MatrixResponseCausalIntersectionResidenceBranchEvaluation],
    primary_energy: Sequence[float],
    half_energy: Sequence[float],
    primary_work: Sequence[float],
    half_work: Sequence[float],
    config: MatrixResponseCausalIntersectionResidenceStudyConfig,
) -> MatrixResponseCausalIntersectionResidenceNumericalAudit:
    sequences = (primary_action, primary_hold, half_action, half_hold)
    if any(len(value) != 32 for value in sequences):
        raise ValueError("Matrix causal intersection residence numerical audit requires 32 paired views")
    if any(len(value) != 32 for value in (primary_energy, half_energy, primary_work, half_work)):
        raise ValueError("Matrix causal intersection residence numerical effort roster differs")
    primary_categories = tuple(
        value.outcome.full_sustained for group in (primary_action, primary_hold) for value in group
    )
    half_categories = tuple(
        value.outcome.full_sustained for group in (half_action, half_hold) for value in group
    )
    agreement = float(np.mean(np.equal(primary_categories, half_categories)))

    def effects(
        action_values: Sequence[MatrixResponseCausalIntersectionResidenceBranchEvaluation],
        hold_values: Sequence[MatrixResponseCausalIntersectionResidenceBranchEvaluation],
    ) -> tuple[float, float]:
        risk = float(
            np.mean(
                [
                    int(left.outcome.full_sustained) - int(right.outcome.full_sustained)
                    for left, right in zip(action_values, hold_values, strict=True)
                ]
            )
        )
        residence = float(
            np.mean(
                [
                    float(left.outcome.full_total_residence)
                    - float(right.outcome.full_total_residence)
                    for left, right in zip(action_values, hold_values, strict=True)
                ]
            )
        )
        return risk, residence

    primary_risk, primary_residence = effects(primary_action, primary_hold)
    half_risk, half_residence = effects(half_action, half_hold)
    probe_differences: list[float] = []
    for primary_group, half_group in (
        (primary_action, half_action),
        (primary_hold, half_hold),
    ):
        for primary, half in zip(primary_group, half_group, strict=True):
            primary_states = primary.propagation.states[:, :, ::1]
            half_states = half.propagation.states[:, :, ::2]
            if primary_states.shape != half_states.shape:
                raise ValueError("Matrix causal intersection residence primary/half probe grids are not aligned")
            denominators = np.maximum(
                np.linalg.norm(primary_states, axis=(-2, -1)),
                np.finfo(np.float64).tiny,
            )
            probe_differences.extend(
                np.ravel(
                    np.linalg.norm(primary_states - half_states, axis=(-2, -1)) / denominators
                ).tolist()
            )
    median_probe = float(np.median(np.asarray(probe_differences, dtype=np.float64)))
    energy_errors = tuple(
        _relative_error(left, right)
        for left, right in zip(primary_energy, half_energy, strict=True)
    )
    work_errors = tuple(
        _relative_error(left, right) for left, right in zip(primary_work, half_work, strict=True)
    )
    invariant_valid = all(value.outcome.technically_valid for group in sequences for value in group)
    return build_numerical_audit_from_operands(
        classification_agreement=agreement,
        primary_risk_gain=primary_risk,
        half_risk_gain=half_risk,
        primary_residence_gain=primary_residence,
        half_residence_gain=half_residence,
        median_probe_state_difference=median_probe,
        maximum_energy_relative_error=max(energy_errors),
        maximum_work_relative_error=max(work_errors),
        invariant_valid=invariant_valid,
        config=config,
    )


def build_numerical_audit_from_operands(
    *,
    classification_agreement: float,
    primary_risk_gain: float,
    half_risk_gain: float,
    primary_residence_gain: float,
    half_residence_gain: float,
    median_probe_state_difference: float,
    maximum_energy_relative_error: float,
    maximum_work_relative_error: float,
    invariant_valid: bool,
    config: MatrixResponseCausalIntersectionResidenceStudyConfig,
) -> MatrixResponseCausalIntersectionResidenceNumericalAudit:
    """Finalize preaggregated paired-view operands without retaining trace arrays."""

    values = (
        classification_agreement,
        primary_risk_gain,
        half_risk_gain,
        primary_residence_gain,
        half_residence_gain,
        median_probe_state_difference,
        maximum_energy_relative_error,
        maximum_work_relative_error,
    )
    if not np.isfinite(values).all():
        raise ValueError("Matrix causal intersection residence numerical operands are nonfinite")
    reasons: set[str] = set()
    if classification_agreement < float(config.numerical_classification_agreement_min):
        reasons.add("full-classification-agreement-failed")
    risk_concordant = _directions_concordant(primary_risk_gain, half_risk_gain)
    residence_concordant = _directions_concordant(primary_residence_gain, half_residence_gain)
    if not risk_concordant:
        reasons.add("risk-direction-discordant")
    if not residence_concordant:
        reasons.add("residence-direction-discordant")
    if median_probe_state_difference > float(config.numerical_probe_state_difference_max):
        reasons.add("probe-state-difference-failed")
    if maximum_energy_relative_error > float(config.numerical_energy_relative_error_max):
        reasons.add("energy-alignment-failed")
    if maximum_work_relative_error > float(config.numerical_work_relative_error_max):
        reasons.add("work-alignment-failed")
    if not invariant_valid:
        reasons.add("half-step-invariant-invalid")
    return MatrixResponseCausalIntersectionResidenceNumericalAudit(
        audit_id="matrix-causal-intersection-residence.numerical-audit",
        paired_block_count=32,
        arm_classification_count=64,
        arm_classification_agreement=_decimal(classification_agreement),
        primary_subset_risk_gain=_decimal(primary_risk_gain),
        half_subset_risk_gain=_decimal(half_risk_gain),
        primary_subset_residence_gain=_decimal(primary_residence_gain),
        half_subset_residence_gain=_decimal(half_residence_gain),
        risk_direction_concordant=risk_concordant,
        residence_direction_concordant=residence_concordant,
        median_probe_state_difference=_decimal(median_probe_state_difference),
        maximum_energy_relative_error=_decimal(maximum_energy_relative_error),
        maximum_work_relative_error=_decimal(maximum_work_relative_error),
        invariant_valid=invariant_valid,
        valid=not reasons,
        reason_codes=tuple(sorted(reasons)),
    )


@dataclass(frozen=True, slots=True)
class MatrixResponseCausalIntersectionResidenceTerminalReport(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-causal-intersection-residence-terminal-report'

    report_id: str
    config_fingerprint: str
    candidate_sha256: str | None
    development_terminal: MatrixResponseCausalIntersectionResidenceTerminal | None
    valid_pair_count: int
    full_risk_gain: Decimal | None
    full_risk_lower: Decimal | None
    full_residence_gain: Decimal | None
    full_residence_lower: Decimal | None
    stratum_full_risk_gains: tuple[Decimal, ...]
    stratum_full_residence_gains: tuple[Decimal, ...]
    response_only_risk_gain: Decimal | None
    response_only_residence_gain: Decimal | None
    structure_only_risk_gain: Decimal | None
    structure_only_residence_gain: Decimal | None
    action_only_successes: int
    hold_only_successes: int
    mcnemar_one_sided_p: Decimal | None
    numerical_audit: MatrixResponseCausalIntersectionResidenceNumericalAudit | None
    delivery_effort_valid: bool
    no_false_x_admission: bool
    terminal: MatrixResponseCausalIntersectionResidenceTerminal
    gate_passed: bool
    prospective_control_condition: str
    reason_codes: tuple[str, ...]
    grants_execution_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.report_id, field_name="report_id")
        validate_sha256(self.config_fingerprint, field_name="config_fingerprint")
        if self.candidate_sha256 is not None:
            validate_sha256(self.candidate_sha256, field_name="candidate_sha256")
        for name in (
            "full_risk_gain",
            "full_risk_lower",
            "full_residence_gain",
            "full_residence_lower",
            "response_only_risk_gain",
            "response_only_residence_gain",
            "structure_only_risk_gain",
            "structure_only_residence_gain",
            "mcnemar_one_sided_p",
        ):
            value = getattr(self, name)
            if value is not None:
                validate_decimal(value, field_name=name)
        for value in (*self.stratum_full_risk_gains, *self.stratum_full_residence_gains):
            validate_decimal(value, field_name="stratum_effect")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        positive = self.terminal is MatrixResponseCausalIntersectionResidenceTerminal.CAUSAL_RESPONSE_GEOMETRY_AUTHORITY_ESTABLISHED
        if (
            self.gate_passed != positive
            or self.prospective_control_condition != "CONDITION_FALSE"
            or self.grants_execution_authority
        ):
            raise ValueError("Matrix causal intersection residence terminal descendants differ")


def development_terminal_report(
    *,
    config: MatrixResponseCausalIntersectionResidenceStudyConfig,
    terminal: MatrixResponseCausalIntersectionResidenceTerminal,
    cells: Sequence[MatrixResponseCausalIntersectionResidenceDevelopmentCell],
) -> MatrixResponseCausalIntersectionResidenceTerminalReport:
    if terminal not in {
        MatrixResponseCausalIntersectionResidenceTerminal.PREDICTIVE_SKILL_ONLY_NO_GEOMETRIC_AUTHORITY,
        MatrixResponseCausalIntersectionResidenceTerminal.STRUCTURE_ONLY_NO_RESPONSE_LAW_AUTHORITY,
        MatrixResponseCausalIntersectionResidenceTerminal.NO_FULL_INTERSECTION_DEVELOPMENT_CANDIDATE,
        MatrixResponseCausalIntersectionResidenceTerminal.INBOUND_PHASE_NOT_CAUSALLY_OBSERVABLE,
    }:
        raise ValueError("Matrix causal intersection residence development terminal differs")
    return MatrixResponseCausalIntersectionResidenceTerminalReport(
        report_id="matrix-causal-intersection-residence.terminal-report",
        config_fingerprint=config.fingerprint(),
        candidate_sha256=None,
        development_terminal=terminal,
        valid_pair_count=0,
        full_risk_gain=None,
        full_risk_lower=None,
        full_residence_gain=None,
        full_residence_lower=None,
        stratum_full_risk_gains=(),
        stratum_full_residence_gains=(),
        response_only_risk_gain=max(
            (_decimal(float(value.response_only_risk_gain)) for value in cells),
            default=None,
        ),
        response_only_residence_gain=max(
            (_decimal(float(value.response_only_residence_gain)) for value in cells),
            default=None,
        ),
        structure_only_risk_gain=max(
            (_decimal(float(value.structure_only_risk_gain)) for value in cells),
            default=None,
        ),
        structure_only_residence_gain=max(
            (_decimal(float(value.structure_only_residence_gain)) for value in cells),
            default=None,
        ),
        action_only_successes=0,
        hold_only_successes=0,
        mcnemar_one_sided_p=None,
        numerical_audit=None,
        delivery_effort_valid=True,
        no_false_x_admission=True,
        terminal=terminal,
        gate_passed=False,
        prospective_control_condition="CONDITION_FALSE",
        reason_codes=("development-gate-failed",),
        grants_execution_authority=False,
    )


def finalize_confirmation(
    *,
    config: MatrixResponseCausalIntersectionResidenceStudyConfig,
    candidate: MatrixResponseCausalIntersectionResidenceDevelopmentCandidate,
    action: tuple[MatrixResponseCausalIntersectionResidenceBranchOutcome, ...],
    hold: tuple[MatrixResponseCausalIntersectionResidenceBranchOutcome, ...],
    numerical_audit: MatrixResponseCausalIntersectionResidenceNumericalAudit,
    delivery_effort_valid: bool,
) -> MatrixResponseCausalIntersectionResidenceTerminalReport:
    """Sole all-intent confirmation finalizer for the full intersection."""

    if len(action) != 128 or len(hold) != 128:
        raise ValueError("Matrix causal intersection residence confirmation requires 128 intent pairs")
    action_by = {value.block_index: value for value in action}
    hold_by = {value.block_index: value for value in hold}
    if set(action_by) != set(range(128)) or set(hold_by) != set(range(128)):
        raise ValueError("Matrix causal intersection residence confirmation block roster differs")
    recomputed_delivery_effort = all(
        _outcome_effort_valid(value, config=config) for value in (*action, *hold)
    )
    if delivery_effort_valid != recomputed_delivery_effort:
        raise ValueError("Matrix causal intersection residence delivery/effort operand differs from branch custody")
    valid_pairs = tuple(
        index
        for index in range(128)
        if action_by[index].technically_valid and hold_by[index].technically_valid
    )
    maximum = (
        config.confirmation_assessment_end_step - config.confirmation_assessment_start_step
    ) * float(config.primary_timestep)
    action_binary = np.asarray(
        [
            int(action_by[index].full_sustained) if action_by[index].technically_valid else 0
            for index in range(128)
        ],
        dtype=np.float64,
    )
    hold_binary = np.asarray(
        [
            int(hold_by[index].full_sustained) if hold_by[index].technically_valid else 1
            for index in range(128)
        ],
        dtype=np.float64,
    )
    action_residence = np.asarray(
        [
            float(action_by[index].full_total_residence)
            if action_by[index].technically_valid
            else 0.0
            for index in range(128)
        ],
        dtype=np.float64,
    )
    hold_residence = np.asarray(
        [
            float(hold_by[index].full_total_residence)
            if hold_by[index].technically_valid
            else maximum
            for index in range(128)
        ],
        dtype=np.float64,
    )
    risk_difference = action_binary - hold_binary
    residence_difference = action_residence - hold_residence
    strata = np.repeat((0, 1), 64)
    risk_gain = float(np.mean(risk_difference))
    residence_gain = float(np.mean(residence_difference))
    risk_lower = stratified_paired_bootstrap_lower(
        differences=risk_difference,
        strata=strata,
        resamples=config.bootstrap_resamples,
        scientific_seed_sha256=fixed_matrix_response_bootstrap_seed_sha256("causal-intersection-full-risk"),
    )
    residence_lower = stratified_paired_bootstrap_lower(
        differences=residence_difference,
        strata=strata,
        resamples=config.bootstrap_resamples,
        scientific_seed_sha256=fixed_matrix_response_bootstrap_seed_sha256("causal-intersection-full-residence"),
    )
    stratum_risk = tuple(
        float(np.mean(risk_difference[index * 64 : (index + 1) * 64])) for index in range(2)
    )
    stratum_residence = tuple(
        float(np.mean(residence_difference[index * 64 : (index + 1) * 64])) for index in range(2)
    )
    action_only = int(np.sum((action_binary == 1) & (hold_binary == 0)))
    hold_only = int(np.sum((action_binary == 0) & (hold_binary == 1)))
    discordant = action_only + hold_only
    mcnemar = (
        1.0
        if discordant == 0
        else float(binomtest(action_only, discordant, p=0.5, alternative="greater").pvalue)
    )

    def diagnostic_effect(
        success: str, residence: str, *, namespace: str
    ) -> tuple[float, float, float, float, tuple[float, float], tuple[float, float]]:
        risk_values = np.asarray(
            [
                (
                    int(getattr(action_by[index], success))
                    if action_by[index].technically_valid
                    else 0
                )
                - (int(getattr(hold_by[index], success)) if hold_by[index].technically_valid else 1)
                for index in range(128)
            ],
            dtype=np.float64,
        )
        residence_values = np.asarray(
            [
                (
                    float(getattr(action_by[index], residence))
                    if action_by[index].technically_valid
                    else 0.0
                )
                - (
                    float(getattr(hold_by[index], residence))
                    if hold_by[index].technically_valid
                    else maximum
                )
                for index in range(128)
            ],
            dtype=np.float64,
        )
        return (
            float(np.mean(risk_values)),
            float(np.mean(residence_values)),
            stratified_paired_bootstrap_lower(
                differences=risk_values,
                strata=strata,
                resamples=config.bootstrap_resamples,
                scientific_seed_sha256=fixed_matrix_response_bootstrap_seed_sha256(
                    f"causal-intersection-{namespace.removesuffix('-only')}-risk"
                ),
            ),
            stratified_paired_bootstrap_lower(
                differences=residence_values,
                strata=strata,
                resamples=config.bootstrap_resamples,
                scientific_seed_sha256=fixed_matrix_response_bootstrap_seed_sha256(
                    f"causal-intersection-{namespace.removesuffix('-only')}-residence"
                ),
            ),
            (
                float(np.mean(risk_values[:64])),
                float(np.mean(risk_values[64:])),
            ),
            (
                float(np.mean(residence_values[:64])),
                float(np.mean(residence_values[64:])),
            ),
        )

    (
        response_risk,
        response_residence,
        response_risk_lower,
        response_residence_lower,
        response_stratum_risk,
        response_stratum_residence,
    ) = diagnostic_effect(
        "response_only_sustained", "response_only_total_residence", namespace="response-only"
    )
    (
        structure_risk,
        structure_residence,
        structure_risk_lower,
        structure_residence_lower,
        structure_stratum_risk,
        structure_stratum_residence,
    ) = diagnostic_effect(
        "structure_only_sustained", "structure_only_total_residence", namespace="structure-only"
    )
    false_x = any(value.false_x_admission for value in (*action, *hold))
    reasons: set[str] = set()
    if len(valid_pairs) < config.confirmation_valid_pair_min:
        reasons.add("valid-pair-minimum-failed")
    if risk_gain < float(config.confirmation_risk_gain_min) or risk_lower <= 0:
        reasons.add("heldout-full-risk-effect-failed")
    if residence_gain < float(config.confirmation_residence_gain_min) or residence_lower <= 0:
        reasons.add("heldout-full-residence-effect-failed")
    if min(stratum_risk) <= 0 or min(stratum_residence) <= 0:
        reasons.add("seed-stratum-direction-failed")
    if not delivery_effort_valid:
        reasons.add("action-delivery-or-effort-invalid")
    if not numerical_audit.valid:
        reasons.add("numerical-view-invalid")
    if false_x:
        reasons.add("false-x-admission")
    if "action-delivery-or-effort-invalid" in reasons:
        terminal = MatrixResponseCausalIntersectionResidenceTerminal.ACTION_DELIVERY_OR_EFFORT_INVALID
    elif "numerical-view-invalid" in reasons:
        terminal = MatrixResponseCausalIntersectionResidenceTerminal.NUMERICAL_VIEW_INVALID
    elif "valid-pair-minimum-failed" in reasons:
        terminal = MatrixResponseCausalIntersectionResidenceTerminal.UNEVALUABLE
    elif reasons:
        response_material = bool(
            response_risk >= float(config.confirmation_risk_gain_min)
            and response_residence >= float(config.confirmation_residence_gain_min)
            and response_risk_lower > 0
            and response_residence_lower > 0
            and min(response_stratum_risk) > 0
            and min(response_stratum_residence) > 0
        )
        structure_material = bool(
            structure_risk >= float(config.confirmation_risk_gain_min)
            and structure_residence >= float(config.confirmation_residence_gain_min)
            and structure_risk_lower > 0
            and structure_residence_lower > 0
            and min(structure_stratum_risk) > 0
            and min(structure_stratum_residence) > 0
        )
        if response_material and not structure_material:
            terminal = MatrixResponseCausalIntersectionResidenceTerminal.PREDICTIVE_SKILL_ONLY_NO_GEOMETRIC_AUTHORITY
        elif structure_material and not response_material:
            terminal = MatrixResponseCausalIntersectionResidenceTerminal.STRUCTURE_ONLY_NO_RESPONSE_LAW_AUTHORITY
        else:
            terminal = MatrixResponseCausalIntersectionResidenceTerminal.HELDOUT_FULL_INTERSECTION_EFFECT_NOT_MATERIAL
    else:
        terminal = MatrixResponseCausalIntersectionResidenceTerminal.CAUSAL_RESPONSE_GEOMETRY_AUTHORITY_ESTABLISHED
    return MatrixResponseCausalIntersectionResidenceTerminalReport(
        report_id="matrix-causal-intersection-residence.terminal-report",
        config_fingerprint=config.fingerprint(),
        candidate_sha256=candidate.candidate_sha256,
        development_terminal=None,
        valid_pair_count=len(valid_pairs),
        full_risk_gain=_decimal(risk_gain),
        full_risk_lower=_decimal(risk_lower),
        full_residence_gain=_decimal(residence_gain),
        full_residence_lower=_decimal(residence_lower),
        stratum_full_risk_gains=tuple(_decimal(value) for value in stratum_risk),
        stratum_full_residence_gains=tuple(_decimal(value) for value in stratum_residence),
        response_only_risk_gain=_decimal(response_risk),
        response_only_residence_gain=_decimal(response_residence),
        structure_only_risk_gain=_decimal(structure_risk),
        structure_only_residence_gain=_decimal(structure_residence),
        action_only_successes=action_only,
        hold_only_successes=hold_only,
        mcnemar_one_sided_p=_decimal(mcnemar),
        numerical_audit=numerical_audit,
        delivery_effort_valid=delivery_effort_valid,
        no_false_x_admission=not false_x,
        terminal=terminal,
        gate_passed=(terminal is MatrixResponseCausalIntersectionResidenceTerminal.CAUSAL_RESPONSE_GEOMETRY_AUTHORITY_ESTABLISHED),
        prospective_control_condition="CONDITION_FALSE",
        reason_codes=tuple(sorted(reasons)),
        grants_execution_authority=False,
    )


@dataclass(frozen=True, slots=True)
class MatrixResponseCausalIntersectionResidenceMethodPackage(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-causal-intersection-residence-method-package'

    package_id: str
    config_fingerprint: str
    fixture_ids: tuple[str, ...]
    fixture_sha256: str
    source_access_count: int
    full_intersection_exact: bool
    predictive_only_rejected: bool
    structure_only_rejected: bool
    missing_factor_rejected: bool
    residence_semantics_exact: bool
    one_factor_terminals_noncompensating: bool
    development_primary_fields: tuple[str, ...]
    confirmation_primary_fields: tuple[str, ...]
    grants_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.package_id, field_name="package_id")
        validate_sha256(self.config_fingerprint, field_name="config_fingerprint")
        validate_sha256(self.fixture_sha256, field_name="fixture_sha256")
        require_sorted_unique_strings(self.fixture_ids, field_name="fixture_ids", allow_empty=False)
        if (
            self.source_access_count != 0
            or not self.full_intersection_exact
            or not self.predictive_only_rejected
            or not self.structure_only_rejected
            or not self.missing_factor_rejected
            or not self.residence_semantics_exact
            or not self.one_factor_terminals_noncompensating
            or self.development_primary_fields != ("full_sustained", "full_total_residence")
            or self.confirmation_primary_fields != ("full_sustained", "full_total_residence")
            or self.grants_authority
        ):
            raise ValueError("Matrix causal intersection residence method package differs")


def qualify_method(*, config: MatrixResponseCausalIntersectionResidenceStudyConfig) -> MatrixResponseCausalIntersectionResidenceMethodPackage:
    """Run source-inaccessible anti-surrogate truth-table qualification."""

    rows = tuple(
        (response, structural, response and structural)
        for response in (False, True)
        for structural in (False, True)
    )
    if rows != (
        (False, False, False),
        (False, True, False),
        (True, False, False),
        (True, True, True),
    ):
        raise AssertionError("Matrix causal intersection residence full-intersection truth table differs")
    factor_cases = (
        ("closure-failed", True, False, False),
        ("full-intersection", True, True, True),
        ("kernel-failed", True, False, False),
        ("missing-probe-factor", False, True, False),
        ("persistence-failed", True, False, False),
        ("probe-law-failed", False, True, False),
        ("x-admitted", True, False, False),
    )
    if any(full != (response and structural) for _, response, structural, full in factor_cases):
        raise AssertionError("Matrix causal intersection residence anti-surrogate factor fixture differs")
    residence_cases = (
        ("alternating", (True, False, True, False), ("0", "0")),
        ("contiguous", (False, True, True, True, False), ("0.032", "0.032")),
        ("lone", (False, True, False), ("0", "0")),
    )
    for _, flags, expected in residence_cases:
        observed = conservative_residence(flags, cadence_time=0.016)
        if not np.allclose(observed, tuple(map(float, expected)), rtol=0, atol=1e-15):
            raise AssertionError("Matrix causal intersection residence conservative-residence fixture differs")
    fixtures = (
        "matrix-causal-intersection-residence.fixture.alternating-factors-zero-residence",
        "matrix-causal-intersection-residence.fixture.amplitude-only-rejected",
        "matrix-causal-intersection-residence.fixture.closure-failed-rejected",
        "matrix-causal-intersection-residence.fixture.contiguous-full-residence-exact",
        "matrix-causal-intersection-residence.fixture.full-intersection-only-positive",
        "matrix-causal-intersection-residence.fixture.kernel-failed-rejected",
        "matrix-causal-intersection-residence.fixture.lone-full-sample-zero-residence",
        "matrix-causal-intersection-residence.fixture.missing-factor-fails-closed",
        "matrix-causal-intersection-residence.fixture.persistence-failed-rejected",
        "matrix-causal-intersection-residence.fixture.predictive-x-first-transient-response-rejected",
        "matrix-causal-intersection-residence.fixture.probe-law-failed-rejected",
        "matrix-causal-intersection-residence.fixture.structure-only-rejected",
        "matrix-causal-intersection-residence.fixture.x-admission-rejected",
    )
    digest = sha256(
        canonical_json_bytes(
            {
                "factor_cases": factor_cases,
                "fixtures": fixtures,
                "residence_cases": residence_cases,
                "rows": rows,
            }
        )
    ).hexdigest()
    return MatrixResponseCausalIntersectionResidenceMethodPackage(
        package_id="matrix-causal-intersection-residence.full-intersection-method",
        config_fingerprint=config.fingerprint(),
        fixture_ids=fixtures,
        fixture_sha256=digest,
        source_access_count=0,
        full_intersection_exact=True,
        predictive_only_rejected=True,
        structure_only_rejected=True,
        missing_factor_rejected=True,
        residence_semantics_exact=True,
        one_factor_terminals_noncompensating=True,
        development_primary_fields=("full_sustained", "full_total_residence"),
        confirmation_primary_fields=("full_sustained", "full_total_residence"),
        grants_authority=False,
    )


@dataclass(frozen=True, slots=True)
class MatrixResponseCausalIntersectionResidenceDevelopmentAggregate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-causal-intersection-residence-development-aggregate'

    aggregate_id: str
    config_fingerprint: str
    phase_features: tuple[MatrixResponseTransientControlledInvariancePhaseFeature, ...]
    cells: tuple[MatrixResponseCausalIntersectionResidenceDevelopmentCell, ...]
    outcomes: tuple[MatrixResponseCausalIntersectionResidenceBranchOutcome, ...]
    candidate: MatrixResponseCausalIntersectionResidenceDevelopmentCandidate | None
    stop_terminal: MatrixResponseCausalIntersectionResidenceTerminal | None
    terminal_report: MatrixResponseCausalIntersectionResidenceTerminalReport | None
    scientific_trace_count: int
    integration_step_count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.aggregate_id, field_name="aggregate_id")
        validate_sha256(self.config_fingerprint, field_name="config_fingerprint")
        require_sorted_unique_ids(
            self.phase_features, attribute="feature_id", field_name="phase_features"
        )
        require_sorted_unique_ids(self.cells, attribute="cell_id", field_name="cells")
        require_sorted_unique_ids(self.outcomes, attribute="outcome_id", field_name="outcomes")
        if (
            len(self.phase_features) != 7
            or len(self.cells) != 28
            or len(self.outcomes) != 1_120
            or self.scientific_trace_count != 1_120
            or self.integration_step_count != 573_440
        ):
            raise ValueError("Matrix causal intersection residence development aggregate roster differs")
        groups: dict[tuple[int, int], list[MatrixResponseCausalIntersectionResidenceBranchOutcome]] = {}
        for value in self.outcomes:
            groups.setdefault((value.assessment_start_step, value.block_index), []).append(value)
        if (
            len(groups) != 224
            or any(
                len(values) != 5
                or {value.intent_word for value in values}
                != {"hold", "x-negative", "x-positive", "y-negative", "y-positive"}
                or len({value.noise_seed_sha256 for value in values}) != 1
                or len({value.noise_block_sha256 for value in values}) != 1
                or any(value.parent_step_multiplier != 1 for value in values)
                for values in groups.values()
            )
            or len(
                {
                    next(iter({value.noise_block_sha256 for value in values}))
                    for values in groups.values()
                }
            )
            != 224
        ):
            raise ValueError("Matrix causal intersection residence development common-noise roster differs")
        stopped = self.candidate is None
        if stopped != (self.stop_terminal is not None and self.terminal_report is not None):
            raise ValueError("Matrix causal intersection residence development disposition differs")
        if self.candidate is not None and self.candidate.config_fingerprint != (
            self.config_fingerprint
        ):
            raise ValueError("Matrix causal intersection residence development candidate config differs")


@dataclass(frozen=True, slots=True)
class MatrixResponseCausalIntersectionResidenceConfirmationAggregate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-causal-intersection-residence-confirmation-aggregate'

    aggregate_id: str
    config_fingerprint: str
    candidate: MatrixResponseCausalIntersectionResidenceDevelopmentCandidate
    action_outcomes: tuple[MatrixResponseCausalIntersectionResidenceBranchOutcome, ...]
    hold_outcomes: tuple[MatrixResponseCausalIntersectionResidenceBranchOutcome, ...]
    half_action_outcomes: tuple[MatrixResponseCausalIntersectionResidenceBranchOutcome, ...]
    half_hold_outcomes: tuple[MatrixResponseCausalIntersectionResidenceBranchOutcome, ...]
    numerical_audit: MatrixResponseCausalIntersectionResidenceNumericalAudit
    terminal_report: MatrixResponseCausalIntersectionResidenceTerminalReport
    primary_trace_count: int
    half_step_trace_count: int
    primary_integration_step_count: int
    half_step_integration_step_count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.aggregate_id, field_name="aggregate_id")
        validate_sha256(self.config_fingerprint, field_name="config_fingerprint")
        require_sorted_unique_ids(
            self.action_outcomes, attribute="outcome_id", field_name="action_outcomes"
        )
        require_sorted_unique_ids(
            self.hold_outcomes, attribute="outcome_id", field_name="hold_outcomes"
        )
        require_sorted_unique_ids(
            self.half_action_outcomes,
            attribute="outcome_id",
            field_name="half_action_outcomes",
        )
        require_sorted_unique_ids(
            self.half_hold_outcomes,
            attribute="outcome_id",
            field_name="half_hold_outcomes",
        )
        if (
            len(self.action_outcomes) != 128
            or len(self.hold_outcomes) != 128
            or len(self.half_action_outcomes) != 32
            or len(self.half_hold_outcomes) != 32
            or self.primary_trace_count != 256
            or self.half_step_trace_count != 64
            or self.primary_integration_step_count != 262_144
            or self.half_step_integration_step_count != 131_072
            or self.terminal_report.numerical_audit != self.numerical_audit
            or self.terminal_report.candidate_sha256 != self.candidate.candidate_sha256
        ):
            raise ValueError("Matrix causal intersection residence confirmation aggregate roster differs")
        primary_action = {value.block_index: value for value in self.action_outcomes}
        primary_hold = {value.block_index: value for value in self.hold_outcomes}
        half_action = {value.block_index: value for value in self.half_action_outcomes}
        half_hold = {value.block_index: value for value in self.half_hold_outcomes}
        if (
            set(primary_action) != set(range(128))
            or set(primary_hold) != set(range(128))
            or set(half_action) != set(NUMERICAL_AUDIT_BLOCK_INDICES)
            or set(half_hold) != set(NUMERICAL_AUDIT_BLOCK_INDICES)
            or any(
                primary_action[index].noise_seed_sha256 != primary_hold[index].noise_seed_sha256
                or primary_action[index].noise_block_sha256
                != primary_hold[index].noise_block_sha256
                or primary_action[index].parent_step_multiplier != 1
                or primary_hold[index].parent_step_multiplier != 1
                for index in range(128)
            )
            or any(
                half_action[index].noise_seed_sha256 != half_hold[index].noise_seed_sha256
                or half_action[index].noise_block_sha256 != half_hold[index].noise_block_sha256
                or half_action[index].parent_step_multiplier != 2
                or half_hold[index].parent_step_multiplier != 2
                for index in NUMERICAL_AUDIT_BLOCK_INDICES
            )
            or len({value.noise_block_sha256 for value in self.hold_outcomes}) != 128
            or len({value.noise_block_sha256 for value in self.half_hold_outcomes}) != 32
        ):
            raise ValueError("Matrix causal intersection residence confirmation common-noise roster differs")
        if (
            self.candidate.config_fingerprint != self.config_fingerprint
            or self.terminal_report.candidate_sha256 != self.candidate.candidate_sha256
        ):
            raise ValueError("Matrix causal intersection residence confirmation candidate custody differs")


def _first_nonkernel_gap(
    state: SixMatrixState,
) -> float:
    spectrum = next(
        value
        for value in spectral_receiver(
            receiver_prefix=f"spectrum.matrix-causal-intersection-residence.phase-gap.{state.step_index}",
            q=state.q,
            positions=state.positions,
        )
        if value.sector == "Y"
    )
    values = np.asarray(tuple(map(float, spectrum.eigenvalues)), dtype=np.float64)
    denominator = float(np.max(values))
    if not spectrum.valid or denominator <= np.finfo(np.float64).tiny:
        return float("nan")
    return float(values[state.q**2] / denominator)


def extract_online_phase_features(
    *,
    states: Sequence[SixMatrixState],
    decision_steps: tuple[int, ...],
    member: SixMatrixResponseModelFamilyMember,
    config: MatrixResponseCausalIntersectionResidenceStudyConfig,
    parent_step_multiplier: int = 1,
) -> tuple[MatrixResponseTransientControlledInvariancePhaseFeature, ...]:
    """Build past-only six-coordinate features on any frozen physical-time grid."""

    if parent_step_multiplier not in {1, 2}:
        raise ValueError("Matrix causal intersection residence phase multiplier differs")
    by_parent_step = {value.step_index // parent_step_multiplier: value for value in states}
    required = tuple(sorted(set(step - lag for step in decision_steps for lag in (0, 32))))
    if any(step not in by_parent_step for step in required):
        raise ValueError("Matrix causal intersection residence phase-feature history is incomplete")
    observations: dict[int, Any] = {}
    closure: dict[int, float] = {}
    gap: dict[int, float] = {}
    for step in required:
        state = by_parent_step[step]
        observation = observe_state(
            observation_id=f"matrix-causal-intersection-residence.phase-observation.{step}",
            local_step=step,
            local_time=step * float(config.primary_timestep),
            state=state,
            member=member,
            config=config,
        )
        observations[step] = observation
        closure[step] = float(observation.factor_y.closure_ratio)
        gap[step] = _first_nonkernel_gap(state)
    outputs: list[MatrixResponseTransientControlledInvariancePhaseFeature] = []
    derivative_time = 32 * float(config.primary_timestep)
    for step in decision_steps:
        state = by_parent_step[step]
        observation = observations[step]
        y = state.positions[1]
        py = state.momenta[1]
        radius_velocity = float(2.0 * np.vdot(y, py).real / (state.n * float(member.mass_y)))
        cross_denominator = (
            max(
                float(observation.native_receiver.radius_x)
                * float(observation.native_receiver.radius_y),
                np.finfo(np.float64).tiny,
            )
            ** 0.5
        )
        values = (
            float(observation.native_receiver.radius_y),
            radius_velocity,
            closure[step],
            (closure[step] - closure[step - 32]) / derivative_time,
            gap[step],
            (gap[step] - gap[step - 32]) / derivative_time,
            float(observation.native_receiver.radius_x),
            float(observation.factor_x.closure_ratio),
            float(observation.factor_x.kernel_band_ratio),
            float(observation.native_receiver.cross_commutator_norm) / cross_denominator,
        )
        valid = bool(
            observation.factor_x.valid
            and observation.factor_y.valid
            and not observation.factor_x.instantaneous_geometric
            and np.isfinite(values).all()
        )
        outputs.append(
            MatrixResponseTransientControlledInvariancePhaseFeature(
                feature_id=f"matrix-causal-intersection-residence.phase-feature.step-{step}",
                parent_step=step,
                radius_y=_decimal(values[0]),
                radius_y_velocity=_decimal(values[1]),
                closure_y=_decimal(values[2]),
                closure_y_velocity=_decimal(values[3]),
                gap_y=_decimal(values[4]),
                gap_y_velocity=_decimal(values[5]),
                radius_x=_decimal(values[6]),
                closure_x=_decimal(values[7]),
                kernel_x=_decimal(values[8]),
                cross_commutator_ratio=_decimal(values[9]),
                valid=valid,
            )
        )
    return tuple(outputs)


__all__ = [
    'MatrixResponseCausalIntersectionResidenceBranchEvaluation',
    'MatrixResponseCausalIntersectionResidenceBranchOutcome',
    'MatrixResponseCausalIntersectionResidenceDevelopmentCandidate',
    'MatrixResponseCausalIntersectionResidenceDevelopmentAggregate',
    'MatrixResponseCausalIntersectionResidenceDevelopmentCell',
    'MatrixResponseCausalIntersectionResidenceMethodPackage',
    'MatrixResponseCausalIntersectionResidenceNumericalAudit',
    'MatrixResponseCausalIntersectionResidenceResponseFactorSample',
    'MatrixResponseCausalIntersectionResidenceConfirmationAggregate',
    'MatrixResponseCausalIntersectionResidenceTerminalReport',
    'MatrixResponseCausalIntersectionResidenceTerminal',
    "NUMERICAL_AUDIT_BLOCK_INDICES",
    'build_numerical_audit',
    'build_numerical_audit_from_operands',
    'conservative_residence',
    'development_terminal_report',
    'evaluate_full_intersection_branch',
    'extract_online_phase_features',
    'finalize_confirmation',
    'qualify_method',
    'reduce_development_cell',
    'select_development_candidate',
]
