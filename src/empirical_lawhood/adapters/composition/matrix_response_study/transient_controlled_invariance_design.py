"""Frozen, nonauthorizing design for Matrix transient response transient controlled invariance."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_decimal,
    validate_relative_locator,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)

from .contracts import MatrixResponseDesignBinding
from .lineage_inputs import MatrixResponseLineageInputs


@dataclass(frozen=True, slots=True)
class MatrixResponseTransientControlledInvarianceStudyConfig(CanonicalRecord):
    """Every claim-bearing fixed choice for the gated Matrix transient response study."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/matrix-response-study/matrix-response-transient-controlled-invariance-study-config'

    config_id: str
    config_version: str
    plan_id: str
    accepted_platform_commit: str
    accepted_platform_receipt_locator: str
    accepted_platform_receipt_sha256: str
    parent_design_fingerprint: str
    parent_source_fingerprint: str
    parent_numerical_qualification_report_locator: str
    parent_numerical_qualification_report_sha256: str
    parent_numerical_qualification_report_size_bytes: int
    parent_anisotropic_feasibility_report_locator: str
    parent_anisotropic_feasibility_report_sha256: str
    parent_anisotropic_feasibility_report_size_bytes: int
    parent_shooting_root_locator: str
    parent_shooting_config_sha256: str
    parent_shooting_source_manifest_sha256: str
    parent_shooting_artifact_manifest_sha256: str
    parent_shooting_terminal_sha256: str
    parent_rollout_id: str
    parent_final_state_sha256: str
    member_id: str
    member_fingerprint: str
    q: int
    alpha_x_index: int
    alpha_y_index: int
    target_alpha_tilde_x: Decimal
    target_alpha_tilde_y: Decimal
    history_id: str
    seed_index: int
    primary_view_id: str
    primary_view_fingerprint: str
    secondary_view_id: str
    secondary_view_fingerprint: str
    primary_timestep: Decimal
    secondary_timestep: Decimal
    parent_total_steps: int
    parent_ramp_steps: int
    online_start_step: int
    event_checkpoint_steps: tuple[int, ...]
    probe_start_step: int
    probe_end_step: int
    receiver_cadence_steps: int
    rolling_window_samples: int
    phi_min: Decimal
    phi_max: Decimal
    closure_ratio_max: Decimal
    kernel_band_ratio_max: Decimal
    persistence_pass_count: int
    matched_phi_target: Decimal
    matched_phi_half_width: Decimal
    matched_control_primary_count: int
    matched_control_reserve_count: int
    probe_field_count: int
    probe_development_count: int
    probe_kappas: tuple[Decimal, ...]
    probe_seed_rule_id: str
    probe_pairwise_overlap_max: Decimal
    probe_numeric_floor: Decimal
    probe_hermiticity_residual_max: Decimal
    probe_trace_residual_max: Decimal
    probe_identity_drift_max: Decimal
    probe_norm_increase_max: Decimal
    probe_conjugation_error_max: Decimal
    probe_resolved_fraction_min: Decimal
    probe_kappa_resolved_fraction_min: Decimal
    probe_half_state_difference_max: Decimal
    probe_half_order_agreement_min: Decimal
    forecast_origin_steps: tuple[int, ...]
    forecast_horizon_steps: tuple[int, ...]
    shuffle_count: int
    shuffle_seed_rule_id: str
    generic_coefficient_bound: Decimal
    generic_ridge_grid: tuple[Decimal, ...]
    generic_fit_rule_id: str
    generic_gram_rank: int
    response_prediction_qualification_loss_pooled_max: Decimal
    response_prediction_qualification_loss_kappa_max: Decimal
    response_prediction_qualification_radius_ratio_max: Decimal
    response_prediction_qualification_generic_ratio_max: Decimal
    response_prediction_qualification_skill_ratio_min: Decimal
    response_prediction_qualification_shuffle_pooled_min: int
    response_prediction_qualification_shuffle_kappa_min: int
    response_prediction_qualification_field_win_min: int
    response_prediction_qualification_half_control_count: int
    intervention_confirmation_phase_steps: tuple[int, ...]
    intervention_confirmation_development_blocks: int
    intervention_confirmation_action_words: tuple[str, ...]
    pulse_delta: Decimal
    pulse_ramp_steps: int
    pulse_dwell_steps: int
    pulse_return_steps: int
    pulse_washout_steps: int
    pulse_alpha_min: Decimal
    pulse_alpha_max: Decimal
    pulse_increment_max: Decimal
    pulse_total_variation_max: Decimal
    pulse_squared_energy_max: Decimal
    generalized_work_ceiling: Decimal
    pulse_count_max: int
    intervention_confirmation_development_steps: int
    intervention_confirmation_development_assessment_start: int
    intervention_confirmation_sustained_duration: Decimal
    intervention_confirmation_development_rmst_gain_min: Decimal
    intervention_confirmation_development_risk_gain_min: Decimal
    intervention_confirmation_confirmation_blocks: int
    intervention_confirmation_confirmation_steps: int
    intervention_confirmation_assessment_start_step: int
    intervention_confirmation_assessment_end_step: int
    intervention_confirmation_valid_pair_min: int
    intervention_confirmation_risk_gain_min: Decimal
    intervention_confirmation_rmst_gain_min: Decimal
    intervention_confirmation_numerical_audit_blocks: int
    rule_predicate_max: int
    prospective_control_blocks: int
    prospective_control_policy_words: tuple[str, ...]
    prospective_control_assessment_start_step: int
    prospective_control_assessment_end_step: int
    prospective_control_residence_duration: Decimal
    prospective_control_valid_block_min: int
    prospective_control_risk_gain_min: Decimal
    prospective_control_rmst_gain_min: Decimal
    prospective_control_rmst_lower_min: Decimal
    prospective_control_stratum_risk_gain_min: Decimal
    prospective_control_radius_risk_gain_min: Decimal
    prospective_control_numerical_audit_blocks: int
    bootstrap_resamples: int
    confidence_level: Decimal
    familywise_alpha: Decimal
    preferred_wall_seconds: int
    hard_wall_seconds: int
    maximum_memory_bytes: int
    maximum_output_bytes: int
    total_integration_steps: int
    evidence_world: str
    maximum_claim: str
    nonclaims: tuple[str, ...]
    grants_authority: bool

    def __post_init__(self) -> None:
        for name in (
            "config_id",
            "plan_id",
            "parent_rollout_id",
            "member_id",
            "history_id",
            "primary_view_id",
            "secondary_view_id",
            "probe_seed_rule_id",
            "shuffle_seed_rule_id",
            "generic_fit_rule_id",
            "evidence_world",
            "maximum_claim",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_semantic_version(self.config_version)
        for name in (
            "parent_design_fingerprint",
            "parent_source_fingerprint",
            "accepted_platform_receipt_sha256",
            "parent_numerical_qualification_report_sha256",
            "parent_anisotropic_feasibility_report_sha256",
            "parent_shooting_config_sha256",
            "parent_shooting_source_manifest_sha256",
            "parent_shooting_artifact_manifest_sha256",
            "parent_shooting_terminal_sha256",
            "parent_final_state_sha256",
            "member_fingerprint",
            "primary_view_fingerprint",
            "secondary_view_fingerprint",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        if len(self.accepted_platform_commit) != 40:
            raise ValueError("Matrix transient response accepted platform commit must be a Git SHA-1")
        int(self.accepted_platform_commit, 16)
        for name in (
            "accepted_platform_receipt_locator",
            "parent_numerical_qualification_report_locator",
            "parent_anisotropic_feasibility_report_locator",
            "parent_shooting_root_locator",
        ):
            validate_relative_locator(getattr(self, name))
        if self.parent_anisotropic_feasibility_report_size_bytes <= 0 or self.parent_numerical_qualification_report_size_bytes <= 0:
            raise ValueError("Matrix transient response parent reports require positive byte counts")
        if (self.q, self.alpha_x_index, self.alpha_y_index, self.seed_index) != (2, 1, 11, 1):
            raise ValueError("Matrix transient response selected coordinate differs")
        if (self.target_alpha_tilde_x, self.target_alpha_tilde_y) != (
            Decimal("0.6666666666666666666666666667"),
            Decimal("7.333333333333333333333333334"),
        ):
            raise ValueError("Matrix transient response baseline couplings differ")
        if (
            self.primary_timestep,
            self.secondary_timestep,
            self.parent_total_steps,
            self.parent_ramp_steps,
            self.online_start_step,
        ) != (Decimal("0.001"), Decimal("0.0005"), 1024, 256, 768):
            raise ValueError("Matrix transient response parent/view timing differs")
        if self.event_checkpoint_steps != (768, 816, 848, 880, 896, 928, 960, 1024):
            raise ValueError("Matrix transient response checkpoint roster differs")
        if (
            self.probe_start_step,
            self.probe_end_step,
            self.receiver_cadence_steps,
            self.rolling_window_samples,
        ) != (816, 1024, 16, 16):
            raise ValueError("Matrix transient response probe/receiver timing differs")
        if (
            self.matched_control_primary_count,
            self.matched_control_reserve_count,
            self.probe_field_count,
            self.probe_development_count,
        ) != (20, 2, 12, 6):
            raise ValueError("Matrix transient response control/probe cardinality differs")
        if self.probe_kappas != (Decimal("0.25"), Decimal("0.5"), Decimal("1.0")):
            raise ValueError("Matrix transient response kappa roster differs")
        if self.forecast_origin_steps != (816, 848, 880, 896, 928, 960):
            raise ValueError("Matrix transient response forecast origins differ")
        if self.forecast_horizon_steps != (16, 32, 64) or self.shuffle_count != 32:
            raise ValueError("Matrix transient response forecast/shuffle cardinality differs")
        if self.generic_ridge_grid != (
            Decimal("0"),
            Decimal("1e-8"),
            Decimal("1e-6"),
            Decimal("1e-4"),
        ):
            raise ValueError("Matrix transient response generic ridge grid differs")
        if self.generic_gram_rank != 3:
            raise ValueError("Matrix transient response generic effective-rank contract differs")
        if self.intervention_confirmation_phase_steps != (816, 848, 880, 896, 928, 960, 1024):
            raise ValueError("Matrix transient response Intervention confirmation phase roster differs")
        if self.intervention_confirmation_action_words != (
            "hold",
            "x-negative",
            "x-positive",
            "y-negative",
            "y-positive",
        ):
            raise ValueError("Matrix transient response Intervention confirmation action roster differs")
        if (
            self.pulse_ramp_steps,
            self.pulse_dwell_steps,
            self.pulse_return_steps,
            self.pulse_washout_steps,
            self.pulse_count_max,
        ) != (48, 32, 48, 256, 1):
            raise ValueError("Matrix transient response pulse shape differs")
        if (
            self.intervention_confirmation_development_blocks,
            self.intervention_confirmation_development_steps,
            self.intervention_confirmation_development_assessment_start,
            self.intervention_confirmation_confirmation_blocks,
            self.intervention_confirmation_confirmation_steps,
            self.intervention_confirmation_assessment_start_step,
            self.intervention_confirmation_assessment_end_step,
        ) != (32, 512, 384, 128, 1024, 1408, 1792):
            raise ValueError("Matrix transient response Intervention confirmation timing/cardinality differs")
        if self.prospective_control_policy_words != (
            "hold",
            "open-loop-energy-matched",
            "phase-aware",
            "phase-shuffled-feedback",
            "radius-only-feedback",
            "stratified-random",
        ):
            raise ValueError("Matrix transient response Prospective control policy roster differs")
        if (
            self.prospective_control_blocks,
            self.prospective_control_assessment_start_step,
            self.prospective_control_assessment_end_step,
        ) != (128, 1408, 1792):
            raise ValueError("Matrix transient response Prospective control timing/cardinality differs")
        decimal_fields = (
            "target_alpha_tilde_x",
            "target_alpha_tilde_y",
            "primary_timestep",
            "secondary_timestep",
            "phi_min",
            "phi_max",
            "closure_ratio_max",
            "kernel_band_ratio_max",
            "matched_phi_target",
            "matched_phi_half_width",
            "probe_pairwise_overlap_max",
            "probe_numeric_floor",
            "probe_hermiticity_residual_max",
            "probe_trace_residual_max",
            "probe_identity_drift_max",
            "probe_norm_increase_max",
            "probe_conjugation_error_max",
            "probe_resolved_fraction_min",
            "probe_kappa_resolved_fraction_min",
            "probe_half_state_difference_max",
            "probe_half_order_agreement_min",
            "generic_coefficient_bound",
            "response_prediction_qualification_loss_pooled_max",
            "response_prediction_qualification_loss_kappa_max",
            "response_prediction_qualification_radius_ratio_max",
            "response_prediction_qualification_generic_ratio_max",
            "response_prediction_qualification_skill_ratio_min",
            "pulse_delta",
            "pulse_alpha_min",
            "pulse_alpha_max",
            "pulse_increment_max",
            "pulse_total_variation_max",
            "pulse_squared_energy_max",
            "generalized_work_ceiling",
            "intervention_confirmation_sustained_duration",
            "intervention_confirmation_development_rmst_gain_min",
            "intervention_confirmation_development_risk_gain_min",
            "intervention_confirmation_risk_gain_min",
            "intervention_confirmation_rmst_gain_min",
            "prospective_control_residence_duration",
            "prospective_control_risk_gain_min",
            "prospective_control_rmst_gain_min",
            "prospective_control_rmst_lower_min",
            "prospective_control_stratum_risk_gain_min",
            "prospective_control_radius_risk_gain_min",
            "confidence_level",
            "familywise_alpha",
        )
        for name in decimal_fields:
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        for index, value in enumerate((*self.probe_kappas, *self.generic_ridge_grid)):
            validate_decimal(value, field_name=f"design_decimal[{index}]", minimum=Decimal(0))
        if any(
            value > Decimal(1)
            for value in (
                self.probe_pairwise_overlap_max,
                self.probe_resolved_fraction_min,
                self.probe_kappa_resolved_fraction_min,
                self.probe_half_order_agreement_min,
                self.confidence_level,
                self.familywise_alpha,
            )
        ):
            raise ValueError("Matrix transient response probability/fraction exceeds one")
        if (
            self.preferred_wall_seconds,
            self.hard_wall_seconds,
            self.maximum_memory_bytes,
            self.maximum_output_bytes,
            self.total_integration_steps,
        ) != (14_400, 43_200, 8 * 1024**3, 1024**3, 2_178_048):
            raise ValueError("Matrix transient response resource envelope differs")
        require_sorted_unique_strings(self.nonclaims, field_name="nonclaims", allow_empty=False)
        if self.grants_authority:
            raise ValueError("Matrix transient response config cannot grant authority")


def build_matrix_response_study_transient_controlled_invariance_study_config(
    design: MatrixResponseDesignBinding, lineage: MatrixResponseLineageInputs
) -> MatrixResponseTransientControlledInvarianceStudyConfig:
    source = design.source_config
    member = next(
        value
        for value in source.anisotropic_model.family_members
        if value.member_id == "six-matrix-response.member.mass-0p5.cross-coupling-1"
    )
    return MatrixResponseTransientControlledInvarianceStudyConfig(
        config_id="matrix-transient-response.controlled-invariance-transient-fuzzy-geometry",
        config_version="1.0.0",
        plan_id="matrix-transient-response-study",
        accepted_platform_commit=design.platform_acceptance_commit,
        accepted_platform_receipt_locator=(
            lineage.locator('accepted_platform_receipt_locator')
        ),
        accepted_platform_receipt_sha256=(
            lineage.sha256('accepted_platform_receipt_sha256')
        ),
        parent_design_fingerprint=design.fingerprint(),
        parent_source_fingerprint=source.fingerprint(),
        parent_numerical_qualification_report_locator=(
            lineage.locator('parent_numerical_qualification_report_locator')
        ),
        parent_numerical_qualification_report_sha256=(
            lineage.sha256('parent_numerical_qualification_report_sha256')
        ),
        parent_numerical_qualification_report_size_bytes=lineage.positive_int('parent_numerical_qualification_report_size_bytes'),
        parent_anisotropic_feasibility_report_locator=(
            lineage.locator('parent_anisotropic_feasibility_report_locator')
        ),
        parent_anisotropic_feasibility_report_sha256=(
            lineage.sha256('parent_anisotropic_feasibility_report_sha256')
        ),
        parent_anisotropic_feasibility_report_size_bytes=lineage.positive_int('parent_anisotropic_feasibility_report_size_bytes'),
        parent_shooting_root_locator=(lineage.locator('parent_shooting_root_locator')),
        parent_shooting_config_sha256=(
            lineage.sha256('parent_shooting_config_sha256')
        ),
        parent_shooting_source_manifest_sha256=(
            lineage.sha256('parent_shooting_source_manifest_sha256')
        ),
        parent_shooting_artifact_manifest_sha256=(
            lineage.sha256('parent_shooting_artifact_manifest_sha256')
        ),
        parent_shooting_terminal_sha256=(
            lineage.sha256('parent_shooting_terminal_sha256')
        ),
        parent_rollout_id=(lineage.stable_id('parent_rollout_id')),
        parent_final_state_sha256=(
            lineage.sha256('parent_final_state_sha256')
        ),
        member_id=member.member_id,
        member_fingerprint=member.fingerprint(),
        q=2,
        alpha_x_index=1,
        alpha_y_index=11,
        target_alpha_tilde_x=Decimal("0.6666666666666666666666666667"),
        target_alpha_tilde_y=Decimal("7.333333333333333333333333334"),
        history_id="matrix-history.joint-increasing-coupling",
        seed_index=1,
        primary_view_id=source.primary_view.view_id,
        primary_view_fingerprint=source.primary_view.fingerprint(),
        secondary_view_id=source.secondary_view.view_id,
        secondary_view_fingerprint=source.secondary_view.fingerprint(),
        primary_timestep=source.primary_view.timestep,
        secondary_timestep=source.secondary_view.timestep,
        parent_total_steps=1024,
        parent_ramp_steps=256,
        online_start_step=768,
        event_checkpoint_steps=(768, 816, 848, 880, 896, 928, 960, 1024),
        probe_start_step=816,
        probe_end_step=1024,
        receiver_cadence_steps=16,
        rolling_window_samples=16,
        phi_min=Decimal("0.35"),
        phi_max=Decimal("0.95"),
        closure_ratio_max=Decimal("0.30"),
        kernel_band_ratio_max=Decimal("0.25"),
        persistence_pass_count=12,
        matched_phi_target=Decimal("0.6910748486811329"),
        matched_phi_half_width=Decimal("0.02"),
        matched_control_primary_count=20,
        matched_control_reserve_count=2,
        probe_field_count=12,
        probe_development_count=6,
        probe_kappas=(Decimal("0.25"), Decimal("0.5"), Decimal("1.0")),
        probe_seed_rule_id="matrix-transient-response.probe-field.explicit-original-seed-pcg64dxsm-qr",
        probe_pairwise_overlap_max=Decimal("0.5"),
        probe_numeric_floor=Decimal("1e-12"),
        probe_hermiticity_residual_max=Decimal("1e-10"),
        probe_trace_residual_max=Decimal("1e-10"),
        probe_identity_drift_max=Decimal("1e-10"),
        probe_norm_increase_max=Decimal("1e-10"),
        probe_conjugation_error_max=Decimal("1e-10"),
        probe_resolved_fraction_min=Decimal("0.90"),
        probe_kappa_resolved_fraction_min=Decimal("0.80"),
        probe_half_state_difference_max=Decimal("0.05"),
        probe_half_order_agreement_min=Decimal("0.90"),
        forecast_origin_steps=(816, 848, 880, 896, 928, 960),
        forecast_horizon_steps=(16, 32, 64),
        shuffle_count=32,
        shuffle_seed_rule_id="matrix-transient-response.operator-shuffle.explicit-original-seed-pcg64dxsm-qr",
        generic_coefficient_bound=Decimal("8"),
        generic_ridge_grid=(
            Decimal("0"),
            Decimal("1e-8"),
            Decimal("1e-6"),
            Decimal("1e-4"),
        ),
        generic_fit_rule_id="matrix-transient-response.linearized-generator-gram-rank3",
        generic_gram_rank=3,
        response_prediction_qualification_loss_pooled_max=Decimal("0.25"),
        response_prediction_qualification_loss_kappa_max=Decimal("0.35"),
        response_prediction_qualification_radius_ratio_max=Decimal("0.75"),
        response_prediction_qualification_generic_ratio_max=Decimal("0.80"),
        response_prediction_qualification_skill_ratio_min=Decimal("1.25"),
        response_prediction_qualification_shuffle_pooled_min=31,
        response_prediction_qualification_shuffle_kappa_min=28,
        response_prediction_qualification_field_win_min=5,
        response_prediction_qualification_half_control_count=4,
        intervention_confirmation_phase_steps=(816, 848, 880, 896, 928, 960, 1024),
        intervention_confirmation_development_blocks=32,
        intervention_confirmation_action_words=(
            "hold",
            "x-negative",
            "x-positive",
            "y-negative",
            "y-positive",
        ),
        pulse_delta=Decimal("0.125"),
        pulse_ramp_steps=48,
        pulse_dwell_steps=32,
        pulse_return_steps=48,
        pulse_washout_steps=256,
        pulse_alpha_min=Decimal("0"),
        pulse_alpha_max=Decimal("8"),
        pulse_increment_max=Decimal("0.002604166666666666666666666667"),
        pulse_total_variation_max=Decimal("0.25"),
        pulse_squared_energy_max=Decimal("0.001001"),
        generalized_work_ceiling=Decimal("32"),
        pulse_count_max=1,
        intervention_confirmation_development_steps=512,
        intervention_confirmation_development_assessment_start=384,
        intervention_confirmation_sustained_duration=Decimal("0.064"),
        intervention_confirmation_development_rmst_gain_min=Decimal("0.032"),
        intervention_confirmation_development_risk_gain_min=Decimal("0.10"),
        intervention_confirmation_confirmation_blocks=128,
        intervention_confirmation_confirmation_steps=1024,
        intervention_confirmation_assessment_start_step=1408,
        intervention_confirmation_assessment_end_step=1792,
        intervention_confirmation_valid_pair_min=120,
        intervention_confirmation_risk_gain_min=Decimal("0.10"),
        intervention_confirmation_rmst_gain_min=Decimal("0.032"),
        intervention_confirmation_numerical_audit_blocks=32,
        rule_predicate_max=3,
        prospective_control_blocks=128,
        prospective_control_policy_words=(
            "hold",
            "open-loop-energy-matched",
            "phase-aware",
            "phase-shuffled-feedback",
            "radius-only-feedback",
            "stratified-random",
        ),
        prospective_control_assessment_start_step=1408,
        prospective_control_assessment_end_step=1792,
        prospective_control_residence_duration=Decimal("0.128"),
        prospective_control_valid_block_min=120,
        prospective_control_risk_gain_min=Decimal("0.10"),
        prospective_control_rmst_gain_min=Decimal("0.064"),
        prospective_control_rmst_lower_min=Decimal("0.032"),
        prospective_control_stratum_risk_gain_min=Decimal("0.05"),
        prospective_control_radius_risk_gain_min=Decimal("0.05"),
        prospective_control_numerical_audit_blocks=32,
        bootstrap_resamples=99_999,
        confidence_level=Decimal("0.95"),
        familywise_alpha=Decimal("0.05"),
        preferred_wall_seconds=14_400,
        hard_wall_seconds=43_200,
        maximum_memory_bytes=8 * 1024**3,
        maximum_output_bytes=1024**3,
        total_integration_steps=2_178_048,
        evidence_world="outcome-visible-event-selected-same-implementation-dimension-two",
        maximum_claim="event-conditional-maintained-constitution-response-domain",
        nonclaims=tuple(
            sorted(
                (
                    "asymptotic-stability",
                    "six-matrix-response-rescue",
                    "constitutive-phase-prevalence",
                    "controller-relative-lower-law",
                    "cross-event-or-substrate-transport",
                    "base-system-prospective-evaluation-or-response-law",
                    "physical-control",
                    "production-controller",
                    "recurrent-autonomous-phase",
                )
            )
        ),
        grants_authority=False,
    )


__all__ = [
    'MatrixResponseTransientControlledInvarianceStudyConfig',
    'build_matrix_response_study_transient_controlled_invariance_study_config',
]
