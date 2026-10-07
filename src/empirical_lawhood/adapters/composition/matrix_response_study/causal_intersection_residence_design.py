"Frozen design for Matrix causal intersection residence full-intersection causal authority.\n\ncausal intersection residence is a fresh Intervention confirmation study.  It inherits only fixed numerical/action\ncoordinates from transient response and the positive response-geometry prerequisite from invariant capacity comparison;\nit does not resume either attempt and does not authorize prospective control.\n"

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

from empirical_lawhood.adapters.simulators.six_matrix_response.probe_scientific_input import MatrixResponseProbeScientificInput

from .transient_controlled_invariance_design import MatrixResponseTransientControlledInvarianceStudyConfig
from .invariant_capacity_comparison_design import InvariantCapacityComparatorStudyConfig
from .lineage_inputs import MatrixResponseLineageInputs


@dataclass(frozen=True, slots=True)
class MatrixResponseCausalIntersectionResidenceStudyConfig(CanonicalRecord):
    """All claim-bearing fixed choices for the bounded causal intersection residence Intervention confirmation assay."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/matrix-response-study/matrix-response-causal-intersection-residence-study-config'

    config_id: str
    config_version: str
    plan_id: str
    accepted_platform_commit: str
    accepted_platform_archive_sha256: str
    accepted_platform_receipt_locator: str
    accepted_platform_receipt_sha256: str
    accepted_platform_capability_receipt_ids: tuple[str, ...]
    parent_numerical_qualification_report_locator: str
    parent_numerical_qualification_report_sha256: str
    parent_numerical_qualification_report_size_bytes: int
    parent_anisotropic_feasibility_report_locator: str
    parent_anisotropic_feasibility_report_sha256: str
    parent_anisotropic_feasibility_report_size_bytes: int
    parent_shooting_root_locator: str
    parent_shooting_config_sha256: str
    parent_shooting_source_manifest_sha256: str
    parent_shooting_terminal_sha256: str
    parent_shooting_artifact_manifest_sha256: str
    parent_transient_response_root_locator: str
    parent_transient_response_config_sha256: str
    parent_transient_response_terminal_sha256: str
    parent_transient_response_artifact_manifest_sha256: str
    parent_transient_response_roster_sha256: str
    parent_transient_response_checkpoints_sha256: str
    parent_transient_response_probes_sha256: str
    parent_transient_response_scores_sha256: str
    parent_invariant_capacity_comparison_root_locator: str
    parent_invariant_capacity_comparison_config_sha256: str
    parent_invariant_capacity_comparison_method_package_sha256: str
    parent_invariant_capacity_comparison_source_manifest_sha256: str
    parent_invariant_capacity_comparison_terminal_sha256: str
    parent_invariant_capacity_comparison_artifact_manifest_sha256: str
    required_invariant_capacity_comparison_terminal: str
    parent_rollout_id: str
    parent_final_state_sha256: str
    member_id: str
    member_fingerprint: str
    q: int
    primary_view_id: str
    primary_view_fingerprint: str
    secondary_view_id: str
    secondary_view_fingerprint: str
    primary_timestep: Decimal
    secondary_timestep: Decimal
    online_start_step: int
    development_checkpoint_steps: tuple[int, ...]
    receiver_cadence_steps: int
    rolling_window_samples: int
    phi_min: Decimal
    phi_max: Decimal
    closure_ratio_max: Decimal
    kernel_band_ratio_max: Decimal
    persistence_pass_count: int
    heldout_probe_field_indices: tuple[int, ...]
    probe_kappas: tuple[Decimal, ...]
    probe_seed_rule_id: str
    probe_roster_config_fingerprint: str
    probe_scientific_input: MatrixResponseProbeScientificInput
    probe_numeric_floor: Decimal
    probe_hermiticity_residual_max: Decimal
    probe_trace_residual_max: Decimal
    probe_identity_drift_max: Decimal
    probe_norm_increase_max: Decimal
    probe_forecast_lag_steps: int
    response_geometry_loss_max: Decimal
    response_radius_ratio_max: Decimal
    development_blocks: int
    development_action_words: tuple[str, ...]
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
    development_branch_steps: int
    development_assessment_offset: int
    sustained_duration: Decimal
    development_residence_gain_min: Decimal
    development_risk_gain_min: Decimal
    rule_predicate_max: int
    confirmation_blocks: int
    confirmation_branch_steps: int
    confirmation_assessment_start_step: int
    confirmation_assessment_end_step: int
    confirmation_valid_pair_min: int
    confirmation_risk_gain_min: Decimal
    confirmation_residence_gain_min: Decimal
    numerical_audit_blocks: int
    numerical_classification_agreement_min: Decimal
    numerical_probe_state_difference_max: Decimal
    numerical_energy_relative_error_max: Decimal
    numerical_work_relative_error_max: Decimal
    bootstrap_resamples: int
    confidence_level: Decimal
    worker_count_max: int
    preferred_wall_seconds: int
    hard_wall_seconds: int
    maximum_memory_bytes: int
    maximum_output_bytes: int
    total_integration_steps_ceiling: int
    maximum_scientific_trace_count: int
    evidence_world: str
    maximum_claim: str
    nonclaims: tuple[str, ...]
    source_inaccessible_method_qualification: bool
    prospective_control_condition_false: bool
    grants_authority: bool

    def __post_init__(self) -> None:
        for name in (
            "config_id",
            "plan_id",
            "parent_rollout_id",
            "member_id",
            "primary_view_id",
            "secondary_view_id",
            "probe_seed_rule_id",
            "evidence_world",
            "maximum_claim",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.required_invariant_capacity_comparison_terminal != "TRANSIENT_RESPONSE_GEOMETRY_SUPPORTED":
            raise ValueError("Matrix causal intersection residence controlling invariant capacity comparison terminal differs")
        validate_semantic_version(self.config_version)
        if len(self.accepted_platform_commit) != 40:
            raise ValueError("Matrix causal intersection residence accepted platform commit differs")
        int(self.accepted_platform_commit, 16)
        for name in (
            "accepted_platform_archive_sha256",
            "accepted_platform_receipt_sha256",
            "parent_numerical_qualification_report_sha256",
            "parent_anisotropic_feasibility_report_sha256",
            "parent_shooting_config_sha256",
            "parent_shooting_source_manifest_sha256",
            "parent_shooting_terminal_sha256",
            "parent_shooting_artifact_manifest_sha256",
            "parent_transient_response_config_sha256",
            "parent_transient_response_terminal_sha256",
            "parent_transient_response_artifact_manifest_sha256",
            "parent_transient_response_roster_sha256",
            "parent_transient_response_checkpoints_sha256",
            "parent_transient_response_probes_sha256",
            "parent_transient_response_scores_sha256",
            "parent_invariant_capacity_comparison_config_sha256",
            "parent_invariant_capacity_comparison_method_package_sha256",
            "parent_invariant_capacity_comparison_source_manifest_sha256",
            "parent_invariant_capacity_comparison_terminal_sha256",
            "parent_invariant_capacity_comparison_artifact_manifest_sha256",
            "parent_final_state_sha256",
            "member_fingerprint",
            "primary_view_fingerprint",
            "secondary_view_fingerprint",
            "probe_roster_config_fingerprint",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        if not isinstance(self.probe_scientific_input, MatrixResponseProbeScientificInput):
            raise ValueError("matrix response probe requires an explicit scientific input")
        self.probe_scientific_input.require_config_custody(
            original_config_sha256=self.parent_transient_response_config_sha256,
            target_config_sha256=self.probe_roster_config_fingerprint,
        )
        for name in (
            "accepted_platform_receipt_locator",
            "parent_numerical_qualification_report_locator",
            "parent_anisotropic_feasibility_report_locator",
            "parent_shooting_root_locator",
            "parent_transient_response_root_locator",
            "parent_invariant_capacity_comparison_root_locator",
        ):
            validate_relative_locator(getattr(self, name))
        require_sorted_unique_strings(
            self.accepted_platform_capability_receipt_ids,
            field_name="accepted_platform_capability_receipt_ids",
            allow_empty=False,
        )
        decimal_fields = (
            "primary_timestep",
            "secondary_timestep",
            "phi_min",
            "phi_max",
            "closure_ratio_max",
            "kernel_band_ratio_max",
            "probe_numeric_floor",
            "probe_hermiticity_residual_max",
            "probe_trace_residual_max",
            "probe_identity_drift_max",
            "probe_norm_increase_max",
            "response_geometry_loss_max",
            "response_radius_ratio_max",
            "pulse_delta",
            "pulse_alpha_min",
            "pulse_alpha_max",
            "pulse_increment_max",
            "pulse_total_variation_max",
            "pulse_squared_energy_max",
            "generalized_work_ceiling",
            "sustained_duration",
            "development_residence_gain_min",
            "development_risk_gain_min",
            "confirmation_risk_gain_min",
            "confirmation_residence_gain_min",
            "numerical_classification_agreement_min",
            "numerical_probe_state_difference_max",
            "numerical_energy_relative_error_max",
            "numerical_work_relative_error_max",
            "confidence_level",
        )
        for name in decimal_fields:
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        for index, value in enumerate(self.probe_kappas):
            validate_decimal(value, field_name=f"probe_kappas[{index}]", minimum=Decimal(0))
        if (
            self.q,
            self.primary_timestep,
            self.secondary_timestep,
            self.online_start_step,
        ) != (2, Decimal("0.001"), Decimal("0.0005"), 768):
            raise ValueError("Matrix causal intersection residence model/view denominator differs")
        if self.development_checkpoint_steps != (816, 848, 880, 896, 928, 960, 1024):
            raise ValueError("Matrix causal intersection residence checkpoint roster differs")
        if (
            self.receiver_cadence_steps,
            self.rolling_window_samples,
            self.persistence_pass_count,
            self.probe_forecast_lag_steps,
        ) != (16, 16, 12, 32):
            raise ValueError("Matrix causal intersection residence temporal endpoint differs")
        if (
            self.phi_min,
            self.phi_max,
            self.closure_ratio_max,
            self.kernel_band_ratio_max,
            self.response_geometry_loss_max,
            self.response_radius_ratio_max,
        ) != (
            Decimal("0.35"),
            Decimal("0.95"),
            Decimal("0.30"),
            Decimal("0.25"),
            Decimal("0.25"),
            Decimal("0.75"),
        ):
            raise ValueError("Matrix causal intersection residence full-intersection thresholds differ")
        if self.heldout_probe_field_indices != (6, 7, 8, 9, 10, 11) or self.probe_kappas != (
            Decimal("0.25"),
            Decimal("0.5"),
            Decimal("1.0"),
        ):
            raise ValueError("Matrix causal intersection residence held-out probe roster differs")
        if (
            self.development_blocks,
            self.development_action_words,
            self.development_branch_steps,
            self.development_assessment_offset,
            self.confirmation_blocks,
            self.confirmation_branch_steps,
            self.confirmation_assessment_start_step,
            self.confirmation_assessment_end_step,
            self.confirmation_valid_pair_min,
            self.numerical_audit_blocks,
        ) != (
            32,
            ("hold", "x-negative", "x-positive", "y-negative", "y-positive"),
            512,
            384,
            128,
            1024,
            1408,
            1792,
            120,
            32,
        ):
            raise ValueError("Matrix causal intersection residence branch roster differs")
        if (
            self.pulse_delta,
            self.pulse_ramp_steps,
            self.pulse_dwell_steps,
            self.pulse_return_steps,
            self.pulse_washout_steps,
            self.pulse_count_max,
            self.rule_predicate_max,
        ) != (Decimal("0.125"), 48, 32, 48, 256, 1, 3):
            raise ValueError("Matrix causal intersection residence action/rule capacity differs")
        if (
            self.sustained_duration,
            self.development_residence_gain_min,
            self.development_risk_gain_min,
            self.confirmation_risk_gain_min,
            self.confirmation_residence_gain_min,
        ) != (
            Decimal("0.064"),
            Decimal("0.032"),
            Decimal("0.10"),
            Decimal("0.10"),
            Decimal("0.032"),
        ):
            raise ValueError("Matrix causal intersection residence claim-bearing effect contract differs")
        if (
            self.worker_count_max,
            self.preferred_wall_seconds,
            self.hard_wall_seconds,
            self.maximum_memory_bytes,
            self.maximum_output_bytes,
            self.total_integration_steps_ceiling,
            self.maximum_scientific_trace_count,
        ) != (8, 14_400, 43_200, 8 * 1024**3, 2 * 1024**3, 967_680, 1_441):
            raise ValueError("Matrix causal intersection residence resource envelope differs")
        require_sorted_unique_strings(self.nonclaims, field_name="nonclaims", allow_empty=False)
        if (
            not self.source_inaccessible_method_qualification
            or not self.prospective_control_condition_false
            or self.grants_authority
        ):
            raise ValueError("Matrix causal intersection residence authority boundary differs")


def build_matrix_response_study_causal_intersection_residence_study_config(
    transient_invariance_config: MatrixResponseTransientControlledInvarianceStudyConfig, invariant_capacity_config: InvariantCapacityComparatorStudyConfig, lineage: MatrixResponseLineageInputs,
    *, probe_scientific_input: MatrixResponseProbeScientificInput
) -> MatrixResponseCausalIntersectionResidenceStudyConfig:
    return MatrixResponseCausalIntersectionResidenceStudyConfig(
        config_id="matrix-causal-intersection-residence.full-intersection-causal-authority",
        config_version="1.0.0",
        plan_id="matrix-causal-intersection-residence-study",
        accepted_platform_commit=invariant_capacity_config.accepted_platform_commit,
        accepted_platform_archive_sha256=invariant_capacity_config.accepted_platform_archive_sha256,
        accepted_platform_receipt_locator=invariant_capacity_config.accepted_platform_receipt_locator,
        accepted_platform_receipt_sha256=invariant_capacity_config.accepted_platform_receipt_sha256,
        accepted_platform_capability_receipt_ids=invariant_capacity_config.accepted_platform_capability_receipt_ids,
        parent_numerical_qualification_report_locator=transient_invariance_config.parent_numerical_qualification_report_locator,
        parent_numerical_qualification_report_sha256=transient_invariance_config.parent_numerical_qualification_report_sha256,
        parent_numerical_qualification_report_size_bytes=transient_invariance_config.parent_numerical_qualification_report_size_bytes,
        parent_anisotropic_feasibility_report_locator=transient_invariance_config.parent_anisotropic_feasibility_report_locator,
        parent_anisotropic_feasibility_report_sha256=transient_invariance_config.parent_anisotropic_feasibility_report_sha256,
        parent_anisotropic_feasibility_report_size_bytes=transient_invariance_config.parent_anisotropic_feasibility_report_size_bytes,
        parent_shooting_root_locator=transient_invariance_config.parent_shooting_root_locator,
        parent_shooting_config_sha256=transient_invariance_config.parent_shooting_config_sha256,
        parent_shooting_source_manifest_sha256=transient_invariance_config.parent_shooting_source_manifest_sha256,
        parent_shooting_terminal_sha256=transient_invariance_config.parent_shooting_terminal_sha256,
        parent_shooting_artifact_manifest_sha256=transient_invariance_config.parent_shooting_artifact_manifest_sha256,
        parent_transient_response_root_locator=invariant_capacity_config.parent_transient_response_root_locator,
        parent_transient_response_config_sha256=lineage.sha256('parent_transient_response_config_sha256'),
        parent_transient_response_terminal_sha256=invariant_capacity_config.parent_transient_response_terminal_sha256,
        parent_transient_response_artifact_manifest_sha256=invariant_capacity_config.parent_transient_response_artifact_manifest_sha256,
        parent_transient_response_roster_sha256=invariant_capacity_config.parent_transient_response_roster_sha256,
        parent_transient_response_checkpoints_sha256=invariant_capacity_config.parent_transient_response_checkpoints_sha256,
        parent_transient_response_probes_sha256=invariant_capacity_config.parent_transient_response_probes_sha256,
        parent_transient_response_scores_sha256=invariant_capacity_config.parent_transient_response_scores_sha256,
        parent_invariant_capacity_comparison_root_locator=(lineage.locator('parent_invariant_capacity_comparison_root_locator')),
        parent_invariant_capacity_comparison_config_sha256=lineage.sha256('parent_invariant_capacity_comparison_config_sha256'),
        parent_invariant_capacity_comparison_method_package_sha256=(
            lineage.sha256('parent_invariant_capacity_comparison_method_package_sha256')
        ),
        parent_invariant_capacity_comparison_source_manifest_sha256=(
            lineage.sha256('parent_invariant_capacity_comparison_source_manifest_sha256')
        ),
        parent_invariant_capacity_comparison_terminal_sha256=(
            lineage.sha256('parent_invariant_capacity_comparison_terminal_sha256')
        ),
        parent_invariant_capacity_comparison_artifact_manifest_sha256=(
            lineage.sha256('parent_invariant_capacity_comparison_artifact_manifest_sha256')
        ),
        required_invariant_capacity_comparison_terminal="TRANSIENT_RESPONSE_GEOMETRY_SUPPORTED",
        parent_rollout_id=transient_invariance_config.parent_rollout_id,
        parent_final_state_sha256=transient_invariance_config.parent_final_state_sha256,
        member_id=transient_invariance_config.member_id,
        member_fingerprint=transient_invariance_config.member_fingerprint,
        q=transient_invariance_config.q,
        primary_view_id=transient_invariance_config.primary_view_id,
        primary_view_fingerprint=transient_invariance_config.primary_view_fingerprint,
        secondary_view_id=transient_invariance_config.secondary_view_id,
        secondary_view_fingerprint=transient_invariance_config.secondary_view_fingerprint,
        primary_timestep=transient_invariance_config.primary_timestep,
        secondary_timestep=transient_invariance_config.secondary_timestep,
        online_start_step=transient_invariance_config.online_start_step,
        development_checkpoint_steps=transient_invariance_config.intervention_confirmation_phase_steps,
        receiver_cadence_steps=transient_invariance_config.receiver_cadence_steps,
        rolling_window_samples=transient_invariance_config.rolling_window_samples,
        phi_min=transient_invariance_config.phi_min,
        phi_max=transient_invariance_config.phi_max,
        closure_ratio_max=transient_invariance_config.closure_ratio_max,
        kernel_band_ratio_max=transient_invariance_config.kernel_band_ratio_max,
        persistence_pass_count=transient_invariance_config.persistence_pass_count,
        heldout_probe_field_indices=(6, 7, 8, 9, 10, 11),
        probe_kappas=transient_invariance_config.probe_kappas,
        probe_seed_rule_id=transient_invariance_config.probe_seed_rule_id,
        probe_roster_config_fingerprint=transient_invariance_config.fingerprint(),
        probe_scientific_input=probe_scientific_input,
        probe_numeric_floor=transient_invariance_config.probe_numeric_floor,
        probe_hermiticity_residual_max=transient_invariance_config.probe_hermiticity_residual_max,
        probe_trace_residual_max=transient_invariance_config.probe_trace_residual_max,
        probe_identity_drift_max=transient_invariance_config.probe_identity_drift_max,
        probe_norm_increase_max=transient_invariance_config.probe_norm_increase_max,
        probe_forecast_lag_steps=32,
        response_geometry_loss_max=transient_invariance_config.response_prediction_qualification_loss_pooled_max,
        response_radius_ratio_max=transient_invariance_config.response_prediction_qualification_radius_ratio_max,
        development_blocks=transient_invariance_config.intervention_confirmation_development_blocks,
        development_action_words=transient_invariance_config.intervention_confirmation_action_words,
        pulse_delta=transient_invariance_config.pulse_delta,
        pulse_ramp_steps=transient_invariance_config.pulse_ramp_steps,
        pulse_dwell_steps=transient_invariance_config.pulse_dwell_steps,
        pulse_return_steps=transient_invariance_config.pulse_return_steps,
        pulse_washout_steps=transient_invariance_config.pulse_washout_steps,
        pulse_alpha_min=transient_invariance_config.pulse_alpha_min,
        pulse_alpha_max=transient_invariance_config.pulse_alpha_max,
        pulse_increment_max=transient_invariance_config.pulse_increment_max,
        pulse_total_variation_max=transient_invariance_config.pulse_total_variation_max,
        pulse_squared_energy_max=transient_invariance_config.pulse_squared_energy_max,
        generalized_work_ceiling=transient_invariance_config.generalized_work_ceiling,
        pulse_count_max=transient_invariance_config.pulse_count_max,
        development_branch_steps=transient_invariance_config.intervention_confirmation_development_steps,
        development_assessment_offset=transient_invariance_config.intervention_confirmation_development_assessment_start,
        sustained_duration=transient_invariance_config.intervention_confirmation_sustained_duration,
        development_residence_gain_min=transient_invariance_config.intervention_confirmation_development_rmst_gain_min,
        development_risk_gain_min=transient_invariance_config.intervention_confirmation_development_risk_gain_min,
        rule_predicate_max=transient_invariance_config.rule_predicate_max,
        confirmation_blocks=transient_invariance_config.intervention_confirmation_confirmation_blocks,
        confirmation_branch_steps=transient_invariance_config.intervention_confirmation_confirmation_steps,
        confirmation_assessment_start_step=transient_invariance_config.intervention_confirmation_assessment_start_step,
        confirmation_assessment_end_step=transient_invariance_config.intervention_confirmation_assessment_end_step,
        confirmation_valid_pair_min=transient_invariance_config.intervention_confirmation_valid_pair_min,
        confirmation_risk_gain_min=transient_invariance_config.intervention_confirmation_risk_gain_min,
        confirmation_residence_gain_min=transient_invariance_config.intervention_confirmation_rmst_gain_min,
        numerical_audit_blocks=transient_invariance_config.intervention_confirmation_numerical_audit_blocks,
        numerical_classification_agreement_min=Decimal("0.90"),
        numerical_probe_state_difference_max=Decimal("0.05"),
        numerical_energy_relative_error_max=Decimal("0.01"),
        numerical_work_relative_error_max=Decimal("0.10"),
        bootstrap_resamples=transient_invariance_config.bootstrap_resamples,
        confidence_level=transient_invariance_config.confidence_level,
        worker_count_max=8,
        preferred_wall_seconds=14_400,
        hard_wall_seconds=43_200,
        maximum_memory_bytes=8 * 1024**3,
        maximum_output_bytes=2 * 1024**3,
        total_integration_steps_ceiling=967_680,
        maximum_scientific_trace_count=1_441,
        evidence_world="outcome-visible-event-selected-same-implementation-dimension-two",
        maximum_claim="event-conditional-full-intersection-causal-authority",
        nonclaims=(
            "autonomous-phase",
            "six-matrix-response-rescue",
            "controlled-invariance",
            "controller-relative-constitution",
            "cross-event-or-substrate-transport",
            "prospective-control-execution",
            "independent-event-confirmation",
            "base-system-prospective-evaluation-or-response-law",
            "physical-control",
            "production-controller",
        ),
        source_inaccessible_method_qualification=True,
        prospective_control_condition_false=True,
        grants_authority=False,
    )


__all__ = [
    'MatrixResponseCausalIntersectionResidenceStudyConfig',
    'build_matrix_response_study_causal_intersection_residence_study_config',
]
