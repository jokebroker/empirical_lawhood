"Frozen, nonauthorizing scientific design for the Six-matrix response isolated-event study."

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
class MatrixResponseShootingStudyConfig(CanonicalRecord):
    """All claim-bearing choices for Matrix shooting, with no dispatch or authority."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/matrix-response-study/matrix-response-shooting-study-config'

    config_id: str
    config_version: str
    plan_id: str
    parent_design_fingerprint: str
    parent_source_fingerprint: str
    parent_numerical_qualification_report_sha256: str
    parent_anisotropic_feasibility_report_locator: str
    parent_anisotropic_feasibility_report_sha256: str
    parent_anisotropic_feasibility_report_size_bytes: int
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
    parent_rng_seed_sha256: str
    primary_view_id: str
    primary_view_fingerprint: str
    secondary_view_id: str
    secondary_view_fingerprint: str
    primary_timestep: Decimal
    secondary_timestep: Decimal
    parent_total_steps: int
    parent_ramp_steps: int
    checkpoint_steps: tuple[int, ...]
    checkpoint_times: tuple[Decimal, ...]
    formation_checkpoint_steps: tuple[int, ...]
    source_admitted_checkpoint_steps: tuple[int, ...]
    linearization_checkpoint_step: int
    branch_seed_derivation_rule_id: str
    bridge_seed_derivation_rule_id: str
    half_step_action_interpolation_id: str
    hermitian_coordinate_basis_id: str
    branch_action_disposition: str
    geometric_target_label: str
    return_target_label: str
    branch_count_per_checkpoint: int
    branch_steps: int
    receiver_cadence_steps: int
    rolling_window_samples: int
    phi_min: Decimal
    phi_max: Decimal
    closure_ratio_max: Decimal
    kernel_band_ratio_max: Decimal
    persistence_pass_count: int
    hermiticity_residual_max: Decimal
    bridge_phi_y_difference_max: Decimal
    confidence_level: Decimal
    simultaneous_family_size: int
    sharp_rise_min: Decimal
    low_commitment_upper_max: Decimal
    nonmonotone_change_min: Decimal
    plateau_lower_min: Decimal
    plateau_consecutive_min: int
    unresolved_checkpoint_min: int
    resolution_fraction_min: Decimal
    residence_landmark: Decimal
    residence_rmst_min: Decimal
    residence_survival_min: Decimal
    residence_branch_min: int
    short_loss_time_max: Decimal
    short_loss_fraction_min: Decimal
    oscillation_branch_fraction_min: Decimal
    oscillation_transition_min: int
    oscillation_occupancy_min: Decimal
    oscillation_occupancy_max: Decimal
    hessian_scale: Decimal
    hessian_antisymmetry_max: Decimal
    quotient_relative_cutoff: Decimal
    unstable_real_part_min: Decimal
    unstable_eigenvalue_change_max: Decimal
    receiver_derivative_gap_min: Decimal
    receiver_derivative_change_max: Decimal
    y_projection_min: Decimal
    null_phi_target: Decimal
    null_bands: tuple[Decimal, ...]
    null_strata: tuple[str, ...]
    null_same_member_minimum: int
    expected_null_primary_counts: tuple[int, ...]
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
            "branch_seed_derivation_rule_id",
            "bridge_seed_derivation_rule_id",
            "half_step_action_interpolation_id",
            "hermitian_coordinate_basis_id",
            "branch_action_disposition",
            "evidence_world",
            "maximum_claim",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_semantic_version(self.config_version)
        for name in (
            "parent_design_fingerprint",
            "parent_source_fingerprint",
            "parent_numerical_qualification_report_sha256",
            "parent_anisotropic_feasibility_report_sha256",
            "parent_final_state_sha256",
            "member_fingerprint",
            "parent_rng_seed_sha256",
            "primary_view_fingerprint",
            "secondary_view_fingerprint",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        validate_relative_locator(self.parent_anisotropic_feasibility_report_locator)
        if self.parent_anisotropic_feasibility_report_size_bytes <= 0:
            raise ValueError("Matrix shooting parent anisotropic feasibility report byte count differs")
        if (self.q, self.alpha_x_index, self.alpha_y_index, self.seed_index) != (2, 1, 11, 1):
            raise ValueError("Matrix shooting selected event coordinate differs")
        if (self.target_alpha_tilde_x, self.target_alpha_tilde_y) != (
            Decimal("0.6666666666666666666666666667"),
            Decimal("7.333333333333333333333333334"),
        ):
            raise ValueError("Matrix shooting selected event target couplings differ")
        if self.checkpoint_steps != (816, 848, 880, 896, 928, 960, 1024):
            raise ValueError("Matrix shooting checkpoint roster differs")
        if (
            self.primary_timestep,
            self.secondary_timestep,
            self.parent_total_steps,
            self.parent_ramp_steps,
        ) != (Decimal("0.001"), Decimal("0.0005"), 1024, 256):
            raise ValueError("Matrix shooting parent/view time design differs")
        expected_times = tuple(
            Decimal(step) * self.primary_timestep for step in self.checkpoint_steps
        )
        if self.checkpoint_times != expected_times:
            raise ValueError("Matrix shooting checkpoint physical times differ")
        if (
            self.formation_checkpoint_steps,
            self.source_admitted_checkpoint_steps,
            self.linearization_checkpoint_step,
        ) != (
            (816, 848, 880, 896),
            (848, 880, 896, 928, 960, 1024),
            896,
        ):
            raise ValueError("Matrix shooting checkpoint scientific roles differ")
        if (
            self.branch_seed_derivation_rule_id,
            self.bridge_seed_derivation_rule_id,
            self.half_step_action_interpolation_id,
            self.hermitian_coordinate_basis_id,
            self.branch_action_disposition,
            self.geometric_target_label,
            self.return_target_label,
        ) != (
            "matrix-shooting.branch-scientific-input",
            "matrix-shooting.bridge-scientific-input",
            "matrix-shooting.parent-binary-float-physical-time",
            "matrix-shooting.hs-hermitian-coordinate-basis",
            "hold",
            "01",
            "00",
        ):
            raise ValueError("Matrix shooting branch/bridge semantic identity differs")
        if (
            self.branch_count_per_checkpoint,
            self.branch_steps,
            self.receiver_cadence_steps,
            self.rolling_window_samples,
        ) != (64, 1024, 16, 16):
            raise ValueError("Matrix shooting shooting cardinality differs")
        if self.persistence_pass_count != 12 or self.simultaneous_family_size != 7:
            raise ValueError("Matrix shooting rolling/simultaneous cardinality differs")
        decimal_fields = (
            "target_alpha_tilde_x",
            "target_alpha_tilde_y",
            "primary_timestep",
            "secondary_timestep",
            "phi_min",
            "phi_max",
            "closure_ratio_max",
            "kernel_band_ratio_max",
            "hermiticity_residual_max",
            "bridge_phi_y_difference_max",
            "confidence_level",
            "sharp_rise_min",
            "low_commitment_upper_max",
            "nonmonotone_change_min",
            "plateau_lower_min",
            "resolution_fraction_min",
            "residence_landmark",
            "residence_rmst_min",
            "residence_survival_min",
            "short_loss_time_max",
            "short_loss_fraction_min",
            "oscillation_branch_fraction_min",
            "oscillation_occupancy_min",
            "oscillation_occupancy_max",
            "hessian_scale",
            "hessian_antisymmetry_max",
            "quotient_relative_cutoff",
            "unstable_real_part_min",
            "unstable_eigenvalue_change_max",
            "receiver_derivative_gap_min",
            "receiver_derivative_change_max",
            "y_projection_min",
            "null_phi_target",
        )
        for name in decimal_fields:
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        for index, value in enumerate(self.null_bands):
            validate_decimal(value, field_name=f"null_bands[{index}]", minimum=Decimal(0))
        if self.null_bands != (
            Decimal("0.005"),
            Decimal("0.01"),
            Decimal("0.02"),
            Decimal("0.05"),
        ):
            raise ValueError("Matrix shooting null-band roster differs")
        if self.null_strata != (
            "all-c1a",
            "same-member",
            "same-member-y11",
            "same-member-x01-y11",
        ):
            raise ValueError("Matrix shooting null strata differ")
        if self.expected_null_primary_counts != (588, 1, 74, 1, 0, 70, 1, 31, 1, 0):
            raise ValueError("Matrix shooting planning null calibration differs")
        if (
            self.preferred_wall_seconds,
            self.hard_wall_seconds,
            self.maximum_memory_bytes,
            self.maximum_output_bytes,
            self.total_integration_steps,
        ) != (7200, 14400, 4 * 1024**3, 256 * 1024**2, 461_952):
            raise ValueError("Matrix shooting resource envelope differs")
        require_sorted_unique_strings(self.nonclaims, field_name="nonclaims", allow_empty=False)
        if self.grants_authority:
            raise ValueError("Matrix shooting config cannot grant execution authority")


def build_matrix_response_study_shooting_study_config(
    design: MatrixResponseDesignBinding, lineage: MatrixResponseLineageInputs
) -> MatrixResponseShootingStudyConfig:
    source = design.source_config
    member = next(
        value
        for value in source.anisotropic_model.family_members
        if value.member_id == "six-matrix-response.member.mass-0p5.cross-coupling-1"
    )
    return MatrixResponseShootingStudyConfig(
        config_id="matrix-shooting.isolated-event-shooting-committor",
        config_version="1.0.0",
        plan_id="matrix-isolated-event-shooting-committor-study",
        parent_design_fingerprint=design.fingerprint(),
        parent_source_fingerprint=source.fingerprint(),
        parent_numerical_qualification_report_sha256=(
            lineage.sha256('parent_numerical_qualification_report_sha256')
        ),
        parent_anisotropic_feasibility_report_locator=(
            lineage.locator('parent_anisotropic_feasibility_report_locator')
        ),
        parent_anisotropic_feasibility_report_sha256=(
            lineage.sha256('parent_anisotropic_feasibility_report_sha256')
        ),
        parent_anisotropic_feasibility_report_size_bytes=lineage.positive_int('parent_anisotropic_feasibility_report_size_bytes'),
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
        parent_rng_seed_sha256=(lineage.sha256('parent_rng_seed_sha256')),
        primary_view_id=source.primary_view.view_id,
        primary_view_fingerprint=source.primary_view.fingerprint(),
        secondary_view_id=source.secondary_view.view_id,
        secondary_view_fingerprint=source.secondary_view.fingerprint(),
        primary_timestep=source.primary_view.timestep,
        secondary_timestep=source.secondary_view.timestep,
        parent_total_steps=1024,
        parent_ramp_steps=256,
        checkpoint_steps=(816, 848, 880, 896, 928, 960, 1024),
        checkpoint_times=tuple(
            Decimal(step) / Decimal(1000) for step in (816, 848, 880, 896, 928, 960, 1024)
        ),
        formation_checkpoint_steps=(816, 848, 880, 896),
        source_admitted_checkpoint_steps=(848, 880, 896, 928, 960, 1024),
        linearization_checkpoint_step=896,
        branch_seed_derivation_rule_id="matrix-shooting.branch-scientific-input",
        bridge_seed_derivation_rule_id="matrix-shooting.bridge-scientific-input",
        half_step_action_interpolation_id="matrix-shooting.parent-binary-float-physical-time",
        hermitian_coordinate_basis_id="matrix-shooting.hs-hermitian-coordinate-basis",
        branch_action_disposition="hold",
        geometric_target_label="01",
        return_target_label="00",
        branch_count_per_checkpoint=64,
        branch_steps=1024,
        receiver_cadence_steps=16,
        rolling_window_samples=16,
        phi_min=Decimal("0.35"),
        phi_max=Decimal("0.95"),
        closure_ratio_max=Decimal("0.30"),
        kernel_band_ratio_max=Decimal("0.25"),
        persistence_pass_count=12,
        hermiticity_residual_max=Decimal("1e-12"),
        bridge_phi_y_difference_max=Decimal("0.10"),
        confidence_level=Decimal("0.95"),
        simultaneous_family_size=7,
        sharp_rise_min=Decimal("0.40"),
        low_commitment_upper_max=Decimal("0.25"),
        nonmonotone_change_min=Decimal("0.25"),
        plateau_lower_min=Decimal("0.50"),
        plateau_consecutive_min=3,
        unresolved_checkpoint_min=4,
        resolution_fraction_min=Decimal("0.50"),
        residence_landmark=Decimal("0.512"),
        residence_rmst_min=Decimal("0.384"),
        residence_survival_min=Decimal("0.25"),
        residence_branch_min=8,
        short_loss_time_max=Decimal("0.032"),
        short_loss_fraction_min=Decimal("0.75"),
        oscillation_branch_fraction_min=Decimal("0.50"),
        oscillation_transition_min=2,
        oscillation_occupancy_min=Decimal("0.25"),
        oscillation_occupancy_max=Decimal("0.75"),
        hessian_scale=Decimal("1e-6"),
        hessian_antisymmetry_max=Decimal("1e-5"),
        quotient_relative_cutoff=Decimal("1e-10"),
        unstable_real_part_min=Decimal("1e-6"),
        unstable_eigenvalue_change_max=Decimal("0.10"),
        receiver_derivative_gap_min=Decimal("1e-8"),
        receiver_derivative_change_max=Decimal("0.10"),
        y_projection_min=Decimal("0.50"),
        null_phi_target=Decimal("0.6910748486811329"),
        null_bands=(Decimal("0.005"), Decimal("0.01"), Decimal("0.02"), Decimal("0.05")),
        null_strata=(
            "all-c1a",
            "same-member",
            "same-member-y11",
            "same-member-x01-y11",
        ),
        null_same_member_minimum=32,
        expected_null_primary_counts=(588, 1, 74, 1, 0, 70, 1, 31, 1, 0),
        preferred_wall_seconds=7200,
        hard_wall_seconds=14400,
        maximum_memory_bytes=4 * 1024**3,
        maximum_output_bytes=256 * 1024**2,
        total_integration_steps=461_952,
        evidence_world="outcome-visible-event-selected-same-implementation-dimension-two",
        maximum_claim="same-selected-event-finite-horizon-dynamical-object",
        nonclaims=tuple(
            sorted(
                (
                    "asymptotic-metastability",
                    "six-matrix-response-rescue",
                    "constitutive-phase-prevalence",
                    "controller-efficacy",
                    "independent-event-recurrence",
                    "base-system-prospective-evaluation-or-response-law",
                    "physical-or-independent-substrate-evidence",
                )
            )
        ),
        grants_authority=False,
    )


__all__ = [
    'MatrixResponseShootingStudyConfig',
    'build_matrix_response_study_shooting_study_config',
]
