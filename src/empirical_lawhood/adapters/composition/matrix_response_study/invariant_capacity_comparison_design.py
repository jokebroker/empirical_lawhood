"""Frozen, nonauthorizing design for the Matrix invariant capacity comparison Response-prediction qualification comparator resolution."""

from __future__ import annotations

from dataclasses import dataclass, replace
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

from .transient_controlled_invariance_design import MatrixResponseTransientControlledInvarianceStudyConfig
from .lineage_inputs import MatrixResponseLineageInputs


@dataclass(frozen=True, slots=True)
class InvariantCapacityComparatorStudyConfig(CanonicalRecord):
    "Every claim-bearing fixed choice for the comparator-only study."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/matrix-response/invariant-capacity-comparator-study-config'

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
    parent_transient_response_source_manifest_sha256: str
    parent_transient_response_stage_manifest_sha256: str
    parent_transient_response_terminal_sha256: str
    parent_transient_response_artifact_manifest_sha256: str
    parent_transient_response_roster_sha256: str
    parent_transient_response_checkpoints_sha256: str
    parent_transient_response_probes_sha256: str
    parent_transient_response_scores_sha256: str
    parent_transient_response_joint_sha256: str
    parent_rollout_id: str
    member_id: str
    member_fingerprint: str
    parent_design_fingerprint: str
    parent_source_fingerprint: str
    q: int
    primary_timestep: Decimal
    secondary_timestep: Decimal
    probe_start_step: int
    probe_end_step: int
    probe_field_count: int
    probe_development_count: int
    probe_kappas: tuple[Decimal, ...]
    probe_seed_rule_id: str
    probe_numeric_floor: Decimal
    probe_resolved_fraction_min: Decimal
    probe_kappa_resolved_fraction_min: Decimal
    probe_half_state_difference_max: Decimal
    probe_half_order_agreement_min: Decimal
    probe_conjugation_error_max: Decimal
    probe_hermiticity_residual_max: Decimal
    probe_trace_residual_max: Decimal
    probe_identity_drift_max: Decimal
    probe_norm_increase_max: Decimal
    forecast_origin_steps: tuple[int, ...]
    forecast_horizon_steps: tuple[int, ...]
    matched_control_primary_count: int
    matched_control_reserve_count: int
    half_step_control_count: int
    comparator_family_id: str
    regularization_rule_id: str
    admissibility_rule_id: str
    method_qualification_rule_id: str
    generic_gram_rank: int
    generic_intrinsic_dimension: int
    generic_shape_dimension: int
    generic_lambda_grid: tuple[Decimal, ...]
    operator_tolerance: Decimal
    trace_floor: Decimal
    projection_gap_min: Decimal
    gate_y_ratio_max: Decimal
    equivalence_ratio_upper: Decimal
    gate_field_min: int
    event_rank_required: int
    method_fixture_ids: tuple[str, ...]
    preferred_wall_seconds: int
    hard_wall_seconds: int
    maximum_memory_bytes: int
    maximum_output_bytes: int
    worker_count_max: int
    expected_primary_candidate_fits: int
    expected_half_candidate_fits: int
    expected_conjugation_candidate_fits: int
    expected_total_candidate_fits: int
    evidence_world: str
    maximum_claim: str
    nonclaims: tuple[str, ...]
    source_inaccessible_method_qualification: bool
    intervention_confirmation_scientific_precondition_only: bool
    permits_intervention_confirmation_execution: bool
    permits_parent_action: bool
    grants_authority: bool

    def __post_init__(self) -> None:
        for name in (
            "config_id",
            "plan_id",
            "parent_rollout_id",
            "member_id",
            "probe_seed_rule_id",
            "comparator_family_id",
            "regularization_rule_id",
            "admissibility_rule_id",
            "method_qualification_rule_id",
            "evidence_world",
            "maximum_claim",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_semantic_version(self.config_version)
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
            "parent_transient_response_source_manifest_sha256",
            "parent_transient_response_stage_manifest_sha256",
            "parent_transient_response_terminal_sha256",
            "parent_transient_response_artifact_manifest_sha256",
            "parent_transient_response_roster_sha256",
            "parent_transient_response_checkpoints_sha256",
            "parent_transient_response_probes_sha256",
            "parent_transient_response_scores_sha256",
            "parent_transient_response_joint_sha256",
            "member_fingerprint",
            "parent_design_fingerprint",
            "parent_source_fingerprint",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        if len(self.accepted_platform_commit) != 40:
            raise ValueError("Matrix invariant capacity comparison accepted platform commit must be a Git SHA-1")
        int(self.accepted_platform_commit, 16)
        for name in (
            "accepted_platform_receipt_locator",
            "parent_numerical_qualification_report_locator",
            "parent_anisotropic_feasibility_report_locator",
            "parent_shooting_root_locator",
            "parent_transient_response_root_locator",
        ):
            validate_relative_locator(getattr(self, name))
        require_sorted_unique_strings(
            self.accepted_platform_capability_receipt_ids,
            field_name="accepted_platform_capability_receipt_ids",
            allow_empty=False,
        )
        if self.parent_numerical_qualification_report_size_bytes <= 0 or self.parent_anisotropic_feasibility_report_size_bytes <= 0:
            raise ValueError("Matrix invariant capacity comparison parent report size differs")
        for name in (
            "primary_timestep",
            "secondary_timestep",
            "probe_numeric_floor",
            "probe_resolved_fraction_min",
            "probe_kappa_resolved_fraction_min",
            "probe_half_state_difference_max",
            "probe_half_order_agreement_min",
            "probe_conjugation_error_max",
            "probe_hermiticity_residual_max",
            "probe_trace_residual_max",
            "probe_identity_drift_max",
            "probe_norm_increase_max",
            "operator_tolerance",
            "trace_floor",
            "projection_gap_min",
            "gate_y_ratio_max",
            "equivalence_ratio_upper",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        for index, value in enumerate((*self.probe_kappas, *self.generic_lambda_grid)):
            validate_decimal(value, field_name=f"design_decimal[{index}]", minimum=Decimal(0))
        if (
            self.q,
            self.probe_start_step,
            self.probe_end_step,
            self.probe_field_count,
            self.probe_development_count,
        ) != (2, 816, 1024, 12, 6):
            raise ValueError("Matrix invariant capacity comparison probe denominator differs")
        if self.primary_timestep != Decimal("0.001") or self.secondary_timestep != Decimal(
            "0.0005"
        ):
            raise ValueError("Matrix invariant capacity comparison numerical views differ")
        if self.probe_kappas != (Decimal("0.25"), Decimal("0.5"), Decimal("1")):
            raise ValueError("Matrix invariant capacity comparison kappa roster differs")
        if self.forecast_origin_steps != (816, 848, 880, 896, 928, 960):
            raise ValueError("Matrix invariant capacity comparison origin roster differs")
        if self.forecast_horizon_steps != (16, 32, 64):
            raise ValueError("Matrix invariant capacity comparison horizon roster differs")
        if (
            self.matched_control_primary_count,
            self.matched_control_reserve_count,
            self.half_step_control_count,
        ) != (20, 2, 4):
            raise ValueError("Matrix invariant capacity comparison control roster differs")
        if (
            self.generic_gram_rank,
            self.generic_intrinsic_dimension,
            self.generic_shape_dimension,
        ) != (3, 42, 41):
            raise ValueError("Matrix invariant capacity comparison comparator capacity differs")
        if self.generic_lambda_grid != (
            Decimal("0"),
            Decimal("1e-8"),
            Decimal("1e-7"),
            Decimal("1e-6"),
            Decimal("1e-5"),
            Decimal("1e-4"),
            Decimal("1e-3"),
            Decimal("1e-2"),
            Decimal("1e-1"),
            Decimal("1"),
            Decimal("10"),
            Decimal("100"),
        ):
            raise ValueError("Matrix invariant capacity comparison regularization grid differs")
        if (
            self.operator_tolerance,
            self.trace_floor,
            self.projection_gap_min,
            self.gate_y_ratio_max,
            self.equivalence_ratio_upper,
            self.gate_field_min,
            self.event_rank_required,
        ) != (
            Decimal("1e-10"),
            Decimal("1e-12"),
            Decimal("1e-10"),
            Decimal("0.80"),
            Decimal("1.25"),
            5,
            1,
        ):
            raise ValueError("Matrix invariant capacity comparison adjudication contract differs")
        if any(
            value > Decimal(1)
            for value in (
                self.probe_resolved_fraction_min,
                self.probe_kappa_resolved_fraction_min,
                self.probe_half_order_agreement_min,
            )
        ):
            raise ValueError("Matrix invariant capacity comparison fraction exceeds one")
        if (
            self.preferred_wall_seconds,
            self.hard_wall_seconds,
            self.maximum_memory_bytes,
            self.maximum_output_bytes,
            self.worker_count_max,
        ) != (1800, 7200, 4 * 1024**3, 256 * 1024**2, 8):
            raise ValueError("Matrix invariant capacity comparison resource envelope differs")
        if (
            self.expected_primary_candidate_fits,
            self.expected_half_candidate_fits,
            self.expected_conjugation_candidate_fits,
            self.expected_total_candidate_fits,
        ) != (10_584, 2_520, 504, 13_608):
            raise ValueError("Matrix invariant capacity comparison work accounting differs")
        require_sorted_unique_strings(
            self.method_fixture_ids, field_name="method_fixture_ids", allow_empty=False
        )
        require_sorted_unique_strings(self.nonclaims, field_name="nonclaims", allow_empty=False)
        if (
            not self.source_inaccessible_method_qualification
            or not self.intervention_confirmation_scientific_precondition_only
            or self.permits_intervention_confirmation_execution
            or self.permits_parent_action
            or self.grants_authority
        ):
            raise ValueError("Matrix invariant capacity comparison authority/effect boundary differs")


def build_invariant_capacity_comparator_study(
    parent: MatrixResponseTransientControlledInvarianceStudyConfig, lineage: MatrixResponseLineageInputs
) -> InvariantCapacityComparatorStudyConfig:
    return InvariantCapacityComparatorStudyConfig(
        config_id="matrix-invariant-capacity-comparison.raw-coefficient-study.config",
        config_version="1.0.0",
        plan_id="matrix-invariant-capacity-comparison.raw-coefficient-study.plan",
        accepted_platform_commit=parent.accepted_platform_commit,
        accepted_platform_archive_sha256=(
            lineage.sha256('accepted_platform_archive_sha256')
        ),
        accepted_platform_receipt_locator=parent.accepted_platform_receipt_locator,
        accepted_platform_receipt_sha256=parent.accepted_platform_receipt_sha256,
        accepted_platform_capability_receipt_ids=lineage.stable_ids('accepted_platform_capability_receipt_ids'),
        parent_numerical_qualification_report_locator=parent.parent_numerical_qualification_report_locator,
        parent_numerical_qualification_report_sha256=parent.parent_numerical_qualification_report_sha256,
        parent_numerical_qualification_report_size_bytes=parent.parent_numerical_qualification_report_size_bytes,
        parent_anisotropic_feasibility_report_locator=parent.parent_anisotropic_feasibility_report_locator,
        parent_anisotropic_feasibility_report_sha256=parent.parent_anisotropic_feasibility_report_sha256,
        parent_anisotropic_feasibility_report_size_bytes=parent.parent_anisotropic_feasibility_report_size_bytes,
        parent_shooting_root_locator=parent.parent_shooting_root_locator,
        parent_shooting_config_sha256=parent.parent_shooting_config_sha256,
        parent_shooting_source_manifest_sha256=parent.parent_shooting_source_manifest_sha256,
        parent_shooting_terminal_sha256=parent.parent_shooting_terminal_sha256,
        parent_shooting_artifact_manifest_sha256=parent.parent_shooting_artifact_manifest_sha256,
        parent_transient_response_root_locator=(lineage.locator('parent_transient_response_root_locator')),
        parent_transient_response_config_sha256=parent.fingerprint(),
        parent_transient_response_source_manifest_sha256=(
            lineage.sha256('parent_transient_response_source_manifest_sha256')
        ),
        parent_transient_response_stage_manifest_sha256=(
            lineage.sha256('parent_transient_response_stage_manifest_sha256')
        ),
        parent_transient_response_terminal_sha256=(
            lineage.sha256('parent_transient_response_terminal_sha256')
        ),
        parent_transient_response_artifact_manifest_sha256=(
            lineage.sha256('parent_transient_response_artifact_manifest_sha256')
        ),
        parent_transient_response_roster_sha256=(
            lineage.sha256('parent_transient_response_roster_sha256')
        ),
        parent_transient_response_checkpoints_sha256=(
            lineage.sha256('parent_transient_response_checkpoints_sha256')
        ),
        parent_transient_response_probes_sha256=(
            lineage.sha256('parent_transient_response_probes_sha256')
        ),
        parent_transient_response_scores_sha256=(
            lineage.sha256('parent_transient_response_scores_sha256')
        ),
        parent_transient_response_joint_sha256=(lineage.sha256('parent_transient_response_joint_sha256')),
        parent_rollout_id=parent.parent_rollout_id,
        member_id=parent.member_id,
        member_fingerprint=parent.member_fingerprint,
        parent_design_fingerprint=parent.parent_design_fingerprint,
        parent_source_fingerprint=parent.parent_source_fingerprint,
        q=parent.q,
        primary_timestep=parent.primary_timestep,
        secondary_timestep=parent.secondary_timestep,
        probe_start_step=parent.probe_start_step,
        probe_end_step=parent.probe_end_step,
        probe_field_count=parent.probe_field_count,
        probe_development_count=parent.probe_development_count,
        probe_kappas=parent.probe_kappas,
        probe_seed_rule_id=parent.probe_seed_rule_id,
        probe_numeric_floor=parent.probe_numeric_floor,
        probe_resolved_fraction_min=parent.probe_resolved_fraction_min,
        probe_kappa_resolved_fraction_min=parent.probe_kappa_resolved_fraction_min,
        probe_half_state_difference_max=parent.probe_half_state_difference_max,
        probe_half_order_agreement_min=parent.probe_half_order_agreement_min,
        probe_conjugation_error_max=parent.probe_conjugation_error_max,
        probe_hermiticity_residual_max=parent.probe_hermiticity_residual_max,
        probe_trace_residual_max=parent.probe_trace_residual_max,
        probe_identity_drift_max=parent.probe_identity_drift_max,
        probe_norm_increase_max=parent.probe_norm_increase_max,
        forecast_origin_steps=parent.forecast_origin_steps,
        forecast_horizon_steps=parent.forecast_horizon_steps,
        matched_control_primary_count=parent.matched_control_primary_count,
        matched_control_reserve_count=parent.matched_control_reserve_count,
        half_step_control_count=parent.response_prediction_qualification_half_control_count,
        comparator_family_id="matrix-invariant-capacity-comparison.rank-three-gram-commutator",
        regularization_rule_id="matrix-invariant-capacity-comparison.raw-coefficient-frobenius-one-se",
        admissibility_rule_id="matrix-invariant-capacity-comparison.invariant-operator-admissibility",
        method_qualification_rule_id="matrix-invariant-capacity-comparison.raw-coefficient-reference-world",
        generic_gram_rank=3,
        generic_intrinsic_dimension=42,
        generic_shape_dimension=41,
        generic_lambda_grid=(
            Decimal("0"),
            Decimal("1e-8"),
            Decimal("1e-7"),
            Decimal("1e-6"),
            Decimal("1e-5"),
            Decimal("1e-4"),
            Decimal("1e-3"),
            Decimal("1e-2"),
            Decimal("1e-1"),
            Decimal("1"),
            Decimal("10"),
            Decimal("100"),
        ),
        operator_tolerance=Decimal("1e-10"),
        trace_floor=Decimal("1e-12"),
        projection_gap_min=Decimal("1e-10"),
        gate_y_ratio_max=Decimal("0.80"),
        equivalence_ratio_upper=Decimal("1.25"),
        gate_field_min=5,
        event_rank_required=1,
        method_fixture_ids=(
            "matrix-invariant-capacity-comparison.fixture.candidate-local-invalidity",
            "matrix-invariant-capacity-comparison.fixture.factor-gauge-invariance",
            "matrix-invariant-capacity-comparison.fixture.generic-dominant-terminal",
            "matrix-invariant-capacity-comparison.fixture.generic-equivalent-terminal",
            "matrix-invariant-capacity-comparison.fixture.identity-kernel",
            "matrix-invariant-capacity-comparison.fixture.induced-norm-bound",
            "matrix-invariant-capacity-comparison.fixture.no-admissible-candidate",
            "matrix-invariant-capacity-comparison.fixture.operator-map-direct-equality",
            "matrix-invariant-capacity-comparison.fixture.semigroup-contractivity",
            "matrix-invariant-capacity-comparison.fixture.unitary-covariance",
            "matrix-invariant-capacity-comparison.fixture.y-privileged-terminal",
        ),
        preferred_wall_seconds=1800,
        hard_wall_seconds=7200,
        maximum_memory_bytes=4 * 1024**3,
        maximum_output_bytes=256 * 1024**2,
        worker_count_max=8,
        expected_primary_candidate_fits=10_584,
        expected_half_candidate_fits=2_520,
        expected_conjugation_candidate_fits=504,
        expected_total_candidate_fits=13_608,
        evidence_world="result-selected-same-implementation-event-conditional",
        maximum_claim="event-conditional-transient-response-geometry-supported",
        nonclaims=(
            "autonomous-phase",
            "six-matrix-response-rescue",
            "controlled-invariance",
            "controller-relative-constitution",
            "cross-event-or-substrate-transport",
            "intervention-confirmation-execution",
            "independent-confirmation",
            "base-system-prospective-evaluation-or-response-law",
            "parent-control-authority",
            "physical-control",
        ),
        source_inaccessible_method_qualification=True,
        intervention_confirmation_scientific_precondition_only=True,
        permits_intervention_confirmation_execution=False,
        permits_parent_action=False,
        grants_authority=False,
    )


def bind_covariant_operator_comparator(parent: InvariantCapacityComparatorStudyConfig) -> InvariantCapacityComparatorStudyConfig:
    """Bind the covariance-correct operator-coordinate implementation identity."""

    return replace(
        parent,
        config_id="matrix-invariant-capacity-comparison.covariance-correct-operator-study.config",
        config_version="1.0.0",
        plan_id="matrix-invariant-capacity-comparison.covariance-correct-operator-study.plan",
        regularization_rule_id="matrix-invariant-capacity-comparison.covariance-correct-augmented-svd-one-se",
        method_qualification_rule_id=(
            "matrix-invariant-capacity-comparison.covariance-correct-fitted-reference-world"
        ),
    )


__all__ = [
    'InvariantCapacityComparatorStudyConfig',
    'build_invariant_capacity_comparator_study',
    'bind_covariant_operator_comparator',
]
