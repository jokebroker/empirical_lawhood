"""uniform electron gas analytic acquisition and independent-panel falsifier on realized SI actions."""

from decimal import Decimal

from .analytic_contracts import UniformElectronGasAnalyticCheck, UniformElectronGasAnalyticPanel
from .native_quickstart import UniformElectronGasAnalyticReferenceConfig, build_analytic_development_rows
from .physics import finite_q_estimate, fit_penetration_depth_from_slab, magnetic_field_amplitude, q_intercept_stability, relative_vector_difference, row_map, slab_profile, ueg_scales


def generate_analytic_panel(config: UniformElectronGasAnalyticReferenceConfig) -> UniformElectronGasAnalyticPanel:
    """Produce one disclosed synthetic acquisition; q/u cells are nested conditions."""

    return UniformElectronGasAnalyticPanel(
        panel_id=f"panel.{config.config_id}",
        config_sha256=config.fingerprint(),
        independent_unit_id=config.independent_unit_id,
        rows=build_analytic_development_rows(config),
        slab_profile=slab_profile(
            thickness_m=config.slab_thickness_m,
            penetration_depth_m=config.reference_penetration_depth_m,
            points=config.analytic_profile_points,
        ),
        synthetic_reference=True,
    )


def check_analytic_panel(
    config: UniformElectronGasAnalyticReferenceConfig, panel: UniformElectronGasAnalyticPanel
) -> UniformElectronGasAnalyticCheck:
    """Evaluate stage, transverse, q-limit and slab falsifiers without promotion."""

    if (
        panel.config_sha256 != config.fingerprint()
        or panel.independent_unit_id != config.independent_unit_id
    ):
        raise ValueError("uniform electron gas panel is bound to another frozen configuration or unit")
    if len(panel.slab_profile) != config.analytic_profile_points:
        raise ValueError("uniform electron gas slab profile point count differs from config")
    expected = {
        (q, u)
        for q in config.q_over_kf
        for u in (-config.u0, -config.u0 / 2, Decimal(0), config.u0 / 2, config.u0)
    }
    mapped = row_map(panel.rows)
    if set(mapped) != expected:
        raise ValueError("uniform electron gas panel q/action chart differs from frozen design")
    reasons: list[str] = []
    for row in panel.rows:
        if not row.valid or not row.accepted:
            reasons.append("ACTION_NOT_ACCEPTED")
        if (
            row.requested_clock,
            row.accepted_clock,
            row.applied_clock,
            row.receiver_clock,
        ) != (
            config.requested_clock,
            config.accepted_clock,
            config.applied_clock,
            config.receiver_clock,
        ):
            reasons.append("ACTION_RECEIVER_CLOCK_DRIFT")
        if row.requested_A_T != row.accepted_A_T or row.accepted_A_T != row.applied_A_T:
            reasons.append("ACTION_DELIVERY_DRIFT")
        if relative_vector_difference(
            row.applied_A_T,
            row.realized_A_T,
            floor=Decimal("1e-300"),
        ) > config.threshold("action_realization_relative"):
            reasons.append("REALIZED_ACTION_DRIFT")
    scales = ueg_scales(config.r_s)
    if (
        max(
            magnetic_field_amplitude(row.q_over_kf, row.realized_A_T, scales)
            for row in panel.rows
        )
        > config.field_ceiling_T
    ):
        reasons.append("FIELD_CEILING_EXCEEDED")
    estimates = tuple(
        finite_q_estimate(
            panel_id=panel.panel_id,
            q_over_kf=q,
            rows=mapped,
            config=config,
            action_direction=config.action_direction,
        )
        for q in config.q_over_kf
    )
    if any(
        not (
            estimate.locality_pass
            and estimate.even_remainder_pass
            and estimate.zero_offset_pass
        )
        for estimate in estimates
    ):
        reasons.append("FINITE_Q_LOCALITY_OR_EVEN_REMAINDER_FAILED")
    fit, _, stability = q_intercept_stability(
        estimates,
        kernel_floor=scales.diamagnetic_kernel_A_T_m3
        * config.threshold("response_floor_relative_to_diamagnetic"),
    )
    if fit.kernel0 <= fit.error_bound:
        reasons.append("NONPOSITIVE_KERNEL_LOWER_BOUND")
    if stability > config.threshold("q_intercept_stability_relative"):
        reasons.append("Q_INTERCEPT_UNSTABLE")
    depth: Decimal | None
    try:
        depth = fit_penetration_depth_from_slab(
            thickness_m=config.slab_thickness_m, profile=panel.slab_profile
        )
    except ValueError:
        depth = None
        reasons.append("SLAB_INVERSE_FAILED")
    if depth is not None and abs(
        depth - config.reference_penetration_depth_m
    ) / config.reference_penetration_depth_m > config.threshold("slab_fit_relative"):
        reasons.append("SLAB_FIT_FAILED")
    return UniformElectronGasAnalyticCheck(
        check_id=f"check.{panel.panel_id}",
        independent_unit_id=panel.independent_unit_id,
        kernel_intercept_A_T_m3=fit.kernel0,
        q_intercept_stability_relative=stability,
        slab_depth_m=depth,
        passed=not reasons,
        reason_codes=tuple(sorted(set(reasons))),
        synthetic_reference=True,
    )


__all__ = ["check_analytic_panel", "generate_analytic_panel"]
