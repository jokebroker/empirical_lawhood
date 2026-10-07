"""Bounded target-owned analytic uniform electron gas development input.

This exercises the retained transverse-response calculation without treating
the source project's truth-known roster as a fresh prospective experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    validate_decimal,
    validate_stable_id,
)

from .contracts import ActionCurrentRow, Vector3
from .physics import finite_q_estimate, fit_penetration_depth_from_slab, kernel_from_penetration_depth, magnetic_field_amplitude, norm, q_intercept_stability, row_map, slab_profile, transverse_relative_dot, ueg_scales, vector_potential_from_u, vector_scale

_THRESHOLD_KEYS = frozenset(
    (
        "action_realization_relative",
        "amplitude_locality_relative",
        "baseline_relative",
        "even_remainder_floor_multiplier",
        "even_remainder_response_relative",
        "normal_cancellation_relative",
        "order_preservation_fraction",
        "q_intercept_stability_relative",
        "response_floor_relative_to_diamagnetic",
        "shielding_score_minimum",
        "slab_fit_relative",
        "transverse_dot_relative",
        "view_k0_agreement_relative",
        "ward_residual",
    )
)


@dataclass(frozen=True, slots=True)
class UniformElectronGasAnalyticReferenceConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/uniform-electron-gas-response/analytic-reference-config'

    config_id: str
    independent_unit_id: str
    r_s: Decimal
    temperature_K: Decimal
    u0: Decimal
    q_over_kf: tuple[Decimal, Decimal, Decimal, Decimal]
    action_direction: Vector3
    q_direction: Vector3
    requested_clock: int
    accepted_clock: int
    applied_clock: int
    receiver_clock: int
    slab_thickness_m: Decimal
    reference_penetration_depth_m: Decimal
    analytic_profile_points: int
    field_ceiling_T: Decimal
    deterministic_kernel_relative_error: Decimal
    thresholds: tuple[tuple[str, Decimal], ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_stable_id(self.independent_unit_id, field_name="independent_unit_id")
        if not self.config_id.startswith(
            "empirical-lawhood-"
        ) or not self.independent_unit_id.startswith("unit.empirical-lawhood-"):
            raise ValueError("uniform electron gas native check needs new target-owned identities")
        for name in (
            "r_s",
            "temperature_K",
            "u0",
            "slab_thickness_m",
            "reference_penetration_depth_m",
            "field_ceiling_T",
            "deterministic_kernel_relative_error",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        if not (
            Decimal("0.1") <= self.r_s <= Decimal(100)
            and self.temperature_K <= Decimal(10000)
            and Decimal(0) < self.u0 <= Decimal("0.001")
            and Decimal(0) < self.slab_thickness_m <= Decimal("0.01")
            and Decimal(0) < self.reference_penetration_depth_m <= Decimal("0.01")
            and self.field_ceiling_T > 0
            and Decimal(0) <= self.deterministic_kernel_relative_error < Decimal("0.1")
        ):
            raise ValueError(
                "uniform electron gas denominator, action or reference geometry is outside bounds"
            )
        for index, q_value in enumerate(self.q_over_kf):
            validate_decimal(q_value, field_name=f"q_over_kf[{index}]")
        if not (
            Decimal(0)
            < self.q_over_kf[0]
            < self.q_over_kf[1]
            < self.q_over_kf[2]
            < self.q_over_kf[3]
            <= Decimal("0.25")
        ):
            raise ValueError(
                "uniform electron gas finite-q chart needs four increasing positive values"
            )
        for name in ("action_direction", "q_direction"):
            direction = getattr(self, name)
            if len(direction) != 3:
                raise ValueError(f"{name} must be a three-vector")
            for value in direction:
                validate_decimal(value, field_name=name)
            if abs(norm(direction) - Decimal(1)) > Decimal("1e-12"):
                raise ValueError(f"{name} must be a unit vector")
        if transverse_relative_dot(self.q_direction, self.action_direction) > Decimal(
            "1e-12"
        ):
            raise ValueError(
                "uniform electron gas q and transverse action directions are not orthogonal"
            )
        if any(
            type(clock) is not int
            for clock in (
                self.requested_clock,
                self.accepted_clock,
                self.applied_clock,
                self.receiver_clock,
            )
        ) or not (
            0
            <= self.requested_clock
            < self.accepted_clock
            < self.applied_clock
            < self.receiver_clock
        ):
            raise ValueError(
                "requested, accepted, applied and receiver clocks must be ordered"
            )
        if (
            type(self.analytic_profile_points) is not int
            or not 3 <= self.analytic_profile_points <= 1001
            or self.analytic_profile_points % 2 == 0
        ):
            raise ValueError("slab profile needs an odd bounded point count")
        keys = tuple(key for key, _ in self.thresholds)
        if keys != tuple(sorted(_THRESHOLD_KEYS)):
            raise ValueError("uniform electron gas threshold keys must be complete, sorted and unique")
        for key, value in self.thresholds:
            validate_decimal(value, field_name=key, minimum=Decimal(0))

    def threshold(self, key: str) -> Decimal:
        try:
            return dict(self.thresholds)[key]
        except KeyError as error:
            raise KeyError(f"unknown uniform electron gas threshold {key}") from error


def build_analytic_development_rows(
    config: UniformElectronGasAnalyticReferenceConfig,
) -> tuple[ActionCurrentRow, ...]:
    """One synthetic positive-reference acquisition; q and u are nested conditions."""

    scales = ueg_scales(config.r_s)
    kernel0 = kernel_from_penetration_depth(config.reference_penetration_depth_m)
    u_values = (-config.u0, -config.u0 / 2, Decimal(0), config.u0 / 2, config.u0)
    maximum_amplitude = abs(vector_potential_from_u(config.u0, scales))
    rows: list[ActionCurrentRow] = []
    for q_index, q_value in enumerate(config.q_over_kf):
        kernel_q = kernel0 * (Decimal(1) + Decimal(4) * q_value**2)
        for u_index, u_value in enumerate(u_values):
            amplitude = vector_potential_from_u(u_value, scales)
            action = vector_scale(config.action_direction, amplitude)
            rows.append(
                ActionCurrentRow(
                    row_id=f"row.uniform-electron-gas-native-q{q_index:02d}-u{u_index:02d}",
                    q_over_kf=q_value,
                    u=u_value,
                    requested_A_T=action,
                    accepted_A_T=action,
                    applied_A_T=action,
                    realized_A_T=action,
                    current_density=vector_scale(
                        config.action_direction, -kernel_q * amplitude
                    ),
                    current_error_bound_A_m2=(
                        kernel_q
                        * maximum_amplitude
                        * config.deterministic_kernel_relative_error
                    ),
                    requested_clock=config.requested_clock,
                    accepted_clock=config.accepted_clock,
                    applied_clock=config.applied_clock,
                    receiver_clock=config.receiver_clock,
                    accepted=True,
                    valid=True,
                )
            )
    if (
        max(
            magnetic_field_amplitude(row.q_over_kf, row.realized_A_T, scales)
            for row in rows
        )
        > config.field_ceiling_T
    ):
        raise ValueError("realized transverse field exceeds the predeclared ceiling")
    return tuple(rows)


def run_analytic_development_check(config: UniformElectronGasAnalyticReferenceConfig) -> dict[str, object]:
    """Exercise the retained SI finite-q and slab transforms without native contact."""

    scales = ueg_scales(config.r_s)
    rows = build_analytic_development_rows(config)
    mapped = row_map(rows)
    estimates = tuple(
        finite_q_estimate(
            panel_id=config.config_id,
            q_over_kf=q_value,
            rows=mapped,
            config=config,
            action_direction=config.action_direction,
        )
        for q_value in config.q_over_kf
    )
    all_fit, small_fit, stability = q_intercept_stability(
        estimates,
        kernel_floor=scales.diamagnetic_kernel_A_T_m3
        * config.threshold("response_floor_relative_to_diamagnetic"),
    )
    profile = slab_profile(
        thickness_m=config.slab_thickness_m,
        penetration_depth_m=config.reference_penetration_depth_m,
        points=config.analytic_profile_points,
    )
    fitted_depth = fit_penetration_depth_from_slab(
        thickness_m=config.slab_thickness_m, profile=profile
    )
    return {
        "config_id": config.config_id,
        "independent_unit_id": config.independent_unit_id,
        "independent_units": 1,
        "nested_q_u_conditions": len(rows),
        "density_m3": str(scales.density_m3),
        "k_f_m1": str(scales.k_f_m1),
        "requested_u_values": [
            str(value)
            for value in (
                -config.u0,
                -config.u0 / 2,
                Decimal(0),
                config.u0 / 2,
                config.u0,
            )
        ],
        "action_unit": "T*m",
        "receiver_unit": "A/m^2",
        "kernel_unit": "A/(T*m^3)",
        "requested_clock": config.requested_clock,
        "accepted_clock": config.accepted_clock,
        "applied_clock": config.applied_clock,
        "receiver_clock": config.receiver_clock,
        "finite_q_kernel_A_T_m3": [str(value.kernel_full) for value in estimates],
        "kernel_intercept_A_T_m3": str(all_fit.kernel0),
        "three_q_intercept_A_T_m3": str(small_fit.kernel0),
        "intercept_stability_relative": str(stability),
        "slab_centre_field_ratio": str(profile[len(profile) // 2][1]),
        "fitted_penetration_depth_m": str(fitted_depth),
        "analytic_reference_only": True,
        "campaign_candidate_compiled": False,
        "campaign_issued": False,
    }


__all__ = [
    'UniformElectronGasAnalyticReferenceConfig',
    "build_analytic_development_rows",
    "run_analytic_development_check",
]
