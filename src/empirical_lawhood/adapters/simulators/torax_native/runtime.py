"""Late-imported direct TORAX 1.4.2 native execution."""

from __future__ import annotations

from decimal import Decimal
from hashlib import sha256
import math
from typing import Any

import numpy as np

from .contracts import NativeToraxAction, NativeToraxEpisodeDisposition, NativeToraxEpisode, NativeToraxPreparation, NativeToraxTrajectory, NativeToraxView


def _profile(
    radial_coordinates: tuple[Decimal, ...],
    values: tuple[Decimal, ...],
    *,
    scale: Decimal = Decimal(1),
) -> dict[float, dict[float, float]]:
    return {
        0.0: {
            float(radius): float(value * scale)
            for radius, value in zip(radial_coordinates, values, strict=True)
        }
    }


def build_native_torax_config(
    preparation: NativeToraxPreparation,
    action: NativeToraxAction,
    view: NativeToraxView,
) -> object:
    """Build an exact direct-TORAX configuration without execution."""

    view.operational_admission()
    if action.duration_s != view.horizon_s:
        raise ValueError("native TORAX action and numerical horizon differ")
    from torax._src.torax_pydantic import model_config  # type: ignore[import-untyped]
    ion_symbols = {"deuterium": "D", "tritium": "T"}
    impurity_symbols = {"argon": "Ar", "carbon": "C", "neon": "Ne"}
    config: dict[str, Any] = {
        "profile_conditions": {
            "Ip": float(preparation.plasma_current_a),
            "T_i": _profile(
                preparation.radial_coordinates,
                preparation.ion_temperature_ev,
                scale=Decimal("0.001"),
            ),
            "T_e": _profile(
                preparation.radial_coordinates,
                preparation.electron_temperature_ev,
                scale=Decimal("0.001"),
            ),
            "n_e": _profile(
                preparation.radial_coordinates,
                preparation.electron_density_m3,
            ),
            "T_i_right_bc": float(preparation.ion_temperature_ev[-1] * Decimal("0.001")),
            "T_e_right_bc": float(preparation.electron_temperature_ev[-1] * Decimal("0.001")),
            "n_e_right_bc": float(preparation.electron_density_m3[-1]),
            "n_e_nbar_is_fGW": False,
            "n_e_right_bc_is_fGW": False,
            "normalize_n_e_to_nbar": False,
            "initial_j_is_total_current": True,
            "initial_psi_from_j": True,
            "initial_psi_mode": "j",
        },
        "plasma_composition": {
            "main_ion": ion_symbols[preparation.main_ion],
            "impurity": {
                "impurity_mode": "fractions",
                "species": {impurity_symbols[preparation.impurity]: 1.0},
            },
            "Z_eff": float(preparation.zeff),
        },
        "numerics": {
            "t_initial": 0.0,
            "t_final": float(view.horizon_s),
            "exact_t_final": True,
            "fixed_dt": float(view.timestep_s),
            "adaptive_dt": False,
            "evolve_ion_heat": True,
            "evolve_electron_heat": True,
            "evolve_current": False,
            "evolve_density": False,
        },
        "geometry": {
            "geometry_type": "circular",
            "n_rho": view.radial_cells,
            "R_major": float(preparation.major_radius_m),
            "a_minor": float(preparation.minor_radius_m),
            "B_0": float(preparation.toroidal_field_t),
            "elongation_LCFS": float(preparation.elongation_lcfs),
        },
        "sources": {
            "generic_current": {"mode": "ZERO"},
            "generic_heat": {
                "model_name": "gaussian",
                "gaussian_width": float(preparation.source_width),
                "gaussian_location": float(preparation.source_radial_location),
                "P_total": float(action.realized_power_w),
                "electron_heat_fraction": float(preparation.electron_heat_fraction),
                "absorption_fraction": float(preparation.absorbed_power_fraction),
                "mode": "MODEL_BASED",
            },
            "ei_exchange": {"mode": "MODEL_BASED", "Qei_multiplier": 1.0},
        },
        "neoclassical": {
            "bootstrap_current": {"model_name": "zeros"},
            "transport": {"model_name": "zeros"},
        },
        "pedestal": {"model_name": "no_pedestal"},
        "transport": {
            "model_name": "constant",
            "chi_i": float(preparation.chi_i_m2_s),
            "chi_e": float(preparation.chi_e_m2_s),
            "D_e": float(preparation.particle_diffusivity_m2_s),
            "V_e": float(preparation.particle_convection_m_s),
        },
        "solver": {
            "solver_type": "linear",
            "theta_implicit": 1.0,
            "use_predictor_corrector": False,
            "use_pereverzev": False,
            "implicit_solver_type": "thomas",
        },
        "time_step_calculator": {"calculator_type": "fixed"},
    }
    return model_config.ToraxConfig(**config)


def _decimal(value: float) -> Decimal:
    if not math.isfinite(value):
        raise ValueError("native TORAX produced a non-finite retained value")
    return Decimal(str(float(value)))


def execute_native_torax(
    preparation: NativeToraxPreparation,
    action: NativeToraxAction,
    view: NativeToraxView,
) -> tuple[NativeToraxTrajectory, NativeToraxEpisode]:
    """Execute one native preparation/action/view and retain a compact trajectory."""

    view.operational_admission()
    import torax  # type: ignore[import-untyped]

    config = build_native_torax_config(preparation, action, view)
    _tree, history = torax.run_simulation(config, progress_bar=False)
    times = np.asarray(history.times, dtype=np.float64)
    core: list[float] = []
    radial_mean: list[float] = []
    contrast: list[float] = []
    minimum_e: list[float] = []
    minimum_i: list[float] = []
    for profiles in history.core_profiles:
        electron = np.asarray(profiles.T_e.value, dtype=np.float64) * 1000.0
        ion = np.asarray(profiles.T_i.value, dtype=np.float64) * 1000.0
        core.append(float(electron[0]))
        radial_mean.append(float(np.mean(electron)))
        contrast.append(float(electron[0] - electron[-1]))
        minimum_e.append(float(np.min(electron)))
        minimum_i.append(float(np.min(ion)))
    trajectory = NativeToraxTrajectory(
        trajectory_id=f"trajectory.{preparation.preparation_id}.{view.view_id}.{action.action_label}",
        preparation_id=preparation.preparation_id,
        action_id=action.action_id,
        view_id=view.view_id,
        time_s=tuple(_decimal(value) for value in times),
        core_electron_temperature_ev=tuple(_decimal(value) for value in core),
        radial_mean_electron_temperature_ev=tuple(_decimal(value) for value in radial_mean),
        core_edge_contrast_ev=tuple(_decimal(value) for value in contrast),
        minimum_electron_temperature_ev=tuple(_decimal(value) for value in minimum_e),
        minimum_ion_temperature_ev=tuple(_decimal(value) for value in minimum_i),
        torax_sim_error=history.sim_error.name,
    )
    reasons: set[str] = set()
    if history.sim_error.name != "NO_ERROR":
        reasons.add("TORAX_SIM_ERROR")
    if trajectory.time_s[0] != 0 or trajectory.time_s[-1] != view.horizon_s:
        reasons.add("TORAX_CLOCK_OR_HORIZON_INVALID")
    if any(value <= 0 for value in trajectory.minimum_electron_temperature_ev):
        reasons.add("NONPOSITIVE_ELECTRON_TEMPERATURE")
    if any(value <= 0 for value in trajectory.minimum_ion_temperature_ev):
        reasons.add("NONPOSITIVE_ION_TEMPERATURE")
    peak_core_temperature = max(trajectory.core_electron_temperature_ev)
    if peak_core_temperature > preparation.maximum_core_temperature_ev:
        reasons.add("CORE_TEMPERATURE_SAFETY_CEILING_EXCEEDED")
    if reasons:
        disposition = (
            NativeToraxEpisodeDisposition.SOLVER_FAILURE
            if "TORAX_SIM_ERROR" in reasons
            else NativeToraxEpisodeDisposition.NUMERICAL_INVALID
        )
        return trajectory, NativeToraxEpisode(
            episode_id=f"episode.{preparation.preparation_id}.{view.view_id}.{action.action_label}",
            preparation_id=preparation.preparation_id,
            action_id=action.action_id,
            view_id=view.view_id,
            disposition=disposition,
            endpoint_delta_core_temperature_ev=None,
            endpoint_core_edge_contrast_ev=None,
            minimum_electron_temperature_ev=None,
            minimum_ion_temperature_ev=None,
            safety_margin_ev=None,
            effort_j=None,
            trajectory_sha256=None,
            reason_codes=tuple(sorted(reasons)),
        )
    payload_hash = sha256(trajectory.canonical_bytes()).hexdigest()
    return trajectory, NativeToraxEpisode(
        episode_id=f"episode.{preparation.preparation_id}.{view.view_id}.{action.action_label}",
        preparation_id=preparation.preparation_id,
        action_id=action.action_id,
        view_id=view.view_id,
        disposition=NativeToraxEpisodeDisposition.COMPLETE,
        endpoint_delta_core_temperature_ev=(
            trajectory.core_electron_temperature_ev[-1] - trajectory.core_electron_temperature_ev[0]
        ),
        endpoint_core_edge_contrast_ev=trajectory.core_edge_contrast_ev[-1],
        minimum_electron_temperature_ev=min(trajectory.minimum_electron_temperature_ev),
        minimum_ion_temperature_ev=min(trajectory.minimum_ion_temperature_ev),
        safety_margin_ev=(
            preparation.maximum_core_temperature_ev - peak_core_temperature
        ),
        effort_j=action.realized_power_w * action.duration_s,
        trajectory_sha256=payload_hash,
        reason_codes=(),
    )


__all__ = ['build_native_torax_config', 'execute_native_torax']
