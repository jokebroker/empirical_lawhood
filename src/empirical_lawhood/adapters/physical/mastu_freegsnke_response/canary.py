"""Reusable short-horizon MAST-U/FreeGSNKE canary calculation.

The caller supplies preparation, static-target, and machine data. This module
does not load historical inputs, assert source authority, or publish receipts.
"""

from __future__ import annotations

import contextlib
import hashlib
import math
import sys
import time
from typing import Any, Final

import numpy as np

from empirical_lawhood.adapters.physical.mast_freegsnke_static_transport.experiment import _build_tokamak, _primary_xpoint, _set_currents


SCHEMA: Final = 'empirical-lawhood/physical/mastu-freegsnke-response/canary-evaluation'
DT_S: Final = 0.0005
HORIZONS_S: Final = (0.005, 0.010, 0.015)
FRACTIONS: Final = (0.0625, 0.125, 0.25)
VIEWS: Final = (65, 129)
ACTIVE_LABELS: Final = (
    "Solenoid",
    "px",
    "d1",
    "d2",
    "d3",
    "dp",
    "d5",
    "d6",
    "d7",
    "p4",
    "p5",
    "p6",
)
CURRENT_LIMITS_A: Final = {
    "px": (-5000.0, 5000.0),
    "d1": (-9000.0, 9000.0),
    "d2": (-9000.0, 9000.0),
    "d3": (-7000.0, 7000.0),
    "dp": (-7000.0, 7000.0),
    "d5": (-5000.0, 5000.0),
    "d6": (-4000.0, 4000.0),
    "d7": (-5000.0, 5000.0),
    "p4": (-10000.0, 0.0),
    "p5": (-10000.0, 0.0),
}


def action_roster(full_increment_v: float) -> tuple[tuple[str, float], ...]:
    if not math.isfinite(full_increment_v) or full_increment_v <= 0:
        raise ValueError("Admission voltage limit must be positive and finite")
    values = [("hold", 0.0)]
    for fraction in FRACTIONS:
        amplitude = full_increment_v * fraction
        values.extend(((f"minus-{fraction:g}", -amplitude), (f"plus-{fraction:g}", amplitude)))
    return tuple(values)


def select_preparation_slice(
    times_s: np.ndarray[Any, Any],
    baseline_valid: np.ndarray[Any, Any],
    refined_valid: np.ndarray[Any, Any],
) -> int:
    """Select the earliest source slice valid in both frozen numerical views."""

    if (
        times_s.ndim != 1
        or baseline_valid.shape != times_s.shape
        or refined_valid.shape != times_s.shape
    ):
        raise ValueError("static-validity rosters do not align")
    eligible = np.flatnonzero(
        np.asarray(baseline_valid, dtype=bool) & np.asarray(refined_valid, dtype=bool)
    )
    if not eligible.size:
        raise ValueError("no public preparation is valid in both numerical views")
    if np.any(np.diff(np.asarray(times_s, dtype=float)) <= 0):
        raise ValueError("public preparation times are not strictly increasing")
    return int(eligible[0])


def select_canary(
    episodes: list[dict[str, Any]], *, materiality_t: float = 0.0017
) -> dict[str, object]:
    """Apply the predeclared smallest-amplitude/earliest-horizon rule."""

    if not math.isfinite(materiality_t) or materiality_t <= 0:
        raise ValueError("materiality threshold must be positive and finite")
    by_key = {(int(row["view"]), str(row["word"])): row for row in episodes}
    for fraction in FRACTIONS:
        words = (f"minus-{fraction:g}", "hold", f"plus-{fraction:g}")
        if any((view, word) not in by_key for view in VIEWS for word in words):
            raise ValueError("canary episode roster is incomplete")
        if any(not bool(by_key[view, word]["valid"]) for view in VIEWS for word in words):
            continue
        for horizon in HORIZONS_S:
            key = format(horizon, ".3f")
            qualified = True
            contrasts: list[float] = []
            for view in VIEWS:
                minus = float(by_key[view, words[0]]["pickup_1_t"][key])
                hold = float(by_key[view, "hold"]["pickup_1_t"][key])
                plus = float(by_key[view, words[2]]["pickup_1_t"][key])
                initial = float(by_key[view, "hold"]["pickup_1_t"]["0.000"])
                contrast = min(plus - hold, hold - minus)
                hold_drift = abs(hold - initial)
                other = VIEWS[1] if view == VIEWS[0] else VIEWS[0]
                view_gap = abs(hold - float(by_key[other, "hold"]["pickup_1_t"][key]))
                qualified &= contrast > 0 and contrast > 2 * max(hold_drift, view_gap)
                contrasts.append(contrast)
            if qualified:
                minimum = min(contrasts)
                return {
                    "disposition": (
                        "SUPPORTED" if minimum >= materiality_t else "METHOD_MATERIALITY_INCOMPATIBLE"
                    ),
                    "selected_fraction": fraction,
                    "selected_horizon_s": horizon,
                    "minimum_signed_contrast_t": minimum,
                    "reason_codes": (
                        [] if minimum >= materiality_t else ["CONFIRMATORY_G1R_MATERIALITY_NOT_REACHED"]
                    ),
                }
    return {
        "disposition": "SHORT_HORIZON_SOURCE_UNEVALUABLE",
        "selected_fraction": None,
        "selected_horizon_s": None,
        "minimum_signed_contrast_t": None,
        "reason_codes": ["NO_VALID_SIGNED_VIEW_STABLE_SHORT_HORIZON"],
    }


def _pickup_1(eq: Any) -> tuple[float, bool]:
    pickups = sorted(
        eq.tokamak.probes.pickups,
        key=lambda value: (float(value["position"][2]), str(value["name"])),
    )
    if len(pickups) < 2:
        raise ValueError("public machine lacks the declared pickup pair")
    eq.tokamak.probes.initialise_setup(eq)
    values = dict(
        zip(
            eq.tokamak.probes.pickup_order,
            eq.tokamak.probes.calculate_pickup_value(eq),
            strict=True,
        )
    )
    _point, _mismatch, topology = _primary_xpoint(eq)
    valid = (
        topology != "UNQUALIFIED" and not bool(eq.intersectsWall()) and not bool(eq.flag_limiter)
    )
    return float(values[str(pickups[0]["name"])]), valid


def _admission_limit(stepping: Any, labels: tuple[str, ...], baseline: np.ndarray[Any, Any]) -> float:
    from scipy.linalg import expm

    inductance = np.asarray(stepping.evol_metal_curr.coil_self_ind, dtype=float)[:12, :12]
    resistance = np.asarray(stepping.evol_metal_curr.coil_resist, dtype=float)[:12]
    dynamics = np.linalg.solve(inductance, np.diag(resistance))
    forcing = np.linalg.solve(inductance, np.eye(12))
    p4 = labels.index("p4")
    maximum = np.zeros(12, dtype=float)
    for clock in np.linspace(0.0, HORIZONS_S[-1], 61):
        kernel = np.linalg.solve(dynamics, np.eye(12) - expm(-dynamics * clock)) @ forcing
        maximum = np.maximum(maximum, np.abs(kernel[:, p4]))
    candidates = []
    for index, label in enumerate(labels):
        if label not in CURRENT_LIMITS_A or maximum[index] == 0:
            continue
        lower, upper = CURRENT_LIMITS_A[label]
        headroom = min(upper - baseline[index], baseline[index] - lower)
        if headroom <= 0:
            raise ValueError(f"{label} lacks symmetric current headroom")
        candidates.append(0.05 * headroom / maximum[index])
    value = math.floor(min(candidates) * 1_000_000) / 1_000_000
    if value <= 0:
        raise ValueError("derived admission voltage limit is not positive")
    return float(value)


def _currents_valid(
    labels: tuple[str, ...],
    minimum: np.ndarray[Any, Any],
    maximum: np.ndarray[Any, Any],
) -> bool:
    return all(
        minimum[labels.index(label)] >= lower and maximum[labels.index(label)] <= upper
        for label, (lower, upper) in CURRENT_LIMITS_A.items()
    )


def _episode(
    preparation: dict[str, np.ndarray[Any, Any]],
    machine: dict[str, object],
    *,
    view: int,
    word: str,
    increment_v: float,
    source_slice_index: int,
    qualification_only: bool = False,
) -> dict[str, object]:
    from freegsnke import (  # type: ignore[import-untyped]
        GSstaticsolver,
        equilibrium_update,
        jtor_update,
        nonlinear_solve,
    )

    started = time.monotonic()
    grid = np.asarray(preparation["grid"], dtype=float)
    tokamak = _build_tokamak(machine, "SYMMETRY_COLLAPSED")
    labels = np.asarray(preparation["coil_labels"]).astype(str).tolist()
    _set_currents(
        tokamak,
        machine,
        "SYMMETRY_COLLAPSED",
        labels,
        np.asarray(preparation["currents_output_a"], dtype=float)[source_slice_index],
    )
    eq = equilibrium_update.Equilibrium(
        tokamak=tokamak,
        Rmin=float(grid[0]),
        Rmax=float(grid[1]),
        Zmin=float(grid[2]),
        Zmax=float(grid[3]),
        nx=view,
        ny=view,
    )
    profiles = jtor_update.Lao85(
        eq=eq,
        Ip=float(preparation["plasma_current_a"][source_slice_index]),
        fvac=float(preparation["fvac_t_m"][source_slice_index]),
        alpha=np.asarray(preparation["alpha"], dtype=float)[source_slice_index],
        beta=np.asarray(preparation["beta"], dtype=float)[source_slice_index],
        alpha_logic=bool(preparation["alpha_logic"][source_slice_index]),
        beta_logic=bool(preparation["beta_logic"][source_slice_index]),
    )
    solver = GSstaticsolver.NKGSsolver(eq)
    with contextlib.redirect_stdout(sys.stderr):
        solver.solve(
            eq=eq,
            profiles=profiles,
            constrain=None,
            target_relative_tolerance=1e-6,
            max_solving_iterations=100,
            verbose=False,
        )
        initial_pickup, initial_valid = _pickup_1(eq)
        stepping = nonlinear_solve.nl_solver(
            eq=eq,
            profiles=profiles,
            GSStaticSolver=solver,
            full_timestep=DT_S,
            max_internal_timestep=DT_S,
            plasma_resistivity=1e-6,
            max_mode_frequency=10**2.5,
        )
        stepping.initialize_from_ICs(eq, profiles)
        resistance = np.asarray(stepping.evol_metal_curr.coil_resist, dtype=float)[:12]
        baseline = np.asarray(stepping.vessel_currents_vec, dtype=float)
        active_labels = tuple(tokamak.coils_list[:12])
        if active_labels != ACTIVE_LABELS:
            raise ValueError("public symmetric circuit order differs")
        full_increment = _admission_limit(stepping, active_labels, baseline[:12])
        if qualification_only:
            return {
                "eligible": bool(initial_valid),
                "full_p4_increment_v": full_increment,
                "initial_pickup_1_t": initial_pickup,
                "view": view,
                "runtime_s": time.monotonic() - started,
            }
        voltage = baseline[:12] * resistance
        voltage[active_labels.index("p4")] += increment_v
        vector_sha = hashlib.sha256(np.asarray(voltage, dtype="<f8").tobytes()).hexdigest()
        snapshots = {"0.000": initial_pickup}
        validity = bool(initial_valid)
        max_relative_change = 0.0
        domain_departed = False
        completed_steps = 0
        extrema_min = baseline[:12].copy()
        extrema_max = baseline[:12].copy()
        for step in range(1, round(HORIZONS_S[-1] / DT_S) + 1):
            if bool(stepping.handleMyy.check_Myy(stepping.hatIy)):
                domain_departed = True
                break
            stepping.nlstepper(
                active_voltage_vec=voltage,
                linear_only=False,
                verbose=0,
                max_solving_iterations=50,
            )
            completed_steps = step
            physical = np.asarray(stepping.vessel_currents_vec, dtype=float)
            extrema_min = np.minimum(extrema_min, physical[:12])
            extrema_max = np.maximum(extrema_max, physical[:12])
            relative = float(getattr(stepping.NK, "relative_change", 0.0))
            max_relative_change = max(max_relative_change, relative)
            clock = step * DT_S
            if any(abs(clock - target) < 1e-12 for target in HORIZONS_S):
                pickup, observed = _pickup_1(stepping.eq1)
                snapshots[format(clock, ".3f")] = pickup
                validity &= observed
    current_valid = _currents_valid(active_labels, extrema_min, extrema_max)
    valid = bool(
        validity
        and not domain_departed
        and current_valid
        and set(snapshots) == {"0.000", "0.005", "0.010", "0.015"}
    )
    return {
        "episode_id": f"canary.view-{view}.{word}",
        "view": view,
        "word": word,
        "requested_increment_v": increment_v,
        "accepted_increment_v": increment_v,
        "applied_vector_sha256": vector_sha,
        "realized_vector_sha256": vector_sha if completed_steps else None,
        "applied_clocks_s": [format(step * DT_S, ".4f") for step in range(completed_steps)],
        "pickup_1_t": snapshots,
        "valid": valid,
        "domain_departed": domain_departed,
        "current_limits_passed": bool(current_valid),
        "maximum_nk_relative_change": max_relative_change,
        "runtime_s": time.monotonic() - started,
    }


def _scientific_episode(
    preparation: dict[str, np.ndarray[Any, Any]],
    machine: dict[str, object],
    *,
    view: int,
    word: str,
    increment_v: float,
    source_slice_index: int,
    qualification_only: bool = False,
) -> dict[str, object]:
    try:
        return _episode(
            preparation,
            machine,
            view=view,
            word=word,
            increment_v=increment_v,
            source_slice_index=source_slice_index,
            qualification_only=qualification_only,
        )
    except (ArithmeticError, RuntimeError, np.linalg.LinAlgError) as error:
        if qualification_only:
            return {
                "eligible": False,
                "error_code": f"FREEGSNKE_{type(error).__name__.upper()}",
                "view": view,
            }
        return {
            "episode_id": f"canary.view-{view}.{word}",
            "view": view,
            "word": word,
            "requested_increment_v": increment_v,
            "valid": False,
            "error_code": f"FREEGSNKE_{type(error).__name__.upper()}",
            "pickup_1_t": {},
        }


def evaluate_canary(
    preparation: dict[str, np.ndarray[Any, Any]],
    static_target: dict[str, np.ndarray[Any, Any]],
    machine: dict[str, object],
    *,
    materiality_t: float = 0.0017,
) -> dict[str, object]:
    """Evaluate supplied MAST-U inputs without asserting release authority.

    Native FreeGSNKE dependencies are loaded only when this calculation runs.
    The result is a calculation; callers must separately establish source,
    input, execution, and claim authority before publishing a scientific claim.
    """

    selected_slice = select_preparation_slice(
        np.asarray(static_target["slice_times_s"], dtype=float),
        np.asarray(static_target["symmetry_collapsed__baseline_65__valid"], dtype=bool),
        np.asarray(static_target["symmetry_collapsed__refined_129__valid"], dtype=bool),
    )
    if selected_slice >= len(preparation["slice_times_s"]):
        raise ValueError("selected static-target slice is absent from preparation")
    view_qualifications = [
        _scientific_episode(
            preparation,
            machine,
            view=view,
            word="qualification",
            increment_v=0,
            source_slice_index=selected_slice,
            qualification_only=True,
        )
        for view in VIEWS
    ]
    qualified = all(bool(item.get("eligible")) for item in view_qualifications)
    qualification: dict[str, object] = {
        "eligible": qualified,
        "views": view_qualifications,
        "selection_rule": "EARLIEST_STATIC_VALID_IN_BOTH_FROZEN_VIEWS",
    }
    if not qualified:
        episodes: list[dict[str, object]] = []
    else:
        raw_limits = [item.get("full_p4_increment_v") for item in view_qualifications]
        if any(not isinstance(item, (float, int)) for item in raw_limits):
            raise TypeError("qualified admission voltage limits are not numeric")
        full = min(float(item) for item in raw_limits if isinstance(item, (float, int)))
        qualification["full_p4_increment_v"] = full
        episodes = [
            _scientific_episode(
                preparation,
                machine,
                view=view,
                word=word,
                increment_v=increment,
                source_slice_index=selected_slice,
            )
            for view in VIEWS
            for word, increment in action_roster(full)
        ]
    selection = (
        select_canary(episodes, materiality_t=materiality_t)
        if len(episodes) == 14
        else {
            "disposition": "SHORT_HORIZON_SOURCE_UNEVALUABLE",
            "selected_fraction": None,
            "selected_horizon_s": None,
            "minimum_signed_contrast_t": None,
            "reason_codes": ["ZERO_ACTION_PROBE_INVALID"],
        }
    )
    result = {
        "schema": SCHEMA,
        "experiment_id": "mastu-freegsnke-short-horizon-voltage-canary",
        "preparation_selection_rule": "EARLIEST_STATIC_VALID_IN_BOTH_FROZEN_VIEWS",
        "source_slice_index": selected_slice,
        "source_time_s": format(float(preparation["slice_times_s"][selected_slice]), ".2f"),
        "coordinate": "SYMMETRY_COLLAPSED",
        "timestep_s": DT_S,
        "materiality_t": materiality_t,
        "qualification": qualification,
        "episodes": episodes,
        "selection": selection,
    }
    return result
