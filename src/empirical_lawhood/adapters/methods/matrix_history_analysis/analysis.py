"""Complete declared sample censuses and history-level descriptive reduction."""

import numpy as np

from . import algebra, passive

SECTOR_METRICS = (
    "mean_slow_rate", "band_ratio", "gap", "lie_leakage",
    "minimum_commutator_coupling", "minimum_commutator_norm_squared",
    "associative_leakage", "casimir_relative", "jordan", "eigenpair",
    "factor_angle_degrees", "commutant_mixing", "left_ancestry_angle_degrees",
    "right_ancestry_angle_degrees", "trace_fraction",
)
ALGEBRA_METRICS = tuple(f"{sector}.{name}" for sector in ("x", "y") for name in SECTOR_METRICS) + (
    "y_slow_x_rate", "x_slow_y_rate", "xy_angle_degrees", "x_norm", "y_norm",
)
PASSIVE_METRICS = tuple(f"{name}.{part}" for name in (*passive.CAUSAL_PREDICTORS, *passive.EVALUATOR_ORACLES) for part in ("full", "worst", "slow", "fast")) + (
    "slow_fast_leakage", "generator_drift", "endpoint_commutator",
)


def algebra_sample_indices(root, multiplier):
    """The prebound slot receives every native state, including fine half ticks."""
    return tuple(range(0, 1024 * multiplier + 1, 1 if root.dense_role else 16 * multiplier))


def passive_cells(root):
    origins = (256, 512, 768, 832, 864, 880, 896, 912, 928, 944) if root.dense_role else (256, 512, 768, 864)
    horizons = (16, 32, 64, 128) if root.dense_role else (32,)
    kappas = (.25, .5, 1.) if root.dense_role else (1.,)
    return tuple((origin, horizon, kappa) for origin in origins for horizon in horizons if origin + horizon <= 1024 for kappa in kappas)


def algebra_history(arrays, root, *, compare_commutant_altered=True):
    metadata, results, altered_sectors = [], [], []
    for view, multiplier in (("primary", 1), ("fine", 2)):
        states = arrays[f"{view}_positions"]
        for index in algebra_sample_indices(root, multiplier):
            measured = algebra.measure(states[index])
            metadata.append((multiplier, index / multiplier, 0))
            results.append(measured)
            altered_sectors.append((False, False))
            if compare_commutant_altered:
                # A distinct fitted-commutant diagnostic never replaces original data.
                altered = None
                changed = tuple(measured[s]["identifiable"] for s in ("x", "y"))
                if any(changed):
                    altered = algebra.measure(np.asarray([measured[s]["reconstructed"] if changed[col] else states[index, col] for col, s in enumerate(("x", "y"))]))
                metadata.append((multiplier, index / multiplier, 1))
                results.append(altered)
                altered_sectors.append(changed)
    count = len(results)
    out = {"sample_role": np.asarray(metadata, dtype="<f8"),
           "metrics": np.full((count, len(ALGEBRA_METRICS)), np.nan, dtype="<f8"),
           "spectra": np.full((count, 2, 15), np.nan, dtype="<f8"),
           "joint_spectra": np.full((count, 15), np.nan, dtype="<f8"),
           "projectors": np.full((count, 2, 15, 15), np.nan, dtype="<f8"),
           "factors": np.full((count, 2, 3, 4, 4), np.nan, dtype="<c16"),
           "commutant_reconstructions": np.full((count, 2, 3, 4, 4), np.nan, dtype="<c16"),
           "identifiable": np.zeros((count, 2), dtype="|b1"),
           "commutant_altered_sectors": np.asarray(altered_sectors, dtype="|b1"),
           "sample_available": np.zeros(count, dtype="|b1")}
    for row, result in enumerate(results):
        if result is None:
            continue
        out["sample_available"][row] = True
        metrics = result["cross_metrics"].copy()
        for col, sector in enumerate(("x", "y")):
            value = result[sector]
            out["spectra"][row, col] = value["spectrum"]
            out["projectors"][row, col] = value["projector"]
            out["factors"][row, col] = value["factor"]
            out["commutant_reconstructions"][row, col] = value["reconstructed"]
            out["identifiable"][row, col] = value["identifiable"]
            metrics.update({f"{sector}.{name}": number for name, number in value["metrics"].items()})
        out["joint_spectra"][row] = result["joint_spectrum"]
        out["metrics"][row] = [metrics.get(name, np.nan) for name in ALGEBRA_METRICS]
    original = out["sample_role"][:, 2] == 0
    return out, int(original.sum()), int(out["identifiable"][original, 1].sum()), {}


def passive_history(arrays, root):
    metadata, outcomes, actuals, controls, seals = [], [], [], [], []
    for view, multiplier in (("primary", 1), ("fine", 2)):
        states = arrays[f"{view}_positions"][:, 1]
        for origin, horizon, kappa in passive_cells(root):
            # This seal is constructed before any future is read or propagated.
            seal = passive.seal_causal_forecast(states[origin * multiplier], kappa=kappa, horizon=horizon * .001)
            path = states[origin * multiplier:(origin + horizon) * multiplier + 1]
            props, rates, integral = passive.propagators(path, .001 / multiplier, (kappa,))
            actual = props[-1, 0]
            outcome = passive.evaluate_forecast(actual, seal=seal, initial_operator=rates[0], endpoint_operator=rates[-1], integrated_operator=integral[-1])
            metadata.append((multiplier, origin, horizon, kappa))
            actuals.append(actual)
            controls.append((rates[0], rates[-1], integral[-1]))
            outcomes.append(outcome)
            seals.append(np.frombuffer(bytes.fromhex(seal.sha256), dtype="<i8"))
    out = {"cell_role": np.asarray(metadata, dtype="<f8"),
           "metrics": np.asarray([[r["metrics"][m] for m in PASSIVE_METRICS] for r in outcomes], dtype="<f8"),
           "actual_operators": np.asarray(actuals, dtype="<f8"),
           "origin_endpoint_integral": np.asarray(controls, dtype="<f8"),
           "causal_forecasts": np.asarray([r["predictions"][:4] for r in outcomes], dtype="<f8"),
           "evaluator_oracles": np.asarray([r["predictions"][4:] for r in outcomes], dtype="<f8"),
           "causal_seal_sha256_words": np.asarray(seals, dtype="<i8"),
           "increment_singular_values": np.asarray([r["increment_singular_values"] for r in outcomes], dtype="<f8"),
           "survival_singular_values": np.asarray([r["survival_singular_values"] for r in outcomes], dtype="<f8"),
           "increment_rank_absolute": np.asarray([r["increment_rank_absolute"] for r in outcomes], dtype="<i8"),
           "increment_unresolved": np.asarray([r["increment_unresolved"] for r in outcomes], dtype="|b1")}
    summaries = {}
    for multiplier in (1, 2):
        cohort = (out["cell_role"][:, 0] == multiplier) & (out["cell_role"][:, 2] == 32) & (out["cell_role"][:, 3] == 1) & np.isin(out["cell_role"][:, 1], (256, 512, 768, 864))
        if int(cohort.sum()) != 4:
            raise ValueError("Passive root aggregation loses a declared cohort origin")
        for col, name in enumerate(PASSIVE_METRICS):
            summaries[f"view{multiplier}.{name}"] = float(out["metrics"][cohort, col].mean())
        summaries[f"view{multiplier}.unresolved_origins"] = int(out["increment_unresolved"][cohort].sum())
    return out, len(outcomes), 0, summaries


def result_column_contract(mode):
    if mode == "ALGEBRA":
        return {"sample_role": ("view_multiplier", "primary_equivalent_tick", "commutant_altered"),
                "metrics": ALGEBRA_METRICS,
                "sector_order": ("X", "Y"), "missing_factor_reason": "SLOW_SECTOR_NOT_IDENTIFIABLE",
                "commutant_altered_sectors": ("X_altered", "Y_altered"),
                "missing_altered_reason": "NO_ORIGINAL_FACTOR_IDENTIFIABLE"}
    return {"cell_role": ("view_multiplier", "origin_primary_tick", "horizon_primary_ticks", "kappa"),
            "metrics": PASSIVE_METRICS, "causal_forecasts": passive.CAUSAL_PREDICTORS,
            "evaluator_oracles": passive.EVALUATOR_ORACLES,
            "origin_endpoint_integral": ("origin_operator", "endpoint_operator", "simpson_generator_integral"),
            "rank_absolute_tolerance": "1e-12", "pseudoinverse_relative_rcond": "1e-12"}


def validate_analysis_arrays(arrays, root, *, mode, compare_commutant_altered):
    """Validate saved census/layout without recalculating spectra or forecasts."""
    if mode == "ALGEBRA":
        expected = [(m, i/m, altered) for m in (1, 2) for i in algebra_sample_indices(root, m)
                    for altered in ((0, 1) if compare_commutant_altered else (0,))]
        count = len(expected)
        shapes = {"sample_role": (count, 3), "metrics": (count, len(ALGEBRA_METRICS)),
            "spectra": (count, 2, 15), "joint_spectra": (count, 15),
            "projectors": (count, 2, 15, 15), "factors": (count, 2, 3, 4, 4),
            "commutant_reconstructions": (count, 2, 3, 4, 4),
            "identifiable": (count, 2), "commutant_altered_sectors": (count, 2), "sample_available": (count,)}
        metadata = "sample_role"
    else:
        expected = [(m, *cell) for m in (1, 2) for cell in passive_cells(root)]
        count = len(expected)
        shapes = {"cell_role": (count, 4), "metrics": (count, len(PASSIVE_METRICS)),
            "actual_operators": (count, 15, 15), "origin_endpoint_integral": (count, 3, 15, 15),
            "causal_forecasts": (count, 4, 15, 15), "evaluator_oracles": (count, 2, 15, 15),
            "causal_seal_sha256_words": (count, 4), "increment_singular_values": (count, 15),
            "survival_singular_values": (count, 15), "increment_rank_absolute": (count,),
            "increment_unresolved": (count,)}
        metadata = "cell_role"
    if set(arrays) != set(shapes) or any(arrays[name].shape != shape for name, shape in shapes.items()) or not np.array_equal(arrays[metadata], np.asarray(expected)):
        raise ValueError("Saved history analysis loses its full declared sample/direction/view census")
