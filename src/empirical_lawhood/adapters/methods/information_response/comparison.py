"information response prediction eight predeclared tests, shared-root diagnostics and separate verdicts."

import json
from math import comb
from typing import Any

import numpy as np

from empirical_lawhood.kernel.provenance import ObjectIdentity
from .models import Array, MODEL_IDS, primary_contrast
from .records import InformationResponseEvaluationConfig, InformationResponseEvaluation, InformationResponseViewObservation, combined_status

PRIMARY = {"preparation-contrast": (("history", "snapshot"), ("history-with-mechanism", "history"), ("mechanism-with-residual", "history-with-mechanism")), "signed-gain": (("mechanism-with-residual", "training-mean"),)}


def tail(wins: int) -> float:
    return sum(comb(32, j) for j in range(wins, 33)) / 2**32


def finite(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return finite(value.tolist())
    if isinstance(value, (tuple, list)):
        return [finite(v) for v in value]
    return None if isinstance(value, float) and not np.isfinite(value) else value


def holm(pvalues: dict[str, float], tie_ranks: dict[str, int]) -> dict[str, bool]:
    if len(pvalues) != 8 or any(not 0 <= p <= 1 for p in pvalues.values()):
        raise ValueError("information response prediction Holm family must contain its eight fixed tests")
    if set(tie_ranks) != set(pvalues) or tuple(sorted(tie_ranks.values())) != tuple(range(8)) or any(type(rank) is not int for rank in tie_ranks.values()):
        raise ValueError("Information prediction Holm family requires eight distinct numerical tie ranks")
    rejected, continuing = {}, True
    for rank, (name, p) in enumerate(
        sorted(pvalues.items(), key=lambda item: (item[1], tie_ranks[item[0]]))
    ):
        continuing = continuing and p <= 0.05 / (8 - rank)
        rejected[name] = continuing
    return rejected


def compare_losses(
    losses: Array, richer: str, simpler: str, valid: bool
) -> dict[str, Any]:
    if losses.shape != (32, 2, 7):
        raise ValueError("information response prediction losses change the assigned root/view/model denominator")
    a, b = losses[:, :, MODEL_IDS.index(richer)], losses[:, :, MODEL_IDS.index(simpler)]
    complete = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    wins = int((complete & (a < b).all(axis=1)).sum())
    reverse = int((complete & (b < a).all(axis=1)).sum())
    return {
        "richer": richer,
        "simpler": simpler,
        "valid": bool(valid and complete.all()),
        "assigned_roots": 32,
        "complete_roots": int(complete.sum()),
        "concordant_wins": wins,
        "reverse_concordant_wins": reverse,
        "ties_disagreements_or_missing": 32 - wins - reverse,
        "exact_directional_tail": tail(wins),
        "reverse_tail_descriptive": tail(reverse),
        "richer_mean_mse_by_view": finite(a.mean(axis=0)),
        "simpler_mean_mse_by_view": finite(b.mean(axis=0)),
        "observed_relative_mse_reduction": finite(
            np.divide(
                b.mean(axis=0) - a.mean(axis=0),
                b.mean(axis=0),
                out=np.full(2, np.nan),
                where=b.mean(axis=0) > 0,
            )
        ),
    }


def diagnostics(observed: Array, predicted: Array) -> tuple[dict[str, Any], Array]:
    # n,view,future,property; n,view,model,property
    if observed.shape[:3] != (32, 2, 2) or predicted.shape != (
        32,
        2,
        7,
        observed.shape[-1],
    ):
        raise ValueError("information response prediction diagnostic operand shapes differ")
    errors = predicted[:, :, :, None] - observed[:, :, None]
    losses = np.mean(errors**2, axis=(3, 4))
    signal = np.sqrt(np.mean(observed**2, axis=(2, 3)))
    numerical = np.sqrt(np.mean((observed[:, 0] - observed[:, 1]) ** 2, axis=(1, 2)))
    mean_subtracted = observed - predicted[:, :, 0, None]
    return {
        "root_signal_rms_by_view": finite(signal),
        "root_numerical_rms": finite(numerical),
        "roots_signal_above_eightfold_floor": int(
            (signal[:, 0] > 8 * np.maximum(1e-6, numerical)).sum()
        ),
        "root_training_mean_subtracted_energy": finite(
            np.mean(mean_subtracted**2, axis=(2, 3))
        ),
        "root_within_microscopic_state_future_variance": finite(
            np.mean((observed[:, :, 0] - observed[:, :, 1]) ** 2 / 2, axis=-1)
        ),
        "models": {
            arm: {
                "root_mse_by_view": finite(losses[:, :, mi]),
                "mean_mse_by_view": finite(losses[:, :, mi].mean(axis=0)),
                "rmse_by_view": finite(np.sqrt(losses[:, :, mi].mean(axis=0))),
                "signed_bias_by_view": finite(errors[:, :, mi].mean(axis=(0, 2, 3))),
                "root_conditional_mean_error_cross_product": finite(
                    np.mean(errors[:, :, mi, 0] * errors[:, :, mi, 1], axis=-1)
                ),
            }
            for mi, arm in enumerate(MODEL_IDS)
        },
    }, losses


def evaluate(
    config: InformationResponseEvaluationConfig, reports: tuple[InformationResponseViewObservation, ...]
) -> InformationResponseEvaluation:
    identity = ObjectIdentity.from_record
    source = config.projection.native_spec
    if (
        len(reports) != 128
        or {(r.root, r.refinement) for r in reports}
        != {(r, v) for r in source.roots for v in (1, 2)}
        or any(
            r.projection_config
            != identity(config.projection.config_id, config.projection)
            for r in reports
        )
    ):
        raise ValueError("information response prediction evaluator changes its complete projection census")
    slots = {(r.root.context, r.root.index, r.refinement): r for r in reports}
    results: dict[str, Any] = {}
    tests: dict[str, dict[str, Any]] = {}
    test_tie_ranks: dict[str, int] = {}
    for name in ("assembling", "prepared"):
        observed = np.empty((32, 2, 2, 5, 5, 2, 2))
        predicted = np.empty((32, 2, 7, 5, 5, 2, 2))
        reasons: set[str] = set()
        for i in range(32):
            for vi in range(2):
                report = slots[name, i, vi + 1]
                observed[i, vi] = np.swapaxes(
                    np.asarray(report.observed_gain, dtype=float).reshape(
                        5, 2, 5, 2, 2
                    ),
                    0,
                    1,
                )
                predicted[i, vi] = np.swapaxes(
                    np.asarray(report.predicted_gain, dtype=float).reshape(
                        5, 7, 5, 2, 2
                    ),
                    0,
                    1,
                )
                reasons.update(report.reasons)
        result: dict[str, Any] = {
            "assigned_roots": 32,
            "reasons": sorted(reasons),
            "experiments": {},
        }
        for number in ("preparation-contrast", "signed-gain"):
            obs = (
                primary_contrast(observed)
                if number == "preparation-contrast"
                else observed[..., (2, 4), :, :]
            )
            pred = (
                primary_contrast(predicted)
                if number == "preparation-contrast"
                else predicted[..., (2, 4), :, :]
            )
            detail, losses = diagnostics(
                obs.reshape(32, 2, 2, -1), pred.reshape(32, 2, 7, -1)
            )
            primary = {}
            for a, b in PRIMARY[number]:
                test_id = f"{name}.{number}.{a}-over-{b}"
                test_tie_ranks[test_id] = len(test_tie_ranks)
                tests[test_id] = compare_losses(losses, a, b, not reasons)
                primary[test_id] = tests[test_id]
            secondary = (
                (("snapshot", "training-mean"), ("snapshot-with-mechanism", "snapshot"), ("history-with-mechanism", "snapshot-with-mechanism"), ("mechanism", "training-mean"))
                if number == "preparation-contrast"
                else (("mechanism", "training-mean"), ("history", "snapshot"), ("history-with-mechanism", "history"))
            )
            detail.update(
                primary=primary,
                descriptive_comparisons={
                    f"{a}-over-{b}": compare_losses(losses, a, b, not reasons)
                    for a, b in secondary
                },
            )
            result["experiments"][number] = detail
        # Retain all five horizons and parent components without fresh selection.
        error = predicted[:, :, :, None] - observed[:, :, None]
        result["full_gain_rmse_by_view_model_parent_horizon"] = finite(
            np.sqrt(np.mean(error**2, axis=(0, 3, 6, 7)))
        )
        results[name] = result
    rejected = holm(
        {
            name: float(t["exact_directional_tail"]) if t["valid"] else 1.0
            for name, t in tests.items()
        },
        test_tie_ranks,
    )
    for name, test in tests.items():
        test["holm_rejected"] = rejected[name]
        test["status"] = (
            "UNEVALUABLE"
            if not test["valid"]
            else ("SUPPORTED" if rejected[name] else "NOT_SUPPORTED")
        )
    for context in results.values():
        for experiment in context["experiments"].values():
            experiment["status"] = combined_status(
                [test["status"] for test in experiment["primary"].values()]
            ).value
    return InformationResponseEvaluation(
        identity(config.config_id, config),
        tuple(
            identity(r.report_id, r) for r in sorted(reports, key=lambda r: r.report_id)
        ),
        json.dumps(results, sort_keys=True, separators=(",", ":"), allow_nan=False),
    )


def concordant_win_power() -> dict[str, float]:
    """Retain the original fixed32-root,24-concordant-win binomial grid.

    This diagnostic assumes independent roots with the supplied concordant-win
    probability, including both views in each root's event. It describes the
    first frozen Holm threshold only, not joint power for all eight tests. It
    draws no outcomes and supplies no entry, selection or qualification gate.
    """
    return {
        str(p): sum((comb(32, j) * p**j * (1 - p) ** (32 - j) for j in range(24, 33)))
        for p in (0.6, 0.7, 0.8, 0.9)
    }
