"Root-level causal response prediction decision rule, independent of acquisition and authority."

from math import comb
from typing import Any

import numpy as np

from .models import Array


def compare_context(observed: Array, causal: Array, baseline: Array) -> dict[str, Any]:
    """C arrays: 32 roots, 2 views, (2 futures for observations), 32 properties."""
    if (
        observed.shape != (32, 2, 2, 32)
        or causal.shape != (32, 2, 32)
        or baseline.shape != causal.shape
    ):
        raise ValueError(
            "causal response prediction comparison changes its assigned root/view/innovation/property census"
        )
    complete = (
        np.isfinite(observed).all(axis=(1, 2, 3))
        & np.isfinite(causal).all(axis=(1, 2))
        & np.isfinite(baseline).all(axis=(1, 2))
    )
    c_loss = np.mean((causal[:, :, None] - observed) ** 2, axis=(2, 3))
    b_loss = np.mean((baseline[:, :, None] - observed) ** 2, axis=(2, 3))
    c_wins = complete & (c_loss < b_loss).all(axis=1)
    b_wins = complete & (b_loss < c_loss).all(axis=1)
    signal = np.sqrt(np.mean(observed[:, 0] ** 2, axis=(1, 2)))
    discrepancy = np.sqrt(np.mean((observed[:, 0] - observed[:, 1]) ** 2, axis=(1, 2)))
    resolved = complete & (signal > 8 * np.maximum(1e-6, discrepancy))
    # Undefined aggregate losses cannot satisfy a practical effect requirement.
    cm, bm = c_loss.mean(axis=0), b_loss.mean(axis=0)
    c_effect = bool(
        np.isfinite(cm).all()
        and np.isfinite(bm).all()
        and (bm > 0).all()
        and (cm <= 0.8 * bm).all()
    )
    b_effect = bool(
        np.isfinite(cm).all()
        and np.isfinite(bm).all()
        and (cm > 0).all()
        and (bm <= 0.8 * cm).all()
    )
    nw_c, nw_b = int(c_wins.sum()), int(b_wins.sum())
    face = (
        "UNINFORMATIVE"
        if resolved.sum() < 24
        else (
            "CAUSAL_ADVANTAGE"
            if nw_c >= 23 and c_effect
            else "BASELINE_ADVANTAGE"
            if nw_b >= 23 and b_effect
            else "UNRESOLVED"
        )
    )

    def finite_list(values: Array) -> Any:
        return [
            finite_list(x) if isinstance(x, np.ndarray) else float(x) if np.isfinite(x) else None
            for x in values
        ]

    return {
        "face": face,
        "assigned_roots": 32,
        "complete_roots": int(complete.sum()),
        "resolved_roots": int(resolved.sum()),
        "causal_wins": nw_c,
        "baseline_wins": nw_b,
        "neither_wins": 32 - nw_c - nw_b,
        "causal_exact_tail": sum(comb(32, j) for j in range(nw_c, 33)) / 2**32,
        "baseline_exact_tail": sum(comb(32, j) for j in range(nw_b, 33)) / 2**32,
        "causal_practical_requirement": c_effect,
        "baseline_practical_requirement": b_effect,
        "causal_mean_mse": finite_list(cm),
        "baseline_mean_mse": finite_list(bm),
        "root_causal_mse": finite_list(c_loss),
        "root_baseline_mse": finite_list(b_loss),
        "root_signal_rms": finite_list(signal),
        "root_numerical_rms": finite_list(discrepancy),
        "within_state_variance": finite_list(
            np.mean((observed[:, :, 0] - observed[:, :, 1]) ** 2 / 2, axis=-1)
        ),
        "causal_mean_error_cross_product": finite_list(
            np.mean((causal - observed[:, :, 0]) * (causal - observed[:, :, 1]), axis=-1)
        ),
        "baseline_mean_error_cross_product": finite_list(
            np.mean((baseline - observed[:, :, 0]) * (baseline - observed[:, :, 1]), axis=-1)
        ),
    }
