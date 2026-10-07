"Predeclared paired-root contribution estimands, separate from controller-use ownership."

from dataclasses import dataclass
import math
import numpy as np
from empirical_lawhood.adapters.methods.prepared_response.statistics import paired_difference_bounds
from .config import ARMS, COMPARISONS, CONFIRMATION_ROOTS, COMPARATIVE_SCOPE
from .numerical import FrozenFit


@dataclass(frozen=True)
class EpisodeEndpoints:
    root: str
    arm: str
    complete: bool
    task_success: bool
    unsafe_observed: bool
    unsafe_or_unobserved: bool
    restricted_time: float
    peak: float | None
    minimum_margin: float | None
    violation_duration: float | None
    time_ratio: float | None
    online_cpu: float | None
    temperature_mae: float | None = None
    conversion_mae: float | None = None


def endpoints(
    root: str,
    arm: str,
    grid: np.ndarray,
    online_cpu: float | None,
    reference_time: float | None,
    *,
    completed: bool,
) -> EpisodeEndpoints:
    """Native grid columns: time,T,Tj,dose,conversion,nA,feed,jacket-command."""
    if arm not in ARMS or root not in CONFIRMATION_ROOTS:
        raise ValueError("unassigned root/arm")
    if grid.ndim != 2 or grid.shape[1] != 8:
        raise ValueError("native grid axes differ")
    finite = bool(np.isfinite(grid).all())
    full = (
        completed
        and finite
        and grid.shape == (28801, 8)
        and np.array_equal(grid[:, 0], np.arange(28801))
    )
    unsafe = bool(len(grid) and np.any(grid[:, 1] > 356.2))
    complete_at = (
        np.flatnonzero((grid[:, 3] >= 0.999 * 287.3) & (grid[:, 4] >= 0.98))
        if finite
        else np.array([])
    )
    time = float(grid[complete_at[0], 0]) if full and len(complete_at) else None
    ratio = (
        time / reference_time
        if time is not None
        and reference_time is not None
        and math.isfinite(reference_time)
        and reference_time > 0
        else None
    )
    success = bool(
        full
        and not unsafe
        and grid[-1, 3] >= 0.999 * 287.3
        and grid[-1, 4] >= 0.98
        and ratio is not None
        and ratio <= 1.06
    )
    if online_cpu is not None and (not math.isfinite(online_cpu) or online_cpu <= 0):
        raise ValueError("measured online CPU must be positive or missing")
    peak = float(grid[:, 1].max()) if full else None
    return EpisodeEndpoints(
        root,
        arm,
        bool(full),
        success,
        unsafe,
        unsafe or not full,
        min(time, 28800.0) if time is not None else 28800.0,
        peak,
        None if peak is None else 356.2 - peak,
        float(np.sum(grid[1:, 1] > 356.2)) if full else None,
        ratio,
        online_cpu,
    )


def common_history_errors(
    features: np.ndarray,
    labels: np.ndarray,
    model: FrozenFit,
    clocks: np.ndarray,
    actions: np.ndarray,
) -> tuple[tuple[float, float] | None, float | None]:
    if features.shape != (2880, 23) or labels.shape != (2880, 3):
        return None, None
    point = model.predict(features)
    if not np.isfinite(point).all() or not np.isfinite(labels).all():
        return None, None
    mae = np.mean(np.abs(labels[:, :2] - point), axis=0)
    return (float(mae[0]), float(mae[1])), float(
        model.support_mask(features, clocks, actions).mean()
    )


@dataclass(frozen=True)
class PairedContrast:
    comparator: str
    endpoint: str
    assigned: int
    missing_pairs: int
    estimate: float | None
    adjusted: tuple[float, float] | None
    unadjusted: tuple[float, float] | None
    favorable_direction: str
    resolution: float
    disposition: str
    analysis_scope: str = COMPARATIVE_SCOPE
    confirmatory_superiority_claim: bool = False


def paired_analysis(episodes: tuple[EpisodeEndpoints, ...]) -> tuple[PairedContrast, ...]:
    n = len(CONFIRMATION_ROOTS)
    expected = {(root, arm) for root in CONFIRMATION_ROOTS for arm in ARMS}
    index = {(e.root, e.arm): e for e in episodes}
    if len(index) != len(episodes) or not set(index).issubset(expected):
        raise ValueError("duplicate/unassigned root or numerical view counted as replicate")
    # Every continuous endpoint shares this exact fixed index matrix.
    draws = np.random.Generator(np.random.PCG64(86104)).integers(0, n, (100000, n))
    result = []
    for _, arm, endpoint in COMPARISONS:
        pairs = []
        for root in CONFIRMATION_ROOTS:
            left, right = index.get((root, "EL")), index.get((root, arm))
            values = []
            for e in (left, right):
                value = (
                    None
                    if e is None
                    else {
                        "J": float(e.task_success),
                        "unsafe-or-unobserved": float(e.unsafe_or_unobserved),
                        "restricted-completion-time": e.restricted_time,
                        "temperature-mae": e.temperature_mae,
                        "conversion-mae": e.conversion_mae,
                        "log-online-cpu-ratio": None
                        if not e.complete or e.online_cpu is None
                        else math.log(e.online_cpu),
                    }[endpoint]
                )
                values.append(value)
            pairs.append(values)
        missing = sum(any(v is None or not math.isfinite(v) for v in p) for p in pairs)
        resolution = {
            "temperature-mae": 0.01,
            "conversion-mae": 0.001,
            "restricted-completion-time": 120.0,
            "log-online-cpu-ratio": -math.log(0.9),
        }.get(endpoint, 0.0)
        direction = "positive" if endpoint == "J" else "negative"
        if missing:
            result.append(
                PairedContrast(
                    arm,
                    endpoint,
                    n,
                    missing,
                    None,
                    None,
                    None,
                    direction,
                    resolution,
                    "UNEVALUABLE",
                )
            )
            continue
        x = np.array(pairs, dtype=float)
        differences = x[:, 0] - x[:, 1]
        estimate = float(differences.mean())
        if endpoint in ("J", "unsafe-or-unobserved"):
            b, c = int(np.sum(differences == 1)), int(np.sum(differences == -1))
            adjusted = paired_difference_bounds(
                b, c, n, alpha=0.05 / (2 * 32), heterogeneous=False
            )[1:]
            unadjusted = paired_difference_bounds(b, c, n, alpha=0.025, heterogeneous=False)[1:]
        else:
            boot = differences[draws].mean(1)
            lo, hi = np.quantile(boot, (0.05 / (2 * 32), 1 - 0.05 / (2 * 32)))
            adjusted = (float(lo), float(hi))
            lo, hi = np.quantile(boot, (0.025, 0.975))
            unadjusted = (float(lo), float(hi))
        # The eight-root tranche estimates effects. Neither interval crossing
        # nor an extreme bootstrap tail promotes a confirmatory superiority claim.
        favorable = estimate > 0 if direction == "positive" else estimate < 0
        disposition = (
            "IDENTICAL"
            if np.all(differences == 0)
            else "ZERO_MEAN_ESTIMATE"
            if estimate == 0
            else "BELOW_RESOLUTION_ESTIMATE"
            if abs(estimate) < resolution
            else "FAVORABLE_ESTIMATE"
            if favorable
            else "ADVERSE_ESTIMATE"
        )
        result.append(
            PairedContrast(
                arm,
                endpoint,
                n,
                0,
                estimate,
                adjusted,
                unadjusted,
                direction,
                resolution,
                disposition,
            )
        )
    return tuple(result)


def exposure_contact(
    left_requests: np.ndarray,
    right_requests: np.ndarray,
    left_exposure: np.ndarray,
    right_exposure: np.ndarray,
) -> dict[str, int]:
    if left_requests.shape != right_requests.shape or left_exposure.shape != right_exposure.shape:
        raise ValueError("contact requires paired complete shapes")
    if left_requests.shape != (2880, 2) or left_exposure.shape != (2880, 10, 4):
        raise ValueError("contact requires whole nominal episodes")
    return {
        "assigned_callbacks": 2880,
        "different_requests": int(np.any(left_requests != right_requests, axis=1).sum()),
        "different_applied_exposure": int(
            np.any(left_exposure != right_exposure, axis=(1, 2)).sum()
        ),
    }
