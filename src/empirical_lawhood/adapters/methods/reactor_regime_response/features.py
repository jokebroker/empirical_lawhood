"Causal, action-independent features for the reactor regime-response study.\n\nThe physical observation clock is ten seconds behind callback time after the\ninitial observation.  A caller supplies only the prefix available at its\ncommitment callback; this module never opens an outcome or scenario record.\n"

from __future__ import annotations

import numpy as np


def causal_features(observations: np.ndarray, stages: np.ndarray, callback: int) -> np.ndarray:
    """Return the declared six current and twelve history coordinates.

    ``stages[i, 2:4]`` are the actual feed and jacket applied at callback i.
    The current callback has not yet applied an action and is excluded.
    """
    if (
        observations.ndim != 2
        or observations.shape[1] != 4
        or stages.ndim != 2
        or stages.shape[1] != 4
        or type(callback) is not int
        or not 0 <= callback < len(observations)
        or len(stages) < callback
    ):
        raise ValueError("causal prefix shape or callback differs")
    y = np.asarray(observations[: callback + 1], dtype=np.float64)
    applied = np.asarray(stages[:callback, 2:4], dtype=np.float64)
    if not np.isfinite(y).all() or not np.isfinite(applied).all():
        raise ValueError("causal prefix has nonfinite values")
    if not np.array_equal(y[:, 0], np.arange(callback + 1, dtype=np.float64) * 10):
        raise ValueError("causal observation clocks differ")
    if np.any(np.diff(y[:, 3]) < -1e-10):
        raise ValueError("observed dose decreases")
    previous = applied[-1] if callback else np.asarray((0.0, 316.0))
    current = y[-1]
    result = [
        current[0] / 28800,
        current[3] / 287.3,
        (current[1] - 318.4) / 40,
        (current[2] - 316) / 40,
        previous[0] / 0.032,
        (previous[1] - 316) / 40,
    ]
    for column in (1, 2):
        for steps in (6, 30):
            result.append((current[column] - y[max(0, callback - steps), column]) / 40)
    for steps in (6, 30):
        duration = min(callback, steps)
        result.append(float(np.mean(applied[-duration:, 0])) / 0.032 if duration else 0.0)
    start = max(0, callback - 30)
    # Left-constant observed integrands use only completed callback intervals.
    result.append(float(np.sum(y[start:callback, 1] - y[start:callback, 2])) / (40 * 30))
    result.append(float(np.sum(np.maximum(y[start:callback, 1] - 330, 0))) / (40 * 30))
    for column, initial in ((0, 0.0), (1, 316.0)):
        last = initial
        since = callback * 10
        for index in range(callback):
            value = float(applied[index, column])
            if abs(value - last) > 1e-12:
                since = (callback - index) * 10
            last = value
        result.append(min(300, since) / 300)
    availability = min(callback, 6) / 30, min(callback, 30) / 30
    result.extend(availability)
    return np.round(np.asarray(result, dtype=np.float64), 9)
