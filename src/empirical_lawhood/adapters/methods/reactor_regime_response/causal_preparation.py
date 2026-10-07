"""Permitted causal preparation checks; private grids are evaluator inputs only."""

from __future__ import annotations

import numpy as np

from empirical_lawhood.adapters.simulators.reactor_causal_response.interface import project_tape_stages

from .causal_contexts import causal_contexts
from .records import RegimeCausalPreparation


def causal_delivery_prefix(
    observations: np.ndarray,
    requests: np.ndarray,
    stages: np.ndarray,
    exposure: np.ndarray,
    callback: int,
    dt: float,
) -> bool:
    """Validate only completed native actions and the available observation."""
    steps = int(10 / dt)
    if (
        not 0 < callback < 2880 or dt not in (1.0, .5)
        or observations.shape[1:] != (4,) or len(observations) <= callback
        or requests.shape[1:] != (2,) or len(requests) < callback
        or stages.shape[1:] != (4,) or len(stages) < callback
        or exposure.shape[1:] != (steps, 4) or len(exposure) < callback
        or not all(np.isfinite(a).all() for a in (
            observations[:callback + 1], requests[:callback],
            stages[:callback], exposure[:callback],
        ))
        or not np.array_equal(observations[:callback + 1, 0], np.arange(callback + 1) * 10)
    ):
        return False
    tape = np.empty((2880, 2))
    tape[:callback] = requests[:callback]
    tape[callback:] = (0.0, float(stages[callback - 1, 3]))
    projected_stages, projected_exposure = project_tape_stages(tape, dt)
    if (not np.array_equal(stages[:callback], projected_stages[:callback])
            or not np.array_equal(exposure[:callback], projected_exposure[:callback])):
        return False
    dose = 0.0
    doses = [dose]
    for interval in exposure[:callback]:
        for _, duration, feed, _ in interval:
            dose += duration * feed
        doses.append(dose)
    return bool(np.array_equal(observations[:callback + 1, 3], doses))


def causal_preparation_validity(
    causal: RegimeCausalPreparation, route: str,
) -> tuple[bool, tuple[str, ...]]:
    """Online checks use the nominal permitted stream and C-qualified recipe."""
    if route not in ("prepared_t0", "c_q", "p_q"):
        raise ValueError("undeclared selected preparation")
    context = next(item for item in causal_contexts(causal) if item.name == route)
    if context.callback is None or context.input_sha256 is None:
        return False, ("NO_SELECTED_CAUSAL_PREFIX",)
    arrays = causal.arrays.unpack()
    base = "exploration_unshifted" if route == "prepared_t0" else route[0]
    values = tuple(arrays[f"{base}_v0_{name}"] for name in (
        "observations", "requests", "stages", "exposure",
    ))
    reasons = []
    observations, requests, stages, exposure = values
    if not causal_delivery_prefix(observations, requests, stages, exposure, context.callback, 1.0):
        reasons.append("INVALID_SELECTED_CAUSAL_DELIVERY")
    if route != "prepared_t0":
        t0 = next(callback for name, callback, _ in causal.anchors if name == "prepared_t0")
        if t0 is None:
            reasons.append("NO_PREPARED_ANCHOR")
        else:
            observations, requests, _, _ = values
            if ("exploration_unshifted_v0_requests" not in arrays or "exploration_unshifted_v0_observations" not in arrays
                    or not np.array_equal(requests[:t0], arrays["exploration_unshifted_v0_requests"][:t0])
                    or not np.array_equal(observations[:t0 + 1], arrays["exploration_unshifted_v0_observations"][:t0 + 1])):
                reasons.append("SELECTED_CAUSAL_UNSHIFTED_EXPLORATION_PREFIX_CHANGED")
    return not reasons, tuple(reasons)
