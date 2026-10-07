"""Prefix-only native validity and safety operands for selected preparations."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from empirical_lawhood.adapters.simulators.reactor_causal_response.interface import project_tape_stages
from empirical_lawhood.kernel.provenance import ObjectIdentity

from .causal_contexts import causal_contexts
from .records import RegimeCausalPreparation, RegimePrivatePreparation

ROUTES = ("prepared_t0", "c_q", "p_q")


@dataclass(frozen=True)
class PreparationEvidence:
    root: str
    route: str
    callback: int | None
    prefix_valid: bool
    nominal_max_K: float | None
    refined_max_K: float | None
    reasons: tuple[str, ...]

    @property
    def safe(self) -> bool:
        return (
            self.prefix_valid
            and self.nominal_max_K is not None
            and self.refined_max_K is not None
            and max(self.nominal_max_K, self.refined_max_K) <= 356.2
        )


def preparation_evidence(
    causal: RegimeCausalPreparation,
    private: RegimePrivatePreparation,
    route: str,
) -> PreparationEvidence:
    """Use only native prefixes available at the fixed decision callback.

    The persisted preparation may contain a full native continuation.  This
    reducer never tests its post-decision samples, so future poisoning cannot
    change an already sealed choice or selected-prefix qualification.
    """
    if route not in ROUTES:
        raise ValueError("selected preparation route differs")
    if (
        (private.root, private.role, private.seed)
        != (causal.root, causal.role, causal.seed)
        or private.causal_preparation
        != ObjectIdentity.from_record(f"{causal.root}.causal-preparation", causal)
    ):
        raise ValueError("private native preparation changed its causal parent")
    context = next(row for row in causal_contexts(causal) if row.name == route)
    callback = context.callback
    if callback is None or context.input_sha256 is None:
        return PreparationEvidence(
            causal.root, route, callback, False, None, None,
            tuple(sorted((*context.reasons, "NO_SELECTED_PREPARATION_CONTACT"))),
        )
    causal_arrays = causal.arrays.unpack()
    private_arrays = private.arrays.unpack()
    base = "exploration_unshifted" if route == "prepared_t0" else route[0]
    t0 = dict((name, value) for name, value, _ in causal.anchors)["prepared_t0"]
    reasons: set[str] = set()
    maxima: list[float | None] = []
    for view, dt in ((0, 1.0), (1, 0.5)):
        stem = f"{base}_v{view}"
        names = tuple(f"{stem}_{name}" for name in ("observations", "requests", "stages", "exposure"))
        grid_name = f"{stem}_grid"
        if not all(name in causal_arrays for name in names) or grid_name not in private_arrays:
            reasons.add(f"MISSING_SELECTED_PREPARATION_V{view}")
            maxima.append(None)
            continue
        observations, requests, stages, exposure = (causal_arrays[name] for name in names)
        grid = private_arrays[grid_name]
        steps = int(10 / dt)
        stop = callback * steps + 1
        if (
            observations.shape[1:] != (4,) or len(observations) <= callback
            or requests.shape[1:] != (2,) or len(requests) < callback
            or stages.shape[1:] != (4,) or len(stages) < callback
            or exposure.shape[1:] != (steps, 4) or len(exposure) < callback
            or grid.ndim != 2
            or grid.shape[1] != 8
            or len(grid) < stop
            or not all(
                np.isfinite(array).all()
                for array in (
                    observations[: callback + 1], requests[:callback],
                    stages[:callback], exposure[:callback], grid[:stop],
                )
            )
            or not np.array_equal(observations[: callback + 1, 0], np.arange(callback + 1) * 10)
            or not np.array_equal(grid[:stop, 0], np.arange(stop) * dt)
        ):
            reasons.add(f"INVALID_SELECTED_PREPARATION_PREFIX_V{view}")
            maxima.append(None)
            continue
        # Replace every future request before projecting the known actuator.
        # The projection is deterministic and may inspect no outcome samples.
        safe_tape = np.empty((2880, 2))
        safe_tape[:callback] = requests[:callback]
        safe_tape[callback:] = (0.0, float(stages[callback - 1, 3]))
        expected_stages, expected_exposure = project_tape_stages(safe_tape, dt)
        if (
            not np.array_equal(stages[:callback], expected_stages[:callback])
            or not np.array_equal(exposure[:callback], expected_exposure[:callback])
            or not np.array_equal(observations[: callback + 1, 3], grid[:stop:steps, 3])
        ):
            reasons.add(f"INVALID_SELECTED_PREPARATION_DELIVERY_V{view}")
        if route != "prepared_t0":
            if t0 is None:
                reasons.add("NO_PREPARED_ANCHOR")
            else:
                unshifted_exploration_requests = causal_arrays.get(f"exploration_unshifted_v{view}_requests")
                unshifted_exploration_observations = causal_arrays.get(f"exploration_unshifted_v{view}_observations")
                if (
                    unshifted_exploration_requests is None
                    or unshifted_exploration_observations is None
                    or not np.array_equal(requests[:t0], unshifted_exploration_requests[:t0])
                    or not np.array_equal(observations[: t0 + 1], unshifted_exploration_observations[: t0 + 1])
                ):
                    reasons.add(f"SELECTED_UNSHIFTED_EXPLORATION_PREFIX_CHANGED_V{view}")
        maximum = float(np.max(grid[:stop, 1]))
        if not isfinite(maximum):
            reasons.add(f"NONFINITE_SELECTED_TEMPERATURE_V{view}")
            maxima.append(None)
        else:
            maxima.append(maximum)
            if maximum > 356.2:
                reasons.add(f"UNSAFE_SELECTED_TEMPERATURE_V{view}")
    return PreparationEvidence(
        causal.root,
        route,
        callback,
        not reasons,
        maxima[0],
        maxima[1],
        tuple(sorted(reasons)),
    )
