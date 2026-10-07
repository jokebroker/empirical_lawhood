"""Eight outcome-blind prediction inputs from exact available native prefixes."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite

import numpy as np

from empirical_lawhood.adapters.methods.reactor_causal_response.numerical import Observation
from empirical_lawhood.adapters.simulators.reactor_causal_response.interface import Actuator

from .features import causal_features
from .measured_panel import CONTEXTS
from .records import RegimeCausalPreparation

FEED_WORDS = (0.0, 0.016, 0.032)


@dataclass(frozen=True)
class CausalContext:
    root: str
    name: str
    callback: int | None
    input_sha256: str | None
    current: tuple[float, ...] | None
    history: tuple[float, ...] | None
    projected_masses_kg: tuple[float, float, float] | None
    projection_valid: tuple[bool, bool, bool]
    reasons: tuple[str, ...]


def _empty(root: str, name: str, callback: int | None, reason: str) -> CausalContext:
    return CausalContext(root, name, callback, None, None, None, None, (False,) * 3, (reason,))


def causal_contexts(preparation: RegimeCausalPreparation) -> tuple[CausalContext, ...]:
    """Future callbacks and private grids never enter any context digest/feature."""
    arrays = preparation.arrays.unpack()
    anchors = {name: callback for name, callback, _ in preparation.anchors}
    t0 = anchors["prepared_t0"]
    rows = []
    for name in CONTEXTS:
        callback = (
            anchors[name]
            if name in anchors
            else None if t0 is None else t0 + (33 if name.endswith("_q") else 93)
        )
        if callback is None:
            rows.append(_empty(preparation.root, name, None, "NO_CAUSAL_CONTACT"))
            continue
        base = "exploration_unshifted" if name in CONTEXTS[:4] else name[0]
        prefix = f"{base}_v0"
        keys = tuple(f"{prefix}_{field}" for field in ("observations", "requests", "stages", "exposure"))
        if not all(key in arrays for key in keys):
            rows.append(_empty(preparation.root, name, callback, "MISSING_CAUSAL_NATIVE_PREFIX"))
            continue
        observations, requests, stages, exposure = (arrays[key] for key in keys)
        if (
            not 0 < callback < 2880
            or observations.shape[1:] != (4,) or len(observations) <= callback
            or requests.shape[1:] != (2,) or len(requests) < callback
            or stages.shape[1:] != (4,) or len(stages) < callback
            or exposure.ndim != 3
            or exposure.shape[0] < callback
            or exposure.shape[2] != 4
            or not np.isfinite(observations[: callback + 1]).all()
            or not np.isfinite(requests[:callback]).all()
            or not np.isfinite(stages[:callback]).all()
            or not np.isfinite(exposure[:callback]).all()
        ):
            rows.append(_empty(preparation.root, name, callback, "INVALID_CAUSAL_NATIVE_PREFIX"))
            continue
        if not np.array_equal(observations[: callback + 1, 0], np.arange(callback + 1) * 10):
            rows.append(_empty(preparation.root, name, callback, "SHIFTED_CAUSAL_CALLBACK_CLOCK"))
            continue
        digest = sha256()
        digest.update(observations[: callback + 1].tobytes())
        digest.update(requests[:callback].tobytes())
        digest.update(stages[:callback].tobytes())
        digest.update(exposure[:callback].tobytes())
        try:
            features = causal_features(observations, stages, callback)
            observation = Observation(*map(float, observations[callback]))
            previous = (
                float(stages[callback - 1, 2]),
                float(stages[callback - 1, 3]),
            )
            projections = tuple(
                Actuator().project_request(
                    observation, previous, (feed, previous[1]), 1 + 3 * index
                )
                for index, feed in enumerate(FEED_WORDS)
            )
            masses = tuple(
                float(sum(dt * realized for _, dt, realized, _ in projection.exposure))
                for projection in projections
            )
            valid = tuple(
                projection.requested == (FEED_WORDS[index], previous[1])
                and projection.applied[1] == previous[1]
                and all(jacket == previous[1] and realized >= 0 for _, _, realized, jacket in projection.exposure)
                and isfinite(masses[index])
                for index, projection in enumerate(projections)
            )
            if not (
                len(features) == 18
                and np.isfinite(features).all()
                and masses[0] == 0 < masses[1] < masses[2]
                and all(valid)
            ):
                raise ValueError("causal feature or three-word native projection invalid")
        except (ValueError, np.linalg.LinAlgError) as error:
            rows.append(_empty(preparation.root, name, callback, f"CAUSAL_PROJECTION_FAILED:{error}"))
            continue
        rows.append(
            CausalContext(
                preparation.root,
                name,
                callback,
                digest.hexdigest(),
                tuple(float(value) for value in features[:6]),
                tuple(float(value) for value in features),
                masses,  # type: ignore[arg-type]
                valid,  # type: ignore[arg-type]
                (),
            )
        )
    return tuple(rows)
