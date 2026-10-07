"Fixed preparation-policy development census and causal observation contract."

from typing import Any

import numpy as np
from ..science import FiniteResponseLawScienceSpec, development_requests
from empirical_lawhood.adapters.simulators.finite_response_law.instruments import REFERENCE_INSTRUMENT
from empirical_lawhood.adapters.simulators.finite_response_law.preparation_policy_contracts import PREPARATION_POLICY_SCHEDULE_IDS

CANDIDATES = ("fitted-baseline-radial-shift", "observed-features-radial-shift", "position-momentum-enriched-baseline-radial-shift", "schedule-map-comparator")
FOLDS = np.arange(24, dtype=np.int64) % 4
SCHEDULES = PREPARATION_POLICY_SCHEDULE_IDS
FEATURE_IDS = (*REFERENCE_INSTRUMENT.feature_ids, "native-x-momentum-hs", "native-y-momentum-hs")
FEATURE_UNITS = (
    *REFERENCE_INSTRUMENT.feature_units,
    "native-reduced-position*native-reduced-momentum",
    "native-reduced-position*native-reduced-momentum",
)


def inner_folds(train: Any) -> Any:
    labels = np.empty(len(train), dtype=np.int64)
    for cohort in (train < 8, train >= 8):
        positions = np.flatnonzero(cohort)
        labels[positions] = np.arange(len(positions)) % 3
    return labels


def requests() -> Any:
    source = development_requests(FiniteResponseLawScienceSpec())
    rows = np.r_[np.arange(8), 16 + np.arange(16)]
    return source["direction"][rows], source["lower"][rows]


def enriched_inputs(x: Any, positions: Any, momenta: Any, ticks: Any) -> Any:
    shape = (24, 9, 2, 26, 2, 3, 4, 4)
    if x.shape != (24, 24, 2) or positions.shape != shape or momenta.shape != shape:
        raise ValueError("preparation-policy development native axes differ")
    if not np.array_equal(ticks, 4096 + np.arange(26) * 16):
        raise ValueError("preparation-policy development trajectory clocks differ")
    for states in (positions, momenta):
        if not np.isfinite(states).all() or not np.array_equal(
            states[:, :, :, 0], np.broadcast_to(states[:, :1, :, 0], states[:, :, :, 0].shape)
        ):
            raise ValueError("preparation-policy development t0 states differ across schedules")
    contractions = np.real(
        np.sum(positions[:, 0, 0, 0].conj() * momenta[:, 0, 0, 0], axis=(2, 3, 4))
    )
    result = np.column_stack((x[..., 0], contractions))
    if result.shape != (24, 26) or len(set(FEATURE_IDS)) != 26:
        raise ValueError("position-momentum-enriched baseline requires 26 unique inputs")
    return result
