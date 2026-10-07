"Fixed reactor regime-response study anchors, preparation tapes and branch identities.\n\nThese functions have no native effect. An issued provider must persist the\npreparation and sealed predictions before calling the separate assay stage.\n"

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from empirical_lawhood.adapters.methods.reactor_causal_response.numerical import Observation
from empirical_lawhood.adapters.methods.reactor_prepared_feed_qualification.records import FeedDomain
from empirical_lawhood.adapters.simulators.reactor_causal_response.acquisition import NativeEpisode
from empirical_lawhood.adapters.simulators.reactor_causal_response.interface import Actuator
from empirical_lawhood.adapters.simulators.reactor_prepared_feed_qualification.acquisition import candidate_features

BASELINE_WINDOWS = ((600, 1200), (7200, 9000), (14400, 16200))
PREPARED_WINDOW = (600, 6000)
FEED_WORDS = (0.0, 0.016, 0.032)
PROBE_BLOCKS = (0.0, 0.032, 0.032, 0.0, 0.032, 0.0)


@dataclass(frozen=True)
class Anchor:
    name: str
    callback: int | None
    reason: str


def _three_word_chart(donor: NativeEpisode, callback: int) -> bool:
    if (not 0 < callback < min(2880, len(donor.observations))
            or len(donor.stages) < callback
            or not np.isfinite(donor.observations[:callback + 1]).all()
            or not np.isfinite(donor.stages[:callback]).all()):
        return False
    previous = (
        (0.0, 316.0)
        if callback == 0
        else (float(donor.stages[callback - 1, 2]), float(donor.stages[callback - 1, 3]))
    )
    row = donor.observations[callback]
    if previous[0] > 0.016 or not 600 <= row[0] < 25200:
        return False
    observation = Observation(*map(float, row))
    projections = tuple(
        Actuator().project_request(
            observation, previous, (feed, previous[1]), 1 + 3 * index
        )
        for index, feed in enumerate(FEED_WORDS)
    )
    return (
        projections[0].mean_feed == 0
        and len({p.delivery_key for p in projections}) == 3
        and all(p.applied[1] == previous[1] for p in projections)
    )


def select_anchors(donor: NativeEpisode, prepared_domain: FeedDomain) -> tuple[Anchor, ...]:
    """First eligible callback in each frozen window; no response labels read."""
    if prepared_domain.domain_id != "d11010" or prepared_domain.actions != (1, 4, 7):
        raise ValueError("prepared interior eligibility binding differs")
    anchors = []
    for name, (low, high) in zip(
        ("early", "middle", "late"), BASELINE_WINDOWS, strict=True
    ):
        callback = next(
            (k for k in range(low // 10, high // 10 + 1) if _three_word_chart(donor, k)),
            None,
        )
        anchors.append(Anchor(name, callback, "CONTACT" if callback is not None else "NO_CONTACT"))
    prepared = None
    for k in range(PREPARED_WINDOW[0] // 10, PREPARED_WINDOW[1] // 10 + 1):
        if not _three_word_chart(donor, k):
            continue
        x, _ = candidate_features(donor, k)
        if prepared_domain.proposed_support(x[[1, 4, 7]], np.asarray((1, 4, 7))).all():
            prepared = k
            break
    anchors.append(
        Anchor("prepared_t0", prepared, "CONTACT" if prepared is not None else "NO_CONTACT")
    )
    return tuple(anchors)


def preparation_tape(
    donor_requests: np.ndarray,
    t0_callback: int,
    path: str,
    pre_t0_applied_jacket_K: float,
) -> np.ndarray:
    "C/P share the exact initial native preparation prefix and fixed 300 s requested-feed integral."
    if (
        donor_requests.shape != (2880, 2)
        or not np.isfinite(donor_requests).all()
        or type(t0_callback) is not int
        or not 60 <= t0_callback <= 600
        or t0_callback + 94 >= 2880
        or path not in ("C", "P")
        or not np.isfinite(pre_t0_applied_jacket_K)
    ):
        raise ValueError("preparation tape input differs")
    tape = donor_requests.copy()
    held_jacket = float(pre_t0_applied_jacket_K)
    for offset in range(95):
        feed = (
            0.016
            if path == "C" or offset >= 30
            else PROBE_BLOCKS[offset // 5]
        )
        tape[t0_callback + offset] = (feed, held_jacket)
    if not np.array_equal(tape[:t0_callback], donor_requests[:t0_callback]):
        raise ValueError("preparation changed its initial causal prefix")
    if not np.isclose(np.sum(tape[t0_callback : t0_callback + 30, 0]) * 10, 4.8):
        raise ValueError("C/P requested feed integral differs")
    return tape


def assay_tape(episode: NativeEpisode, callback: int, feed: float) -> np.ndarray:
    """One ten-second branch from the episode's unaltered prepared context."""
    if (feed not in FEED_WORDS or not 0 < callback < min(2880, len(episode.observations))
            or len(episode.stages) < callback
            or episode.requests.shape != (2880, 2)
            or not np.isfinite(episode.requests).all()):
        raise ValueError("assay lacks a causal context or declared continuation tape")
    tape = episode.requests.copy()
    jacket = float(episode.stages[callback - 1, 3])
    tape[callback] = (feed, jacket)
    return tape
