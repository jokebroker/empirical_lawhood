"""Two-stage native source: preparation, then assays after a sealed-model barrier.

The installed campaign provider must keep the private preparation out of the
prediction task's input ports and verify the assay task's dependency receipt.
This module has no authority to issue or run itself.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Callable

import numpy as np

from empirical_lawhood.adapters.methods.reactor_prepared_feed_qualification.records import FeedDomain
from empirical_lawhood.adapters.methods.reactor_regime_response.config import ROOTS, ReactorRegimeResponseDesign
from empirical_lawhood.adapters.methods.reactor_regime_response.records import RegimeAssayPanel, RegimeCausalPreparation, RegimePrivatePreparation
from empirical_lawhood.adapters.methods.reactor_local_domain_qualification.records import LocalArrayPayload
from empirical_lawhood.adapters.methods.reactor_regime_response.causal_preparation import causal_delivery_prefix
from empirical_lawhood.adapters.simulators.reactor_causal_response.acquisition import ExplorationController, NativeEpisode, TapeController, acquire_episode, acquire_with_progress as _acquire_with_progress
from empirical_lawhood.adapters.simulators.reactor_causal_response.design import draw_scenario
from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_design import ReactorBatchSource
from empirical_lawhood.kernel.provenance import ObjectIdentity

from .panel import FEED_WORDS, assay_tape, preparation_tape, select_anchors

Acquire = Callable[..., NativeEpisode]
Progress = Callable[[int], None]


def _root_assignment(root: str) -> tuple[str, int]:
    found = [(role, seed) for name, role, _, seed in ROOTS if name == root]
    if len(found) != 1:
        raise ValueError("root is outside the frozen regime-response study assignment")
    return found[0]


def acquire_preparation(
    design: ReactorRegimeResponseDesign,
    source: ReactorBatchSource,
    prepared_domain: FeedDomain,
    root: str,
    *,
    selected_route: str | None = None,
    acquire: Acquire = acquire_episode,
    progress: Progress | None = None,
) -> tuple[RegimeCausalPreparation, RegimePrivatePreparation]:
    """At most six B/C or four route-selected D native preparations."""
    role, seed = _root_assignment(root)
    if (role == "prospective" and selected_route not in ("prepared_t0", "c_q", "p_q")) or (
        role != "prospective" and selected_route is not None
    ):
        raise ValueError("native preparation route differs from its assigned phase")
    scenario = draw_scenario(
        root, "heldout" if role in ("qualification", "prospective") else "calibration", seed
    )
    episodes: dict[str, NativeEpisode] = {}
    failures: list[tuple[str, str]] = []
    calls = completed = 0

    def run(
        key: str, name: str, controller: ExplorationController | TapeController, dt: float
    ) -> None:
        nonlocal calls, completed
        calls += 1
        episode, completed = _acquire_with_progress(
            source, scenario, name, controller, dt, acquire, completed, progress
        )
        if episode.root != root or episode.dt != dt:
            raise ValueError("native preparation returned another root or numerical view")
        episodes[key] = episode
        if not episode.complete:
            failures.append((key, episode.failure or "INCOMPLETE_NATIVE_BATCH"))

    run("exploration_unshifted_v0", "exploration-unshifted", ExplorationController(0, 0), 1.0)
    donor = episodes["exploration_unshifted_v0"]
    donor_tape_available = donor.requests.shape == (2880, 2) and np.isfinite(donor.requests).all()
    if donor_tape_available:
        run("exploration_unshifted_v1", "exploration-unshifted", TapeController(donor.requests), 0.5)
    else:
        failures.append(("exploration_unshifted_v1", "DECLARED_DONOR_CONTINUATION_UNAVAILABLE"))
    anchors = select_anchors(donor, prepared_domain)
    t0 = anchors[3].callback
    selected_paths = (
        ("C", "P")
        if selected_route is None
        else ()
        if selected_route == "prepared_t0"
        else (selected_route[0].upper(),)
    )
    if t0 is not None and donor_tape_available:
        jacket = float(donor.stages[t0 - 1, 3])
        for path in selected_paths:
            tape = preparation_tape(donor.requests, t0, path, jacket)
            key = path.lower()
            run(f"{key}_v0", path, TapeController(tape), 1.0)
            nominal = episodes[f"{key}_v0"]
            if not np.array_equal(nominal.requests, tape[: len(nominal.requests)]):
                failures.append((f"{key}_v0", "PREPARATION_REQUEST_TAPE_CHANGED"))
            if not np.array_equal(nominal.observations[: t0 + 1], donor.observations[: t0 + 1]):
                failures.append((f"{key}_v0", "PREPARATION_PREFIX_CHANGED"))
            run(f"{key}_v1", path, TapeController(tape), 0.5)
            refined = episodes[f"{key}_v1"]
            if not np.array_equal(refined.requests, tape[: len(refined.requests)]):
                failures.append((f"{key}_v1", "REFINED_REQUEST_TAPE_CHANGED"))
    else:
        failures.extend(
            (
                f"{path.lower()}_v{view}",
                "PREPARED_INTERIOR_NO_CONTACT"
                if t0 is None
                else "DECLARED_DONOR_CONTINUATION_UNAVAILABLE",
            )
            for path in selected_paths
            for view in (0, 1)
        )
    causal_arrays: dict[str, np.ndarray] = {}
    private_arrays: dict[str, np.ndarray] = {}
    for key, episode in episodes.items():
        for field in ("observations", "requests", "stages", "exposure"):
            causal_arrays[f"{key}_{field}"] = getattr(episode, field).copy()
        for field in ("grid", "callback_cpu"):
            private_arrays[f"{key}_{field}"] = getattr(episode, field).copy()
    anchor_values = tuple((a.name, a.callback, a.reason) for a in anchors)
    causal = RegimeCausalPreparation(
        root,
        role,
        seed,
        ObjectIdentity.from_record(design.config_id, design),
        source.fingerprint(),
        anchor_values,
        LocalArrayPayload.pack(causal_arrays),
        calls,
        tuple(failures),
    )
    private = RegimePrivatePreparation(
        root,
        role,
        seed,
        ObjectIdentity.from_record(f"{root}.causal-preparation", causal),
        LocalArrayPayload.pack(private_arrays),
        calls,
        tuple(failures),
    )
    return causal, private


def _episode(
    root: str,
    key: str,
    causal: dict[str, np.ndarray],
    private: dict[str, np.ndarray],
) -> NativeEpisode | None:
    if f"{key}_observations" not in causal:
        return None
    return NativeEpisode(
        root,
        key,
        0.5 if key.endswith("v1") else 1.0,
        causal[f"{key}_observations"],
        causal[f"{key}_requests"],
        causal[f"{key}_stages"],
        causal[f"{key}_exposure"],
        private[f"{key}_grid"],
        private[f"{key}_callback_cpu"],
        None if len(causal[f"{key}_observations"]) == 2880 else "INCOMPLETE_NATIVE_BATCH",
    )


def acquire_assays(
    source: ReactorBatchSource,
    causal_record: RegimeCausalPreparation,
    private_record: RegimePrivatePreparation,
    sealed_prediction: ObjectIdentity,
    *,
    acquire: Acquire = acquire_episode,
    progress: Progress | None = None,
) -> RegimeAssayPanel:
    """Exactly 48 assigned slots, after the campaign verifies a sealed receipt."""
    root = causal_record.root
    if (
        (private_record.root, private_record.role, private_record.seed)
        != (root, causal_record.role, causal_record.seed)
        or private_record.causal_preparation
        != ObjectIdentity.from_record(f"{root}.causal-preparation", causal_record)
        or source.fingerprint() != causal_record.source_sha256
    ):
        raise ValueError("assay source or persisted preparation identity differs")
    scenario = draw_scenario(
        root,
        "heldout" if causal_record.role in ("qualification", "prospective") else "calibration",
        causal_record.seed,
    )
    causal = causal_record.arrays.unpack()
    private = private_record.arrays.unpack()
    arrays: dict[str, np.ndarray] = {}
    slots: list[tuple[str, int | None, int, int, str]] = []
    failures: list[tuple[str, str]] = []
    calls = completed = 0
    anchor = {name: callback for name, callback, _ in causal_record.anchors}
    contexts = [(name, anchor[name], "exploration_unshifted") for name in ("early", "middle", "late", "prepared_t0")]
    t0 = anchor["prepared_t0"]
    for path in ("c", "p"):
        for label, offset in (("q", 33), ("q600", 93)):
            contexts.append((f"{path}_{label}", None if t0 is None else t0 + offset, path))
    for name, callback, base in contexts:
        for view in (0, 1):
            episode = _episode(root, f"{base}_v{view}", causal, private)
            for word, feed in enumerate(FEED_WORDS):
                key = f"{name}_v{view}_a{word}"
                if callback is None or episode is None or len(episode.observations) <= callback:
                    status = "NO_CAUSAL_PREPARATION_CONTACT"
                    slots.append((name, callback, view, word, status))
                    failures.append((key, status))
                    continue
                try:
                    tape = assay_tape(episode, callback, feed)
                except ValueError:
                    slots.append((name, callback, view, word, "DECLARED_CONTINUATION_UNAVAILABLE"))
                    failures.append((key, "DECLARED_CONTINUATION_UNAVAILABLE"))
                    continue
                calls += 1
                branch, completed = _acquire_with_progress(
                    source,
                    scenario,
                    key,
                    TapeController(tape),
                    0.5 if view else 1.0,
                    acquire,
                    completed,
                    progress,
                )
                if branch.failure is not None:
                    failures.append((f"{key}.continuation", branch.failure))
                step = int(10 / branch.dt)
                if (
                    branch.grid.shape[1:] != (8,)
                    or len(branch.grid) < (callback + 1) * step + 1
                    or len(branch.exposure) <= callback
                    or len(branch.stages) <= callback
                    or len(branch.requests) <= callback
                ):
                    status = "LOCAL_MEASUREMENT_UNAVAILABLE"
                    slots.append((name, callback, view, word, status))
                    failures.append((key, status))
                    continue
                window = branch.grid[callback * step : (callback + 1) * step + 1].copy()
                arrays[f"{key}_grid"] = window
                arrays[f"{key}_exposure"] = branch.exposure[callback].copy()
                arrays[f"{key}_stages"] = branch.stages[callback].copy()
                arrays[f"{key}_request"] = branch.requests[callback].copy()
                # A finite local endpoint cannot redeem an invalid earlier
                # requested/applied/realized prefix used by its causal features.
                arrays[f"{key}_valid"] = np.asarray(
                    [
                        causal_delivery_prefix(
                            branch.observations,
                            branch.requests,
                            branch.stages,
                            branch.exposure,
                            callback + 1,
                            branch.dt,
                        )
                    ],
                    dtype=np.uint8,
                )
                arrays[f"{key}_full_grid_sha256"] = np.frombuffer(
                    sha256(branch.grid.tobytes()).digest(), dtype=np.uint8
                ).copy()
                arrays[f"{key}_prefix_sha256"] = np.frombuffer(
                    sha256(branch.observations[: callback + 1].tobytes()).digest(), dtype=np.uint8
                ).copy()
                if not np.array_equal(
                    branch.observations[: callback + 1], episode.observations[: callback + 1]
                ):
                    failures.append((key, "PREPARED_PREFIX_CHANGED"))
                if not np.array_equal(branch.requests[: callback + 1], tape[: callback + 1]):
                    failures.append((key, "ASSAY_CONTINUATION_CHANGED"))
                slots.append((name, callback, view, word, "MEASURED"))
    return RegimeAssayPanel(
        root,
        causal_record.role,
        causal_record.seed,
        ObjectIdentity.from_record(f"{root}.causal-preparation", causal_record),
        ObjectIdentity.from_record(f"{root}.private-preparation", private_record),
        sealed_prediction,
        tuple(slots),
        LocalArrayPayload.pack(arrays),
        calls,
        tuple(failures),
    )
