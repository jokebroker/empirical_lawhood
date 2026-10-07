"""Receipt-gated selected-action acquisition through the existing native bridge."""

from __future__ import annotations

from hashlib import sha256
from typing import Callable

import numpy as np

from empirical_lawhood.adapters.methods.reactor_selected_action_response.config import assignment
from empirical_lawhood.adapters.methods.reactor_selected_action_response.records import ClassicalAssay, ClassicalCausal, ClassicalDecision, ClassicalPrivate
from empirical_lawhood.adapters.methods.reactor_local_domain_qualification.records import LocalArrayPayload
from empirical_lawhood.adapters.methods.reactor_regime_response.causal_preparation import causal_delivery_prefix
from empirical_lawhood.adapters.simulators.reactor_causal_response.acquisition import ExplorationController, NativeEpisode, TapeController, acquire_episode
from empirical_lawhood.adapters.simulators.reactor_causal_response.design import draw_scenario
from empirical_lawhood.adapters.simulators.reactor_regime_response.acquisition import _acquire_with_progress
from empirical_lawhood.adapters.simulators.reactor_regime_response.panel import assay_tape, select_anchors
from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_design import ReactorBatchSource
from empirical_lawhood.kernel.provenance import ObjectIdentity
from .config import ClassicalNativeConfig

FIELDS = ("observations", "requests", "stages", "exposure", "grid", "callback_cpu")


def acquire_preparation(
    config: ClassicalNativeConfig,
    source: ReactorBatchSource,
    root: str,
    *,
    released: bool,
    acquire: Callable[..., NativeEpisode] = acquire_episode,
    progress: Callable[[int], None] | None = None,
) -> tuple[ClassicalCausal, ClassicalPrivate]:
    role, seed = assignment(root)
    if source.fingerprint() != config.source_sha256:
        raise ValueError("classical preparation changed its pinned source")
    if role == "qualification" and not released:
        raise ValueError("qualification preparation requires its issued experiment")
    episodes: list[NativeEpisode] = []
    reasons: list[str] = []
    callback = None
    calls = completed = 0
    if released:
        scenario = draw_scenario(root, "heldout", seed)
        nominal, completed = _acquire_with_progress(
            source,
            scenario,
            "exploration-unshifted",
            ExplorationController(0, 0),
            1.0,
            acquire,
            completed,
            progress,
        )
        episodes.append(nominal)
        calls += 1
        if nominal.complete:
            refined, completed = _acquire_with_progress(
                source,
                scenario,
                "exploration-unshifted",
                TapeController(nominal.requests),
                0.5,
                acquire,
                completed,
                progress,
            )
            calls += 1
            episodes.append(refined)
            if refined.complete and np.array_equal(nominal.requests, refined.requests):
                callback = next(
                    a.callback
                    for a in select_anchors(nominal, config.prepared_domain)
                    if a.name == "prepared_t0"
                )
                if callback is None:
                    reasons.append("NO_CAUSAL_PREPARATION_CONTACT")
            else:
                reasons.append("REFINED_PREPARATION_INVALID")
        else:
            reasons.append("NOMINAL_PREPARATION_INVALID")
    else:
        reasons.append("LOCAL_LAW_PREREQUISITE_NONENTRY")
    causal_arrays, private_arrays = {}, {}
    for view, episode in enumerate(episodes):
        if episode.root != root or episode.dt != (1.0 if view == 0 else 0.5):
            raise ValueError("classical preparation substituted root or view")
        for field in FIELDS:
            value = getattr(episode, field)
            private_arrays[f"exploration_unshifted_v{view}_{field}"] = value
            if callback is not None and field not in ("grid", "callback_cpu"):
                causal_arrays[f"exploration_unshifted_v{view}_{field}"] = value[
                    : callback + int(field == "observations")
                ].copy()
    causal = ClassicalCausal(
        root,
        role,
        seed,
        ObjectIdentity.from_record(config.design.config_id, config.design),
        source.fingerprint(),
        callback,
        LocalArrayPayload.pack(causal_arrays),
        calls,
        tuple(reasons),
    )
    private = ClassicalPrivate(
        root,
        ObjectIdentity.from_record(f"{root}.causal-preparation", causal),
        LocalArrayPayload.pack(private_arrays),
    )
    return causal, private


def donor_episode(
    causal: ClassicalCausal, private: ClassicalPrivate, view: int
) -> NativeEpisode:
    if private.causal_preparation != ObjectIdentity.from_record(
        f"{causal.root}.causal-preparation", causal
    ):
        raise ValueError("classical private/causal preparation differs")
    arrays = private.arrays.unpack()
    return NativeEpisode(
        causal.root,
        "exploration-unshifted",
        0.5 if view else 1.0,
        **{field: arrays[f"exploration_unshifted_v{view}_{field}"] for field in FIELDS},
        failure=None,
    )


def acquire_assays(
    source: ReactorBatchSource,
    causal: ClassicalCausal,
    private: ClassicalPrivate,
    decision: ClassicalDecision,
    *,
    acquire: Callable[..., NativeEpisode] = acquire_episode,
    progress: Callable[[int], None] | None = None,
) -> ClassicalAssay:
    parent = ObjectIdentity.from_record(f"{causal.root}.causal-preparation", causal)
    if (
        decision.root != causal.root
        or decision.causal_preparation != parent
        or private.causal_preparation != parent
        or source.fingerprint() != causal.source_sha256
    ):
        raise ValueError("classical assay changed its sealed causal input")
    arrays, reasons = {}, []
    calls = completed = 0
    if decision.causal_preparation_valid:
        callback = decision.callback
        assert callback is not None
        scenario = draw_scenario(causal.root, "heldout", causal.seed)
        for view in (0, 1):
            donor = donor_episode(causal, private, view)
            arrays[f"v{view}_prefix_peak_K"] = np.asarray(
                (donor.grid[: callback * int(10 / donor.dt) + 1, 1].max(),)
            )
            for word, feed in enumerate((0.0, 0.016)):
                key = f"v{view}_a{word}"
                tape = assay_tape(donor, callback, feed)
                branch, completed = _acquire_with_progress(
                    source,
                    scenario,
                    key,
                    TapeController(tape),
                    donor.dt,
                    acquire,
                    completed,
                    progress,
                )
                calls += 1
                if not branch.complete:
                    reasons.append(f"{key}.INCOMPLETE_NATIVE_ASSAY")
                    continue
                steps = int(10 / branch.dt)
                for field in ("requests", "stages", "exposure"):
                    arrays[f"{key}_{field}"] = getattr(branch, field)[callback].copy()
                arrays[f"{key}_grid"] = branch.grid[
                    callback * steps : (callback + 1) * steps + 1
                ].copy()
                prefix_valid = (
                    np.array_equal(
                        branch.observations[: callback + 1], donor.observations[: callback + 1]
                    )
                    and np.array_equal(branch.requests, tape)
                    and causal_delivery_prefix(
                        branch.observations,
                        branch.requests,
                        branch.stages,
                        branch.exposure,
                        callback + 1,
                        branch.dt,
                    )
                )
                arrays[f"{key}_valid"] = np.asarray((prefix_valid,), dtype=np.uint8)
                arrays[f"{key}_prefix_sha256"] = np.frombuffer(
                    sha256(branch.observations[: callback + 1].tobytes()).digest(), dtype=np.uint8
                ).copy()
                arrays[f"{key}_full_grid_sha256"] = np.frombuffer(
                    sha256(branch.grid.tobytes()).digest(), dtype=np.uint8
                ).copy()
    else:
        reasons.extend(decision.reasons)
    return ClassicalAssay(
        causal.root,
        parent,
        ObjectIdentity.from_record(decision.decision_id, decision),
        LocalArrayPayload.pack(arrays),
        calls,
        tuple(sorted(set(reasons))),
    )
