"""Versioned pulse assay using the existing source and acquisition owner."""

from hashlib import sha256
from typing import Callable

import numpy as np

from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.config import CONTEXTS, GUARD, WORDS, FrontierDesign, assignment
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.records import CAUSAL_FIELDS, NATIVE_FIELDS, FrontierAssay, FrontierContext, FrontierPreparation, FrontierPrivate
from empirical_lawhood.adapters.methods.reactor_local_domain_qualification.records import LocalArrayPayload
from empirical_lawhood.adapters.methods.reactor_regime_response.causal_preparation import causal_delivery_prefix
from empirical_lawhood.adapters.simulators.reactor_causal_response.acquisition import ExplorationController, NativeEpisode, TapeController, acquire_episode
from empirical_lawhood.adapters.simulators.reactor_causal_response.design import draw_scenario
from empirical_lawhood.adapters.simulators.reactor_causal_response.acquisition import acquire_with_progress
from empirical_lawhood.adapters.simulators.reactor_regime_response.panel import select_anchors
from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_design import ReactorBatchSource, ReactorAssignedScenario
from empirical_lawhood.kernel.provenance import ObjectIdentity
from .config import FrontierNativeConfig
from .words import pulse_tape


def pack_preparation(
    config: FrontierNativeConfig,
    root: str,
    episodes: tuple[NativeEpisode, ...],
    *,
    native_calls: int,
    retained_parents: tuple[ObjectIdentity, ...] = (),
    reasons: tuple[str, ...] = (),
) -> tuple[FrontierPreparation, FrontierPrivate]:
    """Cut each policy input independently; complete tapes/grids stay private."""
    anchors = (
        {a.name: a for a in select_anchors(episodes[0], config.prepared_domain)}
        if len(episodes) == 2 and all(e.complete for e in episodes)
        else {}
    )
    contexts = []
    for name in CONTEXTS:
        anchor = anchors.get(name)
        callback = None if anchor is None else anchor.callback
        ca = {}
        failures = list(reasons)
        if callback is not None:
            nominal = episodes[0]
            if not causal_delivery_prefix(
                nominal.observations,
                nominal.requests,
                nominal.stages,
                nominal.exposure,
                callback,
                1.0,
            ):
                failures.append("INVALID_CAUSAL_DELIVERY_PREFIX")
            for field in CAUSAL_FIELDS:
                ca[field] = getattr(nominal, field)[
                    : callback + int(field == "observations")
                ].copy()
        else:
            failures.append("NO_CAUSAL_CONTACT")
        contexts.append(
            FrontierContext(
                root, name, callback, LocalArrayPayload.pack(ca), tuple(sorted(set(failures)))
            )
        )
    data = {}
    for view, episode in enumerate(episodes):
        if episode.root != root or episode.dt != (1.0 if view == 0 else 0.5):
            raise ValueError("preparation changed its physical root or numerical view")
        data.update({f"v{view}_{field}": getattr(episode, field) for field in NATIVE_FIELDS})
    design = FrontierDesign()
    preparation = FrontierPreparation(
        root,
        ObjectIdentity.from_record(design.config_id, design),
        config.source_sha256,
        tuple(contexts),
        native_calls,
        retained_parents,
        reasons,
    )
    private = FrontierPrivate(
        root,
        ObjectIdentity.from_record(preparation.record_id, preparation),
        LocalArrayPayload.pack(data),
        tuple(e.failure for e in episodes),
    )
    return preparation, private


def acquire_preparation(
    config: FrontierNativeConfig,
    source: ReactorBatchSource,
    root: str,
    *,
    released: bool,
    acquire: Callable[..., NativeEpisode] = acquire_episode,
    progress: Callable[[int], None] | None = None,
) -> tuple[FrontierPreparation, FrontierPrivate]:
    role, seed = assignment(root)
    if role == "development" or source.fingerprint() != config.source_sha256:
        raise ValueError("fresh preparation cannot reacquire a retained root or change source")
    if not released:
        return pack_preparation(
            config, root, (), native_calls=0, reasons=("LOCAL_LAW_PREREQUISITE_NONENTRY",)
        )
    scenario = draw_scenario(root, "heldout", seed)
    nominal, completed = acquire_with_progress(
        source, scenario, "exploration-unshifted", ExplorationController(0, 0), 1.0, acquire, 0, progress
    )
    episodes = [nominal]
    if nominal.complete:
        refined, _ = acquire_with_progress(
            source,
            scenario,
            "exploration-unshifted",
            TapeController(nominal.requests),
            0.5,
            acquire,
            completed,
            progress,
        )
        episodes.append(refined)
    reasons = tuple(f"VIEW_{view}_INCOMPLETE" for view, e in enumerate(episodes) if not e.complete)
    return pack_preparation(
        config, root, tuple(episodes), native_calls=len(episodes), reasons=reasons
    )


def donor_episodes(
    preparation: FrontierPreparation, private: FrontierPrivate
) -> tuple[NativeEpisode, ...]:
    if private.root != preparation.root or private.preparation != ObjectIdentity.from_record(
        preparation.record_id, preparation
    ):
        raise ValueError("private preparation changed its exact causal parent")
    arrays = private.arrays.unpack()
    return tuple(
        NativeEpisode(
            preparation.root,
            "exploration-unshifted",
            0.5 if view else 1.0,
            **{field: arrays[f"v{view}_{field}"] for field in NATIVE_FIELDS},
            failure=failure,
        )
        for view, failure in enumerate(private.failures)
    )


def scenario_for(root: str) -> ReactorAssignedScenario:
    role, seed = assignment(root)
    return draw_scenario(root, "calibration" if role == "development" else "heldout", seed)


def native_window(
    donor: NativeEpisode, branch: NativeEpisode, callback: int, tape: np.ndarray
) -> dict[str, np.ndarray]:
    """Retain the complete local native measurement and full-episode digests."""
    if not branch.complete or donor.root != branch.root or donor.dt != branch.dt:
        return {}
    prefix_valid = (
        np.array_equal(branch.observations[: callback + 1], donor.observations[: callback + 1])
        and np.array_equal(branch.requests, tape)
        and np.array_equal(branch.stages[:callback], donor.stages[:callback])
        and np.array_equal(branch.exposure[:callback], donor.exposure[:callback])
    )
    first = int(callback * 10 / branch.dt)
    result = {
        "grid": branch.grid[first : first + int(GUARD / branch.dt) + 1].copy(),
        "valid": np.asarray([prefix_valid], dtype=np.uint8),
    }
    result.update(
        {
            field: getattr(branch, field)[callback : callback + GUARD // 10].copy()
            for field in ("requests", "stages", "exposure")
        }
    )
    for name, array in (
        ("prefix", branch.observations[: callback + 1]),
        ("full_grid", branch.grid),
    ):
        result[f"{name}_sha256"] = np.frombuffer(
            sha256(array.tobytes()).digest(), dtype=np.uint8
        ).copy()
    return result


def acquire_assay(
    source: ReactorBatchSource,
    preparation: FrontierPreparation,
    private: FrontierPrivate,
    seal: ObjectIdentity,
    *,
    acquire: Callable[..., NativeEpisode] = acquire_episode,
    progress: Callable[[int], None] | None = None,
) -> FrontierAssay:
    if (
        source.fingerprint() != preparation.source_sha256
        or seal.object_schema != 'empirical-lawhood/methods/reactor-finite-control-frontier/frontier-prediction-seal'
    ):
        raise ValueError("assay lacks its source/prediction barrier")
    donors = donor_episodes(preparation, private)
    data, failures = {}, []
    calls = completed = 0
    scenario = scenario_for(preparation.root)
    for context in preparation.contexts:
        callback = context.callback
        if callback is None or context.reasons:
            failures.append((context.context, "NO_VALID_CAUSAL_CONTACT"))
            continue
        if len(donors) != 2 or not all(e.complete for e in donors):
            raise ValueError("contacted context lacks its complete paired preparation")
        jacket = float(donors[0].stages[callback - 1, 3])
        for view, donor in enumerate(donors):
            data[f"{context.context}_v{view}_anchor"] = np.asarray(
                (*donor.observations[callback], *donor.stages[callback - 1, 2:])
            )
            data[f"{context.context}_v{view}_prefix_peak"] = np.asarray(
                [donor.grid[: int(callback * 10 / donor.dt) + 1, 1].max()]
            )
        for pulse in WORDS:
            tape = pulse_tape(donors[0].requests, callback, pulse, jacket)
            for view, donor in enumerate(donors):
                key = f"{context.context}_{pulse.word_id}_v{view}"
                branch, completed = acquire_with_progress(
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
                    failures.append((key, branch.failure or "INCOMPLETE_NATIVE_ASSAY"))
                    continue
                data.update(
                    {
                        f"{key}_{field}": array
                        for field, array in native_window(donor, branch, callback, tape).items()
                    }
                )
    return FrontierAssay(
        preparation.root,
        ObjectIdentity.from_record(preparation.record_id, preparation),
        seal,
        LocalArrayPayload.pack(data),
        calls,
        tuple(failures),
    )
