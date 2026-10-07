"""Native batches, causal anchor nomination and matched-input local action assays."""

from __future__ import annotations

from empirical_lawhood.adapters.methods.reactor_causal_response.numerical import EXPLORATION_EPISODES, EXPLORATION_ARRAY_STEMS
from hashlib import sha256
from typing import Callable

import numpy as np

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.adapters.methods.reactor_causal_response.numerical import CausalFeatures, Observation
from empirical_lawhood.adapters.methods.reactor_causal_response.development_readout import episode_validity
from empirical_lawhood.adapters.methods.reactor_local_domain_qualification.config import LocalQualificationDesign, ROOTS
from empirical_lawhood.adapters.methods.reactor_local_domain_qualification.records import LocalRootEvidence, LocalArrayPayload
from empirical_lawhood.adapters.simulators.reactor_causal_response.acquisition import NativeEpisode, NativeController, acquire_episode, ExplorationController, TapeController
from empirical_lawhood.adapters.simulators.reactor_causal_response.design import draw_scenario
from empirical_lawhood.adapters.simulators.reactor_causal_response.interface import Actuator, measured_labels
from empirical_lawhood.adapters.simulators.reactor_causal_response.projection import project_episode
from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_design import ReactorBatchSource


def candidate_features(episode: NativeEpisode, callback: int) -> tuple[np.ndarray, np.ndarray]:
    """Causal prefix only; no native concentrations or post-run truth read here."""
    state = CausalFeatures()
    for row in episode.observations[: callback + 1]:
        state.append(Observation(*map(float, row)))
    previous = (
        (0.0, 316.0) if callback == 0 else tuple(map(float, episode.stages[callback - 1, 2:]))
    )
    observation = Observation(*map(float, episode.observations[callback]))
    projections = Actuator().project(observation, (previous[0], previous[1]))
    return np.asarray([state.features(previous[1], p) for p in projections]), np.asarray(
        [p.requested for p in projections]
    )


def assay_anchors(
    design: LocalQualificationDesign,
    donors: tuple[NativeEpisode, ...],
) -> tuple[tuple[str, int | None, int | None, tuple[int, ...]], ...]:
    """Choose earliest causal encounter; same-clock tie uses policy 0, 1, 2."""
    projected: list[tuple[np.ndarray, np.ndarray] | None] = []
    for episode in donors:
        if not episode.complete:
            projected.append(None)
        else:
            projected.append(
                project_episode(
                    tuple(Observation(*map(float, r)) for r in episode.observations),
                    tuple((float(r[0]), float(r[1])) for r in episode.requests),
                )
            )
    result: list[tuple[str, int | None, int | None, tuple[int, ...]]] = []
    for domain in design.atlas.candidates:
        encounters = []
        for policy, projection in enumerate(projected):
            if projection is None:
                continue
            x, actions = projection
            eligible = np.flatnonzero(domain.proposed_support(x, actions))
            # Prefer a real intervention opportunity. This predicate uses only
            # causal observations and known actuator delivery, never responses.
            for callback in eligible:
                # Partition/support coordinates exclude the candidate action.
                # The eligible donor therefore shares its causal context with
                # all measured words; no quadratic prefix reconstruction.
                actions_here = domain.actions
                observation = Observation(*map(float, donors[policy].observations[callback]))
                previous = (
                    (0.0, 316.0)
                    if callback == 0
                    else tuple(map(float, donors[policy].stages[callback - 1, 2:]))
                )
                projections = Actuator().project(observation, (previous[0], previous[1]))
                if len({projections[a].delivery_key for a in actions_here}) < 2:
                    continue
                encounters.append((int(callback), policy, actions_here))
                break
        if encounters:
            chosen_callback, chosen_policy, chosen_actions = min(encounters)
            result.append((domain.domain_id, chosen_policy, chosen_callback, chosen_actions))
        else:
            result.append((domain.domain_id, None, None, ()))
    return tuple(result)


def acquire_local_root(
    design: LocalQualificationDesign,
    source: ReactorBatchSource,
    root: str,
    *,
    acquire: Callable[..., NativeEpisode] = acquire_episode,
    progress: Callable[[int], None] | None = None,
) -> LocalRootEvidence:
    _, role, ordinal, seed = next(row for row in ROOTS if row[0] == root)
    scenario = draw_scenario(root, "calibration" if role == "calibration" else "heldout", seed)
    arrays: dict[str, np.ndarray] = {}
    donors = []
    failures: list[tuple[str, str]] = []
    calls = 0
    completed = 0

    def run(name: str, controller: NativeController, dt: float) -> NativeEpisode:
        nonlocal calls, completed
        calls += 1
        last = 0

        def emit(n: int) -> None:
            nonlocal last
            if not last < n <= 2880:
                raise ValueError("native episode progress is outside its monotone callback census")
            last = n
            if progress is not None:
                progress(completed + n)

        episode = (
            acquire(source, scenario, name, controller, dt=dt)
            if progress is None
            else acquire(source, scenario, name, controller, dt=dt, progress=emit)
        )
        # acquire_episode reports each completed decision, including the final
        # one. Do not duplicate that report or pad an incomplete batch to 2880.
        completed += last
        return episode

    def retain_donor(episode: NativeEpisode, prefix: str) -> None:
        for field in ("observations", "requests", "stages", "exposure", "grid", "callback_cpu"):
            arrays[f"{prefix}_{field}"] = getattr(episode, field)
        if not episode.complete:
            failures.append((prefix, episode.failure or "INCOMPLETE_DONOR"))
            return
        arrays[f"{prefix}_valid"] = episode_validity(episode).astype(np.uint8)
        arrays[f"{prefix}_labels"] = measured_labels(
            episode.grid[:, 0],
            episode.grid[:, 1],
            episode.grid[:, 4],
            episode.grid[:, 3],
            episode.dt,
        )

    for donor_policy in range(3):
        nominal = run(EXPLORATION_EPISODES[donor_policy], ExplorationController(ordinal % 3, donor_policy), 1.0)
        retain_donor(nominal, f"{EXPLORATION_ARRAY_STEMS[donor_policy]}_v0")
        donors.append(nominal)
        if nominal.complete:
            refined = run(EXPLORATION_EPISODES[donor_policy], TapeController(nominal.requests), 0.5)
            retain_donor(refined, f"{EXPLORATION_ARRAY_STEMS[donor_policy]}_v1")
        else:
            failures.append((f"{EXPLORATION_ARRAY_STEMS[donor_policy]}_v1", "NOMINAL_DONOR_INCOMPLETE"))
    assays = assay_anchors(design, tuple(donors))
    for domain, policy, callback, actions in assays:
        if policy is None or callback is None:
            continue
        donor = donors[policy]
        features, requests = candidate_features(donor, callback)
        model = next(d for d in design.atlas.candidates if d.domain_id == domain)
        if not model.proposed_support(features[list(actions)], np.asarray(actions)).all():
            raise ValueError("anchor candidate context differs from its eligible donor")
        arrays[f"{domain}_features"] = features
        for action in actions:
            tape = donor.requests.copy()
            tape[callback] = requests[action]
            for view, dt in enumerate((1.0, 0.5)):
                key = f"{domain}_a{action}_v{view}"
                episode = run(f"{domain}-a{action}", TapeController(tape), dt)
                if not episode.complete:
                    failures.append((key, episode.failure or "INCOMPLETE_ASSAY"))
                    continue
                # Persist the raw physical measurement window, actual exposure,
                # commands and prefix identity; no need to retain its unused
                # post-assay response as a predictive input or fitted label.
                step = int(10 / dt)
                arrays[f"{key}_grid"] = episode.grid[
                    callback * step : (callback + 1) * step + 1
                ].copy()
                arrays[f"{key}_exposure"] = episode.exposure[callback].copy()
                arrays[f"{key}_stages"] = episode.stages[callback].copy()
                arrays[f"{key}_request"] = episode.requests[callback].copy()
                arrays[f"{key}_valid"] = episode_validity(episode)[callback : callback + 1].astype(
                    np.uint8
                )
                arrays[f"{key}_grid_sha256"] = np.frombuffer(
                    sha256(episode.grid.tobytes()).digest(), dtype=np.uint8
                ).copy()
                arrays[f"{key}_prefix_sha256"] = np.frombuffer(
                    sha256(episode.observations[: callback + 1].tobytes()).digest(), dtype=np.uint8
                ).copy()
                prefix = arrays[f"{EXPLORATION_ARRAY_STEMS[policy]}_v{view}_observations"][: callback + 1]
                if not np.array_equal(prefix, episode.observations[: callback + 1]):
                    failures.append((key, "DONOR_PREFIX_CHANGED"))
                if not np.array_equal(episode.requests, tape):
                    failures.append((key, "DONOR_CONTINUATION_CHANGED"))
    return LocalRootEvidence(
        root,
        role,
        seed,
        ObjectIdentity.from_record(design.config_id, design),
        LocalArrayPayload.pack(arrays),
        assays,
        tuple(failures),
        calls,
    )
