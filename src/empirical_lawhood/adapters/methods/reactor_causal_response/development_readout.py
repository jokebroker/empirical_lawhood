"""Assigned branch and nomination opportunity readouts from retained measurements."""

from dataclasses import asdict
from hashlib import sha256
from typing import Any
import numpy as np
from empirical_lawhood.adapters.simulators.reactor_causal_response.acquisition import NativeEpisode
from empirical_lawhood.adapters.simulators.reactor_causal_response.interface import Actuator, measured_labels, project_tape_stages
from .numerical import CausalFeatures, Observation, FrozenFit, select
from .opportunity import branch_readout


def episode_validity(episode: NativeEpisode) -> np.ndarray:
    """No hidden outcome constructs features; truth validates only post-run evidence."""
    invalid = np.zeros(2880, dtype=bool)
    if (
        not episode.complete
        or episode.observations.shape != (2880, 4)
        or episode.stages.shape != (2880, 4)
        or episode.callback_cpu.shape != (2880,)
        or any(
            not np.isfinite(a).all()
            for a in (
                episode.observations,
                episode.requests,
                episode.stages,
                episode.exposure,
                episode.grid,
                episode.callback_cpu,
            )
        )
        or (episode.callback_cpu < 0).any()
        or not np.array_equal(episode.observations[:, 0], np.arange(2880) * 10)
        or not np.array_equal(
            episode.grid[:, 0], np.arange(int(28800 / episode.dt) + 1) * episode.dt
        )
    ):
        return invalid
    stages, exposure = project_tape_stages(episode.requests, episode.dt)
    valid = np.all(stages == episode.stages, axis=1) & np.all(
        exposure == episode.exposure, axis=(1, 2)
    )
    expected_dose = np.empty(2881)
    expected_dose[0] = 0.0
    dose = 0.0
    # Preserve native accumulation order, including the near-cap final substep.
    for k in range(2880):
        for _, duration, feed, _ in exposure[k]:
            dose += duration * feed
        expected_dose[k + 1] = dose
    valid &= episode.observations[:, 3] == expected_dose[:-1]
    valid &= episode.grid[:: int(10 / episode.dt), 3][1:] == expected_dose[1:]
    return np.asarray(valid, dtype=bool)


def branch_summary(
    pairs: dict[tuple[str, str], tuple[NativeEpisode, NativeEpisode]],
) -> dict[str, Any]:
    differences = np.zeros((24, 2, 3, 2))
    contact = np.zeros((24, 2), dtype=bool)
    valid = np.zeros((24, 2), dtype=bool)
    details = []
    for i in range(24):
        root = (
            f"reactor-empirical-fit-{i:03d}"
            if i < 16
            else f"reactor-empirical-nomination-{i - 16:03d}"
        )
        anchor = (3600, 9000, 18000, 24000)[i % 4]
        donor = pairs[(root, "exploration-shifted-one-position")]
        for j, name in enumerate(("feed-intervention", "jacket-intervention")):
            pair = pairs[(root, name)]
            good = all(episode_validity(e).all() for e in (*donor, *pair))
            numerical = False
            if good:
                numerical = all(
                    np.all(
                        np.abs(
                            measured_labels(
                                p[0].grid[:, 0],
                                p[0].grid[:, 1],
                                p[0].grid[:, 4],
                                p[0].grid[:, 3],
                                1.0,
                            )
                            - measured_labels(
                                p[1].grid[:, 0],
                                p[1].grid[:, 1],
                                p[1].grid[:, 4],
                                p[1].grid[:, 3],
                                0.5,
                            )
                        )
                        <= (0.01, 0.0002, 0.000001)
                    )
                    for p in (donor, pair)
                )
                differences[i, j] = [
                    [pair[0].grid[anchor + h, c] - donor[0].grid[anchor + h, c] for c in (1, 4)]
                    for h in (10, 60, 300)
                ]
            valid[i, j] = good and numerical
            contact[i, j] = (
                len(pair[0].exposure) > anchor // 10
                and len(donor[0].exposure) > anchor // 10
                and not np.array_equal(
                    pair[0].exposure[anchor // 10], donor[0].exposure[anchor // 10]
                )
            )
            details.append(
                dict(
                    root=root,
                    branch=name,
                    anchor_s=anchor,
                    contact=bool(contact[i, j]),
                    valid=bool(valid[i, j]),
                    differences=differences[i, j].tolist() if good else None,
                    nominal_exposure_sha256=sha256(pair[0].exposure.tobytes()).hexdigest(),
                    donor_exposure_sha256=sha256(donor[0].exposure.tobytes()).hexdigest(),
                )
            )
    return dict(
        summary=asdict(branch_readout(differences, contact, valid)),
        pairs=details,
        axes=("root", "branch-F-J", "elapsed-10-60-300", "endpoint-T-C"),
    )


def nomination_shadow(
    pairs: dict[tuple[str, str], tuple[NativeEpisode, NativeEpisode]],
    models: tuple[FrozenFit, FrozenFit, FrozenFit],
    q: float,
) -> dict[str, Any]:
    actuator = Actuator()
    rows: list[dict[str, Any]] = []
    for i in range(8):
        root = f"reactor-empirical-nomination-{i:03d}"
        for name in ("exploration-unshifted", "exploration-shifted-one-position", "exploration-shifted-two-positions", "feed-intervention", "jacket-intervention"):
            episode = pairs[(root, name)][0]
            state = CausalFeatures()
            previous = (0.0, 316.0)
            counts = np.zeros((3, 512), dtype=int)
            refused = [0] * 3
            alternative = 0
            differences = [0, 0]
            for k, raw in enumerate(episode.observations):
                o = Observation(*map(float, raw))
                state.append(o)
                projections = actuator.project(o, previous)
                choices = tuple(
                    select((o,), previous[1], projections, m, q, causal_state=state) for m in models
                )
                for m, d in enumerate(choices):
                    mask = sum(1 << c.projection.action for c in d.candidates if not c.reasons)
                    counts[m, mask] += 1
                    refused[m] += d.selected is None
                alternative += (
                    len({c.projection.delivery_key for c in choices[0].candidates if not c.reasons})
                    >= 2
                )
                for r in range(2):
                    differences[r] += choices[0].selected != choices[r + 1].selected
                previous = actuator.project_request(
                    o, previous, (float(episode.requests[k, 0]), float(episode.requests[k, 1])), -1
                ).applied
            rows.append(
                dict(
                    root=root,
                    episode=name,
                    queries=len(episode.observations),
                    refused=refused,
                    distinct_alternative_queries=alternative,
                    rival_choice_differences=differences,
                    admitted_mask_counts=counts.tolist(),
                )
            )
    queries = sum(r["queries"] for r in rows)
    return dict(
        status="UNUSABLE_NO_ADMITTED_QUERY"
        if sum(r["refused"][0] for r in rows) == queries
        else "SHADOW_OBSERVED",
        q_dev=q,
        queries=queries,
        distinct_alternative_fraction=sum(r["distinct_alternative_queries"] for r in rows)
        / queries,
        episodes=rows,
    )
