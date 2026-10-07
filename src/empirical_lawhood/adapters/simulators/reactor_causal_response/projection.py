"Causal development trace binding with exact exposure aliases."

from dataclasses import dataclass, replace
import numpy as np
from empirical_lawhood.adapters.methods.reactor_causal_response.numerical import Observation, CausalFeatures, menu
from empirical_lawhood.adapters.methods.reactor_causal_response.discovery import Rows, validate_development_census
from .interface import Actuator


class UnmappedBranchWord(ValueError):
    "Donor continuation has no nominal action ID in the current menu; amendment needed."


def nominal_action_id(previous_jacket: float, requested: tuple[float, float]) -> int:
    matches = [i for i, word in enumerate(menu(previous_jacket)) if word == requested]
    if not matches:
        raise UnmappedBranchWord(
            "Unmapped donor action: unchanged donor request outside current menu"
        )
    return min(matches)


@dataclass(frozen=True)
class DevelopmentEpisode:
    root: str
    role: str
    episode: str
    observations: tuple[Observation, ...]
    requests: tuple[tuple[float, float], ...]
    # Post-run labels are handed to the fitter only after causal projection.
    labels: np.ndarray
    delivery_valid: np.ndarray


def project_episode(
    observations: tuple[Observation, ...], requests: tuple[tuple[float, float], ...]
) -> tuple[np.ndarray, np.ndarray]:
    if len(observations) != 2880 or len(requests) != 2880:
        raise ValueError("complete development callback/request census required")
    actuator = Actuator()
    previous = (0.0, 316.0)
    state = CausalFeatures()
    x, ids = [], []
    for k, (observation, request) in enumerate(zip(observations, requests, strict=True)):
        projection = actuator.project_request(observation, previous, request, -1)
        aliases = [
            p.action
            for p in actuator.project(observation, previous)
            if p.exposure == projection.exposure
        ]
        if not aliases:
            raise UnmappedBranchWord(f'Exact exposure alias absent at callback {k}')
        direct = [i for i, word in enumerate(menu(previous[1])) if word == request]
        action = min(direct) if direct else min(aliases)
        projection = replace(projection, action=action)
        state.append(observation)
        x.append(state.features(previous[1], projection))
        ids.append(action)
        previous = projection.applied
    return np.asarray(x), np.asarray(ids)


def development_rows(episodes: tuple[DevelopmentEpisode, ...]) -> Rows:
    if len(episodes) != 120:
        raise ValueError("all 24 roots by five episodes required")
    roots: list[str] = []
    roles: list[str] = []
    names: list[str] = []
    clocks: list[float] = []
    actions, x, y, valid = [], [], [], []
    for e in episodes:
        causal, ids = project_episode(e.observations, e.requests)
        roots.extend((e.root,) * 2880)
        roles.extend((e.role,) * 2880)
        names.extend((e.episode,) * 2880)
        clocks.extend(o.time for o in e.observations)
        actions.append(ids)
        x.append(causal)
        y.append(e.labels)
        valid.append(e.delivery_valid)
    result = Rows(
        tuple(roots),
        tuple(roles),
        np.array(clocks),
        np.concatenate(actions),
        np.concatenate(x),
        np.concatenate(y),
        np.concatenate(valid),
        tuple(names),
    )
    validate_development_census(result)
    return result
