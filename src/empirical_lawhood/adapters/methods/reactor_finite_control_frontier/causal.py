"""Permitted prefix operands shared by prediction and control; no labels."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from empirical_lawhood.adapters.methods.reactor_causal_response.numerical import Observation, CausalFeatures
from empirical_lawhood.adapters.methods.reactor_prepared_feed_qualification.records import FeedDomain
from empirical_lawhood.adapters.methods.reactor_regime_response.causal_preparation import causal_delivery_prefix
from empirical_lawhood.adapters.simulators.reactor_causal_response.interface import Actuator
from empirical_lawhood.adapters.methods.reactor_local_domain_qualification.records import LocalArrayPayload


class CausalFeedContext(Protocol):
    """Only the permitted prefix fields consumed by this shared calculation."""

    @property
    def callback(self) -> int | None: ...

    @property
    def context(self) -> str: ...

    @property
    def arrays(self) -> LocalArrayPayload: ...

    @property
    def reasons(self) -> tuple[str, ...]: ...


@dataclass(frozen=True)
class CausalOperands:
    observation: Observation
    previous: tuple[float, float]
    domain_features: tuple[float, ...]


def causal_operands(context: CausalFeedContext, domain: FeedDomain) -> CausalOperands | None:
    k = context.callback
    if k is None or context.reasons:
        return None
    data = context.arrays.unpack()
    if not causal_delivery_prefix(
        data["observations"], data["requests"], data["stages"], data["exposure"], k, 1.0
    ):
        return None
    observation = Observation(*map(float, data["observations"][k]))
    previous = (float(data["stages"][k - 1, 2]), float(data["stages"][k - 1, 3]))
    if not 0 <= previous[0] <= 0.016 or not 600 <= observation.time <= 25080:
        return None
    features = CausalFeatures()
    for row in data["observations"]:
        features.append(Observation(*map(float, row)))
    projections = Actuator().project(observation, previous)
    chart = tuple(projections[i] for i in (1, 4, 7))
    if (
        chart[0].mean_feed != 0
        or len({p.delivery_key for p in chart}) != 3
        or any(p.applied[1] != previous[1] for p in chart)
    ):
        return None
    x = np.asarray([features.features(previous[1], p) for p in chart])
    if (
        context.context == "prepared_t0"
        and not domain.proposed_support(x, np.asarray((1, 4, 7))).all()
    ):
        return None
    return CausalOperands(observation, previous, tuple(map(float, x[1])))
