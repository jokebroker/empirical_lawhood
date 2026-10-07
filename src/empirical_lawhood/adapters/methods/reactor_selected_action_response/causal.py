"Only permitted, completed native observations can qualify a selected-action handoff."

from decimal import Decimal as D
import numpy as np

from empirical_lawhood.adapters.methods.reactor_causal_response.numerical import CausalFeatures, Observation
from empirical_lawhood.adapters.methods.reactor_regime_response.causal_preparation import causal_delivery_prefix
from empirical_lawhood.adapters.methods.reactor_regime_response.features import causal_features
from empirical_lawhood.adapters.simulators.reactor_causal_response.interface import Actuator
from empirical_lawhood.adapters.simulators.reactor_selected_action_response.config import ClassicalNativeConfig
from empirical_lawhood.kernel.provenance import ObjectIdentity
from .records import ClassicalCausal, ClassicalDecision


def inputs(
    causal: ClassicalCausal,
) -> tuple[np.ndarray, np.ndarray, Observation, tuple[float, float]]:
    callback = causal.callback
    if callback is None:
        raise ValueError("no selected classical causal callback")
    arrays = causal.arrays.unpack()
    observations, stages = arrays["exploration_unshifted_v0_observations"], arrays["exploration_unshifted_v0_stages"]
    state = CausalFeatures()
    for row in observations:
        state.append(Observation(*map(float, row)))
    previous = (float(stages[-1, 2]), float(stages[-1, 3]))
    observation = Observation(*map(float, observations[-1]))
    projection = Actuator().project_request(observation, previous, (0.016, previous[1]), 4)
    return (
        causal_features(observations, stages, callback),
        np.asarray(state.features(previous[1], projection)),
        observation,
        previous,
    )


def seal_decision(
    config: ClassicalNativeConfig,
    causal: ClassicalCausal,
    authority: ObjectIdentity,
    qualified_law: ObjectIdentity | None,
) -> ClassicalDecision:
    reasons = set(causal.reasons)
    observed = None
    if causal.callback is not None:
        arrays = causal.arrays.unpack()
        for view in (0, 1):
            values = tuple(
                arrays[f"exploration_unshifted_v{view}_{key}"]
                for key in ("observations", "requests", "stages", "exposure")
            )
            if not causal_delivery_prefix(
                values[0], values[1], values[2], values[3], causal.callback, 0.5 if view else 1.0
            ):
                reasons.add("INVALID_CAUSAL_DELIVERY_PREFIX")
        _, domain_features, observation, previous = inputs(causal)
        observed = D(repr(observation.temperature))
        if not config.prepared_domain.proposed_support(
            domain_features[None, :], np.asarray((4,))
        ).all():
            reasons.add("OUTSIDE_INHERITED_CAUSAL_DOMAIN")
        if not 600 <= observation.time <= 6000 or previous[0] > 0.016:
            reasons.add("WRONG_PREPARATION_TIME_OR_PREVIOUS_FEED")
        if (
            observed + config.design.causal_temperature_halfwidth_K
            > config.design.temperature_limit_K
        ):
            reasons.add("CAUSAL_TEMPERATURE_UPPER_ABOVE_LIMIT")
        for feed in (0.0, 0.016):
            projection = Actuator().project_request(
                observation, previous, (feed, previous[1]), 1 if not feed else 4
            )
            mass = sum(dt * realized for _, dt, realized, _ in projection.exposure)
            if (
                abs(mass - (0.0 if not feed else 0.16)) > 1e-12
                or projection.applied[1] != previous[1]
                or any(j != previous[1] for _, _, _, j in projection.exposure)
            ):
                reasons.add("INVALID_FIXED_NATIVE_WORD")
    if causal.role == "prospective" and qualified_law is None:
        reasons.add("LOCAL_LAW_PREREQUISITE_NONENTRY")
    return ClassicalDecision(
        causal.root,
        ObjectIdentity.from_record(f"{causal.root}.causal-preparation", causal),
        causal.recipe,
        qualified_law,
        causal.callback,
        not reasons,
        observed,
        authority,
        tuple(sorted(reasons)),
    )
