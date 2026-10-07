"""Published observer over permitted prefixes, for explicit finite request tapes."""

from decimal import Decimal as D
from typing import TYPE_CHECKING

import numpy as np

from .batch_design import ReactorBatchSource
from .contracts import ReactorCommand, ReactorMeasurement
from .forecast import ReferenceObserverReactorForecast

if TYPE_CHECKING:
    from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.causal import CausalFeedContext


def causal_feed_forecast(
    source: ReactorBatchSource,
    root: str,
    context: "CausalFeedContext",
    words: tuple[tuple[str, tuple[D, ...]], ...],
) -> tuple[dict[str, np.ndarray], tuple[str, ...]]:
    if context.callback is None or context.reasons:
        return {}, ("NO_CAUSAL_CONTACT",)
    if (
        not words
        or len({key for key, _ in words}) != len(words)
        or any(not 1 <= len(rates) <= 24 for _, rates in words)
    ):
        raise ValueError("mechanistic comparator requires an explicit finite tape census")
    model = ReferenceObserverReactorForecast(
        plant_bytes=source.plant_source.encode(),
        params_bytes=source.plant_params.encode(),
        reference_bytes=source.reference_controller.encode(),
    )
    data = context.arrays.unpack()
    state = None
    try:
        for k, row in enumerate(data["observations"]):
            measurement = ReactorMeasurement(
                D(k * 10), D(10), *(D(repr(float(v))) for v in row[1:])
            )
            state = model.observe(measurement)
            if k < context.callback:
                command = ReactorCommand(
                    f"{root}.{context.context}.mech-prefix-{k}",
                    D(k * 10),
                    *(D(repr(float(v))) for v in data["requests"][k]),
                )
                model.commit(state, command)
                if (float(model._feed), float(model._jacket)) != tuple(data["stages"][k, 2:]):
                    return {}, ("MECHANISTIC_PREFIX_PROJECTION_MISMATCH",)
        if state is None:
            return {}, ("NO_OBSERVER_STATE",)
        jacket = D(repr(float(data["stages"][-1, 3])))
        grids = {}
        for name, rates in words:
            commands = tuple(
                ReactorCommand(
                    f"{root}.{context.context}.{name}.mech-{i}",
                    D(context.callback * 10 + i * 10),
                    rate,
                    jacket,
                )
                for i, rate in enumerate(rates)
            )
            grids[name] = model.predict_requested_tape(state, commands, plant_dt_s=D(1))
        if any(not np.isfinite(grid).all() for grid in grids.values()):
            return {}, ("NONFINITE_MECHANISTIC_FORECAST",)
        return grids, ()
    except (OverflowError, FloatingPointError):
        return {}, ("NONFINITE_MECHANISTIC_FORECAST",)
