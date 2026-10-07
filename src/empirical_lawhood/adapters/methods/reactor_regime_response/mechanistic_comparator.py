"""Declared nominal mechanistic shadow on the same permitted reactor prefix."""

from __future__ import annotations

from decimal import Decimal as D
from math import isfinite

import numpy as np

from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_design import ReactorBatchSource
from empirical_lawhood.adapters.simulators.reactor_prefix_response.contracts import ReactorCommand, ReactorMeasurement
from empirical_lawhood.adapters.simulators.reactor_prefix_response.forecast import ReferenceObserverReactorForecast

from .records import RegimeCausalPreparation
from .measured_panel import CONTEXTS


def mechanistic_chart(
    source: ReactorBatchSource,
    causal: RegimeCausalPreparation,
    route: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """Replay observations/requests only; hidden scenario parameters never enter."""
    if route not in CONTEXTS:
        raise ValueError("MECH comparator changed the nominated preparation")
    anchors = {name: callback for name, callback, _ in causal.anchors}
    t0 = anchors["prepared_t0"]
    callback = anchors[route] if route in anchors else None if t0 is None else t0 + (33 if route.endswith("_q") else 93)
    if callback is None:
        return None
    stem = "exploration_unshifted_v0" if route in anchors else f"{route[0]}_v0"
    arrays = causal.arrays.unpack()
    if any(f"{stem}_{name}" not in arrays for name in ("observations", "requests", "stages")):
        return None
    observations = arrays[f"{stem}_observations"]
    requests = arrays[f"{stem}_requests"]
    stages = arrays[f"{stem}_stages"]
    if (
        observations.shape[1:] != (4,) or len(observations) <= callback
        or requests.shape[1:] != (2,) or len(requests) < callback
        or stages.shape[1:] != (4,) or len(stages) < callback
        or not np.isfinite(observations[: callback + 1]).all()
        or not np.isfinite(requests[:callback]).all()
        or not np.isfinite(stages[:callback]).all()
    ):
        return None
    model = ReferenceObserverReactorForecast(
        plant_bytes=source.plant_source.encode(),
        params_bytes=source.plant_params.encode(),
        reference_bytes=source.reference_controller.encode(),
    )
    state = None
    for index in range(callback + 1):
        row = observations[index]
        measurement = ReactorMeasurement(
            D(repr(float(row[0]))), D(10), D(repr(float(row[1]))),
            D(repr(float(row[2]))), D(repr(float(row[3]))),
        )
        state = model.observe(measurement)
        if index < callback:
            command = ReactorCommand(
                f"{causal.root}.mech-prefix.{index:04d}",
                D(index * 10),
                D(repr(float(requests[index, 0]))),
                D(repr(float(requests[index, 1]))),
            )
            model.commit(state, command)
            if (
                float(model._feed) != float(stages[index, 2])
                or float(model._jacket) != float(stages[index, 3])
            ):
                return None
    if state is None:
        return None
    jacket = float(stages[callback - 1, 3])
    peaks = []
    for word, feed in enumerate((0.0, 0.016, 0.032)):
        command = ReactorCommand(
            f"{causal.root}.mech-word.{word}",
            D(callback * 10), D(repr(feed)), D(repr(jacket)),
        )
        try:
            forecast = model.predict(state, command, plant_dt_s=D(1))
        except (OverflowError, FloatingPointError):
            return None
        peak = float(forecast.peak_temperature_k)
        if not isfinite(peak):
            return None
        peaks.append(peak)
    return (tuple(peaks), (0.0, peaks[0] - peaks[1], peaks[0] - peaks[2]))  # type: ignore[return-value]
