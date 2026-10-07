"""Unrefitted nominal native dynamics conditioned on a fixed public history/tape."""

from dataclasses import replace
from decimal import Decimal as D
from typing import cast
import numpy as np

from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_trace import ReactorBatchTrace
from empirical_lawhood.adapters.simulators.reactor_prefix_response.forecast import ReferenceObserverReactorForecast
from empirical_lawhood.adapters.simulators.reactor_prefix_response.bridge import _decimal
from .design import ReactorBatchSource, PULSE_ONSET, RECOVERY_END
from .trace import ReactorHistoryForecast, ResponseRow, pulse_command, tape_digest


def forecast_history(
    source: ReactorBatchSource, donor: ReactorBatchTrace, dt: D
) -> ReactorHistoryForecast:
    # This native adapter uses the existing pinned nominal equations and actuator
    # projection. No scenario parameters, true state or future pulse outcomes enter.
    model = ReferenceObserverReactorForecast(
        plant_bytes=source.plant_source.encode(),
        params_bytes=source.plant_params.encode(),
        reference_bytes=source.reference_controller.encode(),
    )
    state = donor.states[PULSE_ONSET // 10]
    p = replace(model._params, ua0_w_per_k=float(state.ua_estimate_w_per_k))
    scenario = model._native.Scenario(
        "history-nominal-forecast",
        0,
        1.0,
        1e9,
        0.0,
        (1e9, 1e9),
        0.0,
        1e9,
        float(state.kinetic_multiplier),
    )
    paths = []
    for pulse in (False, True):
        x = np.asarray(
            [
                float(v)
                for v in (
                    state.n_a_mol,
                    state.n_b_mol,
                    state.volume_m3,
                    state.t_estimate_k,
                    state.tj_estimate_k,
                )
            ]
        )
        dose = float(state.measurement.dosed_kg)
        feed, jacket = float(state.previous_feed_kg_s), float(state.previous_jacket_k)

        def row(time: int) -> ResponseRow:
            return cast(
                ResponseRow,
                tuple(_decimal(v) for v in (time, x[3], x[4], x[0], 1 - x[1] / p.n_b0_mol)),
            )

        rows = [row(PULSE_ONSET)]
        for time in range(PULSE_ONSET, RECOVERY_END, 10):
            command = donor.forecasts[time // 10].command
            if pulse:
                command = pulse_command(command)
            _, _, feed, jacket = model._actuator(
                p,
                float(time),
                dose,
                feed,
                jacket,
                float(command.feed_kg_s),
                float(command.jacket_k),
            )
            for sub in range(int(10 / dt)):
                delivered = min(feed, max(p.dose_total_kg - dose, 0.0) / float(dt))
                x = model._native.rk4_step(
                    p, scenario, time + sub * float(dt), x, delivered, jacket, float(dt)
                )
                dose += delivered * float(dt)
            rows.append(row(time + 10))
        paths.append(tuple(rows))
    return ReactorHistoryForecast(state.fingerprint(), tape_digest(donor), dt, paths[0], paths[1])
