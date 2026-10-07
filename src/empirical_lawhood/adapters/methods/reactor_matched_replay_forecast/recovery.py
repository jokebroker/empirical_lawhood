"""Paired, fixed-clock response losses; these are operands, not a law finalizer."""

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar, cast
import numpy as np
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.adapters.simulators.reactor_matched_replay_forecast.design import PULSE_ONSET, PULSE_RETURN, RECOVERY_END
from empirical_lawhood.adapters.simulators.reactor_matched_replay_forecast.trace import ReactorBatchTrace, ResponseRow

Quad = tuple[D, D, D, D]


@dataclass(frozen=True, slots=True)
class RecoveryViewScore(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-matched-replay-forecast/recovery-view-score'
    plant_dt_s: D
    common_applied_input_from_s: D | None
    common_input_window_valid: bool
    prefix_max_discrepancy: Quad
    maximum_applied_pulse_k: D
    observed_baseline: tuple[ResponseRow, ...]
    observed_pulse: tuple[ResponseRow, ...]
    predicted_baseline: tuple[ResponseRow, ...]
    predicted_pulse: tuple[ResponseRow, ...]
    increment_mse: Quad
    zero_increment_mse: Quad
    held_increment_mse: Quad
    baseline_mse: Quad
    absolute_pulse_mse: Quad
    baseline_increment_cross: Quad

    def __post_init__(self) -> None:
        if (
            self.plant_dt_s not in (D(1), D(".5"))
            or type(self.common_input_window_valid) is not bool
        ):
            raise ValueError("recovery view or common-input validity differs")
        for rows in (
            self.observed_baseline,
            self.observed_pulse,
            self.predicted_baseline,
            self.predicted_pulse,
        ):
            if len(rows) != 181 or any(
                len(r) != 5 or r[0] != PULSE_ONSET + i * 10 for i, r in enumerate(rows)
            ):
                raise ValueError("recovery changes fixed receiver sample clocks")
        for values in (
            self.prefix_max_discrepancy,
            self.increment_mse,
            self.zero_increment_mse,
            self.held_increment_mse,
            self.baseline_mse,
            self.absolute_pulse_mse,
        ):
            if len(values) != 4 or any(not x.is_finite() or x < 0 for x in values):
                raise ValueError(
                    "recovery losses must be finite nonnegative four-receiver operands"
                )
        if len(self.baseline_increment_cross) != 4 or any(
            not x.is_finite() for x in self.baseline_increment_cross
        ):
            raise ValueError("recovery cross terms must be finite signed operands")


def recovery_scores(trace: ReactorBatchTrace) -> tuple[RecoveryViewScore, ...]:
    from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_trace import ReactorBatchTrace
    from empirical_lawhood.adapters.simulators.reactor_matched_replay_forecast.trace import ReactorReplay

    if (
        trace.failure_code
        or trace.matched_refined is None
        or trace.pulse_native is None
        or trace.pulse_refined is None
    ):
        raise ValueError("incomplete assigned unit has no recovery comparison")
    baselines: tuple[ReactorBatchTrace | ReactorReplay, ...] = (
        trace.reference[0],
        trace.matched_refined,
    )
    result = []
    for baseline, pulse, prediction in zip(
        baselines,
        (trace.pulse_native, trace.pulse_refined),
        trace.history_forecasts,
        strict=True,
    ):
        dt = baseline.plant_dt_s
        a, b = (
            np.asarray(baseline.native_grid, dtype=float),
            np.asarray(pulse.native_grid, dtype=float),
        )
        # Response rows explicitly reorder native coordinates: T,Tj,nA,conversion.
        columns = [0, 1, 2, 5, 4]
        start, end, stride = int(PULSE_ONSET / dt), int(RECOVERY_END / dt), int(10 / dt)
        obs_a, obs_b = a[start : end + 1 : stride, columns], b[start : end + 1 : stride, columns]
        pred_a, pred_b = (
            np.asarray(prediction.baseline, dtype=float),
            np.asarray(prediction.pulse, dtype=float),
        )
        observed = obs_b[:, 1:] - obs_a[:, 1:]
        predicted = pred_b[:, 1:] - pred_a[:, 1:]
        i_return = (PULSE_RETURN - PULSE_ONSET) // 10
        selection = slice(i_return + 1, None)
        error = (predicted - observed)[selection]
        base_error = (pred_a[:, 1:] - obs_a[:, 1:])[selection]
        pulse_error = (pred_b[:, 1:] - obs_b[:, 1:])[selection]
        # Applied callback stages are distinct from truth samples at a boundary:
        # the stage at t acts over [t,t+10); the grid row at t ends the prior step.
        stages_a = np.asarray(baseline.observed_stages, dtype=float)
        stages_b = np.asarray(pulse.observed_stages, dtype=float)
        equal = np.all(stages_a[:, 2:] == stages_b[:, 2:], axis=1)
        window_end = RECOVERY_END // 10
        bad = np.flatnonzero(~equal[PULSE_RETURN // 10 : window_end])
        common = PULSE_RETURN if not len(bad) else PULSE_RETURN + int(bad[-1] + 1) * 10
        valid = common <= PULSE_RETURN + 10
        # Independent realized-input check over the scored common-input window.
        valid = valid and bool(
            np.array_equal(a[int(7810 / dt) + 1 : end + 1, 6:], b[int(7810 / dt) + 1 : end + 1, 6:])
        )

        def quad(values: np.ndarray) -> Quad:
            return cast(Quad, tuple(D(str(float(v))) for v in values))

        def rows(values: np.ndarray) -> tuple[ResponseRow, ...]:
            return tuple(cast(ResponseRow, tuple(D(str(float(v))) for v in row)) for row in values)

        result.append(
            RecoveryViewScore(
                dt,
                D(common) if common < RECOVERY_END else None,
                valid,
                quad(
                    np.max(
                        np.abs(b[: start + 1, [1, 2, 5, 4]] - a[: start + 1, [1, 2, 5, 4]]), axis=0
                    )
                ),
                D(
                    str(
                        float(
                            np.max(
                                np.abs(
                                    stages_b[PULSE_ONSET // 10 : PULSE_RETURN // 10, 3]
                                    - stages_a[PULSE_ONSET // 10 : PULSE_RETURN // 10, 3]
                                )
                            )
                        )
                    )
                ),
                rows(obs_a),
                rows(obs_b),
                prediction.baseline,
                prediction.pulse,
                quad(np.mean(error**2, axis=0)),
                quad(np.mean(observed[selection] ** 2, axis=0)),
                quad(np.mean((predicted[i_return] - observed[selection]) ** 2, axis=0)),
                quad(np.mean(base_error**2, axis=0)),
                quad(np.mean(pulse_error**2, axis=0)),
                quad(2 * np.mean(base_error * error, axis=0)),
            )
        )
    return tuple(result)
