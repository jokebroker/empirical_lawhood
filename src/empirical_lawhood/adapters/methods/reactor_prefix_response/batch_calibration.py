"""Whole-episode forecast operands for the existing qualification owners.

These records contain measurements, bounds and heldout coverage, not terminal
law/admission verdicts. The complete root is the calibration unit; every time,
receiver and numerical view belongs inside that unit's score.
"""

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar, cast

from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_design import assigned_scenarios
from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_trace import ReactorBatchTrace
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord

Triple = tuple[D, D, D]
RECEIVERS = ("interval-peak-temperature", "endpoint-dose", "endpoint-conversion")
UNITS = ("K", "kg", "1")
NUMERICAL_TOLERANCE: Triple = (D(".035"), D(".0359125"), D(".0025"))


@dataclass(frozen=True, slots=True)
class ReactorUnitForecastScore(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-prefix-response/reactor-unit-forecast-score'
    unit_id: str
    split: str
    traces: tuple[ObjectIdentity, ObjectIdentity]
    maximum_absolute_errors: Triple | None
    rmse: Triple | None
    persistence_maximum_absolute_errors: Triple | None
    persistence_rmse: Triple | None
    maximum_numerical_discrepancy: Triple | None
    callback_count: int
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        scenario = next((s for s in assigned_scenarios() if s.unit_id == self.unit_id), None)
        if scenario is None or scenario.split != self.split or len(self.traces) != 2:
            raise ValueError("forecast score changes its whole-unit assignment")
        values = (
            self.maximum_absolute_errors,
            self.rmse,
            self.persistence_maximum_absolute_errors,
            self.persistence_rmse,
            self.maximum_numerical_discrepancy,
        )
        if any(v is None for v in values) != bool(self.reason_codes):
            raise ValueError("unavailable forecast scores require explicit reasons")
        if any(
            v is not None and (len(v) != 3 or any(not x.is_finite() or x < 0 for x in v))
            for v in values
        ):
            raise ValueError("forecast score has invalid native-unit values")
        if not 0 <= self.callback_count <= 5760 or (
            not self.reason_codes and self.callback_count != 5760
        ):
            raise ValueError("forecast score changes its nested callback census")


def score_unit(
    primary: ReactorBatchTrace, refined: ReactorBatchTrace
) -> ReactorUnitForecastScore:
    if (
        primary.scenario != refined.scenario
        or primary.plant_dt_s != 1
        or refined.plant_dt_s != D(".5")
        or primary.source_sha256 != refined.source_sha256
    ):
        raise ValueError("forecast score requires exactly one coupled native/refined unit")
    traces = (primary, refined)
    identities = tuple(
        ObjectIdentity.from_record(
            f"{t.scenario.unit_id}.{'native' if t.plant_dt_s == 1 else 'refined'}", t
        )
        for t in traces
    )
    count = sum(len(t.states) for t in traces)
    if any(t.failure_code is not None for t in traces):
        return ReactorUnitForecastScore(
            primary.scenario.unit_id,
            primary.scenario.split,
            cast(tuple[ObjectIdentity, ObjectIdentity], identities),
            None,
            None,
            None,
            None,
            None,
            count,
            ("ASSIGNED_NATIVE_OR_FORECAST_FAILURE",),
        )
    errors: list[list[D]] = [[], [], []]
    baseline_errors: list[list[D]] = [[], [], []]
    for trace in traces:
        sub = int(D(10) / trace.plant_dt_s)
        for index, (state, forecast) in enumerate(zip(trace.states, trace.forecasts, strict=True)):
            interval = trace.native_grid[index * sub : (index + 1) * sub + 1]
            observed = (max(row[1] for row in interval), interval[-1][3], interval[-1][4])
            point = (
                forecast.peak_temperature_k,
                forecast.endpoint_dosed_kg,
                forecast.endpoint_conversion_b,
            )
            # Same causal observer information, held constant over the next interval.
            persistence = (
                state.t_estimate_k,
                state.measurement.dosed_kg,
                D(1) - state.n_b_mol / D(3170),
            )
            for column in range(3):
                errors[column].append(abs(observed[column] - point[column]))
                baseline_errors[column].append(abs(observed[column] - persistence[column]))
    discrepancy = cast(
        Triple,
        tuple(
            max(
                abs(a[column] - b[column])
                for a, b in zip(primary.native_grid, refined.native_grid[::2], strict=True)
            )
            for column in (1, 3, 4)
        ),
    )
    reasons = (
        ("NUMERICAL_VIEW_DISCREPANCY_EXCEEDS_DECLARED_ALLOWANCE",)
        if any(d > tolerance for d, tolerance in zip(discrepancy, NUMERICAL_TOLERANCE, strict=True))
        else ()
    )

    def maxima(values: list[list[D]]) -> Triple:
        return cast(Triple, tuple(max(v) for v in values))

    def rmse(values: list[list[D]]) -> Triple:
        return cast(
            Triple, tuple((sum((x * x for x in v), D(0)) / D(len(v))).sqrt() for v in values)
        )

    # An invalid numerical unit remains an infinite calibration score. Preserve
    # its numerical discrepancy; do not silently exclude it from the denominator.
    return ReactorUnitForecastScore(
        primary.scenario.unit_id,
        primary.scenario.split,
        cast(tuple[ObjectIdentity, ObjectIdentity], identities),
        None if reasons else maxima(errors),
        rmse(errors),
        maxima(baseline_errors),
        rmse(baseline_errors),
        discrepancy,
        count,
        reasons,
    )


@dataclass(frozen=True, slots=True)
class ReactorForecastCalibration(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-prefix-response/reactor-forecast-calibration'
    unit_scores: tuple[ReactorUnitForecastScore, ...]

    def __post_init__(self) -> None:
        if tuple(v.unit_id for v in self.unit_scores) != tuple(
            s.unit_id for s in assigned_scenarios()
        ):
            raise ValueError(
                "calibration must account for all 32 calibration and ten heldout units"
            )

    @property
    def bounds(self) -> Triple | None:
        calibration = tuple(
            s.maximum_absolute_errors for s in self.unit_scores if s.split == "calibration"
        )
        if any(s is None for s in calibration):
            return None
        complete = tuple(s for s in calibration if s is not None)
        return cast(
            Triple, tuple(max(s[k] for s in complete) + NUMERICAL_TOLERANCE[k] for k in range(3))
        )

    @property
    def heldout_coverage(self) -> tuple[tuple[str, bool], ...]:
        bounds = self.bounds
        return tuple(
            (
                s.unit_id,
                bounds is not None
                and s.maximum_absolute_errors is not None
                and all(
                    error <= bound
                    for error, bound in zip(s.maximum_absolute_errors, bounds, strict=True)
                ),
            )
            for s in self.unit_scores
            if s.split == "heldout"
        )
