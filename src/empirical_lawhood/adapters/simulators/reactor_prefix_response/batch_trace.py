"""Bounded full-batch acquisition with sealed native truth and causal forecasts."""

from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, ClassVar, cast

from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_decimal
from .batch_bridge import ReactorBatchBridge
from .batch_design import ReactorAssignedScenario, ReactorBatchSource, validate_nominal_spec
from .bridge import ReactorWorkerStopError, _decimal
from .contracts import ReactorCommand, ReactorDelivery, ReactorExposure, ReactorFinalDelivery
from .forecast import ReferenceObserverReactorForecast, ReactorForecastState, ReactorPointForecast

GRID_COLUMNS = (
    "time_s",
    "t_reactor_k",
    "t_jacket_k",
    "dosed_kg",
    "conversion_b",
    "n_a_mol",
    "feed_applied_kg_s",
    "tj_cmd_applied_k",
)
MAXIMUM_BATCH_BYTES = 96 * 1024**2
GridRow = tuple[Decimal, Decimal, Decimal, Decimal, Decimal, Decimal, Decimal, Decimal]
StageRow = tuple[Decimal, Decimal, Decimal, Decimal]


@dataclass(frozen=True, slots=True)
class ReactorBatchTrace(CanonicalRecord):
    """Completed or explicitly invalid assigned episode; nested rows are not n."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/reactor-prefix-response/reactor-batch-trace'
    scenario: ReactorAssignedScenario
    plant_dt_s: Decimal
    source_sha256: str
    states: tuple[ReactorForecastState, ...]
    forecasts: tuple[ReactorPointForecast, ...]
    observed_stages: tuple[StageRow, ...]
    native_grid: tuple[GridRow, ...]
    failure_code: str | None

    def __post_init__(self) -> None:
        from empirical_lawhood.kernel.serialization import validate_sha256

        validate_sha256(self.source_sha256, field_name="source_sha256")
        if self.plant_dt_s not in (Decimal(1), Decimal(".5")):
            raise ValueError("batch trace uses an undeclared numerical view")
        if not 0 <= len(self.observed_stages) <= len(self.forecasts) == len(self.states) <= 2880:
            raise ValueError("batch trace changes the ordered forecast/delivery census")
        if self.failure_code is None:
            if (
                len(self.states) != 2880
                or len(self.observed_stages) != 2880
                or len(self.native_grid) != int(Decimal(28800) / self.plant_dt_s) + 1
            ):
                raise ValueError("complete batch trace requires the full native horizon")
        elif (
            self.failure_code not in {"NATIVE_BATCH_FAILURE", "FORECAST_NUMERICAL_FAILURE"}
            or self.native_grid
        ):
            raise ValueError(
                "invalid batch must retain its explicit failure and no fabricated grid"
            )
        for index, (state, prediction) in enumerate(zip(self.states, self.forecasts, strict=True)):
            if (
                state.measurement.time_s != 10 * index
                or prediction.state_sha256 != state.fingerprint()
                or prediction.command.time_s != 10 * index
                or prediction.command.feed_kg_s != state.reference_feed_kg_s
                or prediction.command.jacket_k != state.reference_jacket_k
                or prediction.plant_dt_s != self.plant_dt_s
            ):
                raise ValueError(
                    "batch forecast changes causal state, clock, reference policy or view"
                )
        for stage_row in self.observed_stages:
            if len(stage_row) != 4:
                raise ValueError("batch action stages have another column count")
            for value in stage_row:
                validate_decimal(value, field_name="native-stage", minimum=Decimal(0))
        for index, row in enumerate(self.native_grid):
            if len(row) != len(GRID_COLUMNS) or row[0] != index * self.plant_dt_s:
                raise ValueError("native batch grid changes its columns or clock")
            for value in row:
                validate_decimal(value, field_name="native-grid", minimum=Decimal(0))


def acquire_reference_batch(
    source: ReactorBatchSource,
    scenario: ReactorAssignedScenario,
    plant_dt_s: Decimal,
    progress: Callable[[int], None] | None = None,
) -> ReactorBatchTrace:
    """Called only by the authorized native task runner, never by a diagnostic."""
    validate_nominal_spec(source)
    model = ReferenceObserverReactorForecast(
        plant_bytes=source.plant_source.encode(),
        params_bytes=source.plant_params.encode(),
        reference_bytes=source.reference_controller.encode(),
    )
    states: list[ReactorForecastState] = []
    predictions: list[ReactorPointForecast] = []
    stages: list[StageRow] = []
    exposures: list[ReactorExposure] = []
    failure = None
    grid: tuple[GridRow, ...] = ()
    with ReactorBatchBridge(
        plant_bytes=source.plant_source.encode(),
        params_bytes=source.plant_params.encode(),
        public_scenarios_bytes=source.public_scenarios.encode(),
        assigned_scenario=scenario,
        plant_dt_s=plant_dt_s,
    ) as bridge:
        try:
            measurement = bridge.start()
            for index in range(2880):
                try:
                    state = model.observe(measurement)
                    command = ReactorCommand(
                        f"{scenario.unit_id}.decision.{index:04d}",
                        measurement.time_s,
                        state.reference_feed_kg_s,
                        state.reference_jacket_k,
                    )
                    prediction = model.predict(state, command, plant_dt_s=plant_dt_s)
                except (OverflowError, FloatingPointError):
                    # Source/config/programming ValueErrors deliberately propagate.
                    failure = "FORECAST_NUMERICAL_FAILURE"
                    break
                except ValueError as error:
                    if str(error) != "native reactor emitted a nonfinite value":
                        raise
                    failure = "FORECAST_NUMERICAL_FAILURE"
                    break
                states.append(state)
                predictions.append(prediction)
                delivery = bridge.advance(command)
                model.commit(state, command)
                stages.append(
                    (
                        delivery.accepted_feed_kg_s,
                        delivery.accepted_jacket_k,
                        delivery.applied_feed_kg_s,
                        delivery.applied_jacket_k,
                    )
                )
                exposures.extend(delivery.exposures)
                if progress is not None and ((index + 1) % 100 == 0 or index == 2879):
                    progress(index + 1)
                if isinstance(delivery, ReactorDelivery):
                    measurement = delivery.next_measurement
                elif not isinstance(delivery, ReactorFinalDelivery) or index != 2879:
                    raise ValueError("native batch terminated at another decision")
            if failure is None:
                result = bridge.completed_result
                columns = tuple(getattr(result, name) for name in GRID_COLUMNS)
                grid = tuple(
                    cast(GridRow, tuple(_decimal(v) for v in row))
                    for row in zip(*columns, strict=True)
                )
                if len(exposures) != len(grid) - 1 or any(
                    e.time_s + e.duration_s != row[0]
                    or e.feed_kg_s != row[6]
                    or e.jacket_k != row[7]
                    for e, row in zip(exposures, grid[1:], strict=True)
                ):
                    raise ValueError("native batch arrays contradict observed realized delivery")
        except ReactorWorkerStopError:
            raise
        except ValueError as error:
            if str(error) != "native reactor emitted a nonfinite value":
                raise
            failure = "NATIVE_BATCH_FAILURE"
            grid = ()
        except RuntimeError as error:
            if str(error) not in {
                "native reactor failed; no completion evidence",
                "reactor delivery timed out; no completion evidence",
            }:
                raise
            if str(error) == "native reactor failed; no completion evidence":
                cause = error.__cause__
                numerical = isinstance(cause, (OverflowError, FloatingPointError)) or (
                    isinstance(cause, ValueError)
                    and (
                        str(cause) == "native reactor emitted a nonfinite value"
                        or str(cause).startswith("non-finite controller output at t=")
                    )
                )
                if not numerical:
                    # A binding/contract defect is not a failed scientific assay.
                    raise
            failure = "NATIVE_BATCH_FAILURE"
    return ReactorBatchTrace(
        scenario,
        plant_dt_s,
        source.fingerprint(),
        tuple(states),
        tuple(predictions),
        tuple(stages),
        grid,
        failure,
    )
