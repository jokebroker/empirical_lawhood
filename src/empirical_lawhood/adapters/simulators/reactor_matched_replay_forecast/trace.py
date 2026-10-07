"""Five nested episodes per fresh root: feedback, matched commands and a fixed pulse.

The native donor command tape is an explicit prescribed input to replay assays.
It is never advertised as an available future feedback command forecast.
"""

from dataclasses import dataclass
from decimal import Decimal as D
from typing import Callable, ClassVar, cast

from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_decimal
from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_trace import GRID_COLUMNS, GridRow, StageRow, ReactorBatchTrace, acquire_reference_batch
from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_bridge import ReactorBatchBridge
from empirical_lawhood.adapters.simulators.reactor_prefix_response.bridge import _decimal
from empirical_lawhood.adapters.simulators.reactor_prefix_response.contracts import ReactorCommand
from .design import ReactorAssignedScenario, ReactorBatchSource, BATCH_SOURCE_SHA256, PULSE_ONSET, PULSE_RETURN, PULSE_JACKET_K

MAXIMUM_BATCH_BYTES = 192 * 1024**2
# time, T, Tj, nA, conversion; predictions are only at declared ten-second clocks.
ResponseRow = tuple[D, D, D, D, D]


def pulse_command(command: ReactorCommand) -> ReactorCommand:
    jacket = command.jacket_k
    if PULSE_ONSET <= command.time_s < PULSE_RETURN:
        jacket = max(D("279.3"), jacket + PULSE_JACKET_K)
    return ReactorCommand(
        command.decision_id + ".pulse", command.time_s, command.feed_kg_s, jacket
    )


@dataclass(frozen=True, slots=True)
class ReactorReplay(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/reactor-matched-replay-forecast/reactor-replay'
    plant_dt_s: D
    pulse: bool
    observed_stages: tuple[StageRow, ...]
    native_grid: tuple[GridRow, ...]

    def __post_init__(self) -> None:
        if self.plant_dt_s not in (D(1), D(".5")) or type(self.pulse) is not bool:
            raise ValueError("undeclared replay view or intervention")
        if (
            len(self.observed_stages) != 2880
            or len(self.native_grid) != int(28800 / self.plant_dt_s) + 1
        ):
            raise ValueError("replay must retain the complete native batch")
        for i, row in enumerate(self.native_grid):
            if len(row) != 8 or row[0] != i * self.plant_dt_s:
                raise ValueError("replay changes grid columns or clocks")
            for value in row:
                validate_decimal(value, field_name="replay-native-grid", minimum=D(0))
        for stages in self.observed_stages:
            if len(stages) != 4:
                raise ValueError("replay changes action stages")
            for value in stages:
                validate_decimal(value, field_name="replay-action-stage", minimum=D(0))


@dataclass(frozen=True, slots=True)
class ReactorHistoryForecast(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/reactor-matched-replay-forecast/reactor-history-forecast'
    state_sha256: str
    command_tape_sha256: str
    plant_dt_s: D
    baseline: tuple[ResponseRow, ...]
    pulse: tuple[ResponseRow, ...]

    def __post_init__(self) -> None:
        from empirical_lawhood.kernel.serialization import validate_sha256

        validate_sha256(self.state_sha256, field_name="state_sha256")
        validate_sha256(self.command_tape_sha256, field_name="command_tape_sha256")
        if self.plant_dt_s not in (D(1), D(".5")):
            raise ValueError("history forecast view differs")
        for rows in (self.baseline, self.pulse):
            if len(rows) != 181:
                raise ValueError("history forecast requires all prescribed clocks")
            for i, row in enumerate(rows):
                if len(row) != 5 or row[0] != PULSE_ONSET + 10 * i:
                    raise ValueError("history forecast changes its clock")
                for value in row:
                    validate_decimal(value, field_name="history-forecast", minimum=D(0))
        if self.baseline[0] != self.pulse[0]:
            raise ValueError("paired forecasts must share the pre-pulse causal state")


def tape_digest(trace: ReactorBatchTrace) -> str:
    from hashlib import sha256
    from empirical_lawhood.kernel.serialization import canonical_json_bytes

    return sha256(
        canonical_json_bytes(tuple(f.command.to_document() for f in trace.forecasts))
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class ReactorBatchTrace(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/reactor-matched-replay-forecast/reactor-batch-trace'
    scenario: ReactorAssignedScenario
    source_sha256: str
    reference: tuple[ReactorBatchTrace, ReactorBatchTrace]
    matched_refined: ReactorReplay | None
    pulse_native: ReactorReplay | None
    pulse_refined: ReactorReplay | None
    history_forecasts: tuple[ReactorHistoryForecast, ...]
    failure_code: str | None = None

    def __post_init__(self) -> None:
        if self.source_sha256 != BATCH_SOURCE_SHA256 or len(self.reference) != 2:
            raise ValueError("history unit source or reference roster differs")
        if self.failure_code is not None:
            if self.failure_code not in {
                "ASSIGNED_REFERENCE_FAILURE",
                "ASSIGNED_REPLAY_OR_SHADOW_FAILURE",
            }:
                raise ValueError("unknown assigned history failure")
            if any(
                t.scenario != self.scenario or t.source_sha256 != self.source_sha256
                for t in self.reference
            ):
                raise ValueError("failed history unit changed its assignment")
            return
        if self.matched_refined is None or self.pulse_native is None or self.pulse_refined is None:
            raise ValueError("complete history unit lacks an assigned replay")
        if any(
            t.scenario != self.scenario or t.source_sha256 != self.source_sha256 or t.failure_code
            for t in self.reference
        ):
            raise ValueError("history unit requires both complete same-root reference episodes")
        if tuple(t.plant_dt_s for t in self.reference) != (D(1), D(".5")):
            raise ValueError("history unit changes feedback views")
        if tuple(
            (t.plant_dt_s, t.pulse)
            for t in (self.matched_refined, self.pulse_native, self.pulse_refined)
        ) != ((D(".5"), False), (D(1), True), (D(".5"), True)):
            raise ValueError("history unit changes replay roster")
        if len(self.history_forecasts) != 2 or tuple(
            t.plant_dt_s for t in self.history_forecasts
        ) != (D(1), D(".5")):
            raise ValueError("history unit forecast roster differs")
        donor = self.reference[0]
        if any(
            f.state_sha256 != donor.states[PULSE_ONSET // 10].fingerprint()
            or f.command_tape_sha256 != tape_digest(donor)
            for f in self.history_forecasts
        ):
            raise ValueError("history forecast changes pre-pulse state or prescribed commands")
        # Same requests plus the native actuator map must yield identical callback
        # accepted/applied stages for the fixed-input comparison.
        if donor.observed_stages != self.matched_refined.observed_stages:
            raise ValueError("matched-command replay does not match applied callback stages")
        if self.pulse_native.observed_stages != self.pulse_refined.observed_stages:
            raise ValueError("pulse numerical views do not match applied callback stages")


def acquire_replay(
    source: ReactorBatchSource,
    donor: ReactorBatchTrace,
    dt: D,
    pulse: bool,
    progress: Callable[[int], None] | None = None,
) -> ReactorReplay:
    stages = []
    with ReactorBatchBridge(
        plant_bytes=source.plant_source.encode(),
        params_bytes=source.plant_params.encode(),
        public_scenarios_bytes=source.public_scenarios.encode(),
        assigned_scenario=donor.scenario,
        plant_dt_s=dt,
    ) as bridge:
        bridge.start()
        for i, forecast in enumerate(donor.forecasts):
            command = pulse_command(forecast.command) if pulse else forecast.command
            delivery = bridge.advance(command)
            stages.append(
                (
                    delivery.accepted_feed_kg_s,
                    delivery.accepted_jacket_k,
                    delivery.applied_feed_kg_s,
                    delivery.applied_jacket_k,
                )
            )
            if progress is not None and ((i + 1) % 100 == 0 or i == 2879):
                progress(i + 1)
        result = bridge.completed_result
        grid = tuple(
            cast(GridRow, tuple(_decimal(v) for v in row))
            for row in zip(*(getattr(result, name) for name in GRID_COLUMNS), strict=True)
        )
    return ReactorReplay(dt, pulse, tuple(stages), grid)


def acquire_history_unit(
    source: ReactorBatchSource,
    scenario: ReactorAssignedScenario,
    progress: Callable[[int], None] | None = None,
) -> ReactorBatchTrace:
    from .forecast import forecast_history

    def tick(episode: int) -> Callable[[int], None] | None:
        return None if progress is None else lambda n: progress(episode * 2880 + n)

    native = acquire_reference_batch(source, scenario, D(1), tick(0))
    refined = acquire_reference_batch(source, scenario, D(".5"), tick(1))
    if native.failure_code or refined.failure_code:
        return ReactorBatchTrace(
            scenario,
            source.fingerprint(),
            (native, refined),
            None,
            None,
            None,
            (),
            "ASSIGNED_REFERENCE_FAILURE",
        )
    predictions: tuple[ReactorHistoryForecast, ...] = ()
    matched = pulse_native = pulse_refined = None
    failure = None
    try:
        # Predicted shadow paths precede contact with either pulse outcome.
        predictions = tuple(forecast_history(source, native, dt) for dt in (D(1), D(".5")))
        matched = acquire_replay(source, native, D(".5"), False, tick(2))
        pulse_native = acquire_replay(source, native, D(1), True, tick(3))
        pulse_refined = acquire_replay(source, native, D(".5"), True, tick(4))
    except (OverflowError, FloatingPointError):
        failure = "ASSIGNED_REPLAY_OR_SHADOW_FAILURE"
    except ValueError as error:
        if str(error) != "native reactor emitted a nonfinite value":
            raise
        failure = "ASSIGNED_REPLAY_OR_SHADOW_FAILURE"
    except RuntimeError as error:
        cause = error.__cause__
        numerical = isinstance(cause, (OverflowError, FloatingPointError)) or (
            isinstance(cause, ValueError)
            and str(cause) == "native reactor emitted a nonfinite value"
        )
        if str(error) != "native reactor failed; no completion evidence" or not numerical:
            raise
        failure = "ASSIGNED_REPLAY_OR_SHADOW_FAILURE"
    return ReactorBatchTrace(
        scenario,
        source.fingerprint(),
        (native, refined),
        matched,
        pulse_native,
        pulse_refined,
        predictions,
        failure,
    )
