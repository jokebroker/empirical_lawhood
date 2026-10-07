"""Native measurement binding; effects are invoked only by an issued task runner."""

from dataclasses import asdict, dataclass
from decimal import Decimal as D
import json
from time import process_time, monotonic
from typing import Any, Callable, Protocol
import numpy as np
from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_bridge import ReactorBatchBridge
from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_design import ReactorAssignedScenario, ReactorBatchSource
from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_trace import GRID_COLUMNS
from empirical_lawhood.adapters.simulators.reactor_prefix_response.contracts import ReactorCommand, ReactorDelivery, ReactorFinalDelivery
from empirical_lawhood.adapters.methods.reactor_causal_response.numerical import Observation, FrozenFit, branch_tape, exploration, digest
from empirical_lawhood.adapters.methods.reactor_causal_response.controller import NumericalController, EmpiricalNonattempt
from empirical_lawhood.adapters.methods.reactor_causal_response.calibration import RootOperands
from empirical_lawhood.adapters.methods.reactor_causal_response.experiment_records import EmpiricalEpisodeEnvelope, EmpiricalArrayPayload
from .interface import Actuator, measured_labels, project_tape


class NativeController(Protocol):
    def reset(self, spec: dict[str, Any]) -> None: ...
    def step(self, t_s: float, y: dict[str, float], dt_s: float) -> tuple[float, float]: ...


class CallbackOwner(Protocol):
    failed: bool

    def advance(
        self, controller: NativeController, previous: tuple[float, float], session: Any
    ) -> ReactorDelivery | ReactorFinalDelivery | None: ...


@dataclass(frozen=True)
class NativeEpisode:
    root: str
    episode: str
    dt: float
    observations: np.ndarray
    requests: np.ndarray
    stages: np.ndarray
    exposure: np.ndarray
    grid: np.ndarray
    callback_cpu: np.ndarray
    failure: str | None

    @property
    def complete(self) -> bool:
        return (
            self.failure is None
            and self.observations.shape == (2880, 4)
            and self.stages.shape == (2880, 4)
            and self.callback_cpu.shape == (2880,)
            and all(
                a.dtype == np.dtype("float64")
                for a in (
                    self.observations,
                    self.requests,
                    self.stages,
                    self.exposure,
                    self.grid,
                    self.callback_cpu,
                )
            )
            and self.requests.shape == (2880, 2)
            and self.exposure.shape == (2880, int(10 / self.dt), 4)
            and self.grid.shape == (int(28800 / self.dt) + 1, 8)
        )

    def envelope(self) -> 'EmpiricalEpisodeEnvelope':
        return EmpiricalEpisodeEnvelope(
            self.root,
            self.episode,
            self.dt == 0.5,
            self.failure,
            EmpiricalArrayPayload.pack(
                {
                    name: getattr(self, name)
                    for name in (
                        "observations",
                        "requests",
                        "stages",
                        "exposure",
                        "grid",
                        "callback_cpu",
                    )
                }
            ),
        )

    @classmethod
    def from_envelope(cls, value: 'EmpiricalEpisodeEnvelope') -> "NativeEpisode":
        arrays = value.arrays.unpack()
        if set(arrays) != {
            "observations",
            "requests",
            "stages",
            "exposure",
            "grid",
            "callback_cpu",
        }:
            raise ValueError("native episode arrays differ")
        return cls(
            value.root,
            value.episode,
            0.5 if value.refined else 1.0,
            **arrays,
            failure=value.failure,
        )


class ExplorationController:
    def __init__(self, root: int, policy: int) -> None:
        self.root, self.policy = root, policy
        self.history: tuple[Observation, ...] = ()
        self.previous = (0.0, 316.0)
        self.actuator = Actuator()

    def reset(self, spec: dict[str, Any]) -> None:
        self.history, self.previous = (), (0.0, 316.0)

    def step(self, t_s: float, y: dict[str, float], dt_s: float) -> tuple[float, float]:
        o = Observation(
            float(t_s), float(y["t_reactor_k"]), float(y["t_jacket_k"]), float(y["dosed_kg"])
        )
        self.history += (o,)
        a = exploration(self.history, self.previous[1], self.root, self.policy)
        p = self.actuator.project(o, self.previous)[a]
        self.previous = p.applied
        return p.requested


class TapeController:
    def __init__(self, tape: np.ndarray) -> None:
        if tape.ndim != 2 or tape.shape[1] != 2 or len(tape) > 2880 or not np.isfinite(tape).all():
            raise ValueError("invalid donor request tape")
        self.tape = tape.copy()

    def reset(self, spec: dict[str, Any]) -> None:
        pass

    def step(self, t_s: float, y: dict[str, float], dt_s: float) -> tuple[float, float]:
        k = int(t_s / 10)
        if k >= len(self.tape):
            raise EmpiricalNonattempt("DONOR_TAPE_INCOMPLETE")
        return float(self.tape[k, 0]), float(self.tape[k, 1])


def acquire_episode(
    source: ReactorBatchSource,
    scenario: ReactorAssignedScenario,
    episode: str,
    controller: NativeController,
    *,
    dt: float = 1.0,
    bridge_factory: Callable[..., Any] = ReactorBatchBridge,
    owner: CallbackOwner | None = None,
    progress: Callable[[int], None] | None = None,
    callback_wall: list[float] | None = None,
    retain_stopped_prefix: bool = False,
) -> NativeEpisode:
    if dt not in (1.0, 0.5):
        raise ValueError("undeclared numerical view")
    if retain_stopped_prefix and bridge_factory is ReactorBatchBridge:
        from empirical_lawhood.adapters.simulators.reactor_prefix_response.prefix_bridge import ReactorPrefixBridge

        bridge_factory = ReactorPrefixBridge
    observations, requests, exposure, cpu = [], [], [], []
    stages: list[tuple[float, ...]] = []
    grid = np.empty((0, 8))
    failure = None
    controller.reset(json.loads(source.plant_params))
    with bridge_factory(
        plant_bytes=source.plant_source.encode(),
        params_bytes=source.plant_params.encode(),
        public_scenarios_bytes=source.public_scenarios.encode(),
        assigned_scenario=scenario,
        plant_dt_s=D(str(dt)),
    ) as bridge:
        try:
            measurement = bridge.start()
        except RuntimeError as error:
            if str(error) not in {
                "reactor delivery timed out; no completion evidence",
                "native reactor failed; no completion evidence",
            }:
                raise
            return NativeEpisode(
                scenario.unit_id,
                episode,
                dt,
                np.empty((0, 4)),
                np.empty((0, 2)),
                np.empty((0, 4)),
                np.empty((0, int(10 / dt), 4)),
                grid,
                np.empty((0,)),
                "NATIVE_INITIAL_OBSERVATION_FAILURE",
            )
        for k in range(2880):
            previous = (0.0, 316.0) if not stages else (stages[-1][2], stages[-1][3])
            o = (
                float(measurement.time_s),
                float(measurement.t_reactor_k),
                float(measurement.t_jacket_k),
                float(measurement.dosed_kg),
            )
            if o[0] != k * 10:
                raise ValueError("native callback order changed")
            observations.append(o)
            wall_start = monotonic()
            start = process_time()
            try:
                requested = controller.step(
                    o[0], dict(t_reactor_k=o[1], t_jacket_k=o[2], dosed_kg=o[3]), 10.0
                )
            except EmpiricalNonattempt:
                cpu.append(process_time() - start)
                if callback_wall is not None:
                    callback_wall.append(monotonic() - wall_start)
                if owner is not None and owner.advance(controller, previous, bridge) is not None:
                    raise ValueError("a refusing policy delivered a native action")
                failure = "POLICY_NONATTEMPT"
                break
            cpu.append(process_time() - start)
            if callback_wall is not None:
                callback_wall.append(monotonic() - wall_start)
            if len(requested) != 2 or not np.isfinite(requested).all():
                failure = "NONFINITE_NATIVE_COMMAND"
                break
            command = ReactorCommand(
                f"{scenario.unit_id}.{episode.lower()}.{k:04d}",
                D(k * 10),
                D(repr(float(requested[0]))),
                D(repr(float(requested[1]))),
            )
            try:
                delivered = (
                    bridge.advance(command)
                    if owner is None
                    else owner.advance(controller, previous, bridge)
                )
            except RuntimeError as error:
                if str(error) not in {
                    "reactor delivery timed out; no completion evidence",
                    "native reactor failed; no completion evidence",
                }:
                    raise
                failure = "NATIVE_DELIVERY_FAILURE"
                break
            if delivered is None:
                failure = (
                    "OWNER_DELIVERY_FAILED"
                    if owner is not None and owner.failed
                    else "OWNER_NONATTEMPT"
                )
                break
            if owner is not None:
                command = delivered.command
                if (float(command.feed_kg_s), float(command.jacket_k)) != tuple(
                    requested
                ) or command.time_s != k * 10:
                    raise ValueError("owner delivery differs from numerical callback")
            if delivered.command != command:
                raise ValueError("native delivery changed the requested command")
            requests.append(requested)
            stages.append(
                tuple(
                    float(v)
                    for v in (
                        delivered.accepted_feed_kg_s,
                        delivered.accepted_jacket_k,
                        delivered.applied_feed_kg_s,
                        delivered.applied_jacket_k,
                    )
                )
            )
            exposure.append(
                [
                    (float(e.time_s), float(e.duration_s), float(e.feed_kg_s), float(e.jacket_k))
                    for e in delivered.exposures
                ]
            )
            if owner is not None and owner.failed:
                failure = "OWNER_INEXACT_DELIVERY"
                break
            if isinstance(delivered, ReactorDelivery):
                measurement = delivered.next_measurement
            elif not isinstance(delivered, ReactorFinalDelivery) or k != 2879:
                raise ValueError("final native interval is missing or misplaced")
            if progress is not None:
                progress(k + 1)
        if failure is None:
            result = bridge.completed_result
            grid = np.column_stack([getattr(result, name) for name in GRID_COLUMNS])
            actual = np.asarray(exposure).reshape(-1, 4)
            if not np.array_equal(actual[:, 0] + actual[:, 1], grid[1:, 0]) or not np.array_equal(
                actual[:, 2:], grid[1:, 6:8]
            ):
                raise ValueError("native result contradicts actual delivery exposures")
        elif retain_stopped_prefix:
            bridge.close()
            captured = bridge.closed_prefix_grid
            # A failed macro can have advanced farther than the recorder. Only
            # the prefix with retained actual command/exposure correspondence
            # is available here; its unknown remainder cannot become success.
            count = min(len(captured), len(requests) * int(10 / dt) + 1)
            grid = captured[:count].copy()
            actual = np.asarray(exposure).reshape(-1, 4)[: max(0, count - 1)]
            if count and (
                not np.array_equal(actual[:, 0] + actual[:, 1], grid[1:, 0])
                or not np.array_equal(actual[:, 2:], grid[1:, 6:8])
            ):
                raise ValueError("native stopped-prefix contradicts actual delivery exposures")
    return NativeEpisode(
        scenario.unit_id,
        episode,
        dt,
        np.asarray(observations).reshape(-1, 4),
        np.asarray(requests).reshape(-1, 2),
        np.asarray(stages).reshape(-1, 4),
        np.asarray(exposure).reshape(-1, int(10 / dt), 4),
        grid,
        np.asarray(cpu),
        failure,
    )


def paired_operands(
    nominal: NativeEpisode, refined: NativeEpisode, model: FrozenFit, q: float
) -> RootOperands:
    if nominal.root != refined.root or nominal.dt != 1 or refined.dt != 0.5:
        raise ValueError("paired views change assigned root or denominator")
    controller = NumericalController(model, q, Actuator())
    rows, actions = [], []
    for t, y, j, dose in nominal.observations:
        try:
            controller.step(
                float(t),
                dict(t_reactor_k=float(y), t_jacket_k=float(j), dosed_kg=float(dose)),
                10.0,
            )
        except EmpiricalNonattempt:
            break
        decision = controller.decisions[-1]
        assert decision.selected is not None
        rows.append(decision.candidates[decision.selected].features)
        actions.append(decision.selected)
        controller.decisions.clear()
    labels = np.empty((0, 2, 3))
    requests = np.empty((0, 2, 2))
    pn, pr = np.empty((0, 10, 4)), np.empty((0, 20, 4))
    if nominal.complete and refined.complete:
        labels = np.stack(
            [
                measured_labels(e.grid[:, 0], e.grid[:, 1], e.grid[:, 4], e.grid[:, 3], e.dt)
                for e in (nominal, refined)
            ],
            axis=1,
        )
        requests = np.stack([nominal.requests, refined.requests], axis=1)
        pn, pr = project_tape(nominal.requests, 1.0), project_tape(refined.requests, 0.5)
    return RootOperands(
        nominal.root,
        nominal.observations[:, 0],
        np.asarray(actions, dtype=int),
        np.asarray(rows).reshape(-1, 23),
        labels,
        pn,
        nominal.exposure,
        pr,
        refined.exposure,
        requests,
        nominal.observations,
        np.array([q]),
        np.array(list(bytes.fromhex(digest(asdict(model)))), dtype=np.int64),
        np.stack([nominal.stages, refined.stages], axis=1)
        if nominal.stages.shape == refined.stages.shape
        else np.empty((0, 2, 4)),
    )


def development_root(
    source: ReactorBatchSource,
    scenario: ReactorAssignedScenario,
    ordinal: int,
    *,
    bridge_factory: Callable[..., Any] = ReactorBatchBridge,
    progress: Callable[[int], None] | None = None,
) -> tuple[NativeEpisode, ...]:
    if ordinal not in range(24):
        raise ValueError("undeclared development ordinal")
    result: list[NativeEpisode] = []
    nominals: dict[str, NativeEpisode] = {}
    completed = 0

    def emit(n: int) -> None:
        if progress is not None:
            progress(completed + n)

    exploration_policies = {
        "exploration-unshifted": 0,
        "exploration-shifted-one-position": 1,
        "exploration-shifted-two-positions": 2,
    }
    for name in ("exploration-unshifted", "exploration-shifted-one-position", "exploration-shifted-two-positions", "feed-intervention", "jacket-intervention"):
        controller: NativeController
        if name in exploration_policies:
            controller = ExplorationController(ordinal, exploration_policies[name])
        else:
            donor = nominals["exploration-shifted-one-position"]
            if not donor.complete:
                result.extend(
                    NativeEpisode(
                        scenario.unit_id,
                        name,
                        dt,
                        np.empty((0, 4)),
                        np.empty((0, 2)),
                        np.empty((0, 4)),
                        np.empty((0, int(10 / dt), 4)),
                        np.empty((0, 8)),
                        np.empty((0,)),
                        "DONOR_TAPE_INCOMPLETE",
                    )
                    for dt in (1.0, 0.5)
                )
                continue
            anchor = (3600, 9000, 18000, 24000)[ordinal % 4] // 10
            previous_jacket = 316.0 if anchor == 0 else float(donor.stages[anchor - 1, 3])
            controller = TapeController(
                np.array(
                    branch_tape(tuple(map(tuple, donor.requests)), ordinal, name, previous_jacket)
                )
            )
        nominal = acquire_episode(
            source,
            scenario,
            name,
            controller,
            bridge_factory=bridge_factory,
            progress=None if progress is None else emit,
        )
        completed += len(nominal.requests)
        nominals[name] = nominal
        refined = acquire_episode(
            source,
            scenario,
            name,
            TapeController(nominal.requests),
            dt=0.5,
            bridge_factory=bridge_factory,
            progress=None if progress is None else emit,
        )
        completed += len(refined.requests)
        result.extend((nominal, refined))
    return tuple(result)


def acquire_with_progress(
    source: ReactorBatchSource,
    scenario: object,
    name: str,
    controller: ExplorationController | TapeController,
    dt: float,
    acquire: Callable[..., NativeEpisode],
    completed_callbacks: int,
    progress: Callable[[int], None] | None,
) -> tuple[NativeEpisode, int]:
    last = 0

    def emit(value: int) -> None:
        nonlocal last
        if not last < value <= 2880:
            raise ValueError("native callback progress is nonmonotone")
        last = value
        if progress is not None:
            progress(completed_callbacks + value)

    # The real callable is the installed native bridge; fixtures may inject a
    # no-effect fake. A failed call is retained and never silently retried.
    episode = (
        acquire(source, scenario, name, controller, dt=dt)
        if progress is None
        else acquire(source, scenario, name, controller, dt=dt, progress=emit)
    )
    return episode, completed_callbacks + last
