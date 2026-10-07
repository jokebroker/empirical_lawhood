"""Predeclared, bounded reactor acquisition roster and native callback outputs."""

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from typing import ClassVar

from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id

from .contracts import PARAMS_SHA256, PLANT_SHA256, PUBLIC_SCENARIOS_SHA256, ReactorCommand, ReactorDelivery, ReactorMeasurement

CALIBRATION = ("cooling_loss", "fouling", "nominal")
HELDOUT = ("feed_temp", "kinetic_hot")
SCENARIOS = tuple(sorted((*CALIBRATION, *HELDOUT)))
VIEWS = (("native", Decimal(1)), ("refined", Decimal("0.5")))
ARMS = (("comparator", Decimal(316)), ("pulse", Decimal(315)))


BRANCHES = tuple((s, v, a) for s in SCENARIOS for v, _ in VIEWS for a, _ in ARMS)


def native_task_id(scenario: str, view: str, arm: str, *, config=None) -> str:
    if (scenario, view, arm) not in (BRANCHES if config is None else config.branches):
        raise ValueError("reactor task requires a declared public scenario")
    return f"reactor-native-prefix.{scenario.replace('_', '-')}.{view}.{arm}"


@dataclass(frozen=True, slots=True)
class ReactorPrefixConfig(CanonicalRecord):
    """Exactly 20 paired branches, five units and two views; no private panel.

    Commands are feed=0 at both decisions, jacket=315 or 316 K at t=0,
    jacket=316 K at t=10. The observation at t=20 contains delayed temperature
    from t=10 and current dose. Numerical refinements and action pairs retain
    their parent's unit; they are never counted as additional replication.
    """

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/reactor-prefix-response/reactor-prefix-config'
    config_id: str
    calibration_scenarios: tuple[str, ...]
    heldout_scenarios: tuple[str, ...]
    horizon_s: Decimal
    numpy_version: str
    python_version: str

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        if (
            self.calibration_scenarios != CALIBRATION
            or self.heldout_scenarios != HELDOUT
            or self.horizon_s != Decimal(20)
            or self.numpy_version != "2.4.6"
            or self.python_version != "3.11.14"
        ):
            raise ValueError(
                "reactor prefix changes the declared roster, horizon or numerical runtime"
            )

    @property
    def branches(self) -> tuple[tuple[str, str, str], ...]:
        return BRANCHES


@dataclass(frozen=True, slots=True)
class ReactorSourceBundle(CanonicalRecord):
    """Authenticated source artifact, never an executable configuration value."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/reactor-prefix-response/reactor-source-bundle'
    plant_source: str
    plant_params: str
    public_scenarios: str

    def __post_init__(self) -> None:
        for value, maximum, digest in (
            (self.plant_source, 32_768, PLANT_SHA256),
            (self.plant_params, 4_096, PARAMS_SHA256),
            (self.public_scenarios, 8_192, PUBLIC_SCENARIOS_SHA256),
        ):
            data = value.encode("utf-8")
            if len(data) > maximum or sha256(data).hexdigest() != digest:
                raise ValueError(
                    "reactor source artifact differs from the installed exact pin"
                )


@dataclass(frozen=True, slots=True)
class ReactorPrefixEpisode(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/reactor-prefix-response/reactor-prefix-episode'
    scenario_id: str
    view_id: str
    arm_id: str
    initial_measurement: ReactorMeasurement | None
    deliveries: tuple[ReactorDelivery, ...]
    failure_code: str | None

    def __post_init__(self) -> None:
        validate_prefix_episode(self, SCENARIOS)


def validate_prefix_episode(
    self: ReactorPrefixEpisode, scenarios: tuple[str, ...]
) -> None:
    if (
        self.scenario_id not in scenarios
        or self.view_id not in dict(VIEWS)
        or self.arm_id not in dict(ARMS)
    ):
        raise ValueError("reactor episode is outside the declared panel")
    if len(self.deliveries) > 2:
        raise ValueError("reactor episode exceeds its two-decision bound")
    if self.initial_measurement is None and self.deliveries:
        raise ValueError("reactor deliveries require an initial observation")
    if self.initial_measurement is not None and self.initial_measurement.time_s != 0:
        raise ValueError("reactor initial observation uses another clock")
    for index, delivery in enumerate(self.deliveries):
        expected_jacket = dict(ARMS)[self.arm_id] if index == 0 else Decimal(316)
        if (
            delivery.command.time_s != 10 * index
            or delivery.command.feed_kg_s != 0
            or delivery.command.jacket_k != expected_jacket
        ):
            raise ValueError("reactor episode changed its predeclared command")
        if any(e.duration_s != dict(VIEWS)[self.view_id] for e in delivery.exposures):
            raise ValueError("reactor episode changed its numerical view")
    if self.failure_code is None:
        if self.initial_measurement is None or len(self.deliveries) != 2:
            raise ValueError(
                "complete reactor prefix requires both observed deliveries"
            )
    elif self.failure_code != "REACTOR_NATIVE_PREFIX_FAILED":
        raise ValueError("unknown reactor prefix failure")


@dataclass(frozen=True, slots=True)
class ReactorPrefixPanel(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/reactor-prefix-response/reactor-prefix-panel'
    config: ReactorPrefixConfig
    episodes: tuple[ReactorPrefixEpisode, ...]
    branch: tuple[str, str, str] | None = None

    def __post_init__(self) -> None:
        if self.branch is not None and self.branch not in BRANCHES:
            raise ValueError("reactor panel names an undeclared branch")
        expected = BRANCHES if self.branch is None else (self.branch,)
        if (
            tuple((e.scenario_id, e.view_id, e.arm_id) for e in self.episodes)
            != expected
        ):
            raise ValueError(
                "reactor panel must retain every assigned unit, view and arm exactly once"
            )


def acquire_prefix_branch(
    config: ReactorPrefixConfig,
    source: ReactorSourceBundle,
    branch: tuple[str, str, str],
    progress: Callable[[int], None] | None = None,
) -> ReactorPrefixPanel:
    """Native runner body, called only after execution closure and input authority."""
    import platform
    from importlib.metadata import version

    from .bridge import ReactorBridge, ReactorWorkerStopError

    if (
        version("numpy") != config.numpy_version
        or platform.python_version() != config.python_version
    ):
        raise ValueError(
            "reactor numerical runtime differs before native source execution"
        )
    scenario, view, arm = branch
    native_task_id(*branch, config=config)
    dt, jacket = dict(VIEWS)[view], dict(ARMS)[arm]
    first = None
    deliveries = []
    failure = None
    completed = 0
    from .assigned import AssignedPrefixBridge, ReactorAssignedPrefixConfig, ReactorAssignedPrefixEpisode, ReactorAssignedPrefixPanel

    assigned = type(config) is ReactorAssignedPrefixConfig
    unit = (
        next(u for u in config.assignment.units if u.unit_id == scenario)
        if assigned
        else None
    )
    bridge_inputs = {
        "plant_bytes": source.plant_source.encode(),
        "params_bytes": source.plant_params.encode(),
        "public_scenarios_bytes": source.public_scenarios.encode(),
        "plant_dt_s": dt,
    }
    try:
        bridge = (
            AssignedPrefixBridge.create(unit=unit, **bridge_inputs)
            if assigned
            else ReactorBridge(scenario_id=scenario, **bridge_inputs)
        )
        with bridge:
            first = bridge.start()
            for index, value in enumerate((jacket, Decimal(316))):
                deliveries.append(
                    bridge.deliver(
                        ReactorCommand(
                            f"decision.{scenario}.{view}.{arm}.{index}",
                            Decimal(10 * index),
                            Decimal(0),
                            value,
                        )
                    )
                )
                completed += len(deliveries[-1].exposures)
                if progress is not None:
                    progress(completed)
    except ReactorWorkerStopError:
        raise
    except (ValueError, RuntimeError):
        # No retry, substitute branch, dropped unit or fabricated end state.
        failure = "REACTOR_NATIVE_PREFIX_FAILED"
    if assigned:
        episode = ReactorAssignedPrefixEpisode(
            scenario, view, arm, first, tuple(deliveries), failure, unit
        )
        return ReactorAssignedPrefixPanel(config, (episode,), branch)
    episode = ReactorPrefixEpisode(
        scenario, view, arm, first, tuple(deliveries), failure
    )
    return ReactorPrefixPanel(config, (episode,), branch)
