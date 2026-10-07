"""Exact outcome-blind finite expansion of classical reactor EL."""

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar

from empirical_lawhood.kernel.serialization import CanonicalRecord

PREFIX = "reactor-finite-control-frontier-programme"
CONTEXTS = ("early", "prepared_t0", "middle", "late")
HORIZONS = (10, 30, 60)
GUARD = 120
RATES = (D(".008"), D(".016"), D(".032"))
DURATIONS = (10, 30, 60)
REQUESTS = tuple(
    map(
        D,
        (
            ".0004",
            ".0008",
            ".0012",
            ".0016",
            ".0024",
            ".0032",
            ".0048",
            ".0064",
            ".0096",
            ".0128",
            ".0192",
            ".0256",
            ".0384",
            ".0512",
        ),
    )
)
BUDGETS = tuple(map(D, (".08", ".16", ".32", ".64", "1.28", "2.10")))
POLICIES = ("EL_BOUND", "FIXED_SELECTED_ACTION", "FIXED_TUNED", "DIRECT_MEAN", "MECH_BOUND")
ROOTS = (
    *((f"reactor-regime-response-fit-{i:03d}", "development", i, 98000 + i) for i in range(32)),
    *(
        (f"reactor-finite-control-frontier-{role}-{i:03d}", role, i, start + i)
        for role, count, start in (("qualification", 96, 98700), ("prospective", 64, 98800))
        for i in range(count)
    ),
)


@dataclass(frozen=True, slots=True)
class Pulse(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-finite-control-frontier/pulse'
    rate_kg_s: D
    duration_s: int

    def __post_init__(self) -> None:
        if (
            type(self.rate_kg_s) is not D
            or type(self.duration_s) is not int
            or not (
                (self.rate_kg_s == 0 and self.duration_s == 0)
                or (self.rate_kg_s in RATES and self.duration_s in DURATIONS)
            )
        ):
            raise ValueError("pulse is outside the declared frontier menu")

    @property
    def word_id(self) -> str:
        return f"f{int(self.rate_kg_s * 1000):03d}-d{self.duration_s:03d}"

    def request(self, offset_s: int) -> D:
        if type(offset_s) is not int or offset_s not in range(0, GUARD, 10):
            raise ValueError("pulse callback is outside its precommitted window")
        return self.rate_kg_s if offset_s < self.duration_s else D(0)


ZERO = Pulse(D(0), 0)
PULSES = tuple(Pulse(rate, duration) for duration in DURATIONS for rate in RATES)
WORDS = (ZERO, *PULSES)


@dataclass(frozen=True, slots=True)
class Coordinate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-finite-control-frontier/coordinate'
    context: str
    pulse: Pulse
    horizon_s: int

    def __post_init__(self) -> None:
        if (
            self.context not in CONTEXTS
            or self.pulse not in PULSES
            or type(self.horizon_s) is not int
            or self.horizon_s not in HORIZONS
            or self.pulse.duration_s > self.horizon_s
        ):
            raise ValueError("undeclared response coordinate or truncated positive pulse")

    @property
    def coordinate_id(self) -> str:
        return f"{self.context}.{self.pulse.word_id}.t{self.horizon_s:03d}"

    @property
    def epsilon_K(self) -> D:
        return D(".000001") * D(self.horizon_s) / 10


COORDINATES = tuple(
    Coordinate(c, w, t) for c in CONTEXTS for t in HORIZONS for w in PULSES if w.duration_s <= t
)


@dataclass(frozen=True, slots=True)
class FrontierDesign(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-finite-control-frontier/frontier-design'
    config_id: str = PREFIX
    assigned_roots: tuple[tuple[str, str, int, int], ...] = ROOTS
    coordinates: tuple[Coordinate, ...] = COORDINATES
    guard_s: int = GUARD
    requests_K: tuple[D, ...] = REQUESTS
    budgets_kg: tuple[D, ...] = BUDGETS
    policies: tuple[str, ...] = POLICIES
    temperature_limit_K: D = D("356.2")
    peak_view_tolerance_K: D = D(".01")
    minimum_development_contact: int = 29
    qualification_probability: D = D(".90")
    qualification_alpha: D = D(".05") / 72
    prospective_evaluation_joined_probability: D = D(".70")
    prospective_evaluation_false_admission_ceiling: D = D(".10")
    prospective_evaluation_alpha: D = D(".05") / 24
    bootstrap_draws: int = 20000
    bootstrap_seed: int = 20260926
    native_call_cap: int = 18752
    wall_seconds: int = 129600
    cpu_seconds: int = 388800

    def __post_init__(self) -> None:
        for name, field in self.__dataclass_fields__.items():
            if name != "SCHEMA" and (
                getattr(self, name) != field.default
                or type(getattr(self, name)) is not type(field.default)
            ):
                raise ValueError(f"frontier design changed its predeclared {name}")


def assignment(root: str) -> tuple[str, int]:
    matches = [(role, seed) for name, role, _, seed in ROOTS if name == root]
    if len(matches) != 1:
        raise ValueError("root is outside the declared frontier assignment")
    return matches[0]
