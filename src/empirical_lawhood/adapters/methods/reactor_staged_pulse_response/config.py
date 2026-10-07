"""Closed numerical specification; no outcome-dependent design choices."""

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id

PREFIX = "reactor-staged-pulse-response-programme"
CONTEXTS = ("early", "prepared_t0")
BASE_RATES = tuple(map(D, (".008", ".016", ".032")))
ADDED_RATES = tuple(map(D, (".012", ".020", ".024")))
GAMMAS = tuple(map(D, ("1", ".9", ".8", ".6", ".4")))
LADDER = tuple(map(D, (".0004", ".0008", ".0012", ".0016", ".0024", ".0032", ".0048", ".0064")))
ARMS = ("EL_INTERVAL", "EL_SERVICE", "DIRECT_SAFE", "MECH_SAFE")
MENUS = ("BASE", "EXPANDED")
SEQUENCES = ("EL_SEQUENCE", "FIXED_SEQUENCE", "ONE_PULSE")
STAGES = tuple(f"{block}-{role}" for block in ("base-menu-comparison", "expanded-menu-comparison", "staged-sequence-comparison") for role in ("NOMINATION", "QUALIFICATION", "PROSPECTIVE"))
DEVELOPMENT_ROOTS = tuple(f"reactor-regime-response-fit-{i:03d}" for i in range(32))
CALIBRATION_ROOTS = tuple(f"reactor-finite-control-frontier-qualification-{i:03d}" for i in range(96))


def retained_key(root: str, role: str) -> str:
    if root in DEVELOPMENT_ROOTS and role in ("assay", "preparation", "private", "measurement"):
        return f"development-{DEVELOPMENT_ROOTS.index(root):03d}-{role}"
    if root in CALIBRATION_ROOTS and role in ("measurement", "prediction"):
        return f"calibration-{CALIBRATION_ROOTS.index(root):03d}-{role}"
    raise ValueError("undeclared retained root/input role")


ROOTS = tuple(
    (f"reactor-staged-pulse-response-{block}-{role}-{i:03d}", block, role, start + i)
    for block, qstart, dstart in (
        ("base-menu-comparison", 110000, 110100),
        ("expanded-menu-comparison", 111100, 111200),
        ("staged-sequence-comparison", 112100, 112200),
    )
    for role, count, start in (("qualification", 96, qstart), ("prospective", 64, dstart))
    for i in range(count)
)
CALL_CAPS = (0, 2880, 2432, 768, 5184, 1408, 320, 1152, 1152)
# Per-stage reservations partition the fixed 18 wall / 54 CPU-hour allocation.
CPU_SECONDS = (1200, 19800, 43800, 10800, 30000, 45600, 3300, 8100, 31800)
WALL_SECONDS = (400, 6600, 14600, 3600, 10000, 15200, 1100, 2700, 10600)


@dataclass(frozen=True, slots=True)
class Pulse(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-staged-pulse-response/pulse'
    rate_kg_s: D
    duration_s: int

    def __post_init__(self) -> None:
        if (
            type(self.rate_kg_s) is not D
            or type(self.duration_s) is not int
            or not (
                (self.rate_kg_s == 0 and self.duration_s == 0)
                or (self.rate_kg_s in (*BASE_RATES, *ADDED_RATES) and self.duration_s in (10, 30))
            )
        ):
            raise ValueError("pulse is outside the declared staged-pulse finite menu")

    @property
    def word_id(self) -> str:
        return f"f{int(self.rate_kg_s * 1000):03d}-d{self.duration_s:03d}"

    def rates(self, guard_s: int = 120) -> tuple[D, ...]:
        if guard_s not in (120, 240):
            raise ValueError("undeclared staged-pulse guard")
        return tuple(self.rate_kg_s if t < self.duration_s else D(0) for t in range(0, guard_s, 10))


ZERO = Pulse(D(0), 0)
FIRST = Pulse(D(".032"), 10)
BASE = tuple(Pulse(r, d) for d in (10, 30) for r in BASE_RATES)
ADDED = tuple(Pulse(r, d) for d in (10, 30) for r in ADDED_RATES)
EXPANDED = tuple(sorted((*BASE, *ADDED), key=lambda w: (w.duration_s, w.rate_kg_s)))
SECONDS = tuple(Pulse(r, 10) for r in BASE_RATES)


@dataclass(frozen=True, slots=True)
class Coordinate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-staged-pulse-response/coordinate'
    kind: str
    context: str
    pulse: Pulse
    horizon_s: int

    def __post_init__(self) -> None:
        local = (
            self.kind == "local"
            and self.context in CONTEXTS
            and self.pulse in EXPANDED
            and self.horizon_s in (10, 30)
            and self.pulse.duration_s <= self.horizon_s
        )
        first = (
            self.kind in ("first", "baseline")
            and self.context == "early"
            and self.pulse == FIRST
            and self.horizon_s == 10
        )
        joint = (
            self.kind == "joint"
            and self.context == "induced"
            and self.pulse in SECONDS
            and self.horizon_s == 10
        )
        if type(self.horizon_s) is not int or not (local or first or joint):
            raise ValueError("coordinate changes its declared action/history/receiver")

    @property
    def coordinate_id(self) -> str:
        return f"{self.kind}.{self.context}.{self.pulse.word_id}.t{self.horizon_s:03d}"

    @property
    def episode_guard_s(self) -> int:
        return 240 if self.kind in ("baseline", "joint") else 120

    @property
    def epsilon_K(self) -> D:
        return D(".000001") * self.horizon_s / 10


LOCAL_BASE = tuple(
    Coordinate("local", c, w, t)
    for c in CONTEXTS
    for t in (10, 30)
    for w in BASE
    if w.duration_s <= t
)
LOCAL_EXPANDED = tuple(
    Coordinate("local", c, w, t)
    for c in CONTEXTS
    for t in (10, 30)
    for w in EXPANDED
    if w.duration_s <= t
)
JOINT = (
    Coordinate("first", "early", FIRST, 10),
    Coordinate("baseline", "early", FIRST, 10),
    *(Coordinate("joint", "induced", w, 10) for w in SECONDS),
)


@dataclass(frozen=True, slots=True)
class Request(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-staged-pulse-response/request'
    request_id: str
    context: str
    horizon_s: int
    required_K: D
    budget_kg: D
    second_K: D | None = None

    def __post_init__(self) -> None:
        expected = {
            "EARLY_TEN_SECOND_REQUEST": ("early", 10, D(".0012"), D(".32"), None),
            "EARLY_THIRTY_SECOND_REQUEST": ("early", 30, D(".0032"), D("1.28"), None),
            "PREPARED_TEN_SECOND_REQUEST": ("prepared_t0", 10, D(".0012"), D(".32"), None),
            "PREPARED_THIRTY_SECOND_REQUEST": ("prepared_t0", 30, D(".0032"), D("1.28"), None),
            "LOW": ("induced", 10, D(".0012"), D(".32"), D(".0004")),
            "HIGH": ("induced", 10, D(".0012"), D(".80"), D(".0012")),
        }
        if expected.get(self.request_id) != (
            self.context,
            self.horizon_s,
            self.required_K,
            self.budget_kg,
            self.second_K,
        ):
            raise ValueError("request changes its externally frozen staged-pulse task")


LOCAL_REQUESTS = tuple(
    Request(name, context, horizon, D(benefit), D(budget))
    for name, context, horizon, benefit, budget in (
        ("EARLY_TEN_SECOND_REQUEST", "early", 10, ".0012", ".32"),
        ("EARLY_THIRTY_SECOND_REQUEST", "early", 30, ".0032", "1.28"),
        ("PREPARED_TEN_SECOND_REQUEST", "prepared_t0", 10, ".0012", ".32"),
        ("PREPARED_THIRTY_SECOND_REQUEST", "prepared_t0", 30, ".0032", "1.28"),
    )
)
PAIR_REQUESTS = (
    Request("LOW", "induced", 10, D(".0012"), D(".32"), D(".0004")),
    Request("HIGH", "induced", 10, D(".0012"), D(".80"), D(".0012")),
)


@dataclass(frozen=True, slots=True)
class ClassicalDesign(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-staged-pulse-response/classical-design'
    config_id: str = PREFIX
    roots: tuple[tuple[str, str, str, int], ...] = ROOTS
    local_requests: tuple[Request, ...] = LOCAL_REQUESTS
    pair_requests: tuple[Request, ...] = PAIR_REQUESTS
    gammas: tuple[D, ...] = GAMMAS
    ladder: tuple[D, ...] = LADDER
    first: Pulse = FIRST
    native_call_caps: tuple[int, ...] = CALL_CAPS
    cpu_seconds: tuple[int, ...] = CPU_SECONDS
    wall_seconds: tuple[int, ...] = WALL_SECONDS
    minimum_contact: int = 29
    outward_fraction: D = D(".10")
    temperature_limit_K: D = D("356.2")
    peak_tolerance_K: D = D(".01")
    mass_tolerance_kg: D = D("1e-12")
    budget_tolerance_kg: D = D("1e-10")
    minimum_qualification_successes: int = 95
    qualification_probability: D = D(".90")
    minimum_prospective_evaluation_successes: int = 56
    prospective_evaluation_probability: D = D(".70")
    risk_ceiling: D = D(".10")
    bootstrap_draws: int = 20000
    bootstrap_seeds: tuple[int, ...] = (20260927, 20260928, 20260929)

    def __post_init__(self) -> None:
        for name, field in self.__dataclass_fields__.items():
            if name != "SCHEMA" and (
                getattr(self, name) != field.default
                or type(getattr(self, name)) is not type(field.default)
            ):
                raise ValueError(f"staged-pulse design changed its predeclared {name}")


def roots(block: str, role: str) -> tuple[str, ...]:
    if block not in ("base-menu-comparison", "expanded-menu-comparison", "staged-sequence-comparison") or role not in ("qualification", "prospective"):
        raise ValueError("undeclared staged-pulse block or independent-root role")
    return tuple(root for root, b, r, _ in ROOTS if (b, r) == (block, role))


def assignment(root: str) -> tuple[str, str, int]:
    if root in DEVELOPMENT_ROOTS:
        return "historical", "development", 98000 + DEVELOPMENT_ROOTS.index(root)
    if root in CALIBRATION_ROOTS:
        return "historical", "calibration", 98700 + CALIBRATION_ROOTS.index(root)
    for name, block, role, seed in ROOTS:
        if name == root:
            return block, role, seed
    raise ValueError("unassigned reactor staged-pulse physical root")


@dataclass(frozen=True, slots=True)
class ClassicalUpstream(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-staged-pulse-response/classical-upstream'
    key: str
    artifact: ArtifactIdentity
    task_receipt: ObjectIdentity
    source_run_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.key, field_name="key")
        validate_stable_id(self.source_run_id, field_name="source_run_id")
        if self.task_receipt.object_schema != 'empirical-lawhood/runtime/canonical-task-receipt':
            raise ValueError("upstream input requires an authenticated task receipt")


@dataclass(frozen=True, slots=True)
class ClassicalStage(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-staged-pulse-response/classical-stage'
    design: ClassicalDesign
    stage: str
    upstream: tuple[ClassicalUpstream, ...]

    def __post_init__(self) -> None:
        if self.stage not in STAGES or len({u.key for u in self.upstream}) != len(self.upstream):
            raise ValueError("stage changes the declared progression or repeats a parent")
        if self.stage == "base-menu-comparison-NOMINATION":
            expected = (
                "historical-b",
                *(
                    f"calibration-{i:03d}-{role}"
                    for i in range(96)
                    for role in ("measurement", "prediction")
                ),
            )
        elif self.stage in ("expanded-menu-comparison-NOMINATION", "staged-sequence-comparison-NOMINATION"):
            roles = (
                ("assay", "preparation", "private")
                if self.stage == "expanded-menu-comparison-NOMINATION"
                else ("measurement", "preparation", "private")
            )
            expected = (
                "historical-b",
                *(f"development-{i:03d}-{role}" for i in range(32) for role in roles),
            )
        else:
            expected = ("nomination",) if self.role == "QUALIFICATION" else ("laws", "nomination")
        expected = tuple(sorted(expected))
        if tuple(u.key for u in self.upstream) != expected:
            raise ValueError("stage does not bind its exact immutable upstream roles")

    @property
    def config_id(self) -> str:
        return f"{PREFIX}.{self.stage.lower()}"

    @property
    def block(self) -> str:
        return self.stage.rsplit("-", 1)[0]

    @property
    def role(self) -> str:
        return self.stage.rsplit("-", 1)[1]

    @property
    def root_ids(self) -> tuple[str, ...]:
        return (
            DEVELOPMENT_ROOTS
            if self.role == "NOMINATION"
            else roots(self.block, "qualification" if self.role == "QUALIFICATION" else "prospective")
        )

    @property
    def coordinates(self) -> tuple[Coordinate, ...]:
        return {"base-menu-comparison": LOCAL_BASE, "expanded-menu-comparison": LOCAL_EXPANDED, "staged-sequence-comparison": JOINT}[self.block]

    @property
    def policies(self) -> tuple[str, ...]:
        return {"base-menu-comparison": ARMS, "expanded-menu-comparison": MENUS, "staged-sequence-comparison": SEQUENCES}[self.block]

    @property
    def requests(self) -> tuple[Request, ...]:
        return PAIR_REQUESTS if self.block == "staged-sequence-comparison" else LOCAL_REQUESTS

    @property
    def cpu_seconds(self) -> int:
        return self.design.cpu_seconds[STAGES.index(self.stage)]

    @property
    def wall_seconds(self) -> int:
        return self.design.wall_seconds[STAGES.index(self.stage)]

    @property
    def native_call_cap(self) -> int:
        return self.design.native_call_caps[STAGES.index(self.stage)]
