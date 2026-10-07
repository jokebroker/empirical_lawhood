"Finite native declarations, distinct from exposed matrix preparation."

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.adapters.simulators.six_matrix_response.contracts import BetaCouplingRule, SixMatrixResponseModelFamilyMember, SixMatrixResponseNumericalView, MatrixIntegratorKind, MatrixPrecision


CAMPAIGN = "prepared-response"
CONTEXTS = ("assembling", "prepared")
PARENTS = ("hold", "y-negative-128", "y-positive-128", "y-negative-256", "y-positive-256")
READOUTS = (64, 128, 192, 256, 320)
DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))
STAGE_COUNTS = {
    "qualification": 16,
    "development": 64,
    "calibration": 96,
    "conditional-risk-calibration": 96,
    "prospective-evaluation": 128,
    "policy-reuse-evaluation": 96,
    "policy-reuse-qualification": 96,
}
RNG_RULE = f"{CAMPAIGN}.explicit-numeric-stream-commitment"
PREPARED_STREAM_ROLES = (
    ("initial-ramp", False), ("initial-ramp", True),
    ("initial-post-ramp", False), ("initial-post-ramp", True),
    ("parent", False), ("parent", True),
    ("common-response", False), ("common-response", True),
    ("independent-response-1", False), ("independent-response-1", True),
    ("independent-response-2", False), ("independent-response-2", True),
    ("independent-response-3", False), ("independent-response-3", True),
    ("prospective-task", False), ("prospective-task", True),
    ("passive-probes", False),
)


def _root_count(stage: str) -> int:
    return 2 if stage == "excluded-canary" else STAGE_COUNTS.get(stage, 0)


def _frozen_seed_values(
    stage: str, seed_sha256: str, context: str, index: int
) -> tuple[str, ...] | None:
    from .seed_commitments import FROZEN_SEED_COMMITMENTS

    rows = FROZEN_SEED_COMMITMENTS.get(seed_sha256, {}).get(stage)
    return None if rows is None else rows[CONTEXTS.index(context) * _root_count(stage) + index]


@dataclass(frozen=True, slots=True)
class PreparedRootRandomness(CanonicalRecord):
    """Complete numeric allocation supplied before a native root is acquired.

    Stream tuple positions follow PREPARED_STREAM_ROLES. The final digest owns
    the separate twelve-field probe draw, which has no further label hash.
    """

    SCHEMA: ClassVar[str] = "empirical-lawhood/simulators/prepared-response/prepared-root-randomness"
    stage: str
    context: str
    index: int
    source_seed_sha256: str
    stream_seed_sha256s: tuple[str, ...]
    passive_probe_seed_sha256: str

    def __post_init__(self) -> None:
        validate_sha256(self.source_seed_sha256, field_name="source_seed_sha256")
        if (
            self.context not in CONTEXTS
            or type(self.index) is not int
            or not 0 <= self.index < _root_count(self.stage)
            or not isinstance(self.stream_seed_sha256s, tuple)
            or len(self.stream_seed_sha256s) != len(PREPARED_STREAM_ROLES)
        ):
            raise ValueError("prepared randomness requires the complete ordered root allocation")
        values = (*self.stream_seed_sha256s, self.passive_probe_seed_sha256)
        for value in values:
            validate_sha256(value, field_name="scientific_seed_sha256")
        if len({value[:32] for value in values}) != len(values):
            raise ValueError("prepared root repeats a numeric PCG allocation")
        frozen = _frozen_seed_values(self.stage, self.source_seed_sha256, self.context, self.index)
        if frozen is not None and values != frozen:
            raise ValueError("prepared frozen scientific input changes its numeric commitments")

    def stream_seed(self, purpose: str, bridge: bool) -> str:
        if type(bridge) is not bool or (purpose, bridge) not in PREPARED_STREAM_ROLES:
            raise ValueError("prepared randomness purpose is outside its exact allocation")
        return self.stream_seed_sha256s[PREPARED_STREAM_ROLES.index((purpose, bridge))]


def _default_root_randomness(
    stage: str, seed_sha256: str, context: str, index: int
) -> PreparedRootRandomness:
    values = _frozen_seed_values(stage, seed_sha256, context, index)
    if values is None:
        raise ValueError("prepared caller input requires explicit numeric root commitments")
    return PreparedRootRandomness(stage, context, index, seed_sha256, values[:17], values[17])


def validate_prepared_seed_census(
    stage: str, seed_sha256: str, census: tuple[PreparedRootRandomness, ...]
) -> tuple[PreparedRootRandomness, ...]:
    """Require the complete declaration in physical root acquisition order."""
    validate_sha256(seed_sha256, field_name="source_seed_sha256")
    count = _root_count(stage)
    if not count or not isinstance(census, tuple):
        raise ValueError("prepared source requires a typed numeric root census")
    coordinates = tuple((context, index) for context in CONTEXTS for index in range(count))
    if not census:
        census = tuple(_default_root_randomness(stage, seed_sha256, c, i) for c, i in coordinates)
    if len(census) != len(coordinates) or any(
        not isinstance(row, PreparedRootRandomness)
        or (row.stage, row.source_seed_sha256, row.context, row.index)
        != (stage, seed_sha256, context, index)
        for row, (context, index) in zip(census, coordinates, strict=True)
    ):
        raise ValueError("prepared numeric census changes its stage/root acquisition order")
    allocations = tuple(
        value[:32] for row in census
        for value in (*row.stream_seed_sha256s, row.passive_probe_seed_sha256)
    )
    if len(set(allocations)) != len(allocations):
        raise ValueError("prepared numeric census repeats a PCG allocation across roots")
    return census


def prepared_native_member() -> SixMatrixResponseModelFamilyMember:
    return SixMatrixResponseModelFamilyMember(
        f"{CAMPAIGN}.member",
        Decimal("0.5"),
        Decimal("0.5"),
        Decimal(1),
        BetaCouplingRule.PUBLISHED_DETERMINISTIC,
    )


def prepared_numerical_view(refinement: int) -> SixMatrixResponseNumericalView:
    if type(refinement) is not int or refinement not in (1, 2):
        raise ValueError("prepared source has exactly primary and half numerical views")
    return SixMatrixResponseNumericalView(
        f"{CAMPAIGN}.view.r{refinement}",
        MatrixIntegratorKind.BAOAB_UNDERDAMPED_LANGEVIN,
        Decimal("0.001") / refinement,
        "dimensionless-langevin-time",
        Decimal(1),
        Decimal(1),
        MatrixPrecision.COMPLEX128,
        "numpy.pcg64dxsm",
        "1.0.0",
        RNG_RULE,
        512 * refinement,
    )


@dataclass(frozen=True, slots=True)
class PreparedNativeSpec(CanonicalRecord):
    """Mechanical source contract. Preservation readiness is a separate binding."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/prepared-response/prepared-native-spec'
    stage: str
    seed_sha256: str
    implementation_plan_sha256: str
    dependency_lock_sha256: str
    native_implementation: ObjectIdentity
    member: SixMatrixResponseModelFamilyMember
    numerical_views: tuple[SixMatrixResponseNumericalView, ...]
    selected_amplitude: Decimal | None
    root_seed_census: tuple[PreparedRootRandomness, ...] = ()
    source_initialization: str = "INHERITED_IDEAL_00_OR_11_LINEAR_256_RAMP_NO_POSTLANDMARK_SWEEP"
    kinetic_mass: Decimal = Decimal(1)
    parent_end_ticks: int = 256
    handoff_delay_ticks: int = 16
    pulse_ticks: int = 64
    readouts: tuple[int, ...] = READOUTS
    observation_cadence_ticks: int = 16
    source_window_samples: int = 31
    port_rule: str = "PRIMARY_PREPARENT_ORIENTED_PCA_AND_ORTHOGONAL_RESIDUAL_PCA"
    numerical_noise_rule: str = "ISOTROPIC_HERMITIAN_HS_OU_BROWNIAN_BRIDGE"
    independent_unit: str = "FRESH_STAGE_CONTEXT_STOCHASTIC_ROOT"
    grants_authority: bool = False

    def __post_init__(self) -> None:
        for name in ("seed_sha256", "implementation_plan_sha256", "dependency_lock_sha256"):
            validate_sha256(getattr(self, name), field_name=name)
        if self.stage not in (*STAGE_COUNTS, 'excluded-canary'):
            raise ValueError("prepared native source stage is outside its fixed census")
        if self.stage in ('qualification', 'excluded-canary'):
            if self.selected_amplitude is not None:
                raise ValueError("source qualification/excluded canary must retain all seventeen qualification words")
        elif not isinstance(self.selected_amplitude, Decimal) or self.selected_amplitude not in (
            8,
            16,
        ):
            raise ValueError("post-source qualification native stage requires the frozen nominated amplitude")
        expected_ints = {
            "parent_end_ticks": 256,
            "handoff_delay_ticks": 16,
            "pulse_ticks": 64,
            "observation_cadence_ticks": 16,
            "source_window_samples": 31,
        }
        for name, expected in expected_ints.items():
            value = getattr(self, name)
            if type(value) is not int or value != expected:
                raise ValueError(f"prepared native source changes {name}")
        if (
            self.member != prepared_native_member()
            or self.numerical_views != tuple(prepared_numerical_view(r) for r in (1, 2))
            or self.source_initialization
            != "INHERITED_IDEAL_00_OR_11_LINEAR_256_RAMP_NO_POSTLANDMARK_SWEEP"
            or not isinstance(self.kinetic_mass, Decimal)
            or self.kinetic_mass != 1
            or self.readouts != READOUTS
            or any(type(v) is not int for v in self.readouts)
            or self.port_rule != "PRIMARY_PREPARENT_ORIENTED_PCA_AND_ORTHOGONAL_RESIDUAL_PCA"
            or self.numerical_noise_rule != "ISOTROPIC_HERMITIAN_HS_OU_BROWNIAN_BRIDGE"
            or self.independent_unit != "FRESH_STAGE_CONTEXT_STOCHASTIC_ROOT"
            or self.grants_authority is not False
        ):
            raise ValueError("prepared native source changes its exact mechanical denominator")
        object.__setattr__(self, "root_seed_census", validate_prepared_seed_census(
            self.stage, self.seed_sha256, self.root_seed_census
        ))

    @property
    def spec_id(self) -> str:
        return f"{CAMPAIGN}.{self.stage}.native-spec"

    @property
    def roots(self) -> tuple['PreparedRoot', ...]:
        return tuple(
            PreparedRoot(self.stage, row.context, row.index, self.seed_sha256, row)
            for row in self.root_seed_census
        )

    @property
    def words(self) -> tuple['PreparedForceWord', ...]:
        return prepared_words(self.selected_amplitude)


@dataclass(frozen=True, slots=True)
class PreparedRoot(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/simulators/prepared-response/prepared-root"
    stage: str
    context: str
    index: int
    seed_sha256: str
    randomness: PreparedRootRandomness | None = None

    def __post_init__(self) -> None:
        validate_sha256(self.seed_sha256, field_name="seed_sha256")
        count = 2 if self.stage == 'excluded-canary' else STAGE_COUNTS.get(self.stage, 0)
        if (
            self.context not in CONTEXTS
            or type(self.index) is not int
            or not 0 <= self.index < count
        ):
            raise ValueError("prepared root is outside its fixed fresh-stage census")
        randomness = self.randomness
        if randomness is None:
            randomness = _default_root_randomness(self.stage, self.seed_sha256, self.context, self.index)
        if not isinstance(randomness, PreparedRootRandomness) or (
            randomness.stage, randomness.context, randomness.index, randomness.source_seed_sha256
        ) != (self.stage, self.context, self.index, self.seed_sha256):
            raise ValueError("prepared root differs from its explicit numeric allocation")
        object.__setattr__(self, "randomness", randomness)

    @property
    def root_id(self) -> str:
        return f"{CAMPAIGN}.{self.stage}.{self.context}.r{self.index:03d}"

    @property
    def physical_unit_id(self) -> str:
        return f"{self.root_id}.seed.{self.seed_sha256}"

    @property
    def landmark(self) -> int:
        return 1024 if self.context == "assembling" else 4096

    @property
    def handoff(self) -> int:
        return self.landmark + 272

    @property
    def problem_index(self) -> int | None:
        return self.index // 16 if self.stage in ('policy-reuse-evaluation', 'policy-reuse-qualification') else None


@dataclass(frozen=True, slots=True)
class PreparedForceWord(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/prepared-response/prepared-force-word'
    magnitude: Decimal
    direction_index: int
    sign: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.magnitude, Decimal)
            or not self.magnitude.is_finite()
            or type(self.direction_index) is not int
            or type(self.sign) is not int
            or (self.magnitude, self.direction_index, self.sign) != (Decimal(0), 0, 0)
            and not (
                self.magnitude in (8, 16)
                and self.direction_index in range(4)
                and self.sign in (-1, 1)
            )
        ):
            raise ValueError("prepared word changes the seventeen-word native qualification chart")

    @property
    def word_id(self) -> str:
        if self.sign == 0:
            return f"{CAMPAIGN}.word.hold"
        polarity = "negative" if self.sign < 0 else "positive"
        return f"{CAMPAIGN}.word.a{int(self.magnitude)}.d{self.direction_index}.{polarity}"


def prepared_words(amplitude: Decimal | None = None) -> tuple[PreparedForceWord, ...]:
    if amplitude is not None and (not isinstance(amplitude, Decimal) or amplitude not in (8, 16)):
        raise ValueError("prepared amplitude must be the source qualification-selected 8 or 16")
    return (PreparedForceWord(Decimal(0), 0, 0),) + tuple(
        PreparedForceWord(value, direction, sign)
        for value in ((Decimal(8), Decimal(16)) if amplitude is None else (amplitude,))
        for direction in range(4)
        for sign in (-1, 1)
    )


def validate_prepared_future_role(stage: str, word: PreparedForceWord, purpose: str) -> None:
    """One owner for the predeclared per-stage future-purpose restriction."""
    if (
        stage not in (*STAGE_COUNTS, 'excluded-canary')
        or purpose not in ('common-response', 'independent-response-1', 'independent-response-2', 'independent-response-3', 'prospective-task')
        or stage in ('qualification', 'calibration', 'policy-reuse-qualification')
        and purpose != 'common-response'
        or stage == 'development'
        and (purpose == 'prospective-task' or purpose != 'common-response' and word.sign != 0)
        or stage in ('conditional-risk-calibration', 'prospective-evaluation')
        and purpose not in ('common-response', 'prospective-task')
        or stage == 'policy-reuse-evaluation'
        and purpose != 'prospective-task'
    ):
        raise ValueError("prepared future role changes its stage's declared native acquisition menu")
