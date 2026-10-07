"""Strict unissued recipe and fixed comparative allocation; no verdict inputs."""

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.adapters.simulators.reactor_prefix_response.contracts import UPSTREAM_COMMIT, PLANT_SHA256, PARAMS_SHA256

ARMS = ("EL", "F0", "F1", "MARGIN", "FIXED", "REF", "EKF", "SCHEDULED_BACKOFF_ZERO", "SCHEDULED_BACKOFF_HALF")
NATIVE_BENCHMARK_ARMS = ("EL", "REF", "EKF", "SCHEDULED_BACKOFF_ZERO", "SCHEDULED_BACKOFF_HALF")
CONFIRMATION_ROOT_COUNT = 8
CONFIRMATION_ROOTS = tuple(
    f"reactor-empirical-confirmation-{i:03d}" for i in range(CONFIRMATION_ROOT_COUNT)
)
COMPARATIVE_SCOPE = "DESCRIPTIVE_EFFECT_ESTIMATION_AND_MECHANISM_DIAGNOSTICS"
ROLES = (
    ("fit", 16, 81000),
    ("nomination", 8, 82000),
    ("calibration", 32, 83000),
    ("qualification", 32, 84000),
    ("confirmation", CONFIRMATION_ROOT_COUNT, 85000),
)

COMPARISONS = (
    tuple(
        ("EL", arm, endpoint)
        for endpoint in ("J", "unsafe-or-unobserved", "restricted-completion-time")
        for arm in ARMS[1:]
    )
    + tuple(
        ("EL", arm, endpoint)
        for arm in ("F0", "F1")
        for endpoint in ("temperature-mae", "conversion-mae")
    )
    + tuple(("EL", arm, "log-online-cpu-ratio") for arm in ("REF", "EKF", "SCHEDULED_BACKOFF_ZERO", "SCHEDULED_BACKOFF_HALF"))
)
COMPARATOR_SOURCES = (
    (
        "REF",
        "solution/controller.py",
        "f19d7a1ce9e70c654eefe31a4adfcf2dcb4241dfccddadbe5b6479649ed1efe8",
    ),
    (
        "EKF",
        "authoring/evidence/independent_controller.py",
        "820b0782b409480099e80e29c041a9e44de43be47ff5bea0ee6f459060c497c6",
    ),
    (
        "SCHEDULED_BACKOFF_ZERO",
        "authoring/evidence/fixed_schedule_baseline.py",
        "133a14605e7ae06b6f451aaf29d0c09dfddd90a5ebf4d8683d222350ccfd0718",
    ),
    (
        "SCHEDULED_BACKOFF_HALF",
        "authoring/evidence/fixed_schedule_baseline.py",
        "133a14605e7ae06b6f451aaf29d0c09dfddd90a5ebf4d8683d222350ccfd0718",
    ),
)


@dataclass(frozen=True, slots=True)
class EmpiricalRecipe(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-causal-response/empirical-recipe'
    config_id: str = "reactor-empirical-eight-root"
    source_commit: str = UPSTREAM_COMMIT
    plant_sha256: str = PLANT_SHA256
    params_sha256: str = PARAMS_SHA256
    roles: tuple[tuple[str, int, int], ...] = ROLES
    families: tuple[int, ...] = (7, 13, 23)
    penalties: tuple[D, ...] = (D(".000001"), D(".001"), D(".1"))
    arms: tuple[str, ...] = ARMS
    native_benchmark_arms: tuple[str, ...] = NATIVE_BENCHMARK_ARMS
    comparative_scope: str = COMPARATIVE_SCOPE
    confirmatory_superiority_claims: bool = False
    scheduled_backoffs: tuple[D, ...] = (D(0), D(".5"))
    callback_count: int = 2880
    numerical_tolerance: tuple[D, ...] = (D(".01"), D(".0002"), D(".000001"))
    scales: tuple[D, ...] = (D(".25"), D(".01"))
    calibration_alpha: D = D(".05")
    comparisons: tuple[tuple[str, str, str], ...] = COMPARISONS
    comparator_sources: tuple[tuple[str, str, str], ...] = COMPARATOR_SOURCES
    effect_resolutions: tuple[D, ...] = (D(".01"), D(".001"), D(120), D(".10"))
    comparison_count: int = 32
    binary_alpha: D = D(".00078125")
    bootstrap_seed: int = 86104
    bootstrap_draws: int = 100000
    branch_seed: int = 86101
    branch_draws: int = 10000
    maximum_batches: int = 733
    maximum_artifact_bytes: int = 256 * 1024**3
    maximum_worker_bytes: int = 8 * 1024**3
    maximum_workers: int = 4
    numerical_threads: int = 1
    aggregate_cpu_seconds: int = 36 * 3600
    wall_seconds: int = 12 * 3600

    def __post_init__(self) -> None:
        # Compare to declared dataclass defaults without recursive construction.
        for name, field in self.__dataclass_fields__.items():
            if name != "SCHEMA" and (
                getattr(self, name) != field.default
                or type(getattr(self, name)) is not type(field.default)
            ):
                raise ValueError(f"unamended empirical recipe: {name}")


def decode_recipe(data: bytes) -> EmpiricalRecipe:
    return decode_canonical_bytes(data, EmpiricalRecipe, maximum_bytes=32768)


def arm_order(root: int) -> tuple[str, ...]:
    if type(root) is not int or root not in range(CONFIRMATION_ROOT_COUNT):
        raise ValueError("undeclared confirmation root")
    shift = root % 9
    return ARMS[shift:] + ARMS[:shift]
