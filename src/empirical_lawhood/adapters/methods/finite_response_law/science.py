"""The finite response law science freeze and deterministic request construction."""

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256
from .seed_commitments import FIXED_PASSIVE_PROBE_SEEDS, FIXED_SEEDS

PROGRAMME = "finite-response-law"
PLAN_SHA256 = "90baf62da60fd3f925f9212429e5b724184d6ea29da33a4da6d5bc42aa825d5f"
PARENTS = (
    "hold",
    "y-negative-128",
    "y-positive-128",
    "y-negative-256",
    "y-positive-256",
)
STAGES = (
    ("retained-development", 48),
    ("canary", 2),
    ("supplemental-development", 16),
    ("calibration", 32),
    ("prospective-evaluation", 64),
    ("preparation-screening", 24),
    ("preparation-calibration", 32),
    ("preparation-evaluation", 96),
)
RNG_PURPOSES = (
    "development-request",
    "calibration-request",
    "prospective-evaluation-request",
    "preparation-calibration-request",
    "preparation-evaluation-request",
    "parent-allocation",
    "fold",
    "bootstrap",
    "prefix",
    "parent",
    "future-1",
    "future-2",
)


@dataclass(frozen=True, slots=True)
class FiniteResponseLawScienceSpec(CanonicalRecord):
    """Bind the packaged public protocol, plus the operands consumed by specification binding.

    Later executable recipes derive from the same public protocol. They are not
    free parameters merely because their codecs have not yet been implemented.
    """

    SCHEMA: ClassVar[str] = (
        "empirical-lawhood/methods/finite-response-law/finite-response-law-science-spec"
    )
    plan_sha256: str = PLAN_SHA256
    master_seed: int = 20260914
    request_pairs_per_root: int = 256
    lower_ranges: tuple[tuple[Decimal, Decimal], ...] = (
        (Decimal("0.02"), Decimal("0.12")),
        (Decimal("0.02"), Decimal("0.06")),
    )
    upper: tuple[Decimal, ...] = (Decimal("0.16"), Decimal("0.10"))
    transverse: tuple[Decimal, ...] = (Decimal("0.02"), Decimal("0.01"))
    preservation: tuple[Decimal, ...] = (
        Decimal("0.125"),
        Decimal("0.05"),
        Decimal("0.05"),
    )
    delta: tuple[Decimal, ...] = tuple(
        map(
            Decimal,
            (
                "0.01",
                "0.01",
                "0.0125",
                "0.005",
                "0.005",
                "0.0125",
                "0.005",
                "0.005",
            ),
        )
    )
    stage_root_counts: tuple[tuple[str, int], ...] = STAGES
    joint_opportunity_minimum: Decimal = Decimal("0.80")
    magnitude_discrimination_minimum: Decimal = Decimal("0.20")
    parent_work_maximum: Decimal = Decimal(32)
    horizon_ticks: int = 192

    def __post_init__(self) -> None:
        validate_sha256(self.plan_sha256, field_name="plan_sha256")
        # A closed nomination: construction cannot silently retune the benchmark.
        for name, field in self.__dataclass_fields__.items():
            if name != "SCHEMA" and getattr(self, name) != field.default:
                raise ValueError(
                    f"Finite response law science differs from the adopted nomination: {name}"
                )
        if (
            type(self.master_seed) is not int
            or type(self.request_pairs_per_root) is not int
        ):
            raise ValueError("integer seed and request census required")


def seed_for(purpose: str, unit: str, *, committed_seed: int | None = None) -> int:
    """Return a fixed scientific operand or an explicitly supplied caller seed.

    Public labels never determine numerical draws. Assigned cohorts must supply
    their complete outcome-blind seed census; renaming a unit grants no receipt.
    """
    if purpose not in RNG_PURPOSES or not unit or len(unit) > 256:
        raise ValueError("undeclared RNG purpose or invalid unit")
    fixed = FIXED_SEEDS.get(unit, {}).get(purpose)
    if committed_seed is None:
        if fixed is None:
            raise ValueError("assigned scientific stream requires an explicit seed commitment")
        return fixed
    if type(committed_seed) is not int or not 0 <= committed_seed < 2**128:
        raise ValueError("scientific seed commitment must be a 128-bit unsigned integer")
    if fixed is not None and fixed != committed_seed:
        raise ValueError("scientific seed commitment differs from the fixed roster")
    return committed_seed


def passive_probe_seed_for(unit: str, *, committed_seed: int | None = None) -> int:
    """The probe allocation is an explicit 256-bit scientific RNG operand."""
    fixed = FIXED_PASSIVE_PROBE_SEEDS.get(unit)
    if committed_seed is None:
        if fixed is None:
            raise ValueError("assigned passive probes require an explicit seed commitment")
        return fixed
    if type(committed_seed) is not int or not 0 <= committed_seed < 2**256:
        raise ValueError("passive probe commitment must be a 256-bit unsigned integer")
    if fixed is not None and fixed != committed_seed:
        raise ValueError("passive probe commitment differs from the fixed roster")
    return committed_seed


def validate_assigned_seeds(stage: str, rows: tuple[tuple[str, int, int], ...]) -> None:
    """Bind every native, parent, request and probe allocation before outcomes."""
    counts = {"calibration": 32, "prospective-evaluation": 64}
    if stage not in counts or not isinstance(rows, tuple):
        raise ValueError("assigned scientific seed census differs from its stage")
    expected = {
        (purpose, index)
        for index in range(counts[stage])
        for purpose in (*NATIVE_SEED_PURPOSES, "parent-allocation", f"{stage}-request", "passive-probes")
    }
    if stage == "prospective-evaluation":
        expected.add(("bootstrap", -1))
    if (
        any(not isinstance(row, tuple) or len(row) != 3 for row in rows)
        or rows != tuple(sorted(rows))
        or len(rows) != len(expected)
        or {(p, i) for p, i, _ in rows} != expected
        or any(type(i) is not int or type(s) is not int or not 0 <= s < 2**(256 if p == "passive-probes" else 128) for p, i, s in rows)
        or len({s for p, _, s in rows if p != "passive-probes"}) != sum(p != "passive-probes" for p, _, _ in rows)
    ):
        raise ValueError("assigned scientific seed census is incomplete, duplicated or invalid")


NATIVE_SEED_PURPOSES = ("prefix", "parent", "future-1", "future-2")


def root_seed(root: object, purpose: str) -> int:
    """Read the seed committed by a typed root, falling back to the fixed census."""
    rows = dict(getattr(root, "scientific_seeds", ()))
    return seed_for(purpose, getattr(root, "stage_unit"), committed_seed=rows.get(purpose))


def validate_root_seeds(stage: str, rows: tuple[tuple[str, int], ...]) -> None:
    expected = {*NATIVE_SEED_PURPOSES, "parent-allocation", f"{stage}-request", "passive-probes"}
    if stage == "prospective-evaluation":
        expected.add("bootstrap")
    if (
        not isinstance(rows, tuple)
        or any(not isinstance(row, tuple) or len(row) != 2 for row in rows)
        or rows != tuple(sorted(rows))
        or len(rows) != len(expected)
        or {p for p, _ in rows} != expected
        or any(type(s) is not int or not 0 <= s < 2**(256 if p == "passive-probes" else 128) for p, s in rows)
    ):
        raise ValueError("assigned root scientific seed census is incomplete or invalid")


def development_requests(
    spec: FiniteResponseLawScienceSpec,
) -> dict[str, NDArray[np.generic]]:
    """Root, request pair, consumer; orientations +e1,-e1,+e2,-e2."""
    units = tuple(
        f"{cohort}.prepared.r{r:03d}"
        for cohort, n in (("prepared-response", 16), ("information-response-prediction", 32))
        for r in range(n)
    )
    direction = np.empty((48, spec.request_pairs_per_root, 2), dtype=np.int64)
    lower = np.empty(direction.shape, dtype=np.float64)
    for r, unit in enumerate(units):
        rng = np.random.Generator(
            np.random.PCG64(seed_for("development-request", unit))
        )
        direction[r] = rng.integers(0, 4, size=(spec.request_pairs_per_root, 2))
        for consumer, (low, high) in enumerate(spec.lower_ranges):
            lower[r, :, consumer] = rng.uniform(
                float(low), float(high), spec.request_pairs_per_root
            )
    return {"root_ids": np.asarray(units), "direction": direction, "lower": lower}
