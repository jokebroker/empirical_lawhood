"""Complete numerical seed commitments for the fixed source-assessment roots.

The full digests are scientific inputs. Public identity changes do not allocate
new streams, and the original digest remains available for source custody.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256


@dataclass(frozen=True, slots=True)
class GymToraxSourceAssessmentSeed(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        "empirical-lawhood/simulators/gym-torax-native/source-assessment-scientific-seed"
    )
    VERSION: ClassVar[str] = "1.0.0"
    role: str
    counter: int
    cell_kind: str
    full_seed_sha256: str

    def __post_init__(self) -> None:
        validate_sha256(self.full_seed_sha256, field_name="full_seed_sha256")
        if type(self.counter) is not int or self.counter < 0:
            raise ValueError("scientific seed counter must be a nonnegative integer")
        if self.role == "excluded-metadata-canary":
            if self.counter != 0 or self.cell_kind != "native-canary":
                raise ValueError("metadata canary seed has another scientific role")
        elif self.role == "source-qualification":
            if self.counter not in (0, 1, 2) or self.cell_kind not in (
                "b",
                "bt",
                "c",
                "t",
            ):
                raise ValueError(
                    "source-qualification seed has another scientific role"
                )
        else:
            raise ValueError("scientific seed role is not supported")

    @property
    def environment_seed(self) -> int:
        return (int(self.full_seed_sha256[:16], 16) & (2**63 - 1)) or 1


_SOURCE_ASSESSMENT_SEEDS = (
    GymToraxSourceAssessmentSeed(
        "excluded-metadata-canary",
        0,
        "native-canary",
        "1a66b589fa283fd57833a310474a63ec55392d92db7c8478d3ab7d6e8e73c88f",
    ),
    GymToraxSourceAssessmentSeed(
        "source-qualification",
        0,
        "b",
        "f8f1d14ca269ad596d08694ac22640cae304e6c9a866c48ea018d04ba155d567",
    ),
    GymToraxSourceAssessmentSeed(
        "source-qualification",
        0,
        "bt",
        "e294b8a97d91b8a382b03d6b3a489d7d9ae3f06ad88aa4d0e6f86d017d662fbc",
    ),
    GymToraxSourceAssessmentSeed(
        "source-qualification",
        0,
        "c",
        "c1d048a6fb4ea121387e0b90387bcd51f2b3eee425b11c249fcb4cc31fdba4d5",
    ),
    GymToraxSourceAssessmentSeed(
        "source-qualification",
        0,
        "t",
        "16ddcffb9fac47b68bccd6264eee2f531823b53f6201a0159e26a026bbbed7fe",
    ),
    GymToraxSourceAssessmentSeed(
        "source-qualification",
        1,
        "b",
        "3a97c22382dca906714b36326e6a3fd4ac00cc6ae9cd0457e06ad1c55a2c5e3b",
    ),
    GymToraxSourceAssessmentSeed(
        "source-qualification",
        1,
        "bt",
        "6a8c0d665b08f09c3188cc98c9e1742ef8fdf2d4c44cdbbf9d6b5f3a1ee92ca3",
    ),
    GymToraxSourceAssessmentSeed(
        "source-qualification",
        1,
        "c",
        "de346d4db0d69a48aa852944ea9ef1f51cdccf3569460f858ce2d228922a0ff2",
    ),
    GymToraxSourceAssessmentSeed(
        "source-qualification",
        1,
        "t",
        "5ab1c71fd34510d0d8e3ab436a68de37ba4a7937ed9fccb6afa1637850dc98be",
    ),
    GymToraxSourceAssessmentSeed(
        "source-qualification",
        2,
        "b",
        "3e58c29fdae66761ad3683099b1ad0afa0b88e6852f1750aa8b0e02e983bb999",
    ),
    GymToraxSourceAssessmentSeed(
        "source-qualification",
        2,
        "bt",
        "a98ae74b3405fce8620ee6184125f984f2b19674a632705022be00eaddfa0b99",
    ),
    GymToraxSourceAssessmentSeed(
        "source-qualification",
        2,
        "c",
        "a30aa449a852695339306a2384dd45c0e2a3a596ed94db104654c55c13e2c17b",
    ),
    GymToraxSourceAssessmentSeed(
        "source-qualification",
        2,
        "t",
        "0d7aaa3d9796dfbb5cdf555b738fbe4572e1c0491920c40e2ef319f3b2007150",
    ),
)


def gym_torax_metadata_canary_seed() -> GymToraxSourceAssessmentSeed:
    return _SOURCE_ASSESSMENT_SEEDS[0]


def gym_torax_source_qualification_seeds() -> tuple[GymToraxSourceAssessmentSeed, ...]:
    return _SOURCE_ASSESSMENT_SEEDS[1:]


def require_source_qualification_seed_census(
    seeds: tuple[GymToraxSourceAssessmentSeed, ...],
) -> None:
    expected = tuple(
        (counter, kind) for counter in range(3) for kind in ("b", "bt", "c", "t")
    )
    if not isinstance(seeds, tuple) or any(
        not isinstance(seed, GymToraxSourceAssessmentSeed) for seed in seeds
    ):
        raise ValueError("source qualification requires a typed scientific seed census")
    if tuple((seed.counter, seed.cell_kind) for seed in seeds) != expected or any(
        seed.role != "source-qualification" for seed in seeds
    ):
        raise ValueError(
            "source qualification requires its complete ordered twelve-root seed census"
        )
