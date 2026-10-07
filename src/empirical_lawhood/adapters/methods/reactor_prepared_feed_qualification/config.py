"""Prospective prepared-feed specification; exposed candidate, fresh uncertainty."""

from dataclasses import dataclass, fields
from decimal import Decimal as D
from typing import ClassVar

from empirical_lawhood.adapters.methods.reactor_local_domain_qualification.config import freeze_discovery as freeze_local_discovery
from empirical_lawhood.kernel.serialization import CanonicalRecord

from .records import FeedAtlas, FeedDomain

PREFIX = "reactor-prepared-feed-qualification"
ROOTS = tuple(
    (f"reactor-prepared-feed-{role}-{i:03d}", role, i, base + i)
    for role, base in (("calibration", 95000), ("qualification", 96000))
    for i in range(32)
)
SOURCE_TASKS = tuple(f"feed.{r}" for r, _, _, _ in ROOTS)


def freeze_discovery(payload: bytes) -> FeedAtlas:
    local = freeze_local_discovery(payload)
    original = next(d for d in local.domains if d.domain_id == "d11010")
    values = {f.name: getattr(original, f.name) for f in fields(original)}
    values.update(
        path=original.path[:-1] + ((3, D(".500001"), True),),
        action_fit_roots=tuple(
            n if i in (1, 4, 7) else 0 for i, n in enumerate(original.action_fit_roots)
        ),
        nominated_receivers=(0, 1),
    )
    return FeedAtlas(
        "reactor-feed-primary-candidate",
        local.development_sha256,
        (FeedDomain(**values),),
    )


@dataclass(frozen=True, slots=True)
class FeedQualificationDesign(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-prepared-feed-qualification/feed-qualification-design'
    atlas: FeedAtlas
    config_id: str = "reactor-feed-qualification"
    preparation_phase: int = 0
    assigned_roots: tuple[tuple[str, str, int, int], ...] = ROOTS
    # Receiver 1 is a normalized contrast error, not conversion or kelvin.
    absolute_scale: tuple[D, D] = (D(".25"), D(1))
    numerical_padding: tuple[D, D] = (D(".01"), D(".000001"))
    response_scale: tuple[D, D] = (D(".00005"), D(".5"))
    minimum_contact_roots: int = 29
    qualification_probability_floor: D = D(".90")
    confidence: D = D(".95")
    native_call_cap: int = 512
    wall_seconds: int = 43200
    cpu_seconds: int = 129600

    def __post_init__(self) -> None:
        for name, field in self.__dataclass_fields__.items():
            if name not in ("SCHEMA", "atlas") and (
                getattr(self, name) != field.default
                or type(getattr(self, name)) is not type(field.default)
            ):
                raise ValueError(f"unamended feed specification: {name}")
        domain = self.atlas.domains[0]
        if (
            domain.fingerprint()
            != "7ae75d6a3431e53f0b43955e1a31a1f6b77c1d58550a71dc1537a8fac1d2c0dd"
            or domain.actions != (1, 4, 7)
            or domain.nominated_receivers != (0, 1)
            or domain.path[-1] != (3, D(".500001"), True)
        ):
            raise ValueError("feed candidate support differs")
