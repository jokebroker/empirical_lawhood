"""Fresh local qualification design, fixed before any new native contact."""

from dataclasses import dataclass
from decimal import Decimal as D
from hashlib import sha256
from typing import ClassVar

from empirical_lawhood.kernel.serialization import CanonicalRecord

from .records import LocalAtlas, LocalDomain


def freeze_discovery(payload: bytes) -> LocalAtlas:
    """Authenticate the exposed development publication before freezing its model."""
    import json

    if len(payload) > 4 * 1024**2:
        raise ValueError("local discovery publication identity differs")
    try:
        record = json.loads(payload)
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError("local discovery publication identity differs") from error
    assessments = {d["identifier"]: d for d in record["assessment"]["domains"]}
    domains = []
    for item in record["model"]["domains"]:
        assessment = assessments[item["identifier"]]
        fit = item["fit"]

        def decimal(v: float) -> D:
            return D(repr(float(v)))

        domains.append(
            LocalDomain(
                item["identifier"],
                tuple((c, decimal(t), left) for c, t, left in item["path"]),
                tuple(map(decimal, fit["mean"])),
                tuple(map(decimal, fit["scale"])),
                tuple((decimal(a), decimal(b)) for a, b in fit["operator"]),
                tuple(map(decimal, assessment["support_lower"])),
                tuple(map(decimal, assessment["support_upper"])),
                tuple(assessment["fit_roots_by_action"]),
                tuple(
                    r
                    for r, status in enumerate(assessment["receiver_nomination"])
                    if status == "CANDIDATE_FOR_FRESH_QUALIFICATION"
                ),
            )
        )
    domains = tuple(domains)
    if nomination_sha256(domains) != NOMINATION_SHA256:
        raise ValueError("local discovery publication changes the target nomination")
    return LocalAtlas(
        "reactor-local-primary-discovery", sha256(payload).hexdigest(), domains
    )


PREFIX = "reactor-local-domain-qualification"
# Target-schema scientific domain records, independent of an unpublished raw wrapper.
NOMINATION_SHA256 = "bf897876fce58896a81fdfd2dfe9684532ba350f4780e05422e964d1fcd3a93d"


def nomination_sha256(domains: tuple[LocalDomain, ...]) -> str:
    return sha256(b"".join(domain.canonical_bytes() for domain in domains)).hexdigest()


ROOTS = tuple(
    (f"reactor-local-domain-{role}-{i:03d}", role, i, base + i)
    for role, base in (("calibration", 93000), ("qualification", 94000))
    for i in range(32)
)
SOURCE_TASKS = tuple(f"local.{r}" for r, _, _, _ in ROOTS)


@dataclass(frozen=True, slots=True)
class LocalQualificationDesign(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/methods/reactor-local-domain-qualification/local-qualification-design'
    )
    atlas: LocalAtlas
    config_id: str = "reactor-local-domain-qualification-design"
    assigned_roots: tuple[tuple[str, str, int, int], ...] = ROOTS
    absolute_scale: tuple[D, D] = (D(".25"), D(".01"))
    numerical_padding: tuple[D, D] = (D(".01"), D(".0002"))
    response_scale: tuple[D, D] = (D(".01"), D(".0002"))
    minimum_contact_roots: int = 29
    qualification_probability_floor: D = D(".90")
    confidence: D = D(".95")
    native_call_cap: int = 12288
    wall_seconds: int = 43200
    cpu_seconds: int = 129600

    def __post_init__(self) -> None:
        for name, field in self.__dataclass_fields__.items():
            if name not in ("SCHEMA", "atlas") and (
                getattr(self, name) != field.default
                or type(getattr(self, name)) is not type(field.default)
            ):
                raise ValueError(f"unamended local qualification design: {name}")
        if (
            nomination_sha256(self.atlas.domains) != NOMINATION_SHA256
            or len(self.atlas.domains) != 16
            or len(self.atlas.candidates) != 15
            or sum(len(d.actions) for d in self.atlas.candidates) != 93
        ):
            raise ValueError(
                "qualification requires the exact retained primary local discovery"
            )
        if tuple(
            d.domain_id for d in self.atlas.candidates if 1 in d.nominated_receivers
        ) != ("d100",):
            raise ValueError("conversion nomination roster differs")
