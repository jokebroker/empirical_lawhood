"Fresh, fixed selected-action design; no inherited scientific verdict."

from dataclasses import dataclass
from decimal import Decimal as D
from hashlib import sha256
from typing import ClassVar

from empirical_lawhood.kernel.serialization import CanonicalRecord

PREFIX = "reactor-selected-action-response-programme"
# Target-owned exposed nomination, not a receipt for earlier experiments.
# Historical raw development is an explicit separate authoring input.
DEVELOPMENT_SHA = sha256(
    b"empirical-lawhood/reactor-selected-action-response/exposed-nomination;"
    b"feed_kg_s=.016;mass_kg=.16;cooling_K=[.0012,.0044];"
    b"causal_temperature_halfwidth_K=1;request_K=.0012;limit_K=356.2"
).hexdigest()
ROOTS = tuple(
    (f"reactor-selected-action-response-{role}-{i:03d}", role, i, first + i)
    for role, count, first in (("qualification", 64, 98500), ("prospective", 32, 98600))
    for i in range(count)
)
DEVELOPMENT_ROOTS = tuple(
    sorted(
        f"reactor-regime-response-{role}-{i:03d}"
        for role, count in (
            ("fit", 32),
            ("nomination", 16),
            ("calibration", 32),
            ("qualification", 64),
        )
        for i in range(count)
    )
)


@dataclass(frozen=True, slots=True)
class ClassicalDesign(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-selected-action-response/classical-design'
    config_id: str = "reactor-selected-action-response-design"
    development_sha256: str = DEVELOPMENT_SHA
    assigned_roots: tuple[tuple[str, str, int, int], ...] = ROOTS
    requested_feed_kg_s: D = D(".016")
    applied_mass_kg: D = D(".16")
    response_lower_K: D = D(".0012")
    response_upper_K: D = D(".0044")
    causal_temperature_halfwidth_K: D = D(1)
    request_K: D = D(".0012")
    temperature_limit_K: D = D("356.2")
    numerical_peak_tolerance_K: D = D(".01")
    numerical_cooling_tolerance_K: D = D(".000001")
    qualification_probability_floor: D = D(".90")
    confidence: D = D(".95")
    native_call_cap: int = 1024
    wall_seconds: int = 43200
    cpu_seconds: int = 129600

    def __post_init__(self) -> None:
        for name, field in self.__dataclass_fields__.items():
            if name != "SCHEMA" and (
                getattr(self, name) != field.default
                or type(getattr(self, name)) is not type(field.default)
            ):
                raise ValueError(f"fixed selected-action response design differs: {name}")


def assignment(root: str) -> tuple[str, int]:
    matches = [(role, seed) for name, role, _, seed in ROOTS if name == root]
    if len(matches) != 1:
        raise ValueError("root is outside the fresh selected-action response assignment")
    return matches[0]
