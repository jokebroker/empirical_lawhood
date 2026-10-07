"""Additive scientific identities; no closed historical census is reopened."""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256, validate_stable_id

PREFIX = "empirical-lawhood.preparation-applicability"
LOWER_SHA256 = "ab880c96a5e4d4e5070e4e2268331ea542f5cc0bedf18da43cded949d2462b1b"
MENU = ("hold", "y-negative-256-recovery-016", "y-positive-256-recovery-144")
DELTA = (0.01, 0.01, 0.0125, 0.005, 0.005, 0.0125, 0.005, 0.005)
CAPS = (0.125, 0.05, 0.05) * 2


def numbers(values: object) -> tuple[Decimal, ...]:
    import numpy as np

    return tuple(Decimal(str(float(v))) for v in np.asarray(values).ravel())


def finite(values: tuple[Decimal, ...], count: int) -> None:
    if len(values) != count or any(type(v) is not Decimal or not v.is_finite() for v in values):
        raise ValueError("applicability record changes its finite native geometry")


@dataclass(frozen=True, slots=True)
class PreparationApplicabilityDesign(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/preparation-applicability/design"
    lower_sha256: str = LOWER_SHA256
    development_roots: int = 32
    evaluation_roots: int = 64
    ridge: Decimal = Decimal(10)
    menu: tuple[str, ...] = MENU
    request_pairs: int = 256

    def __post_init__(self) -> None:
        for name, f in self.__dataclass_fields__.items():
            if name != "SCHEMA" and getattr(self, name) != f.default:
                raise ValueError(f"applicability science is frozen: {name}")


@dataclass(frozen=True, slots=True)
class PreparationApplicabilitySelection(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/preparation-applicability/selection"
    root_id: str
    prefix_sha256: str
    policy_sha256: str | None
    selected_index: int | None
    forecast_margins: tuple[Decimal, ...]
    cutoff_tick: int = 4096

    def __post_init__(self) -> None:
        validate_stable_id(self.root_id)
        validate_sha256(self.prefix_sha256, field_name="prefix_sha256")
        if self.policy_sha256 is not None:
            validate_sha256(self.policy_sha256, field_name="policy_sha256")
        finite(self.forecast_margins, 0 if self.selected_index is None else 3)
        if (
            self.selected_index is not None
            and (type(self.selected_index) is not int or self.selected_index not in range(3))
        ) or self.cutoff_tick != 4096:
            raise ValueError("selection changes its menu or causal cutoff")


@dataclass(frozen=True, slots=True)
class PreparationApplicabilityMeasuredRoot(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/preparation-applicability/measured-root"
    root_id: str
    prefix_sha256: str
    panel_sha256: str
    lower_sha256: str
    complete: bool
    maxima: tuple[Decimal, ...]
    conjuncts: tuple[bool, ...]
    margins: tuple[Decimal, ...]
    work: tuple[Decimal, ...]
    joint_counts: tuple[int, ...]
    covered_counts: tuple[int, ...]
    attempts: tuple[int, ...]
    false_admissions: tuple[int, ...]
    marginal_counts: tuple[int, ...]
    handoff: tuple[Decimal, ...] = field(default=(), kw_only=True)

    def __post_init__(self) -> None:
        validate_stable_id(self.root_id)
        if type(self.complete) is not bool:
            raise ValueError("measurement loses its actual completeness flag")
        for v in (self.prefix_sha256, self.panel_sha256, self.lower_sha256):
            validate_sha256(v, field_name="measurement_sha256")
        finite(self.handoff, 144 if self.complete else 0)
        finite(self.maxima, 21 if self.complete else 0)
        finite(self.margins, 3 if self.complete else 0)
        finite(self.work, 6 if self.complete else 0)
        if len(self.conjuncts) != (21 if self.complete else 0):
            raise ValueError("measurement changes its full-validity conjunction")
        if any(type(value) is not bool for value in self.conjuncts) or self.conjuncts != tuple(value <= 1 for value in self.maxima):
            raise ValueError("measurement changes its noncompensating validity thresholds")
        for values in (
            self.joint_counts,
            self.covered_counts,
            self.attempts,
            self.false_admissions,
        ):
            if len(values) not in (0, 3) or any(
                type(v) is not int or not 0 <= v <= 512 for v in values
            ):
                raise ValueError("service count changes its nested request census")
        service_present=bool(self.joint_counts)
        if (any(bool(values)!=service_present for values in (self.covered_counts,self.attempts,self.false_admissions,self.marginal_counts))
            or service_present and not self.complete
            or len(self.marginal_counts)!=(6 if service_present else 0)
            or any(type(value) is not int or not 0<=value<=256 for value in self.marginal_counts)):
            raise ValueError("measurement omits its complete paired service census")
        if service_present:
            for arm in range(3):
                joint,covered=self.joint_counts[arm],self.covered_counts[arm]
                marginal=self.marginal_counts[2*arm:2*arm+2]
                if (joint>min(marginal) or covered!=(joint if all(self.conjuncts[7*arm:7*arm+7]) else 0)
                    or self.attempts[arm]!=sum(marginal)+self.false_admissions[arm]):
                    raise ValueError("measurement changes actual attempts, covered service or false admissions")
