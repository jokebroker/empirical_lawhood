"""New identities with explicit compatibility to the unchanged measurement chart."""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256, require_sorted_unique_strings
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.adapters.simulators.finite_response_law.instruments import REFERENCE_INSTRUMENT
from empirical_lawhood.adapters.methods.finite_response_law.law_payloads import FiniteResponseLawResponseChart
from empirical_lawhood.adapters.methods.preparation_applicability.records import (
    PreparationApplicabilitySelection, PreparationApplicabilityMeasuredRoot, LOWER_SHA256 as LOWER_SHA256,
    MENU as MENU, DELTA as DELTA, CAPS as CAPS, finite, numbers as numbers,
)
from empirical_lawhood.adapters.methods.preparation_applicability.seals import PreparationApplicabilityLowerSeal

PREFIX = "empirical-lawhood.constructed-preparation-applicability"


@dataclass(frozen=True, slots=True)
class ConstructedPreparationCompatibility(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/constructed-preparation-applicability/compatibility"
    instrument: ObjectIdentity = REFERENCE_INSTRUMENT.identity
    chart: FiniteResponseLawResponseChart = FiniteResponseLawResponseChart()
    native_chart_map: str = "identity-on-q2-units-returned-couplings-SKR24-signed-words-paired-receiver-horizon192"
    clocks: tuple[int, ...] = (4096, 4496, 4464, 4688)
    changed_denominator: str = "pre4496-prescribed-bath-mean-plus-.002-independent-perturbation"
    authority_ceiling: str = "fresh-measured-validity-only-no-automatic-history-qualification-transport"

    def __post_init__(self) -> None:
        for name, spec in self.__dataclass_fields__.items():
            if name != "SCHEMA" and getattr(self, name) != spec.default:
                raise ValueError(f"compatibility map changes frozen scientific semantics: {name}")



@dataclass(frozen=True, slots=True)
class ConstructedPreparationDesign(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/constructed-preparation-applicability/design"
    lower_sha256: str = LOWER_SHA256
    qualification_roots: int = 8
    evaluation_roots: int = 32
    innovation_fraction: Decimal = Decimal(".002")
    menu: tuple[str, ...] = MENU
    request_pairs: int = 256
    nominal_programme: str = "cc1-applicability-r1-v1"
    nominal_phase: str = "D"
    nominal_index: int = 16
    compatibility: ConstructedPreparationCompatibility = ConstructedPreparationCompatibility()

    def __post_init__(self) -> None:
        for name, spec in self.__dataclass_fields__.items():
            if name != "SCHEMA" and getattr(self, name) != spec.default:
                raise ValueError(f"boundary construction is frozen: {name}")


@dataclass(frozen=True, slots=True)
class ConstructedPreparationSelection(PreparationApplicabilitySelection):
    SCHEMA: ClassVar[str] = "empirical-lawhood/constructed-preparation-applicability/selection"


@dataclass(frozen=True, slots=True)
class ConstructedPreparationLowerSeal(PreparationApplicabilityLowerSeal):
    SCHEMA: ClassVar[str] = "empirical-lawhood/constructed-preparation-applicability/lower-seal"


@dataclass(frozen=True, slots=True)
class ConstructedPreparationMeasuredRoot(PreparationApplicabilityMeasuredRoot):
    SCHEMA: ClassVar[str] = "empirical-lawhood/constructed-preparation-applicability/measured-root"
    handoff: tuple[Decimal, ...] = field(default=(), kw_only=True)

    def __post_init__(self) -> None:
        PreparationApplicabilityMeasuredRoot.__post_init__(self)
        finite(self.handoff, 144 if self.complete else 0)


@dataclass(frozen=True, slots=True)
class ConstructedPreparationReport(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/constructed-preparation-applicability/report"
    phase: str
    root_ids: tuple[str, ...]
    measurement_sha256s: tuple[str, ...]
    complete: bool
    constructor_qualified: bool
    disposition: str
    full_valid_counts: tuple[int, ...]
    support_counts: tuple[int, ...]
    covered_counts: tuple[int, ...]
    joint_counts: tuple[int, ...]
    false_admissions: tuple[int, ...]
    effect: Decimal | None
    wins: int
    losses: int
    p: Decimal | None
    covered_effect: Decimal | None
    covered_interval: tuple[Decimal, ...]
    lower_sha256: str = field(kw_only=True)
    design_sha256: str = field(kw_only=True)
    source_sha256: str = field(kw_only=True)
    allocation_sha256: str = field(kw_only=True)
    effective_seed_ids: tuple[str, ...] = field(kw_only=True)
    constructor_crossings: tuple[bool, ...] = field(default=(), kw_only=True)
    numerical_maxima: tuple[Decimal, ...] = field(default=(), kw_only=True)

    def __post_init__(self) -> None:
        for digest in (self.lower_sha256, self.design_sha256, self.source_sha256, self.allocation_sha256, *self.measurement_sha256s):
            validate_sha256(digest)
        require_sorted_unique_strings(self.effective_seed_ids, field_name="effective_seed_ids", allow_empty=False)
        expected = self.root_ids
        if self.phase not in ("Q", "E") or len(expected) != (8 if self.phase == "Q" else 32) or len(set(expected)) != len(expected):
            raise ValueError("report changes independent native preparation census")
        if len(self.measurement_sha256s) != len(expected):
            raise ValueError("report omits immutable measurement identities")
        if self.disposition not in ("QUALIFIED", "CONSTRUCTOR_NOT_QUALIFIED", "SUPPORTED", "OPPOSED", "UNRESOLVED", "UNEVALUABLE"):
            raise ValueError("report loses its typed scientific outcome")
        finite(self.covered_interval, 2 if self.complete else 0)
        if self.complete and self.phase == "Q":
            finite(self.numerical_maxima, 24)
            if len(self.constructor_crossings) != 8 or any(type(value) is not bool for value in self.constructor_crossings):
                raise ValueError("Q report loses its complete crossing census")
            qualified = sum(self.constructor_crossings) >= 6 and all(value <= 1 for value in self.numerical_maxima)
            if self.constructor_qualified != qualified or self.disposition != ("QUALIFIED" if qualified else "CONSTRUCTOR_NOT_QUALIFIED"):
                raise ValueError("Q report changes its measured constructor qualification")
        elif self.constructor_qualified or self.constructor_crossings or self.numerical_maxima:
            raise ValueError("only complete Q can qualify a constructor")
        if not self.complete and self.disposition != "UNEVALUABLE":
            raise ValueError("incomplete report loses its unevaluable disposition")
