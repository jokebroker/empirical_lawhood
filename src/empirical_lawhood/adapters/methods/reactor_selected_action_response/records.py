"""Causal/native separation for one selected finite action on fresh roots."""

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar

from empirical_lawhood.adapters.methods.reactor_local_domain_qualification.records import LocalArrayPayload
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from .config import ClassicalDesign, assignment


@dataclass(frozen=True, slots=True)
class ClassicalCausal(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-selected-action-response/classical-causal'
    root: str
    role: str
    seed: int
    recipe: ObjectIdentity
    source_sha256: str
    callback: int | None
    arrays: LocalArrayPayload
    native_calls: int
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            (self.role, self.seed) != assignment(self.root)
            or self.recipe
            != ObjectIdentity.from_record(ClassicalDesign().config_id, ClassicalDesign())
            or not 0 <= self.native_calls <= 2
            or self.callback is not None
            and not 60 <= self.callback <= 600
            or self.callback is None
            and not self.reasons
        ):
            raise ValueError("classical causal preparation changes its assignment or recipe")
        arrays = self.arrays.unpack()
        expected_keys = (
            {
                f"exploration_unshifted_v{view}_{field}"
                for view in (0, 1)
                for field in ("observations", "requests", "stages", "exposure")
            }
            if self.callback is not None
            else set()
        )
        if set(arrays) != expected_keys:
            raise ValueError("classical causal record contains undeclared information")
        if self.callback is not None:
            for view in (0, 1):
                for field in ("observations", "requests", "stages", "exposure"):
                    key = f"exploration_unshifted_v{view}_{field}"
                    expected = self.callback + int(field == "observations")
                    if key not in arrays or len(arrays[key]) != expected:
                        raise ValueError(
                            "classical causal record contains a future or missing prefix"
                        )


@dataclass(frozen=True, slots=True)
class ClassicalPrivate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-selected-action-response/classical-private'
    root: str
    causal_preparation: ObjectIdentity
    arrays: LocalArrayPayload

    def __post_init__(self) -> None:
        assignment(self.root)
        if self.causal_preparation.object_schema != ClassicalCausal.SCHEMA:
            raise ValueError("classical private preparation changed its parent")


@dataclass(frozen=True, slots=True)
class ClassicalDecision(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-selected-action-response/classical-decision'
    root: str
    causal_preparation: ObjectIdentity
    recipe: ObjectIdentity
    qualified_law: ObjectIdentity | None
    callback: int | None
    causal_preparation_valid: bool
    observed_temperature_K: D | None
    issue_authority: ObjectIdentity
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        role, _ = assignment(self.root)
        if (
            self.recipe
            != ObjectIdentity.from_record(ClassicalDesign().config_id, ClassicalDesign())
            or self.issue_authority.object_schema
            != 'empirical-lawhood/planning/durable-authorization-record'
            or self.causal_preparation.object_schema != ClassicalCausal.SCHEMA
            or (
                self.causal_preparation_valid
                and (self.callback is None or self.observed_temperature_K is None or self.reasons)
            )
            or (not self.causal_preparation_valid and not self.reasons)
            or role == "prospective"
            and self.causal_preparation_valid
            and self.qualified_law is None
        ):
            raise ValueError("classical decision changed its frozen prerequisites")

    @property
    def decision_id(self) -> str:
        return f"{self.root}.classical-decision"

    @property
    def route(self) -> str:
        return "prepared_t0"


@dataclass(frozen=True, slots=True)
class ClassicalAssay(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-selected-action-response/classical-assay'
    root: str
    causal_preparation: ObjectIdentity
    sealed_decision: ObjectIdentity
    arrays: LocalArrayPayload
    native_calls: int
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        assignment(self.root)
        if (
            self.causal_preparation.object_schema != ClassicalCausal.SCHEMA
            or self.sealed_decision.object_schema != ClassicalDecision.SCHEMA
            or not 0 <= self.native_calls <= 4
        ):
            raise ValueError("classical assay changed its causal seal or call census")


@dataclass(frozen=True, slots=True)
class ClassicalRootScore(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-selected-action-response/classical-root-score'
    root: str
    evaluable: bool
    adequate: bool
    safe: bool
    cooling_K: tuple[D, ...]
    peak_K: tuple[D, ...]
    mass_kg: tuple[D, ...]
    temperature_error_K: tuple[D, ...]
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        assignment(self.root)
        if (
            (
                self.evaluable
                and any(
                    len(x) != 2
                    for x in (self.cooling_K, self.peak_K, self.mass_kg, self.temperature_error_K)
                )
            )
            or self.adequate
            and (not self.evaluable or not self.safe or self.reasons)
            or not self.adequate
            and not self.reasons
        ):
            raise ValueError("classical root score lost its two-view result or refusal")


@dataclass(frozen=True, slots=True)
class ClassicalQualification(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-selected-action-response/classical-qualification'
    recipe: ObjectIdentity
    assays: tuple[ObjectIdentity, ...]
    roots: tuple[ClassicalRootScore, ...]
    successes: int
    lower_95: D
    mean_K: D | None
    mean_lower_95_K: D | None
    evaluable: bool
    qualifies: bool

    def __post_init__(self) -> None:
        from .config import ROOTS
        from scipy.stats import beta

        design = ClassicalDesign()
        expected_lower = (
            D(0)
            if not self.successes
            else D(repr(float(beta.ppf(0.05, self.successes, 65 - self.successes))))
        )
        expected = tuple(root for root, role, _, _ in ROOTS if role == "qualification")
        if (
            tuple(row.root for row in self.roots) != expected
            or len(self.assays) != 64
            or self.recipe != ObjectIdentity.from_record(design.config_id, design)
            or tuple(x.object_id for x in self.assays) != tuple(f"{r}.assay" for r in expected)
            or any(x.object_schema != ClassicalAssay.SCHEMA for x in self.assays)
            or self.lower_95 != expected_lower
            or self.successes != sum(row.adequate for row in self.roots)
            or self.evaluable != all(row.evaluable for row in self.roots)
            or self.qualifies
            != (
                self.evaluable and self.lower_95 >= D(".90") and all(row.safe for row in self.roots)
            )
        ):
            raise ValueError("classical qualification changes its all-assigned denominator")

    @property
    def package_id(self) -> str:
        return "reactor-classical-selected-action-qualification"
