"""All-assigned coordinate operands for the existing sole law finalizer."""

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar

from scipy.stats import beta

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from .config import COORDINATES, ROOTS, FrontierDesign
from .discovery import FrontierBound, FrontierDevelopment, valid
from .measurement import FrontierMeasuredRoot, FrontierObservation

QUALIFICATION_ROOTS = tuple(r for r, role, _, _ in ROOTS if role == "qualification")


def cp(successes: int, n: int, alpha: D, *, lower: bool) -> D:
    if not 0 <= successes <= n or n <= 0 or not 0 < alpha < 1:
        raise ValueError("invalid exact binomial count or tail")
    if lower:
        return (
            D(0)
            if not successes
            else D(repr(float(beta.ppf(float(alpha), successes, n - successes + 1))))
        )
    return (
        D(1)
        if successes == n
        else D(repr(float(beta.ppf(1 - float(alpha), successes + 1, n - successes))))
    )


def bound_event(bound: FrontierBound, row: FrontierObservation) -> tuple[bool, tuple[str, ...]]:
    failures = list(row.reasons)
    if not bound.nominated:
        failures.append("DEVELOPMENT_NOMINATION_NONENTRY")
    if not valid(row):
        failures.append("INVALID_CONTACT_DELIVERY_OR_NUMERICS")
    if row.contact and row.evaluable:
        if (
            bound.lower_K is None
            or bound.upper_K is None
            or not all(bound.lower_K <= v <= bound.upper_K for v in row.effects_K)
        ):
            failures.append("RESPONSE_INTERVAL_MISS")
        if (
            row.observed_temperature_K is None
            or bound.thermal_allowance_K is None
            or max(row.action_peaks_K) > row.observed_temperature_K + bound.thermal_allowance_K
        ):
            failures.append("ACTION_ENVELOPE_MISS")
        if (
            row.observed_temperature_K is None
            or bound.zero_thermal_allowance_K is None
            or max(row.reference_peaks_K)
            > row.observed_temperature_K + bound.zero_thermal_allowance_K
        ):
            failures.append("REFERENCE_ENVELOPE_MISS")
        if (
            row.observed_temperature_K is None
            or bound.thermal_allowance_K is None
            or row.observed_temperature_K + bound.thermal_allowance_K > D("356.2")
        ):
            failures.append("CAUSAL_SAFETY_BOUND_TOO_HIGH")
    return not failures, tuple(sorted(set(failures)))


@dataclass(frozen=True, slots=True)
class FrontierQualificationRow(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-finite-control-frontier/frontier-qualification-row'
    bound: FrontierBound
    roots: tuple[tuple[str, bool, bool, bool, tuple[str, ...]], ...]
    lower: D

    def __post_init__(self) -> None:
        if (
            tuple(row[0] for row in self.roots) != QUALIFICATION_ROOTS
            or self.lower
            != cp(self.successes, 96, FrontierDesign().qualification_alpha, lower=True)
            or any(
                passed and (not evaluable or unsafe or reasons)
                for _, evaluable, passed, unsafe, reasons in self.roots
            )
        ):
            raise ValueError("Local law operands changed the 96-root count or frozen exact interval")

    @property
    def successes(self) -> int:
        return sum(int(row[2]) for row in self.roots)

    @property
    def evaluable(self) -> bool:
        return all(row[1] for row in self.roots)

    @property
    def qualifies(self) -> bool:
        return (
            self.bound.nominated
            and self.evaluable
            and self.lower >= D(".90")
            and not any(row[3] for row in self.roots)
        )

    @property
    def record_id(self) -> str:
        return f"{self.bound.coordinate.coordinate_id}.qualification-operands"


@dataclass(frozen=True, slots=True)
class FrontierQualification(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-finite-control-frontier/frontier-qualification'
    development: ObjectIdentity
    measurements: tuple[ObjectIdentity, ...]
    assays: tuple[ObjectIdentity, ...]
    rows: tuple[FrontierQualificationRow, ...]

    def __post_init__(self) -> None:
        if (
            self.development.object_schema != FrontierDevelopment.SCHEMA
            or len(self.measurements) != 96
            or len(self.assays) != 96
            or tuple(row.bound.coordinate for row in self.rows) != COORDINATES
        ):
            raise ValueError("Local law lost its development, assigned roots or 72-coordinate denominator")

    @property
    def record_id(self) -> str:
        return "reactor-finite-control-frontier.qualification"


def qualification_operands(
    development: FrontierDevelopment, measurements: tuple[FrontierMeasuredRoot, ...]
) -> FrontierQualification:
    if tuple(m.root for m in measurements) != QUALIFICATION_ROOTS:
        raise ValueError("Local law cannot replace or omit an assigned root")
    rows = []
    for index, bound in enumerate(development.bounds):
        roots = []
        for m in measurements:
            row = m.observations[index]
            passed, reasons = bound_event(bound, row)
            unsafe = next(
                g.unsafe
                for g in m.guards
                if (g.context, g.pulse) == (bound.coordinate.context, bound.coordinate.pulse)
            ) or (row.contact and row.evaluable and (not row.prefix_safe or not row.action_safe))
            roots.append((m.root, row.evaluable, passed, unsafe, reasons))
        lower = cp(
            sum(int(row[2]) for row in roots),
            96,
            FrontierDesign().qualification_alpha,
            lower=True,
        )
        rows.append(FrontierQualificationRow(bound, tuple(roots), lower))
    return FrontierQualification(
        ObjectIdentity.from_record(development.record_id, development),
        tuple(ObjectIdentity.from_record(f"{m.root}.measurement", m) for m in measurements),
        tuple(m.assay for m in measurements),
        tuple(rows),
    )
