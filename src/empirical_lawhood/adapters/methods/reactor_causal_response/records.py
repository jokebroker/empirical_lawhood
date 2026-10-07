"""Typed qualification operands, distinct from a scientific status."""

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.provenance import ObjectIdentity
from .calibration import RootOperands, calibrate, qualification_operands
from .numerical import FrozenFit


@dataclass(frozen=True, slots=True)
class ReactorQualificationOperands(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/reactor-response/reactor-qualification-operands'
    trace_identities: tuple[ObjectIdentity, ...]
    scores: tuple[D | None, ...]
    invalidity: tuple[tuple[str, ...], ...]
    adequacy: tuple[bool, ...]
    adequacy_reasons: tuple[tuple[str, ...], ...]
    q: D | None
    lower_bound: D

    def __post_init__(self) -> None:
        expected = tuple(
            f"reactor-empirical-{role}-{i:03d}"
            for role in ("calibration", "qualification")
            for i in range(32)
        )
        if tuple(t.object_id for t in self.trace_identities) != expected:
            raise ValueError("exact 64-root calibration/qualification identities required")
        if any(
            len(x) != 32
            for x in (self.scores, self.invalidity, self.adequacy, self.adequacy_reasons)
        ):
            raise ValueError("assigned-unit denominator changed")
        if any((s is None) != bool(r) for s, r in zip(self.scores, self.invalidity, strict=True)):
            raise ValueError("score invalidity must be retained")
        if any(a == bool(r) for a, r in zip(self.adequacy, self.adequacy_reasons, strict=True)):
            raise ValueError("adequacy conjuncts differ")
        if any(s is not None and (not s.is_finite() or s < 0) for s in self.scores):
            raise ValueError("invalid finite root score")
        expected_q = (
            None
            if any(s is None for s in self.scores)
            else max(s for s in self.scores if s is not None)
        )
        if self.q != expected_q:
            raise ValueError("rank32 score differs")
        if not self.lower_bound.is_finite() or not 0 <= self.lower_bound <= 1:
            raise ValueError("invalid binomial bound")

    @property
    def calibration_digest(self) -> str:
        from .numerical import digest

        return digest(
            {
                "traces": [t.object_fingerprint for t in self.trace_identities[:32]],
                "scores": [None if s is None else format(s.normalize(), "f") for s in self.scores],
                "q": None if self.q is None else format(self.q.normalize(), "f"),
            }
        )

    @property
    def bounds(self) -> tuple[D, D] | None:
        return (
            None
            if self.q is None
            else (self.q * D(".25") + D(".01"), self.q * D(".01") + D(".0002"))
        )


@dataclass(frozen=True, slots=True)
class DevelopmentBoundReactorQualificationOperands(ReactorQualificationOperands):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/reactor-response/development-bound-reactor-qualification-operands'
    VERSION: ClassVar[str] = '1.0.0'
    development_q: D

    def __post_init__(self) -> None:
        ReactorQualificationOperands.__post_init__(self)
        if not self.development_q.is_finite() or not 0 <= self.development_q <= 1:
            raise ValueError("unusable frozen development consumer")


def build_operands(
    calibration: tuple[RootOperands, ...],
    qualification: tuple[RootOperands, ...],
    model: FrozenFit,
    identities: tuple[ObjectIdentity, ...],
    *,
    development_q: float,
) -> DevelopmentBoundReactorQualificationOperands:
    calibrated = calibrate(calibration, model, consumer_q=development_q)
    assessed, bound = qualification_operands(qualification, model, calibrated.q)
    return DevelopmentBoundReactorQualificationOperands(
        identities,
        tuple(None if s.value is None else D(repr(s.value)) for s in calibrated.scores),
        tuple(s.reasons for s in calibrated.scores),
        tuple(a.adequate for a in assessed),
        tuple(a.reasons for a in assessed),
        None if calibrated.q is None else D(repr(calibrated.q)),
        D(repr(bound)),
        D(repr(development_q)),
    )
