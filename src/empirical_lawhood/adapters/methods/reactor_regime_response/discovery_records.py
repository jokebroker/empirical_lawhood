"""Additive scientific operands; existing B/C package schemas stay unchanged."""

from __future__ import annotations

from dataclasses import dataclass, fields
from decimal import Decimal as D
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord

from .config import ROOTS
from .measured_panel import CONTEXTS, MeasuredContext


@dataclass(frozen=True, slots=True)
class RegimeContextMeasurement(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/regime-context-measurement'
    root: str
    context: str
    callback: int | None
    nominal_mass_kg: tuple[D, ...] | None
    refined_mass_kg: tuple[D, ...] | None
    nominal_peak_K: tuple[D, ...] | None
    refined_peak_K: tuple[D, ...] | None
    nominal_cooling_K: tuple[D, ...] | None
    refined_cooling_K: tuple[D, ...] | None
    coefficient_K_per_kg: D | None
    scalar_residual_K: D | None
    causal_current: tuple[D, ...] | None
    causal_history: tuple[D, ...] | None
    reasons: tuple[str, ...]

    @classmethod
    def from_context(cls, row: MeasuredContext) -> RegimeContextMeasurement:
        def number(value: object) -> object:
            if isinstance(value, float):
                return D(repr(value))
            if isinstance(value, tuple):
                return tuple(number(item) for item in value)
            return value
        return cls(**{field.name: number(getattr(row, field.name)) for field in fields(row)})  # type: ignore[arg-type]

    def to_context(self) -> MeasuredContext:
        def number(value: object) -> object:
            if isinstance(value, D):
                return float(value)
            if isinstance(value, tuple):
                return tuple(number(item) for item in value)
            return value
        return MeasuredContext(**{field.name: number(getattr(self, field.name)) for field in fields(self)})  # type: ignore[arg-type]

    def __post_init__(self) -> None:
        if self.context not in CONTEXTS or self.root not in {row[0] for row in ROOTS}:
            raise ValueError("discovery measurement changed its declared context/root")
        for name, width in (("nominal_mass_kg", 3), ("refined_mass_kg", 3),
                            ("nominal_peak_K", 3), ("refined_peak_K", 3),
                            ("nominal_cooling_K", 3), ("refined_cooling_K", 3),
                            ("causal_current", 6), ("causal_history", 18)):
            value = getattr(self, name)
            if value is not None and (len(value) != width or any(not x.is_finite() for x in value)):
                raise ValueError("discovery measurement has invalid units/shape/finiteness")
        if any(value is not None and not value.is_finite()
               for value in (self.coefficient_K_per_kg, self.scalar_residual_K)):
            raise ValueError("discovery scalar is nonfinite")


@dataclass(frozen=True, slots=True)
class RegimeDiscoveryTraining(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/regime-discovery-training'
    fit_package: ObjectIdentity
    contexts: tuple[RegimeContextMeasurement, ...]

    def __post_init__(self) -> None:
        expected = tuple((root, context) for root, role, _, _ in ROOTS if role == "fit"
                         for context in CONTEXTS)
        if tuple((row.root, row.context) for row in self.contexts) != expected:
            raise ValueError("discovery training lost its 32-root eight-context census")


@dataclass(frozen=True, slots=True)
class RegimePartitionRefit(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/regime-partition-refit'
    draw: int
    root_ordinals: tuple[int, ...]
    distinct_contact_roots: int
    paths: tuple[tuple[tuple[int, D, bool], ...], ...]
    membership: tuple[int, ...]
    leaf_fit_contacts: tuple[int, ...]
    adjusted_rand: D | None
    stable: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if (not 0 <= self.draw < 200 or len(self.root_ordinals) != 32
                or any(not 0 <= value < 32 for value in self.root_ordinals)
                or self.distinct_contact_roots > len(set(self.root_ordinals))
                or (self.adjusted_rand is not None and (not self.adjusted_rand.is_finite()
                                                       or not -1 <= self.adjusted_rand <= 1))
                or (self.stable and (self.reasons or self.adjusted_rand is None
                                    or self.adjusted_rand < D('.8')))):
            raise ValueError("invalid ordinary whole-root refit evidence")


@dataclass(frozen=True, slots=True)
class RegimeLocalStability(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/regime-local-stability'
    model_id: str
    nomination_contexts: tuple[tuple[str, str], ...]
    original_membership: tuple[int, ...]
    original_leaf_count: int
    original_arithmetic_stable: bool
    refits: tuple[RegimePartitionRefit, ...]
    stable_refits: int
    stability_pass: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if (len(self.original_membership) != len(self.nomination_contexts)
                or len(self.refits) not in (0, 200)
                or self.stable_refits != sum(row.stable for row in self.refits)
                or self.stability_pass != (len(self.refits) == 200 and self.stable_refits >= 160
                                           and self.original_arithmetic_stable)
                or (not self.refits and not self.reasons)):
            raise ValueError("local stability changed its fixed 200/160 denominator")


@dataclass(frozen=True, slots=True)
class RegimeDiscoveryDevelopment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/regime-discovery-development'
    training: ObjectIdentity
    fit_package: ObjectIdentity
    nomination: ObjectIdentity
    bootstrap_seed: int
    local_stability: tuple[RegimeLocalStability, ...]
    contexts: tuple[RegimeContextMeasurement, ...]
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        expected = tuple((root, context) for root, role, _, _ in ROOTS if role == "nomination"
                         for context in CONTEXTS)
        if (self.bootstrap_seed != 20260923
                or tuple((row.root, row.context) for row in self.contexts) != expected
                or len({row.model_id for row in self.local_stability}) != len(self.local_stability)):
            raise ValueError("discovery development lost its fixed nomination census")
