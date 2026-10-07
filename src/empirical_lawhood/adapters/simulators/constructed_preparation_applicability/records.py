"""Additive native records; preserved clocks/chart, changed preparation bath."""

from dataclasses import dataclass
from hashlib import sha256
from decimal import Decimal
from typing import Any, ClassVar
from empirical_lawhood.kernel.serialization import validate_sha256
from empirical_lawhood.adapters.simulators.preparation_applicability.records import (
    PreparationApplicabilityStream, PreparationApplicabilityNativePhase, PreparationApplicabilityPrefix, PreparationApplicabilityPanel,
)
from empirical_lawhood.adapters.methods.preparation_applicability.seals import PreparationApplicabilityParents


def nominal_digest(purpose: str) -> str:
    """Frozen deterministic recipe metadata; decoding never imports a marcher."""
    if purpose not in ("prefix", "prefix-bridge", "parent", "parent-bridge", "prefix-probes"):
        raise ValueError("nominal waveform purpose is outside its frozen recipe")
    return sha256(f"cc1-applicability-r1-v1:D:016:{purpose}".encode()).hexdigest()


class ConditionedGenerator:
    """A declared deterministic mean waveform plus independent Gaussian noise."""

    def __init__(self, nominal: Any, fresh: Any) -> None:
        self.nominal, self.fresh = nominal, fresh

    def normal(self, *args: Any, **kwargs: Any) -> Any:
        return self.nominal.normal(*args, **kwargs) + .002 * self.fresh.normal(*args, **kwargs)


@dataclass(frozen=True, slots=True)
class ConstructedPreparationStream(PreparationApplicabilityStream):
    SCHEMA: ClassVar[str] = "empirical-lawhood/constructed-preparation-applicability/stream"
    scientific_seed: int
    nominal_seed_sha256: str | None
    innovation_fraction: Decimal

    def __post_init__(self) -> None:
        PreparationApplicabilityStream.__post_init__(self)
        if self.nominal_seed_sha256 is not None:
            validate_sha256(self.nominal_seed_sha256, field_name="nominal_seed_sha256")
        if self.innovation_fraction != (Decimal(1) if self.nominal_seed_sha256 is None else Decimal(".002")):
            raise ValueError("stream changes construction or ordinary-future variance")

    def generator(self) -> Any:
        import numpy as np
        fresh = np.random.Generator(np.random.PCG64(self.scientific_seed))
        if self.nominal_seed_sha256 is None:
            return fresh
        nominal = np.random.Generator(np.random.PCG64(int(self.nominal_seed_sha256[:32], 16)))
        return ConditionedGenerator(nominal, fresh)


@dataclass(frozen=True, slots=True)
class ConstructedPreparationNativePhase(PreparationApplicabilityNativePhase):
    SCHEMA: ClassVar[str] = "empirical-lawhood/constructed-preparation-applicability/native-phase"
    streams: tuple[ConstructedPreparationStream, ...]

    def __post_init__(self) -> None:
        PreparationApplicabilityNativePhase.__post_init__(self)
        purpose = f"future-{self.future_index + 1}" if self.phase == "future" else self.phase
        expected = tuple(
            ConstructedPreparationStream(
                self.allocation.seed_for(name),
                None if self.phase == "future" else nominal_digest(name),
                Decimal(1) if self.phase == "future" else Decimal(".002"),
            ) for name in (purpose, purpose + "-bridge")
        )
        if self.streams != expected:
            raise ValueError("native stream receipt changes prescribed mean, variance or fresh allocation")



@dataclass(frozen=True, slots=True)
class ConstructedPreparationPrefix(PreparationApplicabilityPrefix):
    SCHEMA: ClassVar[str] = "empirical-lawhood/constructed-preparation-applicability/prefix"
    phases: tuple[ConstructedPreparationNativePhase, ...]


@dataclass(frozen=True, slots=True)
class ConstructedPreparationPanel(PreparationApplicabilityPanel):
    SCHEMA: ClassVar[str] = "empirical-lawhood/constructed-preparation-applicability/panel"
    phases: tuple[ConstructedPreparationNativePhase, ...]


@dataclass(frozen=True, slots=True)
class ConstructedPreparationParents(PreparationApplicabilityParents):
    SCHEMA: ClassVar[str] = "empirical-lawhood/constructed-preparation-applicability/parents"
    phases: tuple[ConstructedPreparationNativePhase, ...]
