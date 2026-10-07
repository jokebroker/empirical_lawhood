"""Prepared feed experiment operands, without inherited local qualification claims."""

from dataclasses import dataclass
from typing import ClassVar
import numpy as np

from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.adapters.methods.reactor_local_domain_qualification.records import LocalDomain, LocalArrayPayload as FeedResponseArrayPayload


__all__ = ['FeedResponseArrayPayload', 'FeedDomain', 'FeedAtlas', 'FeedRootEvidence']


@dataclass(frozen=True, slots=True)
class FeedDomain(LocalDomain):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-prepared-feed-qualification/feed-domain'

    def predict(self, x: np.ndarray) -> np.ndarray:
        temperature = LocalDomain.predict(self, x)[:, 0]
        reference = x.copy()
        reference[:, (5, 11, 21)] = 0
        cooling = LocalDomain.predict(self, reference)[:, 0] - temperature
        return np.column_stack((temperature, cooling))

    def proposed_support(self, x: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return np.asarray(
            LocalDomain.proposed_support(self, x, actions)
            & (np.abs(self.predict(x)[:, 1]) <= 0.01)
        )


@dataclass(frozen=True, slots=True)
class FeedAtlas(CanonicalRecord):
    """One nominated local domain; deliberately no complete-partition claim."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-prepared-feed-qualification/feed-atlas'
    atlas_id: str
    development_sha256: str
    domains: tuple[FeedDomain, ...]

    def __post_init__(self) -> None:
        if len(self.domains) != 1 or self.domains[0].domain_id != "d11010":
            raise ValueError("exact prepared feed candidate required")

    @property
    def candidates(self) -> tuple[FeedDomain, ...]:
        return self.domains


@dataclass(frozen=True, slots=True)
class FeedRootEvidence(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-prepared-feed-qualification/feed-root-evidence'
    root: str
    role: str
    seed: int
    recipe: ObjectIdentity
    arrays: FeedResponseArrayPayload
    assays: tuple[tuple[str, int | None, int | None, tuple[int, ...]], ...]
    failures: tuple[tuple[str, str], ...]
    native_calls: int

    def __post_init__(self) -> None:
        from .config import ROOTS

        if (self.root, self.role, self.seed) not in {(r, role, seed) for r, role, _, seed in ROOTS}:
            raise ValueError("feed root is outside the frozen assignment")
        if not 0 <= self.native_calls <= 8 or len(self.assays) != 1:
            raise ValueError("feed source census differs")
        domain, policy, callback, actions = self.assays[0]
        if domain != "d11010" or (policy is None) != (callback is None):
            raise ValueError("feed causal assay identity differs")
        if policy is None:
            if actions:
                raise ValueError("noncontact cannot have measured words")
        elif policy != 0 or callback is None or not 60 <= callback <= 600 or actions != (1, 4, 7):
            raise ValueError("feed preparation/clock/word census differs")
