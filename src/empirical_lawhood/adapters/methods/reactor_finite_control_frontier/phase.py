"""Exact separate phase issues and retained immutable upstream artifacts."""

from dataclasses import dataclass
from typing import ClassVar
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from .config import FrontierDesign, ROOTS, PREFIX
from .records import FrontierRetainedRoot
from .discovery import FrontierDevelopment
from .law_terminal import FrontierLaws


@dataclass(frozen=True, slots=True)
class FrontierUpstream(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-finite-control-frontier/frontier-upstream'
    key: str
    artifact: ArtifactIdentity
    receipt: ObjectIdentity | None
    source_run_id: str | None


@dataclass(frozen=True, slots=True)
class FrontierPhase(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-finite-control-frontier/frontier-phase'
    design: FrontierDesign
    phase: str
    upstream: tuple[FrontierUpstream, ...]

    def __post_init__(self) -> None:
        expected = (
            tuple(
                (r, FrontierRetainedRoot.SCHEMA)
                for r, role, _, _ in ROOTS
                if role == "development"
            )
            if self.phase == "B"
            else (("development", FrontierDevelopment.SCHEMA),)
            if self.phase == "C"
            else (("development", FrontierDevelopment.SCHEMA), ("laws", FrontierLaws.SCHEMA))
            if self.phase == "D"
            else ()
        )
        if (
            not expected
            or tuple((r.key, r.artifact.payload_schema) for r in self.upstream) != expected
            or any((r.receipt is None) != (self.phase == "B") for r in self.upstream)
        ):
            raise ValueError("phase issue changed its complete immutable upstream census")
        if any((r.receipt is None) != (r.source_run_id is None) for r in self.upstream):
            raise ValueError("retained phase lost its publication namespace")
        if any(
            r.receipt is not None and r.receipt.object_schema != 'empirical-lawhood/runtime/canonical-task-receipt'
            for r in self.upstream
        ):
            raise ValueError("phase upstream lacks its native production receipt")

    @property
    def config_id(self) -> str:
        return f"{PREFIX}.phase-{self.phase.lower()}"

    @property
    def roots(self) -> tuple[str, ...]:
        return tuple(
            r
            for r, role, _, _ in ROOTS
            if role == {"B": "development", "C": "qualification", "D": "prospective"}[self.phase]
        )

    @property
    def cpu_seconds(self) -> int:
        return {"B": 20, "C": 36, "D": 52}[self.phase] * 3600

    @property
    def wall_seconds(self) -> int:
        return (8 if self.phase == "B" else 14) * 3600
