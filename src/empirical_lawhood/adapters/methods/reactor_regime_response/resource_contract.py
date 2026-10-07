"""One retained B/C/D acquisition allocation across separate phase issues."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal as D
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.time import validate_utc_timestamp
from empirical_lawhood.planning.study_issue import StudyOperationAuthority
from empirical_lawhood.runtime.execution_envelope import ExecutionResourceEnvelopeSpec
from empirical_lawhood.runtime.study_issue import IssuedExecutableStudyManifest

from .config import PREFIX, ReactorRegimeResponseDesign


@dataclass(frozen=True, slots=True)
class RegimeWholeStudyAllocation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/regime-whole-study-allocation'

    allocation_id: str
    design: ObjectIdentity
    initial_execution_authority: ObjectIdentity
    started_at_utc: str

    def __post_init__(self) -> None:
        design = ReactorRegimeResponseDesign()
        validate_utc_timestamp(self.started_at_utc)
        if (
            self.allocation_id != f"{PREFIX}.whole-study-allocation"
            or self.design != ObjectIdentity.from_record(design.config_id, design)
            or self.initial_execution_authority.object_schema
            != StudyOperationAuthority.SCHEMA
        ):
            raise ValueError("reactor study allocation changed its first authority or design")


@dataclass(frozen=True, slots=True)
class RegimePhaseAllocation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/regime-phase-allocation'

    allocation_id: str
    phase: str
    study: ObjectIdentity
    issued_study: ObjectIdentity
    execution_authority: ObjectIdentity
    resource_envelope: ObjectIdentity
    run_id: str
    started_at_utc: str
    prior_elapsed_seconds: D
    previous_closeout: ObjectIdentity | None

    def __post_init__(self) -> None:
        validate_utc_timestamp(self.started_at_utc)
        if (
            self.phase not in ("B", "C", "D")
            or self.allocation_id != f"{PREFIX}.phase-{self.phase.lower()}.allocation"
            or self.run_id != f"{PREFIX}-phase-{self.phase.lower()}"
            or self.study.object_schema != RegimeWholeStudyAllocation.SCHEMA
            or self.study.object_id != f"{PREFIX}.whole-study-allocation"
            or self.issued_study.object_schema != IssuedExecutableStudyManifest.SCHEMA
            or self.execution_authority.object_schema != StudyOperationAuthority.SCHEMA
            or self.resource_envelope.object_schema != ExecutionResourceEnvelopeSpec.SCHEMA
            or not self.prior_elapsed_seconds.is_finite()
            or not D(0) <= self.prior_elapsed_seconds
                < D(ReactorRegimeResponseDesign().acquisition_wall_seconds)
            or (self.phase == "B") != (self.previous_closeout is None)
            or self.previous_closeout is not None
                and self.previous_closeout.object_schema != RegimePhaseCloseout.SCHEMA
            or self.previous_closeout is not None
                and self.previous_closeout.object_id != (
                    f"{PREFIX}.phase-{'b' if self.phase == 'C' else 'c'}.closeout"
                )
        ):
            raise ValueError("reactor phase allocation changed its issued resource identity")


@dataclass(frozen=True, slots=True)
class RegimePhaseCloseout(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/regime-phase-closeout'

    closeout_id: str
    phase: str
    allocation: ObjectIdentity
    started_at_utc: str
    ended_at_utc: str
    elapsed_seconds: D
    cumulative_elapsed_seconds: D

    def __post_init__(self) -> None:
        validate_utc_timestamp(self.started_at_utc)
        validate_utc_timestamp(self.ended_at_utc)
        measured = D(str((
            datetime.fromisoformat(self.ended_at_utc.replace("Z", "+00:00"))
            - datetime.fromisoformat(self.started_at_utc.replace("Z", "+00:00"))
        ).total_seconds()))
        if (
            self.phase not in ("B", "C", "D")
            or self.closeout_id != f"{PREFIX}.phase-{self.phase.lower()}.closeout"
            or self.allocation.object_schema != RegimePhaseAllocation.SCHEMA
            or self.allocation.object_id != f"{PREFIX}.phase-{self.phase.lower()}.allocation"
            or measured != self.elapsed_seconds
            or not self.elapsed_seconds.is_finite()
            or self.elapsed_seconds < 0
            or not self.cumulative_elapsed_seconds.is_finite()
            or not self.elapsed_seconds <= self.cumulative_elapsed_seconds
                <= D(ReactorRegimeResponseDesign().acquisition_wall_seconds)
        ):
            raise ValueError("reactor phase closeout changed its measured cumulative elapsed time")


def close_phase_allocation(
    allocation: RegimePhaseAllocation, ended_at_utc: str
) -> RegimePhaseCloseout:
    "Seal one phase's elapsed duration before allocating its next phase."
    validate_utc_timestamp(ended_at_utc)
    elapsed = D(str((
        datetime.fromisoformat(ended_at_utc.replace("Z", "+00:00"))
        - datetime.fromisoformat(allocation.started_at_utc.replace("Z", "+00:00"))
    ).total_seconds()))
    return RegimePhaseCloseout(
        f"{PREFIX}.phase-{allocation.phase.lower()}.closeout",
        allocation.phase,
        ObjectIdentity.from_record(allocation.allocation_id, allocation),
        allocation.started_at_utc,
        ended_at_utc,
        elapsed,
        allocation.prior_elapsed_seconds + elapsed,
    )
