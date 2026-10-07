"""Complete ordinary-source screen operands, without learned-policy qualification."""

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar
import numpy as np

from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256
from .records import numbers, finite


@dataclass(frozen=True, slots=True)
class PreparationApplicabilityScreenReport(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/preparation-applicability/screen-report'
    phase: str
    root_ids: tuple[str, ...]
    measurement_sha256s: tuple[str, ...]
    complete: bool
    disposition: str
    full_valid_counts: tuple[int, ...]
    support_counts: tuple[int, ...]
    covered_counts: tuple[int, ...]
    joint_counts: tuple[int, ...]
    false_admissions: tuple[int, ...]
    handoff: tuple[Decimal, ...]
    lower_sha256: str
    design_sha256: str
    source_sha256: str
    allocation_sha256: str

    def __post_init__(self) -> None:
        if type(self.complete) is not bool:
            raise ValueError('ordinary report loses its complete assigned measurement flag')
        if self.phase not in ('D','E') or len(self.root_ids) != (32 if self.phase == 'D' else 64) or len(set(self.root_ids)) != len(self.root_ids) or len(self.measurement_sha256s) != len(self.root_ids):
            raise ValueError('ordinary report changes its assigned cohort')
        for digest in (*self.measurement_sha256s,self.lower_sha256,self.design_sha256,self.source_sha256,self.allocation_sha256):
            validate_sha256(digest)
        finite(self.handoff, len(self.root_ids)*144 if self.complete else 0)
        if self.disposition != ('EXPOSED_SCREEN_OPERANDS_READY' if self.complete else 'UNEVALUABLE'):
            raise ValueError('ordinary screen loses its evidence ceiling or missingness')
        for name,limit in (('full_valid_counts',len(self.root_ids)),('support_counts',len(self.root_ids)),
                           ('covered_counts',256*len(self.root_ids)),('joint_counts',256*len(self.root_ids)),
                           ('false_admissions',512*len(self.root_ids))):
            values=getattr(self,name)
            expected=3 if self.complete and (self.phase=='E' or name in ('full_valid_counts','support_counts')) else 0
            if len(values)!=expected or any(type(value) is not int or not 0<=value<=limit for value in values):
                raise ValueError('ordinary report changes complete independent-root/service denominators')
        if any(valid>support for valid,support in zip(self.full_valid_counts,self.support_counts,strict=True)):
            raise ValueError('ordinary full validity exceeds its support census')
        if self.phase=='E' and any(covered>joint for covered,joint in zip(self.covered_counts,self.joint_counts,strict=True)):
            raise ValueError('ordinary covered service exceeds actual joint service')


def report(stage, measured) -> PreparationApplicabilityScreenReport:
    if tuple(row.root_id for row in measured) != stage.root_ids:
        raise ValueError('ordinary screen changes its complete assigned independent units')
    complete = all(row.complete for row in measured)
    counts = ((),)*5
    if complete:
        maxima = np.asarray([row.maxima for row in measured],dtype=float).reshape(-1,3,7)
        counts = (
            tuple(map(int,(maxima<=1).all(axis=2).sum(axis=0))),
            tuple(map(int,(maxima[:,:,0]<=1).sum(axis=0))),
            tuple(map(int,np.asarray([row.covered_counts for row in measured]).sum(axis=0))) if stage.phase=='E' else (),
            tuple(map(int,np.asarray([row.joint_counts for row in measured]).sum(axis=0))) if stage.phase=='E' else (),
            tuple(map(int,np.asarray([row.false_admissions for row in measured]).sum(axis=0))) if stage.phase=='E' else (),
        )
    return PreparationApplicabilityScreenReport(
        stage.phase,stage.root_ids,tuple(row.fingerprint() for row in measured),complete,
        'EXPOSED_SCREEN_OPERANDS_READY' if complete else 'UNEVALUABLE',*counts,
        numbers([row.handoff for row in measured]) if complete else (),
        stage.upstream[0].artifact.sha256,stage.design.fingerprint(),
        stage.source.object_fingerprint,stage.allocation.fingerprint(),
    )
