"""Shot-level assembly and protected outcome-blind projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id

from .contracts import (
    FairMastMaterializationReceipt,
    MastCausalState,
    MastEndpointResult,
    MastEvent,
    MastProtectedPreActionPackage,
    MastShotAssignment,
    MastShotRole,
)


@dataclass(frozen=True, slots=True)
class MastArchiveDevelopmentUnit(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive/mast-archive-development-unit'

    unit_id: str
    assignment: ObjectIdentity
    materialization: ObjectIdentity
    event: MastEvent
    state: MastCausalState
    endpoint: MastEndpointResult
    independent_unit_count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.unit_id, field_name="unit_id")
        identities = {self.event.shot_id, self.state.shot_id, self.endpoint.shot_id}
        if len(identities) != 1:
            raise ValueError("archive unit crosses physical shot identities")
        if self.independent_unit_count != 1:
            raise ValueError("one MAST shot is exactly one independent unit")
        if self.state.role is MastShotRole.PROTECTED_EVALUATION:
            raise ValueError("protected shot endpoint cannot enter a development unit")


def assemble_development_unit(
    *,
    assignment: MastShotAssignment,
    materialization: FairMastMaterializationReceipt,
    event: MastEvent,
    state: MastCausalState,
    endpoint: MastEndpointResult,
) -> MastArchiveDevelopmentUnit:
    if assignment.shot_id != materialization.shot_id or assignment.shot_id != state.shot_id:
        raise ValueError("development unit shot identity differs")
    if assignment.shot_id != event.shot_id or assignment.shot_id != endpoint.shot_id:
        raise ValueError("development event/endpoint shot identity differs")
    return MastArchiveDevelopmentUnit(
        unit_id=f"mast-development-unit-{assignment.shot_id}",
        assignment=ObjectIdentity.from_record(assignment.assignment_id, assignment),
        materialization=ObjectIdentity.from_record(materialization.receipt_id, materialization),
        event=event,
        state=state,
        endpoint=endpoint,
        independent_unit_count=1,
    )


def build_protected_preaction_package(
    *, assignment: MastShotAssignment, state: MastCausalState
) -> MastProtectedPreActionPackage:
    if assignment.role is not MastShotRole.PROTECTED_EVALUATION:
        raise ValueError("protected package requires a protected assignment")
    if assignment.challenge_steward_id is None:
        raise ValueError("protected assignment has no challenge steward")
    if assignment.shot_id != state.shot_id:
        raise ValueError("protected assignment/state shot identity differs")
    return MastProtectedPreActionPackage(
        package_id=f"mast-protected-package-{assignment.shot_id}",
        assignment=ObjectIdentity.from_record(assignment.assignment_id, assignment),
        state=state,
        challenge_steward_id=assignment.challenge_steward_id,
        action_sealed=True,
        endpoint_sealed=True,
    )


__all__ = [
    "MastArchiveDevelopmentUnit",
    "assemble_development_unit",
    "build_protected_preaction_package",
]
