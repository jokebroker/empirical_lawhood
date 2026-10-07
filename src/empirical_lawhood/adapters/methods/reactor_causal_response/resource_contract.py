"""Injected, effectful process limit port; scientific code never sets OS limits."""

from dataclasses import dataclass
from datetime import datetime
from typing import ContextManager, Protocol, ClassVar
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.time import validate_utc_timestamp
from .config import EmpiricalRecipe


@dataclass(frozen=True, slots=True)
class EmpiricalRunAllocation(CanonicalRecord):
    """Retained whole-study start; recovery cannot restart its wall allocation."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-causal-response/empirical-run-allocation'
    allocation_id: str
    recipe: ObjectIdentity
    execution_authority: ObjectIdentity
    resource_envelope: ObjectIdentity
    started_at_utc: str
    deadline_utc: str

    def __post_init__(self) -> None:
        validate_stable_id(self.allocation_id)
        validate_utc_timestamp(self.started_at_utc)
        validate_utc_timestamp(self.deadline_utc)
        recipe = EmpiricalRecipe()
        if self.recipe != ObjectIdentity.from_record(recipe.config_id, recipe):
            raise ValueError("run allocation changed the frozen study recipe")
        if (
            self.execution_authority.object_schema
            != 'empirical-lawhood/planning/study-operation-authority'
            or self.resource_envelope.object_schema
            != 'empirical-lawhood/runtime/execution-resource-envelope-spec'
        ):
            raise ValueError("run allocation lacks exact production authority/envelope types")
        seconds = (
            datetime.fromisoformat(self.deadline_utc.replace("Z", "+00:00"))
            - datetime.fromisoformat(self.started_at_utc.replace("Z", "+00:00"))
        ).total_seconds()
        if seconds != recipe.wall_seconds:
            raise ValueError("whole-study wall allocation changed")


class EmpiricalResourceGuard(Protocol):
    def task(self, task_id: str) -> ContextManager[None]: ...
