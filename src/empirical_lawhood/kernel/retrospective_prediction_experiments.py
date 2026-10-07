"""Historical evaluation compatibility, without fresh biological evidence.

The inherited field vocabulary is an explicit operational compatibility map:
``development_unit_ids`` identifies analysis records, not physical replication;
``sealed_outcome_artifact_ids`` identifies software-withheld label ports, not
previously unseen outcomes. Their access and freshness fields state the latter
distinction explicitly. These schemas never decode as the prospective schemas.
"""

from dataclasses import dataclass, field
from typing import ClassVar

from .evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from .experiments import AssignmentKind, ExperimentSpec, RevealBarrierSpec
from .serialization import (
    require_sorted_unique_strings,
    validate_sha256,
    validate_stable_id,
)


@dataclass(frozen=True, slots=True)
class RetrospectiveEvaluationBarrier(RevealBarrierSpec):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/retrospective-evaluation-barrier'

    historical_visibility_ceiling: VisibilityCeiling = field(kw_only=True)

    def __post_init__(self) -> None:
        validate_stable_id(self.barrier_id, field_name="barrier_id")
        validate_stable_id(self.evaluation_cohort_id, field_name="evaluation_cohort_id")
        validate_sha256(self.evaluation_manifest_sha256, field_name="evaluation_manifest_sha256")
        for name, values in (
            ("development_unit_ids", self.development_unit_ids),
            ("sealed_outcome_artifact_ids", self.sealed_outcome_artifact_ids),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)
        if self.fresh_evidence:
            raise ValueError("historical evaluation cannot claim fresh evidence")
        if self.evaluation_outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
            raise ValueError("historical labels retain development-visible access")
        if self.historical_visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE:
            raise ValueError("historical evaluation must retain prior outcome exposure")


@dataclass(frozen=True, slots=True)
class RetrospectiveExperimentSpec(ExperimentSpec):
    """Read-only, nonpromotable historical prediction with explicit replication.

    The inherited independent-unit ID denotes the required unit *definition* in
    the prepared system. Only the explicit roster below records known instances;
    an empty roster means unknown replication, never one unit per table row.
    """

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/retrospective-experiment-spec'

    reveal_barrier: RetrospectiveEvaluationBarrier
    known_physical_independent_unit_ids: tuple[str, ...] = field(kw_only=True)
    new_biological_evidence: bool = field(kw_only=True, default=False)

    def __post_init__(self) -> None:
        ExperimentSpec.__post_init__(self)
        if type(self.reveal_barrier) is not RetrospectiveEvaluationBarrier:
            raise ValueError("historical experiment requires its exact historical barrier")
        require_sorted_unique_strings(
            self.known_physical_independent_unit_ids,
            field_name="known_physical_independent_unit_ids",
        )
        if self.new_biological_evidence:
            raise ValueError("historical reanalysis creates no new biological evidence")
        if self.assignment.kind is not AssignmentKind.OBSERVATIONAL:
            raise ValueError("historical prediction does not authorize interventions")
        if self.relation.action_quantity_ids or self.assignment.action_quantity_ids:
            raise ValueError("historical prediction cannot invent a native action chart")
        if self.obligations.support.physical_unit_count != len(
            self.known_physical_independent_unit_ids
        ):
            raise ValueError("historical support count differs from known physical units")

    def _validate_evidence_separation(self) -> None:
        if (
            self.design_visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE
            or self.evaluation_visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE
        ):
            raise ValueError("historical experiment cannot erase outcome exposure")
        for claim in self.claims:
            if (
                claim.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
                or claim.requested_rung is not None
                or claim.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
                or claim.visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE
            ):
                raise ValueError("historical claims must remain exposed and nonpromotable")
