"""Fresh-evidence experiment, assignment, control and reveal contracts."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from .evidence import ClaimSpec, OutcomeAccess, VisibilityCeiling
from .obligations import ScientificObligations
from .serialization import (
    CanonicalRecord,
    ExtensionBinding,
    require_extensions,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_sha256,
    validate_stable_id,
)
from .status import ReadinessStatus
from .systems import RelationalIdentity, SystemSpec
from .time import InformationCutoff


class AssignmentKind(StrEnum):
    OBSERVATIONAL = "OBSERVATIONAL"
    LOGGED_INTERVENTION = "LOGGED_INTERVENTION"
    RANDOMIZED_INTERVENTION = "RANDOMIZED_INTERVENTION"
    SIMULATOR_INTERVENTION = "SIMULATOR_INTERVENTION"
    SHADOW_DECISION = "SHADOW_DECISION"


@dataclass(frozen=True, slots=True)
class AssignmentSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/assignment-spec'

    assignment_id: str
    kind: AssignmentKind
    independent_unit_id: str
    action_quantity_ids: tuple[str, ...]
    mechanism: str
    support_restriction_ids: tuple[str, ...]
    randomization_unit_id: str | None = None

    def __post_init__(self) -> None:
        validate_stable_id(self.assignment_id, field_name="assignment_id")
        validate_stable_id(self.independent_unit_id, field_name="independent_unit_id")
        require_sorted_unique_strings(
            self.action_quantity_ids,
            field_name="action_quantity_ids",
            allow_empty=self.kind is AssignmentKind.OBSERVATIONAL,
        )
        validate_nonempty(self.mechanism, field_name="mechanism")
        require_sorted_unique_strings(
            self.support_restriction_ids, field_name="support_restriction_ids"
        )
        if self.kind is AssignmentKind.RANDOMIZED_INTERVENTION:
            if self.randomization_unit_id is None:
                raise ValueError("randomized assignment requires a randomization unit")
        elif self.randomization_unit_id is not None:
            raise ValueError("only randomized assignment may name a randomization unit")
        if not self.action_quantity_ids and self.kind is not AssignmentKind.OBSERVATIONAL:
            raise ValueError("only observational assignment may omit an action chart")
        if self.randomization_unit_id is not None:
            validate_stable_id(self.randomization_unit_id, field_name="randomization_unit_id")
        if (
            self.kind
            in {
                AssignmentKind.LOGGED_INTERVENTION,
                AssignmentKind.OBSERVATIONAL,
            }
            and not self.support_restriction_ids
        ):
            raise ValueError("logged/observational assignment requires support limits")


class ControlKind(StrEnum):
    NEGATIVE_ACTION = "NEGATIVE_ACTION"
    WRONG_ACTION = "WRONG_ACTION"
    PLACEBO = "PLACEBO"
    PERSISTENCE = "PERSISTENCE"
    BASELINE_COMPARATOR = "BASELINE_COMPARATOR"
    TIME_REVERSAL = "TIME_REVERSAL"


@dataclass(frozen=True, slots=True)
class ControlSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/control-spec'

    control_id: str
    kind: ControlKind
    capability_key: str
    target_quantity_ids: tuple[str, ...]
    decisive_rule: str

    def __post_init__(self) -> None:
        validate_stable_id(self.control_id, field_name="control_id")
        validate_stable_id(self.capability_key, field_name="capability_key")
        require_sorted_unique_strings(
            self.target_quantity_ids,
            field_name="target_quantity_ids",
            allow_empty=False,
        )
        validate_nonempty(self.decisive_rule, field_name="decisive_rule")


@dataclass(frozen=True, slots=True)
class PrecisionGoal(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/precision-goal'

    goal_id: str
    metric_id: str
    target_width: Decimal
    native_unit: str
    maximum_independent_units: int
    stopping_rule: str

    def __post_init__(self) -> None:
        validate_stable_id(self.goal_id, field_name="goal_id")
        validate_stable_id(self.metric_id, field_name="metric_id")
        validate_decimal(self.target_width, field_name="target_width", minimum=Decimal("0"))
        if self.target_width == 0:
            raise ValueError("precision target width must be positive")
        validate_nonempty(self.native_unit, field_name="native_unit")
        if self.maximum_independent_units <= 0:
            raise ValueError("maximum_independent_units must be positive")
        validate_nonempty(self.stopping_rule, field_name="stopping_rule")


@dataclass(frozen=True, slots=True)
class RevealBarrierSpec(CanonicalRecord):
    """Outcome separation between development and fresh evaluation evidence."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/reveal-barrier-spec'

    barrier_id: str
    development_unit_ids: tuple[str, ...]
    evaluation_cohort_id: str
    evaluation_manifest_sha256: str
    sealed_outcome_artifact_ids: tuple[str, ...]
    evaluation_outcome_access: OutcomeAccess
    fresh_evidence: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.barrier_id, field_name="barrier_id")
        validate_stable_id(self.evaluation_cohort_id, field_name="evaluation_cohort_id")
        validate_sha256(
            self.evaluation_manifest_sha256,
            field_name="evaluation_manifest_sha256",
        )
        require_sorted_unique_strings(
            self.development_unit_ids,
            field_name="development_unit_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.sealed_outcome_artifact_ids,
            field_name="sealed_outcome_artifact_ids",
            allow_empty=False,
        )
        if self.evaluation_outcome_access is not OutcomeAccess.EVALUATION_SEALED:
            raise ValueError("a reveal barrier must keep evaluation outcomes sealed")
        if not self.fresh_evidence:
            raise ValueError("a prospective reveal barrier requires fresh evidence")


@dataclass(frozen=True, slots=True)
class ExperimentSpec(CanonicalRecord):
    """Immutable observation/intervention design prior to outcome reveal."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/experiment-spec'

    experiment_id: str
    system_id: str
    world_id: str
    relation: RelationalIdentity
    independent_unit_id: str
    claims: tuple[ClaimSpec, ...]
    assignment: AssignmentSpec
    measurement_quantity_ids: tuple[str, ...]
    controls: tuple[ControlSpec, ...]
    precision_goals: tuple[PrecisionGoal, ...]
    information_cutoffs: tuple[InformationCutoff, ...]
    reveal_barrier: RevealBarrierSpec
    obligations: ScientificObligations
    design_visibility_ceiling: VisibilityCeiling
    evaluation_visibility_ceiling: VisibilityCeiling
    authority_policy_id: str
    readiness: ReadinessStatus
    authorization_record_id: str | None = None
    predecessor_experiment_ids: tuple[str, ...] = ()
    extensions: tuple[ExtensionBinding, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("experiment_id", self.experiment_id),
            ("system_id", self.system_id),
            ("world_id", self.world_id),
            ("independent_unit_id", self.independent_unit_id),
            ("authority_policy_id", self.authority_policy_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_ids(self.claims, attribute="claim_id", field_name="claims")
        if not self.claims:
            raise ValueError("an experiment requires at least one claim")
        require_sorted_unique_strings(
            self.measurement_quantity_ids,
            field_name="measurement_quantity_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(self.controls, attribute="control_id", field_name="controls")
        if not self.controls:
            raise ValueError("an experiment requires predeclared controls")
        require_sorted_unique_ids(
            self.precision_goals, attribute="goal_id", field_name="precision_goals"
        )
        if not self.precision_goals:
            raise ValueError("an experiment requires precision/stopping goals")
        require_sorted_unique_ids(
            self.information_cutoffs,
            attribute="cutoff_id",
            field_name="information_cutoffs",
        )
        if not self.information_cutoffs:
            raise ValueError("an experiment requires causal cutoffs")
        require_sorted_unique_strings(
            self.predecessor_experiment_ids,
            field_name="predecessor_experiment_ids",
        )
        self._validate_relational_bindings()
        self._validate_evidence_separation()
        self._validate_authorization_state()
        require_extensions(self.extensions)

    def _validate_relational_bindings(self) -> None:
        if self.assignment.independent_unit_id != self.independent_unit_id:
            raise ValueError("assignment and experiment independent units differ")
        if self.assignment.action_quantity_ids != self.relation.action_quantity_ids:
            raise ValueError("assignment actions differ from the relational action chart")
        if not set(self.relation.receiver_quantity_ids).issubset(self.measurement_quantity_ids):
            raise ValueError("experiment measurements omit a relational receiver")
        if self.obligations.support.relation_id != self.relation.relation_id:
            raise ValueError("support and experiment relation identities differ")
        if self.obligations.support.independent_unit_id != self.independent_unit_id:
            raise ValueError("support and experiment independent units differ")
        if self.obligations.closure.retained_history_ids != (self.relation.history_quantity_ids):
            raise ValueError("closure history differs from the relational identity")
        for claim in self.claims:
            if claim.world_id != self.world_id:
                raise ValueError("experiment claim world differs")
            if claim.relation_id != self.relation.relation_id:
                raise ValueError("experiment claim relation differs")
            if claim.physical_independent_unit_id != self.independent_unit_id:
                raise ValueError("experiment claim independent unit differs")

    def _validate_evidence_separation(self) -> None:
        if type(self.reveal_barrier) is not RevealBarrierSpec:
            raise ValueError("prospective experiment requires its original reveal barrier")
        if self.evaluation_visibility_ceiling is not VisibilityCeiling.PROSPECTIVE:
            raise ValueError("fresh evaluation visibility must be prospective")
        for claim in self.claims:
            if claim.outcome_access is not OutcomeAccess.EVALUATION_SEALED:
                raise ValueError("experiment claims must bind sealed evaluation outcomes")
            if claim.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE:
                raise ValueError("fresh-evidence claims must begin prospectively")

    def _validate_authorization_state(self) -> None:
        if self.authorization_record_id is None:
            if self.readiness is ReadinessStatus.READY:
                raise ValueError("an unauthorized experiment cannot be ready")
            return
        validate_stable_id(self.authorization_record_id, field_name="authorization_record_id")
        if self.readiness is ReadinessStatus.AUTHORITY_REQUIRED:
            raise ValueError("an authorized experiment cannot require authority")


def validate_experiment_against_system(experiment: ExperimentSpec, system: SystemSpec) -> None:
    if experiment.system_id != system.system_id:
        raise ValueError("experiment binds the wrong system")
    if experiment.world_id != system.world.world_id:
        raise ValueError("experiment binds the wrong evidence world")
    if experiment.relation != system.relation:
        raise ValueError("experiment relation differs from the prepared system")
    if experiment.independent_unit_id != system.independent_unit.unit_id:
        raise ValueError("experiment physical independent unit differs")
    if experiment.authority_policy_id != system.authority_policy.policy_id:
        raise ValueError("experiment authority policy differs from the system")
    quantities = {quantity.quantity_id for quantity in system.quantities}
    if not set(experiment.measurement_quantity_ids).issubset(quantities):
        raise ValueError("experiment measurements include unknown quantities")
    clocks = {clock.clock_id for clock in system.clocks}
    if any(cutoff.clock_id not in clocks for cutoff in experiment.information_cutoffs):
        raise ValueError("experiment cutoff uses an unknown clock")
    known_views = {view.view_id for view in system.numerical_views}
    for claim in experiment.claims:
        system.validate_claim(claim)
        if not set(claim.numerical_view_ids).issubset(known_views):
            raise ValueError("experiment claim uses an unknown numerical view")
