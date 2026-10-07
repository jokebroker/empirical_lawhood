"""Target-native description and generator-independence records for target construct validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_stable_id,
)


@dataclass(frozen=True, slots=True)
class TargetConstructValidationTargetNativeDossier(CanonicalRecord):
    'Construct description written without structural recurrence role assignments.'

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-target-native-dossier'

    dossier_id: str
    target_id: str
    source_candidate: ObjectIdentity
    domain_id: str
    generator_family_id: str
    task_statement: str
    preparation_unit_id: str
    preparation_unit_definition: str
    causal_cutoff_definition: str
    native_action_ids: tuple[str, ...]
    hold_action_id: str
    receiver_ids: tuple[str, ...]
    receiver_direction_definition: str
    horizon_ids: tuple[str, ...]
    native_unit_ids: tuple[str, ...]
    native_frame_ids: tuple[str, ...]
    baseline_policy_id: str
    decisive_falsifier_ids: tuple[str, ...]
    requested_action_recorded: bool
    accepted_action_recorded: bool
    applied_action_recorded: bool
    realized_action_recorded: bool
    uses_structural_recurrence_role_vocabulary: bool
    protected_outcome_access_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in (
            "dossier_id",
            "target_id",
            "domain_id",
            "generator_family_id",
            "preparation_unit_id",
            "hold_action_id",
            "baseline_policy_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in (
            "task_statement",
            "preparation_unit_definition",
            "causal_cutoff_definition",
            "receiver_direction_definition",
        ):
            validate_nonempty(getattr(self, name), field_name=name)
        for name in (
            "native_action_ids",
            "receiver_ids",
            "horizon_ids",
            "native_unit_ids",
            "native_frame_ids",
            "decisive_falsifier_ids",
        ):
            require_sorted_unique_strings(
                getattr(self, name),
                field_name=name,
                allow_empty=False,
            )
        if self.hold_action_id not in self.native_action_ids:
            raise ValueError("the target-native action alphabet must contain hold")
        if not all(
            (
                self.requested_action_recorded,
                self.accepted_action_recorded,
                self.applied_action_recorded,
                self.realized_action_recorded,
            )
        ):
            raise ValueError("all four action stages must remain separately observable")
        if self.uses_structural_recurrence_role_vocabulary:
            raise ValueError('a native dossier cannot be authored in structural recurrence role vocabulary')
        if self.protected_outcome_access_count != 0:
            raise ValueError("native dossier authoring cannot access protected outcomes")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("native dossier authoring must remain outcome-blind")


@dataclass(frozen=True, slots=True)
class TargetConstructValidationIndependenceDossier(CanonicalRecord):
    """Outcome-blind evidence about source and generator independence."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-independence-dossier'

    dossier_id: str
    target_id: str
    source_candidate: ObjectIdentity
    generator_family_id: str
    authorship_evidence: str
    chronology_evidence: str
    solver_family_evidence: str
    generator_not_authored_for_construct_validation: bool
    generator_predates_construct_validation: bool
    domain_independent_of_other_target: bool
    solver_family_independent_of_other_target: bool
    generator_family_independent_of_other_target: bool
    direct_echo_of_structural_recurrence_fixture: bool
    protected_outcome_access_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("dossier_id", "target_id", "generator_family_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in (
            "authorship_evidence",
            "chronology_evidence",
            "solver_family_evidence",
        ):
            validate_nonempty(getattr(self, name), field_name=name)
        if self.protected_outcome_access_count != 0:
            raise ValueError("independence qualification cannot access protected outcomes")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("independence qualification must remain outcome-blind")
