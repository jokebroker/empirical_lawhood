"""Conditional-parent, sealed-evaluation and progress boundary records."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_sha256,
    validate_stable_id,
)


@dataclass(frozen=True, slots=True)
class TargetConstructValidationDevelopmentParent(CanonicalRecord):
    """Sole canonical science parent of a target evaluation package."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-development-parent'

    parent_id: str
    target_id: str
    source_binding: ObjectIdentity
    relation_binding: ObjectIdentity
    selected_map: ObjectIdentity
    selected_denominator: ObjectIdentity
    comparator_encodings: tuple[ObjectIdentity, ...]
    power_freeze: ObjectIdentity
    prediction_issue: ObjectIdentity
    primary_cell_ids: tuple[str, ...]
    development_complete_unit_ids: tuple[str, ...]
    evaluation_complete_unit_ids: tuple[str, ...]
    reserve_complete_unit_ids: tuple[str, ...]
    eligible_for_evaluation: bool
    evaluation_outcome_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("parent_id", "target_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_ids(
            self.comparator_encodings,
            attribute="object_id",
            field_name="comparator_encodings",
        )
        for name in (
            "primary_cell_ids",
            "development_complete_unit_ids",
            "evaluation_complete_unit_ids",
            "reserve_complete_unit_ids",
        ):
            require_sorted_unique_strings(
                getattr(self, name),
                field_name=name,
                allow_empty=name == "reserve_complete_unit_ids",
            )
        if not 6 <= len(self.primary_cell_ids) <= 24:
            raise ValueError("development parent must retain 6--24 primary cells")
        development = set(self.development_complete_unit_ids)
        evaluation = set(self.evaluation_complete_unit_ids)
        reserve = set(self.reserve_complete_unit_ids)
        if development & evaluation or development & reserve or evaluation & reserve:
            raise ValueError("development parent unit rosters overlap")
        if len(self.comparator_encodings) != 10:
            raise ValueError("development parent must carry the exact comparator family")
        if not self.eligible_for_evaluation:
            raise ValueError("an ineligible development result cannot parent evaluation")
        if self.evaluation_outcome_count:
            raise ValueError("development parent cannot access evaluation outcomes")
        if self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
            raise ValueError("development parent must retain development visibility")


@dataclass(frozen=True, slots=True)
class TargetConstructValidationEvaluationPackage(CanonicalRecord):
    """Conditional follow-up with one science parent and operational fields only."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-evaluation-package'

    package_id: str
    development_parent: TargetConstructValidationDevelopmentParent
    parent_receipt: ObjectIdentity
    authority_id: str
    issue_id: str
    run_id: str
    task_id: str
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("package_id", "authority_id", "issue_id", "run_id", "task_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.outcome_access is not OutcomeAccess.EVALUATION_SEALED:
            raise ValueError("evaluation generation must remain sealed")


def derive_evaluation_package(
    parent: TargetConstructValidationDevelopmentParent,
    *,
    parent_receipt: ObjectIdentity,
    authority_id: str,
    issue_id: str,
    run_id: str,
    task_id: str,
) -> TargetConstructValidationEvaluationPackage:
    """Derive evaluation without any target-specific science override slot."""

    return TargetConstructValidationEvaluationPackage(
        package_id=f"{parent.target_id}.evaluation-package",
        development_parent=parent,
        parent_receipt=parent_receipt,
        authority_id=authority_id,
        issue_id=issue_id,
        run_id=run_id,
        task_id=task_id,
        outcome_access=OutcomeAccess.EVALUATION_SEALED,
    )


@dataclass(frozen=True, slots=True)
class TargetConstructValidationSealedUnitResult(CanonicalRecord):
    """One generator result that exposes identity/custody but not outcome bytes."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-sealed-unit-result'

    result_id: str
    package: ObjectIdentity
    complete_unit_id: str
    task_id: str
    artifact: ObjectIdentity
    receipt: ObjectIdentity
    sealed_payload_sha256: str
    outcome_value_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("result_id", "complete_unit_id", "task_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_sha256(self.sealed_payload_sha256, field_name="sealed_payload_sha256")
        if self.outcome_value_count:
            raise ValueError("sealed generator record cannot expose outcome values")
        if self.outcome_access is not OutcomeAccess.EVALUATION_SEALED:
            raise ValueError("evaluation generators must remain sealed")


@dataclass(frozen=True, slots=True)
class TargetConstructValidationEvaluationReveal(CanonicalRecord):
    """Single evaluator barrier over the full issued evaluation roster."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-evaluation-reveal'

    reveal_id: str
    package: ObjectIdentity
    sealed_results: tuple[TargetConstructValidationSealedUnitResult, ...]
    expected_complete_unit_ids: tuple[str, ...]
    evaluator_id: str
    reveal_authority_id: str
    single_evaluator: bool
    all_generators_complete: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("reveal_id", "evaluator_id", "reveal_authority_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_ids(
            self.sealed_results,
            attribute="result_id",
            field_name="sealed_results",
        )
        require_sorted_unique_strings(
            self.expected_complete_unit_ids,
            field_name="expected_complete_unit_ids",
            allow_empty=False,
        )
        observed = tuple(sorted(value.complete_unit_id for value in self.sealed_results))
        if observed != self.expected_complete_unit_ids:
            raise ValueError("reveal barrier requires exact full evaluation fan-in")
        if any(value.package != self.package for value in self.sealed_results):
            raise ValueError("reveal barrier results cross evaluation packages")
        if not self.single_evaluator or not self.all_generators_complete:
            raise ValueError("reveal requires one evaluator after every generator completes")
        if self.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL:
            raise ValueError("only the evaluator may cross the reveal boundary")


class TargetConstructValidationProgressPhase(StrEnum):
    CANARY = "CANARY"
    DEVELOPMENT = "DEVELOPMENT"
    EVALUATION = "EVALUATION"


@dataclass(frozen=True, slots=True)
class TargetConstructValidationProgressRecord(CanonicalRecord):
    """Operational progress counters, explicitly distinct from scientific tau."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-progress-record'

    progress_id: str
    target_id: str
    task_id: str
    phase: TargetConstructValidationProgressPhase
    complete_units_finished: int
    complete_units_total: int
    elapsed_milliseconds: int
    cpu_milliseconds: int
    rss_high_water_bytes: int
    scratch_bytes: int
    scientific_horizon_value_count: int

    def __post_init__(self) -> None:
        for name in ("progress_id", "target_id", "task_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in (
            "complete_units_finished",
            "complete_units_total",
            "elapsed_milliseconds",
            "cpu_milliseconds",
            "rss_high_water_bytes",
            "scratch_bytes",
            "scientific_horizon_value_count",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.complete_units_finished > self.complete_units_total:
            raise ValueError("progress cannot exceed the issued complete-unit roster")
        if self.scientific_horizon_value_count:
            raise ValueError("operational progress cannot carry scientific horizon values")
