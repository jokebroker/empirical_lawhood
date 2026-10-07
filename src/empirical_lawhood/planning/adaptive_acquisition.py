"""Static fixed-round acquisition authoring; selectors never execute tasks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)


class FixedRoundStopRule(StrEnum):
    EXHAUST_ROUNDS = "EXHAUST_ROUNDS"
    PREDECLARED_CHECKPOINT = "PREDECLARED_CHECKPOINT"


class AcquisitionCausalDomain(StrEnum):
    ACQUISITION = "ACQUISITION"


@dataclass(frozen=True, slots=True)
class FixedRoundArmSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/fixed-round-arm-spec'

    arm_id: str
    query: ObjectIdentity
    maximum_selections: int

    def __post_init__(self) -> None:
        validate_stable_id(self.arm_id, field_name="arm_id")
        if self.maximum_selections < 1:
            raise ValueError("fixed-round arm maximum must be positive")


@dataclass(frozen=True, slots=True)
class FixedRoundAcquisitionPlan(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/fixed-round-acquisition-plan'

    acquisition_plan_id: str
    arms: tuple[FixedRoundArmSpec, ...]
    maximum_rounds: int
    selections_per_round: int
    selector_capability_key: str
    selector_capability_version: str
    selector_config_sha256: str
    selector_implementation_sha256: str
    stop_rule: FixedRoundStopRule
    checkpoint_ids: tuple[str, ...]
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.acquisition_plan_id, field_name="acquisition_plan_id")
        validate_stable_id(
            self.selector_capability_key,
            field_name="selector_capability_key",
        )
        validate_semantic_version(self.selector_capability_version)
        validate_sha256(self.selector_config_sha256, field_name="selector_config_sha256")
        validate_sha256(
            self.selector_implementation_sha256,
            field_name="selector_implementation_sha256",
        )
        require_sorted_unique_ids(self.arms, attribute="arm_id", field_name="arms")
        require_sorted_unique_strings(self.checkpoint_ids, field_name="checkpoint_ids")
        if not self.arms or self.maximum_rounds < 1 or self.selections_per_round < 1:
            raise ValueError("fixed-round acquisition dimensions must be positive")
        if self.selections_per_round > len(self.arms):
            raise ValueError("fixed-round selection width exceeds the frozen arm roster")
        if self.checkpoint_ids and len(self.checkpoint_ids) != self.maximum_rounds:
            raise ValueError("checkpoint roster must cover every static round")
        if self.stop_rule is FixedRoundStopRule.PREDECLARED_CHECKPOINT and not self.checkpoint_ids:
            raise ValueError("checkpoint stop requires a frozen checkpoint roster")
        if self.outcome_access not in {
            OutcomeAccess.OUTCOME_BLIND,
            OutcomeAccess.DEVELOPMENT_VISIBLE,
        }:
            raise ValueError("fixed-round acquisition cannot inspect evaluation outcomes")
        if not self.visibility_ceiling.is_promotable:
            raise ValueError("outcome-visible acquisition cannot author promotable evidence")


@dataclass(frozen=True, slots=True)
class AcquisitionQueryCoordinate(CanonicalRecord):
    """One statically issued task/arm/preparation/action/member query cell."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/acquisition-query-coordinate'

    coordinate_id: str
    query_bundle_id: str
    task_id: str
    arm_id: str
    preparation_unit_id: str
    action_word: ObjectIdentity
    model_member_id: str
    causal_domain: AcquisitionCausalDomain
    maximum_episode_charge: int

    def __post_init__(self) -> None:
        for name, value in (
            ("coordinate_id", self.coordinate_id),
            ("query_bundle_id", self.query_bundle_id),
            ("task_id", self.task_id),
            ("arm_id", self.arm_id),
            ("preparation_unit_id", self.preparation_unit_id),
            ("model_member_id", self.model_member_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.action_word.object_schema != OccurrenceActionWord.SCHEMA:
            raise ValueError("acquisition query requires an exact ActionWord")
        if self.causal_domain is not AcquisitionCausalDomain.ACQUISITION:
            raise ValueError("bounded queries must remain in the acquisition domain")
        if self.maximum_episode_charge != 1:
            raise ValueError("one acquisition coordinate must charge exactly one episode")

    @property
    def action_word_id(self) -> str:
        return self.action_word.object_id

    @property
    def scientific_key(self) -> tuple[str, str, str, str, str]:
        return (
            self.task_id,
            self.arm_id,
            self.preparation_unit_id,
            self.action_word_id,
            self.model_member_id,
        )


@dataclass(frozen=True, slots=True)
class BoundedQueryAcquisitionPlan(CanonicalRecord):
    """Two-round static maximum graph with adaptive bundle selection only."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/bounded-query-acquisition-plan'

    acquisition_plan_id: str
    task_ids: tuple[str, ...]
    arm_ids: tuple[str, ...]
    action_words: tuple[ObjectIdentity, ...]
    model_member_ids: tuple[str, ...]
    common_panel_coordinates: tuple[AcquisitionQueryCoordinate, ...]
    query_coordinates: tuple[AcquisitionQueryCoordinate, ...]
    query_bundle_priority_ids: tuple[str, ...]
    maximum_rounds: int
    maximum_query_bundles: int
    maximum_episode_charge: int
    minimum_distinct_preparations_per_active_action: int
    matched_hold_action_word: ObjectIdentity
    selector_capability_key: str
    selector_capability_version: str
    selector_config_sha256: str
    selector_implementation_sha256: str
    sufficiency_rule_id: str
    information_cutoff_id: str
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        for name, value in (
            ("acquisition_plan_id", self.acquisition_plan_id),
            ("selector_capability_key", self.selector_capability_key),
            ("sufficiency_rule_id", self.sufficiency_rule_id),
            ("information_cutoff_id", self.information_cutoff_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_semantic_version(self.selector_capability_version)
        validate_sha256(self.selector_config_sha256, field_name="selector_config_sha256")
        validate_sha256(
            self.selector_implementation_sha256,
            field_name="selector_implementation_sha256",
        )
        for name, values in (
            ("task_ids", self.task_ids),
            ("arm_ids", self.arm_ids),
            ("model_member_ids", self.model_member_ids),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)
        if len(self.task_ids) != 1 or len(self.arm_ids) != 1:
            raise ValueError("bounded query owns exactly one task/arm static graph")
        require_sorted_unique_ids(
            self.action_words,
            attribute="object_id",
            field_name="action_words",
        )
        if not self.action_words or any(
            value.object_schema != OccurrenceActionWord.SCHEMA for value in self.action_words
        ):
            raise ValueError("bounded-query action roster requires exact ActionWords")
        if self.matched_hold_action_word.object_schema != OccurrenceActionWord.SCHEMA:
            raise ValueError("bounded-query matched HOLD requires an exact ActionWord")
        require_sorted_unique_ids(
            self.common_panel_coordinates,
            attribute="coordinate_id",
            field_name="common_panel_coordinates",
        )
        require_sorted_unique_ids(
            self.query_coordinates,
            attribute="coordinate_id",
            field_name="query_coordinates",
        )
        require_sorted_unique_strings(
            self.query_bundle_priority_ids,
            field_name="query_bundle_priority_ids",
            allow_empty=False,
        )
        if self.maximum_rounds != 2:
            raise ValueError("bounded query requires exactly two static rounds")
        if not 1 <= self.maximum_query_bundles <= self.maximum_rounds:
            raise ValueError("bounded-query bundle ceiling is invalid")
        if self.minimum_distinct_preparations_per_active_action != 2:
            raise ValueError("bounded query requires two active-action preparations")
        action_word_ids = {value.object_id for value in self.action_words}
        if self.matched_hold_action_word.object_id not in action_word_ids:
            raise ValueError("bounded-query matched HOLD is outside the action roster")
        common_bundles = {value.query_bundle_id for value in self.common_panel_coordinates}
        if len(common_bundles) != 4:
            raise ValueError("bounded query requires exactly four common-panel bundles")
        common_preparations = {value.preparation_unit_id for value in self.common_panel_coordinates}
        if len(common_preparations) != 2:
            raise ValueError("common panel requires exactly two acquisition preparations")
        for bundle_id in common_bundles:
            cells = tuple(
                value
                for value in self.common_panel_coordinates
                if value.query_bundle_id == bundle_id
            )
            bundle_axes = {
                (
                    value.task_id,
                    value.arm_id,
                    value.preparation_unit_id,
                    value.action_word,
                )
                for value in cells
            }
            if (
                len(bundle_axes) != 1
                or {value.model_member_id for value in cells} != set(self.model_member_ids)
                or len(cells) != len(self.model_member_ids)
            ):
                raise ValueError(
                    "one common bundle must be one preparation/action over every member"
                )
        hold_preparations = {
            value.preparation_unit_id
            for value in self.common_panel_coordinates
            if value.action_word == self.matched_hold_action_word
        }
        active_pairs = {
            (value.preparation_unit_id, value.action_word)
            for value in self.common_panel_coordinates
            if value.action_word != self.matched_hold_action_word
        }
        if (
            hold_preparations != common_preparations
            or len(active_pairs) != 2
            or len({value[0] for value in active_pairs}) != 1
            or len({value[1] for value in active_pairs}) != 2
        ):
            raise ValueError(
                "common panel must be recurrent HOLD plus two distinct first-preparation probes"
            )
        bundles = {value.query_bundle_id for value in self.query_coordinates}
        if common_bundles & bundles:
            raise ValueError("common and adaptive acquisition bundle identities overlap")
        if set(self.query_bundle_priority_ids) != bundles:
            raise ValueError("bounded-query priority differs from the exact bundle roster")
        if len(bundles) < self.maximum_query_bundles:
            raise ValueError("bounded-query bundle ceiling exceeds the issued bundle roster")
        expected_episode_charge = self.maximum_query_bundles * len(self.model_member_ids)
        if self.maximum_episode_charge != expected_episode_charge:
            raise ValueError("bounded-query episode ceiling differs from two complete bundles")
        all_coordinates = (*self.common_panel_coordinates, *self.query_coordinates)
        scientific_keys = {value.scientific_key for value in all_coordinates}
        if len(scientific_keys) != len(all_coordinates):
            raise ValueError("bounded-query roster duplicates a scientific query coordinate")
        if any(
            value.task_id not in self.task_ids
            or value.arm_id not in self.arm_ids
            or value.action_word not in self.action_words
            or value.model_member_id not in self.model_member_ids
            for value in all_coordinates
        ):
            raise ValueError("bounded-query coordinate is outside a declared axis")
        for bundle_id in self.query_bundle_priority_ids:
            cells = tuple(
                value for value in self.query_coordinates if value.query_bundle_id == bundle_id
            )
            bundle_axes = {
                (
                    value.task_id,
                    value.arm_id,
                    value.preparation_unit_id,
                    value.action_word,
                )
                for value in cells
            }
            if (
                len(bundle_axes) != 1
                or {value.model_member_id for value in cells} != set(self.model_member_ids)
                or len(cells) != len(self.model_member_ids)
            ):
                raise ValueError(
                    "one query bundle must be one task/arm/preparation/action over every member"
                )
        if self.outcome_access not in {
            OutcomeAccess.OUTCOME_BLIND,
            OutcomeAccess.DEVELOPMENT_VISIBLE,
        }:
            raise ValueError("bounded-query plan cannot inspect evaluation outcomes")
        if not self.visibility_ceiling.is_promotable:
            raise ValueError("bounded-query outcome visibility cannot promote evidence")

    @property
    def matched_hold_action_word_id(self) -> str:
        return self.matched_hold_action_word.object_id


@dataclass(frozen=True, slots=True)
class ActionPreparationRecurrenceQualificationRequirement(CanonicalRecord):
    "Frozen action/preparation recurrence roster that must precede one law finalization."

    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/planning/action-preparation-recurrence-qualification-requirement'
    )

    requirement_id: str
    task_id: str
    arm_id: str
    active_action_words: tuple[ObjectIdentity, ...]
    matched_hold_action_word: ObjectIdentity
    model_member_ids: tuple[str, ...]
    method_config: ObjectIdentity
    proof_owner: ObjectIdentity

    def __post_init__(self) -> None:
        for name, value in (
            ("requirement_id", self.requirement_id),
            ("task_id", self.task_id),
            ("arm_id", self.arm_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_ids(
            self.active_action_words,
            attribute="object_id",
            field_name="active_action_words",
        )
        if not self.active_action_words or any(
            value.object_schema != OccurrenceActionWord.SCHEMA for value in self.active_action_words
        ):
            raise ValueError("recurrence qualification requires exact active ActionWords")
        if self.matched_hold_action_word.object_schema != OccurrenceActionWord.SCHEMA:
            raise ValueError("recurrence qualification requires an exact HOLD ActionWord")
        if self.matched_hold_action_word in self.active_action_words:
            raise ValueError("recurrence qualification cannot claim HOLD as active")
        require_sorted_unique_strings(
            self.model_member_ids,
            field_name="model_member_ids",
            allow_empty=False,
        )
        if self.proof_owner.object_schema != (
            'empirical-lawhood/runtime/action-preparation-recurrence-proof-owner'
        ):
            raise ValueError("recurrence qualification requires its exact proof owner")

    @property
    def active_action_word_ids(self) -> tuple[str, ...]:
        return tuple(value.object_id for value in self.active_action_words)


__all__ = [
    "AcquisitionCausalDomain",
    "AcquisitionQueryCoordinate",
    'ActionPreparationRecurrenceQualificationRequirement',
    'BoundedQueryAcquisitionPlan',
    "FixedRoundAcquisitionPlan",
    "FixedRoundArmSpec",
    "FixedRoundStopRule",
]
