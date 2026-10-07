"""Additive action, terminal-axis and effect truth for evidence projections."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.action_contracts import ActionOccurrence, OccurrenceActionWord
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_schema,
    validate_stable_id,
)

from .identification_evidence import (
    IdentificationEvidenceManifest,
    IdentificationEvidenceProjection,
)


MAX_EPISODES = 100_000
MAX_NATIVE_OPERANDS_PER_EPISODE = 64


class PhysicalSinkAxis(StrEnum):
    NONE = "NONE"
    SINK_TERMINATED = "SINK_TERMINATED"
    SAFETY_STOP = "SAFETY_STOP"


class ObservationValidityAxis(StrEnum):
    VALID = "VALID"
    INVALID = "INVALID"
    MISSING = "MISSING"


class NumericalValidityAxis(StrEnum):
    VALID = "VALID"
    INVALID = "INVALID"
    NOT_EVALUATED = "NOT_EVALUATED"


class AuthorityAxis(StrEnum):
    NOT_REQUIRED = "NOT_REQUIRED"
    AUTHORIZED = "AUTHORIZED"
    REFUSED = "REFUSED"
    NOT_PRESENTED = "NOT_PRESENTED"


class ScientificTerminalAxis(StrEnum):
    OBSERVED = "OBSERVED"
    NEGATIVE = "NEGATIVE"
    UNEVALUABLE = "UNEVALUABLE"
    CONDITION_FALSE = "CONDITION_FALSE"
    NONATTEMPT = "NONATTEMPT"


class EpisodeTerminalReasonCode(StrEnum):
    EPISODE_OBSERVED = "EPISODE_OBSERVED"
    SCIENTIFIC_NEGATIVE = "SCIENTIFIC_NEGATIVE"
    PHYSICAL_SINK = "PHYSICAL_SINK"
    SAFETY_STOP = "SAFETY_STOP"
    OBSERVATION_INVALID = "OBSERVATION_INVALID"
    OBSERVATION_MISSING = "OBSERVATION_MISSING"
    NUMERICAL_INVALID = "NUMERICAL_INVALID"
    AUTHORITY_REFUSED = "AUTHORITY_REFUSED"
    AUTHORITY_NOT_PRESENTED = "AUTHORITY_NOT_PRESENTED"
    UPSTREAM_CONDITION_FALSE = "UPSTREAM_CONDITION_FALSE"
    EFFECT_UNEVALUABLE = "EFFECT_UNEVALUABLE"


class ComputabilityEffectStatus(StrEnum):
    REPRESENTED = "REPRESENTED"
    UNRESOLVED = "UNRESOLVED"
    ASSUMPTION_CLOSED = "ASSUMPTION_CLOSED"
    INAPPLICABLE = "INAPPLICABLE"
    OBSERVATION_ONLY = "OBSERVATION_ONLY"


class EffectReasonCode(StrEnum):
    EFFECT_REPRESENTED = "EFFECT_REPRESENTED"
    EFFECT_UNRESOLVED = "EFFECT_UNRESOLVED"
    EFFECT_ASSUMPTION_CLOSED = "EFFECT_ASSUMPTION_CLOSED"
    EFFECT_INAPPLICABLE = "EFFECT_INAPPLICABLE"
    EFFECT_OBSERVATION_ONLY = "EFFECT_OBSERVATION_ONLY"


class ProjectionCompatibilityStatus(StrEnum):
    EXACT = "EXACT"
    COMPATIBLE_WITH_DECLARED_LOSS = "COMPATIBLE_WITH_DECLARED_LOSS"
    INCOMPATIBLE_REQUIRED_FIELD = "INCOMPATIBLE_REQUIRED_FIELD"


class IdentificationMethodEvidenceKind(StrEnum):
    POINT_METHOD = "POINT_METHOD"
    FINITE_ACTION = "FINITE_ACTION"
    CONTROLLED_IO = "CONTROLLED_IO"


@dataclass(frozen=True, slots=True)
class ActionOccurrenceBinding(CanonicalRecord):
    """One explicit four-stage occurrence attached to one projected episode."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/action-occurrence-binding'

    binding_id: str
    episode_id: str
    projected_observation_id: str
    action_word: ObjectIdentity
    occurrence: ActionOccurrence
    stage_receipt_ids: tuple[str, str, str, str]
    stage_clock_binding_ids: tuple[str, str, str, str]

    def __post_init__(self) -> None:
        for name, value in (
            ("binding_id", self.binding_id),
            ("episode_id", self.episode_id),
            ("projected_observation_id", self.projected_observation_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.action_word.object_schema != OccurrenceActionWord.SCHEMA:
            raise ValueError("action occurrence binding requires the current ActionWord")
        for receipt_id in self.stage_receipt_ids:
            validate_stable_id(receipt_id, field_name="stage_receipt_ids")
        if len(set(self.stage_receipt_ids)) != 4:
            raise ValueError("action stages require four distinct receipt identities")
        for clock_binding_id in self.stage_clock_binding_ids:
            validate_stable_id(clock_binding_id, field_name="stage_clock_binding_ids")
        if len(set(self.stage_clock_binding_ids)) != 4:
            raise ValueError("action stages require four distinct clock-binding identities")
        events = (
            self.occurrence.requested,
            self.occurrence.accepted,
            self.occurrence.applied,
            self.occurrence.realized,
        )
        if len({value.quantity_id for value in events}) != 4:
            raise ValueError("action stages require four explicit quantity identities")
        # A native medium may legitimately use one clock for all four stages
        # (direct TORAX does).  Distinct stage-clock bindings retain that fact
        # without inventing four native clocks.


@dataclass(frozen=True, slots=True)
class EpisodeTerminalDisposition(CanonicalRecord):
    """Orthogonal terminal axes plus bounded native operands."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/episode-terminal-disposition'

    episode_id: str
    projected_observation_id: str
    physical_independent_unit_id: str
    physical_sink: PhysicalSinkAxis
    observation_validity: ObservationValidityAxis
    numerical_validity: NumericalValidityAxis
    authority: AuthorityAxis
    scientific_status: ScientificTerminalAxis
    sink_operands: tuple[NamedDecimal, ...]
    score_operands: tuple[NamedDecimal, ...]
    reason_codes: tuple[EpisodeTerminalReasonCode, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("episode_id", self.episode_id),
            ("projected_observation_id", self.projected_observation_id),
            ("physical_independent_unit_id", self.physical_independent_unit_id),
        ):
            validate_stable_id(value, field_name=name)
        for name, values in (
            ("sink_operands", self.sink_operands),
            ("score_operands", self.score_operands),
        ):
            require_sorted_unique_ids(values, attribute="value_id", field_name=name)
            if len(values) > MAX_NATIVE_OPERANDS_PER_EPISODE:
                raise ValueError(f"{name} exceeds its bounded operand count")
        if {value.value_id for value in self.sink_operands} & {
            value.value_id for value in self.score_operands
        }:
            raise ValueError("sink and score operands must remain distinct")
        if self.physical_sink is PhysicalSinkAxis.NONE:
            if self.sink_operands:
                raise ValueError("non-sink episode cannot carry sink operands")
        elif not self.sink_operands:
            raise ValueError("physical/safety sink must retain its native operands")
        invalid_score = (
            self.observation_validity is not ObservationValidityAxis.VALID
            or self.numerical_validity is not NumericalValidityAxis.VALID
            or self.authority in {AuthorityAxis.REFUSED, AuthorityAxis.NOT_PRESENTED}
            or self.physical_sink is not PhysicalSinkAxis.NONE
        )
        if invalid_score and self.score_operands:
            raise ValueError("invalid or nonattempted episode cannot carry score operands")
        if self.scientific_status in {
            ScientificTerminalAxis.OBSERVED,
            ScientificTerminalAxis.NEGATIVE,
        }:
            if invalid_score or not self.score_operands:
                raise ValueError("observed/negative episode requires valid score operands")
        if self.authority in {AuthorityAxis.REFUSED, AuthorityAxis.NOT_PRESENTED}:
            if self.scientific_status is not ScientificTerminalAxis.NONATTEMPT:
                raise ValueError("absent authority must remain a scientific nonattempt")
        require_sorted_unique_strings(
            tuple(value.value for value in self.reason_codes),
            field_name="reason_codes",
            allow_empty=False,
        )
        required_reasons = {
            EpisodeTerminalReasonCode.PHYSICAL_SINK
            if self.physical_sink is PhysicalSinkAxis.SINK_TERMINATED
            else EpisodeTerminalReasonCode.SAFETY_STOP
            if self.physical_sink is PhysicalSinkAxis.SAFETY_STOP
            else None,
            EpisodeTerminalReasonCode.OBSERVATION_INVALID
            if self.observation_validity is ObservationValidityAxis.INVALID
            else EpisodeTerminalReasonCode.OBSERVATION_MISSING
            if self.observation_validity is ObservationValidityAxis.MISSING
            else None,
            EpisodeTerminalReasonCode.NUMERICAL_INVALID
            if self.numerical_validity is NumericalValidityAxis.INVALID
            else None,
            EpisodeTerminalReasonCode.AUTHORITY_REFUSED
            if self.authority is AuthorityAxis.REFUSED
            else EpisodeTerminalReasonCode.AUTHORITY_NOT_PRESENTED
            if self.authority is AuthorityAxis.NOT_PRESENTED
            else None,
            EpisodeTerminalReasonCode.UPSTREAM_CONDITION_FALSE
            if self.scientific_status is ScientificTerminalAxis.CONDITION_FALSE
            else None,
            EpisodeTerminalReasonCode.EPISODE_OBSERVED
            if self.scientific_status is ScientificTerminalAxis.OBSERVED
            else EpisodeTerminalReasonCode.SCIENTIFIC_NEGATIVE
            if self.scientific_status is ScientificTerminalAxis.NEGATIVE
            else None,
        } - {None}
        if not required_reasons.issubset(set(self.reason_codes)):
            raise ValueError("terminal axes are missing their typed terminal reasons")


@dataclass(frozen=True, slots=True)
class ComputabilityEffectCoordinate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/computability-effect-coordinate'

    coordinate_id: str
    effect_id: str
    denominator_member_id: str
    numerical_view_id: str

    def __post_init__(self) -> None:
        for name, value in (
            ("coordinate_id", self.coordinate_id),
            ("effect_id", self.effect_id),
            ("denominator_member_id", self.denominator_member_id),
            ("numerical_view_id", self.numerical_view_id),
        ):
            validate_stable_id(value, field_name=name)


@dataclass(frozen=True, slots=True)
class ComputabilityEffectEntry(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/computability-effect-entry'

    entry_id: str
    coordinate: ComputabilityEffectCoordinate
    status: ComputabilityEffectStatus
    assumption_ids: tuple[str, ...]
    reason_code: EffectReasonCode

    def __post_init__(self) -> None:
        validate_stable_id(self.entry_id, field_name="entry_id")
        require_sorted_unique_strings(self.assumption_ids, field_name="assumption_ids")
        expected = {
            ComputabilityEffectStatus.REPRESENTED: EffectReasonCode.EFFECT_REPRESENTED,
            ComputabilityEffectStatus.UNRESOLVED: EffectReasonCode.EFFECT_UNRESOLVED,
            ComputabilityEffectStatus.ASSUMPTION_CLOSED: (
                EffectReasonCode.EFFECT_ASSUMPTION_CLOSED
            ),
            ComputabilityEffectStatus.INAPPLICABLE: EffectReasonCode.EFFECT_INAPPLICABLE,
            ComputabilityEffectStatus.OBSERVATION_ONLY: (EffectReasonCode.EFFECT_OBSERVATION_ONLY),
        }
        if self.reason_code is not expected[self.status]:
            raise ValueError("computability effect reason differs from its status")
        if self.status is ComputabilityEffectStatus.ASSUMPTION_CLOSED:
            if not self.assumption_ids:
                raise ValueError("assumption-closed effect requires explicit assumptions")
        elif self.assumption_ids:
            raise ValueError("only assumption-closed effects may bind assumptions")


@dataclass(frozen=True, slots=True)
class IdentificationEvidenceProjectionExtension(CanonicalRecord):
    """Exact companion to one immutable current evidence projection."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/identification-evidence-projection-extension'

    extension_id: str
    projection: ObjectIdentity
    manifest: ObjectIdentity
    entered_episode_ids: tuple[str, ...]
    independent_unit_ids: tuple[str, ...]
    action_occurrences: tuple[ActionOccurrenceBinding, ...]
    terminal_dispositions: tuple[EpisodeTerminalDisposition, ...]
    declared_effect_coordinates: tuple[ComputabilityEffectCoordinate, ...]
    computability_entries: tuple[ComputabilityEffectEntry, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.extension_id, field_name="extension_id")
        if self.projection.object_schema != IdentificationEvidenceProjection.SCHEMA:
            raise ValueError("projection extension binds another projection schema")
        if self.manifest.object_schema != IdentificationEvidenceManifest.SCHEMA:
            raise ValueError("projection extension binds another manifest schema")
        require_sorted_unique_strings(
            self.entered_episode_ids, field_name="entered_episode_ids", allow_empty=False
        )
        if len(self.entered_episode_ids) > MAX_EPISODES:
            raise ValueError("projection extension exceeds its episode bound")
        require_sorted_unique_strings(
            self.independent_unit_ids, field_name="independent_unit_ids", allow_empty=False
        )
        require_sorted_unique_ids(
            self.action_occurrences, attribute="binding_id", field_name="action_occurrences"
        )
        require_sorted_unique_ids(
            self.terminal_dispositions,
            attribute="episode_id",
            field_name="terminal_dispositions",
        )
        require_sorted_unique_ids(
            self.declared_effect_coordinates,
            attribute="coordinate_id",
            field_name="declared_effect_coordinates",
        )
        require_sorted_unique_ids(
            self.computability_entries, attribute="entry_id", field_name="computability_entries"
        )
        episodes = set(self.entered_episode_ids)
        if {value.episode_id for value in self.terminal_dispositions} != episodes:
            raise ValueError("terminal dispositions must cover every entered episode exactly once")
        if {value.episode_id for value in self.action_occurrences} != episodes:
            raise ValueError("every entered episode requires explicit complete action occurrence")
        coordinates = {
            (
                value.coordinate.effect_id,
                value.coordinate.denominator_member_id,
                value.coordinate.numerical_view_id,
            )
            for value in self.computability_entries
        }
        declared = {
            (value.effect_id, value.denominator_member_id, value.numerical_view_id)
            for value in self.declared_effect_coordinates
        }
        if len(coordinates) != len(self.computability_entries) or coordinates != declared:
            raise ValueError("computability entries must cover the declared effect product exactly")
        declared_by_id = {value.coordinate_id: value for value in self.declared_effect_coordinates}
        if any(
            declared_by_id.get(value.coordinate.coordinate_id) != value.coordinate
            for value in self.computability_entries
        ):
            raise ValueError("computability entry changes its declared coordinate")
        if not {
            value.physical_independent_unit_id for value in self.terminal_dispositions
        }.issubset(set(self.independent_unit_ids)):
            raise ValueError("nested views cannot be borrowed as independent units")
        observation_by_episode = {
            value.episode_id: value.projected_observation_id for value in self.terminal_dispositions
        }
        if any(
            observation_by_episode[value.episode_id] != value.projected_observation_id
            for value in self.action_occurrences
        ):
            raise ValueError("action and terminal truth bind different projected observations")
        occurrence_coordinates = {
            (value.episode_id, value.occurrence.occurrence_id) for value in self.action_occurrences
        }
        if len(occurrence_coordinates) != len(self.action_occurrences):
            raise ValueError("projection extension duplicates an episode/action occurrence")


@dataclass(frozen=True, slots=True)
class ProjectionTruthRequirement(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/projection-truth-requirement'

    requirement_id: str
    method_kind: IdentificationMethodEvidenceKind
    required_field_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.requirement_id, field_name="requirement_id")
        require_sorted_unique_strings(
            self.required_field_ids, field_name="required_field_ids", allow_empty=False
        )


@dataclass(frozen=True, slots=True)
class ProjectionExtensionCompatibilityReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/projection-extension-compatibility-receipt'

    receipt_id: str
    extension: ObjectIdentity
    target_schema: str
    consumer_id: str
    required_field_ids: tuple[str, ...]
    exact_field_ids: tuple[str, ...]
    lost_field_ids: tuple[str, ...]
    status: ProjectionCompatibilityStatus

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        if self.extension.object_schema != IdentificationEvidenceProjectionExtension.SCHEMA:
            raise ValueError("compatibility receipt requires the projection extension")
        validate_schema(self.target_schema)
        validate_stable_id(self.consumer_id, field_name="consumer_id")
        for name, values in (
            ("required_field_ids", self.required_field_ids),
            ("exact_field_ids", self.exact_field_ids),
            ("lost_field_ids", self.lost_field_ids),
        ):
            require_sorted_unique_strings(values, field_name=name)
        if set(self.exact_field_ids) & set(self.lost_field_ids):
            raise ValueError("compatibility exact and lost fields overlap")
        incompatible = bool(set(self.required_field_ids) & set(self.lost_field_ids))
        expected = (
            ProjectionCompatibilityStatus.INCOMPATIBLE_REQUIRED_FIELD
            if incompatible
            else ProjectionCompatibilityStatus.COMPATIBLE_WITH_DECLARED_LOSS
            if self.lost_field_ids
            else ProjectionCompatibilityStatus.EXACT
        )
        if self.status is not expected:
            raise ValueError("compatibility status differs from declared loss")
