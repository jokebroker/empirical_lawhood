"""Outcome-bounded external evidence manifests for reusable law qualification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord
from empirical_lawhood.kernel.evidence import (
    EvidenceCeiling,
    OutcomeAccess,
    VisibilityCeiling,
    inherited_visibility,
)
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity, NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_relative_locator,
    validate_stable_id,
)
from empirical_lawhood.kernel.systems import RelationalIdentity


MAX_PROJECTED_OBSERVATIONS = 100_000
MAX_PROJECTED_VALUES_PER_OBSERVATION = 256


class IdentificationEvidenceDomain(StrEnum):
    ACQUISITION = "ACQUISITION"
    DEVELOPMENT = "DEVELOPMENT"
    REFERENCE_SEALED = "REFERENCE_SEALED"
    CONFIRMATORY_SEALED = "CONFIRMATORY_SEALED"
    EVALUATOR_REVEAL = "EVALUATOR_REVEAL"


class ObservationDisposition(StrEnum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    SINK_TERMINATED = "SINK_TERMINATED"
    TECHNICAL_INVALID = "TECHNICAL_INVALID"
    CENSORED = "CENSORED"
    MISSING = "MISSING"
    UNEVALUABLE = "UNEVALUABLE"


class NestedCoordinateKind(StrEnum):
    TASK = "TASK"
    PREPARATION = "PREPARATION"
    TRAJECTORY = "TRAJECTORY"
    SEED = "SEED"
    REPEATED_DELIVERY = "REPEATED_DELIVERY"
    DENOMINATOR_MEMBER = "DENOMINATOR_MEMBER"
    CANDIDATE_VERSION = "CANDIDATE_VERSION"
    QUALIFICATION_VIEW = "QUALIFICATION_VIEW"
    RECEIVER = "RECEIVER"
    CLOCK = "CLOCK"


@dataclass(frozen=True, slots=True)
class ExternalEvidencePayload(CanonicalRecord):
    """Authoritative artifact plus bounded external-root-relative locator."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/external-evidence-payload'

    payload_id: str
    artifact: ArtifactIdentity
    external_root_contract_id: str
    relative_locator: str
    publication_receipt: ObjectIdentity
    recovery_identity: ObjectIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.payload_id, field_name="payload_id")
        validate_stable_id(
            self.external_root_contract_id,
            field_name="external_root_contract_id",
        )
        validate_relative_locator(self.relative_locator)
        if self.artifact.artifact_id != self.payload_id:
            raise ValueError("payload identity must equal the authoritative artifact identity")


@dataclass(frozen=True, slots=True)
class NestedEvidenceCoordinate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/nested-evidence-coordinate'

    coordinate_id: str
    kind: NestedCoordinateKind
    parent_coordinate_id: str | None

    def __post_init__(self) -> None:
        validate_stable_id(self.coordinate_id, field_name="coordinate_id")
        if self.parent_coordinate_id is not None:
            validate_stable_id(
                self.parent_coordinate_id,
                field_name="parent_coordinate_id",
            )
            if self.parent_coordinate_id == self.coordinate_id:
                raise ValueError("nested coordinate cannot parent itself")


@dataclass(frozen=True, slots=True)
class QualificationScopeSpec(CanonicalRecord):
    """Claim-specific physical-unit and aggregation scope."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/qualification-scope-spec'

    scope_id: str
    claim_id: str
    population_id: str
    physical_unit_type_id: str
    independent_unit_instance_ids: tuple[str, ...]
    nested_coordinate_ids: tuple[str, ...]
    aggregation_level_id: str
    uncertainty_unit_id: str
    locality_scope_id: str
    maximum_evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        for name, value in (
            ("scope_id", self.scope_id),
            ("claim_id", self.claim_id),
            ("population_id", self.population_id),
            ("physical_unit_type_id", self.physical_unit_type_id),
            ("aggregation_level_id", self.aggregation_level_id),
            ("uncertainty_unit_id", self.uncertainty_unit_id),
            ("locality_scope_id", self.locality_scope_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(
            self.independent_unit_instance_ids,
            field_name="independent_unit_instance_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.nested_coordinate_ids,
            field_name="nested_coordinate_ids",
        )


@dataclass(frozen=True, slots=True)
class ClaimUnitBinding(CanonicalRecord):
    """Exact claim-to-scope binding; nested axes cannot be borrowed as units."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/claim-unit-binding'

    binding_id: str
    claim_id: str
    scope: ObjectIdentity
    independent_unit_instance_ids: tuple[str, ...]
    aggregation_level_id: str

    def __post_init__(self) -> None:
        for name, value in (
            ("binding_id", self.binding_id),
            ("claim_id", self.claim_id),
            ("aggregation_level_id", self.aggregation_level_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(
            self.independent_unit_instance_ids,
            field_name="independent_unit_instance_ids",
            allow_empty=False,
        )
        if self.scope.object_schema != QualificationScopeSpec.SCHEMA:
            raise ValueError("claim-unit binding requires a QualificationScopeSpec")


@dataclass(frozen=True, slots=True)
class ObservationActionDeliveryBinding(CanonicalRecord):
    """Exact current action word plus the observed four delivery-stage identities."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/observation-action-delivery-binding'

    binding_id: str
    action_word: ObjectIdentity
    occurrence_ids: tuple[str, ...]
    requested_event_ids: tuple[str, ...]
    accepted_event_ids: tuple[str, ...]
    applied_event_ids: tuple[str, ...]
    realized_event_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")
        if self.action_word.object_schema != OccurrenceActionWord.SCHEMA:
            raise ValueError("observation action binding requires the current ActionWord schema")
        for name, values in (
            ("occurrence_ids", self.occurrence_ids),
            ("requested_event_ids", self.requested_event_ids),
            ("accepted_event_ids", self.accepted_event_ids),
            ("applied_event_ids", self.applied_event_ids),
            ("realized_event_ids", self.realized_event_ids),
        ):
            require_sorted_unique_strings(values, field_name=name)
        occurrence_ids = set(self.occurrence_ids)
        for values in (
            self.requested_event_ids,
            self.accepted_event_ids,
            self.applied_event_ids,
            self.realized_event_ids,
        ):
            if len(values) > len(occurrence_ids):
                raise ValueError("delivery-stage evidence exceeds the occurrence roster")


@dataclass(frozen=True, slots=True)
class IdentificationManifestObservation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/identification-manifest-observation'

    observation_id: str
    physical_unit_instance_id: str
    nested_coordinate_ids: tuple[str, ...]
    denominator_cell_id: str
    chart_id: str
    split_id: str
    role_id: str
    member_id: str
    candidate_version_id: str
    qualification_view_id: str
    receiver_id: str
    receiver_clock_id: str
    native_frame_id: str
    payload_id: str
    payload_member_locator: str
    action_delivery: ObservationActionDeliveryBinding | None
    disposition: ObservationDisposition
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("observation_id", self.observation_id),
            ("physical_unit_instance_id", self.physical_unit_instance_id),
            ("denominator_cell_id", self.denominator_cell_id),
            ("chart_id", self.chart_id),
            ("split_id", self.split_id),
            ("role_id", self.role_id),
            ("member_id", self.member_id),
            ("candidate_version_id", self.candidate_version_id),
            ("qualification_view_id", self.qualification_view_id),
            ("receiver_id", self.receiver_id),
            ("receiver_clock_id", self.receiver_clock_id),
            ("native_frame_id", self.native_frame_id),
            ("payload_id", self.payload_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(
            self.nested_coordinate_ids,
            field_name="nested_coordinate_ids",
        )
        validate_relative_locator(self.payload_member_locator)
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.disposition is ObservationDisposition.COMPLETE:
            if self.reason_codes:
                raise ValueError("complete observation cannot carry failure reasons")
            if self.action_delivery is not None:
                count = len(self.action_delivery.occurrence_ids)
                if any(
                    len(values) != count
                    for values in (
                        self.action_delivery.requested_event_ids,
                        self.action_delivery.accepted_event_ids,
                        self.action_delivery.applied_event_ids,
                        self.action_delivery.realized_event_ids,
                    )
                ):
                    raise ValueError("complete observation requires all delivery stages")
        elif not self.reason_codes:
            raise ValueError("non-complete observation requires reasons")


@dataclass(frozen=True, slots=True)
class IdentificationEvidenceManifest(CanonicalRecord):
    """Authoritative external evidence topology; contains no dense trajectories."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/identification-evidence-manifest'

    manifest_id: str
    system: ObjectIdentity
    relation: RelationalIdentity
    source_qualification: ObjectIdentity
    runtime_qualification: ObjectIdentity
    observation_operator_qualification: ObjectIdentity
    evidence_world_id: str
    evidence_domain: IdentificationEvidenceDomain
    allowed_consumer_ids: tuple[str, ...]
    information_cutoff_id: str
    payloads: tuple[ExternalEvidencePayload, ...]
    coordinates: tuple[NestedEvidenceCoordinate, ...]
    action_words: tuple[OccurrenceActionWord, ...]
    observations: tuple[IdentificationManifestObservation, ...]
    qualification_scopes: tuple[QualificationScopeSpec, ...]
    outcome_access: OutcomeAccess
    parent_visibility_ceilings: tuple[VisibilityCeiling, ...]
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        for name, value in (
            ("manifest_id", self.manifest_id),
            ("evidence_world_id", self.evidence_world_id),
            ("information_cutoff_id", self.information_cutoff_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(
            self.allowed_consumer_ids,
            field_name="allowed_consumer_ids",
            allow_empty=False,
        )
        for name, values, attribute in (
            ("payloads", self.payloads, "payload_id"),
            ("coordinates", self.coordinates, "coordinate_id"),
            ("action_words", self.action_words, "word_id"),
            ("observations", self.observations, "observation_id"),
            ("qualification_scopes", self.qualification_scopes, "scope_id"),
        ):
            require_sorted_unique_ids(values, attribute=attribute, field_name=name)
            if not values:
                raise ValueError(f"identification manifest requires {name}")
        coordinate_ids = {value.coordinate_id for value in self.coordinates}
        for coordinate in self.coordinates:
            if (
                coordinate.parent_coordinate_id is not None
                and coordinate.parent_coordinate_id not in coordinate_ids
            ):
                raise ValueError("nested coordinate names an unknown parent")
        self._validate_coordinate_graph()
        payload_ids = {value.payload_id for value in self.payloads}
        words = {value.word_id: value for value in self.action_words}
        observed_unit_ids = {value.physical_unit_instance_id for value in self.observations}
        for scope in self.qualification_scopes:
            if not set(scope.nested_coordinate_ids).issubset(coordinate_ids):
                raise ValueError("qualification scope names an unknown nested coordinate")
            if not set(scope.independent_unit_instance_ids).issubset(observed_unit_ids):
                raise ValueError("qualification scope names an unobserved physical unit")
        for observation in self.observations:
            if observation.payload_id not in payload_ids:
                raise ValueError("manifest observation names an unknown payload")
            if not set(observation.nested_coordinate_ids).issubset(coordinate_ids):
                raise ValueError("manifest observation names an unknown nested coordinate")
            if observation.action_delivery is not None:
                self._validate_action_delivery(observation.action_delivery, words)
        inherited = inherited_visibility(self.parent_visibility_ceilings, self.outcome_access)
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(inherited):
            raise ValueError("identification manifest visibility cannot be lowered")

    def _validate_coordinate_graph(self) -> None:
        parents = {value.coordinate_id: value.parent_coordinate_id for value in self.coordinates}
        for coordinate_id in parents:
            seen: set[str] = set()
            current: str | None = coordinate_id
            while current is not None:
                if current in seen:
                    raise ValueError("nested evidence coordinate graph contains a cycle")
                seen.add(current)
                current = parents[current]

    @staticmethod
    def _validate_action_delivery(
        delivery: ObservationActionDeliveryBinding,
        words: dict[str, OccurrenceActionWord],
    ) -> None:
        try:
            word = words[delivery.action_word.object_id]
        except KeyError as error:
            raise ValueError("manifest observation names an unknown action word") from error
        if delivery.action_word != ObjectIdentity.from_record(word.word_id, word):
            raise ValueError("manifest observation changes the exact action-word identity")
        if delivery.occurrence_ids != tuple(sorted(word.chronological_occurrence_ids)):
            raise ValueError("manifest delivery changes the action occurrence roster")
        for name, observed, expected in (
            (
                "requested",
                delivery.requested_event_ids,
                tuple(
                    sorted(f"event.{value.occurrence_id}.requested" for value in word.occurrences)
                ),
            ),
            (
                "accepted",
                delivery.accepted_event_ids,
                tuple(
                    sorted(f"event.{value.occurrence_id}.accepted" for value in word.occurrences)
                ),
            ),
            (
                "applied",
                delivery.applied_event_ids,
                tuple(sorted(f"event.{value.occurrence_id}.applied" for value in word.occurrences)),
            ),
            (
                "realized",
                delivery.realized_event_ids,
                tuple(
                    sorted(f"event.{value.occurrence_id}.realized" for value in word.occurrences)
                ),
            ),
        ):
            if observed != expected:
                raise ValueError(f"manifest delivery changes the {name} event roster")


@dataclass(frozen=True, slots=True)
class ProjectedIdentificationObservation(CanonicalRecord):
    """Small method input row with exact lineage to one manifest coordinate."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/projected-identification-observation'

    projected_observation_id: str
    manifest_observation_id: str
    physical_unit_instance_id: str
    nested_coordinate_ids: tuple[str, ...]
    denominator_cell_id: str
    chart_id: str
    split_id: str
    role_id: str
    member_id: str
    candidate_version_id: str
    qualification_view_id: str
    receiver_id: str
    receiver_clock_id: str
    native_frame_id: str
    payload_id: str
    payload_member_locator: str
    action_delivery: ObservationActionDeliveryBinding | None
    disposition: ObservationDisposition
    compact_values: tuple[NamedDecimal, ...]
    denominator_value_ids: tuple[str, ...]
    history_value_ids: tuple[str, ...]
    action_value_ids: tuple[str, ...]
    receiver_value_ids: tuple[str, ...]
    tags: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("projected_observation_id", self.projected_observation_id),
            ("manifest_observation_id", self.manifest_observation_id),
            ("physical_unit_instance_id", self.physical_unit_instance_id),
            ("denominator_cell_id", self.denominator_cell_id),
            ("chart_id", self.chart_id),
            ("split_id", self.split_id),
            ("role_id", self.role_id),
            ("member_id", self.member_id),
            ("candidate_version_id", self.candidate_version_id),
            ("qualification_view_id", self.qualification_view_id),
            ("receiver_id", self.receiver_id),
            ("receiver_clock_id", self.receiver_clock_id),
            ("native_frame_id", self.native_frame_id),
            ("payload_id", self.payload_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_relative_locator(self.payload_member_locator)
        require_sorted_unique_strings(
            self.nested_coordinate_ids,
            field_name="nested_coordinate_ids",
        )
        require_sorted_unique_ids(
            self.compact_values,
            attribute="value_id",
            field_name="compact_values",
        )
        if len(self.compact_values) > MAX_PROJECTED_VALUES_PER_OBSERVATION:
            raise ValueError("projected observation exceeds its compact-value bound")
        for name, values, allow_empty in (
            ("denominator_value_ids", self.denominator_value_ids, False),
            ("history_value_ids", self.history_value_ids, True),
            ("action_value_ids", self.action_value_ids, False),
            ("receiver_value_ids", self.receiver_value_ids, False),
            ("tags", self.tags, True),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=allow_empty)
        partitions = (
            set(self.denominator_value_ids),
            set(self.history_value_ids),
            set(self.action_value_ids),
            set(self.receiver_value_ids),
        )
        if any(
            left & right
            for index, left in enumerate(partitions)
            for right in partitions[index + 1 :]
        ):
            raise ValueError("projected value partitions overlap")
        partition_ids = set().union(*partitions)
        compact_ids = {value.value_id for value in self.compact_values}
        if self.disposition is ObservationDisposition.COMPLETE:
            if partition_ids != compact_ids:
                raise ValueError("projected value partitions do not cover compact values")
        elif self.compact_values:
            raise ValueError("non-complete projected observation cannot carry scientific values")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.disposition is ObservationDisposition.COMPLETE:
            if not self.compact_values or self.reason_codes:
                raise ValueError("complete projected observation requires values and no reasons")
        elif not self.reason_codes:
            raise ValueError("non-complete projected observation requires reasons")

    @property
    def action_word(self) -> ObjectIdentity | None:
        if self.action_delivery is None:
            return None
        return self.action_delivery.action_word


@dataclass(frozen=True, slots=True)
class IdentificationEvidenceProjection(CanonicalRecord):
    """Bounded deterministic projection consumed by one registered method."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/identification-evidence-projection'

    projection_id: str
    manifest: ObjectIdentity
    allowed_consumer_id: str
    projection_capability: ObjectIdentity
    projection_config: ObjectIdentity
    projection_implementation: ObjectIdentity
    claim_unit_binding: ClaimUnitBinding
    information_cutoff_id: str
    observations: tuple[ProjectedIdentificationObservation, ...]
    source_payload_ids: tuple[str, ...]
    outcome_access: OutcomeAccess
    parent_visibility_ceilings: tuple[VisibilityCeiling, ...]
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        for name, value in (
            ("projection_id", self.projection_id),
            ("allowed_consumer_id", self.allowed_consumer_id),
            ("information_cutoff_id", self.information_cutoff_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.manifest.object_schema != IdentificationEvidenceManifest.SCHEMA:
            raise ValueError("evidence projection must bind an identification manifest")
        require_sorted_unique_ids(
            self.observations,
            attribute="projected_observation_id",
            field_name="observations",
        )
        if not self.observations:
            raise ValueError("evidence projection requires observations")
        if len(self.observations) > MAX_PROJECTED_OBSERVATIONS:
            raise ValueError("evidence projection exceeds its row bound")
        require_sorted_unique_strings(
            self.source_payload_ids,
            field_name="source_payload_ids",
            allow_empty=False,
        )
        if not {value.payload_id for value in self.observations}.issubset(
            set(self.source_payload_ids)
        ):
            raise ValueError("projection observation uses an undeclared source payload")
        projected_units = {
            value.physical_unit_instance_id
            for value in self.observations
            if value.disposition is ObservationDisposition.COMPLETE
        }
        if not projected_units.issubset(set(self.claim_unit_binding.independent_unit_instance_ids)):
            raise ValueError("projection borrows nested rows as independent units")
        inherited = inherited_visibility(self.parent_visibility_ceilings, self.outcome_access)
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(inherited):
            raise ValueError("evidence projection visibility cannot be lowered")
