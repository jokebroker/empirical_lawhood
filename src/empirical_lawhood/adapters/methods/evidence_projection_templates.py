"Prospective, outcome-blind templates for exact evidence projections.\n\n``IdentificationProjectionPlan`` correctly binds the content identity of an\nalready materialized evidence manifest.  That makes it a post-acquisition\nrecord: its manifest fingerprint cannot honestly be placed in a preissue\nconfiguration.  This module supplies the missing lifecycle seam.  A template\nfreezes every topology, action, scope, value-partition and consumer choice;\nthe binder later verifies an authenticated manifest and authors the ordinary\nobservation-local plan without reading compact scientific values.\n"

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
import hashlib
from typing import ClassVar

from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    canonical_json_bytes,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_relative_locator,
    validate_schema,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.systems import RelationalIdentity
from empirical_lawhood.planning.identification_evidence import (
    ClaimUnitBinding,
    ExternalEvidencePayload,
    IdentificationEvidenceDomain,
    IdentificationEvidenceManifest,
    IdentificationManifestObservation,
    NestedEvidenceCoordinate,
    QualificationScopeSpec,
)
from empirical_lawhood.planning.source_pipelines import SourceRuntimeQualificationBindingReceipt

from .evidence_projection import TaggedIdentificationProjectionPlan, ObservationTagBinding, ObservationValuePartition


MAX_IDENTIFICATION_PROJECTION_TEMPLATES = 16
MAX_IDENTIFICATION_PROJECTION_TEMPLATE_OBSERVATIONS = 100_000


class IdentificationProjectionRole(StrEnum):
    """Claim-separated role; a role cannot borrow another projection."""

    PRIMARY_LAW = "PRIMARY_LAW"
    OPTIONAL_INTERACTION = "OPTIONAL_INTERACTION"
    ACTION_LOCAL_HOLD_CALIBRATION = "ACTION_LOCAL_HOLD_CALIBRATION"


class IdentificationProjectionBindingStage(StrEnum):
    """The only current lifecycle supported by this exact template seam."""

    EVALUATOR_REVEAL = "EVALUATOR_REVEAL"


class ConditionalQualificationSelectionKind(StrEnum):
    """Closed choice for one predeclared whole-unit qualification slot."""

    PRIMARY = "PRIMARY"
    CONDITIONAL_ALTERNATIVE = "CONDITIONAL_ALTERNATIVE"


class ConditionalQualificationTechnicalDisposition(StrEnum):
    """Outcome-blind operational evidence allowed to control a slot choice."""

    PRIMARY_RETAINED = "PRIMARY_RETAINED"
    ELIGIBLE_TECHNICAL_FAILURE = "ELIGIBLE_TECHNICAL_FAILURE"


class ConditionalQualificationTechnicalReason(StrEnum):
    """Closed outcome-blind reasons that may activate a whole-unit reserve."""

    WORKER_LOSS_NO_AUTHENTICATED_COMPLETE_PAYLOAD = "WORKER_LOSS_NO_AUTHENTICATED_COMPLETE_PAYLOAD"
    EXECUTOR_INTERRUPTION_NO_AUTHENTICATED_COMPLETE_PAYLOAD = (
        "EXECUTOR_INTERRUPTION_NO_AUTHENTICATED_COMPLETE_PAYLOAD"
    )
    ATOMIC_PUBLICATION_FAILURE_NO_AUTHENTICATED_COMPLETE_PAYLOAD = (
        "ATOMIC_PUBLICATION_FAILURE_NO_AUTHENTICATED_COMPLETE_PAYLOAD"
    )


@dataclass(frozen=True, slots=True)
class ConditionalQualificationWholeUnitOption(CanonicalRecord):
    """One indivisible primary or conditional-alternative acquisition block."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/conditional-qualification-whole-unit-option'

    option_id: str
    whole_unit_id: str
    physical_unit_instance_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.option_id, field_name="option_id")
        validate_stable_id(self.whole_unit_id, field_name="whole_unit_id")
        require_sorted_unique_strings(
            self.physical_unit_instance_ids,
            field_name="physical_unit_instance_ids",
            allow_empty=False,
        )


@dataclass(frozen=True, slots=True)
class ConditionalQualificationSlot(CanonicalRecord):
    """One frozen slot with exactly one same-tier whole-unit alternative."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/conditional-qualification-slot'

    slot_id: str
    tier_id: str
    primary: ConditionalQualificationWholeUnitOption
    conditional_alternative: ConditionalQualificationWholeUnitOption

    def __post_init__(self) -> None:
        validate_stable_id(self.slot_id, field_name="slot_id")
        validate_stable_id(self.tier_id, field_name="tier_id")
        if (
            self.primary.option_id == self.conditional_alternative.option_id
            or self.primary.whole_unit_id == self.conditional_alternative.whole_unit_id
            or set(self.primary.physical_unit_instance_ids)
            & set(self.conditional_alternative.physical_unit_instance_ids)
        ):
            raise ValueError("conditional qualification primary/alternative aliases")


@dataclass(frozen=True, slots=True)
class ConditionalQualificationRosterTemplate(CanonicalRecord):
    """Preissue six-slots-per-tier roster with no selected alternative."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/conditional-qualification-roster-template'

    template_id: str
    slots: tuple[ConditionalQualificationSlot, ...]
    slots_per_tier: int
    selection_rule_id: str
    outcome_access: OutcomeAccess = OutcomeAccess.OUTCOME_BLIND
    visibility_ceiling: VisibilityCeiling = VisibilityCeiling.PROSPECTIVE

    def __post_init__(self) -> None:
        validate_stable_id(self.template_id, field_name="template_id")
        validate_stable_id(self.selection_rule_id, field_name="selection_rule_id")
        require_sorted_unique_ids(self.slots, attribute="slot_id", field_name="slots")
        if self.slots_per_tier != 6 or not self.slots:
            raise ValueError("conditional qualification requires exactly six slots per tier")
        tier_ids = tuple(sorted({value.tier_id for value in self.slots}))
        if any(
            sum(value.tier_id == tier_id for value in self.slots) != self.slots_per_tier
            for tier_id in tier_ids
        ):
            raise ValueError("conditional qualification tier does not contain six slots")
        primary_whole_ids = tuple(value.primary.whole_unit_id for value in self.slots)
        if len(set(primary_whole_ids)) != len(primary_whole_ids):
            raise ValueError("conditional qualification repeats a primary whole unit")
        primary_units = tuple(
            unit_id for value in self.slots for unit_id in value.primary.physical_unit_instance_ids
        )
        if len(set(primary_units)) != len(primary_units):
            raise ValueError("conditional qualification primary physical units alias")
        alternatives_by_tier = {
            tier_id: {
                (
                    value.conditional_alternative.whole_unit_id,
                    value.conditional_alternative.physical_unit_instance_ids,
                )
                for value in self.slots
                if value.tier_id == tier_id
            }
            for tier_id in tier_ids
        }
        if any(len(values) != 1 for values in alternatives_by_tier.values()):
            raise ValueError("conditional qualification tier has more than one reserve block")
        alternative_units_by_tier = {
            tier_id: next(iter(values))[1] for tier_id, values in alternatives_by_tier.items()
        }
        if any(
            set(units) & set(primary_units) for units in alternative_units_by_tier.values()
        ) or len(
            {unit_id for units in alternative_units_by_tier.values() for unit_id in units}
        ) != sum(len(units) for units in alternative_units_by_tier.values()):
            raise ValueError("conditional qualification reserve physical units alias")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("conditional qualification roster must remain outcome-blind")
        if self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE:
            raise ValueError("conditional qualification roster must remain prospective")


@dataclass(frozen=True, slots=True)
class ConditionalQualificationTechnicalEvidence(CanonicalRecord):
    """Closed technical-only evidence for retaining or replacing one slot."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/conditional-qualification-technical-evidence'

    evidence_id: str
    slot_id: str
    whole_unit_id: str
    affected_physical_unit_instance_ids: tuple[str, ...]
    source_receipts: tuple[ObjectIdentity, ...]
    disposition: ConditionalQualificationTechnicalDisposition
    reason_codes: tuple[ConditionalQualificationTechnicalReason, ...]
    scientific_values_read: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.evidence_id, field_name="evidence_id")
        validate_stable_id(self.slot_id, field_name="slot_id")
        validate_stable_id(self.whole_unit_id, field_name="whole_unit_id")
        require_sorted_unique_strings(
            self.affected_physical_unit_instance_ids,
            field_name="affected_physical_unit_instance_ids",
            allow_empty=False,
        )
        source_ids = tuple(value.object_id for value in self.source_receipts)
        if not source_ids or source_ids != tuple(sorted(set(source_ids))):
            raise ValueError("conditional selection source receipts must be sorted and unique")
        if self.reason_codes != tuple(
            sorted(set(self.reason_codes), key=lambda value: value.value)
        ):
            raise ValueError("conditional technical reason codes must be sorted and unique")
        if self.disposition is ConditionalQualificationTechnicalDisposition.PRIMARY_RETAINED:
            if self.reason_codes:
                raise ValueError("retained conditional primary cannot carry failure reasons")
        elif not self.reason_codes:
            raise ValueError("conditional reserve activation requires a technical reason")
        if self.scientific_values_read:
            raise ValueError("conditional qualification selection cannot read scientific values")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("conditional qualification evidence must remain outcome-blind")


@dataclass(frozen=True, slots=True)
class ConditionalQualificationSlotSelection(CanonicalRecord):
    """One exact whole-unit choice; no component-level selection is representable."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/conditional-qualification-slot-selection'

    selection_id: str
    slot_id: str
    kind: ConditionalQualificationSelectionKind
    selected_option: ConditionalQualificationWholeUnitOption
    technical_evidence_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.selection_id, field_name="selection_id")
        validate_stable_id(self.slot_id, field_name="slot_id")
        validate_stable_id(self.technical_evidence_id, field_name="technical_evidence_id")


@dataclass(frozen=True, slots=True)
class ConditionalQualificationRosterSelectionReceipt(CanonicalRecord):
    """Outcome-blind exact selection over every predeclared whole-unit slot."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/conditional-qualification-roster-selection-receipt'

    receipt_id: str
    roster_template: ConditionalQualificationRosterTemplate
    selections: tuple[ConditionalQualificationSlotSelection, ...]
    technical_evidence: tuple[ConditionalQualificationTechnicalEvidence, ...]
    technical_only: bool
    whole_unit_selection: bool
    scientific_values_read: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        require_sorted_unique_ids(
            self.selections,
            attribute="selection_id",
            field_name="selections",
        )
        require_sorted_unique_ids(
            self.technical_evidence,
            attribute="evidence_id",
            field_name="technical_evidence",
        )
        selections_by_slot = {value.slot_id: value for value in self.selections}
        evidence_by_slot = {value.slot_id: value for value in self.technical_evidence}
        slot_ids = {value.slot_id for value in self.roster_template.slots}
        if (
            len(selections_by_slot) != len(self.selections)
            or len(evidence_by_slot) != len(self.technical_evidence)
            or set(selections_by_slot) != slot_ids
            or set(evidence_by_slot) != slot_ids
        ):
            raise ValueError("conditional qualification requires one choice/evidence per slot")
        selected_alternatives_by_tier: dict[str, int] = {}
        for slot in self.roster_template.slots:
            selection = selections_by_slot[slot.slot_id]
            evidence = evidence_by_slot[slot.slot_id]
            if selection.technical_evidence_id != evidence.evidence_id:
                raise ValueError("conditional selection names another technical evidence record")
            if selection.kind is ConditionalQualificationSelectionKind.PRIMARY:
                expected = slot.primary
                disposition = ConditionalQualificationTechnicalDisposition.PRIMARY_RETAINED
            else:
                expected = slot.conditional_alternative
                disposition = (
                    ConditionalQualificationTechnicalDisposition.ELIGIBLE_TECHNICAL_FAILURE
                )
                selected_alternatives_by_tier[slot.tier_id] = (
                    selected_alternatives_by_tier.get(slot.tier_id, 0) + 1
                )
            if (
                selection.selected_option != expected
                or evidence.disposition is not disposition
                or evidence.whole_unit_id != slot.primary.whole_unit_id
                or evidence.affected_physical_unit_instance_ids
                != slot.primary.physical_unit_instance_ids
            ):
                raise ValueError("conditional selection differs from its frozen option/evidence")
        if any(value > 1 for value in selected_alternatives_by_tier.values()):
            raise ValueError("conditional qualification selects more than one reserve per tier")
        if not self.technical_only or not self.whole_unit_selection:
            raise ValueError("conditional qualification selection must be technical/whole-unit")
        if self.scientific_values_read:
            raise ValueError("conditional qualification receipt cannot read scientific values")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("conditional qualification receipt must remain outcome-blind")


@dataclass(frozen=True, slots=True)
class IdentificationProjectionPayloadExtensionTemplate(CanonicalRecord):
    """Outcome-free schema identity for one future artifact extension."""

    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/methods/identification-projection-payload-extension-template'
    )

    namespace: str
    schema: str

    def __post_init__(self) -> None:
        validate_stable_id(self.namespace, field_name="namespace")
        validate_schema(self.schema)


@dataclass(frozen=True, slots=True)
class IdentificationProjectionPayloadTemplate(CanonicalRecord):
    """Prospective payload topology without future content identities or sizes."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/identification-projection-payload-template'

    payload_id: str
    artifact_role: str
    artifact_payload_schema: str
    artifact_media_type: str
    artifact_extensions: tuple[IdentificationProjectionPayloadExtensionTemplate, ...]
    external_root_contract_id: str
    relative_locator: str
    publication_receipt_id: str
    publication_receipt_schema: str
    publication_receipt_version: str
    recovery_identity_id: str
    recovery_identity_schema: str
    recovery_identity_version: str

    def __post_init__(self) -> None:
        for field_name, value in (
            ("payload_id", self.payload_id),
            ("artifact_role", self.artifact_role),
            ("external_root_contract_id", self.external_root_contract_id),
            ("publication_receipt_id", self.publication_receipt_id),
            ("recovery_identity_id", self.recovery_identity_id),
        ):
            validate_stable_id(value, field_name=field_name)
        validate_schema(self.artifact_payload_schema)
        validate_nonempty(self.artifact_media_type, field_name="artifact_media_type")
        require_sorted_unique_ids(
            self.artifact_extensions,
            attribute="namespace",
            field_name="artifact_extensions",
        )
        validate_relative_locator(self.relative_locator)
        for value in (self.publication_receipt_schema, self.recovery_identity_schema):
            validate_schema(value)
        for value in (self.publication_receipt_version, self.recovery_identity_version):
            validate_semantic_version(value)


@dataclass(frozen=True, slots=True)
class IdentificationProjectionObservationTemplate(CanonicalRecord):
    """Outcome-free expected topology for one future manifest observation."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/identification-projection-observation-template'

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
    action_word_id: str | None
    occurrence_ids: tuple[str, ...]
    tags: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name, value in (
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
            validate_stable_id(value, field_name=field_name)
        validate_relative_locator(self.payload_member_locator)
        require_sorted_unique_strings(
            self.nested_coordinate_ids,
            field_name="nested_coordinate_ids",
        )
        require_sorted_unique_strings(
            self.occurrence_ids,
            field_name="occurrence_ids",
        )
        require_sorted_unique_strings(self.tags, field_name="tags")
        if self.action_word_id is None:
            if self.occurrence_ids:
                raise ValueError("action-free observation template has occurrences")
        else:
            validate_stable_id(self.action_word_id, field_name="action_word_id")


@dataclass(frozen=True, slots=True)
class IdentificationProjectionTemplate(CanonicalRecord):
    "Complete prospective template for one claim-scoped observation-local projection."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/identification-projection-template'

    template_id: str
    role: IdentificationProjectionRole
    expected_manifest_id: str
    system: ObjectIdentity
    relation: RelationalIdentity
    source_qualification: ObjectIdentity
    runtime_qualification: ObjectIdentity
    observation_operator_qualification: ObjectIdentity
    evidence_world_id: str
    evidence_domain: IdentificationEvidenceDomain
    expected_allowed_consumer_ids: tuple[str, ...]
    information_cutoff_id: str
    payloads: tuple[IdentificationProjectionPayloadTemplate, ...]
    coordinates: tuple[NestedEvidenceCoordinate, ...]
    action_words: tuple[OccurrenceActionWord, ...]
    qualification_scopes: tuple[QualificationScopeSpec, ...]
    qualification_scope_id: str
    claim_unit_binding_id: str
    allowed_consumer_id: str
    projection_capability: ObjectIdentity
    projection_config: ObjectIdentity
    projection_implementation: ObjectIdentity
    value_partition: ObservationValuePartition
    observations: tuple[IdentificationProjectionObservationTemplate, ...]
    binder_implementation_id: str
    binder_implementation_sha256: str
    binding_stage: IdentificationProjectionBindingStage
    outcome_access: OutcomeAccess
    expected_manifest_parent_visibility_ceilings: tuple[VisibilityCeiling, ...]
    expected_manifest_visibility_ceiling: VisibilityCeiling
    visibility_ceiling: VisibilityCeiling
    conditional_roster_template: ObjectIdentity | None = None
    conditional_roster_selection: ObjectIdentity | None = None

    def __post_init__(self) -> None:
        for field_name, value in (
            ("template_id", self.template_id),
            ("expected_manifest_id", self.expected_manifest_id),
            ("evidence_world_id", self.evidence_world_id),
            ("information_cutoff_id", self.information_cutoff_id),
            ("qualification_scope_id", self.qualification_scope_id),
            ("claim_unit_binding_id", self.claim_unit_binding_id),
            ("allowed_consumer_id", self.allowed_consumer_id),
            ("binder_implementation_id", self.binder_implementation_id),
        ):
            validate_stable_id(value, field_name=field_name)
        validate_sha256(
            self.binder_implementation_sha256,
            field_name="binder_implementation_sha256",
        )
        require_sorted_unique_strings(
            self.expected_allowed_consumer_ids,
            field_name="expected_allowed_consumer_ids",
            allow_empty=False,
        )
        if self.allowed_consumer_id not in self.expected_allowed_consumer_ids:
            raise ValueError("projection consumer is absent from the expected manifest roster")
        require_sorted_unique_ids(
            self.payloads,
            attribute="payload_id",
            field_name="payloads",
        )
        require_sorted_unique_ids(
            self.coordinates,
            attribute="coordinate_id",
            field_name="coordinates",
        )
        require_sorted_unique_ids(
            self.action_words,
            attribute="word_id",
            field_name="action_words",
        )
        require_sorted_unique_ids(
            self.qualification_scopes,
            attribute="scope_id",
            field_name="qualification_scopes",
        )
        require_sorted_unique_ids(
            self.observations,
            attribute="observation_id",
            field_name="observations",
        )
        if (
            not self.payloads
            or not self.coordinates
            or not self.action_words
            or not self.qualification_scopes
            or not self.observations
            or len(self.observations) > MAX_IDENTIFICATION_PROJECTION_TEMPLATE_OBSERVATIONS
        ):
            raise ValueError("projection template action/observation roster is empty or too large")
        scopes = {value.scope_id: value for value in self.qualification_scopes}
        try:
            selected_scope = scopes[self.qualification_scope_id]
        except KeyError as error:
            raise ValueError(
                "projection template selects an unknown qualification scope"
            ) from error
        coordinate_ids = {value.coordinate_id for value in self.coordinates}
        for coordinate in self.coordinates:
            if (
                coordinate.parent_coordinate_id is not None
                and coordinate.parent_coordinate_id not in coordinate_ids
            ):
                raise ValueError("projection template coordinate names an unknown parent")
        self._validate_coordinate_graph()
        words = {value.word_id: value for value in self.action_words}
        payload_ids = {value.payload_id for value in self.payloads}
        observed_units = {value.physical_unit_instance_id for value in self.observations}
        for scope in self.qualification_scopes:
            if not set(scope.independent_unit_instance_ids).issubset(observed_units):
                raise ValueError("projection template scope names an unobserved physical unit")
            if not set(scope.nested_coordinate_ids).issubset(coordinate_ids):
                raise ValueError("projection template scope names an unknown coordinate")
        if not set(selected_scope.independent_unit_instance_ids).issubset(observed_units):
            raise ValueError("selected projection scope names an unobserved physical unit")
        for observation in self.observations:
            if observation.payload_id not in payload_ids:
                raise ValueError("projection template observation names an unknown payload")
            if not set(observation.nested_coordinate_ids).issubset(coordinate_ids):
                raise ValueError("projection template observation names an unknown coordinate")
            if observation.action_word_id is None:
                continue
            try:
                word = words[observation.action_word_id]
            except KeyError as error:
                raise ValueError("projection template names an unknown action word") from error
            if observation.occurrence_ids != tuple(sorted(word.chronological_occurrence_ids)):
                raise ValueError("projection template changes the action occurrence roster")
        if self.binding_stage is not IdentificationProjectionBindingStage.EVALUATOR_REVEAL:
            raise ValueError("projection template has an unsupported binding stage")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("projection template must remain outcome-blind")
        if not self.expected_manifest_parent_visibility_ceilings:
            raise ValueError("projection template requires the future manifest parent ceilings")
        if self.expected_manifest_visibility_ceiling is not VisibilityCeiling.PROSPECTIVE:
            raise ValueError("projection template requires a prospective future manifest")
        if self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE:
            raise ValueError("projection template must remain prospective")
        if self.conditional_roster_template is None:
            if self.conditional_roster_selection is not None:
                raise ValueError("projection template selection lacks its roster template")
        else:
            if self.conditional_roster_template.object_schema != (
                ConditionalQualificationRosterTemplate.SCHEMA
            ):
                raise ValueError("projection template names another conditional roster schema")
            if self.conditional_roster_selection is not None and (
                self.conditional_roster_selection.object_schema
                != ConditionalQualificationRosterSelectionReceipt.SCHEMA
            ):
                raise ValueError("projection template names another selection receipt schema")

    def _validate_coordinate_graph(self) -> None:
        parents = {value.coordinate_id: value.parent_coordinate_id for value in self.coordinates}
        for coordinate_id in parents:
            seen: set[str] = set()
            current: str | None = coordinate_id
            while current is not None:
                if current in seen:
                    raise ValueError("projection template coordinate graph contains a cycle")
                seen.add(current)
                current = parents[current]


def _observation_semantic_key(
    value: IdentificationProjectionObservationTemplate,
) -> tuple[object, ...]:
    return (
        value.chart_id,
        value.split_id,
        value.role_id,
        value.member_id,
        value.candidate_version_id,
        value.qualification_view_id,
        value.receiver_id,
        value.receiver_clock_id,
        value.native_frame_id,
        value.payload_member_locator,
        value.action_word_id,
        value.occurrence_ids,
        value.tags,
    )


def _observation_semantic_sha256(
    value: IdentificationProjectionObservationTemplate,
) -> str:
    return hashlib.sha256(canonical_json_bytes(_observation_semantic_key(value))).hexdigest()


@dataclass(frozen=True, slots=True)
class ConditionalIdentificationProjectionObservationPair(CanonicalRecord):
    """Corresponding complete observation topology for one slot choice."""

    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/methods/conditional-identification-projection-observation-pair'
    )

    pair_id: str
    slot_id: str
    primary_observation_id: str
    primary_observation_sha256: str
    scientific_semantics_sha256: str
    conditional_alternative: IdentificationProjectionObservationTemplate

    def __post_init__(self) -> None:
        validate_stable_id(self.pair_id, field_name="pair_id")
        validate_stable_id(self.slot_id, field_name="slot_id")
        validate_stable_id(
            self.primary_observation_id,
            field_name="primary_observation_id",
        )
        validate_sha256(
            self.primary_observation_sha256,
            field_name="primary_observation_sha256",
        )
        validate_sha256(
            self.scientific_semantics_sha256,
            field_name="scientific_semantics_sha256",
        )
        if self.scientific_semantics_sha256 != _observation_semantic_sha256(
            self.conditional_alternative
        ):
            raise ValueError("conditional projection pair changes scientific row semantics")
        if self.primary_observation_id == self.conditional_alternative.observation_id:
            raise ValueError("conditional projection pair aliases primary and alternative")


@dataclass(frozen=True, slots=True)
class ConditionalIdentificationProjectionTemplate(CanonicalRecord):
    """Preissue union topology from which one ordinary role template is selected."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/conditional-identification-projection-template'

    template_id: str
    role: IdentificationProjectionRole
    primary_template_id: str
    primary_template_sha256: str
    roster_template: ConditionalQualificationRosterTemplate
    observation_pairs: tuple[ConditionalIdentificationProjectionObservationPair, ...]
    alternative_payloads: tuple[IdentificationProjectionPayloadTemplate, ...]
    alternative_coordinates: tuple[NestedEvidenceCoordinate, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.template_id, field_name="template_id")
        validate_stable_id(self.primary_template_id, field_name="primary_template_id")
        validate_sha256(
            self.primary_template_sha256,
            field_name="primary_template_sha256",
        )
        if self.template_id != f"conditional.{self.primary_template_id}":
            raise ValueError("conditional projection identity differs from its primary template")
        require_sorted_unique_ids(
            self.observation_pairs,
            attribute="pair_id",
            field_name="observation_pairs",
        )
        require_sorted_unique_ids(
            self.alternative_payloads,
            attribute="payload_id",
            field_name="alternative_payloads",
        )
        require_sorted_unique_ids(
            self.alternative_coordinates,
            attribute="coordinate_id",
            field_name="alternative_coordinates",
        )
        slots = {value.slot_id: value for value in self.roster_template.slots}
        paired_primary_ids = tuple(value.primary_observation_id for value in self.observation_pairs)
        if len(set(paired_primary_ids)) != len(paired_primary_ids):
            raise ValueError("conditional projection repeats a primary observation")
        alternatives_by_id: dict[
            str,
            IdentificationProjectionObservationTemplate,
        ] = {}
        for pair in self.observation_pairs:
            alternative = pair.conditional_alternative
            previous = alternatives_by_id.setdefault(
                alternative.observation_id,
                alternative,
            )
            if previous != alternative:
                raise ValueError(
                    "conditional projection reuses an alternative observation "
                    "identity with changed content"
                )
        payloads = {value.payload_id for value in self.alternative_payloads}
        coordinates = {value.coordinate_id for value in self.alternative_coordinates}
        for pair in self.observation_pairs:
            slot = slots.get(pair.slot_id)
            if slot is None:
                raise ValueError("conditional projection pair names an unknown slot")
            if pair.conditional_alternative.physical_unit_instance_id not in (
                slot.conditional_alternative.physical_unit_instance_ids
            ):
                raise ValueError("conditional projection pair mixes whole-unit options")
            observation = pair.conditional_alternative
            if observation.payload_id not in payloads or not set(
                observation.nested_coordinate_ids
            ).issubset(coordinates):
                raise ValueError("conditional projection observation topology is incomplete")


def _selected_projection_template(
    *,
    conditional: ConditionalIdentificationProjectionTemplate,
    primary_template: IdentificationProjectionTemplate,
    selection: ConditionalQualificationRosterSelectionReceipt,
) -> IdentificationProjectionTemplate:
    if selection.roster_template != conditional.roster_template:
        raise ValueError("conditional projection selection names another roster template")
    selected_by_slot = {value.slot_id: value for value in selection.selections}
    primary_by_id = {value.observation_id: value for value in primary_template.observations}
    selected_observations = tuple(
        primary_by_id[pair.primary_observation_id]
        if selected_by_slot[pair.slot_id].kind is ConditionalQualificationSelectionKind.PRIMARY
        else pair.conditional_alternative
        for pair in conditional.observation_pairs
    )
    if len({value.observation_id for value in selected_observations}) != len(selected_observations):
        raise ValueError("conditional projection selected roster aliases observations")
    observations = tuple(sorted(selected_observations, key=lambda value: value.observation_id))
    payload_union = {
        value.payload_id: value
        for value in (
            *primary_template.payloads,
            *conditional.alternative_payloads,
        )
    }
    selected_payload_ids = {value.payload_id for value in observations}
    payloads = tuple(
        sorted(
            (payload_union[value] for value in selected_payload_ids),
            key=lambda value: value.payload_id,
        )
    )
    coordinate_union = {
        value.coordinate_id: value
        for value in (
            *primary_template.coordinates,
            *conditional.alternative_coordinates,
        )
    }
    selected_coordinate_ids = {
        coordinate_id for value in observations for coordinate_id in value.nested_coordinate_ids
    }
    pending = list(selected_coordinate_ids)
    while pending:
        coordinate = coordinate_union[pending.pop()]
        parent = coordinate.parent_coordinate_id
        if parent is not None and parent not in selected_coordinate_ids:
            selected_coordinate_ids.add(parent)
            pending.append(parent)
    coordinates = tuple(
        sorted(
            (coordinate_union[value] for value in selected_coordinate_ids),
            key=lambda value: value.coordinate_id,
        )
    )
    primary_scope = primary_template.qualification_scopes[0]
    scope = replace(
        primary_scope,
        independent_unit_instance_ids=tuple(
            sorted({value.physical_unit_instance_id for value in observations})
        ),
        nested_coordinate_ids=tuple(sorted(selected_coordinate_ids)),
    )
    roster_identity = ObjectIdentity.from_record(
        conditional.roster_template.template_id,
        conditional.roster_template,
    )
    selection_identity = ObjectIdentity.from_record(selection.receipt_id, selection)
    return replace(
        primary_template,
        payloads=payloads,
        coordinates=coordinates,
        qualification_scopes=(scope,),
        observations=observations,
        conditional_roster_template=roster_identity,
        conditional_roster_selection=selection_identity,
    )


@dataclass(frozen=True, slots=True)
class IdentificationProjectionTemplateSet(CanonicalRecord):
    """Closed, independently fingerprintable set of claim-scoped templates."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/identification-projection-template-set'

    template_set_id: str
    templates: tuple[IdentificationProjectionTemplate, ...]
    required_roles: tuple[IdentificationProjectionRole, ...]
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    conditional_roster_template: ConditionalQualificationRosterTemplate | None = None
    conditional_templates: tuple[ConditionalIdentificationProjectionTemplate, ...] = ()
    conditional_roster_selection: ObjectIdentity | None = None

    def __post_init__(self) -> None:
        validate_stable_id(self.template_set_id, field_name="template_set_id")
        require_sorted_unique_ids(
            self.templates,
            attribute="template_id",
            field_name="templates",
        )
        if not self.templates or len(self.templates) > MAX_IDENTIFICATION_PROJECTION_TEMPLATES:
            raise ValueError("projection template set has an invalid size")
        expected_roles = tuple(
            sorted(IdentificationProjectionRole, key=lambda value: value.value)
        )
        if self.required_roles != expected_roles:
            raise ValueError("projection template set requires all three exact claim roles")
        if tuple(
            sorted((value.role for value in self.templates), key=lambda value: value.value)
        ) != (self.required_roles):
            raise ValueError("projection template set does not contain exactly its required roles")
        for field_name, values in (
            (
                "expected_manifest_id",
                tuple(value.expected_manifest_id for value in self.templates),
            ),
            (
                "qualification_scope_id",
                tuple(value.qualification_scope_id for value in self.templates),
            ),
            (
                "claim_unit_binding_id",
                tuple(value.claim_unit_binding_id for value in self.templates),
            ),
            (
                "allowed_consumer_id",
                tuple(value.allowed_consumer_id for value in self.templates),
            ),
            (
                "value_partition.partition_id",
                tuple(value.value_partition.partition_id for value in self.templates),
            ),
            (
                "projection_config.object_id",
                tuple(value.projection_config.object_id for value in self.templates),
            ),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"projection template set aliases {field_name} across roles")
        if any(
            value.expected_allowed_consumer_ids != (value.allowed_consumer_id,)
            for value in self.templates
        ):
            raise ValueError("each projection role requires exactly its one declared consumer")
        common = self.templates[0]
        for template in self.templates[1:]:
            if (
                template.system != common.system
                or template.relation != common.relation
                or template.source_qualification != common.source_qualification
                or template.runtime_qualification != common.runtime_qualification
                or template.observation_operator_qualification
                != common.observation_operator_qualification
                or template.evidence_world_id != common.evidence_world_id
                or template.information_cutoff_id != common.information_cutoff_id
                or template.action_words != common.action_words
                or template.conditional_roster_template != common.conditional_roster_template
                or template.conditional_roster_selection != common.conditional_roster_selection
            ):
                raise ValueError("projection templates change the common evidence lineage")
        if self.conditional_roster_template is None:
            if (
                self.conditional_templates
                or self.conditional_roster_selection is not None
                or any(value.conditional_roster_template is not None for value in self.templates)
            ):
                raise ValueError("ordinary projection set contains conditional roster state")
        else:
            roster_identity = ObjectIdentity.from_record(
                self.conditional_roster_template.template_id,
                self.conditional_roster_template,
            )
            if self.conditional_roster_selection is None:
                self._validate_unresolved_conditional_templates()
            elif self.conditional_templates:
                raise ValueError(
                    "selected projection set must not retain unresolved conditional templates"
                )
            if any(
                value.conditional_roster_template != roster_identity
                or value.conditional_roster_selection != self.conditional_roster_selection
                for value in self.templates
            ):
                raise ValueError("projection set/templates differ on conditional selection")
            if self.conditional_roster_selection is not None and (
                self.conditional_roster_selection.object_schema
                != ConditionalQualificationRosterSelectionReceipt.SCHEMA
            ):
                raise ValueError("projection set names another selection receipt schema")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("projection template set must remain outcome-blind")
        if self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE:
            raise ValueError("projection template set must remain prospective")

    def _validate_unresolved_conditional_templates(self) -> None:
        require_sorted_unique_ids(
            self.conditional_templates,
            attribute="template_id",
            field_name="conditional_templates",
        )
        if (
            tuple(
                sorted(
                    (value.role for value in self.conditional_templates),
                    key=lambda value: value.value,
                )
            )
            != self.required_roles
        ):
            raise ValueError("conditional projection set does not cover all three roles")
        conditional_by_role = {value.role: value for value in self.conditional_templates}
        ordinary_by_role = {value.role: value for value in self.templates}
        if any(
            conditional.roster_template != self.conditional_roster_template
            or conditional.primary_template_id != ordinary_by_role[role].template_id
            or conditional.primary_template_sha256 != ordinary_by_role[role].fingerprint()
            for role, conditional in conditional_by_role.items()
        ):
            raise ValueError("conditional projection set changes role/roster topology")
        for role, conditional in conditional_by_role.items():
            primary = ordinary_by_role[role]
            if len(primary.qualification_scopes) != 1:
                raise ValueError("conditional projection requires one exact scope")
            primary_observations = {value.observation_id: value for value in primary.observations}
            pair_ids = {value.primary_observation_id for value in conditional.observation_pairs}
            if pair_ids != set(primary_observations):
                raise ValueError(
                    "conditional projection pairs do not partition primary observations"
                )
            primary_payloads = {value.payload_id: value for value in primary.payloads}
            alternative_payloads = {
                value.payload_id: value for value in conditional.alternative_payloads
            }
            if set(primary_payloads) & set(alternative_payloads):
                raise ValueError("conditional projection aliases primary payload identities")
            payloads = {
                value.payload_id: value
                for value in (*primary.payloads, *conditional.alternative_payloads)
            }
            primary_coordinates = {value.coordinate_id: value for value in primary.coordinates}
            alternative_coordinates = {
                value.coordinate_id: value for value in conditional.alternative_coordinates
            }
            if any(
                primary_coordinates[coordinate_id] != alternative_coordinates[coordinate_id]
                for coordinate_id in set(primary_coordinates) & set(alternative_coordinates)
            ):
                raise ValueError("conditional projection changes a shared coordinate identity")
            coordinates = {
                value.coordinate_id: value
                for value in (*primary.coordinates, *conditional.alternative_coordinates)
            }
            slots = {value.slot_id: value for value in conditional.roster_template.slots}
            for pair in conditional.observation_pairs:
                primary_observation = primary_observations[pair.primary_observation_id]
                if (
                    pair.primary_observation_sha256 != primary_observation.fingerprint()
                    or pair.scientific_semantics_sha256
                    != _observation_semantic_sha256(primary_observation)
                    or primary_observation.physical_unit_instance_id
                    not in slots[pair.slot_id].primary.physical_unit_instance_ids
                    or primary_observation.physical_unit_instance_id
                    == pair.conditional_alternative.physical_unit_instance_id
                    or primary_observation.denominator_cell_id
                    == pair.conditional_alternative.denominator_cell_id
                    or primary_observation.payload_id == pair.conditional_alternative.payload_id
                ):
                    raise ValueError(
                        "conditional projection pair changes primary or scientific semantics"
                    )
                for observation in (primary_observation, pair.conditional_alternative):
                    if observation.payload_id not in payloads or not set(
                        observation.nested_coordinate_ids
                    ).issubset(coordinates):
                        raise ValueError(
                            "conditional projection observation topology is incomplete"
                        )


def materialize_conditional_identification_projection_template_set(
    *,
    template_set: IdentificationProjectionTemplateSet,
    selection: ConditionalQualificationRosterSelectionReceipt,
) -> IdentificationProjectionTemplateSet:
    """Select complete role topologies without reading a scientific value."""

    if template_set.conditional_roster_template is None or not (template_set.conditional_templates):
        raise ValueError("projection template set has no conditional roster topology")
    if template_set.conditional_roster_selection is not None:
        raise ValueError("projection template set is already conditionally materialized")
    if selection.roster_template != template_set.conditional_roster_template:
        raise ValueError("projection template set selection names another roster")
    primary_by_role = {value.role: value for value in template_set.templates}
    templates = tuple(
        sorted(
            (
                _selected_projection_template(
                    conditional=value,
                    primary_template=primary_by_role[value.role],
                    selection=selection,
                )
                for value in template_set.conditional_templates
            ),
            key=lambda value: value.template_id,
        )
    )
    return replace(
        template_set,
        templates=templates,
        conditional_templates=(),
        conditional_roster_selection=ObjectIdentity.from_record(
            selection.receipt_id,
            selection,
        ),
    )


@dataclass(frozen=True, slots=True)
class IdentificationProjectionTemplateBindingReceipt(CanonicalRecord):
    """Post-manifest topology proof; it is not a projection or science result."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/identification-projection-template-binding-receipt'

    binding_id: str
    template: ObjectIdentity
    manifest: ObjectIdentity
    projection_plan: ObjectIdentity
    binder_implementation_id: str
    binder_implementation_sha256: str
    topology_exact: bool
    compact_values_read: bool
    grants_scientific_promotion: bool
    outcome_access: OutcomeAccess
    source_runtime_qualification_binding: ObjectIdentity | None = None

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")
        validate_stable_id(
            self.binder_implementation_id,
            field_name="binder_implementation_id",
        )
        validate_sha256(
            self.binder_implementation_sha256,
            field_name="binder_implementation_sha256",
        )
        if self.template.object_schema != IdentificationProjectionTemplate.SCHEMA:
            raise ValueError("projection binding receipt names another template schema")
        if self.manifest.object_schema != IdentificationEvidenceManifest.SCHEMA:
            raise ValueError("projection binding receipt names another manifest schema")
        if self.projection_plan.object_schema != TaggedIdentificationProjectionPlan.SCHEMA:
            raise ValueError("projection binding receipt names another plan schema")
        if self.source_runtime_qualification_binding is not None and (
            self.source_runtime_qualification_binding.object_schema
            != SourceRuntimeQualificationBindingReceipt.SCHEMA
        ):
            raise ValueError("projection binding receipt names another source/runtime mapping")
        if not self.topology_exact:
            raise ValueError("projection template binding requires exact topology")
        if self.compact_values_read:
            raise ValueError("projection template binder cannot read compact values")
        if self.grants_scientific_promotion:
            raise ValueError("projection template binding cannot promote science")
        if self.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL:
            raise ValueError("projection binding receipt requires evaluator-reveal access")


def _observation_topology(
    observation: IdentificationManifestObservation,
) -> IdentificationProjectionObservationTemplate:
    delivery = observation.action_delivery
    return IdentificationProjectionObservationTemplate(
        observation_id=observation.observation_id,
        physical_unit_instance_id=observation.physical_unit_instance_id,
        nested_coordinate_ids=observation.nested_coordinate_ids,
        denominator_cell_id=observation.denominator_cell_id,
        chart_id=observation.chart_id,
        split_id=observation.split_id,
        role_id=observation.role_id,
        member_id=observation.member_id,
        candidate_version_id=observation.candidate_version_id,
        qualification_view_id=observation.qualification_view_id,
        receiver_id=observation.receiver_id,
        receiver_clock_id=observation.receiver_clock_id,
        native_frame_id=observation.native_frame_id,
        payload_id=observation.payload_id,
        payload_member_locator=observation.payload_member_locator,
        action_word_id=None if delivery is None else delivery.action_word.object_id,
        occurrence_ids=() if delivery is None else delivery.occurrence_ids,
        tags=(),
    )


def _payload_topology(
    payload: ExternalEvidencePayload,
) -> IdentificationProjectionPayloadTemplate:
    return IdentificationProjectionPayloadTemplate(
        payload_id=payload.payload_id,
        artifact_role=payload.artifact.role,
        artifact_payload_schema=payload.artifact.payload_schema,
        artifact_media_type=payload.artifact.media_type,
        artifact_extensions=tuple(
            IdentificationProjectionPayloadExtensionTemplate(
                namespace=value.namespace,
                schema=value.schema,
            )
            for value in payload.artifact.extensions
        ),
        external_root_contract_id=payload.external_root_contract_id,
        relative_locator=payload.relative_locator,
        publication_receipt_id=payload.publication_receipt.object_id,
        publication_receipt_schema=payload.publication_receipt.object_schema,
        publication_receipt_version=payload.publication_receipt.object_version,
        recovery_identity_id=payload.recovery_identity.object_id,
        recovery_identity_schema=payload.recovery_identity.object_schema,
        recovery_identity_version=payload.recovery_identity.object_version,
    )


def bind_identification_projection_template(
    *,
    binding_id: str,
    template: IdentificationProjectionTemplate,
    manifest: IdentificationEvidenceManifest,
    binder_implementation_id: str,
    binder_implementation_sha256: str,
    source_runtime_qualification_binding: (
        SourceRuntimeQualificationBindingReceipt | None
    ) = None,
) -> tuple[TaggedIdentificationProjectionPlan, IdentificationProjectionTemplateBindingReceipt]:
    """Bind one authenticated manifest without reading any external payload value."""

    if (
        template.conditional_roster_template is not None
        and template.conditional_roster_selection is None
    ):
        raise ValueError("conditional projection template requires a selection receipt")
    if (
        binder_implementation_id != template.binder_implementation_id
        or binder_implementation_sha256 != template.binder_implementation_sha256
    ):
        raise ValueError("projection template binder implementation differs")
    source_qualification = template.source_qualification
    runtime_qualification = template.runtime_qualification
    if source_runtime_qualification_binding is not None:
        if (
            source_runtime_qualification_binding.source_expectation != template.source_qualification
            or source_runtime_qualification_binding.runtime_expectation
            != template.runtime_qualification
        ):
            raise ValueError(
                "source/runtime qualification mapping names another prospective template"
            )
        source_qualification = source_runtime_qualification_binding.source_qualification
        runtime_qualification = source_runtime_qualification_binding.runtime_qualification
    if (
        manifest.manifest_id != template.expected_manifest_id
        or manifest.system != template.system
        or manifest.relation != template.relation
        or manifest.source_qualification != source_qualification
        or manifest.runtime_qualification != runtime_qualification
        or manifest.observation_operator_qualification
        != template.observation_operator_qualification
        or manifest.evidence_world_id != template.evidence_world_id
        or manifest.evidence_domain is not template.evidence_domain
        or manifest.allowed_consumer_ids != template.expected_allowed_consumer_ids
        or manifest.information_cutoff_id != template.information_cutoff_id
        or tuple(_payload_topology(value) for value in manifest.payloads) != template.payloads
        or manifest.coordinates != template.coordinates
        or manifest.action_words != template.action_words
        or manifest.qualification_scopes != template.qualification_scopes
        or manifest.parent_visibility_ceilings
        != template.expected_manifest_parent_visibility_ceilings
        or manifest.visibility_ceiling is not template.expected_manifest_visibility_ceiling
        or manifest.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL
    ):
        raise ValueError("revealed manifest differs from the prospective projection template")
    scopes = {value.scope_id: value for value in manifest.qualification_scopes}
    qualification_scope = scopes[template.qualification_scope_id]
    observed_topology = tuple(_observation_topology(value) for value in manifest.observations)
    expected_without_tags = tuple(
        IdentificationProjectionObservationTemplate(
            observation_id=value.observation_id,
            physical_unit_instance_id=value.physical_unit_instance_id,
            nested_coordinate_ids=value.nested_coordinate_ids,
            denominator_cell_id=value.denominator_cell_id,
            chart_id=value.chart_id,
            split_id=value.split_id,
            role_id=value.role_id,
            member_id=value.member_id,
            candidate_version_id=value.candidate_version_id,
            qualification_view_id=value.qualification_view_id,
            receiver_id=value.receiver_id,
            receiver_clock_id=value.receiver_clock_id,
            native_frame_id=value.native_frame_id,
            payload_id=value.payload_id,
            payload_member_locator=value.payload_member_locator,
            action_word_id=value.action_word_id,
            occurrence_ids=value.occurrence_ids,
            tags=(),
        )
        for value in template.observations
    )
    if observed_topology != expected_without_tags:
        raise ValueError("revealed manifest observation topology differs from the template")
    claim_binding = ClaimUnitBinding(
        binding_id=template.claim_unit_binding_id,
        claim_id=qualification_scope.claim_id,
        scope=ObjectIdentity.from_record(
            qualification_scope.scope_id,
            qualification_scope,
        ),
        independent_unit_instance_ids=qualification_scope.independent_unit_instance_ids,
        aggregation_level_id=qualification_scope.aggregation_level_id,
    )
    plan = TaggedIdentificationProjectionPlan(
        plan_id=f"plan.{template.template_id}",
        manifest=ObjectIdentity.from_record(manifest.manifest_id, manifest),
        allowed_consumer_id=template.allowed_consumer_id,
        projection_capability=template.projection_capability,
        projection_config=template.projection_config,
        projection_implementation=template.projection_implementation,
        claim_unit_binding=claim_binding,
        value_partition=template.value_partition,
        observation_tags=tuple(
            ObservationTagBinding(
                observation_id=value.observation_id,
                tags=value.tags,
            )
            for value in template.observations
        ),
    )
    receipt = IdentificationProjectionTemplateBindingReceipt(
        binding_id=binding_id,
        template=ObjectIdentity.from_record(template.template_id, template),
        manifest=ObjectIdentity.from_record(manifest.manifest_id, manifest),
        projection_plan=ObjectIdentity.from_record(plan.plan_id, plan),
        binder_implementation_id=binder_implementation_id,
        binder_implementation_sha256=binder_implementation_sha256,
        topology_exact=True,
        compact_values_read=False,
        grants_scientific_promotion=False,
        outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        source_runtime_qualification_binding=(
            None
            if source_runtime_qualification_binding is None
            else ObjectIdentity.from_record(
                source_runtime_qualification_binding.receipt_id,
                source_runtime_qualification_binding,
            )
        ),
    )
    return plan, receipt


__all__ = [
    'ConditionalIdentificationProjectionObservationPair',
    'ConditionalIdentificationProjectionTemplate',
    'ConditionalQualificationRosterSelectionReceipt',
    'ConditionalQualificationRosterTemplate',
    'ConditionalQualificationSelectionKind',
    'ConditionalQualificationSlotSelection',
    'ConditionalQualificationSlot',
    'ConditionalQualificationTechnicalDisposition',
    'ConditionalQualificationTechnicalEvidence',
    'ConditionalQualificationTechnicalReason',
    'ConditionalQualificationWholeUnitOption',
    'IdentificationProjectionBindingStage',
    'IdentificationProjectionObservationTemplate',
    'IdentificationProjectionPayloadExtensionTemplate',
    'IdentificationProjectionPayloadTemplate',
    'IdentificationProjectionRole',
    'IdentificationProjectionTemplateBindingReceipt',
    'IdentificationProjectionTemplateSet',
    'IdentificationProjectionTemplate',
    'bind_identification_projection_template',
    'materialize_conditional_identification_projection_template_set',
]
