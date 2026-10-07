"Outcome-blind authoring and result contracts for finite native-HOLD cells.\n\nThe records in this module calibrate a measured native HOLD on a finite set of\npredeclared preparation coordinates.  They deliberately do not qualify a law,\ncreate raw admission evidence, derive reachability, or compile a controller.  Runtime\nowns the evaluator that reduces revealed action-local evidence into the\naggregate receipt defined here.\n"

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
from typing import ClassVar

from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord, ActionWordSupportStatus, ObservedActionOccurrence
from empirical_lawhood.kernel.admission import GateStatus
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import EvidenceLink, ObjectIdentity
from empirical_lawhood.kernel.references import (
    ArtifactIdentity,
    ExecutableReference,
    NamedDecimal,
)
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    canonical_json_bytes,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.planning.controller_study import DeliveryEquivalenceSpec, StageAwareDeliveryEquivalenceSpec
from empirical_lawhood.planning.source_pipelines import SourceRuntimeQualificationBindingReceipt


MAX_NATIVE_HOLD_ANCHOR_SLOTS = 64
MAX_NATIVE_HOLD_MODEL_MEMBERS = 32
MAX_NATIVE_HOLD_OCCURRENCE_ROLES = 8
MAX_NATIVE_HOLD_PREDICATES = 64
NATIVE_HOLD_ACTION_LOCAL_PROJECTION_SCHEMA = (
    'empirical-lawhood/runtime/native-hold-action-local-projection'
)


class NativeHoldCalibrationDisposition(StrEnum):
    SUPPORTED = "SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    UNEVALUABLE = "UNEVALUABLE"


class NativeHoldAnchorSelectionKind(StrEnum):
    PRIMARY = "PRIMARY"
    CONDITIONAL_ALTERNATIVE = "CONDITIONAL_ALTERNATIVE"


class NativeHoldSiblingSelectorRule(StrEnum):
    BYTEWISE_FIRST = "BYTEWISE_FIRST"


class NativeHoldCalibrationPredicateKind(StrEnum):
    SCALAR_GREATER_THAN = "SCALAR_GREATER_THAN"
    SCALAR_AT_LEAST = "SCALAR_AT_LEAST"
    SCALAR_AT_MOST = "SCALAR_AT_MOST"
    SCALAR_WITHIN_CLOSED_INTERVAL = "SCALAR_WITHIN_CLOSED_INTERVAL"
    BOOLEAN_EQUALS = "BOOLEAN_EQUALS"
    IDENTITY_EQUALS = "IDENTITY_EQUALS"


def preparation_values_fingerprint(values: tuple[NamedDecimal, ...]) -> str:
    """Return the exact byte identity used for finite preparation matching."""

    return sha256(canonical_json_bytes(values)).hexdigest()


@dataclass(frozen=True, slots=True)
class NativeHoldCalibrationAnchorOption(CanonicalRecord):
    """One predeclared primary or conditional preparation coordinate."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/native-hold-calibration-anchor-option'

    option_id: str
    coordinate_id: str
    preparation_values: tuple[NamedDecimal, ...]
    preparation_fingerprint: str

    def __post_init__(self) -> None:
        validate_stable_id(self.option_id, field_name="option_id")
        validate_stable_id(self.coordinate_id, field_name="coordinate_id")
        require_sorted_unique_ids(
            self.preparation_values,
            attribute="value_id",
            field_name="preparation_values",
        )
        if not self.preparation_values:
            raise ValueError("native-HOLD anchor requires preparation values")
        validate_sha256(
            self.preparation_fingerprint,
            field_name="preparation_fingerprint",
        )
        if self.preparation_fingerprint != preparation_values_fingerprint(self.preparation_values):
            raise ValueError("native-HOLD anchor fingerprint differs from exact values")


@dataclass(frozen=True, slots=True)
class NativeHoldCalibrationAnchorSlot(CanonicalRecord):
    """One fixed slot with at most one outcome-blind technical alternative."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/native-hold-calibration-anchor-slot'

    slot_id: str
    primary: NativeHoldCalibrationAnchorOption
    conditional_alternative: NativeHoldCalibrationAnchorOption | None

    def __post_init__(self) -> None:
        validate_stable_id(self.slot_id, field_name="slot_id")
        alternative = self.conditional_alternative
        if alternative is not None and (
            alternative.option_id == self.primary.option_id
            or alternative.coordinate_id == self.primary.coordinate_id
            or alternative.preparation_fingerprint == self.primary.preparation_fingerprint
        ):
            raise ValueError("native-HOLD alternative must be a distinct coordinate")


@dataclass(frozen=True, slots=True)
class NativeHoldCalibrationSelectedAnchor(CanonicalRecord):
    """The authenticated post-substitution selection for one fixed slot."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/native-hold-calibration-selected-anchor'

    selection_id: str
    slot_id: str
    selected: NativeHoldCalibrationAnchorOption
    selection_kind: NativeHoldAnchorSelectionKind

    def __post_init__(self) -> None:
        validate_stable_id(self.selection_id, field_name="selection_id")
        validate_stable_id(self.slot_id, field_name="slot_id")


@dataclass(frozen=True, slots=True)
class NativeHoldCalibrationSelectedRoster(CanonicalRecord):
    """Authenticated complete anchor roster selected without outcome access."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/native-hold-calibration-selected-roster'

    roster_id: str
    calibration_spec: ObjectIdentity
    selections: tuple[NativeHoldCalibrationSelectedAnchor, ...]
    selection_receipt: ObjectIdentity
    input_artifacts: tuple[ArtifactIdentity, ...]
    evidence_links: tuple[EvidenceLink, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.roster_id, field_name="roster_id")
        if self.calibration_spec.object_schema != NativeHoldDecisionCellCalibrationSpec.SCHEMA:
            raise ValueError("selected native-HOLD roster binds another specification")
        require_sorted_unique_ids(
            self.selections,
            attribute="slot_id",
            field_name="selections",
        )
        if not self.selections:
            raise ValueError("selected native-HOLD roster cannot be empty")
        if len({value.selected.coordinate_id for value in self.selections}) != len(self.selections):
            raise ValueError("selected native-HOLD roster reuses a coordinate")
        if len({value.selected.preparation_fingerprint for value in self.selections}) != len(
            self.selections
        ):
            raise ValueError("selected native-HOLD roster reuses a preparation")
        require_sorted_unique_ids(
            self.input_artifacts,
            attribute="artifact_id",
            field_name="input_artifacts",
        )
        require_sorted_unique_ids(
            self.evidence_links,
            attribute="link_id",
            field_name="evidence_links",
        )
        if not self.input_artifacts or not self.evidence_links:
            raise ValueError("selected native-HOLD roster requires authenticated evidence")
        artifact_ids = {value.artifact_id for value in self.input_artifacts}
        linked_artifact_ids = {
            artifact_id for link in self.evidence_links for artifact_id in link.artifact_ids
        }
        if artifact_ids != linked_artifact_ids:
            raise ValueError("selected native-HOLD artifacts differ from its evidence")
        if not any(
            self.selection_receipt in {value.source, value.target} for value in self.evidence_links
        ):
            raise ValueError("selected native-HOLD roster lacks its authenticated receipt link")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("native-HOLD anchor selection must remain outcome-blind")


@dataclass(frozen=True, slots=True)
class NativeHoldSiblingSelector(CanonicalRecord):
    """Frozen branch-neutral selection over exact sibling-HOLD keys."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/native-hold-sibling-selector'

    selector_id: str
    candidate_key_ids: tuple[str, ...]
    selected_candidate_key_id: str
    rule: NativeHoldSiblingSelectorRule

    def __post_init__(self) -> None:
        validate_stable_id(self.selector_id, field_name="selector_id")
        require_sorted_unique_strings(
            self.candidate_key_ids,
            field_name="candidate_key_ids",
            allow_empty=False,
        )
        for candidate_key_id in self.candidate_key_ids:
            validate_stable_id(candidate_key_id, field_name="candidate_key_ids")
        validate_stable_id(
            self.selected_candidate_key_id,
            field_name="selected_candidate_key_id",
        )
        if self.rule is not NativeHoldSiblingSelectorRule.BYTEWISE_FIRST:
            raise ValueError("native-HOLD selector rule is not registered")
        if self.selected_candidate_key_id != self.candidate_key_ids[0]:
            raise ValueError("native-HOLD selector must choose the bytewise-first key")


@dataclass(frozen=True, slots=True)
class NativeHoldCalibrationPredicateSpec(CanonicalRecord):
    """One noncompensating native calibration predicate."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/native-hold-calibration-predicate-spec'

    predicate_id: str
    quantity_id: str
    predicate_kind: NativeHoldCalibrationPredicateKind
    native_unit: str | None
    lower: Decimal | None
    upper: Decimal | None
    expected_boolean: bool | None
    expected_identity: ObjectIdentity | None

    def __post_init__(self) -> None:
        validate_stable_id(self.predicate_id, field_name="predicate_id")
        validate_stable_id(self.quantity_id, field_name="quantity_id")
        scalar = self.predicate_kind in {
            NativeHoldCalibrationPredicateKind.SCALAR_GREATER_THAN,
            NativeHoldCalibrationPredicateKind.SCALAR_AT_LEAST,
            NativeHoldCalibrationPredicateKind.SCALAR_AT_MOST,
            NativeHoldCalibrationPredicateKind.SCALAR_WITHIN_CLOSED_INTERVAL,
        }
        if scalar:
            if self.native_unit is None:
                raise ValueError("scalar native-HOLD predicate requires a native unit")
            validate_nonempty(self.native_unit, field_name="native_unit")
            if self.expected_boolean is not None or self.expected_identity is not None:
                raise ValueError("scalar native-HOLD predicate carries a categorical value")
            if self.lower is not None:
                validate_decimal(self.lower, field_name="lower")
            if self.upper is not None:
                validate_decimal(self.upper, field_name="upper")
            if self.predicate_kind in {
                NativeHoldCalibrationPredicateKind.SCALAR_GREATER_THAN,
                NativeHoldCalibrationPredicateKind.SCALAR_AT_LEAST,
            } and (self.lower is None or self.upper is not None):
                raise ValueError("lower native-HOLD predicate requires only a lower bound")
            if self.predicate_kind is NativeHoldCalibrationPredicateKind.SCALAR_AT_MOST and (
                self.upper is None or self.lower is not None
            ):
                raise ValueError("at-most native-HOLD predicate requires only an upper bound")
            if self.predicate_kind is (
                NativeHoldCalibrationPredicateKind.SCALAR_WITHIN_CLOSED_INTERVAL
            ) and (self.lower is None or self.upper is None or self.lower > self.upper):
                raise ValueError("closed native-HOLD interval is absent or reversed")
        elif self.predicate_kind is NativeHoldCalibrationPredicateKind.BOOLEAN_EQUALS:
            if (
                self.native_unit is not None
                or self.lower is not None
                or self.upper is not None
                or self.expected_boolean is None
                or self.expected_identity is not None
            ):
                raise ValueError("Boolean native-HOLD predicate requires only its expectation")
        elif (
            self.native_unit is not None
            or self.lower is not None
            or self.upper is not None
            or self.expected_boolean is not None
            or self.expected_identity is None
        ):
            raise ValueError("identity native-HOLD predicate requires only its expectation")


@dataclass(frozen=True, slots=True)
class NativeHoldDecisionCellCalibrationSpec(CanonicalRecord):
    """Preissued finite native-HOLD calibration design."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/native-hold-decision-cell-calibration-spec'

    calibration_spec_id: str
    hold_decision_cell_id: str
    active_decision_cell_ids: tuple[str, ...]
    anchor_slots: tuple[NativeHoldCalibrationAnchorSlot, ...]
    model_member_ids: tuple[str, ...]
    occurrence_role_ids: tuple[str, ...]
    source: ObjectIdentity
    schedule: ObjectIdentity
    native_hold_action_word: OccurrenceActionWord
    retained_history: ObjectIdentity
    horizon: ObjectIdentity
    delivery_equivalence: DeliveryEquivalenceSpec | StageAwareDeliveryEquivalenceSpec
    calibration_predicates: tuple[NativeHoldCalibrationPredicateSpec, ...]
    sibling_hold_selector: NativeHoldSiblingSelector
    evaluator: ExecutableReference
    evaluator_implementation_sha256: str
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.calibration_spec_id, field_name="calibration_spec_id")
        validate_stable_id(self.hold_decision_cell_id, field_name="hold_decision_cell_id")
        require_sorted_unique_strings(
            self.active_decision_cell_ids,
            field_name="active_decision_cell_ids",
            allow_empty=False,
        )
        for decision_cell_id in self.active_decision_cell_ids:
            validate_stable_id(decision_cell_id, field_name="active_decision_cell_ids")
        if self.hold_decision_cell_id in self.active_decision_cell_ids:
            raise ValueError("native-HOLD decision cell cannot be an active decision cell")
        require_sorted_unique_ids(
            self.anchor_slots,
            attribute="slot_id",
            field_name="anchor_slots",
        )
        require_sorted_unique_strings(
            self.model_member_ids,
            field_name="model_member_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.occurrence_role_ids,
            field_name="occurrence_role_ids",
            allow_empty=False,
        )
        for member_id in self.model_member_ids:
            validate_stable_id(member_id, field_name="model_member_ids")
        for role_id in self.occurrence_role_ids:
            validate_stable_id(role_id, field_name="occurrence_role_ids")
        for values, maximum, field_name in (
            (self.anchor_slots, MAX_NATIVE_HOLD_ANCHOR_SLOTS, "anchor_slots"),
            (self.model_member_ids, MAX_NATIVE_HOLD_MODEL_MEMBERS, "model_member_ids"),
            (
                self.occurrence_role_ids,
                MAX_NATIVE_HOLD_OCCURRENCE_ROLES,
                "occurrence_role_ids",
            ),
            (
                self.calibration_predicates,
                MAX_NATIVE_HOLD_PREDICATES,
                "calibration_predicates",
            ),
        ):
            if not values or len(values) > maximum:
                raise ValueError(f"{field_name} is empty or exceeds its bound")
        require_sorted_unique_ids(
            self.calibration_predicates,
            attribute="predicate_id",
            field_name="calibration_predicates",
        )
        primary_options = tuple(slot.primary for slot in self.anchor_slots)
        if len({value.option_id for value in primary_options}) != len(primary_options):
            raise ValueError("native-HOLD primary anchor option identities are reused")
        if len({value.coordinate_id for value in primary_options}) != len(primary_options):
            raise ValueError("native-HOLD primary anchor coordinates are reused")
        if len({value.preparation_fingerprint for value in primary_options}) != len(
            primary_options
        ):
            raise ValueError("native-HOLD primary anchor preparations are reused")

        alternatives = tuple(
            slot.conditional_alternative
            for slot in self.anchor_slots
            if slot.conditional_alternative is not None
        )
        primary_coordinate_ids = {value.coordinate_id for value in primary_options}
        primary_preparation_fingerprints = {
            value.preparation_fingerprint for value in primary_options
        }
        if any(
            value.coordinate_id in primary_coordinate_ids
            or value.preparation_fingerprint in primary_preparation_fingerprints
            for value in alternatives
        ):
            raise ValueError("native-HOLD alternative reuses a primary anchor")

        alternative_by_option_id: dict[str, NativeHoldCalibrationAnchorOption] = {}
        alternative_by_coordinate_id: dict[str, NativeHoldCalibrationAnchorOption] = {}
        alternative_by_preparation: dict[str, NativeHoldCalibrationAnchorOption] = {}
        for alternative in alternatives:
            previous_by_option = alternative_by_option_id.setdefault(
                alternative.option_id,
                alternative,
            )
            if previous_by_option != alternative:
                raise ValueError("repeated native-HOLD alternative option identity changes bytes")
            previous_by_coordinate = alternative_by_coordinate_id.setdefault(
                alternative.coordinate_id,
                alternative,
            )
            if (
                previous_by_coordinate.preparation_values != alternative.preparation_values
                or previous_by_coordinate.preparation_fingerprint
                != alternative.preparation_fingerprint
            ):
                raise ValueError("repeated native-HOLD alternative coordinate changes preparation")
            previous_by_preparation = alternative_by_preparation.setdefault(
                alternative.preparation_fingerprint,
                alternative,
            )
            if (
                previous_by_preparation.coordinate_id != alternative.coordinate_id
                or previous_by_preparation.preparation_values != alternative.preparation_values
            ):
                raise ValueError("repeated native-HOLD alternative preparation changes coordinate")
        word = self.native_hold_action_word
        if word.support_status is not ActionWordSupportStatus.SUPPORTED or not word.occurrences:
            raise ValueError("native-HOLD calibration requires a supported nonempty action word")
        if (
            self.retained_history.object_id != word.retained_history_id
            or self.horizon.object_id != word.horizon_id
        ):
            raise ValueError("native-HOLD action changes retained history or horizon")
        if len({value.channel.native_unit for value in word.occurrences}) != 1:
            raise ValueError("native-HOLD word mixes action units under one tolerance")
        delivery_units = (
            (self.delivery_equivalence.stage_value_tolerance.unit,)
            if isinstance(self.delivery_equivalence, DeliveryEquivalenceSpec)
            else tuple(
                value.tolerance.unit for value in self.delivery_equivalence.stage_value_tolerances
            )
        )
        if set(delivery_units) != {word.occurrences[0].channel.native_unit}:
            raise ValueError("native-HOLD delivery tolerance uses another action unit")
        validate_sha256(
            self.evaluator_implementation_sha256,
            field_name="evaluator_implementation_sha256",
        )
        if not self.evaluator.deterministic:
            raise ValueError("native-HOLD calibration evaluator must be deterministic")
        if (
            self.evaluator.input_schema != NATIVE_HOLD_ACTION_LOCAL_PROJECTION_SCHEMA
            or self.evaluator.output_schema != NativeHoldDecisionCellCalibrationReceipt.SCHEMA
        ):
            raise ValueError("native-HOLD evaluator uses another calibration port")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("native-HOLD calibration design must be outcome-blind")

    @property
    def expected_entry_keys(self) -> tuple[tuple[str, str, str], ...]:
        return tuple(
            sorted(
                (slot.slot_id, member_id, role_id)
                for slot in self.anchor_slots
                for member_id in self.model_member_ids
                for role_id in self.occurrence_role_ids
            )
        )

    @property
    def expected_entries(self) -> tuple[NativeHoldCalibrationEntryCoordinate, ...]:
        return tuple(
            NativeHoldCalibrationEntryCoordinate(
                entry_id=f"calibration-entry.{slot_id}.{member_id}.{role_id}",
                slot_id=slot_id,
                model_member_id=member_id,
                occurrence_role_id=role_id,
            )
            for slot_id, member_id, role_id in self.expected_entry_keys
        )


@dataclass(frozen=True, slots=True)
class NativeHoldCalibrationEntryCoordinate(CanonicalRecord):
    """One predeclared slot/member/occurrence-role product coordinate."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/native-hold-calibration-entry-coordinate'

    entry_id: str
    slot_id: str
    model_member_id: str
    occurrence_role_id: str

    def __post_init__(self) -> None:
        for name, value in (
            ("entry_id", self.entry_id),
            ("slot_id", self.slot_id),
            ("model_member_id", self.model_member_id),
            ("occurrence_role_id", self.occurrence_role_id),
        ):
            validate_stable_id(value, field_name=name)

    @property
    def entry_key(self) -> tuple[str, str, str]:
        return (self.slot_id, self.model_member_id, self.occurrence_role_id)


@dataclass(frozen=True, slots=True)
class NativeHoldCalibrationPredicateReceipt(CanonicalRecord):
    """One mechanically evaluated calibration predicate and its native operand."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/native-hold-calibration-predicate-receipt'

    receipt_id: str
    predicate: NativeHoldCalibrationPredicateSpec
    observed_scalar: NamedDecimal | None
    observed_boolean: bool | None
    observed_identity: ObjectIdentity | None
    status: GateStatus
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        supplied = sum(
            value is not None
            for value in (
                self.observed_scalar,
                self.observed_boolean,
                self.observed_identity,
            )
        )
        expected: tuple[GateStatus, tuple[str, ...]]
        if supplied == 0:
            expected = (GateStatus.UNEVALUABLE, ("CALIBRATION_OPERAND_UNAVAILABLE",))
        elif supplied != 1:
            raise ValueError("native-HOLD predicate receipt has several operand kinds")
        else:
            passed = _native_hold_predicate_passed(
                self.predicate,
                observed_scalar=self.observed_scalar,
                observed_boolean=self.observed_boolean,
                observed_identity=self.observed_identity,
            )
            expected = (
                GateStatus.PASS if passed else GateStatus.FAIL,
                () if passed else ("CALIBRATION_PREDICATE_FAILED",),
            )
        if (self.status, self.reason_codes) != expected:
            raise ValueError("native-HOLD predicate status is not mechanically derived")


def _native_hold_predicate_passed(
    predicate: NativeHoldCalibrationPredicateSpec,
    *,
    observed_scalar: NamedDecimal | None,
    observed_boolean: bool | None,
    observed_identity: ObjectIdentity | None,
) -> bool:
    if predicate.predicate_kind in {
        NativeHoldCalibrationPredicateKind.SCALAR_GREATER_THAN,
        NativeHoldCalibrationPredicateKind.SCALAR_AT_LEAST,
        NativeHoldCalibrationPredicateKind.SCALAR_AT_MOST,
        NativeHoldCalibrationPredicateKind.SCALAR_WITHIN_CLOSED_INTERVAL,
    }:
        if observed_scalar is None or observed_scalar.unit != predicate.native_unit:
            raise ValueError("native-HOLD scalar predicate operand uses another unit or kind")
        if predicate.predicate_kind is NativeHoldCalibrationPredicateKind.SCALAR_GREATER_THAN:
            return predicate.lower is not None and observed_scalar.value > predicate.lower
        if predicate.predicate_kind is NativeHoldCalibrationPredicateKind.SCALAR_AT_LEAST:
            return predicate.lower is not None and observed_scalar.value >= predicate.lower
        if predicate.predicate_kind is NativeHoldCalibrationPredicateKind.SCALAR_AT_MOST:
            return predicate.upper is not None and observed_scalar.value <= predicate.upper
        return (
            predicate.lower is not None
            and predicate.upper is not None
            and predicate.lower <= observed_scalar.value <= predicate.upper
        )
    if predicate.predicate_kind is NativeHoldCalibrationPredicateKind.BOOLEAN_EQUALS:
        if observed_boolean is None:
            raise ValueError("native-HOLD Boolean predicate requires a Boolean operand")
        return observed_boolean is predicate.expected_boolean
    if observed_identity is None:
        raise ValueError("native-HOLD identity predicate requires an identity operand")
    return observed_identity == predicate.expected_identity


@dataclass(frozen=True, slots=True)
class NativeHoldCalibrationOccurrenceReceipt(CanonicalRecord):
    """One slot/member/role calibration result retaining every decisive operand."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/native-hold-calibration-occurrence-receipt'

    receipt_id: str
    slot_id: str
    coordinate_id: str
    preparation_fingerprint: str
    model_member_id: str
    occurrence_role_id: str
    source: ObjectIdentity
    schedule: ObjectIdentity
    native_hold_action_word: ObjectIdentity
    retained_history: ObjectIdentity
    horizon: ObjectIdentity
    delivery_equivalence: DeliveryEquivalenceSpec | StageAwareDeliveryEquivalenceSpec
    observed_occurrences: tuple[ObservedActionOccurrence, ...]
    expected_occurrence_ids: tuple[str, ...]
    observed_occurrence_ids: tuple[str, ...]
    expected_predicate_ids: tuple[str, ...]
    predicate_receipts: tuple[NativeHoldCalibrationPredicateReceipt, ...]
    input_artifacts: tuple[ArtifactIdentity, ...]
    evidence_links: tuple[EvidenceLink, ...]
    disposition: NativeHoldCalibrationDisposition
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("receipt_id", self.receipt_id),
            ("slot_id", self.slot_id),
            ("coordinate_id", self.coordinate_id),
            ("model_member_id", self.model_member_id),
            ("occurrence_role_id", self.occurrence_role_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_sha256(
            self.preparation_fingerprint,
            field_name="preparation_fingerprint",
        )
        if self.native_hold_action_word.object_schema != OccurrenceActionWord.SCHEMA:
            raise ValueError("native-HOLD occurrence receipt binds another action schema")
        require_sorted_unique_strings(
            self.expected_occurrence_ids,
            field_name="expected_occurrence_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.observed_occurrence_ids,
            field_name="observed_occurrence_ids",
        )
        occurrence_ids = tuple(value.expected_occurrence_id for value in self.observed_occurrences)
        if occurrence_ids != tuple(sorted(set(occurrence_ids))):
            raise ValueError("native-HOLD observed occurrences must be sorted and unique")
        if self.observed_occurrence_ids != occurrence_ids:
            raise ValueError("native-HOLD observed occurrence IDs differ from their records")
        require_sorted_unique_strings(
            self.expected_predicate_ids,
            field_name="expected_predicate_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(
            self.predicate_receipts,
            attribute="receipt_id",
            field_name="predicate_receipts",
        )
        require_sorted_unique_ids(
            self.input_artifacts,
            attribute="artifact_id",
            field_name="input_artifacts",
        )
        require_sorted_unique_ids(
            self.evidence_links,
            attribute="link_id",
            field_name="evidence_links",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if not self.input_artifacts or not self.evidence_links:
            raise ValueError("native-HOLD occurrence receipt requires exact evidence")
        artifact_ids = {value.artifact_id for value in self.input_artifacts}
        linked_artifact_ids = {
            artifact_id for link in self.evidence_links for artifact_id in link.artifact_ids
        }
        if artifact_ids != linked_artifact_ids:
            raise ValueError("native-HOLD occurrence artifacts differ from evidence links")
        if (
            tuple(sorted(value.predicate.predicate_id for value in self.predicate_receipts))
            != self.expected_predicate_ids
        ):
            raise ValueError("native-HOLD occurrence predicate roster is incomplete")
        occurrence_roster_incomplete = (
            self.observed_occurrence_ids != self.expected_occurrence_ids
            or any(not value.complete for value in self.observed_occurrences)
        )
        predicate_unevaluable = any(
            value.status is GateStatus.UNEVALUABLE for value in self.predicate_receipts
        )
        if (
            occurrence_roster_incomplete or predicate_unevaluable
        ) and self.disposition is not NativeHoldCalibrationDisposition.UNEVALUABLE:
            raise ValueError("incomplete native-HOLD occurrence must remain unevaluable")
        if self.disposition is NativeHoldCalibrationDisposition.SUPPORTED:
            if self.reason_codes or any(
                value.status is not GateStatus.PASS for value in self.predicate_receipts
            ):
                raise ValueError("supported native-HOLD occurrence contains a failed operand")
        elif not self.reason_codes:
            raise ValueError("non-supported native-HOLD occurrence requires decisive reasons")

    @property
    def entry_key(self) -> tuple[str, str, str]:
        return (self.slot_id, self.model_member_id, self.occurrence_role_id)


@dataclass(frozen=True, slots=True)
class NativeHoldDecisionCellCalibrationReceipt(CanonicalRecord):
    """Complete noncompensating aggregate over the finite HOLD-only domain."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/native-hold-decision-cell-calibration-receipt'

    receipt_id: str
    calibration_spec: ObjectIdentity
    selected_roster: ObjectIdentity
    hold_decision_cell_id: str
    active_decision_cell_ids: tuple[str, ...]
    selected_sibling_hold_candidate_key_id: str
    expected_entries: tuple[NativeHoldCalibrationEntryCoordinate, ...]
    occurrence_receipts: tuple[NativeHoldCalibrationOccurrenceReceipt, ...]
    evaluator: ExecutableReference
    input_artifacts: tuple[ArtifactIdentity, ...]
    evidence_links: tuple[EvidenceLink, ...]
    disposition: NativeHoldCalibrationDisposition
    reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess
    source_runtime_qualification_binding: ObjectIdentity | None = None

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        if self.calibration_spec.object_schema != NativeHoldDecisionCellCalibrationSpec.SCHEMA:
            raise ValueError("native-HOLD receipt binds another calibration specification")
        if self.selected_roster.object_schema != NativeHoldCalibrationSelectedRoster.SCHEMA:
            raise ValueError("native-HOLD receipt binds another selected roster")
        if self.source_runtime_qualification_binding is not None and (
            self.source_runtime_qualification_binding.object_schema
            != SourceRuntimeQualificationBindingReceipt.SCHEMA
        ):
            raise ValueError("native-HOLD receipt names another source/runtime mapping")
        validate_stable_id(self.hold_decision_cell_id, field_name="hold_decision_cell_id")
        validate_stable_id(
            self.selected_sibling_hold_candidate_key_id,
            field_name="selected_sibling_hold_candidate_key_id",
        )
        require_sorted_unique_strings(
            self.active_decision_cell_ids,
            field_name="active_decision_cell_ids",
            allow_empty=False,
        )
        if self.hold_decision_cell_id in self.active_decision_cell_ids:
            raise ValueError("aggregate HOLD cell aliases an active decision cell")
        require_sorted_unique_ids(
            self.expected_entries,
            attribute="entry_id",
            field_name="expected_entries",
        )
        if not self.expected_entries:
            raise ValueError("native-HOLD aggregate expected-entry roster cannot be empty")
        identifiers = tuple(value.receipt_id for value in self.occurrence_receipts)
        if identifiers != tuple(sorted(set(identifiers))) or not identifiers:
            raise ValueError("native-HOLD occurrence receipt IDs must be sorted and complete")
        if len({value.entry_key for value in self.occurrence_receipts}) != len(
            self.occurrence_receipts
        ):
            raise ValueError("native-HOLD aggregate duplicates a slot/member/role entry")
        if {value.entry_key for value in self.expected_entries} != {
            value.entry_key for value in self.occurrence_receipts
        } or len(self.expected_entries) != len(self.occurrence_receipts):
            raise ValueError("native-HOLD aggregate omits or adds expected entries")
        require_sorted_unique_ids(
            self.input_artifacts,
            attribute="artifact_id",
            field_name="input_artifacts",
        )
        require_sorted_unique_ids(
            self.evidence_links,
            attribute="link_id",
            field_name="evidence_links",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if not self.input_artifacts or not self.evidence_links:
            raise ValueError("native-HOLD aggregate requires complete evidence")
        artifact_ids = {value.artifact_id for value in self.input_artifacts}
        linked_artifact_ids = {
            artifact_id for link in self.evidence_links for artifact_id in link.artifact_ids
        }
        if artifact_ids != linked_artifact_ids:
            raise ValueError("native-HOLD aggregate artifacts differ from evidence links")
        expected_disposition = (
            NativeHoldCalibrationDisposition.UNEVALUABLE
            if any(
                value.disposition is NativeHoldCalibrationDisposition.UNEVALUABLE
                for value in self.occurrence_receipts
            )
            else NativeHoldCalibrationDisposition.NOT_SUPPORTED
            if any(
                value.disposition is NativeHoldCalibrationDisposition.NOT_SUPPORTED
                for value in self.occurrence_receipts
            )
            else NativeHoldCalibrationDisposition.SUPPORTED
        )
        if self.disposition is not expected_disposition:
            raise ValueError("native-HOLD aggregate disposition is not the entry intersection")
        if self.disposition is NativeHoldCalibrationDisposition.SUPPORTED:
            if self.reason_codes:
                raise ValueError("supported native-HOLD aggregate cannot carry reasons")
        elif not self.reason_codes:
            raise ValueError("non-supported native-HOLD aggregate requires reasons")
        if self.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL:
            raise ValueError("native-HOLD aggregate requires evaluator-reveal access")


__all__ = [
    "MAX_NATIVE_HOLD_ANCHOR_SLOTS",
    "MAX_NATIVE_HOLD_MODEL_MEMBERS",
    "MAX_NATIVE_HOLD_OCCURRENCE_ROLES",
    "MAX_NATIVE_HOLD_PREDICATES",
    'NativeHoldAnchorSelectionKind',
    'NativeHoldCalibrationAnchorOption',
    'NativeHoldCalibrationAnchorSlot',
    'NativeHoldCalibrationDisposition',
    'NativeHoldCalibrationEntryCoordinate',
    'NativeHoldCalibrationOccurrenceReceipt',
    'NativeHoldCalibrationPredicateKind',
    'NativeHoldCalibrationPredicateReceipt',
    'NativeHoldCalibrationPredicateSpec',
    'NativeHoldCalibrationSelectedAnchor',
    'NativeHoldCalibrationSelectedRoster',
    'NativeHoldDecisionCellCalibrationReceipt',
    'NativeHoldDecisionCellCalibrationSpec',
    'NativeHoldSiblingSelectorRule',
    'NativeHoldSiblingSelector',
    "preparation_values_fingerprint",
]
