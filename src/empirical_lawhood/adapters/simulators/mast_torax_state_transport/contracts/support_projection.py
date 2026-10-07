"Mapped development two-donor projection into the shared action/preparation recurrence receipt."

from __future__ import annotations

from dataclasses import dataclass
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.runtime.adaptive_acquisition_validation import AcquisitionEvidenceRole, ActionPreparationEvidenceCell, ActionPreparationRecurrenceEvidenceSource, ActionPreparationRecurrenceProofOwner, ActionPreparationRecurrenceReceipt, ActionPreparationRecurrenceStatus, RecurrenceEvidenceDomain, RecurrenceEvidenceSourceKind, evaluate_action_preparation_recurrence


_METRIC_CONTEXT = Context(prec=50, rounding=ROUND_HALF_EVEN)


class MappedSupportEvidenceRole(StrEnum):
    DEVELOPMENT_PRECOMMITMENT = "DEVELOPMENT_PRECOMMITMENT"
    NOMINAL_REFERENCE = "NOMINAL_REFERENCE"
    CONFIRMATORY = "CONFIRMATORY"


@dataclass(frozen=True, slots=True)
class MappedSupportMetricAxis(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/mast-torax-state-transport/contracts/mapped-support-metric-axis'

    axis_id: str
    native_unit: str
    whitening_scale: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.axis_id, field_name="axis_id")
        validate_nonempty(self.native_unit, field_name="native_unit")
        validate_decimal(
            self.whitening_scale,
            field_name="whitening_scale",
            minimum=Decimal(0),
        )
        if self.whitening_scale == 0:
            raise ValueError("mapped support whitening scale must be strictly positive")


@dataclass(frozen=True, slots=True)
class MappedStateCoordinate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/mast-torax-state-transport/contracts/mapped-state-coordinate'

    axis_id: str
    native_value: NamedDecimal

    def __post_init__(self) -> None:
        validate_stable_id(self.axis_id, field_name="axis_id")
        if self.native_value.value_id != f"mapped-coordinate.{self.axis_id}":
            raise ValueError("mapped coordinate value does not bind its axis")


@dataclass(frozen=True, slots=True)
class MappedDevelopmentPanelCell(CanonicalRecord):
    "One precommitment mapped-development TORAX action/member response cell."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/mast-torax-state-transport/contracts/mapped-development-panel-cell'

    cell_id: str
    requested_action_word: ObjectIdentity
    model_member_id: str
    accepted_action: ObjectIdentity
    applied_action: ObjectIdentity
    realized_action: ObjectIdentity
    delivery_receipt: ObjectIdentity
    response_evidence: ObjectIdentity
    recurrence_predicate: ObjectIdentity
    uncertainty_rule: ObjectIdentity
    matched_clone: bool
    delivery_valid: bool
    recurrence_passed: bool
    evidence_role: MappedSupportEvidenceRole
    causal_cutoff: ObjectIdentity
    outcome_access: OutcomeAccess
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.cell_id, field_name="cell_id")
        validate_stable_id(self.model_member_id, field_name="model_member_id")
        if any(
            value.object_schema != OccurrenceActionWord.SCHEMA
            for value in (
                self.requested_action_word,
                self.accepted_action,
                self.applied_action,
                self.realized_action,
            )
        ):
            raise ValueError("mapped panel requires exact requested-to-realized ActionWords")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.matched_clone and self.delivery_valid and self.recurrence_passed:
            if self.reason_codes:
                raise ValueError("valid mapped development cell cannot carry failure reasons")
        elif not self.reason_codes:
            raise ValueError("invalid mapped development cell requires a typed reason")

    @property
    def action_word_id(self) -> str:
        return self.requested_action_word.object_id


@dataclass(frozen=True, slots=True)
class MappedDevelopmentDonorState(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/mast-torax-state-transport/contracts/mapped-development-donor-state'

    donor_state_id: str
    preparation_unit_id: str
    source_fingerprint: str
    state_coordinates: tuple[MappedStateCoordinate, ...]
    panel_cells: tuple[MappedDevelopmentPanelCell, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.donor_state_id, field_name="donor_state_id")
        validate_stable_id(self.preparation_unit_id, field_name="preparation_unit_id")
        validate_sha256(self.source_fingerprint, field_name="source_fingerprint")
        require_sorted_unique_ids(
            self.state_coordinates,
            attribute="axis_id",
            field_name="state_coordinates",
        )
        require_sorted_unique_ids(
            self.panel_cells,
            attribute="cell_id",
            field_name="panel_cells",
        )


@dataclass(frozen=True, slots=True)
class MappedActionSupportProjection(CanonicalRecord):
    "Frozen 24-donor mapped development metric and complete action/member support contract."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/mast-torax-state-transport/contracts/mapped-action-support-projection'

    projection_config_id: str
    metric_axes: tuple[MappedSupportMetricAxis, ...]
    action_words: tuple[ObjectIdentity, ...]
    matched_hold_action_word: ObjectIdentity
    model_member_ids: tuple[str, ...]
    donor_state_count: int
    minimum_finite_second_neighbour_distances: int
    support_quantile_numerator: int
    support_quantile_denominator: int
    causal_cutoff: ObjectIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.projection_config_id, field_name="projection_config_id")
        require_sorted_unique_ids(
            self.metric_axes,
            attribute="axis_id",
            field_name="metric_axes",
        )
        if not self.metric_axes:
            raise ValueError("mapped support projection requires a native metric")
        require_sorted_unique_ids(
            self.action_words,
            attribute="object_id",
            field_name="action_words",
        )
        if len(self.action_words) != 3 or any(
            value.object_schema != OccurrenceActionWord.SCHEMA for value in self.action_words
        ):
            raise ValueError("mapped support projection requires the exact three-word chart")
        if self.matched_hold_action_word not in self.action_words:
            raise ValueError("mapped support HOLD is outside the exact action chart")
        require_sorted_unique_strings(
            self.model_member_ids,
            field_name="model_member_ids",
            allow_empty=False,
        )
        if len(self.model_member_ids) != 6:
            raise ValueError("mapped support projection requires all six TORAX members")
        if self.donor_state_count != 24:
            raise ValueError("mapped support projection requires the exact 24-donor development roster")
        if self.minimum_finite_second_neighbour_distances != 20:
            raise ValueError("mapped support projection requires the frozen finite-distance floor")
        if self.support_quantile_numerator != 9 or self.support_quantile_denominator != 10:
            raise ValueError("mapped support projection requires the nearest-rank 0.90 quantile")


@dataclass(frozen=True, slots=True)
class MappedDonorDistanceReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/mast-torax-state-transport/contracts/mapped-donor-distance-receipt'

    receipt_id: str
    subject_state_id: str
    donor_state_id: str
    donor_preparation_unit_id: str
    donor_source_fingerprint: str
    standardized_distance: Decimal
    support_radius: Decimal
    within_support: bool
    metric_config: ObjectIdentity

    def __post_init__(self) -> None:
        for name, value in (
            ("receipt_id", self.receipt_id),
            ("subject_state_id", self.subject_state_id),
            ("donor_state_id", self.donor_state_id),
            ("donor_preparation_unit_id", self.donor_preparation_unit_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_sha256(
            self.donor_source_fingerprint,
            field_name="donor_source_fingerprint",
        )
        validate_decimal(
            self.standardized_distance,
            field_name="standardized_distance",
            minimum=Decimal(0),
        )
        validate_decimal(
            self.support_radius,
            field_name="support_radius",
            minimum=Decimal(0),
        )
        if self.within_support is not (self.standardized_distance <= self.support_radius):
            raise ValueError("mapped donor support disposition is not distance-derived")


@dataclass(frozen=True, slots=True)
class MappedDevelopmentSupportAtlas(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/mast-torax-state-transport/contracts/mapped-development-support-atlas'

    atlas_id: str
    config: MappedActionSupportProjection
    donors: tuple[MappedDevelopmentDonorState, ...]
    second_neighbour_distances: tuple[MappedDonorDistanceReceipt, ...]
    support_radius: Decimal
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.atlas_id, field_name="atlas_id")
        require_sorted_unique_ids(
            self.donors,
            attribute="donor_state_id",
            field_name="donors",
        )
        if len(self.donors) != self.config.donor_state_count:
            raise ValueError("mapped development atlas does not contain the exact 24-donor development roster")
        if len({value.preparation_unit_id for value in self.donors}) != len(self.donors):
            raise ValueError("mapped development donors reuse a preparation identity")
        if len({value.source_fingerprint for value in self.donors}) != len(self.donors):
            raise ValueError("mapped development donors duplicate a native state fingerprint")
        require_sorted_unique_ids(
            self.second_neighbour_distances,
            attribute="receipt_id",
            field_name="second_neighbour_distances",
        )
        if len(self.second_neighbour_distances) != len(self.donors):
            raise ValueError("mapped atlas lacks one second-neighbour distance per donor")
        validate_decimal(
            self.support_radius,
            field_name="support_radius",
            minimum=Decimal(0),
        )
        if self.support_radius == 0:
            raise ValueError("mapped support radius must be strictly positive")
        if self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
            raise ValueError("mapped donor atlas must remain excluded-development evidence")
        expected_axes = tuple(value.axis_id for value in self.config.metric_axes)
        expected_panel = {
            (action.object_id, member)
            for action in self.config.action_words
            for member in self.config.model_member_ids
        }
        all_cells: list[MappedDevelopmentPanelCell] = []
        for donor in self.donors:
            if tuple(value.axis_id for value in donor.state_coordinates) != expected_axes:
                raise ValueError("mapped donor coordinates differ from the frozen metric")
            for axis, coordinate in zip(
                self.config.metric_axes,
                donor.state_coordinates,
                strict=True,
            ):
                if coordinate.native_value.unit != axis.native_unit:
                    raise ValueError("mapped donor coordinate changes native unit")
            observed_panel = {
                (value.action_word_id, value.model_member_id) for value in donor.panel_cells
            }
            if len(donor.panel_cells) != len(expected_panel) or observed_panel != expected_panel:
                raise ValueError("mapped donor lacks the complete action-by-member panel")
            if any(
                value.evidence_role is not MappedSupportEvidenceRole.DEVELOPMENT_PRECOMMITMENT
                or value.causal_cutoff != self.config.causal_cutoff
                or value.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
                or not value.matched_clone
                or not value.delivery_valid
                or not value.recurrence_passed
                for value in donor.panel_cells
            ):
                raise ValueError("post-reference or invalid evidence cannot enter 24-donor mapped development support")
            all_cells.extend(donor.panel_cells)
        if len({value.cell_id for value in all_cells}) != len(all_cells):
            raise ValueError("24-donor mapped development panel duplicates a scientific cell")
        if len({value.delivery_receipt for value in all_cells}) != len(all_cells) or len(
            {value.response_evidence for value in all_cells}
        ) != len(all_cells):
            raise ValueError("24-donor mapped development panel reuses delivery or response evidence")
        config_identity = ObjectIdentity.from_record(
            self.config.projection_config_id,
            self.config,
        )
        expected_distances: dict[str, tuple[Decimal, MappedDevelopmentDonorState]] = {}
        for subject in self.donors:
            neighbours = tuple(
                sorted(
                    (
                        (
                            _distance(
                                self.config,
                                subject.state_coordinates,
                                other.state_coordinates,
                            ),
                            other,
                        )
                        for other in self.donors
                        if other.donor_state_id != subject.donor_state_id
                    ),
                    key=lambda value: (value[0], value[1].donor_state_id),
                )
            )
            expected_distances[subject.donor_state_id] = neighbours[1]
        finite = tuple(value[0] for value in expected_distances.values())
        expected_radius = _nearest_rank_quantile(
            finite,
            numerator=self.config.support_quantile_numerator,
            denominator=self.config.support_quantile_denominator,
        )
        if self.support_radius != expected_radius:
            raise ValueError("mapped support radius is not the frozen nearest-rank quantile")
        receipt_by_subject = {
            value.subject_state_id: value for value in self.second_neighbour_distances
        }
        if set(receipt_by_subject) != {value.donor_state_id for value in self.donors}:
            raise ValueError("mapped second-neighbour receipts change the 24-donor mapped development subject roster")
        donor_by_id = {value.donor_state_id: value for value in self.donors}
        for subject_id, (distance, neighbour) in expected_distances.items():
            receipt = receipt_by_subject[subject_id]
            if (
                receipt.donor_state_id != neighbour.donor_state_id
                or receipt.donor_preparation_unit_id != neighbour.preparation_unit_id
                or receipt.donor_source_fingerprint != neighbour.source_fingerprint
                or receipt.standardized_distance != distance
                or receipt.support_radius != expected_radius
                or receipt.within_support is not (distance <= expected_radius)
                or receipt.metric_config != config_identity
                or receipt.subject_state_id not in donor_by_id
            ):
                raise ValueError("mapped second-neighbour receipt is not metric-derived")


@dataclass(frozen=True, slots=True)
class MappedStateSupportQuery(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/mast-torax-state-transport/contracts/mapped-state-support-query'

    query_id: str
    target_state_id: str
    task_id: str
    arm_id: str
    state_coordinates: tuple[MappedStateCoordinate, ...]
    claimed_active_action_word: ObjectIdentity
    method_config: ObjectIdentity
    precommitment: ObjectIdentity
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name, value in (
            ("query_id", self.query_id),
            ("target_state_id", self.target_state_id),
            ("task_id", self.task_id),
            ("arm_id", self.arm_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_ids(
            self.state_coordinates,
            attribute="axis_id",
            field_name="state_coordinates",
        )
        if self.claimed_active_action_word.object_schema != OccurrenceActionWord.SCHEMA:
            raise ValueError("mapped support query requires an exact active ActionWord")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("mapped support query must precede every protected outcome")


@dataclass(frozen=True, slots=True)
class MappedActionSupportProjectionReceipt(CanonicalRecord):
    "Distance proof in mapped coordinates, carrying the shared action/preparation recurrence receipt."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/mast-torax-state-transport/contracts/mapped-action-support-projection-receipt'

    receipt_id: str
    atlas: MappedDevelopmentSupportAtlas
    query: MappedStateSupportQuery
    selected_donor_distances: tuple[MappedDonorDistanceReceipt, ...]
    recurrence_receipt: ActionPreparationRecurrenceReceipt

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        require_sorted_unique_ids(
            self.selected_donor_distances,
            attribute="receipt_id",
            field_name="selected_donor_distances",
        )
        if len(self.selected_donor_distances) != 2 or any(
            not value.within_support for value in self.selected_donor_distances
        ):
            raise ValueError("mapped support receipt requires two in-radius donors")
        donor_preparations = tuple(
            sorted(value.donor_preparation_unit_id for value in self.selected_donor_distances)
        )
        recurrence = self.recurrence_receipt
        if (
            recurrence.status is not ActionPreparationRecurrenceStatus.SUPPORTED
            or recurrence.evidence_source.source_record
            != ObjectIdentity.from_record(self.atlas.atlas_id, self.atlas)
            or recurrence.evidence_source.source_kind
            is not RecurrenceEvidenceSourceKind.MAPPED_DEVELOPMENT_SUPPORT_ATLAS
            or recurrence.task_id != self.query.task_id
            or recurrence.arm_id != self.query.arm_id
            or recurrence.active_action_word != self.query.claimed_active_action_word
            or recurrence.matched_hold_action_word != self.atlas.config.matched_hold_action_word
            or recurrence.preparation_unit_ids != donor_preparations
            or recurrence.model_member_ids != self.atlas.config.model_member_ids
            or recurrence.method_config != self.query.method_config
        ):
            raise ValueError("mapped support wrapper changes its shared recurrence truth")
        metric_identity = ObjectIdentity.from_record(
            self.atlas.config.projection_config_id,
            self.atlas.config,
        )
        if any(
            value.subject_state_id != self.query.target_state_id
            or value.metric_config != metric_identity
            or value.support_radius != self.atlas.support_radius
            for value in self.selected_donor_distances
        ):
            raise ValueError("mapped support receipt changes distance or boundary identity")
        ranked = tuple(
            sorted(
                (
                    (
                        _distance(
                            self.atlas.config,
                            self.query.state_coordinates,
                            donor.state_coordinates,
                        ),
                        donor,
                    )
                    for donor in self.atlas.donors
                ),
                key=lambda value: (value[0], value[1].donor_state_id),
            )
        )[:2]
        expected = {donor.donor_state_id: (distance, donor) for distance, donor in ranked}
        observed = {value.donor_state_id: value for value in self.selected_donor_distances}
        if set(observed) != set(expected):
            raise ValueError("mapped support receipt substitutes its nearest donors")
        for donor_id, (distance, donor) in expected.items():
            receipt = observed[donor_id]
            if (
                receipt.standardized_distance != distance
                or receipt.donor_preparation_unit_id != donor.preparation_unit_id
                or receipt.donor_source_fingerprint != donor.source_fingerprint
            ):
                raise ValueError("mapped query distance is not metric-derived")


def _coordinate_map(
    coordinates: tuple[MappedStateCoordinate, ...],
) -> dict[str, NamedDecimal]:
    return {value.axis_id: value.native_value for value in coordinates}


def _distance(
    config: MappedActionSupportProjection,
    left: tuple[MappedStateCoordinate, ...],
    right: tuple[MappedStateCoordinate, ...],
) -> Decimal:
    left_map = _coordinate_map(left)
    right_map = _coordinate_map(right)
    expected = {value.axis_id for value in config.metric_axes}
    if set(left_map) != expected or set(right_map) != expected:
        raise ValueError("mapped distance operand differs from the frozen metric axes")
    with localcontext(_METRIC_CONTEXT) as context:
        squared = Decimal(0)
        for axis in config.metric_axes:
            left_value = left_map[axis.axis_id]
            right_value = right_map[axis.axis_id]
            if left_value.unit != axis.native_unit or right_value.unit != axis.native_unit:
                raise ValueError("mapped distance operand changes native unit")
            delta = (left_value.value - right_value.value) / axis.whitening_scale
            squared += delta * delta
        return context.sqrt(squared)


def _nearest_rank_quantile(
    values: tuple[Decimal, ...],
    *,
    numerator: int,
    denominator: int,
) -> Decimal:
    if not values:
        raise ValueError("nearest-rank quantile requires observations")
    rank = (numerator * len(values) + denominator - 1) // denominator
    return tuple(sorted(values))[rank - 1]


def build_mapped_development_support_atlas(
    *,
    atlas_id: str,
    config: MappedActionSupportProjection,
    donors: tuple[MappedDevelopmentDonorState, ...],
) -> MappedDevelopmentSupportAtlas:
    ordered = tuple(sorted(donors, key=lambda value: value.donor_state_id))
    if len(ordered) != config.donor_state_count:
        raise ValueError("mapped development atlas requires exactly 24 donors")
    config_identity = ObjectIdentity.from_record(config.projection_config_id, config)
    distances: list[MappedDonorDistanceReceipt] = []
    for subject in ordered:
        neighbours = tuple(
            sorted(
                (
                    (_distance(config, subject.state_coordinates, other.state_coordinates), other)
                    for other in ordered
                    if other.donor_state_id != subject.donor_state_id
                ),
                key=lambda value: (value[0], value[1].donor_state_id),
            )
        )
        if len(neighbours) < 2:
            raise ValueError("mapped donor has fewer than two distinct neighbours")
        second_distance, second = neighbours[1]
        distances.append(
            MappedDonorDistanceReceipt(
                receipt_id=f"mapped-second-neighbour.{subject.donor_state_id}",
                subject_state_id=subject.donor_state_id,
                donor_state_id=second.donor_state_id,
                donor_preparation_unit_id=second.preparation_unit_id,
                donor_source_fingerprint=second.source_fingerprint,
                standardized_distance=second_distance,
                support_radius=second_distance,
                within_support=True,
                metric_config=config_identity,
            )
        )
    finite = tuple(
        value.standardized_distance
        for value in distances
        if value.standardized_distance.is_finite()
    )
    if len(finite) < config.minimum_finite_second_neighbour_distances:
        raise ValueError("mapped support metric lacks twenty finite distances")
    radius = _nearest_rank_quantile(
        finite,
        numerator=config.support_quantile_numerator,
        denominator=config.support_quantile_denominator,
    )
    normalized = tuple(
        MappedDonorDistanceReceipt(
            receipt_id=value.receipt_id,
            subject_state_id=value.subject_state_id,
            donor_state_id=value.donor_state_id,
            donor_preparation_unit_id=value.donor_preparation_unit_id,
            donor_source_fingerprint=value.donor_source_fingerprint,
            standardized_distance=value.standardized_distance,
            support_radius=radius,
            within_support=value.standardized_distance <= radius,
            metric_config=value.metric_config,
        )
        for value in distances
    )
    return MappedDevelopmentSupportAtlas(
        atlas_id=atlas_id,
        config=config,
        donors=ordered,
        second_neighbour_distances=tuple(sorted(normalized, key=lambda value: value.receipt_id)),
        support_radius=radius,
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
    )


def project_mapped_action_support(
    *,
    atlas: MappedDevelopmentSupportAtlas,
    query: MappedStateSupportQuery,
    proof_owner: ActionPreparationRecurrenceProofOwner,
) -> MappedActionSupportProjectionReceipt:
    "Select two precommitted donors and delegate recurrence evaluation to the shared owner."

    config = atlas.config
    if (
        query.claimed_active_action_word not in config.action_words
        or query.claimed_active_action_word == config.matched_hold_action_word
    ):
        raise ValueError("mapped query claims an action outside active support")
    if tuple(value.axis_id for value in query.state_coordinates) != tuple(
        value.axis_id for value in config.metric_axes
    ):
        raise ValueError("mapped protected state differs from the frozen metric")
    all_panel_evidence = {
        identity
        for donor in atlas.donors
        for cell in donor.panel_cells
        for identity in (cell.delivery_receipt, cell.response_evidence)
    }
    evidence_source = ActionPreparationRecurrenceEvidenceSource(
        source_id=f"recurrence-evidence-source.{atlas.atlas_id}",
        source_kind=RecurrenceEvidenceSourceKind.MAPPED_DEVELOPMENT_SUPPORT_ATLAS,
        source_record=ObjectIdentity.from_record(atlas.atlas_id, atlas),
        causal_domain=RecurrenceEvidenceDomain.EXCLUDED_MAPPED_DEVELOPMENT,
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        evidence_inputs=tuple(sorted(all_panel_evidence, key=lambda value: value.object_id)),
    )
    ranked = tuple(
        sorted(
            (
                (_distance(config, query.state_coordinates, donor.state_coordinates), donor)
                for donor in atlas.donors
            ),
            key=lambda value: (value[0], value[1].donor_state_id),
        )
    )
    selected = ranked[:2]
    if len(selected) != 2 or any(value[0] > atlas.support_radius for value in selected):
        raise ValueError("MAPPED_ACTION_NOT_CLAIMED_UNSUPPORTED: fewer than two local donors")
    metric_identity = ObjectIdentity.from_record(
        config.projection_config_id,
        config,
    )
    distance_receipts = tuple(
        sorted(
            (
                MappedDonorDistanceReceipt(
                    receipt_id=(f"mapped-query-distance.{query.query_id}.{donor.donor_state_id}"),
                    subject_state_id=query.target_state_id,
                    donor_state_id=donor.donor_state_id,
                    donor_preparation_unit_id=donor.preparation_unit_id,
                    donor_source_fingerprint=donor.source_fingerprint,
                    standardized_distance=distance,
                    support_radius=atlas.support_radius,
                    within_support=True,
                    metric_config=metric_identity,
                )
                for distance, donor in selected
            ),
            key=lambda value: value.receipt_id,
        )
    )
    requested = {
        query.claimed_active_action_word,
        config.matched_hold_action_word,
    }
    cells: list[ActionPreparationEvidenceCell] = []
    for _, donor in selected:
        relevant = tuple(
            value for value in donor.panel_cells if value.requested_action_word in requested
        )
        expected = {(action, member) for action in requested for member in config.model_member_ids}
        if {
            (value.requested_action_word, value.model_member_id) for value in relevant
        } != expected or len(relevant) != len(expected):
            raise ValueError("MAPPED_ACTION_NOT_CLAIMED_UNSUPPORTED: incomplete donor panel")
        cells.extend(
            ActionPreparationEvidenceCell(
                cell_id=f"mapped-recurrence.{query.arm_id}.{value.cell_id}",
                task_id=query.task_id,
                arm_id=query.arm_id,
                preparation_unit_id=donor.preparation_unit_id,
                requested_action_word=value.requested_action_word,
                model_member_id=value.model_member_id,
                source_fingerprint=donor.source_fingerprint,
                accepted_action=value.accepted_action,
                applied_action=value.applied_action,
                realized_action=value.realized_action,
                delivery_receipt=value.delivery_receipt,
                response_evidence=value.response_evidence,
                method_config=query.method_config,
                recurrence_predicate=value.recurrence_predicate,
                uncertainty_rule=value.uncertainty_rule,
                acquisition_role=AcquisitionEvidenceRole.MAPPED_DEVELOPMENT_DONOR,
                matched_clone=value.matched_clone,
                delivery_valid=value.delivery_valid,
                recurrence_passed=value.recurrence_passed,
                causal_domain=RecurrenceEvidenceDomain.EXCLUDED_MAPPED_DEVELOPMENT,
                outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
                reason_codes=value.reason_codes,
            )
            for value in relevant
        )
    recurrence = evaluate_action_preparation_recurrence(
        evidence_source=evidence_source,
        task_id=query.task_id,
        arm_id=query.arm_id,
        active_action_word=query.claimed_active_action_word,
        matched_hold_action_word=config.matched_hold_action_word,
        model_member_ids=config.model_member_ids,
        evidence_cells=tuple(sorted(cells, key=lambda value: value.cell_id)),
        method_config=query.method_config,
        proof_owner=proof_owner,
    )
    if recurrence.status is not ActionPreparationRecurrenceStatus.SUPPORTED:
        raise ValueError(
            "MAPPED_ACTION_NOT_CLAIMED_UNSUPPORTED: " + ",".join(recurrence.reason_codes)
        )
    return MappedActionSupportProjectionReceipt(
        receipt_id=f"mapped-action-support.{query.query_id}",
        atlas=atlas,
        query=query,
        selected_donor_distances=distance_receipts,
        recurrence_receipt=recurrence,
    )


__all__ = [
    'MappedActionSupportProjectionReceipt',
    'MappedActionSupportProjection',
    'MappedDevelopmentDonorState',
    'MappedDevelopmentPanelCell',
    'MappedDevelopmentSupportAtlas',
    'MappedDonorDistanceReceipt',
    'MappedStateCoordinate',
    'MappedStateSupportQuery',
    "MappedSupportEvidenceRole",
    'MappedSupportMetricAxis',
    "build_mapped_development_support_atlas",
    "project_mapped_action_support",
]
