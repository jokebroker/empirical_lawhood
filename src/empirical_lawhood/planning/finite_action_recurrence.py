"Frozen type-distinct assignment and sealed products for finite-action recurrence.\n\nThe schedule contains only already-assigned ACTIVE, matched native-HOLD and\nHOLD-control routes.  It has no online choice, controller tick, policy,\nprogramme, compiler, reference decision or admission/current-controller-use identity.\n"

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord, ObservedActionOccurrence
from empirical_lawhood.kernel.admission import GateStatus
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity, NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_schema,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.time import ClockCoordinate
from empirical_lawhood.planning.native_hold_decision_cell_calibration import NativeHoldCalibrationDisposition, NativeHoldDecisionCellCalibrationReceipt
from empirical_lawhood.planning.preissue_branch_selection import PreissueBranchSelectionReceipt
from empirical_lawhood.planning.finite_action_alternate_branch_selection import FiniteActionAlternateBranchSelectionBridge

from .finite_action_occurrence import FINITE_ACTION_ACTIVE_RESPONSE_COORDINATES, FINITE_ACTION_TOKAMAK_ACTIVE_WORD_ID, FINITE_ACTION_TOKAMAK_HOLD_WORD_ID, FINITE_ACTION_TOKAMAK_LOCAL_SUPPORT_IDS, FINITE_ACTION_OCCURRENCE_COUNT, FINITE_ACTION_REQUIRED_PREDICATE_KINDS, FiniteActionOccurrenceDisposition, FiniteActionOccurrencePredicateKind, FiniteActionOccurrenceQualificationReceipt, FiniteActionOccurrenceQualificationTemplateSet, FiniteActionBranchSelection, _is_historical_fidelity_recurrence_selection, finite_action_branch_operator_identity, finite_action_branch_selection_identity, finite_action_branch_selection_receipt_id


FINITE_ACTION_RECURRENCE_EFFICACY_UNIT_COUNT = 18
FINITE_ACTION_RECURRENCE_CONTROL_UNIT_COUNT = 4
FINITE_ACTION_RECURRENCE_MEMBER_COUNT = 2
FINITE_ACTION_RECURRENCE_STRATUM_COUNT = 2
FINITE_ACTION_RECURRENCE_EFFICACY_PER_STRATUM = 9
FINITE_ACTION_RECURRENCE_RESPONSE_SAMPLE_COUNT = 6
FINITE_ACTION_RECURRENCE_PRIMARY_EPISODE_COUNT = 80
FINITE_ACTION_RECURRENCE_EFFICACY_EPISODE_COUNT = 72
FINITE_ACTION_RECURRENCE_CONTROL_EPISODE_COUNT = 8
FINITE_ACTION_RECURRENCE_DEGREES_OF_FREEDOM = 17
FINITE_ACTION_RECURRENCE_ONE_SIDED_CRITICAL_VALUE = Decimal("1.7396067260750725")
FINITE_ACTION_RECURRENCE_MATERIALITY = Decimal("0.0017")
FINITE_ACTION_RECURRENCE_CONTROL_ANCHOR_SLOT_IDS = (
    "c-hold-anchor-01",
    "c-hold-anchor-02",
    "c-hold-anchor-03",
    "c-hold-anchor-04",
)
FINITE_ACTION_RECURRENCE_HOLD_RESERVE_SLOT_ID = "c-hold-anchor-05"
FINITE_ACTION_RECURRENCE_CALIBRATION_OCCURRENCE_ROLE_ID = "c-native-hold-calibration"
FINITE_ACTION_RECURRENCE_STRATUM_IDS = (
    'stratum.tokamak-control.inner',
    'stratum.tokamak-control.outer',
)
FINITE_ACTION_RECURRENCE_RESERVE_REASON_CODES = (
    "ATOMIC_PUBLICATION_FAILURE_NO_AUTHENTICATED_COMPLETE_PAYLOAD",
    "EXECUTOR_INTERRUPTION_NO_AUTHENTICATED_COMPLETE_PAYLOAD",
    "WORKER_LOSS_NO_AUTHENTICATED_COMPLETE_PAYLOAD",
)
FINITE_ACTION_RECURRENCE_NONRESERVABLE_REASON_CODES = (
    "ACTION_CLIPPED_OR_REJECTED",
    "EARLY_TERMINATION",
    "MISSING_SCIENTIFIC_FIELD",
    "NONFINITE_STATE",
    "SCIENTIFIC_INVALIDITY_OR_UNFAVORABLE_OUTCOME",
    "SIMULATOR_NONCONVERGENCE",
    "SINK_OR_VALIDITY_FAILURE",
)
FINITE_ACTION_RECURRENCE_REDUCTION_ORDER = (
    "ACTIVE_MINUS_MATCHED_NATIVE_HOLD_WITHIN_MEMBER",
    "MEMBERWISE_MINIMUM_WITHIN_PHYSICAL_UNIT",
    "PHYSICAL_INDEPENDENT_UNIT_STUDENT_INFERENCE",
)
FINITE_ACTION_RECURRENCE_TERMINAL_PRECEDENCE = (
    "AUTHORITY_OR_RESOURCE_STOP",
    "RECEIPT_CUSTODY_OR_REVEAL_INVALID",
    "FINITE_ACTION_RECURRENCE_PREACTION_NONATTEMPT",
    "FINITE_ACTION_RECURRENCE_DELIVERY_INVALID",
    "FINITE_ACTION_RECURRENCE_TECHNICAL_PARTIAL",
    "FINITE_ACTION_RECURRENCE_EFFICACY_UNEVALUABLE",
    "FINITE_ACTION_RECURRENCE_MEMBER_OR_STRATUM_MIXED",
    "FINITE_ACTION_RECURRENCE_VALID_NEGATIVE_OR_SUBMATERIAL_RECURRENCE",
    "FINITE_ACTION_RECURRENCE_HOLD_CONTROL_FAILED",
    "TERMINAL_FINITE_ACTION_RECURRENCE_POSITIVE",
)


class FiniteActionRecurrenceUnitRole(StrEnum):
    EFFICACY_ACTIVE = "EFFICACY_ACTIVE"
    HOLD_CONTROL = "HOLD_CONTROL"


class FiniteActionRecurrenceRouteRole(StrEnum):
    ACTIVE = "ACTIVE"
    HOLD_CONTROL = "HOLD_CONTROL"
    MATCHED_HOLD = "MATCHED_HOLD"


class FiniteActionRecurrenceReserveRole(StrEnum):
    EFFICACY_WHOLE_UNIT = "EFFICACY_WHOLE_UNIT"
    HOLD_CONTROL_WHOLE_UNIT = "HOLD_CONTROL_WHOLE_UNIT"


class FiniteActionPreactionDisposition(StrEnum):
    NONATTEMPT = "NONATTEMPT"
    READY = "READY"


class FiniteActionRecurrenceRevealIntegrity(StrEnum):
    AUTHORITY_OR_RESOURCE_STOP = "AUTHORITY_OR_RESOURCE_STOP"
    RECEIPT_CUSTODY_OR_REVEAL_INVALID = "RECEIPT_CUSTODY_OR_REVEAL_INVALID"
    VALID = "VALID"


class FiniteActionRouteCanaryCoordinateRole(StrEnum):
    ACTIVE_COORDINATE = "ACTIVE_COORDINATE"
    HOLD_ANCHOR_01 = "HOLD_ANCHOR_01"


class FiniteActionRouteCanaryEpisodeRole(StrEnum):
    ACTIVE = "ACTIVE"
    HOLD_CONTROL = "HOLD_CONTROL"
    HOLD_QUALIFICATION_REPEAT = "HOLD_QUALIFICATION_REPEAT"
    MATCHED_HOLD = "MATCHED_HOLD"


class FiniteActionRouteCanaryDisposition(StrEnum):
    NOT_QUALIFIED = "NOT_QUALIFIED"
    QUALIFIED = "QUALIFIED"
    UNEVALUABLE = "UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class FiniteActionRouteCanaryEpisodeReceipt(CanonicalRecord):
    """One of the eight fresh, nonclaim-bearing assigned-route checks."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-action-route-canary-episode-receipt'

    canary_episode_id: str
    execution_id: str
    physical_unit_id: str
    model_member_id: str
    coordinate_role: FiniteActionRouteCanaryCoordinateRole
    episode_role: FiniteActionRouteCanaryEpisodeRole
    action_word: ObjectIdentity
    preparation_fingerprint: str
    preaction_prefix_sha256: str
    preaction_disposition: FiniteActionPreactionDisposition
    delivery_status: GateStatus
    publication_recovery_status: GateStatus
    seal_reveal_decoding_status: GateStatus
    delivery_trace: ObjectIdentity
    evidence_identities: tuple[ObjectIdentity, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("canary_episode_id", self.canary_episode_id),
            ("execution_id", self.execution_id),
            ("physical_unit_id", self.physical_unit_id),
            ("model_member_id", self.model_member_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_sha256(
            self.preparation_fingerprint,
            field_name="preparation_fingerprint",
        )
        validate_sha256(
            self.preaction_prefix_sha256,
            field_name="preaction_prefix_sha256",
        )
        require_sorted_unique_ids(
            self.evidence_identities,
            attribute="object_id",
            field_name="evidence_identities",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        statuses = (
            self.delivery_status,
            self.publication_recovery_status,
            self.seal_reveal_decoding_status,
        )
        if all(value is GateStatus.PASS for value in statuses):
            if (
                self.preaction_disposition is not FiniteActionPreactionDisposition.READY
                or not self.evidence_identities
                or self.reason_codes
            ):
                raise ValueError("passing finite-action recurrence route canary episode is incomplete")
        elif not self.reason_codes:
            raise ValueError("nonpassing finite-action recurrence route canary episode requires reasons")

    @property
    def product_key(
        self,
    ) -> tuple[
        FiniteActionRouteCanaryCoordinateRole,
        str,
        FiniteActionRouteCanaryEpisodeRole,
    ]:
        return self.coordinate_role, self.model_member_id, self.episode_role


@dataclass(frozen=True, slots=True)
class FiniteActionRecurrenceRouteQualificationReceipt(CanonicalRecord):
    "Eight-episode assigned-route canary; never part of the finite-action recurrence statistic."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-action-recurrence-route-qualification-receipt'

    receipt_id: str
    branch_selection: FiniteActionBranchSelection
    occurrence_receipts: tuple[ObjectIdentity, ...]
    native_hold_calibration: ObjectIdentity
    active_local_support_id: str
    hold_anchor_slot_id: str
    model_member_ids: tuple[str, ...]
    episodes: tuple[FiniteActionRouteCanaryEpisodeReceipt, ...]
    nonattempt_refusal_conformance: GateStatus
    evaluator_decoding_conformance: GateStatus
    disposition: FiniteActionRouteCanaryDisposition
    reason_codes: tuple[str, ...]
    g2_operator_feasibility: ObjectIdentity
    g2_controlling_reason_code: str
    claim_bearing: bool
    evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        validate_stable_id(
            self.active_local_support_id,
            field_name="active_local_support_id",
        )
        validate_stable_id(self.hold_anchor_slot_id, field_name="hold_anchor_slot_id")
        require_sorted_unique_ids(
            self.occurrence_receipts,
            attribute="object_id",
            field_name="occurrence_receipts",
        )
        require_sorted_unique_strings(
            self.model_member_ids,
            field_name="model_member_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(
            self.episodes,
            attribute="canary_episode_id",
            field_name="episodes",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if (
            not _is_historical_fidelity_recurrence_selection(self.branch_selection)
            or self.active_local_support_id not in FINITE_ACTION_TOKAMAK_LOCAL_SUPPORT_IDS
            or self.hold_anchor_slot_id != FINITE_ACTION_RECURRENCE_CONTROL_ANCHOR_SLOT_IDS[0]
            or len(self.model_member_ids) != FINITE_ACTION_RECURRENCE_MEMBER_COUNT
        ):
            raise ValueError("Finite-action recurrence route canary changes its frozen selected route")
        expected_products = {
            *(
                (
                    FiniteActionRouteCanaryCoordinateRole.ACTIVE_COORDINATE,
                    member_id,
                    role,
                )
                for member_id in self.model_member_ids
                for role in (
                    FiniteActionRouteCanaryEpisodeRole.ACTIVE,
                    FiniteActionRouteCanaryEpisodeRole.MATCHED_HOLD,
                )
            ),
            *(
                (
                    FiniteActionRouteCanaryCoordinateRole.HOLD_ANCHOR_01,
                    member_id,
                    role,
                )
                for member_id in self.model_member_ids
                for role in (
                    FiniteActionRouteCanaryEpisodeRole.HOLD_CONTROL,
                    FiniteActionRouteCanaryEpisodeRole.HOLD_QUALIFICATION_REPEAT,
                )
            ),
        }
        if (
            len(self.episodes) != 8
            or {value.product_key for value in self.episodes} != expected_products
        ):
            raise ValueError("Finite-action recurrence route qualification requires the exact eight canary episodes")
        if (
            len({value.execution_id for value in self.episodes}) != 8
            or len({value.delivery_trace.object_id for value in self.episodes}) != 8
            or len({value.canary_episode_id for value in self.episodes}) != 8
            or len({value.physical_unit_id for value in self.episodes}) != 3
        ):
            raise ValueError("Finite-action recurrence route canary changes three-unit/eight-delivery nesting")
        by_product = {value.product_key: value for value in self.episodes}
        active_episodes = tuple(
            value
            for value in self.episodes
            if value.coordinate_role is FiniteActionRouteCanaryCoordinateRole.ACTIVE_COORDINATE
        )
        anchor_episodes = tuple(
            value
            for value in self.episodes
            if value.coordinate_role is FiniteActionRouteCanaryCoordinateRole.HOLD_ANCHOR_01
        )
        if (
            len({value.physical_unit_id for value in active_episodes}) != 1
            or len({value.physical_unit_id for value in anchor_episodes}) != 2
            or len({value.preparation_fingerprint for value in active_episodes}) != 1
            or len({value.preparation_fingerprint for value in anchor_episodes}) != 1
        ):
            raise ValueError("Finite-action recurrence route canary changes its exact active/anchor unit topology")
        anchor_units_by_role = {
            role: {
                value.physical_unit_id for value in anchor_episodes if value.episode_role is role
            }
            for role in (
                FiniteActionRouteCanaryEpisodeRole.HOLD_CONTROL,
                FiniteActionRouteCanaryEpisodeRole.HOLD_QUALIFICATION_REPEAT,
            )
        }
        if any(
            len(values) != 1 for values in anchor_units_by_role.values()
        ) or not anchor_units_by_role[FiniteActionRouteCanaryEpisodeRole.HOLD_CONTROL].isdisjoint(
            anchor_units_by_role[FiniteActionRouteCanaryEpisodeRole.HOLD_QUALIFICATION_REPEAT]
        ):
            raise ValueError("Finite-action recurrence route anchor canary/repeat must use two fresh units")
        for member_id in self.model_member_ids:
            active = by_product[
                (
                    FiniteActionRouteCanaryCoordinateRole.ACTIVE_COORDINATE,
                    member_id,
                    FiniteActionRouteCanaryEpisodeRole.ACTIVE,
                )
            ]
            matched_hold = by_product[
                (
                    FiniteActionRouteCanaryCoordinateRole.ACTIVE_COORDINATE,
                    member_id,
                    FiniteActionRouteCanaryEpisodeRole.MATCHED_HOLD,
                )
            ]
            hold_control = by_product[
                (
                    FiniteActionRouteCanaryCoordinateRole.HOLD_ANCHOR_01,
                    member_id,
                    FiniteActionRouteCanaryEpisodeRole.HOLD_CONTROL,
                )
            ]
            hold_repeat = by_product[
                (
                    FiniteActionRouteCanaryCoordinateRole.HOLD_ANCHOR_01,
                    member_id,
                    FiniteActionRouteCanaryEpisodeRole.HOLD_QUALIFICATION_REPEAT,
                )
            ]
            if (
                active.physical_unit_id != matched_hold.physical_unit_id
                or hold_control.physical_unit_id == hold_repeat.physical_unit_id
                or active.preparation_fingerprint != matched_hold.preparation_fingerprint
                or hold_control.preparation_fingerprint != hold_repeat.preparation_fingerprint
                or active.preaction_prefix_sha256 != matched_hold.preaction_prefix_sha256
                or hold_control.preaction_prefix_sha256 != hold_repeat.preaction_prefix_sha256
            ):
                raise ValueError("Finite-action recurrence route canary pair changes unit/preparation/prefix rules")
        statuses = (
            self.nonattempt_refusal_conformance,
            self.evaluator_decoding_conformance,
            *(
                status
                for episode in self.episodes
                for status in (
                    episode.delivery_status,
                    episode.publication_recovery_status,
                    episode.seal_reveal_decoding_status,
                )
            ),
        )
        expected_disposition = (
            FiniteActionRouteCanaryDisposition.UNEVALUABLE
            if any(value is GateStatus.UNEVALUABLE for value in statuses)
            else FiniteActionRouteCanaryDisposition.NOT_QUALIFIED
            if any(value is GateStatus.FAIL for value in statuses)
            else FiniteActionRouteCanaryDisposition.QUALIFIED
        )
        expected_reasons = tuple(
            sorted({reason for value in self.episodes for reason in value.reason_codes})
        )
        if self.nonattempt_refusal_conformance is not GateStatus.PASS:
            expected_reasons = tuple(
                sorted({*expected_reasons, "FINITE_ACTION_RECURRENCE_PREACTION_NONATTEMPT_CONFORMANCE_FAILED"})
            )
        if self.evaluator_decoding_conformance is not GateStatus.PASS:
            expected_reasons = tuple(
                sorted({*expected_reasons, "FINITE_ACTION_RECURRENCE_EVALUATOR_DECODING_CONFORMANCE_FAILED"})
            )
        if self.disposition is not expected_disposition or self.reason_codes != expected_reasons:
            raise ValueError("Finite-action recurrence route qualification is not the exact canary intersection")
        branch = self.branch_selection
        if (
            self.g2_operator_feasibility != finite_action_branch_operator_identity(branch)
            or self.g2_controlling_reason_code != branch.controlling_reason_code
        ):
            raise ValueError("Finite-action recurrence route canary drops its controlling input-output operator feasibility obstruction")
        if (
            self.claim_bearing
            or self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
            or self.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL
            or self.visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE
        ):
            raise ValueError("Finite-action recurrence route canary cannot enter the recurrence claim")


@dataclass(frozen=True, slots=True)
class FiniteActionRecurrenceUnitSpec(CanonicalRecord):
    """One physical independent unit; members and routes remain nested."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-action-recurrence-unit-spec'

    unit_id: str
    physical_independent_unit_id: str
    preparation_coordinate_id: str
    preparation_sha256: str
    role: FiniteActionRecurrenceUnitRole
    local_support_id: str | None
    stratum_id: str | None
    hold_anchor_slot_id: str | None

    def __post_init__(self) -> None:
        for name, value in (
            ("unit_id", self.unit_id),
            ("physical_independent_unit_id", self.physical_independent_unit_id),
            ("preparation_coordinate_id", self.preparation_coordinate_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_sha256(self.preparation_sha256, field_name="preparation_sha256")
        if self.role is FiniteActionRecurrenceUnitRole.EFFICACY_ACTIVE:
            if (
                self.local_support_id not in FINITE_ACTION_TOKAMAK_LOCAL_SUPPORT_IDS
                or self.stratum_id not in FINITE_ACTION_RECURRENCE_STRATUM_IDS
                or self.hold_anchor_slot_id is not None
            ):
                raise ValueError("Finite-action recurrence efficacy unit requires one J3 region and one stratum")
        elif (
            self.local_support_id is not None
            or self.stratum_id is not None
            or self.hold_anchor_slot_id not in FINITE_ACTION_RECURRENCE_CONTROL_ANCHOR_SLOT_IDS
        ):
            raise ValueError("Finite-action recurrence HOLD control requires exactly one scheduled anchor slot")


@dataclass(frozen=True, slots=True)
class FiniteActionRecurrenceRouteSpec(CanonicalRecord):
    """One fresh, preassigned member-local episode route."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-action-recurrence-route-spec'

    episode_id: str
    execution_id: str
    environment_seed_id: str
    unit_id: str
    model_member_id: str
    role: FiniteActionRecurrenceRouteRole
    action_word_id: str

    def __post_init__(self) -> None:
        for name, value in (
            ("episode_id", self.episode_id),
            ("execution_id", self.execution_id),
            ("environment_seed_id", self.environment_seed_id),
            ("unit_id", self.unit_id),
            ("model_member_id", self.model_member_id),
            ("action_word_id", self.action_word_id),
        ):
            validate_stable_id(value, field_name=name)

    @property
    def product_key(self) -> tuple[str, str, FiniteActionRecurrenceRouteRole]:
        return self.unit_id, self.model_member_id, self.role


@dataclass(frozen=True, slots=True)
class FiniteActionRecurrenceStratumSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-action-recurrence-stratum-spec'

    stratum_id: str
    unit_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.stratum_id, field_name="stratum_id")
        require_sorted_unique_strings(self.unit_ids, field_name="unit_ids", allow_empty=False)
        if len(self.unit_ids) != FINITE_ACTION_RECURRENCE_EFFICACY_PER_STRATUM:
            raise ValueError("Finite-action recurrence stratum requires exactly nine efficacy units")


@dataclass(frozen=True, slots=True)
class FiniteActionRecurrenceReserveSpec(CanonicalRecord):
    """One frozen whole-unit reserve; never a scientific-value replacement."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-action-recurrence-reserve-spec'

    reserve_id: str
    role: FiniteActionRecurrenceReserveRole
    stratum_id: str | None
    hold_anchor_slot_id: str | None
    preparation_coordinate_id: str
    preparation_sha256: str
    eligible_reason_codes: tuple[str, ...]
    nonreservable_reason_codes: tuple[str, ...]
    whole_unit_only: bool
    selection_before_any_cohort_reveal: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.reserve_id, field_name="reserve_id")
        validate_stable_id(
            self.preparation_coordinate_id,
            field_name="preparation_coordinate_id",
        )
        validate_sha256(self.preparation_sha256, field_name="preparation_sha256")
        if self.role is FiniteActionRecurrenceReserveRole.EFFICACY_WHOLE_UNIT:
            if (
                self.stratum_id not in FINITE_ACTION_RECURRENCE_STRATUM_IDS
                or self.hold_anchor_slot_id is not None
            ):
                raise ValueError("Finite-action recurrence efficacy reserve requires one frozen stratum")
        elif (
            self.stratum_id is not None
            or self.hold_anchor_slot_id != FINITE_ACTION_RECURRENCE_HOLD_RESERVE_SLOT_ID
        ):
            raise ValueError("Finite-action recurrence HOLD reserve must be exact anchor slot 05")
        if (
            self.eligible_reason_codes != FINITE_ACTION_RECURRENCE_RESERVE_REASON_CODES
            or self.nonreservable_reason_codes
            != FINITE_ACTION_RECURRENCE_NONRESERVABLE_REASON_CODES
            or not self.whole_unit_only
            or not self.selection_before_any_cohort_reveal
        ):
            raise ValueError("Finite-action recurrence reserve semantics differ from the frozen policy")


def _same_clock_domain(left: ClockCoordinate, right: ClockCoordinate) -> bool:
    return (
        left.clock_id == right.clock_id
        and left.time_unit == right.time_unit
        and left.coordinate_frame == right.coordinate_frame
        and left.origin is right.origin
    )


@dataclass(frozen=True, slots=True)
class FiniteActionRecurrenceAssignmentTemplate(CanonicalRecord):
    "Complete pre-qualification 18+4/80 finite-action recurrence assignment with no future receipt or outcome."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-action-recurrence-assignment-template'

    template_id: str
    expected_assignment_id: str
    frozen_roster: ObjectIdentity
    roster_collision_audit: ObjectIdentity
    expected_branch_selection_receipt_id: str
    expected_branch_selection_receipt_schema: str
    expected_occurrence_receipt_ids: tuple[str, ...]
    expected_occurrence_receipt_schema: str
    expected_native_hold_calibration_receipt_id: str
    expected_native_hold_calibration_receipt_schema: str
    expected_route_qualification_receipt_id: str
    expected_route_qualification_receipt_schema: str
    units: tuple[FiniteActionRecurrenceUnitSpec, ...]
    routes: tuple[FiniteActionRecurrenceRouteSpec, ...]
    model_member_ids: tuple[str, ...]
    strata: tuple[FiniteActionRecurrenceStratumSpec, ...]
    active_word: OccurrenceActionWord
    matched_hold_word: OccurrenceActionWord
    source: ObjectIdentity
    schedule: ObjectIdentity
    retained_history: ObjectIdentity
    receiver: ObjectIdentity
    horizon: ObjectIdentity
    causal_cutoff: ClockCoordinate
    response_coordinates: tuple[ClockCoordinate, ...]
    effect_quantity_id: str
    effect_native_unit: str
    control_anchor_slot_ids: tuple[str, ...]
    hold_reserve_anchor_slot_id: str
    reserves: tuple[FiniteActionRecurrenceReserveSpec, ...]
    reserve_order: tuple[str, ...]
    minimum_evaluable_units: int
    minimum_active_coverage: Decimal
    alpha: Decimal
    degrees_of_freedom: int
    one_sided_critical_value: Decimal
    materiality: NamedDecimal
    strict_materiality: bool
    reduction_order: tuple[str, ...]
    terminal_precedence: tuple[str, ...]
    preaction_prefix_must_be_byte_equal: bool
    route_preassigned_without_online_choice: bool
    claim_ceiling: str
    evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        for field_name, value in (
            ("template_id", self.template_id),
            ("expected_assignment_id", self.expected_assignment_id),
            (
                "expected_branch_selection_receipt_id",
                self.expected_branch_selection_receipt_id,
            ),
            (
                "expected_native_hold_calibration_receipt_id",
                self.expected_native_hold_calibration_receipt_id,
            ),
            (
                "expected_route_qualification_receipt_id",
                self.expected_route_qualification_receipt_id,
            ),
        ):
            validate_stable_id(value, field_name=field_name)
        require_sorted_unique_strings(
            self.expected_occurrence_receipt_ids,
            field_name="expected_occurrence_receipt_ids",
            allow_empty=False,
        )
        if len(self.expected_occurrence_receipt_ids) != 6:
            raise ValueError("Finite-action recurrence template requires six expected occurrence receipt IDs")
        for value in (
            self.expected_branch_selection_receipt_schema,
            self.expected_occurrence_receipt_schema,
            self.expected_native_hold_calibration_receipt_schema,
            self.expected_route_qualification_receipt_schema,
        ):
            validate_schema(value)
        if (
            self.expected_branch_selection_receipt_schema
            not in {
                PreissueBranchSelectionReceipt.SCHEMA,
                FiniteActionAlternateBranchSelectionBridge.SCHEMA,
            }
            or self.expected_occurrence_receipt_schema
            != FiniteActionOccurrenceQualificationReceipt.SCHEMA
            or self.expected_native_hold_calibration_receipt_schema
            != NativeHoldDecisionCellCalibrationReceipt.SCHEMA
            or self.expected_route_qualification_receipt_schema
            != FiniteActionRecurrenceRouteQualificationReceipt.SCHEMA
        ):
            raise ValueError("Finite-action recurrence template expects another post-parent record schema")
        _validate_recurrence_template(self)


def validate_finite_action_occurrence_recurrence_expectations(
    *,
    occurrence_templates: FiniteActionOccurrenceQualificationTemplateSet,
    recurrence_template: FiniteActionRecurrenceAssignmentTemplate,
) -> None:
    """Validate the exact pre-Q occurrence-template -> recurrence receipt join."""

    occurrence_member_ids = tuple(
        sorted({value.model_member_id for value in occurrence_templates.templates})
    )
    if recurrence_template.model_member_ids != occurrence_member_ids:
        raise ValueError("Finite-action recurrence member roster differs from the occurrence templates")
    if (
        recurrence_template.expected_occurrence_receipt_schema
        != FiniteActionOccurrenceQualificationReceipt.SCHEMA
        or recurrence_template.expected_occurrence_receipt_ids
        != occurrence_templates.expected_qualification_receipt_ids
    ):
        raise ValueError("Finite-action recurrence receipt expectations differ from the occurrence templates")


def _validate_recurrence_template(
    template: FiniteActionRecurrenceAssignmentTemplate,
) -> None:
    validate_stable_id(template.effect_quantity_id, field_name="effect_quantity_id")
    validate_nonempty(template.effect_native_unit, field_name="effect_native_unit")
    validate_nonempty(template.claim_ceiling, field_name="claim_ceiling")
    if (
        template.frozen_roster.object_schema != 'empirical-lawhood/composition/tokamak-control-replication/gym-torax-roster'
        or template.roster_collision_audit.object_schema
        != 'empirical-lawhood/composition/tokamak-control-replication/gym-torax-roster-collision-audit'
    ):
        raise ValueError("Finite-action recurrence template requires the exact frozen current roster inputs")
    require_sorted_unique_ids(template.units, attribute="unit_id", field_name="units")
    require_sorted_unique_ids(template.routes, attribute="episode_id", field_name="routes")
    require_sorted_unique_strings(
        template.model_member_ids,
        field_name="model_member_ids",
        allow_empty=False,
    )
    require_sorted_unique_ids(template.strata, attribute="stratum_id", field_name="strata")
    require_sorted_unique_ids(template.reserves, attribute="reserve_id", field_name="reserves")
    if len(template.model_member_ids) != FINITE_ACTION_RECURRENCE_MEMBER_COUNT:
        raise ValueError("Finite-action recurrence requires exactly two numerical members")
    efficacy = tuple(
        value
        for value in template.units
        if value.role is FiniteActionRecurrenceUnitRole.EFFICACY_ACTIVE
    )
    controls = tuple(
        value
        for value in template.units
        if value.role is FiniteActionRecurrenceUnitRole.HOLD_CONTROL
    )
    if (
        len(efficacy) != FINITE_ACTION_RECURRENCE_EFFICACY_UNIT_COUNT
        or len(controls) != FINITE_ACTION_RECURRENCE_CONTROL_UNIT_COUNT
        or len({value.physical_independent_unit_id for value in template.units})
        != len(template.units)
        or len({value.preparation_coordinate_id for value in template.units}) != len(template.units)
    ):
        raise ValueError("Finite-action recurrence requires 18+4 unique physical units")
    if (
        tuple(sorted(value.hold_anchor_slot_id for value in controls if value.hold_anchor_slot_id))
        != FINITE_ACTION_RECURRENCE_CONTROL_ANCHOR_SLOT_IDS
        or template.control_anchor_slot_ids != FINITE_ACTION_RECURRENCE_CONTROL_ANCHOR_SLOT_IDS
        or template.hold_reserve_anchor_slot_id != FINITE_ACTION_RECURRENCE_HOLD_RESERVE_SLOT_ID
    ):
        raise ValueError("Finite-action recurrence controls/reserve change exact anchor slots 01--05")
    if (
        len(template.strata) != FINITE_ACTION_RECURRENCE_STRATUM_COUNT
        or tuple(value.stratum_id for value in template.strata)
        != FINITE_ACTION_RECURRENCE_STRATUM_IDS
    ):
        raise ValueError("Finite-action recurrence requires exact inner/outer strata")
    efficacy_map = {value.unit_id: value for value in efficacy}
    covered: set[str] = set()
    for stratum in template.strata:
        if any(
            efficacy_map.get(unit_id) is None
            or efficacy_map[unit_id].stratum_id != stratum.stratum_id
            for unit_id in stratum.unit_ids
        ):
            raise ValueError("Finite-action recurrence stratum changes an issued unit")
        covered.update(stratum.unit_ids)
    if covered != set(efficacy_map):
        raise ValueError("Finite-action recurrence strata do not partition all 18 efficacy units")
    expected_products: set[tuple[str, str, FiniteActionRecurrenceRouteRole]] = set()
    for unit in template.units:
        roles = (
            (
                FiniteActionRecurrenceRouteRole.ACTIVE,
                FiniteActionRecurrenceRouteRole.MATCHED_HOLD,
            )
            if unit.role is FiniteActionRecurrenceUnitRole.EFFICACY_ACTIVE
            else (FiniteActionRecurrenceRouteRole.HOLD_CONTROL,)
        )
        expected_products.update(
            (unit.unit_id, member_id, role)
            for member_id in template.model_member_ids
            for role in roles
        )
    if (
        len(template.routes) != FINITE_ACTION_RECURRENCE_PRIMARY_EPISODE_COUNT
        or {value.product_key for value in template.routes} != expected_products
        or len({value.execution_id for value in template.routes}) != len(template.routes)
        or len({value.environment_seed_id for value in template.routes}) != len(template.units)
    ):
        raise ValueError("Finite-action recurrence route product must be exactly 72 efficacy plus eight controls")
    unit_map = {value.unit_id: value for value in template.units}
    seeds_by_unit = {
        unit_id: {
            route.environment_seed_id for route in template.routes if route.unit_id == unit_id
        }
        for unit_id in unit_map
    }
    if any(len(seeds) != 1 for seeds in seeds_by_unit.values()):
        raise ValueError("Finite-action recurrence nested routes must share one environment seed per physical unit")
    for route in template.routes:
        unit = unit_map[route.unit_id]
        expected_word_id = (
            template.active_word.word_id
            if route.role is FiniteActionRecurrenceRouteRole.ACTIVE
            else template.matched_hold_word.word_id
        )
        if route.model_member_id not in template.model_member_ids:
            raise ValueError("Finite-action recurrence route uses an undeclared member")
        if route.action_word_id != expected_word_id:
            raise ValueError("Finite-action recurrence route substitutes another action word")
        if (
            unit.role is FiniteActionRecurrenceUnitRole.EFFICACY_ACTIVE
            and route.role is FiniteActionRecurrenceRouteRole.HOLD_CONTROL
        ) or (
            unit.role is FiniteActionRecurrenceUnitRole.HOLD_CONTROL
            and route.role is not FiniteActionRecurrenceRouteRole.HOLD_CONTROL
        ):
            raise ValueError("Finite-action recurrence route changes its issued unit role")
    if (
        template.active_word.word_id != FINITE_ACTION_TOKAMAK_ACTIVE_WORD_ID
        or template.matched_hold_word.word_id != FINITE_ACTION_TOKAMAK_HOLD_WORD_ID
        or len(template.active_word.occurrences) != FINITE_ACTION_OCCURRENCE_COUNT
        or len(template.matched_hold_word.occurrences) != FINITE_ACTION_OCCURRENCE_COUNT
        or template.active_word.denominator_id != template.matched_hold_word.denominator_id
        or template.retained_history.object_id != template.active_word.retained_history_id
        or template.receiver.object_id != template.active_word.receiver_id
        or template.horizon.object_id != template.active_word.horizon_id
        or template.active_word.retained_history_id
        != template.matched_hold_word.retained_history_id
        or template.active_word.receiver_id != template.matched_hold_word.receiver_id
        or template.active_word.horizon_id != template.matched_hold_word.horizon_id
    ):
        raise ValueError("Finite-action recurrence template changes the exact active/HOLD occurrence pair")
    efficacy_reserves = tuple(
        value
        for value in template.reserves
        if value.role is FiniteActionRecurrenceReserveRole.EFFICACY_WHOLE_UNIT
    )
    hold_reserves = tuple(
        value
        for value in template.reserves
        if value.role is FiniteActionRecurrenceReserveRole.HOLD_CONTROL_WHOLE_UNIT
    )
    if (
        len(efficacy_reserves) != 2
        or {value.stratum_id for value in efficacy_reserves}
        != set(FINITE_ACTION_RECURRENCE_STRATUM_IDS)
        or len(hold_reserves) != 1
        or set(template.reserve_order) != {value.reserve_id for value in template.reserves}
        or len(template.reserve_order) != 3
    ):
        raise ValueError("Finite-action recurrence requires two efficacy reserves and one HOLD reserve")
    if not template.reserve_order or len(set(template.reserve_order)) != len(
        template.reserve_order
    ):
        raise ValueError("Finite-action recurrence reserve order must contain unique issued reserve IDs")
    for reserve_id in template.reserve_order:
        validate_stable_id(reserve_id, field_name="reserve_order")
    occupied_coordinates = {value.preparation_coordinate_id for value in template.units}
    occupied_hashes = {value.preparation_sha256 for value in template.units}
    if any(
        value.preparation_coordinate_id in occupied_coordinates
        or value.preparation_sha256 in occupied_hashes
        for value in efficacy_reserves
    ):
        raise ValueError("Finite-action recurrence efficacy reserve aliases a scheduled scientific unit")
    if len(template.response_coordinates) != FINITE_ACTION_RECURRENCE_RESPONSE_SAMPLE_COUNT:
        raise ValueError("Finite-action recurrence requires response states 105--110 exactly")
    if tuple(value.coordinate for value in template.response_coordinates) != (
        FINITE_ACTION_ACTIVE_RESPONSE_COORDINATES
    ) or any(
        not _same_clock_domain(template.causal_cutoff, value)
        or value.coordinate <= template.causal_cutoff.coordinate
        for value in template.response_coordinates
    ):
        raise ValueError("Finite-action recurrence response coordinates change its native state clocks")
    for name, value in (
        ("minimum_active_coverage", template.minimum_active_coverage),
        ("alpha", template.alpha),
        ("one_sided_critical_value", template.one_sided_critical_value),
    ):
        validate_decimal(value, field_name=name, minimum=Decimal(0))
    if (
        template.minimum_evaluable_units != FINITE_ACTION_RECURRENCE_EFFICACY_UNIT_COUNT
        or template.minimum_active_coverage != Decimal(1)
        or template.alpha != Decimal("0.05")
        or template.degrees_of_freedom != FINITE_ACTION_RECURRENCE_DEGREES_OF_FREEDOM
        or template.one_sided_critical_value != FINITE_ACTION_RECURRENCE_ONE_SIDED_CRITICAL_VALUE
        or template.materiality.value != FINITE_ACTION_RECURRENCE_MATERIALITY
        or template.materiality.unit != template.effect_native_unit
        or not template.strict_materiality
    ):
        raise ValueError("Finite-action recurrence inference differs from frozen n=18 Student design")
    if (
        template.reduction_order != FINITE_ACTION_RECURRENCE_REDUCTION_ORDER
        or template.terminal_precedence != FINITE_ACTION_RECURRENCE_TERMINAL_PRECEDENCE
        or not template.preaction_prefix_must_be_byte_equal
        or not template.route_preassigned_without_online_choice
    ):
        raise ValueError("Finite-action recurrence route/reducer/terminal rules differ from the frozen design")
    if (
        template.claim_ceiling != "FINITE_ACTION_RECURRENCE_ONLY"
        or template.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
        or template.outcome_access is not OutcomeAccess.OUTCOME_BLIND
        or template.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
    ):
        raise ValueError("Finite-action recurrence assignment template must remain preissue and outcome-blind")


@dataclass(frozen=True, slots=True)
class FiniteActionRecurrenceAssignment(CanonicalRecord):
    "Exact 18+4, 80-episode finite-action recurrence scientific assignment."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-action-recurrence-assignment'

    assignment_id: str
    frozen_template: ObjectIdentity
    frozen_roster: ObjectIdentity
    roster_collision_audit: ObjectIdentity
    branch_selection: FiniteActionBranchSelection
    units: tuple[FiniteActionRecurrenceUnitSpec, ...]
    routes: tuple[FiniteActionRecurrenceRouteSpec, ...]
    model_member_ids: tuple[str, ...]
    strata: tuple[FiniteActionRecurrenceStratumSpec, ...]
    occurrence_receipts: tuple[FiniteActionOccurrenceQualificationReceipt, ...]
    native_hold_calibration: NativeHoldDecisionCellCalibrationReceipt
    route_qualification: FiniteActionRecurrenceRouteQualificationReceipt
    active_word: OccurrenceActionWord
    matched_hold_word: OccurrenceActionWord
    source: ObjectIdentity
    schedule: ObjectIdentity
    retained_history: ObjectIdentity
    receiver: ObjectIdentity
    horizon: ObjectIdentity
    causal_cutoff: ClockCoordinate
    response_coordinates: tuple[ClockCoordinate, ...]
    effect_quantity_id: str
    effect_native_unit: str
    control_anchor_slot_ids: tuple[str, ...]
    hold_reserve_anchor_slot_id: str
    reserves: tuple[FiniteActionRecurrenceReserveSpec, ...]
    reserve_order: tuple[str, ...]
    minimum_evaluable_units: int
    minimum_active_coverage: Decimal
    alpha: Decimal
    degrees_of_freedom: int
    one_sided_critical_value: Decimal
    materiality: NamedDecimal
    strict_materiality: bool
    reduction_order: tuple[str, ...]
    terminal_precedence: tuple[str, ...]
    preaction_prefix_must_be_byte_equal: bool
    route_preassigned_without_online_choice: bool
    claim_ceiling: str
    evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.assignment_id, field_name="assignment_id")
        validate_stable_id(self.effect_quantity_id, field_name="effect_quantity_id")
        validate_nonempty(self.effect_native_unit, field_name="effect_native_unit")
        validate_nonempty(self.claim_ceiling, field_name="claim_ceiling")
        if (
            self.frozen_roster.object_schema != 'empirical-lawhood/composition/tokamak-control-replication/gym-torax-roster'
            or self.roster_collision_audit.object_schema
            != 'empirical-lawhood/composition/tokamak-control-replication/gym-torax-roster-collision-audit'
        ):
            raise ValueError("Finite-action recurrence assignment requires the exact frozen current roster inputs")
        branch = self.branch_selection
        if not _is_historical_fidelity_recurrence_selection(branch):
            raise ValueError("Finite-action recurrence assignment requires the sealed alternate branch selection")
        require_sorted_unique_ids(self.units, attribute="unit_id", field_name="units")
        require_sorted_unique_ids(self.routes, attribute="episode_id", field_name="routes")
        require_sorted_unique_strings(
            self.model_member_ids,
            field_name="model_member_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(self.strata, attribute="stratum_id", field_name="strata")
        require_sorted_unique_ids(
            self.occurrence_receipts,
            attribute="receipt_id",
            field_name="occurrence_receipts",
        )
        require_sorted_unique_ids(self.reserves, attribute="reserve_id", field_name="reserves")
        if not self.reserve_order or len(set(self.reserve_order)) != len(self.reserve_order):
            raise ValueError("Finite-action recurrence reserve order must contain unique issued reserve IDs")
        for reserve_id in self.reserve_order:
            validate_stable_id(reserve_id, field_name="reserve_order")
        if len(self.model_member_ids) != FINITE_ACTION_RECURRENCE_MEMBER_COUNT:
            raise ValueError("Finite-action recurrence requires exactly two numerical members")
        efficacy = tuple(
            value
            for value in self.units
            if value.role is FiniteActionRecurrenceUnitRole.EFFICACY_ACTIVE
        )
        controls = tuple(
            value
            for value in self.units
            if value.role is FiniteActionRecurrenceUnitRole.HOLD_CONTROL
        )
        if (
            len(efficacy) != FINITE_ACTION_RECURRENCE_EFFICACY_UNIT_COUNT
            or len(controls) != FINITE_ACTION_RECURRENCE_CONTROL_UNIT_COUNT
            or len({value.physical_independent_unit_id for value in self.units}) != len(self.units)
            or len({value.preparation_coordinate_id for value in self.units}) != len(self.units)
        ):
            raise ValueError("Finite-action recurrence requires 18+4 unique physical units")
        if (
            tuple(
                sorted(value.hold_anchor_slot_id for value in controls if value.hold_anchor_slot_id)
            )
            != self.control_anchor_slot_ids
        ):
            raise ValueError("Finite-action recurrence controls must occupy anchor slots 01--04 exactly")

        if (
            len(self.strata) != FINITE_ACTION_RECURRENCE_STRATUM_COUNT
            or tuple(value.stratum_id for value in self.strata)
            != FINITE_ACTION_RECURRENCE_STRATUM_IDS
        ):
            raise ValueError("Finite-action recurrence requires exact inner/outer strata")
        efficacy_map = {value.unit_id: value for value in efficacy}
        covered: set[str] = set()
        for stratum in self.strata:
            if any(
                efficacy_map.get(unit_id) is None
                or efficacy_map[unit_id].stratum_id != stratum.stratum_id
                for unit_id in stratum.unit_ids
            ):
                raise ValueError("Finite-action recurrence stratum changes an issued unit")
            covered.update(stratum.unit_ids)
        if covered != set(efficacy_map):
            raise ValueError("Finite-action recurrence strata do not partition all 18 efficacy units")

        expected_products: set[tuple[str, str, FiniteActionRecurrenceRouteRole]] = set()
        for unit in self.units:
            roles = (
                (
                    FiniteActionRecurrenceRouteRole.ACTIVE,
                    FiniteActionRecurrenceRouteRole.MATCHED_HOLD,
                )
                if unit.role is FiniteActionRecurrenceUnitRole.EFFICACY_ACTIVE
                else (FiniteActionRecurrenceRouteRole.HOLD_CONTROL,)
            )
            expected_products.update(
                (unit.unit_id, member_id, role)
                for member_id in self.model_member_ids
                for role in roles
            )
        if (
            len(self.routes) != FINITE_ACTION_RECURRENCE_PRIMARY_EPISODE_COUNT
            or {value.product_key for value in self.routes} != expected_products
            or len({value.execution_id for value in self.routes}) != len(self.routes)
            or len({value.environment_seed_id for value in self.routes}) != len(self.units)
        ):
            raise ValueError("Finite-action recurrence route product must be exactly 72 efficacy plus eight controls")
        if any(
            len(
                {
                    route.environment_seed_id
                    for route in self.routes
                    if route.unit_id == unit.unit_id
                }
            )
            != 1
            for unit in self.units
        ):
            raise ValueError("Finite-action recurrence nested routes must share one environment seed per physical unit")
        unit_map = {value.unit_id: value for value in self.units}
        for route in self.routes:
            unit = unit_map[route.unit_id]
            if route.model_member_id not in self.model_member_ids:
                raise ValueError("Finite-action recurrence route uses an undeclared member")
            expected_word_id = (
                self.active_word.word_id
                if route.role is FiniteActionRecurrenceRouteRole.ACTIVE
                else self.matched_hold_word.word_id
            )
            if route.action_word_id != expected_word_id:
                raise ValueError("Finite-action recurrence route substitutes another action word")
            if (
                unit.role is FiniteActionRecurrenceUnitRole.EFFICACY_ACTIVE
                and route.role is FiniteActionRecurrenceRouteRole.HOLD_CONTROL
            ) or (
                unit.role is FiniteActionRecurrenceUnitRole.HOLD_CONTROL
                and route.role is not FiniteActionRecurrenceRouteRole.HOLD_CONTROL
            ):
                raise ValueError("Finite-action recurrence route changes its issued unit role")

        expected_occurrence_products = {
            (support_id, member_id)
            for support_id in FINITE_ACTION_TOKAMAK_LOCAL_SUPPORT_IDS
            for member_id in self.model_member_ids
        }
        observed_occurrence_products = {
            (value.qualification_spec.local_support_id, value.qualification_spec.model_member_id)
            for value in self.occurrence_receipts
        }
        if (
            len(self.occurrence_receipts) != len(expected_occurrence_products)
            or observed_occurrence_products != expected_occurrence_products
            or any(
                value.disposition is not FiniteActionOccurrenceDisposition.SUPPORTED
                or value.qualification_spec.active_word != self.active_word
                or value.qualification_spec.matched_hold_word != self.matched_hold_word
                or value.qualification_spec.branch_selection != self.branch_selection
                for value in self.occurrence_receipts
            )
        ):
            raise ValueError("FINITE_ACTION_RECURRENCE_OCCURRENCE_NOT_QUALIFIED")
        route_qualification = self.route_qualification
        occurrence_identities = tuple(
            sorted(
                (
                    ObjectIdentity.from_record(value.receipt_id, value)
                    for value in self.occurrence_receipts
                ),
                key=lambda value: value.object_id,
            )
        )
        if (
            route_qualification.disposition is not FiniteActionRouteCanaryDisposition.QUALIFIED
            or route_qualification.branch_selection != self.branch_selection
            or route_qualification.model_member_ids != self.model_member_ids
            or route_qualification.occurrence_receipts != occurrence_identities
            or route_qualification.native_hold_calibration
            != ObjectIdentity.from_record(
                self.native_hold_calibration.receipt_id,
                self.native_hold_calibration,
            )
        ):
            raise ValueError("FINITE_ACTION_RECURRENCE_ROUTE_NOT_QUALIFIED")
        active_identity = ObjectIdentity.from_record(self.active_word.word_id, self.active_word)
        hold_identity = ObjectIdentity.from_record(
            self.matched_hold_word.word_id,
            self.matched_hold_word,
        )
        if any(
            episode.action_word
            != (
                active_identity
                if episode.episode_role is FiniteActionRouteCanaryEpisodeRole.ACTIVE
                else hold_identity
            )
            for episode in route_qualification.episodes
        ):
            raise ValueError("Finite-action recurrence route canary substitutes another active/HOLD word")
        if (
            self.active_word.word_id != FINITE_ACTION_TOKAMAK_ACTIVE_WORD_ID
            or self.matched_hold_word.word_id != FINITE_ACTION_TOKAMAK_HOLD_WORD_ID
            or len(self.active_word.occurrences) != FINITE_ACTION_OCCURRENCE_COUNT
            or len(self.matched_hold_word.occurrences) != FINITE_ACTION_OCCURRENCE_COUNT
            or self.retained_history.object_id != self.active_word.retained_history_id
            or self.receiver.object_id != self.active_word.receiver_id
            or self.horizon.object_id != self.active_word.horizon_id
        ):
            raise ValueError("Finite-action recurrence assignment changes the exact active/HOLD occurrence pair")
        calibration = self.native_hold_calibration
        if calibration.disposition is not NativeHoldCalibrationDisposition.SUPPORTED:
            raise ValueError("Finite-action recurrence requires complete supported C-anchor HOLD calibration")
        calibration_slots = {value.slot_id for value in calibration.expected_entries}
        calibration_members = {value.model_member_id for value in calibration.expected_entries}
        if (
            calibration_slots
            != {
                *FINITE_ACTION_RECURRENCE_CONTROL_ANCHOR_SLOT_IDS,
                FINITE_ACTION_RECURRENCE_HOLD_RESERVE_SLOT_ID,
            }
            or calibration_members != set(self.model_member_ids)
            or len(calibration.expected_entries) != 10
            or {value.occurrence_role_id for value in calibration.expected_entries}
            != {FINITE_ACTION_RECURRENCE_CALIBRATION_OCCURRENCE_ROLE_ID}
        ):
            raise ValueError("Finite-action recurrence calibration omits an exact anchor/member coordinate")
        expected_hold_identity = ObjectIdentity.from_record(
            self.matched_hold_word.word_id,
            self.matched_hold_word,
        )
        if any(
            value.source != self.source
            or value.schedule != self.schedule
            or value.native_hold_action_word != expected_hold_identity
            or value.retained_history != self.retained_history
            or value.horizon != self.horizon
            for value in calibration.occurrence_receipts
        ):
            raise ValueError("Finite-action recurrence C calibration changes source/schedule/HOLD/history/horizon")
        calibration_fingerprints = {
            value.slot_id: value.preparation_fingerprint
            for value in calibration.occurrence_receipts
        }
        if any(
            control.preparation_sha256 != calibration_fingerprints[control.hold_anchor_slot_id]
            for control in controls
            if control.hold_anchor_slot_id is not None
        ):
            raise ValueError("Finite-action recurrence HOLD control does not alias its selected anchor values")

        if (
            self.control_anchor_slot_ids != FINITE_ACTION_RECURRENCE_CONTROL_ANCHOR_SLOT_IDS
            or self.hold_reserve_anchor_slot_id != FINITE_ACTION_RECURRENCE_HOLD_RESERVE_SLOT_ID
        ):
            raise ValueError("Finite-action recurrence HOLD anchor/reserve slots differ from the frozen tokamak-control roster")
        efficacy_reserves = tuple(
            value
            for value in self.reserves
            if value.role is FiniteActionRecurrenceReserveRole.EFFICACY_WHOLE_UNIT
        )
        hold_reserves = tuple(
            value
            for value in self.reserves
            if value.role is FiniteActionRecurrenceReserveRole.HOLD_CONTROL_WHOLE_UNIT
        )
        if (
            len(efficacy_reserves) != 2
            or {value.stratum_id for value in efficacy_reserves}
            != set(FINITE_ACTION_RECURRENCE_STRATUM_IDS)
            or len(hold_reserves) != 1
            or set(self.reserve_order) != {value.reserve_id for value in self.reserves}
            or len(self.reserve_order) != 3
        ):
            raise ValueError("Finite-action recurrence requires two efficacy reserves and one HOLD reserve")
        if (
            hold_reserves[0].preparation_sha256
            != calibration_fingerprints[FINITE_ACTION_RECURRENCE_HOLD_RESERVE_SLOT_ID]
        ):
            raise ValueError("Finite-action recurrence HOLD reserve does not alias selected anchor slot 05")
        occupied_coordinates = {value.preparation_coordinate_id for value in self.units}
        occupied_hashes = {value.preparation_sha256 for value in self.units}
        if any(
            value.preparation_coordinate_id in occupied_coordinates
            or value.preparation_sha256 in occupied_hashes
            for value in efficacy_reserves
        ):
            raise ValueError("Finite-action recurrence efficacy reserve aliases a scheduled scientific unit")

        if len(self.response_coordinates) != FINITE_ACTION_RECURRENCE_RESPONSE_SAMPLE_COUNT:
            raise ValueError("Finite-action recurrence requires response states 105--110 exactly")
        if tuple(
            value.coordinate for value in self.response_coordinates
        ) != FINITE_ACTION_ACTIVE_RESPONSE_COORDINATES or any(
            not _same_clock_domain(self.causal_cutoff, value)
            or value.coordinate <= self.causal_cutoff.coordinate
            for value in self.response_coordinates
        ):
            raise ValueError("Finite-action recurrence response coordinates change its native state clocks")
        for name, value in (
            ("minimum_active_coverage", self.minimum_active_coverage),
            ("alpha", self.alpha),
            ("one_sided_critical_value", self.one_sided_critical_value),
        ):
            validate_decimal(value, field_name=name, minimum=Decimal(0))
        if (
            self.minimum_evaluable_units != FINITE_ACTION_RECURRENCE_EFFICACY_UNIT_COUNT
            or self.minimum_active_coverage != Decimal(1)
            or self.alpha != Decimal("0.05")
            or self.degrees_of_freedom != FINITE_ACTION_RECURRENCE_DEGREES_OF_FREEDOM
            or self.one_sided_critical_value != FINITE_ACTION_RECURRENCE_ONE_SIDED_CRITICAL_VALUE
            or self.materiality.value != FINITE_ACTION_RECURRENCE_MATERIALITY
            or self.materiality.unit != self.effect_native_unit
            or not self.strict_materiality
        ):
            raise ValueError("Finite-action recurrence inference differs from frozen n=18 Student design")
        if (
            self.reduction_order != FINITE_ACTION_RECURRENCE_REDUCTION_ORDER
            or self.terminal_precedence != FINITE_ACTION_RECURRENCE_TERMINAL_PRECEDENCE
            or not self.preaction_prefix_must_be_byte_equal
            or not self.route_preassigned_without_online_choice
        ):
            raise ValueError("Finite-action recurrence route/reducer/terminal rules differ from the frozen design")
        if (
            self.claim_ceiling != "FINITE_ACTION_RECURRENCE_ONLY"
            or self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
            or self.outcome_access is not OutcomeAccess.EVALUATION_REVEALED
            or self.visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE
        ):
            raise ValueError("Finite-action recurrence assignment exceeds its lower recurrence claim ceiling")
        template = _template_from_recurrence_assignment(self)
        if self.frozen_template != ObjectIdentity.from_record(template.template_id, template):
            raise ValueError("Finite-action recurrence assignment differs from its frozen preissue template")

    @property
    def efficacy_units(self) -> tuple[FiniteActionRecurrenceUnitSpec, ...]:
        return tuple(
            value
            for value in self.units
            if value.role is FiniteActionRecurrenceUnitRole.EFFICACY_ACTIVE
        )

    @property
    def hold_controls(self) -> tuple[FiniteActionRecurrenceUnitSpec, ...]:
        return tuple(
            value
            for value in self.units
            if value.role is FiniteActionRecurrenceUnitRole.HOLD_CONTROL
        )


def _template_from_recurrence_assignment(
    assignment: FiniteActionRecurrenceAssignment,
) -> FiniteActionRecurrenceAssignmentTemplate:
    return FiniteActionRecurrenceAssignmentTemplate(
        template_id=assignment.frozen_template.object_id,
        expected_assignment_id=assignment.assignment_id,
        frozen_roster=assignment.frozen_roster,
        roster_collision_audit=assignment.roster_collision_audit,
        expected_branch_selection_receipt_id=(
            finite_action_branch_selection_receipt_id(assignment.branch_selection)
        ),
        expected_branch_selection_receipt_schema=assignment.branch_selection.SCHEMA,
        expected_occurrence_receipt_ids=tuple(
            value.receipt_id for value in assignment.occurrence_receipts
        ),
        expected_occurrence_receipt_schema=(FiniteActionOccurrenceQualificationReceipt.SCHEMA),
        expected_native_hold_calibration_receipt_id=(assignment.native_hold_calibration.receipt_id),
        expected_native_hold_calibration_receipt_schema=(
            NativeHoldDecisionCellCalibrationReceipt.SCHEMA
        ),
        expected_route_qualification_receipt_id=(assignment.route_qualification.receipt_id),
        expected_route_qualification_receipt_schema=(
            FiniteActionRecurrenceRouteQualificationReceipt.SCHEMA
        ),
        units=assignment.units,
        routes=assignment.routes,
        model_member_ids=assignment.model_member_ids,
        strata=assignment.strata,
        active_word=assignment.active_word,
        matched_hold_word=assignment.matched_hold_word,
        source=assignment.source,
        schedule=assignment.schedule,
        retained_history=assignment.retained_history,
        receiver=assignment.receiver,
        horizon=assignment.horizon,
        causal_cutoff=assignment.causal_cutoff,
        response_coordinates=assignment.response_coordinates,
        effect_quantity_id=assignment.effect_quantity_id,
        effect_native_unit=assignment.effect_native_unit,
        control_anchor_slot_ids=assignment.control_anchor_slot_ids,
        hold_reserve_anchor_slot_id=assignment.hold_reserve_anchor_slot_id,
        reserves=assignment.reserves,
        reserve_order=assignment.reserve_order,
        minimum_evaluable_units=assignment.minimum_evaluable_units,
        minimum_active_coverage=assignment.minimum_active_coverage,
        alpha=assignment.alpha,
        degrees_of_freedom=assignment.degrees_of_freedom,
        one_sided_critical_value=assignment.one_sided_critical_value,
        materiality=assignment.materiality,
        strict_materiality=assignment.strict_materiality,
        reduction_order=assignment.reduction_order,
        terminal_precedence=assignment.terminal_precedence,
        preaction_prefix_must_be_byte_equal=(assignment.preaction_prefix_must_be_byte_equal),
        route_preassigned_without_online_choice=(
            assignment.route_preassigned_without_online_choice
        ),
        claim_ceiling=assignment.claim_ceiling,
        evidence_ceiling=assignment.evidence_ceiling,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )


@dataclass(frozen=True, slots=True)
class FiniteActionRecurrenceAssignmentBindingReceipt(CanonicalRecord):
    """Deterministic post-route attachment receipt for one preissue assignment."""

    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/planning/finite-action-recurrence-assignment-binding-receipt'
    )

    receipt_id: str
    template: ObjectIdentity
    branch_selection: ObjectIdentity
    occurrence_receipts: tuple[ObjectIdentity, ...]
    native_hold_calibration: ObjectIdentity
    route_qualification: ObjectIdentity
    assignment: ObjectIdentity
    binding_implementation_id: str
    binding_implementation_sha256: str
    outcome_dependent_choice: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    scientific_verdict: None = None

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        validate_stable_id(
            self.binding_implementation_id,
            field_name="binding_implementation_id",
        )
        validate_sha256(
            self.binding_implementation_sha256,
            field_name="binding_implementation_sha256",
        )
        require_sorted_unique_ids(
            self.occurrence_receipts,
            attribute="object_id",
            field_name="occurrence_receipts",
        )
        if len(self.occurrence_receipts) != 6:
            raise ValueError("Finite-action recurrence assignment binding requires six occurrence receipts")
        expected_schemas = (
            (self.template, FiniteActionRecurrenceAssignmentTemplate.SCHEMA),
            (
                self.native_hold_calibration,
                NativeHoldDecisionCellCalibrationReceipt.SCHEMA,
            ),
            (
                self.route_qualification,
                FiniteActionRecurrenceRouteQualificationReceipt.SCHEMA,
            ),
            (self.assignment, FiniteActionRecurrenceAssignment.SCHEMA),
        )
        if (
            self.branch_selection.object_schema
            not in {
                PreissueBranchSelectionReceipt.SCHEMA,
                FiniteActionAlternateBranchSelectionBridge.SCHEMA,
            }
            or any(identity.object_schema != schema for identity, schema in expected_schemas)
            or any(
                value.object_schema != FiniteActionOccurrenceQualificationReceipt.SCHEMA
                for value in self.occurrence_receipts
            )
        ):
            raise ValueError("Finite-action recurrence assignment binding contains another record schema")
        if self.outcome_dependent_choice or self.scientific_verdict is not None:
            raise ValueError("Finite-action recurrence assignment binding cannot change or adjudicate the schedule")
        if (
            self.outcome_access is not OutcomeAccess.EVALUATION_REVEALED
            or self.visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE
        ):
            raise ValueError("Finite-action recurrence assignment binding has the wrong post-route access")


def bind_finite_action_recurrence_assignment_template(
    *,
    receipt_id: str,
    template: FiniteActionRecurrenceAssignmentTemplate,
    branch_selection: FiniteActionBranchSelection,
    occurrence_receipts: tuple[FiniteActionOccurrenceQualificationReceipt, ...],
    native_hold_calibration: NativeHoldDecisionCellCalibrationReceipt,
    route_qualification: FiniteActionRecurrenceRouteQualificationReceipt,
    binding_implementation_id: str,
    binding_implementation_sha256: str,
) -> tuple[
    FiniteActionRecurrenceAssignment,
    FiniteActionRecurrenceAssignmentBindingReceipt,
]:
    "Attach authenticated finite-action recurrence prerequisites without changing one route byte."

    require_sorted_unique_ids(
        occurrence_receipts,
        attribute="receipt_id",
        field_name="occurrence_receipts",
    )
    if (
        finite_action_branch_selection_receipt_id(branch_selection)
        != template.expected_branch_selection_receipt_id
        or branch_selection.SCHEMA != template.expected_branch_selection_receipt_schema
        or tuple(value.receipt_id for value in occurrence_receipts)
        != template.expected_occurrence_receipt_ids
        or native_hold_calibration.receipt_id
        != template.expected_native_hold_calibration_receipt_id
        or route_qualification.receipt_id != template.expected_route_qualification_receipt_id
    ):
        raise ValueError("Finite-action recurrence assignment binding substitutes a predeclared record ID")
    assignment = FiniteActionRecurrenceAssignment(
        assignment_id=template.expected_assignment_id,
        frozen_template=ObjectIdentity.from_record(template.template_id, template),
        frozen_roster=template.frozen_roster,
        roster_collision_audit=template.roster_collision_audit,
        branch_selection=branch_selection,
        units=template.units,
        routes=template.routes,
        model_member_ids=template.model_member_ids,
        strata=template.strata,
        occurrence_receipts=occurrence_receipts,
        native_hold_calibration=native_hold_calibration,
        route_qualification=route_qualification,
        active_word=template.active_word,
        matched_hold_word=template.matched_hold_word,
        source=template.source,
        schedule=template.schedule,
        retained_history=template.retained_history,
        receiver=template.receiver,
        horizon=template.horizon,
        causal_cutoff=template.causal_cutoff,
        response_coordinates=template.response_coordinates,
        effect_quantity_id=template.effect_quantity_id,
        effect_native_unit=template.effect_native_unit,
        control_anchor_slot_ids=template.control_anchor_slot_ids,
        hold_reserve_anchor_slot_id=template.hold_reserve_anchor_slot_id,
        reserves=template.reserves,
        reserve_order=template.reserve_order,
        minimum_evaluable_units=template.minimum_evaluable_units,
        minimum_active_coverage=template.minimum_active_coverage,
        alpha=template.alpha,
        degrees_of_freedom=template.degrees_of_freedom,
        one_sided_critical_value=template.one_sided_critical_value,
        materiality=template.materiality,
        strict_materiality=template.strict_materiality,
        reduction_order=template.reduction_order,
        terminal_precedence=template.terminal_precedence,
        preaction_prefix_must_be_byte_equal=(template.preaction_prefix_must_be_byte_equal),
        route_preassigned_without_online_choice=(template.route_preassigned_without_online_choice),
        claim_ceiling=template.claim_ceiling,
        evidence_ceiling=template.evidence_ceiling,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
    )
    occurrence_identities = tuple(
        sorted(
            (ObjectIdentity.from_record(value.receipt_id, value) for value in occurrence_receipts),
            key=lambda value: value.object_id,
        )
    )
    binding = FiniteActionRecurrenceAssignmentBindingReceipt(
        receipt_id=receipt_id,
        template=ObjectIdentity.from_record(template.template_id, template),
        branch_selection=finite_action_branch_selection_identity(branch_selection),
        occurrence_receipts=occurrence_identities,
        native_hold_calibration=ObjectIdentity.from_record(
            native_hold_calibration.receipt_id,
            native_hold_calibration,
        ),
        route_qualification=ObjectIdentity.from_record(
            route_qualification.receipt_id,
            route_qualification,
        ),
        assignment=ObjectIdentity.from_record(assignment.assignment_id, assignment),
        binding_implementation_id=binding_implementation_id,
        binding_implementation_sha256=binding_implementation_sha256,
        outcome_dependent_choice=False,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
    )
    return assignment, binding


class FiniteActionRecurrenceAssignmentTemplateBinder:
    """Code-owned deterministic wrapper for the post-route assignment binding."""

    def __init__(self, *, implementation_id: str, implementation_sha256: str) -> None:
        validate_stable_id(implementation_id, field_name="implementation_id")
        validate_sha256(implementation_sha256, field_name="implementation_sha256")
        self._implementation_id = implementation_id
        self._implementation_sha256 = implementation_sha256

    def bind(
        self,
        *,
        receipt_id: str,
        template: FiniteActionRecurrenceAssignmentTemplate,
        branch_selection: FiniteActionBranchSelection,
        occurrence_receipts: tuple[FiniteActionOccurrenceQualificationReceipt, ...],
        native_hold_calibration: NativeHoldDecisionCellCalibrationReceipt,
        route_qualification: FiniteActionRecurrenceRouteQualificationReceipt,
    ) -> tuple[
        FiniteActionRecurrenceAssignment,
        FiniteActionRecurrenceAssignmentBindingReceipt,
    ]:
        return bind_finite_action_recurrence_assignment_template(
            receipt_id=receipt_id,
            template=template,
            branch_selection=branch_selection,
            occurrence_receipts=occurrence_receipts,
            native_hold_calibration=native_hold_calibration,
            route_qualification=route_qualification,
            binding_implementation_id=self._implementation_id,
            binding_implementation_sha256=self._implementation_sha256,
        )


def _artifact_bytes(value: ArtifactIdentity) -> tuple[str, str, str, int]:
    return value.sha256, value.payload_schema, value.media_type, value.size_bytes


@dataclass(frozen=True, slots=True)
class FiniteActionRecurrenceSealedEpisode(CanonicalRecord):
    """One scheduled route sealed without revealing its scientific response."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-action-recurrence-sealed-episode'

    episode_id: str
    route: FiniteActionRecurrenceRouteSpec
    preparation_artifact: ArtifactIdentity
    config_artifact: ArtifactIdentity
    preaction_prefix_artifact: ArtifactIdentity
    prefix_complete_through_state_104: bool
    preaction_disposition: FiniteActionPreactionDisposition
    observed_occurrences: tuple[ObservedActionOccurrence, ...]
    clipped: bool
    rejected: bool
    substituted: bool
    early_terminated: bool
    delivery_trace: ObjectIdentity
    sealed_outcome_artifact: ArtifactIdentity
    custody_evidence: tuple[ObjectIdentity, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.episode_id, field_name="episode_id")
        if self.episode_id != self.route.episode_id:
            raise ValueError("Finite-action recurrence sealed episode changes its scheduled route")
        require_sorted_unique_ids(
            self.observed_occurrences,
            attribute="expected_occurrence_id",
            field_name="observed_occurrences",
        )
        require_sorted_unique_ids(
            self.custody_evidence,
            attribute="object_id",
            field_name="custody_evidence",
        )
        if not self.custody_evidence:
            raise ValueError("Finite-action recurrence sealed episode requires custody evidence")
        if self.preaction_disposition is FiniteActionPreactionDisposition.NONATTEMPT:
            if self.observed_occurrences or any(
                (self.clipped, self.rejected, self.substituted, self.early_terminated)
            ):
                raise ValueError("pre-action NONATTEMPT cannot carry post-104 action evidence")
        elif not self.prefix_complete_through_state_104:
            raise ValueError("Finite-action recurrence ready route requires a complete pre-action prefix")
        if self.outcome_access is not OutcomeAccess.EVALUATION_SEALED:
            raise ValueError("Finite-action recurrence prospective episode must remain sealed")


@dataclass(frozen=True, slots=True)
class FiniteActionRecurrenceProspectiveBundle(CanonicalRecord):
    "Exact 80-episode sealed finite-action recurrence product, with no controller/tick/reference."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-action-recurrence-prospective-bundle'

    bundle_id: str
    assignment: FiniteActionRecurrenceAssignment
    episodes: tuple[FiniteActionRecurrenceSealedEpisode, ...]
    issue_receipt: ObjectIdentity
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.bundle_id, field_name="bundle_id")
        require_sorted_unique_ids(self.episodes, attribute="episode_id", field_name="episodes")
        if len(self.episodes) != FINITE_ACTION_RECURRENCE_PRIMARY_EPISODE_COUNT or tuple(
            value.episode_id for value in self.episodes
        ) != tuple(value.episode_id for value in self.assignment.routes):
            raise ValueError("Finite-action recurrence prospective bundle differs from the exact 80-route assignment")
        if len({value.delivery_trace.object_id for value in self.episodes}) != len(self.episodes):
            raise ValueError("Finite-action recurrence episodes reuse a delivery trace")
        for attribute in (
            "preparation_artifact",
            "config_artifact",
            "preaction_prefix_artifact",
            "sealed_outcome_artifact",
        ):
            ids = tuple(getattr(value, attribute).artifact_id for value in self.episodes)
            if len(set(ids)) != len(ids):
                raise ValueError("Finite-action recurrence episodes reuse a fresh artifact identity")
        units = {value.unit_id: value for value in self.assignment.units}
        by_product = {value.route.product_key: value for value in self.episodes}
        for episode in self.episodes:
            unit = units[episode.route.unit_id]
            if episode.preparation_artifact.sha256 != unit.preparation_sha256:
                raise ValueError("Finite-action recurrence episode changes its issued preparation bytes")
        for unit in self.assignment.efficacy_units:
            for member_id in self.assignment.model_member_ids:
                active = by_product[
                    (unit.unit_id, member_id, FiniteActionRecurrenceRouteRole.ACTIVE)
                ]
                hold = by_product[
                    (unit.unit_id, member_id, FiniteActionRecurrenceRouteRole.MATCHED_HOLD)
                ]
                equal_complete_prefix = (
                    active.prefix_complete_through_state_104
                    and hold.prefix_complete_through_state_104
                    and _artifact_bytes(active.preparation_artifact)
                    == _artifact_bytes(hold.preparation_artifact)
                    and _artifact_bytes(active.config_artifact)
                    == _artifact_bytes(hold.config_artifact)
                    and _artifact_bytes(active.preaction_prefix_artifact)
                    == _artifact_bytes(hold.preaction_prefix_artifact)
                )
                paired_nonattempt = (
                    active.preaction_disposition is FiniteActionPreactionDisposition.NONATTEMPT
                    and hold.preaction_disposition is FiniteActionPreactionDisposition.NONATTEMPT
                )
                if not equal_complete_prefix and not paired_nonattempt:
                    raise ValueError("unequal/missing efficacy prefix requires paired NONATTEMPT")
                if equal_complete_prefix and paired_nonattempt:
                    raise ValueError("complete equal efficacy pair cannot be recoded NONATTEMPT")
        for control in self.assignment.hold_controls:
            for member_id in self.assignment.model_member_ids:
                episode = by_product[
                    (
                        control.unit_id,
                        member_id,
                        FiniteActionRecurrenceRouteRole.HOLD_CONTROL,
                    )
                ]
                if (
                    not episode.prefix_complete_through_state_104
                    and episode.preaction_disposition
                    is not FiniteActionPreactionDisposition.NONATTEMPT
                ):
                    raise ValueError("incomplete HOLD-control prefix requires NONATTEMPT")
        if self.outcome_access is not OutcomeAccess.EVALUATION_SEALED:
            raise ValueError("Finite-action recurrence prospective bundle must remain sealed")


@dataclass(frozen=True, slots=True)
class FiniteActionRecurrenceResponseSample(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-action-recurrence-response-sample'

    sample_id: str
    quantity_id: str
    coordinate: ClockCoordinate
    value: NamedDecimal

    def __post_init__(self) -> None:
        validate_stable_id(self.sample_id, field_name="sample_id")
        validate_stable_id(self.quantity_id, field_name="quantity_id")


@dataclass(frozen=True, slots=True)
class FiniteActionRecurrencePredicateResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-action-recurrence-predicate-result'

    result_id: str
    kind: FiniteActionOccurrencePredicateKind
    status: GateStatus
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.status is GateStatus.PASS and self.reason_codes:
            raise ValueError("passing finite-action recurrence predicate cannot carry reasons")
        if self.status is not GateStatus.PASS and not self.reason_codes:
            raise ValueError("nonpassing finite-action recurrence predicate requires reasons")


@dataclass(frozen=True, slots=True)
class FiniteActionRecurrenceRevealedEpisode(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-action-recurrence-revealed-episode'

    reveal_id: str
    episode_id: str
    sealed_episode: ObjectIdentity
    delivery_trace: ObjectIdentity
    sealed_outcome_artifact: ArtifactIdentity
    outcome_trace_id: str
    response_samples: tuple[FiniteActionRecurrenceResponseSample, ...]
    predicate_results: tuple[FiniteActionRecurrencePredicateResult, ...]
    technical_reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.reveal_id, field_name="reveal_id")
        validate_stable_id(self.episode_id, field_name="episode_id")
        validate_stable_id(self.outcome_trace_id, field_name="outcome_trace_id")
        if self.sealed_episode.object_schema != FiniteActionRecurrenceSealedEpisode.SCHEMA:
            raise ValueError("Finite-action recurrence reveal binds another sealed-episode schema")
        require_sorted_unique_ids(
            self.response_samples,
            attribute="sample_id",
            field_name="response_samples",
        )
        require_sorted_unique_ids(
            self.predicate_results,
            attribute="result_id",
            field_name="predicate_results",
        )
        require_sorted_unique_strings(
            self.technical_reason_codes,
            field_name="technical_reason_codes",
        )
        if self.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL:
            raise ValueError("Finite-action recurrence revealed episode is evaluator-only")


@dataclass(frozen=True, slots=True)
class FiniteActionRecurrenceRevealedBundle(CanonicalRecord):
    """Authenticated complete reveal of the frozen type-distinct assignment."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-action-recurrence-revealed-bundle'

    reveal_id: str
    sealed_bundle: FiniteActionRecurrenceProspectiveBundle
    integrity: FiniteActionRecurrenceRevealIntegrity
    reveal_authorization: ObjectIdentity | None
    integrity_evidence: tuple[ObjectIdentity, ...]
    integrity_reason_codes: tuple[str, ...]
    episodes: tuple[FiniteActionRecurrenceRevealedEpisode, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.reveal_id, field_name="reveal_id")
        require_sorted_unique_ids(
            self.integrity_evidence,
            attribute="object_id",
            field_name="integrity_evidence",
        )
        require_sorted_unique_strings(
            self.integrity_reason_codes,
            field_name="integrity_reason_codes",
        )
        require_sorted_unique_ids(self.episodes, attribute="episode_id", field_name="episodes")
        valid = self.integrity is FiniteActionRecurrenceRevealIntegrity.VALID
        if valid:
            if (
                self.reveal_authorization is None
                or not self.integrity_evidence
                or self.integrity_reason_codes
            ):
                raise ValueError("valid finite-action recurrence reveal requires authority and clean custody")
        elif not self.integrity_reason_codes:
            raise ValueError("invalid finite-action recurrence reveal requires a decisive reason")
        sealed = {value.episode_id: value for value in self.sealed_bundle.episodes}
        if not set(value.episode_id for value in self.episodes) <= set(sealed):
            raise ValueError("Finite-action recurrence reveal adds an unsealed episode")
        if len({value.outcome_trace_id for value in self.episodes}) != len(self.episodes):
            raise ValueError("Finite-action recurrence revealed episodes reuse an outcome trace")
        assignment = self.sealed_bundle.assignment
        routes = {value.episode_id: value for value in assignment.routes}
        for value in self.episodes:
            source = sealed[value.episode_id]
            if (
                value.sealed_episode != ObjectIdentity.from_record(source.episode_id, source)
                or value.delivery_trace != source.delivery_trace
                or value.sealed_outcome_artifact != source.sealed_outcome_artifact
            ):
                raise ValueError("Finite-action recurrence reveal changes its seal or delivery trace")
            nonattempt = (
                source.preaction_disposition is FiniteActionPreactionDisposition.NONATTEMPT
            )
            route = routes[value.episode_id]
            if nonattempt:
                if value.response_samples or value.predicate_results:
                    raise ValueError("Finite-action recurrence NONATTEMPT fabricates post-104 outcomes")
                continue
            if tuple(sorted(result.kind for result in value.predicate_results)) != tuple(
                sorted(FINITE_ACTION_REQUIRED_PREDICATE_KINDS)
            ):
                raise ValueError("Finite-action recurrence executed route lacks the seven decisive predicates")
            if route.role is FiniteActionRecurrenceRouteRole.HOLD_CONTROL:
                if value.response_samples:
                    raise ValueError("Finite-action recurrence HOLD control cannot enter the efficacy statistic")
            elif (
                len(value.response_samples) != FINITE_ACTION_RECURRENCE_RESPONSE_SAMPLE_COUNT
                or tuple(sample.coordinate for sample in value.response_samples)
                != assignment.response_coordinates
                or any(
                    sample.quantity_id != assignment.effect_quantity_id
                    or sample.value.unit != assignment.effect_native_unit
                    for sample in value.response_samples
                )
            ):
                raise ValueError("Finite-action recurrence efficacy route changes its exact phase response")
        if valid and (
            len(self.episodes) != FINITE_ACTION_RECURRENCE_PRIMARY_EPISODE_COUNT
            or set(value.episode_id for value in self.episodes) != set(sealed)
        ):
            raise ValueError("valid finite-action recurrence reveal is not the complete 80-episode product")
        if self.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL:
            raise ValueError("Finite-action recurrence revealed bundle is evaluator-only")


__all__ = [
    "FINITE_ACTION_RECURRENCE_CALIBRATION_OCCURRENCE_ROLE_ID",
    "FINITE_ACTION_RECURRENCE_CONTROL_ANCHOR_SLOT_IDS",
    "FINITE_ACTION_RECURRENCE_CONTROL_EPISODE_COUNT",
    "FINITE_ACTION_RECURRENCE_CONTROL_UNIT_COUNT",
    "FINITE_ACTION_RECURRENCE_DEGREES_OF_FREEDOM",
    "FINITE_ACTION_RECURRENCE_EFFICACY_EPISODE_COUNT",
    "FINITE_ACTION_RECURRENCE_EFFICACY_UNIT_COUNT",
    "FINITE_ACTION_RECURRENCE_HOLD_RESERVE_SLOT_ID",
    "FINITE_ACTION_RECURRENCE_MATERIALITY",
    "FINITE_ACTION_RECURRENCE_MEMBER_COUNT",
    "FINITE_ACTION_RECURRENCE_NONRESERVABLE_REASON_CODES",
    "FINITE_ACTION_RECURRENCE_ONE_SIDED_CRITICAL_VALUE",
    "FINITE_ACTION_RECURRENCE_PRIMARY_EPISODE_COUNT",
    "FINITE_ACTION_RECURRENCE_REDUCTION_ORDER",
    "FINITE_ACTION_RECURRENCE_RESERVE_REASON_CODES",
    "FINITE_ACTION_RECURRENCE_RESPONSE_SAMPLE_COUNT",
    "FINITE_ACTION_RECURRENCE_STRATUM_IDS",
    "FINITE_ACTION_RECURRENCE_TERMINAL_PRECEDENCE",
    'FiniteActionPreactionDisposition',
    'FiniteActionRecurrenceAssignmentBindingReceipt',
    'FiniteActionRecurrenceAssignmentTemplateBinder',
    'FiniteActionRecurrenceAssignmentTemplate',
    'FiniteActionRecurrenceRouteQualificationReceipt',
    'FiniteActionRecurrenceAssignment',
    'FiniteActionRecurrencePredicateResult',
    'FiniteActionRecurrenceProspectiveBundle',
    'FiniteActionRecurrenceReserveRole',
    'FiniteActionRecurrenceReserveSpec',
    'FiniteActionRecurrenceRevealIntegrity',
    'FiniteActionRecurrenceRevealedBundle',
    'FiniteActionRecurrenceRevealedEpisode',
    'FiniteActionRecurrenceResponseSample',
    'FiniteActionRecurrenceRouteRole',
    'FiniteActionRecurrenceRouteSpec',
    'FiniteActionRecurrenceSealedEpisode',
    'FiniteActionRecurrenceStratumSpec',
    'FiniteActionRecurrenceUnitRole',
    'FiniteActionRecurrenceUnitSpec',
    'FiniteActionRouteCanaryCoordinateRole',
    'FiniteActionRouteCanaryDisposition',
    'FiniteActionRouteCanaryEpisodeReceipt',
    'FiniteActionRouteCanaryEpisodeRole',
    'bind_finite_action_recurrence_assignment_template',
    'validate_finite_action_occurrence_recurrence_expectations',
]
