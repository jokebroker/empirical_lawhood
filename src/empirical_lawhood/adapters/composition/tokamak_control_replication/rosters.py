"Pure, outcome-blind tokamak-control replication roster authoring and collision accounting.\n\nThis bounded authoring module owns only inert experiment records. It opens no\nsource, constructs no provider, and grants no issue, execution, reveal, or\nscientific authority.\n"

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
from enum import StrEnum
import hashlib
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess
from .scientific_inputs import GymToraxRosterScientificInputs, gym_torax_default_roster_scientific_inputs
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    canonical_json_bytes,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.planning.native_hold_decision_cell_calibration import NativeHoldCalibrationAnchorOption, NativeHoldCalibrationAnchorSlot, preparation_values_fingerprint

from empirical_lawhood.adapters.simulators.gym_torax_native.action_word import GYM_TORAX_ACTION_WORD_IDS, GYM_TORAX_LOWER_IP_ACTION_WORD_ID, GYM_TORAX_NATIVE_HOLD_ACTION_WORD_ID
from empirical_lawhood.adapters.simulators.gym_torax_native.diagnostic_contracts import GYM_TORAX_PRIMARY_MEMBER_ID, GYM_TORAX_REFINED_MEMBER_ID


GYM_TORAX_QUALIFICATION_ROSTER_DOMAIN = 'tokamak-control/source-qualification'
GYM_TORAX_EVALUATION_PRIMARY_ROSTER_DOMAIN = 'tokamak-control/matched-evaluation-primary'
GYM_TORAX_EVALUATION_RESERVE_ROSTER_DOMAIN = 'tokamak-control/matched-evaluation-reserve'
GYM_TORAX_PROSPECTIVE_EFFICACY_ROSTER_DOMAIN = 'tokamak-control/prospective-efficacy'
GYM_TORAX_PROSPECTIVE_HOLD_ROSTER_DOMAIN = 'tokamak-control/prospective-hold'
GYM_TORAX_ROUTE_ACTIVE_ROSTER_DOMAIN = 'tokamak-control/route-active-canary'
GYM_TORAX_ROUTE_HOLD_ROSTER_DOMAIN = 'tokamak-control/route-hold-canary'
GYM_TORAX_ROSTER_GENERATOR_ID = 'roster-generator.tokamak-control.committed-words-round-half-even'
GYM_TORAX_ROSTER_ID = 'roster.tokamak-control.complete-maximal'

_QUANTUM = Decimal("0.0000001")
_TWO_64 = Decimal(2**64)
_MODEL_MEMBER_IDS = tuple(sorted((GYM_TORAX_PRIMARY_MEMBER_ID, GYM_TORAX_REFINED_MEMBER_ID)))
_CELL_SUFFIX = {
    "C": "c",
    "B": "b",
    "T": "t",
    "BT": "bt",
}
_PREPARATION_VALUE_IDS = frozenset(
    {
        'tokamak-control.preparation.bootstrap-multiplier',
        'tokamak-control.preparation.initial-density-nbar',
        'tokamak-control.preparation.initial-temperature-scale',
        'tokamak-control.preparation.inner-transport-scale',
    }
)


class GymToraxRosterTier(StrEnum):
    TIER_01 = "TIER_01"
    TIER_02 = "TIER_02"
    TIER_03 = "TIER_03"


class GymToraxQuartetRole(StrEnum):
    QUALIFICATION = "QUALIFICATION"
    EVALUATION_PRIMARY = "EVALUATION_PRIMARY"
    EVALUATION_RESERVE = "EVALUATION_RESERVE"


class GymToraxCellKind(StrEnum):
    C = "C"
    B = "B"
    T = "T"
    BT = "BT"


class GymToraxProspectiveRole(StrEnum):
    EFFICACY_PRIMARY = "EFFICACY_PRIMARY"
    EFFICACY_RESERVE = "EFFICACY_RESERVE"
    HOLD_CONTROL = "HOLD_CONTROL"
    HOLD_RESERVE = "HOLD_RESERVE"
    ROUTE_ACTIVE_CANARY = "ROUTE_ACTIVE_CANARY"
    ROUTE_HOLD_CANARY = "ROUTE_HOLD_CANARY"


class GymToraxProspectiveStratum(StrEnum):
    NONE = "NONE"
    INTERFACE = "INTERFACE"
    OUTER = "OUTER"


class GymToraxEpisodeStage(StrEnum):
    EXCLUDED_QUALIFICATION = "EXCLUDED_QUALIFICATION"
    MATCHED_EVALUATION_PRIMARY = 'MATCHED_EVALUATION_PRIMARY'
    MATCHED_EVALUATION_RESERVE = 'MATCHED_EVALUATION_RESERVE'
    ROUTE_QUALIFICATION = "ROUTE_QUALIFICATION"
    PROSPECTIVE_PRIMARY = "PROSPECTIVE_PRIMARY"
    PROSPECTIVE_RESERVE = "PROSPECTIVE_RESERVE"


class GymToraxEpisodeBranchRole(StrEnum):
    FINITE_ACTION_WORD = "FINITE_ACTION_WORD"
    EFFICACY_ACTIVE = "EFFICACY_ACTIVE"
    EFFICACY_HOLD = "EFFICACY_HOLD"
    HOLD_CONTROL = "HOLD_CONTROL"
    ROUTE_HOLD_REPEAT = "ROUTE_HOLD_REPEAT"


class GymToraxCollisionKind(StrEnum):
    HISTORICAL_PREPARATION = "HISTORICAL_PREPARATION"
    NEW_PREPARATION = "NEW_PREPARATION"
    HISTORICAL_CELL_ID = "HISTORICAL_CELL_ID"
    NEW_CELL_ID = "NEW_CELL_ID"
    HISTORICAL_ENVIRONMENT_SEED = "HISTORICAL_ENVIRONMENT_SEED"
    NEW_ENVIRONMENT_SEED = "NEW_ENVIRONMENT_SEED"
    QUANTIZED_INTERVAL = "QUANTIZED_INTERVAL"


@dataclass(frozen=True, slots=True)
class GymToraxExposedSeedCollisionDomain(CanonicalRecord):
    """Authenticated compact collision namespace supplied before generation."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/tokamak-control-replication/gym-torax-exposed-seed-collision-domain'

    domain_id: str
    source_closure_sha256: str
    preparation_fingerprints: tuple[str, ...]
    exposed_cell_identity_sha256: tuple[str, ...]
    environment_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.domain_id, field_name="domain_id")
        validate_sha256(self.source_closure_sha256, field_name="source_closure_sha256")
        require_sorted_unique_strings(
            self.preparation_fingerprints,
            field_name="preparation_fingerprints",
        )
        for value in self.preparation_fingerprints:
            validate_sha256(value, field_name="preparation_fingerprint")
        require_sorted_unique_strings(self.exposed_cell_identity_sha256, field_name="exposed_cell_identity_sha256")
        for value in self.exposed_cell_identity_sha256:
            validate_sha256(value, field_name="exposed_cell_identity_sha256")
        if self.environment_seeds != tuple(sorted(set(self.environment_seeds))):
            raise ValueError("historical environment seeds must be sorted and unique")
        # Earlier recorded environment seeds used all eight SHA-256 bytes or
        # a 32-bit prefix. Retain that complete historical uint64 namespace;
        # only newly generated tokamak replication seeds are constrained
        # to the positive int63 domain.
        if any(value < 0 or value >= 2**64 for value in self.environment_seeds):
            raise ValueError("historical environment seed is outside the uint64 domain")


def _preparation_values(
    *,
    temperature: Decimal,
    density: Decimal,
    bootstrap: Decimal,
    transport: Decimal,
) -> tuple[NamedDecimal, ...]:
    return tuple(
        sorted(
            (
                NamedDecimal(
                    value_id='tokamak-control.preparation.bootstrap-multiplier',
                    value=bootstrap,
                    unit="1",
                ),
                NamedDecimal(
                    value_id='tokamak-control.preparation.initial-density-nbar',
                    value=density,
                    unit="1",
                ),
                NamedDecimal(
                    value_id='tokamak-control.preparation.initial-temperature-scale',
                    value=temperature,
                    unit="1",
                ),
                NamedDecimal(
                    value_id='tokamak-control.preparation.inner-transport-scale',
                    value=transport,
                    unit="1",
                ),
            ),
            key=lambda value: value.value_id,
        )
    )


@dataclass(frozen=True, slots=True)
class GymToraxRosterCell(CanonicalRecord):
    "One fresh preparation cell in the exact maximal tokamak-control replication roster."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/tokamak-control-replication/gym-torax-roster-cell'

    cell_id: str
    physical_unit_instance_id: str
    block_id: str
    roster_domain: str
    candidate_counter: int
    tier: GymToraxRosterTier
    quartet_role: GymToraxQuartetRole | None
    cell_kind: GymToraxCellKind
    prospective_role: GymToraxProspectiveRole | None
    stratum: GymToraxProspectiveStratum
    reserve: bool
    source_anchor_slot_id: str | None
    preparation_values: tuple[NamedDecimal, ...]
    preparation_fingerprint: str
    source_preparation_fingerprint: str
    source_cell_identity_sha256: str

    def __post_init__(self) -> None:
        for field_name, value in (
            ("cell_id", self.cell_id),
            ("physical_unit_instance_id", self.physical_unit_instance_id),
            ("block_id", self.block_id),
        ):
            validate_stable_id(value, field_name=field_name)
        if self.roster_domain not in _ROSTER_DOMAINS:
            raise ValueError("roster cell uses an unregistered deterministic domain")
        if self.candidate_counter < 0:
            raise ValueError("roster candidate counter must be nonnegative")
        require_sorted_unique_ids(
            self.preparation_values,
            attribute="value_id",
            field_name="preparation_values",
        )
        if len(self.preparation_values) != 4:
            raise ValueError("Gym--TORAX preparation cell requires exactly four coordinates")
        if {value.value_id for value in self.preparation_values} != _PREPARATION_VALUE_IDS or any(
            value.unit != "1" for value in self.preparation_values
        ):
            raise ValueError("Gym--TORAX preparation coordinate identities or units differ")
        validate_sha256(self.source_preparation_fingerprint, field_name="source_preparation_fingerprint")
        validate_sha256(self.source_cell_identity_sha256, field_name="source_cell_identity_sha256")
        validate_sha256(self.preparation_fingerprint, field_name="preparation_fingerprint")
        if self.preparation_fingerprint != preparation_values_fingerprint(self.preparation_values):
            raise ValueError("roster cell fingerprint differs from exact preparation values")
        quartet = self.quartet_role is not None
        prospective = self.prospective_role is not None
        if quartet == prospective:
            raise ValueError("roster cell must be exactly one of quartet or prospective")
        if quartet:
            if (
                self.stratum is not GymToraxProspectiveStratum.NONE
                or self.source_anchor_slot_id is not None
            ):
                raise ValueError("quartet cell cannot carry prospective metadata")
            expected_reserve = self.quartet_role is GymToraxQuartetRole.EVALUATION_RESERVE
        else:
            role = self.prospective_role
            if role is None:  # pragma: no cover - narrowed above
                raise AssertionError("prospective role disappeared")
            expected_reserve = role in {
                GymToraxProspectiveRole.EFFICACY_RESERVE,
                GymToraxProspectiveRole.HOLD_RESERVE,
            }
            efficacy = role in {
                GymToraxProspectiveRole.EFFICACY_PRIMARY,
                GymToraxProspectiveRole.EFFICACY_RESERVE,
                GymToraxProspectiveRole.ROUTE_ACTIVE_CANARY,
            }
            if efficacy != (self.stratum is not GymToraxProspectiveStratum.NONE):
                raise ValueError("prospective efficacy role/stratum differ")
            calibrated_hold = role in {
                GymToraxProspectiveRole.HOLD_CONTROL,
                GymToraxProspectiveRole.HOLD_RESERVE,
            }
            if calibrated_hold != (self.source_anchor_slot_id is not None):
                raise ValueError("prospective HOLD calibration role/slot differ")
            if not efficacy and self.cell_kind is not GymToraxCellKind.C:
                raise ValueError("prospective HOLD cells must remain C coordinates")
            if efficacy and self.cell_kind is not GymToraxCellKind.BT:
                raise ValueError("prospective efficacy cells must remain BT coordinates")
        if self.reserve != expected_reserve:
            raise ValueError("roster cell reserve flag differs from its exact role")
        values = _coordinates_by_id(self.preparation_values)
        temperature = values['tokamak-control.preparation.initial-temperature-scale']
        density = values['tokamak-control.preparation.initial-density-nbar']
        bootstrap = values['tokamak-control.preparation.bootstrap-multiplier']
        transport = values['tokamak-control.preparation.inner-transport-scale']
        if not (
            Decimal("0.848") <= density <= Decimal("0.852")
            and Decimal("0.990") <= transport <= Decimal("1.010")
        ):
            raise ValueError("Gym--TORAX density/transport coordinate is outside core")
        core_temperature = Decimal("0.990") <= temperature <= Decimal("1.010")
        core_bootstrap = Decimal("0.995") <= bootstrap <= Decimal("1.005")
        face_temperature = Decimal("1.010") < temperature <= Decimal("1.020")
        face_bootstrap = Decimal("0.990") <= bootstrap < Decimal("0.995")
        expected_axes = {
            GymToraxCellKind.C: (core_temperature, core_bootstrap),
            GymToraxCellKind.B: (core_temperature, face_bootstrap),
            GymToraxCellKind.T: (face_temperature, core_bootstrap),
            GymToraxCellKind.BT: (face_temperature, face_bootstrap),
        }[self.cell_kind]
        if not all(expected_axes):
            raise ValueError("Gym--TORAX preparation cell is outside its exact chart role")
        if self.cell_kind is GymToraxCellKind.BT and not _joint_tier_valid(
            self.preparation_values,
            self.tier,
        ):
            raise ValueError("joint preparation cell has ambiguous or wrong tier ownership")

    def coordinate(self, value_id: str) -> Decimal:
        return next(value.value for value in self.preparation_values if value.value_id == value_id)


@dataclass(frozen=True, slots=True)
class GymToraxMatchedQuartet(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/tokamak-control-replication/gym-torax-matched-quartet'

    quartet_id: str
    roster_domain: str
    candidate_counter: int
    tier: GymToraxRosterTier
    role: GymToraxQuartetRole
    cells: tuple[GymToraxRosterCell, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.quartet_id, field_name="quartet_id")
        require_sorted_unique_ids(self.cells, attribute="cell_id", field_name="cells")
        if len(self.cells) != 4 or {value.cell_kind for value in self.cells} != set(
            GymToraxCellKind
        ):
            raise ValueError("matched quartet requires exact C/B/T/BT cells")
        if any(
            value.block_id != self.quartet_id
            or value.roster_domain != self.roster_domain
            or value.candidate_counter != self.candidate_counter
            or value.tier is not self.tier
            or value.quartet_role is not self.role
            for value in self.cells
        ):
            raise ValueError("matched quartet cell metadata differs")
        by_kind = {value.cell_kind: value for value in self.cells}
        density_id = 'tokamak-control.preparation.initial-density-nbar'
        transport_id = 'tokamak-control.preparation.inner-transport-scale'
        if (
            len({value.coordinate(density_id) for value in self.cells}) != 1
            or len({value.coordinate(transport_id) for value in self.cells}) != 1
        ):
            raise ValueError("matched quartet density/transport coordinates differ")
        temperature_id = 'tokamak-control.preparation.initial-temperature-scale'
        bootstrap_id = 'tokamak-control.preparation.bootstrap-multiplier'
        if not (
            by_kind[GymToraxCellKind.C].coordinate(temperature_id)
            == by_kind[GymToraxCellKind.B].coordinate(temperature_id)
            and by_kind[GymToraxCellKind.T].coordinate(temperature_id)
            == by_kind[GymToraxCellKind.BT].coordinate(temperature_id)
            and by_kind[GymToraxCellKind.C].coordinate(bootstrap_id)
            == by_kind[GymToraxCellKind.T].coordinate(bootstrap_id)
            and by_kind[GymToraxCellKind.B].coordinate(bootstrap_id)
            == by_kind[GymToraxCellKind.BT].coordinate(bootstrap_id)
        ):
            raise ValueError("matched quartet does not preserve its factorization")


@dataclass(frozen=True, slots=True)
class GymToraxEpisodeCoordinate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/tokamak-control-replication/gym-torax-episode-coordinate'

    coordinate_id: str
    stage: GymToraxEpisodeStage
    cell_id: str
    physical_unit_instance_id: str
    model_member_id: str
    branch_role: GymToraxEpisodeBranchRole
    action_word_id: str
    reserve: bool
    environment_seed: int

    def __post_init__(self) -> None:
        for field_name, value in (
            ("coordinate_id", self.coordinate_id),
            ("cell_id", self.cell_id),
            ("physical_unit_instance_id", self.physical_unit_instance_id),
            ("model_member_id", self.model_member_id),
            ("action_word_id", self.action_word_id),
        ):
            validate_stable_id(value, field_name=field_name)
        if self.model_member_id not in _MODEL_MEMBER_IDS:
            raise ValueError("episode coordinate uses another numerical member")
        if self.action_word_id not in GYM_TORAX_ACTION_WORD_IDS:
            raise ValueError("episode coordinate uses another current action word")
        if self.environment_seed <= 0 or self.environment_seed >= 2**63:
            raise ValueError("episode environment seed is outside positive int63")


@dataclass(frozen=True, slots=True)
class GymToraxRosterCollision(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/tokamak-control-replication/gym-torax-roster-collision'

    collision_id: str
    roster_domain: str
    candidate_counter: int
    kind: GymToraxCollisionKind
    collided_value_sha256: str

    def __post_init__(self) -> None:
        validate_stable_id(self.collision_id, field_name="collision_id")
        if self.roster_domain not in _ROSTER_DOMAINS:
            raise ValueError("collision record uses an unregistered roster domain")
        if self.candidate_counter < 0:
            raise ValueError("collision candidate counter must be nonnegative")
        validate_sha256(self.collided_value_sha256, field_name="collided_value_sha256")


@dataclass(frozen=True, slots=True)
class GymToraxRosterCollisionAudit(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/tokamak-control-replication/gym-torax-roster-collision-audit'

    audit_id: str
    historical_domain: GymToraxExposedSeedCollisionDomain
    checked_namespace_ids: tuple[str, ...]
    rejected_candidates: tuple[GymToraxRosterCollision, ...]
    collision_free: bool
    outcome_access: OutcomeAccess = OutcomeAccess.OUTCOME_BLIND

    def __post_init__(self) -> None:
        validate_stable_id(self.audit_id, field_name="audit_id")
        require_sorted_unique_strings(
            self.checked_namespace_ids,
            field_name="checked_namespace_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(
            self.rejected_candidates,
            attribute="collision_id",
            field_name="rejected_candidates",
        )
        if not self.collision_free:
            raise ValueError("a frozen tokamak-control replication roster must be collision-free after rejection")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("roster collision audit cannot read outcomes")


@dataclass(frozen=True, slots=True)
class GymToraxRoster(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/tokamak-control-replication/gym-torax-roster'

    roster_id: str
    generator_id: str
    historical_collision_domain: GymToraxExposedSeedCollisionDomain
    qualification_quartets: tuple[GymToraxMatchedQuartet, ...]
    evaluation_primary_quartets: tuple[GymToraxMatchedQuartet, ...]
    evaluation_reserve_quartets: tuple[GymToraxMatchedQuartet, ...]
    prospective_cells: tuple[GymToraxRosterCell, ...]
    anchor_slots: tuple[NativeHoldCalibrationAnchorSlot, ...]
    episode_coordinates: tuple[GymToraxEpisodeCoordinate, ...]
    collision_audit: GymToraxRosterCollisionAudit
    scientific_inputs: GymToraxRosterScientificInputs
    primary_episode_count: int
    maximum_episode_count: int
    maximum_operational_launches: int
    outcome_access: OutcomeAccess = OutcomeAccess.OUTCOME_BLIND
    frozen: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.scientific_inputs, GymToraxRosterScientificInputs):
            raise ValueError("roster requires its typed scientific input census")
        validate_stable_id(self.roster_id, field_name="roster_id")
        validate_stable_id(self.generator_id, field_name="generator_id")
        for name, values, expected in (
            ("qualification_quartets", self.qualification_quartets, 3),
            ("evaluation_primary_quartets", self.evaluation_primary_quartets, 18),
            ("evaluation_reserve_quartets", self.evaluation_reserve_quartets, 3),
        ):
            require_sorted_unique_ids(values, attribute="quartet_id", field_name=name)
            if len(values) != expected:
                raise ValueError(f"{name} differs from its frozen count")
        tier_counts = {
            tier: sum(value.tier is tier for value in self.evaluation_primary_quartets)
            for tier in GymToraxRosterTier
        }
        if set(tier_counts.values()) != {6}:
            raise ValueError("evaluation roster must contain six quartets per tier")
        if any(
            sum(value.tier is tier for value in self.qualification_quartets) != 1
            or sum(value.tier is tier for value in self.evaluation_reserve_quartets) != 1
            for tier in GymToraxRosterTier
        ):
            raise ValueError("qualification/reserve rosters require one quartet per tier")
        require_sorted_unique_ids(
            self.prospective_cells,
            attribute="cell_id",
            field_name="prospective_cells",
        )
        role_counts = {
            role: sum(value.prospective_role is role for value in self.prospective_cells)
            for role in GymToraxProspectiveRole
        }
        if role_counts != {
            GymToraxProspectiveRole.EFFICACY_PRIMARY: 18,
            GymToraxProspectiveRole.EFFICACY_RESERVE: 2,
            GymToraxProspectiveRole.HOLD_CONTROL: 4,
            GymToraxProspectiveRole.HOLD_RESERVE: 1,
            GymToraxProspectiveRole.ROUTE_ACTIVE_CANARY: 1,
            GymToraxProspectiveRole.ROUTE_HOLD_CANARY: 1,
        }:
            raise ValueError("prospective roster differs from exact 18+4/reserve/canary design")
        require_sorted_unique_ids(self.anchor_slots, attribute="slot_id", field_name="anchor_slots")
        if len(self.anchor_slots) != 5:
            raise ValueError("native-HOLD roster requires five exact anchor slots")
        slots = {value.slot_id: value for value in self.anchor_slots}
        calibrated_holds = tuple(
            value for value in self.prospective_cells if value.source_anchor_slot_id is not None
        )
        if len(calibrated_holds) != 5:
            raise ValueError("native-HOLD roster must contain five fresh non-aliased cells")
        for cell in calibrated_holds:
            slot_id = cell.source_anchor_slot_id
            if slot_id is None:  # pragma: no cover - narrowed above
                raise AssertionError("native-HOLD cell lost its anchor slot")
            slot = slots.get(slot_id)
            if (
                slot is None
                or cell.cell_id != slot.primary.coordinate_id
                or cell.preparation_values != slot.primary.preparation_values
                or cell.preparation_fingerprint != slot.primary.preparation_fingerprint
            ):
                raise ValueError("native-HOLD cell and calibration slot differ")
        require_sorted_unique_ids(
            self.episode_coordinates,
            attribute="coordinate_id",
            field_name="episode_coordinates",
        )
        episodes_by_id = {episode.coordinate_id: episode for episode in self.episode_coordinates}
        cells = (
            *(cell for quartet in self.qualification_quartets for cell in quartet.cells),
            *(cell for quartet in self.evaluation_primary_quartets for cell in quartet.cells),
            *(cell for quartet in self.evaluation_reserve_quartets for cell in quartet.cells),
            *self.prospective_cells,
        )
        for cell in cells:
            preparation_input = self.scientific_inputs.preparation_input(_scientific_preparation_coordinates(cell.preparation_values))
            cell_input = self.scientific_inputs.cell_identity_input(_SCIENTIFIC_ROSTER_ROLES[cell.roster_domain], cell.candidate_counter, _CELL_SUFFIX[cell.cell_kind.value])
            if (
                cell.source_preparation_fingerprint != preparation_input.source_preparation_fingerprint
                or cell.source_cell_identity_sha256 != cell_input.source_cell_identity_sha256
            ):
                raise ValueError("roster cell changes its original scientific custody inputs")
            for plan in _episode_plans_for_cell(cell):
                episode = episodes_by_id.get(_episode_coordinate_id(cell, plan))
                if episode is None or episode.environment_seed != _environment_seed(
                    cell, plan, self.scientific_inputs
                ):
                    raise ValueError("roster episode differs from its declared scientific seed input")
        if self.primary_episode_count != 760 or self.maximum_episode_count != 866:
            raise ValueError("tokamak-control replication scientific episode arithmetic differs from 760/866")
        if len(self.episode_coordinates) != self.maximum_episode_count:
            raise ValueError("maximal tokamak-control replication episode roster differs from 866 coordinates")
        if self.maximum_operational_launches != 2 * self.maximum_episode_count:
            raise ValueError("tokamak-control replication operational launch ceiling differs from two attempts")
        if self.collision_audit.historical_domain != self.historical_collision_domain:
            raise ValueError("roster and collision audit use another historical domain")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND or not self.frozen:
            raise ValueError("tokamak-control replication roster must be frozen and outcome-blind")


_ROSTER_DOMAINS = frozenset(
    {
        GYM_TORAX_QUALIFICATION_ROSTER_DOMAIN,
        GYM_TORAX_EVALUATION_PRIMARY_ROSTER_DOMAIN,
        GYM_TORAX_EVALUATION_RESERVE_ROSTER_DOMAIN,
        GYM_TORAX_PROSPECTIVE_EFFICACY_ROSTER_DOMAIN,
        GYM_TORAX_PROSPECTIVE_HOLD_ROSTER_DOMAIN,
        GYM_TORAX_ROUTE_ACTIVE_ROSTER_DOMAIN,
        GYM_TORAX_ROUTE_HOLD_ROSTER_DOMAIN,
    }
)


_SCIENTIFIC_ROSTER_ROLES = {
    GYM_TORAX_QUALIFICATION_ROSTER_DOMAIN: "source-qualification",
    GYM_TORAX_EVALUATION_PRIMARY_ROSTER_DOMAIN: "matched-evaluation-primary",
    GYM_TORAX_EVALUATION_RESERVE_ROSTER_DOMAIN: "matched-evaluation-reserve",
    GYM_TORAX_PROSPECTIVE_EFFICACY_ROSTER_DOMAIN: "prospective-efficacy",
    GYM_TORAX_PROSPECTIVE_HOLD_ROSTER_DOMAIN: "prospective-hold",
    GYM_TORAX_ROUTE_ACTIVE_ROSTER_DOMAIN: "active-route-canary",
    GYM_TORAX_ROUTE_HOLD_ROSTER_DOMAIN: "hold-route-canary",
}
_SCIENTIFIC_STAGE_ROLES = {
    GymToraxEpisodeStage.EXCLUDED_QUALIFICATION: "excluded-source-qualification",
    GymToraxEpisodeStage.MATCHED_EVALUATION_PRIMARY: "primary-matched-evaluation",
    GymToraxEpisodeStage.MATCHED_EVALUATION_RESERVE: "reserve-matched-evaluation",
    GymToraxEpisodeStage.ROUTE_QUALIFICATION: "excluded-route-qualification",
    GymToraxEpisodeStage.PROSPECTIVE_PRIMARY: "prospective-primary",
    GymToraxEpisodeStage.PROSPECTIVE_RESERVE: "prospective-reserve",
}
_SCIENTIFIC_MEMBER_ROLES = {
    GYM_TORAX_PRIMARY_MEMBER_ID: "primary",
    GYM_TORAX_REFINED_MEMBER_ID: "refined",
}


def _quantize(value: Decimal) -> Decimal:
    return value.quantize(_QUANTUM, rounding=ROUND_HALF_EVEN)


def _stream_words(
    domain: str, counter: int, scientific_inputs: GymToraxRosterScientificInputs
) -> Iterator[int]:
    row = scientific_inputs.stream_input(_SCIENTIFIC_ROSTER_ROLES[domain], counter)
    yield from row.words
    raise ValueError("roster candidate exhausted its explicit scientific word blocks")


def _unit_fraction(word: int) -> Decimal:
    return Decimal(word) / _TWO_64


def _map(word: int, lower: Decimal, upper: Decimal) -> Decimal:
    return _quantize(lower + (upper - lower) * _unit_fraction(word))


def _tier_bounds(tier: GymToraxRosterTier) -> tuple[Decimal, Decimal]:
    index = tuple(GymToraxRosterTier).index(tier)
    return Decimal(index) / Decimal(3), Decimal(index + 1) / Decimal(3)


def _fingerprint_text(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _cell_id(domain_slug: str, counter: int, kind: GymToraxCellKind) -> str:
    return f'cell.tokamak-control.{domain_slug}.{counter:08d}.{_CELL_SUFFIX[kind.value]}'


def _cell_unit_id(domain_slug: str, counter: int, kind: GymToraxCellKind) -> str:
    # The base deterministic unit namespace is extended by cell kind because
    # every preparation cell, not the four-cell blocking quartet, is the measurement unit.
    return f'unit.tokamak-control.{domain_slug}.{counter:08d}.{_CELL_SUFFIX[kind.value]}'


def _coordinates_by_id(values: tuple[NamedDecimal, ...]) -> dict[str, Decimal]:
    return {value.value_id: value.value for value in values}


def _scientific_preparation_coordinates(values: tuple[NamedDecimal, ...]) -> tuple[Decimal, Decimal, Decimal, Decimal]:
    coordinates = _coordinates_by_id(values)
    return (
        coordinates['tokamak-control.preparation.initial-temperature-scale'],
        coordinates['tokamak-control.preparation.initial-density-nbar'],
        coordinates['tokamak-control.preparation.bootstrap-multiplier'],
        coordinates['tokamak-control.preparation.inner-transport-scale'],
    )


@dataclass(frozen=True, slots=True)
class _DepthInterval:
    lower: Decimal
    upper: Decimal
    lower_inclusive: bool
    upper_inclusive: bool


def _round_half_even_preimage(value: Decimal) -> tuple[Decimal, Decimal, bool]:
    integer = int(value / _QUANTUM)
    midpoint_inclusive = integer % 2 == 0
    half = _QUANTUM / Decimal(2)
    return value - half, value + half, midpoint_inclusive


def _temperature_depth_interval(value: Decimal) -> _DepthInterval:
    lower, upper, endpoints_inclusive = _round_half_even_preimage(value)
    return _DepthInterval(
        lower=(lower - Decimal("1.010")) / Decimal("0.010"),
        upper=(upper - Decimal("1.010")) / Decimal("0.010"),
        lower_inclusive=endpoints_inclusive,
        upper_inclusive=endpoints_inclusive,
    )


def _bootstrap_depth_interval(value: Decimal) -> _DepthInterval:
    lower, upper, endpoints_inclusive = _round_half_even_preimage(value)
    return _DepthInterval(
        lower=(Decimal("0.995") - upper) / Decimal("0.005"),
        upper=(Decimal("0.995") - lower) / Decimal("0.005"),
        lower_inclusive=endpoints_inclusive,
        upper_inclusive=endpoints_inclusive,
    )


def _intersect_depths(
    left: _DepthInterval,
    right: _DepthInterval,
) -> _DepthInterval | None:
    lower = max(left.lower, right.lower)
    upper = min(left.upper, right.upper)
    lower_inclusive = (left.lower != lower or left.lower_inclusive) and (
        right.lower != lower or right.lower_inclusive
    )
    upper_inclusive = (left.upper != upper or left.upper_inclusive) and (
        right.upper != upper or right.upper_inclusive
    )
    if lower > upper or (lower == upper and not (lower_inclusive and upper_inclusive)):
        return None
    return _DepthInterval(lower, upper, lower_inclusive, upper_inclusive)


def _owned_tier(interval: _DepthInterval) -> GymToraxRosterTier | None:
    owners = []
    for tier in GymToraxRosterTier:
        lower, upper = _tier_bounds(tier)
        lower_contained = interval.lower > lower or (
            interval.lower == lower and not interval.lower_inclusive
        )
        if lower_contained and interval.upper <= upper:
            owners.append(tier)
    return owners[0] if len(owners) == 1 else None


def _joint_tier_valid(
    values: tuple[NamedDecimal, ...],
    tier: GymToraxRosterTier,
) -> bool:
    by_id = _coordinates_by_id(values)
    temperature = by_id['tokamak-control.preparation.initial-temperature-scale']
    bootstrap = by_id['tokamak-control.preparation.bootstrap-multiplier']
    intersection = _intersect_depths(
        _temperature_depth_interval(temperature),
        _bootstrap_depth_interval(bootstrap),
    )
    return intersection is not None and _owned_tier(intersection) is tier


def _quartet_candidate(
    *,
    domain: str,
    scientific_inputs: GymToraxRosterScientificInputs,
    domain_slug: str,
    counter: int,
    tier: GymToraxRosterTier,
    role: GymToraxQuartetRole,
) -> GymToraxMatchedQuartet | None:
    words = _stream_words(domain, counter, scientific_inputs)
    core_temperature = _map(next(words), Decimal("0.990"), Decimal("1.010"))
    core_bootstrap = _map(next(words), Decimal("0.995"), Decimal("1.005"))
    density = _map(next(words), Decimal("0.848"), Decimal("0.852"))
    transport = _map(next(words), Decimal("0.990"), Decimal("1.010"))
    lower, upper = _tier_bounds(tier)
    depth = lower + (upper - lower) * _unit_fraction(next(words))
    bootstrap = _quantize(Decimal("0.995") - Decimal("0.005") * depth)
    temperature = _quantize(Decimal("1.010") + Decimal("0.010") * depth)
    quartet_id = f'quartet.tokamak-control.{domain_slug}.{counter:08d}'
    coordinates = {
        GymToraxCellKind.C: (core_temperature, core_bootstrap),
        GymToraxCellKind.B: (core_temperature, bootstrap),
        GymToraxCellKind.T: (temperature, core_bootstrap),
        GymToraxCellKind.BT: (temperature, bootstrap),
    }
    cells = tuple(
        sorted(
            (
                GymToraxRosterCell(
                    cell_id=_cell_id(domain_slug, counter, kind),
                    physical_unit_instance_id=_cell_unit_id(domain_slug, counter, kind),
                    block_id=quartet_id,
                    roster_domain=domain,
                    candidate_counter=counter,
                    tier=tier,
                    quartet_role=role,
                    cell_kind=kind,
                    prospective_role=None,
                    stratum=GymToraxProspectiveStratum.NONE,
                    reserve=role is GymToraxQuartetRole.EVALUATION_RESERVE,
                    source_anchor_slot_id=None,
                    preparation_values=(
                        values := _preparation_values(
                            temperature=coordinates[kind][0],
                            density=density,
                            bootstrap=coordinates[kind][1],
                            transport=transport,
                        )
                    ),
                    preparation_fingerprint=preparation_values_fingerprint(values),
                    source_preparation_fingerprint=scientific_inputs.preparation_input(_scientific_preparation_coordinates(values)).source_preparation_fingerprint,
                    source_cell_identity_sha256=scientific_inputs.cell_identity_input(_SCIENTIFIC_ROSTER_ROLES[domain], counter, _CELL_SUFFIX[kind.value]).source_cell_identity_sha256,
                )
                for kind in GymToraxCellKind
            ),
            key=lambda value: value.cell_id,
        )
    )
    joint = next(value for value in cells if value.cell_kind is GymToraxCellKind.BT)
    if not _joint_tier_valid(joint.preparation_values, tier):
        return None
    return GymToraxMatchedQuartet(
        quartet_id=quartet_id,
        roster_domain=domain,
        candidate_counter=counter,
        tier=tier,
        role=role,
        cells=cells,
    )


def _prospective_candidate(
    *,
    domain: str,
    scientific_inputs: GymToraxRosterScientificInputs,
    domain_slug: str,
    counter: int,
    role: GymToraxProspectiveRole,
    stratum: GymToraxProspectiveStratum,
) -> GymToraxRosterCell | None:
    words = _stream_words(domain, counter, scientific_inputs)
    lower, upper = (
        (Decimal(0), Decimal("0.5"))
        if stratum is GymToraxProspectiveStratum.INTERFACE
        else (Decimal("0.5"), Decimal(1))
    )
    depth = lower + (upper - lower) * _unit_fraction(next(words))
    if not lower < depth <= upper:
        return None
    bootstrap = _quantize(Decimal("0.995") - Decimal("0.005") * depth)
    temperature = _quantize(Decimal("1.010") + Decimal("0.010") * depth)
    density = _map(next(words), Decimal("0.848"), Decimal("0.852"))
    transport = _map(next(words), Decimal("0.990"), Decimal("1.010"))
    tier = next(
        candidate
        for candidate in GymToraxRosterTier
        if _tier_bounds(candidate)[0] < depth <= _tier_bounds(candidate)[1]
    )
    values = _preparation_values(
        temperature=temperature,
        density=density,
        bootstrap=bootstrap,
        transport=transport,
    )
    if not _joint_tier_valid(values, tier):
        return None
    cell_id = _cell_id(domain_slug, counter, GymToraxCellKind.BT)
    return GymToraxRosterCell(
        cell_id=cell_id,
        physical_unit_instance_id=_cell_unit_id(domain_slug, counter, GymToraxCellKind.BT),
        block_id=f'block.tokamak-control.{domain_slug}.{counter:08d}',
        roster_domain=domain,
        candidate_counter=counter,
        tier=tier,
        quartet_role=None,
        cell_kind=GymToraxCellKind.BT,
        prospective_role=role,
        stratum=stratum,
        reserve=role is GymToraxProspectiveRole.EFFICACY_RESERVE,
        source_anchor_slot_id=None,
        preparation_values=values,
        preparation_fingerprint=preparation_values_fingerprint(values),
        source_preparation_fingerprint=scientific_inputs.preparation_input(_scientific_preparation_coordinates(values)).source_preparation_fingerprint,
        source_cell_identity_sha256=scientific_inputs.cell_identity_input(_SCIENTIFIC_ROSTER_ROLES[domain], counter, "bt").source_cell_identity_sha256,
    )


def _anchor_option(
    cell: GymToraxRosterCell, *, slot: int, kind: str
) -> NativeHoldCalibrationAnchorOption:
    return NativeHoldCalibrationAnchorOption(
        option_id=f'anchor-option.tokamak-control.c-hold-{slot:02d}.{kind}',
        coordinate_id=cell.cell_id,
        preparation_values=cell.preparation_values,
        preparation_fingerprint=cell.preparation_fingerprint,
    )


def _hold_candidate(
    *,
    domain: str,
    scientific_inputs: GymToraxRosterScientificInputs,
    domain_slug: str,
    counter: int,
    role: GymToraxProspectiveRole,
    anchor_slot_id: str | None,
) -> GymToraxRosterCell | None:
    if role not in {
        GymToraxProspectiveRole.HOLD_CONTROL,
        GymToraxProspectiveRole.HOLD_RESERVE,
        GymToraxProspectiveRole.ROUTE_HOLD_CANARY,
    }:
        raise ValueError("fresh HOLD candidate requires an exact HOLD role")
    words = _stream_words(domain, counter, scientific_inputs)
    depth = Decimal("0.5") + Decimal("0.5") * _unit_fraction(next(words))
    if not Decimal("0.5") < depth <= Decimal(1):
        return None
    core_bootstrap = _quantize(Decimal("0.995") + Decimal("0.005") * depth)
    core_temperature = _quantize(Decimal("1.010") - Decimal("0.010") * depth)
    density = _map(next(words), Decimal("0.848"), Decimal("0.852"))
    transport = _map(next(words), Decimal("0.990"), Decimal("1.010"))
    values = _preparation_values(
        temperature=core_temperature,
        density=density,
        bootstrap=core_bootstrap,
        transport=transport,
    )
    tier = next(
        candidate
        for candidate in GymToraxRosterTier
        if _tier_bounds(candidate)[0] < depth <= _tier_bounds(candidate)[1]
    )
    return GymToraxRosterCell(
        cell_id=f'cell.tokamak-control.{domain_slug}.{counter:08d}.c',
        physical_unit_instance_id=f'unit.tokamak-control.{domain_slug}.{counter:08d}.c',
        block_id=f'block.tokamak-control.{domain_slug}.{counter:08d}',
        roster_domain=domain,
        candidate_counter=counter,
        tier=tier,
        quartet_role=None,
        cell_kind=GymToraxCellKind.C,
        prospective_role=role,
        stratum=GymToraxProspectiveStratum.NONE,
        reserve=role is GymToraxProspectiveRole.HOLD_RESERVE,
        source_anchor_slot_id=anchor_slot_id,
        preparation_values=values,
        preparation_fingerprint=preparation_values_fingerprint(values),
        source_preparation_fingerprint=scientific_inputs.preparation_input(_scientific_preparation_coordinates(values)).source_preparation_fingerprint,
        source_cell_identity_sha256=scientific_inputs.cell_identity_input(_SCIENTIFIC_ROSTER_ROLES[domain], counter, "c").source_cell_identity_sha256,
    )


def _environment_seed(
    cell: GymToraxRosterCell,
    plan: _EpisodePlan,
    scientific_inputs: GymToraxRosterScientificInputs,
) -> int:
    key = (
        _SCIENTIFIC_STAGE_ROLES[plan.stage],
        _SCIENTIFIC_ROSTER_ROLES[cell.roster_domain],
        cell.candidate_counter,
        _CELL_SUFFIX[cell.cell_kind.value],
        _SCIENTIFIC_MEMBER_ROLES[plan.member_id],
        plan.role_slug,
    )
    return scientific_inputs.environment_seed_input(key).environment_seed


@dataclass(frozen=True, slots=True)
class _EpisodePlan:
    stage: GymToraxEpisodeStage
    member_id: str
    branch_role: GymToraxEpisodeBranchRole
    action_word_id: str
    role_slug: str
    reserve: bool


def _episode_plans_for_cell(cell: GymToraxRosterCell) -> tuple[_EpisodePlan, ...]:
    plans: list[_EpisodePlan] = []
    if cell.quartet_role is not None:
        stage = {
            GymToraxQuartetRole.QUALIFICATION: (GymToraxEpisodeStage.EXCLUDED_QUALIFICATION),
            GymToraxQuartetRole.EVALUATION_PRIMARY: GymToraxEpisodeStage.MATCHED_EVALUATION_PRIMARY,
            GymToraxQuartetRole.EVALUATION_RESERVE: GymToraxEpisodeStage.MATCHED_EVALUATION_RESERVE,
        }[cell.quartet_role]
        for member_id in _MODEL_MEMBER_IDS:
            for word_id in GYM_TORAX_ACTION_WORD_IDS:
                plans.append(
                    _EpisodePlan(
                        stage=stage,
                        member_id=member_id,
                        branch_role=GymToraxEpisodeBranchRole.FINITE_ACTION_WORD,
                        action_word_id=word_id,
                        role_slug=word_id.removeprefix('action-word.tokamak-control.'),
                        reserve=cell.reserve,
                    )
                )
        return tuple(plans)
    role = cell.prospective_role
    if role is None:  # pragma: no cover - cell invariant narrows this branch
        raise AssertionError("prospective cell role disappeared")
    if role in {
        GymToraxProspectiveRole.EFFICACY_PRIMARY,
        GymToraxProspectiveRole.EFFICACY_RESERVE,
        GymToraxProspectiveRole.ROUTE_ACTIVE_CANARY,
    }:
        stage = (
            GymToraxEpisodeStage.ROUTE_QUALIFICATION
            if role is GymToraxProspectiveRole.ROUTE_ACTIVE_CANARY
            else GymToraxEpisodeStage.PROSPECTIVE_RESERVE
            if role is GymToraxProspectiveRole.EFFICACY_RESERVE
            else GymToraxEpisodeStage.PROSPECTIVE_PRIMARY
        )
        for member_id in _MODEL_MEMBER_IDS:
            plans.extend(
                (
                    _EpisodePlan(
                        stage=stage,
                        member_id=member_id,
                        branch_role=GymToraxEpisodeBranchRole.EFFICACY_ACTIVE,
                        action_word_id=GYM_TORAX_LOWER_IP_ACTION_WORD_ID,
                        role_slug="efficacy-active",
                        reserve=cell.reserve,
                    ),
                    _EpisodePlan(
                        stage=stage,
                        member_id=member_id,
                        branch_role=GymToraxEpisodeBranchRole.EFFICACY_HOLD,
                        action_word_id=GYM_TORAX_NATIVE_HOLD_ACTION_WORD_ID,
                        role_slug="efficacy-hold",
                        reserve=cell.reserve,
                    ),
                )
            )
        return tuple(plans)
    stage = (
        GymToraxEpisodeStage.ROUTE_QUALIFICATION
        if role is GymToraxProspectiveRole.ROUTE_HOLD_CANARY
        else GymToraxEpisodeStage.PROSPECTIVE_RESERVE
        if role is GymToraxProspectiveRole.HOLD_RESERVE
        else GymToraxEpisodeStage.PROSPECTIVE_PRIMARY
    )
    branch_specs = (
        (
            (GymToraxEpisodeBranchRole.HOLD_CONTROL, "hold-control"),
            (GymToraxEpisodeBranchRole.ROUTE_HOLD_REPEAT, "route-hold-repeat"),
        )
        if role is GymToraxProspectiveRole.ROUTE_HOLD_CANARY
        else ((GymToraxEpisodeBranchRole.HOLD_CONTROL, "hold-control"),)
    )
    return tuple(
        _EpisodePlan(
            stage=stage,
            member_id=member_id,
            branch_role=branch,
            action_word_id=GYM_TORAX_NATIVE_HOLD_ACTION_WORD_ID,
            role_slug=slug,
            reserve=cell.reserve,
        )
        for member_id in _MODEL_MEMBER_IDS
        for branch, slug in branch_specs
    )


def _episode_coordinate_id(cell: GymToraxRosterCell, plan: _EpisodePlan) -> str:
    return (
        f"episode-coordinate.tokamak-control.{plan.stage.value.lower().replace('_', '-')}"
        f".{cell.cell_id.removeprefix('cell.tokamak-control.')}"
        f".{plan.member_id.removeprefix('member.tokamak-control.')}.{plan.role_slug}"
    )


class GymToraxRosterGenerator:
    """Generate the exact maximal roster, rejecting collisions before freeze."""

    generator_id = GYM_TORAX_ROSTER_GENERATOR_ID

    @staticmethod
    def _collision(
        *,
        index: int,
        domain: str,
        counter: int,
        kind: GymToraxCollisionKind,
        value: object,
    ) -> GymToraxRosterCollision:
        return GymToraxRosterCollision(
            collision_id=f'collision.tokamak-control.{index:08d}',
            roster_domain=domain,
            candidate_counter=counter,
            kind=kind,
            collided_value_sha256=(
                value if kind in {GymToraxCollisionKind.HISTORICAL_CELL_ID, GymToraxCollisionKind.NEW_CELL_ID} and isinstance(value, str)
                else _fingerprint_text(value)
            ),
        )

    def generate(
        self,
        historical: GymToraxExposedSeedCollisionDomain,
        *,
        scientific_inputs: GymToraxRosterScientificInputs | None = None,
    ) -> GymToraxRoster:
        inputs = (
            gym_torax_default_roster_scientific_inputs()
            if scientific_inputs is None else scientific_inputs
        )
        if not isinstance(inputs, GymToraxRosterScientificInputs):
            raise ValueError("roster generation requires a typed scientific input census")
        collisions: list[GymToraxRosterCollision] = []
        historical_fingerprints = set(historical.preparation_fingerprints)
        historical_cell_ids = set(historical.exposed_cell_identity_sha256)
        known_fingerprints = set(historical_fingerprints)
        known_cell_ids = set(historical_cell_ids)
        historical_environment_seeds = set(historical.environment_seeds)
        known_environment_seeds = set(historical.environment_seeds)

        def accept_cells(
            cells: tuple[GymToraxRosterCell, ...],
        ) -> bool:
            local_fingerprints = [value.source_preparation_fingerprint for value in cells]
            local_ids = [value.source_cell_identity_sha256 for value in cells]
            local_seeds = [
                _environment_seed(cell, plan, inputs)
                for cell in cells
                for plan in _episode_plans_for_cell(cell)
            ]
            reasons: list[tuple[GymToraxCollisionKind, object]] = []
            if len(set(local_fingerprints)) != len(local_fingerprints):
                reasons.append((GymToraxCollisionKind.NEW_PREPARATION, local_fingerprints))
            if len(set(local_ids)) != len(local_ids):
                reasons.append((GymToraxCollisionKind.NEW_CELL_ID, local_ids))
            if len(set(local_seeds)) != len(local_seeds):
                reasons.append((GymToraxCollisionKind.NEW_ENVIRONMENT_SEED, local_seeds))
            for cell in cells:
                if cell.source_preparation_fingerprint in known_fingerprints:
                    reasons.append(
                        (
                            GymToraxCollisionKind.HISTORICAL_PREPARATION
                            if cell.source_preparation_fingerprint in historical_fingerprints
                            else GymToraxCollisionKind.NEW_PREPARATION,
                            cell.source_preparation_fingerprint,
                        )
                    )
                if cell.source_cell_identity_sha256 in known_cell_ids:
                    reasons.append(
                        (
                            GymToraxCollisionKind.HISTORICAL_CELL_ID
                            if cell.source_cell_identity_sha256 in historical_cell_ids
                            else GymToraxCollisionKind.NEW_CELL_ID,
                            cell.source_cell_identity_sha256,
                        )
                    )
            for seed in local_seeds:
                if seed in known_environment_seeds:
                    reasons.append(
                        (
                            GymToraxCollisionKind.HISTORICAL_ENVIRONMENT_SEED
                            if seed in historical_environment_seeds
                            else GymToraxCollisionKind.NEW_ENVIRONMENT_SEED,
                            seed,
                        )
                    )
            if reasons:
                for kind, value in reasons:
                    collisions.append(
                        self._collision(
                            index=len(collisions),
                            domain=cells[0].roster_domain,
                            counter=cells[0].candidate_counter,
                            kind=kind,
                            value=value,
                        )
                    )
                return False
            known_fingerprints.update(local_fingerprints)
            known_cell_ids.update(local_ids)
            known_environment_seeds.update(local_seeds)
            return True

        def quartets(
            *,
            domain: str,
            slug: str,
            role: GymToraxQuartetRole,
            per_tier: int,
        ) -> tuple[GymToraxMatchedQuartet, ...]:
            accepted: list[GymToraxMatchedQuartet] = []
            counter = 0
            for tier in GymToraxRosterTier:
                tier_values: list[GymToraxMatchedQuartet] = []
                while len(tier_values) < per_tier:
                    candidate = _quartet_candidate(
                        scientific_inputs=inputs,
                        domain=domain,
                        domain_slug=slug,
                        counter=counter,
                        tier=tier,
                        role=role,
                    )
                    if candidate is None:
                        collisions.append(
                            self._collision(
                                index=len(collisions),
                                domain=domain,
                                counter=counter,
                                kind=GymToraxCollisionKind.QUANTIZED_INTERVAL,
                                value=(tier.value, counter),
                            )
                        )
                    elif accept_cells(candidate.cells):
                        tier_values.append(candidate)
                    counter += 1
                accepted.extend(tier_values)
            return tuple(sorted(accepted, key=lambda value: value.quartet_id))

        qualification = quartets(
            domain=GYM_TORAX_QUALIFICATION_ROSTER_DOMAIN,
            slug='source-qualification',
            role=GymToraxQuartetRole.QUALIFICATION,
            per_tier=1,
        )
        evaluation_primary = quartets(
            domain=GYM_TORAX_EVALUATION_PRIMARY_ROSTER_DOMAIN,
            slug='matched-evaluation-primary',
            role=GymToraxQuartetRole.EVALUATION_PRIMARY,
            per_tier=6,
        )
        evaluation_reserve = quartets(
            domain=GYM_TORAX_EVALUATION_RESERVE_ROSTER_DOMAIN,
            slug='matched-evaluation-reserve',
            role=GymToraxQuartetRole.EVALUATION_RESERVE,
            per_tier=1,
        )

        prospective: list[GymToraxRosterCell] = []

        def prospective_cells(
            *,
            role: GymToraxProspectiveRole,
            stratum: GymToraxProspectiveStratum,
            count: int,
            start_counter: int,
        ) -> int:
            counter = start_counter
            accepted = 0
            while accepted < count:
                candidate = _prospective_candidate(
                    scientific_inputs=inputs,
                    domain=GYM_TORAX_PROSPECTIVE_EFFICACY_ROSTER_DOMAIN,
                    domain_slug='prospective-efficacy',
                    counter=counter,
                    role=role,
                    stratum=stratum,
                )
                if candidate is None:
                    collisions.append(
                        self._collision(
                            index=len(collisions),
                            domain=GYM_TORAX_PROSPECTIVE_EFFICACY_ROSTER_DOMAIN,
                            counter=counter,
                            kind=GymToraxCollisionKind.QUANTIZED_INTERVAL,
                            value=(role.value, stratum.value, counter),
                        )
                    )
                elif accept_cells((candidate,)):
                    prospective.append(candidate)
                    accepted += 1
                counter += 1
            return counter

        counter = prospective_cells(
            role=GymToraxProspectiveRole.EFFICACY_PRIMARY,
            stratum=GymToraxProspectiveStratum.INTERFACE,
            count=9,
            start_counter=0,
        )
        counter = prospective_cells(
            role=GymToraxProspectiveRole.EFFICACY_PRIMARY,
            stratum=GymToraxProspectiveStratum.OUTER,
            count=9,
            start_counter=counter,
        )
        counter = prospective_cells(
            role=GymToraxProspectiveRole.EFFICACY_RESERVE,
            stratum=GymToraxProspectiveStratum.INTERFACE,
            count=1,
            start_counter=counter,
        )
        prospective_cells(
            role=GymToraxProspectiveRole.EFFICACY_RESERVE,
            stratum=GymToraxProspectiveStratum.OUTER,
            count=1,
            start_counter=counter,
        )

        hold_counter = 0
        hold_slot = 1
        while hold_slot <= 5:
            role = (
                GymToraxProspectiveRole.HOLD_CONTROL
                if hold_slot <= 4
                else GymToraxProspectiveRole.HOLD_RESERVE
            )
            candidate = _hold_candidate(
                scientific_inputs=inputs,
                domain=GYM_TORAX_PROSPECTIVE_HOLD_ROSTER_DOMAIN,
                domain_slug='prospective-hold',
                counter=hold_counter,
                role=role,
                anchor_slot_id=f"c-hold-anchor-{hold_slot:02d}",
            )
            if candidate is None:
                collisions.append(
                    self._collision(
                        index=len(collisions),
                        domain=GYM_TORAX_PROSPECTIVE_HOLD_ROSTER_DOMAIN,
                        counter=hold_counter,
                        kind=GymToraxCollisionKind.QUANTIZED_INTERVAL,
                        value=(role.value, hold_counter),
                    )
                )
            elif accept_cells((candidate,)):
                prospective.append(candidate)
                hold_slot += 1
            hold_counter += 1

        calibrated_holds = tuple(
            sorted(
                (value for value in prospective if value.source_anchor_slot_id is not None),
                key=lambda value: value.source_anchor_slot_id or "",
            )
        )
        anchor_slots = tuple(
            NativeHoldCalibrationAnchorSlot(
                slot_id=cell.source_anchor_slot_id or "",
                primary=_anchor_option(cell, slot=index, kind="primary"),
                conditional_alternative=None,
            )
            for index, cell in enumerate(calibrated_holds, start=1)
        )

        route_counter = 0
        while True:
            route_active = _prospective_candidate(
                scientific_inputs=inputs,
                domain=GYM_TORAX_ROUTE_ACTIVE_ROSTER_DOMAIN,
                domain_slug='route-active-canary',
                counter=route_counter,
                role=GymToraxProspectiveRole.ROUTE_ACTIVE_CANARY,
                stratum=GymToraxProspectiveStratum.OUTER,
            )
            if route_active is not None and accept_cells((route_active,)):
                prospective.append(route_active)
                break
            if route_active is None:
                collisions.append(
                    self._collision(
                        index=len(collisions),
                        domain=GYM_TORAX_ROUTE_ACTIVE_ROSTER_DOMAIN,
                        counter=route_counter,
                        kind=GymToraxCollisionKind.QUANTIZED_INTERVAL,
                        value=("p5q-active", route_counter),
                    )
                )
            route_counter += 1

        route_hold_counter = 0
        while True:
            route_hold = _hold_candidate(
                scientific_inputs=inputs,
                domain=GYM_TORAX_ROUTE_HOLD_ROSTER_DOMAIN,
                domain_slug='route-hold-canary',
                counter=route_hold_counter,
                role=GymToraxProspectiveRole.ROUTE_HOLD_CANARY,
                anchor_slot_id=None,
            )
            if route_hold is None:
                collisions.append(
                    self._collision(
                        index=len(collisions),
                        domain=GYM_TORAX_ROUTE_HOLD_ROSTER_DOMAIN,
                        counter=route_hold_counter,
                        kind=GymToraxCollisionKind.QUANTIZED_INTERVAL,
                        value=("p5q-hold", route_hold_counter),
                    )
                )
            elif accept_cells((route_hold,)):
                prospective.append(route_hold)
                break
            route_hold_counter += 1

        all_cells = tuple(
            sorted(
                (
                    *(cell for value in qualification for cell in value.cells),
                    *(cell for value in evaluation_primary for cell in value.cells),
                    *(cell for value in evaluation_reserve for cell in value.cells),
                    *prospective,
                ),
                key=lambda value: value.cell_id,
            )
        )
        by_cell = {value.cell_id: value for value in all_cells}
        episodes: list[GymToraxEpisodeCoordinate] = []

        def add_episode(
            *,
            stage: GymToraxEpisodeStage,
            cell: GymToraxRosterCell,
            member_id: str,
            branch_role: GymToraxEpisodeBranchRole,
            action_word_id: str,
            role_slug: str,
            reserve: bool,
        ) -> None:
            plan = _EpisodePlan(
                stage=stage,
                member_id=member_id,
                branch_role=branch_role,
                action_word_id=action_word_id,
                role_slug=role_slug,
                reserve=reserve,
            )
            coordinate_id = _episode_coordinate_id(cell, plan)
            episodes.append(
                GymToraxEpisodeCoordinate(
                    coordinate_id=coordinate_id,
                    stage=stage,
                    cell_id=cell.cell_id,
                    physical_unit_instance_id=cell.physical_unit_instance_id,
                    model_member_id=member_id,
                    branch_role=branch_role,
                    action_word_id=action_word_id,
                    reserve=reserve,
                    environment_seed=_environment_seed(cell, plan, inputs),
                )
            )

        for cell in all_cells:
            for plan in _episode_plans_for_cell(cell):
                add_episode(
                    stage=plan.stage,
                    cell=cell,
                    member_id=plan.member_id,
                    branch_role=plan.branch_role,
                    action_word_id=plan.action_word_id,
                    role_slug=plan.role_slug,
                    reserve=plan.reserve,
                )

        episodes_sorted = tuple(sorted(episodes, key=lambda value: value.coordinate_id))
        seeds = tuple(value.environment_seed for value in episodes_sorted)
        if len(set(seeds)) != len(seeds) or historical_environment_seeds.intersection(seeds):
            raise AssertionError("accepted tokamak-control replication episode seed namespace is not collision-free")

        audit = GymToraxRosterCollisionAudit(
            audit_id='collision-audit.tokamak-control.complete-maximal',
            historical_domain=historical,
            checked_namespace_ids=(
                'namespace.tokamak-control.cell-id',
                'namespace.tokamak-control.environment-seed',
                'namespace.tokamak-control.preparation-fingerprint',
            ),
            rejected_candidates=tuple(sorted(collisions, key=lambda value: value.collision_id)),
            collision_free=True,
        )
        # Defensive linkage: every episode cell exists in the exact roster.
        if any(value.cell_id not in by_cell for value in episodes_sorted):
            raise AssertionError("episode coordinate names a cell outside the roster")
        primary_count = sum(not value.reserve for value in episodes_sorted)
        return GymToraxRoster(
            roster_id=GYM_TORAX_ROSTER_ID,
            generator_id=self.generator_id,
            historical_collision_domain=historical,
            qualification_quartets=qualification,
            evaluation_primary_quartets=evaluation_primary,
            evaluation_reserve_quartets=evaluation_reserve,
            prospective_cells=tuple(sorted(prospective, key=lambda value: value.cell_id)),
            anchor_slots=anchor_slots,
            episode_coordinates=episodes_sorted,
            collision_audit=audit,
            scientific_inputs=inputs,
            primary_episode_count=primary_count,
            maximum_episode_count=len(episodes_sorted),
            maximum_operational_launches=2 * len(episodes_sorted),
        )


__all__ = [
    'GYM_TORAX_EVALUATION_PRIMARY_ROSTER_DOMAIN',
    'GYM_TORAX_EVALUATION_RESERVE_ROSTER_DOMAIN',
    'GYM_TORAX_PROSPECTIVE_EFFICACY_ROSTER_DOMAIN',
    'GYM_TORAX_PROSPECTIVE_HOLD_ROSTER_DOMAIN',
    'GYM_TORAX_QUALIFICATION_ROSTER_DOMAIN',
    'GYM_TORAX_ROSTER_GENERATOR_ID',
    'GYM_TORAX_ROSTER_ID',
    'GYM_TORAX_ROUTE_ACTIVE_ROSTER_DOMAIN',
    'GYM_TORAX_ROUTE_HOLD_ROSTER_DOMAIN',
    'GymToraxCellKind',
    'GymToraxCollisionKind',
    'GymToraxEpisodeBranchRole',
    'GymToraxEpisodeCoordinate',
    'GymToraxEpisodeStage',
    'GymToraxExposedSeedCollisionDomain',
    'GymToraxMatchedQuartet',
    'GymToraxProspectiveRole',
    'GymToraxProspectiveStratum',
    'GymToraxQuartetRole',
    'GymToraxRosterCell',
    'GymToraxRosterCollisionAudit',
    'GymToraxRosterCollision',
    'GymToraxRosterGenerator',
    'GymToraxRosterTier',
    'GymToraxRoster',
]
