"Prospective Gym--TORAX denominator qualification.\n\nThis module is additive to the receipt-bound preparation, preparation-path and numerical-denominator implementations.  It owns\nthe exact outcome-blind denominator-qualification design and its pure validation/selection\nsemantics.  Simulator acquisition is added only after these records are\ncovered by disk-independent conformance.\n"

from __future__ import annotations

import base64
import copy
import hashlib
import itertools
import math
import time
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
from enum import StrEnum
from typing import Any, ClassVar, Iterable, Mapping, Sequence
from types import MappingProxyType

import numpy as np
import numpy.typing as npt

from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_relative_locator,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.status import ReadinessStatus
from empirical_lawhood.kernel.time import parse_utc_timestamp
from empirical_lawhood.kernel.worlds import EvidenceUnitScope
from empirical_lawhood.planning.formal_analysis import (
    FormalGapSourceCapabilityInventory,
    derive_formal_gap_applicability,
)
from empirical_lawhood.planning.formal_gaps import (
    FormalGapCoverage,
    FormalGapCoverageAssignment,
    FormalGapCoverageDisposition,
    FormalGapEvidenceWorld,
    FormalGapRegister,
    validate_formal_gap_coverage,
)

from .open_campaigns import OpenSimulatorSourceManifest
from .prepared_base import EpisodeDisposition


DENOMINATOR_RESPONSE_PARENT_ID = 'design.denominator-qualification.denominator-qualification'
DENOMINATOR_RESPONSE_QUALIFICATION_CONFIG_ID = 'config.denominator-qualification.source-runtime-qualification'
DENOMINATOR_RESPONSE_DEVELOPMENT_CONFIG_ID = 'config.denominator-qualification.numerical-factorial-development'
DENOMINATOR_RESPONSE_EVALUATION_CONFIG_PREFIX = 'config.denominator-qualification.numerical-'
DENOMINATOR_RESPONSE_EVALUATOR_ID = 'evaluator.denominator-qualification.denominator-qualification'
DENOMINATOR_RESPONSE_DECISION_RULE_ID = 'decision-rule.denominator-qualification-development-to-evaluation'
DENOMINATOR_RESPONSE_HANDOFF_ID = 'handoff.denominator-qualification-to-followup-study-or-stop'
DENOMINATOR_RESPONSE_BASELINE_WORD_ID = 'word.denominator-response.midrange-baseline'

DENOMINATOR_RESPONSE_HORIZON_S = 180
DENOMINATOR_RESPONSE_ACTION_ANCHOR_S = 120
DENOMINATOR_RESPONSE_COMMON_GRID_POINTS = 201
DENOMINATOR_RESPONSE_DECISIVE_CLOCKS = tuple(range(110, 181))
DENOMINATOR_RESPONSE_MAX_PROVISIONAL_EPISODE_BYTES = 16 * 1024**2
DENOMINATOR_RESPONSE_IN_MEMORY_GUARD_BYTES = 256 * 1024**2
DENOMINATOR_RESPONSE_EXTERNAL_FREE_FLOOR_BYTES = 8 * 1024**3
DENOMINATOR_RESPONSE_NUMERICAL_ZERO_NORMALIZED = Decimal("1e-12")
DENOMINATOR_RESPONSE_ACTION_COMPONENT_IDS = (
    "ip_a",
    "nbi_power_w",
    "nbi_location",
    "nbi_width",
    "ecrh_power_w",
    "ecrh_location",
    "ecrh_width",
)

_QUANTUM = Decimal("0.0000001")
_COARSE_LEVELS = (0, 0, 0)
_MIDDLE_LEVELS = (1, 1, 1)
_FINE_LEVELS = (2, 2, 2)
_MEASUREMENT_DIRECT_GAP_IDS = frozenset({"gap.calculus.numerical-view-convergence"})
_DENOMINATOR_INAPPLICABLE_GAP_IDS = frozenset(
    {
        "gap.dynamics.preservation-barriers",
        "gap.geometry.admission-margins",
        "gap.geometry.cohomology",
        "gap.geometry.decision-quotients",
        "gap.geometry.information-geometry",
        "gap.geometry.reachability-viability",
        "gap.geometry.symmetry-gauge",
        "gap.geometry.topology-restriction",
    }
)


class DenominatorResponseStage(StrEnum):
    QUALIFICATION = "QUALIFICATION"
    NUMERICAL_QUALIFICATION_DEVELOPMENT = 'NUMERICAL_QUALIFICATION_DEVELOPMENT'
    NUMERICAL_QUALIFICATION_EVALUATION = 'NUMERICAL_QUALIFICATION_EVALUATION'
    FOLLOWUP_STUDY_DEVELOPMENT = 'FOLLOWUP_STUDY_DEVELOPMENT'
    FOLLOWUP_STUDY_EVALUATION = 'FOLLOWUP_STUDY_EVALUATION'


class DenominatorResponsePairId(StrEnum):
    COARSE_TO_MIDDLE = 'coarse-to-middle'
    MIDDLE_TO_FINE = 'middle-to-fine'


class DenominatorResponseQualificationResult(StrEnum):
    DENOMINATOR_QUALIFIED = "DENOMINATOR_QUALIFIED"
    NO_COMPATIBLE_DENOMINATOR = "NO_COMPATIBLE_DENOMINATOR"
    NUMERICAL_INTERACTION = "NUMERICAL_INTERACTION"
    NO_REFINEMENT_EVIDENCE = "NO_REFINEMENT_EVIDENCE"
    OBSERVATION_MAP_SENSITIVE = "OBSERVATION_MAP_SENSITIVE"
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
    PARTIAL_SCIENTIFIC = "PARTIAL_SCIENTIFIC"
    TECHNICAL_INVALID = "TECHNICAL_INVALID"


@dataclass(frozen=True, slots=True)
class DenominatorResponseCell(CanonicalRecord):
    """One physical preparation/acquisition unit."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/denominator-response-cell'

    cell_id: str
    stage: DenominatorResponseStage
    reserve: bool
    environment_seed: int
    initial_temperature_scale: Decimal
    initial_density_nbar: Decimal
    bootstrap_multiplier: Decimal
    inner_transport_scale: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.cell_id, field_name="cell_id")
        if self.environment_seed < 0:
            raise ValueError("denominator response environment seed cannot be negative")
        for name, value, lower, upper in (
            (
                "initial_temperature_scale",
                self.initial_temperature_scale,
                Decimal("0.990"),
                Decimal("1.010"),
            ),
            (
                "initial_density_nbar",
                self.initial_density_nbar,
                Decimal("0.848"),
                Decimal("0.852"),
            ),
            (
                "bootstrap_multiplier",
                self.bootstrap_multiplier,
                Decimal("0.995"),
                Decimal("1.005"),
            ),
            (
                "inner_transport_scale",
                self.inner_transport_scale,
                Decimal("0.990"),
                Decimal("1.010"),
            ),
        ):
            validate_decimal(value, field_name=name, minimum=lower)
            if value > upper:
                raise ValueError(f"{name} must be at most {upper}")

    @property
    def coordinate_key(self) -> tuple[Decimal, Decimal, Decimal, Decimal]:
        return (
            self.initial_temperature_scale,
            self.initial_density_nbar,
            self.bootstrap_multiplier,
            self.inner_transport_scale,
        )


@dataclass(frozen=True, slots=True)
class DenominatorResponseNumericalView(CanonicalRecord):
    """One numerical view inside the simulator denominator."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/denominator-response-numerical-view'

    view_id: str
    timestep_level: int
    grid_level: int
    corrector_level: int
    internal_timestep_s: Decimal
    radial_cells: int
    corrector_steps: int
    solver_id: str = "solver.torax.linear-theta"
    predictor_corrector: bool = True
    pereverzev_enabled: bool = True
    pereverzev_chi: Decimal = Decimal("30")
    pereverzev_d: Decimal = Decimal("15")
    precision: str = "float64"
    backend: str = "cpu"

    def __post_init__(self) -> None:
        validate_stable_id(self.view_id, field_name="view_id")
        validate_stable_id(self.solver_id, field_name="solver_id")
        if any(level not in {0, 1, 2} for level in self.levels):
            raise ValueError("denominator response view has an unknown factor level")
        expected = (
            (Decimal("1"), Decimal("0.5"), Decimal("0.25"))[self.timestep_level],
            (25, 33, 41)[self.grid_level],
            (10, 20, 40)[self.corrector_level],
        )
        if expected != (
            self.internal_timestep_s,
            self.radial_cells,
            self.corrector_steps,
        ):
            raise ValueError("denominator response view identity differs from its factor levels")
        suffix = (
            f"dt{str(self.internal_timestep_s).replace('.', 'p')}-"
            f"rho{self.radial_cells}-c{self.corrector_steps}"
        )
        if self.view_id != f"view.denominator-response.{suffix}":
            raise ValueError("denominator response view ID differs from its numerical coordinates")
        if (
            not self.predictor_corrector
            or not self.pereverzev_enabled
            or self.pereverzev_chi != Decimal("30")
            or self.pereverzev_d != Decimal("15")
            or self.precision != "float64"
            or self.backend != "cpu"
        ):
            raise ValueError("denominator response view changed the frozen numerical family")

    @property
    def levels(self) -> tuple[int, int, int]:
        return (self.timestep_level, self.grid_level, self.corrector_level)

    @property
    def native_internal_steps(self) -> int:
        return int(Decimal(DENOMINATOR_RESPONSE_HORIZON_S) / self.internal_timestep_s)


@dataclass(frozen=True, slots=True)
class DenominatorResponseViewPair(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/denominator-response-view-pair'

    pair_id: DenominatorResponsePairId
    primary_view_id: str
    finer_view_id: str
    other_ladder_view_id: str
    evaluation_view_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("primary_view_id", self.primary_view_id),
            ("finer_view_id", self.finer_view_id),
            ("other_ladder_view_id", self.other_ladder_view_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(
            self.evaluation_view_ids,
            field_name="evaluation_view_ids",
            allow_empty=False,
        )
        views = {value.view_id: value for value in all_views()}
        primary = views[self.primary_view_id]
        finer = views[self.finer_view_id]
        if not all(left < right for left, right in zip(primary.levels, finer.levels)):
            raise ValueError("denominator response pair sentinel is not strictly finer")
        low = primary.levels
        high = finer.levels
        expected = tuple(
            sorted(
                views_by_levels()[levels].view_id
                for levels in itertools.product(
                    (low[0], high[0]),
                    (low[1], high[1]),
                    (low[2], high[2]),
                )
            )
        )
        if self.evaluation_view_ids != expected:
            raise ValueError("denominator response pair does not bind its complete local 2^3 stencil")


@dataclass(frozen=True, slots=True)
class DenominatorResponseNativeAction(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/denominator-response-native-action'

    ip_a: Decimal
    nbi_power_w: Decimal
    nbi_location: Decimal
    nbi_width: Decimal
    ecrh_power_w: Decimal
    ecrh_location: Decimal
    ecrh_width: Decimal

    def __post_init__(self) -> None:
        for name, value, lower, upper in (
            ("ip_a", self.ip_a, Decimal("100000"), Decimal("15000000")),
            ("nbi_power_w", self.nbi_power_w, Decimal(0), Decimal("33000000")),
            ("nbi_location", self.nbi_location, Decimal(0), Decimal(1)),
            ("nbi_width", self.nbi_width, Decimal("0.01"), Decimal(1)),
            ("ecrh_power_w", self.ecrh_power_w, Decimal(0), Decimal("20000000")),
            ("ecrh_location", self.ecrh_location, Decimal(0), Decimal(1)),
            ("ecrh_width", self.ecrh_width, Decimal("0.01"), Decimal(1)),
        ):
            validate_decimal(value, field_name=name, minimum=lower)
            if value > upper:
                raise ValueError(f"{name} must be at most {upper}")
        if (
            self.nbi_location != Decimal("0.25")
            or self.nbi_width != Decimal("0.25")
            or self.ecrh_width != Decimal("0.05")
        ):
            raise ValueError("denominator response action changed a frozen preparation coordinate")


@dataclass(frozen=True, slots=True)
class DenominatorResponseActionRow(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/denominator-response-action-row'

    row_id: str
    request_clock_s: int
    action: DenominatorResponseNativeAction

    def __post_init__(self) -> None:
        validate_stable_id(self.row_id, field_name="row_id")
        if not 0 <= self.request_clock_s < DENOMINATOR_RESPONSE_HORIZON_S:
            raise ValueError("denominator response action row is outside the frozen horizon")


@dataclass(frozen=True, slots=True)
class DenominatorResponseActionWord(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/denominator-response-action-word'

    word_id: str
    family_id: str
    rows: tuple[DenominatorResponseActionRow, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.word_id, field_name="word_id")
        validate_stable_id(self.family_id, field_name="family_id")
        require_sorted_unique_ids(self.rows, attribute="row_id", field_name="rows")
        if tuple(value.request_clock_s for value in self.rows) != tuple(range(DENOMINATOR_RESPONSE_HORIZON_S)):
            raise ValueError("denominator response word must cover request clocks 0--179")


@dataclass(frozen=True, slots=True)
class DenominatorResponseReceiverGroup(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/denominator-response-receiver-group'

    group_id: str
    source_field_id: str
    native_unit: str
    common_grid_points: int
    hybrid_absolute_scale: Decimal
    relative_tolerance: Decimal
    map_tolerance: Decimal
    decisive_clock_s: tuple[int, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.group_id, field_name="group_id")
        if not self.source_field_id or any(
            character.isspace() for character in self.source_field_id
        ):
            raise ValueError("denominator response receiver group requires a native source field")
        if not self.native_unit:
            raise ValueError("denominator response receiver group requires a native unit")
        for name, value in (
            ("hybrid_absolute_scale", self.hybrid_absolute_scale),
            ("relative_tolerance", self.relative_tolerance),
            ("map_tolerance", self.map_tolerance),
        ):
            validate_decimal(value, field_name=name, minimum=Decimal("0.0000001"))
        if (
            self.common_grid_points != DENOMINATOR_RESPONSE_COMMON_GRID_POINTS
            or self.relative_tolerance != Decimal("0.02")
            or self.map_tolerance != Decimal("0.002")
            or self.decisive_clock_s != DENOMINATOR_RESPONSE_DECISIVE_CLOCKS
        ):
            raise ValueError("denominator response receiver group changed the frozen denominator-qualification rule")


@dataclass(frozen=True, slots=True)
class DenominatorResponseQualificationTolerances(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/denominator-response-qualification-tolerances'

    contraction_ratio: Decimal = Decimal("0.9")
    interaction_normalized: Decimal = Decimal("0.01")
    map_normalized: Decimal = Decimal("0.002")

    def __post_init__(self) -> None:
        for name, value in (
            ("contraction_ratio", self.contraction_ratio),
            ("interaction_normalized", self.interaction_normalized),
            ("map_normalized", self.map_normalized),
        ):
            validate_decimal(value, field_name=name, minimum=Decimal("0.0000001"))
            if value > Decimal("0.9999999"):
                raise ValueError(f"{name} must be at most 0.9999999")
        if (
            self.contraction_ratio != Decimal("0.9")
            or self.interaction_normalized != Decimal("0.01")
            or self.map_normalized != Decimal("0.002")
        ):
            raise ValueError("denominator qualification tolerances changed after freeze")


@dataclass(frozen=True, slots=True)
class DenominatorResponseQualificationConfig(CanonicalRecord):
    """Exact prospective/excluded Act A child."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/denominator-response-qualification-config'

    config_id: str
    parent_design_id: str
    stage: DenominatorResponseStage
    source_manifest: ObjectIdentity
    formal_coverage: ObjectIdentity
    cells: tuple[DenominatorResponseCell, ...]
    primary_cell_ids: tuple[str, ...]
    reserve_cell_ids: tuple[str, ...]
    views: tuple[DenominatorResponseNumericalView, ...]
    baseline_word: DenominatorResponseActionWord
    candidate_pairs: tuple[DenominatorResponseViewPair, ...]
    selected_pair_id: DenominatorResponsePairId | None
    receiver_groups: tuple[DenominatorResponseReceiverGroup, ...]
    tolerances: DenominatorResponseQualificationTolerances
    evaluator_id: str
    decision_rule_id: str
    outcome_access: OutcomeAccess
    maximum_evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        for name, value in (
            ("config_id", self.config_id),
            ("parent_design_id", self.parent_design_id),
            ("evaluator_id", self.evaluator_id),
            ("decision_rule_id", self.decision_rule_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(
            self.primary_cell_ids,
            field_name="primary_cell_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.reserve_cell_ids,
            field_name="reserve_cell_ids",
        )
        require_sorted_unique_ids(self.cells, attribute="cell_id", field_name="cells")
        require_sorted_unique_ids(self.views, attribute="view_id", field_name="views")
        require_sorted_unique_ids(
            self.receiver_groups,
            attribute="group_id",
            field_name="receiver_groups",
        )
        if self.parent_design_id != DENOMINATOR_RESPONSE_PARENT_ID:
            raise ValueError("denominator response config binds another parent design")
        if self.evaluator_id != DENOMINATOR_RESPONSE_EVALUATOR_ID or self.decision_rule_id != DENOMINATOR_RESPONSE_DECISION_RULE_ID:
            raise ValueError("denominator response config changed its evaluator or decision rule")
        by_id = {value.cell_id: value for value in self.cells}
        if set(by_id) != {*self.primary_cell_ids, *self.reserve_cell_ids}:
            raise ValueError("denominator response config roster differs from its cell inventory")
        if set(self.primary_cell_ids) & set(self.reserve_cell_ids):
            raise ValueError("denominator response primary and reserve cells overlap")
        if any(by_id[value].reserve for value in self.primary_cell_ids) or any(
            not by_id[value].reserve for value in self.reserve_cell_ids
        ):
            raise ValueError("denominator response primary/reserve flags differ")
        if any(value.stage is not self.stage for value in self.cells):
            raise ValueError("denominator response cell stage differs from its config")
        expected_counts = {
            DenominatorResponseStage.QUALIFICATION: (1, 0, 3),
            DenominatorResponseStage.NUMERICAL_QUALIFICATION_DEVELOPMENT: (6, 2, 27),
            DenominatorResponseStage.NUMERICAL_QUALIFICATION_EVALUATION: (12, 3, 8),
        }.get(self.stage)
        if expected_counts is None:
            raise ValueError("Act A config cannot bind an Act B cohort")
        if (
            len(self.primary_cell_ids),
            len(self.reserve_cell_ids),
            len(self.views),
        ) != expected_counts:
            raise ValueError("denominator qualification config changed its frozen cohort or views")
        expected_access = {
            DenominatorResponseStage.QUALIFICATION: OutcomeAccess.DEVELOPMENT_VISIBLE,
            DenominatorResponseStage.NUMERICAL_QUALIFICATION_DEVELOPMENT: OutcomeAccess.DEVELOPMENT_VISIBLE,
            DenominatorResponseStage.NUMERICAL_QUALIFICATION_EVALUATION: OutcomeAccess.EVALUATION_SEALED,
        }[self.stage]
        expected_ceiling = (
            EvidenceCeiling.NON_PROMOTABLE
            if self.stage is DenominatorResponseStage.QUALIFICATION
            else EvidenceCeiling.MEASUREMENT
        )
        if (
            self.outcome_access is not expected_access
            or self.maximum_evidence_ceiling is not expected_ceiling
        ):
            raise ValueError("denominator qualification config changed its visibility or ceiling")
        if self.stage is DenominatorResponseStage.NUMERICAL_QUALIFICATION_EVALUATION:
            if self.selected_pair_id is None:
                raise ValueError("denominator response evaluation requires a selected pair")
            selected = next(
                value for value in self.candidate_pairs if value.pair_id is self.selected_pair_id
            )
            if tuple(value.view_id for value in self.views) != selected.evaluation_view_ids:
                raise ValueError("denominator response evaluation views differ from selected template")
        elif self.selected_pair_id is not None:
            raise ValueError("denominator response pre-evaluation config cannot select a pair")

    @property
    def primary_episode_count(self) -> int:
        return len(self.primary_cell_ids) * len(self.views)

    @property
    def maximum_episode_count(self) -> int:
        return len(self.cells) * len(self.views)


@dataclass(frozen=True, slots=True)
class DenominatorResponseQualificationResourceEnvelope(CanonicalRecord):
    """Pre-canary allocation; qualification replaces provisional byte/time fields."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/denominator-response-qualification-resource-envelope'

    envelope_id: str
    config: ObjectIdentity
    provisional: bool
    primary_episode_count: int
    maximum_episode_count: int
    primary_transition_count: int
    maximum_transition_count: int
    maximum_native_internal_steps: int
    maximum_episode_output_bytes: int
    maximum_total_output_bytes: int
    in_memory_guard_bytes: int
    worker_process_count: int
    cpu_cores: int
    peak_memory_bytes: int
    wall_time_seconds: int
    minimum_free_external_bytes: int

    def __post_init__(self) -> None:
        validate_stable_id(self.envelope_id, field_name="envelope_id")
        values = (
            self.primary_episode_count,
            self.maximum_episode_count,
            self.primary_transition_count,
            self.maximum_transition_count,
            self.maximum_native_internal_steps,
            self.maximum_episode_output_bytes,
            self.maximum_total_output_bytes,
            self.in_memory_guard_bytes,
            self.worker_process_count,
            self.cpu_cores,
            self.peak_memory_bytes,
            self.wall_time_seconds,
            self.minimum_free_external_bytes,
        )
        if any(value <= 0 for value in values):
            raise ValueError("denominator response resource allocations must be positive")
        if (
            self.primary_episode_count > self.maximum_episode_count
            or self.primary_transition_count > self.maximum_transition_count
            or self.maximum_episode_output_bytes > self.in_memory_guard_bytes
            or self.maximum_total_output_bytes
            < self.maximum_episode_count * self.maximum_episode_output_bytes
        ):
            raise ValueError("denominator response resource envelope is internally inconsistent")


@dataclass(frozen=True, slots=True)
class DenominatorResponsePlanTimeSourceInventory(CanonicalRecord):
    """Outcome-blind facts established from installed source and predecessor bytes."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/denominator-response-plan-time-source-inventory'

    inventory_id: str
    gymtorax_version: str
    torax_version: str
    jax_version: str
    jaxlib_version: str
    backend: str
    precision: str
    configurable_terminal_horizon: bool
    full_profile_scalar_state_exposed: bool
    action_bound_ids: tuple[str, ...]
    clock_semantic_ids: tuple[str, ...]
    known_source_unavailable_operand_ids: tuple[str, ...]
    source_inspection_basis_ids: tuple[str, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.inventory_id, field_name="inventory_id")
        for name, values in (
            ("action_bound_ids", self.action_bound_ids),
            ("clock_semantic_ids", self.clock_semantic_ids),
            (
                "known_source_unavailable_operand_ids",
                self.known_source_unavailable_operand_ids,
            ),
            ("source_inspection_basis_ids", self.source_inspection_basis_ids),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)
        if (
            self.gymtorax_version != "1.1.1"
            or self.torax_version != "1.4.2"
            or self.jax_version != "0.10.2"
            or self.jaxlib_version != "0.10.2"
            or self.backend != "cpu"
            or self.precision != "float64"
            or not self.configurable_terminal_horizon
            or not self.full_profile_scalar_state_exposed
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
        ):
            raise ValueError("denominator response plan-time source inventory differs from its freeze")


@dataclass(frozen=True, slots=True)
class DenominatorResponseFloat64Block(CanonicalRecord):
    """One exact little-endian float64 array embedded in canonical JSON."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/denominator-response-float64-block'

    block_id: str
    category: str
    native_field_id: str
    native_unit: str
    dimension_ids: tuple[str, ...]
    shape: tuple[int, ...]
    clock_s: tuple[int, ...]
    data_base64: str
    data_sha256: str
    finite_value_count: int
    nonfinite_value_count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.block_id, field_name="block_id")
        validate_stable_id(self.category, field_name="category")
        if (
            not self.native_field_id
            or any(character.isspace() for character in self.native_field_id)
            or not self.native_unit
        ):
            raise ValueError("denominator response block requires a native field and unit")
        if (
            not self.dimension_ids
            or any(not value for value in self.dimension_ids)
            or len(set(self.dimension_ids)) != len(self.dimension_ids)
        ):
            raise ValueError("denominator response block dimensions must be nonempty and unique")
        if (
            not self.shape
            or any(value <= 0 for value in self.shape)
            or len(self.shape) != len(self.dimension_ids)
        ):
            raise ValueError("denominator response block shape differs from its dimensions")
        if self.clock_s:
            if self.dimension_ids[0] != "clock_s" or self.shape[0] != len(self.clock_s):
                raise ValueError("denominator response block clock axis differs from its clock identity")
            if tuple(sorted(set(self.clock_s))) != self.clock_s:
                raise ValueError("denominator response block clocks must be sorted and unique")
        elif self.dimension_ids[0] == "clock_s":
            raise ValueError("denominator response time block omitted its clock coordinates")
        validate_sha256(self.data_sha256, field_name="data_sha256")
        if min(self.finite_value_count, self.nonfinite_value_count) < 0:
            raise ValueError("denominator response block finite counts cannot be negative")
        expected_values = math.prod(self.shape)
        if self.finite_value_count + self.nonfinite_value_count != expected_values:
            raise ValueError("denominator response block finite counts differ from its shape")
        try:
            payload = base64.b64decode(self.data_base64.encode("ascii"), validate=True)
        except (UnicodeEncodeError, ValueError) as error:
            raise ValueError("denominator response block is not canonical base64") from error
        if (
            len(payload) != expected_values * np.dtype("<f8").itemsize
            or hashlib.sha256(payload).hexdigest() != self.data_sha256
        ):
            raise ValueError("denominator response block bytes differ from its shape or digest")

    def array(self) -> npt.NDArray[np.float64]:
        payload = base64.b64decode(self.data_base64.encode("ascii"), validate=True)
        return np.frombuffer(payload, dtype="<f8").reshape(self.shape).copy()


@dataclass(frozen=True, slots=True)
class DenominatorResponseCapturedEpisode(CanonicalRecord):
    """Lossless all-clock source capture for one cell/view/word."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/denominator-response-captured-episode'

    episode_id: str
    config: ObjectIdentity
    cell: ObjectIdentity
    view: ObjectIdentity
    word: ObjectIdentity
    state_clock_s: tuple[int, ...]
    action_request_clock_s: tuple[int, ...]
    blocks: tuple[DenominatorResponseFloat64Block, ...]
    source_profile_field_ids: tuple[str, ...]
    source_scalar_field_ids: tuple[str, ...]
    source_numerics_field_ids: tuple[str, ...]
    source_unavailable_operand_ids: tuple[str, ...]
    action_clipped_clock_s: tuple[int, ...]
    disposition: EpisodeDisposition
    last_valid_clock_s: int | None
    missing_required_state_clock_s: tuple[int, ...]
    reason_codes: tuple[str, ...]
    runtime_seconds: Decimal
    backend: str
    precision: str
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.episode_id, field_name="episode_id")
        require_sorted_unique_ids(self.blocks, attribute="block_id", field_name="blocks")
        for name, string_values in (
            ("source_profile_field_ids", self.source_profile_field_ids),
            ("source_scalar_field_ids", self.source_scalar_field_ids),
            ("source_numerics_field_ids", self.source_numerics_field_ids),
            (
                "source_unavailable_operand_ids",
                self.source_unavailable_operand_ids,
            ),
            ("reason_codes", self.reason_codes),
        ):
            require_sorted_unique_strings(string_values, field_name=name)
        for name, integer_values in (
            ("state_clock_s", self.state_clock_s),
            ("action_request_clock_s", self.action_request_clock_s),
            ("action_clipped_clock_s", self.action_clipped_clock_s),
            (
                "missing_required_state_clock_s",
                self.missing_required_state_clock_s,
            ),
        ):
            if tuple(sorted(set(integer_values))) != integer_values:
                raise ValueError(f"{name} must be sorted and unique")
        validate_decimal(
            self.runtime_seconds,
            field_name="runtime_seconds",
            minimum=Decimal(0),
        )
        if self.backend != "cpu" or self.precision != "float64":
            raise ValueError("denominator response capture requires CPU float64")
        if self.disposition is EpisodeDisposition.COMPLETE:
            if (
                self.state_clock_s != tuple(range(DENOMINATOR_RESPONSE_HORIZON_S + 1))
                or self.action_request_clock_s != tuple(range(DENOMINATOR_RESPONSE_HORIZON_S))
                or self.last_valid_clock_s != DENOMINATOR_RESPONSE_HORIZON_S
                or self.missing_required_state_clock_s
                or self.reason_codes
            ):
                raise ValueError("complete denominator response episode is not all-clock complete")
        if self.last_valid_clock_s is not None and self.last_valid_clock_s not in (
            self.state_clock_s
        ):
            raise ValueError("denominator response last-valid clock is absent from capture")
        block_keys = {(value.category, value.native_field_id) for value in self.blocks}
        if len(block_keys) != len(self.blocks):
            raise ValueError("denominator response capture repeats a category/native field")
        if self.state_clock_s and not {
            ("action-requested", "native-action"),
            ("action-accepted", "native-action"),
            ("action-applied", "native-action"),
            ("action-realized", "native-action"),
        }.issubset(block_keys):
            raise ValueError("denominator response capture omitted an action-stage ledger")

    def block(self, category: str, native_field_id: str) -> DenominatorResponseFloat64Block:
        try:
            return next(
                value
                for value in self.blocks
                if value.category == category and value.native_field_id == native_field_id
            )
        except StopIteration as error:
            raise KeyError((category, native_field_id)) from error


@dataclass(frozen=True, slots=True)
class DenominatorResponseQualificationCellPairMetrics(CanonicalRecord):
    """All noncompensating Act A predicates for one cell/pair."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/denominator-response-qualification-cell-pair-metrics'

    metric_id: str
    cell_id: str
    pair_id: DenominatorResponsePairId
    complete_bundle: bool
    source_complete: bool
    map_pass: bool
    compatibility_pass: bool
    refinement_pass: bool | None
    interaction_pass: bool
    compatibility_max_by_group: tuple[tuple[str, Decimal], ...]
    coarse_middle_max_by_group: tuple[tuple[str, Decimal], ...]
    middle_fine_max_by_group: tuple[tuple[str, Decimal], ...]
    map_max_by_group: tuple[tuple[str, Decimal], ...]
    selected_cube_interaction_max: Decimal
    coarse_cube_interaction_max: Decimal | None
    fine_cube_interaction_max: Decimal | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.metric_id, field_name="metric_id")
        validate_stable_id(self.cell_id, field_name="cell_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        for name, values in (
            ("compatibility_max_by_group", self.compatibility_max_by_group),
            ("coarse_middle_max_by_group", self.coarse_middle_max_by_group),
            ("middle_fine_max_by_group", self.middle_fine_max_by_group),
            ("map_max_by_group", self.map_max_by_group),
        ):
            keys = tuple(value[0] for value in values)
            if tuple(sorted(set(keys))) != keys:
                raise ValueError(f"{name} group IDs must be sorted and unique")
            for group_id, value in values:
                validate_stable_id(group_id, field_name=f"{name}.group_id")
                validate_decimal(value, field_name=f"{name}.value", minimum=Decimal(0))
        for name, optional_value in (
            (
                "selected_cube_interaction_max",
                self.selected_cube_interaction_max,
            ),
            ("coarse_cube_interaction_max", self.coarse_cube_interaction_max),
            ("fine_cube_interaction_max", self.fine_cube_interaction_max),
        ):
            if optional_value is not None:
                validate_decimal(
                    optional_value,
                    field_name=name,
                    minimum=Decimal(0),
                )


@dataclass(frozen=True, slots=True)
class DenominatorResponseQualificationDevelopmentDecision(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/denominator-response-qualification-development-decision'

    decision_id: str
    config: ObjectIdentity
    selected_pair_id: DenominatorResponsePairId | None
    terminal_result: DenominatorResponseQualificationResult | None
    cell_pair_metrics: tuple[DenominatorResponseQualificationCellPairMetrics, ...]
    pair_pass_ids: tuple[DenominatorResponsePairId, ...]
    evaluation_template_id: str | None
    reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.decision_id, field_name="decision_id")
        require_sorted_unique_ids(
            self.cell_pair_metrics,
            attribute="metric_id",
            field_name="cell_pair_metrics",
        )
        if tuple(sorted(set(self.pair_pass_ids), key=lambda value: value.value)) != (
            self.pair_pass_ids
        ):
            raise ValueError("denominator response pair-pass IDs must be sorted and unique")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        selected = self.selected_pair_id is not None
        if selected != (self.terminal_result is None):
            raise ValueError("denominator response development must select a pair or terminal result")
        if selected:
            if (
                self.selected_pair_id not in self.pair_pass_ids
                or self.evaluation_template_id
                != f"template.denominator-qualification.numerical-{self.selected_pair_id.value}-evaluation"
            ):
                raise ValueError("denominator response development selection differs from its pass set")
        elif self.evaluation_template_id is not None:
            raise ValueError("terminal denominator response development cannot open evaluation")
        if self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
            raise ValueError("denominator response development decision has invalid outcome access")


@dataclass(frozen=True, slots=True)
class DenominatorResponseQualificationEvaluationAdjudication(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/denominator-response-qualification-evaluation-adjudication'

    adjudication_id: str
    config: ObjectIdentity
    selected_pair_id: DenominatorResponsePairId
    primary_result: DenominatorResponseQualificationResult
    cell_pair_metrics: tuple[DenominatorResponseQualificationCellPairMetrics, ...]
    complete_cell_ids: tuple[str, ...]
    handoff_id: str | None
    reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess
    maximum_evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.adjudication_id, field_name="adjudication_id")
        require_sorted_unique_ids(
            self.cell_pair_metrics,
            attribute="metric_id",
            field_name="cell_pair_metrics",
        )
        require_sorted_unique_strings(
            self.complete_cell_ids,
            field_name="complete_cell_ids",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        qualified = self.primary_result is DenominatorResponseQualificationResult.DENOMINATOR_QUALIFIED
        if qualified != (self.handoff_id == DENOMINATOR_RESPONSE_HANDOFF_ID):
            raise ValueError("denominator qualification handoff differs from its primary result")
        if (
            self.outcome_access is not OutcomeAccess.EVALUATION_REVEALED
            or self.maximum_evidence_ceiling is not EvidenceCeiling.MEASUREMENT
        ):
            raise ValueError("denominator qualification adjudication changed its reveal or ceiling")


@dataclass(frozen=True, slots=True)
class DenominatorResponseSourceFile(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/denominator-response-source-file'

    relative_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        validate_relative_locator(self.relative_path)
        validate_sha256(self.sha256, field_name="sha256")
        if self.size_bytes <= 0:
            raise ValueError("denominator response source file must be nonempty")


@dataclass(frozen=True, slots=True)
class DenominatorResponseImplementationManifest(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/denominator-response-implementation-manifest'

    manifest_id: str
    source_files: tuple[DenominatorResponseSourceFile, ...]
    implementation_sha256: str

    def __post_init__(self) -> None:
        validate_stable_id(self.manifest_id, field_name="manifest_id")
        validate_sha256(
            self.implementation_sha256,
            field_name="implementation_sha256",
        )
        paths = tuple(value.relative_path for value in self.source_files)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)) or not paths:
            raise ValueError("denominator response source closure must be nonempty, sorted and unique")


@dataclass(frozen=True, slots=True)
class DenominatorResponseEpisodeArtifactReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/denominator-response-episode-artifact-receipt'

    receipt_id: str
    config: ObjectIdentity
    episode: ObjectIdentity
    cell_id: str
    view_id: str
    relative_path: str
    physical_sha256: str
    size_bytes: int
    runtime_seconds: Decimal
    disposition: EpisodeDisposition
    committed_at_utc: str
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        validate_stable_id(self.cell_id, field_name="cell_id")
        validate_stable_id(self.view_id, field_name="view_id")
        validate_relative_locator(self.relative_path)
        validate_sha256(self.physical_sha256, field_name="physical_sha256")
        validate_decimal(
            self.runtime_seconds,
            field_name="runtime_seconds",
            minimum=Decimal(0),
        )
        parse_utc_timestamp(self.committed_at_utc, field_name="committed_at_utc")
        if self.size_bytes <= 0:
            raise ValueError("denominator response episode receipt has no payload")


@dataclass(frozen=True, slots=True)
class DenominatorResponseCellSubstitution(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/denominator-response-cell-substitution'

    substitution_id: str
    excluded_cell_id: str
    reserve_cell_id: str
    reason_code: str
    classified_before_outcome_access: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.substitution_id, field_name="substitution_id")
        validate_stable_id(self.excluded_cell_id, field_name="excluded_cell_id")
        validate_stable_id(self.reserve_cell_id, field_name="reserve_cell_id")
        if (
            self.excluded_cell_id == self.reserve_cell_id
            or self.reason_code != "TECHNICAL_OBSERVATION_FAILURE"
            or not self.classified_before_outcome_access
        ):
            raise ValueError("denominator response reserve substitution is not technical")


@dataclass(frozen=True, slots=True)
class DenominatorResponseAcquisitionIndex(CanonicalRecord):
    """Terminal receipt index; episode payloads remain independently recoverable."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/denominator-response-acquisition-index'

    index_id: str
    config: ObjectIdentity
    episode_receipts: tuple[DenominatorResponseEpisodeArtifactReceipt, ...]
    attempted_cell_ids: tuple[str, ...]
    effective_cell_ids: tuple[str, ...]
    cell_substitutions: tuple[DenominatorResponseCellSubstitution, ...]
    expected_episode_count: int
    total_size_bytes: int
    maximum_episode_size_bytes: int
    terminal_acquisition: bool
    recovery_verified_without_reacquisition: bool
    completed_at_utc: str
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.index_id, field_name="index_id")
        for name, values in (
            ("attempted_cell_ids", self.attempted_cell_ids),
            ("effective_cell_ids", self.effective_cell_ids),
        ):
            require_sorted_unique_strings(
                values,
                field_name=name,
                allow_empty=False,
            )
        require_sorted_unique_ids(
            self.cell_substitutions,
            attribute="substitution_id",
            field_name="cell_substitutions",
        )
        keys = tuple((value.cell_id, value.view_id) for value in self.episode_receipts)
        if (
            keys != tuple(sorted(keys))
            or len(keys) != len(set(keys))
            or self.expected_episode_count <= 0
            or len(self.episode_receipts) != self.expected_episode_count
            or self.total_size_bytes != sum(value.size_bytes for value in self.episode_receipts)
            or self.maximum_episode_size_bytes
            != max(value.size_bytes for value in self.episode_receipts)
            or not self.terminal_acquisition
            or not self.recovery_verified_without_reacquisition
        ):
            raise ValueError("denominator response acquisition index is not terminal and exact")
        parse_utc_timestamp(self.completed_at_utc, field_name="completed_at_utc")
        if any(
            value.config != self.config
            or value.cell_id not in self.attempted_cell_ids
            or value.outcome_access is not self.outcome_access
            for value in self.episode_receipts
        ):
            raise ValueError("denominator response acquisition index mixes children or visibility")
        excluded = {value.excluded_cell_id for value in self.cell_substitutions}
        reserves = {value.reserve_cell_id for value in self.cell_substitutions}
        if (
            not self.effective_cell_ids
            or not set(self.effective_cell_ids).issubset(self.attempted_cell_ids)
            or not excluded.issubset(self.attempted_cell_ids)
            or not reserves.issubset(self.effective_cell_ids)
            or excluded & set(self.effective_cell_ids)
            or len(excluded) != len(self.cell_substitutions)
            or len(reserves) != len(self.cell_substitutions)
        ):
            raise ValueError("denominator response acquisition substitutions are inconsistent")


@dataclass(frozen=True, slots=True)
class DenominatorResponseQualificationReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/denominator-response-qualification-receipt'

    receipt_id: str
    config: ObjectIdentity
    source_manifest: ObjectIdentity
    implementation: ObjectIdentity
    provisional_resource_envelope: ObjectIdentity
    acquisition_index: ObjectIdentity
    installed_distribution_record_sha256: tuple[str, ...]
    source_profile_field_ids: tuple[str, ...]
    source_scalar_field_ids: tuple[str, ...]
    source_numerics_field_ids: tuple[str, ...]
    source_unavailable_operand_ids: tuple[str, ...]
    maximum_episode_size_bytes: int
    maximum_episode_runtime_seconds: Decimal
    complete_route: bool
    recovery_verified_without_reacquisition: bool
    reason_codes: tuple[str, ...]
    maximum_evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        for name, values in (
            (
                "installed_distribution_record_sha256",
                self.installed_distribution_record_sha256,
            ),
            ("source_profile_field_ids", self.source_profile_field_ids),
            ("source_scalar_field_ids", self.source_scalar_field_ids),
            ("source_numerics_field_ids", self.source_numerics_field_ids),
            (
                "source_unavailable_operand_ids",
                self.source_unavailable_operand_ids,
            ),
            ("reason_codes", self.reason_codes),
        ):
            require_sorted_unique_strings(
                values,
                field_name=name,
                allow_empty=name in {"source_numerics_field_ids", "reason_codes"},
            )
        for value in self.installed_distribution_record_sha256:
            validate_sha256(
                value,
                field_name="installed_distribution_record_sha256",
            )
        validate_decimal(
            self.maximum_episode_runtime_seconds,
            field_name="maximum_episode_runtime_seconds",
            minimum=Decimal(0),
        )
        if (
            self.maximum_episode_size_bytes <= 0
            or not self.complete_route
            or not self.recovery_verified_without_reacquisition
            or self.reason_codes
            or self.maximum_evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
            or self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
        ):
            raise ValueError("denominator response qualification did not close the exact route")


@dataclass(frozen=True, slots=True)
class DenominatorResponseQualificationScientificApproval(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/denominator-response-qualification-scientific-approval'

    approval_id: str
    stage: DenominatorResponseStage
    config: ObjectIdentity
    approver: ObjectIdentity
    proposer: ObjectIdentity
    passed_gate_ids: tuple[str, ...]
    authorization_basis_sha256: str
    approved_at_utc: str
    codex_or_chat_is_approver_or_issuer: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.approval_id, field_name="approval_id")
        require_sorted_unique_strings(
            self.passed_gate_ids,
            field_name="passed_gate_ids",
            allow_empty=False,
        )
        validate_sha256(
            self.authorization_basis_sha256,
            field_name="authorization_basis_sha256",
        )
        parse_utc_timestamp(self.approved_at_utc, field_name="approved_at_utc")
        if (
            self.stage
            not in {
                DenominatorResponseStage.NUMERICAL_QUALIFICATION_DEVELOPMENT,
                DenominatorResponseStage.NUMERICAL_QUALIFICATION_EVALUATION,
            }
            or self.approver == self.proposer
            or self.codex_or_chat_is_approver_or_issuer
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
        ):
            raise ValueError("denominator qualification approval lacks an independent outcome-blind gate")


@dataclass(frozen=True, slots=True)
class DenominatorResponseQualificationIssue(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/denominator-response-qualification-issue'

    issue_id: str
    stage: DenominatorResponseStage
    config: ObjectIdentity
    implementation: ObjectIdentity
    source_closure: ObjectIdentity
    qualification_receipt: ObjectIdentity
    scientific_approval: ObjectIdentity
    custody_authority: ObjectIdentity
    proposer_role_id: str
    approver_role_id: str
    executor_role_id: str
    custodian_role_id: str
    evaluator_role_id: str
    issued_at_utc: str
    execution_authority_required: bool
    reveal_authority_required: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.issue_id, field_name="issue_id")
        roles = (
            self.proposer_role_id,
            self.approver_role_id,
            self.executor_role_id,
            self.custodian_role_id,
            self.evaluator_role_id,
        )
        for role_id in roles:
            validate_stable_id(role_id, field_name="role_id")
        parse_utc_timestamp(self.issued_at_utc, field_name="issued_at_utc")
        if (
            self.stage
            not in {
                DenominatorResponseStage.NUMERICAL_QUALIFICATION_DEVELOPMENT,
                DenominatorResponseStage.NUMERICAL_QUALIFICATION_EVALUATION,
            }
            or len(set(roles)) != len(roles)
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or not self.execution_authority_required
            or self.reveal_authority_required != (self.stage is DenominatorResponseStage.NUMERICAL_QUALIFICATION_EVALUATION)
        ):
            raise ValueError("denominator qualification issue merged or omitted an authority gate")


@dataclass(frozen=True, slots=True)
class DenominatorResponseQualificationExecutionReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/denominator-response-qualification-execution-receipt'

    receipt_id: str
    run_id: str
    issue: ObjectIdentity
    execution_authority: ObjectIdentity
    config: ObjectIdentity
    acquisition_index: ObjectIdentity
    completed_at_utc: str
    terminal_acquisition: bool
    artifacts_complete: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        validate_stable_id(self.run_id, field_name="run_id")
        parse_utc_timestamp(self.completed_at_utc, field_name="completed_at_utc")
        if (
            not self.terminal_acquisition
            or not self.artifacts_complete
            or self.outcome_access
            not in {
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                OutcomeAccess.EVALUATION_SEALED,
            }
        ):
            raise ValueError("denominator qualification execution receipt is not terminal")


@dataclass(frozen=True, slots=True)
class DenominatorResponseQualificationHandoff(CanonicalRecord):
    """Finite conditional handoff; it confers no Act B issue authority."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/denominator-response-qualification-handoff'

    handoff_id: str
    adjudication: ObjectIdentity
    evaluation_index: ObjectIdentity
    selected_pair: DenominatorResponseViewPair
    primary_denominator_view: ObjectIdentity
    finer_sentinel_view: ObjectIdentity
    other_ladder_view: ObjectIdentity
    source_manifest: ObjectIdentity
    formal_coverage: ObjectIdentity
    receiver_groups: tuple[DenominatorResponseReceiverGroup, ...]
    tolerances: DenominatorResponseQualificationTolerances
    development_cell_ids: tuple[str, ...]
    evaluation_cell_ids: tuple[str, ...]
    eligible_for_followup_study_design: bool
    issues_followup_study: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        if self.handoff_id != DENOMINATOR_RESPONSE_HANDOFF_ID:
            raise ValueError("denominator response handoff identity changed")
        for name, values in (
            ("development_cell_ids", self.development_cell_ids),
            ("evaluation_cell_ids", self.evaluation_cell_ids),
        ):
            require_sorted_unique_strings(
                values,
                field_name=name,
                allow_empty=False,
            )
        if set(self.development_cell_ids) & set(self.evaluation_cell_ids):
            raise ValueError("denominator response handoff cohorts overlap")
        if (
            not self.eligible_for_followup_study_design
            or self.issues_followup_study
            or self.outcome_access is not OutcomeAccess.EVALUATION_REVEALED
        ):
            raise ValueError("denominator response handoff merged eligibility with issue")


_FROZEN_ENVIRONMENT_SEEDS: Mapping[str, int] = MappingProxyType({
    'cell.denominator-response-numerical-qualification-development-01': 4251491899948046663,
    'cell.denominator-response-numerical-qualification-development-02': 1235247759025486967,
    'cell.denominator-response-numerical-qualification-development-03': 4124377970947156157,
    'cell.denominator-response-numerical-qualification-development-04': 5613148603468219831,
    'cell.denominator-response-numerical-qualification-development-05': 6289487943884317736,
    'cell.denominator-response-numerical-qualification-development-06': 13410453598975003917,
    'cell.denominator-response-numerical-qualification-development-reserve-01': 6749005868454754787,
    'cell.denominator-response-numerical-qualification-development-reserve-02': 9858717552418721363,
    'cell.denominator-response-numerical-qualification-evaluation-01': 15204536892768561783,
    'cell.denominator-response-numerical-qualification-evaluation-02': 17312416481520127159,
    'cell.denominator-response-numerical-qualification-evaluation-03': 2702641919840414747,
    'cell.denominator-response-numerical-qualification-evaluation-04': 12667531203049096614,
    'cell.denominator-response-numerical-qualification-evaluation-05': 3302975760861412895,
    'cell.denominator-response-numerical-qualification-evaluation-06': 13599499567358908747,
    'cell.denominator-response-numerical-qualification-evaluation-07': 7113608403265818051,
    'cell.denominator-response-numerical-qualification-evaluation-08': 2949984342286354471,
    'cell.denominator-response-numerical-qualification-evaluation-09': 5223254799711547182,
    'cell.denominator-response-numerical-qualification-evaluation-10': 16165396253433229355,
    'cell.denominator-response-numerical-qualification-evaluation-11': 7198086710532209486,
    'cell.denominator-response-numerical-qualification-evaluation-12': 15234073397298557256,
    'cell.denominator-response-numerical-qualification-evaluation-reserve-01': 16943063901424380385,
    'cell.denominator-response-numerical-qualification-evaluation-reserve-02': 5017593165435083143,
    'cell.denominator-response-numerical-qualification-evaluation-reserve-03': 9005314702113198059,
    'cell.denominator-response-followup-study-development-01': 15692788349648582447,
    'cell.denominator-response-followup-study-development-02': 10617032220740638251,
    'cell.denominator-response-followup-study-development-03': 13918783836952720071,
    'cell.denominator-response-followup-study-development-04': 9372730925467463695,
    'cell.denominator-response-followup-study-development-05': 192956893063987490,
    'cell.denominator-response-followup-study-development-06': 13133596062281905466,
    'cell.denominator-response-followup-study-development-reserve-01': 5101093491478936770,
    'cell.denominator-response-followup-study-development-reserve-02': 4592261730645699889,
    'cell.denominator-response-followup-study-evaluation-01': 13400516925459311339,
    'cell.denominator-response-followup-study-evaluation-02': 10232342505929015821,
    'cell.denominator-response-followup-study-evaluation-03': 10875338715272662086,
    'cell.denominator-response-followup-study-evaluation-04': 5746152451403201125,
    'cell.denominator-response-followup-study-evaluation-05': 12510181726293614622,
    'cell.denominator-response-followup-study-evaluation-06': 17813309065919147204,
    'cell.denominator-response-followup-study-evaluation-07': 12214437419085333031,
    'cell.denominator-response-followup-study-evaluation-08': 9637773407093764167,
    'cell.denominator-response-followup-study-evaluation-09': 16442217935157187794,
    'cell.denominator-response-followup-study-evaluation-10': 454338592547010276,
    'cell.denominator-response-followup-study-evaluation-11': 3510841824260185180,
    'cell.denominator-response-followup-study-evaluation-12': 7921181074122782813,
    'cell.denominator-response-followup-study-evaluation-13': 11041399122653190990,
    'cell.denominator-response-followup-study-evaluation-14': 8743978471116402509,
    'cell.denominator-response-followup-study-evaluation-15': 6633747013857683652,
    'cell.denominator-response-followup-study-evaluation-16': 17007713729005523790,
    'cell.denominator-response-followup-study-evaluation-17': 5144811989599065447,
    'cell.denominator-response-followup-study-evaluation-18': 17252605342590775363,
    'cell.denominator-response-followup-study-evaluation-19': 2506414268612327285,
    'cell.denominator-response-followup-study-evaluation-20': 2728239190684303983,
    'cell.denominator-response-followup-study-evaluation-21': 9232065075208926296,
    'cell.denominator-response-followup-study-evaluation-22': 3738446251381969546,
    'cell.denominator-response-followup-study-evaluation-23': 10297289175452555199,
    'cell.denominator-response-followup-study-evaluation-24': 5055560120600449204,
    'cell.denominator-response-followup-study-evaluation-reserve-01': 10759914675195839959,
    'cell.denominator-response-followup-study-evaluation-reserve-02': 6163710391667731607,
    'cell.denominator-response-followup-study-evaluation-reserve-03': 2080202840121024087,
    'cell.denominator-response-qualification-excluded-01': 10593951447664632348,
})


def _seed(value: str) -> int:
    return _FROZEN_ENVIRONMENT_SEEDS[value]


def _lhs_value(
    lower: Decimal,
    upper: Decimal,
    rank: int,
    count: int,
    offset: Decimal,
) -> Decimal:
    return (
        lower + (upper - lower) * (Decimal(rank) + Decimal("0.5")) / Decimal(count) + offset
    ).quantize(_QUANTUM, rounding=ROUND_HALF_EVEN)


def _roster_cell(
    *,
    cell_id: str,
    stage: DenominatorResponseStage,
    reserve: bool,
    global_rank: int,
) -> DenominatorResponseCell:
    count = 58
    ranks = (
        global_rank,
        (17 * global_rank + 3) % count,
        (31 * global_rank + 11) % count,
        (43 * global_rank + 19) % count,
    )
    intervals = (
        (Decimal("0.990"), Decimal("1.010")),
        (Decimal("0.848"), Decimal("0.852")),
        (Decimal("0.995"), Decimal("1.005")),
        (Decimal("0.990"), Decimal("1.010")),
    )
    offsets = (
        Decimal("0.0000001"),
        Decimal("0.0000002"),
        Decimal("0.0000003"),
        Decimal("0.0000004"),
    )
    values = tuple(
        _lhs_value(lower, upper, rank, count, offset)
        for (lower, upper), rank, offset in zip(intervals, ranks, offsets, strict=True)
    )
    return DenominatorResponseCell(
        cell_id=cell_id,
        stage=stage,
        reserve=reserve,
        environment_seed=_seed(cell_id),
        initial_temperature_scale=values[0],
        initial_density_nbar=values[1],
        bootstrap_multiplier=values[2],
        inner_transport_scale=values[3],
    )


def _cohort(
    stage: DenominatorResponseStage,
    *,
    active_count: int,
    reserve_count: int,
    start_rank: int,
) -> tuple[DenominatorResponseCell, ...]:
    stem = stage.value.lower().replace("_", "-")
    values = tuple(
        _roster_cell(
            cell_id=f"cell.denominator-response-{stem}-{index + 1:02d}",
            stage=stage,
            reserve=False,
            global_rank=start_rank + index,
        )
        for index in range(active_count)
    ) + tuple(
        _roster_cell(
            cell_id=f"cell.denominator-response-{stem}-reserve-{index + 1:02d}",
            stage=stage,
            reserve=True,
            global_rank=start_rank + active_count + index,
        )
        for index in range(reserve_count)
    )
    return tuple(sorted(values, key=lambda value: value.cell_id))


def denominator_qualification_development_cells() -> tuple[DenominatorResponseCell, ...]:
    return _cohort(
        DenominatorResponseStage.NUMERICAL_QUALIFICATION_DEVELOPMENT,
        active_count=6,
        reserve_count=2,
        start_rank=0,
    )


def denominator_qualification_evaluation_cells() -> tuple[DenominatorResponseCell, ...]:
    return _cohort(
        DenominatorResponseStage.NUMERICAL_QUALIFICATION_EVALUATION,
        active_count=12,
        reserve_count=3,
        start_rank=8,
    )


def followup_study_development_cells() -> tuple[DenominatorResponseCell, ...]:
    return _cohort(
        DenominatorResponseStage.FOLLOWUP_STUDY_DEVELOPMENT,
        active_count=6,
        reserve_count=2,
        start_rank=23,
    )


def followup_study_evaluation_cells() -> tuple[DenominatorResponseCell, ...]:
    return _cohort(
        DenominatorResponseStage.FOLLOWUP_STUDY_EVALUATION,
        active_count=24,
        reserve_count=3,
        start_rank=31,
    )


def qualification_cells() -> tuple[DenominatorResponseCell, ...]:
    return (
        DenominatorResponseCell(
            cell_id='cell.denominator-response-qualification-excluded-01',
            stage=DenominatorResponseStage.QUALIFICATION,
            reserve=False,
            environment_seed=_seed('cell.denominator-response-qualification-excluded-01'),
            initial_temperature_scale=Decimal("1.0000061"),
            initial_density_nbar=Decimal("0.8500062"),
            bootstrap_multiplier=Decimal("1.0000063"),
            inner_transport_scale=Decimal("1.0000064"),
        ),
    )


def all_claim_bearing_cells() -> tuple[DenominatorResponseCell, ...]:
    values = (
        *denominator_qualification_development_cells(),
        *denominator_qualification_evaluation_cells(),
        *followup_study_development_cells(),
        *followup_study_evaluation_cells(),
    )
    validate_roster(values)
    return values


def validate_roster(cells: Sequence[DenominatorResponseCell]) -> None:
    ids = tuple(value.cell_id for value in cells)
    keys = tuple(value.coordinate_key for value in cells)
    if len(ids) != len(set(ids)) or len(keys) != len(set(keys)):
        raise ValueError("denominator response roster repeats an identity or preparation tuple")
    if any(all(value == value.quantize(Decimal("0.000001")) for value in key) for key in keys):
        raise ValueError("denominator response cell lacks a predecessor-disjoint residue")

    from .numerical_denominator import (
        development_cells as numerical_denominator_development_cells,
        evaluation_cells as numerical_denominator_evaluation_cells,
        qualification_cells as numerical_denominator_qualification_cells,
    )
    from .prepared_base import (
        development_cells as prepared_base_development_cells,
        evaluation_master_cells as prepared_base_evaluation_cells,
    )
    from .preparation_path_sensitivity import evaluation_cells as preparation_path_evaluation_cells, qualification_cell as preparation_path_qualification_cell

    predecessor_ids = {
        *(value.cell_id for value in prepared_base_development_cells()),
        *(value.cell_id for value in prepared_base_evaluation_cells()),
        *(value.cell_id for value in preparation_path_evaluation_cells()),
        preparation_path_qualification_cell().cell_id,
        *(value.cell_id for value in numerical_denominator_development_cells()),
        *(value.cell_id for value in numerical_denominator_evaluation_cells()),
        *(value.cell_id for value in numerical_denominator_qualification_cells()),
    }
    predecessor_keys = {
        *(value.coordinate_key for value in prepared_base_development_cells()),
        *(value.coordinate_key for value in prepared_base_evaluation_cells()),
        *(value.coordinate_key for value in preparation_path_evaluation_cells()),
        preparation_path_qualification_cell().coordinate_key,
        *(value.coordinate_key for value in numerical_denominator_development_cells()),
        *(value.coordinate_key for value in numerical_denominator_evaluation_cells()),
        *(value.coordinate_key for value in numerical_denominator_qualification_cells()),
    }
    if set(ids) & predecessor_ids or set(keys) & predecessor_keys:
        raise ValueError("denominator response roster overlaps a preparation, preparation-path or numerical-denominator preparation")


def _view(levels: tuple[int, int, int]) -> DenominatorResponseNumericalView:
    timestep = (Decimal("1"), Decimal("0.5"), Decimal("0.25"))[levels[0]]
    radial_cells = (25, 33, 41)[levels[1]]
    correctors = (10, 20, 40)[levels[2]]
    return DenominatorResponseNumericalView(
        view_id=(
            f"view.denominator-response.dt{str(timestep).replace('.', 'p')}-rho{radial_cells}-c{correctors}"
        ),
        timestep_level=levels[0],
        grid_level=levels[1],
        corrector_level=levels[2],
        internal_timestep_s=timestep,
        radial_cells=radial_cells,
        corrector_steps=correctors,
    )


def all_views() -> tuple[DenominatorResponseNumericalView, ...]:
    return tuple(
        sorted(
            (
                _view((timestep, grid, corrector))
                for timestep in range(3)
                for grid in range(3)
                for corrector in range(3)
            ),
            key=lambda value: value.view_id,
        )
    )


def views_by_levels() -> dict[tuple[int, int, int], DenominatorResponseNumericalView]:
    return {value.levels: value for value in all_views()}


def diagonal_views() -> tuple[DenominatorResponseNumericalView, ...]:
    by_levels = views_by_levels()
    return tuple(
        sorted(
            (by_levels[value] for value in (_COARSE_LEVELS, _MIDDLE_LEVELS, _FINE_LEVELS)),
            key=lambda value: value.view_id,
        )
    )


def candidate_pairs() -> tuple[DenominatorResponseViewPair, ...]:
    by_levels = views_by_levels()

    def pair(
        pair_id: DenominatorResponsePairId,
        low: tuple[int, int, int],
        high: tuple[int, int, int],
        other: tuple[int, int, int],
    ) -> DenominatorResponseViewPair:
        stencil = tuple(
            sorted(
                by_levels[levels].view_id
                for levels in itertools.product(
                    (low[0], high[0]),
                    (low[1], high[1]),
                    (low[2], high[2]),
                )
            )
        )
        return DenominatorResponseViewPair(
            pair_id=pair_id,
            primary_view_id=by_levels[low].view_id,
            finer_view_id=by_levels[high].view_id,
            other_ladder_view_id=by_levels[other].view_id,
            evaluation_view_ids=stencil,
        )

    return (
        pair(DenominatorResponsePairId.COARSE_TO_MIDDLE, _COARSE_LEVELS, _MIDDLE_LEVELS, _FINE_LEVELS),
        pair(DenominatorResponsePairId.MIDDLE_TO_FINE, _MIDDLE_LEVELS, _FINE_LEVELS, _COARSE_LEVELS),
    )


def baseline_action(request_clock_s: int) -> DenominatorResponseNativeAction:
    if not 0 <= request_clock_s < DENOMINATOR_RESPONSE_HORIZON_S:
        raise ValueError("denominator response baseline clock is outside 0--179")
    ip = (
        Decimal("3000000")
        + Decimal(request_clock_s + 1) * (Decimal("12500000") - Decimal("3000000")) / Decimal(100)
        if request_clock_s < 99
        else Decimal("12500000")
    )
    heating = request_clock_s >= 30
    return DenominatorResponseNativeAction(
        ip_a=ip,
        nbi_power_w=Decimal("16500000") if heating else Decimal(0),
        nbi_location=Decimal("0.25"),
        nbi_width=Decimal("0.25"),
        ecrh_power_w=Decimal("10000000") if heating else Decimal(0),
        ecrh_location=Decimal("0.35"),
        ecrh_width=Decimal("0.05"),
    )


def baseline_word() -> DenominatorResponseActionWord:
    return DenominatorResponseActionWord(
        word_id=DENOMINATOR_RESPONSE_BASELINE_WORD_ID,
        family_id='word-family.denominator-response.identity',
        rows=tuple(
            DenominatorResponseActionRow(
                row_id=f"{DENOMINATOR_RESPONSE_BASELINE_WORD_ID}.row-{clock:03d}",
                request_clock_s=clock,
                action=baseline_action(clock),
            )
            for clock in range(DENOMINATOR_RESPONSE_HORIZON_S)
        ),
    )


def receiver_groups() -> tuple[DenominatorResponseReceiverGroup, ...]:
    values = (
        (
            'receiver-group.denominator-response.electron-temperature',
            "T_e",
            "keV",
            Decimal("0.1"),
        ),
        (
            'receiver-group.denominator-response.ion-temperature',
            "T_i",
            "keV",
            Decimal("0.1"),
        ),
        (
            'receiver-group.denominator-response.electron-density',
            "n_e",
            "m-3",
            Decimal("1e18"),
        ),
        (
            'receiver-group.denominator-response.safety-factor',
            "q",
            "1",
            Decimal("0.1"),
        ),
        (
            'receiver-group.denominator-response.poloidal-flux',
            "psi",
            "Wb",
            Decimal("0.05"),
        ),
        (
            'receiver-group.denominator-response.enclosed-current',
            "Ip_profile",
            "A",
            Decimal("1e5"),
        ),
    )
    return tuple(
        sorted(
            (
                DenominatorResponseReceiverGroup(
                    group_id=group_id,
                    source_field_id=field_id,
                    native_unit=unit,
                    common_grid_points=DENOMINATOR_RESPONSE_COMMON_GRID_POINTS,
                    hybrid_absolute_scale=scale,
                    relative_tolerance=Decimal("0.02"),
                    map_tolerance=Decimal("0.002"),
                    decisive_clock_s=DENOMINATOR_RESPONSE_DECISIVE_CLOCKS,
                )
                for group_id, field_id, unit, scale in values
            ),
            key=lambda value: value.group_id,
        )
    )


def plan_time_source_inventory() -> DenominatorResponsePlanTimeSourceInventory:
    return DenominatorResponsePlanTimeSourceInventory(
        inventory_id='source-inventory.denominator-response.plan-time',
        gymtorax_version="1.1.1",
        torax_version="1.4.2",
        jax_version="0.10.2",
        jaxlib_version="0.10.2",
        backend="cpu",
        precision="float64",
        configurable_terminal_horizon=True,
        full_profile_scalar_state_exposed=True,
        action_bound_ids=(
            'bound.denominator-response.ecrh-location-0-1',
            'bound.denominator-response.ecrh-power-0-20mw',
            'bound.denominator-response.ip-maximum-15ma',
            'bound.denominator-response.ip-ramp-0p2ma-per-s',
            'bound.denominator-response.nbi-power-0-33mw',
        ),
        clock_semantic_ids=(
            'clock.denominator-response.accepted-equals-requested',
            'clock.denominator-response.applied-receiver-request-plus-one',
            'clock.denominator-response.piecewise-linear-within-window',
            'clock.denominator-response.realized-receiver-request-plus-one',
        ),
        known_source_unavailable_operand_ids=(
            "operand.closed-particle-balance",
            "operand.electron-density-source-component",
            "operand.total-electron-density-flux",
        ),
        source_inspection_basis_ids=(
            "source.gymtorax.action-handler-v1.1.1",
            "source.gymtorax.iter-hybrid-env-v1.1.1",
            "source.gymtorax.observation-handler-v1.1.1",
            "source.gymtorax.torax-wrapper-v1.1.1",
            'source.numerical-denominator.accepted-qualification',
            "source.torax.output-state-history-v1.4.2",
        ),
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


def build_denominator_qualification_formal_coverage(
    *,
    register: FormalGapRegister,
    source_manifest: OpenSimulatorSourceManifest,
    denominator_id: str,
    candidate_act_id: str,
    independent_unit_ids: tuple[str, ...],
    numerical_view_ids: tuple[str, ...],
) -> tuple[FormalGapSourceCapabilityInventory, FormalGapCoverage]:
    "Lower the one feasible measurement formal row and type every other disposition."

    inventory = FormalGapSourceCapabilityInventory(
        inventory_id=(f"formal-source-inventory.{candidate_act_id.removeprefix('draft.')}"),
        denominator_id=denominator_id,
        evidence_world=FormalGapEvidenceWorld.NUMERICAL_SIMULATOR,
        source_materializations=(
            ObjectIdentity.from_record(source_manifest.source_id, source_manifest),
        ),
        present_operand_ids=("numerical.views", "response.local"),
        satisfied_prerequisite_ids=(
            "support.nested-view-identity",
            "support.refinement-order",
        ),
        independent_unit_ids=tuple(sorted(independent_unit_ids)),
        independent_unit_scope=EvidenceUnitScope.PHYSICAL_INDEPENDENT_UNIT,
        numerical_view_ids=tuple(sorted(numerical_view_ids)),
        available_estimator_family_ids=("estimator.calculus.local-convergence",),
        available_control_ids=(
            "control.negative-action",
            "control.support-matched-comparator",
        ),
        multiplicity_family_ids=("multiplicity.formal.calculus",),
        denominator_inapplicable_gap_ids=tuple(sorted(_DENOMINATOR_INAPPLICABLE_GAP_IDS)),
        resource_blocked_gap_ids=(),
        requested_claim_ceiling=EvidenceCeiling.MEASUREMENT,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    applicability = derive_formal_gap_applicability(register, inventory)
    assignments = []
    for gap in register.gaps:
        if gap.gap_id in _MEASUREMENT_DIRECT_GAP_IDS:
            assignments.append(
                FormalGapCoverageAssignment(
                    gap_id=gap.gap_id,
                    disposition=FormalGapCoverageDisposition.TEST_IN_THIS_ACT,
                    readiness_reason=None,
                    reason_codes=(),
                    selected_estimator_family_id=gap.estimator_family_ids[0],
                    selected_control_ids=gap.control_ids,
                    selected_multiplicity_family_id=gap.multiplicity_family_id,
                    obligation_ids=(f"obligation.denominator-qualification.{gap.gap_id.removeprefix('gap.')}",),
                    output_ids=(f"output.denominator-qualification.{gap.gap_id.removeprefix('gap.')}",),
                    adjudication_owner_ids=(DENOMINATOR_RESPONSE_EVALUATOR_ID,),
                )
            )
        elif gap.gap_id in _DENOMINATOR_INAPPLICABLE_GAP_IDS:
            assignments.append(
                FormalGapCoverageAssignment(
                    gap_id=gap.gap_id,
                    disposition=(FormalGapCoverageDisposition.NOT_APPLICABLE_TO_DENOMINATOR),
                    readiness_reason=None,
                    reason_codes=("DENOMINATOR_RESPONSE_REQUIRED_MATHEMATICAL_OBJECT_ABSENT",),
                    selected_estimator_family_id=None,
                    selected_control_ids=(),
                    selected_multiplicity_family_id=None,
                    obligation_ids=(),
                    output_ids=(),
                    adjudication_owner_ids=(),
                )
            )
        else:
            assignments.append(
                FormalGapCoverageAssignment(
                    gap_id=gap.gap_id,
                    disposition=(FormalGapCoverageDisposition.DEFER_WITH_TYPED_PREREQUISITE),
                    readiness_reason=ReadinessStatus.PREREQUISITE_NOT_MET,
                    reason_codes=("DENOMINATOR_RESPONSE_CONTROLLED_LOCAL_LAW_PREREQUISITE_ABSENT",),
                    selected_estimator_family_id=None,
                    selected_control_ids=(),
                    selected_multiplicity_family_id=None,
                    obligation_ids=(),
                    output_ids=(),
                    adjudication_owner_ids=(),
                )
            )
    coverage = FormalGapCoverage(
        coverage_id=f"formal-gap-coverage.{candidate_act_id.removeprefix('draft.')}",
        register=ObjectIdentity.from_record(register.register_id, register),
        denominator_id=denominator_id,
        candidate_act_id=candidate_act_id,
        applicability=applicability,
        assignments=tuple(assignments),
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    validate_formal_gap_coverage(register, coverage)
    return inventory, coverage


def _active_and_reserves(
    cells: Iterable[DenominatorResponseCell],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    values = tuple(cells)
    return (
        tuple(sorted(value.cell_id for value in values if not value.reserve)),
        tuple(sorted(value.cell_id for value in values if value.reserve)),
    )


def build_denominator_qualification_config(
    *,
    stage: DenominatorResponseStage,
    source_manifest: OpenSimulatorSourceManifest,
    formal_coverage: FormalGapCoverage,
    selected_pair_id: DenominatorResponsePairId | None = None,
) -> DenominatorResponseQualificationConfig:
    if stage is DenominatorResponseStage.QUALIFICATION:
        config_id = DENOMINATOR_RESPONSE_QUALIFICATION_CONFIG_ID
        cells = qualification_cells()
        views = diagonal_views()
    elif stage is DenominatorResponseStage.NUMERICAL_QUALIFICATION_DEVELOPMENT:
        config_id = DENOMINATOR_RESPONSE_DEVELOPMENT_CONFIG_ID
        cells = denominator_qualification_development_cells()
        views = all_views()
    elif stage is DenominatorResponseStage.NUMERICAL_QUALIFICATION_EVALUATION:
        if selected_pair_id is None:
            raise ValueError("denominator response evaluation config requires a selected pair")
        config_id = f"{DENOMINATOR_RESPONSE_EVALUATION_CONFIG_PREFIX}{selected_pair_id.value}-evaluation"
        cells = denominator_qualification_evaluation_cells()
        selected = next(value for value in candidate_pairs() if value.pair_id is selected_pair_id)
        by_id = {value.view_id: value for value in all_views()}
        views = tuple(by_id[value] for value in selected.evaluation_view_ids)
    else:
        raise ValueError("Act A config cannot bind an Act B stage")
    active, reserves = _active_and_reserves(cells)
    return DenominatorResponseQualificationConfig(
        config_id=config_id,
        parent_design_id=DENOMINATOR_RESPONSE_PARENT_ID,
        stage=stage,
        source_manifest=ObjectIdentity.from_record(source_manifest.source_id, source_manifest),
        formal_coverage=ObjectIdentity.from_record(formal_coverage.coverage_id, formal_coverage),
        cells=cells,
        primary_cell_ids=active,
        reserve_cell_ids=reserves,
        views=views,
        baseline_word=baseline_word(),
        candidate_pairs=candidate_pairs(),
        selected_pair_id=selected_pair_id,
        receiver_groups=receiver_groups(),
        tolerances=DenominatorResponseQualificationTolerances(),
        evaluator_id=DENOMINATOR_RESPONSE_EVALUATOR_ID,
        decision_rule_id=DENOMINATOR_RESPONSE_DECISION_RULE_ID,
        outcome_access=(
            OutcomeAccess.EVALUATION_SEALED
            if stage is DenominatorResponseStage.NUMERICAL_QUALIFICATION_EVALUATION
            else OutcomeAccess.DEVELOPMENT_VISIBLE
        ),
        maximum_evidence_ceiling=(
            EvidenceCeiling.NON_PROMOTABLE
            if stage is DenominatorResponseStage.QUALIFICATION
            else EvidenceCeiling.MEASUREMENT
        ),
    )


def compile_denominator_qualification_resource_envelope(
    config: DenominatorResponseQualificationConfig,
) -> DenominatorResponseQualificationResourceEnvelope:
    maximum_output = config.maximum_episode_count * DENOMINATOR_RESPONSE_MAX_PROVISIONAL_EPISODE_BYTES
    maximum_native_steps = len(config.cells) * sum(
        value.native_internal_steps for value in config.views
    )
    wall_time = {
        DenominatorResponseStage.QUALIFICATION: 3_600,
        DenominatorResponseStage.NUMERICAL_QUALIFICATION_DEVELOPMENT: 12 * 3_600,
        DenominatorResponseStage.NUMERICAL_QUALIFICATION_EVALUATION: 8 * 3_600,
    }[config.stage]
    return DenominatorResponseQualificationResourceEnvelope(
        envelope_id=f"resource-envelope.{config.config_id.removeprefix('config.')}",
        config=ObjectIdentity.from_record(config.config_id, config),
        provisional=True,
        primary_episode_count=config.primary_episode_count,
        maximum_episode_count=config.maximum_episode_count,
        primary_transition_count=config.primary_episode_count * DENOMINATOR_RESPONSE_HORIZON_S,
        maximum_transition_count=config.maximum_episode_count * DENOMINATOR_RESPONSE_HORIZON_S,
        maximum_native_internal_steps=maximum_native_steps,
        maximum_episode_output_bytes=DENOMINATOR_RESPONSE_MAX_PROVISIONAL_EPISODE_BYTES,
        maximum_total_output_bytes=maximum_output,
        in_memory_guard_bytes=DENOMINATOR_RESPONSE_IN_MEMORY_GUARD_BYTES,
        worker_process_count=1,
        cpu_cores=4,
        peak_memory_bytes=24 * 1024**3,
        wall_time_seconds=wall_time,
        minimum_free_external_bytes=max(DENOMINATOR_RESPONSE_EXTERNAL_FREE_FLOOR_BYTES, 3 * maximum_output),
    )


def compile_qualified_denominator_qualification_resource_envelope(
    config: DenominatorResponseQualificationConfig,
    qualification_index: DenominatorResponseAcquisitionIndex,
) -> DenominatorResponseQualificationResourceEnvelope:
    """Freeze a conservative exact child allocation from the excluded canary."""

    if (
        qualification_index.expected_episode_count != 3
        or qualification_index.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
        or any(
            value.disposition is not EpisodeDisposition.COMPLETE
            for value in qualification_index.episode_receipts
        )
    ):
        raise ValueError("denominator response resource compiler requires complete qualification")
    observed_size = qualification_index.maximum_episode_size_bytes
    size_with_margin = math.ceil(observed_size * 1.25 / 1024**2) * 1024**2
    episode_bound = max(size_with_margin, 12 * 1024**2)
    if episode_bound > DENOMINATOR_RESPONSE_MAX_PROVISIONAL_EPISODE_BYTES:
        raise ValueError("denominator response canary exceeded the admitted artifact surface")
    maximum_output = config.maximum_episode_count * episode_bound
    maximum_runtime = max(
        float(value.runtime_seconds) for value in qualification_index.episode_receipts
    )
    wall_time = (
        math.ceil((maximum_runtime * config.maximum_episode_count * 1.5 + 3_600) / 3_600) * 3_600
    )
    maximum_native_steps = len(config.cells) * sum(
        value.native_internal_steps for value in config.views
    )
    return DenominatorResponseQualificationResourceEnvelope(
        envelope_id=f"resource-envelope.{config.config_id.removeprefix('config.')}",
        config=ObjectIdentity.from_record(config.config_id, config),
        provisional=False,
        primary_episode_count=config.primary_episode_count,
        maximum_episode_count=config.maximum_episode_count,
        primary_transition_count=config.primary_episode_count * DENOMINATOR_RESPONSE_HORIZON_S,
        maximum_transition_count=config.maximum_episode_count * DENOMINATOR_RESPONSE_HORIZON_S,
        maximum_native_internal_steps=maximum_native_steps,
        maximum_episode_output_bytes=episode_bound,
        maximum_total_output_bytes=maximum_output,
        in_memory_guard_bytes=DENOMINATOR_RESPONSE_IN_MEMORY_GUARD_BYTES,
        worker_process_count=1,
        cpu_cores=4,
        peak_memory_bytes=24 * 1024**3,
        wall_time_seconds=wall_time,
        minimum_free_external_bytes=max(
            DENOMINATOR_RESPONSE_EXTERNAL_FREE_FLOOR_BYTES,
            3 * maximum_output,
        ),
    )


def _float64_block(
    *,
    category: str,
    native_field_id: str,
    native_unit: str,
    dimension_ids: tuple[str, ...],
    values: npt.ArrayLike,
    clock_s: tuple[int, ...] = (),
) -> DenominatorResponseFloat64Block:
    array = np.ascontiguousarray(values, dtype="<f8")
    if array.ndim == 0:
        array = array.reshape(1)
    payload = array.tobytes(order="C")
    token = hashlib.sha256(f"{category}\0{native_field_id}".encode("utf-8")).hexdigest()[:24]
    finite_count = int(np.isfinite(array).sum())
    return DenominatorResponseFloat64Block(
        block_id=f"block.denominator-response.{token}",
        category=category,
        native_field_id=native_field_id,
        native_unit=native_unit or "source-unit-unspecified",
        dimension_ids=dimension_ids,
        shape=tuple(int(value) for value in array.shape),
        clock_s=clock_s,
        data_base64=base64.b64encode(payload).decode("ascii"),
        data_sha256=hashlib.sha256(payload).hexdigest(),
        finite_value_count=finite_count,
        nonfinite_value_count=array.size - finite_count,
    )


def _scale_profile(profile: Any, scale: float) -> Any:
    result = copy.deepcopy(profile)
    for radial_values in result.values():
        for radius in tuple(radial_values):
            if float(radius) < 1.0:
                radial_values[radius] = float(radial_values[radius]) * scale
    return result


def _environment(cell: DenominatorResponseCell, view: DenominatorResponseNumericalView) -> Any:
    from gymtorax.envs.iter_hybrid_env import (  # type: ignore[import-untyped]
        CONFIG,
        IterHybridEnv,
    )

    source_config = copy.deepcopy(CONFIG)
    source_config["numerics"]["t_final"] = DENOMINATOR_RESPONSE_HORIZON_S
    source_config["numerics"]["fixed_dt"] = float(view.internal_timestep_s)
    source_config["geometry"]["n_rho"] = view.radial_cells
    source_config["solver"]["n_corrector_steps"] = view.corrector_steps
    source_config["profile_conditions"]["T_i"] = _scale_profile(
        source_config["profile_conditions"]["T_i"],
        float(cell.initial_temperature_scale),
    )
    source_config["profile_conditions"]["T_e"] = _scale_profile(
        source_config["profile_conditions"]["T_e"],
        float(cell.initial_temperature_scale),
    )
    source_config["profile_conditions"]["nbar"] = float(cell.initial_density_nbar)
    source_config["neoclassical"]["bootstrap_current"]["bootstrap_multiplier"] = float(
        cell.bootstrap_multiplier
    )
    for key in ("D_e_inner", "chi_i_inner", "chi_e_inner"):
        source_config["transport"][key] = float(source_config["transport"][key]) * float(
            cell.inner_transport_scale
        )

    class DenominatorResponseEnvironment(IterHybridEnv):  # type: ignore[misc]
        def _get_torax_config(self) -> dict[str, Any]:
            return {
                "config": source_config,
                "discretization": "fixed",
                "ratio_a_sim": int(round(1.0 / float(view.internal_timestep_s))),
            }

        def _compute_reward(self, state: Any, next_state: Any, action: Any) -> float:
            del state, next_state, action
            return 0.0

    return DenominatorResponseEnvironment(render_mode=None, store_history=False, log_level="warning")


def _action_array(action: DenominatorResponseNativeAction) -> npt.NDArray[np.float64]:
    return np.asarray(
        (
            float(action.ip_a),
            float(action.nbi_power_w),
            float(action.nbi_location),
            float(action.nbi_width),
            float(action.ecrh_power_w),
            float(action.ecrh_location),
            float(action.ecrh_width),
        ),
        dtype=np.float64,
    )


def _action_mapping(
    action: DenominatorResponseNativeAction,
) -> dict[str, npt.NDArray[np.float64]]:
    return {
        "Ip": np.asarray((float(action.ip_a),), dtype=np.float64),
        "NBI": np.asarray(
            (
                float(action.nbi_power_w),
                float(action.nbi_location),
                float(action.nbi_width),
            ),
            dtype=np.float64,
        ),
        "ECRH": np.asarray(
            (
                float(action.ecrh_power_w),
                float(action.ecrh_location),
                float(action.ecrh_width),
            ),
            dtype=np.float64,
        ),
    }


def _native_action(values: Mapping[str, Any]) -> DenominatorResponseNativeAction:
    ip = np.asarray(values["Ip"], dtype=np.float64).reshape(-1)
    nbi = np.asarray(values["NBI"], dtype=np.float64).reshape(-1)
    ecrh = np.asarray(values["ECRH"], dtype=np.float64).reshape(-1)
    return DenominatorResponseNativeAction(
        ip_a=Decimal(str(float(ip[0]))),
        nbi_power_w=Decimal(str(float(nbi[0]))),
        nbi_location=Decimal(str(float(nbi[1]))),
        nbi_width=Decimal(str(float(nbi[2]))),
        ecrh_power_w=Decimal(str(float(ecrh[0]))),
        ecrh_location=Decimal(str(float(ecrh[1]))),
        ecrh_width=Decimal(str(float(ecrh[2]))),
    )


def _state_scalar(state: Mapping[str, Any], field_id: str) -> Decimal | None:
    raw = state.get("scalars", {}).get(field_id)
    if raw is None:
        return None
    values = np.asarray(raw, dtype=np.float64).reshape(-1)
    if values.size != 1:
        return None
    return Decimal(str(float(values[0])))


def _realized_action(environment: Any, applied: DenominatorResponseNativeAction) -> DenominatorResponseNativeAction:
    state = environment.state or {}
    ip = _state_scalar(state, "Ip")
    nbi = _state_scalar(state, "P_aux_generic_total")
    ecrh = _state_scalar(state, "P_ecrh_e")
    return DenominatorResponseNativeAction(
        ip_a=applied.ip_a if ip is None else ip,
        nbi_power_w=applied.nbi_power_w if nbi is None else nbi,
        nbi_location=applied.nbi_location,
        nbi_width=applied.nbi_width,
        ecrh_power_w=applied.ecrh_power_w if ecrh is None else ecrh,
        ecrh_location=applied.ecrh_location,
        ecrh_width=applied.ecrh_width,
    )


def _source_metadata(
    environment: Any,
) -> tuple[
    dict[tuple[str, str], tuple[str, tuple[str, ...]]],
    dict[str, npt.NDArray[np.float64]],
]:
    """Read the source schema once; receiver values continue through ``state``."""

    tree = environment.torax_app.get_state_data()
    metadata: dict[tuple[str, str], tuple[str, tuple[str, ...]]] = {}
    for category in ("profiles", "scalars"):
        dataset = tree[f"/{category}/"].ds
        variables = dataset.variables
        coordinate_names = set(dataset.coords)
        for field_id, variable in variables.items():
            if field_id in coordinate_names:
                continue
            unit = str(
                variable.attrs.get(
                    "units",
                    variable.attrs.get("unit", "source-unit-unspecified"),
                )
            )
            dimensions = tuple(str(value) for value in variable.dims)
            if dimensions and dimensions[0] != "time":
                species_dimension = dimensions[0]
                labels = (
                    variables[species_dimension].values
                    if species_dimension in variables
                    else np.arange(variable.shape[0])
                )
                for index in range(variable.shape[0]):
                    split_id = f"{field_id}_{str(labels[index])}"
                    metadata[(category, split_id)] = (
                        unit,
                        dimensions[2:] if dimensions[1:2] == ("time",) else dimensions[1:],
                    )
            else:
                metadata[(category, field_id)] = (
                    unit,
                    dimensions[1:] if dimensions[:1] == ("time",) else dimensions,
                )
    coordinates = {
        str(name): np.asarray(value.values, dtype=np.float64).reshape(-1)
        for name, value in tree.coords.items()
        if str(name) in {"rho_norm", "rho_face_norm", "rho_cell_norm"}
    }
    return metadata, coordinates


def _capture_state(
    state: Mapping[str, Any],
    *,
    metadata: Mapping[tuple[str, str], tuple[str, tuple[str, ...]]],
    field_values: dict[tuple[str, str], list[npt.NDArray[np.float64]]],
    expected_shapes: dict[tuple[str, str], tuple[int, ...]],
    reason_codes: set[str],
) -> None:
    for category in ("profiles", "scalars"):
        observed = state.get(category, {})
        for field_id, raw in observed.items():
            key = (category, str(field_id))
            array = np.asarray(raw, dtype=np.float64)
            if array.ndim == 0:
                array = array.reshape(1)
            shape = tuple(int(value) for value in array.shape)
            if key not in expected_shapes:
                expected_shapes[key] = shape
                field_values[key] = []
            if shape != expected_shapes[key]:
                reason_codes.add("SOURCE_FIELD_SHAPE_DRIFT")
                array = np.full(expected_shapes[key], np.nan, dtype=np.float64)
            field_values[key].append(np.ascontiguousarray(array, dtype=np.float64))
        missing = set(expected_shapes).difference(
            (category, str(field_id)) for field_id in observed
        )
        for key in missing:
            if key[0] != category:
                continue
            reason_codes.add("SOURCE_FIELD_MISSING_AFTER_INITIALIZATION")
            field_values[key].append(np.full(expected_shapes[key], np.nan, dtype=np.float64))
    unknown_metadata = set(field_values).difference(metadata)
    if unknown_metadata:
        reason_codes.add("SOURCE_FIELD_METADATA_UNAVAILABLE")


def _numerics_values(environment: Any) -> dict[str, npt.NDArray[np.float64]]:
    current = environment.torax_app.current_sim_state
    if current is None:
        return {}
    outputs = current.solver_numeric_outputs
    result = {}
    for field_id in (
        "inner_solver_iterations",
        "outer_solver_iterations",
        "sawtooth_crash",
    ):
        raw = getattr(outputs, field_id, None)
        if raw is not None:
            result[field_id] = np.asarray(raw, dtype=np.float64).reshape(-1)
    return result


def _make_state_blocks(
    *,
    state_clock_s: tuple[int, ...],
    metadata: Mapping[tuple[str, str], tuple[str, tuple[str, ...]]],
    coordinates: Mapping[str, npt.NDArray[np.float64]],
    field_values: Mapping[tuple[str, str], list[npt.NDArray[np.float64]]],
    numerics_values: Mapping[str, list[npt.NDArray[np.float64]]],
    requested: Sequence[DenominatorResponseNativeAction],
    accepted: Sequence[DenominatorResponseNativeAction],
    applied: Sequence[DenominatorResponseNativeAction],
    realized: Sequence[DenominatorResponseNativeAction],
    reason_codes: set[str],
) -> tuple[DenominatorResponseFloat64Block, ...]:
    blocks: list[DenominatorResponseFloat64Block] = []
    for coordinate_id, coordinate_values in sorted(coordinates.items()):
        blocks.append(
            _float64_block(
                category="coordinate",
                native_field_id=coordinate_id,
                native_unit="1",
                dimension_ids=(coordinate_id,),
                values=coordinate_values,
            )
        )
    for (category, field_id), captured_values in sorted(field_values.items()):
        if len(captured_values) != len(state_clock_s):
            reason_codes.add("SOURCE_FIELD_CLOCK_COUNT_MISMATCH")
            continue
        unit, dimensions = metadata.get(
            (category, field_id),
            (
                "source-unit-unspecified",
                tuple(f"axis_{i}" for i in range(captured_values[0].ndim)),
            ),
        )
        if category == "scalars" and not dimensions and captured_values[0].ndim == 1:
            dimensions = ("value",)
        if len(dimensions) != captured_values[0].ndim:
            reason_codes.add("SOURCE_FIELD_DIMENSION_METADATA_MISMATCH")
            dimensions = tuple(f"source_axis_{index}" for index in range(captured_values[0].ndim))
        stacked = np.stack(captured_values)
        blocks.append(
            _float64_block(
                category=f"source-{category.removesuffix('s')}",
                native_field_id=field_id,
                native_unit=unit,
                dimension_ids=("clock_s", *dimensions),
                values=stacked,
                clock_s=state_clock_s,
            )
        )
    for field_id, captured_numerics in sorted(numerics_values.items()):
        if len(captured_numerics) != len(state_clock_s):
            reason_codes.add("NUMERICS_FIELD_CLOCK_COUNT_MISMATCH")
            continue
        blocks.append(
            _float64_block(
                category="source-numerics",
                native_field_id=field_id,
                native_unit="1",
                dimension_ids=("clock_s", "value"),
                values=np.stack(captured_numerics),
                clock_s=state_clock_s,
            )
        )
    action_clock_s = tuple(range(len(requested)))
    for category, action_values in (
        ("action-requested", requested),
        ("action-accepted", accepted),
        ("action-applied", applied),
        ("action-realized", realized),
    ):
        if not action_values:
            continue
        blocks.append(
            _float64_block(
                category=category,
                native_field_id="native-action",
                native_unit='mixed-native-see-denominator-response-action-component-contract',
                dimension_ids=("clock_s", "action_component"),
                values=np.stack([_action_array(value) for value in action_values]),
                clock_s=action_clock_s,
            )
        )

    common_grid = np.linspace(0.0, 1.0, DENOMINATOR_RESPONSE_COMMON_GRID_POINTS, dtype=np.float64)
    blocks.append(
        _float64_block(
            category="coordinate",
            native_field_id="common_rho_norm",
            native_unit="1",
            dimension_ids=("common_rho_norm",),
            values=common_grid,
        )
    )
    by_key = {(value.category, value.native_field_id): value for value in blocks}
    receiver_by_field = {value.source_field_id: value for value in receiver_groups()}
    for field_id, group in sorted(receiver_by_field.items()):
        source = by_key.get(("source-profile", field_id))
        if source is None:
            reason_codes.add(f"DECISIVE_SOURCE_FIELD_ABSENT_{field_id.upper()}")
            continue
        if len(source.dimension_ids) != 2:
            reason_codes.add(f"DECISIVE_SOURCE_FIELD_RANK_INVALID_{field_id.upper()}")
            continue
        coordinate_id = source.dimension_ids[1]
        coordinate = by_key.get(("coordinate", coordinate_id))
        if coordinate is None:
            reason_codes.add(f"DECISIVE_SOURCE_COORDINATE_ABSENT_{coordinate_id.upper()}")
            continue
        native_x = coordinate.array().reshape(-1)
        native_y = source.array()
        if native_y.shape[1] != native_x.size or native_x[0] > 0 or native_x[-1] < 1:
            reason_codes.add(f"DECISIVE_SOURCE_SUPPORT_INVALID_{field_id.upper()}")
            continue
        common = np.stack([np.interp(common_grid, native_x, row) for row in native_y])
        blocks.append(
            _float64_block(
                category="common-profile",
                native_field_id=field_id,
                native_unit=group.native_unit,
                dimension_ids=("clock_s", "common_rho_norm"),
                values=common,
                clock_s=state_clock_s,
            )
        )
    return tuple(sorted(blocks, key=lambda value: value.block_id))


def acquire_denominator_qualification_episode(
    *,
    config: DenominatorResponseQualificationConfig,
    cell_id: str,
    view_id: str,
) -> DenominatorResponseCapturedEpisode:
    """Execute one exact Act A path and retain the complete source surface."""

    from .prepared_base import verify_cpu_runtime

    runtime_reasons = verify_cpu_runtime()
    if runtime_reasons:
        raise RuntimeError(",".join(runtime_reasons))
    cell = next(value for value in config.cells if value.cell_id == cell_id)
    view = next(value for value in config.views if value.view_id == view_id)
    word = config.baseline_word
    started = time.monotonic()
    reason_codes: set[str] = set()
    disposition = EpisodeDisposition.COMPLETE
    state_clock_s: list[int] = []
    field_values: dict[tuple[str, str], list[npt.NDArray[np.float64]]] = {}
    numerics_values: dict[str, list[npt.NDArray[np.float64]]] = {}
    expected_shapes: dict[tuple[str, str], tuple[int, ...]] = {}
    requested_values: list[DenominatorResponseNativeAction] = []
    accepted_values: list[DenominatorResponseNativeAction] = []
    applied_values: list[DenominatorResponseNativeAction] = []
    realized_values: list[DenominatorResponseNativeAction] = []
    clipped_clock_s: list[int] = []
    metadata: dict[tuple[str, str], tuple[str, tuple[str, ...]]] = {}
    coordinates: dict[str, npt.NDArray[np.float64]] = {}
    environment: Any | None = None
    try:
        environment = _environment(cell, view)
        environment.reset(seed=cell.environment_seed)
        metadata, coordinates = _source_metadata(environment)
        state_clock_s.append(0)
        _capture_state(
            environment.state or {},
            metadata=metadata,
            field_values=field_values,
            expected_shapes=expected_shapes,
            reason_codes=reason_codes,
        )
        for field_id, values in _numerics_values(environment).items():
            numerics_values.setdefault(field_id, []).append(values)

        for row in word.rows:
            request_clock = row.request_clock_s
            if int(round(float(environment.current_time))) != request_clock:
                disposition = EpisodeDisposition.TECHNICAL_OBSERVATION_FAILURE
                reason_codes.add("NATIVE_CLOCK_DIVERGENCE")
                break
            requested = row.action
            _, reward, terminated, truncated, info = environment.step(_action_mapping(requested))
            accepted = requested
            applied = _native_action(environment.torax_app.config.get_current_action_values())
            realized = _realized_action(environment, applied)
            requested_values.append(requested)
            accepted_values.append(accepted)
            applied_values.append(applied)
            realized_values.append(realized)
            receiver_clock = request_clock + 1
            state_clock_s.append(receiver_clock)
            _capture_state(
                environment.state or {},
                metadata=metadata,
                field_values=field_values,
                expected_shapes=expected_shapes,
                reason_codes=reason_codes,
            )
            observed_numerics = _numerics_values(environment)
            prior_clock_count = len(state_clock_s) - 1
            for field_id in set(numerics_values) | set(observed_numerics):
                if field_id in observed_numerics:
                    if field_id not in numerics_values:
                        numerics_values[field_id] = [
                            np.full_like(observed_numerics[field_id], np.nan)
                            for _ in range(prior_clock_count)
                        ]
                        reason_codes.add("NUMERICS_FIELD_APPEARED_AFTER_INITIALIZATION")
                    numerics_values[field_id].append(observed_numerics[field_id])
                else:
                    reason_codes.add("NUMERICS_FIELD_MISSING_AFTER_INITIALIZATION")
                    previous = numerics_values[field_id][-1]
                    numerics_values[field_id].append(np.full_like(previous, np.nan))
            if bool(info.get("action_clipped", False)):
                clipped_clock_s.append(request_clock)
                disposition = EpisodeDisposition.DELIVERY_INVALID
                reason_codes.add("UNEXPECTED_NATIVE_CLIPPING")
            decisive_field_ids = {value.source_field_id for value in config.receiver_groups}
            state_finite = all(
                np.isfinite(np.asarray(raw, dtype=np.float64)).all()
                for field_id, raw in (environment.state or {}).get("profiles", {}).items()
                if field_id in decisive_field_ids
            )
            if float(reward) == -1000.0 or not state_finite:
                disposition = EpisodeDisposition.NUMERICAL_INVALID
                reason_codes.add("NONFINITE_OR_FAILED_SIMULATION")
            if terminated or truncated:
                if receiver_clock < DENOMINATOR_RESPONSE_HORIZON_S:
                    if disposition is EpisodeDisposition.COMPLETE:
                        disposition = EpisodeDisposition.SIMULATOR_TERMINATED
                    reason_codes.add(
                        "SIMULATOR_TERMINATED_BEFORE_HORIZON"
                        if terminated
                        else "SIMULATOR_TRUNCATED_BEFORE_HORIZON"
                    )
                break
    except Exception as error:
        if state_clock_s:
            disposition = EpisodeDisposition.NUMERICAL_INVALID
            reason_codes.add(f"SIMULATOR_{type(error).__name__.upper()}")
        else:
            disposition = EpisodeDisposition.TECHNICAL_OBSERVATION_FAILURE
            reason_codes.add(f"TECHNICAL_INITIALIZATION_{type(error).__name__.upper()}")
    finally:
        if environment is not None:
            environment.close()

    state_clocks = tuple(state_clock_s)
    blocks = _make_state_blocks(
        state_clock_s=state_clocks,
        metadata=metadata,
        coordinates=coordinates,
        field_values=field_values,
        numerics_values=numerics_values,
        requested=requested_values,
        accepted=accepted_values,
        applied=applied_values,
        realized=realized_values,
        reason_codes=reason_codes,
    )
    missing_clocks = tuple(value for value in range(DENOMINATOR_RESPONSE_HORIZON_S + 1) if value not in state_clocks)
    if missing_clocks and disposition is EpisodeDisposition.COMPLETE:
        disposition = EpisodeDisposition.PARTIAL_VALID_PREFIX
        reason_codes.add("REQUIRED_CLOCKS_ABSENT")
    decisive_fields = {value.source_field_id for value in config.receiver_groups}
    available_profiles = {
        value.native_field_id for value in blocks if value.category == "source-profile"
    }
    if decisive_fields - available_profiles:
        disposition = EpisodeDisposition.UNEVALUABLE_OPERAND
    invalid_nonfinite_categories = {
        "action-accepted",
        "action-applied",
        "action-realized",
        "action-requested",
        "common-profile",
        "coordinate",
    }
    decisive_nonfinite = any(
        value.nonfinite_value_count
        and (
            value.category in invalid_nonfinite_categories
            or (value.category == "source-profile" and value.native_field_id in decisive_fields)
        )
        for value in blocks
    )
    if decisive_nonfinite:
        if disposition is EpisodeDisposition.COMPLETE:
            disposition = EpisodeDisposition.NUMERICAL_INVALID
        reason_codes.add("NONFINITE_DECISIVE_VALUE")
    if disposition is EpisodeDisposition.COMPLETE and reason_codes:
        disposition = EpisodeDisposition.UNEVALUABLE_OPERAND

    return DenominatorResponseCapturedEpisode(
        episode_id=(
            f"episode.denominator-qualification.{config.stage.value.lower().replace('_', '-')}."
            f"{cell.cell_id.removeprefix('cell.denominator-response-')}."
            f"{view.view_id.removeprefix('view.denominator-response.')}"
        ),
        config=ObjectIdentity.from_record(config.config_id, config),
        cell=ObjectIdentity.from_record(cell.cell_id, cell),
        view=ObjectIdentity.from_record(view.view_id, view),
        word=ObjectIdentity.from_record(word.word_id, word),
        state_clock_s=state_clocks,
        action_request_clock_s=tuple(range(len(requested_values))),
        blocks=blocks,
        source_profile_field_ids=tuple(
            sorted(value.native_field_id for value in blocks if value.category == "source-profile")
        ),
        source_scalar_field_ids=tuple(
            sorted(value.native_field_id for value in blocks if value.category == "source-scalar")
        ),
        source_numerics_field_ids=tuple(
            sorted(value.native_field_id for value in blocks if value.category == "source-numerics")
        ),
        source_unavailable_operand_ids=plan_time_source_inventory().known_source_unavailable_operand_ids,
        action_clipped_clock_s=tuple(clipped_clock_s),
        disposition=disposition,
        last_valid_clock_s=max(state_clocks) if state_clocks else None,
        missing_required_state_clock_s=missing_clocks,
        reason_codes=tuple(sorted(reason_codes)),
        runtime_seconds=Decimal(f"{time.monotonic() - started:.9f}"),
        backend="cpu",
        precision="float64",
        outcome_access=config.outcome_access,
    )


def _normalized_discrepancy(
    left: npt.NDArray[np.float64],
    right: npt.NDArray[np.float64],
    *,
    scale: Decimal,
) -> Decimal:
    denominator = np.maximum(np.abs(right), float(scale))
    value = float(np.max(np.abs(left - right) / denominator))
    return Decimal(str(value))


def _common_profile(
    episode: DenominatorResponseCapturedEpisode,
    field_id: str,
) -> npt.NDArray[np.float64]:
    return episode.block("common-profile", field_id).array()


def _native_map_discrepancy(
    episode: DenominatorResponseCapturedEpisode,
    group: DenominatorResponseReceiverGroup,
) -> Decimal:
    source = episode.block("source-profile", group.source_field_id)
    common = episode.block("common-profile", group.source_field_id)
    coordinate = episode.block("coordinate", source.dimension_ids[1]).array().reshape(-1)
    common_coordinate = episode.block("coordinate", "common_rho_norm").array().reshape(-1)
    native_values = source.array()
    common_values = common.array()
    clock_indices = [
        episode.state_clock_s.index(value)
        for value in group.decisive_clock_s
        if value in episode.state_clock_s
    ]
    maximum = 0.0
    for index in clock_indices:
        native_row = native_values[index]
        common_row = common_values[index]
        native_integral = float(np.trapezoid(native_row, coordinate))
        common_integral = float(np.trapezoid(common_row, common_coordinate))
        denominator = max(abs(native_integral), float(group.hybrid_absolute_scale))
        residuals = (
            abs(native_integral - common_integral) / denominator,
            abs(native_row[0] - common_row[0])
            / max(abs(native_row[0]), float(group.hybrid_absolute_scale)),
            abs(native_row[-1] - common_row[-1])
            / max(abs(native_row[-1]), float(group.hybrid_absolute_scale)),
        )
        maximum = max(maximum, *residuals)
    result = Decimal(str(maximum))
    return Decimal(0) if result <= DENOMINATOR_RESPONSE_NUMERICAL_ZERO_NORMALIZED else result


def _cube_interaction_max(
    *,
    episodes_by_levels: Mapping[tuple[int, int, int], DenominatorResponseCapturedEpisode],
    low: int,
    high: int,
    groups: Sequence[DenominatorResponseReceiverGroup],
) -> Decimal:
    maximum = 0.0
    for group in groups:
        arrays = {
            levels: _common_profile(episode, group.source_field_id)
            for levels, episode in episodes_by_levels.items()
            if set(levels).issubset({low, high})
        }
        expected_levels = set(itertools.product((low, high), (low, high), (low, high)))
        if set(arrays) != expected_levels:
            return Decimal("1e999")
        clock_indices = tuple(value for value in group.decisive_clock_s)
        sample = next(iter(arrays.values()))
        indices = [
            episodes_by_levels[(low, low, low)].state_clock_s.index(value)
            for value in clock_indices
        ]
        sliced = {key: value[indices] for key, value in arrays.items()}
        reference = sliced[(low, low, low)]
        denominator = np.maximum(np.abs(reference), float(group.hybrid_absolute_scale))
        for axes in ((0, 1), (0, 2), (1, 2)):
            context_axis = next(value for value in range(3) if value not in axes)
            for context in (low, high):
                corners = {}
                for left in (low, high):
                    for right in (low, high):
                        levels = [low, low, low]
                        levels[axes[0]] = left
                        levels[axes[1]] = right
                        levels[context_axis] = context
                        levels_key = (levels[0], levels[1], levels[2])
                        corners[(left, right)] = sliced[levels_key]
                interaction = (
                    corners[(high, high)]
                    - corners[(high, low)]
                    - corners[(low, high)]
                    + corners[(low, low)]
                )
                maximum = max(
                    maximum,
                    float(np.max(np.abs(interaction) / denominator)),
                )
        third = (
            sliced[(high, high, high)]
            - sliced[(high, high, low)]
            - sliced[(high, low, high)]
            - sliced[(low, high, high)]
            + sliced[(high, low, low)]
            + sliced[(low, high, low)]
            + sliced[(low, low, high)]
            - sliced[(low, low, low)]
        )
        maximum = max(maximum, float(np.max(np.abs(third) / denominator)))
        if sample.shape[0] != len(episodes_by_levels[(low, low, low)].state_clock_s):
            raise ValueError("denominator response interaction source clock count changed")
    result = Decimal(str(maximum))
    return Decimal(0) if result <= DENOMINATOR_RESPONSE_NUMERICAL_ZERO_NORMALIZED else result


def _pair_metrics(
    *,
    config: DenominatorResponseQualificationConfig,
    pair: DenominatorResponseViewPair,
    cell_id: str,
    episodes: Sequence[DenominatorResponseCapturedEpisode],
    development: bool,
) -> DenominatorResponseQualificationCellPairMetrics:
    by_view_id = {value.view.object_id: value for value in episodes}
    expected_view_ids = (
        {value.view_id for value in config.views} if development else set(pair.evaluation_view_ids)
    )
    all_view_records = {value.view_id: value for value in all_views()}
    by_levels = {
        all_view_records[view_id].levels: episode for view_id, episode in by_view_id.items()
    }
    groups = config.receiver_groups
    reasons: set[str] = set()
    complete_bundle = (
        len(episodes) == len(expected_view_ids)
        and set(by_view_id) == expected_view_ids
        and all(value.disposition is EpisodeDisposition.COMPLETE for value in episodes)
    )
    source_complete = bool(episodes) and all(
        all(
            group.source_field_id in value.source_profile_field_ids
            and ("common-profile", group.source_field_id)
            in {(block.category, block.native_field_id) for block in value.blocks}
            for group in groups
        )
        for value in episodes
    )
    compatibility: list[tuple[str, Decimal]] = []
    coarse_middle: list[tuple[str, Decimal]] = []
    middle_fine: list[tuple[str, Decimal]] = []
    map_values: list[tuple[str, Decimal]] = []
    if complete_bundle and source_complete:
        primary = by_view_id[pair.primary_view_id]
        finer = by_view_id[pair.finer_view_id]
        coarse = by_levels.get(_COARSE_LEVELS)
        middle = by_levels.get(_MIDDLE_LEVELS)
        fine = by_levels.get(_FINE_LEVELS)
        for group in groups:
            indices = [primary.state_clock_s.index(value) for value in group.decisive_clock_s]
            compatibility.append(
                (
                    group.group_id,
                    _normalized_discrepancy(
                        _common_profile(primary, group.source_field_id)[indices],
                        _common_profile(finer, group.source_field_id)[indices],
                        scale=group.hybrid_absolute_scale,
                    ),
                )
            )
            if development and coarse is not None and middle is not None and fine is not None:
                coarse_indices = [
                    coarse.state_clock_s.index(value) for value in group.decisive_clock_s
                ]
                coarse_middle.append(
                    (
                        group.group_id,
                        _normalized_discrepancy(
                            _common_profile(coarse, group.source_field_id)[coarse_indices],
                            _common_profile(middle, group.source_field_id)[coarse_indices],
                            scale=group.hybrid_absolute_scale,
                        ),
                    )
                )
                middle_fine.append(
                    (
                        group.group_id,
                        _normalized_discrepancy(
                            _common_profile(middle, group.source_field_id)[coarse_indices],
                            _common_profile(fine, group.source_field_id)[coarse_indices],
                            scale=group.hybrid_absolute_scale,
                        ),
                    )
                )
            map_values.append(
                (
                    group.group_id,
                    max(_native_map_discrepancy(value, group) for value in episodes),
                )
            )
    if not complete_bundle:
        reasons.add("INCOMPLETE_CELL_VIEW_BUNDLE")
    if not source_complete:
        reasons.add("DECISIVE_SOURCE_FIELD_ABSENT")
    compatibility_pass = bool(compatibility) and all(
        value <= Decimal("0.02") for _, value in compatibility
    )
    map_pass = bool(map_values) and all(
        value <= config.tolerances.map_normalized for _, value in map_values
    )
    refinement_pass: bool | None = None
    coarse_interaction: Decimal | None = None
    fine_interaction: Decimal | None = None
    selected_interaction = Decimal("1e999")
    if complete_bundle and source_complete:
        selected_low = 0 if pair.pair_id is DenominatorResponsePairId.COARSE_TO_MIDDLE else 1
        selected_high = selected_low + 1
        selected_interaction = _cube_interaction_max(
            episodes_by_levels=by_levels,
            low=selected_low,
            high=selected_high,
            groups=groups,
        )
        if development:
            coarse_interaction = _cube_interaction_max(
                episodes_by_levels=by_levels,
                low=0,
                high=1,
                groups=groups,
            )
            fine_interaction = _cube_interaction_max(
                episodes_by_levels=by_levels,
                low=1,
                high=2,
                groups=groups,
            )
            refinement_pass = bool(coarse_middle) and all(
                fine_value <= config.tolerances.contraction_ratio * coarse_value
                for (_, coarse_value), (_, fine_value) in zip(
                    coarse_middle, middle_fine, strict=True
                )
            )
    interaction_pass = selected_interaction <= config.tolerances.interaction_normalized
    if development and coarse_interaction is not None and fine_interaction is not None:
        interaction_pass = interaction_pass and (
            fine_interaction <= config.tolerances.contraction_ratio * coarse_interaction
        )
    if not compatibility_pass:
        reasons.add("PAIR_COMPATIBILITY_FAILED")
    if refinement_pass is False:
        reasons.add("THREE_LEVEL_REFINEMENT_FAILED")
    if not interaction_pass:
        reasons.add("LOCAL_INTERACTION_FAILED")
    if not map_pass:
        reasons.add("OBSERVATION_MAP_FAILED")
    return DenominatorResponseQualificationCellPairMetrics(
        metric_id=(f"metric.denominator-qualification.{cell_id.removeprefix('cell.denominator-response-')}.{pair.pair_id.value}"),
        cell_id=cell_id,
        pair_id=pair.pair_id,
        complete_bundle=complete_bundle,
        source_complete=source_complete,
        map_pass=map_pass,
        compatibility_pass=compatibility_pass,
        refinement_pass=refinement_pass,
        interaction_pass=interaction_pass,
        compatibility_max_by_group=tuple(sorted(compatibility)),
        coarse_middle_max_by_group=tuple(sorted(coarse_middle)),
        middle_fine_max_by_group=tuple(sorted(middle_fine)),
        map_max_by_group=tuple(sorted(map_values)),
        selected_cube_interaction_max=selected_interaction,
        coarse_cube_interaction_max=coarse_interaction,
        fine_cube_interaction_max=fine_interaction,
        reason_codes=tuple(sorted(reasons)),
    )


def _terminal_result(metrics: Sequence[DenominatorResponseQualificationCellPairMetrics]) -> DenominatorResponseQualificationResult:
    if not metrics or not any(value.complete_bundle for value in metrics):
        return DenominatorResponseQualificationResult.TECHNICAL_INVALID
    if any(not value.complete_bundle for value in metrics):
        return DenominatorResponseQualificationResult.PARTIAL_SCIENTIFIC
    if any(not value.source_complete for value in metrics):
        return DenominatorResponseQualificationResult.SOURCE_UNAVAILABLE
    if any(not value.map_pass for value in metrics):
        return DenominatorResponseQualificationResult.OBSERVATION_MAP_SENSITIVE
    if all(not value.compatibility_pass for value in metrics):
        return DenominatorResponseQualificationResult.NO_COMPATIBLE_DENOMINATOR
    if any(value.refinement_pass is False for value in metrics):
        return DenominatorResponseQualificationResult.NO_REFINEMENT_EVIDENCE
    if any(not value.interaction_pass for value in metrics):
        return DenominatorResponseQualificationResult.NUMERICAL_INTERACTION
    return DenominatorResponseQualificationResult.NO_COMPATIBLE_DENOMINATOR


def evaluate_denominator_qualification_development_cell(
    config: DenominatorResponseQualificationConfig,
    cell_id: str,
    episodes: Sequence[DenominatorResponseCapturedEpisode],
) -> tuple[DenominatorResponseQualificationCellPairMetrics, ...]:
    if config.stage is not DenominatorResponseStage.NUMERICAL_QUALIFICATION_DEVELOPMENT:
        raise ValueError("denominator response development evaluator received another stage")
    if cell_id not in {value.cell_id for value in config.cells}:
        raise ValueError("denominator response development evaluator received an inactive cell")
    expected_config = ObjectIdentity.from_record(config.config_id, config)
    if any(
        value.config != expected_config or value.cell.object_id != cell_id for value in episodes
    ):
        raise ValueError("denominator response development episode binds another config")
    return tuple(
        sorted(
            (
                _pair_metrics(
                    config=config,
                    pair=pair,
                    cell_id=cell_id,
                    episodes=episodes,
                    development=True,
                )
                for pair in config.candidate_pairs
            ),
            key=lambda value: value.metric_id,
        )
    )


def adjudicate_denominator_qualification_development_metrics(
    config: DenominatorResponseQualificationConfig,
    metrics: Sequence[DenominatorResponseQualificationCellPairMetrics],
    *,
    effective_cell_ids: tuple[str, ...] | None = None,
) -> DenominatorResponseQualificationDevelopmentDecision:
    if config.stage is not DenominatorResponseStage.NUMERICAL_QUALIFICATION_DEVELOPMENT:
        raise ValueError("denominator response development evaluator received another stage")
    expected_config = ObjectIdentity.from_record(config.config_id, config)
    cells = (
        config.primary_cell_ids if effective_cell_ids is None else tuple(sorted(effective_cell_ids))
    )
    if len(cells) != len(config.primary_cell_ids) or not set(cells).issubset(
        {value.cell_id for value in config.cells}
    ):
        raise ValueError("denominator response development effective cohort is invalid")
    expected_keys = {
        (cell_id, pair.pair_id) for cell_id in cells for pair in config.candidate_pairs
    }
    observed_keys = {(value.cell_id, value.pair_id) for value in metrics}
    if (
        observed_keys != expected_keys
        or len(metrics) != len(expected_keys)
        or len({value.metric_id for value in metrics}) != len(metrics)
    ):
        raise ValueError("denominator response development metrics do not cover the exact cohort")
    ordered_metrics = tuple(sorted(metrics, key=lambda value: value.metric_id))
    pair_passes = tuple(
        pair.pair_id
        for pair in config.candidate_pairs
        if all(
            value.complete_bundle
            and value.source_complete
            and value.map_pass
            and value.compatibility_pass
            and value.refinement_pass is True
            and value.interaction_pass
            for value in ordered_metrics
            if value.pair_id is pair.pair_id
        )
    )
    selected = pair_passes[0] if pair_passes else None
    terminal = None if selected is not None else _terminal_result(ordered_metrics)
    return DenominatorResponseQualificationDevelopmentDecision(
        decision_id='decision.denominator-qualification.development-to-evaluation',
        config=expected_config,
        selected_pair_id=selected,
        terminal_result=terminal,
        cell_pair_metrics=ordered_metrics,
        pair_pass_ids=pair_passes,
        evaluation_template_id=(
            None
            if selected is None
            else f"template.denominator-qualification.numerical-{selected.value}-evaluation"
        ),
        reason_codes=(
            () if selected is not None else (f"TERMINAL_{terminal.value}",)  # type: ignore[union-attr]
        ),
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
    )


def adjudicate_denominator_qualification_development(
    config: DenominatorResponseQualificationConfig,
    episodes: Sequence[DenominatorResponseCapturedEpisode],
) -> DenominatorResponseQualificationDevelopmentDecision:
    metrics = tuple(
        metric
        for cell_id in config.primary_cell_ids
        for metric in evaluate_denominator_qualification_development_cell(
            config,
            cell_id,
            tuple(value for value in episodes if value.cell.object_id == cell_id),
        )
    )
    return adjudicate_denominator_qualification_development_metrics(config, metrics)


def evaluate_denominator_qualification_evaluation_cell(
    config: DenominatorResponseQualificationConfig,
    cell_id: str,
    episodes: Sequence[DenominatorResponseCapturedEpisode],
) -> DenominatorResponseQualificationCellPairMetrics:
    if config.stage is not DenominatorResponseStage.NUMERICAL_QUALIFICATION_EVALUATION or config.selected_pair_id is None:
        raise ValueError("denominator response evaluation evaluator received another stage")
    if cell_id not in {value.cell_id for value in config.cells}:
        raise ValueError("denominator response evaluation evaluator received an inactive cell")
    expected_config = ObjectIdentity.from_record(config.config_id, config)
    if any(
        value.config != expected_config or value.cell.object_id != cell_id for value in episodes
    ):
        raise ValueError("denominator response evaluation episode binds another config")
    pair = next(
        value for value in config.candidate_pairs if value.pair_id is config.selected_pair_id
    )
    return _pair_metrics(
        config=config,
        pair=pair,
        cell_id=cell_id,
        episodes=episodes,
        development=False,
    )


def adjudicate_denominator_qualification_evaluation_metrics(
    config: DenominatorResponseQualificationConfig,
    metrics: Sequence[DenominatorResponseQualificationCellPairMetrics],
    *,
    effective_cell_ids: tuple[str, ...] | None = None,
) -> DenominatorResponseQualificationEvaluationAdjudication:
    if config.stage is not DenominatorResponseStage.NUMERICAL_QUALIFICATION_EVALUATION or config.selected_pair_id is None:
        raise ValueError("denominator response evaluation evaluator received another stage")
    expected_config = ObjectIdentity.from_record(config.config_id, config)
    cells = (
        config.primary_cell_ids if effective_cell_ids is None else tuple(sorted(effective_cell_ids))
    )
    if len(cells) != len(config.primary_cell_ids) or not set(cells).issubset(
        {value.cell_id for value in config.cells}
    ):
        raise ValueError("denominator response evaluation effective cohort is invalid")
    expected_keys = {(cell_id, config.selected_pair_id) for cell_id in cells}
    observed_keys = {(value.cell_id, value.pair_id) for value in metrics}
    if (
        observed_keys != expected_keys
        or len(metrics) != len(expected_keys)
        or len({value.metric_id for value in metrics}) != len(metrics)
    ):
        raise ValueError("denominator response evaluation metrics do not cover the exact cohort")
    ordered_metrics = tuple(sorted(metrics, key=lambda value: value.metric_id))
    qualified = all(
        value.complete_bundle
        and value.source_complete
        and value.map_pass
        and value.compatibility_pass
        and value.interaction_pass
        for value in ordered_metrics
    )
    primary = DenominatorResponseQualificationResult.DENOMINATOR_QUALIFIED if qualified else _terminal_result(ordered_metrics)
    complete_cells = tuple(
        sorted(value.cell_id for value in ordered_metrics if value.complete_bundle)
    )
    return DenominatorResponseQualificationEvaluationAdjudication(
        adjudication_id='adjudication.denominator-qualification.denominator-qualification',
        config=expected_config,
        selected_pair_id=config.selected_pair_id,
        primary_result=primary,
        cell_pair_metrics=ordered_metrics,
        complete_cell_ids=complete_cells,
        handoff_id=DENOMINATOR_RESPONSE_HANDOFF_ID if qualified else None,
        reason_codes=(() if qualified else (f"TERMINAL_{primary.value}",)),
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        maximum_evidence_ceiling=EvidenceCeiling.MEASUREMENT,
    )


def adjudicate_denominator_qualification_evaluation(
    config: DenominatorResponseQualificationConfig,
    episodes: Sequence[DenominatorResponseCapturedEpisode],
) -> DenominatorResponseQualificationEvaluationAdjudication:
    metrics = tuple(
        evaluate_denominator_qualification_evaluation_cell(
            config,
            cell_id,
            tuple(value for value in episodes if value.cell.object_id == cell_id),
        )
        for cell_id in config.primary_cell_ids
    )
    return adjudicate_denominator_qualification_evaluation_metrics(config, metrics)


__all__ = [
    "DENOMINATOR_RESPONSE_ACTION_COMPONENT_IDS",
    "DENOMINATOR_RESPONSE_ACTION_ANCHOR_S",
    "DENOMINATOR_RESPONSE_HANDOFF_ID",
    "DENOMINATOR_RESPONSE_HORIZON_S",
    "DENOMINATOR_RESPONSE_NUMERICAL_ZERO_NORMALIZED",
    'DenominatorResponseAcquisitionIndex',
    'DenominatorResponseQualificationConfig',
    'DenominatorResponseQualificationCellPairMetrics',
    'DenominatorResponseQualificationDevelopmentDecision',
    'DenominatorResponseQualificationEvaluationAdjudication',
    'DenominatorResponseQualificationExecutionReceipt',
    'DenominatorResponseQualificationHandoff',
    'DenominatorResponseQualificationIssue',
    'DenominatorResponseQualificationResourceEnvelope',
    'DenominatorResponseQualificationResult',
    'DenominatorResponseQualificationScientificApproval',
    'DenominatorResponseQualificationTolerances',
    'DenominatorResponseActionRow',
    'DenominatorResponseActionWord',
    'DenominatorResponseCell',
    'DenominatorResponseCapturedEpisode',
    'DenominatorResponseCellSubstitution',
    'DenominatorResponseEpisodeArtifactReceipt',
    'DenominatorResponseFloat64Block',
    'DenominatorResponseImplementationManifest',
    'DenominatorResponseNativeAction',
    'DenominatorResponseNumericalView',
    'DenominatorResponsePairId',
    'DenominatorResponsePlanTimeSourceInventory',
    'DenominatorResponseReceiverGroup',
    'DenominatorResponseSourceFile',
    'DenominatorResponseStage',
    'DenominatorResponseViewPair',
    'DenominatorResponseQualificationReceipt',
    'denominator_qualification_development_cells',
    'denominator_qualification_evaluation_cells',
    'followup_study_development_cells',
    'followup_study_evaluation_cells',
    'acquire_denominator_qualification_episode',
    'adjudicate_denominator_qualification_development',
    'adjudicate_denominator_qualification_development_metrics',
    'adjudicate_denominator_qualification_evaluation',
    'adjudicate_denominator_qualification_evaluation_metrics',
    "all_claim_bearing_cells",
    "all_views",
    "baseline_action",
    "baseline_word",
    'build_denominator_qualification_config',
    'build_denominator_qualification_formal_coverage',
    "candidate_pairs",
    'compile_denominator_qualification_resource_envelope',
    'compile_qualified_denominator_qualification_resource_envelope',
    "diagonal_views",
    "plan_time_source_inventory",
    "qualification_cells",
    "receiver_groups",
    'evaluate_denominator_qualification_development_cell',
    'evaluate_denominator_qualification_evaluation_cell',
    "validate_roster",
    "views_by_levels",
]
