"""Prepared-base Gym--TORAX campaign records, acquisition, and adjudication.

This adapter is the bounded current-version implementation for preparation-local
and paired-numerical-view experiments.  It owns Gym--TORAX source semantics but
does not own scheduling, persistence, issue, authority, or reveal.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
from enum import StrEnum
import math
import statistics
import time
from typing import Any, ClassVar, Mapping, Sequence
from types import MappingProxyType

import numpy as np

from empirical_lawhood.kernel.evidence import (
    EvidenceCeiling,
    OutcomeAccess,
    VisibilityCeiling,
)
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.status import ReadinessStatus
from empirical_lawhood.kernel.time import parse_utc_timestamp
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
)
from empirical_lawhood.kernel.worlds import EvidenceUnitScope

from .open_campaigns import OpenSimulatorSourceManifest


PREPARED_SOURCE_PARENT_ID = 'design.prepared-base.gym-torax-preparation-view-base'
PREPARED_SOURCE_DEVELOPMENT_CONFIG_ID = 'config.prepared-base.development'
PREPARED_SOURCE_EVALUATION_TEMPLATE_ID = 'template.prepared-base.evaluation'
PREPARED_SOURCE_EVALUATOR_ID = 'evaluator.prepared-base.preparation-view-base'
PREPARED_SOURCE_DECISION_RULE_ID = 'decision-rule.prepared-base-to-preparation-path-sensitivity'

PREPARED_SOURCE_REFERENCE_WORD_ID = 'word.prepared-base.installed-reference'
PREPARED_SOURCE_EARLY_ECRH_WORD_ID = 'word.prepared-base.early-ecrh'
PREPARED_SOURCE_EARLY_FULL_WORD_ID = 'word.prepared-base.early-full-heating'
PREPARED_SOURCE_CANDIDATE_WORD_IDS = (PREPARED_SOURCE_EARLY_ECRH_WORD_ID, PREPARED_SOURCE_EARLY_FULL_WORD_ID)

PREPARED_SOURCE_PRIMARY_VIEW_ID = 'view.prepared-base.primary-dt1-rho25-c10-linear-pc'
PREPARED_SOURCE_REFINED_VIEW_ID = 'view.prepared-base.refined-dt0p5-rho33-c20-linear-pc'
PREPARED_SOURCE_ALLOWED_EVALUATION_COUNTS = (14, 18, 24)
PREPARED_SOURCE_SOURCE_UNEVALUABLE_GAP_IDS = frozenset(
    {
        "gap.geometry.cohomology",
        "gap.geometry.information-geometry",
        "gap.geometry.metric-structure",
        "gap.geometry.topology-restriction",
    }
)

_QUANTUM = Decimal("0.0000001")
_ACTION_UNITS = {
    "action.ecrh-location": "rho_norm",
    "action.ecrh-power": "W",
    "action.ecrh-width": "rho_norm",
    "action.ip": "A",
    "action.nbi-location": "rho_norm",
    "action.nbi-power": "W",
    "action.nbi-width": "rho_norm",
}
_SCALAR_SOURCES = {
    "receiver.beta-n": ("beta_N", "1"),
    "receiver.fgw-line-average": ("fgw_n_e_line_avg", "1"),
    "receiver.fgw-volume-average": ("fgw_n_e_volume_avg", "1"),
    "receiver.h98": ("H98", "1"),
    "receiver.lcfs-current": ("Ip", "A"),
    "receiver.volume-average-density": ("n_e_volume_avg", "m-3"),
    "receiver.p-fusion": ("P_fusion", "W"),
    "receiver.p-heat-total": ("P_heat_total", "W"),
    "receiver.p-radiation-electron": ("P_radiation_e", "W"),
    "receiver.q-fusion": ("Q_fusion", "1"),
    "receiver.q-min": ("q_min", "1"),
    "receiver.q95": ("q95", "1"),
}
_PROFILE_SOURCES = {
    "profile.density-electron": ("n_e", "m-3"),
    "profile.poloidal-flux": ("psi", "Wb"),
    "profile.q": ("q", "1"),
}
_DIAGNOSTIC_CLOCKS = frozenset((*range(1, 31), *range(105, 115), 150))


class PreparedBaseStage(StrEnum):
    QUALIFICATION = "QUALIFICATION"
    DEVELOPMENT = "DEVELOPMENT"
    EVALUATION = "EVALUATION"


class PreparationWordKind(StrEnum):
    REFERENCE = "REFERENCE"
    PREPARATION = "PREPARATION"


class EpisodeDisposition(StrEnum):
    COMPLETE = "COMPLETE"
    TECHNICAL_OBSERVATION_FAILURE = "TECHNICAL_OBSERVATION_FAILURE"
    DELIVERY_INVALID = "DELIVERY_INVALID"
    SIMULATOR_TERMINATED = "SIMULATOR_TERMINATED"
    PRESERVATION_CROSSING = "PRESERVATION_CROSSING"
    NUMERICAL_INVALID = "NUMERICAL_INVALID"
    PARTIAL_VALID_PREFIX = "PARTIAL_VALID_PREFIX"
    UNEVALUABLE_OPERAND = "UNEVALUABLE_OPERAND"


class PreparedBasePrimaryResult(StrEnum):
    SUPPORTED_CROSS_VIEW_BASE = "SUPPORTED_CROSS_VIEW_BASE"
    VIEW_SENSITIVE_BASE = "VIEW_SENSITIVE_BASE"
    NO_SUPPORTED_PREPARATION = "NO_SUPPORTED_PREPARATION"
    PARTIAL_OR_TERMINATED = "PARTIAL_OR_TERMINATED"
    TECHNICAL_OBSERVATION_FAILURE = "TECHNICAL_OBSERVATION_FAILURE"
    UNEVALUABLE_OPERAND = "UNEVALUABLE_OPERAND"


@dataclass(frozen=True, slots=True)
class PreparedBaseCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/prepared-base-cell'

    cell_id: str
    stage: PreparedBaseStage
    reserve: bool
    environment_seed: int
    initial_temperature_scale: Decimal
    initial_density_nbar: Decimal
    bootstrap_multiplier: Decimal
    inner_transport_scale: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.cell_id, field_name="cell_id")
        if self.environment_seed < 0:
            raise ValueError("prepared-base environment seed must be nonnegative")
        for name, lower, upper in (
            ("initial_temperature_scale", Decimal("0.990"), Decimal("1.010")),
            ("initial_density_nbar", Decimal("0.848"), Decimal("0.852")),
            ("bootstrap_multiplier", Decimal("0.995"), Decimal("1.005")),
            ("inner_transport_scale", Decimal("0.990"), Decimal("1.010")),
        ):
            value = getattr(self, name)
            validate_decimal(value, field_name=name, minimum=lower)
            if value > upper:
                raise ValueError(f"{name} exceeds the declared prepared-base coordinate bounds")

    @property
    def coordinate_key(self) -> tuple[Decimal, Decimal, Decimal, Decimal]:
        return (
            self.initial_temperature_scale,
            self.initial_density_nbar,
            self.bootstrap_multiplier,
            self.inner_transport_scale,
        )


@dataclass(frozen=True, slots=True)
class PreparedBaseView(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/prepared-base-view'

    view_id: str
    internal_timestep_s: Decimal
    radial_cells: int
    corrector_steps: int
    solver_id: str
    predictor_corrector: bool
    pereverzev_enabled: bool
    pereverzev_chi: Decimal
    pereverzev_d: Decimal
    precision: str
    backend: str

    def __post_init__(self) -> None:
        validate_stable_id(self.view_id, field_name="view_id")
        if (
            self.internal_timestep_s not in {Decimal("1"), Decimal("0.5")}
            or self.solver_id != "solver.torax.linear-theta"
            or not self.predictor_corrector
            or not self.pereverzev_enabled
            or self.pereverzev_chi != Decimal("30")
            or self.pereverzev_d != Decimal("15")
            or self.precision != "float64"
            or self.backend != "cpu"
        ):
            raise ValueError("prepared-base view differs from the frozen CPU contract")
        expected = {
            PREPARED_SOURCE_PRIMARY_VIEW_ID: (25, 10, Decimal("1")),
            PREPARED_SOURCE_REFINED_VIEW_ID: (33, 20, Decimal("0.5")),
        }.get(self.view_id)
        if expected != (self.radial_cells, self.corrector_steps, self.internal_timestep_s):
            raise ValueError("prepared-base view identity differs from its resolution")


@dataclass(frozen=True, slots=True)
class NativeAction(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/native-action'

    ip_a: Decimal
    nbi_power_w: Decimal
    nbi_location: Decimal
    nbi_width: Decimal
    ecrh_power_w: Decimal
    ecrh_location: Decimal
    ecrh_width: Decimal

    def __post_init__(self) -> None:
        for name in (
            "ip_a",
            "nbi_power_w",
            "nbi_location",
            "nbi_width",
            "ecrh_power_w",
            "ecrh_location",
            "ecrh_width",
        ):
            validate_decimal(getattr(self, name), field_name=name)
        if (
            self.nbi_location != Decimal("0.25")
            or self.nbi_width != Decimal("0.25")
            or self.ecrh_location != Decimal("0.35")
            or self.ecrh_width != Decimal("0.05")
        ):
            raise ValueError("prepared-base action changes frozen deposition coordinates")

    def named(self) -> tuple[NamedDecimal, ...]:
        values = {
            "action.ecrh-location": self.ecrh_location,
            "action.ecrh-power": self.ecrh_power_w,
            "action.ecrh-width": self.ecrh_width,
            "action.ip": self.ip_a,
            "action.nbi-location": self.nbi_location,
            "action.nbi-power": self.nbi_power_w,
            "action.nbi-width": self.nbi_width,
        }
        return tuple(
            NamedDecimal(value_id=key, value=value, unit=_ACTION_UNITS[key])
            for key, value in sorted(values.items())
        )


@dataclass(frozen=True, slots=True)
class PreparationActionRow(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/preparation-action-row'

    row_id: str
    request_clock_s: int
    action: NativeAction

    def __post_init__(self) -> None:
        validate_stable_id(self.row_id, field_name="row_id")
        if not 0 <= self.request_clock_s < 150:
            raise ValueError("prepared-base action clock is outside the horizon")


@dataclass(frozen=True, slots=True)
class PreparationWord(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/preparation-word'

    word_id: str
    kind: PreparationWordKind
    added_pre_clock_100_energy_j: Decimal
    rows: tuple[PreparationActionRow, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.word_id, field_name="word_id")
        validate_decimal(
            self.added_pre_clock_100_energy_j,
            field_name="added_pre_clock_100_energy_j",
            minimum=Decimal(0),
        )
        require_sorted_unique_ids(self.rows, attribute="row_id", field_name="rows")
        if tuple(value.request_clock_s for value in self.rows) != tuple(range(150)):
            raise ValueError("prepared-base word must cover request clocks 0--149")
        if self.kind is PreparationWordKind.REFERENCE:
            if self.word_id != PREPARED_SOURCE_REFERENCE_WORD_ID:
                raise ValueError("prepared-base reference word identity changed")
        elif self.word_id not in PREPARED_SOURCE_CANDIDATE_WORD_IDS:
            raise ValueError("prepared-base candidate word is outside the frozen family")


@dataclass(frozen=True, slots=True)
class PreparedBaseConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/prepared-base-config'

    config_id: str
    parent_design_id: str
    stage: PreparedBaseStage
    source_manifest: ObjectIdentity
    cells: tuple[PreparedBaseCell, ...]
    primary_cell_ids: tuple[str, ...]
    reserve_cell_ids: tuple[str, ...]
    words: tuple[PreparationWord, ...]
    views: tuple[PreparedBaseView, ...]
    formal_coverage: ObjectIdentity
    evaluator_id: str
    decision_rule_id: str
    evaluation_count: int | None
    selected_word_id: str | None
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
        require_sorted_unique_ids(self.cells, attribute="cell_id", field_name="cells")
        require_sorted_unique_ids(self.words, attribute="word_id", field_name="words")
        require_sorted_unique_ids(self.views, attribute="view_id", field_name="views")
        require_sorted_unique_strings(
            self.primary_cell_ids, field_name="primary_cell_ids", allow_empty=False
        )
        require_sorted_unique_strings(
            self.reserve_cell_ids,
            field_name="reserve_cell_ids",
            allow_empty=self.stage is PreparedBaseStage.QUALIFICATION,
        )
        cell_ids = {value.cell_id for value in self.cells}
        if (
            set(self.primary_cell_ids) & set(self.reserve_cell_ids)
            or set(self.primary_cell_ids) | set(self.reserve_cell_ids) != cell_ids
            or any(
                value.reserve != (value.cell_id in set(self.reserve_cell_ids))
                or value.stage is not self.stage
                for value in self.cells
            )
        ):
            raise ValueError("prepared-base config has an invalid primary/reserve roster")
        if tuple(value.view_id for value in self.views) != (
            PREPARED_SOURCE_PRIMARY_VIEW_ID,
            PREPARED_SOURCE_REFINED_VIEW_ID,
        ):
            raise ValueError("prepared-base config changed the paired views")
        word_ids = {value.word_id for value in self.words}
        if self.stage is PreparedBaseStage.DEVELOPMENT:
            if (
                self.config_id != PREPARED_SOURCE_DEVELOPMENT_CONFIG_ID
                or word_ids != {PREPARED_SOURCE_REFERENCE_WORD_ID, *PREPARED_SOURCE_CANDIDATE_WORD_IDS}
                or len(self.primary_cell_ids) != 6
                or len(self.reserve_cell_ids) != 2
                or self.evaluation_count is not None
                or self.selected_word_id is not None
                or self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
            ):
                raise ValueError("prepared-base development config changed")
        elif self.stage is PreparedBaseStage.EVALUATION:
            if (
                self.evaluation_count not in PREPARED_SOURCE_ALLOWED_EVALUATION_COUNTS
                or self.selected_word_id not in PREPARED_SOURCE_CANDIDATE_WORD_IDS
                or word_ids != {self.selected_word_id}
                or len(self.primary_cell_ids) != self.evaluation_count
                or len(self.reserve_cell_ids) != math.ceil(0.1 * self.evaluation_count)
                or self.outcome_access is not OutcomeAccess.EVALUATION_SEALED
                or self.config_id
                != (
                    "config.prepared-base.evaluation-"
                    f"{self.selected_word_id.removeprefix('word.prepared-base.')}-"
                    f"{self.evaluation_count}"
                )
            ):
                raise ValueError("prepared-base evaluation config changed")
        elif (
            self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
            or self.maximum_evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
        ):
            raise ValueError("qualification config must remain non-promotable")
        if (
            self.stage is not PreparedBaseStage.QUALIFICATION
            and self.maximum_evidence_ceiling is not EvidenceCeiling.ORDER_RELATION
        ):
            raise ValueError("prepared-base primary exceeds its measurement or order-relation ceiling")


@dataclass(frozen=True, slots=True)
class DenseField(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/dense-field'

    field_id: str
    unit: str
    values: tuple[Decimal, ...]
    source_available: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.field_id, field_name="field_id")
        for value in self.values:
            validate_decimal(value, field_name=f"{self.field_id}.value")
        if self.source_available != bool(self.values):
            raise ValueError("dense field source-availability flag differs from values")


@dataclass(frozen=True, slots=True)
class ActionLedger(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/action-ledger'

    request_clock_s: int
    accepted_clock_s: int
    applied_clock_s: int
    realized_clock_s: int
    requested: NativeAction
    accepted: NativeAction
    applied: NativeAction
    realized: NativeAction
    clipped: bool
    realization_basis_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.realization_basis_id, field_name="realization_basis_id")
        if (
            self.accepted_clock_s != self.request_clock_s
            or self.applied_clock_s != self.request_clock_s + 1
            or self.realized_clock_s != self.request_clock_s + 1
        ):
            raise ValueError("prepared-base action-stage clocks changed")


@dataclass(frozen=True, slots=True)
class PreparedBaseTransition(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/prepared-base-transition'

    transition_id: str
    receiver_clock_s: int
    action: ActionLedger
    receiver_values: tuple[NamedDecimal, ...]
    diagnostic_fields: tuple[DenseField, ...]
    valid: bool
    terminated: bool
    truncated: bool
    missing_primary_operand_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.transition_id, field_name="transition_id")
        if not 1 <= self.receiver_clock_s <= 150:
            raise ValueError("prepared-base receiver clock is outside the horizon")
        require_sorted_unique_ids(
            self.receiver_values,
            attribute="value_id",
            field_name="receiver_values",
        )
        require_sorted_unique_ids(
            self.diagnostic_fields,
            attribute="field_id",
            field_name="diagnostic_fields",
        )
        require_sorted_unique_strings(
            self.missing_primary_operand_ids,
            field_name="missing_primary_operand_ids",
        )


@dataclass(frozen=True, slots=True)
class PreparedBaseEpisode(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/prepared-base-episode'

    episode_id: str
    config: ObjectIdentity
    cell: ObjectIdentity
    word: ObjectIdentity
    view: ObjectIdentity
    transitions: tuple[PreparedBaseTransition, ...]
    disposition: EpisodeDisposition
    last_valid_clock_s: int | None
    missing_required_clocks_s: tuple[int, ...]
    reason_codes: tuple[str, ...]
    runtime_seconds: Decimal
    backend: str
    precision: str
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.episode_id, field_name="episode_id")
        validate_decimal(self.runtime_seconds, field_name="runtime_seconds", minimum=Decimal(0))
        require_sorted_unique_ids(
            self.transitions, attribute="transition_id", field_name="transitions"
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if tuple(value.receiver_clock_s for value in self.transitions) != tuple(
            range(1, len(self.transitions) + 1)
        ):
            raise ValueError("prepared-base transitions are not a complete native prefix")
        if self.last_valid_clock_s is not None and not 1 <= self.last_valid_clock_s <= 150:
            raise ValueError("prepared-base last-valid clock is invalid")
        expected_missing = tuple(range(len(self.transitions) + 1, 151))
        if self.missing_required_clocks_s != expected_missing:
            raise ValueError("prepared-base missing-clock ledger changed")
        if self.disposition is EpisodeDisposition.COMPLETE and (
            len(self.transitions) != 150 or self.reason_codes
        ):
            raise ValueError("complete prepared-base episode is incomplete")
        if self.backend != "cpu" or self.precision != "float64":
            raise ValueError("prepared-base episode drifted from CPU float64")


@dataclass(frozen=True, slots=True)
class EpisodeAssessment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/episode-assessment'

    assessment_id: str
    episode: ObjectIdentity
    cell_id: str
    word_id: str
    view_id: str
    complete_contract_pass: bool
    worst_normalized_margin: Decimal | None
    maximum_phase_fgw: Decimal | None
    first_failed_predicate_id: str | None
    disposition: EpisodeDisposition
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("assessment_id", self.assessment_id),
            ("cell_id", self.cell_id),
            ("word_id", self.word_id),
            ("view_id", self.view_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.first_failed_predicate_id is not None:
            validate_stable_id(
                self.first_failed_predicate_id,
                field_name="first_failed_predicate_id",
            )
        if self.worst_normalized_margin is not None:
            validate_decimal(self.worst_normalized_margin, field_name="worst_normalized_margin")
        if self.maximum_phase_fgw is not None:
            validate_decimal(self.maximum_phase_fgw, field_name="maximum_phase_fgw")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")


@dataclass(frozen=True, slots=True)
class DevelopmentDecision(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/development-decision'

    decision_id: str
    development_config: ObjectIdentity
    selected_word_id: str | None
    selected_pass_count: int
    recurrence_lower: Decimal
    recurrence_upper: Decimal
    nuisance_sd: Decimal | None
    nuisance_sigma_upper: Decimal | None
    n_required: int | None
    evaluation_count: int | None
    disposition: str
    reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.decision_id, field_name="decision_id")
        if self.selected_word_id is not None:
            validate_stable_id(self.selected_word_id, field_name="selected_word_id")
        for name in (
            "recurrence_lower",
            "recurrence_upper",
            "nuisance_sd",
            "nuisance_sigma_upper",
        ):
            value = getattr(self, name)
            if value is not None:
                validate_decimal(value, field_name=name)
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
            raise ValueError("development decision must remain development-visible")


@dataclass(frozen=True, slots=True)
class EvaluationAdjudication(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/evaluation-adjudication'

    adjudication_id: str
    evaluation_config: ObjectIdentity
    primary_result: PreparedBasePrimaryResult
    pass_count: int
    unit_count: int
    recurrence_lower: Decimal
    recurrence_upper: Decimal
    one_sided_lower: Decimal
    confirmatory_view_difference_mean: Decimal | None
    selected_preparation_branch_id: str
    first_failed_predicate_id: str | None
    reason_codes: tuple[str, ...]
    maximum_evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        for name, value in (
            ("adjudication_id", self.adjudication_id),
            ("selected_preparation_branch_id", self.selected_preparation_branch_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.first_failed_predicate_id is not None:
            validate_stable_id(
                self.first_failed_predicate_id,
                field_name="first_failed_predicate_id",
            )
        for name in (
            "recurrence_lower",
            "recurrence_upper",
            "one_sided_lower",
            "confirmatory_view_difference_mean",
        ):
            value = getattr(self, name)
            if value is not None:
                validate_decimal(value, field_name=name)
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if (
            self.maximum_evidence_ceiling is not EvidenceCeiling.ORDER_RELATION
            or self.outcome_access is not OutcomeAccess.EVALUATION_REVEALED
            or self.visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE
        ):
            raise ValueError("Prepared-base adjudication changes its ceiling or reveal state")


@dataclass(frozen=True, slots=True)
class PreparedBaseQualificationReceipt(CanonicalRecord):
    """Excluded route receipt whose episode tuples align with ``view_ids``."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/prepared-base-qualification-receipt'

    receipt_id: str
    config: ObjectIdentity
    source_manifest: ObjectIdentity
    implementation_sha256: str
    episode_sha256: tuple[str, ...]
    episode_size_bytes: tuple[int, ...]
    episode_runtime_seconds: tuple[Decimal, ...]
    view_ids: tuple[str, ...]
    backend: str
    precision: str
    complete_route: bool
    reason_codes: tuple[str, ...]
    maximum_evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        validate_sha256(self.implementation_sha256, field_name="implementation_sha256")
        if len(set(self.episode_sha256)) != len(self.episode_sha256):
            raise ValueError("qualification episode hashes must be unique")
        for value in self.episode_sha256:
            validate_sha256(value, field_name="episode_sha256")
        require_sorted_unique_strings(self.view_ids, field_name="view_ids")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if (
            len(self.episode_sha256) != 2
            or len(self.episode_size_bytes) != 2
            or len(self.episode_runtime_seconds) != 2
            or self.view_ids != (PREPARED_SOURCE_PRIMARY_VIEW_ID, PREPARED_SOURCE_REFINED_VIEW_ID)
            or any(value <= 0 for value in self.episode_size_bytes)
            or self.backend != "cpu"
            or self.precision != "float64"
            or not self.complete_route
            or self.reason_codes
            or self.maximum_evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
            or self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
        ):
            raise ValueError("prepared-base qualification receipt is not readiness-only closure")
        for runtime_value in self.episode_runtime_seconds:
            validate_decimal(
                runtime_value,
                field_name="episode_runtime_seconds",
                minimum=Decimal(0),
            )


@dataclass(frozen=True, slots=True)
class PreparedBaseScientificApproval(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/prepared-base-scientific-approval'

    approval_id: str
    config: ObjectIdentity
    approver: ObjectIdentity
    approver_id: str
    authorization_basis_sha256: str
    passed_gate_ids: tuple[str, ...]
    approved_at_utc: str
    codex_or_chat_is_approver_or_issuer: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.approval_id, field_name="approval_id")
        validate_stable_id(self.approver_id, field_name="approver_id")
        if self.approver.object_id != self.approver_id:
            raise ValueError("prepared-base approval identity differs")
        validate_sha256(
            self.authorization_basis_sha256,
            field_name="authorization_basis_sha256",
        )
        require_sorted_unique_strings(
            self.passed_gate_ids,
            field_name="passed_gate_ids",
            allow_empty=False,
        )
        parse_utc_timestamp(self.approved_at_utc, field_name="approved_at_utc")
        if (
            self.codex_or_chat_is_approver_or_issuer
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
        ):
            raise ValueError("prepared-base approval is not independent and outcome-blind")


@dataclass(frozen=True, slots=True)
class PreparedBaseIssue(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/prepared-base-issue'

    issue_id: str
    config: ObjectIdentity
    scientific_approval: ObjectIdentity
    custody_authority: ObjectIdentity
    source_closure: ObjectIdentity
    qualification_receipt: ObjectIdentity
    parent_development_decision: ObjectIdentity | None
    proposer_role_id: str
    approver_role_id: str
    executor_role_id: str
    custodian_role_id: str
    evaluator_role_id: str
    issued_at_utc: str
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name, value in (
            ("issue_id", self.issue_id),
            ("proposer_role_id", self.proposer_role_id),
            ("approver_role_id", self.approver_role_id),
            ("executor_role_id", self.executor_role_id),
            ("custodian_role_id", self.custodian_role_id),
            ("evaluator_role_id", self.evaluator_role_id),
        ):
            validate_stable_id(value, field_name=name)
        parse_utc_timestamp(self.issued_at_utc, field_name="issued_at_utc")
        if (
            len(
                {
                    self.proposer_role_id,
                    self.approver_role_id,
                    self.executor_role_id,
                    self.custodian_role_id,
                    self.evaluator_role_id,
                }
            )
            != 5
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
        ):
            raise ValueError("prepared-base issue lacks role separation or blindness")


@dataclass(frozen=True, slots=True)
class PreparedBasePanel(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/prepared-base-panel'

    panel_id: str
    issue: ObjectIdentity
    config: ObjectIdentity
    episode_refs: tuple[ObjectIdentity, ...]
    expected_episode_count: int
    complete_roster: bool
    custodian_role_id: str
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.panel_id, field_name="panel_id")
        validate_stable_id(self.custodian_role_id, field_name="custodian_role_id")
        require_sorted_unique_ids(
            self.episode_refs,
            attribute="object_id",
            field_name="episode_refs",
        )
        if (
            self.expected_episode_count <= 0
            or len(self.episode_refs) != self.expected_episode_count
            or not self.complete_roster
            or self.outcome_access
            not in {OutcomeAccess.DEVELOPMENT_VISIBLE, OutcomeAccess.EVALUATION_SEALED}
        ):
            raise ValueError("prepared-base panel is incomplete")


@dataclass(frozen=True, slots=True)
class PreparedBaseCellSubstitution(CanonicalRecord):
    """Outcome-blind replacement of one complete technical-failure cell bundle."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/prepared-base-cell-substitution'

    substitution_id: str
    intended_cell_id: str
    replacement_cell_id: str
    excluded_episode_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    receiver_outcome_released_to_evaluator: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("substitution_id", self.substitution_id),
            ("intended_cell_id", self.intended_cell_id),
            ("replacement_cell_id", self.replacement_cell_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(
            self.excluded_episode_ids,
            field_name="excluded_episode_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.reason_codes,
            field_name="reason_codes",
            allow_empty=False,
        )
        if (
            self.intended_cell_id == self.replacement_cell_id
            or self.receiver_outcome_released_to_evaluator
        ):
            raise ValueError("prepared-base reserve substitution violates blinding")


@dataclass(frozen=True, slots=True)
class PreparedBaseAcquisitionBatch(CanonicalRecord):
    """One immutable stage acquisition with exact active-cell substitutions."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/prepared-base-acquisition-batch'

    batch_id: str
    config: ObjectIdentity
    stage: PreparedBaseStage
    episodes: tuple[PreparedBaseEpisode, ...]
    active_cell_ids: tuple[str, ...]
    excluded_cell_ids: tuple[str, ...]
    substitutions: tuple[PreparedBaseCellSubstitution, ...]
    expected_active_cell_count: int
    complete_active_roster: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.batch_id, field_name="batch_id")
        require_sorted_unique_ids(
            self.episodes,
            attribute="episode_id",
            field_name="episodes",
        )
        for name, values in (
            ("active_cell_ids", self.active_cell_ids),
            ("excluded_cell_ids", self.excluded_cell_ids),
        ):
            require_sorted_unique_strings(values, field_name=name)
        require_sorted_unique_ids(
            self.substitutions,
            attribute="substitution_id",
            field_name="substitutions",
        )
        if (
            self.expected_active_cell_count <= 0
            or bool(len(self.active_cell_ids) == self.expected_active_cell_count)
            != self.complete_active_roster
            or set(self.active_cell_ids) & set(self.excluded_cell_ids)
        ):
            raise ValueError("prepared-base acquisition batch has an invalid active roster")
        episode_cell_ids = {value.cell.object_id for value in self.episodes}
        if not set(self.active_cell_ids).issubset(episode_cell_ids):
            raise ValueError("prepared-base active cell lacks acquired evidence")
        if not set(self.excluded_cell_ids).issubset(episode_cell_ids):
            raise ValueError("prepared-base excluded cell lacks preserved evidence")
        if self.outcome_access not in {
            OutcomeAccess.DEVELOPMENT_VISIBLE,
            OutcomeAccess.EVALUATION_SEALED,
        }:
            raise ValueError("prepared-base batch has invalid outcome access")


_FROZEN_ENVIRONMENT_SEEDS: Mapping[str, int] = MappingProxyType({
    'cell.prepared-base-development-01': 480051934265821193,
    'cell.prepared-base-development-02': 7470899217754495341,
    'cell.prepared-base-development-03': 13099849997252847968,
    'cell.prepared-base-development-04': 12141567045527329554,
    'cell.prepared-base-development-05': 11008798650188809922,
    'cell.prepared-base-development-06': 16667914853026140043,
    'cell.prepared-base-development-07': 1053664285418010813,
    'cell.prepared-base-development-08': 4800880561995160760,
    'cell.prepared-base-evaluation-01': 11535152603732891706,
    'cell.prepared-base-evaluation-02': 16573738058044165555,
    'cell.prepared-base-evaluation-03': 16902535817555464338,
    'cell.prepared-base-evaluation-04': 8241526963166583401,
    'cell.prepared-base-evaluation-05': 15727333043277449930,
    'cell.prepared-base-evaluation-06': 1751674175265034832,
    'cell.prepared-base-evaluation-07': 5226039540576226438,
    'cell.prepared-base-evaluation-08': 1573516590911345631,
    'cell.prepared-base-evaluation-09': 5811180549030043708,
    'cell.prepared-base-evaluation-10': 901983844261481320,
    'cell.prepared-base-evaluation-11': 14307108284554216832,
    'cell.prepared-base-evaluation-12': 3884370378267703779,
    'cell.prepared-base-evaluation-13': 11454770605589154051,
    'cell.prepared-base-evaluation-14': 7612811648095166208,
    'cell.prepared-base-evaluation-15': 230059171168705055,
    'cell.prepared-base-evaluation-16': 9900217566957862349,
    'cell.prepared-base-evaluation-17': 7579811674181121190,
    'cell.prepared-base-evaluation-18': 5330418080786128077,
    'cell.prepared-base-evaluation-19': 12113345035637008276,
    'cell.prepared-base-evaluation-20': 2048976523963060298,
    'cell.prepared-base-evaluation-21': 14574035104709769785,
    'cell.prepared-base-evaluation-22': 15155030945044383645,
    'cell.prepared-base-evaluation-23': 16830781564542662757,
    'cell.prepared-base-evaluation-24': 10355956055550899646,
    'cell.prepared-base-reserve-01': 8000547873835674494,
    'cell.prepared-base-reserve-02': 12620590138014298910,
    'cell.prepared-base-reserve-03': 14182455442816593976,
    'cell.prepared-base-qualification-01': 5017894497388729670,
})


def _seed(cell_id: str) -> int:
    return _FROZEN_ENVIRONMENT_SEEDS[cell_id]


def _lhs(
    lower: Decimal,
    upper: Decimal,
    rank: int,
    count: int,
    offset: Decimal,
) -> Decimal:
    return (
        lower + (upper - lower) * (Decimal(rank) + Decimal("0.5")) / Decimal(count) + offset
    ).quantize(_QUANTUM, rounding=ROUND_HALF_EVEN)


def _cell(
    *,
    cell_id: str,
    stage: PreparedBaseStage,
    reserve: bool,
    ranks: tuple[int, int, int, int],
    count: int,
    offsets: tuple[Decimal, Decimal, Decimal, Decimal],
) -> PreparedBaseCell:
    intervals = (
        (Decimal("0.990"), Decimal("1.010")),
        (Decimal("0.848"), Decimal("0.852")),
        (Decimal("0.995"), Decimal("1.005")),
        (Decimal("0.990"), Decimal("1.010")),
    )
    values = tuple(
        _lhs(lower, upper, rank, count, offset)
        for (lower, upper), rank, offset in zip(intervals, ranks, offsets, strict=True)
    )
    return PreparedBaseCell(
        cell_id=cell_id,
        stage=stage,
        reserve=reserve,
        environment_seed=_seed(cell_id),
        initial_temperature_scale=values[0],
        initial_density_nbar=values[1],
        bootstrap_multiplier=values[2],
        inner_transport_scale=values[3],
    )


def development_cells() -> tuple[PreparedBaseCell, ...]:
    values = tuple(
        _cell(
            cell_id=f"cell.prepared-base-development-{index + 1:02d}",
            stage=PreparedBaseStage.DEVELOPMENT,
            reserve=index >= 6,
            ranks=(index, (5 * index + 1) % 8, (3 * index + 3) % 8, (7 * index + 2) % 8),
            count=8,
            offsets=(
                Decimal("0.0000001"),
                Decimal("0.0000002"),
                Decimal("0.0000003"),
                Decimal("0.0000004"),
            ),
        )
        for index in range(8)
    )
    _validate_cell_roster(values)
    return values


def evaluation_master_cells() -> tuple[PreparedBaseCell, ...]:
    values = tuple(
        _cell(
            cell_id=(
                f"cell.prepared-base-evaluation-{index + 1:02d}"
                if index < 24
                else f"cell.prepared-base-reserve-{index - 23:02d}"
            ),
            stage=PreparedBaseStage.EVALUATION,
            reserve=index >= 24,
            ranks=(
                index,
                (5 * index + 1) % 27,
                (7 * index + 3) % 27,
                (11 * index + 5) % 27,
            ),
            count=27,
            offsets=(
                Decimal("0.0000002"),
                Decimal("0.0000003"),
                Decimal("0.0000004"),
                Decimal("0.0000005"),
            ),
        )
        for index in range(27)
    )
    _validate_cell_roster((*development_cells(), *values))
    return values


def _validate_cell_roster(cells: Sequence[PreparedBaseCell]) -> None:
    keys = tuple(value.coordinate_key for value in cells)
    if len(keys) != len(set(keys)):
        raise ValueError("prepared-base roster repeats a denominator")
    # All immediate predecessor rosters were frozen on a six-decimal grid.
    # Each current tuple has at least one deliberate seventh-decimal residue.
    if any(all(value == value.quantize(Decimal("0.000001")) for value in key) for key in keys):
        raise ValueError("prepared-base cell lacks the frozen predecessor-disjoint residue")


def paired_views() -> tuple[PreparedBaseView, ...]:
    return (
        PreparedBaseView(
            view_id=PREPARED_SOURCE_PRIMARY_VIEW_ID,
            internal_timestep_s=Decimal("1"),
            radial_cells=25,
            corrector_steps=10,
            solver_id="solver.torax.linear-theta",
            predictor_corrector=True,
            pereverzev_enabled=True,
            pereverzev_chi=Decimal("30"),
            pereverzev_d=Decimal("15"),
            precision="float64",
            backend="cpu",
        ),
        PreparedBaseView(
            view_id=PREPARED_SOURCE_REFINED_VIEW_ID,
            internal_timestep_s=Decimal("0.5"),
            radial_cells=33,
            corrector_steps=20,
            solver_id="solver.torax.linear-theta",
            predictor_corrector=True,
            pereverzev_enabled=True,
            pereverzev_chi=Decimal("30"),
            pereverzev_d=Decimal("15"),
            precision="float64",
            backend="cpu",
        ),
    )


def _baseline_action(clock_s: int) -> NativeAction:
    ip = (
        Decimal("3000000")
        + Decimal(clock_s + 1) * (Decimal("12500000") - Decimal("3000000")) / Decimal(100)
        if clock_s < 99
        else Decimal("12500000")
    )
    full = clock_s >= 99
    return NativeAction(
        ip_a=ip,
        nbi_power_w=Decimal("33000000") if full else Decimal(0),
        nbi_location=Decimal("0.25"),
        nbi_width=Decimal("0.25"),
        ecrh_power_w=Decimal("20000000") if full else Decimal(0),
        ecrh_location=Decimal("0.35"),
        ecrh_width=Decimal("0.05"),
    )


def preparation_words() -> tuple[PreparationWord, ...]:
    specs = (
        (PREPARED_SOURCE_REFERENCE_WORD_ID, PreparationWordKind.REFERENCE, Decimal(0), False, False),
        (
            PREPARED_SOURCE_EARLY_ECRH_WORD_ID,
            PreparationWordKind.PREPARATION,
            Decimal("1400000000"),
            True,
            False,
        ),
        (
            PREPARED_SOURCE_EARLY_FULL_WORD_ID,
            PreparationWordKind.PREPARATION,
            Decimal("3710000000"),
            True,
            True,
        ),
    )
    result = []
    for word_id, kind, energy, early_ecrh, early_nbi in specs:
        rows = []
        for clock_s in range(150):
            baseline = _baseline_action(clock_s)
            early = 30 <= clock_s < 100
            action = NativeAction(
                ip_a=baseline.ip_a,
                nbi_power_w=(Decimal("33000000") if early and early_nbi else baseline.nbi_power_w),
                nbi_location=baseline.nbi_location,
                nbi_width=baseline.nbi_width,
                ecrh_power_w=(
                    Decimal("20000000") if early and early_ecrh else baseline.ecrh_power_w
                ),
                ecrh_location=baseline.ecrh_location,
                ecrh_width=baseline.ecrh_width,
            )
            rows.append(
                PreparationActionRow(
                    row_id=f"{word_id}.row-{clock_s:03d}",
                    request_clock_s=clock_s,
                    action=action,
                )
            )
        result.append(
            PreparationWord(
                word_id=word_id,
                kind=kind,
                added_pre_clock_100_energy_j=energy,
                rows=tuple(rows),
            )
        )
    return tuple(sorted(result, key=lambda value: value.word_id))


def build_prepared_source_formal_coverage(
    *,
    register: FormalGapRegister,
    source_manifest: OpenSimulatorSourceManifest,
    denominator_id: str,
    candidate_act_id: str,
    independent_unit_ids: tuple[str, ...],
) -> tuple[FormalGapSourceCapabilityInventory, FormalGapCoverage]:
    act_suffix = candidate_act_id.removeprefix("draft.")
    inventory = FormalGapSourceCapabilityInventory(
        inventory_id=f"formal-source-inventory.{act_suffix}",
        denominator_id=denominator_id,
        evidence_world=FormalGapEvidenceWorld.NUMERICAL_SIMULATOR,
        source_materializations=(
            ObjectIdentity.from_record(source_manifest.source_id, source_manifest),
        ),
        present_operand_ids=(),
        satisfied_prerequisite_ids=(),
        independent_unit_ids=tuple(sorted(independent_unit_ids)),
        independent_unit_scope=EvidenceUnitScope.PHYSICAL_INDEPENDENT_UNIT,
        numerical_view_ids=(PREPARED_SOURCE_PRIMARY_VIEW_ID, PREPARED_SOURCE_REFINED_VIEW_ID),
        available_estimator_family_ids=(),
        available_control_ids=(),
        multiplicity_family_ids=(),
        denominator_inapplicable_gap_ids=(),
        resource_blocked_gap_ids=(),
        requested_claim_ceiling=EvidenceCeiling.LOCAL_LAW,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    applicability = derive_formal_gap_applicability(register, inventory)
    assignments = tuple(
        FormalGapCoverageAssignment(
            gap_id=gap.gap_id,
            disposition=(
                FormalGapCoverageDisposition.UNEVALUABLE_FROM_SOURCE
                if gap.gap_id in PREPARED_SOURCE_SOURCE_UNEVALUABLE_GAP_IDS
                else FormalGapCoverageDisposition.DEFER_WITH_TYPED_PREREQUISITE
            ),
            readiness_reason=(
                ReadinessStatus.SOURCE_PREREQUISITE_NOT_MET
                if gap.gap_id in PREPARED_SOURCE_SOURCE_UNEVALUABLE_GAP_IDS
                else ReadinessStatus.PREREQUISITE_NOT_MET
            ),
            reason_codes=(
                ("PREPARED_RESPONSE_SOURCE_MATHEMATICAL_OBJECT_ABSENT",)
                if gap.gap_id in PREPARED_SOURCE_SOURCE_UNEVALUABLE_GAP_IDS
                else ("PREPARED_RESPONSE_CONTROLLED_LOCAL_LAW_PREREQUISITE_ABSENT",)
            ),
            selected_estimator_family_id=None,
            selected_control_ids=(),
            selected_multiplicity_family_id=None,
            obligation_ids=(),
            output_ids=(),
            adjudication_owner_ids=(),
        )
        for gap in register.gaps
    )
    coverage = FormalGapCoverage(
        coverage_id=f"formal-gap-coverage.{act_suffix}",
        register=ObjectIdentity.from_record(register.register_id, register),
        denominator_id=denominator_id,
        candidate_act_id=candidate_act_id,
        applicability=applicability,
        assignments=assignments,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    return inventory, coverage


def development_config(
    *,
    source_manifest: OpenSimulatorSourceManifest,
    formal_coverage: FormalGapCoverage,
) -> PreparedBaseConfig:
    cells = development_cells()
    return PreparedBaseConfig(
        config_id=PREPARED_SOURCE_DEVELOPMENT_CONFIG_ID,
        parent_design_id=PREPARED_SOURCE_PARENT_ID,
        stage=PreparedBaseStage.DEVELOPMENT,
        source_manifest=ObjectIdentity.from_record(source_manifest.source_id, source_manifest),
        cells=tuple(sorted(cells, key=lambda value: value.cell_id)),
        primary_cell_ids=tuple(sorted(value.cell_id for value in cells if not value.reserve)),
        reserve_cell_ids=tuple(sorted(value.cell_id for value in cells if value.reserve)),
        words=preparation_words(),
        views=paired_views(),
        formal_coverage=ObjectIdentity.from_record(formal_coverage.coverage_id, formal_coverage),
        evaluator_id=PREPARED_SOURCE_EVALUATOR_ID,
        decision_rule_id=PREPARED_SOURCE_DECISION_RULE_ID,
        evaluation_count=None,
        selected_word_id=None,
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        maximum_evidence_ceiling=EvidenceCeiling.ORDER_RELATION,
    )


def qualification_config(
    *,
    source_manifest: OpenSimulatorSourceManifest,
    formal_coverage: FormalGapCoverage,
) -> PreparedBaseConfig:
    cell = PreparedBaseCell(
        cell_id='cell.prepared-base-qualification-01',
        stage=PreparedBaseStage.QUALIFICATION,
        reserve=False,
        environment_seed=_seed('cell.prepared-base-qualification-01'),
        initial_temperature_scale=Decimal("1.0000001"),
        initial_density_nbar=Decimal("0.8500002"),
        bootstrap_multiplier=Decimal("1.0000003"),
        inner_transport_scale=Decimal("1.0000004"),
    )
    word = next(value for value in preparation_words() if value.word_id == PREPARED_SOURCE_EARLY_ECRH_WORD_ID)
    return PreparedBaseConfig(
        config_id='config.prepared-base.qualification',
        parent_design_id=PREPARED_SOURCE_PARENT_ID,
        stage=PreparedBaseStage.QUALIFICATION,
        source_manifest=ObjectIdentity.from_record(source_manifest.source_id, source_manifest),
        cells=(cell,),
        primary_cell_ids=(cell.cell_id,),
        reserve_cell_ids=(),
        words=(word,),
        views=paired_views(),
        formal_coverage=ObjectIdentity.from_record(formal_coverage.coverage_id, formal_coverage),
        evaluator_id=PREPARED_SOURCE_EVALUATOR_ID,
        decision_rule_id=PREPARED_SOURCE_DECISION_RULE_ID,
        evaluation_count=None,
        selected_word_id=None,
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        maximum_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
    )


def evaluation_config(
    *,
    source_manifest: OpenSimulatorSourceManifest,
    formal_coverage: FormalGapCoverage,
    selected_word_id: str,
    evaluation_count: int,
) -> PreparedBaseConfig:
    master = evaluation_master_cells()
    primary = tuple(
        value
        for value in master
        if not value.reserve and int(value.cell_id.rsplit("-", 1)[-1]) <= evaluation_count
    )
    reserve_count = math.ceil(0.1 * evaluation_count)
    reserves = tuple(value for value in master if value.reserve)[:reserve_count]
    word = next(value for value in preparation_words() if value.word_id == selected_word_id)
    return PreparedBaseConfig(
        config_id=(
            "config.prepared-base.evaluation-"
            f"{selected_word_id.removeprefix('word.prepared-base.')}-{evaluation_count}"
        ),
        parent_design_id=PREPARED_SOURCE_PARENT_ID,
        stage=PreparedBaseStage.EVALUATION,
        source_manifest=ObjectIdentity.from_record(source_manifest.source_id, source_manifest),
        cells=tuple(sorted((*primary, *reserves), key=lambda value: value.cell_id)),
        primary_cell_ids=tuple(sorted(value.cell_id for value in primary)),
        reserve_cell_ids=tuple(sorted(value.cell_id for value in reserves)),
        words=(word,),
        views=paired_views(),
        formal_coverage=ObjectIdentity.from_record(formal_coverage.coverage_id, formal_coverage),
        evaluator_id=PREPARED_SOURCE_EVALUATOR_ID,
        decision_rule_id=PREPARED_SOURCE_DECISION_RULE_ID,
        evaluation_count=evaluation_count,
        selected_word_id=selected_word_id,
        outcome_access=OutcomeAccess.EVALUATION_SEALED,
        maximum_evidence_ceiling=EvidenceCeiling.ORDER_RELATION,
    )


def _decimal(value: Any) -> Decimal:
    number = float(np.asarray(value, dtype=np.float64).reshape(-1)[0])
    if not math.isfinite(number):
        raise ValueError("Gym--TORAX produced a non-finite scalar")
    return Decimal(f"{number:.17g}")


def _native_action(values: Mapping[str, Any]) -> NativeAction:
    if set(values) != {"Ip", "NBI", "ECRH"}:
        raise ValueError("Gym--TORAX action mapping changed")
    ip = np.asarray(values["Ip"], dtype=np.float64).reshape(-1)
    nbi = np.asarray(values["NBI"], dtype=np.float64).reshape(-1)
    ecrh = np.asarray(values["ECRH"], dtype=np.float64).reshape(-1)
    if ip.shape != (1,) or nbi.shape != (3,) or ecrh.shape != (3,):
        raise ValueError("Gym--TORAX action vector shape changed")

    def fixed(value: Any, expected: Decimal) -> Decimal:
        observed = _decimal(value)
        return expected if abs(observed - expected) <= Decimal("0.000000000000001") else observed

    return NativeAction(
        ip_a=_decimal(ip[0]),
        nbi_power_w=_decimal(nbi[0]),
        nbi_location=fixed(nbi[1], Decimal("0.25")),
        nbi_width=fixed(nbi[2], Decimal("0.25")),
        ecrh_power_w=_decimal(ecrh[0]),
        ecrh_location=fixed(ecrh[1], Decimal("0.35")),
        ecrh_width=fixed(ecrh[2], Decimal("0.05")),
    )


def _action_mapping(action: NativeAction) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
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


def _scale_profile(profile: Any, scale: float) -> Any:
    result = copy.deepcopy(profile)
    for radial_values in result.values():
        for radius in tuple(radial_values):
            if float(radius) < 1.0:
                radial_values[radius] = float(radial_values[radius]) * scale
    return result


def _environment(cell: PreparedBaseCell, view: PreparedBaseView) -> Any:
    from gymtorax.envs.iter_hybrid_env import CONFIG, IterHybridEnv  # type: ignore[import-untyped]

    config = copy.deepcopy(CONFIG)
    config["numerics"]["fixed_dt"] = float(view.internal_timestep_s)
    config["geometry"]["n_rho"] = view.radial_cells
    config["solver"]["n_corrector_steps"] = view.corrector_steps
    config["profile_conditions"]["T_i"] = _scale_profile(
        config["profile_conditions"]["T_i"], float(cell.initial_temperature_scale)
    )
    config["profile_conditions"]["T_e"] = _scale_profile(
        config["profile_conditions"]["T_e"], float(cell.initial_temperature_scale)
    )
    config["profile_conditions"]["nbar"] = float(cell.initial_density_nbar)
    config["neoclassical"]["bootstrap_current"]["bootstrap_multiplier"] = float(
        cell.bootstrap_multiplier
    )
    for key in ("D_e_inner", "chi_i_inner", "chi_e_inner"):
        config["transport"][key] = float(config["transport"][key]) * float(
            cell.inner_transport_scale
        )

    class PreparedBaseEnvironment(IterHybridEnv):  # type: ignore[misc]
        def _get_torax_config(self) -> dict[str, Any]:
            return {
                "config": config,
                "discretization": "fixed",
                "ratio_a_sim": int(round(1.0 / float(view.internal_timestep_s))),
            }

        def _compute_reward(self, state: Any, next_state: Any, action: Any) -> float:
            del state, next_state, action
            return 0.0

    return PreparedBaseEnvironment(render_mode=None, store_history=False, log_level="warning")


def verify_cpu_runtime() -> tuple[str, ...]:
    import importlib.metadata
    import jax

    reasons = []
    try:
        jax.config.update("jax_platforms", "cpu")  # type: ignore[no-untyped-call]
        jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
    except (RuntimeError, ValueError):
        reasons.append("CPU_FLOAT64_CONFIGURATION_FAILED")
    if importlib.metadata.version("gymtorax") != "1.1.1":
        reasons.append("GYM_TORAX_VERSION_MISMATCH")
    if importlib.metadata.version("torax") != "1.4.2":
        reasons.append("TORAX_VERSION_MISMATCH")
    if importlib.metadata.version("jax") != "0.10.2":
        reasons.append("JAX_VERSION_MISMATCH")
    if importlib.metadata.version("jaxlib") != "0.10.2":
        reasons.append("JAXLIB_VERSION_MISMATCH")
    if jax.default_backend() != "cpu":
        reasons.append("CPU_BACKEND_REQUIRED_NO_FALLBACK")
    if not bool(getattr(jax.config, "x64_enabled", False)):
        reasons.append("JAX_FLOAT64_REQUIRED")
    return tuple(sorted(reasons))


def _state_scalar(state: Mapping[str, Any], source_id: str) -> Decimal | None:
    value = state.get("scalars", {}).get(source_id)
    return None if value is None else _decimal(value)


def _receiver_values(state: Mapping[str, Any]) -> tuple[tuple[NamedDecimal, ...], tuple[str, ...]]:
    values: dict[str, NamedDecimal] = {}
    missing = []
    for value_id, (source_id, unit) in _SCALAR_SOURCES.items():
        value = _state_scalar(state, source_id)
        if value is None:
            if value_id in {
                "receiver.beta-n",
                "receiver.fgw-volume-average",
                "receiver.h98",
                "receiver.p-heat-total",
                "receiver.p-radiation-electron",
                "receiver.q-min",
                "receiver.q95",
            }:
                missing.append(value_id)
            continue
        values[value_id] = NamedDecimal(value_id=value_id, value=value, unit=unit)
    heat = values.get("receiver.p-heat-total")
    radiation = values.get("receiver.p-radiation-electron")
    if heat is None or radiation is None or heat.value <= 0:
        missing.append("receiver.radiated-fraction")
    else:
        values["receiver.radiated-fraction"] = NamedDecimal(
            value_id="receiver.radiated-fraction",
            value=radiation.value / heat.value,
            unit="1",
        )
    return tuple(sorted(values.values(), key=lambda value: value.value_id)), tuple(sorted(missing))


def _diagnostic_fields(state: Mapping[str, Any], clock_s: int) -> tuple[DenseField, ...]:
    if clock_s not in _DIAGNOSTIC_CLOCKS:
        return ()
    profiles = state.get("profiles", {})
    fields = []
    for field_id, (source_id, unit) in _PROFILE_SOURCES.items():
        raw = profiles.get(source_id)
        values = (
            () if raw is None else tuple(_decimal(value) for value in np.asarray(raw).reshape(-1))
        )
        fields.append(
            DenseField(
                field_id=field_id,
                unit=unit,
                values=values,
                source_available=bool(values),
            )
        )
    for field_id, unit in (
        ("profile.current-boundary", "A"),
        ("profile.density-boundary", "m-3"),
        ("profile.density-flux", "m-2.s-1"),
        ("profile.density-source", "m-3.s-1"),
        ("profile.particle-balance", "m-3.s-1"),
    ):
        fields.append(
            DenseField(
                field_id=field_id,
                unit=unit,
                values=(),
                source_available=False,
            )
        )
    return tuple(sorted(fields, key=lambda value: value.field_id))


def _realized_action(environment: Any, applied: NativeAction) -> NativeAction:
    state = environment.state or {}
    ip = _state_scalar(state, "Ip")
    nbi = _state_scalar(state, "P_aux_generic_total")
    ecrh = _state_scalar(state, "P_ecrh_e")
    if ip is None or nbi is None or ecrh is None:
        return applied
    return NativeAction(
        ip_a=ip,
        nbi_power_w=nbi,
        nbi_location=applied.nbi_location,
        nbi_width=applied.nbi_width,
        ecrh_power_w=ecrh,
        ecrh_location=applied.ecrh_location,
        ecrh_width=applied.ecrh_width,
    )


def acquire_episode(
    *,
    config: PreparedBaseConfig,
    cell_id: str,
    word_id: str,
    view_id: str,
) -> PreparedBaseEpisode:
    runtime_reasons = verify_cpu_runtime()
    if runtime_reasons:
        raise RuntimeError(",".join(runtime_reasons))
    cell = next(value for value in config.cells if value.cell_id == cell_id)
    word = next(value for value in config.words if value.word_id == word_id)
    view = next(value for value in config.views if value.view_id == view_id)
    environment = _environment(cell, view)
    transitions: list[PreparedBaseTransition] = []
    reasons: set[str] = set()
    disposition = EpisodeDisposition.COMPLETE
    started = time.monotonic()
    try:
        observation, _ = environment.reset(seed=cell.environment_seed)
        del observation
        for request_clock, row in enumerate(word.rows):
            if int(round(float(environment.current_time))) != request_clock:
                disposition = EpisodeDisposition.TECHNICAL_OBSERVATION_FAILURE
                reasons.add("NATIVE_CLOCK_DIVERGENCE")
                break
            requested = row.action
            _, reward, terminated, truncated, info = environment.step(_action_mapping(requested))
            accepted = requested
            applied = _native_action(environment.torax_app.config.get_current_action_values())
            realized = _realized_action(environment, applied)
            receiver_clock = request_clock + 1
            state = environment.state or {}
            receivers, missing = _receiver_values(state)
            valid = (
                float(reward) != -1000.0
                and not missing
                and all(math.isfinite(float(value.value)) for value in receivers)
            )
            clipped = bool(info.get("action_clipped", False))
            transitions.append(
                PreparedBaseTransition(
                    transition_id=(
                        f"transition.{cell.cell_id.removeprefix('cell.')}."
                        f"{word.word_id.removeprefix('word.')}."
                        f"{view.view_id.removeprefix('view.')}.{receiver_clock:03d}"
                    ),
                    receiver_clock_s=receiver_clock,
                    action=ActionLedger(
                        request_clock_s=request_clock,
                        accepted_clock_s=request_clock,
                        applied_clock_s=receiver_clock,
                        realized_clock_s=receiver_clock,
                        requested=requested,
                        accepted=accepted,
                        applied=applied,
                        realized=realized,
                        clipped=clipped,
                        realization_basis_id=(
                            "realization.gym-torax.state-scalars"
                            if realized != applied
                            else "realization.gym-torax.current-action-values"
                        ),
                    ),
                    receiver_values=receivers,
                    diagnostic_fields=_diagnostic_fields(state, receiver_clock),
                    valid=valid,
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    missing_primary_operand_ids=missing,
                )
            )
            if clipped:
                disposition = EpisodeDisposition.DELIVERY_INVALID
                reasons.add("UNEXPECTED_NATIVE_CLIPPING")
            if missing:
                disposition = EpisodeDisposition.UNEVALUABLE_OPERAND
                reasons.add("PRIMARY_RECEIVER_OPERAND_ABSENT")
            if not valid and not missing:
                disposition = EpisodeDisposition.NUMERICAL_INVALID
                reasons.add("NONFINITE_OR_FAILED_SIMULATION")
            if terminated or truncated:
                if receiver_clock < 150:
                    if disposition is EpisodeDisposition.COMPLETE:
                        disposition = EpisodeDisposition.SIMULATOR_TERMINATED
                    reasons.add(
                        "SIMULATOR_TERMINATED_BEFORE_HORIZON"
                        if terminated
                        else "SIMULATOR_TRUNCATED_BEFORE_HORIZON"
                    )
                break
    except Exception as error:
        disposition = EpisodeDisposition.TECHNICAL_OBSERVATION_FAILURE
        reasons.add(f"TECHNICAL_{type(error).__name__.upper()}")
    finally:
        environment.close()
    if len(transitions) < 150 and disposition is EpisodeDisposition.COMPLETE:
        disposition = EpisodeDisposition.PARTIAL_VALID_PREFIX
        reasons.add("REQUIRED_CLOCKS_ABSENT")
    last_valid = max(
        (value.receiver_clock_s for value in transitions if value.valid),
        default=None,
    )
    runtime = Decimal(f"{time.monotonic() - started:.9f}")
    return PreparedBaseEpisode(
        episode_id=(
            f"episode.{config.stage.value.lower()}."
            f"{cell.cell_id.removeprefix('cell.')}."
            f"{word.word_id.removeprefix('word.')}."
            f"{view.view_id.removeprefix('view.')}"
        ),
        config=ObjectIdentity.from_record(config.config_id, config),
        cell=ObjectIdentity.from_record(cell.cell_id, cell),
        word=ObjectIdentity.from_record(word.word_id, word),
        view=ObjectIdentity.from_record(view.view_id, view),
        transitions=tuple(transitions),
        disposition=disposition,
        last_valid_clock_s=last_valid,
        missing_required_clocks_s=tuple(range(len(transitions) + 1, 151)),
        reason_codes=tuple(sorted(reasons)),
        runtime_seconds=runtime,
        backend="cpu",
        precision="float64",
        outcome_access=config.outcome_access,
    )


_TECHNICAL_RESERVE_DISPOSITIONS = frozenset(
    {
        EpisodeDisposition.TECHNICAL_OBSERVATION_FAILURE,
        EpisodeDisposition.DELIVERY_INVALID,
    }
)


def acquire_stage(
    *,
    config: PreparedBaseConfig,
) -> PreparedBaseAcquisitionBatch:
    """Acquire primary bundles and consume reserves only for blinded technical failures."""

    if config.stage is PreparedBaseStage.QUALIFICATION:
        raise ValueError("qualification uses the explicit excluded paired-route acquisition")
    episodes: list[PreparedBaseEpisode] = []
    active_cell_ids: list[str] = []
    excluded_cell_ids: list[str] = []
    substitutions: list[PreparedBaseCellSubstitution] = []
    reserve_ids = iter(config.reserve_cell_ids)

    for intended_cell_id in config.primary_cell_ids:
        current_cell_id = intended_cell_id
        while True:
            bundle: list[PreparedBaseEpisode] = []
            technical_failure = False
            for word in config.words:
                for view in config.views:
                    episode = acquire_episode(
                        config=config,
                        cell_id=current_cell_id,
                        word_id=word.word_id,
                        view_id=view.view_id,
                    )
                    bundle.append(episode)
                    episodes.append(episode)
                    if episode.disposition in _TECHNICAL_RESERVE_DISPOSITIONS:
                        technical_failure = True
                        break
                if technical_failure:
                    break
            if not technical_failure:
                active_cell_ids.append(current_cell_id)
                break
            excluded_cell_ids.append(current_cell_id)
            try:
                replacement_cell_id = next(reserve_ids)
            except StopIteration:
                break
            reasons = tuple(
                sorted({reason for episode in bundle for reason in episode.reason_codes})
            )
            substitutions.append(
                PreparedBaseCellSubstitution(
                    substitution_id=(
                        f"substitution.{config.config_id.removeprefix('config.')}."
                        f"{intended_cell_id.removeprefix('cell.')}."
                        f"{replacement_cell_id.removeprefix('cell.')}"
                    ),
                    intended_cell_id=intended_cell_id,
                    replacement_cell_id=replacement_cell_id,
                    excluded_episode_ids=tuple(sorted(value.episode_id for value in bundle)),
                    reason_codes=reasons or ("TECHNICAL_BUNDLE_FAILURE",),
                    receiver_outcome_released_to_evaluator=False,
                )
            )
            current_cell_id = replacement_cell_id

    return PreparedBaseAcquisitionBatch(
        batch_id=f"batch.{config.config_id.removeprefix('config.')}",
        config=ObjectIdentity.from_record(config.config_id, config),
        stage=config.stage,
        episodes=tuple(sorted(episodes, key=lambda value: value.episode_id)),
        active_cell_ids=tuple(sorted(active_cell_ids)),
        excluded_cell_ids=tuple(sorted(excluded_cell_ids)),
        substitutions=tuple(sorted(substitutions, key=lambda value: value.substitution_id)),
        expected_active_cell_count=len(config.primary_cell_ids),
        complete_active_roster=len(active_cell_ids) == len(config.primary_cell_ids),
        outcome_access=config.outcome_access,
    )


_PREDICATES = (
    ("predicate.prefix-q-min", "receiver.q-min", range(1, 31), Decimal("1.02"), "LOWER"),
    (
        "predicate.prefix-fgw-volume",
        "receiver.fgw-volume-average",
        range(1, 31),
        Decimal("0.95"),
        "UPPER",
    ),
    ("predicate.phase-q-min", "receiver.q-min", range(105, 115), Decimal("1.02"), "LOWER"),
    ("predicate.phase-h98", "receiver.h98", range(105, 115), Decimal("0.8"), "LOWER"),
    (
        "predicate.phase-fgw-volume",
        "receiver.fgw-volume-average",
        range(105, 115),
        Decimal("0.95"),
        "UPPER",
    ),
    ("predicate.full-q95", "receiver.q95", range(1, 151), Decimal("3"), "LOWER"),
    ("predicate.full-beta-n", "receiver.beta-n", range(1, 151), Decimal("3.5"), "UPPER"),
    (
        "predicate.full-radiated-fraction",
        "receiver.radiated-fraction",
        range(1, 151),
        Decimal("0.5"),
        "UPPER",
    ),
    ("predicate.terminal-h98", "receiver.h98", (150,), Decimal("0.8"), "LOWER"),
)


def assess_episode(episode: PreparedBaseEpisode) -> EpisodeAssessment:
    by_clock = {
        value.receiver_clock_s: {item.value_id: item.value for item in value.receiver_values}
        for value in episode.transitions
    }
    reasons = set(episode.reason_codes)
    margins: list[Decimal] = []
    first_failure = None
    maximum_phase_fgw = None
    if episode.disposition is EpisodeDisposition.COMPLETE:
        for predicate_id, receiver_id, clocks, threshold, direction in _PREDICATES:
            observed = [by_clock.get(clock, {}).get(receiver_id) for clock in clocks]
            if any(value is None for value in observed):
                reasons.add("PREDICATE_OPERAND_ABSENT")
                if first_failure is None:
                    first_failure = predicate_id
                continue
            values = [value for value in observed if value is not None]
            local = (
                [((value - threshold) / max(abs(threshold), Decimal(1))) for value in values]
                if direction == "LOWER"
                else [((threshold - value) / max(abs(threshold), Decimal(1))) for value in values]
            )
            margins.extend(local)
            if any(value < 0 for value in local) and first_failure is None:
                first_failure = predicate_id
                reasons.add(
                    predicate_id.replace("predicate.", "").upper().replace("-", "_") + "_FAILED"
                )
        maximum_phase_fgw = max(
            by_clock[clock]["receiver.fgw-volume-average"] for clock in range(105, 115)
        )
    return EpisodeAssessment(
        assessment_id=f"assessment.{episode.episode_id.removeprefix('episode.')}",
        episode=ObjectIdentity.from_record(episode.episode_id, episode),
        cell_id=episode.cell.object_id,
        word_id=episode.word.object_id,
        view_id=episode.view.object_id,
        complete_contract_pass=(
            episode.disposition is EpisodeDisposition.COMPLETE
            and first_failure is None
            and not reasons
        ),
        worst_normalized_margin=min(margins) if margins else None,
        maximum_phase_fgw=maximum_phase_fgw,
        first_failed_predicate_id=first_failure,
        disposition=episode.disposition,
        reason_codes=tuple(sorted(reasons)),
    )


def _binomial_interval(k: int, n: int) -> tuple[Decimal, Decimal]:
    from scipy.stats import beta

    lower = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return Decimal(f"{lower:.15g}"), Decimal(f"{upper:.15g}")


def _one_sided_lower(k: int, n: int) -> Decimal:
    from scipy.stats import beta

    value = 0.0 if k == 0 else float(beta.ppf(0.05, k, n - k + 1))
    return Decimal(f"{value:.15g}")


def adjudicate_development(
    *,
    config: PreparedBaseConfig,
    episodes: Sequence[PreparedBaseEpisode],
    active_cell_ids: Sequence[str] | None = None,
) -> DevelopmentDecision:
    if config.stage is not PreparedBaseStage.DEVELOPMENT:
        raise ValueError("development adjudicator requires a development config")
    adjudication_cells = tuple(
        config.primary_cell_ids if active_cell_ids is None else active_cell_ids
    )
    if len(adjudication_cells) != len(config.primary_cell_ids):
        return DevelopmentDecision(
            decision_id='decision.prepared-base.development',
            development_config=ObjectIdentity.from_record(config.config_id, config),
            selected_word_id=None,
            selected_pass_count=0,
            recurrence_lower=Decimal(0),
            recurrence_upper=Decimal(1),
            nuisance_sd=None,
            nuisance_sigma_upper=None,
            n_required=None,
            evaluation_count=None,
            disposition="TECHNICAL_OBSERVATION_FAILURE",
            reason_codes=("DEVELOPMENT_ACTIVE_ROSTER_INCOMPLETE",),
            outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        )
    primary_cells = set(adjudication_cells)
    assessments = tuple(
        assess_episode(value) for value in episodes if value.cell.object_id in primary_cells
    )
    by_key = {(value.cell_id, value.word_id, value.view_id): value for value in assessments}
    expected = {
        (cell_id, word_id, view.view_id)
        for cell_id in adjudication_cells
        for word_id in (PREPARED_SOURCE_REFERENCE_WORD_ID, *PREPARED_SOURCE_CANDIDATE_WORD_IDS)
        for view in config.views
    }
    if set(by_key) != expected:
        return DevelopmentDecision(
            decision_id='decision.prepared-base.development',
            development_config=ObjectIdentity.from_record(config.config_id, config),
            selected_word_id=None,
            selected_pass_count=0,
            recurrence_lower=Decimal(0),
            recurrence_upper=Decimal(1),
            nuisance_sd=None,
            nuisance_sigma_upper=None,
            n_required=None,
            evaluation_count=None,
            disposition="TECHNICAL_OBSERVATION_FAILURE",
            reason_codes=("DEVELOPMENT_ROSTER_INCOMPLETE",),
            outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        )
    ranked = []
    for word_id in PREPARED_SOURCE_CANDIDATE_WORD_IDS:
        rows = [
            (
                by_key[(cell_id, word_id, PREPARED_SOURCE_PRIMARY_VIEW_ID)],
                by_key[(cell_id, word_id, PREPARED_SOURCE_REFINED_VIEW_ID)],
            )
            for cell_id in adjudication_cells
        ]
        technical = all(
            item.disposition
            not in {
                EpisodeDisposition.TECHNICAL_OBSERVATION_FAILURE,
                EpisodeDisposition.DELIVERY_INVALID,
                EpisodeDisposition.UNEVALUABLE_OPERAND,
            }
            for pair in rows
            for item in pair
        )
        pass_count = sum(
            left.complete_contract_pass and right.complete_contract_pass for left, right in rows
        )
        margins = [
            min(left.worst_normalized_margin, right.worst_normalized_margin)
            for left, right in rows
            if left.worst_normalized_margin is not None
            and right.worst_normalized_margin is not None
        ]
        median_margin = (
            Decimal(str(statistics.median(margins))) if margins else Decimal("-Infinity")
        )
        energy = next(
            value.added_pre_clock_100_energy_j for value in config.words if value.word_id == word_id
        )
        ranked.append((technical, pass_count, median_margin, -energy, word_id, rows))
    eligible = [value for value in ranked if value[0]]
    if not eligible:
        return DevelopmentDecision(
            decision_id='decision.prepared-base.development',
            development_config=ObjectIdentity.from_record(config.config_id, config),
            selected_word_id=None,
            selected_pass_count=0,
            recurrence_lower=Decimal(0),
            recurrence_upper=Decimal(1),
            nuisance_sd=None,
            nuisance_sigma_upper=None,
            n_required=None,
            evaluation_count=None,
            disposition="TECHNICAL_OBSERVATION_FAILURE",
            reason_codes=("NO_TECHNICALLY_INTERPRETABLE_CANDIDATE",),
            outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        )
    selected = sorted(
        eligible,
        key=lambda value: (-value[1], -value[2], -value[3], value[4]),
    )[0]
    _, pass_count, _, _, word_id, rows = selected
    lower, upper = _binomial_interval(pass_count, 6)
    differences = [
        right.maximum_phase_fgw - left.maximum_phase_fgw
        for left, right in rows
        if left.maximum_phase_fgw is not None and right.maximum_phase_fgw is not None
    ]
    reasons: set[str] = set()
    if len(differences) == 6:
        from scipy.stats import chi2

        sd = Decimal(str(statistics.stdev(differences)))
        floor = max(sd, Decimal("0.0001"))
        sigma_upper = floor * Decimal(str(math.sqrt(5 / float(chi2.ppf(0.05, 5)))))
        n_required = math.ceil(
            (
                (Decimal("1.959963984540054") + Decimal("1.281551565544601"))
                * sigma_upper
                / Decimal("0.03")
            )
            ** 2
        )
    else:
        sd = None
        sigma_upper = None
        n_required = 24
        reasons.add("INCOMPLETE_SCIENTIFIC_PREFIX_SELECTS_FROZEN_MAXIMUM")
    evaluation_count = next(
        (value for value in PREPARED_SOURCE_ALLOWED_EVALUATION_COUNTS if value >= max(14, n_required)),
        None,
    )
    disposition = (
        "DESIGN_INFEASIBLE_AT_FROZEN_MAXIMUM" if n_required > 24 else "EVALUATION_NOMINATED"
    )
    if n_required > 24:
        reasons.add("N_REQUIRED_EXCEEDS_24")
    return DevelopmentDecision(
        decision_id='decision.prepared-base.development',
        development_config=ObjectIdentity.from_record(config.config_id, config),
        selected_word_id=word_id,
        selected_pass_count=pass_count,
        recurrence_lower=lower,
        recurrence_upper=upper,
        nuisance_sd=sd,
        nuisance_sigma_upper=sigma_upper,
        n_required=n_required,
        evaluation_count=evaluation_count,
        disposition=disposition,
        reason_codes=tuple(sorted(reasons)),
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
    )


_BRANCHES = {
    PreparedBasePrimaryResult.SUPPORTED_CROSS_VIEW_BASE: "branch.prepared-response.response-admission",
    PreparedBasePrimaryResult.VIEW_SENSITIVE_BASE: "branch.prepared-response.numerical-preparation",
    PreparedBasePrimaryResult.NO_SUPPORTED_PREPARATION: "branch.prepared-response.preparation",
    PreparedBasePrimaryResult.PARTIAL_OR_TERMINATED: "branch.prepared-response.termination-base-path",
    PreparedBasePrimaryResult.TECHNICAL_OBSERVATION_FAILURE: "branch.prepared-response.technical-repair",
    PreparedBasePrimaryResult.UNEVALUABLE_OPERAND: "branch.prepared-response.alternate-denominator-or-operand-repair",
}


def adjudicate_evaluation(
    *,
    config: PreparedBaseConfig,
    episodes: Sequence[PreparedBaseEpisode],
    active_cell_ids: Sequence[str] | None = None,
) -> EvaluationAdjudication:
    if config.stage is not PreparedBaseStage.EVALUATION:
        raise ValueError("evaluation adjudicator requires an evaluation config")
    adjudication_cells = tuple(
        config.primary_cell_ids if active_cell_ids is None else active_cell_ids
    )
    primary_cells = set(adjudication_cells)
    assessments = tuple(
        assess_episode(value) for value in episodes if value.cell.object_id in primary_cells
    )
    by_key = {(value.cell_id, value.view_id): value for value in assessments}
    expected = {(cell_id, view.view_id) for cell_id in adjudication_cells for view in config.views}
    reason_codes: set[str] = set()
    pairs: list[tuple[EpisodeAssessment, EpisodeAssessment]]
    if len(adjudication_cells) != len(config.primary_cell_ids):
        result = PreparedBasePrimaryResult.TECHNICAL_OBSERVATION_FAILURE
        reason_codes.add("EVALUATION_ACTIVE_ROSTER_INCOMPLETE")
        pairs = []
    elif set(by_key) != expected:
        result = PreparedBasePrimaryResult.TECHNICAL_OBSERVATION_FAILURE
        reason_codes.add("EVALUATION_ROSTER_INCOMPLETE")
        pairs = []
    else:
        pairs = [
            (
                by_key[(cell_id, PREPARED_SOURCE_PRIMARY_VIEW_ID)],
                by_key[(cell_id, PREPARED_SOURCE_REFINED_VIEW_ID)],
            )
            for cell_id in adjudication_cells
        ]
        dispositions = {item.disposition for pair in pairs for item in pair}
        if dispositions & {
            EpisodeDisposition.TECHNICAL_OBSERVATION_FAILURE,
            EpisodeDisposition.DELIVERY_INVALID,
        }:
            result = PreparedBasePrimaryResult.TECHNICAL_OBSERVATION_FAILURE
        elif EpisodeDisposition.UNEVALUABLE_OPERAND in dispositions:
            result = PreparedBasePrimaryResult.UNEVALUABLE_OPERAND
        elif dispositions & {
            EpisodeDisposition.SIMULATOR_TERMINATED,
            EpisodeDisposition.PRESERVATION_CROSSING,
            EpisodeDisposition.NUMERICAL_INVALID,
            EpisodeDisposition.PARTIAL_VALID_PREFIX,
        }:
            result = PreparedBasePrimaryResult.PARTIAL_OR_TERMINATED
        elif all(
            left.complete_contract_pass and right.complete_contract_pass for left, right in pairs
        ):
            result = PreparedBasePrimaryResult.SUPPORTED_CROSS_VIEW_BASE
        elif any(
            left.complete_contract_pass != right.complete_contract_pass for left, right in pairs
        ):
            result = PreparedBasePrimaryResult.VIEW_SENSITIVE_BASE
        else:
            result = PreparedBasePrimaryResult.NO_SUPPORTED_PREPARATION
    pass_count = sum(
        left.complete_contract_pass and right.complete_contract_pass for left, right in pairs
    )
    unit_count = len(config.primary_cell_ids)
    lower, upper = _binomial_interval(pass_count, unit_count)
    one_sided = _one_sided_lower(pass_count, unit_count)
    if result is PreparedBasePrimaryResult.SUPPORTED_CROSS_VIEW_BASE and (
        pass_count != unit_count or one_sided < Decimal("0.80")
    ):
        result = PreparedBasePrimaryResult.NO_SUPPORTED_PREPARATION
        reason_codes.add("RECURRENCE_LOWER_BOUND_BELOW_0P80")
    differences = [
        right.maximum_phase_fgw - left.maximum_phase_fgw
        for left, right in pairs
        if left.maximum_phase_fgw is not None and right.maximum_phase_fgw is not None
    ]
    first_failed = next(
        (
            item.first_failed_predicate_id
            for pair in pairs
            for item in pair
            if item.first_failed_predicate_id is not None
        ),
        None,
    )
    return EvaluationAdjudication(
        adjudication_id=f"adjudication.{config.config_id.removeprefix('config.')}",
        evaluation_config=ObjectIdentity.from_record(config.config_id, config),
        primary_result=result,
        pass_count=pass_count,
        unit_count=unit_count,
        recurrence_lower=lower,
        recurrence_upper=upper,
        one_sided_lower=one_sided,
        confirmatory_view_difference_mean=(
            sum(differences, Decimal(0)) / Decimal(len(differences)) if differences else None
        ),
        selected_preparation_branch_id=_BRANCHES[result],
        first_failed_predicate_id=first_failed,
        reason_codes=tuple(sorted(reason_codes)),
        maximum_evidence_ceiling=EvidenceCeiling.ORDER_RELATION,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
    )


__all__ = [
    "ActionLedger",
    "DenseField",
    "DevelopmentDecision",
    "EpisodeAssessment",
    "EpisodeDisposition",
    "EvaluationAdjudication",
    "NativeAction",
    "PreparationActionRow",
    "PreparationWord",
    "PreparationWordKind",
    "PreparedBaseAcquisitionBatch",
    "PreparedBaseCellSubstitution",
    "PreparedBaseIssue",
    "PreparedBasePanel",
    "PreparedBaseQualificationReceipt",
    "PreparedBaseScientificApproval",
    "PreparedBaseCell",
    "PreparedBaseConfig",
    "PreparedBaseEpisode",
    "PreparedBaseStage",
    "PreparedBaseTransition",
    "PreparedBaseView",
    'PreparedBasePrimaryResult',
    "PREPARED_SOURCE_ALLOWED_EVALUATION_COUNTS",
    "PREPARED_SOURCE_CANDIDATE_WORD_IDS",
    "PREPARED_SOURCE_DEVELOPMENT_CONFIG_ID",
    "PREPARED_SOURCE_EARLY_ECRH_WORD_ID",
    "PREPARED_SOURCE_EARLY_FULL_WORD_ID",
    "PREPARED_SOURCE_EVALUATOR_ID",
    "PREPARED_SOURCE_PARENT_ID",
    "PREPARED_SOURCE_PRIMARY_VIEW_ID",
    "PREPARED_SOURCE_REFERENCE_WORD_ID",
    "PREPARED_SOURCE_REFINED_VIEW_ID",
    "acquire_episode",
    "acquire_stage",
    "adjudicate_development",
    "adjudicate_evaluation",
    "assess_episode",
    "build_prepared_source_formal_coverage",
    "development_cells",
    "development_config",
    "evaluation_config",
    "evaluation_master_cells",
    "paired_views",
    "preparation_words",
    "qualification_config",
    "verify_cpu_runtime",
]
