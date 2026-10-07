"Theory-aligned Gym--TORAX numerical-denominator experiment.\n\nThe bounded experiment treats timestep, grid, corrector and observation gauge\nas denominator coordinates.  Numerical configurations are nested comparisons\nwithin one preparation cell, never independent replications.  The module is\nadditive to the retained preparation implementation and has a maximum evidence ceiling of measurement.\n"

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
from enum import StrEnum
from typing import Any, ClassVar, Mapping, Sequence, cast
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
    require_sorted_unique_strings,
    validate_relative_locator,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.time import parse_utc_timestamp
from empirical_lawhood.planning.formal_analysis import FormalGapSourceCapabilityInventory
from empirical_lawhood.planning.formal_gaps import (
    FormalGapCoverage,
    FormalGapCoverageAssignment,
    FormalGapCoverageDisposition,
    FormalGapEvidenceWorld,
    FormalGapRegister,
)
from empirical_lawhood.kernel.status import ReadinessStatus
from empirical_lawhood.kernel.worlds import EvidenceUnitScope

from .open_campaigns import OpenSimulatorSourceManifest
from .prepared_base import (
    ActionLedger,
    DenseField,
    EpisodeDisposition,
    PreparationWord,
    PreparedBaseCellSubstitution,
    PREPARED_SOURCE_EARLY_FULL_WORD_ID,
    _action_mapping,
    _environment,
    _native_action,
    _realized_action,
    _receiver_values,
    preparation_words,
    verify_cpu_runtime,
)


NUMERICAL_DENOMINATOR_PARENT_ID = 'design.numerical-denominator.gym-torax-numerical-denominator'
NUMERICAL_DENOMINATOR_DEVELOPMENT_CONFIG_ID = 'config.numerical-denominator.numerical-factorial-development'
NUMERICAL_DENOMINATOR_QUALIFICATION_CONFIG_ID = 'config.numerical-denominator.numerical-factorial-qualification'
NUMERICAL_DENOMINATOR_EVALUATION_CONFIG_PREFIX = 'config.numerical-denominator.numerical-'
NUMERICAL_DENOMINATOR_EVALUATOR_ID = 'evaluator.numerical-denominator.numerical-denominator'
NUMERICAL_DENOMINATOR_DECISION_RULE_ID = 'decision-rule.numerical-denominator-development-to-evaluation'
NUMERICAL_DENOMINATOR_HANDOFF_ID = 'handoff.numerical-denominator-to-followup-study-or-stop'
NUMERICAL_DENOMINATOR_MAX_CANONICAL_EPISODE_BYTES = 2 * 1024**2
NUMERICAL_DENOMINATOR_BATCH_OVERHEAD_BYTES = 8 * 1024**2
NUMERICAL_DENOMINATOR_REQUIRED_IN_MEMORY_GUARD_BYTES = 256 * 1024**2

NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID = 'view.numerical-denominator.timestep-one-second.radial-cells-25.corrector-steps-10'
NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID = 'view.numerical-denominator.timestep-half-second.radial-cells-25.corrector-steps-10'
NUMERICAL_DENOMINATOR_TIMESTEP_QUARTER_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID = 'view.numerical-denominator.timestep-quarter-second.radial-cells-25.corrector-steps-10'
NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_10_VIEW_ID = 'view.numerical-denominator.timestep-one-second.radial-cells-33.corrector-steps-10'
NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_41_CORRECTOR_STEPS_10_VIEW_ID = 'view.numerical-denominator.timestep-one-second.radial-cells-41.corrector-steps-10'
NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_20_VIEW_ID = 'view.numerical-denominator.timestep-one-second.radial-cells-25.corrector-steps-20'
NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_40_VIEW_ID = 'view.numerical-denominator.timestep-one-second.radial-cells-25.corrector-steps-40'
NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_10_VIEW_ID = 'view.numerical-denominator.timestep-half-second.radial-cells-33.corrector-steps-10'
NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_20_VIEW_ID = 'view.numerical-denominator.timestep-half-second.radial-cells-25.corrector-steps-20'
NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_20_VIEW_ID = 'view.numerical-denominator.timestep-one-second.radial-cells-33.corrector-steps-20'
NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_20_VIEW_ID = 'view.numerical-denominator.timestep-half-second.radial-cells-33.corrector-steps-20'

_QUANTUM = Decimal("0.0000001")
_PHASE_CLOCKS = tuple(range(105, 115))
_DIAGNOSTIC_CLOCKS = frozenset((*range(1, 31), *_PHASE_CLOCKS, 150))
_SOURCE_UNEVALUABLE_GAP_IDS = frozenset(
    {
        "gap.geometry.cohomology",
        "gap.geometry.information-geometry",
        "gap.geometry.metric-structure",
        "gap.geometry.topology-restriction",
    }
)


class NumericalDenominatorStage(StrEnum):
    QUALIFICATION = "QUALIFICATION"
    DEVELOPMENT = "DEVELOPMENT"
    EVALUATION = "EVALUATION"


class NumericalDenominatorBranch(StrEnum):
    FACTORIAL = "factorial"
    TIMESTEP = "timestep"
    GRID = "grid"
    CORRECTOR = "corrector"
    NUMERICAL_INTERACTION = "numerical-interaction"
    OBSERVATION_GAUGE = "observation-gauge"
    STABLE_DENOMINATOR = "stable-denominator"
    SCIENTIFIC_PARTIAL = "scientific-partial"


class NumericalDenominatorPrimaryResult(StrEnum):
    QUALIFIED_STABLE_PASS = "QUALIFIED_STABLE_PASS"
    QUALIFIED_STABLE_FAIL = "QUALIFIED_STABLE_FAIL"
    OBSERVATION_GAUGE_SENSITIVE = "OBSERVATION_GAUGE_SENSITIVE"
    TIMESTEP_SENSITIVE = "TIMESTEP_SENSITIVE"
    GRID_SENSITIVE = "GRID_SENSITIVE"
    CORRECTOR_SENSITIVE = "CORRECTOR_SENSITIVE"
    NUMERICAL_INTERACTION = "NUMERICAL_INTERACTION"
    NONCONVERGENT_OR_UNRESOLVED = "NONCONVERGENT_OR_UNRESOLVED"
    SCIENTIFIC_PARTIAL_OR_TERMINATED = "SCIENTIFIC_PARTIAL_OR_TERMINATED"
    UNEVALUABLE_SOURCE_OPERAND = "UNEVALUABLE_SOURCE_OPERAND"
    TECHNICAL_OBSERVATION_FAILURE = "TECHNICAL_OBSERVATION_FAILURE"


@dataclass(frozen=True, slots=True)
class NumericalDenominatorTolerances(CanonicalRecord):
    """Frozen numerical/gauge decision tolerances."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/numerical-denominator-tolerances'

    observation_fgw_absolute: Decimal = Decimal("0.002")
    material_fgw_absolute: Decimal = Decimal("0.01")
    interaction_fgw_absolute: Decimal = Decimal("0.01")
    stable_fgw_absolute: Decimal = Decimal("0.01")
    stable_qmin_normalized: Decimal = Decimal("0.01")
    stable_h98_absolute: Decimal = Decimal("0.01")
    stable_density_relative: Decimal = Decimal("0.01")
    stable_current_relative: Decimal = Decimal("0.001")
    contraction_ratio: Decimal = Decimal("0.9")
    identity_residual_absolute: Decimal = Decimal("0.0000000001")

    def __post_init__(self) -> None:
        values = (
            self.observation_fgw_absolute,
            self.material_fgw_absolute,
            self.interaction_fgw_absolute,
            self.stable_fgw_absolute,
            self.stable_qmin_normalized,
            self.stable_h98_absolute,
            self.stable_density_relative,
            self.stable_current_relative,
            self.identity_residual_absolute,
        )
        if any(value <= 0 for value in values):
            raise ValueError("numerical denominator tolerances must be positive")
        if not Decimal(0) < self.contraction_ratio < Decimal(1):
            raise ValueError("numerical denominator contraction ratio must lie in (0, 1)")


@dataclass(frozen=True, slots=True)
class NumericalDenominatorCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/numerical-denominator-cell'

    cell_id: str
    stage: NumericalDenominatorStage
    reserve: bool
    environment_seed: int
    initial_temperature_scale: Decimal
    initial_density_nbar: Decimal
    bootstrap_multiplier: Decimal
    inner_transport_scale: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.cell_id, field_name="cell_id")
        if self.environment_seed < 0:
            raise ValueError("numerical denominator environment seed cannot be negative")
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
            if not lower <= value <= upper:
                raise ValueError(f'{name} is outside the frozen numerical denominator local box')

    @property
    def coordinate_key(self) -> tuple[Decimal, Decimal, Decimal, Decimal]:
        return (
            self.initial_temperature_scale,
            self.initial_density_nbar,
            self.bootstrap_multiplier,
            self.inner_transport_scale,
        )


@dataclass(frozen=True, slots=True)
class NumericalDenominatorView(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/numerical-denominator-view'

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
        if any(value not in {0, 1, 2} for value in self.levels):
            raise ValueError("numerical denominator view has an unknown factor level")
        expected_dt = (Decimal("1"), Decimal("0.5"), Decimal("0.25"))[
            self.timestep_level
        ]
        expected_grid = (25, 33, 41)[self.grid_level]
        expected_correctors = (10, 20, 40)[self.corrector_level]
        if (
            self.internal_timestep_s != expected_dt
            or self.radial_cells != expected_grid
            or self.corrector_steps != expected_correctors
            or not self.predictor_corrector
            or not self.pereverzev_enabled
            or self.pereverzev_chi != Decimal("30")
            or self.pereverzev_d != Decimal("15")
            or self.precision != "float64"
            or self.backend != "cpu"
        ):
            raise ValueError("numerical denominator view changed the frozen source/numerical family")

    @property
    def levels(self) -> tuple[int, int, int]:
        return (self.timestep_level, self.grid_level, self.corrector_level)

    @property
    def native_internal_steps(self) -> int:
        return int(Decimal(150) / self.internal_timestep_s)


@dataclass(frozen=True, slots=True)
class NumericalDenominatorConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/numerical-denominator-config'

    config_id: str
    parent_design_id: str
    stage: NumericalDenominatorStage
    branch: NumericalDenominatorBranch
    source_manifest: ObjectIdentity
    formal_coverage: ObjectIdentity
    cells: tuple[NumericalDenominatorCell, ...]
    primary_cell_ids: tuple[str, ...]
    reserve_cell_ids: tuple[str, ...]
    word: PreparationWord
    views: tuple[NumericalDenominatorView, ...]
    tolerances: NumericalDenominatorTolerances
    common_grid_cells: int
    evaluator_id: str
    decision_rule_id: str
    outcome_access: OutcomeAccess
    maximum_evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_stable_id(self.parent_design_id, field_name="parent_design_id")
        validate_stable_id(self.evaluator_id, field_name="evaluator_id")
        validate_stable_id(self.decision_rule_id, field_name="decision_rule_id")
        require_sorted_unique_strings(self.primary_cell_ids, field_name="primary_cell_ids")
        require_sorted_unique_strings(self.reserve_cell_ids, field_name="reserve_cell_ids")
        if set(self.primary_cell_ids) & set(self.reserve_cell_ids):
            raise ValueError("numerical denominator primary and reserve cells overlap")
        by_id = {value.cell_id: value for value in self.cells}
        if len(by_id) != len(self.cells) or set(by_id) != {
            *self.primary_cell_ids,
            *self.reserve_cell_ids,
        }:
            raise ValueError("numerical denominator config cell inventory differs from its roster")
        if any(by_id[value].reserve for value in self.primary_cell_ids) or any(
            not by_id[value].reserve for value in self.reserve_cell_ids
        ):
            raise ValueError("numerical denominator primary/reserve flags differ")
        if any(value.stage is not self.stage for value in self.cells):
            raise ValueError("numerical denominator cell stage differs from config")
        if self.word.word_id != PREPARED_SOURCE_EARLY_FULL_WORD_ID:
            raise ValueError("numerical denominator must retain the exact preparation early-full-heating word")
        view_ids = tuple(value.view_id for value in self.views)
        if len(view_ids) != len(set(view_ids)):
            raise ValueError("numerical denominator config repeats a numerical denominator")
        expected = _branch_view_ids(self.stage, self.branch)
        if view_ids != expected:
            raise ValueError("numerical denominator config views differ from the frozen branch")
        expected_counts = {
            NumericalDenominatorStage.QUALIFICATION: (1, 0),
            NumericalDenominatorStage.DEVELOPMENT: (4, 1),
            NumericalDenominatorStage.EVALUATION: (8, 2),
        }[self.stage]
        if (len(self.primary_cell_ids), len(self.reserve_cell_ids)) != expected_counts:
            raise ValueError("numerical denominator config changed the frozen cohort size")
        expected_access = {
            NumericalDenominatorStage.QUALIFICATION: OutcomeAccess.DEVELOPMENT_VISIBLE,
            NumericalDenominatorStage.DEVELOPMENT: OutcomeAccess.DEVELOPMENT_VISIBLE,
            NumericalDenominatorStage.EVALUATION: OutcomeAccess.EVALUATION_SEALED,
        }[self.stage]
        if (
            self.outcome_access is not expected_access
            or self.maximum_evidence_ceiling is not EvidenceCeiling.MEASUREMENT
            or self.common_grid_cells != 200
            or self.parent_design_id != NUMERICAL_DENOMINATOR_PARENT_ID
            or self.evaluator_id != NUMERICAL_DENOMINATOR_EVALUATOR_ID
            or self.decision_rule_id != NUMERICAL_DENOMINATOR_DECISION_RULE_ID
        ):
            raise ValueError("numerical denominator config changed its frozen evidence contract")

    @property
    def maximum_episode_count(self) -> int:
        return len(self.cells) * len(self.views)

    @property
    def primary_episode_count(self) -> int:
        return len(self.primary_cell_ids) * len(self.views)

    @property
    def maximum_native_internal_steps(self) -> int:
        return len(self.cells) * sum(value.native_internal_steps for value in self.views)


@dataclass(frozen=True, slots=True)
class NumericalDenominatorResourceEnvelope(CanonicalRecord):
    "Frozen single-process execution allocation for one numerical denominator child."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/numerical-denominator-resource-envelope'

    envelope_id: str
    config: ObjectIdentity
    primary_episode_count: int
    maximum_episode_count: int
    primary_transition_count: int
    maximum_transition_count: int
    maximum_native_internal_steps: int
    maximum_corrector_step_count: int
    maximum_grid_corrector_update_count: int
    maximum_diagnostic_snapshot_count: int
    maximum_canonical_output_bytes: int
    required_in_memory_guard_bytes: int
    worker_process_count: int
    cpu_cores: int
    peak_memory_bytes: int
    wall_time_seconds: int
    minimum_free_external_bytes: int

    def __post_init__(self) -> None:
        validate_stable_id(self.envelope_id, field_name="envelope_id")
        positive = (
            self.primary_episode_count,
            self.maximum_episode_count,
            self.primary_transition_count,
            self.maximum_transition_count,
            self.maximum_native_internal_steps,
            self.maximum_corrector_step_count,
            self.maximum_grid_corrector_update_count,
            self.maximum_diagnostic_snapshot_count,
            self.maximum_canonical_output_bytes,
            self.required_in_memory_guard_bytes,
            self.worker_process_count,
            self.cpu_cores,
            self.peak_memory_bytes,
            self.wall_time_seconds,
            self.minimum_free_external_bytes,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("numerical denominator resource allocations must be positive")
        if (
            self.primary_episode_count > self.maximum_episode_count
            or self.primary_transition_count > self.maximum_transition_count
            or self.maximum_canonical_output_bytes
            > self.required_in_memory_guard_bytes
            or self.worker_process_count != 1
            or self.cpu_cores != 4
            or self.peak_memory_bytes != 24 * 1024**3
            or self.wall_time_seconds != 3600
        ):
            raise ValueError("numerical denominator resource envelope changed its bounded execution model")


def compile_resource_envelope(config: NumericalDenominatorConfig) -> NumericalDenominatorResourceEnvelope:
    maximum_episodes = config.maximum_episode_count
    maximum_output = (
        NUMERICAL_DENOMINATOR_BATCH_OVERHEAD_BYTES
        + maximum_episodes * NUMERICAL_DENOMINATOR_MAX_CANONICAL_EPISODE_BYTES
    )
    return NumericalDenominatorResourceEnvelope(
        envelope_id=f"resource-envelope.{config.config_id.removeprefix('config.')}",
        config=ObjectIdentity.from_record(config.config_id, config),
        primary_episode_count=config.primary_episode_count,
        maximum_episode_count=maximum_episodes,
        primary_transition_count=config.primary_episode_count * 150,
        maximum_transition_count=maximum_episodes * 150,
        maximum_native_internal_steps=config.maximum_native_internal_steps,
        maximum_corrector_step_count=len(config.cells)
        * sum(
            value.native_internal_steps * value.corrector_steps
            for value in config.views
        ),
        maximum_grid_corrector_update_count=len(config.cells)
        * sum(
            value.native_internal_steps
            * value.corrector_steps
            * value.radial_cells
            for value in config.views
        ),
        maximum_diagnostic_snapshot_count=maximum_episodes
        * len(_DIAGNOSTIC_CLOCKS),
        maximum_canonical_output_bytes=maximum_output,
        required_in_memory_guard_bytes=NUMERICAL_DENOMINATOR_REQUIRED_IN_MEMORY_GUARD_BYTES,
        worker_process_count=1,
        cpu_cores=4,
        peak_memory_bytes=24 * 1024**3,
        wall_time_seconds=3600,
        minimum_free_external_bytes=1024**3 + 3 * maximum_output,
    )


@dataclass(frozen=True, slots=True)
class NumericalDenominatorSourceFile(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/numerical-denominator-source-file'

    relative_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        validate_relative_locator(self.relative_path)
        validate_sha256(self.sha256, field_name="sha256")
        if self.size_bytes <= 0:
            raise ValueError("numerical denominator source file must be nonempty")


@dataclass(frozen=True, slots=True)
class NumericalDenominatorImplementationManifest(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/numerical-denominator-implementation-manifest'

    manifest_id: str
    source_files: tuple[NumericalDenominatorSourceFile, ...]
    implementation_sha256: str

    def __post_init__(self) -> None:
        validate_stable_id(self.manifest_id, field_name="manifest_id")
        validate_sha256(
            self.implementation_sha256,
            field_name="implementation_sha256",
        )
        paths = tuple(value.relative_path for value in self.source_files)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)) or not paths:
            raise ValueError("numerical denominator source closure must be nonempty, sorted and unique")


@dataclass(frozen=True, slots=True)
class NumericalDenominatorQualificationEpisode(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/numerical-denominator-qualification-episode'

    view_id: str
    episode: ObjectIdentity
    sha256: str
    size_bytes: int
    runtime_seconds: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.view_id, field_name="view_id")
        validate_sha256(self.sha256, field_name="sha256")
        if self.size_bytes <= 0 or self.runtime_seconds < 0:
            raise ValueError("numerical denominator qualification episode has invalid resources")


@dataclass(frozen=True, slots=True)
class NumericalDenominatorQualificationReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/numerical-denominator-qualification-receipt'

    receipt_id: str
    config: ObjectIdentity
    source_manifest: ObjectIdentity
    implementation: ObjectIdentity
    resource_envelope: ObjectIdentity
    installed_distribution_record_sha256: tuple[str, ...]
    episodes: tuple[NumericalDenominatorQualificationEpisode, ...]
    available_field_ids: tuple[str, ...]
    unavailable_field_ids: tuple[str, ...]
    maximum_episode_size_bytes: int
    maximum_episode_bound_bytes: int
    backend: str
    precision: str
    complete_route: bool
    recovery_verified_without_reacquisition: bool
    reason_codes: tuple[str, ...]
    maximum_evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        require_sorted_unique_strings(
            self.installed_distribution_record_sha256,
            field_name="installed_distribution_record_sha256",
            allow_empty=False,
        )
        for value in self.installed_distribution_record_sha256:
            validate_sha256(
                value,
                field_name="installed_distribution_record_sha256",
            )
        view_ids = tuple(value.view_id for value in self.episodes)
        if view_ids != tuple(sorted(view_ids)) or len(view_ids) != len(set(view_ids)):
            raise ValueError("numerical denominator qualification views must be sorted and unique")
        require_sorted_unique_strings(
            self.available_field_ids,
            field_name="available_field_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.unavailable_field_ids,
            field_name="unavailable_field_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if set(self.available_field_ids) & set(self.unavailable_field_ids):
            raise ValueError("numerical denominator qualification field dispositions overlap")
        if (
            len(self.episodes) != 5
            or self.maximum_episode_size_bytes
            != max(value.size_bytes for value in self.episodes)
            or self.maximum_episode_size_bytes > self.maximum_episode_bound_bytes
            or self.maximum_episode_bound_bytes != NUMERICAL_DENOMINATOR_MAX_CANONICAL_EPISODE_BYTES
            or self.backend != "cpu"
            or self.precision != "float64"
            or not self.complete_route
            or not self.recovery_verified_without_reacquisition
            or self.reason_codes
            or self.maximum_evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
            or self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
        ):
            raise ValueError("numerical denominator qualification did not close the exact route")


@dataclass(frozen=True, slots=True)
class NumericalDenominatorApproval(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/numerical-denominator-approval'

    approval_id: str
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
            self.approver == self.proposer
            or self.codex_or_chat_is_approver_or_issuer
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
        ):
            raise ValueError("numerical denominator approval lacks independent outcome-blind approval")


@dataclass(frozen=True, slots=True)
class NumericalDenominatorIssue(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/numerical-denominator-issue'

    issue_id: str
    stage: NumericalDenominatorStage
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
        for value in roles:
            validate_stable_id(value, field_name="role_id")
        parse_utc_timestamp(self.issued_at_utc, field_name="issued_at_utc")
        if (
            len(set(roles)) != len(roles)
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or not self.execution_authority_required
            or self.reveal_authority_required
            != (self.stage is NumericalDenominatorStage.EVALUATION)
        ):
            raise ValueError("numerical denominator issue merged or omitted an authority gate")


@dataclass(frozen=True, slots=True)
class NumericalDenominatorExecutionReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/numerical-denominator-execution-receipt'

    receipt_id: str
    run_id: str
    issue: ObjectIdentity
    execution_authority: ObjectIdentity
    config: ObjectIdentity
    batch: ObjectIdentity
    batch_relative_path: str
    batch_sha256: str
    batch_size_bytes: int
    active_cell_ids: tuple[str, ...]
    excluded_cell_ids: tuple[str, ...]
    completed_at_utc: str
    terminal_acquisition: bool
    artifacts_complete: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        validate_stable_id(self.run_id, field_name="run_id")
        validate_relative_locator(self.batch_relative_path)
        validate_sha256(self.batch_sha256, field_name="batch_sha256")
        require_sorted_unique_strings(
            self.active_cell_ids,
            field_name="active_cell_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.excluded_cell_ids,
            field_name="excluded_cell_ids",
        )
        parse_utc_timestamp(self.completed_at_utc, field_name="completed_at_utc")
        if (
            self.batch_size_bytes <= 0
            or not self.terminal_acquisition
            or not self.artifacts_complete
            or self.outcome_access
            not in {
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                OutcomeAccess.EVALUATION_SEALED,
            }
        ):
            raise ValueError("numerical denominator execution receipt is not terminal and complete")


def _view(
    view_id: str,
    timestep_level: int,
    grid_level: int,
    corrector_level: int,
) -> NumericalDenominatorView:
    return NumericalDenominatorView(
        view_id=view_id,
        timestep_level=timestep_level,
        grid_level=grid_level,
        corrector_level=corrector_level,
        internal_timestep_s=(Decimal("1"), Decimal("0.5"), Decimal("0.25"))[
            timestep_level
        ],
        radial_cells=(25, 33, 41)[grid_level],
        corrector_steps=(10, 20, 40)[corrector_level],
    )


def all_views() -> tuple[NumericalDenominatorView, ...]:
    return (
        _view(NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID, 0, 0, 0),
        _view(NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_20_VIEW_ID, 0, 0, 1),
        _view(NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_40_VIEW_ID, 0, 0, 2),
        _view(NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_10_VIEW_ID, 0, 1, 0),
        _view(NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_20_VIEW_ID, 0, 1, 1),
        _view(NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_41_CORRECTOR_STEPS_10_VIEW_ID, 0, 2, 0),
        _view(NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID, 1, 0, 0),
        _view(NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_20_VIEW_ID, 1, 0, 1),
        _view(NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_10_VIEW_ID, 1, 1, 0),
        _view(NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_20_VIEW_ID, 1, 1, 1),
        _view(NUMERICAL_DENOMINATOR_TIMESTEP_QUARTER_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID, 2, 0, 0),
    )


def _view_by_id() -> dict[str, NumericalDenominatorView]:
    return {value.view_id: value for value in all_views()}


def _branch_view_ids(stage: NumericalDenominatorStage, branch: NumericalDenominatorBranch) -> tuple[str, ...]:
    if stage in {NumericalDenominatorStage.QUALIFICATION, NumericalDenominatorStage.DEVELOPMENT}:
        if branch is not NumericalDenominatorBranch.FACTORIAL:
            raise ValueError("numerical denominator qualification/development must use the factorial branch")
        return (NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID, NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID, NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_10_VIEW_ID, NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_20_VIEW_ID, NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_20_VIEW_ID)
    mapping = {
        NumericalDenominatorBranch.TIMESTEP: (NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID, NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID, NUMERICAL_DENOMINATOR_TIMESTEP_QUARTER_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID),
        NumericalDenominatorBranch.GRID: (NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID, NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_10_VIEW_ID, NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_41_CORRECTOR_STEPS_10_VIEW_ID),
        NumericalDenominatorBranch.CORRECTOR: (NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID, NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_20_VIEW_ID, NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_40_VIEW_ID),
        NumericalDenominatorBranch.NUMERICAL_INTERACTION: (
            NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID,
            NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID,
            NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_10_VIEW_ID,
            NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_20_VIEW_ID,
            NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_10_VIEW_ID,
            NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_20_VIEW_ID,
            NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_20_VIEW_ID,
            NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_20_VIEW_ID,
        ),
        NumericalDenominatorBranch.OBSERVATION_GAUGE: (NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID, NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_20_VIEW_ID),
        NumericalDenominatorBranch.STABLE_DENOMINATOR: (NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID, NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_20_VIEW_ID),
        NumericalDenominatorBranch.SCIENTIFIC_PARTIAL: (
            NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID,
            NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID,
            NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_10_VIEW_ID,
            NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_20_VIEW_ID,
            NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_20_VIEW_ID,
        ),
    }
    try:
        return mapping[branch]
    except KeyError as error:
        raise ValueError("numerical denominator evaluation has an invalid branch") from error


def views_for(stage: NumericalDenominatorStage, branch: NumericalDenominatorBranch) -> tuple[NumericalDenominatorView, ...]:
    by_id = _view_by_id()
    return tuple(by_id[value] for value in _branch_view_ids(stage, branch))


_FROZEN_ENVIRONMENT_SEEDS: Mapping[str, int] = MappingProxyType({
    'cell.numerical-denominator-development-01': 3437341236442414511,
    'cell.numerical-denominator-development-02': 12065850033608006739,
    'cell.numerical-denominator-development-03': 7498101499735733000,
    'cell.numerical-denominator-development-04': 11865967029166264239,
    'cell.numerical-denominator-development-reserve-01': 4692997795719339579,
    'cell.numerical-denominator-evaluation-01': 10620631481293701260,
    'cell.numerical-denominator-evaluation-02': 10145030238417357578,
    'cell.numerical-denominator-evaluation-03': 11615574333934064101,
    'cell.numerical-denominator-evaluation-04': 8971997029467971251,
    'cell.numerical-denominator-evaluation-05': 7840456778838460423,
    'cell.numerical-denominator-evaluation-06': 2665384003611617283,
    'cell.numerical-denominator-evaluation-07': 14103759009056369881,
    'cell.numerical-denominator-evaluation-08': 13062693799545943345,
    'cell.numerical-denominator-evaluation-reserve-01': 7554790044384178837,
    'cell.numerical-denominator-evaluation-reserve-02': 6349410649264591446,
    'cell.numerical-denominator-qualification-01': 9058584383385139637,
})


def _seed(value: str) -> int:
    return _FROZEN_ENVIRONMENT_SEEDS[value]


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


def _lhs_cell(
    *,
    cell_id: str,
    stage: NumericalDenominatorStage,
    reserve: bool,
    ranks: tuple[int, int, int, int],
    count: int,
    offsets: tuple[Decimal, Decimal, Decimal, Decimal],
) -> NumericalDenominatorCell:
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
    return NumericalDenominatorCell(
        cell_id=cell_id,
        stage=stage,
        reserve=reserve,
        environment_seed=_seed(cell_id),
        initial_temperature_scale=values[0],
        initial_density_nbar=values[1],
        bootstrap_multiplier=values[2],
        inner_transport_scale=values[3],
    )


def development_cells() -> tuple[NumericalDenominatorCell, ...]:
    primary = tuple(
        _lhs_cell(
            cell_id=f"cell.numerical-denominator-development-{index + 1:02d}",
            stage=NumericalDenominatorStage.DEVELOPMENT,
            reserve=False,
            ranks=(
                index,
                (3 * index + 1) % 4,
                (index + 2) % 4,
                (3 * index + 3) % 4,
            ),
            count=4,
            offsets=(
                Decimal("0.0000006"),
                Decimal("0.0000007"),
                Decimal("0.0000008"),
                Decimal("0.0000009"),
            ),
        )
        for index in range(4)
    )
    reserve = (
        NumericalDenominatorCell(
            cell_id='cell.numerical-denominator-development-reserve-01',
            stage=NumericalDenominatorStage.DEVELOPMENT,
            reserve=True,
            environment_seed=_seed('cell.numerical-denominator-development-reserve-01'),
            initial_temperature_scale=Decimal("1.0000011"),
            initial_density_nbar=Decimal("0.8500012"),
            bootstrap_multiplier=Decimal("1.0000013"),
            inner_transport_scale=Decimal("1.0000014"),
        ),
    )
    result = (*primary, *reserve)
    _validate_roster(result)
    validate_predecessor_disjointness(result)
    return result


def evaluation_cells() -> tuple[NumericalDenominatorCell, ...]:
    primary = tuple(
        _lhs_cell(
            cell_id=f"cell.numerical-denominator-evaluation-{index + 1:02d}",
            stage=NumericalDenominatorStage.EVALUATION,
            reserve=False,
            ranks=(
                index,
                (5 * index + 1) % 8,
                (3 * index + 2) % 8,
                (7 * index + 4) % 8,
            ),
            count=8,
            offsets=(
                Decimal("0.0000011"),
                Decimal("0.0000012"),
                Decimal("0.0000013"),
                Decimal("0.0000014"),
            ),
        )
        for index in range(8)
    )
    reserves = (
        NumericalDenominatorCell(
            cell_id='cell.numerical-denominator-evaluation-reserve-01',
            stage=NumericalDenominatorStage.EVALUATION,
            reserve=True,
            environment_seed=_seed('cell.numerical-denominator-evaluation-reserve-01'),
            initial_temperature_scale=Decimal("0.9975016"),
            initial_density_nbar=Decimal("0.8510017"),
            bootstrap_multiplier=Decimal("0.9987518"),
            inner_transport_scale=Decimal("1.0075019"),
        ),
        NumericalDenominatorCell(
            cell_id='cell.numerical-denominator-evaluation-reserve-02',
            stage=NumericalDenominatorStage.EVALUATION,
            reserve=True,
            environment_seed=_seed('cell.numerical-denominator-evaluation-reserve-02'),
            initial_temperature_scale=Decimal("1.0025021"),
            initial_density_nbar=Decimal("0.8490022"),
            bootstrap_multiplier=Decimal("1.0012523"),
            inner_transport_scale=Decimal("0.9925024"),
        ),
    )
    result = (*primary, *reserves)
    _validate_roster((*development_cells(), *result))
    validate_predecessor_disjointness(result)
    return result


def qualification_cells() -> tuple[NumericalDenominatorCell, ...]:
    value = NumericalDenominatorCell(
        cell_id='cell.numerical-denominator-qualification-01',
        stage=NumericalDenominatorStage.QUALIFICATION,
        reserve=False,
        environment_seed=_seed('cell.numerical-denominator-qualification-01'),
        initial_temperature_scale=Decimal("1.0000031"),
        initial_density_nbar=Decimal("0.8500032"),
        bootstrap_multiplier=Decimal("1.0000033"),
        inner_transport_scale=Decimal("1.0000034"),
    )
    _validate_roster((*development_cells(), *evaluation_cells(), value))
    validate_predecessor_disjointness((value,))
    return (value,)


def _validate_roster(cells: Sequence[NumericalDenominatorCell]) -> None:
    ids = tuple(value.cell_id for value in cells)
    keys = tuple(value.coordinate_key for value in cells)
    if len(ids) != len(set(ids)) or len(keys) != len(set(keys)):
        raise ValueError("numerical denominator roster repeats an identity or denominator")
    if any(
        all(value == value.quantize(Decimal("0.000001")) for value in key)
        for key in keys
    ):
        raise ValueError("numerical denominator cell lacks a predecessor-disjoint residue")


def validate_predecessor_disjointness(cells: Sequence[NumericalDenominatorCell]) -> None:
    "Prove exact disjointness from the current Gym--TORAX preparation and preparation-path cohorts.\n\n    Earlier preparation families were frozen on a six-decimal coordinate lattice; the\n    residue check in ``_validate_roster`` covers that earlier family.\n    "

    from .prepared_base import (
        development_cells as prepared_response_development_cells,
        evaluation_master_cells as prepared_response_evaluation_cells,
    )
    from .preparation_path_sensitivity import evaluation_cells as preparation_path_evaluation_cells, qualification_cell as preparation_path_qualification_cell

    current_ids = {value.cell_id for value in cells}
    current_keys = {value.coordinate_key for value in cells}
    prepared_response_cohort = (*prepared_response_development_cells(), *prepared_response_evaluation_cells())
    preparation_path_cohort = (*preparation_path_evaluation_cells(), preparation_path_qualification_cell())
    predecessor_ids = {
        *(value.cell_id for value in prepared_response_cohort),
        *(value.cell_id for value in preparation_path_cohort),
    }
    predecessor_keys = {
        *(value.coordinate_key for value in prepared_response_cohort),
        *(value.coordinate_key for value in preparation_path_cohort),
    }
    if current_ids.intersection(predecessor_ids):
        raise ValueError("numerical denominator cell identity overlaps preparation and preparation-path cohorts")
    if current_keys.intersection(predecessor_keys):
        raise ValueError("numerical denominator cell coordinates overlap preparation and preparation-path cohorts")


def _early_full_word() -> PreparationWord:
    return next(
        value for value in preparation_words() if value.word_id == PREPARED_SOURCE_EARLY_FULL_WORD_ID
    )


def build_numerical_denominator_formal_coverage(
    *,
    register: FormalGapRegister,
    source_manifest: OpenSimulatorSourceManifest,
    denominator_id: str,
    candidate_act_id: str,
    independent_unit_ids: tuple[str, ...],
    numerical_view_ids: tuple[str, ...],
) -> tuple[FormalGapSourceCapabilityInventory, FormalGapCoverage]:
    "Disposition all formal gaps without manufacturing local law action operands."

    inventory = FormalGapSourceCapabilityInventory(
        inventory_id=f"formal-source-inventory.{candidate_act_id.removeprefix('draft.')}",
        denominator_id=denominator_id,
        evidence_world=FormalGapEvidenceWorld.NUMERICAL_SIMULATOR,
        source_materializations=(
            ObjectIdentity.from_record(source_manifest.source_id, source_manifest),
        ),
        present_operand_ids=(),
        satisfied_prerequisite_ids=(),
        independent_unit_ids=tuple(sorted(independent_unit_ids)),
        independent_unit_scope=EvidenceUnitScope.PHYSICAL_INDEPENDENT_UNIT,
        numerical_view_ids=tuple(sorted(numerical_view_ids)),
        available_estimator_family_ids=(),
        available_control_ids=(),
        multiplicity_family_ids=(),
        denominator_inapplicable_gap_ids=(),
        resource_blocked_gap_ids=(),
        requested_claim_ceiling=EvidenceCeiling.MEASUREMENT,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    from empirical_lawhood.planning.formal_analysis import derive_formal_gap_applicability

    applicability = derive_formal_gap_applicability(register, inventory)
    assignments = tuple(
        FormalGapCoverageAssignment(
            gap_id=gap.gap_id,
            disposition=(
                FormalGapCoverageDisposition.UNEVALUABLE_FROM_SOURCE
                if gap.gap_id in _SOURCE_UNEVALUABLE_GAP_IDS
                else FormalGapCoverageDisposition.DEFER_WITH_TYPED_PREREQUISITE
            ),
            readiness_reason=(
                ReadinessStatus.SOURCE_PREREQUISITE_NOT_MET
                if gap.gap_id in _SOURCE_UNEVALUABLE_GAP_IDS
                else ReadinessStatus.PREREQUISITE_NOT_MET
            ),
            reason_codes=(
                ("NUMERICAL_DENOMINATOR_SOURCE_MATHEMATICAL_OBJECT_ABSENT",)
                if gap.gap_id in _SOURCE_UNEVALUABLE_GAP_IDS
                else ("NUMERICAL_DENOMINATOR_CONTROLLED_LOCAL_LAW_ACTION_PREREQUISITE_ABSENT",)
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
        coverage_id=f"formal-gap-coverage.{candidate_act_id.removeprefix('draft.')}",
        register=ObjectIdentity.from_record(register.register_id, register),
        denominator_id=denominator_id,
        candidate_act_id=candidate_act_id,
        applicability=applicability,
        assignments=assignments,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    return inventory, coverage


def build_config(
    *,
    stage: NumericalDenominatorStage,
    branch: NumericalDenominatorBranch,
    source_manifest: OpenSimulatorSourceManifest,
    formal_coverage: FormalGapCoverage,
) -> NumericalDenominatorConfig:
    cells = {
        NumericalDenominatorStage.QUALIFICATION: qualification_cells,
        NumericalDenominatorStage.DEVELOPMENT: development_cells,
        NumericalDenominatorStage.EVALUATION: evaluation_cells,
    }[stage]()
    if stage is NumericalDenominatorStage.QUALIFICATION:
        config_id = NUMERICAL_DENOMINATOR_QUALIFICATION_CONFIG_ID
    elif stage is NumericalDenominatorStage.DEVELOPMENT:
        config_id = NUMERICAL_DENOMINATOR_DEVELOPMENT_CONFIG_ID
    else:
        config_id = f"{NUMERICAL_DENOMINATOR_EVALUATION_CONFIG_PREFIX}{branch.value}-evaluation"
    return NumericalDenominatorConfig(
        config_id=config_id,
        parent_design_id=NUMERICAL_DENOMINATOR_PARENT_ID,
        stage=stage,
        branch=branch,
        source_manifest=ObjectIdentity.from_record(source_manifest.source_id, source_manifest),
        formal_coverage=ObjectIdentity.from_record(
            formal_coverage.coverage_id,
            formal_coverage,
        ),
        cells=tuple(sorted(cells, key=lambda value: value.cell_id)),
        primary_cell_ids=tuple(
            sorted(value.cell_id for value in cells if not value.reserve)
        ),
        reserve_cell_ids=tuple(sorted(value.cell_id for value in cells if value.reserve)),
        word=_early_full_word(),
        views=views_for(stage, branch),
        tolerances=NumericalDenominatorTolerances(),
        common_grid_cells=200,
        evaluator_id=NUMERICAL_DENOMINATOR_EVALUATOR_ID,
        decision_rule_id=NUMERICAL_DENOMINATOR_DECISION_RULE_ID,
        outcome_access=(
            OutcomeAccess.EVALUATION_SEALED
            if stage is NumericalDenominatorStage.EVALUATION
            else OutcomeAccess.DEVELOPMENT_VISIBLE
        ),
        maximum_evidence_ceiling=EvidenceCeiling.MEASUREMENT,
    )


@dataclass(frozen=True, slots=True)
class NumericalDenominatorGaugeObservation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/numerical-denominator-gauge-observation'

    receiver_clock_s: int
    source_volume_density_m3: Decimal
    source_line_density_m3: Decimal
    source_volume_fgw: Decimal
    source_line_fgw: Decimal
    common_volume_density_m3: Decimal
    common_line_density_m3: Decimal
    common_volume_fgw: Decimal
    common_line_fgw: Decimal
    lcfs_current_a: Decimal
    source_volume_identity_residual: Decimal
    source_line_identity_residual: Decimal

    def __post_init__(self) -> None:
        if not 1 <= self.receiver_clock_s <= 150:
            raise ValueError("numerical denominator gauge observation clock is outside the horizon")
        positive = (
            self.source_volume_density_m3,
            self.source_line_density_m3,
            self.source_volume_fgw,
            self.source_line_fgw,
            self.common_volume_density_m3,
            self.common_line_density_m3,
            self.common_volume_fgw,
            self.common_line_fgw,
            self.lcfs_current_a,
        )
        if any(value <= 0 or not value.is_finite() for value in positive):
            raise ValueError("numerical denominator gauge observation must be positive and finite")
        if (
            self.source_volume_identity_residual < 0
            or self.source_line_identity_residual < 0
        ):
            raise ValueError("numerical denominator identity residual cannot be negative")


@dataclass(frozen=True, slots=True)
class NumericalDenominatorTransition(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/numerical-denominator-transition'

    transition_id: str
    receiver_clock_s: int
    action: ActionLedger
    receiver_values: tuple[NamedDecimal, ...]
    gauge: NumericalDenominatorGaugeObservation | None
    diagnostic_fields: tuple[DenseField, ...]
    valid: bool
    terminated: bool
    truncated: bool
    missing_primary_operand_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.transition_id, field_name="transition_id")
        require_sorted_unique_strings(
            tuple(value.value_id for value in self.receiver_values),
            field_name="receiver_values",
        )
        require_sorted_unique_strings(
            tuple(value.field_id for value in self.diagnostic_fields),
            field_name="diagnostic_fields",
        )
        require_sorted_unique_strings(
            self.missing_primary_operand_ids,
            field_name="missing_primary_operand_ids",
        )
        if self.gauge is not None and self.gauge.receiver_clock_s != self.receiver_clock_s:
            raise ValueError("numerical denominator transition/gauge clocks differ")


@dataclass(frozen=True, slots=True)
class NumericalDenominatorEpisode(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/numerical-denominator-episode'

    episode_id: str
    config: ObjectIdentity
    cell: ObjectIdentity
    word: ObjectIdentity
    view: ObjectIdentity
    transitions: tuple[NumericalDenominatorTransition, ...]
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
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if tuple(value.receiver_clock_s for value in self.transitions) != tuple(
            range(1, len(self.transitions) + 1)
        ):
            raise ValueError("numerical denominator episode clocks are not contiguous")
        if self.missing_required_clocks_s != tuple(range(len(self.transitions) + 1, 151)):
            raise ValueError("numerical denominator episode missing-clock inventory differs")
        if self.backend != "cpu" or self.precision != "float64":
            raise ValueError("numerical denominator episode changed backend or precision")
        if self.runtime_seconds < 0:
            raise ValueError("numerical denominator runtime cannot be negative")
        if self.last_valid_clock_s is not None and not 1 <= self.last_valid_clock_s <= 150:
            raise ValueError("numerical denominator last-valid clock is invalid")
        if self.disposition is EpisodeDisposition.COMPLETE and (
            len(self.transitions) != 150
            or self.reason_codes
            or self.last_valid_clock_s != 150
        ):
            raise ValueError("complete numerical denominator episode is incomplete")


@dataclass(frozen=True, slots=True)
class NumericalDenominatorBatch(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/numerical-denominator-batch'

    batch_id: str
    config: ObjectIdentity
    stage: NumericalDenominatorStage
    episodes: tuple[NumericalDenominatorEpisode, ...]
    active_cell_ids: tuple[str, ...]
    excluded_cell_ids: tuple[str, ...]
    substitutions: tuple[PreparedBaseCellSubstitution, ...]
    expected_active_cell_count: int
    complete_active_roster: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.batch_id, field_name="batch_id")
        require_sorted_unique_strings(self.active_cell_ids, field_name="active_cell_ids")
        require_sorted_unique_strings(
            self.excluded_cell_ids,
            field_name="excluded_cell_ids",
        )
        if set(self.active_cell_ids) & set(self.excluded_cell_ids):
            raise ValueError("numerical denominator active/excluded cells overlap")
        if self.complete_active_roster != (
            len(self.active_cell_ids) == self.expected_active_cell_count
        ):
            raise ValueError("numerical denominator batch completeness differs from roster")
        episode_ids = tuple(value.episode_id for value in self.episodes)
        substitution_ids = tuple(value.substitution_id for value in self.substitutions)
        if len(episode_ids) != len(set(episode_ids)) or len(substitution_ids) != len(
            set(substitution_ids)
        ):
            raise ValueError("numerical denominator batch repeats an episode or substitution")
        expected_access = (
            OutcomeAccess.EVALUATION_SEALED
            if self.stage is NumericalDenominatorStage.EVALUATION
            else OutcomeAccess.DEVELOPMENT_VISIBLE
        )
        if self.outcome_access is not expected_access or any(
            value.outcome_access is not expected_access for value in self.episodes
        ):
            raise ValueError("numerical denominator batch has the wrong stage outcome access")


def _decimal(value: Any) -> Decimal:
    return Decimal(str(float(np.asarray(value).reshape(-1)[0])))


def _relative(numerator: Decimal, denominator: Decimal) -> Decimal:
    if denominator == 0:
        return Decimal("Infinity")
    return abs(numerator / denominator)


def _common_gauge(
    environment: Any,
    state: Mapping[str, Any],
    receiver_clock_s: int,
    common_grid_cells: int,
) -> NumericalDenominatorGaugeObservation:
    sim_state = environment.torax_app.current_sim_state
    if sim_state is None:
        raise ValueError("numerical denominator TORAX state is absent")
    rho = np.asarray(sim_state.geometry.rho_norm, dtype=np.float64)
    density = np.asarray(sim_state.core_profiles.n_e.value, dtype=np.float64)
    vpr = np.asarray(sim_state.geometry.vpr, dtype=np.float64)
    common_rho = (np.arange(common_grid_cells, dtype=np.float64) + 0.5) / float(
        common_grid_cells
    )
    common_density = np.interp(common_rho, rho, density)
    common_vpr = np.interp(common_rho, rho, vpr)
    volume_density = float(np.sum(common_density * common_vpr) / np.sum(common_vpr))
    line_density = float(np.mean(common_density))
    current = float(np.asarray(sim_state.core_profiles.Ip_profile_face[-1]))
    minor_radius = float(np.asarray(sim_state.geometry.a_minor))
    greenwald_limit_m3 = current * 1e-6 / (math.pi * minor_radius**2) * 1e20
    if not math.isfinite(greenwald_limit_m3) or greenwald_limit_m3 <= 0:
        raise ValueError("numerical denominator Greenwald denominator is invalid")
    source_volume_density = _decimal(state["scalars"]["n_e_volume_avg"])
    source_line_density = _decimal(state["scalars"]["n_e_line_avg"])
    source_volume_fgw = _decimal(state["scalars"]["fgw_n_e_volume_avg"])
    source_line_fgw = _decimal(state["scalars"]["fgw_n_e_line_avg"])
    common_volume_fgw = Decimal(str(volume_density / greenwald_limit_m3))
    common_line_fgw = Decimal(str(line_density / greenwald_limit_m3))
    calculated_source_volume = source_volume_density / Decimal(str(greenwald_limit_m3))
    calculated_source_line = source_line_density / Decimal(str(greenwald_limit_m3))
    return NumericalDenominatorGaugeObservation(
        receiver_clock_s=receiver_clock_s,
        source_volume_density_m3=source_volume_density,
        source_line_density_m3=source_line_density,
        source_volume_fgw=source_volume_fgw,
        source_line_fgw=source_line_fgw,
        common_volume_density_m3=Decimal(str(volume_density)),
        common_line_density_m3=Decimal(str(line_density)),
        common_volume_fgw=common_volume_fgw,
        common_line_fgw=common_line_fgw,
        lcfs_current_a=Decimal(str(current)),
        source_volume_identity_residual=abs(
            source_volume_fgw - calculated_source_volume
        ),
        source_line_identity_residual=abs(source_line_fgw - calculated_source_line),
    )


def _array_field(field_id: str, unit: str, raw: Any) -> DenseField:
    if raw is None:
        return DenseField(
            field_id=field_id,
            unit=unit,
            values=(),
            source_available=False,
        )
    values = tuple(_decimal(value) for value in np.asarray(raw).reshape(-1))
    return DenseField(
        field_id=field_id,
        unit=unit,
        values=values,
        source_available=bool(values),
    )


def _diagnostic_fields(environment: Any, receiver_clock_s: int) -> tuple[DenseField, ...]:
    if receiver_clock_s not in _DIAGNOSTIC_CLOCKS:
        return ()
    sim_state = environment.torax_app.current_sim_state
    if sim_state is None:
        return ()
    fields = [
        _array_field(
            "profile.electron-density",
            "m-3",
            sim_state.core_profiles.n_e.value,
        ),
        _array_field("profile.q", "1", sim_state.core_profiles.q_face),
        _array_field("profile.psi", "Wb", sim_state.core_profiles.psi.value),
        _array_field(
            "profile.current",
            "A",
            sim_state.core_profiles.Ip_profile_face,
        ),
        _array_field("grid.rho-cell", "1", sim_state.geometry.rho_norm),
        _array_field("grid.rho-face", "1", sim_state.geometry.rho_face_norm),
        _array_field("geometry.vpr", "m3", sim_state.geometry.vpr),
        _array_field(
            "boundary.electron-density-right",
            "m-3",
            sim_state.core_profiles.n_e.right_face_constraint,
        ),
        _array_field(
            "transport.d-electron",
            "m2.s-1",
            sim_state.core_transport.d_face_el,
        ),
        _array_field(
            "transport.v-electron",
            "m.s-1",
            sim_state.core_transport.v_face_el,
        ),
        _array_field(
            "transport.d-electron-pereverzev",
            "m2.s-1",
            sim_state.core_transport.d_face_el_pereverzev,
        ),
        _array_field(
            "transport.v-electron-pereverzev",
            "m.s-1",
            sim_state.core_transport.v_face_el_pereverzev,
        ),
    ]
    for source_id, raw in sorted(sim_state.core_sources.n_e.items()):
        stable = str(source_id).lower().replace("_", "-")
        fields.append(_array_field(f"source.density.{stable}", "m-3.s-1", raw))
    fields.extend(
        (
            _array_field("profile.density-source", "m-3.s-1", None),
            _array_field("profile.density-flux", "m-2.s-1", None),
            _array_field("profile.particle-balance", "m-3.s-1", None),
        )
    )
    return tuple(sorted(fields, key=lambda value: value.field_id))


def acquire_episode(
    *,
    config: NumericalDenominatorConfig,
    cell_id: str,
    view_id: str,
) -> NumericalDenominatorEpisode:
    runtime_reasons = verify_cpu_runtime()
    if runtime_reasons:
        raise RuntimeError(",".join(runtime_reasons))
    cell = next(value for value in config.cells if value.cell_id == cell_id)
    view = next(value for value in config.views if value.view_id == view_id)
    word = config.word
    environment = _environment(cast(Any, cell), cast(Any, view))
    transitions: list[NumericalDenominatorTransition] = []
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
            _, reward, terminated, truncated, info = environment.step(
                _action_mapping(requested)
            )
            accepted = requested
            applied = _native_action(environment.torax_app.config.get_current_action_values())
            realized = _realized_action(environment, applied)
            receiver_clock = request_clock + 1
            state = environment.state or {}
            receivers, missing = _receiver_values(state)
            gauge: NumericalDenominatorGaugeObservation | None
            try:
                gauge = _common_gauge(
                    environment,
                    state,
                    receiver_clock,
                    config.common_grid_cells,
                )
            except (KeyError, TypeError, ValueError, FloatingPointError):
                gauge = None
                missing = tuple(
                    sorted(
                        {
                            *missing,
                            "receiver.common-gauge-density-fgw",
                        }
                    )
                )
            valid = (
                float(reward) != -1000.0
                and not missing
                and all(math.isfinite(float(value.value)) for value in receivers)
                and gauge is not None
            )
            clipped = bool(info.get("action_clipped", False))
            transitions.append(
                NumericalDenominatorTransition(
                    transition_id=(
                        f"transition.{cell.cell_id.removeprefix('cell.')}."
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
                    gauge=gauge,
                    diagnostic_fields=_diagnostic_fields(environment, receiver_clock),
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
                reasons.add("PRIMARY_OR_COMMON_GAUGE_OPERAND_ABSENT")
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
        # Once a simulator trajectory exists, an untyped source exception is
        # conservatively scientific/numerical evidence.  It cannot consume a
        # reserve as though it were an outcome-blind infrastructure failure.
        if transitions:
            disposition = EpisodeDisposition.NUMERICAL_INVALID
            reasons.add(f"SIMULATOR_{type(error).__name__.upper()}")
        else:
            disposition = EpisodeDisposition.TECHNICAL_OBSERVATION_FAILURE
            reasons.add(f"TECHNICAL_INITIALIZATION_{type(error).__name__.upper()}")
    finally:
        environment.close()
    if len(transitions) < 150 and disposition is EpisodeDisposition.COMPLETE:
        disposition = EpisodeDisposition.PARTIAL_VALID_PREFIX
        reasons.add("REQUIRED_CLOCKS_ABSENT")
    last_valid = max(
        (value.receiver_clock_s for value in transitions if value.valid),
        default=None,
    )
    return NumericalDenominatorEpisode(
        episode_id=(
            f"episode.{config.stage.value.lower()}."
            f"{cell.cell_id.removeprefix('cell.')}."
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
        runtime_seconds=Decimal(f"{time.monotonic() - started:.9f}"),
        backend="cpu",
        precision="float64",
        outcome_access=config.outcome_access,
    )


_TECHNICAL_RESERVE_DISPOSITIONS = frozenset(
    {
        EpisodeDisposition.TECHNICAL_OBSERVATION_FAILURE,
    }
)


def acquire_stage(config: NumericalDenominatorConfig) -> NumericalDenominatorBatch:
    """Acquire whole view bundles with outcome-blind technical replacement."""

    episodes: list[NumericalDenominatorEpisode] = []
    active: list[str] = []
    excluded: list[str] = []
    substitutions: list[PreparedBaseCellSubstitution] = []
    reserve_ids = iter(config.reserve_cell_ids)
    for intended_cell_id in config.primary_cell_ids:
        current_cell_id = intended_cell_id
        while True:
            bundle: list[NumericalDenominatorEpisode] = []
            technical_failure = False
            for view in config.views:
                episode = acquire_episode(
                    config=config,
                    cell_id=current_cell_id,
                    view_id=view.view_id,
                )
                bundle.append(episode)
                episodes.append(episode)
                if episode.disposition in _TECHNICAL_RESERVE_DISPOSITIONS:
                    technical_failure = True
                    break
            if not technical_failure:
                active.append(current_cell_id)
                break
            excluded.append(current_cell_id)
            try:
                replacement = next(reserve_ids)
            except StopIteration:
                break
            substitutions.append(
                PreparedBaseCellSubstitution(
                    substitution_id=(
                        f"substitution.{config.config_id.removeprefix('config.')}."
                        f"{intended_cell_id.removeprefix('cell.')}."
                        f"{replacement.removeprefix('cell.')}"
                    ),
                    intended_cell_id=intended_cell_id,
                    replacement_cell_id=replacement,
                    excluded_episode_ids=tuple(
                        sorted(value.episode_id for value in bundle)
                    ),
                    reason_codes=tuple(
                        sorted(
                            {
                                reason
                                for episode in bundle
                                for reason in episode.reason_codes
                            }
                        )
                    )
                    or ("TECHNICAL_BUNDLE_FAILURE",),
                    receiver_outcome_released_to_evaluator=False,
                )
            )
            current_cell_id = replacement
    return NumericalDenominatorBatch(
        batch_id=f"batch.{config.config_id.removeprefix('config.')}",
        config=ObjectIdentity.from_record(config.config_id, config),
        stage=config.stage,
        episodes=tuple(sorted(episodes, key=lambda value: value.episode_id)),
        active_cell_ids=tuple(sorted(active)),
        excluded_cell_ids=tuple(sorted(excluded)),
        substitutions=tuple(
            sorted(substitutions, key=lambda value: value.substitution_id)
        ),
        expected_active_cell_count=len(config.primary_cell_ids),
        complete_active_roster=len(active) == len(config.primary_cell_ids),
        outcome_access=config.outcome_access,
    )


def _receivers(episode: NumericalDenominatorEpisode) -> dict[int, dict[str, Decimal]]:
    return {
        transition.receiver_clock_s: {
            item.value_id: item.value for item in transition.receiver_values
        }
        for transition in episode.transitions
    }


def _gauges(episode: NumericalDenominatorEpisode) -> dict[int, NumericalDenominatorGaugeObservation]:
    return {
        transition.receiver_clock_s: transition.gauge
        for transition in episode.transitions
        if transition.gauge is not None
    }


def _complete_receiver_pass(episode: NumericalDenominatorEpisode, *, common: bool) -> bool:
    if episode.disposition is not EpisodeDisposition.COMPLETE:
        return False
    receivers = _receivers(episode)
    gauges = _gauges(episode)
    if set(gauges) != set(range(1, 151)):
        return False

    def fgw(clock: int) -> Decimal:
        return (
            gauges[clock].common_volume_fgw
            if common
            else gauges[clock].source_volume_fgw
        )

    return (
        all(receivers[clock]["receiver.q-min"] >= Decimal("1.02") for clock in range(1, 31))
        and all(fgw(clock) <= Decimal("0.95") for clock in range(1, 31))
        and all(
            receivers[clock]["receiver.q-min"] >= Decimal("1.02")
            for clock in _PHASE_CLOCKS
        )
        and all(
            receivers[clock]["receiver.h98"] >= Decimal("0.8")
            for clock in _PHASE_CLOCKS
        )
        and all(fgw(clock) <= Decimal("0.95") for clock in _PHASE_CLOCKS)
        and all(receivers[clock]["receiver.q95"] >= Decimal("3") for clock in range(1, 151))
        and all(
            receivers[clock]["receiver.beta-n"] <= Decimal("3.5")
            for clock in range(1, 151)
        )
        and all(
            receivers[clock]["receiver.radiated-fraction"] <= Decimal("0.5")
            for clock in range(1, 151)
        )
        and receivers[150]["receiver.h98"] >= Decimal("0.8")
    )


def _max_phase_path_difference(left: NumericalDenominatorEpisode, right: NumericalDenominatorEpisode) -> Decimal:
    left_gauges = _gauges(left)
    right_gauges = _gauges(right)
    return max(
        abs(
            right_gauges[clock].common_volume_fgw
            - left_gauges[clock].common_volume_fgw
        )
        for clock in _PHASE_CLOCKS
    )


def _clock_contract_status(episode: NumericalDenominatorEpisode, clock: int) -> tuple[bool, ...]:
    receivers = _receivers(episode)[clock]
    gauge = _gauges(episode)[clock]
    statuses = [
        receivers["receiver.q95"] >= Decimal("3"),
        receivers["receiver.beta-n"] <= Decimal("3.5"),
        receivers["receiver.radiated-fraction"] <= Decimal("0.5"),
    ]
    if clock in range(1, 31) or clock in _PHASE_CLOCKS:
        statuses.extend(
            (
                receivers["receiver.q-min"] >= Decimal("1.02"),
                gauge.common_volume_fgw <= Decimal("0.95"),
            )
        )
    if clock in _PHASE_CLOCKS or clock == 150:
        statuses.append(receivers["receiver.h98"] >= Decimal("0.8"))
    return tuple(statuses)


def _first_decision_change(left: NumericalDenominatorEpisode, right: NumericalDenominatorEpisode) -> int | None:
    return next(
        (
            clock
            for clock in range(1, 151)
            if _clock_contract_status(left, clock)
            != _clock_contract_status(right, clock)
        ),
        None,
    )


def _decision_margin_min(episode: NumericalDenominatorEpisode) -> Decimal:
    receivers = _receivers(episode)
    gauges = _gauges(episode)
    margins: list[Decimal] = []
    for clock in range(1, 151):
        row = receivers[clock]
        margins.extend(
            (
                (row["receiver.q95"] - Decimal("3")) / Decimal("3"),
                (Decimal("3.5") - row["receiver.beta-n"]) / Decimal("3.5"),
                (
                    Decimal("0.5") - row["receiver.radiated-fraction"]
                )
                / Decimal("0.5"),
            )
        )
        if clock in range(1, 31) or clock in _PHASE_CLOCKS:
            margins.extend(
                (
                    (row["receiver.q-min"] - Decimal("1.02")) / Decimal("1.02"),
                    (
                        Decimal("0.95") - gauges[clock].common_volume_fgw
                    )
                    / Decimal("0.95"),
                )
            )
        if clock in _PHASE_CLOCKS or clock == 150:
            margins.append(
                (row["receiver.h98"] - Decimal("0.8")) / Decimal("0.8")
            )
    return min(margins)


@dataclass(frozen=True, slots=True)
class _CompatibilityMetrics:
    fgw_difference_max: Decimal
    qmin_difference_max: Decimal
    h98_difference_max: Decimal
    density_relative_difference_max: Decimal
    current_relative_difference_max: Decimal
    decisions_match: bool
    first_decision_changing_clock_s: int | None
    fine_decision_margin_min: Decimal
    line_volume_disagreement_max: Decimal
    identity_residual_max: Decimal


def _compatibility_metrics(
    left: NumericalDenominatorEpisode,
    right: NumericalDenominatorEpisode,
) -> _CompatibilityMetrics:
    left_gauges = _gauges(left)
    right_gauges = _gauges(right)
    left_receivers = _receivers(left)
    right_receivers = _receivers(right)
    return _CompatibilityMetrics(
        fgw_difference_max=_max_phase_path_difference(left, right),
        qmin_difference_max=max(
            abs(
                right_receivers[clock]["receiver.q-min"]
                - left_receivers[clock]["receiver.q-min"]
            )
            / Decimal("1.02")
            for clock in _PHASE_CLOCKS
        ),
        h98_difference_max=max(
            abs(
                right_receivers[clock]["receiver.h98"]
                - left_receivers[clock]["receiver.h98"]
            )
            for clock in _PHASE_CLOCKS
        ),
        density_relative_difference_max=max(
            _relative_difference(
                left_gauges[clock].common_volume_density_m3,
                right_gauges[clock].common_volume_density_m3,
            )
            for clock in _PHASE_CLOCKS
        ),
        current_relative_difference_max=max(
            _relative_difference(
                left_gauges[clock].lcfs_current_a,
                right_gauges[clock].lcfs_current_a,
            )
            for clock in _PHASE_CLOCKS
        ),
        decisions_match=(
            _complete_receiver_pass(left, common=True)
            == _complete_receiver_pass(right, common=True)
        ),
        first_decision_changing_clock_s=_first_decision_change(left, right),
        fine_decision_margin_min=_decision_margin_min(right),
        line_volume_disagreement_max=max(
            abs(value.common_line_fgw - value.common_volume_fgw)
            for value in right_gauges.values()
        ),
        identity_residual_max=max(
            max(
                value.source_volume_identity_residual,
                value.source_line_identity_residual,
            )
            for value in (*left_gauges.values(), *right_gauges.values())
        ),
    )


def _interaction_residual(episodes: Mapping[str, NumericalDenominatorEpisode]) -> Decimal:
    gauges = {key: _gauges(value) for key, value in episodes.items()}
    return max(
        abs(
            (gauges[NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_20_VIEW_ID][clock].common_volume_fgw - gauges[NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID][clock].common_volume_fgw)
            - (gauges[NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID][clock].common_volume_fgw - gauges[NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID][clock].common_volume_fgw)
            - (gauges[NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_10_VIEW_ID][clock].common_volume_fgw - gauges[NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID][clock].common_volume_fgw)
            - (gauges[NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_20_VIEW_ID][clock].common_volume_fgw - gauges[NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID][clock].common_volume_fgw)
        )
        for clock in _PHASE_CLOCKS
    )


@dataclass(frozen=True, slots=True)
class NumericalDenominatorDevelopmentCellSummary(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/numerical-denominator-development-cell-summary'

    summary_id: str
    cell_id: str
    observation_difference_max: Decimal
    timestep_effect_max: Decimal
    grid_effect_max: Decimal
    corrector_effect_max: Decimal
    interaction_residual_max: Decimal
    material_axis_ids: tuple[str, ...]
    observation_sensitive: bool
    interaction_material: bool
    complete: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.summary_id, field_name="summary_id")
        validate_stable_id(self.cell_id, field_name="cell_id")
        require_sorted_unique_strings(
            self.material_axis_ids,
            field_name="material_axis_ids",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")


@dataclass(frozen=True, slots=True)
class NumericalDenominatorDevelopmentDecision(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/numerical-denominator-development-decision'

    decision_id: str
    config: ObjectIdentity
    summaries: tuple[NumericalDenominatorDevelopmentCellSummary, ...]
    selected_branch: NumericalDenominatorBranch | None
    reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess
    maximum_evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.decision_id, field_name="decision_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if (
            self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
            or self.maximum_evidence_ceiling is not EvidenceCeiling.MEASUREMENT
        ):
            raise ValueError("numerical denominator development decision exceeds measurement/development access")


def _development_summary(
    cell_id: str,
    episodes: Mapping[str, NumericalDenominatorEpisode],
    tolerances: NumericalDenominatorTolerances,
) -> NumericalDenominatorDevelopmentCellSummary:
    reasons: set[str] = set()
    expected = {NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID, NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID, NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_10_VIEW_ID, NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_20_VIEW_ID, NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_20_VIEW_ID}
    complete = set(episodes) == expected and all(
        value.disposition is EpisodeDisposition.COMPLETE for value in episodes.values()
    )
    if not complete:
        reasons.add("NUMERICAL_DENOMINATOR_DEVELOPMENT_CELL_INCOMPLETE")
        zero = Decimal(0)
        return NumericalDenominatorDevelopmentCellSummary(
            summary_id=f"summary.numerical-denominator.development.{cell_id.removeprefix('cell.')}",
            cell_id=cell_id,
            observation_difference_max=zero,
            timestep_effect_max=zero,
            grid_effect_max=zero,
            corrector_effect_max=zero,
            interaction_residual_max=zero,
            material_axis_ids=(),
            observation_sensitive=False,
            interaction_material=False,
            complete=False,
            reason_codes=tuple(sorted(reasons)),
        )
    observation_difference = max(
        abs(gauge.source_volume_fgw - gauge.common_volume_fgw)
        for episode in episodes.values()
        for gauge in _gauges(episode).values()
    )
    observation_decision_change = any(
        _complete_receiver_pass(value, common=False)
        != _complete_receiver_pass(value, common=True)
        for value in episodes.values()
    )
    effects = {
        "axis.corrector": _max_phase_path_difference(
            episodes[NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID], episodes[NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_20_VIEW_ID]
        ),
        "axis.grid": _max_phase_path_difference(
            episodes[NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID], episodes[NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_10_VIEW_ID]
        ),
        "axis.timestep": _max_phase_path_difference(
            episodes[NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID], episodes[NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID]
        ),
    }
    view_by_axis = {
        "axis.corrector": NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_20_VIEW_ID,
        "axis.grid": NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_10_VIEW_ID,
        "axis.timestep": NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID,
    }
    material = tuple(
        sorted(
            axis
            for axis, effect in effects.items()
            if effect > tolerances.material_fgw_absolute
            or _complete_receiver_pass(episodes[NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID], common=True)
            != _complete_receiver_pass(episodes[view_by_axis[axis]], common=True)
        )
    )
    interaction = _interaction_residual(episodes)
    return NumericalDenominatorDevelopmentCellSummary(
        summary_id=f"summary.numerical-denominator.development.{cell_id.removeprefix('cell.')}",
        cell_id=cell_id,
        observation_difference_max=observation_difference,
        timestep_effect_max=effects["axis.timestep"],
        grid_effect_max=effects["axis.grid"],
        corrector_effect_max=effects["axis.corrector"],
        interaction_residual_max=interaction,
        material_axis_ids=material,
        observation_sensitive=(
            observation_decision_change
            or observation_difference > tolerances.observation_fgw_absolute
        ),
        interaction_material=interaction > tolerances.interaction_fgw_absolute,
        complete=True,
        reason_codes=(),
    )


def adjudicate_development(config: NumericalDenominatorConfig, batch: NumericalDenominatorBatch) -> NumericalDenominatorDevelopmentDecision:
    if config.stage is not NumericalDenominatorStage.DEVELOPMENT:
        raise ValueError("numerical denominator development adjudicator requires development config")
    reasons: set[str] = set()
    if not batch.complete_active_roster:
        reasons.add("NUMERICAL_DENOMINATOR_DEVELOPMENT_ACTIVE_ROSTER_INCOMPLETE")
    by_cell: dict[str, dict[str, NumericalDenominatorEpisode]] = {
        cell_id: {} for cell_id in batch.active_cell_ids
    }
    for episode in batch.episodes:
        if episode.cell.object_id in by_cell:
            by_cell[episode.cell.object_id][episode.view.object_id] = episode
    summaries = tuple(
        _development_summary(cell_id, by_cell[cell_id], config.tolerances)
        for cell_id in sorted(by_cell)
    )
    selected: NumericalDenominatorBranch | None
    if reasons:
        selected = None
    elif any(
        episode.disposition in _TECHNICAL_RESERVE_DISPOSITIONS
        for episode in batch.episodes
        if episode.cell.object_id in set(batch.active_cell_ids)
    ):
        reasons.add("NUMERICAL_DENOMINATOR_TECHNICAL_OBSERVATION_FAILURE")
        selected = None
    elif any(
        episode.disposition is EpisodeDisposition.UNEVALUABLE_OPERAND
        for episode in batch.episodes
        if episode.cell.object_id in set(batch.active_cell_ids)
    ):
        reasons.add("NUMERICAL_DENOMINATOR_UNEVALUABLE_SOURCE_OPERAND")
        selected = None
    elif any(not value.complete for value in summaries):
        selected = NumericalDenominatorBranch.SCIENTIFIC_PARTIAL
    elif any(value.observation_sensitive for value in summaries):
        selected = NumericalDenominatorBranch.OBSERVATION_GAUGE
    else:
        axes = {axis for value in summaries for axis in value.material_axis_ids}
        if any(value.interaction_material for value in summaries) or len(axes) > 1:
            selected = NumericalDenominatorBranch.NUMERICAL_INTERACTION
        elif axes == {"axis.timestep"}:
            selected = NumericalDenominatorBranch.TIMESTEP
        elif axes == {"axis.grid"}:
            selected = NumericalDenominatorBranch.GRID
        elif axes == {"axis.corrector"}:
            selected = NumericalDenominatorBranch.CORRECTOR
        else:
            selected = NumericalDenominatorBranch.STABLE_DENOMINATOR
    return NumericalDenominatorDevelopmentDecision(
        decision_id='decision.numerical-denominator.numerical-factorial-development',
        config=ObjectIdentity.from_record(config.config_id, config),
        summaries=summaries,
        selected_branch=selected,
        reason_codes=tuple(sorted(reasons)),
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        maximum_evidence_ceiling=EvidenceCeiling.MEASUREMENT,
    )


@dataclass(frozen=True, slots=True)
class NumericalDenominatorEvaluationCellSummary(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/numerical-denominator-evaluation-cell-summary'

    summary_id: str
    cell_id: str
    stable: bool
    fine_receiver_pass: bool | None
    first_difference_max: Decimal | None
    last_difference_max: Decimal | None
    contraction_ratio: Decimal | None
    observation_difference_max: Decimal | None
    interaction_residual_max: Decimal | None
    reason_codes: tuple[str, ...]
    qmin_difference_max: Decimal | None = None
    h98_difference_max: Decimal | None = None
    density_relative_difference_max: Decimal | None = None
    current_relative_difference_max: Decimal | None = None
    first_decision_changing_clock_s: int | None = None
    fine_decision_margin_min: Decimal | None = None
    line_volume_disagreement_max: Decimal | None = None
    identity_residual_max: Decimal | None = None
    material_axis_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_stable_id(self.summary_id, field_name="summary_id")
        validate_stable_id(self.cell_id, field_name="cell_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        require_sorted_unique_strings(
            self.material_axis_ids,
            field_name="material_axis_ids",
        )
        if (
            self.first_decision_changing_clock_s is not None
            and not 1 <= self.first_decision_changing_clock_s <= 150
        ):
            raise ValueError("numerical denominator first decision-changing clock is invalid")


@dataclass(frozen=True, slots=True)
class NumericalDenominatorEvaluationAdjudication(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/numerical-denominator-evaluation-adjudication'

    adjudication_id: str
    config: ObjectIdentity
    primary_result: NumericalDenominatorPrimaryResult
    branch: NumericalDenominatorBranch
    summaries: tuple[NumericalDenominatorEvaluationCellSummary, ...]
    selected_denominator_view_id: str | None
    handoff_id: str | None
    reason_codes: tuple[str, ...]
    maximum_evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.adjudication_id, field_name="adjudication_id")
        if self.selected_denominator_view_id is not None:
            validate_stable_id(
                self.selected_denominator_view_id,
                field_name="selected_denominator_view_id",
            )
        if self.handoff_id is not None:
            validate_stable_id(self.handoff_id, field_name="handoff_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        selected = self.primary_result is NumericalDenominatorPrimaryResult.QUALIFIED_STABLE_PASS
        if selected != (
            self.selected_denominator_view_id is not None and self.handoff_id is not None
        ):
            raise ValueError("numerical denominator denominator handoff differs from qualified pass")
        if (
            self.maximum_evidence_ceiling is not EvidenceCeiling.MEASUREMENT
            or self.outcome_access is not OutcomeAccess.EVALUATION_REVEALED
            or self.visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE
        ):
            raise ValueError("numerical denominator evaluation adjudication exceeds measurement/reveal")


def _relative_difference(left: Decimal, right: Decimal) -> Decimal:
    denominator = max(abs(left), abs(right), Decimal("1e-30"))
    return abs(right - left) / denominator


def _axis_evaluation_summary(
    config: NumericalDenominatorConfig,
    cell_id: str,
    episodes: Mapping[str, NumericalDenominatorEpisode],
) -> NumericalDenominatorEvaluationCellSummary:
    ordered = _branch_view_ids(NumericalDenominatorStage.EVALUATION, config.branch)
    if set(episodes) != set(ordered) or any(
        value.disposition is not EpisodeDisposition.COMPLETE for value in episodes.values()
    ):
        return NumericalDenominatorEvaluationCellSummary(
            summary_id=f"summary.numerical-denominator.evaluation.{cell_id.removeprefix('cell.')}",
            cell_id=cell_id,
            stable=False,
            fine_receiver_pass=None,
            first_difference_max=None,
            last_difference_max=None,
            contraction_ratio=None,
            observation_difference_max=None,
            interaction_residual_max=None,
            reason_codes=("NUMERICAL_DENOMINATOR_EVALUATION_CELL_INCOMPLETE",),
        )
    coarse, middle, fine = (episodes[value] for value in ordered)
    first = _max_phase_path_difference(coarse, middle)
    metrics = _compatibility_metrics(middle, fine)
    last = metrics.fgw_difference_max
    ratio = Decimal(0) if first == 0 and last == 0 else (
        Decimal("Infinity") if first == 0 else last / first
    )
    fine_gauges = _gauges(fine)
    contracts = (
        last <= config.tolerances.stable_fgw_absolute,
        (
            ratio <= config.tolerances.contraction_ratio
            or (
                first <= config.tolerances.stable_fgw_absolute
                and last <= config.tolerances.stable_fgw_absolute
            )
        ),
        metrics.qmin_difference_max <= config.tolerances.stable_qmin_normalized,
        metrics.h98_difference_max <= config.tolerances.stable_h98_absolute,
        metrics.density_relative_difference_max
        <= config.tolerances.stable_density_relative,
        metrics.current_relative_difference_max
        <= config.tolerances.stable_current_relative,
        metrics.decisions_match,
        metrics.identity_residual_max
        <= config.tolerances.identity_residual_absolute,
    )
    return NumericalDenominatorEvaluationCellSummary(
        summary_id=f"summary.numerical-denominator.evaluation.{cell_id.removeprefix('cell.')}",
        cell_id=cell_id,
        stable=all(contracts),
        fine_receiver_pass=_complete_receiver_pass(fine, common=True),
        first_difference_max=first,
        last_difference_max=last,
        contraction_ratio=ratio,
        observation_difference_max=max(
            abs(value.source_volume_fgw - value.common_volume_fgw)
            for value in fine_gauges.values()
        ),
        interaction_residual_max=None,
        reason_codes=tuple(
            sorted(
                reason
                for passed, reason in zip(
                    contracts,
                    (
                        "FGW_LAST_LEVEL_INCOMPATIBLE",
                        "REFINEMENT_NOT_CONTRACTING",
                        "QMIN_LAST_LEVEL_INCOMPATIBLE",
                        "H98_LAST_LEVEL_INCOMPATIBLE",
                        "DENSITY_LAST_LEVEL_INCOMPATIBLE",
                        "CURRENT_LAST_LEVEL_INCOMPATIBLE",
                        "RECEIVER_DECISION_CHANGED",
                        "DENSITY_CURRENT_IDENTITY_FAILED",
                    ),
                    strict=True,
                )
                if not passed
            )
        ),
        qmin_difference_max=metrics.qmin_difference_max,
        h98_difference_max=metrics.h98_difference_max,
        density_relative_difference_max=metrics.density_relative_difference_max,
        current_relative_difference_max=metrics.current_relative_difference_max,
        first_decision_changing_clock_s=metrics.first_decision_changing_clock_s,
        fine_decision_margin_min=metrics.fine_decision_margin_min,
        line_volume_disagreement_max=metrics.line_volume_disagreement_max,
        identity_residual_max=metrics.identity_residual_max,
    )


def _pair_evaluation_summary(
    config: NumericalDenominatorConfig,
    cell_id: str,
    episodes: Mapping[str, NumericalDenominatorEpisode],
) -> NumericalDenominatorEvaluationCellSummary:
    if set(episodes) != {NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID, NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_20_VIEW_ID} or any(
        value.disposition is not EpisodeDisposition.COMPLETE for value in episodes.values()
    ):
        return NumericalDenominatorEvaluationCellSummary(
            summary_id=f"summary.numerical-denominator.evaluation.{cell_id.removeprefix('cell.')}",
            cell_id=cell_id,
            stable=False,
            fine_receiver_pass=None,
            first_difference_max=None,
            last_difference_max=None,
            contraction_ratio=None,
            observation_difference_max=None,
            interaction_residual_max=None,
            reason_codes=("NUMERICAL_DENOMINATOR_EVALUATION_CELL_INCOMPLETE",),
        )
    baseline = episodes[NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID]
    joint = episodes[NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_20_VIEW_ID]
    metrics = _compatibility_metrics(baseline, joint)
    difference = metrics.fgw_difference_max
    observation = max(
        abs(value.source_volume_fgw - value.common_volume_fgw)
        for episode in episodes.values()
        for value in _gauges(episode).values()
    )
    contracts = (
        difference <= config.tolerances.stable_fgw_absolute,
        metrics.qmin_difference_max <= config.tolerances.stable_qmin_normalized,
        metrics.h98_difference_max <= config.tolerances.stable_h98_absolute,
        metrics.density_relative_difference_max
        <= config.tolerances.stable_density_relative,
        metrics.current_relative_difference_max
        <= config.tolerances.stable_current_relative,
        metrics.decisions_match,
        observation <= config.tolerances.observation_fgw_absolute,
        metrics.identity_residual_max
        <= config.tolerances.identity_residual_absolute,
    )
    return NumericalDenominatorEvaluationCellSummary(
        summary_id=f"summary.numerical-denominator.evaluation.{cell_id.removeprefix('cell.')}",
        cell_id=cell_id,
        stable=all(contracts),
        fine_receiver_pass=_complete_receiver_pass(joint, common=True),
        first_difference_max=difference,
        last_difference_max=difference,
        contraction_ratio=None,
        observation_difference_max=observation,
        interaction_residual_max=None,
        reason_codes=tuple(
            sorted(
                reason
                for passed, reason in zip(
                    contracts,
                    (
                        "FGW_PAIR_INCOMPATIBLE",
                        "QMIN_PAIR_INCOMPATIBLE",
                        "H98_PAIR_INCOMPATIBLE",
                        "DENSITY_PAIR_INCOMPATIBLE",
                        "CURRENT_PAIR_INCOMPATIBLE",
                        "RECEIVER_DECISION_CHANGED",
                        "OBSERVATION_GAUGE_INCOMPATIBLE",
                        "DENSITY_CURRENT_IDENTITY_FAILED",
                    ),
                    strict=True,
                )
                if not passed
            )
        ),
        qmin_difference_max=metrics.qmin_difference_max,
        h98_difference_max=metrics.h98_difference_max,
        density_relative_difference_max=metrics.density_relative_difference_max,
        current_relative_difference_max=metrics.current_relative_difference_max,
        first_decision_changing_clock_s=metrics.first_decision_changing_clock_s,
        fine_decision_margin_min=metrics.fine_decision_margin_min,
        line_volume_disagreement_max=metrics.line_volume_disagreement_max,
        identity_residual_max=metrics.identity_residual_max,
    )


def _interaction_evaluation_summary(
    config: NumericalDenominatorConfig,
    cell_id: str,
    episodes: Mapping[str, NumericalDenominatorEpisode],
) -> NumericalDenominatorEvaluationCellSummary:
    expected = set(_branch_view_ids(NumericalDenominatorStage.EVALUATION, config.branch))
    if set(episodes) != expected or any(
        value.disposition is not EpisodeDisposition.COMPLETE for value in episodes.values()
    ):
        return NumericalDenominatorEvaluationCellSummary(
            summary_id=f"summary.numerical-denominator.evaluation.{cell_id.removeprefix('cell.')}",
            cell_id=cell_id,
            stable=False,
            fine_receiver_pass=None,
            first_difference_max=None,
            last_difference_max=None,
            contraction_ratio=None,
            observation_difference_max=None,
            interaction_residual_max=None,
            reason_codes=("NUMERICAL_DENOMINATOR_EVALUATION_CELL_INCOMPLETE",),
        )
    residual = _interaction_residual(episodes)
    baseline = episodes[NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID]
    effects = {
        "axis.corrector": _max_phase_path_difference(
            baseline,
            episodes[NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_20_VIEW_ID],
        ),
        "axis.grid": _max_phase_path_difference(
            baseline,
            episodes[NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_10_VIEW_ID],
        ),
        "axis.timestep": _max_phase_path_difference(
            baseline,
            episodes[NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID],
        ),
    }
    axis_views = {
        "axis.corrector": NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_20_VIEW_ID,
        "axis.grid": NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_10_VIEW_ID,
        "axis.timestep": NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID,
    }
    material_axes = tuple(
        sorted(
            axis
            for axis, effect in effects.items()
            if effect > config.tolerances.material_fgw_absolute
            or _complete_receiver_pass(baseline, common=True)
            != _complete_receiver_pass(episodes[axis_views[axis]], common=True)
        )
    )
    obstruction = bool(material_axes) or (
        residual > config.tolerances.interaction_fgw_absolute
    )
    return NumericalDenominatorEvaluationCellSummary(
        summary_id=f"summary.numerical-denominator.evaluation.{cell_id.removeprefix('cell.')}",
        cell_id=cell_id,
        stable=False,
        fine_receiver_pass=None,
        first_difference_max=None,
        last_difference_max=None,
        contraction_ratio=None,
        observation_difference_max=max(
            abs(value.source_volume_fgw - value.common_volume_fgw)
            for episode in episodes.values()
            for value in _gauges(episode).values()
        ),
        interaction_residual_max=residual,
        reason_codes=(
            ("NUMERICAL_INTERACTION_MATERIAL",)
            if obstruction
            else ("INTERACTION_DEVELOPMENT_NOT_RECURRENT",)
        ),
        material_axis_ids=material_axes,
    )


def adjudicate_evaluation(
    config: NumericalDenominatorConfig,
    batch: NumericalDenominatorBatch,
) -> NumericalDenominatorEvaluationAdjudication:
    if config.stage is not NumericalDenominatorStage.EVALUATION:
        raise ValueError("numerical denominator evaluation adjudicator requires evaluation config")
    reasons: set[str] = set()
    active = set(batch.active_cell_ids)
    by_cell: dict[str, dict[str, NumericalDenominatorEpisode]] = {value: {} for value in active}
    for episode in batch.episodes:
        if episode.cell.object_id in active:
            by_cell[episode.cell.object_id][episode.view.object_id] = episode
    dispositions = {
        episode.disposition
        for episode in batch.episodes
        if episode.cell.object_id in active
    }
    if not batch.complete_active_roster or dispositions & _TECHNICAL_RESERVE_DISPOSITIONS:
        result = NumericalDenominatorPrimaryResult.TECHNICAL_OBSERVATION_FAILURE
        reasons.add("NUMERICAL_DENOMINATOR_EVALUATION_TECHNICAL_ROSTER_FAILURE")
        summaries: tuple[NumericalDenominatorEvaluationCellSummary, ...] = ()
    elif EpisodeDisposition.UNEVALUABLE_OPERAND in dispositions:
        result = NumericalDenominatorPrimaryResult.UNEVALUABLE_SOURCE_OPERAND
        reasons.add("NUMERICAL_DENOMINATOR_EVALUATION_SOURCE_OPERAND_ABSENT")
        summaries = ()
    elif dispositions - {EpisodeDisposition.COMPLETE}:
        result = NumericalDenominatorPrimaryResult.SCIENTIFIC_PARTIAL_OR_TERMINATED
        reasons.add("NUMERICAL_DENOMINATOR_EVALUATION_SCIENTIFIC_PARTIAL")
        summaries = ()
    else:
        if config.branch in {
            NumericalDenominatorBranch.TIMESTEP,
            NumericalDenominatorBranch.GRID,
            NumericalDenominatorBranch.CORRECTOR,
        }:
            summaries = tuple(
                _axis_evaluation_summary(config, cell_id, by_cell[cell_id])
                for cell_id in sorted(active)
            )
        elif config.branch is NumericalDenominatorBranch.NUMERICAL_INTERACTION:
            summaries = tuple(
                _interaction_evaluation_summary(config, cell_id, by_cell[cell_id])
                for cell_id in sorted(active)
            )
        elif config.branch is NumericalDenominatorBranch.SCIENTIFIC_PARTIAL:
            summaries = tuple(
                NumericalDenominatorEvaluationCellSummary(
                    summary_id=(
                        "summary.numerical-denominator.evaluation."
                        f"{cell_id.removeprefix('cell.')}"
                    ),
                    cell_id=cell_id,
                    stable=False,
                    fine_receiver_pass=None,
                    first_difference_max=None,
                    last_difference_max=None,
                    contraction_ratio=None,
                    observation_difference_max=None,
                    interaction_residual_max=None,
                    reason_codes=("PARTIAL_DEVELOPMENT_NOT_RECURRENT",),
                )
                for cell_id in sorted(active)
            )
        else:
            summaries = tuple(
                _pair_evaluation_summary(config, cell_id, by_cell[cell_id])
                for cell_id in sorted(active)
            )
        if config.branch is NumericalDenominatorBranch.OBSERVATION_GAUGE and any(
            (value.observation_difference_max or Decimal(0))
            > config.tolerances.observation_fgw_absolute
            or "OBSERVATION_GAUGE_INCOMPATIBLE" in value.reason_codes
            for value in summaries
        ):
            result = NumericalDenominatorPrimaryResult.OBSERVATION_GAUGE_SENSITIVE
        elif config.branch is NumericalDenominatorBranch.NUMERICAL_INTERACTION:
            result = (
                NumericalDenominatorPrimaryResult.NUMERICAL_INTERACTION
                if any(
                    value.material_axis_ids
                    or (
                        value.interaction_residual_max is not None
                        and value.interaction_residual_max
                        > config.tolerances.interaction_fgw_absolute
                    )
                    for value in summaries
                )
                else NumericalDenominatorPrimaryResult.NONCONVERGENT_OR_UNRESOLVED
            )
        elif all(value.stable for value in summaries):
            pass_values = {value.fine_receiver_pass for value in summaries}
            if pass_values == {True}:
                result = NumericalDenominatorPrimaryResult.QUALIFIED_STABLE_PASS
            elif pass_values == {False}:
                result = NumericalDenominatorPrimaryResult.QUALIFIED_STABLE_FAIL
            else:
                result = NumericalDenominatorPrimaryResult.NONCONVERGENT_OR_UNRESOLVED
        elif config.branch is NumericalDenominatorBranch.TIMESTEP:
            result = NumericalDenominatorPrimaryResult.TIMESTEP_SENSITIVE
        elif config.branch is NumericalDenominatorBranch.GRID:
            result = NumericalDenominatorPrimaryResult.GRID_SENSITIVE
        elif config.branch is NumericalDenominatorBranch.CORRECTOR:
            result = NumericalDenominatorPrimaryResult.CORRECTOR_SENSITIVE
        elif config.branch is NumericalDenominatorBranch.SCIENTIFIC_PARTIAL:
            result = NumericalDenominatorPrimaryResult.NONCONVERGENT_OR_UNRESOLVED
        else:
            result = NumericalDenominatorPrimaryResult.NONCONVERGENT_OR_UNRESOLVED
    selected_view = None
    handoff = None
    if result is NumericalDenominatorPrimaryResult.QUALIFIED_STABLE_PASS:
        selected_view = {
            NumericalDenominatorBranch.TIMESTEP: NUMERICAL_DENOMINATOR_TIMESTEP_QUARTER_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID,
            NumericalDenominatorBranch.GRID: NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_41_CORRECTOR_STEPS_10_VIEW_ID,
            NumericalDenominatorBranch.CORRECTOR: NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_40_VIEW_ID,
            NumericalDenominatorBranch.STABLE_DENOMINATOR: NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_20_VIEW_ID,
        }.get(config.branch)
        if selected_view is None:
            result = NumericalDenominatorPrimaryResult.NONCONVERGENT_OR_UNRESOLVED
            reasons.add("NUMERICAL_DENOMINATOR_BRANCH_CANNOT_QUALIFY_DENOMINATOR")
        else:
            handoff = NUMERICAL_DENOMINATOR_HANDOFF_ID
    return NumericalDenominatorEvaluationAdjudication(
        adjudication_id=f"adjudication.{config.config_id.removeprefix('config.')}",
        config=ObjectIdentity.from_record(config.config_id, config),
        primary_result=result,
        branch=config.branch,
        summaries=summaries,
        selected_denominator_view_id=selected_view,
        handoff_id=handoff,
        reason_codes=tuple(sorted(reasons)),
        maximum_evidence_ceiling=EvidenceCeiling.MEASUREMENT,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
    )


__all__ = [
    'NumericalDenominatorBatch',
    'NumericalDenominatorBranch',
    'NumericalDenominatorApproval',
    'NumericalDenominatorCell',
    'NumericalDenominatorConfig',
    'NumericalDenominatorDevelopmentDecision',
    'NumericalDenominatorEpisode',
    'NumericalDenominatorEvaluationAdjudication',
    'NumericalDenominatorExecutionReceipt',
    'NumericalDenominatorIssue',
    'NumericalDenominatorImplementationManifest',
    'NumericalDenominatorPrimaryResult',
    'NumericalDenominatorQualificationEpisode',
    'NumericalDenominatorQualificationReceipt',
    'NumericalDenominatorResourceEnvelope',
    'NumericalDenominatorSourceFile',
    'NumericalDenominatorStage',
    'NumericalDenominatorTolerances',
    'NumericalDenominatorView',
    "acquire_episode",
    "acquire_stage",
    "adjudicate_development",
    "adjudicate_evaluation",
    "all_views",
    "build_config",
    'build_numerical_denominator_formal_coverage',
    "compile_resource_envelope",
    "development_cells",
    "evaluation_cells",
    "qualification_cells",
    "validate_predecessor_disjointness",
    "views_for",
]
