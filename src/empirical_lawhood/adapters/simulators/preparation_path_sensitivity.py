"Fresh density/current/Greenwald preparation-path sensitivity experiment.\n\nThe act is selected only by the frozen preparation branch for local Greenwald-gap failure.  It\nretains the preparation word, paired views, phase clocks and 0.03\nGreenwald-gap threshold, but uses fresh physical preparation cells.  Its\nmaximum evidence ceiling is order relation; its claim concerns numerical-view order/path attribution.\n"

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
import math
from typing import Mapping, ClassVar, Sequence
from types import MappingProxyType

from empirical_lawhood.kernel.evidence import (
    EvidenceCeiling,
    OutcomeAccess,
    VisibilityCeiling,
)
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.time import parse_utc_timestamp

from .open_campaigns import OpenSimulatorSourceManifest
from .prepared_base import (
    EpisodeDisposition,
    PreparationWord,
    PreparedBaseCellSubstitution,
    PreparedBaseEpisode,
    PreparedBaseView,
    PREPARED_SOURCE_EARLY_FULL_WORD_ID,
    PREPARED_SOURCE_PRIMARY_VIEW_ID,
    PREPARED_SOURCE_REFINED_VIEW_ID,
    acquire_episode,
    paired_views,
    preparation_words,
)


PREPARATION_PATH_SENSITIVITY_CONFIG_ID = 'config.preparation-path-sensitivity.gym-density-current-greenwald'
PREPARATION_PATH_SENSITIVITY_QUALIFICATION_CONFIG_ID = 'config.preparation-path-sensitivity.gym-density-current-greenwald-qualification'
PREPARATION_PATH_SENSITIVITY_NOMINATION_ID = 'nomination.preparation-path-sensitivity.gym-density-current-greenwald'
PREPARATION_PATH_SENSITIVITY_RUN_ID = 'run.preparation-path-sensitivity.gym-density-current-greenwald'
PREPARATION_PATH_SENSITIVITY_GAP_THRESHOLD = Decimal("0.03")
PREPARATION_PATH_SENSITIVITY_DOMINANCE_FRACTION = Decimal("0.80")
PREPARATION_PATH_SENSITIVITY_RESIDUAL_TOLERANCE = Decimal("0.0000000001")
PREPARATION_PATH_SENSITIVITY_PHASE_CLOCKS = tuple(range(105, 115))
_TECHNICAL = frozenset(
    {
        EpisodeDisposition.TECHNICAL_OBSERVATION_FAILURE,
        EpisodeDisposition.DELIVERY_INVALID,
    }
)


class PreparationPathSensitivityStage(StrEnum):
    QUALIFICATION = "QUALIFICATION"
    EVALUATION = "EVALUATION"


class PreparationPathSensitivityCellAttribution(StrEnum):
    DENSITY_PATH_DOMINANT = "DENSITY_PATH_DOMINANT"
    CURRENT_PATH_DOMINANT = "CURRENT_PATH_DOMINANT"
    MIXED_DENSITY_CURRENT = "MIXED_DENSITY_CURRENT"
    OBSERVATION_RECONSTRUCTION_MISMATCH = "OBSERVATION_RECONSTRUCTION_MISMATCH"
    FGW_GAP_NOT_RECURRENT = "FGW_GAP_NOT_RECURRENT"


class PreparationPathSensitivityPrimaryResult(StrEnum):
    DENSITY_PATH_RECURRENT = "DENSITY_PATH_RECURRENT"
    CURRENT_PATH_RECURRENT = "CURRENT_PATH_RECURRENT"
    MIXED_DENSITY_CURRENT_PATH = "MIXED_DENSITY_CURRENT_PATH"
    OBSERVATION_RECONSTRUCTION_MISMATCH = "OBSERVATION_RECONSTRUCTION_MISMATCH"
    FGW_GAP_NOT_RECURRENT = "FGW_GAP_NOT_RECURRENT"
    PARTIAL_OR_TERMINATED = "PARTIAL_OR_TERMINATED"
    TECHNICAL_OBSERVATION_FAILURE = "TECHNICAL_OBSERVATION_FAILURE"
    UNEVALUABLE_OPERAND = "UNEVALUABLE_OPERAND"


@dataclass(frozen=True, slots=True)
class PreparationPathSensitivityNomination(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/preparation-path-sensitivity-nomination'

    nomination_id: str
    parent_prepared_response_adjudication: ObjectIdentity
    selected_branch_id: str
    diagnostic_id: str
    first_failed_predicate_id: str
    question: str
    fresh_evidence_required: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        for name in (
            "nomination_id",
            "selected_branch_id",
            "diagnostic_id",
            "first_failed_predicate_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if (
            self.nomination_id != PREPARATION_PATH_SENSITIVITY_NOMINATION_ID
            or self.selected_branch_id != "branch.prepared-response.numerical-preparation"
            or self.diagnostic_id != 'diagnostic.prepared-base-greenwald-fraction-local'
            or self.first_failed_predicate_id != "predicate.phase-fgw-volume"
            or not self.fresh_evidence_required
            or self.outcome_access is not OutcomeAccess.EVALUATION_REVEALED
            or self.visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE
        ):
            raise ValueError("Preparation-path sensitivity nomination differs from the frozen preparation branch")


@dataclass(frozen=True, slots=True)
class PreparationPathSensitivityCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/preparation-path-sensitivity-cell'

    cell_id: str
    stage: PreparationPathSensitivityStage
    reserve: bool
    environment_seed: int
    initial_temperature_scale: Decimal
    initial_density_nbar: Decimal
    bootstrap_multiplier: Decimal
    inner_transport_scale: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.cell_id, field_name="cell_id")
        if self.environment_seed < 0:
            raise ValueError("Preparation-path sensitivity environment seed must be nonnegative")
        for name, lower, upper in (
            ("initial_temperature_scale", Decimal("0.990"), Decimal("1.010")),
            ("initial_density_nbar", Decimal("0.848"), Decimal("0.852")),
            ("bootstrap_multiplier", Decimal("0.995"), Decimal("1.005")),
            ("inner_transport_scale", Decimal("0.990"), Decimal("1.010")),
        ):
            value = getattr(self, name)
            validate_decimal(value, field_name=name, minimum=lower)
            if value > upper:
                raise ValueError(f"{name} exceeds the inherited local box")

    @property
    def coordinate_key(self) -> tuple[Decimal, Decimal, Decimal, Decimal]:
        return (
            self.initial_temperature_scale,
            self.initial_density_nbar,
            self.bootstrap_multiplier,
            self.inner_transport_scale,
        )


@dataclass(frozen=True, slots=True)
class PreparationPathSensitivityConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/preparation-path-sensitivity-config'

    config_id: str
    stage: PreparationPathSensitivityStage
    nomination: ObjectIdentity
    parent_prepared_response_adjudication: ObjectIdentity
    source_manifest: ObjectIdentity
    cells: tuple[PreparationPathSensitivityCell, ...]
    primary_cell_ids: tuple[str, ...]
    reserve_cell_ids: tuple[str, ...]
    words: tuple[PreparationWord, ...]
    views: tuple[PreparedBaseView, ...]
    phase_clocks_s: tuple[int, ...]
    fgw_gap_threshold: Decimal
    dominance_fraction: Decimal
    reconstruction_residual_tolerance: Decimal
    outcome_access: OutcomeAccess
    maximum_evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        require_sorted_unique_ids(self.cells, attribute="cell_id", field_name="cells")
        require_sorted_unique_strings(
            self.primary_cell_ids,
            field_name="primary_cell_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.reserve_cell_ids,
            field_name="reserve_cell_ids",
            allow_empty=self.stage is PreparationPathSensitivityStage.QUALIFICATION,
        )
        cell_ids = {value.cell_id for value in self.cells}
        if (
            set(self.primary_cell_ids) & set(self.reserve_cell_ids)
            or set(self.primary_cell_ids) | set(self.reserve_cell_ids) != cell_ids
            or any(
                value.stage is not self.stage
                or value.reserve != (value.cell_id in set(self.reserve_cell_ids))
                for value in self.cells
            )
        ):
            raise ValueError("Preparation-path sensitivity config has an invalid cell roster")
        if (
            tuple(value.word_id for value in self.words) != (PREPARED_SOURCE_EARLY_FULL_WORD_ID,)
            or tuple(value.view_id for value in self.views)
            != (PREPARED_SOURCE_PRIMARY_VIEW_ID, PREPARED_SOURCE_REFINED_VIEW_ID)
            or self.phase_clocks_s != PREPARATION_PATH_SENSITIVITY_PHASE_CLOCKS
            or self.fgw_gap_threshold != PREPARATION_PATH_SENSITIVITY_GAP_THRESHOLD
            or self.dominance_fraction != PREPARATION_PATH_SENSITIVITY_DOMINANCE_FRACTION
            or self.reconstruction_residual_tolerance != PREPARATION_PATH_SENSITIVITY_RESIDUAL_TOLERANCE
        ):
            raise ValueError("Preparation-path sensitivity changed its inherited word/view/path rule")
        if self.stage is PreparationPathSensitivityStage.QUALIFICATION:
            expected = (
                self.config_id == PREPARATION_PATH_SENSITIVITY_QUALIFICATION_CONFIG_ID
                and len(self.primary_cell_ids) == 1
                and not self.reserve_cell_ids
                and self.outcome_access is OutcomeAccess.DEVELOPMENT_VISIBLE
                and self.maximum_evidence_ceiling is EvidenceCeiling.NON_PROMOTABLE
            )
        else:
            expected = (
                self.config_id == PREPARATION_PATH_SENSITIVITY_CONFIG_ID
                and len(self.primary_cell_ids) == 6
                and len(self.reserve_cell_ids) == 1
                and self.outcome_access is OutcomeAccess.EVALUATION_SEALED
                and self.maximum_evidence_ceiling is EvidenceCeiling.ORDER_RELATION
            )
        if not expected:
            raise ValueError("Preparation-path sensitivity config changed its frozen stage contract")


@dataclass(frozen=True, slots=True)
class PreparationPathSensitivityBatch(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/preparation-path-sensitivity-batch'

    batch_id: str
    config: ObjectIdentity
    stage: PreparationPathSensitivityStage
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
            or self.complete_active_roster
            != (len(self.active_cell_ids) == self.expected_active_cell_count)
            or set(self.active_cell_ids) & set(self.excluded_cell_ids)
        ):
            raise ValueError("Preparation-path sensitivity batch roster is inconsistent")
        expected_access = (
            OutcomeAccess.DEVELOPMENT_VISIBLE
            if self.stage is PreparationPathSensitivityStage.QUALIFICATION
            else OutcomeAccess.EVALUATION_SEALED
        )
        if self.outcome_access is not expected_access:
            raise ValueError("Preparation-path sensitivity batch has the wrong stage access")


@dataclass(frozen=True, slots=True)
class PreparationPathSensitivityCellSummary(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/preparation-path-sensitivity-cell-summary'

    summary_id: str
    cell_id: str
    maximum_fgw_gap: Decimal
    mean_log_fgw_difference: Decimal
    mean_log_density_difference: Decimal
    mean_negative_log_current_difference: Decimal
    maximum_absolute_reconstruction_residual: Decimal
    attribution: PreparationPathSensitivityCellAttribution

    def __post_init__(self) -> None:
        validate_stable_id(self.summary_id, field_name="summary_id")
        validate_stable_id(self.cell_id, field_name="cell_id")
        for name in (
            "maximum_fgw_gap",
            "mean_log_fgw_difference",
            "mean_log_density_difference",
            "mean_negative_log_current_difference",
            "maximum_absolute_reconstruction_residual",
        ):
            validate_decimal(getattr(self, name), field_name=name)


@dataclass(frozen=True, slots=True)
class PreparationPathSensitivityAdjudication(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/preparation-path-sensitivity-adjudication'

    adjudication_id: str
    config: ObjectIdentity
    batch: ObjectIdentity
    primary_result: PreparationPathSensitivityPrimaryResult
    unit_count: int
    attribution_counts: tuple[tuple[str, int], ...]
    cell_summaries: tuple[PreparationPathSensitivityCellSummary, ...]
    reason_codes: tuple[str, ...]
    maximum_evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.adjudication_id, field_name="adjudication_id")
        require_sorted_unique_ids(
            self.cell_summaries,
            attribute="summary_id",
            field_name="cell_summaries",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if (
            self.unit_count != len(self.cell_summaries)
            or tuple(key for key, _ in self.attribution_counts)
            != tuple(sorted(key for key, _ in self.attribution_counts))
            or self.maximum_evidence_ceiling is not EvidenceCeiling.ORDER_RELATION
            or self.outcome_access is not OutcomeAccess.EVALUATION_REVEALED
            or self.visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE
        ):
            raise ValueError("Preparation-path sensitivity adjudication changes its claim or reveal contract")


@dataclass(frozen=True, slots=True)
class PreparationPathSensitivityQualificationReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/preparation-path-sensitivity-qualification-receipt'

    receipt_id: str
    config: ObjectIdentity
    implementation_sha256: str
    episode_sha256: tuple[str, ...]
    complete_route: bool
    reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess
    maximum_evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        validate_sha256(self.implementation_sha256, field_name="implementation_sha256")
        for value in self.episode_sha256:
            validate_sha256(value, field_name="episode_sha256")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if (
            len(self.episode_sha256) != 2
            or not self.complete_route
            or self.reason_codes
            or self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
            or self.maximum_evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
        ):
            raise ValueError("Preparation-path sensitivity qualification is not excluded route evidence")


@dataclass(frozen=True, slots=True)
class PreparationPathSensitivityApproval(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/preparation-path-sensitivity-approval'

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
            raise ValueError("Preparation-path sensitivity approval lacks independent outcome-blind approval")


@dataclass(frozen=True, slots=True)
class PreparationPathSensitivityIssue(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/preparation-path-sensitivity-issue'

    issue_id: str
    nomination: ObjectIdentity
    config: ObjectIdentity
    implementation_manifest: ObjectIdentity
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
        if len(set(roles)) != len(roles) or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("Preparation-path sensitivity issue lacks role separation or child blindness")


@dataclass(frozen=True, slots=True)
class PreparationPathSensitivityExecutionReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/preparation-path-sensitivity-execution-receipt'

    receipt_id: str
    run_id: str
    issue: ObjectIdentity
    execution_authority: ObjectIdentity
    batch: ObjectIdentity
    batch_relative_path: str
    active_cell_ids: tuple[str, ...]
    excluded_cell_ids: tuple[str, ...]
    completed_at_utc: str
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        validate_stable_id(self.run_id, field_name="run_id")
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
        if self.outcome_access is not OutcomeAccess.EVALUATION_SEALED:
            raise ValueError("Preparation-path sensitivity execution receipt revealed an outcome")


_FROZEN_ENVIRONMENT_SEEDS: Mapping[str, int] = MappingProxyType({
    'cell.preparation-path-sensitivity-density-current-01': 2577986942,
    'cell.preparation-path-sensitivity-density-current-02': 2016873144,
    'cell.preparation-path-sensitivity-density-current-03': 4287291252,
    'cell.preparation-path-sensitivity-density-current-04': 1450584195,
    'cell.preparation-path-sensitivity-density-current-05': 2475028105,
    'cell.preparation-path-sensitivity-density-current-06': 1437380105,
    'cell.preparation-path-sensitivity-density-current-reserve-01': 2390567192,
    'cell.preparation-path-sensitivity-density-current-qualification-01': 3546228484,
})


def _seed(value: str) -> int:
    return _FROZEN_ENVIRONMENT_SEEDS[value]


def _cell(
    cell_id: str,
    *,
    stage: PreparationPathSensitivityStage,
    reserve: bool,
    index: int,
) -> PreparationPathSensitivityCell:
    return PreparationPathSensitivityCell(
        cell_id=cell_id,
        stage=stage,
        reserve=reserve,
        environment_seed=_seed(cell_id),
        initial_temperature_scale=Decimal("1") + Decimal(13 * index + 1) * Decimal("1e-7"),
        initial_density_nbar=Decimal("0.85") + Decimal(17 * index + 2) * Decimal("1e-7"),
        bootstrap_multiplier=Decimal("1") + Decimal(19 * index + 3) * Decimal("1e-7"),
        inner_transport_scale=Decimal("1") + Decimal(23 * index + 4) * Decimal("1e-7"),
    )


def evaluation_cells() -> tuple[PreparationPathSensitivityCell, ...]:
    primary = tuple(
        _cell(
            f"cell.preparation-path-sensitivity-density-current-{index:02d}",
            stage=PreparationPathSensitivityStage.EVALUATION,
            reserve=False,
            index=index,
        )
        for index in range(1, 7)
    )
    reserve = _cell(
        'cell.preparation-path-sensitivity-density-current-reserve-01',
        stage=PreparationPathSensitivityStage.EVALUATION,
        reserve=True,
        index=7,
    )
    return (*primary, reserve)


def qualification_cell() -> PreparationPathSensitivityCell:
    return _cell(
        'cell.preparation-path-sensitivity-density-current-qualification-01',
        stage=PreparationPathSensitivityStage.QUALIFICATION,
        reserve=False,
        index=101,
    )


def build_config(
    *,
    nomination: PreparationPathSensitivityNomination,
    source_manifest: OpenSimulatorSourceManifest,
    qualification: bool,
) -> PreparationPathSensitivityConfig:
    stage = PreparationPathSensitivityStage.QUALIFICATION if qualification else PreparationPathSensitivityStage.EVALUATION
    cells = (qualification_cell(),) if qualification else evaluation_cells()
    word = next(value for value in preparation_words() if value.word_id == PREPARED_SOURCE_EARLY_FULL_WORD_ID)
    return PreparationPathSensitivityConfig(
        config_id=(PREPARATION_PATH_SENSITIVITY_QUALIFICATION_CONFIG_ID if qualification else PREPARATION_PATH_SENSITIVITY_CONFIG_ID),
        stage=stage,
        nomination=ObjectIdentity.from_record(nomination.nomination_id, nomination),
        parent_prepared_response_adjudication=nomination.parent_prepared_response_adjudication,
        source_manifest=ObjectIdentity.from_record(
            source_manifest.source_id,
            source_manifest,
        ),
        cells=tuple(sorted(cells, key=lambda value: value.cell_id)),
        primary_cell_ids=tuple(sorted(value.cell_id for value in cells if not value.reserve)),
        reserve_cell_ids=tuple(sorted(value.cell_id for value in cells if value.reserve)),
        words=(word,),
        views=paired_views(),
        phase_clocks_s=PREPARATION_PATH_SENSITIVITY_PHASE_CLOCKS,
        fgw_gap_threshold=PREPARATION_PATH_SENSITIVITY_GAP_THRESHOLD,
        dominance_fraction=PREPARATION_PATH_SENSITIVITY_DOMINANCE_FRACTION,
        reconstruction_residual_tolerance=PREPARATION_PATH_SENSITIVITY_RESIDUAL_TOLERANCE,
        outcome_access=(
            OutcomeAccess.DEVELOPMENT_VISIBLE if qualification else OutcomeAccess.EVALUATION_SEALED
        ),
        maximum_evidence_ceiling=(
            EvidenceCeiling.NON_PROMOTABLE if qualification else EvidenceCeiling.ORDER_RELATION
        ),
    )


def acquire_preparation_path_sensitivity(config: PreparationPathSensitivityConfig) -> PreparationPathSensitivityBatch:
    episodes: list[PreparedBaseEpisode] = []
    active: list[str] = []
    excluded: list[str] = []
    substitutions: list[PreparedBaseCellSubstitution] = []
    reserves = iter(config.reserve_cell_ids)
    for intended_id in config.primary_cell_ids:
        current_id = intended_id
        while True:
            local = []
            technical = False
            for view in config.views:
                episode = acquire_episode(
                    config=config,  # type: ignore[arg-type]
                    cell_id=current_id,
                    word_id=config.words[0].word_id,
                    view_id=view.view_id,
                )
                local.append(episode)
                episodes.append(episode)
                if episode.disposition in _TECHNICAL:
                    technical = True
                    break
            if not technical:
                active.append(current_id)
                break
            excluded.append(current_id)
            try:
                replacement = next(reserves)
            except StopIteration:
                break
            substitutions.append(
                PreparedBaseCellSubstitution(
                    substitution_id=(
                        f"substitution.preparation-path-sensitivity.{intended_id.removeprefix('cell.')}."
                        f"{replacement.removeprefix('cell.')}"
                    ),
                    intended_cell_id=intended_id,
                    replacement_cell_id=replacement,
                    excluded_episode_ids=tuple(sorted(value.episode_id for value in local)),
                    reason_codes=tuple(
                        sorted({reason for value in local for reason in value.reason_codes})
                    )
                    or ("TECHNICAL_BUNDLE_FAILURE",),
                    receiver_outcome_released_to_evaluator=False,
                )
            )
            current_id = replacement
    return PreparationPathSensitivityBatch(
        batch_id=(
            'batch.preparation-path-sensitivity-density-current-qualification'
            if config.stage is PreparationPathSensitivityStage.QUALIFICATION
            else 'batch.preparation-path-sensitivity-density-current-evaluation'
        ),
        config=ObjectIdentity.from_record(config.config_id, config),
        stage=config.stage,
        episodes=tuple(sorted(episodes, key=lambda value: value.episode_id)),
        active_cell_ids=tuple(sorted(active)),
        excluded_cell_ids=tuple(sorted(excluded)),
        substitutions=tuple(sorted(substitutions, key=lambda value: value.substitution_id)),
        expected_active_cell_count=len(config.primary_cell_ids),
        complete_active_roster=len(active) == len(config.primary_cell_ids),
        outcome_access=config.outcome_access,
    )


def _receiver(episode: PreparedBaseEpisode, clock: int, value_id: str) -> Decimal:
    transition = episode.transitions[clock - 1]
    values = {value.value_id: value.value for value in transition.receiver_values}
    try:
        return values[value_id]
    except KeyError as error:
        raise ValueError(f'Preparation-path sensitivity receiver operand absent: {value_id}') from error


def _log_ratio(numerator: Decimal, denominator: Decimal) -> Decimal:
    if numerator <= 0 or denominator <= 0:
        raise ValueError("Preparation-path sensitivity log-ratio operand must be positive")
    return Decimal(str(math.log(float(numerator / denominator))))


def summarize_cell(
    config: PreparationPathSensitivityConfig,
    primary: PreparedBaseEpisode,
    refined: PreparedBaseEpisode,
) -> PreparationPathSensitivityCellSummary:
    total = []
    density = []
    current = []
    residuals = []
    for clock in config.phase_clocks_s:
        total_value = _log_ratio(
            _receiver(refined, clock, "receiver.fgw-volume-average"),
            _receiver(primary, clock, "receiver.fgw-volume-average"),
        )
        density_value = _log_ratio(
            _receiver(refined, clock, "receiver.volume-average-density"),
            _receiver(primary, clock, "receiver.volume-average-density"),
        )
        current_value = -_log_ratio(
            _receiver(refined, clock, "receiver.lcfs-current"),
            _receiver(primary, clock, "receiver.lcfs-current"),
        )
        total.append(total_value)
        density.append(density_value)
        current.append(current_value)
        residuals.append(total_value - density_value - current_value)
    divisor = Decimal(len(config.phase_clocks_s))
    total_mean = sum(total, Decimal(0)) / divisor
    density_mean = sum(density, Decimal(0)) / divisor
    current_mean = sum(current, Decimal(0)) / divisor
    residual = max(abs(value) for value in residuals)
    gap = max(
        _receiver(refined, clock, "receiver.fgw-volume-average") for clock in config.phase_clocks_s
    ) - max(
        _receiver(primary, clock, "receiver.fgw-volume-average") for clock in config.phase_clocks_s
    )
    if residual > config.reconstruction_residual_tolerance:
        attribution = PreparationPathSensitivityCellAttribution.OBSERVATION_RECONSTRUCTION_MISMATCH
    elif abs(gap) < config.fgw_gap_threshold:
        attribution = PreparationPathSensitivityCellAttribution.FGW_GAP_NOT_RECURRENT
    elif (
        total_mean * density_mean > 0
        and abs(density_mean) >= config.dominance_fraction * abs(total_mean)
        and abs(current_mean) <= (Decimal(1) - config.dominance_fraction) * abs(total_mean)
    ):
        attribution = PreparationPathSensitivityCellAttribution.DENSITY_PATH_DOMINANT
    elif (
        total_mean * current_mean > 0
        and abs(current_mean) >= config.dominance_fraction * abs(total_mean)
        and abs(density_mean) <= (Decimal(1) - config.dominance_fraction) * abs(total_mean)
    ):
        attribution = PreparationPathSensitivityCellAttribution.CURRENT_PATH_DOMINANT
    else:
        attribution = PreparationPathSensitivityCellAttribution.MIXED_DENSITY_CURRENT
    return PreparationPathSensitivityCellSummary(
        summary_id=f"summary.preparation-path-sensitivity.{primary.cell.object_id.removeprefix('cell.')}",
        cell_id=primary.cell.object_id,
        maximum_fgw_gap=gap,
        mean_log_fgw_difference=total_mean,
        mean_log_density_difference=density_mean,
        mean_negative_log_current_difference=current_mean,
        maximum_absolute_reconstruction_residual=residual,
        attribution=attribution,
    )


def adjudicate_preparation_path_sensitivity(config: PreparationPathSensitivityConfig, batch: PreparationPathSensitivityBatch) -> PreparationPathSensitivityAdjudication:
    reasons: set[str] = set()
    summaries: tuple[PreparationPathSensitivityCellSummary, ...] = ()
    if not batch.complete_active_roster:
        result = PreparationPathSensitivityPrimaryResult.TECHNICAL_OBSERVATION_FAILURE
        reasons.add("PREPARATION_PATH_SENSITIVITY_ACTIVE_ROSTER_INCOMPLETE")
    else:
        active = set(batch.active_cell_ids)
        episodes = tuple(value for value in batch.episodes if value.cell.object_id in active)
        dispositions = {value.disposition for value in episodes}
        if dispositions & _TECHNICAL:
            result = PreparationPathSensitivityPrimaryResult.TECHNICAL_OBSERVATION_FAILURE
        elif EpisodeDisposition.UNEVALUABLE_OPERAND in dispositions:
            result = PreparationPathSensitivityPrimaryResult.UNEVALUABLE_OPERAND
        elif dispositions != {EpisodeDisposition.COMPLETE}:
            result = PreparationPathSensitivityPrimaryResult.PARTIAL_OR_TERMINATED
        else:
            by_key = {(value.cell.object_id, value.view.object_id): value for value in episodes}
            expected = {
                (cell_id, view.view_id)
                for cell_id in batch.active_cell_ids
                for view in config.views
            }
            if set(by_key) != expected:
                result = PreparationPathSensitivityPrimaryResult.TECHNICAL_OBSERVATION_FAILURE
                reasons.add("PREPARATION_PATH_SENSITIVITY_EPISODE_ROSTER_INCOMPLETE")
            else:
                try:
                    summaries = tuple(
                        summarize_cell(
                            config,
                            by_key[(cell_id, PREPARED_SOURCE_PRIMARY_VIEW_ID)],
                            by_key[(cell_id, PREPARED_SOURCE_REFINED_VIEW_ID)],
                        )
                        for cell_id in batch.active_cell_ids
                    )
                except ValueError:
                    result = PreparationPathSensitivityPrimaryResult.UNEVALUABLE_OPERAND
                    reasons.add("PREPARATION_PATH_SENSITIVITY_DENSITY_CURRENT_OPERAND_ABSENT")
                else:
                    attributions = {value.attribution for value in summaries}
                    if attributions == {PreparationPathSensitivityCellAttribution.DENSITY_PATH_DOMINANT}:
                        result = PreparationPathSensitivityPrimaryResult.DENSITY_PATH_RECURRENT
                    elif attributions == {PreparationPathSensitivityCellAttribution.CURRENT_PATH_DOMINANT}:
                        result = PreparationPathSensitivityPrimaryResult.CURRENT_PATH_RECURRENT
                    elif PreparationPathSensitivityCellAttribution.OBSERVATION_RECONSTRUCTION_MISMATCH in attributions:
                        result = PreparationPathSensitivityPrimaryResult.OBSERVATION_RECONSTRUCTION_MISMATCH
                    elif attributions == {PreparationPathSensitivityCellAttribution.FGW_GAP_NOT_RECURRENT}:
                        result = PreparationPathSensitivityPrimaryResult.FGW_GAP_NOT_RECURRENT
                    else:
                        result = PreparationPathSensitivityPrimaryResult.MIXED_DENSITY_CURRENT_PATH
    counts = Counter(value.attribution.value for value in summaries)
    return PreparationPathSensitivityAdjudication(
        adjudication_id='adjudication.preparation-path-sensitivity.gym-density-current-greenwald',
        config=ObjectIdentity.from_record(config.config_id, config),
        batch=ObjectIdentity.from_record(batch.batch_id, batch),
        primary_result=result,
        unit_count=len(summaries),
        attribution_counts=tuple(sorted(counts.items())),
        cell_summaries=tuple(sorted(summaries, key=lambda value: value.summary_id)),
        reason_codes=tuple(sorted(reasons)),
        maximum_evidence_ceiling=EvidenceCeiling.ORDER_RELATION,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
    )


def validate_no_base_preparation_overlap(
    cells: Sequence[PreparationPathSensitivityCell],
    *,
    base_preparation_cell_ids: Sequence[str],
    base_preparation_coordinate_keys: Sequence[tuple[Decimal, Decimal, Decimal, Decimal]],
) -> None:
    if {value.cell_id for value in cells}.intersection(base_preparation_cell_ids) or {
        value.coordinate_key for value in cells
    }.intersection(base_preparation_coordinate_keys):
        raise ValueError("Preparation-path sensitivity cells overlap preparation identities or coordinates")


__all__ = [
    'PreparationPathSensitivityAdjudication',
    'PreparationPathSensitivityApproval',
    'PreparationPathSensitivityBatch',
    'PreparationPathSensitivityCell',
    'PreparationPathSensitivityCellAttribution',
    'PreparationPathSensitivityCellSummary',
    'PreparationPathSensitivityConfig',
    'PreparationPathSensitivityExecutionReceipt',
    'PreparationPathSensitivityIssue',
    'PreparationPathSensitivityNomination',
    'PreparationPathSensitivityPrimaryResult',
    'PreparationPathSensitivityQualificationReceipt',
    "PREPARATION_PATH_SENSITIVITY_CONFIG_ID",
    "PREPARATION_PATH_SENSITIVITY_NOMINATION_ID",
    "PREPARATION_PATH_SENSITIVITY_RUN_ID",
    "acquire_preparation_path_sensitivity",
    "adjudicate_preparation_path_sensitivity",
    "build_config",
    "evaluation_cells",
    "qualification_cell",
    "summarize_cell",
    "validate_no_base_preparation_overlap",
]
