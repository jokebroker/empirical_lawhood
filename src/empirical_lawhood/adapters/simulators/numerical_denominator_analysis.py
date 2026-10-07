"Descriptive, non-promoting analysis of completed numerical-denominator evidence."

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar, Mapping

from empirical_lawhood.kernel.evidence import (
    EvidenceCeiling,
    OutcomeAccess,
    VisibilityCeiling,
)
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_stable_id,
)

from .numerical_denominator import NumericalDenominatorBatch, NumericalDenominatorDevelopmentDecision, NumericalDenominatorEpisode, NumericalDenominatorEvaluationAdjudication, NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID, NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_20_VIEW_ID, NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_10_VIEW_ID, NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_20_VIEW_ID, NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID, NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_20_VIEW_ID, NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_10_VIEW_ID, NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_20_VIEW_ID, _complete_receiver_pass, _decision_margin_min, _gauges, _receivers


_PHASE_CLOCKS = tuple(range(105, 115))


@dataclass(frozen=True, slots=True)
class NumericalDenominatorPathAnalysis(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/numerical-denominator-path-analysis'

    analysis_id: str
    cohort: str
    cell_id: str
    view_id: str
    complete_common_receiver_pass: bool
    complete_source_receiver_pass: bool
    phase_common_volume_fgw_min: Decimal
    phase_common_volume_fgw_max: Decimal
    phase_source_volume_fgw_max: Decimal
    observation_difference_max: Decimal
    line_volume_difference_max: Decimal
    phase_qmin_min: Decimal
    phase_h98_min: Decimal
    terminal_h98: Decimal
    decision_margin_min: Decimal
    phase_density_min_m3: Decimal
    phase_density_max_m3: Decimal
    phase_lcfs_current_min_a: Decimal
    phase_lcfs_current_max_a: Decimal
    identity_residual_max: Decimal
    available_field_ids: tuple[str, ...]
    unavailable_field_ids: tuple[str, ...]
    runtime_seconds: Decimal

    def __post_init__(self) -> None:
        for name, value in (
            ("analysis_id", self.analysis_id),
            ("cell_id", self.cell_id),
            ("view_id", self.view_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.cohort not in {"development", "evaluation"}:
            raise ValueError("Numerical-denominator path analysis has an invalid cohort")
        require_sorted_unique_strings(
            self.available_field_ids,
            field_name="available_field_ids",
        )
        require_sorted_unique_strings(
            self.unavailable_field_ids,
            field_name="unavailable_field_ids",
        )
        if set(self.available_field_ids) & set(self.unavailable_field_ids):
            raise ValueError("Numerical-denominator path field dispositions overlap")
        if self.runtime_seconds < 0:
            raise ValueError("Numerical-denominator path runtime cannot be negative")


@dataclass(frozen=True, slots=True)
class NumericalDenominatorCellInteractionAnalysis(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/numerical-denominator-cell-interaction-analysis'

    analysis_id: str
    cell_id: str
    timestep_effect_max: Decimal
    grid_effect_max: Decimal
    corrector_effect_max: Decimal
    joint_minus_additive_max: Decimal
    timestep_grid_interaction_max: Decimal
    timestep_corrector_interaction_max: Decimal
    grid_corrector_interaction_max: Decimal
    three_way_interaction_max: Decimal
    timestep_context_effect_min: Decimal
    timestep_context_effect_max: Decimal
    grid_context_effect_min: Decimal
    grid_context_effect_max: Decimal
    corrector_context_effect_min: Decimal
    corrector_context_effect_max: Decimal
    baseline_to_joint_mean_signed: Decimal
    passing_view_ids: tuple[str, ...]
    failing_view_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.analysis_id, field_name="analysis_id")
        validate_stable_id(self.cell_id, field_name="cell_id")
        require_sorted_unique_strings(
            self.passing_view_ids,
            field_name="passing_view_ids",
        )
        require_sorted_unique_strings(
            self.failing_view_ids,
            field_name="failing_view_ids",
        )
        if set(self.passing_view_ids) & set(self.failing_view_ids):
            raise ValueError("Numerical-denominator view decision dispositions overlap")


@dataclass(frozen=True, slots=True)
class NumericalDenominatorDetailedAnalysis(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/numerical-denominator-detailed-analysis'

    analysis_id: str
    development_decision: ObjectIdentity
    development_batch: ObjectIdentity
    evaluation_adjudication: ObjectIdentity
    evaluation_batch: ObjectIdentity
    development_paths: tuple[NumericalDenominatorPathAnalysis, ...]
    evaluation_paths: tuple[NumericalDenominatorPathAnalysis, ...]
    evaluation_cells: tuple[NumericalDenominatorCellInteractionAnalysis, ...]
    development_episode_count: int
    evaluation_episode_count: int
    evaluation_independent_unit_count: int
    technical_reserve_count: int
    incomplete_episode_count: int
    common_receiver_pass_path_count: int
    source_receiver_pass_path_count: int
    maximum_observation_difference: Decimal
    minimum_interaction_residual: Decimal
    maximum_interaction_residual: Decimal
    total_simulator_runtime_seconds: Decimal
    maximum_evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.analysis_id, field_name="analysis_id")
        if (
            self.development_episode_count != len(self.development_paths)
            or self.evaluation_episode_count != len(self.evaluation_paths)
            or self.evaluation_independent_unit_count != len(self.evaluation_cells)
            or self.technical_reserve_count < 0
            or self.incomplete_episode_count < 0
            or self.maximum_evidence_ceiling is not EvidenceCeiling.MEASUREMENT
            or self.outcome_access is not OutcomeAccess.EVALUATION_REVEALED
            or self.visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE
        ):
            raise ValueError("Numerical-denominator detailed analysis changed its evidence contract")


def _path(episode: NumericalDenominatorEpisode, cohort: str) -> NumericalDenominatorPathAnalysis:
    gauges = _gauges(episode)
    receivers = _receivers(episode)
    fields = {
        field.field_id: field.source_available
        for transition in episode.transitions
        for field in transition.diagnostic_fields
    }
    phase_gauges = tuple(gauges[clock] for clock in _PHASE_CLOCKS)
    return NumericalDenominatorPathAnalysis(
        analysis_id=(
            f"analysis.numerical-denominator.{cohort}."
            f"{episode.cell.object_id.removeprefix('cell.')}."
            f"{episode.view.object_id.removeprefix('view.')}"
        ),
        cohort=cohort,
        cell_id=episode.cell.object_id,
        view_id=episode.view.object_id,
        complete_common_receiver_pass=_complete_receiver_pass(
            episode,
            common=True,
        ),
        complete_source_receiver_pass=_complete_receiver_pass(
            episode,
            common=False,
        ),
        phase_common_volume_fgw_min=min(value.common_volume_fgw for value in phase_gauges),
        phase_common_volume_fgw_max=max(value.common_volume_fgw for value in phase_gauges),
        phase_source_volume_fgw_max=max(value.source_volume_fgw for value in phase_gauges),
        observation_difference_max=max(
            abs(value.source_volume_fgw - value.common_volume_fgw) for value in gauges.values()
        ),
        line_volume_difference_max=max(
            abs(value.common_line_fgw - value.common_volume_fgw) for value in gauges.values()
        ),
        phase_qmin_min=min(receivers[clock]["receiver.q-min"] for clock in _PHASE_CLOCKS),
        phase_h98_min=min(receivers[clock]["receiver.h98"] for clock in _PHASE_CLOCKS),
        terminal_h98=receivers[150]["receiver.h98"],
        decision_margin_min=_decision_margin_min(episode),
        phase_density_min_m3=min(value.common_volume_density_m3 for value in phase_gauges),
        phase_density_max_m3=max(value.common_volume_density_m3 for value in phase_gauges),
        phase_lcfs_current_min_a=min(value.lcfs_current_a for value in phase_gauges),
        phase_lcfs_current_max_a=max(value.lcfs_current_a for value in phase_gauges),
        identity_residual_max=max(
            max(
                value.source_volume_identity_residual,
                value.source_line_identity_residual,
            )
            for value in gauges.values()
        ),
        available_field_ids=tuple(
            sorted(field_id for field_id, available in fields.items() if available)
        ),
        unavailable_field_ids=tuple(
            sorted(field_id for field_id, available in fields.items() if not available)
        ),
        runtime_seconds=episode.runtime_seconds,
    )


def _values(
    episodes: Mapping[str, NumericalDenominatorEpisode],
) -> dict[str, dict[int, Decimal]]:
    return {
        view_id: {clock: gauge.common_volume_fgw for clock, gauge in _gauges(episode).items()}
        for view_id, episode in episodes.items()
    }


def _effect(
    values: Mapping[str, Mapping[int, Decimal]],
    left: str,
    right: str,
) -> tuple[Decimal, ...]:
    return tuple(values[right][clock] - values[left][clock] for clock in _PHASE_CLOCKS)


def _max_abs(values: tuple[Decimal, ...]) -> Decimal:
    return max(abs(value) for value in values)


def _cell(
    cell_id: str,
    episodes: Mapping[str, NumericalDenominatorEpisode],
) -> NumericalDenominatorCellInteractionAnalysis:
    values = _values(episodes)
    dt = _effect(values, NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID, NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID)
    grid = _effect(values, NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID, NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_10_VIEW_ID)
    corrector = _effect(values, NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID, NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_20_VIEW_ID)
    joint = _effect(values, NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID, NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_20_VIEW_ID)
    dt_grid = tuple(
        values[NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_10_VIEW_ID][clock]
        - values[NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID][clock]
        - values[NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_10_VIEW_ID][clock]
        + values[NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID][clock]
        for clock in _PHASE_CLOCKS
    )
    dt_corrector = tuple(
        values[NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_20_VIEW_ID][clock]
        - values[NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID][clock]
        - values[NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_20_VIEW_ID][clock]
        + values[NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID][clock]
        for clock in _PHASE_CLOCKS
    )
    grid_corrector = tuple(
        values[NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_20_VIEW_ID][clock]
        - values[NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_10_VIEW_ID][clock]
        - values[NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_20_VIEW_ID][clock]
        + values[NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID][clock]
        for clock in _PHASE_CLOCKS
    )
    three_way = tuple(
        values[NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_20_VIEW_ID][clock]
        - values[NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_10_VIEW_ID][clock]
        - values[NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_20_VIEW_ID][clock]
        - values[NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_20_VIEW_ID][clock]
        + values[NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID][clock]
        + values[NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_10_VIEW_ID][clock]
        + values[NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_20_VIEW_ID][clock]
        - values[NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID][clock]
        for clock in _PHASE_CLOCKS
    )
    joint_minus_additive = tuple(
        joint[index] - dt[index] - grid[index] - corrector[index]
        for index in range(len(_PHASE_CLOCKS))
    )
    dt_context = (
        *_effect(values, NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID, NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID),
        *_effect(values, NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_10_VIEW_ID, NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_10_VIEW_ID),
        *_effect(values, NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_20_VIEW_ID, NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_20_VIEW_ID),
        *_effect(values, NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_20_VIEW_ID, NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_20_VIEW_ID),
    )
    grid_context = (
        *_effect(values, NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID, NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_10_VIEW_ID),
        *_effect(values, NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID, NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_10_VIEW_ID),
        *_effect(values, NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_20_VIEW_ID, NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_20_VIEW_ID),
        *_effect(values, NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_20_VIEW_ID, NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_20_VIEW_ID),
    )
    corrector_context = (
        *_effect(values, NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID, NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_20_VIEW_ID),
        *_effect(values, NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_10_VIEW_ID, NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_25_CORRECTOR_STEPS_20_VIEW_ID),
        *_effect(values, NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_10_VIEW_ID, NUMERICAL_DENOMINATOR_TIMESTEP_ONE_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_20_VIEW_ID),
        *_effect(values, NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_10_VIEW_ID, NUMERICAL_DENOMINATOR_TIMESTEP_HALF_SECOND_RADIAL_CELLS_33_CORRECTOR_STEPS_20_VIEW_ID),
    )
    passing = tuple(
        sorted(
            view_id
            for view_id, episode in episodes.items()
            if _complete_receiver_pass(episode, common=True)
        )
    )
    failing = tuple(sorted(set(episodes) - set(passing)))
    return NumericalDenominatorCellInteractionAnalysis(
        analysis_id=f"analysis.numerical-denominator.interaction.{cell_id.removeprefix('cell.')}",
        cell_id=cell_id,
        timestep_effect_max=_max_abs(dt),
        grid_effect_max=_max_abs(grid),
        corrector_effect_max=_max_abs(corrector),
        joint_minus_additive_max=_max_abs(joint_minus_additive),
        timestep_grid_interaction_max=_max_abs(dt_grid),
        timestep_corrector_interaction_max=_max_abs(dt_corrector),
        grid_corrector_interaction_max=_max_abs(grid_corrector),
        three_way_interaction_max=_max_abs(three_way),
        timestep_context_effect_min=min(dt_context),
        timestep_context_effect_max=max(dt_context),
        grid_context_effect_min=min(grid_context),
        grid_context_effect_max=max(grid_context),
        corrector_context_effect_min=min(corrector_context),
        corrector_context_effect_max=max(corrector_context),
        baseline_to_joint_mean_signed=sum(joint, Decimal(0)) / Decimal(len(joint)),
        passing_view_ids=passing,
        failing_view_ids=failing,
    )


def analyze_numerical_denominator(
    *,
    development_decision: NumericalDenominatorDevelopmentDecision,
    development_batch: NumericalDenominatorBatch,
    evaluation_adjudication: NumericalDenominatorEvaluationAdjudication,
    evaluation_batch: NumericalDenominatorBatch,
) -> NumericalDenominatorDetailedAnalysis:
    development_paths = tuple(
        _path(value, "development")
        for value in sorted(
            development_batch.episodes,
            key=lambda value: value.episode_id,
        )
    )
    evaluation_paths = tuple(
        _path(value, "evaluation")
        for value in sorted(
            evaluation_batch.episodes,
            key=lambda value: value.episode_id,
        )
    )
    by_cell: dict[str, dict[str, NumericalDenominatorEpisode]] = {
        value: {} for value in evaluation_batch.active_cell_ids
    }
    for episode in evaluation_batch.episodes:
        if episode.cell.object_id in by_cell:
            by_cell[episode.cell.object_id][episode.view.object_id] = episode
    cells = tuple(_cell(cell_id, by_cell[cell_id]) for cell_id in sorted(by_cell))
    all_paths = (*development_paths, *evaluation_paths)
    residuals = tuple(value.joint_minus_additive_max for value in cells)
    return NumericalDenominatorDetailedAnalysis(
        analysis_id='analysis.numerical-denominator.numerical-interaction',
        development_decision=ObjectIdentity.from_record(
            development_decision.decision_id,
            development_decision,
        ),
        development_batch=ObjectIdentity.from_record(
            development_batch.batch_id,
            development_batch,
        ),
        evaluation_adjudication=ObjectIdentity.from_record(
            evaluation_adjudication.adjudication_id,
            evaluation_adjudication,
        ),
        evaluation_batch=ObjectIdentity.from_record(
            evaluation_batch.batch_id,
            evaluation_batch,
        ),
        development_paths=development_paths,
        evaluation_paths=evaluation_paths,
        evaluation_cells=cells,
        development_episode_count=len(development_paths),
        evaluation_episode_count=len(evaluation_paths),
        evaluation_independent_unit_count=len(cells),
        technical_reserve_count=(
            len(development_batch.substitutions) + len(evaluation_batch.substitutions)
        ),
        incomplete_episode_count=sum(
            len(value.transitions) != 150
            for value in (*development_batch.episodes, *evaluation_batch.episodes)
        ),
        common_receiver_pass_path_count=sum(
            value.complete_common_receiver_pass for value in all_paths
        ),
        source_receiver_pass_path_count=sum(
            value.complete_source_receiver_pass for value in all_paths
        ),
        maximum_observation_difference=max(value.observation_difference_max for value in all_paths),
        minimum_interaction_residual=min(residuals),
        maximum_interaction_residual=max(residuals),
        total_simulator_runtime_seconds=sum(
            (value.runtime_seconds for value in all_paths),
            Decimal(0),
        ),
        maximum_evidence_ceiling=EvidenceCeiling.MEASUREMENT,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
    )


__all__ = [
    'NumericalDenominatorCellInteractionAnalysis',
    'NumericalDenominatorDetailedAnalysis',
    'NumericalDenominatorPathAnalysis',
    'analyze_numerical_denominator',
]
