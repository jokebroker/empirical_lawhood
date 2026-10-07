"""Frozen Tier-L0 tuning, validation tournament, and development-only refit."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import math

import numpy as np
from numpy.typing import NDArray

from .analysis import (
    LinearResponseModel,
    ModelFamily,
    ModelFitRecord,
    OfficialAggregateScore,
    TargetFeatureMatrix,
    TargetResponseMatrix,
    TournamentEntry,
    TournamentRecord,
    VirtualCellAnalysisError,
    cross_fitted_prefix_admission,
    deterministic_target_folds,
    evaluate_prediction,
    fit_prefix_admission_model,
    fit_tier_l0_model,
    predict_response,
    realized_admission_labels,
    select_tournament,
)
from .contracts import ModelSelectionContract
from .records import FalsifierPanel


COMPETITION_SELECTION_INTEGRITY_FALSIFIER_IDS = (
    "falsifier.feature-cutoff-and-provenance",
    "falsifier.final-outcome-lineage-empty",
    "falsifier.leaderboard-lineage-empty",
    "falsifier.matched-control-contract",
)


@dataclass(frozen=True, slots=True)
class HyperparameterChoice:
    family: ModelFamily
    alpha: float
    reduced_rank: int | None
    prefix_alpha: float | None

    @property
    def choice_id(self) -> str:
        rank = "full" if self.reduced_rank is None else str(self.reduced_rank)
        prefix = "none" if self.prefix_alpha is None else f"{self.prefix_alpha:g}"
        return f"{self.family.value.lower()}-a{self.alpha:g}-r{rank}-pa{prefix}"


@dataclass(frozen=True, slots=True)
class DevelopedCandidate:
    candidate_id: str
    hyperparameters: HyperparameterChoice
    model: LinearResponseModel
    fit: ModelFitRecord
    cross_validation_mae: float
    cross_validation_cosine: float
    cross_validation_proxy: float

    def __post_init__(self) -> None:
        values = (
            self.cross_validation_mae,
            self.cross_validation_cosine,
            self.cross_validation_proxy,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("candidate metrics must be finite")


@dataclass(frozen=True, slots=True)
class CandidateOfficialValidation:
    """Exact scorer result for one candidate on development-visible validation."""

    candidate_id: str
    target_ids: tuple[str, ...]
    score: OfficialAggregateScore
    score_standard_error: Decimal
    metrics_arrow: bytes
    cell_eval_version: str
    pdex_version: str
    response_outcome_read: bool

    def __post_init__(self) -> None:
        if tuple(sorted(set(self.target_ids))) != self.target_ids:
            raise ValueError("candidate validation targets must be sorted and unique")
        if not self.metrics_arrow:
            raise ValueError("candidate validation requires exact per-target metrics")
        if not self.score_standard_error.is_finite() or self.score_standard_error < 0:
            raise ValueError("candidate validation score SE must be finite and nonnegative")
        if not self.cell_eval_version or not self.pdex_version:
            raise ValueError("candidate validation evaluator versions are required")
        if not self.response_outcome_read:
            raise ValueError("candidate validation must acknowledge its validation outcome read")


@dataclass(frozen=True, slots=True)
class CandidateDevelopmentRefit:
    """Outcome-blind all-development refit retained for every selectable family."""

    candidate_id: str
    model: LinearResponseModel
    fit: ModelFitRecord

    def __post_init__(self) -> None:
        if self.model.model_id != f"{self.candidate_id}-development-refit":
            raise ValueError("development refit model has another candidate identity")
        if self.fit.model_id != self.model.model_id:
            raise ValueError("development refit and fit record differ")


@dataclass(frozen=True, slots=True)
class TierL0Development:
    candidates: tuple[DevelopedCandidate, ...]
    official_validations: tuple[CandidateOfficialValidation, ...]
    development_refits: tuple[CandidateDevelopmentRefit, ...]
    tournament: TournamentRecord
    selected_validation_model: LinearResponseModel
    train_target_ids: tuple[str, ...]
    validation_target_ids: tuple[str, ...]
    development_target_ids: tuple[str, ...]
    realized_admission_repression_scale: float
    final_outcome_read: bool
    leaderboard_read: bool

    def __post_init__(self) -> None:
        ids = tuple(candidate.candidate_id for candidate in self.candidates)
        if tuple(sorted(set(ids))) != ids:
            raise ValueError("developed candidate IDs must be sorted and unique")
        if self.tournament.selected_candidate_id not in ids:
            raise ValueError("tournament selection is absent from candidate models")
        validation_ids = tuple(value.candidate_id for value in self.official_validations)
        if validation_ids != ids:
            raise ValueError("official validation must cover every selectable candidate exactly")
        if self.selected_validation_model.model_id != self.tournament.selected_candidate_id:
            raise ValueError("selected validation model differs from the tournament")
        refit_ids = tuple(value.candidate_id for value in self.development_refits)
        if refit_ids != ids:
            raise ValueError("development refits must cover every selectable candidate exactly")
        if set(self.train_target_ids) & set(self.validation_target_ids):
            raise ValueError("training and validation actions must be disjoint")
        if self.development_target_ids != tuple(
            sorted((*self.train_target_ids, *self.validation_target_ids))
        ):
            raise ValueError("development refit target roster does not close")
        if self.final_outcome_read or self.leaderboard_read:
            raise ValueError("development cannot read final outcomes or leaderboard")

    def refit_for(self, candidate_id: str) -> CandidateDevelopmentRefit:
        return next(
            value for value in self.development_refits if value.candidate_id == candidate_id
        )

    @property
    def final_refit_model(self) -> LinearResponseModel:
        return self.refit_for(self.tournament.selected_candidate_id).model

    @property
    def final_refit(self) -> ModelFitRecord:
        return self.refit_for(self.tournament.selected_candidate_id).fit


def competition_selection_integrity_failures(
    panel: FalsifierPanel,
) -> tuple[str, ...]:
    """Return only causal/provenance failures that can block a prediction freeze.

    Quantitative target/gene permutation results adjudicate response-law support;
    they never replace the official-validation competition winner.
    """

    by_id = {value.falsifier_id: value for value in panel.results}
    missing = set(COMPETITION_SELECTION_INTEGRITY_FALSIFIER_IDS) - set(by_id)
    if missing:
        raise VirtualCellAnalysisError(
            "falsifier panel lacks competition-selection integrity sentinels"
        )
    return tuple(
        falsifier_id
        for falsifier_id in COMPETITION_SELECTION_INTEGRITY_FALSIFIER_IDS
        if not by_id[falsifier_id].passed
    )


def _subset(
    responses: TargetResponseMatrix,
    indices: NDArray[np.int64],
) -> TargetResponseMatrix:
    return TargetResponseMatrix(
        target_ids=tuple(responses.target_ids[index] for index in indices),
        gene_ids=responses.gene_ids,
        values=np.asarray(responses.values[indices], dtype=np.float64),
        target_weights=np.asarray(responses.target_weights[indices], dtype=np.float64),
    )


def merge_development_responses(
    train: TargetResponseMatrix,
    validation: TargetResponseMatrix,
) -> TargetResponseMatrix:
    if train.gene_ids != validation.gene_ids:
        raise VirtualCellAnalysisError("development response gene registries differ")
    if set(train.target_ids) & set(validation.target_ids):
        raise VirtualCellAnalysisError("training and validation target rosters overlap")
    rows = {
        target_id: (values, weight)
        for target_id, values, weight in zip(
            (*train.target_ids, *validation.target_ids),
            np.concatenate((train.values, validation.values), axis=0),
            np.concatenate((train.target_weights, validation.target_weights)),
            strict=True,
        )
    }
    target_ids = tuple(sorted(rows))
    return TargetResponseMatrix(
        target_ids=target_ids,
        gene_ids=train.gene_ids,
        values=np.stack([rows[target_id][0] for target_id in target_ids]),
        target_weights=np.asarray([rows[target_id][1] for target_id in target_ids]),
    )


def _proxy(mae: float, cosine: float) -> float:
    """Frozen bounded response proxy: equal direction and magnitude terms."""

    direction = min(1.0, max(0.0, (cosine + 1.0) / 2.0))
    magnitude = 1.0 / (1.0 + max(0.0, mae))
    return 0.5 * (direction + magnitude)


def _choices(contract: ModelSelectionContract) -> tuple[HyperparameterChoice, ...]:
    alphas = tuple(float(value) for value in contract.ridge_alphas)
    values = [
        HyperparameterChoice(ModelFamily.NO_CHANGE_BASELINE, 0.0, None, None),
        HyperparameterChoice(ModelFamily.WEIGHTED_COMMON_RESPONSE_BASELINE, 0.0, None, None),
    ]
    values.extend(
        HyperparameterChoice(ModelFamily.TARGET_FEATURE_RIDGE_RESPONSE, alpha, None, None) for alpha in alphas
    )
    for family in (ModelFamily.REDUCED_RANK_TARGET_FEATURE_RIDGE_RESPONSE, ModelFamily.ADMISSION_CONDITIONED_REDUCED_RANK_RIDGE_RESPONSE):
        values.extend(
            HyperparameterChoice(
                family,
                alpha,
                rank,
                alpha if family is ModelFamily.ADMISSION_CONDITIONED_REDUCED_RANK_RIDGE_RESPONSE else None,
            )
            for alpha in alphas
            for rank in contract.reduced_ranks
        )
    return tuple(values)


def _fit_choice(
    *,
    choice: HyperparameterChoice,
    model_id: str,
    responses: TargetResponseMatrix,
    features: TargetFeatureMatrix,
    admission_labels: NDArray[np.float64],
    admission_fold_count: int,
    admission_seed: int,
    response_basis: NDArray[np.float64] | None,
) -> tuple[LinearResponseModel, ModelFitRecord]:
    if choice.family in {ModelFamily.NO_CHANGE_BASELINE, ModelFamily.WEIGHTED_COMMON_RESPONSE_BASELINE}:
        return fit_tier_l0_model(
            model_id=model_id,
            family=choice.family,
            responses=responses,
            features=None,
        )
    prefix_values: NDArray[np.float64] | None = None
    prefix_model = None
    if choice.family is ModelFamily.ADMISSION_CONDITIONED_REDUCED_RANK_RIDGE_RESPONSE:
        assert choice.prefix_alpha is not None
        prefix_values = cross_fitted_prefix_admission(
            target_ids=responses.target_ids,
            features=features,
            realized_labels=admission_labels,
            alpha=choice.prefix_alpha,
            fold_count=admission_fold_count,
            seed=admission_seed,
        )
        prefix_model = fit_prefix_admission_model(
            model_id=f"prefix-{model_id}",
            target_ids=responses.target_ids,
            features=features,
            realized_labels=admission_labels,
            alpha=choice.prefix_alpha,
        )
    return fit_tier_l0_model(
        model_id=model_id,
        family=choice.family,
        responses=responses,
        features=features,
        alpha=choice.alpha,
        reduced_rank=choice.reduced_rank,
        prefix_admission=prefix_values,
        prefix_admission_model=prefix_model,
        response_basis_override=response_basis,
    )


def _maximum_basis(
    responses: TargetResponseMatrix,
    maximum_rank: int,
) -> NDArray[np.float64]:
    centered = responses.values - np.average(
        responses.values,
        axis=0,
        weights=responses.target_weights,
    )
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    return np.asarray(right[:maximum_rank], dtype=np.float64)


def fit_tier_l0_candidates(
    *,
    train: TargetResponseMatrix,
    validation: TargetResponseMatrix,
    features: TargetFeatureMatrix,
    contract: ModelSelectionContract,
    realized_admission_repression_scale: float,
) -> tuple[DevelopedCandidate, ...]:
    """Tune hyperparameters on grouped training folds and fit one model per family."""

    if set(train.target_ids) & set(validation.target_ids):
        raise VirtualCellAnalysisError("training and validation targets overlap")
    if train.gene_ids != validation.gene_ids:
        raise VirtualCellAnalysisError("training and validation gene registries differ")
    choices = _choices(contract)
    labels = realized_admission_labels(
        train,
        repression_scale=realized_admission_repression_scale,
    )
    metric_rows: dict[str, list[tuple[float, float, float]]] = {
        choice.choice_id: [] for choice in choices
    }
    all_indices = np.arange(len(train.target_ids), dtype=np.int64)
    maximum_requested_rank = max(contract.reduced_ranks)
    for seed in contract.repeat_seeds:
        folds = deterministic_target_folds(
            train.target_ids,
            fold_count=contract.outer_target_folds,
            seed=seed,
        )
        for fold_index, held_values in enumerate(folds):
            held = np.asarray(held_values, dtype=np.int64)
            training = np.setdiff1d(all_indices, held, assume_unique=True)
            training_response = _subset(train, training)
            held_response = _subset(train, held)
            maximum_rank = min(
                maximum_requested_rank,
                len(training_response.target_ids) - 1,
                len(training_response.gene_ids),
            )
            basis = _maximum_basis(training_response, maximum_rank)
            for choice in choices:
                if choice.reduced_rank is not None and choice.reduced_rank > maximum_rank:
                    continue
                model, _fit = _fit_choice(
                    choice=choice,
                    model_id=f"cv-{choice.choice_id}-{seed}-{fold_index}",
                    responses=training_response,
                    features=features,
                    admission_labels=labels[training],
                    admission_fold_count=min(
                        contract.outer_target_folds,
                        len(training_response.target_ids),
                    ),
                    admission_seed=seed * 100 + fold_index,
                    response_basis=(
                        basis
                        if choice.family
                        in {ModelFamily.REDUCED_RANK_TARGET_FEATURE_RIDGE_RESPONSE, ModelFamily.ADMISSION_CONDITIONED_REDUCED_RANK_RIDGE_RESPONSE}
                        else None
                    ),
                )
                prediction = predict_response(
                    model,
                    target_ids=held_response.target_ids,
                    features=(
                        None
                        if choice.family in {ModelFamily.NO_CHANGE_BASELINE, ModelFamily.WEIGHTED_COMMON_RESPONSE_BASELINE}
                        else features
                    ),
                )
                mae, cosine, _ = evaluate_prediction(held_response, prediction)
                metric_rows[choice.choice_id].append((mae, cosine, _proxy(mae, cosine)))
    best_by_family: dict[ModelFamily, HyperparameterChoice] = {}
    for family in ModelFamily:
        eligible = [
            choice
            for choice in choices
            if choice.family is family and metric_rows[choice.choice_id]
        ]
        if not eligible:
            raise VirtualCellAnalysisError(f"no evaluable hyperparameters for {family.value}")
        best_by_family[family] = min(
            eligible,
            key=lambda choice: (
                -float(np.mean([row[2] for row in metric_rows[choice.choice_id]])),
                choice.reduced_rank or 0,
                choice.alpha,
                choice.choice_id,
            ),
        )
    developed: list[DevelopedCandidate] = []
    full_basis = _maximum_basis(
        train,
        min(maximum_requested_rank, len(train.target_ids) - 1, len(train.gene_ids)),
    )
    for family in ModelFamily:
        choice = best_by_family[family]
        model_id = {
            ModelFamily.NO_CHANGE_BASELINE: "baseline-no-change",
            ModelFamily.WEIGHTED_COMMON_RESPONSE_BASELINE: "baseline-weighted-common-response",
            ModelFamily.TARGET_FEATURE_RIDGE_RESPONSE: "feature-ridge-response",
            ModelFamily.REDUCED_RANK_TARGET_FEATURE_RIDGE_RESPONSE: "feature-ridge-response-with-reduced-rank",
            ModelFamily.ADMISSION_CONDITIONED_REDUCED_RANK_RIDGE_RESPONSE: "receiver-admission-conditioned-reduced-rank-ridge-response",
        }[family]
        model, fit = _fit_choice(
            choice=choice,
            model_id=model_id,
            responses=train,
            features=features,
            admission_labels=labels,
            admission_fold_count=contract.outer_target_folds,
            admission_seed=contract.repeat_seeds[0],
            response_basis=(
                full_basis
                if family in {ModelFamily.REDUCED_RANK_TARGET_FEATURE_RIDGE_RESPONSE, ModelFamily.ADMISSION_CONDITIONED_REDUCED_RANK_RIDGE_RESPONSE}
                else None
            ),
        )
        rows = metric_rows[choice.choice_id]
        developed.append(
            DevelopedCandidate(
                candidate_id=model_id,
                hyperparameters=choice,
                model=model,
                fit=fit,
                cross_validation_mae=float(np.mean([row[0] for row in rows])),
                cross_validation_cosine=float(np.mean([row[1] for row in rows])),
                cross_validation_proxy=float(np.mean([row[2] for row in rows])),
            )
        )
    result = tuple(sorted(developed, key=lambda value: value.candidate_id))
    if tuple(value.candidate_id for value in result) != contract.candidate_ids:
        raise VirtualCellAnalysisError("fitted candidate roster differs from the frozen contract")
    return result


def finalize_tier_l0_development(
    *,
    train: TargetResponseMatrix,
    validation: TargetResponseMatrix,
    features: TargetFeatureMatrix,
    contract: ModelSelectionContract,
    realized_admission_repression_scale: float,
    candidates: tuple[DevelopedCandidate, ...],
    official_validations: tuple[CandidateOfficialValidation, ...],
) -> TierL0Development:
    """Select by exact official validation score, then refit every family."""

    candidate_ids = tuple(value.candidate_id for value in candidates)
    validation_ids = tuple(value.candidate_id for value in official_validations)
    if candidate_ids != contract.candidate_ids or validation_ids != candidate_ids:
        raise VirtualCellAnalysisError("candidate and official-validation rosters do not close")
    if set(train.target_ids) & set(validation.target_ids):
        raise VirtualCellAnalysisError("training and validation targets overlap")
    if train.gene_ids != validation.gene_ids:
        raise VirtualCellAnalysisError("training and validation gene registries differ")
    for evaluation in official_validations:
        if evaluation.target_ids != validation.target_ids:
            raise VirtualCellAnalysisError(
                f"official validation targets differ for {evaluation.candidate_id}"
            )
    entries = tuple(
        TournamentEntry(
            candidate_id=candidate.candidate_id,
            validation_des=evaluation.score.des,
            validation_pds=evaluation.score.pds,
            validation_mae=evaluation.score.mae,
            official_score=evaluation.score.average_score,
            official_score_standard_error=evaluation.score_standard_error,
            parameter_count=candidate.fit.parameter_count,
            eligible=True,
            reason_codes=(),
        )
        for candidate, evaluation in zip(candidates, official_validations, strict=True)
    )
    tournament = select_tournament(
        tournament_id="tournament.virtual-cell-2025-tier-l0",
        entries=entries,
    )
    selected = next(
        value for value in candidates if value.candidate_id == tournament.selected_candidate_id
    )
    development = merge_development_responses(train, validation)
    development_labels = realized_admission_labels(
        development,
        repression_scale=realized_admission_repression_scale,
    )
    final_basis = _maximum_basis(
        development,
        min(
            max(contract.reduced_ranks),
            len(development.target_ids) - 1,
            len(development.gene_ids),
        ),
    )
    refits = []
    for candidate in candidates:
        final_model, final_fit = _fit_choice(
            choice=candidate.hyperparameters,
            model_id=f"{candidate.candidate_id}-development-refit",
            responses=development,
            features=features,
            admission_labels=development_labels,
            admission_fold_count=contract.outer_target_folds,
            admission_seed=contract.repeat_seeds[-1],
            response_basis=(
                final_basis
                if candidate.hyperparameters.family
                in {ModelFamily.REDUCED_RANK_TARGET_FEATURE_RIDGE_RESPONSE, ModelFamily.ADMISSION_CONDITIONED_REDUCED_RANK_RIDGE_RESPONSE}
                else None
            ),
        )
        refits.append(
            CandidateDevelopmentRefit(
                candidate_id=candidate.candidate_id,
                model=final_model,
                fit=final_fit,
            )
        )
    return TierL0Development(
        candidates=candidates,
        official_validations=official_validations,
        development_refits=tuple(refits),
        tournament=tournament,
        selected_validation_model=selected.model,
        train_target_ids=train.target_ids,
        validation_target_ids=validation.target_ids,
        development_target_ids=development.target_ids,
        realized_admission_repression_scale=realized_admission_repression_scale,
        final_outcome_read=False,
        leaderboard_read=False,
    )


__all__ = [
    "COMPETITION_SELECTION_INTEGRITY_FALSIFIER_IDS",
    "CandidateDevelopmentRefit",
    "CandidateOfficialValidation",
    "DevelopedCandidate",
    "HyperparameterChoice",
    "TierL0Development",
    "competition_selection_integrity_failures",
    "finalize_tier_l0_development",
    "fit_tier_l0_candidates",
    "merge_development_responses",
]
