"""Five predeclared fresh C contrasts on whole independent reactor roots."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt

import numpy as np

from .config import ROOTS

CONTRASTS = ("R", "H", "I_READ", "I_EXCITATION", "I_USE")


@dataclass(frozen=True)
class RootComparisonLosses:
    root: str
    complete_baseline_contact: bool
    common_anchor_preparation_pair_contact: bool
    p_at_q_contact: bool
    r_rival_K2: float | None
    r_local_K2: float | None
    h_current_K2: float | None
    h_history_K2: float | None
    constant_preparation_action_only_common_anchor_loss_K2: float | None
    constant_preparation_full_readout_common_anchor_loss_K2: float | None
    probe_preparation_action_only_common_anchor_loss_K2: float | None
    probe_preparation_full_readout_common_anchor_loss_K2: float | None
    p_masked_q_K2: float | None
    p_full_q_K2: float | None
    source_valid: bool
    sealed_before_labels: bool
    probe_preparation_common_anchor_contact: bool | None = None
    source_valid_by_contrast: tuple[bool, bool, bool, bool, bool] | None = None

    def __post_init__(self) -> None:
        values = (
            self.r_rival_K2,
            self.r_local_K2,
            self.h_current_K2,
            self.h_history_K2,
            self.constant_preparation_action_only_common_anchor_loss_K2,
            self.constant_preparation_full_readout_common_anchor_loss_K2,
            self.probe_preparation_action_only_common_anchor_loss_K2,
            self.probe_preparation_full_readout_common_anchor_loss_K2,
            self.p_masked_q_K2,
            self.p_full_q_K2,
        )
        if any(value is not None and (not isfinite(value) or value < 0) for value in values):
            raise ValueError("fresh comparison loss is negative or nonfinite")


@dataclass(frozen=True)
class ContrastResult:
    contrast_id: str
    assigned_roots: int
    complete_contact_roots: int
    model_failure_roots: int
    mean_advantage_K2: float | None
    lower_99_one_sided_K2: float | None
    upper_99_one_sided_K2: float | None
    arm_rmse_K: tuple[float, ...]
    arm_loss_ratio: float | None
    verdict: str
    reasons: tuple[str, ...]


def _contact(row: RootComparisonLosses, contrast: str) -> bool:
    if contrast in ("R", "H"):
        return row.complete_baseline_contact
    if contrast == "I_READ":
        return row.common_anchor_preparation_pair_contact
    if contrast == "I_EXCITATION":
        return row.common_anchor_preparation_pair_contact
    return row.p_at_q_contact


def _source_valid(row: RootComparisonLosses, contrast: str) -> bool:
    if row.source_valid_by_contrast is None:
        return row.source_valid
    return row.source_valid_by_contrast[CONTRASTS.index(contrast)]


def _arms(row: RootComparisonLosses, contrast: str) -> tuple[float | None, ...]:
    if contrast == "R":
        return row.r_rival_K2, row.r_local_K2
    if contrast == "H":
        return row.h_current_K2, row.h_history_K2
    if contrast == "I_READ":
        return row.probe_preparation_action_only_common_anchor_loss_K2, row.probe_preparation_full_readout_common_anchor_loss_K2
    if contrast == "I_EXCITATION":
        return (
            row.probe_preparation_action_only_common_anchor_loss_K2,
            row.probe_preparation_full_readout_common_anchor_loss_K2,
            row.constant_preparation_action_only_common_anchor_loss_K2,
            row.constant_preparation_full_readout_common_anchor_loss_K2,
        )
    if contrast == "I_USE":
        return row.p_masked_q_K2, row.p_full_q_K2
    raise ValueError("undeclared C contrast")


def _advantage(contrast: str, arms: tuple[float, ...]) -> float:
    if contrast == "I_EXCITATION":
        p_masked, p_full, c_masked, c_full = arms
        return (p_masked - p_full) - (c_masked - c_full) - 0.2 * p_masked
    baseline, improved = arms
    return 0.8 * baseline - improved


def reduce_contrasts(roots: tuple[RootComparisonLosses, ...]) -> tuple[ContrastResult, ...]:
    expected = tuple(root for root, role, _, _ in ROOTS if role == "qualification")
    if tuple(row.root for row in roots) != expected:
        raise ValueError("five contrasts require all 64 fresh qualification roots")
    results = []
    for contrast in CONTRASTS:
        contact = tuple(row for row in roots if _contact(row, contrast))
        model_failure = sum(
            not _source_valid(row, contrast)
            or not row.sealed_before_labels
            or any(value is None for value in _arms(row, contrast))
            for row in contact
        )
        valid = tuple(
            row
            for row in contact
            if _source_valid(row, contrast)
            and row.sealed_before_labels
            and all(value is not None for value in _arms(row, contrast))
        )
        reasons = set()
        if len(contact) < 48:
            reasons.add("INSUFFICIENT_COMPLETE_CONTEXT_CONTACT")
        if model_failure:
            reasons.add("MODEL_OR_SEAL_FAILURE_ON_CONTACTED_ROOT")
        if not valid:
            reasons.add("NO_EVALUABLE_MATCHED_ROOTS")
            results.append(
                ContrastResult(contrast, 64, len(contact), model_failure, None, None, None, (), None, "UNEVALUABLE", tuple(sorted(reasons)))
            )
            continue
        values = np.asarray([_arms(row, contrast) for row in valid], dtype=np.float64)
        advantage = np.asarray([_advantage(contrast, tuple(value)) for value in values])
        rng = np.random.default_rng(20260924)
        sampled = advantage[rng.integers(0, len(advantage), size=(20000, len(advantage)))]
        mean = float(np.mean(advantage))
        lower = float(np.quantile(np.mean(sampled, axis=1), 0.01))
        upper = float(np.quantile(np.mean(sampled, axis=1), 0.99))
        losses = np.mean(values, axis=0)
        rmse = tuple(sqrt(float(loss)) for loss in losses)
        ratio = float(losses[1] / losses[0]) if losses[0] > 0 else None
        if reasons:
            verdict = "UNEVALUABLE"
        elif lower > 0:
            verdict = "SUPPORTED_20_PERCENT_ADVANTAGE"
        elif upper < 0:
            verdict = "OPPOSED_20_PERCENT_ADVANTAGE"
        else:
            verdict = "UNRESOLVED_20_PERCENT_ADVANTAGE"
        results.append(
            ContrastResult(
                contrast,
                64,
                len(contact),
                model_failure,
                mean,
                lower,
                upper,
                rmse,
                ratio,
                verdict,
                tuple(sorted(reasons)),
            )
        )
    return tuple(results)
