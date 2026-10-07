"""Outcome-free parent and one-shot command rules over locked forecasts."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from empirical_lawhood.adapters.simulators.matrix_preparation.contracts import PARENTS, PreparationRoot

from empirical_lawhood.adapters.simulators.matrix_preparation.scientific_inputs import preparation_scientific_root_inputs


Array = npt.NDArray[np.float64]
TARGET_SEED_SHA256 = '4c4d971e7646e18ac63aedb9acc2c54c2cda5a404663d63138903f937748bdb4'
TARGET_CENTERS = np.array([-0.375, 0.0, 0.375])


def development_targets(roots: tuple[PreparationRoot, ...]) -> Array:
    """Independent target purpose, invoked only after the prediction lock.

    Rejection sampling removes modulo bias; neither handoff nor audit/task
    outcomes are arguments. Root IDs index the concealed draw, never a model.
    """
    values = []
    for root in roots:
        counter = 0
        while True:
            raw = bytes.fromhex(preparation_scientific_root_inputs(root.context, root.index).target_draw(counter))
            value = int.from_bytes(raw, "big")
            if value < (2**256 // 3) * 3:
                values.append(TARGET_CENTERS[value % 3])
                break
            counter += 1
    return np.asarray(values, dtype=np.float64)


@dataclass(frozen=True)
class ParentSelection:
    parent_indices: npt.NDArray[np.int64]
    refused: npt.NDArray[np.bool_]
    eligible: npt.NDArray[np.bool_]


def select_parents(
    probability_usable: Array,
    probability_preservation: Array,
    support: npt.NDArray[np.bool_],
    declared_costs: Array,
) -> ParentSelection:
    if (
        probability_usable.ndim != 2
        or probability_usable.shape[1:] != (5,)
        or probability_preservation.shape != probability_usable.shape
        or support.shape != probability_usable.shape
        or declared_costs.shape != (5,)
        or not np.isfinite(declared_costs).all()
        or np.any(declared_costs < 0)
    ):
        raise ValueError("parent selector received another legal finite-menu forecast chart")
    legal = support & np.isfinite(probability_usable) & np.isfinite(probability_preservation)
    legal &= (
        (probability_usable >= 0.90)
        & (probability_usable <= 1)
        & (probability_preservation >= 0.90)
        & (probability_preservation <= 1)
    )
    selected = np.zeros(len(legal), dtype=np.int64)
    refused = ~np.any(legal, axis=1)
    for root in range(len(legal)):
        eligible = np.flatnonzero(legal[root])
        if len(eligible):
            selected[root] = min(
                eligible,
                key=lambda p: (-probability_usable[root, p], declared_costs[p], PARENTS[p]),
            )
    # HOLD fallback is a recorded refusal. This function makes no safety claim.
    return ParentSelection(selected, refused, legal)


@dataclass(frozen=True)
class CommandDecision:
    action_indices: npt.NDArray[np.int64]
    reasons: npt.NDArray[np.int64]


def select_commands(
    predictions: Array,
    halfwidths: Array,
    targets: Array,
    probability_preservation: Array,
    support: npt.NDArray[np.bool_],
    *,
    description_qualified: bool,
) -> CommandDecision:
    """Only predictions, intervals, target and causal eligibility may enter.

    Commands are NEG=0,HOLD=1,POS=2; -1 means refusal. Both numerical
    intervals must be contained. No realized response or hidden state is an
    input. Reasons: 0 chosen, 1 unqualified, 2 unsupported, 3 preservation,
    4 unavailable interval, 5 no contained action.
    """
    if (
        predictions.ndim != 5
        or predictions.shape[1:] != (5, 2, 3, 5)
        or halfwidths.shape != predictions.shape
        or targets.shape != (len(predictions),)
        or not np.isin(targets, TARGET_CENTERS).all()
        or probability_preservation.shape != (len(predictions), 5)
        or support.shape != probability_preservation.shape
        or type(description_qualified) is not bool
    ):
        raise ValueError("one-shot controller changes its locked observable input chart")
    actions, reasons = (
        np.full((len(targets), 5), -1, dtype=np.int64),
        np.full((len(targets), 5), 1, dtype=np.int64),
    )
    if not description_qualified:
        return CommandDecision(actions, reasons)
    for r in range(len(targets)):
        for p in range(5):
            if not support[r, p]:
                reasons[r, p] = 2
                continue
            if not 0.90 <= probability_preservation[r, p] <= 1:
                reasons[r, p] = 3
                continue
            center, width = predictions[r, p, :, :, -1], halfwidths[r, p, :, :, -1]
            known = np.all(np.isfinite(center) & np.isfinite(width) & (width >= 0), axis=0)
            if not known.any():
                reasons[r, p] = 4
                continue
            contained = known & np.all(
                (center - width >= targets[r] - 0.125) & (center + width <= targets[r] + 0.125),
                axis=0,
            )
            choices = np.flatnonzero(contained)
            if not len(choices):
                reasons[r, p] = 5
                continue
            actions[r, p] = min(choices, key=lambda s: (0 if s == 1 else 1, s))
            reasons[r, p] = 0
    return CommandDecision(actions, reasons)
