"""Join the authenticated missing future without changing retained observations."""

from typing import TYPE_CHECKING

import numpy as np

from .retained import FiniteResponseLawObservedPanel, FloatArray, BoolArray

if TYPE_CHECKING:
    from .native_records import FiniteResponseLawNativeEvaluation


def join_missing_future(
    retained: FiniteResponseLawObservedPanel,
    values: FloatArray,
    observed: BoolArray,
    valid: BoolArray,
) -> FiniteResponseLawObservedPanel:
    """Input axes: root, view, parent, nine native words, seven native outputs.

    Word order: HOLD, 8/d0/-, 8/d0/+, 8/d1/-, 8/d1/+,
    16/d0/-, 16/d0/+, 16/d1/-, 16/d1/+.
    The caller authenticates the complete assigned source/receipt lineage.
    """
    shape = (16, 2, 5, 9, 7)
    if values.shape != shape or values.dtype != np.float64:
        raise ValueError("Supplement native axes or units differ")
    if any(a.shape != shape or a.dtype != np.bool_ for a in (observed, valid)):
        raise ValueError("Supplement observed/valid masks differ")
    if np.any(valid & (~observed | ~np.isfinite(values))):
        raise ValueError("Valid supplement values must be observed and finite")
    if (
        retained.observed[:16, :, :, :, 1].any()
        or retained.hold_observed[:16, :, 1].any()
    ):
        raise ValueError("The missing future is already observed; never replace it")
    y, o, d = retained.y.copy(), retained.observed.copy(), retained.valid.copy()
    hold = retained.hold_observed.copy()
    for pair in range(4):
        negative, positive = 1 + 2 * pair, 2 + 2 * pair
        # Paired odd response is in native units, not gain per force magnitude.
        response = (values[:, :, :, positive, :2] - values[:, :, :, negative, :2]) / 2
        y[:16, :, pair, :2, 1] = response.transpose(0, 2, 3, 1)
        for mask, target in ((observed, o), (valid, d)):
            paired = mask[:, :, :, positive, :2] & mask[:, :, :, negative, :2]
            target[:16, :, pair, :2, 1] = paired.transpose(0, 2, 3, 1)
        for word, output_slice in ((positive, slice(2, 5)), (negative, slice(5, 8))):
            y[:16, :, pair, output_slice, 1] = values[:, :, :, word, 2:5].transpose(
                0, 2, 3, 1
            )
            o[:16, :, pair, output_slice, 1] = observed[:, :, :, word, 2:5].transpose(
                0, 2, 3, 1
            )
            d[:16, :, pair, output_slice, 1] = valid[:, :, :, word, 2:5].transpose(
                0, 2, 3, 1
            )
    measured_hold = np.asarray((observed[:, :, :, 0] & valid[:, :, :, 0]).all(axis=3))
    hold[:16, :, 1] = measured_hold.transpose(0, 2, 1)
    return FiniteResponseLawObservedPanel(
        y, o, d, retained.parent_work.copy(), hold, retained.root_ids
    )


def supplement_from_evaluation(
    evaluation: 'FiniteResponseLawNativeEvaluation',
) -> tuple[FloatArray, BoolArray, BoolArray]:
    """Reduce the original 16-root D1 views into the missing-future panel.

    Callers authenticate the source, successful task/receipt census and sole
    scientific adjudication before entry. Incomplete native evaluations refuse
    entry. The reduction preserves the original output values and validity
    rules without imputation.
    This reducer performs no native acquisition, authority act or publication.
    """
    from .science import PARENTS

    if (
        evaluation.config.projection.native_spec.stage != "supplemental-development"
        or evaluation.reasons
        or evaluation.completed_native_updates != 414720
    ):
        raise ValueError("D1 native measurements incomplete")
    values = np.full((16, 2, 5, 9, 7), np.nan)
    observed = np.zeros(values.shape, dtype=bool)
    valid = observed.copy()
    slots = set()
    for view in evaluation.views:
        for row in view.words:
            task, word = (row.invocation, row.invocation.word)
            if (
                word is None
                or task.purpose != "future-2"
                or task.clocks != (4368, 4560)
            ):
                raise ValueError("D1 word changes future/clock")
            w = (
                0
                if word.sign == 0
                else 1
                + (4 if word.magnitude == 16 else 0)
                + 2 * word.direction_index
                + (word.sign == 1)
            )
            slot = (task.root.index, view.refinement - 1, PARENTS.index(task.parent), w)
            if slot in slots:
                raise ValueError("D1 repeats a native word/view")
            slots.add(slot)
            vals = np.asarray([np.nan if x is None else float(x) for x in row.outputs])
            values[slot], observed[slot] = (vals, True)
            valid[slot] = (
                np.isfinite(vals) & row.complete & (row.maximum_force_error == 0)
            )
    if len(slots) != 16 * 2 * 5 * 9:
        raise ValueError("D1 projection census differs")
    return (values, observed, valid)
