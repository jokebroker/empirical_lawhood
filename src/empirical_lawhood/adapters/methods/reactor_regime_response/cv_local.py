"Fourfold whole-root greedy local discovery for fresh regime-response study fit roots.\n\nSplit coordinate and quantile are the only proposals shared across folds.\nEvery numerical threshold, normalizer and leaf fit is reconstructed from the\ntraining roots of that fold before its held-out roots are scored.\n"

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import CoefficientRows, Fit, LAMBDAS, LocalLeaf, _ridge, _standardize, fit_affine, split_stable

QUANTILES = (0.25, 0.5, 0.75)


@dataclass(frozen=True)
class SplitProposal:
    parent: tuple[bool, ...]
    coordinate: int
    quantile: float


def _threshold(values: np.ndarray, quantile: float) -> float:
    distinct = np.unique(values)
    if len(distinct) < 2:
        raise ValueError("candidate split coordinate has no distinct fit values")
    index = min(len(distinct) - 2, max(0, int(np.floor(quantile * (len(distinct) - 1)))))
    return float((distinct[index] + distinct[index + 1]) / 2)


def _built(
    rows: CoefficientRows,
    penalty: float,
    proposals: tuple[SplitProposal, ...],
    *,
    min_child_roots: int,
) -> Fit:
    if not proposals:
        return fit_affine(rows, penalty)
    mean, scale, z = _standardize(rows.features, rows.context_weight)
    leaves: dict[tuple[bool, ...], tuple[np.ndarray, tuple[tuple[int, float, bool], ...]]] = {
        (): (np.ones(len(rows.roots), dtype=bool), ())
    }
    for proposal in proposals:
        if proposal.parent not in leaves:
            raise ValueError("local split parent is absent")
        mask, path = leaves.pop(proposal.parent)
        threshold = _threshold(rows.features[mask, proposal.coordinate], proposal.quantile)
        if not split_stable(rows.features[mask], proposal.coordinate, threshold):
            raise ValueError("local split changes under canonical perturbation")
        left = mask & (rows.features[:, proposal.coordinate] <= threshold)
        right = mask & ~left
        if (
            len({root for root, selected in zip(rows.roots, left, strict=True) if selected})
            < min_child_roots
            or len({root for root, selected in zip(rows.roots, right, strict=True) if selected})
            < min_child_roots
        ):
            raise ValueError("local split has insufficient distinct fit roots")
        leaves[proposal.parent + (True,)] = (
            left,
            path + ((proposal.coordinate, threshold, True),),
        )
        leaves[proposal.parent + (False,)] = (
            right,
            path + ((proposal.coordinate, threshold, False),),
        )
    fitted = []
    for _, (mask, path) in sorted(leaves.items()):
        if not mask.any():
            raise ValueError("local partition has an empty leaf")
        fitted.append(LocalLeaf(path, _ridge(z[mask], rows.coefficient[mask], rows.fit_weight[mask], penalty)))
    return Fit("local", mean, scale, np.empty(0), penalty, leaves=tuple(fitted))


def _fold_indices(rows: CoefficientRows) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    folds = np.asarray([int(root.rsplit("-", 1)[-1]) % 4 for root in rows.roots])
    result = []
    for fold in range(4):
        train = np.where(folds != fold)[0]
        validation = np.where(folds == fold)[0]
        if not len(train) or not len(validation):
            raise ValueError("whole-root discovery lacks one of four folds")
        result.append((train, validation))
    return tuple(result)


def _cv_loss(
    rows: CoefficientRows,
    penalty: float,
    proposals: tuple[SplitProposal, ...],
    parent: tuple[bool, ...] | None = None,
) -> float:
    root_losses: dict[str, float] = {}
    for train, validation in _fold_indices(rows):
        trained = rows.subset(train)
        heldout = rows.subset(validation)
        fit = _built(trained, penalty, proposals, min_child_roots=12)
        if parent:
            path = next(leaf.path[:len(parent)] for leaf in fit.leaves
                        if tuple(item[2] for item in leaf.path[:len(parent)]) == parent)
            mask = np.ones(len(heldout.roots), dtype=bool)
            for coordinate, threshold, left in path:
                mask &= (heldout.features[:, coordinate] <= threshold) == left
            if not mask.any():
                continue
            heldout = heldout.subset(np.where(mask)[0])
        loss = heldout.root_losses(fit.predict(heldout.features))
        if root_losses.keys() & loss.keys():
            raise ValueError("whole-root discovery reused a validation root")
        root_losses.update(loss)
    if parent is None and len(root_losses) != len(set(rows.roots)):
        raise ValueError("whole-root discovery omitted a fit root")
    if not root_losses:
        raise ValueError("local parent has no held-out context")
    return rows.average_root_loss(root_losses)


def fit_local_cv(rows: CoefficientRows, penalty: float) -> Fit:
    """Select up to two stable splits by held-out root contrast loss."""
    if penalty not in LAMBDAS:
        raise ValueError("undeclared local ridge penalty")
    selected: tuple[SplitProposal, ...] = ()
    for _ in range(2):
        baseline = _cv_loss(rows, penalty, selected)
        parents = ((),) if not selected else ((True,), (False,))
        best: tuple[float, float, SplitProposal] | None = None
        for parent in parents:
            parent_baseline = baseline if not parent else _cv_loss(rows, penalty, selected, parent)
            if len(selected) == 2:
                continue
            for coordinate in range(rows.features.shape[1]):
                for quantile in QUANTILES:
                    proposal = SplitProposal(parent, coordinate, quantile)
                    candidate = selected + (proposal,)
                    try:
                        # A split unavailable in any fold is not selected from
                        # the remaining folds or from held-out features.
                        loss = _cv_loss(rows, penalty, candidate)
                        parent_loss = loss if not parent else _cv_loss(rows, penalty, candidate, parent)
                        complete_fit = _built(rows, penalty, candidate, min_child_roots=12)
                    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                        continue
                    gain = baseline - loss
                    parent_gain = parent_baseline - parent_loss
                    if gain <= 0 or parent_gain <= 1e-12 or parent_gain < 0.1 * parent_baseline:
                        continue
                    threshold = next(leaf.path[len(parent)][1] for leaf in complete_fit.leaves
                                     if len(leaf.path) > len(parent)
                                     and tuple(item[2] for item in leaf.path[:len(parent)]) == parent)
                    if best is None or (-gain, coordinate, threshold, parent, quantile) < (
                        -best[0], best[2].coordinate, best[1], best[2].parent, best[2].quantile
                    ):
                        best = gain, threshold, proposal
        if best is None:
            break
        selected += (best[2],)
    if not selected:
        raise ValueError("no fourfold stable local gain reaches the declared threshold")
    return _built(rows, penalty, selected, min_child_roots=12)
