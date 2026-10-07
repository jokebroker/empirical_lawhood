"""Root-blocked model-tree discovery with explicitly paired action challenges.

This is a candidate producer. It neither constructs ResponseLaw nor grants
support, control admission or a scientific qualification status.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


# Candidate-action coordinates cannot determine the pre-action operating domain.
CONTEXT = (0, 1, 3, 4, 7, 8, 13, 14, 15, 16, 17, 18, 19, 20)
ABSOLUTE_SCALE = np.array((0.10, 0.005))
RESPONSE_SCALE = np.array((0.01, 0.0002))


@dataclass(frozen=True)
class DiscoveryDesign:
    max_leaves: int = 16
    max_depth: int = 6
    minimum_roots: int = 8
    minimum_rows: int = 256
    folds: int = 4
    minimum_gain: float = 0.05
    minimum_shared_loss: float = 0.0001
    ridge: float = 0.000001
    quantiles: tuple[float, ...] = (0.25, 0.5, 0.75)

    def __post_init__(self) -> None:
        if (
            not 1 <= self.max_leaves <= 32
            or not 1 <= self.max_depth <= 8
            or self.minimum_roots < self.folds * 2
            or self.minimum_rows < 32
            or self.folds < 2
            or not 0 < self.minimum_gain < 1
            or not 0 < self.minimum_shared_loss < 1
            or not 0 < self.ridge <= 1
            or self.quantiles != (0.25, 0.5, 0.75)
        ):
            raise ValueError("invalid bounded discovery design")


@dataclass(frozen=True)
class PairedContrasts:
    """Indices share a root and the identical pre-intervention causal history."""

    donor: np.ndarray
    branch: np.ndarray
    channel: np.ndarray
    contact: np.ndarray

    def validate(self, x: np.ndarray, roots: np.ndarray) -> None:
        n = len(self.donor)
        if any(a.shape != (n,) for a in (self.branch, self.channel, self.contact)):
            raise ValueError("paired contrast axes differ")
        if (
            self.donor.dtype.kind not in "iu"
            or self.branch.dtype.kind not in "iu"
            or np.any(self.donor < 0)
            or np.any(self.branch < 0)
            or np.any(self.donor >= len(x))
            or np.any(self.branch >= len(x))
            or not np.isin(self.channel, (0, 1)).all()
            or self.contact.dtype.kind != "b"
            or not np.array_equal(roots[self.donor], roots[self.branch])
            or not np.array_equal(x[self.donor][:, CONTEXT], x[self.branch][:, CONTEXT])
            or len(set(zip(self.donor.tolist(), self.branch.tolist(), strict=True))) != n
        ):
            raise ValueError("contrast is not a unique same-root, same-history intervention")


@dataclass(frozen=True)
class LocalFit:
    mean: np.ndarray
    scale: np.ndarray
    operator: np.ndarray
    roots: tuple[int, ...]
    rows: int
    pairs: int

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(((x - self.mean) / self.scale) @ self.operator[:-1] + self.operator[-1])


@dataclass(frozen=True)
class Domain:
    identifier: str
    # A left branch uses <=; the complementary right branch uses >.
    path: tuple[tuple[int, float, bool], ...]
    fit: LocalFit
    cv_loss: float

    def contains(self, x: np.ndarray) -> np.ndarray:
        mask = np.ones(len(x), dtype=bool)
        for coordinate, threshold, left in self.path:
            mask &= (x[:, coordinate] <= threshold) if left else (x[:, coordinate] > threshold)
        return mask


@dataclass(frozen=True)
class DiscoveredAtlas:
    domains: tuple[Domain, ...]
    paired_fit: bool
    split_records: tuple[dict[str, object], ...]

    def assignments(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2 or x.shape[1] != 23 or not np.isfinite(x).all():
            raise ValueError("finite causal feature vectors required")
        assignments = np.full(len(x), -1, dtype=int)
        for index, domain in enumerate(self.domains):
            mask = domain.contains(x)
            if np.any(assignments[mask] != -1):
                raise ValueError("overlapping discovered domains")
            assignments[mask] = index
        if np.any(assignments == -1):
            raise ValueError("incomplete discovery partition")
        # Partition membership is NOT support or receiver admission.
        return assignments

    def predict(self, x: np.ndarray) -> np.ndarray:
        assignment = self.assignments(x)
        result = np.empty((len(x), 2))
        for index, domain in enumerate(self.domains):
            mask = assignment == index
            result[mask] = domain.fit.predict(x[mask])
        return result


class _Learner:
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        roots: np.ndarray,
        pairs: PairedContrasts,
        design: DiscoveryDesign,
        paired: bool,
    ) -> None:
        self.x, self.y, self.roots, self.pairs = x, y, roots, pairs
        self.design, self.paired = design, paired
        unique = np.unique(roots)
        self.fold = np.searchsorted(unique, roots) % design.folds

    def eligible(self, mask: np.ndarray) -> bool:
        return (
            int(mask.sum()) >= self.design.minimum_rows
            and len(np.unique(self.roots[mask])) >= self.design.minimum_roots
        )

    def fit(self, mask: np.ndarray) -> LocalFit:
        x, y, root = self.x[mask], self.y[mask], self.roots[mask]
        _, counts = np.unique(root, return_counts=True)
        if not len(counts):
            raise ValueError("empty local fit")
        # Equal root weight, even when roots have different residence in a domain.
        weight = 1 / counts[np.searchsorted(np.unique(root), root)] / len(counts)
        mean = np.sum(x * weight[:, None], axis=0)
        scale = np.sqrt(np.sum((x - mean) ** 2 * weight[:, None], axis=0))
        scale = np.where(scale < 1e-12, 1, scale)
        z = np.column_stack(((x - mean) / scale, np.ones(len(x))))
        gram = z.T @ (z * weight[:, None])
        rhs = z.T @ (y * weight[:, None])
        selected = mask[self.pairs.donor] & mask[self.pairs.branch] & self.pairs.contact
        pair_count = int(selected.sum())
        operators = []
        for receiver in range(2):
            g, b = gram.copy(), rhs[:, receiver].copy()
            if self.paired and pair_count:
                donor, branch = self.pairs.donor[selected], self.pairs.branch[selected]
                dz = np.column_stack(
                    ((self.x[branch] - self.x[donor]) / scale, np.zeros(pair_count))
                )
                dy = self.y[branch, receiver] - self.y[donor, receiver]
                pair_roots = self.roots[donor]
                unique, counts = np.unique(pair_roots, return_counts=True)
                pw = 1 / counts[np.searchsorted(unique, pair_roots)] / len(unique)
                pw *= (ABSOLUTE_SCALE[receiver] / RESPONSE_SCALE[receiver]) ** 2
                g += dz.T @ (dz * pw[:, None])
                b += dz.T @ (dy * pw)
            penalty = np.eye(z.shape[1]) * self.design.ridge
            penalty[-1, -1] = 0
            operators.append(np.linalg.solve(g + penalty, b))
        return LocalFit(
            mean,
            scale,
            np.stack(operators, axis=1),
            tuple(map(int, np.unique(root))),
            len(x),
            pair_count,
        )

    def cross_predictions(self, mask: np.ndarray) -> np.ndarray | None:
        """Whole roots are excluded from fitting; rows/views never supply n."""
        predictions = np.full_like(self.y, np.nan)
        for fold in range(self.design.folds):
            train, test = mask & (self.fold != fold), mask & (self.fold == fold)
            if (
                not test.any()
                or len(np.unique(self.roots[train]))
                < self.design.minimum_roots * (self.design.folds - 1) // self.design.folds
            ):
                return None
            model = self.fit(train)
            predictions[test] = model.predict(self.x[test])
        return predictions

    def score(self, mask: np.ndarray, predictions: np.ndarray | None) -> float:
        if predictions is None:
            return float("inf")
        absolute: list[float] = []
        contrasts: list[float] = []
        for fold in range(self.design.folds):
            test = mask & (self.fold == fold)
            ids = np.flatnonzero(test)
            errors = (predictions[ids] - self.y[ids]) / ABSOLUTE_SCALE
            for root in np.unique(self.roots[ids]):
                absolute.append(float(np.mean(errors[self.roots[ids] == root] ** 2)))
            # Both absolute and paired candidates face the same contrast challenge.
            p = test[self.pairs.donor] & test[self.pairs.branch] & self.pairs.contact
            donor, branch = self.pairs.donor[p], self.pairs.branch[p]
            if len(donor):
                error = (
                    predictions[branch] - predictions[donor] - (self.y[branch] - self.y[donor])
                ) / RESPONSE_SCALE
                for root in np.unique(self.roots[donor]):
                    contrasts.append(float(np.mean(error[self.roots[donor] == root] ** 2)))
        return float(np.mean(absolute) + (np.mean(contrasts) if contrasts else 0))

    def loss(self, mask: np.ndarray) -> float:
        return self.score(mask, self.cross_predictions(mask))


def discover_local_laws(
    x: np.ndarray,
    y: np.ndarray,
    roots: np.ndarray,
    pairs: PairedContrasts,
    *,
    paired: bool,
    design: DiscoveryDesign = DiscoveryDesign(),
    progress: Callable[[dict[str, object]], None] | None = None,
) -> DiscoveredAtlas:
    """Discover a finite partition on FIT roots only. No nomination inputs exist."""
    if (
        x.ndim != 2
        or x.shape[1] != 23
        or y.shape != (len(x), 2)
        or roots.shape != (len(x),)
        or roots.dtype.kind not in "iu"
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or np.any(roots < 0)
    ):
        raise ValueError("invalid finite fit operands")
    pairs.validate(x, roots)
    learner = _Learner(x, y, roots, pairs, design, paired)
    full = np.ones(len(x), dtype=bool)
    if not learner.eligible(full):
        raise ValueError("insufficient independent fit roots or observations")
    initial_loss = learner.loss(full)
    if not np.isfinite(initial_loss):
        raise ValueError("insufficient root-blocked cross-validation coverage")
    domains = [Domain("d", (), learner.fit(full), initial_loss)]
    records: list[dict[str, object]] = []
    # Breadth first: a finite, declared search; no repeated tune-until-success.
    cursor = 0
    while cursor < len(domains) and len(domains) < design.max_leaves:
        parent = domains[cursor]
        if len(parent.path) >= design.max_depth:
            cursor += 1
            continue
        mask = parent.contains(x)
        shared_predictions = learner.cross_predictions(mask)
        best = None
        candidates = []
        for coordinate in CONTEXT:
            for threshold in np.unique(np.quantile(x[mask, coordinate], design.quantiles)):
                left = mask & (x[:, coordinate] <= threshold)
                right = mask & ~left
                if not learner.eligible(left) or not learner.eligible(right):
                    continue
                a, b = learner.loss(left), learner.loss(right)
                # Compare the same child-weighted estimand for shared vs split.
                # Parent's global score would change its weighting as residence changes.
                shared = (
                    learner.score(left, shared_predictions)
                    + learner.score(right, shared_predictions)
                ) / 2
                split = (a + b) / 2
                gain = 0.0 if shared < design.minimum_shared_loss else (shared - split) / shared
                candidate = dict(
                    coordinate=coordinate,
                    threshold=float(threshold),
                    shared_loss=shared,
                    split_loss=split,
                    gain=gain,
                )
                if np.isfinite(split) and np.isfinite(shared):
                    candidates.append(candidate)
                    key = (-gain, coordinate, float(threshold))
                    if best is None or key < best[0]:
                        best = (key, coordinate, float(threshold), left, right, a, b, gain)
        record: dict[str, object] = dict(
            domain=parent.identifier, candidates=candidates, accepted=False
        )
        if best is not None and best[-1] >= design.minimum_gain:
            _, coordinate, threshold, left, right, a, b, gain = best
            record.update(accepted=True, coordinate=coordinate, threshold=threshold, gain=gain)
            children = [
                Domain(
                    parent.identifier + suffix,
                    parent.path + ((coordinate, threshold, side),),
                    learner.fit(child),
                    loss,
                )
                for suffix, side, child, loss in (("0", True, left, a), ("1", False, right, b))
            ]
            domains.pop(cursor)
            domains.extend(children)
        else:
            cursor += 1
        records.append(record)
        if progress is not None:
            progress(
                dict(domain=parent.identifier, accepted=record["accepted"], leaves=len(domains))
            )
    return DiscoveredAtlas(tuple(domains), paired, tuple(records))
