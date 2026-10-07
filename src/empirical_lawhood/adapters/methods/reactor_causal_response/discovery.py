"""Nine training-only affine fits, frozen support and nomination operands."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
from empirical_lawhood.adapters.methods.response_formalization import fit_affine_operator
from .numerical import DIMENSIONS, LAMBDAS, FrozenFit, SupportCell, digest, stratum


@dataclass(frozen=True)
class Rows:
    roots: tuple[str, ...]
    roles: tuple[str, ...]
    clocks: np.ndarray
    actions: np.ndarray
    features: np.ndarray
    # nominal/refined, T/C/dose; no responses enter the feature interface.
    labels: np.ndarray
    delivery_valid: np.ndarray
    episodes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        n = len(self.roots)
        if (
            len(self.roles) != n
            or self.clocks.shape != (n,)
            or self.actions.shape != (n,)
            or self.features.shape != (n, 23)
            or self.labels.shape != (n, 2, 3)
            or self.delivery_valid.shape != (n,)
        ):
            raise ValueError("row axes must be row,view,receiver with explicit roles")
        if any(r not in ("fit", "nomination") for r in self.roles):
            raise ValueError("protected/calibration outcomes cannot enter discovery")
        if not np.isfinite(self.features).all() or not np.isin(self.actions, range(9)).all():
            raise ValueError("invalid causal feature/action")
        if any(
            len({role for root, role in zip(self.roots, self.roles, strict=True) if root == r}) != 1
            for r in set(self.roots)
        ):
            raise ValueError("root crosses scientific split")


def validate_development_census(rows: Rows) -> None:
    expected = {
        f"reactor-empirical-{role}-{r:03d}"
        for role, n in (("fit", 16), ("nomination", 8))
        for r in range(n)
    }
    if set(rows.roots) != expected or len(rows.episodes) != len(rows.roots):
        raise ValueError("complete development root/episode census required")
    roots, episodes = np.asarray(rows.roots), np.asarray(rows.episodes)
    for root in sorted(expected):
        for episode in ("exploration-unshifted", "exploration-shifted-one-position", "exploration-shifted-two-positions", "feed-intervention", "jacket-intervention"):
            selected = (roots == root) & (episodes == episode)
            if not np.array_equal(np.sort(rows.clocks[selected]), np.arange(2880) * 10):
                raise ValueError("missing/duplicate development callback; no deletion fit")
    if len(rows.roots) != 24 * 5 * 2880:
        raise ValueError("unassigned development rows")


def support_cells(rows: Rows, mask: np.ndarray, dimension: int) -> tuple[SupportCell, ...]:
    result = []
    strata = np.array([stratum(t) for t in rows.clocks])
    for s in range(3):
        for a in range(9):
            selected = mask & (strata == s) & (rows.actions == a)
            roots = tuple(sorted({r for r, m in zip(rows.roots, selected, strict=True) if m}))
            if not np.any(selected):
                continue
            x = rows.features[selected, :dimension]
            lo, hi = x.min(axis=0), x.max(axis=0)
            padding = np.where(hi == lo, 1e-12, 0.05 * (hi - lo))
            result.append(SupportCell(s, a, roots, tuple(lo - padding), tuple(hi + padding)))
    return tuple(result)


@dataclass(frozen=True)
class FitCall:
    family: int
    penalty: float
    rows: int
    ridge: float
    input_digest: str
    output_digest: str


@dataclass(frozen=True)
class Nomination:
    family: int
    penalty: float
    eligible: bool
    maximum: tuple[float, float] | None
    rmse: tuple[float, float] | None
    q_dev: float | None
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Discovery:
    fits: tuple[FrozenFit, ...]
    calls: tuple[FitCall, ...]
    nominations: tuple[Nomination, ...]
    selected: int | None
    rivals: tuple[int, int]
    disposition: str
    fit_mask: tuple[bool, ...]
    nomination_mask: tuple[bool, ...]


def nominate(scores: tuple[Nomination, ...]) -> tuple[int | None, tuple[int, int]]:
    if tuple((s.family, s.penalty) for s in scores) != tuple(
        (f, penalty) for f in range(3) for penalty in LAMBDAS
    ):
        raise ValueError("nine candidate results required")

    def loss(i: int) -> tuple[float, float]:
        s = scores[i]
        value = float("inf") if s.rmse is None else s.rmse[0] / 0.1 + s.rmse[1] / 0.005
        return value, -s.penalty

    eligible = [i for i, s in enumerate(scores) if s.eligible]
    selected = min(eligible, key=lambda i: (scores[i].family, *loss(i))) if eligible else None
    rivals = tuple(
        min((i for i, s in enumerate(scores) if s.family == f), key=loss) for f in (0, 1)
    )
    return selected, (rivals[0], rivals[1])


def discover(
    rows: Rows, on_fit: Callable[[FitCall], None] | None = None, *, require_full_census: bool = True
) -> Discovery:
    if require_full_census:
        validate_development_census(rows)
    fit_mask = np.array([r == "fit" for r in rows.roles])
    nom_mask = ~fit_mask
    if not fit_mask.any() or not nom_mask.any():
        raise ValueError("both declared development roles required")
    # Missing labels do not create a deletion fit.
    if not np.isfinite(rows.labels[fit_mask]).all():
        raise ValueError("missing required fit label; no deletion fitting")
    x, y = rows.features[fit_mask], rows.labels[fit_mask, 0, :2]
    training_digest = digest(
        {
            "x": x.tolist(),
            "y": y.tolist(),
            "roots": [r for r, m in zip(rows.roots, fit_mask, strict=True) if m],
        }
    )
    valid = (
        np.isfinite(rows.labels).all(axis=(1, 2))
        & (np.abs(rows.labels[:, 0] - rows.labels[:, 1]) <= (0.01, 0.0002, 0.000001)).all(axis=1)
        & rows.delivery_valid
    )
    fits, calls, scores = [], [], []
    for family, d in enumerate(DIMENSIONS):
        mean, sd = x[:, :d].mean(axis=0), x[:, :d].std(axis=0)
        sd = np.where(sd < 1e-12, 1, sd)
        z = (x[:, :d] - mean) / sd
        support = support_cells(rows, fit_mask, d)
        for penalty in LAMBDAS:
            operator = fit_affine_operator(z, y, ridge=len(x) * penalty)
            call = FitCall(
                family,
                penalty,
                len(x),
                len(x) * penalty,
                training_digest,
                digest(operator.tolist()),
            )
            calls.append(call)
            if on_fit is not None:
                on_fit(call)
            model = FrozenFit(
                family,
                penalty,
                tuple(mean),
                tuple(sd),
                tuple(map(tuple, operator)),
                support,
                training_digest,
                len(x),
            )
            fits.append(model)
            queries = rows.features[nom_mask]
            supported = bool(
                model.support_mask(queries, rows.clocks[nom_mask], rows.actions[nom_mask]).all()
            )
            reasons = []
            if not valid.all():
                reasons.append("INVALID_ASSIGNED_VIEW_OR_DELIVERY")
            if not supported:
                reasons.append("OUTSIDE_SUPPORT")
            errors = rows.labels[nom_mask, :, :2] - model.predict(queries)[:, None, :]
            maximum = np.max(np.abs(errors), axis=(0, 1)) if np.isfinite(errors).all() else None
            rmse = (
                np.sqrt(np.mean(errors[:, 0] ** 2, axis=0))
                if np.isfinite(errors[:, 0]).all()
                else None
            )
            if maximum is None or np.any(maximum > (0.25, 0.01)):
                reasons.append("MAXIMUM_ERROR")
            if rmse is None or np.any(rmse > (0.1, 0.005)):
                reasons.append("RMSE")
            scores.append(
                Nomination(
                    family,
                    penalty,
                    not reasons,
                    None if maximum is None else tuple(maximum),
                    None if rmse is None else tuple(rmse),
                    None if reasons or maximum is None else float(max(maximum / (0.25, 0.01))),
                    tuple(reasons),
                )
            )
    selected, rivals = nominate(tuple(scores))
    return Discovery(
        tuple(fits),
        tuple(calls),
        tuple(scores),
        selected,
        rivals,
        "NOMINATED" if selected is not None else "IDENTIFICATION_NOT_SUPPORTED",
        tuple(bool(m) for m in fit_mask),
        tuple(bool(m) for m in nom_mask),
    )
