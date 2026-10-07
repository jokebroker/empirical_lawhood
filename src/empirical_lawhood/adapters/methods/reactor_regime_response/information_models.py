"Frozen, root-separated smooth information contrasts for the regime-response study.\n\nThe input rows are made from causal preparation alone.  Assay labels are joined\nonly in the fitting/reduction role named by the root assignment.  In\nparticular the two q paths share the same prepared-t0 target for I-read and\nI-excitation; P-at-q is a different, explicitly named I-use target.\n"

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from .config import ROOTS
from .measured_panel import MeasuredContext
from .models import CoefficientRows, Fit, LAMBDAS, RBF_MULTIPLIERS, fit_affine, fit_rbf
from .probe_features import action_only_features, full_readout_features
from .records import RegimeCausalPreparation

BASELINE = ("early", "middle", "late", "prepared_t0")
INFO_ARMS = ("c_masked", "c_full", "p_masked", "p_full")


@dataclass(frozen=True)
class InformationInput:
    root: str
    context: str
    masked: tuple[float, ...]
    full: tuple[float, ...]


@dataclass(frozen=True)
class MatchedSmoothSpec:
    family: str
    penalty: float
    multiplier: float | None

    @property
    def spec_id(self) -> str:
        return f"{self.family}.l{self.penalty:.6g}.r{self.multiplier if self.multiplier is not None else 'none'}"


@dataclass(frozen=True)
class FrozenInformationRoster:
    pair_id: str
    spec: MatchedSmoothSpec
    arms: tuple[str, ...]
    fits: tuple[Fit, ...]
    support_lower: tuple[np.ndarray, ...]
    support_upper: tuple[np.ndarray, ...]
    nomination_losses: tuple[tuple[str, float], ...]

    def fitted(self, arm: str) -> Fit:
        return self.fits[self.arms.index(arm)]


@dataclass(frozen=True)
class FittedInformationSpec:
    pair_id: str
    spec: MatchedSmoothSpec
    arms: tuple[str, ...]
    fits: tuple[Fit, ...]
    support_lower: tuple[np.ndarray, ...]
    support_upper: tuple[np.ndarray, ...]


def _arms_for(pair_id: str) -> tuple[tuple[str, ...], str]:
    rosters = {
        "S_H": (("current", "history"), "baseline"),
        "S_I": (INFO_ARMS, "prepared_t0"),
        "S_Q": (("p_masked", "p_full"), "p_q"),
    }
    if pair_id not in rosters:
        raise ValueError("undeclared information pair")
    return rosters[pair_id]


def causal_inputs(preparation: RegimeCausalPreparation) -> tuple[InformationInput, ...]:
    """Read q inputs without opening the private grid or any assay labels."""
    t0 = dict((name, callback) for name, callback, _ in preparation.anchors)["prepared_t0"]
    if t0 is None:
        return ()
    arrays = preparation.arrays.unpack()
    rows = []
    for path in ("c", "p"):
        key = f"{path}_v0"
        required = tuple(f"{key}_{name}" for name in ("observations", "requests", "stages", "exposure"))
        if not all(name in arrays for name in required):
            continue
        observations, requests, stages, exposure = (arrays[name] for name in required)
        for suffix, offset in (("q", 33), ("q600", 93)):
            callback = t0 + offset
            try:
                masked = action_only_features(
                    observations, requests, stages, exposure, t0 * 10, callback
                )
                # q+600 retains the readout actually available and sealed at q.
                if suffix == "q600":
                    from .probe_features import probe_readout

                    stored = probe_readout(
                        observations, stages, exposure, t0 * 10, t0 + 33
                    )
                else:
                    stored = None
                full = full_readout_features(
                    observations, requests, stages, exposure, t0 * 10, callback, stored
                )
            except (ValueError, np.linalg.LinAlgError):
                continue
            if not np.isfinite(masked).all() or not np.isfinite(full).all():
                continue
            rows.append(
                InformationInput(
                    preparation.root,
                    f"{path}_{suffix}",
                    tuple(float(value) for value in masked),
                    tuple(float(value) for value in full),
                )
            )
    return tuple(rows)


def _role_roots(role: str) -> set[str]:
    return {root for root, assigned, _, _ in ROOTS if assigned == role}


def _smooth_specs() -> tuple[MatchedSmoothSpec, ...]:
    return (
        *(MatchedSmoothSpec("affine", penalty, None) for penalty in LAMBDAS),
        *(
            MatchedSmoothSpec("rbf", penalty, multiplier)
            for penalty in LAMBDAS
            for multiplier in RBF_MULTIPLIERS
        ),
    )


def _fit(rows: CoefficientRows, spec: MatchedSmoothSpec) -> Fit:
    if spec.family == "affine":
        return fit_affine(rows, spec.penalty)
    if spec.family == "rbf" and spec.multiplier is not None:
        return fit_rbf(rows, spec.penalty, spec.multiplier)
    raise ValueError("information model is outside the fixed smooth roster")


def _feature(
    input_row: InformationInput | MeasuredContext, arm: str
) -> tuple[float, ...] | None:
    if isinstance(input_row, InformationInput):
        return input_row.masked if arm.endswith("masked") else input_row.full
    return input_row.causal_current if arm == "current" else input_row.causal_history


def _rows(
    role: str,
    arm: str,
    target: str,
    inputs: tuple[InformationInput, ...],
    charts: tuple[MeasuredContext, ...],
) -> CoefficientRows:
    assigned = _role_roots(role)
    if any(row.root not in assigned for row in inputs) or any(
        row.root not in assigned for row in charts
    ):
        raise ValueError("information fit/nomination has a whole-root role leak")
    chart_index = {(row.root, row.context): row for row in charts}
    if len(chart_index) != len(charts):
        raise ValueError("information charts duplicate a root/context")
    if target == "baseline":
        source: tuple[InformationInput | MeasuredContext, ...] = tuple(
            row for row in charts if row.context in BASELINE
        )
    else:
        source = tuple(
            row
            for row in inputs
            if row.context in (("p_q",) if target == "p_q" else ("c_q", "p_q"))
            and row.context.startswith(arm[0])
        )
    features = []
    masses = []
    contrasts = []
    roots = []
    for row in source:
        label_context = (
            row.context
            if target in ("baseline", "p_q")
            else "prepared_t0"
        )
        chart = chart_index.get((row.root, label_context))
        feature = _feature(row, arm)
        if (
            chart is None
            or not chart.valid
            or chart.nominal_mass_kg is None
            or chart.nominal_cooling_K is None
            or feature is None
        ):
            continue
        features.append(feature)
        masses.append(chart.nominal_mass_kg[1:])
        contrasts.append(chart.nominal_cooling_K[1:])
        roots.append(row.root)
    if not roots:
        raise ValueError("information arm has no measured target with causal input")
    return CoefficientRows(
        np.asarray(features, dtype=np.float64),
        np.asarray(masses, dtype=np.float64),
        np.asarray(contrasts, dtype=np.float64),
        tuple(roots),
    )


def _match_whole_roots(rows: tuple[CoefficientRows, ...]) -> tuple[CoefficientRows, ...]:
    common = set(rows[0].roots).intersection(*(set(row.roots) for row in rows[1:]))
    if not common:
        raise ValueError("matched information arms share no complete root")
    return tuple(
        row.subset(np.asarray([i for i, root in enumerate(row.roots) if root in common]))
        for row in rows
    )


def fit_information_roster(
    pair_id: str,
    fit_inputs: tuple[InformationInput, ...],
    fit_charts: tuple[MeasuredContext, ...],
) -> tuple[tuple[FittedInformationSpec, ...], tuple[str, ...]]:
    """Fit every matched S specification on fit roots before nomination."""
    arms, target = _arms_for(pair_id)
    try:
        fit_rows = _match_whole_roots(
            tuple(_rows("fit", arm, target, fit_inputs, fit_charts) for arm in arms)
        )
    except ValueError as error:
        return (), (f"MISSING_MATCHED_INFORMATION_FIT_ROWS:{error}",)
    fitted = []
    failures = []
    for spec in _smooth_specs():
        try:
            arm_fits = tuple(_fit(rows, spec) for rows in fit_rows)
            lower = tuple(
                np.min(rows.features, axis=0)
                - 0.05 * (np.max(rows.features, axis=0) - np.min(rows.features, axis=0))
                for rows in fit_rows
            )
            upper = tuple(
                np.max(rows.features, axis=0)
                + 0.05 * (np.max(rows.features, axis=0) - np.min(rows.features, axis=0))
                for rows in fit_rows
            )
            fitted.append(FittedInformationSpec(pair_id, spec, arms, arm_fits, lower, upper))
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            failures.append(f"{spec.spec_id}:{type(error).__name__}:{error}")
    return tuple(fitted), tuple(sorted(failures))


def score_information_nomination(
    pair_id: str,
    fitted_roster: tuple[FittedInformationSpec, ...],
    nomination_inputs: tuple[InformationInput, ...],
    nomination_charts: tuple[MeasuredContext, ...],
) -> tuple[FrozenInformationRoster | None, tuple[str, ...]]:
    """Select an already fitted form by all matched nomination-arm losses."""
    arms, target = _arms_for(pair_id)
    if any(item.pair_id != pair_id or item.arms != arms for item in fitted_roster):
        raise ValueError("information roster changed its frozen arms")
    if not fitted_roster:
        return None, ("NO_FITTED_INFORMATION_SPECIFICATION",)
    try:
        nomination_rows = _match_whole_roots(
            tuple(
                _rows("nomination", arm, target, nomination_inputs, nomination_charts)
                for arm in arms
            )
        )
    except ValueError as error:
        return None, (f"MISSING_MATCHED_INFORMATION_NOMINATION_ROWS:{error}",)
    candidates = []
    failures = []
    for item in fitted_roster:
        try:
            losses = tuple(
                rows.root_losses(model.predict(rows.features))
                for rows, model in zip(nomination_rows, item.fits, strict=True)
            )
            if not all(losses) or not all(
                isfinite(value) for arm_losses in losses for value in arm_losses.values()
            ):
                raise ValueError("nonfinite or empty information nomination loss")
            mean_loss = float(np.mean(tuple(float(np.mean(tuple(x.values()))) for x in losses)))
            candidates.append((mean_loss, item, losses))
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            failures.append(f"{item.spec.spec_id}:{type(error).__name__}:{error}")
    if not candidates:
        return None, tuple(sorted(failures))
    minimum = min(row[0] for row in candidates)
    tied = [row for row in candidates if row[0] <= 1.01 * minimum]
    _, selected, losses = min(
        tied,
        key=lambda row: (
            sum(len(model.operator) for model in row[1].fits),
            -row[1].spec.penalty,
            row[1].spec.spec_id,
        ),
    )
    return (
        FrozenInformationRoster(
            pair_id,
            selected.spec,
            arms,
            selected.fits,
            selected.support_lower,
            selected.support_upper,
            tuple(
                (arm, float(np.mean(tuple(arm_losses.values()))))
                for arm, arm_losses in zip(arms, losses, strict=True)
            ),
        ),
        tuple(sorted(failures)),
    )


def nominate_information_pair(
    pair_id: str,
    fit_inputs: tuple[InformationInput, ...],
    fit_charts: tuple[MeasuredContext, ...],
    nomination_inputs: tuple[InformationInput, ...],
    nomination_charts: tuple[MeasuredContext, ...],
) -> tuple[FrozenInformationRoster | None, tuple[str, ...]]:
    """Convenience path; production persists the fit stage before nomination."""
    fit_roots = {row.root for row in fit_inputs} | {row.root for row in fit_charts}
    nomination_roots = {row.root for row in nomination_inputs} | {
        row.root for row in nomination_charts
    }
    if fit_roots & nomination_roots:
        raise ValueError("nomination root entered information fit")
    fitted, failures = fit_information_roster(pair_id, fit_inputs, fit_charts)
    selected, nomination_failures = score_information_nomination(
        pair_id, fitted, nomination_inputs, nomination_charts
    )
    return selected, tuple(sorted((*failures, *nomination_failures)))
