"Whole-root prediction and one simultaneous fresh calibration score per assigned root.\n\nThese operands do not finalize laws, choose parents or certify admission. Calibration\ncannot change a coefficient, a dependent refinement scale, a policy roster or an action choice.\n"

from dataclasses import dataclass
from decimal import Decimal

import numpy as np

from empirical_lawhood.adapters.simulators.prepared_response.contracts import PreparedRoot
from empirical_lawhood.adapters.methods.response_formalization import affine_prediction, fit_affine_operator

from .models import Array, RIDGES, STRUCTURES, PreparedTrainingPanel, PreparedFittedModel, PreparedAffineModelSpec, fit_prepared_model
from .statistics import conformal_order_index


def _freeze(value: Array) -> Array:
    return np.frombuffer(np.asarray(value, dtype="<f8").tobytes(), dtype="<f8").reshape(value.shape)


def prepared_root_folds(roots: tuple[PreparedRoot, ...]) -> tuple[int, ...]:
    """Four deterministic folds; the index is source-assigned before outcomes."""
    if (
        len(roots) < 4
        or len(set(roots)) != len(roots)
        or any(r.stage != 'development' for r in roots)
        or len({(r.context, r.seed_sha256) for r in roots}) != 1
    ):
        raise ValueError("four-fold prediction requires at least four distinct whole dependent refinement roots")
    ordered = sorted(roots, key=lambda r: r.index)
    assignments = {root: rank % 4 for rank, root in enumerate(ordered)}
    return tuple(assignments[root] for root in roots)


@dataclass(frozen=True, slots=True)
class PreparedCrossfit:
    roots: tuple[PreparedRoot, ...]
    folds: tuple[int, ...]
    models: tuple[PreparedFittedModel, ...]
    predictions: Array

    def __post_init__(self) -> None:
        if self.folds != prepared_root_folds(self.roots) or len(self.models) != 4:
            raise ValueError("prepared crossfit lost its four whole-root folds")
        for fold, model in enumerate(self.models):
            expected = tuple(
                root for root, f in zip(self.roots, self.folds, strict=True) if f != fold
            )
            if model.assigned_training_roots != expected:
                raise ValueError("held-out root or an altered training census enters a dependent refinement fit")
        if (
            self.predictions.shape != (len(self.roots), 5, 2, 9, 5, 7)
            or self.predictions.dtype != np.dtype("float64")
            or any(m.spec != self.models[0].spec for m in self.models)
        ):
            raise ValueError("prepared crossfit changes its output or specification roster")
        object.__setattr__(self, "predictions", _freeze(self.predictions))


def crossfit_prepared_model(
    panel: PreparedTrainingPanel,
    spec: PreparedAffineModelSpec,
    *,
    structure: str,
    ridge: Decimal,
) -> PreparedCrossfit:
    folds = prepared_root_folds(panel.roots)
    prediction = np.full(panel.observed.shape, np.nan)
    models: list[PreparedFittedModel] = []
    for fold in range(4):
        train = tuple(i for i, f in enumerate(folds) if f != fold)
        model = fit_prepared_model(panel.subset(train), spec, structure=structure, ridge=ridge)
        models.append(model)
        for i, f in enumerate(folds):
            if f != fold:
                continue
            valid = panel.valid[i].copy()
            if structure == "mechanism-i1":
                valid &= np.isfinite(panel.sketch[i]).all(axis=-1)
            if np.any(valid):
                prediction[i, valid] = model.predict(
                    panel.history[i, valid],
                    panel.sketch[i, valid] if structure == "mechanism-i1" else None,
                )
    return PreparedCrossfit(panel.roots, folds, tuple(models), prediction)


def prepared_joint_root_losses(observed: Array, predicted: Array, normalizers: Array) -> Array:
    """Maximum standardized error, all branches/views/words/readouts/channels.

    Normalizers express prediction loss units, never permission to trade one
    scientific gate against another. Missing required rows have infinite loss.
    """
    if (
        observed.shape != predicted.shape
        or observed.ndim != 6
        or observed.shape[1:] != (5, 2, 9, 5, 7)
        or normalizers.shape != (7,)
        or not np.isfinite(normalizers).all()
        or np.any(normalizers <= 0)
    ):
        raise ValueError("dependent refinement joint loss changes its full output roster or native scaling")
    residual = abs(observed - predicted) / normalizers
    residual[~np.isfinite(residual)] = np.inf
    return _freeze(np.max(residual, axis=(1, 2, 3, 4, 5)))


@dataclass(frozen=True, slots=True)
class PreparedConditionalScale:
    """Deployable uncertainty coefficients, without training roots or errors."""

    positive_floor: float
    instrument_tier: str
    feature_mean: Array
    feature_scale: Array
    logvariance_operator: Array

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.positive_floor)
            or self.positive_floor <= 0
            or self.instrument_tier not in ("I0", "I1")
        ):
            raise ValueError("conditional scale changes its positive floor or input tier")
        count = 200 if self.instrument_tier == "I1" else 192
        for name, shape in (
            ("feature_mean", (count,)),
            ("feature_scale", (count,)),
            ("logvariance_operator", (count + 1, 315)),
        ):
            value = getattr(self, name)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError("conditional scale changes its fitted scalar chart")
            object.__setattr__(self, name, _freeze(value))
        if np.any(self.feature_scale <= 0):
            raise ValueError("conditional scale has a nonpositive feature standardization")

    def predict(self, history: Array, sketch: Array | None = None) -> Array:
        if history.ndim != 3 or history.shape[1:] != (16, 12) or not np.isfinite(history).all():
            raise ValueError("conditional scales require only causal sixteen-row I0 history")
        features = history.reshape(len(history), 192)
        if self.instrument_tier == "I1":
            if sketch is None or sketch.shape != (len(history), 8) or not np.isfinite(sketch).all():
                raise ValueError("I1 conditional scale requires its measured handoff sketch")
            features = np.column_stack((features, sketch))
        elif sketch is not None:
            raise ValueError("I0 conditional scale cannot receive an I1 sketch")
        with np.errstate(over="ignore", invalid="ignore"):
            log_variance = affine_prediction(
                self.logvariance_operator, (features - self.feature_mean) / self.feature_scale
            )
            values = np.maximum(np.exp(0.5 * log_variance), self.positive_floor)
        values[~np.isfinite(values)] = np.nan
        return _freeze(values.reshape(len(history), 9, 5, 7))


@dataclass(frozen=True, slots=True)
class PreparedResidualScale(PreparedConditionalScale):
    training_roots: tuple[PreparedRoot, ...]
    reference_rms: Array
    valid_rows: int

    def __post_init__(self) -> None:
        PreparedConditionalScale.__post_init__(self)
        if (
            type(self.training_roots) is not tuple
            or not self.training_roots
            or len(set(self.training_roots)) != len(self.training_roots)
            or any(r.stage != 'development' for r in self.training_roots)
            or len({(r.context, r.seed_sha256) for r in self.training_roots}) != 1
            or self.reference_rms.shape != (9, 5, 7)
            or not np.isfinite(self.reference_rms).all()
            or np.any(self.reference_rms < self.positive_floor)
            or type(self.valid_rows) is not int
            or not 1 <= self.valid_rows <= 10 * len(self.training_roots)
        ):
            raise ValueError("residual scale requires positive finite dependent refinement-only whole-root evidence")
        object.__setattr__(self, "reference_rms", _freeze(self.reference_rms))

    def deployment_scale(self) -> PreparedConditionalScale:
        return PreparedConditionalScale(
            self.positive_floor,
            self.instrument_tier,
            self.feature_mean,
            self.feature_scale,
            self.logvariance_operator,
        )


def fit_prepared_residual_scale(
    panel: PreparedTrainingPanel, crossfit: PreparedCrossfit, *, positive_floor: float
) -> PreparedResidualScale:
    if not np.isfinite(positive_floor) or positive_floor <= 0:
        raise ValueError("conditional scale needs a predeclared strictly positive floor")
    if panel.roots != crossfit.roots:
        raise ValueError("residual scale cannot pair another training population")
    residual = panel.observed - crossfit.predictions
    complete = panel.valid & np.all(np.isfinite(residual), axis=(3, 4, 5))
    if not np.any(complete):
        raise ValueError("no held-out complete residual rows are available for dependent refinement scaling")
    scales = np.maximum(np.sqrt(np.mean(residual[complete] ** 2, axis=0)), positive_floor)
    features = panel.history[complete].reshape(int(complete.sum()), 192)
    tier = "I1" if crossfit.models[0].structure == "mechanism-i1" else "I0"
    if tier == "I1":
        features = np.column_stack((features, panel.sketch[complete]))
    mean = features.mean(axis=0)
    feature_scale = np.maximum(features.std(axis=0), 1e-12)
    targets = np.log(np.maximum(residual[complete] ** 2, positive_floor**2))
    # Fixed mean-loss ridge 1: the shared unnormalized solver receives n rows.
    # No fresh calibration row estimates this operator or a standardization coefficient.
    operator = fit_affine_operator(
        (features - mean) / feature_scale,
        targets.reshape(len(features), 315),
        ridge=float(len(features)),
    )
    return PreparedResidualScale(
        positive_floor,
        tier,
        mean,
        feature_scale,
        operator,
        panel.roots,
        scales,
        int(complete.sum()),
    )


@dataclass(frozen=True, slots=True)
class PreparedCalibration:
    roots: tuple[PreparedRoot, ...]
    scale: PreparedResidualScale
    policy_ids: tuple[str, ...]
    scores: Array
    quantile: float
    order_index: int

    def __post_init__(self) -> None:
        if (
            len(self.roots) != 96
            or len(set(self.roots)) != 96
            or any(r.stage != 'calibration' for r in self.roots)
            or len({(r.context, r.seed_sha256) for r in self.roots}) != 1
            or self.roots[0].context != self.scale.training_roots[0].context
            or not self.policy_ids
            or len(set(self.policy_ids)) != len(self.policy_ids)
            or self.scores.shape != (96,)
            or np.isnan(self.scores).any()
            or np.any(self.scores < 0)
            or self.order_index != conformal_order_index(96)
            or self.quantile != float(np.sort(self.scores)[self.order_index - 1])
        ):
            raise ValueError(
                "calibration changes its fresh calibration root/policy census or joint order statistic"
            )
        object.__setattr__(self, "scores", _freeze(self.scores))

    def halfwidth(self, history: Array, sketch: Array | None = None) -> Array:
        return _freeze(self.quantile * self.scale.predict(history, sketch))


def calibrate_prepared_joint_box(
    *,
    roots: tuple[PreparedRoot, ...],
    scale: PreparedResidualScale,
    policy_ids: tuple[str, ...],
    observed: Array,
    predicted: Array,
    history: Array,
    sketch: Array | None = None,
) -> PreparedCalibration:
    expected = (96, len(policy_ids), 2, 9, 5, 7)
    if observed.shape != expected or predicted.shape != expected:
        raise ValueError("fresh calibration must retain every declared policy/view/action/readout/channel")
    if history.shape != (96, len(policy_ids), 2, 16, 12):
        raise ValueError("fresh calibration conditional scales require the exact causal handoff roster")
    flat_history = history.reshape(-1, 16, 12)
    valid = np.isfinite(flat_history).all(axis=(1, 2))
    flat_sketch = None
    if scale.instrument_tier == "I1":
        if sketch is None or sketch.shape != (96, len(policy_ids), 2, 8):
            raise ValueError("fresh calibration I1 scale is missing its declared sketch roster")
        flat_sketch = sketch.reshape(-1, 8)
        valid &= np.isfinite(flat_sketch).all(axis=-1)
    elif sketch is not None:
        raise ValueError("fresh calibration I0 scale cannot receive an I1 sketch")
    scales = np.full((len(flat_history), 9, 5, 7), np.nan)
    if np.any(valid):
        scales[valid] = scale.predict(
            flat_history[valid], None if flat_sketch is None else flat_sketch[valid]
        )
    errors = abs(observed - predicted) / scales.reshape(expected)
    errors[~np.isfinite(errors)] = np.inf
    scores = np.max(errors, axis=(1, 2, 3, 4, 5))
    order = conformal_order_index(96)
    return PreparedCalibration(
        roots, scale, policy_ids, scores, float(np.sort(scores)[order - 1]), order
    )


@dataclass(frozen=True, slots=True)
class PreparedNestedFamilyFit:
    """One structure's tuned outer predictions and its all-dependent refinement final refit.

    This is a development nominee, not qualification. Instrument/predictor
    cost and the scientific owner's accuracy/sharpness/support screens still
    determine whether any nominee can be the primary.
    """

    outer: PreparedCrossfit
    final: PreparedFittedModel
    normalizers: Array
    inner_rankings: tuple[tuple[tuple[int, float], ...], ...]
    selected_inner_crossfits: tuple[PreparedCrossfit, ...]
    candidate_fit_count: int

    def __post_init__(self) -> None:
        if (
            self.final.assigned_training_roots != self.outer.roots
            or any(m.structure != self.final.structure for m in self.outer.models)
            or self.normalizers.shape != (7,)
            or not np.isfinite(self.normalizers).all()
            or np.any(self.normalizers <= 0)
            or len(self.inner_rankings) != 5
            or len(self.selected_inner_crossfits) != 5
            or any(len(ranks) != 3 for ranks in self.inner_rankings)
            or self.candidate_fit_count != 65
        ):
            raise ValueError("nested dependent refinement selection changes its whole-root/three-ridge fit census")
        for ranks in self.inner_rankings:
            if any(
                type(missing) is not int or missing < 0 or not np.isfinite(loss) or loss < 0
                for missing, loss in ranks
            ):
                raise ValueError("nested dependent refinement ranking lost missingness or its finite loss sum")
        for ranks, model in zip(self.inner_rankings, (*self.outer.models, self.final), strict=True):
            chosen = min(range(3), key=lambda i: (*ranks[i], i))
            if model.ridge != RIDGES[chosen]:
                raise ValueError("nested dependent refinement model disagrees with its frozen ranking/tie order")
        for inner, model in zip(
            self.selected_inner_crossfits, (*self.outer.models, self.final), strict=True
        ):
            if inner.roots != model.assigned_training_roots or any(
                m.structure != model.structure or m.ridge != model.ridge for m in inner.models
            ):
                raise ValueError(
                    "selected inner prediction is detached from its outer training scope"
                )
        object.__setattr__(self, "normalizers", _freeze(self.normalizers))


def fit_prepared_nested_family(
    panel: PreparedTrainingPanel,
    spec: PreparedAffineModelSpec,
    *,
    structure: str,
    normalizers: Array,
) -> PreparedNestedFamilyFit:
    "Four outer folds, four inner folds, three ridges; all views stay together.\n\n    Inner ranking minimizes unresolved root count, then the finite loss sum\n    divided by the complete assigned-root count, then the frozen ridge order.\n    Infinite errors remain visible in the returned outer predictions/losses;\n    no calibration or qualification is inferred from this tuning criterion.\n    "
    if structure not in STRUCTURES or len(panel.roots) < 8:
        raise ValueError("nested finite-family selection requires its declared structure and folds")
    folds = prepared_root_folds(panel.roots)
    prediction = np.full(panel.observed.shape, np.nan)
    models: list[PreparedFittedModel] = []
    rankings: list[tuple[tuple[int, float], ...]] = []
    selected_inner: list[PreparedCrossfit] = []
    for outer in (*range(4), None):
        train = (
            panel
            if outer is None
            else panel.subset(tuple(i for i, f in enumerate(folds) if f != outer))
        )
        ranks: list[tuple[int, float]] = []
        inner_results: list[PreparedCrossfit] = []
        for ridge in RIDGES:
            inner = crossfit_prepared_model(train, spec, structure=structure, ridge=ridge)
            inner_results.append(inner)
            losses = prepared_joint_root_losses(train.observed, inner.predictions, normalizers)
            finite = np.isfinite(losses)
            ranks.append((int((~finite).sum()), float(losses[finite].sum() / len(train.roots))))
        rankings.append(tuple(ranks))
        chosen = min(range(3), key=lambda i: (*ranks[i], i))
        selected_inner.append(inner_results[chosen])
        model = fit_prepared_model(train, spec, structure=structure, ridge=RIDGES[chosen])
        models.append(model)
        if outer is not None:
            for i, fold in enumerate(folds):
                if fold == outer and np.any(panel.valid[i]):
                    valid = panel.valid[i].copy()
                    if structure == "mechanism-i1":
                        valid &= np.isfinite(panel.sketch[i]).all(axis=-1)
                    if not np.any(valid):
                        continue
                    prediction[i, valid] = model.predict(
                        panel.history[i, valid],
                        panel.sketch[i, valid] if structure == "mechanism-i1" else None,
                    )
    return PreparedNestedFamilyFit(
        PreparedCrossfit(panel.roots, folds, tuple(models[:4]), prediction),
        models[4],
        normalizers,
        tuple(rankings),
        tuple(selected_inner),
        65,
    )


@dataclass(frozen=True, slots=True)
class PreparedNestedScales:
    outer: tuple[PreparedResidualScale, ...]
    final: PreparedResidualScale
    out_of_fold_scales: Array

    def __post_init__(self) -> None:
        roots = self.final.training_roots
        folds = prepared_root_folds(roots)
        if len(self.outer) != 4 or self.out_of_fold_scales.shape != (len(roots), 5, 2, 9, 5, 7):
            raise ValueError("nested scales lost their four-fold scalar roster")
        for fold, scale in enumerate(self.outer):
            if scale.training_roots != tuple(
                root for root, f in zip(roots, folds, strict=True) if f != fold
            ):
                raise ValueError("outer scale has access to its held-out root")
        object.__setattr__(self, "out_of_fold_scales", _freeze(self.out_of_fold_scales))


def fit_prepared_nested_scales(
    panel: PreparedTrainingPanel, nested: PreparedNestedFamilyFit, *, positive_floor: float
) -> PreparedNestedScales:
    "dependent refinement adequacy labels cannot use a scale trained on their own error."
    if panel.roots != nested.outer.roots:
        raise ValueError("nested scales require the exact dependent refinement model population")
    outputs = np.full(panel.observed.shape, np.nan)
    scales: list[PreparedResidualScale] = []
    for fold in range(4):
        train = panel.subset(tuple(i for i, f in enumerate(nested.outer.folds) if f != fold))
        scale = fit_prepared_residual_scale(
            train, nested.selected_inner_crossfits[fold], positive_floor=positive_floor
        )
        scales.append(scale)
        for i, f in enumerate(nested.outer.folds):
            if f != fold:
                continue
            valid = panel.valid[i].copy()
            if scale.instrument_tier == "I1":
                valid &= np.isfinite(panel.sketch[i]).all(axis=-1)
            if np.any(valid):
                outputs[i, valid] = scale.predict(
                    panel.history[i, valid],
                    panel.sketch[i, valid] if scale.instrument_tier == "I1" else None,
                )
    final = fit_prepared_residual_scale(panel, nested.outer, positive_floor=positive_floor)
    return PreparedNestedScales(tuple(scales), final, outputs)
