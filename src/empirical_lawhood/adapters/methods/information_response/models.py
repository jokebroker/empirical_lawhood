"""information response prediction matched information masks and small frozen signed-gain predictors.

Numerical fitting and mechanism propagation remain with their existing owners.
Only information selection, parent centering and the finite tournament are new.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

import numpy as np

from empirical_lawhood.adapters.methods.causal_response.models import Array as Array, CausalResponseCoefficientBlock, CausalResponsePanel, central_gain as central_gain, primary_contrast as primary_contrast
from empirical_lawhood.adapters.methods.prepared_response.models import PreparedBilinearModelSpec, prepared_mechanism_mean
from empirical_lawhood.adapters.methods.response_formalization import (
    affine_prediction,
    fit_affine_operator,
)
from empirical_lawhood.adapters.simulators.prepared_response.contracts import CONTEXTS, PreparedRoot
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256

PROGRAMME = "information-response-prediction"
MODEL_IDS = ("training-mean", "snapshot", "history", "snapshot-with-mechanism", "history-with-mechanism", "mechanism", "mechanism-with-residual")
FITTED_IDS = ("snapshot", "history", "snapshot-with-mechanism", "history-with-mechanism", "mechanism-with-residual")
SNAPSHOT_COLUMNS = (0, 1, 2, 3, 4, 5, 9, 10)
RIDGES = (Decimal(10), Decimal(1), Decimal("0.1"))
FEATURE_COUNTS = {"snapshot": 8, "history": 128, "snapshot-with-mechanism": 16, "history-with-mechanism": 136, "mechanism-with-residual": 136}


def features(history: Array, sketch: Array, arm: str) -> Array:
    if history.ndim != 3 or history.shape[1:] != (16, 12) or sketch.shape != (len(history), 8):
        raise ValueError("information response prediction feature input changes the measured scalar chart")
    if arm not in FITTED_IDS or not np.isfinite(history).all() or not np.isfinite(sketch).all():
        raise ValueError("information response prediction features are unavailable or the arm is undeclared")
    value = history[:, :, SNAPSHOT_COLUMNS]
    value = value[:, -1] if arm in ("snapshot", "snapshot-with-mechanism") else value.reshape(len(history), -1)
    return np.concatenate((value, sketch), axis=1) if arm in ("snapshot-with-mechanism", "history-with-mechanism", "mechanism-with-residual") else value


def mechanism_gain(history: Array, sketch: Array) -> Array:
    return central_gain(prepared_mechanism_mean(history, sketch, Decimal(16)))


@dataclass(frozen=True, slots=True)
class InformationResponsePredictor(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/information-response/information-response-predictor'
    arm: str
    ridge: Decimal | None
    # Reuse the existing bounded float coefficient codec with unchanged meaning.
    blocks: tuple[CausalResponseCoefficientBlock, ...]

    def __post_init__(self) -> None:
        if self.arm not in MODEL_IDS or (self.ridge in RIDGES) != (self.arm in FITTED_IDS):
            raise ValueError("information response prediction predictor changes its finite family")
        if self.arm not in FITTED_IDS and self.ridge is not None:
            raise ValueError("information response prediction nonfitted arm has a ridge")
        shapes: dict[str, tuple[int, ...]] = (
            {} if self.arm == "mechanism" else {"baseline_gain": (5, 5, 2, 2)}
        )
        if self.arm in FITTED_IDS:
            n = FEATURE_COUNTS[self.arm]
            shapes.update(feature_mean=(5, n), feature_scale=(n,), output_operator=(n + 1, 20))
        if tuple(b.name for b in self.blocks) != tuple(sorted(shapes)) or any(
            b.shape != shapes[b.name] for b in self.blocks
        ):
            raise ValueError("information response prediction coefficient roster/shape differs")
        if self.arm in FITTED_IDS and (self.arrays()["feature_scale"] <= 0).any():
            raise ValueError("information response prediction scale must be positive")

    def arrays(self) -> dict[str, Array]:
        return {b.name: b.array() for b in self.blocks}

    def predict(self, history: Array, sketch: Array, parents: tuple[int, ...]) -> Array:
        if len(parents) != len(history) or any(
            type(p) is not int or not 0 <= p < 5 for p in parents
        ):
            raise ValueError("information response prediction prediction changes parent labels")
        prior = mechanism_gain(history, sketch) if self.arm in ("mechanism", "mechanism-with-residual") else 0.0
        if self.arm == "mechanism":
            return np.asarray(prior, dtype=np.float64)
        arrays = self.arrays()
        result = arrays["baseline_gain"][list(parents)].copy()
        if self.arm in FITTED_IDS:
            x = features(history, sketch, self.arm) - arrays["feature_mean"][list(parents)]
            x /= arrays["feature_scale"]
            result += affine_prediction(arrays["output_operator"], x).reshape(-1, 5, 2, 2)
        return np.asarray(result + prior, dtype=np.float64)


def fit(panel: CausalResponsePanel, arm: str, ridge: Decimal | None) -> InformationResponsePredictor:
    n = len(panel.roots)
    if arm not in MODEL_IDS:
        raise ValueError("information response prediction fit outside declared arms")
    if arm == "mechanism":
        return InformationResponsePredictor(arm, ridge, ())
    history = panel.history[:, :, 0].reshape(-1, 16, 12)
    sketch = panel.sketch[:, :, 0].reshape(-1, 8)
    y = central_gain(panel.observed[:, :, 0])
    if arm == "mechanism-with-residual":
        y = y - mechanism_gain(history, sketch).reshape(n, 5, 5, 2, 2)
    arrays = {"baseline_gain": y.mean(axis=0)}
    if arm in FITTED_IDS:
        x = features(history, sketch, arm).reshape(n, 5, -1)
        means = x.mean(axis=0)
        centered = (x - means).reshape(n * 5, -1)
        scales = np.maximum(centered.std(axis=0), 1e-12)
        arrays.update(
            feature_mean=means,
            feature_scale=scales,
            output_operator=fit_affine_operator(
                centered / scales,
                (y - arrays["baseline_gain"]).reshape(n * 5, 20),
                float(ridge) if ridge is not None else 0.0,
            ),
        )
    return InformationResponsePredictor(
        arm,
        ridge,
        tuple(CausalResponseCoefficientBlock.from_array(k, v) for k, v in sorted(arrays.items())),
    )


@dataclass(frozen=True, slots=True)
class InformationResponseContextPredictor(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/information-response/information-response-context-predictor'
    recipe: PreparedBilinearModelSpec
    training_roots: tuple[PreparedRoot, ...]
    models: tuple[InformationResponsePredictor, ...]

    def __post_init__(self) -> None:
        if (
            tuple(r.index for r in self.training_roots) != tuple(range(16))
            or any(r.stage != 'qualification' or r.context != self.recipe.context for r in self.training_roots)
            or len({r.seed_sha256 for r in self.training_roots}) != 1
            or tuple(m.arm for m in self.models) != MODEL_IDS
        ):
            raise ValueError("information response prediction changes the original training roots or frozen tournament")

    def predict(self, history: Array, sketch: Array, parents: tuple[int, ...]) -> Array:
        return np.stack([m.predict(history, sketch, parents) for m in self.models], axis=1)


@dataclass(frozen=True, slots=True)
class InformationResponseModelBank(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/information-response/information-response-model-bank'
    plan_sha256: str
    development_manifest_sha256: str
    development_results_sha256: str
    contexts: tuple[InformationResponseContextPredictor, ...]

    def __post_init__(self) -> None:
        for name in ("plan_sha256", "development_manifest_sha256", "development_results_sha256"):
            validate_sha256(getattr(self, name), field_name=name)
        if tuple(c.recipe.context for c in self.contexts) != CONTEXTS:
            raise ValueError("information response prediction bank requires both separately fitted contexts")

    @property
    def bank_id(self) -> str:
        return f"{PROGRAMME}.model-bank"
