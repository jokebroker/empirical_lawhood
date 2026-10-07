"Original-source qualification-labelled development using the existing finite predictor math."

from dataclasses import dataclass
from decimal import Decimal
import base64
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256
from empirical_lawhood.adapters.methods.prepared_response.models import PreparedFinitePredictor, PreparedBilinearModelSpec, fit_prepared_predictor_arrays, prepared_bilinear_model_shapes, validate_prepared_training_scalar_labels
from empirical_lawhood.adapters.simulators.prepared_response.contracts import CONTEXTS, PreparedRoot

Array = npt.NDArray[np.float64]
PROGRAMME = "causal-response-prediction"
STRUCTURES = ("direct", "mechanism-i1", "compact-4", "compact-8")
CANDIDATES = tuple((s, Decimal(r)) for s in STRUCTURES for r in ("10", "1", "0.1"))


def central_gain(values: Array) -> Array:
    """Nine-word handoff increments -> amplitude-multiplied central gain."""
    if values.shape[-3:] not in ((9, 5, 7), (9, 5, 2)):
        raise ValueError("central gain requires the declared nine-word order")
    return np.stack(
        (
            (values[..., 2, :, :2] - values[..., 1, :, :2]) / 2,
            (values[..., 4, :, :2] - values[..., 3, :, :2]) / 2,
        ),
        axis=-1,
    )


def primary_contrast(gain: Array) -> Array:
    """Input ends in parent/horizon/receiver/direction; retain both primary times."""
    if gain.shape[-4:] != (5, 5, 2, 2):
        raise ValueError("contrast changes the five-parent gain chart")
    contrast = gain[..., 1:, :, :, :] - gain[..., :1, :, :, :]
    return contrast[..., (2, 4), :, :]


@dataclass(frozen=True, slots=True)
class CausalResponsePanel:
    roots: tuple[PreparedRoot, ...]
    history: Array  # root,parent,view,16,12
    sketch: Array  # root,parent,view,8
    observed: Array  # root,parent,view,9,5,7
    transitions: Array  # root,parent,view,9,21,12

    def __post_init__(self) -> None:
        n = len(self.roots)
        if (
            not 1 <= n <= 16
            or len(set(self.roots)) != n
            or any(r.stage != 'qualification' for r in self.roots)
            or len({(r.context, r.seed_sha256) for r in self.roots}) != 1
        ):
            raise ValueError("causal response prediction development requires original exposed source qualification roots in one context")
        for name, suffix in (
            ("history", (5, 2, 16, 12)),
            ("sketch", (5, 2, 8)),
            ("observed", (5, 2, 9, 5, 7)),
            ("transitions", (5, 2, 9, 21, 12)),
        ):
            value = getattr(self, name)
            if (
                value.shape != (n, *suffix)
                or value.dtype != np.dtype("float64")
                or not np.isfinite(value).all()
            ):
                raise ValueError(f"causal response prediction unresolved or mis-shaped development {name}")
        validate_prepared_training_scalar_labels(
            self.history, self.observed, self.transitions, np.ones((n, 5, 2), dtype=bool)
        )

    def subset(self, rows: tuple[int, ...]) -> 'CausalResponsePanel':
        if len(set(rows)) != len(rows) or any(
            type(r) is not int or not 0 <= r < len(self.roots) for r in rows
        ):
            raise ValueError("causal response prediction subset changes whole-root assignment")
        return CausalResponsePanel(
            tuple(self.roots[r] for r in rows),
            *(
                getattr(self, name)[list(rows)]
                for name in ("history", "sketch", "observed", "transitions")
            ),
        )


def fit(panel: CausalResponsePanel, candidate: tuple[str, Decimal]) -> PreparedFinitePredictor:
    structure, ridge = candidate
    if candidate not in CANDIDATES:
        raise ValueError("causal response prediction fit outside the frozen finite family")
    spec = PreparedBilinearModelSpec(
        f"{PROGRAMME}.{panel.roots[0].context}.numerical-recipe",
        panel.roots[0].context,
        Decimal(16),
    )
    return fit_prepared_predictor_arrays(
        panel.history[:, :, 0].reshape(-1, 16, 12),
        panel.sketch[:, :, 0].reshape(-1, 8),
        panel.observed[:, :, 0].reshape(-1, 9, 5, 7),
        panel.transitions[:, :, 0].reshape(-1, 9, 21, 12),
        spec,
        structure=structure,
        ridge=ridge,
    )


def predict_gain(model: PreparedFinitePredictor, panel: CausalResponsePanel, view: int = 0) -> Array:
    history = panel.history[:, :, view].reshape(-1, 16, 12)
    sketch = panel.sketch[:, :, view].reshape(-1, 8) if model.structure == "mechanism-i1" else None
    return central_gain(model.predict(history, sketch)).reshape(len(panel.roots), 5, 5, 2, 2)


@dataclass(frozen=True, slots=True)
class CausalResponseCoefficientBlock(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/causal-response/causal-response-coefficient-block'
    name: str
    shape: tuple[int, ...]
    data_base64: str

    def __post_init__(self) -> None:
        if (
            self.name not in (*prepared_bilinear_model_shapes("direct"), "baseline_gain")
            or len(self.data_base64) > 4_000_000
        ):
            raise ValueError("causal response prediction unknown or unbounded coefficient block")
        value = self.array()
        if not np.isfinite(value).all():
            raise ValueError("causal response prediction coefficient block is nonfinite")

    def array(self) -> Array:
        if len(self.shape) > 4 or any(type(n) is not int or not 0 <= n <= 1000 for n in self.shape):
            raise ValueError("causal response prediction coefficient shape outside finite bounds")
        raw = base64.b64decode(self.data_base64, validate=True)
        if len(raw) != 8 * int(np.prod(self.shape)):
            raise ValueError("causal response prediction coefficient byte count differs from shape")
        return np.frombuffer(raw, dtype="<f8").reshape(self.shape)

    @classmethod
    def from_array(cls, name: str, value: Array) -> 'CausalResponseCoefficientBlock':
        return cls(
            name,
            value.shape,
            base64.b64encode(np.asarray(value, dtype="<f8").tobytes()).decode("ascii"),
        )


@dataclass(frozen=True, slots=True)
class CausalResponseContextPredictor(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/causal-response/causal-response-context-predictor'
    recipe: PreparedBilinearModelSpec
    structure: str
    ridge: Decimal
    training_roots: tuple[PreparedRoot, ...]
    blocks: tuple[CausalResponseCoefficientBlock, ...]

    def __post_init__(self) -> None:
        if (
            (self.structure, self.ridge) not in CANDIDATES
            or tuple(r.index for r in self.training_roots) != tuple(range(16))
            or any(r.stage != 'qualification' or r.context != self.recipe.context for r in self.training_roots)
            or len({r.seed_sha256 for r in self.training_roots}) != 1
        ):
            raise ValueError("causal response prediction model bank changes original source qualification training roster")
        shapes = {**prepared_bilinear_model_shapes(self.structure), "baseline_gain": (5, 5, 2, 2)}
        if tuple(b.name for b in self.blocks) != tuple(sorted(shapes)) or any(
            b.shape != shapes[b.name] for b in self.blocks
        ):
            raise ValueError("causal response prediction model bank changes the exact coefficient roster")
        self.predictor()

    def predictor(self) -> PreparedFinitePredictor:
        return PreparedFinitePredictor(
            self.recipe,
            self.structure,
            self.ridge,
            **{b.name: b.array() for b in self.blocks if b.name != "baseline_gain"},
        )

    @property
    def baseline_gain(self) -> Array:
        return next(b.array() for b in self.blocks if b.name == "baseline_gain")


@dataclass(frozen=True, slots=True)
class CausalResponseModelBank(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/causal-response/causal-response-model-bank'
    plan_sha256: str
    development_manifest_sha256: str
    development_results_sha256: str
    contexts: tuple[CausalResponseContextPredictor, ...]

    def __post_init__(self) -> None:
        for name in ("plan_sha256", "development_manifest_sha256", "development_results_sha256"):
            validate_sha256(getattr(self, name), field_name=name)
        if tuple(c.recipe.context for c in self.contexts) != CONTEXTS:
            raise ValueError("causal response prediction model bank requires both separately fitted contexts")

    @property
    def bank_id(self) -> str:
        return f"{PROGRAMME}.model-bank"
