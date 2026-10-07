"""Finite predictor mathematics and the original dependent refinement-only lineage wrapper.

The compact recurrences are internal prediction mechanisms. They produce
finite nine-word outputs and never claim controlled-map reachability.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

import numpy as np
import numpy.typing as npt
from scipy.linalg import expm

from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id
from empirical_lawhood.adapters.methods.response_formalization import (
    affine_prediction,
    fit_affine_operator,
)
from empirical_lawhood.adapters.simulators.prepared_response.contracts import CONTEXTS, READOUTS, PreparedRoot, prepared_words

Array = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
STRUCTURES = ("direct", "compact-4", "compact-8", "mechanism-i1")
RIDGES = (Decimal("0.1"), Decimal(1), Decimal(10))
CHANNELS = (
    "receiver-m1-increment",
    "receiver-m2-increment",
    "maximum-off-force-x-hs",
    "maximum-relative-y",
    "maximum-passive-transfer-difference",
    "signed-force-work",
    "absolute-force-work",
)
STATE_CHANNELS = (0, 9, 1, 10)
LATENT_CHANNELS = (2, 3, 4, 5, 6, 7, 8, 11)


def _freeze(value: Array) -> Array:
    return np.frombuffer(np.asarray(value, dtype="<f8").tobytes(), dtype="<f8").reshape(value.shape)


@dataclass(frozen=True, slots=True)
class PreparedAffineModelSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/prepared-response/prepared-affine-model-spec'
    COMPACT_RULE: ClassVar[str] = (
        "DEVELOPMENT_CAUSAL_SCALAR_TRANSITIONS_16_TICK_AFFINE_20_STEP_CLOSED_FORECAST"
    )
    spec_id: str
    context: str
    amplitude: Decimal
    structures: tuple[str, ...] = STRUCTURES
    ridges: tuple[Decimal, ...] = RIDGES
    readouts: tuple[int, ...] = READOUTS
    output_channels: tuple[str, ...] = CHANNELS
    standard_deviation_floor: Decimal = Decimal("0.000000000001")
    feature_rule: str = "I0_16_BY_12_PLUS_I1_HANDOFF_8_ONLY_FOR_MECHANISM"
    compact_rule: str = "DEVELOPMENT_CAUSAL_SCALAR_TRANSITIONS_16_TICK_AFFINE_20_STEP_CLOSED_FORECAST"
    latent_rule: str = "TRAIN_ONLY_CENTERED_SCALED_OTHER_EIGHT_CHANNEL_PCA_FIRST_FOUR"
    mechanism_rule: str = "FROZEN_TWO_PORT_FORCE_HESSIAN_UNIT_MASS_FRICTION_MEAN_PLUS_DEVELOPMENT_RESIDUAL"
    ridge_rule: str = "UNNORMALIZED_SUM_SQUARED_ERROR_UNPENALIZED_INTERCEPT"

    def __post_init__(self) -> None:
        validate_stable_id(self.spec_id, field_name="spec_id")
        if (
            self.context not in CONTEXTS
            or not isinstance(self.amplitude, Decimal)
            or self.amplitude not in (8, 16)
            or self.structures != STRUCTURES
            or self.ridges != RIDGES
            or any(not isinstance(x, Decimal) for x in self.ridges)
            or self.readouts != READOUTS
            or any(type(x) is not int for x in self.readouts)
            or self.output_channels != CHANNELS
            or not isinstance(self.standard_deviation_floor, Decimal)
            or self.standard_deviation_floor != Decimal("0.000000000001")
            or self.feature_rule != "I0_16_BY_12_PLUS_I1_HANDOFF_8_ONLY_FOR_MECHANISM"
            or self.compact_rule != self.COMPACT_RULE
            or self.latent_rule != "TRAIN_ONLY_CENTERED_SCALED_OTHER_EIGHT_CHANNEL_PCA_FIRST_FOUR"
            or self.mechanism_rule
            != "FROZEN_TWO_PORT_FORCE_HESSIAN_UNIT_MASS_FRICTION_MEAN_PLUS_DEVELOPMENT_RESIDUAL"
            or self.ridge_rule != "UNNORMALIZED_SUM_SQUARED_ERROR_UNPENALIZED_INTERCEPT"
        ):
            raise ValueError("prepared model specification changes its finite family or inputs")


@dataclass(frozen=True, slots=True)
class PreparedBilinearModelSpec(PreparedAffineModelSpec):
    """Same finite family, with state-dependent compact action gains."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/prepared-response/prepared-bilinear-model-spec'
    COMPACT_RULE: ClassVar[str] = (
        "DEVELOPMENT_CAUSAL_SCALAR_TRANSITIONS_16_TICK_BILINEAR_20_STEP_CLOSED_FORECAST"
    )
    compact_rule: str = COMPACT_RULE


def validate_prepared_training_scalar_labels(
    history: Array, observed: Array, transitions: Array, valid: BoolArray
) -> None:
    """Keep scalar trajectory labels attached to their own measured handoff."""
    if not np.allclose(
        transitions[valid, :, 0, :],
        history[valid, -1, :][:, None, :],
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError("dependent refinement transition labels do not start at their own measured handoff")
    expected = (
        transitions[valid][:, :, (4, 8, 12, 16, 20), :][..., (0, 9)]
        - history[valid, -1, :][:, None, None, (0, 9)]
    )
    if not np.allclose(observed[valid][..., :2], expected, rtol=0, atol=1e-12):
        raise ValueError("dependent refinement receiver labels disagree with the causal scalar trajectory")


@dataclass(frozen=True, slots=True)
class PreparedTrainingPanel:
    """One-context scalar panel, indexed root/parent/view/word/time/channel.

    Invalid complete-menu rows remain assigned. Only complete valid rows enter
    a numerical fit; held-out loss and calibration retain invalid rows as inf.
    This in-process operand grants no acquisition or outcome-read authority.
    """

    roots: tuple[PreparedRoot, ...]
    history: Array  # root, 5 parents, 2 views, 16 past samples, 12 channels
    sketch: Array  # root, 5, 2, 8
    observed: Array  # root, 5, 2, 9 words, 5 readouts, 7 channels
    transitions: Array  # root, 5, 2, 9, 21 scalar observations, 12 channels
    valid: BoolArray  # root, 5, 2

    def __post_init__(self) -> None:
        n = len(self.roots)
        if (
            type(self.roots) is not tuple
            or not 1 <= n <= 64
            or len(set(self.roots)) != n
            or any(root.stage != 'development' for root in self.roots)
            or len({(root.context, root.seed_sha256) for root in self.roots}) != 1
            or self.valid.shape != (n, 5, 2)
            or self.valid.dtype != np.dtype("bool")
        ):
            raise ValueError("prepared training requires distinct whole dependent refinement roots in one context")
        for name, shape in (
            ("history", (n, 5, 2, 16, 12)),
            ("sketch", (n, 5, 2, 8)),
            ("observed", (n, 5, 2, 9, 5, 7)),
            ("transitions", (n, 5, 2, 9, 21, 12)),
        ):
            value = getattr(self, name)
            if value.shape != shape or value.dtype != np.dtype("float64"):
                raise ValueError(f"prepared dependent refinement {name} changes its fixed scalar roster")
            if name != "sketch" and not np.isfinite(value[self.valid]).all():
                raise ValueError(f"prepared dependent refinement {name} marks an unresolved row valid")
            object.__setattr__(self, name, _freeze(value))
        validate_prepared_training_scalar_labels(
            self.history, self.observed, self.transitions, self.valid
        )
        object.__setattr__(
            self, "valid", np.frombuffer(self.valid.tobytes(), dtype="bool").reshape(n, 5, 2)
        )

    def subset(self, indices: tuple[int, ...]) -> 'PreparedTrainingPanel':
        if len(set(indices)) != len(indices) or any(
            type(i) is not int or not 0 <= i < len(self.roots) for i in indices
        ):
            raise ValueError("dependent refinement subset changes its whole-root census")
        selection = list(indices)
        return PreparedTrainingPanel(
            tuple(self.roots[i] for i in indices),
            self.history[selection],
            self.sketch[selection],
            self.observed[selection],
            self.transitions[selection],
            self.valid[selection],
        )


def _standardize(values: Array, floor: float) -> tuple[Array, Array, Array]:
    mean = values.mean(axis=0)
    scale = np.maximum(values.std(axis=0), floor)
    return (values - mean) / scale, mean, scale


def _word_controls(amplitude: Decimal) -> Array:
    result = np.zeros((9, 2))
    for i, word in enumerate(prepared_words(amplitude)):
        if word.sign:
            directions = ((1, 0), (0, 1), (1, 1), (1, -1))
            direction = np.array(directions[word.direction_index], dtype=float)
            result[i] = word.sign * float(amplitude) * direction / np.linalg.norm(direction)
    return result


def prepared_mechanism_mean(history: Array, sketch: Array, amplitude: Decimal) -> Array:
    """Frozen local two-port equation, including baseline motion and force-off.

    Only measured scalars enter; out-of-port norms are residual-model features,
    not an invented closure. This approximation alone has no coverage claim.
    """
    if (
        history.ndim != 3
        or history.shape[1:] != (16, 12)
        or sketch.shape != (len(history), 8)
        or not np.isfinite(history).all()
        or not np.isfinite(sketch).all()
        or amplitude not in (8, 16)
    ):
        raise ValueError("mechanism predictor requires only the finite handoff scalar contract")
    result = np.empty((len(history), 9, 5, 2))
    controls = _word_controls(amplitude)
    for row in range(len(history)):
        hessian = sketch[row, 2:6].reshape(2, 2)
        if not np.allclose(hessian, hessian.T, rtol=1e-10, atol=1e-10):
            raise ValueError("measured projected Hessian is not symmetric")
        generator = np.zeros((6, 6))
        generator[:2, 2:4] = np.eye(2)
        generator[2:4, :2] = -hessian
        generator[2:4, 2:4] = -np.eye(2)
        generator[2:4, 4:6] = np.eye(2)
        pulse = expm(generator * 0.064)
        free = tuple(expm(generator * ((tick - 64) * 0.001)) for tick in READOUTS)
        initial = np.zeros((9, 6))
        initial[:, 2:4] = history[row, -1, (1, 10)]
        initial[:, 4:6] = sketch[row, :2] + controls
        after_pulse = initial @ pulse.T
        after_pulse[:, 4:6] = sketch[row, :2]
        for j, propagation in enumerate(free):
            result[row, :, j] = (after_pulse @ propagation.T)[:, :2]
    if not np.isfinite(result).all():
        raise ValueError("frozen mechanism forecast is numerically unresolved")
    return result


def prepared_affine_model_shapes(structure: str) -> dict[str, tuple[int, ...]]:
    if structure not in STRUCTURES:
        raise ValueError("finite predictor has an undeclared model structure")
    features = 200 if structure == "mechanism-i1" else 192
    order = int(structure[-1]) if structure.startswith("compact-") else 0
    return {
        "feature_mean": (features,),
        "feature_scale": (features,),
        "output_operator": (features + 1, 9 * 5 * (5 if order else 7)),
        "latent_mean": (8,) if order == 8 else (0,),
        "latent_scale": (8,) if order == 8 else (0,),
        "latent_basis": (8, 4) if order == 8 else (0, 0),
        "state_mean": (order,),
        "state_scale": (order,),
        "transition_operator": (order + 3, order) if order else (0, 0),
    }


def prepared_bilinear_model_shapes(structure: str) -> dict[str, tuple[int, ...]]:
    shapes = prepared_affine_model_shapes(structure)
    if structure.startswith("compact-"):
        order = int(structure[-1])
        shapes["transition_operator"] = (3 * order + 3, order)
    return shapes


def _model_shapes(spec: PreparedAffineModelSpec, structure: str) -> dict[str, tuple[int, ...]]:
    return (
        prepared_bilinear_model_shapes(structure)
        if isinstance(spec, PreparedBilinearModelSpec)
        else prepared_affine_model_shapes(structure)
    )


def _compact_design(states: Array, controls: Array, spec: PreparedAffineModelSpec) -> Array:
    """One fixed feature order for fitting and all twenty rollout steps."""
    if isinstance(spec, PreparedBilinearModelSpec):
        return np.concatenate(
            (states, controls, controls[..., :1] * states, controls[..., 1:] * states), axis=-1
        )
    return np.concatenate((states, controls), axis=-1)


@dataclass(frozen=True, slots=True)
class PreparedFinitePredictor:
    """Deployable coefficients contain no root seeds, labels or fitting API."""

    spec: PreparedAffineModelSpec
    structure: str
    ridge: Decimal
    feature_mean: Array
    feature_scale: Array
    output_operator: Array
    latent_mean: Array
    latent_scale: Array
    latent_basis: Array
    state_mean: Array
    state_scale: Array
    transition_operator: Array

    def __post_init__(self) -> None:
        if (
            type(self.spec) not in (PreparedAffineModelSpec, PreparedBilinearModelSpec)
            or self.structure not in STRUCTURES
            or self.ridge not in RIDGES
            or not isinstance(self.ridge, Decimal)
        ):
            raise ValueError("finite predictor changes its frozen family")
        for name, shape in _model_shapes(self.spec, self.structure).items():
            value = getattr(self, name)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"fitted {name} changes its finite payload shape/validity")
            if name.endswith("scale") and np.any(value <= 0):
                raise ValueError("fitted standardization scale must be strictly positive")
            object.__setattr__(self, name, _freeze(value))

    def predict(self, history: Array, sketch: Array | None = None) -> Array:
        """Return handoff-relative finite predictions; no outcomes or refit input."""
        if history.ndim != 3 or history.shape[1:] != (16, 12) or not np.isfinite(history).all():
            raise ValueError("deployment requires only sixteen measured twelve-scalar samples")
        features = history.reshape(len(history), 192)
        mechanism: Array | None = None
        if self.structure == "mechanism-i1":
            if sketch is None or sketch.shape != (len(history), 8) or not np.isfinite(sketch).all():
                raise ValueError("I1 deployment requires the exact eight measured sketch scalars")
            features = np.column_stack((features, sketch))
            mechanism = prepared_mechanism_mean(history, sketch, self.spec.amplitude)
        elif sketch is not None:
            raise ValueError("I0 deployment cannot receive a mechanism sketch")
        outputs = affine_prediction(
            self.output_operator, (features - self.feature_mean) / self.feature_scale
        ).reshape(len(history), 9, 5, -1)
        if self.structure.startswith("compact-"):
            state = history[:, -1, STATE_CHANNELS]
            if self.structure == "compact-8":
                extras = history[:, -1, LATENT_CHANNELS]
                latent = ((extras - self.latent_mean) / self.latent_scale) @ self.latent_basis
                state = np.column_stack((state, latent))
            state = (state - self.state_mean) / self.state_scale
            states = np.repeat(state[:, None, :], 9, axis=1)
            origin = history[:, -1, (0, 9)]
            receiver = np.empty((len(history), 9, 5, 2))
            controls = _word_controls(self.spec.amplitude) / float(self.spec.amplitude)
            for step in range(20):
                forcing = np.broadcast_to(
                    controls if step < 4 else 0 * controls, (len(history), 9, 2)
                )
                design = _compact_design(states, forcing, self.spec)
                states = affine_prediction(
                    self.transition_operator, design.reshape(-1, design.shape[-1])
                ).reshape(states.shape)
                if step in (3, 7, 11, 15, 19):
                    physical = states * self.state_scale + self.state_mean
                    receiver[:, :, step // 4] = physical[..., :2] - origin[:, None, :]
            outputs = np.concatenate((receiver, outputs), axis=-1)
        elif mechanism is not None:
            outputs[..., :2] += mechanism
        if not np.isfinite(outputs).all():
            raise ValueError("finite predictor is numerically unresolved")
        return _freeze(outputs)


@dataclass(frozen=True, slots=True)
class PreparedFittedModel(PreparedFinitePredictor):
    assigned_training_roots: tuple[PreparedRoot, ...]
    fitted_rows: int

    def __post_init__(self) -> None:
        PreparedFinitePredictor.__post_init__(self)
        if (
            type(self.assigned_training_roots) is not tuple
            or not self.assigned_training_roots
            or len(set(self.assigned_training_roots)) != len(self.assigned_training_roots)
            or any(
                r.stage != 'development' or r.context != self.spec.context
                for r in self.assigned_training_roots
            )
            or type(self.fitted_rows) is not int
            or not 1 <= self.fitted_rows <= 10 * len(self.assigned_training_roots)
        ):
            raise ValueError("fitted model changes its dependent refinement training lineage")

    def deployment_predictor(self) -> PreparedFinitePredictor:
        return PreparedFinitePredictor(
            self.spec,
            self.structure,
            self.ridge,
            **{name: getattr(self, name) for name in prepared_affine_model_shapes(self.structure)},
        )


def fit_prepared_model(
    panel: PreparedTrainingPanel,
    spec: PreparedAffineModelSpec,
    *,
    structure: str,
    ridge: Decimal,
) -> PreparedFittedModel:
    if (
        structure not in spec.structures
        or ridge not in spec.ridges
        or panel.roots[0].context != spec.context
    ):
        raise ValueError("requested dependent refinement fit is outside its declared family/context")
    fit_mask = panel.valid.copy()
    if structure == "mechanism-i1":
        fit_mask &= np.isfinite(panel.sketch).all(axis=-1)
    if not np.any(fit_mask):
        raise ValueError(
            "no complete dependent refinement rows are available; all assigned roots remain in the census"
        )
    predictor = fit_prepared_predictor_arrays(
        panel.history[fit_mask],
        panel.sketch[fit_mask],
        panel.observed[fit_mask],
        panel.transitions[fit_mask],
        spec,
        structure=structure,
        ridge=ridge,
    )
    return PreparedFittedModel(
        spec,
        structure,
        ridge,
        **{name: getattr(predictor, name) for name in _model_shapes(spec, structure)},
        assigned_training_roots=panel.roots,
        fitted_rows=int(fit_mask.sum()),
    )


def fit_prepared_predictor_arrays(
    history: Array,
    sketch: Array,
    observed: Array,
    transitions: Array,
    spec: PreparedAffineModelSpec,
    *,
    structure: str,
    ridge: Decimal,
) -> PreparedFinitePredictor:
    """Reuse numerical recipes without asserting a training tranche identity.

    The calling scientific owner must bind exposure, whole-root assignment and
    selection. The original dependent refinement wrapper above retains its stricter dependent refinement contract.
    """
    n = len(history)
    if structure not in spec.structures or ridge not in spec.ridges or not 1 <= n <= 640:
        raise ValueError("array fit changes its finite family or bounded row count")
    for value, shape in (
        (history, (n, 16, 12)),
        (sketch, (n, 8)),
        (observed, (n, 9, 5, 7)),
        (transitions, (n, 9, 21, 12)),
    ):
        if value.shape != shape or value.dtype != np.dtype("float64"):
            raise ValueError("array fit changes a numerical operand shape/dtype")
        if (value is not sketch or structure == "mechanism-i1") and not np.isfinite(value).all():
            raise ValueError("array fit received unresolved required operands")
    validate_prepared_training_scalar_labels(history, observed, transitions, np.ones(n, dtype=bool))
    observed = observed.copy()
    features = history.reshape(len(history), 192)
    if structure == "mechanism-i1":
        features = np.column_stack((features, sketch))
        observed[..., :2] -= prepared_mechanism_mean(history, sketch, spec.amplitude)
    floor = float(spec.standard_deviation_floor)
    features, feature_mean, feature_scale = _standardize(features, floor)
    latent_mean, latent_scale, state_mean, state_scale = (np.empty(0) for _ in range(4))
    latent_basis, transition_operator = np.empty((0, 0)), np.empty((0, 0))
    if structure.startswith("compact-"):
        labels = transitions
        states = labels[..., STATE_CHANNELS]
        if structure == "compact-8":
            extras = labels[..., LATENT_CHANNELS]
            standardized, latent_mean, latent_scale = _standardize(extras.reshape(-1, 8), floor)
            _, _, right = np.linalg.svd(standardized, full_matrices=False)
            latent_basis = right[:4].T
            # Fix the sign of each basis column for stable persisted bytes.
            for col in range(4):
                if latent_basis[np.argmax(abs(latent_basis[:, col])), col] < 0:
                    latent_basis[:, col] *= -1
            latent = (standardized @ latent_basis).reshape(*labels.shape[:-1], 4)
            states = np.concatenate((states, latent), axis=-1)
        flat = states.reshape(-1, states.shape[-1])
        _, state_mean, state_scale = _standardize(flat, floor)
        states = (states - state_mean) / state_scale
        forcing = np.zeros((*states.shape[:-2], 20, 2))
        forcing[:, :, :4, :] = _word_controls(spec.amplitude)[None, :, None, :] / float(
            spec.amplitude
        )
        design = _compact_design(states[..., :-1, :], forcing, spec)
        transition_operator = fit_affine_operator(
            design.reshape(-1, design.shape[-1]),
            states[..., 1:, :].reshape(-1, states.shape[-1]),
            ridge=float(ridge),
        )
        observed = observed[..., 2:]
    output_operator = fit_affine_operator(
        features, observed.reshape(len(history), -1), ridge=float(ridge)
    )
    return PreparedFinitePredictor(
        spec,
        structure,
        ridge,
        feature_mean,
        feature_scale,
        output_operator,
        latent_mean,
        latent_scale,
        latent_basis,
        state_mean,
        state_scale,
        transition_operator,
    )


def prepared_response_decomposition(prediction: Array) -> tuple[Array, Array, Array]:
    """Baseline, signed odd response and even remainder in the exact word order."""
    if prediction.shape[-3:] != (9, 5, 7) or not np.isfinite(prediction).all():
        raise ValueError("response decomposition requires the complete finite prediction")
    baseline = prediction[..., 0, :, :]
    pairs = prediction[..., 1:, :, :].reshape(*prediction.shape[:-3], 4, 2, 5, 7)
    odd = (pairs[..., 1, :, :] - pairs[..., 0, :, :]) / 2
    even = pairs.mean(axis=-3) - baseline[..., None, :, :]
    return _freeze(baseline), _freeze(odd), _freeze(even)
