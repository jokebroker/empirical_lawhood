"""development's bounded affine stochastic fit, with whole-root selection and access roles.

This is a method implementation, not an execution route or a law finalizer.
The existing affine ridge owner supplies the regression. development supplies its native
clock, representations, drift-to-transition conversion and residual uncertainty.
"""

from dataclasses import dataclass
from hashlib import sha256
import io
from typing import Iterator, Literal

import h5py  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt
from scipy.linalg import eigh

from empirical_lawhood.adapters.methods.response_formalization import fit_affine_operator
from empirical_lawhood.adapters.simulators.six_matrix_response.response_qualification import PARENTS
from empirical_lawhood.adapters.simulators.six_matrix_response.response_source import response_hdf5_read_text

from .development_projection import DEVELOPMENT_DATA_SCHEMA, DEVELOPMENT_KNOTS, ResponseGeometryDevelopmentViewReport


FloatArray = npt.NDArray[np.float64]
Representation = Literal["typed", "untyped", "alternative", "state_clock", "typed_microstate"]
REPRESENTATIONS: tuple[Representation, ...] = (
    "typed",
    "untyped",
    "alternative",
    "state_clock",
    "typed_microstate",
)
RIDGES = (1e-4, 1e-2, 1.0)
DEVELOPMENT_DELTA = 0.125
ALTERNATIVE_SEED = 20260907
DT = np.diff(DEVELOPMENT_KNOTS) * 0.001
FORCES = np.asarray([-8.0, 0.0, 8.0])[:, None] * (np.asarray(DEVELOPMENT_KNOTS[:-1]) < 64)[None, :]


@dataclass(frozen=True)
class ResponseGeometryDevelopmentMeasuredView:
    report: ResponseGeometryDevelopmentViewReport
    parents: dict[str, dict[str, FloatArray]]


def read_response_geometry_development_measurements(report: ResponseGeometryDevelopmentViewReport, payload: bytes) -> ResponseGeometryDevelopmentMeasuredView:
    if len(payload) > 16 * 1024**2 or sha256(payload).hexdigest() != report.data_sha256:
        raise ValueError("development measurements fail their report digest/size")
    parents = {}
    with h5py.File(io.BytesIO(payload), "r") as artifact:
        if (
            response_hdf5_read_text(artifact, "schema") != DEVELOPMENT_DATA_SCHEMA
            or response_hdf5_read_text(artifact, "root_id") != report.root.root_id
            or response_hdf5_read_text(artifact, "projection_config_sha256")
            != report.projection_config.object_fingerprint
            or artifact.attrs.get("refinement") != report.refinement
            or set(artifact) != {p.parent for p in report.packets if p.signed_responses}
        ):
            raise ValueError("development measurement identity/parent roster differs")
        shapes = {
            "scalar": (3, 8, 2),
            "raw": (3, 8, 192),
            "typed": (3, 8, 192),
            "state_clock": (3, 8, 53),
            "native_ticks": (8,),
            "invocation_clock": (3,),
            "passive_native": (3, 2, 15, 15),
            "passive_forecasts": (3, 2, 15, 15),
            "passive_directions": (15, 6),
        }
        for parent in artifact:
            if not isinstance(artifact.get(parent, getlink=True), h5py.HardLink) or not isinstance(
                artifact[parent], h5py.Group
            ):
                raise ValueError("development measurements cannot link to another artifact")
            group = artifact[parent]
            if set(group) != {*shapes, "invocation_buffer"}:
                raise ValueError("development measured arrays change their declared roster")
            arrays = {}
            for name in group:
                if not isinstance(group.get(name, getlink=True), h5py.HardLink):
                    raise ValueError("development measured arrays require local hard links")
                data = group[name]
                if (
                    not isinstance(data, h5py.Dataset)
                    or data.is_virtual
                    or data.external
                    or data.id.get_create_plist().get_nfilters()
                ):
                    raise ValueError("development measured arrays require unfiltered local numeric datasets")
                shape = data.shape
                if (
                    (name in shapes and shape != shapes[name])
                    or (
                        name == "invocation_buffer"
                        and (len(shape) != 1 or not 1 <= shape[0] <= 8192)
                    )
                    or data.dtype not in (np.dtype("float64"), np.dtype("int64"))
                ):
                    raise ValueError("development measured arrays exceed their numeric geometry")
                arrays[name] = np.asarray(data[...], dtype=np.float64)
            origin = report.root.landmark_tick + report.root.invocation_offset
            if not np.array_equal(
                arrays["native_ticks"], origin + np.asarray(DEVELOPMENT_KNOTS)
            ) or not np.array_equal(
                arrays["invocation_clock"],
                [report.root.landmark_tick, origin, report.root.invocation_offset],
            ):
                raise ValueError("development measured clocks differ from the source root")
            if any(
                not np.isfinite(arrays[name]).all() for name in ("raw", "scalar", "state_clock")
            ) or any(np.isinf(value).any() for value in arrays.values()):
                raise ValueError("development measured features are nonfinite outside explicit unknown masks")
            if np.isfinite(arrays["typed"]).all() != report.typed_projector_resolved or (
                not report.typed_projector_resolved and not np.isnan(arrays["typed"]).all()
            ):
                raise ValueError("development typed feature availability differs from its causal projector")
            parents[parent] = arrays
    return ResponseGeometryDevelopmentMeasuredView(report, parents)


@dataclass(frozen=True)
class ResponseGeometryDevelopmentRootSeries:
    """One independent unit, retaining its two paired numerical views."""

    context: str
    index: int
    parent: str
    scalar: FloatArray
    features: FloatArray

    def __post_init__(self) -> None:
        if self.scalar.shape != (2, 3, 8, 2) or self.features.shape[:3] != (2, 3, 8):
            raise ValueError("development fit rows change the paired root/view/sign/clock geometry")
        if not np.isfinite(self.scalar).all() or not np.isfinite(self.features).all():
            raise ValueError("development fitted feature values must have explicit finite mask encoding")


def root_series(
    views: tuple[ResponseGeometryDevelopmentMeasuredView, ResponseGeometryDevelopmentMeasuredView],
    parent: str,
    representation: Representation,
) -> ResponseGeometryDevelopmentRootSeries | None:
    if views[0].report.root != views[1].report.root or tuple(
        v.report.refinement for v in views
    ) != (1, 2):
        raise ValueError("development model rows must retain both views of one physical root")
    if representation not in REPRESENTATIONS:
        raise ValueError("development representation is outside the predeclared roster")
    needs_typed = representation in ("typed", "typed_microstate")
    if any(
        parent not in view.parents or (needs_typed and not view.report.typed_projector_resolved)
        for view in views
    ):
        return None
    features, scalars = [], []
    for view in views:
        arrays = view.parents[parent]
        if representation == "typed_microstate":
            phase = np.concatenate((arrays["typed"], arrays["raw"]), axis=-1)
        else:
            key = {
                "typed": "typed",
                "untyped": "raw",
                "alternative": "raw",
                "state_clock": "state_clock",
            }[representation]
            phase = arrays[key]
        buffer = arrays["invocation_buffer"]
        # Unknowns remain observable masks; no held-root mean imputes a value.
        finite = np.isfinite(buffer)
        initial = np.concatenate(
            (np.where(finite, buffer, 0), finite.astype(float), arrays["invocation_clock"] * 0.001)
        )
        shared = np.broadcast_to(initial, (3, 8, len(initial)))
        features.append(np.concatenate((phase, shared), axis=-1))
        scalars.append(arrays["scalar"])
    root = views[0].report.root
    return ResponseGeometryDevelopmentRootSeries(root.context, root.index, parent, np.stack(scalars), np.stack(features))


@dataclass(frozen=True)
class ResponseGeometryDevelopmentFeatureMap:
    """Fitted only on supplied fit roots, with two explicit physical coordinates."""

    center: FloatArray
    scale: FloatArray
    scalar_regression: FloatArray
    basis: FloatArray
    singular_values: FloatArray
    training_roots: tuple[int, ...]

    def __post_init__(self) -> None:
        count = len(self.center)
        if (
            not 1 <= count <= 16384
            or self.scale.shape != (count,)
            or self.scalar_regression.shape != (3, count)
            or self.basis.shape[0] != count
            or not 0 <= self.basis.shape[1] <= 6
            or self.singular_values.shape != (self.basis.shape[1],)
        ):
            raise ValueError("development feature map exceeds its declared geometry")
        if (
            self.training_roots != tuple(sorted(set(self.training_roots)))
            or len(self.training_roots) < 6
            or any(not 0 <= i < 16 for i in self.training_roots)
        ):
            raise ValueError("development feature map includes another root role")
        if any(
            not np.isfinite(v).all()
            for v in (
                self.center,
                self.scale,
                self.scalar_regression,
                self.basis,
                self.singular_values,
            )
        ) or np.any(self.scale <= 0):
            raise ValueError("development feature map has nonfinite parameters or nonpositive scale")

    def transform(self, scalar: FloatArray, features: FloatArray, dimension: int) -> FloatArray:
        if dimension not in (4, 8) or dimension - 2 > self.basis.shape[1]:
            raise ValueError("development map lacks the selected identified latent dimension")
        normalized = (features - self.center) / self.scale
        residual = (
            normalized
            - np.concatenate((scalar, np.ones((*scalar.shape[:-1], 1))), axis=-1)
            @ self.scalar_regression
        )
        return np.concatenate((scalar, residual @ self.basis[:, : dimension - 2]), axis=-1)


def fit_feature_map(series: tuple[ResponseGeometryDevelopmentRootSeries, ...], *, alternative: bool) -> ResponseGeometryDevelopmentFeatureMap:
    if (
        len(series) < 6
        or len({s.index for s in series}) != len(series)
        or any(not 0 <= s.index < 16 for s in series)
    ):
        raise ValueError(
            "development basis requires at least six distinct fit roots; calibration/validation access is forbidden"
        )
    if len({(s.context, s.parent) for s in series}) != 1:
        raise ValueError("development coefficients cannot pool contexts or parent charts")
    scalar = np.concatenate([s.scalar.reshape(-1, 2) for s in series])
    raw = np.concatenate([s.features.reshape(-1, s.features.shape[-1]) for s in series])
    center, scale = np.mean(raw, axis=0), np.std(raw, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    normalized = (raw - center) / scale
    design = np.column_stack((scalar, np.ones(len(scalar))))
    regression = np.linalg.lstsq(design, normalized, rcond=1e-10)[0]
    residual = normalized - design @ regression
    # Solve the smaller sample Gram matrix; static invocation buffers repeat
    # within a root and must not create a quadratic feature-space allocation.
    values, vectors = eigh(
        residual @ residual.T, subset_by_index=(max(0, len(residual) - 6), len(residual) - 1)
    )
    order = np.argsort(values)[::-1]
    singular = np.sqrt(np.maximum(values[order], 0))
    usable = values[order] > 1e-12 * max(1.0, float(values[order][0]))
    singular = singular[usable]
    if alternative:
        generator = np.random.Generator(np.random.PCG64DXSM(ALTERNATIVE_SEED))
        trial = generator.normal(size=(residual.shape[1], 6))
        # Only directions with actual training variation are retained.
        active = np.std(residual, axis=0) > 1e-12
        trial[~active] = 0
        basis, triangular = np.linalg.qr(trial, mode="reduced")
        rank = int(np.sum(np.abs(np.diag(triangular)) > 1e-10))
        basis = basis[:, : min(rank, len(singular))]
    else:
        basis = residual.T @ vectors[:, order][:, usable] / singular
    for column in range(basis.shape[1]):
        if basis[np.argmax(np.abs(basis[:, column])), column] < 0:
            basis[:, column] *= -1
    return ResponseGeometryDevelopmentFeatureMap(
        center, scale, regression, basis, singular, tuple(sorted(s.index for s in series))
    )


@dataclass(frozen=True)
class ResponseGeometryDevelopmentAffineModel:
    context: str
    parent: str
    representation: Representation
    dimension: int
    ridge: float
    split_at_force_off: bool
    feature_map: ResponseGeometryDevelopmentFeatureMap
    drift: FloatArray
    diffusion: FloatArray
    design_singular_values: tuple[FloatArray, ...]
    unidentified_directions: tuple[FloatArray, ...]

    def __post_init__(self) -> None:
        phases = 2 if self.split_at_force_off else 1
        if (
            self.context not in ("assembling", "prepared")
            or self.parent not in PARENTS
            or self.representation not in REPRESENTATIONS
            or self.dimension not in (4, 8)
            or self.ridge not in RIDGES
        ):
            raise ValueError("development affine model changes its declared candidate chart")
        if (
            self.drift.shape != (phases, self.dimension + 2, self.dimension)
            or self.diffusion.shape != (phases, self.dimension, self.dimension)
            or len(self.design_singular_values) != phases
            or len(self.unidentified_directions) != phases
        ):
            raise ValueError("development affine model loses its complete drift/noise geometry")
        if any(
            not np.isfinite(v).all()
            for v in (
                self.drift,
                self.diffusion,
                *self.design_singular_values,
                *self.unidentified_directions,
            )
        ):
            raise ValueError("development affine model parameters must be finite")

    def predict(self, scalar: FloatArray, features: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Read invocation coordinates only; propagate initial state, affine drift and noise."""
        if scalar.shape != (2,) or features.ndim != 1:
            raise ValueError("development prediction accepts one invocation state, never a future trajectory")
        initial = self.feature_map.transform(scalar, features, self.dimension)
        return propagate_response_geometry_development_latent(initial, self.drift, self.diffusion)


def propagate_response_geometry_development_latent(
    initial: FloatArray, drift: FloatArray, diffusion: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Complete finite model prediction shared by fitting and the registered law evaluator."""
    dimension = len(initial)
    if initial.shape != (dimension,) or not np.isfinite(initial).all():
        raise ValueError("development latent propagation requires its complete finite model geometry")
    states = np.broadcast_to(initial, (3, dimension)).copy()
    covariance = np.zeros((dimension, dimension))
    predictions, deviations = [], []
    for transition, pulse_input, affine, noise in _response_geometry_development_transitions(dimension, drift, diffusion):
        states = (
            states @ transition.T + np.asarray([-8.0, 0.0, 8.0])[:, None] * pulse_input + affine
        )
        covariance = transition @ covariance @ transition.T + noise
        predictions.append(states[:, 0] - initial[0])
        deviations.append(float(np.sqrt(max(0.0, covariance[0, 0]))))
    mean = np.stack(predictions, axis=-1)
    if not np.isfinite(mean).all() or not np.isfinite(deviations).all():
        raise FloatingPointError("development affine propagation is nonfinite")
    return mean, np.asarray(deviations)


def _response_geometry_development_transitions(
    dimension: int, drift: FloatArray, diffusion: FloatArray
) -> Iterator[tuple[FloatArray, FloatArray, FloatArray, FloatArray]]:
    if (
        dimension not in (4, 8)
        or drift.shape not in ((1, dimension + 2, dimension), (2, dimension + 2, dimension))
        or diffusion.shape != (len(drift), dimension, dimension)
        or any(not np.isfinite(value).all() for value in (drift, diffusion))
    ):
        raise ValueError("development latent propagation requires its complete finite model geometry")
    for k, dt in enumerate(DT):
        phase = int(len(drift) == 2 and DEVELOPMENT_KNOTS[k] >= 64)
        coefficient = drift[phase]
        yield (
            np.eye(dimension) + dt * coefficient[:dimension].T,
            dt * coefficient[-2] if DEVELOPMENT_KNOTS[k] < 64 else np.zeros(dimension),
            dt * coefficient[-1],
            dt * diffusion[phase],
        )


def response_geometry_development_endpoint_coefficients(
    dimension: int, drift: FloatArray, diffusion: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Exact composition for one supported short-pulse-response amplitude, including affine drift/noise.

    No independent force-off input is introduced. The returned state transition,
    pulse input, affine term and covariance share the forecast's seven updates.
    """
    state = np.eye(dimension)
    pulse, affine = np.zeros(dimension), np.zeros(dimension)
    covariance = np.zeros((dimension, dimension))
    for transition, pulse_input, intercept, noise in _response_geometry_development_transitions(dimension, drift, diffusion):
        state = transition @ state
        pulse = transition @ pulse + pulse_input
        affine = transition @ affine + intercept
        covariance = transition @ covariance @ transition.T + noise
    if any(not np.isfinite(value).all() for value in (state, pulse, affine, covariance)):
        raise FloatingPointError("development composed endpoint operator is nonfinite")
    return state, pulse, affine, covariance


def fit_affine_model(
    series: tuple[ResponseGeometryDevelopmentRootSeries, ...],
    feature_map: ResponseGeometryDevelopmentFeatureMap,
    *,
    representation: Representation,
    dimension: int,
    ridge: float,
    split: bool,
) -> ResponseGeometryDevelopmentAffineModel:
    if tuple(sorted(s.index for s in series)) != feature_map.training_roots or any(
        not 0 <= s.index < 16 for s in series
    ):
        raise ValueError("development fit and feature map roots differ or include protected roles")
    if ridge not in RIDGES or len({(s.context, s.parent) for s in series}) != 1:
        raise ValueError("development regression changes its grid or native chart")
    latent = np.stack([feature_map.transform(s.scalar, s.features, dimension) for s in series])
    drift, diffusion, singular, nullspaces = fit_response_geometry_development_latent_drift(latent, ridge=ridge, split=split)
    return ResponseGeometryDevelopmentAffineModel(
        series[0].context,
        series[0].parent,
        representation,
        dimension,
        ridge,
        split,
        feature_map,
        drift,
        diffusion,
        singular,
        nullspaces,
    )


def fit_response_geometry_development_latent_drift(
    latent: FloatArray,
    *,
    ridge: float,
    split: bool,
) -> tuple[FloatArray, FloatArray, tuple[FloatArray, ...], tuple[FloatArray, ...]]:
    """The shared regression arithmetic; this grants no root identity or qualification.

    Native fitting authenticates roots before this call. Hypothetical acquisition
    refits separately retain actual training roots and explicitly simulated rows.
    """
    if (
        latent.ndim != 5
        or latent.shape[1:4] != (2, 3, 8)
        or latent.shape[-1] not in (4, 8)
        or len(latent) < 2
        or not np.isfinite(latent).all()
        or ridge not in RIDGES
    ):
        raise ValueError("development latent refit changes its finite pulse/clock/model geometry")
    dimension = latent.shape[-1]
    start, changes = latent[..., :-1, :], np.diff(latent, axis=-2)
    controls = np.broadcast_to(FORCES[None, None, :, :, None], (*start.shape[:-1], 1))
    full_design = np.concatenate((start, controls), axis=-1)
    drift, diffusion, singular, nullspaces = [], [], [], []
    for phase in range(2 if split else 1):
        take = (
            (np.asarray(DEVELOPMENT_KNOTS[:-1]) < 64)
            if split and phase == 0
            else (np.asarray(DEVELOPMENT_KNOTS[:-1]) >= 64)
            if split
            else np.ones(7, dtype=bool)
        )
        x = full_design[..., take, :].reshape(-1, dimension + 1)
        dt = np.broadcast_to(DT[take], (*start.shape[:-2], int(np.sum(take)))).ravel()
        increments = changes[..., take, :].reshape(-1, dimension)
        center, scale = x.mean(axis=0), x.std(axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        standardized = (x - center) / scale
        fitted = fit_affine_operator(standardized, increments / dt[:, None], ridge=ridge)
        coefficient = np.vstack(
            (fitted[:-1] / scale[:, None], fitted[-1] - (center / scale) @ fitted[:-1])
        )
        residual = increments - dt[:, None] * (x @ coefficient[:-1] + coefficient[-1])
        # Brownian innovation scaling is an explicit reduced-model assumption;
        # held-root calibration covers its endpoint error, not latent truth.
        noise = residual / np.sqrt(dt[:, None])
        covariance = noise.T @ noise / len(noise)
        _, sv, vh = np.linalg.svd(
            np.column_stack((standardized, np.ones(len(x)))), full_matrices=False
        )
        rank = int(np.sum(sv > 1e-10 * max(1.0, float(sv[0]))))
        drift.append(coefficient)
        diffusion.append((covariance + covariance.T) / 2)
        singular.append(sv)
        nullspaces.append(vh[rank:])
    return np.stack(drift), np.stack(diffusion), tuple(singular), tuple(nullspaces)


def endpoint_predictions(
    model: ResponseGeometryDevelopmentAffineModel, series: ResponseGeometryDevelopmentRootSeries
) -> tuple[FloatArray, FloatArray, FloatArray]:
    if (model.context, model.parent) != (series.context, series.parent):
        raise ValueError("development prediction cannot transport coefficients between native charts")
    predictions, deviations = [], []
    for view in range(2):
        # All signs share one invocation; disagreement is a custody error.
        if not np.array_equal(
            series.scalar[view, :, 0], np.broadcast_to(series.scalar[view, 0, 0], (3, 2))
        ) or not np.array_equal(
            series.features[view, :, 0],
            np.broadcast_to(series.features[view, 0, 0], series.features[view, :, 0].shape),
        ):
            raise ValueError("development signs do not share their exact invocation state")
        mean, deviation = model.predict(series.scalar[view, 0, 0], series.features[view, 0, 0])
        predictions.append(mean[:, -1])
        deviations.append(np.full(3, deviation[-1]))
    observed = series.scalar[:, :, -1, 0] - series.scalar[:, :, 0, 0]
    return np.stack(predictions), np.stack(deviations), observed


@dataclass(frozen=True)
class ResponseGeometryDevelopmentCandidateScore:
    dimension: int
    ridge: float
    split_at_force_off: bool
    roots: tuple[int, ...]
    # Root, parent, numerical view, sign; never 30 independent observations.
    errors: FloatArray | None
    reason: str | None

    @property
    def losses(self) -> FloatArray:
        if self.errors is None:
            raise ValueError("development failed candidate has no invented score")
        return np.asarray(np.mean(np.square(self.errors), axis=(1, 2, 3)), dtype=np.float64)

    @property
    def mean_loss(self) -> float:
        return float(np.mean(self.losses))

    @property
    def standard_error(self) -> float:
        return float(np.std(self.losses, ddof=1) / np.sqrt(len(self.roots)))


@dataclass(frozen=True)
class ResponseGeometryDevelopmentModelGroup:
    context: str
    representation: Representation
    roots: tuple[int, ...]
    scores: tuple[ResponseGeometryDevelopmentCandidateScore, ...]
    selected_index: int | None
    models: tuple[ResponseGeometryDevelopmentAffineModel, ...]
    reason: str | None

    def __post_init__(self) -> None:
        if (
            self.context not in ("assembling", "prepared")
            or self.representation not in REPRESENTATIONS
            or self.roots != tuple(sorted(set(self.roots)))
            or any(not 0 <= i < 16 for i in self.roots)
        ):
            raise ValueError("development model group changes its fitting role")
        if self.selected_index is not None and not 0 <= self.selected_index < len(self.scores):
            raise ValueError("development selected model is outside its retained grid")
        if self.models and (
            tuple(m.parent for m in self.models) != PARENTS
            or any(
                m.context != self.context
                or m.representation != self.representation
                or m.feature_map.training_roots != self.roots
                for m in self.models
            )
            or self.selected_index is None
            or self.reason is not None
        ):
            raise ValueError("development model group changes its selected five-parent family")


def organize_response_geometry_development_views(
    views: tuple[ResponseGeometryDevelopmentMeasuredView, ...],
    *,
    context: str,
    role: Literal["fit", "calibration", "validation"],
) -> dict[int, tuple[ResponseGeometryDevelopmentMeasuredView, ResponseGeometryDevelopmentMeasuredView]]:
    expected = {"fit": range(16), "calibration": range(16, 32), "validation": range(32, 64)}[role]
    if (
        context not in ("assembling", "prepared")
        or len(views) != len(expected) * 2
        or len({v.report.projection_config for v in views}) != 1
    ):
        raise ValueError("development role input count/context/config differs")
    by_slot = {(v.report.root.index, v.report.refinement): v for v in views}
    if (
        len(by_slot) != len(views)
        or set(by_slot) != {(i, r) for i in expected for r in (1, 2)}
        or any(v.report.root.context != context for v in views)
    ):
        raise ValueError("development stage input includes another root role or repeats a numerical view")
    return {i: (by_slot[i, 1], by_slot[i, 2]) for i in expected}


def development_series(
    pairs: dict[int, tuple[ResponseGeometryDevelopmentMeasuredView, ResponseGeometryDevelopmentMeasuredView]],
    representation: Representation,
) -> dict[str, dict[int, ResponseGeometryDevelopmentRootSeries]]:
    return {
        parent: {
            i: series
            for i, pair in pairs.items()
            if (series := root_series(pair, parent, representation)) is not None
        }
        for parent in PARENTS
    }


def select_response_geometry_development_models(
    series: dict[str, dict[int, ResponseGeometryDevelopmentRootSeries]],
    *,
    context: str,
    representation: Representation,
) -> ResponseGeometryDevelopmentModelGroup:
    """One bounded search. This entry point accepts fitting roots exclusively.

    Also used inside each outer support fold: its grid, bases and choice then
    see only that outer fold's training roots. No scores select a later refit.
    """
    if set(series) != set(PARENTS) or any(
        s.context != context or s.parent != parent or s.index != index or not 0 <= index < 16
        for parent, values in series.items()
        for index, s in values.items()
    ):
        raise ValueError("development model selection requires its exact fit-root native charts")
    roots = tuple(sorted(set.intersection(*(set(values) for values in series.values()))))
    # Twelve complete roots suffice for the main fit; a nested fold can have
    # nine, leaving at least six roots for every inner coefficient/basis fit.
    if len(roots) < 9:
        return ResponseGeometryDevelopmentModelGroup(
            context, representation, roots, (), None, (), "INSUFFICIENT_COMPLETE_FIT_ROOTS"
        )
    folds = tuple(
        (tuple(i for i in roots if i % 4 != fold), tuple(i for i in roots if i % 4 == fold))
        for fold in range(4)
    )
    folds = tuple((training, testing) for training, testing in folds if testing)
    maps = {}
    failures = {}
    for fold, (training, _) in enumerate(folds):
        for parent in PARENTS:
            try:
                maps[fold, parent] = fit_feature_map(
                    tuple(series[parent][i] for i in training),
                    alternative=representation == "alternative",
                )
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                failures[fold, parent] = type(error).__name__

    def score(dimension: int, ridge: float, split: bool) -> ResponseGeometryDevelopmentCandidateScore:
        errors = np.empty((len(roots), len(PARENTS), 2, 3))
        for fold, (training, testing) in enumerate(folds):
            for parent_index, parent in enumerate(PARENTS):
                if (fold, parent) in failures:
                    return ResponseGeometryDevelopmentCandidateScore(
                        dimension, ridge, split, roots, None, "FOLD_BASIS_UNAVAILABLE"
                    )
                try:
                    model = fit_affine_model(
                        tuple(series[parent][i] for i in training),
                        maps[fold, parent],
                        representation=representation,
                        dimension=dimension,
                        ridge=ridge,
                        split=split,
                    )
                    for index in testing:
                        prediction, _, actual = endpoint_predictions(
                            model, series[parent][index]
                        )
                        errors[roots.index(index), parent_index] = prediction - actual
                except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                    return ResponseGeometryDevelopmentCandidateScore(
                        dimension, ridge, split, roots, None, "FOLD_MODEL_UNAVAILABLE"
                    )
        return ResponseGeometryDevelopmentCandidateScore(dimension, ridge, split, roots, errors, None)

    scores = [score(dimension, ridge, False) for dimension in (4, 8) for ridge in RIDGES]
    available = [i for i, value in enumerate(scores) if value.errors is not None]
    if not available:
        return ResponseGeometryDevelopmentModelGroup(
            context, representation, roots, tuple(scores), None, (), "NO_ESTIMABLE_CANDIDATE"
        )
    best_unsplit = min(available, key=lambda i: scores[i].mean_loss)
    unsplit_errors = scores[best_unsplit].errors
    assert unsplit_errors is not None
    if abs(float(np.mean(unsplit_errors))) > DEVELOPMENT_DELTA / 4:
        scores.extend(score(dimension, ridge, True) for dimension in (4, 8) for ridge in RIDGES)
        improved = [
            i
            for i in range(6, 12)
            if scores[i].errors is not None
            and scores[i].mean_loss <= 0.9 * scores[best_unsplit].mean_loss
        ]
        available.extend(improved)
    else:
        scores.extend(
            ResponseGeometryDevelopmentCandidateScore(d, r, True, roots, None, "TIME_EXTENSION_NOT_TRIGGERED")
            for d in (4, 8)
            for r in RIDGES
        )
    best = min(available, key=lambda i: scores[i].mean_loss)
    ceiling = scores[best].mean_loss + scores[best].standard_error
    eligible = [i for i in available if scores[i].mean_loss <= ceiling]
    selected = min(
        eligible,
        key=lambda i: (
            scores[i].dimension,
            scores[i].split_at_force_off,
            -scores[i].ridge,
            scores[i].mean_loss,
        ),
    )
    chosen = scores[selected]
    models = []
    try:
        for parent in PARENTS:
            training_series = tuple(series[parent][i] for i in roots)
            feature_map = fit_feature_map(
                training_series, alternative=representation == "alternative"
            )
            models.append(
                fit_affine_model(
                    training_series,
                    feature_map,
                    representation=representation,
                    dimension=chosen.dimension,
                    ridge=chosen.ridge,
                    split=chosen.split_at_force_off,
                )
            )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return ResponseGeometryDevelopmentModelGroup(
            context,
            representation,
            roots,
            tuple(scores),
            selected,
            (),
            "SELECTED_FULL_FIT_UNAVAILABLE",
        )
    return ResponseGeometryDevelopmentModelGroup(
        context, representation, roots, tuple(scores), selected, tuple(models), None
    )


def fit_response_geometry_development_context(
    views: tuple[ResponseGeometryDevelopmentMeasuredView, ...], *, context: str
) -> tuple[ResponseGeometryDevelopmentModelGroup, ...]:
    pairs = organize_response_geometry_development_views(views, context=context, role="fit")
    groups = []
    for representation in REPRESENTATIONS:
        series = development_series(pairs, representation)
        complete = set.intersection(*(set(values) for values in series.values()))
        if len(complete) < 12:
            groups.append(
                ResponseGeometryDevelopmentModelGroup(
                    context,
                    representation,
                    tuple(sorted(complete)),
                    (),
                    None,
                    (),
                    "INSUFFICIENT_COMPLETE_FIT_ROOTS",
                )
            )
        else:
            groups.append(
                select_response_geometry_development_models(series, context=context, representation=representation)
            )
    return tuple(groups)
