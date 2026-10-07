"""dependent refinement-only nested fit products, preserving every assigned root and failed attempt.

These are nominal fitting records. Their development multipliers are not
protected calibration, and this module cannot finalize a response law.
"""

import base64
from dataclasses import dataclass
from decimal import Decimal
from math import ceil
from time import perf_counter_ns, process_time_ns
from typing import ClassVar

import numpy as np

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id
from empirical_lawhood.planning.native_source import NativeNestedCrossfitPlan, NativeNestedCrossfitPopulation
from empirical_lawhood.adapters.simulators.prepared_response.development_roster import prepared_response_development_crossfit_plan
from .models import Array, PreparedTrainingPanel, PreparedBilinearModelSpec, STRUCTURES
from .calibration import fit_prepared_nested_family, fit_prepared_nested_scales, prepared_root_folds
from .coefficients import PreparedBilinearPredictorCoefficients, prepared_bilinear_predictor_coefficients
from .development_projection import PreparedResponseDevelopmentProjectionConfig, PreparedResponseDevelopmentViewProjection, collect_prepared_response_development_view_panel


MAXIMUM_DEVELOPMENT_FIT_BYTES = 16 * 1024**2
_ARRAY_SHAPE = (64, 5, 2, 9, 5, 7)
_BLOCK_NAMES = ("outer-predictions", "outer-scales")


@dataclass(frozen=True, slots=True)
class PreparedResponseDevelopmentFitConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/prepared-response/prepared-response-development-fit-config'
    config_id: str
    projection: PreparedResponseDevelopmentProjectionConfig
    crossfit_plan: NativeNestedCrossfitPlan
    scale_positive_floor: Decimal = Decimal("0.00000001")
    normalization_rule: str = (
        "EPSILON_RECEIVERS_INHERITED_PRESERVATION_LIMITS_UNIT_MASS_IMPULSE_ENERGY_WORK"
    )
    development_quantile: Decimal = Decimal("0.95")
    scope: str = "DEVELOPMENT_NESTED_CROSSFIT_NOMINATION_NOT_PROTECTED_CALIBRATION"

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        if (
            type(self.scale_positive_floor) is not Decimal
            or self.scale_positive_floor != Decimal("0.00000001")
            or self.normalization_rule
            != "EPSILON_RECEIVERS_INHERITED_PRESERVATION_LIMITS_UNIT_MASS_IMPULSE_ENERGY_WORK"
            or type(self.development_quantile) is not Decimal
            or self.development_quantile != Decimal("0.95")
            or self.scope != "DEVELOPMENT_NESTED_CROSSFIT_NOMINATION_NOT_PROTECTED_CALIBRATION"
            or self.crossfit_plan != prepared_response_development_crossfit_plan(self.projection.native_spec)
        ):
            raise ValueError("dependent refinement fit changes its predeclared scales, loss or nomination ceiling")

    @property
    def normalizers(self) -> tuple[Decimal, ...]:
        charter = self.projection.source_qualification.selected_charter
        assert charter is not None
        # A dimensional work normalization, not a preservation/effort admission cap.
        work = (charter.amplitude * Decimal("0.064")) ** 2 / 2
        return (
            charter.epsilon,
            charter.epsilon,
            Decimal("0.125"),
            Decimal("0.05"),
            Decimal("0.05"),
            work,
            work,
        )

    def model_spec(self, context: str) -> PreparedBilinearModelSpec:
        amplitude = self.projection.native_spec.selected_amplitude
        assert amplitude is not None
        return PreparedBilinearModelSpec(f"{self.config_id}.{context}.model-family", context, amplitude)

    def crossfit_population(self, context: str) -> NativeNestedCrossfitPopulation:
        values = tuple(
            population
            for population in self.crossfit_plan.populations
            if population.population_id == f"prepared-response.dependent-refinement.crossfit.{context}"
        )
        if len(values) != 1:
            raise ValueError("dependent refinement fit context is outside its frozen cross-fit populations")
        return values[0]


def _planned_fold_assignments(
    config: PreparedResponseDevelopmentFitConfig, context: str, roots: tuple[object, ...]
) -> tuple[int, ...]:
    population = config.crossfit_population(context)
    unit_ids = tuple(getattr(root, "physical_unit_id") for root in roots)
    if unit_ids != population.outer_partition.population_unit_ids:
        raise ValueError("dependent refinement fit roots differ from their frozen cross-fit population")
    heldout = {
        unit_id: fold
        for fold, assignment in enumerate(population.outer_partition.folds)
        for unit_id in assignment.heldout_unit_ids
    }
    return tuple(heldout[unit_id] for unit_id in unit_ids)


def _pack(predictions: Array, scales: Array) -> tuple[tuple[str, str], ...]:
    return tuple(
        (name, base64.b64encode(np.asarray(value, dtype="<f8").tobytes()).decode("ascii"))
        for name, value in zip(_BLOCK_NAMES, (predictions, scales), strict=True)
    )


@dataclass(frozen=True, slots=True)
class PreparedResponseDevelopmentModelFit(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/prepared-response/prepared-response-development-model-fit'
    config: PreparedResponseDevelopmentFitConfig
    context: str
    structure: str
    projections: tuple[ObjectIdentity, ...]
    fold_assignments: tuple[int, ...]
    coefficients: tuple[PreparedBilinearPredictorCoefficients, ...]
    development_multipliers: tuple[Decimal | None, ...]
    blocks: tuple[tuple[str, str], ...]
    failure_reason: str | None
    elapsed_nanoseconds: int
    cpu_nanoseconds: int
    completed_candidate_fits: int | None
    completed_scale_fits: int | None
    encoding: str = "BASE64_LITTLE_ENDIAN_FLOAT64_C_ORDER_NAN_UNKNOWN_FIXED_DEVELOPMENT_ROOTS"

    def __post_init__(self) -> None:
        roots = tuple(
            r for r in self.config.projection.native_spec.roots if r.context == self.context
        )
        if (
            len(roots) != 64
            or self.structure not in STRUCTURES
            or tuple(v.object_id for v in self.projections)
            != tuple(f"{root.root_id}.project.r{refinement}" for root in roots for refinement in (1, 2))
            or any(
                value.object_schema != PreparedResponseDevelopmentViewProjection.SCHEMA
                for value in self.projections
            )
            or self.fold_assignments != _planned_fold_assignments(
                self.config, self.context, roots
            )
            or tuple(name for name, _ in self.blocks) != _BLOCK_NAMES
            or self.encoding != "BASE64_LITTLE_ENDIAN_FLOAT64_C_ORDER_NAN_UNKNOWN_FIXED_DEVELOPMENT_ROOTS"
            or any(
                type(v) is not int or v < 0
                for v in (self.elapsed_nanoseconds, self.cpu_nanoseconds)
            )
            or any(
                value is not None and (type(value) is not int or value < 0)
                for value in (self.completed_candidate_fits, self.completed_scale_fits)
            )
        ):
            raise ValueError(
                "dependent refinement fit product changes its assigned roots, folds, model or measured costs"
            )
        arrays = self.arrays()
        if self.failure_reason is None:
            if (
                len(self.coefficients) != 5
                or len(self.development_multipliers) != 5
                or self.completed_candidate_fits != 65
                or self.completed_scale_fits != 5
                or any(
                    c.model_spec != self.config.model_spec(self.context)
                    or c.structure != self.structure
                    or c.scale_positive_floor != self.config.scale_positive_floor
                    for c in self.coefficients
                )
                or any(
                    q is not None and (type(q) is not Decimal or not q.is_finite() or q < 0)
                    for q in self.development_multipliers
                )
            ):
                raise ValueError(
                    "dependent refinement nominal product loses a nested/final fit or provisional multiplier"
                )
            finite = np.isfinite(arrays[1])
            if np.any(arrays[1][finite] < float(self.config.scale_positive_floor)):
                raise ValueError("dependent refinement outer scales violate their frozen positive floor")
        elif (
            self.failure_reason
            not in (
                "INSUFFICIENT_COMPLETE_DEVELOPMENT_ROWS",
                "NUMERICAL_FIT_FAILURE",
                "HELD_RESIDUALS_UNAVAILABLE",
            )
            or self.coefficients
            or self.development_multipliers
            or any(not np.isnan(a).all() for a in arrays)
            or self.completed_candidate_fits not in (None, 0, 65)
            or self.completed_scale_fits not in (None, 0, 5)
        ):
            raise ValueError("failed dependent refinement attempt cannot fabricate fitted products or completed work")
        if len(self.canonical_bytes()) > MAXIMUM_DEVELOPMENT_FIT_BYTES:
            raise ValueError("dependent refinement fit exceeds its complete sixteen-MiB product bound")

    @property
    def report_id(self) -> str:
        return f"prepared-response.dependent-refinement.fit.{self.context}.{self.structure}"

    @property
    def nominal_coefficients(self) -> PreparedBilinearPredictorCoefficients | None:
        return self.coefficients[-1] if self.coefficients else None

    def outer_halfwidths(self) -> Array:
        """Pair each held root's scale with its outer-training-only multiplier."""
        _, scales = self.arrays()
        multipliers = np.full(64, np.inf)
        if len(self.development_multipliers) == 5:
            for index, fold in enumerate(self.fold_assignments):
                value = self.development_multipliers[fold]
                if value is not None:
                    multipliers[index] = float(value)
        with np.errstate(invalid="ignore"):
            return np.asarray(scales * multipliers.reshape(64, 1, 1, 1, 1, 1), dtype=np.float64)

    def arrays(self) -> tuple[Array, Array]:
        size = int(np.prod(_ARRAY_SHAPE)) * 8
        result = []
        for _, encoded in self.blocks:
            if not isinstance(encoded, str) or len(encoded) != 4 * ((size + 2) // 3):
                raise ValueError("dependent refinement fit scalar block changes its exact byte bound")
            raw = base64.b64decode(encoded, validate=True)
            if len(raw) != size or base64.b64encode(raw).decode("ascii") != encoded:
                raise ValueError("dependent refinement fit scalar block changes its canonical bytes")
            value = np.frombuffer(raw, dtype="<f8").reshape(_ARRAY_SHAPE)
            if np.isinf(value).any():
                raise ValueError("dependent refinement fit prediction/scale must be finite or an explicit unknown")
            result.append(value)
        return result[0], result[1]


def _provisional_multiplier(observed: Array, predicted: Array, scales: Array) -> Decimal | None:
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        residual = abs(observed - predicted) / scales
    residual[~np.isfinite(residual)] = np.inf
    root_scores = np.max(residual, axis=(1, 2, 3, 4, 5))
    order = ceil((len(root_scores) + 1) * 0.95)
    quantile = np.inf if order > len(root_scores) else np.sort(root_scores)[order - 1]
    return Decimal(str(float(quantile))) if np.isfinite(quantile) else None


def _scale_on_panel(
    panel: PreparedTrainingPanel, coefficients: PreparedBilinearPredictorCoefficients
) -> Array:
    _, scale = coefficients.predictors()
    valid = panel.valid.copy()
    if scale.instrument_tier == "I1":
        valid &= np.isfinite(panel.sketch).all(axis=-1)
    values = np.full(panel.observed.shape, np.nan)
    if valid.any():
        values[valid] = scale.predict(
            panel.history[valid], panel.sketch[valid] if scale.instrument_tier == "I1" else None
        )
    return values


def fit_prepared_response_development_structure(
    config: PreparedResponseDevelopmentFitConfig,
    context: str,
    structure: str,
    reports: tuple[PreparedResponseDevelopmentViewProjection, ...],
) -> PreparedResponseDevelopmentModelFit:
    panel = collect_prepared_response_development_view_panel(config.projection, context, reports)
    spec = config.model_spec(context)
    if structure not in STRUCTURES:
        raise ValueError("dependent refinement fitting cannot expand its predeclared structure family")
    folds = _planned_fold_assignments(config, context, panel.roots)
    valid = panel.valid.copy()
    if structure == "mechanism-i1":
        valid &= np.isfinite(panel.sketch).all(axis=-1)
    start, cpu_start = perf_counter_ns(), process_time_ns()
    failure = None
    completed_candidates: int | None = None
    completed_scales: int | None = None
    coefficients: tuple[PreparedBilinearPredictorCoefficients, ...] = ()
    multipliers: tuple[Decimal | None, ...] = ()
    predictions, scales = np.full(_ARRAY_SHAPE, np.nan), np.full(_ARRAY_SHAPE, np.nan)
    # Every inner and outer training slice must retain enough scalar rows for
    # all declared structures. No held root is removed to satisfy this check.
    training_slices: list[tuple[int, ...]] = []
    for outer in (*range(4), None):
        indices = tuple(i for i, f in enumerate(folds) if outer is None or f != outer)
        inner_folds = prepared_root_folds(tuple(panel.roots[i] for i in indices))
        training_slices.extend(
            tuple(i for i, f in zip(indices, inner_folds, strict=True) if f != inner)
            for inner in range(4)
        )
    if any(sum(valid[i].sum() for i in indices) < 4 for indices in training_slices):
        failure = "INSUFFICIENT_COMPLETE_DEVELOPMENT_ROWS"
        completed_candidates = completed_scales = 0
    else:
        try:
            nested = fit_prepared_nested_family(
                panel,
                spec,
                structure=structure,
                normalizers=np.asarray(config.normalizers, dtype=np.float64),
            )
            completed_candidates = 65
            scale_fits = fit_prepared_nested_scales(
                panel, nested, positive_floor=float(config.scale_positive_floor)
            )
            completed_scales = 5
            coefficients = tuple(
                prepared_bilinear_predictor_coefficients(m, s)
                for m, s in zip(
                    (*nested.outer.models, nested.final),
                    (*scale_fits.outer, scale_fits.final),
                    strict=True,
                )
            )
            numbers = []
            for fold in range(5):
                training = (
                    panel
                    if fold == 4
                    else panel.subset(tuple(i for i, f in enumerate(folds) if f != fold))
                )
                training_prediction = (
                    nested.outer.predictions
                    if fold == 4
                    else nested.selected_inner_crossfits[fold].predictions
                )
                numbers.append(
                    _provisional_multiplier(
                        training.observed,
                        training_prediction,
                        _scale_on_panel(training, coefficients[fold]),
                    )
                )
            multipliers = tuple(numbers)
            predictions, scales = nested.outer.predictions, scale_fits.out_of_fold_scales
        except np.linalg.LinAlgError:
            failure = "NUMERICAL_FIT_FAILURE"
        except ValueError as exc:
            if str(exc) == "finite predictor is numerically unresolved":
                failure = "NUMERICAL_FIT_FAILURE"
            elif str(exc) in (
                "no complete dependent refinement rows are available; all assigned roots remain in the census",
                "no held-out complete residual rows are available for dependent refinement scaling",
            ):
                failure = "HELD_RESIDUALS_UNAVAILABLE"
            else:
                raise
    if failure is not None:
        coefficients, multipliers = (), ()
    return PreparedResponseDevelopmentModelFit(
        config,
        context,
        structure,
        tuple(ObjectIdentity.from_record(r.report_id, r) for r in reports),
        folds,
        coefficients,
        multipliers,
        _pack(predictions, scales),
        failure,
        perf_counter_ns() - start,
        process_time_ns() - cpu_start,
        completed_candidates,
        completed_scales,
    )


def decode_prepared_response_development_fit(payload: bytes) -> PreparedResponseDevelopmentModelFit:
    return decode_canonical_bytes(payload, PreparedResponseDevelopmentModelFit, maximum_bytes=MAXIMUM_DEVELOPMENT_FIT_BYTES)
