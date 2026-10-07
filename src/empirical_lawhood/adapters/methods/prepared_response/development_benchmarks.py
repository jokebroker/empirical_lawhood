"Retained dependent refinement constant-gain diagnostics; no new model selection or calibration."

import base64
from dataclasses import dataclass
from time import process_time_ns
from typing import ClassVar

import numpy as np

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord

from .benchmarks import fit_prepared_constant_gain
from .development_fit import PreparedResponseDevelopmentFitConfig, PreparedResponseDevelopmentModelFit
from .development_projection import PreparedResponseDevelopmentViewProjection, collect_prepared_response_development_view_panel
from .models import Array, STRUCTURES, _word_controls


CONTEXTS = ("assembling", "prepared")
_SHAPES = {"gains": (2, 5, 2, 5, 2), "held-receiver-predictions": (2, 4, 64, 5, 2, 9, 5, 2)}
MAXIMUM_CONSTANT_GAIN_REPORT_BYTES = 8 * 1024**2


@dataclass(frozen=True, slots=True)
class PreparedResponseDevelopmentConstantGainReport(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/prepared-response/prepared-response-development-constant-gain-report'
    config: PreparedResponseDevelopmentFitConfig
    fits: tuple[ObjectIdentity, ...]
    blocks: tuple[tuple[str, str], ...]
    completed_gain_fits: int
    cpu_nanoseconds: int
    rule: str = "TRAINING_ONLY_SIGNED_ODD_OLS_REUSE_OUTER_HOLD_BASELINE_FOUR_FOLDS_PLUS_FINAL"
    scope: str = "DEVELOPMENT_POINT_BENCHMARK_ALL_ROOTS_NO_INTERVAL_OR_MODEL_SELECTION_AUTHORITY"

    @property
    def report_id(self) -> str:
        return "prepared-response.dependent-refinement.constant-gain-benchmark"

    def __post_init__(self) -> None:
        if (
            tuple(value.object_id for value in self.fits)
            != tuple(f"prepared-response.dependent-refinement.fit.{context}.{structure}" for context in CONTEXTS for structure in STRUCTURES)
            or any(value.object_schema != PreparedResponseDevelopmentModelFit.SCHEMA for value in self.fits)
            or tuple(name for name, _ in self.blocks) != tuple(_SHAPES)
            or type(self.completed_gain_fits) is not int or not 0 <= self.completed_gain_fits <= 10
            or type(self.cpu_nanoseconds) is not int or self.cpu_nanoseconds < 0
            or self.rule != "TRAINING_ONLY_SIGNED_ODD_OLS_REUSE_OUTER_HOLD_BASELINE_FOUR_FOLDS_PLUS_FINAL"
            or self.scope != "DEVELOPMENT_POINT_BENCHMARK_ALL_ROOTS_NO_INTERVAL_OR_MODEL_SELECTION_AUTHORITY"
        ):
            raise ValueError("dependent refinement constant-gain report changes its frozen census or diagnostic rule")
        arrays = self.arrays()
        gain_available = np.isfinite(arrays["gains"]).all(axis=(2, 3, 4))
        gain_absent = np.isnan(arrays["gains"]).all(axis=(2, 3, 4))
        if not (gain_available | gain_absent).all() or int(gain_available.sum()) != self.completed_gain_fits:
            raise ValueError("dependent refinement constant-gain fit count differs from its retained coefficients")

    def arrays(self) -> dict[str, Array]:
        result = {}
        for name, encoded in self.blocks:
            count = int(np.prod(_SHAPES[name])) * 8
            if len(encoded) != 4 * ((count + 2) // 3):
                raise ValueError("dependent refinement constant-gain block changes its exact full-census size")
            raw = base64.b64decode(encoded, validate=True)
            if len(raw) != count or base64.b64encode(raw).decode("ascii") != encoded:
                raise ValueError("dependent refinement constant-gain block is not canonical base64")
            array = np.frombuffer(raw, dtype="<f8").reshape(_SHAPES[name])
            if np.isinf(array).any():
                raise ValueError("dependent refinement constant-gain missingness must remain explicit NaN")
            result[name] = array
        return result


def build_prepared_response_development_constant_gain_report(
    config: PreparedResponseDevelopmentFitConfig,
    projections: tuple[PreparedResponseDevelopmentViewProjection, ...],
    fits: tuple[PreparedResponseDevelopmentModelFit, ...],
) -> PreparedResponseDevelopmentConstantGainReport:
    """Fit ten tiny gains total, reuse all eight existing baseline prediction products."""
    start = process_time_ns()
    if (
        tuple((fit.context, fit.structure) for fit in fits)
        != tuple((context, structure) for context in CONTEXTS for structure in STRUCTURES)
        or any(fit.config != config for fit in fits)
    ):
        raise ValueError("dependent refinement benchmark requires all eight exact context/structure fits")
    gains = np.full(_SHAPES["gains"], np.nan)
    predictions = np.full(_SHAPES["held-receiver-predictions"], np.nan)
    completed = 0
    for c, context in enumerate(CONTEXTS):
        reports = tuple(report for report in projections if report.root.context == context)
        identities = tuple(ObjectIdentity.from_record(report.report_id, report) for report in reports)
        context_fits = fits[c * 4 : (c + 1) * 4]
        if any(fit.projections != identities for fit in context_fits):
            raise ValueError("dependent refinement benchmark projections differ from the fitted observations")
        panel = collect_prepared_response_development_view_panel(config.projection, context, reports)
        folds = context_fits[0].fold_assignments
        amplitude = config.model_spec(context).amplitude
        for fold in range(5):
            training = panel if fold == 4 else panel.subset(tuple(i for i, f in enumerate(folds) if f != fold))
            if training.valid.any():
                gains[c, fold] = fit_prepared_constant_gain(training, amplitude)
                completed += 1
        for s, fit in enumerate(context_fits):
            baseline, _ = fit.arrays()
            for root, fold in enumerate(folds):
                if not np.isfinite(gains[c, fold]).all():
                    continue
                contrast = np.einsum("wp,ptr->wtr", _word_controls(amplitude), gains[c, fold])
                predictions[c, s, root] = baseline[root, ..., :1, :, :2] + contrast
                predictions[c, s, root, ~panel.valid[root]] = np.nan
    arrays = {"gains": gains, "held-receiver-predictions": predictions}
    return PreparedResponseDevelopmentConstantGainReport(
        config, tuple(ObjectIdentity.from_record(fit.report_id, fit) for fit in fits),
        tuple((name, base64.b64encode(np.asarray(array, dtype="<f8").tobytes()).decode("ascii"))
              for name, array in arrays.items()), completed, process_time_ns() - start,
    )
