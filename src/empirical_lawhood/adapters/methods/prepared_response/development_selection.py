"dependent refinement-only eligibility and measured-cost selection of a nominal finite library.\n\nThe result is a development nomination. It cannot publish a calibrated response\nlaw, construct admission evidence or authorize a protected continuation.\n"

from dataclasses import dataclass
from decimal import Decimal
from math import ceil
from typing import ClassVar

import numpy as np

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_stable_id,
)

from .calibration import prepared_joint_root_losses
from .coefficients import PreparedBilinearPredictorCoefficients
from .development_fit import PreparedResponseDevelopmentFitConfig, PreparedResponseDevelopmentModelFit
from .development_projection import PreparedResponseDevelopmentViewProjection, collect_prepared_response_development_view_panel
from .models import STRUCTURES, PreparedTrainingPanel
from .development_policy import PreparedResponseDevelopmentPolicyLibrary, build_prepared_response_development_policy_library


CONTEXTS = ("assembling", "prepared")


@dataclass(frozen=True, slots=True)
class PreparedResponseDevelopmentCandidateCost(CanonicalRecord):
    """Excluded-canary measurement of the deployable instrument and predictor."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/prepared-response/prepared-response-development-candidate-cost'
    cost_id: str
    structure: str
    instrument_tier: str
    excluded_canary: ObjectIdentity
    measurement_repetitions: int
    instrument_cpu_nanoseconds: int
    predictor_cpu_nanoseconds: int
    measurement_rule: str = "MEDIAN_CPU_TIME_COMPLETE_SYNTHETIC_ROOT_VIEW_NO_NATIVE_EFFECT"

    def __post_init__(self) -> None:
        validate_stable_id(self.cost_id, field_name="cost_id")
        expected_tier = "I1" if self.structure == "mechanism-i1" else "I0"
        if (
            self.structure not in STRUCTURES
            or self.instrument_tier != expected_tier
            or type(self.measurement_repetitions) is not int
            or self.measurement_repetitions < 10
            or any(
                type(value) is not int or value <= 0
                for value in (self.instrument_cpu_nanoseconds, self.predictor_cpu_nanoseconds)
            )
            or self.measurement_rule
            != "MEDIAN_CPU_TIME_COMPLETE_SYNTHETIC_ROOT_VIEW_NO_NATIVE_EFFECT"
        ):
            raise ValueError("dependent refinement candidate cost changes its measured finite canary contract")

    @property
    def total_cpu_nanoseconds(self) -> int:
        return self.instrument_cpu_nanoseconds + self.predictor_cpu_nanoseconds


DEVELOPMENT_SCENARIO_ROOT_INDICES = (
    ('assembling', (2, 52, 13, 38, 6, 53, 11, 55, 40, 20, 7, 35, 56, 63, 49, 41)),
    ('prepared', (49, 7, 3, 43, 57, 8, 31, 4, 48, 20, 60, 15, 6, 13, 0, 63)),
)


@dataclass(frozen=True, slots=True)
class PreparedResponseDevelopmentSelectionConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/prepared-response/prepared-response-development-selection-config'
    config_id: str
    fit: PreparedResponseDevelopmentFitConfig
    candidate_costs: tuple[PreparedResponseDevelopmentCandidateCost, ...]
    minimum_finite_roots: int = 62
    minimum_nominal_coverage: Decimal = Decimal("0.90")
    minimum_sharp_roots: int = 48
    minimum_baseline_loss_improvement: Decimal = Decimal("0.05")
    accuracy_rule: str = "WHOLE_ROOT_MAXIMUM_NORMALIZED_OUTER_HELDOUT_LOSS"
    coverage_rule: str = "OUTER_HELDOUT_RESIDUAL_INSIDE_TRAINING_ONLY_NOMINAL_BOX"
    sharpness_rule: str = "BOTH_RECEIVER_HALFWIDTHS_AT_MOST_SOURCE_QUALIFICATION_EPSILON_COMPLETE_CHART"
    baseline_rule: str = "OUTER_TRAINING_ROOT_CELLWISE_CONTEXT_INTERCEPT"
    selection_rule: str = "BOTH_CONTEXT_ELIGIBILITY_THEN_MEASURED_CPU_COST_THEN_STRUCTURE_ORDER"
    scope: str = "DEVELOPMENT_NOMINATION_ONLY_NOT_CALIBRATED_RESPONSE_LAW"
    scenario_root_indices: tuple[tuple[str, tuple[int, ...]], ...] = DEVELOPMENT_SCENARIO_ROOT_INDICES

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        require_sorted_unique_ids(
            self.candidate_costs, attribute="cost_id", field_name="candidate_costs"
        )
        if (
            tuple(value.structure for value in self.candidate_costs) != STRUCTURES
            or self.scenario_root_indices != DEVELOPMENT_SCENARIO_ROOT_INDICES
            or any(type(index) is not int for _, indices in self.scenario_root_indices for index in indices)
            or self.minimum_finite_roots != 62
            or self.minimum_nominal_coverage != Decimal("0.90")
            or self.minimum_sharp_roots != 48
            or self.minimum_baseline_loss_improvement != Decimal("0.05")
            or self.accuracy_rule != "WHOLE_ROOT_MAXIMUM_NORMALIZED_OUTER_HELDOUT_LOSS"
            or self.coverage_rule
            != "OUTER_HELDOUT_RESIDUAL_INSIDE_TRAINING_ONLY_NOMINAL_BOX"
            or self.sharpness_rule
            != "BOTH_RECEIVER_HALFWIDTHS_AT_MOST_SOURCE_QUALIFICATION_EPSILON_COMPLETE_CHART"
            or self.baseline_rule != "OUTER_TRAINING_ROOT_CELLWISE_CONTEXT_INTERCEPT"
            or self.selection_rule
            != "BOTH_CONTEXT_ELIGIBILITY_THEN_MEASURED_CPU_COST_THEN_STRUCTURE_ORDER"
            or self.scope != "DEVELOPMENT_NOMINATION_ONLY_NOT_CALIBRATED_RESPONSE_LAW"
        ):
            raise ValueError("dependent refinement selection changes its finite eligibility or measured-cost rule")


@dataclass(frozen=True, slots=True)
class PreparedResponseDevelopmentCandidateAssessment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/prepared-response/prepared-response-development-candidate-assessment'
    assessment_id: str
    fit: ObjectIdentity
    cost: ObjectIdentity
    context: str
    structure: str
    assigned_roots: int
    finite_loss_roots: int
    covered_roots: int
    sharp_roots: int
    median_normalized_loss: Decimal | None
    baseline_median_normalized_loss: Decimal | None
    baseline_loss_improvement: Decimal | None
    eligible: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.assessment_id, field_name="assessment_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if (
            self.fit.object_schema != PreparedResponseDevelopmentModelFit.SCHEMA
            or self.cost.object_schema != PreparedResponseDevelopmentCandidateCost.SCHEMA
            or self.context not in CONTEXTS
            or self.structure not in STRUCTURES
            or self.assigned_roots != 64
            or any(
                type(value) is not int or not 0 <= value <= self.assigned_roots
                for value in (self.finite_loss_roots, self.covered_roots, self.sharp_roots)
            )
            or any(
                value is not None
                and (
                    type(value) is not Decimal
                    or not value.is_finite()
                    or name != "baseline_loss_improvement"
                    and value < 0
                )
                for name, value in (
                    ("median_normalized_loss", self.median_normalized_loss),
                    (
                        "baseline_median_normalized_loss",
                        self.baseline_median_normalized_loss,
                    ),
                    ("baseline_loss_improvement", self.baseline_loss_improvement),
                )
            )
            or self.eligible != (not self.reason_codes)
        ):
            raise ValueError("dependent refinement assessment changes its complete-root metrics or eligibility")


@dataclass(frozen=True, slots=True)
class PreparedResponseDevelopmentNominalLibrary(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/prepared-response/prepared-response-development-nominal-library'
    config: PreparedResponseDevelopmentSelectionConfig
    fits: tuple[ObjectIdentity, ...]
    assessments: tuple[PreparedResponseDevelopmentCandidateAssessment, ...]
    selected_structure: str | None
    selected_fits: tuple[ObjectIdentity, ...]
    selected_coefficients: tuple[PreparedBilinearPredictorCoefficients, ...]
    policy_library: PreparedResponseDevelopmentPolicyLibrary | None
    disposition: str
    reason_codes: tuple[str, ...]
    grants_law_or_authority: bool = False

    def __post_init__(self) -> None:
        expected_pairs = tuple(
            (context, structure) for context in CONTEXTS for structure in STRUCTURES
        )
        if (
            len(self.fits) != 8
            or any(value.object_schema != PreparedResponseDevelopmentModelFit.SCHEMA for value in self.fits)
            or tuple((value.context, value.structure) for value in self.assessments)
            != expected_pairs
            or tuple(value.fit for value in self.assessments) != self.fits
        ):
            raise ValueError("dependent refinement nominal library changes its two-context candidate census")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.selected_structure is None:
            if (
                self.selected_fits
                or self.selected_coefficients
                or self.policy_library is not None
                or self.disposition != "NOT_SUPPORTED"
                or self.reason_codes != ("NO_COMMON_ELIGIBLE_ABSOLUTE_DESCRIPTION",)
            ):
                raise ValueError("unsupported dependent refinement library cannot fabricate a selected description")
        elif (
            self.selected_structure not in STRUCTURES
            or len(self.selected_fits) != 2
            or len(self.selected_coefficients) != 2
            or self.selected_fits
            != tuple(
                value.fit for value in self.assessments
                if value.structure == self.selected_structure
            )
            or any(
                coefficient.structure != self.selected_structure
                or coefficient.model_spec != self.config.fit.model_spec(context)
                for context, coefficient in zip(CONTEXTS, self.selected_coefficients, strict=True)
            )
            or self.policy_library is None
            or self.policy_library.selection_config
            != ObjectIdentity.from_record(self.config.config_id, self.config)
            or tuple(value.selected_fit for value in self.policy_library.contexts)
            != self.selected_fits
            or self.disposition != "NOMINATED_FOR_FRESH_CALIBRATION"
            or self.reason_codes
        ):
            raise ValueError("dependent refinement nomination changes its selected structure or provisional ceiling")
        if self.grants_law_or_authority:
            raise ValueError("dependent refinement nominal library cannot grant a law, issue or execution authority")

    @property
    def library_id(self) -> str:
        return "prepared-response.dependent-refinement.nominal-library"


def _decimal(value: float) -> Decimal | None:
    return Decimal(format(value, ".17g")) if np.isfinite(value) else None


def _baseline_predictions(panel: PreparedTrainingPanel, folds: tuple[int, ...]) -> np.ndarray:
    predictions = np.full(panel.observed.shape, np.nan)
    for fold in range(4):
        train = np.asarray([value != fold for value in folds])
        valid = panel.valid[train]
        observed = panel.observed[train].copy()
        observed[~valid] = np.nan
        with np.errstate(invalid="ignore"):
            baseline = np.nanmean(observed, axis=(0, 1, 2))
        for index, assignment in enumerate(folds):
            if assignment == fold and panel.valid[index].all():
                predictions[index] = baseline
    return predictions


def _assessment(
    config: PreparedResponseDevelopmentSelectionConfig,
    panel: PreparedTrainingPanel,
    fit: PreparedResponseDevelopmentModelFit,
    cost: PreparedResponseDevelopmentCandidateCost,
) -> PreparedResponseDevelopmentCandidateAssessment:
    predictions, _ = fit.arrays()
    normalizers = np.asarray(config.fit.normalizers, dtype=np.float64)
    losses = prepared_joint_root_losses(panel.observed, predictions, normalizers)
    baseline = _baseline_predictions(panel, fit.fold_assignments)
    baseline_losses = prepared_joint_root_losses(panel.observed, baseline, normalizers)
    finite = np.isfinite(losses)
    baseline_finite = np.isfinite(baseline_losses)
    median = float(np.median(losses[finite])) if finite.any() else np.inf
    baseline_median = (
        float(np.median(baseline_losses[baseline_finite])) if baseline_finite.any() else np.inf
    )
    improvement = (
        (baseline_median - median) / baseline_median
        if np.isfinite(median) and np.isfinite(baseline_median) and baseline_median > 0
        else np.nan
    )
    halfwidth = fit.outer_halfwidths()
    complete = panel.valid.all(axis=(1, 2)) & np.isfinite(predictions).all(
        axis=(1, 2, 3, 4, 5)
    ) & np.isfinite(halfwidth).all(axis=(1, 2, 3, 4, 5))
    covered = complete & (abs(panel.observed - predictions) <= halfwidth).all(
        axis=(1, 2, 3, 4, 5)
    )
    charter = config.fit.projection.source_qualification.selected_charter
    assert charter is not None
    epsilon = float(charter.epsilon)
    sharp = complete & (halfwidth[..., :2] <= epsilon).all(axis=(1, 2, 3, 4, 5))
    reasons = []
    if fit.failure_reason is not None:
        reasons.append(f"FIT_{fit.failure_reason}")
    if int(finite.sum()) < config.minimum_finite_roots:
        reasons.append("INSUFFICIENT_FINITE_HELDOUT_ROOTS")
    if int(covered.sum()) < ceil(64 * float(config.minimum_nominal_coverage)):
        reasons.append("INSUFFICIENT_NOMINAL_HELDOUT_COVERAGE")
    if int(sharp.sum()) < config.minimum_sharp_roots:
        reasons.append("INSUFFICIENT_NOMINAL_RECEIVER_SHARPNESS")
    if not np.isfinite(improvement) or improvement < float(
        config.minimum_baseline_loss_improvement
    ):
        reasons.append("NO_INFORMATIVE_ABSOLUTE_BASELINE_IMPROVEMENT")
    reason_codes = tuple(sorted(reasons))
    return PreparedResponseDevelopmentCandidateAssessment(
        f"prepared-response.dependent-refinement.assess.{fit.context}.{fit.structure}",
        ObjectIdentity.from_record(fit.report_id, fit),
        ObjectIdentity.from_record(cost.cost_id, cost),
        fit.context,
        fit.structure,
        64,
        int(finite.sum()),
        int(covered.sum()),
        int(sharp.sum()),
        _decimal(median),
        _decimal(baseline_median),
        _decimal(improvement),
        not reason_codes,
        reason_codes,
    )


def select_prepared_response_development_nominal_library(
    config: PreparedResponseDevelopmentSelectionConfig,
    projections: tuple[PreparedResponseDevelopmentViewProjection, ...],
    fits: tuple[PreparedResponseDevelopmentModelFit, ...],
) -> PreparedResponseDevelopmentNominalLibrary:
    expected = tuple((context, structure) for context in CONTEXTS for structure in STRUCTURES)
    if tuple((fit.context, fit.structure) for fit in fits) != expected or any(
        fit.config != config.fit for fit in fits
    ):
        raise ValueError("dependent refinement selection requires all eight exact context/structure fits")
    by_context = {
        context: tuple(
            report
            for report in projections
            if report.root.context == context
        )
        for context in CONTEXTS
    }
    projection_identities = {
        context: tuple(ObjectIdentity.from_record(report.report_id, report) for report in reports)
        for context, reports in by_context.items()
    }
    if any(fit.projections != projection_identities[fit.context] for fit in fits):
        raise ValueError("dependent refinement selection projections differ from the fitted observations")
    panels = {
        context: collect_prepared_response_development_view_panel(
            config.fit.projection, context, by_context[context]
        )
        for context in CONTEXTS
    }
    costs = {value.structure: value for value in config.candidate_costs}
    assessments = tuple(
        _assessment(config, panels[fit.context], fit, costs[fit.structure]) for fit in fits
    )
    common = tuple(
        structure
        for structure in STRUCTURES
        if all(
            next(
                value
                for value in assessments
                if value.context == context and value.structure == structure
            ).eligible
            for context in CONTEXTS
        )
    )
    fit_identities = tuple(ObjectIdentity.from_record(value.report_id, value) for value in fits)
    if not common:
        return PreparedResponseDevelopmentNominalLibrary(
            config,
            fit_identities,
            assessments,
            None,
            (),
            (),
            None,
            "NOT_SUPPORTED",
            ("NO_COMMON_ELIGIBLE_ABSOLUTE_DESCRIPTION",),
        )
    selected = min(
        common,
        key=lambda structure: (costs[structure].total_cpu_nanoseconds, STRUCTURES.index(structure)),
    )
    selected_fits = tuple(value for value in fits if value.structure == selected)
    policy_library = build_prepared_response_development_policy_library(
        config, selected, projections, fits
    )
    return PreparedResponseDevelopmentNominalLibrary(
        config,
        fit_identities,
        assessments,
        selected,
        tuple(ObjectIdentity.from_record(value.report_id, value) for value in selected_fits),
        tuple(value.coefficients[-1] for value in selected_fits),
        policy_library,
        "NOMINATED_FOR_FRESH_CALIBRATION",
        (),
    )
