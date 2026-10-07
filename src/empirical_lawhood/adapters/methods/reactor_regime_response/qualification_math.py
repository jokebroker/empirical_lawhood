"""Selected-route root maxima and noncompensating C law/preparation gates."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .config import ROOTS
from .control_math import WordForecast, one_sided_cp
from .measured_panel import MeasuredContext

ROUTES = ("prepared_t0", "c_q", "p_q")


@dataclass(frozen=True)
class SelectedRootOperand:
    root: str
    route: str
    chart: MeasuredContext
    forecast: tuple[WordForecast, WordForecast, WordForecast] | None
    preparation_prefix_valid: bool
    preparation_grid_max_K: float | None

    def __post_init__(self) -> None:
        if self.route not in ROUTES or self.chart.root != self.root or self.chart.context != self.route:
            raise ValueError("selected route or independent root changed")
        if self.forecast is not None and tuple(word.word for word in self.forecast) != (0, 1, 2):
            raise ValueError("selected root lacks its complete three-word prediction")
        if self.preparation_grid_max_K is not None and not isfinite(self.preparation_grid_max_K):
            raise ValueError("preparation maximum temperature is nonfinite")

    @property
    def preparation_safe(self) -> bool:
        return (
            self.preparation_prefix_valid
            and self.preparation_grid_max_K is not None
            and self.preparation_grid_max_K <= 356.2
        )

    @property
    def contacted_valid(self) -> bool:
        return (
            self.chart.valid
            and self.forecast is not None
            and self.chart.nominal_peak_K is not None
            and self.chart.refined_peak_K is not None
            and self.chart.nominal_cooling_K is not None
            and self.chart.refined_cooling_K is not None
            and all(word.supported and word.projection_valid for word in self.forecast)
        )


@dataclass(frozen=True)
class RootPrecision:
    root: str
    contacted_valid: bool
    temperature_score: float | None
    cooling_score: float | None
    maximum_absolute_error_K: float | None
    maximum_contrast_error_K: float | None
    reasons: tuple[str, ...]


def root_precision(operand: SelectedRootOperand, *, coverage: bool = False) -> RootPrecision:
    """Raw error calibrates q; only coverage uses the numerical padding."""
    reasons = set(operand.chart.reasons)
    if not operand.contacted_valid:
        reasons.add("MISSING_VALID_SUPPORTED_SELECTED_CHART")
        return RootPrecision(operand.root, False, None, None, None, None, tuple(sorted(reasons)))
    forecast = operand.forecast
    assert forecast is not None
    measured = (
        (operand.chart.nominal_peak_K, operand.chart.nominal_cooling_K),
        (operand.chart.refined_peak_K, operand.chart.refined_cooling_K),
    )
    if any(peak is None or cooling is None for peak, cooling in measured):
        raise ValueError("contacted root lost a numerical view")
    temperature_errors = []
    cooling_errors = []
    cooling_scores = []
    for peak, cooling in measured:
        assert peak is not None and cooling is not None
        for word in range(3):
            t_error = abs(forecast[word].predicted_peak_K - peak[word])
            c_error = abs(forecast[word].predicted_cooling_K - cooling[word])
            temperature_errors.append(t_error)
            cooling_errors.append(c_error)
            cooling_scores.append(
                max(c_error - (1e-6 if coverage else 0), 0)
                / (0.00005 + 0.5 * abs(forecast[word].predicted_cooling_K))
            )
    if not all(isfinite(value) for value in (*temperature_errors, *cooling_errors, *cooling_scores)):
        reasons.add("NONFINITE_ROOT_PRECISION_OPERAND")
        return RootPrecision(operand.root, False, None, None, None, None, tuple(sorted(reasons)))
    return RootPrecision(
        operand.root,
        True,
        max(max(error - (0.01 if coverage else 0), 0) / 0.25 for error in temperature_errors),
        max(cooling_scores),
        max(temperature_errors),
        max(cooling_errors),
        tuple(sorted(reasons)),
    )


@dataclass(frozen=True)
class Calibration:
    contacted_roots: tuple[str, ...]
    q_temperature: float | None
    q_cooling: float | None
    precision_pass: bool
    reasons: tuple[str, ...]


def calibrate(scores: tuple[RootPrecision, ...]) -> Calibration:
    expected = tuple(r for r, role, _, _ in ROOTS if role == "calibration")
    if tuple(score.root for score in scores) != expected:
        raise ValueError("calibration must retain all 32 assigned roots in order")
    contact = tuple(score for score in scores if score.contacted_valid)
    reasons = set()
    if len(contact) < 29:
        reasons.add("INSUFFICIENT_CALIBRATION_CONTACT")
    if any(score.temperature_score is None or score.cooling_score is None for score in contact):
        reasons.add("MISSING_CALIBRATION_SCORE")
    if not contact or "MISSING_CALIBRATION_SCORE" in reasons:
        q_t = q_c = None
    else:
        q_t = max(float(score.temperature_score) for score in contact if score.temperature_score is not None)
        q_c = max(float(score.cooling_score) for score in contact if score.cooling_score is not None)
        if q_t > 1:
            reasons.add("ABSOLUTE_PRECISION_FAILED")
        if q_c > 1:
            reasons.add("COOLING_PRECISION_FAILED")
    return Calibration(
        tuple(score.root for score in contact),
        q_t,
        q_c,
        not reasons,
        tuple(sorted(reasons)),
    )


@dataclass(frozen=True)
class RootCoverage:
    root: str
    contacted_valid: bool
    interval_covered: bool
    law_adequate: bool
    preparation_safe: bool
    joined: bool
    reasons: tuple[str, ...]


def qualification_root(operand: SelectedRootOperand, calibrated: Calibration) -> RootCoverage:
    score = root_precision(operand, coverage=True)
    covered = (
        score.contacted_valid
        and calibrated.q_temperature is not None
        and calibrated.q_cooling is not None
        and score.temperature_score is not None
        and score.cooling_score is not None
        and score.temperature_score <= calibrated.q_temperature
        and score.cooling_score <= calibrated.q_cooling
    )
    reasons = set(score.reasons)
    if not covered:
        reasons.add("SELECTED_LAW_NOT_COVERED")
    if not operand.preparation_safe:
        reasons.add("SELECTED_PREPARATION_UNSAFE_OR_INVALID")
    return RootCoverage(
        operand.root,
        score.contacted_valid,
        bool(covered),
        bool(covered and calibrated.precision_pass),
        operand.preparation_safe,
        bool(covered and calibrated.precision_pass and operand.preparation_safe),
        tuple(sorted(reasons)),
    )


@dataclass(frozen=True)
class Qualification:
    assignments: int
    law_successes: int
    joint_successes: int
    law_lower_95: float
    joint_lower_95: float
    law_pass: bool
    release_D: bool
    reasons: tuple[str, ...]


def qualify(roots: tuple[RootCoverage, ...], calibration: Calibration) -> Qualification:
    expected = tuple(r for r, role, _, _ in ROOTS if role == "qualification")
    if tuple(root.root for root in roots) != expected:
        raise ValueError("qualification must retain all 64 assigned roots in order")
    law = sum(root.law_adequate for root in roots)
    joined = sum(root.joined for root in roots)
    law_lower = one_sided_cp(law, 64, lower=True)
    joint_lower = one_sided_cp(joined, 64, lower=True)
    reasons = set(calibration.reasons)
    if law_lower < 0.90:
        reasons.add("JOINT_LAW_ADEQUACY_FAILED")
    if joint_lower < 0.90:
        reasons.add("LAW_AND_PREPARATION_JOINT_FAILED")
    return Qualification(
        64,
        law,
        joined,
        law_lower,
        joint_lower,
        calibration.precision_pass and law_lower >= 0.90,
        calibration.precision_pass and joint_lower >= 0.90,
        tuple(sorted(reasons)),
    )
